# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public creator/follower model bootstrap backed by shared host memory.

The application supplies two model factories.  The elected creator is the
only process that invokes the checkpoint-loading factory; followers invoke the
lightweight skeleton factory after the creator has published its live profile.
FlexTensor owns election, shared tensor publication, restored-state setup, and
resource teardown.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable  # noqa: TC003 - beartype resolves public annotations at runtime.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from flextensor.config import OffloadConfig  # noqa: TC001 - beartype resolves public annotations at runtime.
from flextensor.offload_manager import DEFAULT_MANAGER_NAME, OffloadManager, get_offload_manager
from flextensor.shm.coordinator import ShmCoordinator
from flextensor.shm.namespace import SHM_PROTOCOL_VERSION
from flextensor.shm.tensor_store import SharedTensorStore
from flextensor.state_handler import TensorManagerStateHandler

LOGGER = logging.getLogger(__name__)

_NAMESPACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_MANAGER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


@dataclass(slots=True)
class _LoadResources:
    """Resources acquired by one load and retained until cleanup succeeds."""

    manager: OffloadManager | None = None
    owner_token: object = field(default_factory=object)
    store: SharedTensorStore | None = None
    coordinator: ShmCoordinator | None = None
    preserve_model: bool = False

    def close(self) -> None:
        """Release in dependency order, retaining unfinished work for retry."""
        if self.manager is not None:
            self.manager._release_shared_offload(  # noqa: SLF001
                owner_token=self.owner_token,
                preserve_model=self.preserve_model,
            )
            self.manager = None

        if self.store is not None:
            self.store.close()
            self.store = None

        if self.coordinator is not None:
            self.coordinator.close()
            self.coordinator = None


def _normalize_shared_offload_options(namespace: object, wait_timeout: Any) -> tuple[str, float]:
    """Validate raw session options and return their canonical runtime values."""
    if not isinstance(namespace, str) or _NAMESPACE_RE.fullmatch(namespace) is None:
        raise ValueError(
            "Shared offload namespace must be 1-96 POSIX-safe characters: letters, digits, '.', '_' or '-'"
        )
    if isinstance(wait_timeout, bool):
        raise ValueError("Shared offload wait timeout must be a non-negative finite number")
    try:
        normalized_timeout = float(wait_timeout)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Shared offload wait timeout must be a non-negative finite number") from error
    if not math.isfinite(normalized_timeout) or normalized_timeout < 0:
        raise ValueError("Shared offload wait timeout must be a non-negative finite number")
    return namespace, normalized_timeout


def _resident_parameter_names(model: nn.Module, tensor_names: Iterable[str]) -> list[str]:
    """Split resident parameters from the state's parameter/buffer inventory."""
    parameters = dict(model.named_parameters(recurse=True, remove_duplicate=False))
    buffers = dict(model.named_buffers(recurse=True, remove_duplicate=False))
    names = list(tensor_names)

    if any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError("Shared offload GPU tensor names must be non-empty strings")

    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate_names:
        raise RuntimeError(f"Shared offload GPU tensor names contain duplicates: {duplicate_names!r}")

    registered_names = parameters.keys() | buffers.keys()
    unknown_names = sorted(set(names) - registered_names)
    if unknown_names:
        raise RuntimeError(f"Shared offload GPU tensor names are not registered by the final model: {unknown_names!r}")
    return [name for name in names if name in parameters]


class SharedOffloadSession:
    """Own multiple shared offload managers for one worker process.

    Construct one session independently in every worker after process spawning.
    Workers that use the same base namespace elect a creator independently for
    each manager name passed to :meth:`offload_from_profile`.

    The session is process- and thread-affine: construct it after spawning and
    perform every load and close operation from that same thread.

    The creator factory is invoked only in the elected creator.  Followers wait
    for publication and then invoke the follower factory, which should normally
    construct the same model topology on the ``meta`` device.

    Args:
        namespace: Run-scoped namespace shared by all participating workers.
        wait_timeout: Timeout in seconds for followers waiting for readiness.
            Defaults to 300 seconds. Zero means no overall limit; creator
            heartbeat checks remain active.
    """

    def __init__(self, namespace: str, *, wait_timeout: Any = 300.0) -> None:
        if sys.platform == "win32":
            raise NotImplementedError("Shared memory (SHM) is not supported on Windows")
        normalized_namespace, normalized_timeout = _normalize_shared_offload_options(namespace, wait_timeout)

        self.namespace = normalized_namespace
        self.wait_timeout = normalized_timeout
        self._resources_by_manager_name: dict[str, _LoadResources] = {}
        self._loading_manager_names: set[str] = set()
        self._pending_cleanup_by_manager_name: dict[str, _LoadResources] = {}
        self._closed = False
        self._owner_pid = os.getpid()
        self._owner_thread = threading.get_ident()

    @staticmethod
    def validate_options(namespace: object, *, wait_timeout: Any = 300.0) -> None:
        """Validate raw constructor options without creating a session or SHM resources.

        Args:
            namespace: Candidate run-scoped namespace shared by all workers.
            wait_timeout: Candidate follower-readiness timeout, in seconds.
                Zero means no overall limit.

        Raises:
            ValueError: If the namespace is not POSIX-safe or the timeout is not
                a non-negative finite number.
        """
        _normalize_shared_offload_options(namespace, wait_timeout)

    def _check_owner(self) -> None:
        """Reject inherited or cross-thread use of process-local handles."""
        current_pid = os.getpid()
        current_thread = threading.get_ident()
        if current_pid != self._owner_pid:
            raise RuntimeError(
                "SharedOffloadSession cannot be used after fork; construct a new session inside each worker process"
            )
        if current_thread != self._owner_thread:
            raise RuntimeError(
                f"SharedOffloadSession belongs to thread {self._owner_thread}, "
                f"but was accessed from thread {current_thread}"
            )

    def namespace_for(self, name: str) -> str:
        """Return a compact independent SHM namespace for a named manager."""
        if not isinstance(name, str) or _MANAGER_RE.fullmatch(name) is None:
            raise ValueError(
                "Shared offload manager name must be 1-64 POSIX-safe characters: letters, digits, '.', '_' or '-'"
            )
        digest = hashlib.sha256(f"{self.namespace}\0{name}".encode()).hexdigest()[:12]
        return f"ft_{digest}"

    def offload_from_profile(  # noqa: C901 - keep shared transaction and error handling together.
        self,
        profile_directory: str | os.PathLike[str],
        *,
        config: OffloadConfig,
        name: str = DEFAULT_MANAGER_NAME,
        creator_model_factory: Callable[[], nn.Module],
        follower_model_factory: Callable[[], nn.Module],
        materialize_device: torch.device | str | None = None,
        compile_fn: Callable[[nn.Module], nn.Module] | None = None,
    ) -> nn.Module:
        """Elect a role and restore a model directly into its saved placement.

        Both factories must construct matching registered parameter and buffer
        names.  The creator factory may load checkpoint data; the follower
        factory must not depend on checkpoint tensor payload.

        Args:
            profile_directory: Directory containing the saved FlexTensor
                profile.  Only the elected creator reads it.
            config: Enabled allocation-block transfer configuration.  The
                session copies it and supplies its own SHM namespace and wait
                timeout without mutating the caller's object.
            name: Named offload manager to initialize.  The name also selects
                an independent child namespace and creator election.
            creator_model_factory: Zero-argument factory that constructs the
                checkpoint-backed model.  It runs only in the elected creator.
            follower_model_factory: Zero-argument factory that constructs the
                same registered model topology without checkpoint payload.  It
                runs only in followers and normally builds on ``meta``.
            materialize_device: Device for follower-resident parameters and
                buffers.  Defaults to the CUDA device selected by *config*.
            compile_fn: Optional per-unit compiler applied during restored-state
                initialization.

        Returns:
            The normal FlexTensor model proxy in inference phase.

        Raises:
            ValueError: If *name* or *config* is invalid for shared offload.
            RuntimeError: If the session or manager lifecycle is incompatible
                with the requested load, or shared state is invalid.
            TimeoutError: If a follower exceeds the coordination timeout.
            ShmCreatorFailedError: If the creator publishes an initialization
                failure.
            ShmCreatorDiedError: If the creator heartbeat expires while it is
                still initializing.

        Note:
            Exceptions raised by either model factory or by *compile_fn*
            propagate unchanged after transactional cleanup.
        """
        self._check_owner()
        manager_namespace = self.namespace_for(name)
        if not config.enabled:
            raise ValueError("Shared offload requires OffloadConfig.enabled=True")
        if config.transfer_mode != "allocation_block_transfer":
            raise ValueError("Shared offload requires OffloadConfig.transfer_mode='allocation_block_transfer'")
        if self._closed:
            raise RuntimeError("Shared offload session is closed")
        if (
            name in self._resources_by_manager_name
            or name in self._loading_manager_names
            or name in self._pending_cleanup_by_manager_name
        ):
            raise RuntimeError(f"Shared offload manager {name!r} has already been loaded")
        self._loading_manager_names.add(name)

        resources = _LoadResources()
        ready_published = False
        try:
            manager_config = config.model_copy(
                update={
                    "shm_enabled": True,
                    "shm_namespace": manager_namespace,
                    "shm_wait_timeout": self.wait_timeout,
                },
            )
            manager = get_offload_manager(name)
            manager._prepare_shared_offload(  # noqa: SLF001 - package-internal transaction boundary.
                manager_config,
                owner_token=resources.owner_token,
            )
            resources.manager = manager

            # Election happens here, before either application factory runs.
            coordinator = ShmCoordinator(manager_namespace)
            resources.coordinator = coordinator
            role = "creator" if coordinator.is_creator else "follower"
            LOGGER.info(
                "Shared offload elected role=%s manager=%s namespace=%s protocol=%d",
                role,
                name,
                manager_namespace,
                SHM_PROTOCOL_VERSION,
            )

            if coordinator.is_creator:
                proxy = self._load_creator(
                    resources,
                    manager,
                    coordinator,
                    name=name,
                    profile_directory=profile_directory,
                    model_factory=creator_model_factory,
                    compile_fn=compile_fn,
                )
            else:
                proxy = self._load_follower(
                    resources,
                    manager,
                    coordinator,
                    model_factory=follower_model_factory,
                    materialize_device=materialize_device,
                    config=manager_config,
                    compile_fn=compile_fn,
                )

            if self._closed:
                raise RuntimeError(f"Shared offload session closed during load of manager {name!r}")
            self._resources_by_manager_name[name] = resources
            if coordinator.is_creator:
                coordinator.notify_ready()
                ready_published = True

            LOGGER.info(
                "Shared offload initialized role=%s manager=%s namespace=%s protocol=%d",
                role,
                name,
                manager_namespace,
                SHM_PROTOCOL_VERSION,
            )
            return proxy
        except BaseException as primary_error:
            if resources.store is None:
                resources.store = SharedTensorStore._take_rejected(primary_error)  # noqa: SLF001
            if resources.coordinator is not None and resources.coordinator.is_creator and not ready_published:
                try:
                    resources.coordinator.mark_failed(primary_error)
                except BaseException as publication_error:
                    if add_note := getattr(primary_error, "add_note", None):
                        add_note(
                            f"FlexTensor could not publish this creator initialization failure: {publication_error!r}"
                        )
                    LOGGER.exception(
                        "Failed to publish shared offload creator failure for manager=%s namespace=%s",
                        name,
                        manager_namespace,
                    )
            self._cleanup_failed_load(name, resources)
            raise
        finally:
            self._loading_manager_names.discard(name)

    @staticmethod
    def _load_creator(
        resources: _LoadResources,
        manager: OffloadManager,
        coordinator: ShmCoordinator,
        *,
        name: str,
        profile_directory: str | os.PathLike[str],
        model_factory: Callable[[], nn.Module],
        compile_fn: Callable[[nn.Module], nn.Module] | None,
    ) -> nn.Module:
        """Restore and publish creator data; the outer transaction publishes READY."""
        disk_state = TensorManagerStateHandler.load_from_file(Path(profile_directory) / "profile.json")
        model = model_factory()

        def publish() -> None:
            final_model = manager.model
            if final_model is None:
                raise RuntimeError(f"Shared offload creator manager {name!r} did not retain its final model")
            live_state = manager._export_runtime_state()  # noqa: SLF001 - internal shared restore API.
            parameter_names = _resident_parameter_names(final_model, live_state.gpu_tensors_names)
            resources.store = SharedTensorStore.create_from_model(
                coordinator.namespace,
                final_model,
                parameter_names=parameter_names,
                shm_owner=coordinator.memory_owner,
            )
            coordinator.write_profile(live_state)

        return manager._restore_shared_offload(  # noqa: SLF001 - internal shared restore API.
            model,
            disk_state,
            coordinator.namespace,
            load_from_shm=False,
            owner_token=resources.owner_token,
            shm_owner=coordinator.memory_owner,
            compile_fn=compile_fn,
            before_compile=publish,
        )

    def _load_follower(
        self,
        resources: _LoadResources,
        manager: OffloadManager,
        coordinator: ShmCoordinator,
        *,
        model_factory: Callable[[], nn.Module],
        materialize_device: torch.device | str | None,
        config: OffloadConfig,
        compile_fn: Callable[[nn.Module], nn.Module] | None,
    ) -> nn.Module:
        """Wait for READY and restore a follower, retaining its store for cleanup."""
        coordinator.wait_for_ready(timeout=self.wait_timeout or None)
        live_state = coordinator.read_profile()
        model = model_factory()
        _resident_parameter_names(model, live_state.gpu_tensors_names)
        resources.store = SharedTensorStore.attach(coordinator.namespace, shm_owner=coordinator.memory_owner)
        target_device = (
            materialize_device if materialize_device is not None else torch.device(f"cuda:{config.gpu_device}")
        )
        resources.store.materialize_into(
            model,
            device=target_device,
        )
        proxy = manager._restore_shared_offload(  # noqa: SLF001 - internal shared restore API.
            model,
            live_state,
            coordinator.namespace,
            load_from_shm=True,
            owner_token=resources.owner_token,
            shm_owner=coordinator.memory_owner,
            compile_fn=compile_fn,
        )
        return proxy

    def _cleanup_failed_load(
        self,
        name: str,
        resources: _LoadResources,
    ) -> None:
        """Best-effort rollback without replacing the initialization error."""
        self._resources_by_manager_name.pop(name, None)

        try:
            resources.close()
        except BaseException:
            self._pending_cleanup_by_manager_name[name] = resources
            LOGGER.exception(
                "Failed to clean up shared offload manager=%s after initialization error; "
                "resources retained for close() retry",
                name,
            )

    def close(self, *, preserve_models: bool = False) -> None:
        """Release loaded managers in reverse order.

        Closing is idempotent.  Failed teardown retains the remaining handles
        so another call can retry it.

        Args:
            preserve_models: Clone loader-managed values into private storage
                before teardown.  Keep the default ``False`` for disposable
                serving workers; their returned proxies become unusable.
        """
        self._check_owner()
        if self._closed and not self._resources_by_manager_name and not self._pending_cleanup_by_manager_name:
            return
        self._closed = True
        first_error: BaseException | None = None

        for name in reversed(self._resources_by_manager_name):
            resources = self._resources_by_manager_name[name]
            resources.preserve_model = preserve_models
            self._pending_cleanup_by_manager_name[name] = resources
        self._resources_by_manager_name.clear()

        for name in list(self._pending_cleanup_by_manager_name):
            try:
                self._pending_cleanup_by_manager_name[name].close()
            except BaseException as cleanup_error:
                if first_error is None:
                    first_error = cleanup_error
            else:
                del self._pending_cleanup_by_manager_name[name]
        if first_error is not None:
            raise first_error

    def __enter__(self) -> SharedOffloadSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_type is None:
                raise
            if isinstance(exc, BaseException) and (add_note := getattr(exc, "add_note", None)):
                add_note(
                    f"FlexTensor shared offload cleanup also failed; call close() again to retry: {cleanup_error!r}"
                )
            LOGGER.exception(
                "Failed to close shared offload session while preserving an in-flight exception",
            )


__all__ = ["SharedOffloadSession"]

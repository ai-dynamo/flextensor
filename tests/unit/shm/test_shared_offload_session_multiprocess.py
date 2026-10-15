# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CPU multiprocess coverage for the public shared-offload orchestration."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import tempfile
import traceback
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import torch
from shm.multiprocess_utils import (
    EVENT_TIMEOUT,
    assert_clean_exit,
    drain_results,
    unlink_shared_memory_if_present,
    wait_for_event,
)
from torch import nn

import flextensor.shared_offload as shared_offload_module
from flextensor.config import OffloadConfig
from flextensor.offload_manager import OffloadManager, OffloadModelProxy, OffloadPhase
from flextensor.shared_offload import SharedOffloadSession
from flextensor.shm.coordinator import ShmCoordinatorError
from flextensor.shm.namespace import coord_block_name, profile_block_name
from flextensor.state_handler import TensorManagerState, TensorManagerStateHandler

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as EventType

    from flextensor.shm.flexible_shm import FlexibleSharedMemory


_MANAGER_NAME = "m"
_PROCESS_TIMEOUT = max(20.0, EVENT_TIMEOUT * 4)
_EXPECTED_WEIGHT = [[2.0, -1.0], [0.5, 3.0]]
_EXPECTED_OFFSET = [0.25, -0.75]
_EXPECTED_OUTPUT = [[5.75, -6.75]]


class _TinyCpuModel(nn.Module):
    """Two-tensor topology whose follower starts entirely on ``meta``."""

    def __init__(self, *, device: str) -> None:
        super().__init__()
        if device == "meta":
            weight = torch.empty((2, 2), dtype=torch.float32, device="meta")
            offset = torch.empty(2, dtype=torch.float32, device="meta")
        else:
            weight = torch.tensor(_EXPECTED_WEIGHT, dtype=torch.float32, device=device)
            offset = torch.tensor(_EXPECTED_OFFSET, dtype=torch.float32, device=device)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("offset", offset)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight + self.offset


class _CpuSharedOffloadManager(OffloadManager):
    """Keep real manager ownership while replacing CUDA-dependent restoration."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._published_state: TensorManagerState | None = None

    def _install_cpu_model(
        self,
        model: nn.Module,
        state: TensorManagerState,
        namespace: str,
        *,
        owner_token: object | None,
        compile_fn: Callable[[nn.Module], nn.Module] | None,
    ) -> OffloadModelProxy:
        if self._shared_offload_owner_token is not owner_token:  # noqa: SLF001 - verify real ownership contract.
            raise RuntimeError("CPU model installation does not own the shared-offload reservation")
        self._validate_restored_shm_namespace(namespace)
        if compile_fn is not None:
            raise AssertionError("CPU shared-offload integration does not compile models")
        self._model = model
        self._model_proxy = OffloadModelProxy(model, self)
        self._current_phase = OffloadPhase.INFERENCE
        self._published_state = state
        return self._model_proxy

    def _restore_shared_offload(
        self,
        model: nn.Module,
        state: TensorManagerState,
        namespace: str,
        *,
        load_from_shm: bool,
        owner_token: object | None = None,
        shm_owner: FlexibleSharedMemory | None = None,
        compile_fn: Callable[[nn.Module], nn.Module] | None = None,
        before_compile: Callable[[], None] | None = None,
    ) -> OffloadModelProxy:
        proxy = self._install_cpu_model(
            model,
            state,
            namespace,
            owner_token=owner_token,
            compile_fn=compile_fn,
        )
        if before_compile is not None:
            before_compile()
        return proxy

    def _export_runtime_state(self) -> TensorManagerState:
        if self._published_state is None:
            raise RuntimeError("CPU shared-offload manager has no state to publish")
        return self._published_state


def _install_process_local_cpu_manager() -> None:
    """Replace only the CUDA-dependent named-manager factory in one worker."""
    managers: dict[str, _CpuSharedOffloadManager] = {}

    def get_manager(name: str) -> _CpuSharedOffloadManager:
        return managers.setdefault(name, _CpuSharedOffloadManager(name))

    shared_offload_module.get_offload_manager = get_manager


def _config() -> OffloadConfig:
    return OffloadConfig(
        pinned_memory=False,
        transfer_mode="allocation_block_transfer",
    )


def _creator_process(
    namespace: str,
    profile_directory: str,
    published: EventType,
    close_allowed: EventType,
    result_queue: Queue,
) -> None:
    """Publish the model, then retain every real SHM owner until released."""
    _install_process_local_cpu_manager()
    session: SharedOffloadSession | None = None
    creator_factory_calls = 0
    try:
        session = SharedOffloadSession(namespace, wait_timeout=_PROCESS_TIMEOUT)

        def creator_factory() -> nn.Module:
            nonlocal creator_factory_calls
            creator_factory_calls += 1
            return _TinyCpuModel(device="cpu")

        with patch.object(
            TensorManagerStateHandler, "load_from_file", wraps=TensorManagerStateHandler.load_from_file
        ) as load_profile:
            proxy = session.offload_from_profile(
                profile_directory,
                config=_config(),
                name=_MANAGER_NAME,
                creator_model_factory=creator_factory,
                follower_model_factory=lambda: _raise_wrong_factory("creator invoked follower factory"),
                materialize_device="cpu",
            )
        output = proxy(torch.tensor([[3.0, -1.0]], dtype=torch.float32))
        weight = proxy.weight.detach().tolist()
        offset = proxy.offset.detach().tolist()
        published.set()
        # The parent always opens this gate in its ``finally`` block. Waiting
        # without a competing timeout guarantees that a slow spawn cannot make
        # the creator start teardown before the follower has closed.
        close_allowed.wait()
        session.close()
        session = None
        result_queue.put({
            "success": True,
            "role": "creator",
            "creator_factory_calls": creator_factory_calls,
            "profile_loads": load_profile.call_count,
            "weight": weight,
            "offset": offset,
            "output": output.detach().tolist(),
            "pid": os.getpid(),
        })
    except BaseException as error:
        result_queue.put({
            "success": False,
            "role": "creator",
            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            "pid": os.getpid(),
        })
        raise
    finally:
        if session is not None:
            session.close()


def _follower_process(
    namespace: str,
    profile_directory: str,
    result_queue: Queue,
    building_model: EventType,
    attach_allowed: EventType,
) -> None:
    """Attach after READY, materialize the meta topology, and verify values."""
    _install_process_local_cpu_manager()
    session: SharedOffloadSession | None = None
    follower_factory_calls = 0
    try:
        session = SharedOffloadSession(namespace, wait_timeout=_PROCESS_TIMEOUT)

        def follower_factory() -> nn.Module:
            nonlocal follower_factory_calls
            follower_factory_calls += 1
            building_model.set()
            wait_for_event(attach_allowed, "follower attachment", timeout=_PROCESS_TIMEOUT)
            model = _TinyCpuModel(device="meta")
            if not model.weight.is_meta or not model.offset.is_meta:
                raise AssertionError("follower factory did not construct a meta topology")
            return model

        with patch.object(
            TensorManagerStateHandler, "load_from_file", wraps=TensorManagerStateHandler.load_from_file
        ) as load_profile:
            proxy = session.offload_from_profile(
                profile_directory,
                config=_config(),
                name=_MANAGER_NAME,
                creator_model_factory=lambda: _raise_wrong_factory("follower invoked creator factory"),
                follower_model_factory=follower_factory,
                materialize_device="cpu",
            )
        output = proxy(torch.tensor([[3.0, -1.0]], dtype=torch.float32))
        result = {
            "success": True,
            "role": "follower",
            "follower_factory_calls": follower_factory_calls,
            "profile_loads": load_profile.call_count,
            "weight": proxy.weight.detach().tolist(),
            "offset": proxy.offset.detach().tolist(),
            "output": output.detach().tolist(),
            "pid": os.getpid(),
        }
        session.close()
        session = None
        result_queue.put(result)
    except BaseException as error:
        result_queue.put({
            "success": False,
            "role": "follower",
            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            "pid": os.getpid(),
        })
        raise
    finally:
        if session is not None:
            session.close()


def _raise_wrong_factory(message: str) -> Any:
    raise AssertionError(message)


def _resident_only_state() -> TensorManagerState:
    state = TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={1: "weight", 2: "offset"},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={},
        release_strategy={},
        label_to_block_id={},
        stats=[],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=["weight", "offset"],
        shm_block_name_map={},
    )
    state.validate_internal()
    return state


def _fsm_names(manager_namespace: str) -> tuple[str, ...]:
    return (
        coord_block_name(manager_namespace),
        coord_block_name(manager_namespace) + "_b",
        profile_block_name(manager_namespace),
        f"{manager_namespace}_t",
    )


def _fsm_shm_names(name: str) -> tuple[str, str, str]:
    return name, f"sm_{name}_kd", f"{name}_km"


def _cleanup_fsm_namespace(name: str) -> None:
    for shm_name in _fsm_shm_names(name):
        unlink_shared_memory_if_present(shm_name)

    # The main lock path intentionally survives generations. This test's unique
    # namespace is no longer in use once both processes have exited.
    temp_dir = Path(tempfile.gettempdir())
    for lock_name in (name, f"{name}_ks"):
        (temp_dir / f"{lock_name}.lock").unlink(missing_ok=True)


def _assert_segments_removed(manager_namespace: str) -> None:
    for fsm_name in _fsm_names(manager_namespace):
        # Only the keep-alive lock is generation-owned; the main lock is a stable anchor.
        lock_path = Path(tempfile.gettempdir()) / f"{fsm_name}_ks.lock"
        assert not lock_path.exists(), f"Lock file {lock_path} survived final session close"
        for shm_name in _fsm_shm_names(fsm_name):
            try:
                unexpected = SharedMemory(name=shm_name, create=False)
            except FileNotFoundError:
                continue
            unexpected.close()
            pytest.fail(f"Shared-memory segment {shm_name!r} survived final session close")


@pytest.mark.parametrize("creator_closes_first", [False, True])
def test_creator_publishes_and_spawned_follower_materializes_real_shared_state(
    tmp_path: Path, creator_closes_first: bool
) -> None:
    """Exercise public election, profile publication, tensor copy, and ordered cleanup."""
    context = multiprocessing.get_context("spawn")
    namespace = f"ft_{uuid.uuid4().hex[:8]}"
    manager_namespace = SharedOffloadSession(namespace).namespace_for(_MANAGER_NAME)
    profile_directory = tmp_path / "profile"
    profile_directory.mkdir()
    TensorManagerStateHandler.save_to_file(profile_directory / "profile.json", _resident_only_state())

    published = context.Event()
    close_creator = context.Event()
    building_model = context.Event()
    attach_allowed = context.Event()
    result_queue = context.Queue()
    creator = context.Process(
        target=_creator_process,
        args=(namespace, os.fspath(profile_directory), published, close_creator, result_queue),
    )
    follower = context.Process(
        target=_follower_process,
        args=(namespace, os.fspath(profile_directory), result_queue, building_model, attach_allowed),
    )

    try:
        creator.start()
        try:
            wait_for_event(published, "shared-offload creator publication", proc=creator, timeout=_PROCESS_TIMEOUT)
        except AssertionError as error:
            message = str(error)
            with contextlib.suppress(AssertionError):
                result = drain_results(result_queue, expected_count=1, timeout=EVENT_TIMEOUT)[0]
                message += f"\n{result.get('error', result)}"
            raise AssertionError(message) from error
        if not creator.is_alive():
            creator.join(timeout=_PROCESS_TIMEOUT)
            raise AssertionError(f"shared-offload creator exited early with code {creator.exitcode}")

        # Starting only after offload_from_profile() returned also covers a
        # follower that was not waiting when the READY notification was sent.
        follower.start()
        wait_for_event(building_model, "follower model factory", proc=follower, timeout=_PROCESS_TIMEOUT)
        if creator_closes_first:
            close_creator.set()
            creator.join(timeout=_PROCESS_TIMEOUT)
            assert_clean_exit(creator, "shared-offload creator")
        attach_allowed.set()
        follower.join(timeout=_PROCESS_TIMEOUT)
        assert_clean_exit(follower, "shared-offload follower")
        if not creator_closes_first:
            assert creator.is_alive(), "creator should still be waiting for its close gate"

        close_creator.set()
        creator.join(timeout=_PROCESS_TIMEOUT)
        assert_clean_exit(creator, "shared-offload creator")

        results = drain_results(result_queue, expected_count=2, timeout=_PROCESS_TIMEOUT)
        assert all(result["success"] for result in results), results
        by_role = {result["role"]: result for result in results}
        assert set(by_role) == {"creator", "follower"}
        assert by_role["creator"]["creator_factory_calls"] == 1
        assert by_role["creator"]["profile_loads"] == 1
        assert by_role["follower"]["follower_factory_calls"] == 1
        assert by_role["follower"]["profile_loads"] == 0
        for result in by_role.values():
            assert result["weight"] == _EXPECTED_WEIGHT
            assert result["offset"] == _EXPECTED_OFFSET
            assert result["output"] == _EXPECTED_OUTPUT

        _assert_segments_removed(manager_namespace)
    finally:
        close_creator.set()
        attach_allowed.set()
        for process in (follower, creator):
            if process.pid is None:
                continue
            process.join(timeout=_PROCESS_TIMEOUT)
            if process.is_alive():
                process.terminate()
                process.join(timeout=_PROCESS_TIMEOUT)
        for name in _fsm_names(manager_namespace):
            _cleanup_fsm_namespace(name)


def test_session_rejects_dead_generation_before_profile_read_or_factories(tmp_path, monkeypatch):
    namespace = f"ft_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(shared_offload_module, "get_offload_manager", _CpuSharedOffloadManager)
    profile = tmp_path / "profile.json"
    TensorManagerStateHandler.save_to_file(profile, _resident_only_state())

    with SharedOffloadSession(namespace) as creator, SharedOffloadSession(namespace) as follower:
        original = creator.offload_from_profile(
            tmp_path,
            config=_config(),
            name=_MANAGER_NAME,
            creator_model_factory=lambda: _TinyCpuModel(device="cpu"),
            follower_model_factory=lambda: pytest.fail("expected creator"),
            materialize_device="cpu",
        )
        creator._resources_by_manager_name[_MANAGER_NAME].coordinator.memory_owner.keep_alive.stop()
        with pytest.raises(ShmCoordinatorError, match="no live participants") as exc_info:
            follower.offload_from_profile(
                tmp_path / "missing_profile",
                config=_config(),
                name=_MANAGER_NAME,
                creator_model_factory=lambda: pytest.fail("dead generation invoked creator factory"),
                follower_model_factory=lambda: pytest.fail("dead generation invoked follower factory"),
                materialize_device="cpu",
            )
        assert (
            "Stop workers and prevent new joins, then run "
            f"SharedOffloadSession.cleanup_namespace({namespace!r}, manager_names=[{_MANAGER_NAME!r}]) "
            "before retrying."
        ) in str(exc_info.value)
        assert str(exc_info.value).count("SharedOffloadSession.cleanup_namespace") == 1
        assert original(torch.tensor([[3.0, -1.0]])).detach().tolist() == _EXPECTED_OUTPUT
    _assert_segments_removed(creator.namespace_for(_MANAGER_NAME))

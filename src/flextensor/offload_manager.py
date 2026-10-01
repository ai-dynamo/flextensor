# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import logging
import os
import threading
import types
import warnings
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

if TYPE_CHECKING:
    from flextensor.collectors import LayerStatistics
import psutil
import torch
from proxytypes3 import ObjectWrapper
from torch import nn
from torch.utils.hooks import RemovableHandle  # noqa: TC002 — beartype on @decorated __init__ evaluates this at runtime

from flextensor.benchmark_tensor_mode import PreloadToDevice
from flextensor.compile import COMPILED_WARMUP_FORWARDS, CompiledOffload
from flextensor.compile.module_swap import resolve_offload_units, resolve_resident_units
from flextensor.compile.replan_status import StrategyReplanStatus, build_strategy_replan_status
from flextensor.compile.warmup_tail import CompiledOffloadTailState
from flextensor.compiled_offload import build_compiled_offload_forward
from flextensor.compiler_utils import disable as _compiler_disable
from flextensor.compiler_utils import (
    find_torch_compile_wrapper,
    is_torch_compiled_module,
)
from flextensor.config import OffloadConfig, resolve_load_strategy
from flextensor.helpers import NoOpTensorManager
from flextensor.instrumentation import dump_to_directory, get_registry
from flextensor.offload_timing import (
    OffloadTimingCollector,
    OffloadTimingReport,
    format_offload_timing_table,
)
from flextensor.shm.flexible_shm import FlexibleSharedMemory  # noqa: TC001 - runtime beartype annotation
from flextensor.state_handler import TensorManagerState  # noqa: TC001 - runtime beartype annotation
from flextensor.tensor_discovery import (
    has_offload_modules,
    has_patched_ancestor,
    is_offload_patched_module,
    select_offload_unit_paths,
)
from flextensor.types import GPUMemoryUsage  # noqa: TC001
from flextensor.utils import (
    get_class_matched_module_paths,
    get_module_paths,
    get_tensor_data,
    matches_any_class_pattern,
    matches_any_pattern,
    partition_patterns,
    set_tensor_data,
)

if TYPE_CHECKING:
    from flextensor.state_transition import StateTransitionPlan

LOGGER = logging.getLogger(__name__)
_GiB = 1 << 30


def _collect_matches(
    paths: list[str],
    remaining: set[str],
    matched: set[str],
    name_bodies: Mapping[str, str],
    *,
    recursive_star: bool,
) -> None:
    """Move name patterns from *remaining* to *matched* when they match any path.

    Class patterns are absent from *name_bodies* and skipped here.
    """
    for path in paths:
        for pattern in list(remaining):
            body = name_bodies.get(pattern)
            if body is None:
                continue
            if matches_any_pattern(path, [body], recursive_star=recursive_star):
                matched.add(pattern)
                remaining.discard(pattern)
        if not remaining:
            return


def _collect_class_matches(
    model: nn.Module,
    remaining: set[str],
    matched: set[str],
    class_bodies: Mapping[str, str],
) -> None:
    """Move ``class:`` patterns from *remaining* to *matched* by walking modules.

    Only patterns present in *class_bodies* are considered; non-class patterns
    are left untouched.
    """
    if not class_bodies or not remaining & class_bodies.keys():
        return
    for path, module in model.named_modules():
        if not path:
            continue
        cls = type(module)
        for pattern in list(remaining):
            body = class_bodies.get(pattern)
            if body is None:
                continue
            if matches_any_class_pattern(cls, [body]):
                matched.add(pattern)
                remaining.discard(pattern)
        if not remaining:
            return


def _find_matched_patterns(
    model: nn.Module,
    patterns: list[str],
    module_paths: list[str],
    *,
    recursive_star: bool = True,
    param_recursive_star: bool | None = None,
    include_parameters: bool = False,
) -> set[str]:
    """Return the subset of *patterns* that match at least one module or parameter.

    Accepts raw (optionally prefixed) patterns.  ``class:<body>`` patterns are
    matched against both the short class name and the fully-qualified class
    name (see :func:`flextensor.utils.matches_any_class_pattern`); name
    patterns behave as before.

    Args:
        model: Model to check paths against.
        patterns: Raw patterns to test (may carry ``class:`` / ``name:`` prefixes).
        module_paths: Non-root module paths (e.g. from ``named_modules()``).
        recursive_star: Passed through to ``matches_any_pattern`` for module paths.
        param_recursive_star: Passed through to ``matches_any_pattern`` for parameter
            paths.  When *None* (default), falls back to *recursive_star* so existing
            callers that don't distinguish module/param semantics keep working.
        include_parameters: If True, also check against ``named_parameters()`` paths
            so that parameter-level patterns (e.g. ``layers.*.weight``) are recognised.
    """
    if param_recursive_star is None:
        param_recursive_star = recursive_star

    partitioned = partition_patterns(patterns)
    remaining = set(patterns)
    matched: set[str] = set()

    _collect_matches(module_paths, remaining, matched, partitioned.name_bodies, recursive_star=recursive_star)

    if include_parameters and remaining & partitioned.name_bodies.keys():
        param_paths = [p for p, _ in model.named_parameters()]
        _collect_matches(param_paths, remaining, matched, partitioned.name_bodies, recursive_star=param_recursive_star)

    _collect_class_matches(model, remaining, matched, partitioned.class_bodies)

    return matched


def _log_diagnostic_snapshot(label: str, om: OffloadManager | None) -> None:
    """Capture and log full diagnostic context. Called only on error path.

    Every diagnostic collection is wrapped in its own try/except so that
    capture failures never mask the original exception.

    Args:
        label: Human-readable context for the error (e.g. method name, forward block name).
        om: The OffloadManager instance, or None if unavailable.
    """
    parts = [f"FlexTensor error in {label}"]

    if om is not None:
        parts.append(f"  phase={om._current_phase.value}")  # noqa: SLF001
        parts.append(f"  iteration={om._iteration_count}")  # noqa: SLF001
        parts.append(f"  manager={om.name!r}")

    try:
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = torch.cuda.get_device_properties(i).total_memory
            parts.append(
                f"  gpu{i}: alloc={alloc / _GiB:.2f}GiB reserved={reserved / _GiB:.2f}GiB total={total / _GiB:.2f}GiB"
            )
    except Exception:
        parts.append("  gpu: <unavailable>")

    try:
        vm = psutil.virtual_memory()
        parts.append(
            f"  host: used={vm.used / _GiB:.2f}GiB avail={vm.available / _GiB:.2f}GiB total={vm.total / _GiB:.2f}GiB"
        )
    except Exception:
        parts.append("  host: <unavailable>")

    parts.append(f"  pid={os.getpid()}")

    LOGGER.error("\n".join(parts), exc_info=True)


_F = TypeVar("_F", bound=Callable[..., Any])


def _error_boundary(func: _F) -> _F:
    """Decorator that logs full diagnostics on unhandled exceptions.

    Captures FlexTensor state, GPU/host memory, traceback, then re-raises.
    Zero overhead on the happy path (CPython try/except is free when no
    exception is raised).

    Only for OffloadManager methods — ``self`` must be an OffloadManager.

    Args:
        func: The OffloadManager method to wrap.

    Returns:
        Wrapped method with identical signature.
    """

    @functools.wraps(func)
    def wrapper(self: OffloadManager, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except Exception:
            _log_diagnostic_snapshot(func.__qualname__, self)
            raise

    return wrapper  # type: ignore[return-value]


@runtime_checkable
class _ShmCoordinatorLike(Protocol):
    """Structural interface for ShmCoordinator used by the follower init path."""

    namespace: str
    is_creator: bool

    def wait_for_ready(self, timeout: float | None = None) -> None: ...
    def read_profile(self) -> TensorManagerState: ...


@runtime_checkable
class TensorManagerProtocol(Protocol):
    """Structural interface for tensor managers used by ``OffloadManager``."""

    use_shm: bool
    load_from_shm: bool
    shm_namespace: str | None
    shm_owner: FlexibleSharedMemory | None
    stats: list[LayerStatistics]

    @property
    def replan_source_weights_available(self) -> bool: ...

    def set_model(self, model: nn.Module) -> None: ...
    def build_parameters_mapping(self, model: nn.Module | dict[str, torch.Tensor]) -> None: ...
    def initialize_warmup(self) -> nn.Module: ...
    def initialize_profile(self) -> nn.Module: ...
    def initialize_inference(self) -> nn.Module: ...
    def restore_state(
        self,
        model: nn.Module,
        state: TensorManagerState,
        *,
        load_from_shm: bool = False,
    ) -> None: ...
    def plan_state_adoption(self, model: nn.Module, state: TensorManagerState) -> StateTransitionPlan: ...
    def execute_state_adoption(self, model: nn.Module, plan: StateTransitionPlan) -> None: ...
    def restore_adopted_state(self, model: nn.Module, state: TensorManagerState) -> None: ...
    def prepare_infer_load_mode(self) -> None: ...
    def prepare_final_model(self, model: nn.Module, *, in_place: bool = False) -> nn.Module: ...
    def export_state(self) -> TensorManagerState: ...
    def save_profile(self, profile_directory: str) -> None: ...
    def load_profile(self, profile_directory: str, model: nn.Module) -> None: ...
    def get_gpu_memory_usage(self) -> GPUMemoryUsage: ...
    def get_memory_transfer_stats(self) -> dict[int, float] | None: ...
    def collect_offload_timing(self) -> OffloadTimingReport | None: ...
    def sync_prev_onload(self) -> None: ...
    def join_after_forward(self) -> None: ...
    def trap(self, name: str) -> Any: ...
    def is_profiling_suspended(self) -> bool: ...
    def clear_profiling_durations(self) -> None: ...
    def suspend_profiling(self) -> None: ...
    def resume_profiling(self) -> None: ...
    def pause_profiling(self) -> Any: ...
    def release_memory(self) -> None: ...
    def shutdown(self, *, preserve_model: bool = True) -> None: ...


class OffloadPhase(Enum):
    """Phase of the offload manager during tensor discovery, profiling, and inference."""

    NOT_INITIALIZED = "not_initialized"
    DISCOVERY = "discovery"
    PROFILING = "profiling"
    INFERENCE = "inference"


DEFAULT_MANAGER_NAME: str = "default"
"""Name used by :func:`get_offload_manager` when no explicit name is provided."""

# Global singleton map for OffloadManager instances
OFFLOAD_MANAGER_MAP = {}
_MANAGER_MAP_LOCK = threading.Lock()

# Config fields consumed by ``_initialize_tensor_manager`` when it constructs
# the TensorManager. That runs only once per manager, so a later ``set_config``
# cannot reapply any of them — ``set_config`` warns instead of silently
# accepting the change. ``skip_discovery`` is handled separately (it raises).
_TENSOR_MANAGER_ONESHOT_FIELDS: tuple[str, ...] = (
    "enabled",
    "enable_diagnostics",
    "exclude_patterns",
    "gpu_device",
    "include_patterns",
    "load_strategy",
    "max_gpu_mem_fraction",
    "min_blocks",
    "num_blocks",
    "piecewise_prefetch",
    "pinned_memory",
    "pinned_memory_mode",
    "profile_mode",
    "shm_enabled",
    "transfer_budget_scale",
    "transfer_mode",
    "offload_timing",
)


def _one_shot_value_differs(old: Any, new: Any) -> bool:
    """Compare two one-shot config values, tolerating value-less objects.

    ``Strategy`` implementations define no ``__eq__``, so ``!=`` falls back to
    identity. That makes ``config.model_copy(deep=True)`` — which clones the
    strategy — look like a user-initiated change and emit a spurious warning.
    Compare same-typed instances by their attributes instead, so a clone is
    equal while a genuinely different strategy still differs.

    Args:
        old: Value captured by the active ``TensorManager``.
        new: Value from the incoming config.

    Returns:
        ``True`` if the values represent a real change.
    """
    if old is new:
        return False
    if type(old) is type(new) and hasattr(old, "__dict__") and not isinstance(old, (str, bytes)):
        return bool(vars(old) != vars(new))
    return bool(old != new)


def _safe_remove_hook_handle(handle: RemovableHandle, *, context: str) -> None:
    """Remove a hook handle, logging at WARNING if removal raises.

    A handle becomes stale when its target module's ``_forward_hooks`` is
    mutated externally (model replacement, GC race, monkey-patching).  Letting
    the resulting ``KeyError`` / ``RuntimeError`` propagate would abort phase
    transitions and ``release()`` for what is at worst a leaked reference.
    """
    try:
        handle.remove()
    except Exception as exc:
        LOGGER.warning(
            "OffloadManager: state-update hook handle removal failed during %s (%s: %s); "
            "the previous hook may remain installed on its original module.",
            context,
            type(exc).__name__,
            exc,
        )


class OffloadModelProxy(ObjectWrapper):
    """Proxy that delegates to the current model version during offload phase transitions."""

    def __init__(self, model, offload_manager: OffloadManager):
        ObjectWrapper.__init__(self, model)
        # Store offload_manager directly on the proxy, bypassing ObjectWrapper delegation
        object.__setattr__(self, "offload_manager", offload_manager)

    def _get_model(self):
        """Helper method to get the current model, avoiding code duplication."""
        # Access offload_manager directly from proxy, bypassing ObjectWrapper delegation
        offload_manager = object.__getattribute__(self, "offload_manager")
        model = offload_manager.model
        if model is None:
            message = "Model not initialized. Ensure offload() has been called and the model hasn't been released."
            raise RuntimeError(message)
        return model

    def __call__(self, *args, **kwargs):
        """Invoke the underlying model; phase transitions fire via a forward hook."""
        return self._get_model()(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Invoke the underlying model; phase transitions fire via a forward hook."""
        return self._get_model()(*args, **kwargs)

    # Special methods that need explicit delegation (not handled by __getattr__)
    # These are required for proper proxy behavior:
    # - __iter__: iteration support
    # - __enter__, __exit__: context manager protocol
    # - __dir__: introspection support

    def __iter__(self):
        """Delegate __iter__ to underlying model if it supports it."""
        model = self._get_model()
        if not hasattr(model, "__iter__"):
            raise TypeError(f"'{type(model).__name__}' object is not iterable")
        return iter(model)

    def __enter__(self):
        """Delegate __enter__ to underlying model for context manager protocol."""
        model = self._get_model()
        if not hasattr(model, "__enter__"):
            raise AttributeError(f"'{type(model).__name__}' object has no attribute '__enter__'")
        return model.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Delegate __exit__ to underlying model for context manager protocol."""
        model = self._get_model()
        if not hasattr(model, "__exit__"):
            raise AttributeError(f"'{type(model).__name__}' object has no attribute '__exit__'")
        return model.__exit__(exc_type, exc_val, exc_tb)

    def __dir__(self):
        """Delegate __dir__ to underlying model to include model's attributes."""
        model = self._get_model()
        # Combine proxy's own attributes with model's attributes
        proxy_attrs = set(object.__dir__(self))
        model_attrs = set(dir(model))
        return sorted(proxy_attrs | model_attrs)


class OffloadManager:
    """Simplified offload manager API.

    Provides a high-level interface for weight offloading with automatic
    phase management, profiling, and module patching capabilities.
    """

    def __init__(self, name: str):
        self.name = name
        self._compiled = CompiledOffload(self)
        self.config = OffloadConfig()
        self._tensor_manager: TensorManagerProtocol | None = None
        self._initialized = False
        self._current_phase = OffloadPhase.NOT_INITIALIZED
        self._iteration_count = 0
        self._model: nn.Module | None = None
        self._model_proxy: OffloadModelProxy | None = None
        self._patched_modules: list[nn.Module] = []
        self._state_hook_handle: RemovableHandle | None = None
        # Whether the most recent ``_transition_to_warmup`` honored a
        # requested ``skip_discovery=True``. Set to ``False`` when the
        # warning-and-fallback path fires (no patched modules) so callers
        # have a programmatic signal — log scraping is unreliable in
        # production loggers that route stderr to JSON.
        # ``None`` until a warmup transition determines it; reset to ``None`` by
        # ``release()``. See the :attr:`skip_discovery_honored` property.
        self._skip_discovery_honored: bool | None = None
        # One-shot config values as captured by the live TensorManager; see
        # ``_warn_on_one_shot_field_changes``. Empty until one is constructed.
        self._tensor_manager_oneshot_snapshot: dict[str, Any] = {}
        # When True, the caller drives :meth:`update_state` after each CUDA-graph
        # replay (forward hooks do not run under ``graph.replay()``).
        self._manual_update_state = False
        self._state_takeover_active = False
        self._cleanup_blocked = False
        # Keep post-migration data alive when failed takeover cleanup cannot rebind its parameters.
        self._failed_state_takeover_parameter_data: list[tuple[nn.Parameter, torch.Tensor]] = []

        # Restores are serialized, so retain at most one failed candidate for cleanup retry.
        self._failed_restored_tensor_manager: Any | None = None

        # SharedOffloadSession keeps this reservation until cleanup ends.
        self._shared_offload_owner_token: object | None = None

    @property
    def compiled_offload_manager_id(self) -> int:
        """Stable id wired into compiled-offload custom ops for this manager."""
        return self._compiled.manager_id

    def set_config(self, config: OffloadConfig):
        """Set configuration for the offload manager.

        Most fields take effect on the next phase transition or on the
        operation that reads them. The fields in
        :data:`_TENSOR_MANAGER_ONESHOT_FIELDS` are baked into the active
        :class:`TensorManager` when it is constructed at the first
        ``offload()`` call — ``_initialize_tensor_manager`` does not run
        again — so changing them here cannot affect the live manager. To
        honor a new value, ``release()`` the current manager and
        re-``offload()``.

        ``skip_discovery`` is the one field that *raises* rather than warns:
        the live ``TensorManager`` would keep its captured value while
        ``self.config.skip_discovery`` reported the new one, and
        ``offload_block()``'s guard reads the live config — so a ``True→False``
        flip would silently permit manual blocks while the manager stays in
        skip-mode and never captured their tensor mappings. Every other
        one-shot field emits a ``UserWarning`` naming it.

        Args:
            config: OffloadConfig instance with settings

        Raises:
            RuntimeError: If ``skip_discovery`` differs from the value the
                active ``TensorManager`` captured at first ``offload()``.

        Warns:
            UserWarning: If any other one-shot field differs from the value
                the active ``TensorManager`` captured.
        """
        self._check_shared_offload_reservation()
        if self._tensor_manager is not None:
            captured = self._tensor_manager_oneshot_snapshot
            old_skip = captured.get("skip_discovery", getattr(self.config, "skip_discovery", None))
            new_skip = getattr(config, "skip_discovery", None)
            if old_skip != new_skip:
                raise RuntimeError(
                    f"OffloadManager.set_config: cannot change skip_discovery from "
                    f"{old_skip} to {new_skip} after the TensorManager has been "
                    f"created (during the first offload() call). skip_discovery is "
                    f"a one-shot flag; the manager captured its value at init and "
                    f"cannot pick up a change here. Silently allowing the change "
                    f"would let offload_block() calls execute under skip-mode "
                    f"internals that never captured their tensor mappings. To "
                    f"honor the new value, call release() and re-offload()."
                )
            self._warn_on_one_shot_field_changes(config)
        self.config = config

        # Enable/disable instrumentation based on config
        registry = get_registry()
        registry.enabled = config.enable_instrumentation

    def _check_shared_offload_reservation(self, owner_token: object | None = None) -> None:
        """Reject access that does not own an active shared session."""
        reservation = self._shared_offload_owner_token
        if reservation is not None and reservation is not owner_token:
            raise RuntimeError(
                "OffloadManager is reserved by an active SharedOffloadSession; "
                "close that session before reusing or releasing this manager."
            )

    def _prepare_shared_offload(self, config: OffloadConfig, *, owner_token: object) -> None:
        """Validate an idle manager and install a shared-session config.

        This check runs before role election and, consequently, before either
        application model factory.  It prevents an expensive creator factory
        from running only to discover that the named manager is already active.
        """
        if self._cleanup_blocked:
            raise RuntimeError("Manager cleanup is incomplete; retained resources cannot be reused safely.")
        self._check_shared_offload_reservation()
        if self._failed_restored_tensor_manager is not None:
            raise RuntimeError(
                "Cannot initialize shared offload while candidate cleanup is pending; "
                "call release() to retry cleanup first."
            )
        if self._tensor_manager is not None:
            raise RuntimeError("Cannot initialize shared offload with an existing TensorManager; call release() first.")
        if (
            self._model is not None
            or self._model_proxy is not None
            or self._current_phase is not OffloadPhase.NOT_INITIALIZED
            or self._state_hook_handle is not None
            or self._patched_modules
        ):
            raise RuntimeError(
                "Cannot initialize shared offload while an active offload runtime exists; call release() first."
            )
        self._validate_shm_config(config)
        self.set_config(config)
        self._require_shm_allocation_mode()
        self._shared_offload_owner_token = owner_token

    def _release_shared_offload(
        self,
        *,
        owner_token: object,
        preserve_model: bool,
    ) -> None:
        """Release an owned reservation, raising on absent or mismatched ownership."""
        if self._shared_offload_owner_token is None or self._shared_offload_owner_token is not owner_token:
            raise RuntimeError("Shared offload session no longer owns its manager reservation")
        self.release(
            preserve_model=preserve_model,
            _shared_offload_owner_token=owner_token,
        )
        self._shared_offload_owner_token = None

    def _warn_on_one_shot_field_changes(self, config: OffloadConfig) -> None:
        """Warn for one-shot fields that differ from the live manager's captured values.

        Args:
            config: The incoming config being applied.
        """
        # Fall back to the current config when no snapshot exists (a manager
        # installed without going through ``_initialize_tensor_manager``).
        # Treating a missing snapshot as "captured None" would report every
        # field as changed.
        captured = self._tensor_manager_oneshot_snapshot or {
            name: getattr(self.config, name, None) for name in _TENSOR_MANAGER_ONESHOT_FIELDS
        }
        changed = [
            name
            for name in _TENSOR_MANAGER_ONESHOT_FIELDS
            if _one_shot_value_differs(captured.get(name), getattr(config, name, None))
        ]
        if not changed:
            return
        message = (
            f"OffloadManager.set_config: {', '.join(sorted(changed))} "
            f"{'is' if len(changed) == 1 else 'are'} baked into the active TensorManager at the "
            f"first offload() call; the live manager keeps its original value(s) while self.config "
            f"reports the new one(s). Call release() and re-offload() to apply them."
        )
        # ``include_patterns``/``exclude_patterns`` are the exception: ``offload()``
        # re-runs ``_offload_modules``/``_exclude_modules`` from ``self.config``
        # after this call, so the *patching* honors the new values while the
        # TensorManager's captured copy (used for tensor mapping) does not.
        # Name that split rather than claiming they are ignored outright.
        pattern_fields = sorted(set(changed) & {"include_patterns", "exclude_patterns"})
        if pattern_fields:
            message += (
                f" Note: {', '.join(pattern_fields)} additionally take effect for module patching on the "
                f"next offload() while the TensorManager's captured copy does not, so the two can diverge."
            )
        warnings.warn(message, UserWarning, stacklevel=3)

    @property
    def compiled_offload_active(self) -> bool:
        """Whether this manager's offload run uses the compiled-offload path."""
        return self._compiled.active

    @property
    def compiled_replan_active(self) -> bool:
        """Whether to re-plan strategy from compiled per-trap timings."""
        return self._compiled.replan_active

    @property
    def compiled_profile_active(self) -> bool:
        """Whether view-mode profile runs under ``compile_fn`` (no replan tail)."""
        return self._compiled.profile_active

    @property
    def external_compile_active(self) -> bool:
        """Whether the active run uses ``external_compile`` with no ``compile_fn``.

        Describes the **effective live run** as resolved at ``offload()`` —
        mutating ``config.external_compile`` afterwards cannot flip it (that
        field is re-read only by the next ``offload()``). ``False`` for eager
        offload and for FlexTensor-driven ``compile_fn`` runs, which do not need
        the caller to compile each unit.
        """
        return self._compiled.active and self._compiled.compile_fn is None

    @property
    def strategy_replan_status(self) -> StrategyReplanStatus:
        """Passive snapshot of the current replan lifecycle.

        Reading this never arms or advances the tail, so it is safe both as a
        preflight check before :meth:`request_strategy_replan` and as an outcome
        check after the tail finishes. Field semantics are documented on
        :class:`~flextensor.StrategyReplanStatus`; for the preflight-then-drive
        pattern see ``docs/how-to/torch-compile.md``.

        ``can_request`` is required by both :meth:`request_strategy_replan`
        variants but sufficient only for the default one: the CUDA-graph variant
        (``manual_update_state=True``) adds requirements — ``offload_timing=
        'cuda_graph'`` plus external CUDA timing events — not reflected here.
        """
        return build_strategy_replan_status(
            self._compiled.tail_state,
            armed=self._compiled.tail_armed,
            replan_applied=self._compiled.replan_applied,
            can_request=self._can_request_strategy_replan(),
        )

    def _can_request_strategy_replan(self) -> bool:
        """Whether a default :meth:`request_strategy_replan` would arm a tail now."""
        if not (self._compiled.active and self._compiled.replan_active):
            return False
        return self._strategy_replan_unavailable_reason() is None

    def _strategy_replan_unavailable_reason(self) -> str | None:
        """Why a default replan request cannot arm now, or ``None`` when it can.

        Shared by :attr:`strategy_replan_status` and the guard in
        :meth:`request_strategy_replan` so the advertised ``can_request`` and
        the actual refusal cannot drift apart. Assumes compiled offload is
        active with replan armed; ``CompiledOffload.request_strategy_replan``
        reports those two flags itself.
        """
        # Arming installs compiled profiling on the inference loader, and the
        # tail only advances on INFERENCE forwards.
        if self._current_phase is not OffloadPhase.INFERENCE:
            return (
                f"the lifecycle is still in {self._current_phase.value}; drive "
                "om.iters_before_inference forwards to reach INFERENCE first"
            )
        if self._compiled.tail_state in (
            CompiledOffloadTailState.WARMING,
            CompiledOffloadTailState.MEASURING,
        ):
            return "a replan tail is already in flight; drive the forwards the request returned"
        if self._compiled.tail_state is CompiledOffloadTailState.FAILED:
            return "an earlier replan failed, so inference is already unsafe; restart the process"
        tm = self._tensor_manager
        if tm is None or not tm.replan_source_weights_available:
            return (
                "FlexTensor no longer retains the source weights a rebuild needs "
                "(they are consumed by a successful replan)"
            )
        return None

    @property
    def phase(self) -> OffloadPhase:
        """Current offload lifecycle phase."""
        return self._current_phase

    @property
    def eager_profiling_iters(self) -> int:
        """Profiling forwards required before the PROFILING→INFERENCE transition.

        Floored at ``1``. :meth:`update_state` runs as a post-forward hook, so
        the transition cannot fire until one profile forward has completed —
        the raw ``profiling_iters=0`` would advertise a budget that can never
        reach INFERENCE, and this property is driven directly as a loop bound
        (see :meth:`FlexTensorOffloadWorker.warmup_and_profile_model`).
        """
        return max(1, self._eager_profiling_iters())

    def _resolve_compiled_offload_activation(
        self,
        effective_config: OffloadConfig,
        compile_fn: Callable[[nn.Module], nn.Module] | None,
    ) -> bool:
        """Resolve compiled-offload flags from config or ``compile_fn``."""
        return self._compiled.resolve_activation(effective_config, compile_fn)

    def _resolve_profile_directory(self, profile_directory: str | None) -> str:
        """Resolve profile directory from argument or config.

        Args:
            profile_directory: Explicit directory, or None to use config.profile_storage_dir.

        Returns:
            Resolved profile directory path.

        Raises:
            ValueError: If no directory is provided and config.profile_storage_dir is not set.
        """
        resolved = profile_directory or self.config.profile_storage_dir
        if resolved is None:
            raise ValueError(
                "No profile directory specified. Provide profile_directory argument "
                "or set profile_storage_dir in OffloadConfig."
            )
        return resolved

    def save_profile(self, profile_directory: str | None = None) -> None:
        """Save current offload profile to directory.

        Args:
            profile_directory: Directory to save profile to.
                If None, uses config.profile_storage_dir.

        Raises:
            RuntimeError: If tensor manager is not initialized or no profile data exists.
            ValueError: If no directory is provided and config.profile_storage_dir is not set,
                or if config.profile_read_only is True.
        """
        if self.config.profile_read_only:
            raise ValueError(
                "Cannot save profile: config.profile_read_only is True. "
                "Set profile_read_only=False in OffloadConfig to enable saving."
            )

        if self._tensor_manager is None:
            raise RuntimeError("Tensor manager not initialized. Call offload() first.")

        resolved_dir = self._resolve_profile_directory(profile_directory)
        self._tensor_manager.save_profile(resolved_dir)

    def load_profile(self, profile_directory: str | None = None, model: nn.Module | None = None) -> None:
        """Load offload profile from directory and restore state to model.

        Args:
            profile_directory: Directory containing the profile.
                If None, uses config.profile_storage_dir.
            model: Model to restore state to. If None, uses the current model.

        Raises:
            RuntimeError: If no model is available to restore state to.
            FileNotFoundError: If profile directory or file doesn't exist.
            ValueError: If no directory is provided and config.profile_storage_dir is not set.
        """
        self._check_shared_offload_reservation()
        if self._tensor_manager is None:
            self._initialize_tensor_manager()
        # Keep a stable local so mypy can narrow the optional attribute after the guard.
        tensor_manager = self._tensor_manager

        resolved_dir = self._resolve_profile_directory(profile_directory)
        target_model = model if model is not None else self._model
        if target_model is None:
            raise RuntimeError("No model provided and no model has been offloaded yet.")
        if tensor_manager is None:
            raise RuntimeError("Tensor manager initialization failed")
        tensor_manager.load_profile(resolved_dir, target_model)

    def _require_shm_allocation_mode(self) -> None:
        """Require the only loader mode that supports shared restored blocks."""
        self._validate_shm_config(self.config)
        if not self.config.shm_enabled or self.config.transfer_mode != "allocation_block_transfer":
            raise ValueError(
                "SHM restored-state initialization requires shm_enabled=True and "
                "transfer_mode='allocation_block_transfer'"
            )

    def _validate_restored_shm_namespace(self, namespace: str) -> None:
        """Ensure restored blocks use the configured stable namespace."""
        self._require_shm_allocation_mode()
        if namespace != self.config.shm_namespace:
            raise ValueError(
                f"SHM namespace {namespace!r} does not match configured stable namespace {self.config.shm_namespace!r}"
            )

    def _export_runtime_state(self) -> TensorManagerState:
        """Export the current live state from the TensorManager runtime."""
        if self._tensor_manager is None:
            raise RuntimeError("Cannot export runtime state before TensorManager initialization")
        return self._tensor_manager.export_state()

    def benchmark_transfers(self):
        """Benchmark weight transfer speeds."""
        if self._tensor_manager is None:
            return
        # TODO: Implement transfer benchmarking

    def get_tensor_manager(self):
        """Get the internal tensor manager instance.

        Returns:
            TensorManager instance or None if not initialized
        """
        return self._tensor_manager

    def sync_prev_onload(self) -> None:
        if self._tensor_manager is not None:
            self._tensor_manager.sync_prev_onload()

    def join_after_forward(self) -> None:
        if self._tensor_manager is not None:
            self._tensor_manager.join_after_forward()

    def get_gpu_memory_usage(self) -> GPUMemoryUsage:
        """Get GPU memory usage by FlexTensor in inference mode.

        Returns the memory used by GPU transfer blocks and unmapped tensors
        that were moved to GPU. This method should be called after the manager
        has transitioned to inference mode.

        Returns:
            GPUMemoryUsage: Memory breakdown with per-component bytes and MB values.
                See `GPUMemoryUsage` for field details.

        Raises:
            RuntimeError: If called before the manager has transitioned to
                inference mode.

        Examples:
            >>> om = get_offload_manager()
            >>> config = OffloadConfig(include_patterns=["layers.*"])
            >>> model = om.offload(model, config=config)
            >>> # Run discovery and profiling iterations (path-aware count)
            >>> for _ in range(om.iters_before_inference):
            ...     model(input)
            >>> # Now in inference mode, get memory usage
            >>> usage = om.get_gpu_memory_usage()
            >>> print(f"FlexTensor GPU memory: {usage.total_mb:.2f} MB")
            >>> print(f"  Transfer blocks: {usage.blocks_mb:.2f} MB")
            >>> print(f"  Unmapped tensors: {usage.unmapped_tensors_mb:.2f} MB")
        """
        if self._current_phase != OffloadPhase.INFERENCE:
            msg = (
                f"Cannot get GPU memory usage in phase {self._current_phase.value}. "
                "This method is only available after transitioning to inference mode."
            )
            raise RuntimeError(msg)

        if self._tensor_manager is None:
            msg = "Tensor manager not initialized"
            raise RuntimeError(msg)

        return self._tensor_manager.get_gpu_memory_usage()

    def reset_offload_timing(self) -> None:
        """Start a fresh durable offload-timing measure window.

        Clears accumulated measure passes and the collector's rolling log
        window. Drops any pending **eager** pass on the collector so it cannot
        be published into this new window by a later ``on_pass_start``. Call
        before a measure segment you care about (serving → shutdown
        :meth:`collect_offload_timing`).
        """
        if self._tensor_manager is None:
            return
        if getattr(self.config, "offload_timing", "off") == "off":
            return
        tm = self._tensor_manager
        if hasattr(tm, "_clear_offload_timing_measure"):
            tm._clear_offload_timing_measure()  # noqa: SLF001
        loader = getattr(tm, "tensor_layer_loader", None)
        collector = getattr(loader, "offload_timing_collector", None) if loader is not None else None
        if collector is None or not getattr(collector, "enabled", False):
            return
        if hasattr(collector, "reset"):
            collector.reset()

    def _arm_offload_timing_after_capture(self) -> None:
        """Arm the timing collector to publish after captured/replayed execution."""
        collector = self._offload_timing_collector()
        if collector is None or not getattr(collector, "enabled", False):
            return
        if hasattr(collector, "arm_replay_measure"):
            collector.arm_replay_measure()

    def _offload_timing_collector(self) -> OffloadTimingCollector | None:
        """Return the active loader's offload-timing collector, if any."""
        if self._tensor_manager is None:
            return None
        loader = getattr(self._tensor_manager, "tensor_layer_loader", None)
        return getattr(loader, "offload_timing_collector", None) if loader is not None else None

    def _has_offload_timing_measure(self) -> bool:
        """True when the durable measure store has at least one pass (integrations)."""
        if self._tensor_manager is None:
            return False
        if getattr(self.config, "offload_timing", "off") == "off":
            return False
        # Best-effort: publish a pending replay or eager pass before inspecting.
        self.update_offload_timing()
        collector = self._offload_timing_collector()
        if collector is not None and hasattr(collector, "flush_pending_eager_pass"):
            collector.flush_pending_eager_pass()
        measure = getattr(self._tensor_manager, "_offload_timing_measure", None)
        return bool(measure)

    def collect_offload_timing(self) -> OffloadTimingReport | None:
        """Collect aggregate offload timing from the durable measure store.

        Flushes any pending eager pass before draining (see
        :meth:`TensorManager.collect_offload_timing`). Under CUDA-graph replay,
        call :meth:`update_offload_timing` (or :meth:`update_state` during a
        manual replan) after each replay so passes are published before
        collect — this method does not re-finalize the last replay (that would
        duplicate it).

        Must be called **after** all forward passes have completed (e.g. after
        ``pipe()`` returns or at shutdown). Calling this drains the durable
        measure store, so the next call returns data only for passes since
        :meth:`reset_offload_timing` / this collect. Retention before drain is
        capped by
        :data:`~flextensor.offload_timing.OFFLOAD_TIMING_MEASURE_MAX_PASSES`.

        Returns:
            :class:`~flextensor.offload_timing.OffloadTimingReport` with
            per-trap min/max/avg/std for transfer, compute, and wait, or
            ``None`` when:

            * :attr:`OffloadConfig.offload_timing` is ``"off"``,
            * the :class:`TensorManager` is not initialized yet, or
            * the durable measure store is empty after flush (no passes in
              the current window).
        """
        if self._tensor_manager is None:
            return None
        return self._tensor_manager.collect_offload_timing()

    @property
    def model(self) -> nn.Module | None:
        """Get the current active model.

        Returns:
            The current model instance or None if not initialized
        """
        return self._model

    def _initialize_tensor_manager(self):
        """Initialize TensorManager with config settings."""
        # Import here to avoid circular dependency between offload_manager and tensor_manager modules
        from flextensor.tensor_manager import TensorManager

        self._validate_shm_config(self.config)

        device_gpu = torch.device(f"cuda:{self.config.gpu_device}")

        tensor_manager_load_strategy = resolve_load_strategy(self.config)

        if not self.config.enabled:
            benchmark_cls = PreloadToDevice
            # pyright: ignore[reportUnknownReturnType]
            self._tensor_manager = NoOpTensorManager(device_gpu, benchmark_cls=benchmark_cls)
        else:
            self._tensor_manager = TensorManager(
                device_gpu,
                tensor_manager_load_strategy,
                pinned_memory=self.config.pinned_memory,
                pinned_memory_mode=self.config.pinned_memory_mode,
                loader_type=self.config.transfer_mode,
                blocks=self.config.num_blocks,
                remove_layers_operations=[],
                include_patterns=self.config.include_patterns,
                exclude_patterns=self.config.exclude_patterns,
                use_shm=self.config.shm_enabled,
                enable_diagnostics=self.config.enable_diagnostics,
                max_gpu_mem_fraction=self.config.max_gpu_mem_fraction,
                profile_mode=self.config.profile_mode,
                _offload_timing=self.config.offload_timing,
                _piecewise_prefetch=self.config.piecewise_prefetch,
            )
            self._tensor_manager.set_skip_discovery(self.config.skip_discovery)

        # Snapshot exactly what the manager captured. ``set_config`` diffs
        # against this rather than against ``self.config``, which it overwrites
        # on every call — otherwise re-applying an already-diverged config
        # would compare equal and warn only once.
        self._tensor_manager_oneshot_snapshot = {
            name: getattr(self.config, name, None) for name in (*_TENSOR_MANAGER_ONESHOT_FIELDS, "skip_discovery")
        }
        self._tensor_manager.use_shm = self.config.shm_enabled
        self._tensor_manager.shm_namespace = self.config.shm_namespace

    def _transfer_hooks(self, old_model: nn.Module | None, new_model: nn.Module):  # noqa: C901
        """Transfer hooks from old model to new model, preserving module hierarchy.

        Args:
            old_model: Model to extract hooks from (None if no previous model)
            new_model: Model to register hooks on
        """
        # Skip if old model is None or models are the same object (no transfer needed)
        if old_model is None or old_model is new_model:
            return

        # Get module mappings by name
        old_modules = dict(old_model.named_modules())
        new_modules = dict(new_model.named_modules())

        for module_name, old_module in old_modules.items():
            if module_name not in new_modules:
                continue

            new_module = new_modules[module_name]

            # Transfer forward hooks (create list copy to avoid mutation during iteration).
            # Skip our internal state-update hook: the manager re-installs it explicitly
            # on each phase transition so ``self._state_hook_handle`` tracks the live
            # hook and ``release()`` can remove it correctly.
            # Preserve ``with_kwargs`` / ``always_called`` flags — re-registering without
            # them changes the hook signature (e.g. drops kwargs) and breaks callers.
            forward_hooks_with_kwargs = getattr(old_module, "_forward_hooks_with_kwargs", {})
            forward_hooks_always_called = getattr(old_module, "_forward_hooks_always_called", {})
            # Model copies may already share hook registries. Re-registering
            # duplicates callbacks and leaves entries the original handle cannot remove.
            forward_hooks = getattr(old_module, "_forward_hooks", None)
            if forward_hooks and forward_hooks is not getattr(new_module, "_forward_hooks", None):
                for hook_id, hook_fn in list(forward_hooks.items()):
                    if getattr(hook_fn, "_ft_state_update_hook", False):
                        continue
                    new_module.register_forward_hook(
                        hook_fn,
                        with_kwargs=bool(forward_hooks_with_kwargs.get(hook_id, False)),
                        always_call=bool(forward_hooks_always_called.get(hook_id, False)),
                    )

            forward_pre_hooks_with_kwargs = getattr(old_module, "_forward_pre_hooks_with_kwargs", {})
            forward_pre_hooks = getattr(old_module, "_forward_pre_hooks", None)
            if forward_pre_hooks and forward_pre_hooks is not getattr(new_module, "_forward_pre_hooks", None):
                for hook_id, hook_fn in list(forward_pre_hooks.items()):
                    new_module.register_forward_pre_hook(
                        hook_fn,
                        with_kwargs=bool(forward_pre_hooks_with_kwargs.get(hook_id, False)),
                    )

            # Transfer backward hooks (create list copy to avoid mutation during iteration)
            backward_hooks = getattr(old_module, "_backward_hooks", None)
            if backward_hooks:
                if backward_hooks is getattr(new_module, "_backward_hooks", None):
                    # Late registration updates the shared registry but only one module's mode.
                    if new_module._is_full_backward_hook is None:  # noqa: SLF001
                        new_module._is_full_backward_hook = old_module._is_full_backward_hook  # noqa: SLF001
                else:
                    for _, hook_fn in list(backward_hooks.items()):
                        new_module.register_full_backward_hook(hook_fn)

            # Transfer backward pre-hooks (create list copy to avoid mutation during iteration)
            backward_pre_hooks = getattr(old_module, "_backward_pre_hooks", None)
            if backward_pre_hooks and backward_pre_hooks is not getattr(new_module, "_backward_pre_hooks", None):
                for _, hook_fn in list(backward_pre_hooks.items()):
                    if hasattr(new_module, "register_full_backward_pre_hook"):
                        new_module.register_full_backward_pre_hook(hook_fn)

    def _patch_module_forward(self, module: nn.Module, offload_name: str) -> None:
        """Patch module's forward method to include offload context.

        Args:
            module: The module to patch
            offload_name: Name for the offload block

        Note:
            Stores the original forward class method as module._ft_original_forward_func for restoration.
            Uses the UNBOUND forward function from the class to avoid bound method issues when
            models are copied during state transitions.
        """

        # Skip if already patched
        if hasattr(module, "_ft_original_forward_func"):
            return

        # Get the UNBOUND forward function from the class, not a bound method.
        # This is critical: bound methods capture `self`, which breaks when models are copied.
        original_forward_func = type(module).forward
        offload_manager = self

        def patched_forward(self_module, *args, **kwargs):
            try:
                # Bypass the public ``offload_block()`` so the user-facing
                # ``skip_discovery`` guard does not block auto-trap usage, but
                # keep its null-manager check: a patched forward can outlive
                # ``release()`` (which clears ``_tensor_manager``), and an
                # AttributeError on NoneType would be far less legible.
                tensor_manager = offload_manager._tensor_manager  # noqa: SLF001
                if tensor_manager is None:
                    raise RuntimeError(
                        f"Tensor manager not initialized while running the patched forward for "
                        f"{offload_name!r}. The module's forward is still patched after release(); "
                        f"call offload() again, or restore the original forward."
                    )
                with tensor_manager.trap(offload_name):
                    return original_forward_func(self_module, *args, **kwargs)
            except Exception:
                _log_diagnostic_snapshot(f"forward({offload_name})", offload_manager)
                raise

        # Make patched_forward look like the original for introspection
        patched_forward = functools.wraps(original_forward_func)(patched_forward)

        # Discovery / profiling / non-compiled inference keep this forward
        # outside the compiled graph so the offload trap and loader run eagerly.
        # Compiled inference swaps to ``_ft_compiled_offload_forward`` later.
        patched_forward = _compiler_disable(patched_forward)

        # Store original for restoration (using the UNBOUND function, not bound method)
        module._ft_original_forward_func = original_forward_func  # noqa: SLF001
        module._ft_offload_name = offload_name  # noqa: SLF001

        if self._compiled.active:
            module._ft_compiled_offload_forward = build_compiled_offload_forward(  # noqa: SLF001
                original_forward_func,
                offload_name,
            )

        # Bind patched_forward as a method to the module.
        # This ensures it receives `self` when called.
        module.forward = types.MethodType(patched_forward, module)  # type: ignore[method-assign]

        # Track patched module for cleanup
        self._patched_modules.append(module)
        if self._compiled.active:
            # Stamped for multi-manager dispatch; the stable offload-unit name is
            # closed over by ``_ft_compiled_offload_forward`` (not an index).
            module._ft_manager_id = self._compiled.manager_id  # noqa: SLF001

    def _restore_module_forward(self, module: nn.Module) -> None:
        """Restore module's original forward method.

        Operates strictly on ``module.__dict__`` so that we don't trip over
        custom ``__getattr__`` implementations (e.g. vLLM's ``StageMissingLayer``,
        which is the reason ``tensor_discovery.is_offload_patched_module`` exists)
        and so that cleanup never aborts mid-loop on a half-patched module.

        Args:
            module: The module to restore
        """
        if "_ft_original_forward_func" not in module.__dict__:
            return
        module.__dict__.pop("forward", None)
        module.__dict__.pop("_ft_original_forward_func", None)
        module.__dict__.pop("_ft_offload_name", None)
        module.__dict__.pop("_ft_compiled_offload_forward", None)
        module.__dict__.pop("_ft_manager_id", None)

    def get_layer_label_by_idx(self) -> list[str]:
        """Return ``_ft_offload_name`` of each patched module, in patch order.

        Used for diagnostics / logging (e.g. loader install). Custom ops take
        the offload-unit name directly (no index registry). Prefer
        :meth:`offload_units` for the user-facing ``(name, module)`` listing.
        """
        return [
            module._ft_offload_name  # noqa: SLF001
            for module in self._patched_modules
            if hasattr(module, "_ft_offload_name")
        ]

    def offload_units(self) -> tuple[tuple[str, nn.Module], ...]:
        """Return outermost offloaded modules as ``(qualified_name, module)`` pairs.

        Same set ``compile_fn`` wraps: nested patched modules are omitted so
        each graph would read one rolling slot. Order matches patch order;
        shared modules are identity-deduplicated (first path wins). Available
        after :meth:`offload`; the set is stable once the manager is in
        INFERENCE.

        To ``torch.compile`` these units, enable ``external_compile=True`` (or
        pass ``compile_fn``) and compile only after INFERENCE. Prefer
        ``module.compile(...)`` so the object in the tree stays the same;
        ``torch.compile(module)`` returns a new wrapper and is not written
        back until a later replace helper exists.

        Returns:
            Immutable ``(name, module)`` pairs. Empty when no model is bound
            or nothing was patched.
        """
        model = self._model
        if model is None:
            return ()
        return tuple(resolve_offload_units(model, self._patched_modules))

    def resident_units(self) -> tuple[tuple[str, nn.Module], ...]:
        """Return outermost modules that are not offload units and contain none.

        These subtrees stay GPU-resident at serving time and may be compiled
        without compiled-offload custom ops. The root, containers that hold
        offload units, registry containers without an executable ``forward``
        (``ModuleList``, ``ModuleDict``), and children inside an offload unit
        are omitted.

        Compile only after INFERENCE: earlier phases can swap the live module
        tree. Resident-only compile does not require ``external_compile`` or
        :meth:`request_strategy_replan`.

        Boundaries are the complement of :meth:`offload_units` live identities,
        so a ``compile_fn`` replacement that dropped patch markers is still
        omitted, and a second parent of a shared offload unit is not listed as
        resident.

        Returns:
            Immutable ``(name, module)`` pairs in ``named_children`` walk
            order, identity-deduplicated. Empty when no model is bound.
        """
        model = self._model
        if model is None:
            return ()
        return tuple(resolve_resident_units(model, self._patched_modules))

    def _install_compiled_forwards(self) -> None:
        """Swap each patched module's ``forward`` to the compile-transparent variant.

        Called at the INFERENCE transition when compiled-offload is enabled
        (``FT_EXTERNAL_COMPILE=1``). Discovery and profiling keep the default
        eager ``patched_forward`` (trap + loader outside the compiled graph).
        At inference, bind the pre-built ``_ft_compiled_offload_forward`` so
        ``pre_compute`` / ``post_compute`` mark residency and the unit body can
        compile as one subgraph. No-op unless compiled-offload is enabled.
        """
        if not self._compiled.active:
            return
        for module in self._patched_modules:
            module._ft_manager_id = self._compiled.manager_id  # noqa: SLF001
        swapped = 0
        for module in self._patched_modules:
            compiled_forward = getattr(module, "_ft_compiled_offload_forward", None)
            if compiled_forward is None:
                continue
            module.forward = types.MethodType(compiled_forward, module)  # type: ignore[method-assign]
            swapped += 1
        LOGGER.info(
            "FlexTensor compiled-offload: installed compile-transparent forwards on %d/%d patched modules",
            swapped,
            len(self._patched_modules),
        )

    def request_strategy_replan(self, *, manual_update_state: bool = False) -> int:
        """Remeasure under the current runtime and rebuild the offload strategy.

        * ``manual_update_state=True`` — CUDA graphs: run ``graph.replay()`` then
          :meth:`update_state` for the returned count (recapture graphs after).
        * ``manual_update_state=False`` — external compile: run that many model
          forwards; the forward hook calls :meth:`update_state` for you.

        Needed after external ``torch.compile`` (``external_compile=True``),
        after ``compile_fn`` with ``profile_mode='getter'``, or after CUDA-graph
        capture. Not required for default ``compile_fn`` + ``profile_mode='view'``.

        Args:
            manual_update_state: When True, caller drives each measure step
                (e.g. ``graph.replay()`` + :meth:`update_state`).

        Returns:
            Measure (and warmup) forwards / replays to drive, or ``0`` when
            nothing was armed. Both paths arm only while
            :attr:`strategy_replan_status` reports ``can_request``; otherwise
            they warn and leave the current tail untouched.
        """
        if manual_update_state:
            return self._begin_manual_update_replan()
        if self._compiled.active and self._compiled.replan_active:
            # Flags are on, so ``CompiledOffload`` would arm; check the runtime
            # preconditions it cannot see before it mutates the tail.
            reason = self._strategy_replan_unavailable_reason()
            if reason is not None:
                LOGGER.warning("FlexTensor: request_strategy_replan() ignored — %s.", reason)
                return 0
        return self._compiled.request_strategy_replan()

    def _begin_manual_update_replan(self) -> int:
        """Start caller-driven measure→replan; return iters for the caller's loop.

        Used when forward hooks do not run (today: CUDA-graph replay). Caller
        drives each step then :meth:`update_state`.
        """
        if getattr(self.config, "offload_timing", "off") != "cuda_graph":
            LOGGER.warning(
                "FlexTensor: request_strategy_replan(manual_update_state=True) needs "
                "offload_timing='cuda_graph'; not arming (compiled measure tail "
                "cannot run under graph.replay() without external CUDA timing events)."
            )
            return 0
        if not (self._compiled.active and self._compiled.replan_active):
            LOGGER.warning(
                "FlexTensor: request_strategy_replan(manual_update_state=True) ignored — "
                "compiled replan was not armed (use external_compile=True so source "
                "weights survive the first loader build)."
            )
            return 0

        # Gate on resolved collector capability *before* reset/arm so a soft
        # fallback refuse does not wipe the durable measure store.
        collector = self._offload_timing_collector()
        if collector is None or not getattr(collector, "enabled", False):
            LOGGER.warning(
                "FlexTensor: request_strategy_replan(manual_update_state=True) ignored — "
                "no enabled offload-timing collector on the inference loader."
            )
            return 0
        if not getattr(collector, "_external_events", False):
            LOGGER.warning(
                "FlexTensor: request_strategy_replan(manual_update_state=True) needs "
                "external CUDA timing events on the collector; this build fell back "
                "to internal events (or offload_timing was not 'cuda_graph'). Not arming."
            )
            return 0

        # Same runtime preconditions as the default path, checked before any
        # side effect below: re-arming would reset an in-flight measurement or
        # drop a recorded failure.
        reason = self._strategy_replan_unavailable_reason()
        if reason is not None:
            LOGGER.warning(
                "FlexTensor: request_strategy_replan(manual_update_state=True) ignored — %s.",
                reason,
            )
            return 0

        # Fresh durable window + enable replay timing readback for captured events.
        self.reset_offload_timing()
        self._arm_offload_timing_after_capture()

        self._manual_update_state = True
        # Graph is already captured — skip compile warmup; measure only.
        # Do not enable custom-op profiling; budgets come from offload timing.
        remaining = self._compiled.arm_replan_tail(
            compiled_warm_forwards=COMPILED_WARMUP_FORWARDS,
            enable_profiling=False,
            finish_replan=self._finish_manual_update_replan,
        )
        if remaining == 0:
            # profiling_iters=0 finished inside arm; flag already cleared by finish.
            return 0
        LOGGER.info(
            "FlexTensor: armed CUDA-graph measure replan "
            "(%d replay + update_state() call(s); recapture graphs after rebuild).",
            remaining,
        )
        return remaining

    def update_offload_timing(self, *, replay_generation: int = -1) -> bool:
        """Store the current offload-timing pass into the collector.

        Measurement only — does not advance phase or rebuild strategy.
        Needed under CUDA graphs: each ``graph.replay()`` overwrites the same
        captured event handles, so call this after every replay you want to
        keep (or :meth:`update_state` does it when replan used
        ``manual_update_state=True``).

        Returns:
            ``True`` when a pass was published or same-generation deduped.
            ``False`` when finalize raised or unexpectedly no-op'd (inactive
            pass / capturing) — callers must not advance a measure replan
            counter after ``False``.
        """
        if self._tensor_manager is None:
            return False
        if getattr(self.config, "offload_timing", "off") == "off":
            return True  # timing off: treat as no-op success
        collector = self._offload_timing_collector()
        if collector is None or not getattr(collector, "enabled", False):
            return False
        if not hasattr(collector, "finalize_replay_pass"):
            return False
        try:
            published = collector.finalize_replay_pass(replay_generation=replay_generation)
        except Exception:
            LOGGER.warning(
                "FlexTensor: update_offload_timing failed",
                exc_info=True,
            )
            return False
        # ``None`` from older mocks/stubs: treat as success for compatibility.
        if published is False:
            LOGGER.warning(
                "FlexTensor: update_offload_timing did not publish a pass "
                "(collector inactive or stream capturing); not advancing measure."
            )
            return False
        return True

    def reset_offload_timing_sampling(self) -> None:
        """Reset durable timings and close the integration-owned sample window."""
        self.reset_offload_timing()
        collector = self._offload_timing_collector()
        if collector is not None:
            collector.set_measure_window(False)

    def begin_offload_timing_sample(self) -> None:
        """Open the durable sink for one selected outer inference call."""
        collector = self._offload_timing_collector()
        if collector is not None:
            collector.set_measure_window(True)

    def finish_offload_timing_sample(self, *, replay_generation: int | None) -> bool:
        """Publish one selected eager or captured pass, then close its sink."""
        collector = self._offload_timing_collector()
        if collector is None:
            return False
        try:
            if replay_generation is None:
                return collector.flush_pending_eager_pass(preserve_replay_state=True)
            collector.arm_replay_measure()
            return collector.finalize_replay_pass(replay_generation=replay_generation)
        finally:
            collector.set_measure_window(False)

    def cancel_offload_timing_sample(self) -> None:
        """Close a selected sample window without publishing it."""
        collector = self._offload_timing_collector()
        if collector is not None:
            collector.set_measure_window(False)

    def _fail_manual_update_replan(self, exc: BaseException) -> None:
        """Abort an in-flight CUDA-graph measure replan (strategy unchanged)."""
        self._manual_update_state = False
        self._disarm_offload_timing_after_measure()
        self._compiled._tail.mark_failed(exc)  # noqa: SLF001
        raise exc

    def _disarm_offload_timing_after_measure(self) -> None:
        """Drop pending-pass flags so the next eager enter does not re-sink."""
        collector = self._offload_timing_collector()
        if collector is None or not getattr(collector, "enabled", False):
            return
        if hasattr(collector, "disarm_replay_measure"):
            collector.disarm_replay_measure()

    def _finish_manual_update_replan(self) -> bool:
        """Drain measure and rebuild from offload-timing budgets.

        Returns:
            ``True`` when a new strategy was applied, ``False`` when keeping
            the current loader (empty measure / rebuild refused).
        """
        self._manual_update_state = False
        # Always disarm before drain/rebuild so a following eager on_pass_start
        # cannot re-publish the last replay into the next durable window.
        self._disarm_offload_timing_after_measure()
        report = None
        tm = self._tensor_manager
        if tm is not None and hasattr(tm, "_drain_offload_timing_measure"):
            report = tm._drain_offload_timing_measure()  # noqa: SLF001
        if report is None or getattr(report, "num_passes", 0) <= 0 or not getattr(report, "per_trap", None):
            LOGGER.warning(
                "FlexTensor: CUDA-graph measure replan finished with no offload-timing "
                "passes; keeping the current strategy. Ensure "
                "offload_timing='cuda_graph' and call update_state() after "
                "each graph.replay()."
            )
            return False
        if not self._finish_replan_from_offload_timing(report):
            LOGGER.warning(
                "FlexTensor: CUDA-graph measure replan did not apply a new strategy; keeping the current loader."
            )
            return False
        return True

    def _finish_replan_from_offload_timing(self, report: OffloadTimingReport) -> bool:
        """Map one offload-timing report → **compute** budgets and finish replan.

        Only the compute/hiding-window column is applied. Runtime ``transfer_ms``
        / ``wait_ms`` are diagnostic; transfer costs stay on the profiled
        size→time curve. See
        :meth:`~flextensor.offload_timing.OffloadTimingReport.compute_budgets_by_profile_label`.
        """
        if self._tensor_manager is None:
            return False
        profile_labels = [stat.label for stat in self._tensor_manager.stats]
        durations_by_label = report.compute_budgets_by_profile_label(
            profile_labels,
            conservative=True,
        )
        if not durations_by_label:
            LOGGER.warning("FlexTensor: no compute budgets in offload-timing report; keeping the current strategy.")
            return False
        LOGGER.info(
            "FlexTensor: applying %d per-trap compute budget(s) from offload timing "
            "(initial profile used median eager compute only).\n%s",
            len(durations_by_label),
            format_offload_timing_table(report),
        )
        return self._compiled.finish_replan(durations_by_label)

    def _warn_if_compile_wrapped(self, target_phase: str) -> None:
        """Warn the user when a phase transition fires while ``OffloadModelProxy``
        is wrapped by ``torch.compile``.

        ``OptimizedModule`` binds to the proxy at construction time and caches
        the wrapped reference; a transition that follows silently invalidates
        the cached graph.  See ``docs/how-to/torch-compile.md`` for supported
        flows.
        """
        if self._model_proxy is None:
            return
        wrapper = find_torch_compile_wrapper(self._model_proxy)
        if wrapper is None:
            return
        warnings.warn(
            f"OffloadManager: phase transition to {target_phase} fired while the offload "
            "proxy is wrapped by torch.compile. The compiled graph captured a stale model "
            "reference and may produce incorrect results on subsequent calls. See "
            "docs/how-to/torch-compile.md for supported flows; either complete the eager "
            "lifecycle before torch.compile(proxy), or rebuild the OptimizedModule after "
            "the final transition.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _swap_to_new_model(self, new_model: nn.Module, target_phase: OffloadPhase) -> None:
        """Atomically swap ``self._model`` to ``new_model`` and advance to
        ``target_phase``.

        Runs the side-effecting steps (transfer hooks, install state hook,
        rebind proxy, update phase) and rolls back to the prior consistent
        snapshot if any step raises, so the manager never observes a
        half-transitioned state.
        """
        old_model = self._model
        saved_handle = self._state_hook_handle
        saved_phase = self._current_phase
        saved_iter = self._iteration_count

        self._model = new_model
        try:
            self._transfer_hooks(old_model, new_model)
            self._install_state_update_hook()
            if self._model_proxy is not None:
                self._model_proxy.__subject__ = new_model
            self._current_phase = target_phase
            self._iteration_count = 0
        except Exception:
            self._model = old_model
            self._state_hook_handle = saved_handle
            self._current_phase = saved_phase
            self._iteration_count = saved_iter
            if self._model_proxy is not None:
                self._model_proxy.__subject__ = old_model
            LOGGER.exception(
                "OffloadManager: transition to %s failed; rolled back to %s.",
                target_phase.name,
                saved_phase.name,
            )
            raise

    @_error_boundary
    def _transition_to_warmup(self):
        """Transition to discovery phase.

        Method name kept to align with ``TensorManager.initialize_warmup()``.

        If ``skip_discovery`` is enabled and at least one patched module is
        reachable from ``self._model`` (`has_offload_modules`), immediately
        transitions to profiling.

        If ``skip_discovery=True`` but no patched modules are reachable, logs a
        ``WARNING``, flips :attr:`skip_discovery_honored` to ``False`` so
        callers have a programmatic signal, and falls through to the discovery
        phase. When ``skip_discovery=False``, proceeds to discovery normally
        with no warning.
        """
        # Keep a stable local so mypy can narrow the optional attribute after the guard.
        tensor_manager = self._tensor_manager
        if tensor_manager is None or self._model is None:
            raise RuntimeError("OffloadManager is not initialized")
        tensor_manager.set_model(self._model)
        new_model = tensor_manager.initialize_warmup()
        # Only skip discovery/profiling when a profile was restored from
        # *outside* this manager. Testing ``tensor_manager_state`` is not
        # enough: a normal cycle reaching INFERENCE stores one too, so a second
        # ``offload()`` would replay the previous model's plan and tensor IDs.
        #
        # ``is True`` rather than truthiness — a MagicMock attribute is truthy
        # and must not trigger this path (the same reason the previous
        # ``isinstance`` check existed).
        if getattr(tensor_manager, "state_restored_from_profile", False) is True:
            # Profile was loaded via :func:`load_profile` / :func:`offload_from_profile`.
            # ``initialize_warmup`` already wired the inference loader; skip
            # discovery/profiling (compiled view-profile would arm
            # ``ProfileBlockController`` forwards on the wrong loader).
            #
            # No discovery phase runs here, so the skip is vacuously honored.
            # Assign explicitly: a stale ``False`` carried over from a previous
            # cycle would let ``offload_block()`` past its guard on a path that
            # never captured manual-block tensor mappings.
            self._skip_discovery_honored = True
            self._transition_to_inference()
            return
        self._swap_to_new_model(new_model, OffloadPhase.DISCOVERY)

        if self.config.skip_discovery:
            # Use ``has_offload_modules(self._model)`` rather than the captured
            # ``self._patched_modules`` list so this gate matches the one used
            # by ``TensorManager.initialize_warmup`` — the two signals can
            # desync after model swaps / rebuilds.
            if has_offload_modules(self._model):
                self._skip_discovery_honored = True
                self._transition_to_profile()
            else:
                self._skip_discovery_honored = False
                LOGGER.warning(
                    "skip_discovery=True but no patched modules are reachable "
                    "from the active model; falling back to the discovery phase. "
                    "Check include_patterns or move offload() ahead of any wrapper "
                    "that hides matching modules. Inspect "
                    "``OffloadManager.skip_discovery_honored`` to detect this case "
                    "programmatically."
                )
        else:
            self._skip_discovery_honored = True

    @property
    def skip_discovery_honored(self) -> bool | None:
        """Whether the most recent warmup honored the requested ``skip_discovery``.

        Three states, so "not determined yet" is distinguishable from "the skip
        fired" — conflating them made the signal easy to misread:

        - ``None`` — undetermined. No warmup transition has run: before the
          first ``offload()``, and again after ``release()``.
        - ``True`` — the warmup honored the request. Either
          ``skip_discovery=False`` was configured (nothing to skip), or the
          skip-path short-circuit fired and profiling started immediately.
        - ``False`` — ``skip_discovery=True`` was requested but no patched
          modules were reachable, so the manager fell back to the discovery
          phase.

        Lets callers detect the fallback without relying on log scraping in
        production loggers that route stderr to JSON. Test for the fallback
        explicitly with ``manager.skip_discovery_honored is False``; truthiness
        alone treats the undetermined state as a fallback.

        .. versionchanged:: v0.3.0
            Was ``bool``, defaulting to ``True`` before any transition. Now
            returns ``None`` until determined. Callers doing
            ``if not manager.skip_discovery_honored`` should switch to
            ``is False`` to keep the pre-``offload()`` reading out of the
            fallback branch.
        """
        return self._skip_discovery_honored

    @_error_boundary
    def _transition_to_profile(self):
        """Transition to profiling phase.

        Method name kept to align with ``TensorManager.initialize_profile()``.
        """
        if self._tensor_manager is None:
            return

        self._warn_if_compile_wrapped("PROFILING")
        new_model = self._tensor_manager.initialize_profile()
        self._swap_to_new_model(new_model, OffloadPhase.PROFILING)
        self._compiled.on_enter_profile()

    @_error_boundary
    def _transition_to_inference(self):
        """Transition to inference phase."""
        try:
            if self._tensor_manager is None:
                return

            self._warn_if_compile_wrapped("INFERENCE")
            new_model = self._tensor_manager.initialize_inference()
            self._swap_to_new_model(new_model, OffloadPhase.INFERENCE)
            # Compiled-offload: swap each layer's forward to the compile-transparent
            # variant, then let CompiledOffload install loader / compile_fn / replan.
            self._install_compiled_forwards()
            self._compiled.on_enter_inference()
        finally:
            self._dump_instrumentation()

    def _dump_instrumentation(self) -> None:
        if self._current_phase is not OffloadPhase.INFERENCE:
            return
        if (
            not self.config.enable_instrumentation
            or not self.config.instrumentation_output_dir
            or self._tensor_manager is None
        ):
            return
        try:
            dump_to_directory(
                self.config.instrumentation_output_dir,
                extra={"memory_transfer_stats": self._tensor_manager.get_memory_transfer_stats()},
            )
        except Exception:
            LOGGER.warning("OffloadManager: instrumentation dump failed; continuing.", exc_info=True)

    def update_state(self, *, replay_generation: int = -1) -> None:
        """Advance the phase state machine by one iteration.

        In ``INFERENCE``, phase transitions are finished, but when a replan
        tail is armed this method still advances it:

        * **CUDA-graph** (``request_strategy_replan(manual_update_state=True)``)
          — :meth:`update_offload_timing` then advance the measure counter;
          the last call rebuilds from graph budgets.
        * **Compiled / external-compile** — advance the passive warm→measure
          tail via :meth:`CompiledOffload.on_forward` (usually from the
          forward hook; no explicit call needed).

        Args:
            replay_generation: Forwarded to :meth:`update_offload_timing` on
                the CUDA-graph measure path. Pass the same generation used by
                a gen-aware serving hook so ``-1`` is not required and
                double-publish is avoided.

        No-op in ``NOT_INITIALIZED``.

        When profiling is suspended:

        - During DISCOVERY the iteration counter still advances (discovery
          durations are unused and cleared at the transition anyway).
        - During PROFILING the counter is paused so that suppressed passes
          don't consume the ``profiling_iters`` budget.

        Compiled-profile compile-warmup model forwards (while
        ``profile_compile_warm_remaining > 0``) likewise do not advance the
        PROFILING counter, so the logged ``profiling_iters`` measured samples
        are collected in full after warmup.
        """
        if self._current_phase == OffloadPhase.INFERENCE:
            if self._manual_update_state:
                # CUDA-graph replan: caller invokes this after each graph.replay().
                # Do not advance the measure counter if finalize failed — that
                # would finish replan from partial/skewed budgets.
                if not self.update_offload_timing(replay_generation=replay_generation):
                    self._fail_manual_update_replan(
                        RuntimeError(
                            "FlexTensor: update_offload_timing failed during "
                            "CUDA-graph measure replan; aborting (strategy unchanged)."
                        )
                    )
                self._compiled.advance_tail(finish_replan=self._finish_manual_update_replan)
                return
            # Compiled path: forward hook usually calls us; advance warm/measure.
            self._compiled.on_forward()
            return

        if self._current_phase == OffloadPhase.NOT_INITIALIZED:
            return

        if self._tensor_manager is None:
            raise RuntimeError(
                f"OffloadManager is in phase {self._current_phase.name} but its TensorManager is not set."
            )

        if self._tensor_manager.is_profiling_suspended() and self._current_phase == OffloadPhase.PROFILING:
            return

        # Compile-warmup is one slot per model forward and must not consume the
        # profiling_iters measure budget (see compiled-profile log contract).
        if self._current_phase == OffloadPhase.PROFILING and self._compiled.profile_compile_warm_remaining > 0:
            self._compiled.advance_profile_compile_warmup()
            return

        self._iteration_count += 1

        if self._current_phase == OffloadPhase.DISCOVERY and self._iteration_count >= self.config.discovery_iters:
            self._transition_to_profile()
        elif self._current_phase == OffloadPhase.PROFILING and self._iteration_count >= self._eager_profiling_iters():
            self._transition_to_inference()

    def _eager_profiling_iters(self) -> int:
        """Eager profiling budget for the PROFILING -> INFERENCE transition."""
        return self._compiled.eager_profiling_iters()

    @property
    def iters_before_inference(self) -> int:
        """Forwards to run before the model is ready to serve.

        This reflects the active path:

        - ``discovery_iters`` is *excluded* when ``skip_discovery=True`` was
          honored (the manager short-circuits from DISCOVERY straight to
          PROFILING inside ``offload()``, so no discovery forwards are
          consumed). It is included when ``skip_discovery=False``, or when
          ``skip_discovery=True`` was requested but no patched modules were
          reachable and the manager fell back to full discovery — see
          :attr:`skip_discovery_honored`.
        - The profile budget uses the path-specific eager seed via
          :meth:`CompiledOffload.eager_profiling_iters` and
          :meth:`CompiledOffload.extra_iters_before_inference`.

        Both active components are floored at ``1``. :meth:`update_state` runs
        as a post-forward hook and compares ``_iteration_count >= budget``, so
        each phase it drives consumes at least one forward regardless of the
        configured count. Returning the raw ``0`` (reachable with
        ``profiling_iters=0``, or ``discovery_iters=0`` while discovery is
        active — both are ``ge=0``) would make the documented drive loop stop
        short and strand the model in DISCOVERY or PROFILING, serving through
        the profile-mode model with no error and no warning.

        Replan measure/warmup forwards are **not** included here; drive them
        with :meth:`request_strategy_replan` after INFERENCE when needed.
        """
        # ``is True`` rather than truthiness: while undetermined (``None``) the
        # skip has not fired, so budget for discovery. Over-counting costs a
        # spare forward; under-counting strands the model mid-phase.
        discovery_active = not (self.config.skip_discovery and self._skip_discovery_honored is True)
        discovery_component = max(1, self.config.discovery_iters) if discovery_active else 0
        # Already floored by the property — single source of truth for the
        # profile budget so the two accessors cannot drift apart.
        profile_component = self.eager_profiling_iters
        return discovery_component + profile_component + self._compiled.extra_iters_before_inference()

    @staticmethod
    def _validate_shm_config(config: OffloadConfig) -> None:
        """Validate high-level SHM settings before creating or reusing a manager."""
        if config.shm_enabled and (config.shm_namespace is None or not config.shm_namespace.strip()):
            raise ValueError("OffloadConfig.shm_namespace must be non-empty when shm_enabled=True")

    def _validate_existing_tensor_manager_config(self, config: OffloadConfig) -> None:
        """Reject SHM settings that conflict with the live TensorManager."""
        if self._tensor_manager is None:
            return

        # Protocol doubles and older managers may not expose a concrete boolean;
        # only an explicit True means the live manager owns SHM resources.
        manager_uses_shm = getattr(self._tensor_manager, "use_shm", False) is True
        if manager_uses_shm != config.shm_enabled:
            raise RuntimeError(
                "OffloadConfig.shm_enabled cannot be changed for an existing TensorManager; "
                "call release() before reinitializing"
            )
        if manager_uses_shm and self._tensor_manager.shm_namespace != config.shm_namespace:
            raise RuntimeError(
                "OffloadConfig.shm_namespace cannot be changed for an existing TensorManager; "
                "call release() before reinitializing"
            )

    def init(
        self,
        config: OffloadConfig | None = None,
        *,
        _shared_offload_owner_token: object | None = None,
    ) -> None:
        self._check_shared_offload_reservation(_shared_offload_owner_token)
        if self._cleanup_blocked:
            raise RuntimeError("Manager cleanup is incomplete; retained resources cannot be reused safely.")
        if self._failed_restored_tensor_manager is not None:
            raise RuntimeError(
                "Cannot initialize while candidate cleanup is pending; call release() to retry shutdown first."
            )
        next_config = self.config if config is None else config
        self._validate_shm_config(next_config)
        self._validate_existing_tensor_manager_config(next_config)
        if config is not None:
            self.set_config(config)
        if self._tensor_manager is None:
            try:
                self._initialize_tensor_manager()
            except Exception:
                tensor_manager = self._tensor_manager
                if tensor_manager is not None:
                    try:
                        tensor_manager.shutdown()
                    except Exception:
                        self._cleanup_blocked = True
                        LOGGER.exception("TensorManager cleanup failed after initialization error")
                    else:
                        self._tensor_manager = None
                raise

    @_error_boundary
    def offload_from_state(  # noqa: C901 - takeover is one transactional lifecycle.
        self,
        model: nn.Module,
        state: TensorManagerState,
        config: OffloadConfig | None = None,
        *,
        allow_strategy_replan: bool = True,
    ) -> nn.Module:
        """Adopt a saved state and return the offloaded model proxy.

        If this method raises, cleanup preserves storage ownership and removes
        FlexTensor modifications where possible, but does not roll back storage
        placement. The input model is not guaranteed usable and must be discarded.
        """
        self._check_shared_offload_reservation()
        if self._cleanup_blocked:
            raise RuntimeError("Manager cleanup is incomplete; retained resources cannot be reused safely.")
        if self._state_takeover_active or self._current_phase != OffloadPhase.NOT_INITIALIZED:
            raise RuntimeError("OffloadManager is already active; call release() before adopting another state.")
        if is_torch_compiled_module(model):
            raise RuntimeError("Cannot adopt state into a compiled model; use the eager model first.")

        self._manual_update_state = False
        effective_config = config if config is not None else self.config
        activation_resolved = False
        try:
            self._compiled.resolve_activation(effective_config, None)
            if not allow_strategy_replan:
                self._compiled.replan_active = False
            activation_resolved = True
            self.init(config)
        except Exception:
            if activation_resolved:
                self._compiled.teardown()
            raise
        if self._tensor_manager is None:
            raise RuntimeError("Tensor manager initialization failed")

        try:
            self._compiled.arm_non_destructive_first_loader()
            plan = self._tensor_manager.plan_state_adoption(model, state)
        except Exception:
            self._cleanup_failed_state_takeover(model, [])
            raise

        # Loaders may replace or empty Parameter.data while taking ownership.
        # Retain the bounded post-migration homes until setup commits.
        migrated_parameter_data: list[tuple[nn.Parameter, torch.Tensor]] = []
        try:
            self._tensor_manager.execute_state_adoption(model, plan)
            self._state_takeover_active = True
            for parameter in model.parameters():
                migrated_parameter_data.append((parameter, get_tensor_data(parameter)))
            self._tensor_manager.restore_adopted_state(model, state)
            self._tensor_manager.prepare_infer_load_mode()

            self._model = model
            self._offload_modules(model, self.config.include_patterns)
            self._exclude_modules(model, self.config.exclude_patterns)
            self._check_no_modules_patched()

            final_model = self._tensor_manager.prepare_final_model(model, in_place=True)
            self._tensor_manager.set_model(final_model)
            final_model = self._tensor_manager.initialize_inference()
            self._swap_to_new_model(final_model, OffloadPhase.INFERENCE)
            self._install_compiled_forwards()
            self._compiled.on_enter_inference()
            self._dump_instrumentation()
            if self._model_proxy is None:
                self._model_proxy = OffloadModelProxy(self._model, self)
            return self._model_proxy
        except Exception:
            self._cleanup_failed_state_takeover(model, migrated_parameter_data)
            raise

    def _cleanup_failed_state_takeover(  # noqa: C901
        self,
        model: nn.Module,
        migrated_parameter_data: list[tuple[nn.Parameter, torch.Tensor]],
    ) -> None:
        """Drop setup resources without rolling back completed migrations."""
        self._compiled.teardown()
        if self._state_hook_handle is not None:
            _safe_remove_hook_handle(self._state_hook_handle, context="failed state takeover")
            self._state_hook_handle = None
        for module in model.modules():
            hooks = getattr(module, "_forward_hooks", None)
            if hooks is None:
                continue
            for hook_id, hook in list(hooks.items()):
                if getattr(hook, "_ft_state_update_hook", False):
                    hooks.pop(hook_id, None)

        unrestored_modules = []
        for module in reversed(self._patched_modules):
            try:
                self._restore_module_forward(module)
            except Exception:
                unrestored_modules.append(module)
                LOGGER.exception("Failed-state-takeover cleanup could not restore a module forward")
        self._patched_modules[:] = reversed(unrestored_modules)

        ownership_restored = True
        for parameter, migrated_data in migrated_parameter_data:
            try:
                set_tensor_data(parameter, migrated_data)
            except Exception:
                ownership_restored = False
                LOGGER.exception("Failed-state-takeover cleanup could not restore post-migration parameter data")

        tensor_manager = self._tensor_manager
        cleanup_complete = ownership_restored and not self._patched_modules
        if cleanup_complete and tensor_manager is not None:
            try:
                tensor_manager.release_memory()
            except Exception:
                cleanup_complete = False
                LOGGER.exception("Failed-state-takeover release_memory cleanup failed; preserving the original error")
            if cleanup_complete:
                try:
                    tensor_manager.shutdown()
                except Exception:
                    cleanup_complete = False
                    LOGGER.exception("Failed-state-takeover shutdown cleanup failed; preserving the original error")
        if not cleanup_complete:
            LOGGER.error(
                "Failed-state-takeover cleanup retained TensorManager resources because parameter ownership "
                "or resource teardown could not be completed; restart the process after reporting this error."
            )

        self._tensor_manager = None if cleanup_complete else tensor_manager
        self._cleanup_blocked = not cleanup_complete
        self._failed_state_takeover_parameter_data = [] if cleanup_complete else migrated_parameter_data
        self._initialized = False
        self._current_phase = OffloadPhase.NOT_INITIALIZED
        self._iteration_count = 0
        if cleanup_complete:
            self._model_proxy = None
            self._model = None
        else:
            self._model = model
        self._state_takeover_active = False

    def offload(
        self,
        model: nn.Module,
        config: OffloadConfig | None = None,
        compile_fn: Callable[[nn.Module], nn.Module] | None = None,
    ) -> nn.Module:
        """Offload modules in the model based on config patterns.

        Patches modules matching ``config.include_patterns`` (and applies
        ``config.exclude_patterns``) to automatically manage tensor transfers.
        Supports wildcards in include/exclude patterns.

        Args:
            model: PyTorch model to patch
            config: OffloadConfig to use for offload. Set
                ``config.external_compile=True`` (or ``FT_EXTERNAL_COMPILE=1`` via
                :func:`~flextensor.config.load_config``)
                to enable the compile-transparent ``pre_compute/post_compute`` custom-op
                forwards and auto-register the rolling loader at INFERENCE so the
                caller can apply external per-unit ``torch.compile`` after the eager
                forwards counted by :attr:`iters_before_inference` (one graph per
                offloaded unit — not whole-model compile).
            compile_fn: Optional per-unit compiler. When given, this activates the
                compiled-offload path (no ``FT_EXTERNAL_COMPILE`` env var needed):
                FlexTensor patches the offloaded units so the offload barriers are
                visible to the compiler, and applies ``compile_fn`` to **each
                offloaded unit** -- one compiled graph per unit
                (``lambda m: torch.compile(m, fullgraph=True)``, a Torch-TensorRT
                compile, an eager passthrough ``lambda m: m``, ...). Compiling at the
                offload granularity is slot-alias safe by construction: each graph
                reads exactly one rolling slot, so no graph can ever alias two
                same-slot units (the monolithic-graph miscompile). The callable takes
                one module and returns a callable module with the same forward
                signature; if it raises on a unit, that unit is left eager.
                With the default ``profile_mode='view'``, FlexTensor also runs
                compiled view-profile under ``compile_fn``, so the offload strategy
                is built from compiled timings and **no** post-INFERENCE replan is
                needed. Call :meth:`request_strategy_replan` only when timings were
                not collected under compile (``profile_mode='getter'``, or external
                ``torch.compile`` after ``external_compile=True``). ``None``
                (default) is exactly today's eager offload -- no behavior change.

        Returns:
            A wrapper around the model that tracks state transitions

        Examples:
            Basic eager offload (see ``docs/quick-start.md`` for a runnable
            end-to-end example). With the default ``skip_discovery=False``,
            ``discovery_iters`` + ``profiling_iters`` forwards drive the
            state machine into ``INFERENCE`` (the post-forward hook
            transitions on the Nth call); the extra iteration below runs
            the first real inference forward::

                import torch
                import flextensor as ft

                model = MyModel().to("cpu").eval()
                config = ft.OffloadConfig(include_patterns=["embed", "layers.*", "head"])
                proxy = ft.offload(model, config)
                om = ft.get_offload_manager()

                x = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
                with torch.no_grad():
                    for _ in range(om.iters_before_inference + 1):
                        proxy(x)   # drives discovery -> profile -> inference

            Per-unit ``torch.compile`` via ``compile_fn`` (preferred; see
            ``docs/how-to/torch-compile.md``). Whole-model ``torch.compile(proxy)``
            is not supported across phase transitions::

                def compile_fn(module: torch.nn.Module) -> torch.nn.Module:
                    return torch.compile(module, fullgraph=True)

                proxy = ft.offload(model, config, compile_fn=compile_fn)
                om = ft.get_offload_manager()
                with torch.no_grad():
                    for _ in range(om.iters_before_inference):
                        proxy(x)  # discovery + compiled view-profile -> INFERENCE

            For external per-unit compile after INFERENCE, set
            ``OffloadConfig(external_compile=True)``, drive
            ``om.iters_before_inference`` eager forwards, ``torch.compile`` each
            offloaded unit, then :meth:`request_strategy_replan`.
        """
        self._check_shared_offload_reservation()

        # Reject an already-compiled model up front, before any state mutation.
        # ``torch.compile`` (and backends built on it, e.g. ``torch_tensorrt``)
        # wrap the model in an ``OptimizedModule`` whose self-referential
        # ``__getattr__``/``__setattr__`` proxy to ``_orig_mod`` makes FlexTensor's
        # recursive module preprocessing loop without terminating (``RecursionError``),
        # and whose captured graph predates -- and therefore ignores -- the offload
        # patches. Fail fast with actionable guidance instead of crashing deep in
        # preprocessing or silently serving a stale graph.
        if is_torch_compiled_module(model):
            raise RuntimeError(
                "FlexTensor does not support offloading an already-compiled model "
                f"(received a torch.compile wrapper: {type(model).__name__}). "
                "Offload the eager model first. For FlexTensor-driven per-unit compile "
                "use offload(model, config, compile_fn=<per-unit compiler>). For "
                "external per-unit compile after INFERENCE use "
                "OffloadConfig(external_compile=True), drive "
                "om.iters_before_inference eager forwards, then torch.compile each "
                "offloaded unit (e.g. each block). See docs/how-to/torch-compile.md."
            )

        # Resolve compiled-offload flags *before* init/patching so
        # ``_patch_module_forward`` pre-builds compile-transparent forwards when active.
        # Drop any in-flight CUDA-graph measure arm from a prior session on this manager.
        self._manual_update_state = False
        effective_config = config if config is not None else self.config
        self._compiled.resolve_activation(effective_config, compile_fn)

        self.init(config)

        # Arm a non-destructive first inference loader only when a post-compile
        # replan is intended (``replan_active``); view-profile compile_fn must not
        # retain another model-sized host copy.
        self._compiled.arm_non_destructive_first_loader()
        # Save old model for hook transfer before overwriting
        old_model = self._model
        self._model = model

        self._offload_modules(self._model, self.config.include_patterns)
        self._exclude_modules(self._model, self.config.exclude_patterns)
        self._check_no_modules_patched()

        # Create proxy that delegates to current model
        if self._model_proxy is None:
            self._model_proxy = OffloadModelProxy(self._model, self)

        # Transfer hooks from old model to new model (if re-initializing)
        self._transfer_hooks(old_model, self._model)

        self._transition_to_warmup()

        return self._model_proxy

    def _install_state_update_hook(self) -> None:
        """Register a compiler-disabled forward hook on ``self._model`` that
        advances phase state.

        ``compiler_utils.disable`` makes ``torch.compile`` emit a graph break at the hook's
        call-site and run the body eagerly, so ``update_state`` fires reliably
        even when ``OptimizedModule`` bypasses ``OffloadModelProxy.__call__``
        under ``torch.compile(proxy)``.

        ``prepend=True`` ensures user-registered top-level forward hooks see the
        post-transition phase regardless of registration order.

        Idempotent: any previously installed state-update hook is removed once
        the new one is in place.  No-op when ``self._model is None`` (any
        existing handle is removed first).
        """
        old_handle = self._state_hook_handle

        if self._model is None:
            if old_handle is not None:
                _safe_remove_hook_handle(old_handle, context="install (model=None)")
                self._state_hook_handle = None
            return

        om = self

        # Keep the hook eager under torch.compile; on builds without the
        # public compiler API, the decorator is a no-op.
        @_compiler_disable
        def _update_state_hook(_module: nn.Module, _inputs: Any, _output: Any) -> None:
            om.update_state()

        # Tag the closure so ``_transfer_hooks`` can identify and skip it.
        _update_state_hook._ft_state_update_hook = True  # type: ignore[attr-defined]  # noqa: SLF001

        # Register the new hook *before* removing the old one, so a failure here
        # leaves the old handle intact (still cleanable via ``release()``).
        new_handle = self._model.register_forward_hook(_update_state_hook, prepend=True)
        self._state_hook_handle = new_handle
        if old_handle is not None:
            # Tolerate a stale handle (model already replaced, hook dict
            # mutated externally): the new hook is installed, the old one is
            # untracked, and a hard raise here would fail the transition for
            # what is at worst a one-time stale reference.
            _safe_remove_hook_handle(old_handle, context="install (replacing previous handle)")

    def offload_block(self, name: str) -> Any:
        """Create an offload block context manager.

        Args:
            name: Name of the offload block (used for tracking)

        Returns:
            OffloadBlock context manager (Trap, WarmupTrap, TrapInfer, …)

        Raises:
            RuntimeError: If ``offload()`` has not been called yet, or if
                ``skip_discovery`` was honored. Manual blocks need discovery;
                leave ``skip_discovery=False`` (the default). Allowed on the
                no-patched-modules fallback.

        Examples:
            >>> with om.offload_block("encoder"):
            ...     output = encoder(hidden_states)

        Note:
            Compiled offload is driven by auto-patched forwards that call
            ``pre_compute/post_compute`` directly. Manual ``offload_block`` stays
            on the eager trap path (:class:`~flextensor.trap_tensor_mode.TrapInferDirect`
            / :class:`~flextensor.trap_tensor_mode.TrapInfer`); do not wrap an
            auto-patched compiled unit or transfers will double-schedule.
        """
        if self._tensor_manager is None:
            raise RuntimeError(
                "Tensor manager not initialized. Call offload() on a model first to initialize the manager."
            )
        # Gate on what actually happened, not on what was requested: when
        # skip_discovery was requested but no patched modules were reachable,
        # the manager fell back to a real discovery phase, which is exactly
        # what manual blocks need. Blocking there would reject the one
        # topology the fallback exists to keep working.
        #
        # ``is not False`` deliberately: allow manual blocks only when the
        # fallback is *known* to have happened. While undetermined (``None``)
        # no discovery has run, so permitting them would be unsafe.
        if self.config is not None and self.config.skip_discovery and self._skip_discovery_honored is not False:
            raise RuntimeError(
                "offload_block() is not supported when OffloadConfig.skip_discovery is True. "
                "Manual offload blocks require discovery-phase iterations to map their tensors. "
                "Set skip_discovery=False on the config when the model uses manual offload_block() calls."
            )
        return self._tensor_manager.trap(name)

    @staticmethod
    def _validate_restored_model_has_no_meta_tensors(model: nn.Module) -> None:
        """Fail if restored parameters or buffers still lack backing storage."""
        meta_parameters = sorted(
            name for name, parameter in model.named_parameters(remove_duplicate=False) if parameter.is_meta
        )
        meta_buffers = sorted(name for name, buffer in model.named_buffers(remove_duplicate=False) if buffer.is_meta)
        if not meta_parameters and not meta_buffers:
            return

        details = []
        if meta_parameters:
            details.append(f"parameters={meta_parameters}")
        if meta_buffers:
            details.append(f"buffers={meta_buffers}")
        raise RuntimeError("Restored model still contains meta tensors: " + "; ".join(details))

    @staticmethod
    def _cleanup_failed_restored_tensor_manager(
        tensor_manager: Any,
    ) -> bool:
        """Best-effort cleanup that never masks the restoration failure.

        Restored-state initialization documents the candidate model as
        unusable after failure. Discard its loader-backed tensor data during
        shutdown so rollback never allocates private copies of SHM weights.
        """
        try:
            tensor_manager.shutdown(preserve_model=False)
        except BaseException:
            LOGGER.exception(
                "OffloadManager: candidate TensorManager.shutdown() failed during restored-state rollback; "
                "preserving the original initialization error."
            )
            return False
        return True

    def _retry_failed_restored_tensor_manager_cleanup(self) -> None:
        """Retry shutdown without dropping a candidate that still fails."""
        tensor_manager = self._failed_restored_tensor_manager
        if tensor_manager is not None and self._cleanup_failed_restored_tensor_manager(tensor_manager):
            self._failed_restored_tensor_manager = None

    def _rollback_restored_model_patches(self) -> None:
        """Remove patches installed by the failed restored-state attempt."""
        for module in reversed(self._patched_modules):
            try:
                self._restore_module_forward(module)
            except BaseException:
                LOGGER.exception(
                    "OffloadManager: failed to remove a partial restored-model patch; "
                    "preserving the original initialization error."
                )
        self._patched_modules.clear()

    def _rollback_restored_state_hook(
        self,
        final_model: nn.Module | None,
        saved_hook_ids: dict[int, set[int]],
    ) -> None:
        """Remove a state hook installed before a failed restored-state commit."""
        if self._state_hook_handle is not None:
            try:
                self._state_hook_handle.remove()
            except BaseException:
                LOGGER.exception(
                    "OffloadManager: failed to remove the candidate state hook; "
                    "preserving the original initialization error."
                )

        # A failing swap may reset ``_state_hook_handle`` before its newly
        # registered hook is removed. Remove only FlexTensor-tagged hooks from
        # the uncommitted model so no orphan callback retains this manager.
        if final_model is not None:
            for module in final_model.modules():
                hooks = module.__dict__.get("_forward_hooks")
                if hooks is None:
                    continue
                preserved_hook_ids = saved_hook_ids.get(id(module), set())
                for hook_id, hook in list(hooks.items()):
                    if hook_id not in preserved_hook_ids and getattr(hook, "_ft_state_update_hook", False):
                        hooks.pop(hook_id, None)

        self._state_hook_handle = None

    def _restore_shared_offload(
        self,
        model: nn.Module,
        state: TensorManagerState,
        namespace: str,
        *,
        load_from_shm: bool,
        shm_owner: FlexibleSharedMemory | None = None,
        owner_token: object | None = None,
        compile_fn: Callable[[nn.Module], nn.Module] | None = None,
        before_compile: Callable[[], None] | None = None,
    ) -> OffloadModelProxy:
        """Initialize either SHM restore role through one transactional path."""
        self._check_shared_offload_reservation(owner_token)
        self._validate_restored_shm_namespace(namespace)
        if self._failed_restored_tensor_manager is not None:
            raise RuntimeError(
                "Cannot initialize restored SHM state while candidate cleanup is pending; "
                "call release() to retry cleanup before restoring another model."
            )
        if (
            self._cleanup_blocked
            or self._tensor_manager is not None
            or self._model is not None
            or self._model_proxy is not None
            or self._current_phase is not OffloadPhase.NOT_INITIALIZED
            or self._state_hook_handle is not None
            or self._patched_modules
        ):
            raise RuntimeError(
                "Cannot initialize restored SHM state while an active offload runtime exists; "
                "call release() before restoring another model."
            )

        candidate: TensorManagerProtocol | None = None
        final_model: nn.Module | None = None
        saved_final_model_hook_ids: dict[int, set[int]] = {}
        try:
            # Match offload()/offload_from_state(): resolve the compile mode
            # before building a manager so patched forwards use the right path.
            self._manual_update_state = False
            self._compiled.resolve_activation(self.config, compile_fn)
            if (load_from_shm or shm_owner is not None) and self._compiled.replan_active:
                LOGGER.warning(
                    "FlexTensor compiled-offload: strategy replan is disabled for coordinated SHM loads. "
                    "Published blocks must remain available to followers for the shared lifetime; "
                    "compiled forwards remain enabled."
                )
                self._compiled.replan_active = False
            self.init(_shared_offload_owner_token=owner_token)
            candidate = cast("TensorManagerProtocol | None", self._tensor_manager)
            if candidate is None:
                raise RuntimeError("TensorManager initialization did not produce a manager")
            self._compiled.arm_non_destructive_first_loader()

            # The allocation-block loader is created by initialize_warmup(), so
            # install namespace and role before any allocation can occur.
            candidate.shm_namespace = namespace
            candidate.shm_owner = shm_owner
            candidate.restore_state(model, state, load_from_shm=load_from_shm)
            candidate.initialize_warmup()
            candidate.initialize_profile()
            final_model = candidate.initialize_inference()
            saved_final_model_hook_ids = {
                id(module): set(module.__dict__.get("_forward_hooks", {})) for module in final_model.modules()
            }
            self._validate_restored_model_has_no_meta_tensors(final_model)

            # The saved plan defines the transfer pass boundaries. Expanding
            # discovery patterns again can add unprofiled units whose transfer
            # scopes cross into CUDA graph capture before the first saved unit.
            self._offload_modules(final_model, [stat.label for stat in state.stats])
            self._exclude_modules(final_model, self.config.exclude_patterns)
            self._check_no_modules_patched()

            self._model_proxy = OffloadModelProxy(final_model, self)
            self._swap_to_new_model(final_model, OffloadPhase.INFERENCE)
            # Publish the original tensor names before compiler wrappers can
            # rename modules. Keep publication inside the restore transaction.
            if before_compile is not None:
                before_compile()
            self._install_compiled_forwards()
            self._compiled.on_enter_inference()
            self._skip_discovery_honored = True
            self._dump_instrumentation()
            return self._model_proxy
        except BaseException:
            try:
                self._compiled.teardown()
            except BaseException:
                LOGGER.exception(
                    "OffloadManager: compiled-offload cleanup failed during restored-state rollback; "
                    "preserving the original initialization error."
                )
            self._rollback_restored_model_patches()
            self._rollback_restored_state_hook(final_model, saved_final_model_hook_ids)

            cleanup_target = candidate if candidate is not None else self._tensor_manager
            self._tensor_manager = None
            self._tensor_manager_oneshot_snapshot = {}
            self._model = None
            self._model_proxy = None
            self._current_phase = OffloadPhase.NOT_INITIALIZED
            self._iteration_count = 0
            self._initialized = False
            self._skip_discovery_honored = None
            self._manual_update_state = False
            self._cleanup_blocked = False

            if cleanup_target is not None and not self._cleanup_failed_restored_tensor_manager(cleanup_target):
                self._failed_restored_tensor_manager = cleanup_target
            raise

    @_error_boundary
    def _initialize_from_shm(self, coordinator: _ShmCoordinatorLike, model: nn.Module) -> OffloadModelProxy:
        """Initialize as a follower from shared memory.

        Loads profile, restores state, and jumps directly to INFERENCE.

        Args:
            coordinator: ShmCoordinator in follower mode.
            model: Model constructed on meta device.

        Returns:
            Proxy-wrapped model in INFERENCE phase.

        Raises:
            RuntimeError: If coordinator is a creator (not a follower).
            ValueError: If the profile's loader type is incompatible or tensors
                from the profile are missing from the model.
        """
        if coordinator.is_creator:
            raise RuntimeError("_initialize_from_shm called on creator coordinator")

        coordinator.wait_for_ready(timeout=self.config.shm_wait_timeout or None)
        state = coordinator.read_profile()
        return self._restore_shared_offload(model, state, coordinator.namespace, load_from_shm=True)

    def clear_profiling_durations(self) -> None:
        """Clear all accumulated duration measurements.

        Wipes every duration sample collected so far.  Tensor-to-layer
        mappings (discovery data) are **not** affected.

        Raises:
            RuntimeError: if called before :func:`flextensor.offload` —
                no active ``TensorManager`` to clear.

        .. note::
            With ``OffloadConfig(enabled=False)`` the active manager is
            :class:`NoOpTensorManager`: this method is a no-op because
            no durations are ever recorded.
        """
        if self._tensor_manager is None:
            raise RuntimeError(
                "clear_profiling_durations() called before flextensor.offload(...); "
                "call flextensor.offload(model, config=...) first."
            )
        self._tensor_manager.clear_profiling_durations()

    def suspend_profiling(self) -> None:
        """Suppress profiling recording and (in PROFILING phase) freeze the iteration counter.

        While suspended:

        * ``record_all()`` (the PROFILING recorder) is a **complete no-op**
          — neither tensor IDs nor duration samples are stored, so a paused
          warmup pass cannot widen per-layer tensor sets on data-dependent
          models (MoE / conditional branches / mixed-batch shapes).
        * ``record_duration()`` skips duration recording.
        * ``record_tensors()`` (the DISCOVERY recorder) is **not** affected
          — discovery's tensor-to-layer mapping is a hard prerequisite for
          every later phase.
        * In DISCOVERY the iteration counter **still advances** (discovery
          durations are unused).
        * In PROFILING the counter is **paused** so suppressed passes don't
          consume the ``profiling_iters`` budget.

        Suspensions are reference-counted: each call must be matched by a
        :meth:`resume_profiling`, and recording only resumes once every
        outstanding suspension has been released.  Independent callers can
        therefore bracket their own sections without accidentally resuming
        someone else's suspension.

        Raises:
            RuntimeError: if called before :func:`flextensor.offload` —
                no active ``TensorManager`` to suspend.

        .. note::
            With ``OffloadConfig(enabled=False)`` the active manager is
            :class:`NoOpTensorManager`: these methods are no-ops and the
            profiling-iteration counter is not frozen.
        """
        if self._tensor_manager is None:
            raise RuntimeError(
                "suspend_profiling() called before flextensor.offload(...); "
                "call flextensor.offload(model, config=...) first."
            )
        self._tensor_manager.suspend_profiling()

    def resume_profiling(self) -> None:
        """Release one outstanding :meth:`suspend_profiling` call.

        Recording only resumes once every outstanding suspension has been
        released.

        Raises:
            RuntimeError: if called before :func:`flextensor.offload`
                (no active ``TensorManager``), or if called while not
                suspended (unbalanced ``suspend`` / ``resume``).
        """
        if self._tensor_manager is None:
            raise RuntimeError(
                "resume_profiling() called before flextensor.offload(...); "
                "call flextensor.offload(model, config=...) first."
            )
        self._tensor_manager.resume_profiling()

    # See `flextensor.helpers.ProfilingSuspender.suspended` for the `-> Any` rationale
    # (beartype + @contextmanager cross-version compatibility).
    @contextmanager
    def pause_profiling(self) -> Any:
        """Context-manager form of :meth:`suspend_profiling` / :meth:`resume_profiling`.

        Raises:
            RuntimeError: if called before :func:`flextensor.offload` —
                no active ``TensorManager``, so the ``with`` block cannot
                actually run with profiling paused.
        """
        if self._tensor_manager is None:
            raise RuntimeError(
                "pause_profiling() called before flextensor.offload(...); "
                "call flextensor.offload(model, config=...) first."
            )
        with self._tensor_manager.pause_profiling():
            yield

    def release(  # noqa: C901 - teardown has independent best-effort cleanup stages.
        self,
        *,
        preserve_model: bool = True,
        _shared_offload_owner_token: object | None = None,
    ) -> None:
        """Release this manager and its TensorManager resources.

        Args:
            preserve_model: Restore loader-managed tensor values before
                teardown. Set to ``False`` only when the managed model will
                be discarded; this avoids private copies of SHM-backed
                weights and leaves that model unusable.
        """
        self._check_shared_offload_reservation(_shared_offload_owner_token)
        owns_shared_reservation = (
            _shared_offload_owner_token is not None and self._shared_offload_owner_token is _shared_offload_owner_token
        )
        if self._cleanup_blocked and not owns_shared_reservation:
            raise RuntimeError(
                "Manager cleanup is incomplete; TensorManager resources remain retained and cannot be released safely."
            )
        if owns_shared_reservation:
            self._cleanup_blocked = False
        # Undo any compile_fn substitutions and tear down the compiled-offload
        # tail before restoring forwards, so the model is left as it was handed in.
        self._manual_update_state = False
        self._compiled.teardown()

        # Failed SHM restore candidates are not associated with the live model,
        # so retry their staged cleanup independently before releasing it.
        self._retry_failed_restored_tensor_manager_cleanup()

        if self._tensor_manager is not None:
            teardown_error: Exception | None = None
            try:
                self._tensor_manager.release_memory()
            except Exception as error:
                teardown_error = error
            try:
                if preserve_model:
                    self._tensor_manager.shutdown()
                else:
                    self._tensor_manager.shutdown(preserve_model=False)
            except Exception as error:
                if teardown_error is None:
                    teardown_error = error
            if teardown_error is not None:
                self._cleanup_blocked = True
                raise teardown_error

        # Do not remove transfer traps until their manager has released every
        # storage owner successfully.
        try:
            for module in self._patched_modules:
                self._restore_module_forward(module)
        except Exception:
            self._cleanup_blocked = True
            raise
        self._patched_modules.clear()

        if self._tensor_manager is not None:
            self._tensor_manager = None
        # Nothing has captured a value any more; the next offload() re-snapshots.
        self._tensor_manager_oneshot_snapshot = {}
        # No warmup transition owns this any more. Leaving the previous cycle's
        # verdict in place would make ``skip_discovery_honored`` and
        # ``iters_before_inference`` answer for a manager that no longer exists.
        self._skip_discovery_honored = None
        self._initialized = False
        self._current_phase = OffloadPhase.NOT_INITIALIZED
        self._model_proxy = None
        self._model = None
        self._state_takeover_active = False
        self._failed_state_takeover_parameter_data.clear()
        if self._state_hook_handle is not None:
            _safe_remove_hook_handle(self._state_hook_handle, context="release()")
            self._state_hook_handle = None
        if self._failed_restored_tensor_manager is not None:
            raise RuntimeError("TensorManager cleanup remains pending; call release() again to retry shutdown.")

    def _offload_modules(self, model: nn.Module, patterns: list[str]) -> None:
        """Patch modules matching include patterns, skipping those with patched ancestors.

        Name patterns use flat ``named_modules()`` iteration with
        ``matches_any_pattern(recursive_star=False)`` to preserve existing
        single-segment ``*`` semantics (``layers.*`` matches direct children
        only).  Parameter-level patterns (e.g., ``layers.*.weight``) are
        auto-truncated to derive module-level patterns (e.g., ``layers.*``) via
        ``_derive_module_patterns``.

        Class patterns (``class:<glob>``) match against both
        ``type(module).__name__`` and the fully-qualified class name, letting
        hybrid architectures select modules without relying on fragile
        upstream name conventions while still supporting disambiguation via
        FQCN globs such as ``class:torch.nn.*.Linear``.

        Validation uses ``recursive_star=False`` for module paths but
        ``recursive_star=True`` for parameter paths, matching the runtime
        semantics in ``_any_prefix_matches`` (tensor_discovery.py).

        Modules whose ancestors are already patched are skipped, because they belong to
        their ancestor's offload unit rather than forming independent units. This defines
        offload unit boundaries: only modules with no patched ancestors become offload units.

        Args:
            model: The model whose sub-modules to patch.
            patterns: Include patterns (name-based and/or ``class:`` prefixed).
        """
        module_paths = get_module_paths(model)
        selected_paths = select_offload_unit_paths(model, patterns, [])
        matched_patterns = _find_matched_patterns(
            model,
            patterns,
            module_paths,
            recursive_star=False,
            param_recursive_star=True,
            include_parameters=True,
        )
        for module_path, module in model.named_modules():
            if not module_path:  # skip root
                continue
            if module_path in selected_paths:
                if has_patched_ancestor(model, module_path):
                    continue
                self._patch_module_forward(module, module_path)
        for pattern in patterns:
            if pattern not in matched_patterns:
                LOGGER.warning("Include pattern '%s' did not match any modules or parameters.", pattern)

    def _exclude_modules(self, model: nn.Module, patterns: list[str]) -> None:
        """Un-patch offload units matching any exclude pattern.

        Name patterns use ``matches_any_pattern(recursive_star=True)`` so
        ``foo.*`` matches all descendants. Class patterns (``class:<glob>``)
        un-patch every module whose class name matches, regardless of
        position in the tree.

        Args:
            model: The model whose sub-modules to un-patch.
            patterns: Exclude patterns (name-based and/or ``class:`` prefixed).
        """
        if not patterns:
            return
        patched_before = len(self._patched_modules)
        module_paths = get_module_paths(model)
        name_patterns, class_patterns = partition_patterns(patterns)
        class_matched_paths = get_class_matched_module_paths(model, class_patterns)
        matched_patterns = _find_matched_patterns(
            model,
            patterns,
            module_paths,
            recursive_star=True,
            include_parameters=True,
        )
        for module_path, module in model.named_modules():
            if not is_offload_patched_module(module):
                continue
            matched_by_name = matches_any_pattern(module_path, name_patterns, recursive_star=True)
            matched_by_class = module_path in class_matched_paths
            if matched_by_name or matched_by_class:
                self._restore_module_forward(module)
                if module in self._patched_modules:
                    self._patched_modules.remove(module)
        for pattern in patterns:
            if pattern not in matched_patterns:
                LOGGER.warning("Exclude pattern '%s' did not match any modules or parameters.", pattern)
        if patched_before > 0 and not self._patched_modules:
            LOGGER.error(
                "All %d included modules were removed by exclude_patterns %s. Offloading is effectively disabled.",
                patched_before,
                patterns,
            )

    def _check_no_modules_patched(self) -> None:
        """Emit an error when no modules ended up patched after include/exclude."""
        if not self._patched_modules and self.config.include_patterns:
            LOGGER.error(
                "No modules matched any include pattern %s. "
                "Offloading is effectively disabled. "
                "Check pattern spelling against model.named_modules() paths.",
                self.config.include_patterns,
            )


@dataclass(frozen=True, slots=True)
class _ManagerEntry:
    """Internal entry pairing an OffloadManager with its owner thread."""

    manager: OffloadManager
    owner_thread: int


def get_offload_manager(name: str = DEFAULT_MANAGER_NAME) -> OffloadManager:
    """Get or create an OffloadManager singleton.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.

    Returns:
        OffloadManager instance

    Raises:
        RuntimeError: If the named manager was created by a different thread.

    Examples:
        >>> om = get_offload_manager()
        >>> om.set_config(OffloadConfig(gpu_device=0))
    """
    current_thread = threading.get_ident()
    with _MANAGER_MAP_LOCK:
        if name not in OFFLOAD_MANAGER_MAP:
            OFFLOAD_MANAGER_MAP[name] = _ManagerEntry(
                manager=OffloadManager(name),
                owner_thread=current_thread,
            )
        entry = OFFLOAD_MANAGER_MAP[name]
    if entry.owner_thread != current_thread:
        raise RuntimeError(
            f"OffloadManager '{name}' belongs to thread {entry.owner_thread}, "
            f"but accessed from thread {current_thread}. "
            f"Each named manager must be used from a single thread."
        )
    return entry.manager


# Convenience functions for the default offload manager


def init(config: OffloadConfig | None = None, name: str = DEFAULT_MANAGER_NAME):
    om = get_offload_manager(name)
    om.init(config)


def offload(
    model: nn.Module,
    config: OffloadConfig | None = None,
    name: str = DEFAULT_MANAGER_NAME,
    compile_fn: Callable[[nn.Module], nn.Module] | None = None,
) -> nn.Module:
    """Offload modules using the specified offload manager.

    Args:
        model: PyTorch model to patch.
        config: OffloadConfig to use for offload.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
        compile_fn: Optional per-unit compiler. Passing it activates the
            compiled-offload path without any ``FT_EXTERNAL_COMPILE`` env var: FlexTensor
            applies ``compile_fn`` to **each offloaded unit** (one compiled graph
            per unit -- slot-alias safe by construction). With default
            ``profile_mode='view'``, compiled view-profile builds the strategy from
            compiled timings so no post-INFERENCE replan is needed. Use
            :meth:`OffloadManager.request_strategy_replan` only for getter-profile
            or external-compile paths. ``None`` (default) is exactly today's eager
            offload. See :meth:`OffloadManager.offload` for the full contract.

    Returns:
        The patched model.
    """
    om = get_offload_manager(name)
    return om.offload(model, config, compile_fn=compile_fn)


def set_config(config: OffloadConfig, name: str = DEFAULT_MANAGER_NAME):
    """Set config for the specified offload manager.

    Args:
        config: OffloadConfig to apply.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    om.set_config(config)


def save_profile(profile_directory: str | None = None, name: str = DEFAULT_MANAGER_NAME) -> None:
    """Save profile using the specified offload manager.

    Args:
        profile_directory: Directory to save profile to.
            If None, uses config.profile_storage_dir.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    om.save_profile(profile_directory)


def load_profile(
    profile_directory: str | None = None, model: nn.Module | None = None, name: str = DEFAULT_MANAGER_NAME
) -> None:
    """Load profile using the specified offload manager.

    Args:
        profile_directory: Directory containing the profile.
            If None, uses config.profile_storage_dir.
        model: Model to restore state to. If None, uses the current model.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    om.load_profile(profile_directory, model)


def offload_from_state(
    model: nn.Module,
    state: TensorManagerState,
    config: OffloadConfig | None = None,
    name: str = DEFAULT_MANAGER_NAME,
    *,
    allow_strategy_replan: bool = True,
) -> nn.Module:
    """Adopt an in-memory tensor-manager state and enter inference.

    The state must describe ``model`` and use the transfer mode selected by
    ``config``. Storage placement is adopted before loader metadata is restored;
    the model and its modules keep their identity.

    If this function raises, cleanup preserves storage ownership and removes
    FlexTensor modifications where possible, but does not roll back storage
    placement. The input model is not guaranteed usable and must be discarded.

    Args:
        model: Eager model matching the saved state.
        state: Validated state to adopt.
        config: Offload configuration. Uses the manager's current configuration when omitted.
        name: Offload manager name.
        allow_strategy_replan: Preserve source weights for a later compiled-timing replan.

    Returns:
        An inference-ready proxy for ``model``.
    """
    return get_offload_manager(name).offload_from_state(
        model,
        state,
        config=config,
        allow_strategy_replan=allow_strategy_replan,
    )


def offload_from_profile(
    model: nn.Module,
    profile_directory: str,
    config: OffloadConfig | None = None,
    name: str = DEFAULT_MANAGER_NAME,
    compile_fn: Callable[[nn.Module], nn.Module] | None = None,
) -> nn.Module:
    """Initialize, load a saved profile, and offload the model in one step.

    Convenience wrapper that combines :func:`init`, :func:`load_profile`, and
    :func:`offload`.  The profile must be loaded *before* offloading so that
    ``initialize_warmup`` sees the saved state and skips discovery/profiling,
    going straight to inference mode.

    Args:
        model: PyTorch model to offload.
        profile_directory: Directory containing the saved profile.
        config: OffloadConfig to apply.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
        compile_fn: Optional per-unit compiler (same as :func:`offload`).

    Returns:
        The offloaded model proxy, ready for inference.
    """
    init(config=config, name=name)
    load_profile(profile_directory, model=model, name=name)
    return offload(model, config=config, name=name, compile_fn=compile_fn)


def offload_block(block_name: str, name: str = DEFAULT_MANAGER_NAME) -> Any:
    """Create offload block using the specified offload manager.

    Args:
        block_name: Name for the offload block.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    return om.offload_block(block_name)


def release(name: str = DEFAULT_MANAGER_NAME, *, preserve_model: bool = True):
    """Release the specified offload manager.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
        preserve_model: Restore loader-managed tensor values before teardown.
            Set to ``False`` only when the managed model will be discarded.
    """
    om = get_offload_manager(name)
    om.release(preserve_model=preserve_model)


def get_gpu_memory_usage(name: str = DEFAULT_MANAGER_NAME) -> GPUMemoryUsage:
    """Get GPU memory usage by the specified offload manager.

    Returns the memory used by GPU transfer blocks and unmapped tensors.
    Must be called after the manager has transitioned to inference mode.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.

    Returns:
        GPUMemoryUsage dataclass with memory breakdown in bytes and megabytes.

    Raises:
        RuntimeError: If called before inference mode.
    """
    om = get_offload_manager(name)
    return om.get_gpu_memory_usage()


def reset_offload_timing(name: str = DEFAULT_MANAGER_NAME) -> None:
    """Start a fresh durable offload-timing measure window.

    Convenience wrapper around :meth:`OffloadManager.reset_offload_timing`.
    """
    om = get_offload_manager(name)
    om.reset_offload_timing()


def collect_offload_timing(name: str = DEFAULT_MANAGER_NAME) -> OffloadTimingReport | None:
    """Collect aggregate offload timing from the durable measure store.

    Convenience wrapper around
    :meth:`OffloadManager.collect_offload_timing`.
    After CUDA-graph replay, call :func:`update_offload_timing` first so
    passes are published; this drains the store and does not re-finalize.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.

    Returns:
        :class:`~flextensor.offload_timing.OffloadTimingReport`, or ``None``
        when timing is ``"off"``, the manager has no ``TensorManager`` yet, or
        the durable measure store is empty after flush.
    """
    om = get_offload_manager(name)
    return om.collect_offload_timing()


def update_offload_timing(name: str = DEFAULT_MANAGER_NAME, *, replay_generation: int = -1) -> bool:
    """Store the current offload-timing pass into the collector.

    Convenience wrapper around :meth:`OffloadManager.update_offload_timing`.
    Measurement only. Call after each ``graph.replay()`` you want to keep
    (or :func:`update_state` does it when replan used ``manual_update_state=True``).

    Returns:
        ``True`` on success / timing inactive; ``False`` if finalize failed.
    """
    om = get_offload_manager(name)
    return om.update_offload_timing(replay_generation=replay_generation)


def offload_units(name: str = DEFAULT_MANAGER_NAME) -> tuple[tuple[str, nn.Module], ...]:
    """Return outermost offloaded modules as ``(qualified_name, module)`` pairs.

    Convenience wrapper around :meth:`OffloadManager.offload_units`.
    """
    return get_offload_manager(name).offload_units()


def resident_units(name: str = DEFAULT_MANAGER_NAME) -> tuple[tuple[str, nn.Module], ...]:
    """Return outermost GPU-resident modules as ``(qualified_name, module)`` pairs.

    Convenience wrapper around :meth:`OffloadManager.resident_units`.
    """
    return get_offload_manager(name).resident_units()


def request_strategy_replan(name: str = DEFAULT_MANAGER_NAME, *, manual_update_state: bool = False) -> int:
    """Remeasure under the current runtime and rebuild the offload strategy.

    Convenience wrapper around :meth:`OffloadManager.request_strategy_replan`.

    Args:
        name: Offload manager name. Defaults to DEFAULT_MANAGER_NAME.
        manual_update_state: Caller-driven measure path — arm post-capture
            timing and require :func:`update_state` after each opaque step
            (today: ``graph.replay()``). After rebuild, recapture CUDA graphs
            before serving.

    Returns:
        Forwards / replays to drive, or ``0`` when no replan was armed.
    """
    om = get_offload_manager(name)
    return om.request_strategy_replan(manual_update_state=manual_update_state)


def update_state(name: str = DEFAULT_MANAGER_NAME, *, replay_generation: int = -1) -> None:
    """Advance offload phase / armed replan by one step.

    Convenience wrapper around :meth:`OffloadManager.update_state`. Needed
    explicitly after ``graph.replay()`` when replan was armed with
    ``manual_update_state=True`` (replay does not run module forward hooks).

    Args:
        name: Offload manager name.
        replay_generation: Optional CUDA-graph replay generation for the
            manual measure path (dedupes with a gen-aware serving hook).
    """
    om = get_offload_manager(name)
    om.update_state(replay_generation=replay_generation)


# ---------------------------------------------------------------------------
# Profiling data control API
# ---------------------------------------------------------------------------


def clear_profiling_durations(name: str = DEFAULT_MANAGER_NAME) -> None:
    """Clear all accumulated profiling duration measurements.

    Wipes every duration sample collected so far.  Tensor-to-layer
    mappings (discovery data) are **not** affected.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    om.clear_profiling_durations()


def suspend_profiling(name: str = DEFAULT_MANAGER_NAME) -> None:
    """Suppress profiling recording (durations and tensor IDs).

    In PROFILING, suppression is total — ``TensorManager.record_all()`` is
    a complete no-op, so a paused warmup pass cannot widen per-layer tensor
    sets on data-dependent models (MoE / conditional branches / mixed-batch
    shapes). In DISCOVERY, ``record_tensors()`` is **not** affected (the
    tensor-to-layer mapping is a hard prerequisite for every later phase).
    The iteration counter is frozen in PROFILING so suppressed passes don't
    consume the ``profiling_iters`` budget. See
    :meth:`OffloadManager.suspend_profiling` for the full per-phase contract.

    Suspensions are reference-counted: each :func:`suspend_profiling` must
    be matched by a :func:`resume_profiling`, and recording only resumes
    once every outstanding suspension has been released.  Independent
    callers can therefore bracket their own sections without accidentally
    resuming someone else's suspension.

    .. note::
        With ``OffloadConfig(enabled=False)`` the active manager is
        :class:`NoOpTensorManager`: these functions are no-ops and the
        profiling-iteration counter is not frozen.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.

    Example::

        model = flextensor.offload(model, config=config)

        flextensor.suspend_profiling()
        for size in [1, 8, 32, 128]:
            model(make_dummy_input(size))   # profiling suppressed during warmup
        flextensor.resume_profiling()

        for batch in dataloader:            # durations are now collected for the real workload
            output = model(batch)
    """
    om = get_offload_manager(name)
    om.suspend_profiling()


def resume_profiling(name: str = DEFAULT_MANAGER_NAME) -> None:
    """Resume profiling duration recording after :func:`suspend_profiling`.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    om.resume_profiling()


# See `flextensor.helpers.ProfilingSuspender.suspended` for the `-> Any` rationale
# (beartype + @contextmanager cross-version compatibility).
@contextmanager
def pause_profiling(name: str = DEFAULT_MANAGER_NAME) -> Any:
    """Context-manager form of :func:`suspend_profiling` / :func:`resume_profiling`.

    Shares the refcount with the raw functions, so it nests freely with them.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.

    Example::

        model = flextensor.offload(model, config=config)

        with flextensor.pause_profiling():
            for size in [1, 8, 32, 128]:
                model(make_dummy_input(size))   # profiling suppressed during warmup

        for batch in dataloader:                # durations are now collected for the real workload
            output = model(batch)
    """
    om = get_offload_manager(name)
    with om.pause_profiling():
        yield

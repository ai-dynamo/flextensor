# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Mapping  # noqa: TC003
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from torch.overrides import TorchFunctionMode

from flextensor.trap_tensor_mode import _graph_break
from flextensor.types import GPUMemoryUsage

if TYPE_CHECKING:
    from flextensor.offload_manager import TensorManagerProtocol as _TensorManagerProtocol


def format_tensor_id_hint(
    tensor_ids: Iterable[int],
    id_to_name: Mapping[int, str] | None = None,
    *,
    head: int = 8,
) -> str:
    """Format a tensor-id collection for log hints with optional name resolution.

    Sorts the ids, formats up to ``head`` of them inline (resolving each via
    ``id_to_name`` when supplied), and collapses any remainder to
    ``", ... (N more)"``. Used by warning paths that want to point operators
    at specific affected tensors without dumping arbitrarily long id lists.

    Names make the hint actionable: ``lm_head.weight (id=140234517123920)``
    is greppable in the source; the raw integer is not.

    Args:
        tensor_ids: Tensor ids to format. Sorted for stable log output.
        id_to_name: Optional id → parameter-name mapping used to resolve each
            id to a greppable name. Ids missing from the mapping fall back to
            the bare ``id=<int>`` form.
        head: Maximum number of ids to format inline before collapsing the
            remainder into a ``", ... (N more)"`` suffix.

    Returns:
        A comma-separated hint string, empty when *tensor_ids* is empty.

    Example:
        >>> format_tensor_id_hint([2, 1], {1: "lm_head.weight"}, head=1)
        'lm_head.weight (id=1), ... (1 more)'
    """
    sorted_ids = sorted(tensor_ids)

    def fmt(tid: int) -> str:
        if id_to_name is not None and (name := id_to_name.get(tid)):
            return f"{name} (id={tid})"
        return f"id={tid}"

    hint = ", ".join(fmt(tid) for tid in sorted_ids[:head])
    if len(sorted_ids) > head:
        hint += f", ... ({len(sorted_ids) - head} more)"
    return hint


class TrapNestingGuard:
    """Prevents nested trap entry on a shared pair of CUDA timing events.

    Traps execute sequentially (one per layer in forward()). A second
    ``acquire()`` before the matching ``release()`` would overwrite the
    start-event timestamp and silently corrupt timing, so we fail fast.

    One instance lives on ``TensorManager``; each trap calls
    ``acquire(trace_id)`` / ``release()`` via that shared instance.
    """

    def __init__(self) -> None:
        self._active = False

    def acquire(self, trace_id: str) -> None:
        """Mark a trap as active, or raise if one is already active.

        Args:
            trace_id: Name of the trap being entered (included in the error message).

        Raises:
            RuntimeError: If called while another trap is already active.
        """
        if self._active:
            raise RuntimeError(
                f"Nested traps are not supported: trap '{trace_id}' entered while another "
                f"trap is active. Shared CUDA events would produce incorrect timing."
            )
        self._active = True

    def release(self) -> None:
        self._active = False

    def is_active(self) -> bool:
        """Whether a trap is currently active (between acquire and release)."""
        return self._active


class ProfilingSuspender:
    """Reference-counted suspension of profiling duration recording.

    Owns a single non-negative counter.  :meth:`suspend` increments it and
    :meth:`resume` decrements it; :meth:`is_suspended` returns ``True`` iff
    the counter is above zero.  Recording is therefore only re-enabled once
    every outstanding suspension has been released, which lets independent
    callers bracket their own sections without accidentally resuming
    someone else's suspension.

    Invariants:

    * The counter never goes negative. :meth:`resume` on a non-suspended
      suspender raises :class:`RuntimeError` — almost always a sign of
      unbalanced ``suspend`` / ``resume`` calls, and the traceback points
      directly at the offending call site.
    * Prefer :meth:`suspended` over raw :meth:`suspend` / :meth:`resume`
      where possible; the context manager guarantees balance even on
      exceptions.

    One instance lives on ``TensorManager``; the public
    ``suspend_profiling()`` / ``resume_profiling()`` / ``pause_profiling()``
    API delegates to it.
    """

    def __init__(self) -> None:
        self._count = 0

    def suspend(self) -> None:
        self._count += 1

    def resume(self) -> None:
        if self._count == 0:
            raise RuntimeError(
                "resume_profiling() called while profiling was not suspended; "
                "suspend_profiling() / resume_profiling() calls are unbalanced."
            )
        self._count -= 1

    # Return type is `Any` instead of `Iterator[None]` or `AbstractContextManager[None]` due to
    # beartype compatibility issues across Python versions. The `@contextmanager` decorator returns
    # a `_GeneratorContextManager` which beartype fails to validate against `Iterator`/`Generator`
    # on Python 3.10, while `AbstractContextManager` fails on Python 3.11+.
    @contextmanager
    def suspended(self) -> Any:
        """Context manager form; guarantees balance on exceptions."""
        self.suspend()
        try:
            yield
        finally:
            self.resume()

    def is_suspended(self) -> bool:
        return self._count > 0


class NoOpTrap:
    def __init__(self):
        pass

    def __enter__(self):
        _graph_break()
        return self

    def __exit__(self, _type, _value, _traceback):
        _graph_break()
        return False


class EmptyFunctionModeTrap(TorchFunctionMode):
    def __init__(self):
        pass

    def __enter__(self):
        _graph_break()
        return super().__enter__()

    def __exit__(self, _type, _value, _traceback):
        _graph_break()
        return super().__exit__(_type, _value, _traceback)

    def __torch_function__(self, func, _types, args, kwargs=None):
        return func(*args, **(kwargs or {}))


if TYPE_CHECKING:
    # Keep the structural contract visible near the no-op implementation.
    _NOOP_TENSOR_MANAGER_PROTOCOL: type[_TensorManagerProtocol]


class NoOpTensorManager:
    """Stand-in ``TensorManager`` used when ``OffloadConfig(enabled=False)``.

    All profiling-control hooks (``suspend_profiling`` / ``resume_profiling``
    / ``pause_profiling`` / ``clear_profiling_durations``) are no-ops because
    no durations are ever recorded. In particular, :meth:`is_profiling_suspended`
    permanently returns ``False``, so callers should not assume the
    suspend/iteration semantics documented for the real ``TensorManager``
    (see ``docs/explanation/phases.md``) apply when offloading is disabled —
    ``OffloadManager.update_state()`` will not freeze the profiling counter
    in this mode.
    """

    def __init__(
        self,
        device_gpu,
        benchmark_cls=None,
    ):
        self.device_gpu = device_gpu
        self.benchmark_cls = benchmark_cls
        self.traps_duration_ms = 0
        self.traps_direct_duration_ms = 0
        self.traps_direct_duration_ms = 0
        self.traps_direct_stats = {}
        self.tensor_statistics_map = {}
        self.tensors_map = {}
        self.traced_tensors = set()
        self.loader_type = ""
        self.shm_namespace: str | None = None
        self.stats: list[Any] = []

    def build_parameters_mapping(self, _model):
        pass

    def prepare_warmup_mode(self):
        pass

    def prepare_profile_mode(self):
        pass

    def prepare_profile_direct_mode(self):
        pass

    def prepare_infer_mode(self):
        pass

    def prepare_infer_load_mode(self) -> None:
        pass

    def sync_prev_onload(self) -> None:
        pass

    def join_after_forward(self) -> None:
        pass

    def is_profiling_suspended(self) -> bool:
        return False

    def clear_profiling_durations(self) -> None:
        pass

    def suspend_profiling(self) -> None:
        pass

    def resume_profiling(self) -> None:
        pass

    # See `ProfilingSuspender.suspended` for the `-> Any` rationale
    # (beartype + @contextmanager cross-version compatibility).
    @contextmanager
    def pause_profiling(self) -> Any:
        yield

    def trap(self, _name):
        return NoOpTrap()

    def release_memory(self):
        self.traps_direct_duration_ms = 0

    def arm_non_destructive_first_loader(self) -> None:
        pass

    @property
    def replan_source_weights_available(self) -> bool:
        # Disabled offload never builds a loader, so there is nothing to replan.
        return False

    def clear_replan_state(self) -> None:
        pass

    def prepare_profile_direct_mode_model(self, model):
        return model

    def prepare_model(self, model):
        return model

    def prepare_final_model(self, model: Any, *, in_place: bool = False) -> Any:
        return model

    def benchmark_context(self, _iterations: int = 10):
        return self.benchmark_cls(device_gpu=self.device_gpu)

    def set_model(self, model):
        self.model = model

    def initialize_warmup(self):
        return self.model

    def initialize_profile(self):
        return self.model

    def initialize_inference(self):
        return self.model

    def shutdown(self):
        pass

    def restore_state(self, model: Any, _state: Any) -> None:
        # Match TensorManager's profile-restore contract: later initialize_* calls
        # return the model associated with the restored state.
        self.model = model

    def plan_state_adoption(self, _model: Any, _state: Any) -> Any:
        from flextensor.state_transition import StateTransitionPlan

        return StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)

    def execute_state_adoption(self, _model: Any, _plan: Any) -> None:
        pass

    def restore_adopted_state(self, model: Any, _state: Any) -> None:
        self.model = model

    def save_profile(self, _profile_directory: str) -> None:
        pass

    def load_profile(self, _profile_directory: str, model: Any) -> None:
        # Disabled offload has no profile to load, but callers can still use the
        # load_profile -> initialize_* sequence and expect the target model back.
        self.model = model

    def get_memory_transfer_stats(self) -> dict[int, float] | None:
        return None

    def collect_offload_timing(self) -> Any:
        return None

    def get_gpu_memory_usage(self) -> GPUMemoryUsage:
        """Get GPU memory usage (returns zeros for disabled offload).

        When offload is disabled, no FlexTensor memory is allocated,
        so this returns a GPUMemoryUsage with all zero values.

        Returns:
            GPUMemoryUsage: All zero values
        """
        return GPUMemoryUsage(blocks_bytes=0, unmapped_tensors_bytes=0, total_bytes=0)

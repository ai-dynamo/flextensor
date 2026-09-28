# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import torch

from flextensor.allocation_block import AllocationBlock, AllocationManager
from flextensor.compiler_utils import disable as _compiler_disable
from flextensor.compiler_utils import is_compiling as _is_compiling
from flextensor.construction_rollback import ConstructionRollback
from flextensor.core.transfer import (
    CudaTransferRuntime,
    TransferTicket,
    synchronize_transfer_stream,
)
from flextensor.core.transfer import (
    TransferStreamSynchronizationError as _TransferStreamSynchronizationError,
)
from flextensor.helpers import format_tensor_id_hint
from flextensor.host_pinning import HostPinner
from flextensor.instrumentation import instrumentable
from flextensor.offload_timing import OffloadTimingCollector
from flextensor.piecewise_prefetch_policy import PiecewisePrefetchPolicy
from flextensor.strategy_operations import find_transfers_for_preload

from .collectors import IterativeLayerStatistics, LayerStatistics, TensorStatistics
from .utils import clear_and_delete_tensor, delete_tensor, is_dense_layout

LOGGER = logging.getLogger(__name__)

TransferStreamSynchronizationError = _TransferStreamSynchronizationError
_CONSTRUCTION_CLEANUP_ATTEMPTS = 2


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_to_block_ids(
    label_to_block_id: dict[str, int],
    transfer_to_compute_map: dict[str, str],
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for transfer_label, compute_label in transfer_to_compute_map.items():
        result.setdefault(compute_label, []).append(label_to_block_id[transfer_label])
    return result


def _compute_preload(
    layer_stats: list[LayerStatistics],
    strategy: dict[str, list[TensorStatistics]],
) -> set[int]:
    """Compute tensor IDs that need to be preloaded.

    Args:
        layer_stats: Layer statistics.
        strategy: Strategy mapping layer labels to tensors to offload.

    Returns:
        Set of tensor IDs that are not included in the strategy.
    """
    preload_tensors_ids: set[int] = set()
    strategy_tensors_ids: set[int] = set()
    for layer_stat in layer_stats:
        # add tensors from current layer
        label = layer_stat.label
        if label in strategy:
            layer_strategy = strategy[label]
            for tensor_info in layer_strategy:
                strategy_tensors_ids.add(tensor_info.tensor_id)
        # check tensors which are not included in the strategy layer
        for tensor_info in layer_stat.tensors:
            if tensor_info.tensor_id not in strategy_tensors_ids:
                preload_tensors_ids.add(tensor_info.tensor_id)
    return preload_tensors_ids


def _compute_untimed_traced_preload(
    layer_stats: list[LayerStatistics],
    tensors_map: Mapping[int, torch.Tensor],
) -> set[int]:
    """Tensor IDs in ``tensors_map`` that appear in no layer at all.

    These are *traced but untimed* — their trap fired (so the tensor is
    registered in ``tensors_map`` and ``traced_tensors``) but no duration
    sample was recorded, so the label was dropped by
    :func:`flextensor.tensor_manager.compute_layer_statistics`. Without
    this rescue, ``TensorStrategyLoader.get(...)`` returns ``None`` for
    them and the trap path falls through to the original CPU tensor
    (a device-mismatch crash for parameterized GPU ops, or silent CPU
    compute when all operands happen to be CPU).

    Block loaders don't need an analogous rescue:
    ``MoveUnmappedTensorsToGPUProcessor`` already routes any non-view-mapped
    tensor to GPU as a *routing* decision (mapped → block view, unmapped →
    GPU pin), so the strategy-loader fall-through-to-CPU failure mode
    cannot occur there. The cost on the strategy path is permanent GPU
    residence for the rescued IDs (they sit in ``cpu_to_gpu_map`` for the
    loader's lifetime).

    The typical trigger is a profile-coverage gap (e.g. vLLM's
    ``logits_processor`` firing only at decode-time but profiled with
    prefill-shaped inputs). Anchored by
    ``tests/unit/test_untimed_traps_runtime.py``.

    Note: ``TensorStrategyLoader`` further narrows this set against
    the model's reachable tensor IDs (passed via
    ``reachable_tensor_ids``) so that ``tensors_map`` entries from any
    future non-model code path don't get moved to GPU as a side effect.
    """
    layer_tensor_ids: set[int] = set()
    for layer_stat in layer_stats:
        for tensor_info in layer_stat.tensors:
            layer_tensor_ids.add(tensor_info.tensor_id)
    return {tid for tid in tensors_map if tid not in layer_tensor_ids}


def _compute_untimed_traced_preload_iterative(
    layer_stats: list[IterativeLayerStatistics],
    tensors_map: Mapping[int, torch.Tensor],
) -> set[int]:
    """IDs in ``tensors_map`` absent from every iterative layer's ``tensor_ids``.

    Iterative-stats counterpart of :func:`_compute_untimed_traced_preload`;
    feeds :class:`UntimedTrapRescuer`.
    """
    layer_tensor_ids: set[int] = set()
    for stat in layer_stats:
        layer_tensor_ids.update(stat.tensor_ids)
    return {tid for tid in tensors_map if tid not in layer_tensor_ids}


def _compute_peak_memory_from_strategy(
    strategy_map: dict[str, list[TensorStatistics]],
    release_strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
) -> int:
    """Compute peak GPU memory by simulating the sliding window execution.

    This function simulates the tensor loading and release pattern during inference
    to determine the maximum memory required at any point. It iterates through layers
    in execution order, tracking which tensors are loaded (from strategy_map) and
    released (from release_strategy_map).

    Args:
        strategy_map: Dictionary mapping layer labels to lists of TensorStatistics
            representing tensors to load when entering that layer.
        release_strategy_map: Dictionary mapping layer labels to lists of TensorStatistics
            representing tensors to release when exiting that layer.
        layer_stats: List of LayerStatistics defining the execution order.

    Returns:
        Peak GPU memory in bytes - the maximum total size of tensors simultaneously
        loaded at any point during execution.
    """
    if not layer_stats:
        return 0

    # Track currently loaded tensors by their ID to avoid double-counting
    loaded_tensor_sizes: dict[int, int] = {}
    peak_memory = 0
    current_memory = 0

    for layer in layer_stats:
        label = layer.label

        # Load tensors from strategy_map when entering the layer
        if label in strategy_map:
            for tensor_info in strategy_map[label]:
                tensor_id = tensor_info.tensor_id
                if tensor_id not in loaded_tensor_sizes:
                    loaded_tensor_sizes[tensor_id] = tensor_info.size_bytes
                    current_memory += tensor_info.size_bytes

        # Update peak after loading (this is when memory is highest for this layer)
        peak_memory = max(peak_memory, current_memory)

        # Release tensors from release_strategy_map when exiting the layer
        if label in release_strategy_map:
            for tensor_info in release_strategy_map[label]:
                tensor_id = tensor_info.tensor_id
                if tensor_id in loaded_tensor_sizes:
                    current_memory -= loaded_tensor_sizes[tensor_id]
                    del loaded_tensor_sizes[tensor_id]

    return peak_memory


class Loader(ABC):
    @abstractmethod
    def enter(self, label: str) -> None:
        """
        Called when a particular block/tensor group is ready for compute/transfer.
        """

    @abstractmethod
    def exit(self, label: str) -> None:
        """
        Called when a particular block/tensor group is finished with compute/transfer.
        """

    def shutdown(self) -> None:
        """
        Release resources and clean up allocations.
        """
        return None


class _RawTensorDataBinder:
    """Temporarily point original tensor objects at active materialized storage."""

    def __init__(self, tensors_map: Mapping[int, torch.Tensor]) -> None:
        self.tensors_map = tensors_map
        self._original_data: dict[int, torch.Tensor] = {}
        self._shared_storage_ids = self._find_shared_storage_ids(tensors_map)

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
        return (str(tensor.device), tensor.untyped_storage().data_ptr())

    @classmethod
    def _find_shared_storage_ids(cls, tensors_map: Mapping[int, torch.Tensor]) -> dict[int, set[int]]:
        storage_to_tensors: dict[tuple[str, int], list[tuple[int, torch.Tensor]]] = {}
        for tensor_id, tensor in tensors_map.items():
            storage_to_tensors.setdefault(cls._storage_key(tensor), []).append((tensor_id, tensor))

        shared_storage_ids: dict[int, set[int]] = {}
        for tensors in storage_to_tensors.values():
            for index, (tensor_id, tensor) in enumerate(tensors):
                for other_id, other in tensors[index + 1 :]:
                    if not cls._storage_regions_may_overlap(tensor, other):
                        continue
                    shared_storage_ids.setdefault(tensor_id, {tensor_id}).add(other_id)
                    shared_storage_ids.setdefault(other_id, {other_id}).add(tensor_id)
        return shared_storage_ids

    @staticmethod
    def _storage_regions_may_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
        """Conservatively detect overlap between two views of one storage."""

        def byte_bounds(tensor: torch.Tensor) -> tuple[int, int] | None:
            if tensor.numel() == 0:
                return (0, 0)
            if tensor.layout != torch.strided or any(stride < 0 for stride in tensor.stride()):
                return None
            first_element = tensor.storage_offset()
            last_element = first_element + sum(
                (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            )
            element_size = tensor.element_size()
            return first_element * element_size, (last_element + 1) * element_size

        left_bounds = byte_bounds(left)
        right_bounds = byte_bounds(right)
        if left_bounds is None or right_bounds is None:
            return True
        return max(left_bounds[0], right_bounds[0]) < min(left_bounds[1], right_bounds[1])

    def _validate_bind(self, tensor_id: int, tensor: torch.Tensor, active_tensor: torch.Tensor) -> None:
        shared_ids = self._shared_storage_ids.get(tensor_id)
        if shared_ids is not None:
            msg = (
                f"Cannot bind tensor id {tensor_id}; managed tensor ids {sorted(shared_ids)} share storage "
                "with overlapping byte regions. Overlapping shared-storage parameters are not supported "
                "by raw materialization."
            )
            raise RuntimeError(msg)

        if active_tensor.layout != tensor.layout:
            msg = f"Materialized tensor layout {active_tensor.layout} does not match original layout {tensor.layout}."
            raise RuntimeError(msg)
        if active_tensor.dtype != tensor.dtype:
            msg = f"Materialized tensor dtype {active_tensor.dtype} does not match original dtype {tensor.dtype}."
            raise RuntimeError(msg)
        if active_tensor.shape != tensor.shape:
            msg = f"Materialized tensor shape {active_tensor.shape} does not match original shape {tensor.shape}."
            raise RuntimeError(msg)
        if active_tensor.stride() != tensor.stride():
            active_stride = active_tensor.stride()
            original_stride = tensor.stride()
            msg = f"Materialized tensor stride {active_stride} does not match original stride {original_stride}."
            raise RuntimeError(msg)

    @_compiler_disable
    def bind(self, tensor_id: int, active_tensor: torch.Tensor | None) -> None:
        tensor = self.tensors_map.get(tensor_id)
        if tensor is None or active_tensor is None or tensor_id in self._original_data:
            return
        self._validate_bind(tensor_id, tensor, active_tensor)
        with torch.no_grad():
            self._original_data[tensor_id] = tensor.data
            tensor.data = active_tensor

    @_compiler_disable
    def restore(self, tensor_ids: set[int]) -> None:
        for tensor_id in tensor_ids:
            original_data = self._original_data.pop(tensor_id, None)
            tensor = self.tensors_map.get(tensor_id)
            if tensor is not None and original_data is not None:
                with torch.no_grad():
                    tensor.data = original_data

    @_compiler_disable
    def restore_all(self) -> None:
        self.restore(set(self._original_data))


@instrumentable
class WarmupDirectTensorLoader(Loader):
    """Per-trap tensor loader for direct-mode discovery.

    Discovery normally records tensor IDs without building a full offload
    strategy yet. Direct-mode models still need a loader so attribute access
    sites that bypass torch dispatch, such as Triton custom kernels, see a
    materialized tensor while the current trap is executing.
    """

    def __init__(
        self,
        label_to_tensor_ids: dict[str, set[int]],
        tensors_map: Mapping[int, torch.Tensor],
        device_gpu: torch.device,
    ) -> None:
        self.label_to_tensor_ids = label_to_tensor_ids
        self.tensors_map = tensors_map
        self.device_gpu = device_gpu
        self.cpu_to_gpu_map: dict[int, torch.Tensor] = {}
        self._active_counts: dict[int, int] = {}
        self._active_labels: list[str] = []
        self._borrowed_by_label: dict[str, set[int]] = {}
        self._accessed_by_label: dict[str, set[int]] = {}
        self._data_binder = _RawTensorDataBinder(tensors_map)

    def get_label_tensor_ids(self, label: str) -> set[int]:
        """Return the tensor IDs that belong to *label*."""
        return set(self.label_to_tensor_ids.get(label, set()))

    def get_accessed_tensor_ids(self, label: str) -> set[int]:
        """Return tensor IDs read through direct getters while *label* was active."""
        return set(self._accessed_by_label.get(label, set()))

    @property
    def _active_label(self) -> str | None:
        return self._active_labels[-1] if self._active_labels else None

    def _retain_tensor(self, tensor_id: int) -> torch.Tensor | None:
        tensor = self.tensors_map.get(tensor_id)
        if tensor is None:
            return None

        count = self._active_counts.get(tensor_id, 0)
        if count:
            self._active_counts[tensor_id] = count + 1
            return self.cpu_to_gpu_map[tensor_id]

        active_tensor = tensor.to(device=self.device_gpu, copy=True)
        # Custom kernels may read the original Parameter object directly
        # instead of going through direct-mode property getters.
        if not _is_compiling():
            self._data_binder.bind(tensor_id, active_tensor)
        self.cpu_to_gpu_map[tensor_id] = active_tensor
        self._active_counts[tensor_id] = 1
        return active_tensor

    def _release_tensor(self, tensor_id: int) -> None:
        count = self._active_counts.get(tensor_id, 0)
        if count <= 0:
            return
        if count > 1:
            self._active_counts[tensor_id] = count - 1
            return

        self._active_counts.pop(tensor_id, None)
        if not _is_compiling():
            self._data_binder.restore({tensor_id})
        self.cpu_to_gpu_map.pop(tensor_id, None)

    def enter(self, label: str) -> None:
        """Materialize every tensor owned by the active discovery trap."""
        self._active_labels.append(label)
        tensor_ids = self.get_label_tensor_ids(label)

        for tensor_id in tensor_ids:
            self._retain_tensor(tensor_id)

    def exit(self, label: str) -> None:
        """Release tensors materialized for *label*."""
        tensor_ids = self.get_label_tensor_ids(label) | self._borrowed_by_label.pop(label, set())
        if self.device_gpu.type == "cuda" and tensor_ids:
            torch.cuda.synchronize(self.device_gpu)
        for tensor_id in tensor_ids:
            self._release_tensor(tensor_id)

        self._accessed_by_label.pop(label, None)
        if self._active_labels and self._active_labels[-1] == label:
            self._active_labels.pop()
        elif label in self._active_labels:
            self._active_labels.remove(label)

    def get(self, tensor_id: int) -> torch.Tensor | None:
        """Return the materialized tensor for *tensor_id*, when active."""
        active_label = self._active_label
        if active_label is not None:
            self._accessed_by_label.setdefault(active_label, set()).add(tensor_id)
            if tensor_id not in self.cpu_to_gpu_map and tensor_id in self.tensors_map:
                self._borrowed_by_label.setdefault(active_label, set()).add(tensor_id)
                return self._retain_tensor(tensor_id)
        return self.cpu_to_gpu_map.get(tensor_id)

    def release_memory(self) -> None:
        """Release any tensors left materialized by an interrupted warmup."""
        if self.device_gpu.type == "cuda" and self.cpu_to_gpu_map:
            torch.cuda.synchronize(self.device_gpu)
        if not _is_compiling():
            self._data_binder.restore_all()
        self.cpu_to_gpu_map.clear()
        self._active_counts.clear()
        self._active_labels.clear()
        self._borrowed_by_label.clear()
        self._accessed_by_label.clear()


class UntimedTrapRescuer:
    """Force-pin traced-but-untimed tensors to GPU for a loader's lifetime.

    "Traced but untimed" = present in ``tensors_map`` but absent from every
    layer's ``tensor_ids``. The rescuer copies them to GPU once at
    construction and is consulted by :meth:`TensorLayerLoader.get` on a
    cache miss.

    Complements the runtime cross-module fallback in
    :func:`flextensor.tensor_manager._make_tensor_getter`, which stays as
    the safety net for positional-argument reads (e.g. vLLM
    ``logits_processor(lm_head, …)``) that the static handoff cannot see.
    ``UntimedTrapRescuer`` only covers ids the strategy has *no* row for;
    the view-mode profile path (``ProfileBlockController`` +
    :class:`flextensor.tensor_processors.MoveUnmappedTensorsToGPUProcessor`)
    handles unmapped placement instead and does not use this rescuer.

    Trade-off: permanent GPU residence for rescued ids, in exchange for no
    per-trap H2D copy and no trap-duration taint.
    """

    def __init__(
        self,
        layer_stats: list[IterativeLayerStatistics],
        tensors_map: Mapping[int, torch.Tensor],
        device_gpu: torch.device,
        *,
        reachable_tensor_ids: set[int] | None = None,
        del_tensor_func: Callable[[torch.Tensor], None] = clear_and_delete_tensor,
        id_to_name_map: Mapping[int, str] | None = None,
    ) -> None:
        self.device_gpu = device_gpu
        self.pinned: dict[int, torch.Tensor] = {}
        self.owned_ids: set[int] = set()
        self.del_tensor = del_tensor_func

        rescue_ids = _compute_untimed_traced_preload_iterative(layer_stats, tensors_map)
        if reachable_tensor_ids is not None:
            rescue_ids &= reachable_tensor_ids
        if not rescue_ids:
            return

        rescue_bytes = sum(tensors_map[tid].numel() * tensors_map[tid].element_size() for tid in rescue_ids)

        try:
            for tensor_id in rescue_ids:
                tensor = tensors_map[tensor_id]
                if tensor.device != device_gpu:
                    self.pinned[tensor_id] = tensor.to(device=device_gpu, copy=True)
                    self.owned_ids.add(tensor_id)
                else:
                    self.pinned[tensor_id] = tensor
        except torch.cuda.OutOfMemoryError as e:
            self.shutdown()
            raise torch.cuda.OutOfMemoryError(
                f"GPU out of memory during untimed-trap rescue: pinning {len(rescue_ids)} tensor(s) "
                f"({rescue_bytes / 1024**3:.2f} GiB) missed by profile coverage."
            ) from e
        torch.cuda.synchronize()

        LOGGER.warning(
            "Untimed-trap rescue activated: %d tensor(s) (%.2f MiB) force-pinned to GPU "
            "[tensors: %s]. Likely a profile-coverage gap.",
            len(rescue_ids),
            rescue_bytes / (1024 * 1024),
            format_tensor_id_hint(rescue_ids, id_to_name_map),
        )

    def get(self, tensor_id: int) -> torch.Tensor | None:
        """Return the rescued GPU copy for ``tensor_id``, or ``None``."""
        return self.pinned.get(tensor_id)

    def pin(self, tensor_id: int, tensor: torch.Tensor) -> None:
        """Adopt ``tensor`` into the pinned set as passthrough (shared storage).

        Runtime callers (currently ``TensorLayerLoader.exit``) only ever
        adopt tensors already owned by the model, so the rescuer must not
        run ``del_tensor_func`` on them at shutdown — the model still
        references the same storage. Owned copies are only produced by
        ``__init__``; there is no supported path for a caller to transfer
        ownership after construction.
        """
        self.pinned[tensor_id] = tensor

    def shutdown(self) -> None:
        """Drop references to all rescued GPU tensors. Idempotent.

        Owned entries (fresh copies) go through ``del_tensor_func``;
        passthrough entries share storage with the model, so only the
        local reference is dropped.
        """
        if not self.pinned:
            return
        for tensor_id in list(self.pinned.keys()):
            gpu_tensor = self.pinned.pop(tensor_id)
            if tensor_id in self.owned_ids:
                self.del_tensor(gpu_tensor)
            else:
                del gpu_tensor
        self.owned_ids.clear()


@instrumentable
class TensorLayerLoader(Loader):
    """
    This loader manages tensor loading for specific layers within a trap/block context.
    It loads all required tensors when entering a layer and releases them upon exit.

    Note: This loader uses clear_and_delete_tensor by default for more aggressive memory release.
    It has been observed that the standard delete_tensor function does not work effectively
    in this context, potentially leading to insufficient cleanup.
    Further investigation is needed.
    """

    def __init__(
        self,
        layer_stats: list[IterativeLayerStatistics],
        tensors_map: Mapping[int, torch.Tensor],
        device_gpu: torch.device,
        delete_tensor_func: Callable[[torch.Tensor], None] = clear_and_delete_tensor,
        *,
        rescuer: UntimedTrapRescuer | None = None,
    ) -> None:
        """Construct a layer-scoped tensor loader.

        Args:
            layer_stats: Per-layer tensor-id manifests driving enter/exit.
            tensors_map: Canonical id -> tensor table (CPU master copies).
            device_gpu: Target GPU device.
            delete_tensor_func: Per-tensor release callable used in ``exit``.
            rescuer: Optional :class:`UntimedTrapRescuer` consulted by
                :meth:`get` on a cache miss. Defaults to ``None``.
        """
        self.layer_stats = layer_stats
        self.tensors_map = tensors_map
        self.layer_stats_map: dict[str, list[int]] = {}
        self.load_time_ms = 0.0
        for statistics in layer_stats:
            stats_list: list[int] = []
            if statistics.label in self.layer_stats_map:
                stats_list = self.layer_stats_map[statistics.label]
            stats_list.extend(statistics.tensor_ids)
            self.layer_stats_map[statistics.label] = stats_list
        self.loaded: list[torch.Tensor] = []
        self.device_gpu = device_gpu
        self.cpu_to_gpu_map: dict[int, torch.Tensor] = {}
        self._active_counts: dict[int, int] = {}
        self.del_tensor = delete_tensor_func
        self._data_binder = _RawTensorDataBinder(tensors_map)

        self.model_ids: set[int] = set()
        self.rescuer = rescuer

    def set_model_ids(self, tensor_ids: set[int]) -> None:
        self.model_ids = tensor_ids

    def get_label_tensor_ids(self, label: str) -> set[int]:
        """Return unique tensor IDs associated with *label*."""
        return set(self.layer_stats_map.get(label, []))

    @_compiler_disable
    def enter(self, label: str) -> None:
        """Load tensors for the specified layer."""
        start_time_ns = time.time_ns()
        tensor_ids_list = self.layer_stats_map.get(label, [])
        # TODO: after consolidating stats, remove unique_tensor_ids
        unique_tensor_ids = set()
        for tensor_id in tensor_ids_list:
            if tensor_id in unique_tensor_ids:
                continue
            unique_tensor_ids.add(tensor_id)

            count = self._active_counts.get(tensor_id, 0)
            if count:
                self._active_counts[tensor_id] = count + 1
                continue

            tensor = self.tensors_map[tensor_id]
            if tensor.device != self.device_gpu:
                gpu_tensor = tensor.to(device=self.device_gpu, copy=True)
                if not _is_compiling():
                    self._data_binder.bind(tensor_id, gpu_tensor)
                self.cpu_to_gpu_map[tensor_id] = gpu_tensor
                self._active_counts[tensor_id] = 1
        end_time_ns = time.time_ns()
        duration_ms = (end_time_ns - start_time_ns) / 1e6
        self.load_time_ms += duration_ms

    @_compiler_disable
    def exit(self, label: str) -> None:
        """Release tensors for the specified layer."""
        tensor_ids_list = self.layer_stats_map.get(label, [])
        # TODO: after consolidating stats, remove unique_tensor_ids
        unique_tensor_ids = set()
        for tensor_id in tensor_ids_list:
            if tensor_id in unique_tensor_ids:
                continue
            unique_tensor_ids.add(tensor_id)

            count = self._active_counts.get(tensor_id, 0)
            if count <= 0:
                # ``enter`` saw the canonical already on GPU and skipped it;
                # adopt into the rescuer so ``get`` keeps finding it.
                if self.rescuer is not None:
                    self.rescuer.pin(tensor_id, self.tensors_map[tensor_id])
                continue
            if count > 1:
                self._active_counts[tensor_id] = count - 1
                continue

            self._active_counts.pop(tensor_id, None)
            if not _is_compiling():
                self._data_binder.restore({tensor_id})
            gpu_tensor = self.cpu_to_gpu_map.get(tensor_id)
            if gpu_tensor is None:
                continue
            if tensor_id in self.model_ids:
                self.del_tensor(gpu_tensor)
            else:
                del gpu_tensor
            del self.cpu_to_gpu_map[tensor_id]

    def get(self, tensor_id: int) -> torch.Tensor | None:
        """Get tensor by ID if loaded, falling back to the rescuer."""
        if tensor_id in self.cpu_to_gpu_map:
            return self.cpu_to_gpu_map[tensor_id]
        if self.rescuer is not None:
            return self.rescuer.get(tensor_id)
        return None

    def release_memory(self) -> None:
        """Release memory (placeholder implementation)."""
        if not _is_compiling():
            self._data_binder.restore_all()
        for tensor_id, gpu_tensor in list(self.cpu_to_gpu_map.items()):
            if tensor_id in self.model_ids:
                self.del_tensor(gpu_tensor)
            del self.cpu_to_gpu_map[tensor_id]
        self._active_counts.clear()

    def shutdown(self) -> None:
        """Release loader resources, including any rescuer-pinned tensors."""
        if self.rescuer is not None:
            self.rescuer.shutdown()


@instrumentable
class TensorStrategyLoader(Loader):
    """
    This loader implements strategic tensor management using configurable loading and release strategies.
    It supports preloading, asynchronous transfers, and scheduled memory cleanup based on defined policies.
    """

    def __init__(
        self,
        layer_stats: list[LayerStatistics],
        strategy_map: dict[str, list[TensorStatistics]],
        release_strategy_map: dict[str, list[TensorStatistics]],
        tensors_map: Mapping[int, torch.Tensor],
        device_gpu: torch.device,
        release_tensors: bool,
        stream_priority: int,
        delete_tensor_func: Callable[[torch.Tensor], None] = delete_tensor,
        *,
        reachable_tensor_ids: set[int] | None = None,
    ) -> None:
        self.device_gpu = device_gpu
        self.layer_stats = layer_stats
        self.strategy_map = strategy_map
        self.release_strategy_map = release_strategy_map
        self.tensors_map = tensors_map
        self.cpu_to_gpu_map = {}
        # create transfer stream with higher priority
        self.transfer_stream = torch.cuda.Stream(device=self.device_gpu, priority=stream_priority)
        self.transfer_ongoing = set()
        self.release_tensors = release_tensors
        self.use_events = True

        # preload tensors
        # TODO: create method for preloading tensors
        self.preload_ids = _compute_preload(layer_stats, strategy_map)
        # Rescue traced-but-untimed tensors so .get(...) doesn't return None
        # and fall through to the CPU tensor at trap time.
        rescue_ids = _compute_untimed_traced_preload(layer_stats, self.tensors_map)
        if reachable_tensor_ids is not None:
            # Narrow the rescue to model-reachable tensors only.
            rescue_ids &= reachable_tensor_ids
        if rescue_ids:
            # Only signal on the saved-state path; report_profiling_quality covers fresh runs.
            rescue_bytes = sum(
                self.tensors_map[tid].numel() * self.tensors_map[tid].element_size() for tid in rescue_ids
            )
            # Untimed-rescued tensors aren't in any layer_stat (that's *why*
            # they're rescued), so layer-name lookup isn't possible here.
            # Surface the sorted tensor IDs (truncated) so a user grepping a
            # profile dump can cross-reference which trapped tensors lost
            # their duration sample.
            sorted_ids = sorted(rescue_ids)
            id_hint = ", ".join(str(tid) for tid in sorted_ids[:8])
            if len(sorted_ids) > 8:
                id_hint += f", ... ({len(sorted_ids) - 8} more)"
            LOGGER.warning(
                "Untimed-trap rescue activated: %d tensor(s) (%.2f MiB) force-pinned to GPU "
                "[ids: %s]. Likely a profile-coverage gap.",
                len(rescue_ids),
                rescue_bytes / (1024 * 1024),
                id_hint,
            )
        self.preload_ids |= rescue_ids
        # Ids whose ``cpu_to_gpu_map`` entry is the canonical model tensor rather
        # than a loader-owned copy. Freeing one of these would corrupt the model's
        # weights silently, so ``release_for_label`` must never ``del_tensor`` them.
        self._passthrough_preload_ids: set[int] = set()
        for tensor_id, tensor in self.tensors_map.items():
            if tensor_id in self.preload_ids:
                if tensor.device == device_gpu:
                    gpu_tensor = tensor
                    self._passthrough_preload_ids.add(tensor_id)
                else:
                    gpu_tensor = tensor.to(device=device_gpu, copy=True)
                self.cpu_to_gpu_map[tensor_id] = gpu_tensor

        torch.cuda.synchronize()

        self.scheduled_releases = {}
        self.transfer_events = {}
        self.scheduled_transfers = {}
        self.del_tensor = delete_tensor_func

        self.tensor_id_to_release_label = {}
        self.load_to_release_labels_map = {}
        self.prepare_load_to_release_labels_map()

    def prepare_load_to_release_labels_map(self):
        for label, strategy in self.release_strategy_map.items():
            for tensor_info in strategy:
                self.tensor_id_to_release_label[tensor_info.tensor_id] = label

        for label, strategy in self.strategy_map.items():
            release_labels = set()
            for tensor_info in strategy:
                tensor_id = tensor_info.tensor_id
                release_label = self.tensor_id_to_release_label[tensor_id]
                release_labels.add(release_label)
            self.load_to_release_labels_map[label] = release_labels

    @_compiler_disable
    def enter(self, label: str) -> None:
        if label not in self.strategy_map:
            return
        strategy = self.strategy_map[label]

        release_labels = self.load_to_release_labels_map[label]
        for release_label in release_labels:
            if release_label in self.scheduled_releases:
                del self.scheduled_releases[release_label]

        with torch.cuda.stream(self.transfer_stream):
            for tensor_info in strategy:
                tensor_id = tensor_info.tensor_id
                tensor = self.tensors_map[tensor_id]
                if tensor_id not in self.cpu_to_gpu_map:
                    self.cpu_to_gpu_map[tensor_id] = tensor.to(device=self.device_gpu, non_blocking=True, copy=True)

        # transfer events
        event = torch.cuda.Event()
        event.record(self.transfer_stream)
        self.scheduled_transfers[label] = event
        for tensor_info in strategy:
            tensor_id = tensor_info.tensor_id
            self.transfer_events[tensor_id] = event

    def release_for_label(self, label):
        release_strategy = self.release_strategy_map[label]
        for tensor_info in release_strategy:
            if tensor_info.tensor_id in self.cpu_to_gpu_map:
                if tensor_info.tensor_id in self._passthrough_preload_ids:
                    # Passthrough entry: the map holds the canonical model tensor,
                    # not a loader-owned copy. Deleting its storage would silently
                    # corrupt the model's weights rather than fail loudly.
                    continue
                gpu_tensor = self.cpu_to_gpu_map[tensor_info.tensor_id]
                self.del_tensor(gpu_tensor)
                del self.cpu_to_gpu_map[tensor_info.tensor_id]

    def release_memory(self):
        if not self.release_tensors:
            return
        labels_passed = []
        for key, ev_item in self.scheduled_releases.items():
            if ev_item.query():
                labels_passed.append(key)
        for key in labels_passed:
            self.release_for_label(key)
            del self.scheduled_releases[key]

        # release transfer events
        for label in labels_passed:
            if label not in self.strategy_map:
                continue
            strategy = self.strategy_map[label]
            for tensor_info in strategy:
                tensor_id = tensor_info.tensor_id
                if tensor_id in self.transfer_events:
                    del self.transfer_events[tensor_id]

    @_compiler_disable
    def exit(self, label: str) -> None:
        if not self.release_tensors:
            return

        if label not in self.release_strategy_map:
            if self.use_events:
                self.release_memory()
            return
        if self.use_events:
            self.release_memory_with_events(label)
        else:
            self.release_memory_with_stream_synchronize(label)

    def release_memory_with_events(self, label):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.release_memory()
        self.scheduled_releases[str(label)] = event

    def release_memory_with_stream_synchronize(self, label):
        torch.cuda.current_stream().synchronize()
        self.release_for_label(label)

    def get(self, tensor_id):
        if tensor_id in self.transfer_events:
            event = self.transfer_events[tensor_id]
            torch.cuda.current_stream().wait_event(event)
            del self.transfer_events[tensor_id]

        tensor = None
        if tensor_id in self.cpu_to_gpu_map:
            tensor = self.cpu_to_gpu_map[tensor_id]
        return tensor

    def get_gpu_memory_bytes(self) -> int:
        """Get peak GPU memory used during inference execution.

        Computes the maximum GPU memory that will be used at any point during
        inference by simulating the sliding window pattern of tensor loading
        and releasing.

        Returns:
            Peak GPU memory in bytes - the maximum total size of tensors
            simultaneously loaded at any point during execution.
        """
        return _compute_peak_memory_from_strategy(self.strategy_map, self.release_strategy_map, self.layer_stats)

    def shutdown(self) -> None:
        """Release constructor preloads and interrupted transfer state."""
        synchronize_transfer_stream(self.transfer_stream)
        for tensor_id, gpu_tensor in self.cpu_to_gpu_map.items():
            if tensor_id not in self._passthrough_preload_ids:
                self.del_tensor(gpu_tensor)
        self.cpu_to_gpu_map.clear()
        self._passthrough_preload_ids.clear()
        self.transfer_ongoing.clear()
        self.scheduled_releases.clear()
        self.transfer_events.clear()
        self.scheduled_transfers.clear()


@instrumentable
class RawBlockController:
    def __init__(
        self,
        label_to_size_map,
        block_sizes,
        device_gpu,
        tensors_map,
        strategy_map,
        label_to_block_id,
        *,
        host_pinner: HostPinner,
        release_tensor_memory: bool = True,
    ):
        self.block_map_cpu = {}
        self.block_map_cpu_sizes = {}
        self.block_map_gpu = {}
        self.device_cpu = "cpu"
        self.device_gpu = device_gpu
        self.label_to_block_id = label_to_block_id
        self.host_pinner = host_pinner
        # When True (inference default), each source weight in ``tensors_map`` is
        # freed right after being copied into its pinned CPU block. Compiled
        # re-plan's first (non-destructive) build must pass False so originals
        # survive for the subsequent destructive rebuild.
        self.release_tensor_memory = release_tensor_memory

        # Process each layer: allocate block, copy tensors, release tensors
        # This interleaved approach reduces peak memory compared to allocating all blocks upfront
        self.block_meta_map = {}
        self.label_to_cpu_tensor_id_map = {}
        for label, strategy in strategy_map.items():
            if len(strategy) == 0:
                continue

            # Allocate CPU block for this layer.
            total_size = label_to_size_map[label]
            block = torch.zeros(total_size, dtype=torch.uint8, device=self.device_cpu)
            block = self.host_pinner.pin(block)
            self.block_map_cpu[label] = block
            self.block_map_cpu_sizes[label] = block.size(0)

            # Collect tensors for this layer
            tensors_list = []
            tensor_ids_list = []
            # TODO: after consolidating stats, remove unique_tensor_ids
            unique_tensor_ids = set()
            for tensor_info in strategy:
                tensor_id = tensor_info.tensor_id
                if tensor_id in unique_tensor_ids:
                    continue
                unique_tensor_ids.add(tensor_id)
                tensor = tensors_map[tensor_id]
                tensors_list.append(tensor)
                tensor_ids_list.append(tensor_id)

            # combine_tensors copies tensor data to block and optionally releases
            meta = self.combine_tensors(tensors_list, block)
            self.block_meta_map[label] = meta
            self.label_to_cpu_tensor_id_map[label] = tensor_ids_list

        # Allocate GPU blocks after all CPU blocks are populated
        for block_id, total_size in block_sizes.items():
            block = torch.zeros(total_size, dtype=torch.uint8, device=self.device_gpu)
            self.block_map_gpu[block_id] = block

        self.gpu_block_view_map = {}
        for label, block_id in label_to_block_id.items():
            gpu_block = self.block_map_gpu[block_id]
            cpu_block_size = self.block_map_cpu_sizes[label]
            gpu_block_view = gpu_block[:cpu_block_size]
            self.gpu_block_view_map[label] = gpu_block_view

        self.prepare_tensor_id_to_view_mapping()

    def prepare_tensor_id_to_view_mapping(self):
        self.label_to_tensor_views_map = {}
        for label, meta in self.block_meta_map.items():
            block_id = self.label_to_block_id[label]
            gpu_block = self.block_map_gpu[block_id]
            tensor_views = self.reconstruct_original_shapes(gpu_block, meta)
            self.label_to_tensor_views_map[label] = tensor_views

        self.tensor_id_to_view_map = {}
        for label, views in self.label_to_tensor_views_map.items():
            tensor_ids = self.label_to_cpu_tensor_id_map[label]
            for tensor_id, view in zip(tensor_ids, views, strict=False):
                self.tensor_id_to_view_map[tensor_id] = view

    def combine_tensors(self, tensors, combined):
        metadata = []
        current_idx = 0
        for tensor in tensors:
            tensor_shape = tensor.shape
            tensor_dtype = tensor.dtype
            tensor_stride = tensor.stride()
            num_elements = tensor.numel()
            element_size = tensor.element_size()
            nbytes = num_elements * element_size

            start_idx = current_idx
            end_idx = current_idx + nbytes

            if tensor.is_contiguous():
                byte_tensor = tensor.view(torch.uint8).flatten()
            elif is_dense_layout(tensor):
                # Dense non-C-contiguous (e.g. Fortran-contiguous from
                # weight.t()): access raw storage bytes so the original
                # memory layout is preserved in the combined block.
                offset_bytes = tensor.storage_offset() * element_size
                byte_tensor = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                byte_tensor.set_(tensor.untyped_storage(), offset_bytes, (nbytes,))
            else:
                # Non-dense with gaps: copy to contiguous layout first.
                contig = tensor.contiguous()
                tensor_stride = contig.stride()
                byte_tensor = contig.view(torch.uint8).flatten()

            combined[start_idx:end_idx] = byte_tensor

            metadata.append((start_idx, end_idx, tensor_shape, tensor_dtype, tensor_stride))
            current_idx = end_idx

            if getattr(self, "release_tensor_memory", True):
                tensor.data = torch.empty(0, device="cpu", dtype=tensor_dtype)

        return metadata

    def reconstruct_original_shapes(
        self,
        combined_tensor: torch.Tensor,
        metadata: list[tuple[int, int, torch.Size, torch.dtype, tuple[int, ...]]],
    ) -> list[torch.Tensor]:
        """
        Reconstruct tensors with their original shapes, dtypes and strides
        from the uint8 combined tensor.

        Args:
            combined_tensor: The flattened combined uint8 tensor (reinterpreted bytes)
            metadata: List of (start_idx, end_idx, original_shape, dtype, strides) tuples

        Returns:
            List of tensors with original shapes, dtypes and strides
        """
        ust = combined_tensor.untyped_storage()
        reconstructed = []
        for start_idx, _end_idx, original_shape, dtype, original_strides in metadata:
            t = torch.empty(0, dtype=dtype, device=combined_tensor.device)
            storage_offset_elems = start_idx // t.element_size()
            t.set_(ust, storage_offset_elems, original_shape, original_strides)
            reconstructed.append(t)

        return reconstructed

    def get_tensor_id_to_view_mapping(self):
        return self.tensor_id_to_view_map

    def schedule_transfer(self, label, non_blocking=True):
        cpu_block = self.block_map_cpu[label]
        gpu_block_view = self.gpu_block_view_map[label]
        gpu_block_view.copy_(cpu_block, non_blocking=non_blocking)

    def shutdown(self) -> None:
        """Release any resources associated with this controller."""
        self.block_map_cpu.clear()
        self.block_map_cpu_sizes.clear()
        self.block_map_gpu.clear()
        self.block_meta_map.clear()
        self.label_to_cpu_tensor_id_map.clear()
        self.gpu_block_view_map.clear()
        self.label_to_tensor_views_map.clear()
        self.tensor_id_to_view_map.clear()

    def release_gpu_blocks(self) -> None:
        """Drop every reference to this controller's GPU block allocations.

        Mirrors :meth:`AllocationBlockController.release_gpu_blocks` so compiled
        re-plan can free loader-1 GPU storage before rebuilding. Caller contract:
        only safe once no live tensor (e.g. ``param.data``) aliases these views.
        """
        self.block_map_gpu = {}
        self.gpu_block_view_map = {}
        self.label_to_tensor_views_map = {}
        self.tensor_id_to_view_map = {}

    def get_gpu_memory_bytes(self) -> int:
        """Get total GPU memory used by transfer blocks.

        Returns:
            Total GPU memory in bytes allocated for transfer blocks.
        """
        total_bytes = 0
        for block in self.block_map_gpu.values():
            total_bytes += block.numel() * block.element_size()
        return total_bytes


@instrumentable
class AllocationBlockController:
    def __init__(  # noqa: C901 - allocation and rollback form one construction transaction.
        self,
        allocation_ordered: dict[int, list[str]],
        device_gpu: torch.device,
        tensors_map: Mapping[int, torch.Tensor],
        strategy_map: dict[str, list[TensorStatistics]],
        label_to_block_id: dict[str, int],
        use_shm: bool = False,
        shm_block_name_map: dict[str, str] | None = None,
        block_name_fn: Callable[[int], str] | None = None,
        *,
        host_pinner: HostPinner,
        release_tensor_memory: bool = True,
    ) -> None:
        self.block_map_cpu = {}  # label to cpu block
        self.label_to_cpu_tensor_id_map = {}
        self.device_cpu = "cpu"
        self.device_gpu = device_gpu
        self.device_gpu_str = self._format_device_str(device_gpu)
        self.block_map_gpu = {}  # block_id to gpu block
        self.label_to_gpu_block = {}
        self.label_to_tensor_views_map = {}
        self.gpu_block_view_map = {}
        self.tensor_id_to_view_map: dict[int, torch.Tensor] = {}
        self.use_shm = use_shm
        self.load_model_from_shm = use_shm and shm_block_name_map is not None
        # Treat the caller's profile map as input. Publishing names into it
        # incrementally would leave a failed construction looking complete to
        # a later retry.
        self.shm_block_name_map = dict(shm_block_name_map or {})
        self._block_name_fn = block_name_fn or (lambda index: f"ft_{os.getpid()}_{index}")
        self.host_pinner = host_pinner
        # When True (inference default), source weights are released as they are
        # copied into pinned CPU blocks. Compiled re-plan profiling disables this
        # so a second, destructive controller can be built from the same tensors.
        self.release_tensor_memory = release_tensor_memory
        # Every successfully returned block enters this sole ownership ledger.
        self._owned_blocks: list[AllocationBlock] = []
        self._construction_rollback = ConstructionRollback()

        try:
            if release_tensor_memory and not self.load_model_from_shm:
                self._validate_source_storage(allocation_ordered, strategy_map, tensors_map)
            for index, (_block_id, layers) in enumerate(allocation_ordered.items()):
                shm_prefix = None
                if self.use_shm:
                    if layers[0] not in self.shm_block_name_map:
                        shm_prefix = self._block_name_fn(index)
                    else:
                        shm_prefix = self.shm_block_name_map[layers[0]]
                allocation_manager = AllocationManager(
                    shm_block_name_prefix=shm_prefix,
                    load_from_shm=self.load_model_from_shm,
                    pinned_memory=True,
                    host_pinner=self.host_pinner,
                    release_tensor_memory=self.release_tensor_memory,
                )
                for label in layers:
                    if self.use_shm and label not in self.shm_block_name_map:
                        self.shm_block_name_map[label] = shm_prefix
                    strategy = strategy_map[label]
                    if len(strategy) == 0:
                        continue
                    block = allocation_manager.block()
                    # Record ownership before allocation or map publication:
                    # allocate() is where CPU SHM is created/attached and may
                    # fail after acquiring the handle.
                    self._owned_blocks.append(block)
                    if release_tensor_memory and not self.load_model_from_shm:
                        self._construction_rollback.record(block, strategy, tensors_map)
                    # _add_tensors calls block.allocate(), which may release each
                    # source tensor according to release_tensor_memory.
                    tensor_ids_list = self._add_tensors(label, strategy, tensors_map, block)
                    self.block_map_cpu[label] = block
                    self.label_to_cpu_tensor_id_map[label] = tensor_ids_list

                gpu_block = allocation_manager.create_max_block(device=self.device_gpu_str)
                self._owned_blocks.append(gpu_block)
                for label in layers:
                    self.label_to_gpu_block[label] = gpu_block

            for label, cpu_block in self.block_map_cpu.items():
                gpu_block = self.label_to_gpu_block[label]
                gpu_tensor_views, gpu_block_view = cpu_block.project_views(gpu_block)
                self.label_to_tensor_views_map[label] = gpu_tensor_views
                block_id = label_to_block_id[label]
                self.block_map_gpu[block_id] = gpu_block_view  # TODO: REMOVE!
                self.gpu_block_view_map[label] = gpu_block_view

            self.prepare_tensor_id_to_view_mapping()
        except BaseException as primary_error:
            protected_blocks = self._construction_rollback.restore(attempts=_CONSTRUCTION_CLEANUP_ATTEMPTS)
            protected_block_ids = {id(block) for block in protected_blocks}
            releasable_blocks = [block for block in self._owned_blocks if id(block) not in protected_block_ids]
            failed_releases = self._release_blocks(
                releasable_blocks,
                attempts=_CONSTRUCTION_CLEANUP_ATTEMPTS,
            )
            self._owned_blocks = [*protected_blocks, *failed_releases]
            self._clear_allocation_maps()
            self.shm_block_name_map.clear()
            if self._owned_blocks:
                self._attach_cleanup_controller(primary_error)
            raise

    @staticmethod
    def _validate_source_storage(
        allocation_ordered: dict[int, list[str]],
        strategy_map: dict[str, list[TensorStatistics]],
        tensors_map: Mapping[int, torch.Tensor],
    ) -> None:
        """Reject unsupported storage groups before any source tensor is released."""
        managed_ids = {
            id(tensors_map[info.tensor_id])
            for layers in allocation_ordered.values()
            for label in layers
            for info in strategy_map.get(label, ())
        }
        sources = {
            id(tensor): tensor
            for tensor in tensors_map.values()
            if not tensor.is_meta and tensor.layout == torch.strided
        }
        shared_ids = _RawTensorDataBinder._find_shared_storage_ids(sources)  # noqa: SLF001
        for tensor_id in managed_ids:
            if tensor_id in shared_ids:
                # Disjoint views are normal after release() transfers packed
                # CPU storage back to the model. Only overlap is unsupported.
                raise ValueError(
                    "Block loaders cannot preserve distinct tensor views sharing storage with overlapping byte regions "
                    f"(tensor ids {sorted(shared_ids[tensor_id])}); use the strategy loader"
                )

    @staticmethod
    def _release_blocks(
        blocks: list[AllocationBlock],
        *,
        attempts: int,
    ) -> list[AllocationBlock]:
        """Release owned blocks with bounded retries for staged teardown."""
        pending = list(reversed(blocks))
        for _attempt in range(attempts):
            if not pending:
                break
            failed: list[AllocationBlock] = []
            for block in pending:
                try:
                    block.release()
                except BaseException:
                    # Construction cleanup must never replace the primary
                    # failure. Failed staged handles remain reachable below.
                    LOGGER.warning("Failed to release allocation block during cleanup", exc_info=True)
                    failed.append(block)
            pending = failed
        pending.reverse()
        return pending

    def _attach_cleanup_controller(
        self,
        primary_error: BaseException,
        cleanup_error: BaseException | None = None,
    ) -> None:
        """Keep retryable handles reachable without replacing the primary error."""
        try:
            primary_error.__dict__["_flextensor_cleanup_controller"] = self
            if cleanup_error is not None:
                primary_error.__dict__["_flextensor_cleanup_error"] = cleanup_error
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(
                    "FlexTensor construction cleanup is incomplete; call "
                    "exception._flextensor_cleanup_controller.shutdown() to retry it."
                )
        except BaseException:
            LOGGER.exception("Could not attach retryable cleanup handles to the primary error")

    def _clear_allocation_maps(self) -> None:
        self.block_map_cpu.clear()
        self.label_to_cpu_tensor_id_map.clear()
        self.block_map_gpu.clear()
        self.label_to_gpu_block.clear()
        self.label_to_tensor_views_map.clear()
        self.gpu_block_view_map.clear()
        self.tensor_id_to_view_map.clear()

    def _add_tensors(
        self,
        label: str,
        strategy: list[TensorStatistics],
        tensors_map: Mapping[int, torch.Tensor],
        block: AllocationBlock,
    ) -> list[int]:
        """Add tensors from strategy to an allocation block.

        Collects tensors referenced by the strategy, deduplicates by tensor ID,
        adds them to the allocation block, and allocates the block memory.
        Allocation releases each original tensor's storage after copying.

        Args:
            label: Layer label identifying this tensor group.
            strategy: List of TensorStatistics describing tensors to add.
            tensors_map: Mapping from tensor IDs to actual tensors.
            block: AllocationBlock to add tensors to and allocate.

        Returns:
            List of corresponding tensor IDs in the same order.

        Example:
            >>> block = allocation_manager.block()
            >>> ids = self._add_tensors("layer.0", strategy, tensors_map, block)
        """
        unique_tensor_ids: set[int] = set()
        tensor_ids_list: list[int] = []
        for tensor_info in strategy:
            tensor_id = tensor_info.tensor_id
            # TODO: after consolidating stats, remove unique_tensor_ids
            if tensor_id in unique_tensor_ids:
                continue
            unique_tensor_ids.add(tensor_id)

            tensor = tensors_map[tensor_id]
            tensor_ids_list.append(tensor_id)
            block.add(tensor)
        _views = block.allocate()
        return tensor_ids_list

    def _format_device_str(self, device_gpu: torch.device) -> str:
        """Format a torch device to its string representation.

        Args:
            device_gpu: The torch.device to format (e.g., cuda:0, cuda, cpu).

        Returns:
            String representation of the device, including index if present
            (e.g., "cuda:0" or "cuda").
        """
        device_str = str(device_gpu.type)
        if device_gpu.index is not None:
            device_str += ":" + str(device_gpu.index)
        return device_str

    def prepare_tensor_id_to_view_mapping(self):
        self.tensor_id_to_view_map = {}
        for label, views in self.label_to_tensor_views_map.items():
            tensor_ids = self.label_to_cpu_tensor_id_map[label]
            for tensor_id, view in zip(tensor_ids, views, strict=False):
                self.tensor_id_to_view_map[tensor_id] = view

    def get_tensor_id_to_view_mapping(self):
        return self.tensor_id_to_view_map

    def schedule_transfer(self, label, non_blocking=True):
        cpu_block = self.block_map_cpu[label]
        gpu_block_view = self.gpu_block_view_map[label]
        cpu_block.copy_to(gpu_block_view, non_blocking=non_blocking)

    def commit_construction(self) -> None:
        """Discard rollback metadata after a transfer loader assumes ownership."""
        self._construction_rollback.commit()

    def abort_construction(self, primary_error: BaseException | None = None) -> None:
        """Restore source tensors and release resources after handoff failure."""
        try:
            self.shutdown()
        except BaseException as cleanup_error:
            if primary_error is not None:
                self._attach_cleanup_controller(primary_error, cleanup_error)
            raise

    def shutdown(self) -> None:
        protected_blocks = self._construction_rollback.restore(attempts=_CONSTRUCTION_CLEANUP_ATTEMPTS)
        protected_block_ids = {id(block) for block in protected_blocks}
        failed_releases = self._release_blocks(
            [block for block in self._owned_blocks if id(block) not in protected_block_ids],
            attempts=_CONSTRUCTION_CLEANUP_ATTEMPTS,
        )
        self._owned_blocks = [*protected_blocks, *failed_releases]
        self._clear_allocation_maps()
        if self._owned_blocks:
            raise RuntimeError(
                f"AllocationBlockController cleanup remains incomplete for {len(self._owned_blocks)} block(s)"
            )

    def release_gpu_blocks(self) -> None:
        """Drop references to GPU blocks once no live tensor aliases them."""
        gpu_block_ids = {id(block) for block in self.label_to_gpu_block.values()}
        self._owned_blocks = [block for block in self._owned_blocks if id(block) not in gpu_block_ids]
        self.label_to_gpu_block = {}
        self.block_map_gpu = {}
        self.gpu_block_view_map = {}
        self.label_to_tensor_views_map = {}
        self.tensor_id_to_view_map = {}

    def get_gpu_memory_bytes(self) -> int:
        """Get total GPU memory used by transfer blocks.

        Returns:
            Total GPU memory in bytes allocated for transfer blocks.
        """
        total_bytes = 0
        for gpu_block_view in self.block_map_gpu.values():
            # Use untyped_storage().nbytes() to get the full underlying allocation size,
            # not just the view size (views share storage with the full GPU block)
            total_bytes += gpu_block_view.untyped_storage().nbytes()
        return total_bytes


class PreallocatedLoader(Loader):
    def __init__(
        self,
        allocation_controller: AllocationBlockController | RawBlockController,
        offload_timing_collector: "OffloadTimingCollector | None" = None,
        piecewise_prefetch_policy: "PiecewisePrefetchPolicy | None" = None,
    ):
        self.allocation_controller = allocation_controller
        self.offload_timing_collector = offload_timing_collector or OffloadTimingCollector(enabled=False)
        self.piecewise_prefetch_policy = piecewise_prefetch_policy or PiecewisePrefetchPolicy()

    @abstractmethod
    def preload(self) -> None:
        """
        Method to pre-load tensors or data blocks as per loader strategy.
        """

    @abstractmethod
    def prepare(self) -> None:
        """
        Prepare the loader for transfers or computation.
        """

    def sync_prev_onload(self) -> None:
        """Make the current compute stream wait for prior transfer work."""
        raise NotImplementedError

    def join_after_forward(self) -> None:
        """Join transfer work at an integration-defined forward boundary."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """
        Release resources and clean up allocations.
        """
        self.allocation_controller.shutdown()

    def release_memory(self) -> None:
        """
        Release memory.
        """
        return None

    def get_gpu_memory_bytes(self) -> int:
        """Get total GPU memory used by transfer blocks.

        Delegates to the block controller to compute GPU memory usage.

        Returns:
            Total GPU memory in bytes allocated for transfer blocks.
        """
        return self.allocation_controller.get_gpu_memory_bytes()


@instrumentable
class PreallocatedBatchTransferTensorLoader(PreallocatedLoader):
    def __init__(
        self,
        layer_stats: list[LayerStatistics],
        device_gpu: torch.device,
        label_to_block_id: dict[str, int],
        transfer_to_compute_map: dict[str, str],
        allocation_controller: AllocationBlockController | RawBlockController,
        offload_timing_collector: "OffloadTimingCollector | None" = None,
        piecewise_prefetch_policy: "PiecewisePrefetchPolicy | None" = None,
    ) -> None:
        super().__init__(
            allocation_controller,
            offload_timing_collector=offload_timing_collector,
            piecewise_prefetch_policy=piecewise_prefetch_policy,
        )
        self.device_gpu = device_gpu
        self.cpu_to_gpu_map = {}

        self.label_to_block_id = label_to_block_id
        self.transfer_to_compute_map = transfer_to_compute_map

        self._transfer_runtime = CudaTransferRuntime(device_gpu)
        self.transfer_stream = self._transfer_runtime.transfer_stream
        self.scheduled_transfers: dict[str, TransferTicket] = {}
        self._compute_to_block_ids = _compute_to_block_ids(label_to_block_id, transfer_to_compute_map)

        self.first_layer_label: str | None = None
        self.last_layer_label: str | None = None

        if layer_stats:
            self.first_layer_label = layer_stats[0].label
            self.last_layer_label = layer_stats[-1].label

        self.preload_transfers = find_transfers_for_preload(self.transfer_to_compute_map, layer_stats)
        self.prepare()

    @_compiler_disable
    def sync_prev_onload(self) -> None:
        """Make the current compute stream wait for prior transfer work."""
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

    @_compiler_disable
    def preload(self) -> None:
        if not self.preload_transfers:
            return
        self._transfer_runtime.begin()
        for label in self.preload_transfers:
            block_id = self.label_to_block_id[label]
            self._transfer_runtime.submit(
                block_id,
                lambda label=label: self.allocation_controller.schedule_transfer(label),
                trace_label=f"schedule={label} consume={self.transfer_to_compute_map[label]}",
            )
        self._transfer_runtime.join()

    def prepare(self) -> None:
        self.preload()

    @_compiler_disable
    def enter(self, label):
        tc = self.offload_timing_collector
        pp = self.piecewise_prefetch_policy

        if label == self.first_layer_label:
            recovering = self._transfer_runtime.is_active
            if recovering:
                pending_count = len(self.scheduled_transfers)
                self._transfer_runtime.join()
                LOGGER.debug(
                    "%s: abort recovery — joined transfer stream and clearing %d scheduled entries",
                    type(self).__name__,
                    pending_count,
                )
                self.scheduled_transfers.clear()

            tc.on_pass_start()
            pp.reset()

        if not self._transfer_runtime.is_active:
            self._transfer_runtime.begin()

        if label not in self.label_to_block_id:
            tc.record_compute_start(label)
            return

        block_id = self.label_to_block_id[label]

        def transfer() -> None:
            tc.record_transfer_start(label, self.transfer_stream)
            self.allocation_controller.schedule_transfer(label)

        self.scheduled_transfers[label] = self._transfer_runtime.submit(
            block_id,
            transfer,
            trace_label=f"schedule={label} consume={self.transfer_to_compute_map[label]}",
        )
        pp.on_schedule(schedule_label=label, compute_label=label)
        tc.record_transfer_end(label, self.transfer_stream)
        tc.record_compute_start(label)

    @_compiler_disable
    def exit(self, label):
        tc = self.offload_timing_collector
        pp = self.piecewise_prefetch_policy

        # Record compute_end BEFORE the wait so compute_ms reflects
        # pure forward work, not forward + stall.
        tc.record_compute_end(label)

        ticket = self.scheduled_transfers.get(label)
        if ticket is not None:
            tc.record_wait_start(label)
            self._transfer_runtime.wait_ready(ticket)
            pp.on_wait(label)
            tc.record_wait_end(label)

        slots = self._compute_to_block_ids.get(label)
        if self._transfer_runtime.is_active and slots:
            for slot in slots:
                self._transfer_runtime.finish_use(slot)
        else:
            torch.cuda.current_stream().record_event()

        # Cross-iteration sync: record event on transfer stream after last layer.
        # Also serves as the final join to bring the transfer stream back to the
        # compute stream, required for CUDA graph capture completeness.
        if label == self.last_layer_label:
            from flextensor.piecewise_prefetch_policy import PiecewisePrefetchPolicyError

            if self._transfer_runtime.is_active:
                self._transfer_runtime.join()

            # Clear maps even if strict policy raises (same contract as
            # join_after_forward) so capture-session handles never leak.
            policy_error: BaseException | None = None
            try:
                pp.on_piece_join(at_last_layer=True)
            except PiecewisePrefetchPolicyError as exc:
                policy_error = exc

            # After join: drop completion maps (incl. scheduled_transfers).
            self.scheduled_transfers.clear()

            if policy_error is not None:
                raise policy_error

    @_compiler_disable
    def join_after_forward(self) -> None:
        """Join transfer_stream back to compute_stream at end of forward / piece.

        Call before CUDA-graph ``capture_end`` (FULL once, PIECEWISE every piece)
        so forked H2D is not left unjoined. Under PIECEWISE, last-layer ``exit()``
        alone is not enough — earlier pieces never reach it.

        Always clears transfer/event maps (even when the join itself is skipped):
        capture-session event handles are invalid across piece/iteration
        boundaries. Safe at end of forward / piece — not mid-forward.

        The piecewise-prefetch policy check runs first for diagnostics, but the
        stream join and map clears always run afterward — even if the policy
        raises in strict mode — otherwise ``capture_end`` fails with
        ``cudaErrorStreamCaptureUnjoined``.
        """
        from flextensor.piecewise_prefetch_policy import PiecewisePrefetchPolicyError

        pp = self.piecewise_prefetch_policy
        policy_error: BaseException | None = None
        try:
            pp.on_piece_join(at_last_layer=False)
        except PiecewisePrefetchPolicyError as exc:
            policy_error = exc

        self._transfer_runtime.join()
        self.scheduled_transfers.clear()

        if policy_error is not None:
            raise policy_error

    def release_memory(self):
        pass

    def shutdown(self) -> None:
        self._transfer_runtime.close()
        super().shutdown()


@instrumentable
class PreallocatedBatchTransferTensorLoaderReordered(PreallocatedLoader):
    def __init__(
        self,
        layer_stats: list[LayerStatistics],
        device_gpu: torch.device,
        label_to_block_id: dict[str, int],
        transfer_to_compute_map: dict[str, str],
        allocation_controller: AllocationBlockController | RawBlockController,
        offload_timing_collector: "OffloadTimingCollector | None" = None,
        piecewise_prefetch_policy: "PiecewisePrefetchPolicy | None" = None,
    ) -> None:
        super().__init__(
            allocation_controller,
            offload_timing_collector=offload_timing_collector,
            piecewise_prefetch_policy=piecewise_prefetch_policy,
        )
        self.device_gpu = device_gpu
        self.cpu_to_gpu_map = {}

        self.label_to_block_id = label_to_block_id
        self.transfer_to_compute_map = transfer_to_compute_map

        self._transfer_runtime = CudaTransferRuntime(device_gpu)
        self.transfer_stream = self._transfer_runtime.transfer_stream
        self.scheduled_transfers: dict[str, list[TransferTicket]] = {}
        self._compute_to_block_ids = _compute_to_block_ids(label_to_block_id, transfer_to_compute_map)
        self._preloaded_compute_labels: set[str] = set()

        # Track iteration boundary for CUDA graph capture
        self.first_layer_label: str | None = None
        self.last_layer_label: str | None = None

        if layer_stats:
            self.first_layer_label = layer_stats[0].label
            self.last_layer_label = layer_stats[-1].label

        self.preload_transfers = find_transfers_for_preload(self.transfer_to_compute_map, layer_stats)
        self.prepare()

    @_compiler_disable
    def sync_prev_onload(self) -> None:
        """Make the current compute stream wait for prior transfer work."""
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

    @_compiler_disable
    def preload(self) -> None:
        if not self.preload_transfers:
            return
        self._transfer_runtime.begin()
        for label, compute_label in self.preload_transfers.items():
            block_id = self.label_to_block_id[label]
            self._transfer_runtime.submit(
                block_id,
                lambda label=label: self.allocation_controller.schedule_transfer(label),
                trace_label=f"schedule={label} consume={compute_label}",
            )
            self._preloaded_compute_labels.add(compute_label)
        self._transfer_runtime.join()

    def prepare(self) -> None:
        self.preload()

    @_compiler_disable
    def enter(self, label):
        tc = self.offload_timing_collector
        pp = self.piecewise_prefetch_policy

        if label == self.first_layer_label:
            recovering = self._transfer_runtime.is_active
            if recovering:
                pending_count = len(self.scheduled_transfers)
                self._transfer_runtime.join()
                LOGGER.debug(
                    "%s: abort recovery — joined transfer stream and clearing %d scheduled entries",
                    type(self).__name__,
                    pending_count,
                )
                self.scheduled_transfers.clear()

            tc.on_pass_start()
            pp.reset()

        if not self._transfer_runtime.is_active:
            self._transfer_runtime.begin()

        if label in self._preloaded_compute_labels:
            pp.on_wait(label)

        # Wait for the transfer that provides this layer's data before compute reads it.
        tickets = self.scheduled_transfers.pop(label, None)
        if tickets:
            tc.record_wait_start(label)
            for ticket in tickets:
                self._transfer_runtime.wait_ready(ticket)
            pp.on_wait(label)
            tc.record_wait_end(label)

        if label not in self.label_to_block_id:
            tc.record_compute_start(label)
            return

        self._schedule_block_transfer(label)

        tc.record_compute_start(label)

    def _schedule_block_transfer(self, label) -> None:
        """Issue the rolling-block H2D transfer for ``label`` on the transfer stream."""
        tc = self.offload_timing_collector
        pp = self.piecewise_prefetch_policy
        block_id = self.label_to_block_id[label]
        compute_label = self.transfer_to_compute_map[label]

        def transfer() -> None:
            tc.record_transfer_start(label, self.transfer_stream)
            self.allocation_controller.schedule_transfer(label)

        ticket = self._transfer_runtime.submit(
            block_id,
            transfer,
            trace_label=f"schedule={label} consume={compute_label}",
        )
        tc.record_transfer_end(label, self.transfer_stream)
        self.scheduled_transfers.setdefault(compute_label, []).append(ticket)
        pp.on_schedule(schedule_label=label, compute_label=compute_label)

    @_compiler_disable
    def exit(self, label):
        tc = self.offload_timing_collector
        pp = self.piecewise_prefetch_policy

        # Record compute_end BEFORE the join so compute_ms reflects
        # pure forward work, not forward + stall.
        tc.record_compute_end(label)

        slots = self._compute_to_block_ids.get(label)
        if self._transfer_runtime.is_active and slots:
            for slot in slots:
                self._transfer_runtime.finish_use(slot)
        else:
            torch.cuda.current_stream().record_event()

        # Join: bring transfer stream back to compute stream after last layer.
        # Required for CUDA graph capture completeness.
        if label == self.last_layer_label:
            from flextensor.piecewise_prefetch_policy import PiecewisePrefetchPolicyError

            if self._transfer_runtime.is_active:
                self._transfer_runtime.join()

            # Clear maps even if strict policy raises (same contract as
            # join_after_forward) so capture-session handles never leak.
            policy_error: BaseException | None = None
            try:
                pp.on_piece_join(at_last_layer=True)
            except PiecewisePrefetchPolicyError as exc:
                policy_error = exc

            # After join: drop completion maps (incl. scheduled_transfers).
            self.scheduled_transfers.clear()

            if policy_error is not None:
                raise policy_error

    @_compiler_disable
    def join_after_forward(self) -> None:
        """Join transfer_stream back to compute_stream at end of forward / piece.

        Call before CUDA-graph ``capture_end`` (FULL once, PIECEWISE every piece)
        so forked H2D is not left unjoined. Always clears transfer/event maps.
        Outstanding remapped (schedule≠compute) prefetches here mean rearrange
        overlap was broken by the boundary; see
        :class:`~flextensor.piecewise_prefetch_policy.PiecewisePrefetchPolicy`.

        Stream join / map clears always run even if the policy raises (strict
        mode); otherwise capture ends with ``cudaErrorStreamCaptureUnjoined``.
        """
        from flextensor.piecewise_prefetch_policy import PiecewisePrefetchPolicyError

        pp = self.piecewise_prefetch_policy
        policy_error: BaseException | None = None
        try:
            pp.on_piece_join(at_last_layer=False)
        except PiecewisePrefetchPolicyError as exc:
            policy_error = exc

        self._transfer_runtime.join()
        self.scheduled_transfers.clear()

        if policy_error is not None:
            raise policy_error

    def release_memory(self):
        pass

    def shutdown(self) -> None:
        self._transfer_runtime.close()
        super().shutdown()

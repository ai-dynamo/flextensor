# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Global optimization strategies for FlexTensor.

This module provides global optimization strategies that use metaheuristic
algorithms to find optimal offloading configurations.
"""

import math
import warnings
from typing import Any, cast

import numpy as np

from flextensor.collectors import LayerStatistics, TensorStatistics, extract_memory_transfers_from_layer_stats
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator

from .assignment import AssignmentStrategy, OptimizedRoundRobinAssignment
from .protocol import BlockStrategyData, StrategyComputeError, StrategyResult
from .transfer_window import GapAwareWindow, SingleLayerWindow, TransferWindowCalculator
from .utils import (
    EarlyStopCallback,
    calculate_transfer_to_compute_map,
    compute_label_to_size_map,
    validate_memory_params,
)


class GlobalOffloadStrategy:
    """
    Global offload strategy using assignment-based layer-to-block mapping for pipelined execution.

    Pipeline Execution Model:
    - Layer N computes using Block X (data transferred during Layer N-1)
    - While computing Layer N, we transfer Block Y for Layer N+1
    - Constraint: Block Y ≠ Block X (can't transfer and compute same block simultaneously)
    - Block X becomes free after Layer N finishes → can be reused for Layer N+2

    This means:
    - Minimum 2 blocks required for pipelining
    - With 2 blocks: alternating pattern (A, B, A, B, ...)
    - With 3+ blocks: more flexibility for scheduling

    The strategy uses an AssignmentStrategy (default: OptimizedRoundRobinAssignment) to
    determine layer-to-block assignments, then calculates block sizes based on the
    maximum weight size assigned to each block.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        scale: float = 1.0,
        threshold_mb: float = 0.1,
        min_blocks: int | None = None,
        max_blocks: int | None = None,
        assignment_strategy: AssignmentStrategy | None = None,
    ):
        """
        Initialize GlobalOffloadStrategy.

        Args:
            n_blocks: Number of reusable memory blocks (minimum 2 for pipelining).
            scale: Duration multiplier (baseline). Controls how much of the previous
                layer's compute time is available for transfers. ``1.0`` = exact fit.
            threshold_mb: Minimum weight size threshold in MB. Weights smaller than
                this are kept on GPU and not offloaded.
            min_blocks: Minimum number of blocks to use (default: 2).
                - 2 blocks: With alternating pattern (A,B,A,B,...)
            max_blocks: Maximum number of blocks to use (default: n_blocks).
            assignment_strategy: Strategy for layer-to-block assignment.
                Default: OptimizedRoundRobinAssignment (best GPU utilization).
                Available strategies:
                - OptimizedRoundRobinAssignment: Optimized cyclic pattern (recommended)
                - StrictRoundRobinAssignment: Strict i % n_blocks pattern

        Raises:
            ValueError: If n_blocks < 2 (minimum required for pipelining).
            ValueError: If min_blocks > max_blocks.
            ValueError: If min_blocks < 2 or max_blocks > n_blocks.
            ValueError: If scale <= 0.
        """
        if n_blocks < 2:
            msg = "n_blocks must be at least 2 for pipelined execution"
            raise ValueError(msg)
        validate_memory_params(scale)

        # Set defaults
        if min_blocks is None:
            min_blocks = 2
        if max_blocks is None:
            max_blocks = n_blocks

        # Validate block range
        if min_blocks < 2:
            msg = "min_blocks must be at least 2 for pipelined execution"
            raise ValueError(msg)
        if max_blocks > n_blocks:
            msg = f"max_blocks ({max_blocks}) cannot exceed n_blocks ({n_blocks})"
            raise ValueError(msg)
        if min_blocks > max_blocks:
            msg = f"min_blocks ({min_blocks}) cannot exceed max_blocks ({max_blocks})"
            raise ValueError(msg)

        self.n_blocks = n_blocks
        self.scale = scale
        self.threshold_mb = threshold_mb
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.assignment_strategy = assignment_strategy or OptimizedRoundRobinAssignment(
            min_blocks=min_blocks,
            max_blocks=max_blocks,
        )

        # Attributes set after compute()
        self.optimal_block_sizes: list[int] = []
        self.optimal_layer_to_block: dict[str, int] = {}
        self.optimal_non_offloaded_memory: int = 0
        self.optimal_peak_memory: float = 0.0

    def _collect_layer_tensors(
        self,
        layer_stats: list[LayerStatistics],
        memory_transfer_interpolator: MemoryTransferInterpolator | None = None,
        latency_priority: bool = True,
        scale: float | None = None,
    ) -> list[tuple[str, list[TensorStatistics], float]]:
        """
        Collect tensors per layer that meet the threshold (to be offloaded).

        When latency_priority is True and memory_transfer_interpolator is provided,
        limits offload to what can be transferred within compute time (0% overhead).

        Pipeline model: Transfer for layer N happens during layer N-1's compute.
        So we cap layer N's offload to what fits in layer N-1's duration.

        Args:
            layer_stats: Layer statistics.
            memory_transfer_interpolator: Interpolator for memory transfer timing.
            latency_priority: Whether to cap offload to what fits in compute time.
            scale: Duration multiplier to use. Defaults to ``self.scale``.

        Returns:
            List of (layer_label, offloaded_tensors, layer_duration) tuples.
        """
        effective_scale = scale if scale is not None else self.scale
        result = []
        threshold_bytes = self.threshold_mb * 1024 * 1024

        for layer_idx, layer in enumerate(layer_stats):
            filtered_tensors = [t for t in layer.tensors if t.size_bytes > threshold_bytes]

            # When prioritizing latency, cap offload to what can be transferred in time
            # Transfer for layer N happens during layer N-1's compute
            if latency_priority and memory_transfer_interpolator is not None and layer_idx > 0:
                prev_duration = layer_stats[layer_idx - 1].duration
                if prev_duration > 0:
                    # Calculate max bytes that can be transferred during PREVIOUS layer's compute
                    transfer_bytes = memory_transfer_interpolator.duration_to_bytes(prev_duration * effective_scale)
                    # Handle infinity (can happen with degenerate interpolator data)
                    max_transfer_bytes = 2**63 - 1 if math.isinf(transfer_bytes) else int(transfer_bytes)

                    # Sort tensors by size (largest first) and select what fits
                    sorted_tensors = sorted(filtered_tensors, key=lambda t: t.size_bytes, reverse=True)
                    selected_tensors = []
                    current_size = 0

                    for tensor in sorted_tensors:
                        if current_size + tensor.size_bytes <= max_transfer_bytes:
                            selected_tensors.append(tensor)
                            current_size += tensor.size_bytes

                    filtered_tensors = selected_tensors

            if filtered_tensors:
                result.append((layer.label, filtered_tensors, layer.duration))

        return result

    def _calculate_non_offloaded_memory(
        self,
        layer_stats: list[LayerStatistics],
        offloaded_layer_data: list[tuple[str, list[TensorStatistics], float]] | None = None,
    ) -> int:
        """
        Calculate total memory of tensors that are NOT offloaded (stay on GPU).

        This includes:
        - Tensors below the size threshold (always kept on GPU)
        - Tensors above the threshold that were NOT selected for offload
          (e.g., because they exceed the transfer time budget)

        Args:
            layer_stats: All layer statistics.
            offloaded_layer_data: Output from ``_collect_layer_tensors`` — list of
                (label, offloaded_tensors, duration) tuples.  When provided, any
                tensor NOT in the offloaded list is counted as non-offloaded.
                When ``None``, falls back to only counting below-threshold tensors.

        Returns:
            Total non-offloaded memory across all layers in bytes.
        """
        if offloaded_layer_data is not None:
            # The first layer is always fully on GPU (preloaded — no previous
            # layer to pipeline the transfer).  Even though _collect_layer_tensors
            # may include it, its tensors can never actually be offloaded via the
            # pipeline, so we must NOT mark them as offloaded here.
            first_label = layer_stats[0].label if layer_stats else None
            offloaded_ids_per_layer: dict[str, set[int]] = {
                label: {t.tensor_id for t in tensors}
                for label, tensors, _ in offloaded_layer_data
                if label != first_label
            }
            total_non_offloaded = 0
            for layer in layer_stats:
                offloaded_ids = offloaded_ids_per_layer.get(layer.label, set())
                total_non_offloaded += sum(t.size_bytes for t in layer.tensors if t.tensor_id not in offloaded_ids)
            return total_non_offloaded

        # Fallback: only count below-threshold tensors (legacy behaviour)
        threshold_bytes = self.threshold_mb * 1024 * 1024
        total_non_offloaded = 0
        for layer in layer_stats:
            total_non_offloaded += sum(t.size_bytes for t in layer.tensors if t.size_bytes <= threshold_bytes)
        return total_non_offloaded

    @staticmethod
    def calculate_layer_tensor_size(tensors: list[TensorStatistics]) -> int:
        """Calculate total size of tensors in a layer."""
        return sum(t.size_bytes for t in tensors)

    @staticmethod
    def calculate_peak_memory_pipelined(
        block_sizes: list[int],
        non_offloaded_memory: int,
    ) -> int:
        """
        Calculate peak GPU memory for pipelined execution.

        Peak memory includes:
        - All blocks (must be pre-allocated on GPU for pipelining)
        - Non-offloaded tensors (always stay on GPU)

        Args:
            block_sizes: Size of each block in bytes (integers).
            non_offloaded_memory: Memory used by tensors that are not offloaded.

        Returns:
            Total peak GPU memory in bytes (integer).
        """
        # All blocks are pre-allocated on GPU
        total_block_memory = sum(block_sizes)

        # Non-offloaded tensors always stay on GPU
        return total_block_memory + non_offloaded_memory

    def _compute_with_assignment_strategy(
        self,
        layer_data: list[tuple[str, list[TensorStatistics], float]],
        layer_stats: list[LayerStatistics],
        non_offloaded_memory: int,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """
        Compute strategy using an external assignment strategy.

        Uses the provided assignment_strategy to determine layer-to-block
        assignment, then calculates block sizes accordingly.

        Args:
            layer_data: List of (label, tensors, duration) per layer.
            layer_stats: Original layer statistics.
            non_offloaded_memory: Memory of tensors not being offloaded.
            max_gpu_mem_bytes: Hard GPU memory limit in bytes (never exceed).
                When ``None``, the memory constraint is not checked.

        Returns:
            StrategyResult containing strategy_map and block_data.
        """
        n_blocks = self.n_blocks

        # Get layer sizes
        layer_sizes = [self.calculate_layer_tensor_size(tensors) for _label, tensors, _duration in layer_data]

        # Block sizing uses TRANSFER sizes: during layer i's execution, the
        # data transferred is the NEXT layer's offload (strategy[label_i] =
        # layer_data[i+1]'s tensors).  So the weight for assigning layer i to
        # a block is layer_sizes[i+1], giving transfer_sizes = layer_sizes[1:].
        # This also naturally excludes the last layer (no subsequent transfer).
        transfer_sizes = layer_sizes[1:] if len(layer_sizes) > 1 else layer_sizes

        # Use external assignment strategy (guaranteed non-None by caller check)
        best_assignment = self.assignment_strategy.compute(transfer_sizes, n_blocks)

        # Calculate block sizes based on actual transfer volumes
        block_sizes = self._calculate_block_sizes_for_assignment(transfer_sizes, best_assignment, n_blocks)

        # Build strategy using the same format as KnapsackBlock:
        # strategy[layer_N-1.label] = tensors from layer_N (transferred during layer_N-1's compute)
        strategy: dict[str, list[TensorStatistics]] = {}
        layer_to_block_map: dict[str, int] = {}

        for layer_idx in range(len(best_assignment)):
            label = layer_data[layer_idx][0]
            layer_to_block_map[label] = best_assignment[layer_idx]
            if layer_idx > 0:
                prev_label = layer_data[layer_idx - 1][0]
                strategy[prev_label] = list(layer_data[layer_idx][1])

        # Last layer's tensors are still offloaded — transferred during the
        # second-to-last layer's compute — but the last layer itself has no block.
        if len(layer_data) > 1:
            prev_label = layer_data[-2][0]
            strategy[prev_label] = list(layer_data[-1][1])

        # Store optimal configuration
        self.optimal_block_sizes = block_sizes
        self.optimal_layer_to_block = layer_to_block_map
        self.optimal_non_offloaded_memory = non_offloaded_memory
        self.optimal_peak_memory = self.calculate_peak_memory_pipelined(block_sizes, non_offloaded_memory)

        # Validate against memory budget (only when hard limit is set)
        if max_gpu_mem_bytes is not None and self.optimal_peak_memory > max_gpu_mem_bytes:
            min_required_mb = self.optimal_peak_memory / 1024 / 1024
            budget_mb = max_gpu_mem_bytes / 1024 / 1024
            excess_mb = min_required_mb - budget_mb

            warnings.warn(
                f"Cannot satisfy GPU memory constraint: "
                f"minimum required = {min_required_mb:.1f}MB (determined by tensor sizes), "
                f"budget = {budget_mb:.1f}MB, "
                f"excess = {excess_mb:.1f}MB. "
                f"Consider increasing max_gpu_mem_bytes or reducing tensor sizes.",
                UserWarning,
                stacklevel=2,
            )

        # Build block data from optimized values
        block_data = self._create_block_data(layer_stats, strategy)

        return StrategyResult(strategy_map=strategy, block_data=block_data)

    def _calculate_block_sizes_for_assignment(
        self,
        layer_sizes: list[int],
        assignment: list[int],
        n_blocks: int,
    ) -> list[int]:
        """Calculate block sizes based on layer assignment."""
        block_sizes = [0] * n_blocks
        for layer_idx, block_idx in enumerate(assignment):
            block_sizes[block_idx] = max(block_sizes[block_idx], layer_sizes[layer_idx])
        return block_sizes

    def compute(
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """
        Compute offload strategy using pipelined block optimization.

        When ``max_gpu_mem_bytes`` is set (memory mode), the strategy will
        automatically increase ``scale`` beyond its initial value if the
        initial offloading is insufficient to meet the memory constraint.

        Args:
            layer_stats: List of layer statistics containing tensor info and durations.
            memory_stats: Optional dict mapping tensor size (bytes) to transfer time (ms).
                If provided, uses this data. Otherwise falls back to extraction from layer stats.
            max_gpu_mem_bytes: Hard GPU memory limit in bytes (never exceed).
                When ``None``, latency mode — offload what fits in ``scale * duration``.

        Returns:
            StrategyResult containing strategy_map and block_data.
        """
        # Handle empty layers - nothing to offload
        if not layer_stats:
            return StrategyResult(strategy_map={}, block_data=None)

        # Use provided memory_stats or fall back to extraction from layer stats
        memory_transfers = memory_stats
        if memory_transfers is None:
            memory_transfers = extract_memory_transfers_from_layer_stats(layer_stats)

        if not memory_transfers:
            raise StrategyComputeError(
                "No memory transfer data available. Either provide memory_stats or ensure "
                "layer_stats contains transfer timing data from profiling."
            )

        memory_transfer_interpolator = MemoryTransferInterpolator(memory_transfers)

        # Collect layer data (tensors to be offloaded)
        # When latency_priority=True, caps offload per layer to what fits in scale * compute time
        layer_data = self._collect_layer_tensors(layer_stats, memory_transfer_interpolator, latency_priority=True)

        if not layer_data:
            return StrategyResult(strategy_map={}, block_data=None)

        # Calculate memory of non-offloaded tensors (always on GPU)
        non_offloaded_memory = self._calculate_non_offloaded_memory(layer_stats, offloaded_layer_data=layer_data)

        # When memory mode is active, suppress the "Cannot satisfy GPU memory
        # constraint" warning from the initial compute since we may auto-adjust
        # scale afterward.
        target = max_gpu_mem_bytes
        if target is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Cannot satisfy GPU memory constraint", category=UserWarning)
                result = self._compute_with_assignment_strategy(
                    layer_data, layer_stats, non_offloaded_memory, max_gpu_mem_bytes=max_gpu_mem_bytes
                )
        else:
            result = self._compute_with_assignment_strategy(
                layer_data, layer_stats, non_offloaded_memory, max_gpu_mem_bytes=max_gpu_mem_bytes
            )

        # Memory mode: auto-increase scale if peak memory exceeds target
        if target is not None and self.optimal_peak_memory > target:
            result = self._adjust_scale_for_memory(
                layer_stats, memory_transfer_interpolator, target, max_gpu_mem_bytes=max_gpu_mem_bytes
            )

        return result

    def _adjust_scale_for_memory(
        self,
        layer_stats: list[LayerStatistics],
        interpolator: MemoryTransferInterpolator,
        memory_budget: int,
        max_attempts: int = 40,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """Iteratively increase scale until peak GPU fits within the memory budget.

        Uses binary search to find the lowest scale that meets the constraint,
        minimising latency overhead while satisfying the GPU memory budget.

        ``self.scale`` is only updated once at the end to the final chosen value,
        so the object remains in a consistent state even if an exception is raised
        during the search.

        Args:
            layer_stats: Layer statistics.
            interpolator: Memory transfer interpolator.
            memory_budget: Target peak GPU memory in bytes.
            max_attempts: Maximum binary search iterations.
            max_gpu_mem_bytes: Hard GPU memory limit in bytes, threaded through
                to ``_compute_with_assignment_strategy`` for constraint validation.

        Returns:
            StrategyResult with the adjusted strategy.
        """
        original_scale = self.scale

        # Estimate upper bound: total model size / budget to get rough idea
        total_model_size = sum(sum(t.size_bytes for t in layer.tensors) for layer in layer_stats)
        needed_ratio = total_model_size / memory_budget if memory_budget > 0 else 100.0
        high_scale = max(original_scale * needed_ratio * 2, original_scale * 100)
        low_scale = original_scale

        best_result: StrategyResult | None = None
        best_peak = self.optimal_peak_memory

        for _ in range(max_attempts):
            mid_scale = (low_scale + high_scale) / 2

            layer_data = self._collect_layer_tensors(layer_stats, interpolator, latency_priority=True, scale=mid_scale)
            if not layer_data:
                low_scale = mid_scale
                continue

            non_offloaded = self._calculate_non_offloaded_memory(layer_stats, offloaded_layer_data=layer_data)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Cannot satisfy GPU memory constraint", category=UserWarning)
                result = self._compute_with_assignment_strategy(
                    layer_data, layer_stats, non_offloaded, max_gpu_mem_bytes=max_gpu_mem_bytes
                )

            if self.optimal_peak_memory <= memory_budget:
                best_result = result
                best_peak = self.optimal_peak_memory
                high_scale = mid_scale
            else:
                low_scale = mid_scale

            if abs(high_scale - low_scale) < 0.001:
                break

        # Determine final scale: high_scale is the last known-good value (midpoint
        # can overshoot due to discrete tensor selection boundaries).
        final_scale = high_scale

        if best_result is not None:
            layer_data = self._collect_layer_tensors(
                layer_stats, interpolator, latency_priority=True, scale=final_scale
            )
            if layer_data:
                non_offloaded = self._calculate_non_offloaded_memory(layer_stats, offloaded_layer_data=layer_data)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=r"Cannot satisfy GPU memory constraint", category=UserWarning
                    )
                    best_result = self._compute_with_assignment_strategy(
                        layer_data, layer_stats, non_offloaded, max_gpu_mem_bytes=max_gpu_mem_bytes
                    )

            self.scale = final_scale
            if final_scale > original_scale:
                warnings.warn(
                    f"Original scale ({original_scale}) is insufficient to meet GPU memory target "
                    f"({memory_budget / 1024**3:.2f} GB). "
                    f"Adjusted scale to {final_scale:.4f}.",
                    stacklevel=3,
                )
            return best_result

        # Could not meet constraint — use best-effort scale
        layer_data = self._collect_layer_tensors(layer_stats, interpolator, latency_priority=True, scale=final_scale)
        if layer_data:
            non_offloaded = self._calculate_non_offloaded_memory(layer_stats, offloaded_layer_data=layer_data)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Cannot satisfy GPU memory constraint", category=UserWarning)
                result = self._compute_with_assignment_strategy(
                    layer_data, layer_stats, non_offloaded, max_gpu_mem_bytes=max_gpu_mem_bytes
                )
        else:
            result = StrategyResult(strategy_map={}, block_data=None)

        self.scale = final_scale
        warnings.warn(
            f"Cannot satisfy GPU memory constraint: "
            f"minimum required = {best_peak / 1024 / 1024:.1f}MB (determined by tensor sizes), "
            f"budget = {memory_budget / 1024 / 1024:.1f}MB, "
            f"excess = {(best_peak - memory_budget) / 1024 / 1024:.1f}MB. "
            f"Consider increasing max_gpu_mem_bytes or reducing tensor sizes.",
            UserWarning,
            stacklevel=3,
        )
        return result

    def _create_block_data(
        self,
        layer_stats: list[LayerStatistics],
        strategy_map: dict[str, list[TensorStatistics]],
    ) -> BlockStrategyData:
        """Build BlockStrategyData from the optimized values."""
        label_to_size_map = compute_label_to_size_map(layer_stats, strategy_map)

        # Only include layers that have tensors to offload in block data.
        # The loader uses label_to_block_id to gate schedule_transfer calls,
        # so non-offloading labels must be excluded to avoid invalid transfers.
        allocation_ordered: dict[int, list[str]] = {}
        label_to_block_id_filtered: dict[str, int] = {}
        for label, block_id in self.optimal_layer_to_block.items():
            if label not in strategy_map:
                continue
            label_to_block_id_filtered[label] = block_id
            if block_id not in allocation_ordered:
                allocation_ordered[block_id] = []
            allocation_ordered[block_id].append(label)

        # Convert block_sizes to dict, excluding unused (zero-sized) blocks
        block_sizes_dict: dict[int, int] = {i: size for i, size in enumerate(self.optimal_block_sizes) if size > 0}

        # Compute transfer_to_compute_map
        transfer_to_compute_map = calculate_transfer_to_compute_map(layer_stats, strategy_map)

        return BlockStrategyData(
            label_to_size_map=label_to_size_map,
            allocation_ordered=allocation_ordered,
            block_sizes=block_sizes_dict,
            label_to_block_id=label_to_block_id_filtered,
            transfer_to_compute_map=transfer_to_compute_map,
        )


class GlobalTensorSelectionStrategy:
    """
    Weight-level metaheuristic optimization strategy for GPU memory offloading.

    This strategy optimizes the binary decision of whether to offload each weight,
    maximizing GPU memory utilization while respecting memory limits and ensuring
    transfers fit within pipelined execution windows.

    Unlike GlobalOffloadStrategy which optimizes block sizes and layer assignments,
    this strategy performs fine-grained weight selection to find the optimal
    subset of weights to offload.

    Decision variables:
    - Binary x[i] for each weight: 1 = offload, 0 = keep on GPU
    - Continuous scale: multiplier for compute time (transfer window)

    Constraints:
    - Hard: Peak GPU memory <= max_gpu_mem_bytes (when set)
    - Hard: Transfer time for layer N <= (layer N-1's compute time) * scale

    Objective (maximize):
    - GPU utilization relative to max_gpu_mem (when set)
    - Offload ratio (secondary)
    - Prefer lower scale (less latency overhead)

    Memory Model:
    - Tensors NOT offloaded stay on GPU permanently (preloaded)
    - Peak GPU = sum(all kept tensors) + sum(block sizes)
    - Blocks are pre-allocated for pipelined transfers

    Supported Optimizers (via scipy):
    - DE: Differential Evolution (scipy.optimize.differential_evolution) -
        RECOMMENDED default. Natively handles binary variables via integrality
        constraints. Best for high-dimensional problems with many tensors.
    - SA: Simulated Annealing (scipy.optimize.dual_annealing) -
        Treats variables as continuous; less efficient for binary decisions.
        May be useful for small problems with few tensors.
    """

    def __init__(
        self,
        scale: float = 1.0,
        scale_ub: float | None = None,
        n_blocks: int = 4,
        min_blocks: int | None = None,
        max_blocks: int | None = None,
        threshold_mb: float = 0.1,
        pop_size: int = 50,
        epoch: int = 200,
        max_early_stop: int = 25,
        optimizer: str = "DE",
        seed_ratio: float = 0.25,
        seed: int | None = None,
        assignment_strategy: AssignmentStrategy | None = None,
        transfer_window: TransferWindowCalculator | None = None,
    ):
        """
        Initialize GlobalTensorSelectionStrategy.

        Args:
            scale: Duration multiplier (baseline / lower bound for the optimizer).
                ``0.9`` = 10% safety margin, ``1.0`` = exact fit (default).
            scale_ub: Upper bound for scale the optimizer may explore.
                When ``None`` (default), defaults to ``scale`` (latency mode).
                ``compute()`` automatically widens to ``100.0`` in memory mode
                (``max_gpu_mem_bytes`` set) unless the caller explicitly set
                ``scale_ub``.
                When explicitly set, used as-is. Example: ``scale=0.9, scale_ub=1.0``
                means transfers should fit within 90-100% of compute time.
            n_blocks: Number of memory blocks (max). Used if assignment_strategy is None.
            min_blocks: Minimum number of blocks to try (default: 2).
            max_blocks: Maximum number of blocks to try (default: n_blocks).
            threshold_mb: Minimum tensor size threshold in MB (smaller tensors always kept on GPU).
            pop_size: Population size for optimization.
            epoch: Maximum number of optimization epochs.
            max_early_stop: Stop if no improvement for this many epochs.
            optimizer: Optimizer to use (default: "DE").
                Available options:
                - "DE": Differential Evolution (scipy.optimize.differential_evolution).
                    RECOMMENDED. Natively handles binary variables via integrality
                    constraints. Best for high-dimensional problems with many tensors.
                - "SA": Simulated Annealing (scipy.optimize.dual_annealing).
                    Treats variables as continuous; less efficient for binary decisions.
                    May be useful for small problems with few tensors.
            seed_ratio: Ratio of population to seed with feasible solutions (0.0-1.0).
                Lower ratio = more random exploration.
            seed: Random seed for reproducibility. If None, uses non-deterministic seed.
            assignment_strategy: Strategy for assigning layers to blocks.
                Default: OptimizedRoundRobinAssignment (tries min_blocks to max_blocks).
                Options: StrictRoundRobinAssignment, OptimizedRoundRobinAssignment.
            transfer_window: Calculator for per-layer transfer windows.
                Default: SingleLayerWindow (previous layer's duration only).
                Use GapAwareWindow to exploit empty layers for larger transfer budgets.
        """
        validate_memory_params(scale)
        # For backward compatibility and internal use, keep scale_lb
        self.scale_lb = scale
        # Resolve scale_ub default based on mode
        if scale_ub is not None:
            self.scale_ub = scale_ub
        else:
            self.scale_ub = scale  # Default to latency mode; compute() adjusts for memory mode
        self._user_set_scale_ub = scale_ub is not None
        self.n_blocks = n_blocks
        self.min_blocks = min_blocks if min_blocks is not None else 2
        self.max_blocks = max_blocks if max_blocks is not None else n_blocks
        self.threshold_mb = threshold_mb
        self.pop_size = pop_size
        self.epoch = epoch
        self.max_early_stop = max_early_stop
        self.optimizer_name = optimizer
        self.seed_ratio = seed_ratio
        self.seed = seed
        self.assignment_strategy = assignment_strategy or OptimizedRoundRobinAssignment(
            min_blocks=self.min_blocks,
            max_blocks=self.max_blocks,
        )
        self.transfer_window: TransferWindowCalculator = transfer_window or SingleLayerWindow()

        # Results stored after compute()
        self.optimal_tensor_selection: dict[str, list[bool]] = {}
        self.optimal_peak_memory: int = 0
        self.optimal_latency_overhead: float = 0.0
        self.optimal_scale: float = 1.0
        self.optimal_block_sizes: list[int] = []
        self.optimal_layer_to_block: dict[str, int] = {}

    @property
    def scale(self) -> float:
        """Duration multiplier (lower bound). Alias for ``scale_lb``."""
        return self.scale_lb

    @scale.setter
    def scale(self, value: float) -> None:
        self.scale_lb = value

    def compute(  # noqa: C901
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """Run the core optimizer.

        Args:
            layer_stats: List of layer statistics containing tensor info and durations.
            memory_stats: Optional dict mapping tensor size (bytes) to transfer time (ms).
                If provided, uses this data. Otherwise falls back to extraction from layer stats.
            max_gpu_mem_bytes: Hard GPU memory limit in bytes (never exceed).
                When ``None``, the memory constraint is disabled and the optimizer
                only respects the transfer time constraint (latency mode).

        Returns:
            StrategyResult containing strategy_map and block_data.
        """
        from scipy.optimize import differential_evolution, dual_annealing

        # Compute effective scale_ub locally (do NOT mutate self.scale_ub)
        effective_scale_ub = self.scale_ub
        if max_gpu_mem_bytes is not None and not self._user_set_scale_ub:
            effective_scale_ub = 100.0

        # Handle empty layers - nothing to offload
        if not layer_stats:
            return StrategyResult(strategy_map={}, block_data=None)

        # Get memory transfer interpolator
        memory_transfers = memory_stats
        if memory_transfers is None:
            memory_transfers = extract_memory_transfers_from_layer_stats(layer_stats)

        if not memory_transfers:
            raise StrategyComputeError(
                "No memory transfer data available. Either provide memory_stats or ensure "
                "layer_stats contains transfer timing data from profiling."
            )

        interpolator = MemoryTransferInterpolator(memory_transfers)
        threshold_bytes = self.threshold_mb * 1024 * 1024

        # Pass 1: identify offloadable tensors per layer.
        # This determines permanent gaps (layers with no offloadable tensors)
        # which the transfer window calculator uses to compute effective windows.
        layer_tensor_counts: dict[str, int] = {}
        layer_offloadable: list[tuple[int, list[TensorStatistics]]] = []

        for layer_idx, layer in enumerate(layer_stats):
            # First layer is always on GPU (preloaded — no previous layer to
            # pipeline the transfer).  Its tensors must not be offered to the
            # optimizer; otherwise the peak memory estimate would be too low.
            if layer_idx == 0:
                layer_tensor_counts[layer.label] = 0
                continue
            offloadable_tensors = [t for t in layer.tensors if t.size_bytes > threshold_bytes]
            layer_tensor_counts[layer.label] = len(offloadable_tensors)
            if offloadable_tensors:
                layer_offloadable.append((layer_idx, offloadable_tensors))

        # Build static offload-size indicator for permanent gap detection.
        # Non-zero for layers that have offloadable tensors (always have some
        # offload); zero for permanent gaps (no offloadable tensors at all).
        num_layers = len(layer_stats)
        static_offload_sizes = np.zeros(num_layers, dtype=np.float64)
        for layer_idx, tensors in layer_offloadable:
            static_offload_sizes[layer_idx] = sum(t.size_bytes for t in tensors)
        layer_durations_arr = np.array([layer.duration for layer in layer_stats], dtype=np.float64)

        # Detect permanent gaps and switch to GapAwareWindow for wider
        # transfer budgets.  Gap layers (0 tensors) stay on GPU at zero
        # memory cost; the benefit is purely from the wider window.
        has_permanent_gaps = any(static_offload_sizes[i] == 0.0 and i > 0 for i in range(num_layers))
        transfer_window_calc: TransferWindowCalculator = self.transfer_window
        if has_permanent_gaps and isinstance(self.transfer_window, SingleLayerWindow):
            transfer_window_calc = GapAwareWindow()

        # Pass 2: classify layers as fixed vs variable using effective windows.
        # Layers whose full offload fits within the transfer budget at scale_lb
        # are "fixed" (always offloaded) and excluded from the optimizer to
        # reduce dimensionality.
        # Structure: list of (layer_idx, layer_label, tensor_idx, tensor, layer_duration)
        tensor_info: list[tuple[int, str, int, TensorStatistics, float]] = []
        fixed_offload_tensors: dict[str, list[TensorStatistics]] = {}
        fixed_offload_size_by_layer: dict[int, int] = {}

        for layer_idx, offloadable_tensors in layer_offloadable:
            total_layer_offload = sum(t.size_bytes for t in offloadable_tensors)
            effective_window = transfer_window_calc.compute_window(layer_idx, static_offload_sizes, layer_durations_arr)
            fits = False
            if effective_window > 0.0:
                transfer_time = interpolator.bytes_to_duration(total_layer_offload)
                fits = transfer_time <= effective_window * self.scale_lb * 1.001

            if fits:
                fixed_offload_tensors[layer_stats[layer_idx].label] = offloadable_tensors
                fixed_offload_size_by_layer[layer_idx] = total_layer_offload
            else:
                for tensor_idx, tensor in enumerate(offloadable_tensors):
                    tensor_info.append((
                        layer_idx,
                        layer_stats[layer_idx].label,
                        tensor_idx,
                        tensor,
                        layer_stats[layer_idx].duration,
                    ))

        if not tensor_info and not fixed_offload_tensors:
            return StrategyResult(strategy_map={}, block_data=None)

        # Short-circuit: all layers fit within the transfer budget — skip optimizer
        if not tensor_info:
            return self._build_fixed_only_result(
                layer_stats, fixed_offload_tensors, fixed_offload_size_by_layer, threshold_bytes
            )

        num_tensors = len(tensor_info)

        # Short-circuit: all layers fit within the transfer budget — skip optimizer
        if not tensor_info:
            return self._build_fixed_only_result(
                layer_stats, fixed_offload_tensors, fixed_offload_size_by_layer, threshold_bytes
            )

        num_tensors = len(tensor_info)

        # Pre-compute layer info for objective function
        layer_labels = [layer.label for layer in layer_stats]

        # Calculate per-layer small tensor sizes (tensors below threshold, always on GPU)
        layer_small_tensor_sizes = {
            layer.label: sum(t.size_bytes for t in layer.tensors if t.size_bytes <= threshold_bytes)
            for layer in layer_stats
        }

        # Capture variables for closure
        max_gpu_mem = max_gpu_mem_bytes
        n_blocks = self.n_blocks
        scale_lb = self.scale_lb
        scale_ub = effective_scale_ub

        # Compute layer sizes for assignment strategy.
        # Block sizing uses TRANSFER sizes: during layer i's execution, the data
        # transferred is the NEXT layer's offload.  So the weight for assigning
        # layer i to a block is layer_sizes[i+1], giving layer_sizes[1:].
        layer_sizes = [sum(t.size_bytes for t in layer.tensors) for layer in layer_stats]
        layer_sizes_for_assignment = layer_sizes[1:] if len(layer_sizes) > 1 else layer_sizes

        # Use assignment strategy to compute layer-to-block mapping for N-1 layers
        layer_to_block = self.assignment_strategy.compute(layer_sizes_for_assignment, n_blocks)

        # --- Precompute numpy arrays for vectorized objective function ---
        # Tensor properties as contiguous arrays (avoids per-call dict/list operations)
        tensor_sizes_arr = np.array([t.size_bytes for _, _, _, t, _ in tensor_info], dtype=np.float64)
        tensor_layer_idx_arr = np.array([layer_idx for layer_idx, _, _, _, _ in tensor_info], dtype=np.intp)

        # Constant memory values
        total_small_tensor_memory = sum(layer_small_tensor_sizes.values())
        total_large_tensor_memory = float(np.sum(tensor_sizes_arr))
        # First layer's large tensors are always on GPU (not in tensor_info)
        first_layer_resident = (
            sum(t.size_bytes for t in layer_stats[0].tensors if t.size_bytes > threshold_bytes) if layer_stats else 0
        )
        total_model_size = sum(layer_sizes)

        # Pre-fixed offload sizes per layer (layers whose tensors all fit at scale_lb)
        fixed_offload_sizes_arr = np.zeros(num_layers, dtype=np.float64)
        for layer_idx, size in fixed_offload_size_by_layer.items():
            fixed_offload_sizes_arr[layer_idx] = size

        # Block assignment: precompute per-block layer index masks
        layer_to_block_obj: Any = layer_to_block
        layer_to_block_list = (
            list(layer_to_block_obj.values()) if isinstance(layer_to_block_obj, dict) else list(layer_to_block_obj)
        )
        block_layer_indices: list[np.ndarray] = []
        for b in range(n_blocks):
            indices = np.array([i for i, bid in enumerate(layer_to_block_list) if bid == b], dtype=np.intp)
            block_layer_indices.append(indices)

        # Gap-aware block sizing: for each assignment slot, find the next
        # layer with actual offload data (skipping permanent gaps).  After
        # rearrange_transfers() the block will hold that layer's data.
        # next_real_transfer[i] maps assignment index i to the original
        # layer index whose offload will occupy the block.
        # Without gaps this equals i+1 (standard pipeline).
        # Sentinel: num_layers means "no next real transfer" (last layer or
        # trailing gaps); callers must filter with `sources < num_layers`.
        next_real_transfer = np.arange(1, num_layers + 1, dtype=np.intp)
        if has_permanent_gaps:
            for i in range(num_layers - 1):
                j = i + 1
                while j < num_layers and static_offload_sizes[j] == 0.0:
                    j += 1
                next_real_transfer[i] = min(j, num_layers)

        # Keep this loosely typed to avoid beartype work inside optimizer loops.
        def objective(solution: Any) -> Any:  # noqa: C901
            """Objective function for tensor selection optimization (minimization).

            Solution format: [tensor_0, tensor_1, ..., tensor_N, scale]
            - tensor_i: binary (0 = keep on GPU, 1 = offload)
            - scale: continuous [scale_lb, scale_ub]
            """
            solution_scale = float(solution[-1])

            # Vectorized: compute per-layer offload sizes via bincount
            offload_mask = solution[:num_tensors] > 0.5
            offloaded_sizes = np.where(offload_mask, tensor_sizes_arr, 0.0)
            layer_offload_sizes = np.asarray(
                np.bincount(tensor_layer_idx_arr, weights=offloaded_sizes, minlength=num_layers), dtype=np.float64
            )
            layer_offload_sizes += fixed_offload_sizes_arr

            # Block sizes: max TRANSFER size across layers in each block.
            # next_real_transfer[i] maps assignment index i to the layer
            # whose offload occupies the block (skips permanent gaps).
            total_block_memory = 0.0
            if num_layers > 1:
                for b_indices in block_layer_indices:
                    if len(b_indices) > 0:
                        sources = next_real_transfer[b_indices]
                        valid = sources < num_layers
                        if np.any(valid):
                            total_block_memory += float(np.max(layer_offload_sizes[sources[valid]]))

            # Peak memory: blocks + all non-offloaded tensors
            total_offloaded = float(np.sum(offloaded_sizes))
            total_kept_on_gpu = (
                total_small_tensor_memory + first_layer_resident + total_large_tensor_memory - total_offloaded
            )
            peak_memory = total_block_memory + total_kept_on_gpu

            # HARD CONSTRAINT: Must fit in GPU memory (only when max_gpu_mem is set)
            if max_gpu_mem is not None and peak_memory > max_gpu_mem:
                violation_ratio = (peak_memory - max_gpu_mem) / max_gpu_mem
                return 1000.0 * (1 + violation_ratio)

            # Transfer constraint: check layers with non-zero offload
            # Tolerance accounts for log-log interpolation round-trip imprecision
            # between duration_to_bytes and bytes_to_duration.
            required_scale = 1.0
            constraint_violated = False
            scale_budget = solution_scale * 1.001

            for i in range(1, num_layers):
                offload_size_i = layer_offload_sizes[i]
                if offload_size_i > 0.0:
                    effective_dur = transfer_window_calc.compute_window(i, layer_offload_sizes, layer_durations_arr)
                    if effective_dur > 0.0:
                        transfer_time = interpolator.bytes_to_duration(offload_size_i)
                        layer_scale = transfer_time / effective_dur
                        if layer_scale > required_scale:
                            required_scale = layer_scale
                        if transfer_time > effective_dur * scale_budget:
                            constraint_violated = True
                    else:
                        constraint_violated = True

            # HARD CONSTRAINT: Transfers must fit within solution's chosen scale
            if constraint_violated:
                return 1000.0 * (required_scale / solution_scale)

            # Score: prefer peak memory close to max_gpu_mem (latency mode: offload ratio only)
            gpu_score = (peak_memory / max_gpu_mem if max_gpu_mem > 0 else 1.0) if max_gpu_mem is not None else 1.0

            offload_ratio = total_offloaded / total_model_size if total_model_size > 0 else 0.0
            score = (gpu_score + offload_ratio * 0.5) / solution_scale
            return -score

        # Precompute per-layer tensor index lookup (avoids O(num_tensors) scan per layer)
        label_to_tensor_indices: dict[str, list[tuple[int, TensorStatistics]]] = {}
        for idx, (_, label, _, tensor, _) in enumerate(tensor_info):
            if label not in label_to_tensor_indices:
                label_to_tensor_indices[label] = []
            label_to_tensor_indices[label].append((idx, tensor))

        # Pre-sort by size (ascending) for feasible builder, avoid re-sorting each call
        label_to_tensors_asc = {
            label: sorted(tensors, key=lambda x: x[1].size_bytes) for label, tensors in label_to_tensor_indices.items()
        }

        # Generate initial feasible solutions to help optimizer

        def _build_feasible_at_scale(target_scale: float) -> np.ndarray:
            """Build a tensor selection that fits within transfer budget at given scale."""
            selection = np.zeros(num_tensors, dtype=float)

            for layer_idx in range(1, num_layers):
                effective_dur = transfer_window_calc.compute_window(
                    layer_idx, static_offload_sizes, layer_durations_arr
                )
                budget = interpolator.duration_to_bytes(effective_dur * target_scale) * 0.995

                current_transfer = 0
                for idx, tensor in label_to_tensors_asc.get(layer_labels[layer_idx], []):
                    if current_transfer + tensor.size_bytes <= budget:
                        selection[idx] = 1.0
                        current_transfer += tensor.size_bytes

            return np.append(selection, target_scale)

        def _peak_for_solution(solution: np.ndarray) -> float:
            offload_mask = solution[:num_tensors] > 0.5
            offloaded = np.where(offload_mask, tensor_sizes_arr, 0.0)
            per_layer = np.asarray(
                np.bincount(tensor_layer_idx_arr, weights=offloaded, minlength=num_layers), dtype=np.float64
            )
            per_layer += fixed_offload_sizes_arr
            if num_layers > 1:
                blk_mem = 0.0
                for bi in block_layer_indices:
                    if len(bi) > 0:
                        src = next_real_transfer[bi]
                        v = src < num_layers
                        if np.any(v):
                            blk_mem += float(np.max(per_layer[src[v]]))
            else:
                blk_mem = 0.0
            return blk_mem + total_small_tensor_memory + total_large_tensor_memory - float(np.sum(offloaded))

        def generate_feasible_solution() -> np.ndarray:
            """Generate a solution where both transfer and memory constraints are met."""
            sol = _build_feasible_at_scale(scale_lb)
            if max_gpu_mem is None or _peak_for_solution(sol) <= max_gpu_mem:
                return sol
            # Memory constraint violated at scale_lb — binary search for minimum valid scale
            lo, hi = scale_lb, min(scale_ub, scale_lb * 20)
            best_sol = sol
            for _ in range(20):
                mid = (lo + hi) / 2
                trial = _build_feasible_at_scale(mid)
                if _peak_for_solution(trial) <= max_gpu_mem:
                    hi = mid
                    best_sol = trial
                else:
                    lo = mid
            return best_sol

        def generate_aggressive_solution() -> np.ndarray:
            """Generate a solution that offloads everything (may need higher scale)."""
            selection = np.ones(num_tensors, dtype=float)
            return np.append(selection, scale_ub)

        def generate_conservative_solution() -> np.ndarray:
            """Generate a solution that maximizes GPU utilization (keeps more on GPU)."""
            selection = np.zeros(num_tensors, dtype=float)

            for layer_idx in range(1, num_layers):
                effective_dur = transfer_window_calc.compute_window(
                    layer_idx, static_offload_sizes, layer_durations_arr
                )
                max_transfer_bytes = interpolator.duration_to_bytes(effective_dur * scale_ub)

                layer_tensors = label_to_tensor_indices.get(layer_labels[layer_idx], [])
                # Sort descending by size (largest first for conservative packing)
                layer_tensors_desc = sorted(layer_tensors, key=lambda x: -x[1].size_bytes)

                layer_total_size = sum(t.size_bytes for _, t in layer_tensors_desc)
                current_offload = 0

                if layer_total_size <= max_transfer_bytes:
                    for idx, _ in layer_tensors_desc:
                        selection[idx] = 1.0
                    continue

                for idx, tensor in layer_tensors_desc:
                    if current_offload + tensor.size_bytes <= max_transfer_bytes:
                        selection[idx] = 1.0
                        current_offload += tensor.size_bytes

            return np.append(selection, scale_ub)

        # Build initial population
        starting_solutions: list[np.ndarray] = []

        n_feasible = max(1, int(self.pop_size * self.seed_ratio))
        for _ in range(n_feasible):
            starting_solutions.append(generate_feasible_solution())

        if len(starting_solutions) < self.pop_size:
            starting_solutions.append(generate_conservative_solution())

        if len(starting_solutions) < self.pop_size:
            starting_solutions.append(generate_aggressive_solution())

        rng = np.random.default_rng(self.seed)
        while len(starting_solutions) < self.pop_size:
            random_selection = rng.integers(0, 2, num_tensors).astype(float)
            random_scale = rng.uniform(scale_lb, scale_ub)
            starting_solutions.append(np.append(random_selection, random_scale))

        starting_solutions_arr = np.array(starting_solutions)

        # Variable bounds: binary for each tensor + continuous for scale
        opt_bounds = [(0, 1)] * num_tensors + [(scale_lb, scale_ub)]

        # Run optimizer
        if self.optimizer_name == "SA":
            # Simulated Annealing via dual_annealing
            # dual_annealing requires strict lower < upper for all bounds.
            # When scale_lb == scale_ub (fixed scale), add a tiny epsilon.
            sa_bounds = list(opt_bounds)
            if scale_lb >= scale_ub:
                sa_bounds[-1] = (scale_lb, scale_ub + 1e-10)

            x0 = generate_feasible_solution()
            early_stop = EarlyStopCallback(
                max_stall=self.max_early_stop,
                objective_func=objective,
            )
            result: Any = dual_annealing(
                objective,
                bounds=sa_bounds,
                x0=x0,
                maxiter=self.epoch,
                # Cap total function evaluations to pop_size * epoch. Default
                # maxfun=1e7 is far too large for high-dimensional binary
                # problems and causes the optimizer to run for hours.
                maxfun=self.pop_size * self.epoch,
                # Disable local search: variables are mostly binary so gradient-based
                # local optimization (L-BFGS-B) is useless and very expensive.
                no_local_search=True,
                seed=self.seed if self.seed is not None else 42,
                callback=early_stop,
            )
        else:
            # Differential Evolution (also covers former "GA", "DE", "WOA", "GBO")
            integrality_flags = [True] * num_tensors + [False]
            early_stop = EarlyStopCallback(
                max_stall=self.max_early_stop,
                objective_func=objective,
            )
            result = differential_evolution(
                objective,
                bounds=cast("Any", opt_bounds),
                integrality=integrality_flags,
                init=starting_solutions_arr,
                maxiter=self.epoch,
                tol=0.01,
                seed=self.seed if self.seed is not None else 42,
                polish=False,
                callback=cast("Any", early_stop),
            )

        # Extract solution — fall back to fixed-only tensors if optimizer produced NaN
        solution = result.x
        if np.any(np.isnan(solution)):
            tensor_selection = [False] * num_tensors
            optimal_scale = scale_lb
        else:
            tensor_selection = [bool(x > 0.5) for x in solution[:-1]]
            optimal_scale = float(solution[-1])

        # Clamp scale to bounds (optimizer might produce values slightly outside due to numerics)
        optimal_scale = max(scale_lb, min(scale_ub, optimal_scale))

        # Store optimized scale
        self.optimal_scale = optimal_scale

        # Build strategy_map from selection
        # Format: strategy[prev_layer.label] = tensors from next layer to transfer during prev layer
        strategy: dict[str, list[TensorStatistics]] = {}

        # Group selected tensors by layer (optimizer-chosen + pre-fixed)
        layer_selected_tensors: dict[str, list[TensorStatistics]] = {}
        for idx, (_, label, _, tensor, _) in enumerate(tensor_info):
            if tensor_selection[idx]:
                if label not in layer_selected_tensors:
                    layer_selected_tensors[label] = []
                layer_selected_tensors[label].append(tensor)

        for label, tensors in fixed_offload_tensors.items():
            if label not in layer_selected_tensors:
                layer_selected_tensors[label] = []
            layer_selected_tensors[label].extend(tensors)

        # Convert to strategy format (shifted by one layer for pipeline).
        # Skip gap labels (no offloaded tensors) so transfers bridge across gaps,
        # matching GlobalOffloadStrategy's behaviour.
        layer_labels_list = [layer.label for layer in layer_stats]
        strategy = self._build_pipeline_strategy_map(layer_labels_list, layer_selected_tensors)

        # Store optimal results
        self.optimal_tensor_selection = {
            label: [tensor_selection[idx] for idx, (_, lbl, _, _, _) in enumerate(tensor_info) if lbl == label]
            for label in {lbl for _, lbl, _, _, _ in tensor_info}
        }
        for label, tensors in fixed_offload_tensors.items():
            self.optimal_tensor_selection[label] = [True] * len(tensors)

        # Use the same block assignment the objective function was optimized
        # against so that peak memory is consistent with the optimizer's
        # constraint check.
        gap_layer_to_block: dict[str, int] = {
            layer_labels[i]: layer_to_block_list[i] for i in range(len(layer_to_block_list))
        }

        block_sizes = [0] * self.n_blocks
        for label, tensors in strategy.items():
            blk = gap_layer_to_block[label]
            block_sizes[blk] = max(block_sizes[blk], sum(t.size_bytes for t in tensors))

        # Peak memory = sum(block_sizes) + total_kept_on_gpu
        total_block_memory = sum(block_sizes)
        total_kept_on_gpu = 0
        for layer in layer_stats:
            total_kept_on_gpu += layer_small_tensor_sizes[layer.label]
            if layer.label in layer_selected_tensors:
                offloaded_ids = {t.tensor_id for t in layer_selected_tensors[layer.label]}
                kept_large = sum(
                    t.size_bytes
                    for t in layer.tensors
                    if t.size_bytes > threshold_bytes and t.tensor_id not in offloaded_ids
                )
            else:
                kept_large = sum(t.size_bytes for t in layer.tensors if t.size_bytes > threshold_bytes)
            total_kept_on_gpu += kept_large

        self.optimal_peak_memory = total_block_memory + total_kept_on_gpu
        self.optimal_block_sizes = block_sizes
        self.optimal_layer_to_block = gap_layer_to_block

        # Build block data from optimized values (not recalculated)
        block_data = self._create_block_data(layer_stats, strategy)

        return StrategyResult(strategy_map=strategy, block_data=block_data)

    def _build_fixed_only_result(
        self,
        layer_stats: list[LayerStatistics],
        fixed_offload_tensors: dict[str, list[TensorStatistics]],
        fixed_offload_size_by_layer: dict[int, int],
        threshold_bytes: float,
    ) -> StrategyResult:
        """Build result when all layers fit within transfer budget — no optimizer needed."""
        n_blocks = self.n_blocks
        layer_labels_list = [layer.label for layer in layer_stats]

        # Build strategy_map (shifted by one layer for pipeline).
        # Skip gap labels to bridge transfers across gaps.
        strategy = self._build_pipeline_strategy_map(layer_labels_list, fixed_offload_tensors)

        self.optimal_scale = self.scale_lb
        self.optimal_tensor_selection = {label: [True] * len(ts) for label, ts in fixed_offload_tensors.items()}

        # Re-derive block assignment from the actual transfer pipeline
        # (strategy keys), ensuring proper alternation when gaps are skipped.
        transfer_labels = list(strategy.keys())
        transfer_sizes_final = [sum(t.size_bytes for t in strategy[lb]) for lb in transfer_labels]
        gap_aware_block = list(self.assignment_strategy.compute(transfer_sizes_final, n_blocks))

        block_sizes = [0] * n_blocks
        gap_layer_to_block: dict[str, int] = {}
        for pos, label in enumerate(transfer_labels):
            blk = gap_aware_block[pos]
            gap_layer_to_block[label] = blk
            block_sizes[blk] = max(block_sizes[blk], transfer_sizes_final[pos])

        # Peak memory
        layer_small_tensor_sizes = {
            layer.label: sum(t.size_bytes for t in layer.tensors if t.size_bytes <= threshold_bytes)
            for layer in layer_stats
        }
        total_block_memory = sum(block_sizes)
        total_kept_on_gpu = sum(layer_small_tensor_sizes.values())
        if layer_stats:
            total_kept_on_gpu += sum(t.size_bytes for t in layer_stats[0].tensors if t.size_bytes > threshold_bytes)
        for layer in layer_stats:
            if layer.label in fixed_offload_tensors:
                offloaded_ids = {t.tensor_id for t in fixed_offload_tensors[layer.label]}
                total_kept_on_gpu += sum(
                    t.size_bytes
                    for t in layer.tensors
                    if t.size_bytes > threshold_bytes and t.tensor_id not in offloaded_ids
                )
            elif layer != layer_stats[0]:
                total_kept_on_gpu += sum(t.size_bytes for t in layer.tensors if t.size_bytes > threshold_bytes)

        self.optimal_peak_memory = total_block_memory + total_kept_on_gpu
        self.optimal_block_sizes = block_sizes
        self.optimal_layer_to_block = gap_layer_to_block

        block_data = self._create_block_data(layer_stats, strategy)
        return StrategyResult(strategy_map=strategy, block_data=block_data)

    @staticmethod
    def _build_pipeline_strategy_map(
        layer_labels: list[str],
        tensor_map: dict[str, list[TensorStatistics]],
    ) -> dict[str, list[TensorStatistics]]:
        """Build pipeline-shifted strategy map, skipping gap layers.

        In pipelined offloading, layer *i*'s tensors are transferred during an
        earlier layer's compute.  This method assigns each offloaded layer's
        tensors to the preceding transfer slot (in execution order), skipping
        labels that have no offloaded tensors so that transfers bridge across
        gaps.

        The first offloaded layer's tensors are assigned to the model's first
        layer when it is not itself the first layer.  The last offloaded
        layer's tensors are never assigned a forward slot (they are handled
        by the previous iteration's tail transfer).

        Args:
            layer_labels: All layer labels in execution order.
            tensor_map: Mapping from layer label to its offloaded tensors.

        Returns:
            Strategy map keyed by transfer-slot label.
        """
        strategy: dict[str, list[TensorStatistics]] = {}
        offloaded_order = [label for label in layer_labels if label in tensor_map]
        if offloaded_order:
            first_pos = layer_labels.index(offloaded_order[0])
            if first_pos > 0:
                strategy[layer_labels[0]] = tensor_map[offloaded_order[0]]
            for j in range(1, len(offloaded_order)):
                strategy[offloaded_order[j - 1]] = tensor_map[offloaded_order[j]]
        return strategy

    def _create_block_data(
        self,
        layer_stats: list[LayerStatistics],
        strategy_map: dict[str, list[TensorStatistics]],
    ) -> BlockStrategyData:
        """Build BlockStrategyData from the optimized values."""
        label_to_size_map = compute_label_to_size_map(layer_stats, strategy_map)

        # Only include layers that have tensors to offload in block data.
        # The loader uses label_to_block_id to gate schedule_transfer calls,
        # so non-offloading labels must be excluded to avoid invalid transfers.
        allocation_ordered: dict[int, list[str]] = {}
        label_to_block_id_filtered: dict[str, int] = {}
        for label, block_id in self.optimal_layer_to_block.items():
            if label not in strategy_map:
                continue
            label_to_block_id_filtered[label] = block_id
            if block_id not in allocation_ordered:
                allocation_ordered[block_id] = []
            allocation_ordered[block_id].append(label)

        # Convert block_sizes to dict, excluding unused (zero-sized) blocks
        block_sizes_dict: dict[int, int] = {i: size for i, size in enumerate(self.optimal_block_sizes) if size > 0}

        # Compute transfer_to_compute_map
        transfer_to_compute_map = calculate_transfer_to_compute_map(layer_stats, strategy_map)

        return BlockStrategyData(
            label_to_size_map=label_to_size_map,
            allocation_ordered=allocation_ordered,
            block_sizes=block_sizes_dict,
            label_to_block_id=label_to_block_id_filtered,
            transfer_to_compute_map=transfer_to_compute_map,
        )

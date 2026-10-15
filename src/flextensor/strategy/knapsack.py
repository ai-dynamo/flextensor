# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Knapsack-based offloading strategies for FlexTensor.

This module provides knapsack-based optimization strategies for determining
which tensors to offload to CPU memory during inference.
"""

import warnings
from typing import Any, ClassVar, cast

import numpy

from flextensor.collectors import LayerStatistics, TensorStatistics, extract_memory_transfers_from_layer_stats
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator

from .protocol import BlockStrategyData, StrategyComputeError, StrategyResult
from .utils import EarlyStopCallback, _build_block_data, validate_memory_params

# =============================================================================
# Duration and Scale Functions
# =============================================================================


def _estimate_required_scale(
    layer_stats: list[LayerStatistics],
    memory_transfer_interpolator: MemoryTransferInterpolator,
    needed_offload_bytes: float,
) -> float:
    """
    Estimate the scale factor needed to offload the required bytes.

    Uses the total compute duration and transfer rate to estimate what scale
    would be needed to transfer the required amount of data.

    Args:
        layer_stats: List of layer statistics.
        memory_transfer_interpolator: Interpolator for transfer rate estimation.
        needed_offload_bytes: Total bytes that need to be offloaded.

    Returns:
        Estimated scale factor needed.
    """
    # Calculate total compute duration (excluding first layer which doesn't transfer)
    total_duration = sum(layer.duration for layer in layer_stats[:-1]) if len(layer_stats) > 1 else 1.0

    if total_duration <= 0:
        return 100.0  # Fallback to safe default

    # Estimate transfer capacity at scale=1.0
    capacity_at_scale_1 = memory_transfer_interpolator.duration_to_bytes(total_duration)

    if capacity_at_scale_1 <= 0:
        return 100.0  # Fallback to safe default

    # Estimate scale needed: scale = needed_bytes / capacity_at_scale_1
    # Add 50% safety margin
    estimated_scale = (needed_offload_bytes / capacity_at_scale_1) * 1.5

    # Ensure minimum of 10.0 to handle edge cases
    return max(estimated_scale, 10.0)


def _validate_layer_stats_for_pipelining(layer_stats: list[LayerStatistics]) -> None:
    """Validate that layer_stats has at least 2 layers for pipelined offload."""
    if not layer_stats:
        raise StrategyComputeError("layer_stats must contain at least one layer")
    if len(layer_stats) < 2:
        raise StrategyComputeError(
            "layer_stats must contain at least 2 layers for pipelined offload computation. "
            "With a single layer, there is no previous layer to pipeline transfers against."
        )


def _compute_offload_for_scale(
    layer_stats: list[LayerStatistics],
    memory_transfer_interpolator: MemoryTransferInterpolator,
    test_scale: float,
) -> tuple[float, list[float]]:
    """Compute offloadable memory for a given scale.

    Returns:
        Tuple of (total_offload_bytes, per_layer_offload_list).
    """
    offload_per_layer: list[float] = []
    prev = layer_stats[0]
    for layer in layer_stats[1:]:
        layer_size = sum(t.size_bytes for t in layer.tensors)
        max_offload = memory_transfer_interpolator.duration_to_bytes(prev.duration * test_scale)
        actual_offload = float(min(max_offload, layer_size))
        offload_per_layer.append(actual_offload)
        prev = layer
    return float(sum(offload_per_layer)), offload_per_layer


def _binary_search_optimal_scale(
    layer_stats: list[LayerStatistics],
    memory_transfer_interpolator: MemoryTransferInterpolator,
    initial_scale: float,
    estimated_upper_bound: float,
    needed_total_offload: int | float,
    tolerance_bytes: int,
    max_iterations: int,
) -> tuple[float, float, list[float]]:
    """Binary search to find optimal scale for target offload."""
    low_scale = initial_scale
    high_scale = max(initial_scale * 100, estimated_upper_bound)
    optimal_scale = high_scale

    for _ in range(max_iterations):
        mid_scale = (low_scale + high_scale) / 2
        offload_at_mid, per_layer_at_mid = _compute_offload_for_scale(
            layer_stats, memory_transfer_interpolator, mid_scale
        )

        if abs(offload_at_mid - needed_total_offload) < tolerance_bytes:
            return mid_scale, offload_at_mid, per_layer_at_mid

        if offload_at_mid < needed_total_offload:
            low_scale = mid_scale
        else:
            high_scale = mid_scale
            optimal_scale = mid_scale

    final_offload, per_layer = _compute_offload_for_scale(layer_stats, memory_transfer_interpolator, optimal_scale)
    return optimal_scale, final_offload, per_layer


def _compute_optimal_scale(
    layer_stats: list[LayerStatistics],
    memory_transfer_interpolator: MemoryTransferInterpolator,
    max_gpu_mem_bytes: int,
    initial_scale: float = 1.0,
    tolerance_bytes: int = 1024 * 1024,  # 1MB default tolerance
    max_iterations: int = 50,
) -> tuple[float, float, list[float]]:
    """
    Compute optimal scale to meet GPU memory target using binary search.

    The scale factor controls how aggressively we offload tensors. A higher scale
    allows more memory to be transferred during each layer's compute time.

    Args:
        layer_stats: List of layer statistics containing tensor info and durations.
            Must contain at least 2 layers for pipelined offload computation.
        memory_transfer_interpolator: Interpolator to convert duration to bytes.
        max_gpu_mem_bytes: Target maximum GPU memory usage in bytes.
        initial_scale: Starting scale value (default 1.0).
        tolerance_bytes: Convergence tolerance in bytes (default 1MB).
        max_iterations: Maximum binary search iterations (default 50).

    Returns:
        Tuple of (optimal_scale, total_offload_bytes, per_layer_offload_list).
        - optimal_scale: The scale factor that achieves the target GPU memory.
        - total_offload_bytes: Total bytes that will be offloaded with this scale.
        - per_layer_offload_list: List of offload amounts per layer.

    Raises:
        ValueError: If layer_stats is empty or contains only one layer.

    Examples:
        >>> interpolator = MemoryTransferInterpolator(memory_transfers)
        >>> scale, total, per_layer = _compute_optimal_scale(
        ...     layer_stats, interpolator, max_gpu_mem_bytes=20 * 1024**3
        ... )
        >>> print(f"Optimal scale: {scale}, will offload {total / 1024**3:.2f} GB")
    """
    _validate_layer_stats_for_pipelining(layer_stats)

    all_layers_size = sum(sum(t.size_bytes for t in layer.tensors) for layer in layer_stats)
    max_possible_offload = sum(sum(t.size_bytes for t in layer.tensors) for layer in layer_stats[1:])
    needed_total_offload = all_layers_size - max_gpu_mem_bytes

    # Edge case: already under limit
    if needed_total_offload <= 0:
        total_offload, per_layer = _compute_offload_for_scale(layer_stats, memory_transfer_interpolator, initial_scale)
        return initial_scale, total_offload, per_layer

    estimated_upper_bound = _estimate_required_scale(
        layer_stats, memory_transfer_interpolator, float(needed_total_offload)
    )

    # Edge case: impossible to meet target even with max offloading
    if needed_total_offload > max_possible_offload:
        high_scale = max(initial_scale * 100, estimated_upper_bound)
        total_offload, per_layer = _compute_offload_for_scale(layer_stats, memory_transfer_interpolator, high_scale)
        return high_scale, total_offload, per_layer

    # Check if initial scale already meets the target
    current_offload, current_per_layer = _compute_offload_for_scale(
        layer_stats, memory_transfer_interpolator, initial_scale
    )
    if current_offload >= needed_total_offload:
        return initial_scale, current_offload, current_per_layer

    # Binary search to find optimal scale
    return _binary_search_optimal_scale(
        layer_stats,
        memory_transfer_interpolator,
        initial_scale,
        estimated_upper_bound,
        needed_total_offload,
        tolerance_bytes,
        max_iterations,
    )


# =============================================================================
# Knapsack Solvers
# =============================================================================


def _greedy_pack(sizes: numpy.ndarray, capacity: float, order_indices: list[int]) -> numpy.ndarray:
    """Build a binary selection vector by greedily packing items in the given order.

    Args:
        sizes: Array of item sizes.
        capacity: Maximum total size.
        order_indices: Order in which to consider items.

    Returns:
        Binary numpy array (1 = selected, 0 = not selected).
    """
    import numpy as np

    x = np.zeros(len(sizes))
    remaining = capacity
    for i in order_indices:
        if sizes[i] <= remaining:
            x[i] = 1.0
            remaining -= sizes[i]
    return x


def _compute_solution_block(  # noqa: C901
    duration: float,
    layer: LayerStatistics,
    scale: float,
    threshold_mb: float,
    memory_transfers: dict[int, float],
) -> list[TensorStatistics]:
    """Compute knapsack solution using block-based memory transfer model.

    Since both profit and weight are ``size_bytes`` (maximize total bytes offloaded
    within a transfer capacity), this is a subset-sum problem.  A greedy heuristic
    generates strong initial solutions, then ``differential_evolution`` refines them
    with early stopping so it bails out quickly when no improvement is found.

    For small problems (n <= 5), only the greedy heuristic is used.
    """
    import numpy as np
    from scipy.optimize import differential_evolution

    if not memory_transfers:
        # No transfer data available → cannot offload based on block transfers.
        return []

    memory_transfer_interpolator = MemoryTransferInterpolator(memory_transfers)

    max_capacity_size = memory_transfer_interpolator.duration_to_bytes(duration * scale)

    layer_tensors = [x for x in layer.tensors if (x.size_bytes / 1024 / 1024) > threshold_mb]

    # Check whether we can offload all tensors
    total_size = sum(t.size_bytes for t in layer_tensors)
    if total_size <= max_capacity_size:
        return list(layer_tensors)

    # Check if optimization is possible (at least one tensor fits)
    if not any(t.size_bytes <= max_capacity_size for t in layer_tensors):
        return []

    n = len(layer_tensors)
    if n == 0:
        return []

    sizes = np.array([t.size_bytes for t in layer_tensors], dtype=np.float64)
    cap = float(max_capacity_size)

    # --- Greedy seed solutions (descending and ascending) ---
    desc_order = sorted(range(n), key=lambda i: sizes[i], reverse=True)
    asc_order = sorted(range(n), key=lambda i: sizes[i])

    greedy_desc = _greedy_pack(sizes, cap, desc_order)
    greedy_asc = _greedy_pack(sizes, cap, asc_order)

    greedy_best = greedy_desc if np.dot(greedy_desc, sizes) >= np.dot(greedy_asc, sizes) else greedy_asc
    greedy_value = float(np.dot(greedy_best, sizes))

    # For tiny problems, greedy is likely optimal — skip optimizer
    if n <= 5:
        return [layer_tensors[i] for i in range(n) if greedy_best[i] == 1.0]

    # --- Objective: minimize negative packed size, penalty for over-capacity ---
    def objective(x: np.ndarray) -> float:
        weight = cast("float", np.dot(x, sizes))
        if weight > cap:
            return (weight - cap) / max(cap, 1.0)
        return -weight / max(cap, 1.0)

    # --- Build initial population seeded with greedy solutions ---
    pop_target = max(6, min(20, 250 // n))
    rng = np.random.default_rng(42)

    init_pop = [greedy_desc, greedy_asc]
    for _ in range(pop_target - 2):
        init_pop.append(rng.integers(0, 2, size=n).astype(float))
    init_pop_arr = np.array(init_pop)
    # --- Run optimizer with early stopping ---
    result = differential_evolution(
        objective,
        bounds=cast("Any", [(0, 1)] * n),
        integrality=[True] * n,
        init=init_pop_arr,
        maxiter=200,
        tol=0.01,
        seed=42,
        polish=False,
        callback=cast("Any", EarlyStopCallback(max_stall=25, objective_func=objective)),
    )

    if np.any(np.isnan(result.x)):
        return [layer_tensors[i] for i in range(n) if greedy_best[i] == 1.0]

    solution = np.round(result.x).astype(int)
    opt_value = float(np.dot(solution, sizes))

    # Use optimizer result only if it is feasible and at least as good as greedy
    if opt_value > cap or opt_value < greedy_value:
        return [layer_tensors[i] for i in range(n) if greedy_best[i] == 1.0]

    return [layer_tensors[i] for i in range(n) if solution[i] == 1]


def _greedy_knapsack(
    weights: numpy.ndarray, profits: numpy.ndarray, capacity: float, order_indices: list[int]
) -> numpy.ndarray:
    """Build a binary selection vector by greedily packing items in the given order.

    Args:
        weights: Array of item weights.
        profits: Array of item profits.
        capacity: Maximum total weight.
        order_indices: Order in which to consider items.

    Returns:
        Binary numpy array (1 = selected, 0 = not selected).
    """
    import numpy as np

    x = np.zeros(len(weights))
    remaining = capacity
    for i in order_indices:
        if weights[i] <= remaining:
            x[i] = 1.0
            remaining -= weights[i]
    return x


def _compute_solution(
    duration: float,
    layer: LayerStatistics,
    scale: float,
    threshold_mb: float,
) -> list[TensorStatistics]:
    """Compute knapsack solution using duration-based model.

    A greedy heuristic generates strong initial solutions (by value-density and
    by weight), then ``differential_evolution`` refines them with early stopping.
    For small problems (n <= 5), only the greedy heuristic is used.
    """
    import numpy as np
    from scipy.optimize import differential_evolution

    max_capacity = int(duration * 1e6)
    max_capacity = int(max_capacity * scale)
    layer_tensors = [x for x in layer.tensors if (x.size_bytes / 1024 / 1024) > threshold_mb]
    weights_arr = np.array([int(x.load_time_ms * 1e6) for x in layer_tensors], dtype=np.float64)
    profits_arr = np.array([int(x.size_bytes) for x in layer_tensors], dtype=np.float64)

    # Check whether we can offload all tensors
    if weights_arr.sum() <= max_capacity:
        return list(layer_tensors)

    # Check if optimization is possible (at least one tensor fits)
    if not np.any(weights_arr <= max_capacity):
        return []

    n = len(weights_arr)
    if n == 0:
        return []

    cap = float(max_capacity)

    # --- Greedy seed solutions ---
    # 1. By value/weight ratio descending (classic greedy knapsack heuristic)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(weights_arr > 0, profits_arr / weights_arr, 0.0)
    ratio_order = sorted(range(n), key=lambda i: ratios[i], reverse=True)
    greedy_ratio = _greedy_knapsack(weights_arr, profits_arr, cap, ratio_order)

    # 2. By weight ascending (fit as many small items as possible)
    weight_order = sorted(range(n), key=lambda i: weights_arr[i])
    greedy_light = _greedy_knapsack(weights_arr, profits_arr, cap, weight_order)

    # 3. By profit descending (grab highest-value items first)
    profit_order = sorted(range(n), key=lambda i: profits_arr[i], reverse=True)
    greedy_heavy = _greedy_knapsack(weights_arr, profits_arr, cap, profit_order)

    # Pick the best greedy solution
    greedy_values = [
        (np.dot(greedy_ratio, profits_arr), greedy_ratio),
        (np.dot(greedy_light, profits_arr), greedy_light),
        (np.dot(greedy_heavy, profits_arr), greedy_heavy),
    ]
    greedy_best_value, greedy_best = max(greedy_values, key=lambda t: t[0])

    # For tiny problems, greedy is likely optimal — skip optimizer
    if n <= 5:
        return [layer_tensors[i] for i in range(n) if greedy_best[i] == 1.0]

    # --- Objective ---
    def objective(x: np.ndarray) -> float:
        current_weight = cast("float", np.dot(x, weights_arr))
        if current_weight > cap:
            return (current_weight - cap) / max(cap, 1.0)
        return -cast("float", np.dot(x, profits_arr)) / max(float(np.sum(profits_arr)), 1.0)

    # --- Build initial population seeded with greedy solutions ---
    pop_target = max(6, min(20, 250 // n))
    rng = np.random.default_rng(42)

    init_pop = [greedy_ratio, greedy_light, greedy_heavy]
    for _ in range(pop_target - 3):
        init_pop.append(rng.integers(0, 2, size=n).astype(float))
    init_pop_arr = np.array(init_pop)
    # --- Run optimizer with early stopping ---
    result = differential_evolution(
        objective,
        bounds=cast("Any", [(0, 1)] * n),
        integrality=[True] * n,
        init=init_pop_arr,
        maxiter=200,
        tol=0.01,
        seed=42,
        polish=False,
        callback=cast("Any", EarlyStopCallback(max_stall=25, objective_func=objective)),
    )

    if np.any(np.isnan(result.x)):
        return [layer_tensors[i] for i in range(n) if greedy_best[i] == 1.0]

    solution = np.round(result.x).astype(int)
    opt_weight = float(np.dot(solution, weights_arr))
    opt_value = float(np.dot(solution, profits_arr))

    # Use optimizer result only if it is feasible and at least as good as greedy
    if opt_weight > cap or opt_value < greedy_best_value:
        return [layer_tensors[i] for i in range(n) if greedy_best[i] == 1.0]

    return [layer_tensors[i] for i in range(n) if solution[i] == 1]


# =============================================================================
# Offload Tensor Computation
# =============================================================================


def _compute_offload_tensors_block(
    layer_stats: list[LayerStatistics],
    memory_transfers: dict[int, float],
    scale: float = 1.0,
    threshold_mb: float = 0.1,
) -> dict[str, list[TensorStatistics]]:
    """
    Compute offload strategy using block-based knapsack optimization.

    Uses pre-computed memory transfer benchmarks to determine which tensors
    can be offloaded within the compute time of previous layers.

    Args:
        layer_stats: List of layer statistics containing tensor info and durations.
        memory_transfers: Dict mapping tensor size (bytes) to transfer time (ms).
        scale: Scaling factor for duration calculations.
        threshold_mb: Minimum tensor size threshold in MB.

    Returns:
        Strategy dict mapping layer labels to lists of tensors to offload.
    """
    strategy: dict[str, list[TensorStatistics]] = {}

    if len(layer_stats) < 2:
        return strategy

    prev_layer = layer_stats[0]
    for layer in layer_stats[1:]:
        res = _compute_solution_block(prev_layer.duration, layer, scale, threshold_mb, memory_transfers)
        strategy[prev_layer.label] = res
        prev_layer = layer

    return strategy


def _compute_offload_tensors(
    layer_stats: list[LayerStatistics],
    scale: float = 1.0,
    threshold_mb: float = 0.1,
) -> dict[str, list[TensorStatistics]]:
    """Compute offload strategy for consecutive layers using duration-based model."""
    strategy: dict[str, list[TensorStatistics]] = {}

    # Handle empty layer_stats (can happen with simple models or no offloadable layers)
    if len(layer_stats) == 0:
        return strategy

    prev_layer = layer_stats[0]
    for layer in layer_stats[1:]:
        res = _compute_solution(prev_layer.duration, layer, scale, threshold_mb)
        strategy[prev_layer.label] = res
        prev_layer = layer
    return strategy


# =============================================================================
# Cyclic Strategy Functions
# =============================================================================


def _collect_existing_tensor_ids(strategy: dict[str, list[TensorStatistics]]) -> set[int]:
    """Extract all tensor IDs that are already part of the current strategy."""
    tensor_ids = set()
    for tensors in strategy.values():
        for tensor_info in tensors:
            tensor_ids.add(tensor_info.tensor_id)
    return tensor_ids


def _find_accumulated_duration_from_end(
    layer_stats: list[LayerStatistics],
    strategy: dict[str, list],
) -> tuple[LayerStatistics | None, float]:
    """
    Find the last layer without an active offload strategy and accumulate duration
    from layers without strategies, working backwards from the end.

    Returns:
        Tuple of (last_layer_without_strategy, accumulated_duration_ms)
    """
    last_layer_without_strategy = None
    accumulated_duration = 0.0

    for layer in reversed(layer_stats):
        layer_label = layer.label

        # If layer not in strategy, accumulate its duration
        if layer_label not in strategy:
            last_layer_without_strategy = layer
            accumulated_duration += layer.duration
            continue

        # If layer has active strategy (non-empty), stop accumulating
        if len(strategy[layer_label]) > 0:
            break

        # Layer is in strategy but has empty offload list
        last_layer_without_strategy = layer
        accumulated_duration += layer.duration

    return last_layer_without_strategy, accumulated_duration


def _has_tensor_conflicts(layer: LayerStatistics, existing_tensor_ids: set[int]) -> bool:
    """Check if any tensors in the layer conflict with existing strategy tensors."""
    return any(tensor.tensor_id in existing_tensor_ids for tensor in layer.tensors)


def _calculate_total_offload_memory(tensors: list) -> float:
    """Calculate total memory footprint of weight list in MB."""
    return float(sum(tensor.size_bytes for tensor in tensors) / 1024 / 1024)


def _find_best_offload_opportunity(
    layer_stats: list[LayerStatistics],
    existing_tensor_ids: set[int],
    available_duration: float,
    scale: float,
    threshold_mb: float,
) -> tuple[LayerStatistics | None, list[TensorStatistics], float]:
    """
    Find the layer that offers the best offload opportunity given available duration.

    Returns:
        Tuple of (best_layer, best_tensors_to_offload, best_memory_amount)
    """
    best_layer = None
    best_tensors: list[TensorStatistics] = []
    best_memory_offload = 0.0

    for layer in layer_stats:
        # Stop iteration when we find a layer with offload strategy
        if _has_tensor_conflicts(layer, existing_tensor_ids):
            break

        # Compute optimal offload solution for this layer given available duration
        candidate_tensors = _compute_solution(available_duration, layer, scale, threshold_mb)
        candidate_memory = _calculate_total_offload_memory(candidate_tensors)

        # Keep track of the best option
        if candidate_memory > best_memory_offload:
            best_memory_offload = candidate_memory
            best_tensors = candidate_tensors
            best_layer = layer
        available_duration += layer.duration

    return best_layer, best_tensors, best_memory_offload


def _generate_cyclic_strategy(
    layer_stats: list[LayerStatistics],
    strategy_: dict[str, list[TensorStatistics]],
    scale: float = 1.0,
    threshold_mb: float = 0.1,
) -> dict[str, list[TensorStatistics]]:
    """
    Extend an existing offload strategy by finding additional offload opportunities
    in a cyclic manner. This uses accumulated duration from layers without active
    strategies to create new offload opportunities.

    Args:
        layer_stats: List of layer statistics containing tensor information
        strategy_: Existing offload strategy mapping layer labels to tensors to offload
        scale: Scaling factor for duration calculations
        threshold_mb: Minimum tensor size threshold in MB

    Returns:
        updated_strategy
    """
    # Create working copies to avoid modifying originals
    strategy = strategy_.copy()

    # Step 1: Collect tensor IDs already in use by current strategy
    existing_tensor_ids = _collect_existing_tensor_ids(strategy)

    # Step 2: Find accumulated duration from layers without active strategies
    upload_layer, available_duration = _find_accumulated_duration_from_end(layer_stats, strategy)

    if upload_layer is None or available_duration <= 0:
        # No opportunity for cyclic strategy
        return strategy

    # Step 3: Find the best offload opportunity given the available duration
    best_layer, best_tensors, best_memory = _find_best_offload_opportunity(
        layer_stats,
        existing_tensor_ids,
        available_duration,
        scale,
        threshold_mb,
    )

    # Step 4: Update strategy if we found a beneficial offload opportunity
    if best_layer is not None and best_memory > 0:
        strategy[upload_layer.label] = best_tensors

    return strategy


# =============================================================================
# Layer Merging
# =============================================================================


def _prepare_merged_layers(
    org_layer_stats: list[LayerStatistics],
    group_size: int = 2,
) -> list[LayerStatistics]:
    """
    Merges consecutive layers into groups of specified size.

    Args:
        org_layer_stats: List of LayerStatistics objects to merge
        group_size: Number of layers to merge into each group (default: 2)

    Returns:
        List of merged LayerStatistics objects
    """

    if not org_layer_stats:
        return []

    merged_layers = []

    for i in range(0, len(org_layer_stats), group_size):
        # Get the slice of layers for this group
        group_layers = org_layer_stats[i : i + group_size]

        # Create merged layer with the label from the first layer in the group
        label = group_layers[0].label
        tensors = []
        duration = 0.0

        # Accumulate duration and tensors from all layers in the group
        for layer in group_layers:
            duration += layer.duration
            tensors.extend(layer.tensors)

        merged_layer = LayerStatistics(label=label, tensors=tensors, duration=duration)
        merged_layers.append(merged_layer)

    return merged_layers


# =============================================================================
# Peak GPU Estimation for Block Strategies
# =============================================================================


def _estimate_peak_gpu_block(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    block_data: BlockStrategyData,
) -> int:
    """Estimate peak GPU memory for a block-based strategy.

    For block strategies, peak GPU memory includes both the pre-allocated blocks
    (sized to the maximum offload per layer in each block) and all non-offloaded
    tensors that remain resident on GPU.

    Peak GPU = sum(block_sizes) + (total_model_size - total_offloaded_size)

    Args:
        strategy_map: Layer label to offloaded tensors mapping.
        layer_stats: Layer statistics with tensor information.
        block_data: Block strategy data containing block sizes.

    Returns:
        Estimated peak GPU memory in bytes.
    """
    block_sizes = block_data.block_sizes
    block_memory = sum(block_sizes.values()) if isinstance(block_sizes, dict) else sum(block_sizes)

    total_model = sum(sum(t.size_bytes for t in layer.tensors) for layer in layer_stats)
    total_offloaded = sum(sum(t.size_bytes for t in tensors) for tensors in strategy_map.values())

    return block_memory + total_model - total_offloaded


# =============================================================================
# Strategy Classes
# =============================================================================


class KnapsackStrategy:
    """Knapsack-based offloading strategy using duration-based optimization."""

    def __init__(
        self,
        scale: float = 1.0,
        cyclic: bool = True,
        group_size: int = 1,
        threshold_mb: float = 0.1,
        n_blocks: int = 4,
    ):
        """
        Initialize KnapsackStrategy for weight offloading.

        Args:
            scale: Duration multiplier (baseline). ``0.9`` = 10% safety margin,
                ``1.0`` = exact fit, ``2.0`` = allow 100% overhead. When
                ``max_gpu_mem_bytes`` is set, scale may be increased automatically
                to meet the memory constraint.
            cyclic: Whether to apply cyclic strategy extension.
            group_size: Number of layers to merge into groups.
            threshold_mb: Minimum tensor size threshold in MB.
            n_blocks: Number of blocks for block-based loaders. Set to 0 to skip
                block data computation.
        """
        validate_memory_params(scale)
        self.threshold_mb = threshold_mb
        self.scale = scale
        self.cyclic = cyclic
        self.group_size = group_size
        self.n_blocks = n_blocks

    def compute(  # noqa: C901
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """
        Compute offload strategy using knapsack optimization.

        Args:
            layer_stats: List of layer statistics containing tensor info and durations.
            memory_stats: Optional dict mapping tensor size (bytes) to transfer time (ms).
                If not provided, extracts from layer stats.
            max_gpu_mem_bytes: Hard GPU memory limit in bytes (never exceed).
                When ``None``, latency mode — respect ``scale`` strictly.

        Returns:
            StrategyResult containing strategy_map and optional block_data.
        """
        new_layer_stats = layer_stats
        if self.group_size > 1:
            new_layer_stats = _prepare_merged_layers(layer_stats, self.group_size)

        # Use provided memory_stats or extract from layer stats
        memory_transfers = memory_stats
        if memory_transfers is None:
            memory_transfers = extract_memory_transfers_from_layer_stats(new_layer_stats)

        # Determine effective scale - either use provided scale or compute optimal
        effective_scale = self.scale
        if max_gpu_mem_bytes is not None and memory_transfers:
            memory_transfer_interpolator = MemoryTransferInterpolator(memory_transfers)
            effective_scale, _, _ = _compute_optimal_scale(
                layer_stats=new_layer_stats,
                memory_transfer_interpolator=memory_transfer_interpolator,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
                initial_scale=self.scale,
            )
            if effective_scale > self.scale:
                warnings.warn(
                    f"Original scale ({self.scale}) is insufficient to meet GPU memory target "
                    f"({max_gpu_mem_bytes / 1024**3:.2f} GB). "
                    f"Adjusted scale to {effective_scale:.4f}.",
                    stacklevel=2,
                )

        strategy_map = _compute_offload_tensors(new_layer_stats, effective_scale, self.threshold_mb)
        if self.cyclic:
            strategy_map = _generate_cyclic_strategy(new_layer_stats, strategy_map, effective_scale, self.threshold_mb)

        block_data = _build_block_data(strategy_map, new_layer_stats, self.n_blocks)

        # Post-validation: block pre-allocation overhead may push peak above
        # the hard limit. Adjust scale to meet max_gpu_mem_bytes (hard limit).
        # The soft target is already handled by _compute_optimal_scale above.
        if max_gpu_mem_bytes is not None and block_data is not None:
            limit = max_gpu_mem_bytes
            peak_gpu = _estimate_peak_gpu_block(strategy_map, new_layer_stats, block_data)
            if peak_gpu > limit:
                best_peak = peak_gpu
                best_strategy = strategy_map
                best_block_data: BlockStrategyData | None = block_data
                current_scale = effective_scale
                for _ in range(10):
                    overshoot = peak_gpu / limit
                    current_scale *= max(overshoot, 1.05)
                    test_strategy = _compute_offload_tensors(new_layer_stats, current_scale, self.threshold_mb)
                    if self.cyclic:
                        test_strategy = _generate_cyclic_strategy(
                            new_layer_stats, test_strategy, current_scale, self.threshold_mb
                        )
                    test_block = _build_block_data(test_strategy, new_layer_stats, self.n_blocks)
                    if test_block is None:
                        break
                    peak_gpu = _estimate_peak_gpu_block(test_strategy, new_layer_stats, test_block)
                    if peak_gpu < best_peak:
                        best_peak = peak_gpu
                        best_strategy = test_strategy
                        best_block_data = test_block
                    if peak_gpu <= limit:
                        break
                if best_peak > limit:
                    warnings.warn(
                        f"Knapsack strategy could not meet GPU memory limit "
                        f"({limit / 1024**3:.2f} GB). "
                        f"Estimated peak: {best_peak / 1024**3:.2f} GB after scale adjustment.",
                        stacklevel=2,
                    )
                strategy_map = best_strategy
                block_data = best_block_data

        return StrategyResult(strategy_map=strategy_map, block_data=block_data)


class KnapsackBlockStrategy:
    """
    Block-based knapsack strategy that uses externally provided memory transfer statistics.

    This strategy expects memory_stats to be provided by TensorManager, which handles
    the live GPU benchmarking. This separates concerns: TensorManager manages profiling,
    strategies focus on optimization logic.
    """

    def __init__(
        self,
        scale: float = 1.0,
        group_size: int = 1,
        threshold_mb: float = 0.1,
        n_blocks: int = 4,
    ):
        """
        Initialize KnapsackBlockStrategy for weight offloading.

        Args:
            scale: Duration multiplier (baseline). ``0.9`` = 10% safety margin,
                ``1.0`` = exact fit, ``2.0`` = allow 100% overhead. When
                ``max_gpu_mem_bytes`` is set, scale may be increased automatically
                to meet the memory constraint.
            group_size: Number of layers to merge into groups.
            threshold_mb: Minimum tensor size threshold in MB.
            n_blocks: Number of blocks for block-based loaders.
        """
        validate_memory_params(scale)
        self.threshold_mb = threshold_mb
        self.scale = scale
        self.group_size = group_size
        self.n_blocks = n_blocks

    def compute(
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """
        Compute offload strategy using block-based knapsack optimization.

        Args:
            layer_stats: List of layer statistics containing tensor info and durations.
            memory_stats: Dict mapping tensor size (bytes) to transfer time (ms).
                Should be provided by TensorManager via live GPU benchmarking.
                If not provided, falls back to extracting from layer stats.
            max_gpu_mem_bytes: Hard GPU memory limit in bytes (never exceed).
                When ``None``, latency mode — respect ``scale`` strictly.

        Returns:
            StrategyResult containing strategy_map and block_data.
        """
        new_layer_stats = layer_stats
        if self.group_size > 1:
            new_layer_stats = _prepare_merged_layers(layer_stats, self.group_size)

        # Use provided memory_stats or fall back to extraction from layer stats
        memory_transfers = memory_stats
        if memory_transfers is None:
            memory_transfers = extract_memory_transfers_from_layer_stats(new_layer_stats)

        # Determine effective scale - either use provided scale or compute optimal
        effective_scale = self.scale
        if max_gpu_mem_bytes is not None and memory_transfers:
            memory_transfer_interpolator = MemoryTransferInterpolator(memory_transfers)
            effective_scale, _, _ = _compute_optimal_scale(
                layer_stats=new_layer_stats,
                memory_transfer_interpolator=memory_transfer_interpolator,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
                initial_scale=self.scale,
            )
            if effective_scale > self.scale:
                warnings.warn(
                    f"Original scale ({self.scale}) is insufficient to meet GPU memory target "
                    f"({max_gpu_mem_bytes / 1024**3:.2f} GB). "
                    f"Adjusted scale to {effective_scale:.4f}.",
                    stacklevel=2,
                )

        strategy_map = _compute_offload_tensors_block(
            new_layer_stats, memory_transfers, effective_scale, self.threshold_mb
        )

        block_data = _build_block_data(strategy_map, new_layer_stats, self.n_blocks)

        # Post-validation: _compute_optimal_scale doesn't account for block
        # pre-allocation overhead. Check actual peak GPU and increase scale to
        # offload more aggressively if the hard limit is violated.
        if max_gpu_mem_bytes is not None and block_data is not None:
            strategy_map, block_data = self._adjust_scale_for_block_overhead(
                strategy_map,
                block_data,
                new_layer_stats,
                memory_transfers,
                effective_scale,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
            )

        return StrategyResult(strategy_map=strategy_map, block_data=block_data)

    def _adjust_scale_for_block_overhead(
        self,
        strategy_map: dict[str, list[TensorStatistics]],
        block_data: BlockStrategyData,
        layer_stats: list[LayerStatistics],
        memory_transfers: dict[int, float],
        initial_scale: float,
        max_attempts: int = 10,
        max_gpu_mem_bytes: int | None = None,
    ) -> tuple[dict[str, list[TensorStatistics]], BlockStrategyData | None]:
        """Iteratively increase scale until actual peak GPU fits within memory limit.

        The initial scale from ``_compute_optimal_scale`` doesn't account for block
        pre-allocation overhead (blocks are sized to the maximum offload across layers
        in each block). This method checks the actual peak GPU and increases the scale
        to offload more aggressively, reducing non-offloaded resident memory faster
        than block memory grows.

        Args:
            strategy_map: Current layer-to-tensors offload mapping.
            block_data: Current block strategy data.
            layer_stats: Layer statistics.
            memory_transfers: Memory transfer benchmarks.
            initial_scale: Starting scale value.
            max_attempts: Maximum correction iterations.
            max_gpu_mem_bytes: Hard GPU memory limit in bytes (never exceed).

        Returns:
            Tuple of (adjusted_strategy_map, adjusted_block_data).
        """
        if max_gpu_mem_bytes is None:
            raise RuntimeError("_adjust_scale_for_block_overhead requires max_gpu_mem_bytes to be set")
        limit = max_gpu_mem_bytes

        peak_gpu = _estimate_peak_gpu_block(strategy_map, layer_stats, block_data)
        if peak_gpu <= limit:
            return strategy_map, block_data

        best_peak = peak_gpu
        best_strategy = strategy_map
        best_block_data: BlockStrategyData | None = block_data
        current_scale = initial_scale

        for _ in range(max_attempts):
            overshoot = peak_gpu / limit
            current_scale *= max(overshoot, 1.05)

            test_strategy = _compute_offload_tensors_block(
                layer_stats, memory_transfers, current_scale, self.threshold_mb
            )
            test_block_data = _build_block_data(test_strategy, layer_stats, self.n_blocks)
            if test_block_data is None:
                break

            peak_gpu = _estimate_peak_gpu_block(test_strategy, layer_stats, test_block_data)
            if peak_gpu < best_peak:
                best_peak = peak_gpu
                best_strategy = test_strategy
                best_block_data = test_block_data

            if peak_gpu <= limit:
                break

        if best_peak > limit:
            warnings.warn(
                f"KnapsackBlock strategy could not meet GPU memory limit "
                f"({limit / 1024**3:.2f} GB). "
                f"Estimated peak: {best_peak / 1024**3:.2f} GB after scale adjustment.",
                stacklevel=3,
            )

        return best_strategy, best_block_data


class AdaptiveKnapsackStrategy:
    """
    Adaptive strategy that selects between KnapsackStrategy and KnapsackBlockStrategy
    based on the loader type configuration.

    This provides a unified interface that automatically selects the appropriate
    strategy implementation based on how tensors will be loaded during inference.
    """

    # Loader types that require block-based strategy
    BLOCK_LOADER_TYPES: ClassVar[set[str]] = {"allocation_block_transfer", "raw_block_transfer"}

    def __init__(
        self,
        scale: float = 1.0,
        loader_type: str = "strategy",
        cyclic: bool = True,
        group_size: int = 1,
        threshold_mb: float = 0.1,
    ):
        """
        Initialize AdaptiveKnapsackStrategy.

        Args:
            scale: Duration multiplier (baseline). See ``KnapsackStrategy`` for details.
            loader_type: Type of tensor loader to use. Options:
                - "strategy": Uses KnapsackStrategy (profiled tensor data)
                - "raw_block_transfer": Uses KnapsackBlockStrategy
                - "allocation_block_transfer": Uses KnapsackBlockStrategy
            cyclic: Whether to apply cyclic strategy extension.
            group_size: Number of layers to merge into groups.
            threshold_mb: Minimum tensor size threshold in MB.
        """
        validate_memory_params(scale)
        self.scale = scale
        self.loader_type = loader_type
        self.cyclic = cyclic
        self.group_size = group_size
        self.threshold_mb = threshold_mb

    def _create_strategy(self) -> KnapsackStrategy | KnapsackBlockStrategy:
        """Create the appropriate strategy instance based on loader_type."""
        if self.loader_type in self.BLOCK_LOADER_TYPES:
            return KnapsackBlockStrategy(
                scale=self.scale,
                group_size=self.group_size,
                threshold_mb=self.threshold_mb,
            )
        else:
            return KnapsackStrategy(
                scale=self.scale,
                cyclic=self.cyclic,
                group_size=self.group_size,
                threshold_mb=self.threshold_mb,
            )

    def compute(
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """
        Compute offload strategy using the appropriate strategy implementation.

        Args:
            layer_stats: List of layer statistics containing tensor info and durations.
            memory_stats: Optional dict mapping tensor size (bytes) to transfer time (ms).
            max_gpu_mem_bytes: Hard GPU memory limit in bytes (never exceed).
                When ``None``, latency mode — respect ``scale`` strictly.

        Returns:
            StrategyResult containing strategy_map and block_data.
        """
        strategy_impl = self._create_strategy()
        return strategy_impl.compute(layer_stats, memory_stats, max_gpu_mem_bytes)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Analysis script for comparing offload strategies on DeepSeek R1 model data.

This script loads profiled model data and runs various offload strategies,
providing detailed per-layer and summary statistics for comparison.

Usage:
    python analyze_offload_strategy.py [--strategy STRATEGY] [--n-blocks N] [--gpu-mem GB] [--transfer-ratio R ...]

Strategies:
    - global: GlobalOffloadStrategy with OptimizedRoundRobinAssignment (default)
    - global-strict: GlobalOffloadStrategy with StrictRoundRobinAssignment
    - knapsack: KnapsackStrategy (original per-layer strategy)
    - knapsack-block: KnapsackBlockStrategy (live GPU benchmarking, requires CUDA)
    - budget-fill: BudgetFillStrategy facade (greedy + optional layer DE)
    - budget-fill-greedy: BudgetFillGreedyStrategy alone
    - budget-fill-tensor-de: BudgetFillTensorDEStrategy (per-tensor DE; slow)
    - budget-fill-layer-de: BudgetFillLayerDEStrategy (per-layer DE)
    - budget-fill-compare: facade vs greedy vs layer DE vs tensor DE
    - tensor-select: GlobalTensorSelectionStrategy with OptimizedRoundRobinAssignment (default)
    - tensor-select-strict: GlobalTensorSelectionStrategy with StrictRoundRobinAssignment
    - adaptive: AdaptiveStrategy (fast candidates only)
    - adaptive-extra: AdaptiveStrategy with extra_optimization (includes slower TensorSelection)
    - recommended: Run strategies with OptimizedRoundRobinAssignment (no cyclic violations)
    - all: Run all strategies and compare
"""

from __future__ import annotations

import warnings

from beartype.roar import BeartypeClawDecorWarning

# Fail on beartype decorator warnings - these indicate type hint issues
# This mirrors the pytest filterwarnings configuration in pyproject.toml
# IMPORTANT: This MUST be set before importing flextensor modules, as
# beartype warnings are emitted during import/decoration time.
warnings.filterwarnings("error", category=BeartypeClawDecorWarning)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

# Import data generator functions
from data_generator import TENSOR_MANAGER_STATE_FILE, load_data, scale_memory_stats  # noqa: E402

from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator  # noqa: E402
from flextensor.strategy import (  # noqa: E402
    AdaptiveStrategy,
    BudgetFillGreedyStrategy,
    BudgetFillLayerDEStrategy,
    BudgetFillStrategy,
    BudgetFillTensorDEStrategy,
    GlobalOffloadStrategy,
    GlobalTensorSelectionStrategy,
    KnapsackBlockStrategy,
    KnapsackStrategy,
    OptimizedRoundRobinAssignment,
    StrictRoundRobinAssignment,
)
from flextensor.strategy.protocol import BlockStrategyData  # noqa: E402
from flextensor.strategy.utils import (  # noqa: E402
    calculate_transfer_to_compute_map,
    format_block_table,
    reverse_transfer_to_compute_map,
    strategy_has_transfer_gaps,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from flextensor.collectors import LayerStatistics, TensorStatistics


@dataclass
class LayerStats:
    """Per-layer statistics."""

    label: str
    tensor_count: int
    layer_size_bytes: int
    duration_ms: float
    block_id: int | None
    offload_size_bytes: int
    transfer_time_ms: float
    transfer_window_ms: float  # Available window for the transfer (gap-aware)
    is_async: bool  # True if transfer completes within transfer window


@dataclass
class BlockInfo:
    """Block-level statistics extracted from strategy results."""

    block_sizes: list[int]
    blocks_used: int
    peak_memory: int
    violations: int
    layer_to_block: dict[str, int] | None


@dataclass
class AnalysisResult:
    """Result of running a strategy analysis."""

    strategy_name: str
    layer_stats: list[LayerStats]
    block_sizes: list[int]
    blocks_used: int
    pipeline_violations: int
    scale: float
    peak_memory_bytes: int
    strategy_map: dict[str, list[TensorStatistics]] | None = None
    layer_to_block: dict[str, int] | None = None
    assignment: str = ""  # Assignment strategy: "", "Strict", "Optimized"
    optimization_time_s: float = 0.0  # Wall-clock time to run the strategy optimizer
    has_transfer_gaps: bool = False  # True if strategy has unused transfer slots


def count_pipeline_violations(layer_to_block: dict[str, int]) -> int:
    """Count consecutive layers assigned to the same block.

    Cyclic wrap-around (last layer vs first layer) is NOT counted because
    the PreallocatedBatchTransferTensorLoader handles it via cross-iteration
    synchronization: the runtime's final join orders the previous iteration
    before slot reuse. A cyclic overlap only adds latency (waiting for the
    transfer), which the optimizer already accounts for via the ``scale``
    parameter.

    Args:
        layer_to_block: Mapping of layer label to block ID.

    Returns:
        Number of consecutive pipeline violations.
    """
    labels = list(layer_to_block.keys())
    if len(labels) < 2:
        return 0

    return sum(1 for i in range(len(labels) - 1) if layer_to_block[labels[i]] == layer_to_block[labels[i + 1]])


def compute_peak_memory(layer_stats: list[LayerStats], block_sizes: list[int]) -> int:
    """Compute peak GPU memory including resident tensors + block memory.

    Peak memory = block_memory + total_non_offloaded_tensors

    Non-offloaded tensors stay on GPU permanently throughout inference.
    Layer 0 is always fully on GPU (preloaded). For layers 1..N-1, offloaded
    tensors are in strategy_map[prev_label] and stored on CPU.
    """
    block_memory = sum(block_sizes)
    total_model = sum(ls.layer_size_bytes for ls in layer_stats)
    total_offloaded = sum(ls.offload_size_bytes for ls in layer_stats)
    total_non_offloaded = total_model - total_offloaded
    peak = block_memory + total_non_offloaded
    return peak


def compute_layer_stats(
    layer_stats_list: list[LayerStatistics],
    strategy_map: dict[str, list[TensorStatistics]],
    layer_to_block: dict[str, int] | None,
    interpolator: MemoryTransferInterpolator,
) -> list[LayerStats]:
    """Compute per-layer statistics from strategy results.

    Uses ``transfer_to_compute_map`` to correctly attribute transfers when
    strategies skip gap layers (e.g. ``strategy["4"] = L6's tensors``).
    The transfer window for the async check spans all layers from the
    transfer slot through the layer before the compute slot.

    Args:
        layer_stats_list: List of layer statistics from profiling.
        strategy_map: Mapping of layer label -> tensors to transfer during that layer's compute.
        layer_to_block: Mapping of layer label -> block ID assignment.
        interpolator: Memory transfer interpolator for estimating transfer times.

    Returns:
        List of LayerStats with computed offload metrics.
    """
    t2c_map = calculate_transfer_to_compute_map(layer_stats_list, strategy_map)
    compute_to_transfer = reverse_transfer_to_compute_map(t2c_map)

    label_to_idx = {layer.label: idx for idx, layer in enumerate(layer_stats_list)}

    layer_stats = []
    for i, layer in enumerate(layer_stats_list):
        layer_size = sum(t.size_bytes for t in layer.tensors)
        block_id = layer_to_block.get(layer.label) if layer_to_block else None

        transfer_label = compute_to_transfer.get(layer.label)
        offload_tensors = strategy_map.get(transfer_label, []) if transfer_label is not None else []

        offload_size = sum(t.size_bytes for t in offload_tensors)
        transfer_time = interpolator.bytes_to_duration(offload_size) if offload_size > 0 else 0.0

        if i == 0 or transfer_label is None:
            window = 0.0
            is_async = True
        else:
            transfer_idx = label_to_idx.get(transfer_label, i - 1)
            window = sum(layer_stats_list[j].duration for j in range(transfer_idx, i))
            is_async = transfer_time <= window

        layer_stats.append(
            LayerStats(
                label=layer.label,
                tensor_count=len(layer.tensors),
                layer_size_bytes=layer_size,
                duration_ms=layer.duration,
                block_id=block_id,
                offload_size_bytes=offload_size,
                transfer_time_ms=transfer_time,
                transfer_window_ms=window,
                is_async=is_async,
            )
        )

    return layer_stats


def extract_block_info(
    block_data: object | None,
    layer_stats: list[LayerStats],
    n_blocks: int,
    fallback_peak_memory: int = 0,
) -> BlockInfo:
    """Extract block-level statistics from strategy block_data.

    Args:
        block_data: Block data from strategy compute result (has block_sizes, label_to_block_id).
        layer_stats: List of computed layer statistics.
        n_blocks: Number of blocks configured.
        fallback_peak_memory: Peak memory to use if block_data is None (default: 0).

    Returns:
        BlockInfo with extracted statistics.
    """
    if block_data:
        block_sizes_raw = block_data.block_sizes
        # Convert to list, filtering out zero-sized blocks
        if isinstance(block_sizes_raw, dict):
            # Dict format: only include non-zero blocks
            block_sizes = [size for size in block_sizes_raw.values() if size > 0]
        else:
            # List format: filter out zeros
            block_sizes = [size for size in block_sizes_raw if size > 0]
        blocks_used = len({ls.block_id for ls in layer_stats if ls.block_id is not None})
        peak_memory = sum(block_sizes)
        # Count consecutive pipeline violations (cyclic is handled by loader sync).
        # Track original positions: gap layers (with no block) create natural
        # pipeline barriers, so same-block layers separated by a gap are NOT violations.
        block_layers_with_pos = [(idx, ls) for idx, ls in enumerate(layer_stats) if ls.block_id is not None]
        violations = sum(
            1
            for i in range(len(block_layers_with_pos) - 1)
            if block_layers_with_pos[i][1].block_id == block_layers_with_pos[i + 1][1].block_id
            and block_layers_with_pos[i + 1][0] - block_layers_with_pos[i][0] == 1
        )
        layer_to_block = block_data.label_to_block_id
    else:
        block_sizes = []
        blocks_used = 0
        peak_memory = fallback_peak_memory
        violations = 0
        layer_to_block = None

    return BlockInfo(
        block_sizes=block_sizes,
        blocks_used=blocks_used,
        peak_memory=peak_memory,
        violations=violations,
        layer_to_block=layer_to_block,
    )


def run_global_strategy(
    layer_stats_list: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
    n_blocks: int,
    max_gpu_mem_bytes: int,
    memory_stats: dict[int, float],
    assignment_strategy: object | None = None,
    assignment_name: str = "",
) -> AnalysisResult:
    """Run GlobalOffloadStrategy and collect results."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        strategy = GlobalOffloadStrategy(
            n_blocks=n_blocks,
            threshold_mb=1.0,
            min_blocks=2,
            max_blocks=n_blocks,
            assignment_strategy=assignment_strategy,
        )
        t0 = time.perf_counter()
        compute_result = strategy.compute(
            layer_stats_list, memory_stats=memory_stats, max_gpu_mem_bytes=max_gpu_mem_bytes
        )
        optimization_time = time.perf_counter() - t0
        strategy_map = compute_result.strategy_map

    # Extract scale from warnings if adjusted
    scale = 1.0
    for w in caught_warnings:
        if "Automatically adjusted scale to" in str(w.message):
            # Parse scale from warning message
            msg = str(w.message)
            with contextlib.suppress(IndexError, ValueError):
                scale = float(msg.split("adjusted scale to ")[1].split(" ")[0])

    # Build layer stats using shared helper
    layer_stats = compute_layer_stats(layer_stats_list, strategy_map, strategy.optimal_layer_to_block, interpolator)

    return AnalysisResult(
        strategy_name="GlobalOffload",
        layer_stats=layer_stats,
        block_sizes=strategy.optimal_block_sizes,
        blocks_used=len(set(strategy.optimal_layer_to_block.values())),
        pipeline_violations=count_pipeline_violations(strategy.optimal_layer_to_block),
        scale=scale,
        peak_memory_bytes=strategy.optimal_peak_memory,
        strategy_map=strategy_map,
        layer_to_block=strategy.optimal_layer_to_block,
        assignment=assignment_name,
        optimization_time_s=optimization_time,
    )


def run_knapsack_strategy(
    layer_stats_list: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
    scale: float = 1.0,
    n_blocks: int = 4,
    strategy_name: str = "Knapsack (per-layer)",
    max_gpu_mem_bytes: int | None = None,
) -> AnalysisResult:
    """Run original KnapsackStrategy and collect results.

    Note: Knapsack strategy_map stores tensors under PREVIOUS layer's label:
    - strategy["embed"] = tensors from layer_0 that fit in embed's compute
    - strategy["0"] = tensors from layer_1 that fit in layer_0's compute
    So for layer N, we look up strategy[layer_{N-1}.label] to get the tensors
    that were transferred FOR layer N during layer N-1's compute.
    """
    strategy = KnapsackStrategy(
        scale=scale,
        n_blocks=n_blocks,
    )
    t0 = time.perf_counter()
    compute_result = strategy.compute(layer_stats_list, max_gpu_mem_bytes=max_gpu_mem_bytes)
    optimization_time = time.perf_counter() - t0
    strategy_map = compute_result.strategy_map
    block_data = compute_result.block_data

    # Build layer stats and extract block info using shared helpers
    layer_to_block = block_data.label_to_block_id if block_data else None
    layer_stats = compute_layer_stats(layer_stats_list, strategy_map, layer_to_block, interpolator)
    block_info = extract_block_info(block_data, layer_stats, n_blocks)

    peak_memory = compute_peak_memory(layer_stats, block_info.block_sizes)

    return AnalysisResult(
        strategy_name=strategy_name,
        layer_stats=layer_stats,
        block_sizes=block_info.block_sizes,
        blocks_used=block_info.blocks_used,
        pipeline_violations=block_info.violations,
        scale=scale,
        peak_memory_bytes=peak_memory,
        strategy_map=strategy_map,
        layer_to_block=block_info.layer_to_block,
        optimization_time_s=optimization_time,
    )


def run_knapsack_block_strategy(
    layer_stats_list: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
    memory_stats: dict[int, float],
    scale: float = 1.0,
    n_blocks: int = 4,
    max_gpu_mem_bytes: int | None = None,
) -> AnalysisResult:
    """Run KnapsackBlockStrategy with provided memory stats.

    Note: KnapsackBlock strategy_map stores tensors under PREVIOUS layer's label:
    - strategy["embed"] = tensors from layer_0 that fit in embed's compute
    - strategy["0"] = tensors from layer_1 that fit in layer_0's compute
    So for layer N, we look up strategy[layer_{N-1}.label] to get the tensors
    that were transferred FOR layer N during layer N-1's compute.
    """
    strategy = KnapsackBlockStrategy(
        scale=scale,
        threshold_mb=1.0,
        n_blocks=n_blocks,
    )
    t0 = time.perf_counter()
    compute_result = strategy.compute(layer_stats_list, memory_stats, max_gpu_mem_bytes=max_gpu_mem_bytes)
    optimization_time = time.perf_counter() - t0
    strategy_map = compute_result.strategy_map
    block_data = compute_result.block_data

    # Build layer stats and extract block info using shared helpers
    layer_to_block = block_data.label_to_block_id if block_data else None
    layer_stats = compute_layer_stats(layer_stats_list, strategy_map, layer_to_block, interpolator)
    block_info = extract_block_info(block_data, layer_stats, n_blocks)

    peak_memory = compute_peak_memory(layer_stats, block_info.block_sizes)

    return AnalysisResult(
        strategy_name="KnapsackBlock",
        layer_stats=layer_stats,
        block_sizes=block_info.block_sizes,
        blocks_used=block_info.blocks_used,
        pipeline_violations=block_info.violations,
        scale=scale,
        peak_memory_bytes=peak_memory,
        strategy_map=strategy_map,
        layer_to_block=block_info.layer_to_block,
        optimization_time_s=optimization_time,
    )


def run_budget_fill_strategy(
    layer_stats_list: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
    memory_stats: dict[int, float],
    *,
    scale: float = 1.0,
    n_blocks: int = 4,
    max_gpu_mem_bytes: int | None = None,
    assignment_strategy: object | None = None,
    assignment_name: str = "Optimized",
) -> AnalysisResult:
    """Run BudgetFillStrategy facade (greedy + optional layer DE)."""
    if max_gpu_mem_bytes is None:
        msg = "budget-fill requires --gpu-mem / max_gpu_mem_bytes"
        raise ValueError(msg)
    if assignment_strategy is None:
        assignment_strategy = OptimizedRoundRobinAssignment(min_blocks=2, max_blocks=n_blocks)
        assignment_name = "Optimized"
    strategy = BudgetFillStrategy(
        n_blocks=n_blocks,
        threshold_mb=1.0,
        scale=scale,
        assignment_strategy=assignment_strategy,  # type: ignore[arg-type]
        enable_layer_de=True,
        # enable_tensor_de defaults True (non-prefix fallback; Adaptive sets False).
    )
    t0 = time.perf_counter()
    compute_result = strategy.compute(layer_stats_list, memory_stats, max_gpu_mem_bytes=max_gpu_mem_bytes)
    optimization_time = time.perf_counter() - t0
    strategy_map = compute_result.strategy_map
    block_data = compute_result.block_data

    layer_to_block = block_data.label_to_block_id if block_data else None
    layer_stats = compute_layer_stats(layer_stats_list, strategy_map, layer_to_block, interpolator)
    block_info = extract_block_info(block_data, layer_stats, n_blocks)

    peak_memory = compute_peak_memory(layer_stats, block_info.block_sizes)

    return AnalysisResult(
        strategy_name="BudgetFill",
        layer_stats=layer_stats,
        block_sizes=block_info.block_sizes,
        blocks_used=block_info.blocks_used,
        pipeline_violations=block_info.violations,
        scale=scale,
        peak_memory_bytes=peak_memory,
        strategy_map=strategy_map,
        layer_to_block=block_info.layer_to_block,
        optimization_time_s=optimization_time,
        assignment=assignment_name,
    )


def run_budget_fill_greedy_strategy(
    layer_stats_list: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
    memory_stats: dict[int, float],
    *,
    scale: float = 1.0,
    n_blocks: int = 4,
    max_gpu_mem_bytes: int | None = None,
    assignment_strategy: object | None = None,
    assignment_name: str = "Optimized",
) -> AnalysisResult:
    """Run BudgetFillGreedyStrategy alone (ranked-prefix scan)."""
    if max_gpu_mem_bytes is None:
        msg = "budget-fill-greedy requires --gpu-mem / max_gpu_mem_bytes"
        raise ValueError(msg)
    if assignment_strategy is None:
        assignment_strategy = OptimizedRoundRobinAssignment(min_blocks=2, max_blocks=n_blocks)
        assignment_name = "Optimized"
    strategy = BudgetFillGreedyStrategy(
        n_blocks=n_blocks,
        threshold_mb=1.0,
        scale=scale,
        assignment_strategy=assignment_strategy,  # type: ignore[arg-type]
    )
    t0 = time.perf_counter()
    compute_result = strategy.compute(layer_stats_list, memory_stats, max_gpu_mem_bytes=max_gpu_mem_bytes)
    optimization_time = time.perf_counter() - t0
    strategy_map = compute_result.strategy_map
    block_data = compute_result.block_data

    layer_to_block = block_data.label_to_block_id if block_data else None
    layer_stats = compute_layer_stats(layer_stats_list, strategy_map, layer_to_block, interpolator)
    block_info = extract_block_info(block_data, layer_stats, n_blocks)

    peak_memory = compute_peak_memory(layer_stats, block_info.block_sizes)

    return AnalysisResult(
        strategy_name="BudgetFillGreedy",
        layer_stats=layer_stats,
        block_sizes=block_info.block_sizes,
        blocks_used=block_info.blocks_used,
        pipeline_violations=block_info.violations,
        scale=scale,
        peak_memory_bytes=peak_memory,
        strategy_map=strategy_map,
        layer_to_block=block_info.layer_to_block,
        optimization_time_s=optimization_time,
        assignment=assignment_name,
    )


def run_budget_fill_tensor_de_strategy(
    layer_stats_list: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
    memory_stats: dict[int, float],
    *,
    scale: float = 1.0,
    n_blocks: int = 4,
    max_gpu_mem_bytes: int | None = None,
    assignment_strategy: object | None = None,
    assignment_name: str = "Optimized",
) -> AnalysisResult:
    """Run BudgetFillTensorDEStrategy (DE binary selection under a hard GPU budget)."""
    if max_gpu_mem_bytes is None:
        msg = "budget-fill-tensor-de requires --gpu-mem / max_gpu_mem_bytes"
        raise ValueError(msg)
    if assignment_strategy is None:
        assignment_strategy = OptimizedRoundRobinAssignment(min_blocks=2, max_blocks=n_blocks)
        assignment_name = "Optimized"
    strategy = BudgetFillTensorDEStrategy(
        n_blocks=n_blocks,
        threshold_mb=1.0,
        scale=scale,
        assignment_strategy=assignment_strategy,  # type: ignore[arg-type]
        pop_size=30,
        epoch=80,
        max_early_stop=25,
        seed=42,
    )
    t0 = time.perf_counter()
    compute_result = strategy.compute(layer_stats_list, memory_stats, max_gpu_mem_bytes=max_gpu_mem_bytes)
    optimization_time = time.perf_counter() - t0
    strategy_map = compute_result.strategy_map
    block_data = compute_result.block_data

    layer_to_block = block_data.label_to_block_id if block_data else None
    layer_stats = compute_layer_stats(layer_stats_list, strategy_map, layer_to_block, interpolator)
    block_info = extract_block_info(block_data, layer_stats, n_blocks)

    peak_memory = compute_peak_memory(layer_stats, block_info.block_sizes)

    return AnalysisResult(
        strategy_name="BudgetFillTensorDE",
        layer_stats=layer_stats,
        block_sizes=block_info.block_sizes,
        blocks_used=block_info.blocks_used,
        pipeline_violations=block_info.violations,
        scale=scale,
        peak_memory_bytes=peak_memory,
        strategy_map=strategy_map,
        layer_to_block=block_info.layer_to_block,
        optimization_time_s=optimization_time,
        assignment=assignment_name,
    )


def run_budget_fill_layer_de_strategy(
    layer_stats_list: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
    memory_stats: dict[int, float],
    *,
    scale: float = 1.0,
    n_blocks: int = 4,
    max_gpu_mem_bytes: int | None = None,
    assignment_strategy: object | None = None,
    assignment_name: str = "Optimized",
) -> AnalysisResult:
    """Run BudgetFillLayerDEStrategy (per-layer DE under a hard GPU budget)."""
    if max_gpu_mem_bytes is None:
        msg = "budget-fill-layer-de requires --gpu-mem / max_gpu_mem_bytes"
        raise ValueError(msg)
    if assignment_strategy is None:
        assignment_strategy = OptimizedRoundRobinAssignment(min_blocks=2, max_blocks=n_blocks)
        assignment_name = "Optimized"
    strategy = BudgetFillLayerDEStrategy(
        n_blocks=n_blocks,
        threshold_mb=1.0,
        scale=scale,
        assignment_strategy=assignment_strategy,  # type: ignore[arg-type]
        pop_size=20,
        epoch=60,
        max_early_stop=20,
        seed=42,
    )
    t0 = time.perf_counter()
    compute_result = strategy.compute(layer_stats_list, memory_stats, max_gpu_mem_bytes=max_gpu_mem_bytes)
    optimization_time = time.perf_counter() - t0
    strategy_map = compute_result.strategy_map
    block_data = compute_result.block_data

    layer_to_block = block_data.label_to_block_id if block_data else None
    layer_stats = compute_layer_stats(layer_stats_list, strategy_map, layer_to_block, interpolator)
    block_info = extract_block_info(block_data, layer_stats, n_blocks)

    peak_memory = compute_peak_memory(layer_stats, block_info.block_sizes)

    return AnalysisResult(
        strategy_name="BudgetFillLayerDE",
        layer_stats=layer_stats,
        block_sizes=block_info.block_sizes,
        blocks_used=block_info.blocks_used,
        pipeline_violations=block_info.violations,
        scale=scale,
        peak_memory_bytes=peak_memory,
        strategy_map=strategy_map,
        layer_to_block=block_info.layer_to_block,
        optimization_time_s=optimization_time,
        assignment=assignment_name,
    )


def run_tensor_selection(
    layer_stats_list: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
    memory_stats: dict[int, float],
    n_blocks: int = 4,
    max_gpu_mem_bytes: int | None = None,
    assignment_strategy: object | None = None,
    assignment_name: str = "",
) -> AnalysisResult:
    """Run GlobalTensorSelectionStrategy strategy.

    This strategy optimizes which tensors to offload per layer using
    scipy optimization, with max_gpu_mem_bytes as the primary constraint.

    Args:
        layer_stats_list: Layer statistics.
        interpolator: Memory transfer interpolator.
        memory_stats: Memory transfer benchmark data.
        n_blocks: Number of memory blocks.
        max_gpu_mem_bytes: Maximum GPU memory limit in bytes.
        assignment_strategy: Optional assignment strategy (StrictRoundRobinAssignment,
            OptimizedRoundRobinAssignment, or None for default).
        assignment_name: Name of assignment strategy for display.
    """
    strategy = GlobalTensorSelectionStrategy(
        n_blocks=n_blocks,
        threshold_mb=1.0,
        pop_size=50,
        epoch=100,
        max_early_stop=50,
        scale=0.9,  # Lower bound: 10% margin (transfers within 90% of compute)
        assignment_strategy=assignment_strategy,
    )
    t0 = time.perf_counter()
    compute_result = strategy.compute(
        layer_stats_list, memory_stats, max_gpu_mem_bytes=max_gpu_mem_bytes or (48 * 1024**3)
    )
    optimization_time = time.perf_counter() - t0
    strategy_map = compute_result.strategy_map
    block_data = compute_result.block_data

    # Build layer stats and extract block info using shared helpers
    layer_to_block = block_data.label_to_block_id if block_data else None
    layer_stats = compute_layer_stats(layer_stats_list, strategy_map, layer_to_block, interpolator)
    block_info = extract_block_info(block_data, layer_stats, n_blocks)

    # Get optimized scale from strategy (if available)
    optimized_scale = getattr(strategy, "optimal_scale", 1.0)

    # Use strategy.optimal_peak_memory which includes tensors kept on GPU + block memory
    peak_memory = getattr(strategy, "optimal_peak_memory", block_info.peak_memory)

    return AnalysisResult(
        strategy_name="GlobalTensorSelectionStrategy",
        layer_stats=layer_stats,
        block_sizes=block_info.block_sizes,
        blocks_used=block_info.blocks_used,
        pipeline_violations=block_info.violations,
        scale=optimized_scale,
        peak_memory_bytes=peak_memory,
        strategy_map=strategy_map,
        layer_to_block=block_info.layer_to_block,
        assignment=assignment_name,
        optimization_time_s=optimization_time,
    )


def run_adaptive_strategy(
    layer_stats_list: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
    memory_stats: dict[int, float],
    *,
    scale: float = 1.0,
    n_blocks: int = 4,
    max_gpu_mem_bytes: int | None = None,
    extra_optimization: bool = False,
) -> AnalysisResult:
    """Run AdaptiveStrategy which evaluates all strategies and picks the best.

    Args:
        layer_stats_list: Layer statistics.
        interpolator: Memory transfer interpolator.
        memory_stats: Memory transfer benchmark data.
        scale: Scale factor passed to all sub-strategies.
        n_blocks: Number of memory blocks.
        max_gpu_mem_bytes: Maximum GPU memory limit in bytes.
        extra_optimization: Include slower TensorSelection candidates.
    """
    strategy = AdaptiveStrategy(
        scale=scale,
        loader_type="allocation_block_transfer",
        threshold_mb=1.0,
        n_blocks=n_blocks,
        extra_optimization=extra_optimization,
    )
    t0 = time.perf_counter()
    compute_result = strategy.compute(layer_stats_list, memory_stats, max_gpu_mem_bytes=max_gpu_mem_bytes)
    optimization_time = time.perf_counter() - t0
    strategy_map = compute_result.strategy_map
    block_data = compute_result.block_data

    layer_to_block = block_data.label_to_block_id if block_data else None
    layer_stats = compute_layer_stats(layer_stats_list, strategy_map, layer_to_block, interpolator)
    block_info = extract_block_info(block_data, layer_stats, n_blocks)

    peak_memory = compute_peak_memory(layer_stats, block_info.block_sizes)

    # Report the selected sub-strategy as the assignment name
    selected = strategy.selected_strategy_name

    label = "AdaptiveStrategy(extra)" if extra_optimization else "AdaptiveStrategy"

    return AnalysisResult(
        strategy_name=label,
        layer_stats=layer_stats,
        block_sizes=block_info.block_sizes,
        blocks_used=block_info.blocks_used,
        pipeline_violations=block_info.violations,
        scale=scale,
        peak_memory_bytes=peak_memory,
        strategy_map=strategy_map,
        layer_to_block=block_info.layer_to_block,
        assignment=selected,
        optimization_time_s=optimization_time,
    )


def format_bytes(size_bytes: int | float) -> str:
    """Format bytes to human-readable string."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


def print_layer_statistics(result: AnalysisResult, detailed: bool = False) -> None:
    """Print per-layer statistics with pipeline view.

    Pipeline model (shared transfer bus):
    - During layer N's compute, we transfer layer N+1's data
    - Transfer must complete within layer N's compute time
    - The transfer bus is shared, so only one transfer at a time
    """
    print(f"\n{'=' * 120}")
    print(f"LAYER STATISTICS: {result.strategy_name}")
    print("=" * 120)

    if detailed:
        print(
            f"{'Layer':<15} {'Block':>6} {'Size':>10} {'Compute':>10} | "
            f"{'Prefetch':>12} {'→Block':>7} {'Transfer':>10} {'Fits?':>6} {'Margin':>10}"
        )
        print("-" * 120)

        layers = result.layer_stats

        for i, ls in enumerate(layers):
            block_str = str(ls.block_id) if ls.block_id is not None else "N/A"

            # During this layer's compute, what do we prefetch?
            if i + 1 < len(layers):
                next_layer = layers[i + 1]
                prefetch_label = next_layer.label[:12]
                prefetch_block = str(next_layer.block_id) if next_layer.block_id is not None else "N/A"
                prefetch_time = next_layer.transfer_time_ms

                # Does the prefetch fit within this layer's compute time?
                fits = prefetch_time <= ls.duration_ms
                fits_str = "✓" if fits else "✗"
                margin = ls.duration_ms - prefetch_time
                margin_str = f"{margin:+.2f}ms"
            else:
                # Last layer - no prefetch needed
                prefetch_label = "-"
                prefetch_block = "-"
                prefetch_time = 0.0
                fits_str = "-"
                margin_str = "-"

            print(
                f"{ls.label:<15} {block_str:>6} {format_bytes(ls.layer_size_bytes):>10} "
                f"{ls.duration_ms:>10.2f} | "
                f"{prefetch_label:>12} {prefetch_block:>7} {prefetch_time:>10.2f} "
                f"{fits_str:>6} {margin_str:>10}"
            )
    else:
        # Summary per-layer stats
        print(f"Total layers: {len(result.layer_stats)}")
        sizes = [ls.layer_size_bytes for ls in result.layer_stats]
        durations = [ls.duration_ms for ls in result.layer_stats]
        print(
            f"Layer sizes: min={format_bytes(min(sizes))}, max={format_bytes(max(sizes))}, "
            f"avg={format_bytes(sum(sizes) / len(sizes))}"
        )
        print(
            f"Durations: min={min(durations):.2f}ms, max={max(durations):.2f}ms, "
            f"avg={sum(durations) / len(durations):.2f}ms"
        )


def print_transfer_table(
    layer_stats_list: list[LayerStatistics],
    strategy_map: dict[str, list[TensorStatistics]],
    interpolator: MemoryTransferInterpolator,
    strategy_name: str,
) -> None:
    """Print simple transfer table showing layer size, offload size, and transfer time.

    Pipeline model:
    - First layer is preloaded (transfer = 0)
    - During layer N's compute, we prefetch layer N+1's data
    - So each row shows: current layer's compute time vs NEXT layer's transfer time

    Note: Knapsack strategies store tensors under CURRENT layer's label:
    - strategy["embed"] = tensors for layer_0 (transferred during embed's compute)
    - strategy["0"] = tensors for layer_1 (transferred during layer_0's compute)
    """
    print(f"\n{'=' * 120}")
    print(f"TRANSFER TABLE: {strategy_name}")
    print("=" * 120)
    print(
        f"{'Layer':<12} {'Layer Size':>12} {'Duration':>12} | "
        f"{'Prefetch':>12} {'Prefetch Size':>14} {'Transfer ms':>12} {'Fits?':>8}"
    )
    print("-" * 120)

    for i, layer in enumerate(layer_stats_list):
        layer_size = sum(t.size_bytes for t in layer.tensors)

        # Get transfer that happens DURING this layer's compute
        # strategy[layer.label] = tensors for next layer, transferred during this layer's compute
        if i + 1 < len(layer_stats_list):
            next_layer = layer_stats_list[i + 1]
            # Use CURRENT layer's label to get what's transferred during its compute
            offload_tensors = strategy_map.get(layer.label, [])
            offload_size = sum(t.size_bytes for t in offload_tensors)
            transfer_time = interpolator.bytes_to_duration(offload_size) if offload_size > 0 else 0.0
            fits = "✓" if transfer_time <= layer.duration else "✗"
            prefetch_label = next_layer.label
        else:
            # Last layer - no next layer to prefetch
            offload_size = 0
            transfer_time = 0.0
            fits = "-"
            prefetch_label = "-"

        print(
            f"{layer.label:<12} "
            f"{format_bytes(layer_size):>12} "
            f"{layer.duration:>10.2f}ms | "
            f"{prefetch_label:>12} "
            f"{format_bytes(offload_size) if offload_size > 0 else '-':>14} "
            f"{transfer_time:>10.2f}ms "
            f"{fits:>8}"
        )

    print()


def print_offload_decisions(
    layer_stats_list: list[LayerStatistics],
    strategy_map: dict[str, list[TensorStatistics]],
    layer_to_block: dict[str, int],
    block_sizes: list[int],
    strategy_name: str,
) -> None:
    """Print offloading decisions per layer.

    Shows what gets transferred during each layer's compute time.

    Note: Knapsack strategies store tensors under CURRENT layer's label:
    - strategy["embed"] = tensors for layer_0 (transferred during embed's compute)
    So "Transfer For" shows the next layer whose tensors are being transferred.
    """
    print(f"\n{'=' * 140}")
    print(f"OFFLOAD DECISIONS: {strategy_name}")
    print("=" * 140)

    print(
        f"{'During':<15} {'Block':>6} {'Duration':>10} | "
        f"{'Transfer For':>12} {'Tensors':>8} {'Size':>12} {'Transfer':>10} | "
        f"{'Block Size':>12}"
    )
    print("-" * 140)

    total_offload = 0

    for i, layer in enumerate(layer_stats_list):
        label = layer.label

        # Get offload decisions - these are tensors for the NEXT layer
        offload_tensors = strategy_map.get(label, [])
        offload_count = len(offload_tensors)
        offload_size = sum(t.size_bytes for t in offload_tensors)
        total_offload += offload_size

        # Next layer label (what's being transferred for)
        next_label = layer_stats_list[i + 1].label if i + 1 < len(layer_stats_list) else "-"

        # Block info
        block_id = layer_to_block.get(label)
        block_str = str(block_id) if block_id is not None else "N/A"
        block_size = block_sizes[block_id] if block_id is not None and block_id < len(block_sizes) else 0

        # Estimate transfer time (would need interpolator for accurate calculation)
        transfer_str = f"{offload_size / 1e9 * 10:.1f}ms" if offload_size > 0 else "-"

        print(
            f"{label:<15} {block_str:>6} {layer.duration:>10.2f} | "
            f"{next_label:>12} {offload_count:>8} {format_bytes(offload_size):>12} {transfer_str:>10} | "
            f"{format_bytes(block_size):>12}"
        )

    print("-" * 140)
    print(f"{'TOTAL':<15} {'':>6} {'':>10} | {'':>12} {'':>8} {format_bytes(total_offload):>12} {'':>10} |")

    # Count pipeline violations (consecutive layers in EXECUTION order with same block)
    layers_with_blocks = [
        (layer.label, layer_to_block.get(layer.label))
        for layer in layer_stats_list
        if layer_to_block.get(layer.label) is not None
    ]
    violations = sum(
        1 for i in range(len(layers_with_blocks) - 1) if layers_with_blocks[i][1] == layers_with_blocks[i + 1][1]
    )

    # Block summary
    print("\nBlock Summary:")
    blocks_used = len({b for b in layer_to_block.values() if b is not None})
    print(f"  Blocks used:        {blocks_used}")
    print(f"  Pipeline violations: {violations}")
    print("\nBlock Sizes:")
    for i, size in enumerate(block_sizes):
        if size > 0:
            print(f"  Block {i}: {format_bytes(size)}")


def print_block_assignment_table(
    result: AnalysisResult,
    layer_stats_list: list[LayerStatistics],
) -> None:
    """Print per-layer block assignment table using the shared library formatter."""
    strategy_map = result.strategy_map or {}

    block_data: BlockStrategyData | None = None
    if result.layer_to_block:
        t2c_map = calculate_transfer_to_compute_map(layer_stats_list, strategy_map) if strategy_map else {}
        block_data = BlockStrategyData(
            label_to_block_id=result.layer_to_block,
            block_sizes=list(result.block_sizes),
            label_to_size_map={},
            allocation_ordered={},
            transfer_to_compute_map=t2c_map,
        )

    print(format_block_table(layer_stats_list, strategy_map, block_data, result.strategy_name))


def print_summary_statistics(result: AnalysisResult, model_size_bytes: int) -> None:
    """Print summary statistics for a strategy."""
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {result.strategy_name}")
    print("=" * 80)

    # Calculate totals
    total_duration = sum(ls.duration_ms for ls in result.layer_stats)
    total_offload = sum(ls.offload_size_bytes for ls in result.layer_stats)
    total_transfer = sum(ls.transfer_time_ms for ls in result.layer_stats)
    async_layers = sum(1 for ls in result.layer_stats if ls.is_async)
    sync_layers = len(result.layer_stats) - async_layers

    # Calculate extra time from synchronous (non-overlapped) transfers.
    # Only the portion that exceeds the gap-aware transfer window is overhead.
    extra_time = sum(ls.transfer_time_ms - ls.transfer_window_ms for ls in result.layer_stats if not ls.is_async)

    duration_with_offload = total_duration + extra_time
    actual_scale = duration_with_offload / total_duration if total_duration > 0 else 1.0

    print("\nTiming:")
    print(f"  Duration without offload:     {total_duration:.2f} ms")
    print(f"  Duration with offload:        {duration_with_offload:.2f} ms (actual={actual_scale:.2f}x)")
    print(f"  Total transfer time:          {total_transfer:.2f} ms")
    print(f"  Extra time from sync:         {extra_time:.2f} ms")
    print(f"  Overhead:                     {(actual_scale - 1) * 100:.1f}%")

    print("\nAsync Analysis:")
    print(
        f"  Async layers (hidden):        {async_layers} / {len(result.layer_stats)} "
        f"({async_layers / len(result.layer_stats) * 100:.1f}%)"
    )
    print(f"  Sync layers (visible):        {sync_layers} / {len(result.layer_stats)}")

    print("\nMemory:")
    print(f"  Model size:                   {format_bytes(model_size_bytes)}")
    print(f"  Total offload size:           {format_bytes(total_offload)}")
    print(f"  Peak GPU memory:              {format_bytes(result.peak_memory_bytes)}")
    if result.block_sizes:
        active_blocks = [s for s in result.block_sizes if s > 0]
        print(f"  Block sizes:                  {[format_bytes(s) for s in active_blocks]}")

    print("\nBlocks:")
    print(f"  Blocks used:                  {result.blocks_used}")
    print(f"  Pipeline violations:          {result.pipeline_violations}")
    print(f"  Actual slowdown:              {actual_scale:.2f}x")
    if result.scale != 1.0:
        print(f"  Required scale (from opt):    {result.scale:.2f}x")
    print(f"  Transfer gaps detected:       {'Yes' if result.has_transfer_gaps else 'No'}")

    if result.optimization_time_s > 0:
        print("\nOptimization:")
        print(f"  Optimization time:            {result.optimization_time_s:.2f}s")


def compare_strategies(results: list[AnalysisResult], model_size_bytes: int, title: str | None = None) -> None:
    """Print comparison table of all strategies."""
    print("\n" + "=" * 140)
    print(title or "STRATEGY COMPARISON")
    print("=" * 140)

    headers = [
        "Strategy",
        "Assignment",
        "Blk",
        "Viol",
        "Gaps",
        "Scale",
        "Duration",
        "Overhead",
        "Async%",
        "Peak GPU",
        "Offload",
        "Opt Time",
    ]
    print(
        f"{headers[0]:<20} {headers[1]:<10} {headers[2]:>4} {headers[3]:>4} {headers[4]:>4} {headers[5]:>5} "
        f"{headers[6]:>12} {headers[7]:>8} {headers[8]:>7} {headers[9]:>12} {headers[10]:>12} "
        f"{headers[11]:>10}"
    )
    print("-" * 160)

    for result in results:
        total_duration = sum(ls.duration_ms for ls in result.layer_stats)
        total_offload = sum(ls.offload_size_bytes for ls in result.layer_stats)
        async_pct = sum(1 for ls in result.layer_stats if ls.is_async) / len(result.layer_stats) * 100

        # Only the excess beyond the gap-aware transfer window is overhead
        extra_time = sum(ls.transfer_time_ms - ls.transfer_window_ms for ls in result.layer_stats if not ls.is_async)
        duration_with_offload = total_duration + extra_time
        actual_scale = duration_with_offload / total_duration if total_duration > 0 else 1.0
        overhead = (actual_scale - 1) * 100

        assignment = result.assignment if result.assignment else "-"
        violations = result.pipeline_violations
        gaps = "Yes" if result.has_transfer_gaps else "No"
        opt_time = f"{result.optimization_time_s:.1f}s" if result.optimization_time_s > 0 else "-"
        print(
            f"{result.strategy_name:<20} {assignment:<10} {result.blocks_used:>4} {violations:>4} {gaps:>4} "
            f"{actual_scale:>5.2f} {duration_with_offload:>10.1f}ms {overhead:>7.1f}% "
            f"{async_pct:>6.1f}% {format_bytes(result.peak_memory_bytes):>12} "
            f"{format_bytes(total_offload):>12} {opt_time:>10}"
        )

    # Add legend for violations
    print()
    print("Note: Viol = Consecutive pipeline violations (adjacent layers sharing a block).")
    print("      Cyclic wrap-around (last layer vs first layer) is NOT counted because the")
    print("      loader's cross-iteration sync handles it — it only adds transfer wait time.")


def main() -> None:  # noqa: C901
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze offload strategies on DeepSeek R1 model data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    available_strategies = [
        "global",
        "global-strict",
        "knapsack",
        "knapsack-block",
        "budget-fill",
        "budget-fill-greedy",
        "budget-fill-tensor-de",
        "budget-fill-layer-de",
        "budget-fill-compare",
        "tensor-select",
        "tensor-select-strict",
        "adaptive",
        "adaptive-extra",
        "all",
        "recommended",
    ]

    parser.add_argument(
        "--strategy",
        choices=available_strategies,
        default="all",
        help="Strategy to run (default: all)",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=4,
        help="Number of blocks for global strategies (default: 4)",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=48.0,
        help="GPU memory limit in GB (default: 48.0)",
    )

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Show diagnostic tables (detailed layer stats, offload decisions, transfer table, block assignments)",
    )
    parser.add_argument(
        "--transfer-budget-scale",
        type=float,
        default=1.0,
        help="Transfer budget scale factor for strategy (default: 1.0)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic data instead of loading from file",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=61,
        help="Number of transformer layers for generated data (default: 61)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generated data (default: None = random)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate results and exit with code 1 if any violations or memory exceeded",
    )
    parser.add_argument(
        "--transfer-ratio",
        nargs="+",
        type=float,
        default=[1.0, 0.1],
        help="Transfer speed ratio(s): 1.0 = full speed, 0.1 = 10%% speed. "
        "Multiple values run side-by-side comparison (default: 1.0 0.1)",
    )
    parser.add_argument(
        "--gap-layers",
        nargs="+",
        type=int,
        default=None,
        help="Transformer layer indices to simulate as gaps (no tensors). "
        "Only applies to generated data. Example: --gap-layers 5 10 15",
    )

    args = parser.parse_args()

    # Load or generate model data
    if args.generate:
        print(f"Generating synthetic DeepSeek R1-like model data (seed={args.seed})...")
        layer_stats_list, memory_stats, _interpolator = load_data(
            use_file=False,
            num_layers=args.num_layers,
            seed=args.seed,
            gap_layers=args.gap_layers,
        )
        if args.gap_layers:
            print(f"Gap layers (no tensors): {args.gap_layers}")
    else:
        print("Loading DeepSeek R1 model data...")
        if not TENSOR_MANAGER_STATE_FILE.exists():
            print(f"  Note: {TENSOR_MANAGER_STATE_FILE} not found, generating synthetic data...")
        layer_stats_list, memory_stats, _interpolator = load_data(use_file=True)

    # Calculate model size
    model_size_bytes = sum(sum(t.size_bytes for t in layer.tensors) for layer in layer_stats_list)
    print(f"Model size: {format_bytes(model_size_bytes)}")
    print(f"Total layers: {len(layer_stats_list)}")
    print(f"GPU memory limit: {args.gpu_mem} GB")
    print(f"Number of blocks: {args.n_blocks}")

    max_gpu_mem_bytes = int(args.gpu_mem * 1024**3)

    # Define which strategies to run
    if args.strategy == "recommended":
        strategies_to_run = [
            "global",
            "knapsack",
            "knapsack-block",
            "budget-fill",
            "budget-fill-greedy",
            "budget-fill-layer-de",
            "tensor-select",
            "adaptive",
            "adaptive-extra",
        ]
    elif args.strategy == "budget-fill-compare":
        strategies_to_run = [
            "budget-fill",
            "budget-fill-greedy",
            "budget-fill-layer-de",
            "budget-fill-tensor-de",
        ]
    elif args.strategy == "all":
        strategies_to_run = [
            "global",
            "global-strict",
            "knapsack",
            "knapsack-block",
            "budget-fill",
            "budget-fill-greedy",
            "budget-fill-tensor-de",
            "budget-fill-layer-de",
            "tensor-select",
            "tensor-select-strict",
            "adaptive",
            "adaptive-extra",
        ]
    else:
        strategies_to_run = [args.strategy]

    ratios_to_run: list[float] = args.transfer_ratio
    all_ratio_results: dict[str, list[AnalysisResult]] = {}

    for ratio in ratios_to_run:
        scaled_stats = scale_memory_stats(memory_stats, ratio)
        profile_interpolator = MemoryTransferInterpolator(scaled_stats)
        ratio_label = f"{ratio}x" if ratio != int(ratio) else f"{int(ratio)}x"

        if len(ratios_to_run) > 1:
            print(f"\n{'#' * 80}")
            print(f"# Transfer ratio: {ratio_label}")
            print(f"{'#' * 80}")

        # Build strategy runners for this ratio
        strategies: dict[str, Callable[[], AnalysisResult]] = {
            "global": lambda _i=profile_interpolator, _m=scaled_stats: run_global_strategy(
                layer_stats_list,
                _i,
                args.n_blocks,
                max_gpu_mem_bytes,
                _m,
                assignment_strategy=None,
                assignment_name="Optimized",
            ),
            "global-strict": lambda _i=profile_interpolator, _m=scaled_stats: run_global_strategy(
                layer_stats_list,
                _i,
                args.n_blocks,
                max_gpu_mem_bytes,
                _m,
                assignment_strategy=StrictRoundRobinAssignment(),
                assignment_name="Strict",
            ),
            "knapsack": lambda _i=profile_interpolator: run_knapsack_strategy(
                layer_stats_list,
                _i,
                scale=args.transfer_budget_scale,
                n_blocks=args.n_blocks,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
            ),
            "knapsack-block": lambda _i=profile_interpolator, _m=scaled_stats: run_knapsack_block_strategy(
                layer_stats_list,
                _i,
                _m,
                scale=args.transfer_budget_scale,
                n_blocks=args.n_blocks,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
            ),
            "budget-fill": lambda _i=profile_interpolator, _m=scaled_stats: run_budget_fill_strategy(
                layer_stats_list,
                _i,
                _m,
                scale=args.transfer_budget_scale,
                n_blocks=args.n_blocks,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
            ),
            "budget-fill-greedy": lambda _i=profile_interpolator, _m=scaled_stats: (
                run_budget_fill_greedy_strategy(
                    layer_stats_list,
                    _i,
                    _m,
                    scale=args.transfer_budget_scale,
                    n_blocks=args.n_blocks,
                    max_gpu_mem_bytes=max_gpu_mem_bytes,
                )
            ),
            "budget-fill-tensor-de": lambda _i=profile_interpolator,
            _m=scaled_stats: run_budget_fill_tensor_de_strategy(
                layer_stats_list,
                _i,
                _m,
                scale=args.transfer_budget_scale,
                n_blocks=args.n_blocks,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
            ),
            "budget-fill-layer-de": lambda _i=profile_interpolator, _m=scaled_stats: (
                run_budget_fill_layer_de_strategy(
                    layer_stats_list,
                    _i,
                    _m,
                    scale=args.transfer_budget_scale,
                    n_blocks=args.n_blocks,
                    max_gpu_mem_bytes=max_gpu_mem_bytes,
                )
            ),
            "tensor-select": lambda _i=profile_interpolator, _m=scaled_stats: run_tensor_selection(
                layer_stats_list,
                _i,
                _m,
                n_blocks=args.n_blocks,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
                assignment_strategy=None,
                assignment_name="Optimized",
            ),
            "tensor-select-strict": lambda _i=profile_interpolator, _m=scaled_stats: run_tensor_selection(
                layer_stats_list,
                _i,
                _m,
                n_blocks=args.n_blocks,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
                assignment_strategy=StrictRoundRobinAssignment(),
                assignment_name="Strict",
            ),
            "adaptive": lambda _i=profile_interpolator, _m=scaled_stats: run_adaptive_strategy(
                layer_stats_list,
                _i,
                _m,
                scale=args.transfer_budget_scale,
                n_blocks=args.n_blocks,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
            ),
            "adaptive-extra": lambda _i=profile_interpolator, _m=scaled_stats: run_adaptive_strategy(
                layer_stats_list,
                _i,
                _m,
                scale=args.transfer_budget_scale,
                n_blocks=args.n_blocks,
                max_gpu_mem_bytes=max_gpu_mem_bytes,
                extra_optimization=True,
            ),
        }

        results: list[AnalysisResult] = []

        for name in strategies_to_run:
            runner = strategies[name]
            print(f"\nRunning {name} strategy...")
            result = runner()
            if result.strategy_map:
                result.has_transfer_gaps = strategy_has_transfer_gaps(result.strategy_map, layer_stats_list)
            results.append(result)
            if args.diagnostics:
                print_layer_statistics(result, detailed=True)
                if result.strategy_map and result.layer_to_block:
                    print_offload_decisions(
                        layer_stats_list,
                        result.strategy_map,
                        result.layer_to_block,
                        result.block_sizes,
                        result.strategy_name,
                    )
                if result.strategy_map:
                    print_transfer_table(
                        layer_stats_list, result.strategy_map, profile_interpolator, result.strategy_name
                    )
                print_block_assignment_table(result, layer_stats_list)
            print_summary_statistics(result, model_size_bytes)

        # Print comparison for this ratio
        if len(results) > 1:
            title = f"STRATEGY COMPARISON ({ratio_label} transfer)" if len(ratios_to_run) > 1 else None
            compare_strategies(results, model_size_bytes, title=title)

        all_ratio_results[ratio_label] = results

    # Validate results if requested (across all ratios)
    if args.validate:
        import sys

        failed = False
        print("\n" + "=" * 80)
        print("VALIDATION")
        print("=" * 80)

        for ratio_label, results in all_ratio_results.items():
            if len(all_ratio_results) > 1:
                print(f"\n--- {ratio_label} transfer ---")
            for result in results:
                strategy_name = (
                    f"{result.strategy_name} ({result.assignment})" if result.assignment else result.strategy_name
                )
                violations = result.pipeline_violations
                peak_memory = result.peak_memory_bytes
                memory_exceeded = peak_memory > max_gpu_mem_bytes

                if violations > 0:
                    print(f"FAIL: {strategy_name} has {violations} pipeline violation(s)")
                    failed = True
                if memory_exceeded:
                    print(
                        f"FAIL: {strategy_name} exceeds GPU memory: "
                        f"{format_bytes(peak_memory)} > {format_bytes(max_gpu_mem_bytes)}"
                    )
                    failed = True
                if violations == 0 and not memory_exceeded:
                    print(f"PASS: {strategy_name}")

        if failed:
            print("\nValidation FAILED")
            sys.exit(1)
        else:
            print("\nValidation PASSED")
            sys.exit(0)


if __name__ == "__main__":
    main()

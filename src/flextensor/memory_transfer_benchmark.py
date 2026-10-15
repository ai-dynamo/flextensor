# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory transfer benchmarking and extraction utilities.

This module provides functions for obtaining memory transfer statistics,
either through live GPU benchmarking or extraction from layer statistics.
"""

import gc

import torch

from .collectors import LayerStatistics, extract_memory_transfers_from_layer_stats  # noqa: F401
from .tensor_processors import BenchmarkTensorProcessor
from .utils import calculate_tensor_size


def _format_human_size(size_bytes: int) -> str:
    """Format byte size in human-readable units (B, KiB, MiB, GiB)."""
    for unit, threshold in [("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)]:
        if size_bytes >= threshold:
            return f"{size_bytes / threshold:.1f} {unit}"
    return f"{size_bytes} B"


def format_memory_transfer_table(memory_stats: dict[int, float]) -> str:
    """Format memory transfer statistics as a diagnostic table.

    Args:
        memory_stats: Dictionary mapping tensor size (bytes) to transfer time (ms).

    Returns:
        Formatted table string matching the diagnostic table style.
    """
    if not memory_stats:
        return "No memory transfer statistics available."

    width = 110
    header = "=" * width
    col_header = f"{'Size (bytes)':>14} {'Size':>10} {'Transfer (ms)':>14} {'Bandwidth (GB/s)':>17}"

    lines = [
        header,
        "Memory Transfer Statistics",
        header,
        col_header,
        "-" * width,
    ]

    for size_bytes, time_ms in sorted(memory_stats.items()):
        human = _format_human_size(size_bytes)
        bandwidth = size_bytes / (time_ms / 1000) / 1e9 if time_ms > 0 else 0.0
        lines.append(f"{size_bytes:>14} {human:>10} {time_ms:>14.3f} {bandwidth:>17.3f}")

    lines.append(header)
    return "\n".join(lines)


def benchmark_memory_transfers(
    layer_stats: list[LayerStatistics],
    device_gpu: torch.device,
    max_depth: int = 4,
    max_size_bytes: int = 4 * 1024 * 1024 * 1024,
    pin_memory: bool = True,
) -> dict[int, float]:
    """
    Benchmark memory transfer speeds for different tensor sizes.

    Performs live memory transfer benchmarks by creating tensors of various sizes
    and measuring actual CPU->GPU transfer times.

    Args:
        layer_stats: List of layer statistics to determine max tensor size.
        device_gpu: Target GPU device for transfers.
        max_depth: Number of power-of-2 divisions to benchmark (default 4).
        max_size_bytes: Maximum single tensor size to benchmark (default 4GB).
        pin_memory: Whether to use pinned memory for transfers (default True).

    Returns:
        Dictionary mapping tensor size (bytes) to transfer time (ms).
    """
    # Find max layer size to determine benchmark range
    max_layer_size_bytes = 0
    for layer in layer_stats:
        layer_size = sum(tensor.size_bytes for tensor in layer.tensors)
        max_layer_size_bytes = max(max_layer_size_bytes, layer_size)

    # Determine element size dynamically from dtype
    tensor_dtype = torch.float32
    element_size = torch.ones(1, dtype=tensor_dtype).element_size()

    # Align max_size_bytes to element boundary
    max_size_bytes_aligned = (max_size_bytes // element_size) * element_size

    # Determine tensor sizes to benchmark (in bytes, aligned to element size)
    base_size = min(max_layer_size_bytes, max_size_bytes_aligned)
    base_size_aligned = (base_size // element_size) * element_size
    tensor_sizes_bytes = [
        (int(base_size_aligned / (2**depth)) // element_size) * element_size for depth in range(max_depth)
    ]

    # Benchmark each size
    memory_transfers: dict[int, float] = {}
    benchmark_tensor_processor = BenchmarkTensorProcessor(device_gpu=device_gpu)

    for tensor_size_bytes in tensor_sizes_bytes:
        if tensor_size_bytes <= 0:
            continue
        # Convert bytes to number of elements
        num_elements = tensor_size_bytes // element_size
        if num_elements <= 0:
            continue
        tensor_cpu = torch.rand(num_elements, dtype=tensor_dtype, device="cpu", pin_memory=pin_memory)
        new_tensor = benchmark_tensor_processor.process(tensor_cpu)
        transfer_time = benchmark_tensor_processor.get_results()["tensor_statistics_map"][id(new_tensor)].load_time_ms
        # Store with actual byte size as key (calculated from tensor properties)
        actual_size_bytes = calculate_tensor_size(tensor_cpu)
        memory_transfers[actual_size_bytes] = transfer_time

        # Release memory
        benchmark_tensor_processor.tensors_map.clear()
        del tensor_cpu
        gc.collect()
        torch._C._host_emptyCache()  # type: ignore[attr-defined, unused-ignore]  # noqa: SLF001  # private API, name varies by torch version

    torch._C._host_emptyCache()  # type: ignore[attr-defined, unused-ignore]  # noqa: SLF001  # private API, name varies by torch version
    gc.collect()

    return memory_transfers

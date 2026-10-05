# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch.overrides import TorchFunctionMode

from flextensor.collectors import TensorStatistics
from flextensor.host_pinning import HostPinner
from flextensor.tensor import wrap_trace_tensor
from flextensor.utils import calculate_tensor_size


class TensorBenchmarkMode(TorchFunctionMode, ABC):
    """
    Abstract base class for tensor benchmarking modes.

    This class defines the common interface that all tensor benchmark implementations
    must follow to be used interchangeably as benchmark_cls parameter in TensorManager.
    """

    @abstractmethod
    def __init__(
        self,
        device_gpu: torch.device,
        pinned_memory: bool = True,
        iterations: int = 10,
        pinned_memory_limit_mb: float | None = None,
        host_pinner: HostPinner | None = None,
    ) -> None:
        """
        Initialize tensor benchmarking mode.

        Args:
            device_gpu: Target GPU device for benchmarking
            pinned_memory: Whether to use pinned memory for tensors
            iterations: Number of benchmark iterations
            pinned_memory_limit_mb: Maximum size of tensors that will be copied to pinned_memory
            host_pinner: :class:`HostPinner` controlling how CPU tensors are pinned.
                Defaults to torch mode (PyTorch's caching pinned allocator) when ``None``.
        """

    @abstractmethod
    def get_results(self) -> dict[str, Any]:
        """
        Get benchmark results.

        Returns:
            Dictionary containing benchmark results with keys:
            - tensor_statistics_map: Mapping of tensor IDs to statistics
            - tensors_map: Mapping of tensor IDs to tensors
        """


class BenchmarkReplace(TensorBenchmarkMode):
    """
    Benchmark mode that measures tensor transfer times and collects statistics.

    This implementation intercepts tensor operations, measures transfer times from
    CPU to GPU, and collects comprehensive statistics about tensor usage.
    """

    def __init__(
        self,
        device_gpu: torch.device,
        pinned_memory: bool = True,
        iterations: int = 10,
        pinned_memory_limit_mb: float | None = None,
        *,
        host_pinner: HostPinner,
    ) -> None:
        """
        Initialize BenchmarkReplace for tensor benchmarking.

        Args:
            device_gpu: Target GPU device for benchmarking
            pinned_memory: Whether to use pinned memory for tensors
            iterations: Number of benchmark iterations
            pinned_memory_limit_mb: Maximum size of tensors that will be copied to pinned_memory
            host_pinner: :class:`HostPinner` controlling how CPU tensors are pinned.
                Required (keyword-only) so profiling stats always reflect the
                same pinning primitive that production transfers will use —
                a silent fallback to ``HostPinner()`` would let
                ``pinned_memory_mode="host_register"`` report ``"torch"``-mode
                latencies.
        """
        self.iterations = iterations
        self.warmup_iterations = 5
        self.device_gpu = device_gpu
        self.pinned_memory = pinned_memory
        self.tensor_statistics_map: dict[int, TensorStatistics] = {}
        self.tensors_map: dict[int, Any] = {}
        self.pinned_memory_limit_mb = pinned_memory_limit_mb
        self.host_pinner = host_pinner

    def get_results(self) -> dict[str, Any]:
        """
        Get benchmark results.

        Returns:
            Dictionary containing benchmark results with keys:
            - tensor_statistics_map: Mapping of tensor IDs to statistics
            - tensors_map: Mapping of tensor IDs to tensors
        """
        return {
            "tensor_statistics_map": self.tensor_statistics_map,
            "tensors_map": self.tensors_map,
        }

    def __torch_function__(  # type: ignore[override]
        self,
        func: Callable[..., Any],
        _types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """
        Intercept tensor operations and measure transfer times.

        This method is called by PyTorch when a tensor operation is performed.
        It intercepts the operation, measures the transfer time, and returns
        the result wrapped in TraceTensor.

        Args:
            func: The function to intercept
            _types: The types of the arguments
            args: The arguments to the function
            kwargs: The keyword arguments to the function

        Returns:
            The result of the function, wrapped in TraceTensor if it's a tensor
        """
        result = func(*args, **(kwargs or {}))
        # Only process tensor results, skip non-tensor returns
        # Skip processing meta tensors (they have no data to transfer)
        if not isinstance(result, torch.Tensor) or result.is_meta:
            return result

        result_tensor = result

        if result_tensor.device.type != "cpu":
            return result  # TODO: fix me, We keep some tensors in GPU to exclude from optimization (deepseek model)

        tensor_size = calculate_tensor_size(result_tensor)
        tensor_size_mb = tensor_size / 1024 / 1024
        if (
            self.pinned_memory
            and not self.host_pinner.is_pinned(result_tensor)
            and (self.pinned_memory_limit_mb is None or tensor_size_mb <= self.pinned_memory_limit_mb)
        ):
            result_tensor = self.host_pinner.pin(result_tensor)

        transfer_times = []
        for _ in range(self.warmup_iterations):
            result_tensor.to(device=self.device_gpu)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = result_tensor.to(device=self.device_gpu)
            end_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(end_event)
            transfer_times.append(duration)

        new_tensor = wrap_trace_tensor(result_tensor)
        new_tensor = new_tensor.requires_grad_(requires_grad=False)
        tensor_id = id(new_tensor)
        self.tensor_statistics_map[tensor_id] = TensorStatistics(
            tensor_id=tensor_id,
            name="",
            size_bytes=tensor_size,
            load_time_ms=statistics.median(transfer_times),
        )
        self.tensors_map[tensor_id] = new_tensor
        return new_tensor


class PreloadToDevice(TensorBenchmarkMode):
    """
    Benchmark mode that immediately transfers tensors to GPU without timing measurements.

    This implementation provides a simple preloading strategy without collecting
    detailed benchmark statistics, suitable for scenarios where you just want
    tensors moved to GPU without measurement overhead.
    """

    def __init__(
        self,
        device_gpu: torch.device,
        pinned_memory: bool = True,
        iterations: int = 10,
        pinned_memory_limit_mb: float | None = None,
        host_pinner: HostPinner | None = None,
    ) -> None:
        """
        Initialize PreloadToDevice for simple tensor preloading.

        Args:
            device_gpu: Target GPU device for preloading
            pinned_memory: Whether to use pinned memory for tensors (ignored in this implementation)
            iterations: Number of benchmark iterations (ignored in this implementation)
            pinned_memory_limit_mb: Maximum size of tensors that will be copied to pinned_memory
            host_pinner: Stored for interface compatibility but not used; kept so callers
                like ``TensorManager.benchmark_context`` can pass it uniformly.
        """
        self.device_gpu = device_gpu
        self.pinned_memory = pinned_memory  # Stored for interface compatibility but not used
        self.iterations = iterations  # Stored for interface compatibility but not used
        self.tensor_statistics_map: dict[int, TensorStatistics] = {}
        self.tensors_map: dict[int, Any] = {}
        self.pinned_memory_limit_mb = pinned_memory_limit_mb
        self.host_pinner = host_pinner  # accepted for ABC compatibility, not used

    def get_results(self) -> dict[str, Any]:
        """
        Get benchmark results.

        Returns:
            Dictionary containing benchmark results with keys:
            - tensor_statistics_map: Mapping of tensor IDs to statistics (empty for this implementation)
            - tensors_map: Mapping of tensor IDs to tensors
        """
        return {
            "tensor_statistics_map": self.tensor_statistics_map,
            "tensors_map": self.tensors_map,
        }

    def __torch_function__(  # type: ignore[override]
        self,
        func: Callable[..., Any],
        _types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """
        Intercept tensor operations and immediately transfer to GPU.

        This method is called by PyTorch when a tensor operation is performed.
        It intercepts the operation and immediately transfers the result to GPU
        without timing measurements.

        Args:
            func: The function to intercept
            _types: The types of the arguments
            args: The arguments to the function
            kwargs: The keyword arguments to the function

        Returns:
            The result of the function, transferred to GPU if it's a tensor
        """
        result = func(*args, **(kwargs or {}))
        # Only process tensor results, skip non-tensor returns
        # Skip processing meta tensors (they have no data to transfer)
        if not isinstance(result, torch.Tensor) or result.is_meta:
            return result

        result_tensor = result

        gpu_tensor = result_tensor.to(device=self.device_gpu)

        # Store tensor for interface compatibility
        tensor_id = id(gpu_tensor)
        self.tensors_map[tensor_id] = gpu_tensor

        return gpu_tensor


class NoOpBenchmark(TensorBenchmarkMode):
    def __init__(
        self,
        device_gpu: torch.device,
        pinned_memory: bool = True,
        iterations: int = 10,
        pinned_memory_limit_mb: float | None = None,
        host_pinner: HostPinner | None = None,
    ) -> None:
        self.device_gpu = device_gpu
        self.pinned_memory = pinned_memory  # Stored for interface compatibility but not used
        self.iterations = iterations  # Stored for interface compatibility but not used
        self.tensor_statistics_map: dict[int, TensorStatistics] = {}
        self.tensors_map: dict[int, Any] = {}
        self.pinned_memory_limit_mb = pinned_memory_limit_mb
        self.host_pinner = host_pinner  # accepted for ABC compatibility, not used

    def get_results(self) -> dict[str, Any]:
        """
        Get benchmark results.

        Returns:
            Dictionary containing benchmark results with keys:
            - tensor_statistics_map: Mapping of tensor IDs to statistics (empty for this implementation)
            - tensors_map: Mapping of tensor IDs to tensors
        """
        return {
            "tensor_statistics_map": self.tensor_statistics_map,
            "tensors_map": self.tensors_map,
        }

    def __torch_function__(  # type: ignore[override]
        self,
        func: Callable[..., Any],
        _types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return func(*args, **(kwargs or {}))

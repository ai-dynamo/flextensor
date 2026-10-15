# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory transfer interpolation utilities for predicting transfer times and sizes."""

from typing import cast

import numpy as np


class MemoryTransferInterpolator:
    """
    Bidirectional interpolator for memory transfer size <-> duration mapping.

    This class provides methods to:
    - Predict transfer duration given a size in bytes
    - Predict transfer size given a duration in milliseconds

    Uses log-log linear interpolation for accurate predictions across wide ranges.
    """

    def __init__(self, memory_transfers: dict[int, float]):
        """
        Initialize the interpolator with measured memory transfer data.

        Args:
            memory_transfers: Dictionary mapping bytes (int) to duration_ms (float)
        """
        if not memory_transfers:
            message = "memory_transfers dictionary cannot be empty"
            raise ValueError(message)

        # Sort by bytes for proper interpolation
        sorted_items = sorted(memory_transfers.items())
        self.bytes_array = np.array([item[0] for item in sorted_items], dtype=np.float64)
        self.duration_array = np.array([item[1] for item in sorted_items], dtype=np.float64)

        # Validate all values are positive
        if np.any(self.bytes_array <= 0):
            message = "All byte values must be positive"
            raise ValueError(message)
        if np.any(self.duration_array <= 0):
            message = "All duration values must be positive"
            raise ValueError(message)

        # Store bounds for validation
        self.min_bytes = self.bytes_array[0]
        self.max_bytes = self.bytes_array[-1]
        self.min_duration = self.duration_array[0]
        self.max_duration = self.duration_array[-1]

        # Precompute log values for interpolation
        # Using linear interpolation in log-log space for better accuracy
        # (memory transfers typically have near-linear relationship in log space)
        self.log_bytes = np.log(self.bytes_array)
        self.log_duration = np.log(self.duration_array)

    def bytes_to_duration(self, size_bytes: int | float) -> float:
        """
        Predict transfer duration for a given size.

        Supports both interpolation and extrapolation using linear
        interpolation in log-log space.

        Args:
            size_bytes: Transfer size in bytes (int or float)

        Returns:
            Predicted duration in milliseconds
        """
        if size_bytes <= 0:
            message = f"size_bytes must be positive, got {size_bytes}"
            raise ValueError(message)

        log_size = np.log(float(size_bytes))

        # Handle single data point case
        if len(self.log_bytes) == 1:
            # No interpolation possible, return the single duration value
            return float(self.duration_array[0])

        # Use np.interp which does flat extrapolation by default
        # For better extrapolation, we manually extend the linear trend
        if log_size < self.log_bytes[0]:
            # Extrapolate below: use slope from first two points
            slope = (self.log_duration[1] - self.log_duration[0]) / (self.log_bytes[1] - self.log_bytes[0])
            log_duration = self.log_duration[0] + slope * (log_size - self.log_bytes[0])
        elif log_size > self.log_bytes[-1]:
            # Extrapolate above: use slope from last two points
            slope = (self.log_duration[-1] - self.log_duration[-2]) / (self.log_bytes[-1] - self.log_bytes[-2])
            log_duration = self.log_duration[-1] + slope * (log_size - self.log_bytes[-1])
        else:
            # Interpolate within range
            log_duration = np.interp(log_size, self.log_bytes, self.log_duration)

        return float(np.exp(log_duration))

    def duration_to_bytes(self, duration_ms: float) -> float:
        """
        Predict transfer size for a given duration.

        Supports both interpolation and extrapolation using linear
        interpolation in log-log space.

        Args:
            duration_ms: Transfer duration in milliseconds

        Returns:
            Predicted size in bytes
        """
        if duration_ms <= 0:
            message = f"duration_ms must be positive, got {duration_ms}"
            raise ValueError(message)

        log_dur = np.log(float(duration_ms))

        # Handle single data point case
        if len(self.log_duration) == 1:
            # No interpolation possible, return the single bytes value
            return float(self.bytes_array[0])

        # Use np.interp which does flat extrapolation by default
        # For better extrapolation, we manually extend the linear trend
        if log_dur < self.log_duration[0]:
            # Extrapolate below: use slope from first two points
            dur_diff = self.log_duration[1] - self.log_duration[0]
            if dur_diff != 0:
                slope = (self.log_bytes[1] - self.log_bytes[0]) / dur_diff
                log_bytes = self.log_bytes[0] + slope * (log_dur - self.log_duration[0])
            else:
                log_bytes = self.log_bytes[0]  # Flat extrapolation if no slope
        elif log_dur > self.log_duration[-1]:
            # Extrapolate above: use slope from last two points
            dur_diff = self.log_duration[-1] - self.log_duration[-2]
            if dur_diff != 0:
                slope = (self.log_bytes[-1] - self.log_bytes[-2]) / dur_diff
                log_bytes = self.log_bytes[-1] + slope * (log_dur - self.log_duration[-1])
            else:
                log_bytes = self.log_bytes[-1]  # Flat extrapolation if no slope
        else:
            # Interpolate within range
            log_bytes = np.interp(log_dur, self.log_duration, self.log_bytes)

        return float(np.exp(log_bytes))

    def get_bandwidth_mbps(self) -> float:
        """
        Calculate average bandwidth in MB/s based on the measured data.

        Returns:
            Average bandwidth in megabytes per second
        """
        # Use largest transfer for most accurate bandwidth estimate
        bytes_val = self.bytes_array[-1]
        duration_sec = self.duration_array[-1] / 1000.0  # Convert ms to seconds
        mb = bytes_val / (1024 * 1024)  # Convert bytes to MB
        return cast("float", mb / duration_sec)

    def __repr__(self) -> str:
        return (
            f"MemoryTransferInterpolator("
            f"bytes_range=[{self.min_bytes:.0f}, {self.max_bytes:.0f}], "
            f"duration_range=[{self.min_duration:.2f}, {self.max_duration:.2f}] ms, "
            f"bandwidth≈{self.get_bandwidth_mbps():.2f} MB/s)"
        )


def create_interpolator_from_benchmarks(memory_transfers: dict[int, float]) -> MemoryTransferInterpolator:
    """
    Convenience function to create an interpolator from benchmark results.

    Args:
        memory_transfers: Dictionary mapping bytes to duration_ms

    Returns:
        Configured MemoryTransferInterpolator instance

    Example:
        >>> transfers = {1073741824: 89.67, 536870912: 44.85}
        >>> interp = create_interpolator_from_benchmarks(transfers)
        >>> duration = interp.bytes_to_duration(1000000000)
        >>> size = interp.duration_to_bytes(50.0)
    """
    return MemoryTransferInterpolator(memory_transfers)

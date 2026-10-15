# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks and test-only memory instrumentation for real diffusion pipelines."""

from __future__ import annotations

import resource
import threading
import weakref
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import psutil
import torch


def assert_video_parity(reference: np.ndarray, candidate: np.ndarray, *, min_psnr: float) -> float:
    assert reference.shape == candidate.shape, f"video shape mismatch: {reference.shape} != {candidate.shape}"
    assert reference.ndim == 4 and reference.size > 0, "expected nonempty FHWC video"
    for video in (reference, candidate):
        assert np.isfinite(video).all(), "nonfinite video pixels"
        assert video.min() >= 0 and video.max() <= 1, "expected pixels in [0, 1]"
    mse = np.mean((reference.astype(np.float64) - candidate) ** 2, axis=(1, 2, 3))
    with np.errstate(divide="ignore"):
        scores = -10 * np.log10(mse)
    worst = int(np.argmin(scores))
    assert scores[worst] >= min_psnr, f"frame {worst}: PSNR {scores[worst]:.3f} dB < {min_psnr} dB"
    return float(scores[worst])


def assert_memory_limits(report: dict, limits: dict) -> None:
    for phase, thresholds in limits.items():
        assert phase in report, f"missing memory phase: {phase}"
        for metric, limit in thresholds.items():
            assert metric in report[phase], f"missing {phase}.{metric}"
            value = report[phase][metric]
            assert 0 <= value <= limit, f"{phase}.{metric}: {value / 1024**3:.3f} GiB > {limit / 1024**3:.3f} GiB"


class StorageTracker:
    """Weak references avoid turning the memory check into a memory leak.

    Buffer storage wins over weight views when both point at the same allocation.
    Component peaks are observed at allocation/forward boundaries, not allocator-wide.
    """

    def __init__(self):
        self.refs = {}

    def add(self, tensor: torch.Tensor, kind: str) -> None:
        self.refs[id(tensor)] = (weakref.ref(tensor), kind)

    def snapshot(self) -> dict[str, int]:
        storages = {}
        dead = []
        for key, (ref, kind) in self.refs.items():
            tensor = ref()
            if tensor is None:
                dead.append(key)
                continue
            if tensor.device.type not in {"cpu", "cuda"}:
                continue
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr())
            if identity not in storages or kind == "ft_buffers":
                storages[identity] = (tensor.device.type, kind, storage.nbytes(), tensor.is_pinned())
        for key in dead:
            del self.refs[key]
        result = {
            f"{device}_{kind}_bytes": 0
            for device in ("cpu", "gpu")
            for kind in ("weights", "ft_buffers", "total", "pinned")
        }
        for device, kind, size, pinned in storages.values():
            device = "gpu" if device == "cuda" else device
            result[f"{device}_{kind}_bytes"] += size
            result[f"{device}_total_bytes"] += size
            if pinned:
                result[f"{device}_pinned_bytes"] += size
        return result


class MemoryRecorder:
    """Track component allocations without modifying production classes."""

    def __init__(self):
        self.storage = StorageTracker()
        self.peaks = {}
        self.report = {}

    def sample(self) -> dict[str, int]:
        current = self.storage.snapshot()
        for key, value in current.items():
            self.peaks[key] = max(self.peaks.get(key, 0), value)
        return current

    @contextmanager
    def allocations(self):
        from flextensor.allocation_block import AllocationBlock
        from flextensor.host_pinning import HostPinner
        from flextensor.profile_block_controller import ProfileBlockController

        pin = HostPinner.pin
        allocate = AllocationBlock._make_base_block
        profile_allocate = ProfileBlockController._allocate_blocks

        def record_pin(pinner, tensor):
            self.storage.add(tensor, "ft_buffers")
            self.sample()
            result = pin(pinner, tensor)
            self.storage.add(result, "ft_buffers")
            self.sample()  # Includes the temporary pageable + pinned copy overlap.
            return result

        def record_allocate(block, *args, **kwargs):
            result = allocate(block, *args, **kwargs)
            self.storage.add(result, "ft_buffers")
            self.sample()
            return result

        def record_profile_allocate(controller, *args, **kwargs):
            for tensor in controller._original_data.values():
                self.storage.add(tensor, "weights")
            result = profile_allocate(controller, *args, **kwargs)
            for tensor in result:
                self.storage.add(tensor, "ft_buffers")
            self.sample()
            return result

        with (
            patch.object(HostPinner, "pin", record_pin),
            patch.object(AllocationBlock, "_make_base_block", record_allocate),
            patch.object(ProfileBlockController, "_allocate_blocks", record_profile_allocate),
        ):
            yield

    @contextmanager
    def phase(self, name: str):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.peaks = {}
        self.sample()
        process = psutil.Process()
        rss_peak = process.memory_info().rss
        stop = threading.Event()

        def sample_rss():
            nonlocal rss_peak
            # ponytail: 20 ms RSS sampling can miss shorter peaks; lifetime VmHWM is also recorded.
            while not stop.wait(0.02):
                rss_peak = max(rss_peak, process.memory_info().rss)

        thread = threading.Thread(target=sample_rss, daemon=True)
        thread.start()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            stop.set()
            thread.join()
            self.sample()
            self.report[name] = {
                **{f"{key.removesuffix('_bytes')}_peak_bytes": value for key, value in self.peaks.items()},
                "gpu_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                "gpu_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
                "rss_peak_sampled_bytes": max(rss_peak, process.memory_info().rss),
                "rss_lifetime_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }

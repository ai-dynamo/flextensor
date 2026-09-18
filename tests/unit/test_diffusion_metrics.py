# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The diffusion gate must reject a bad frame, invalid output, and memory regressions."""

import gc
import json
import signal
import time
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
import torch

from tests.integration import _diffusion as diffusion
from tests.integration import _diffusion_metrics as metrics
from tests.integration._diffusion import _track_pipeline


def test_parity_rejects_a_bad_frame_even_when_global_psnr_passes():
    reference = np.zeros((100, 2, 2, 3), dtype=np.float32)
    candidate = reference.copy()
    candidate[50] = 0.02  # Global PSNR is 54 dB, this frame is only 34 dB.
    with pytest.raises(AssertionError, match="frame 50"):
        metrics.assert_video_parity(reference, candidate, min_psnr=48)
    candidate[:] = 1 / 255  # 48.13 dB, independently hand-computed.
    assert metrics.assert_video_parity(reference, candidate, min_psnr=48) == pytest.approx(48.1308, abs=1e-4)
    assert metrics.assert_video_parity(reference, reference, min_psnr=48) == float("inf")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_parity_rejects_invalid_pixels(bad):
    reference = np.zeros((5, 2, 2, 3), dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0, 0, 0] = bad
    with pytest.raises(AssertionError):
        metrics.assert_video_parity(reference, candidate, min_psnr=48)


def test_parity_rejects_missing_frames_and_empty_video():
    reference = np.zeros((5, 2, 2, 3), dtype=np.float32)
    for candidate in (reference[:4], reference[:0]):
        with pytest.raises(AssertionError):
            metrics.assert_video_parity(reference, candidate, min_psnr=48)
    with pytest.raises(AssertionError):
        metrics.assert_video_parity(reference[:0], reference[:0], min_psnr=48)


def test_storage_tracker_counts_aliases_once_and_does_not_retain_allocations():
    tracker = metrics.StorageTracker()
    weight = torch.zeros(16, dtype=torch.float32)
    view = weight[4:]
    tracker.add(weight, "weights")
    tracker.add(view, "ft_buffers")
    snapshot = tracker.snapshot()
    assert snapshot["cpu_total_bytes"] == 64
    assert snapshot["cpu_ft_buffers_bytes"] == 64
    assert snapshot["cpu_weights_bytes"] == 0
    del weight, view
    gc.collect()
    assert tracker.snapshot()["cpu_total_bytes"] == 0


def test_memory_gate_fails_on_peak_even_when_final_usage_is_low():
    report = {"startup": {"gpu_allocated_peak_bytes": 101, "rss_peak_sampled_bytes": 10}}
    with pytest.raises(AssertionError, match=r"startup.*gpu_allocated"):
        metrics.assert_memory_limits(report, {"startup": {"gpu_allocated_peak_bytes": 100}})
    report["startup"]["gpu_allocated_peak_bytes"] = 99
    metrics.assert_memory_limits(report, {"startup": {"gpu_allocated_peak_bytes": 100}})
    with pytest.raises(AssertionError, match="missing"):
        metrics.assert_memory_limits(report, {"inference": {"gpu_allocated_peak_bytes": 100}})


def test_memory_tracks_source_weights_retained_after_parameter_rebinding():
    memory = metrics.MemoryRecorder()
    model = torch.nn.Linear(4, 4, bias=False)
    original = model.weight.data
    model.weight.data = torch.empty(0)
    tensor_manager = SimpleNamespace(tensors_map={}, _replan_source_data={1: original})
    _track_pipeline(
        memory,
        SimpleNamespace(components={"transformer": model}),
        SimpleNamespace(_tensor_manager=tensor_manager),
    )
    assert memory.sample()["cpu_total_bytes"] == 64
    tensor_manager._replan_source_data.clear()
    del original
    gc.collect()
    assert memory.sample()["cpu_total_bytes"] == 0


@pytest.mark.parametrize("worker_exit", ["signal", "error", "success"])
def test_worker_exit_kills_group_and_preserves_async_cuda(monkeypatch, tmp_path, worker_exit):
    # A real lightweight subprocess stands in for model loading, including a compiler child.
    package = tmp_path / "tests" / "integration"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "_diffusion.py").write_text("""
import json, os, signal, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
out = Path(sys.argv[sys.argv.index("--output") + 1])
(out / "child.json").write_text(json.dumps({"pid": child.pid,
    "blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}))
if os.environ["WORKER_EXIT"] == "signal":
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(60)
(out / "metrics.json").write_text("{}")
sys.exit(7 if os.environ["WORKER_EXIT"] == "error" else 0)
""")
    monkeypatch.setattr(diffusion, "ROOT", tmp_path)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    monkeypatch.setenv("WORKER_EXIT", worker_exit)
    previous_handler = signal.getsignal(signal.SIGTERM)
    if worker_exit == "success":
        diffusion.run_mode(tmp_path / "out", mode="baseline", profile_dir=tmp_path / "profiles")
    else:
        with pytest.raises(SystemExit if worker_exit == "signal" else AssertionError):
            diffusion.run_mode(tmp_path / "out", mode="baseline", profile_dir=tmp_path / "profiles")
    child = json.loads((tmp_path / "out" / "child.json").read_text())
    for _ in range(100):
        try:
            if psutil.Process(child["pid"]).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.01)
    else:
        psutil.Process(child["pid"]).kill()
        pytest.fail("worker exit left a compiler child running")
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert child["blocking"] is None
    assert child["cublas"] == ":16:8"

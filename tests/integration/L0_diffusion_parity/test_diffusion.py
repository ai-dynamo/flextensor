# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wan2.1 1.3B parity/memory gate. L4 CI; fresh process per mode.

Native references: eager for eager FT; per-block compiled for compiled FT.
No baseline wrappers or norm changes. Every frame >=48 dB; latents atol=rtol=1e-3.
Cover first request, profile restore, replan, A -> B -> A. CUDA stays asynchronous.
compile_fn skips runtime discovery; external compile warms up eager, then replans.

GPU allocated/reserved caps: 22/23 GiB. RSS startup/inference caps: 48/32 GiB.
FT weight budget: 60%, gate 65%. Live storage growth <=64 MiB.
Component peaks sampled at allocation/denoising boundaries; exclude activations.
RSS sampled every 20 ms; startup also gates lifetime peak. Short spikes may escape.

Run ./test.sh; add -k eager for eager/profile only. Caller selects GPUs.
Artifacts: test_results/ (logs, metrics, raw frames/latents). Direct pytest: --basetemp.
Timeouts: DIFFUSION_TIMEOUT=3600 per worker, TIMEOUT=7200 per suite.
Cached checkpoint: HF_HUB_OFFLINE=1. Wan2.2 deferred until host loading shrinks.
"""

from pathlib import Path

import pytest

from tests.integration._diffusion import check_run, run_mode

# Keep compiled parity on the L4 CI tier.
pytestmark = [pytest.mark.gpu_vram_24g, pytest.mark.gpu_sm_89]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("wan-parity")
    for mode in ("baseline", "eager", "profile"):
        run_mode(root / mode, mode=mode, profile_dir=root / "profiles")
    return root


@pytest.mark.parametrize("mode", ["baseline", "eager", "profile"])
def test_eager_and_profile_parity_and_memory(runs: Path, mode: str):
    check_run(runs / mode, runs / "baseline", offloaded=mode != "baseline")


@pytest.fixture(scope="module")
def compiled_runs(runs):
    run_mode(runs / "baseline-compiled", mode="baseline-compiled", profile_dir=runs / "profiles")
    check_run(runs / "baseline-compiled", runs / "baseline-compiled", offloaded=False)
    return runs


@pytest.mark.parametrize("mode", ["compile-fn", "external-compile"])
def test_compiled_parity_and_memory(compiled_runs: Path, mode: str):
    run_mode(compiled_runs / mode, mode=mode, profile_dir=compiled_runs / "profiles")
    check_run(compiled_runs / mode, compiled_runs / "baseline-compiled", offloaded=True)

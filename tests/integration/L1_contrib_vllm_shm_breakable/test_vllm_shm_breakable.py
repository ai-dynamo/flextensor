# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real breakable CUDA-graph capture with SHM on both DP workers."""

import importlib.metadata
import json
import re
import shutil
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from packaging.version import Version

from flextensor.offload_manager import DEFAULT_MANAGER_NAME
from flextensor.shared_offload import SharedOffloadSession
from flextensor.state_handler import TensorManagerStateHandler
from tests.integration._vllm_server import VllmOffloadSmokeCase, run_vllm_server_test
from tests.integration._vllm_shm import (
    QWEN_ARGS,
    QWEN_CORRECTNESS,
    QWEN_MODEL,
    assert_namespace_removed,
    create_qwen_shm_profile,
)
from tests.integration._vllm_utils import sanitize_test_name

pytestmark = [pytest.mark.gpu_count_min_2, pytest.mark.gpu_vram_min_80g]


@pytest.fixture
def test_output_dir(request: pytest.FixtureRequest) -> Path:
    output_dir = Path(__file__).parent / "test_results" / sanitize_test_name(request.node.name)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    return output_dir


def _assert_breakable_capture_on_both_ranks(log_lines: list[str]) -> dict[str, list[str]]:
    workers: dict[str, tuple[str, str]] = {}
    captures: dict[str, list[str]] = {}
    for line in log_lines:
        pid = re.search(r"\bpid=(\d+)\b", line)
        if pid is None:
            continue
        execution = re.search(
            r"FlexTensor vLLM SHM execution data_parallel_index=(\d+) role=(creator|follower)\b", line
        )
        if execution is not None:
            rank, role = execution.groups()
            assert rank not in workers, f"Duplicate SHM execution marker for rank {rank}"
            workers[rank] = (pid.group(1), role)
        if "Captured breakable cudagraph for " in line:
            segments = re.search(r"BreakableCUDAGraphCapture\(graphs=(\d+), eager_breaks=(\d+)\)", line)
            if segments is not None and int(segments.group(1)) > 1 and int(segments.group(2)) > 0:
                captures.setdefault(pid.group(1), []).append(line)

    assert set(workers) == {"0", "1"}, f"Both DP ranks must execute requests: {workers}"
    assert len({pid for pid, _role in workers.values()}) == 2, f"DP ranks must have distinct worker PIDs: {workers}"
    assert {role for _pid, role in workers.values()} == {"creator", "follower"}, workers
    for rank, (pid, _role) in workers.items():
        assert captures.get(pid), f"No successful breakable capture with eager breaks on DP rank {rank} (pid={pid})"
    return {rank: captures[pid] for rank, (pid, _role) in workers.items()}


def test_v2_dp2_shm_breakable_capture_and_repeated_generation(test_output_dir: Path, tmp_path: Path) -> None:
    assert Version(importlib.metadata.version("vllm")) >= Version("0.25.1"), "Run in the pinned breakable vLLM image"
    assert torch.cuda.device_count() >= 2, "Breakable SHM acceptance requires two visible CUDA devices"
    # PIECEWISE forces attention eager breaks, including for single-token decode.
    # Disabling prefix caching makes every repeated request process its prompt again.
    cli_args = (
        *QWEN_ARGS,
        "--compilation-config",
        '{"cudagraph_mode":"PIECEWISE"}',
        "--no-enable-prefix-caching",
        # vLLM 0.25 defaults to immediate abort, which bypasses SHM teardown.
        "--shutdown-timeout",
        "15",
    )
    env_vars = (("VLLM_USE_BREAKABLE_CUDAGRAPH", "1"), ("VLLM_LOGGING_COLOR", "0"))
    profile_dir = create_qwen_shm_profile(
        test_output_dir, tmp_path / "profile_cache", cli_args=cli_args, env_vars=env_vars
    )
    state = TensorManagerStateHandler.load_from_file(profile_dir / "profile.json")
    assert any(state.load_strategy.values()), "Breakable SHM acceptance requires offloaded weights"

    namespace = f"vllm_dp2_breakable_{uuid.uuid4().hex}"
    manager_namespace = SharedOffloadSession(namespace).namespace_for(DEFAULT_MANAGER_NAME)
    shared_case = (
        VllmOffloadSmokeCase(
            model_name=QWEN_MODEL,
            output_dir_name="qwen2_5_7b_dp2_shm_breakable",
            cli_args=(
                *cli_args,
                "--data-parallel-size",
                "2",
                "--data-parallel-backend",
                "mp",
                "--distributed-executor-backend",
                "mp",
                "--api-server-count",
                "1",
                "--gpu-memory-utilization",
                "0.9",
            ),
        )
        .with_flextensor_offload()
        .with_env_vars(
            *env_vars,
            # Fresh root for both ranks, separate from profile-generation caches.
            ("VLLM_CACHE_ROOT", str(tmp_path / "shm_cache")),
            ("FT_PROFILE_STORAGE_DIR", str(profile_dir)),
            ("FT_PROFILE_READ_ONLY", "1"),
            ("FT_SHM_ENABLED", "1"),
            ("FT_SHM_NAMESPACE", namespace),
            ("FT_SHM_WAIT_TIMEOUT", "1800"),
        )
    )
    correctness = replace(QWEN_CORRECTNESS, request_count=2, repeat_count=3)
    _, metrics, log_lines = run_vllm_server_test(
        shared_case,
        output_dir=test_output_dir / "dp2_shm_breakable",
        correctness_check=correctness,
        server_ready_timeout=1800,
    )
    (test_output_dir / "server.log").write_text("\n".join(log_lines) + "\n")
    (test_output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    assert_namespace_removed(manager_namespace)
    assert sum("checkpoint_load=loaded" in line for line in log_lines) == 1
    assert sum("checkpoint_load=skipped" in line for line in log_lines) == 1
    assert sum("follower post-load storage identity verified" in line for line in log_lines) == 1
    assert sum("saved profile loaded path=" in line for line in log_lines) == 1
    captures = _assert_breakable_capture_on_both_ranks(log_lines)
    (test_output_dir / "breakable_captures.json").write_text(json.dumps(captures, indent=2) + "\n")
    responses = metrics["correctness"]["responses"]
    assert len(responses) == correctness.request_count * correctness.repeat_count
    assert all(response["usage"]["completion_tokens"] > 1 for response in responses), "Every request must decode"

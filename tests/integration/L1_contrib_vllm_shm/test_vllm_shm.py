# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory data-parallel coverage for FlexTensor's vLLM worker."""

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
from flextensor.state_handler import TensorManagerState, TensorManagerStateHandler
from flextensor.utils import _compute_packed_byte_layout
from tests.integration._vllm_server import (
    VllmBenchmarkConfig,
    VllmOffloadSmokeCase,
    _set_cli_arg,
    run_vllm_server_test,
)
from tests.integration._vllm_shm import (
    QWEN_ARGS,
    QWEN_CORRECTNESS,
    QWEN_MODEL,
    assert_namespace_removed,
    create_qwen_shm_profile,
)
from tests.integration._vllm_utils import sanitize_test_name
from tests.integration.L1_contrib_vllm_shm._host_memory import (
    HostMemorySampler,
    qualify_shared_host_memory,
)

# VRAM is per device; the count marker selects a multi-GPU runner.
pytestmark = [pytest.mark.gpu_count_min_2, pytest.mark.gpu_vram_min_80g]


@pytest.fixture
def test_output_dir(request: pytest.FixtureRequest) -> Path:
    output_dir = Path(__file__).parent / "test_results" / sanitize_test_name(request.node.name)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    return output_dir


def _profile_offload_backing(state: TensorManagerState) -> tuple[dict[str, int], int]:
    """Match CPU allocation-block packing, rather than reusable GPU block sizes."""
    backing: dict[str, int] = {}
    unique_weights: dict[int, int] = {}
    for group_index, labels in enumerate(state.allocation_ordered.values()):
        block_index = 0
        for label in labels:
            tensors = {t.tensor_id: t.size_bytes for t in state.load_strategy[label]}
            if not tensors:
                continue
            _, size = _compute_packed_byte_layout(tensors.values())
            backing[f"_w{group_index}_{block_index}"] = size
            unique_weights.update(tensors)
            block_index += 1
    return backing, sum(unique_weights.values())


def test_v2_dp2_reuses_profile_through_shared_memory(test_output_dir: Path, tmp_path: Path) -> None:
    """Serve concurrent DP requests with one checkpoint-backed creator and one SHM follower."""
    if Version(importlib.metadata.version("vllm")) < Version("0.17.0"):
        pytest.skip("vLLM DP=2 SHM acceptance requires vLLM >= 0.17.0")
    if torch.cuda.device_count() < 2:
        pytest.fail("DP=2 SHM acceptance requires a dedicated runner with two visible CUDA devices")

    profile_dir = create_qwen_shm_profile(test_output_dir, tmp_path / "profile_cache")
    state = TensorManagerStateHandler.load_from_file(profile_dir / "profile.json")
    expected_backing, duplicate_weight_bytes = _profile_offload_backing(state)
    assert duplicate_weight_bytes > 0, "Host-memory acceptance needs a profile with offloaded weights"

    baseline_namespace = f"vllm_dp1_{uuid.uuid4().hex}"
    baseline_manager_namespace = SharedOffloadSession(baseline_namespace).namespace_for(DEFAULT_MANAGER_NAME)
    baseline_case = (
        VllmOffloadSmokeCase(
            model_name=QWEN_MODEL,
            output_dir_name="qwen2_5_7b_dp1_shm",
            cli_args=(
                *QWEN_ARGS,
                "--data-parallel-size",
                "1",
                "--data-parallel-backend",
                "mp",
                # Keep the worker executor and API count the same across DP sizes.
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
            ("VLLM_CACHE_ROOT", str(tmp_path / "shm_baseline_cache")),
            ("FT_PROFILE_STORAGE_DIR", str(profile_dir)),
            ("FT_PROFILE_READ_ONLY", "1"),
            ("FT_SHM_ENABLED", "1"),
            ("FT_SHM_NAMESPACE", baseline_namespace),
            ("FT_SHM_WAIT_TIMEOUT", "1800"),
        )
    )

    _, baseline_metrics, _ = run_vllm_server_test(
        baseline_case,
        output_dir=test_output_dir / "dp1_shm",
        correctness_check=replace(QWEN_CORRECTNESS, request_count=2),
        server_ready_timeout=1800,
        host_memory_sampler_factory=lambda pid: HostMemorySampler(pid, baseline_manager_namespace, expected_backing, 1),
    )
    assert_namespace_removed(baseline_manager_namespace)

    namespace = f"vllm_dp2_{uuid.uuid4().hex}"
    manager_namespace = SharedOffloadSession(namespace).namespace_for(DEFAULT_MANAGER_NAME)
    shared_case = replace(
        baseline_case,
        output_dir_name="qwen2_5_7b_dp2_shm",
        cli_args=_set_cli_arg(baseline_case.cli_args, "--data-parallel-size", "2"),
    ).with_env_vars(
        ("FT_SHM_NAMESPACE", namespace),
        # Both ranks must compile the SHM model instead of reusing either
        # single-worker run's AOT graph on rank 0.
        ("VLLM_CACHE_ROOT", str(tmp_path / "shm_cache")),
    )
    _, metrics, log_lines = run_vllm_server_test(
        shared_case,
        output_dir=test_output_dir / "dp2_shm",
        correctness_check=replace(QWEN_CORRECTNESS, request_count=2),
        benchmark_config=VllmBenchmarkConfig(
            request_count=2,
            input_tokens=32,
            output_tokens=16,
            max_concurrency=2,
        ),
        server_ready_timeout=1800,
        host_memory_sampler_factory=lambda pid: HostMemorySampler(
            pid, manager_namespace, expected_backing, 2, baseline=baseline_metrics["host_memory"]
        ),
    )

    assert_namespace_removed(manager_namespace)
    memory_report = qualify_shared_host_memory(
        baseline_metrics["host_memory"], metrics["host_memory"], duplicate_weight_bytes
    )
    memory_report["expected_offload_backing_bytes"] = expected_backing
    memory_report["baseline_artifact"] = "dp1_shm/host_memory.json"
    memory_report["shared_artifact"] = "dp2_shm/host_memory.json"
    (test_output_dir / "host_memory_acceptance.json").write_text(json.dumps(memory_report, indent=2) + "\n")
    assert not memory_report["failures"], memory_report["failures"]
    assert len(metrics["correctness"]["responses"]) == 2
    assert metrics["usage"].get("completion_tokens", 0) > 0
    assert metrics["benchmark"]["completed"] >= 2
    shared_init_lines = [line for line in log_lines if "Shared offload initialized role=" in line]
    assert len(shared_init_lines) == 2
    assert sum("role=creator" in line for line in shared_init_lines) == 1
    assert sum("role=follower" in line for line in shared_init_lines) == 1
    shared_namespaces = {
        match.group(1) for line in shared_init_lines if (match := re.search(r"\bnamespace=([^ ]+)", line)) is not None
    }
    assert shared_namespaces == {manager_namespace}
    assert sum("saved profile loaded path=" in line for line in log_lines) == 1
    assert sum("checkpoint_load=loaded" in line for line in log_lines) == 1
    assert sum("checkpoint_load=skipped" in line for line in log_lines) == 1
    assert sum("follower post-load storage identity verified" in line for line in log_lines) == 1
    assert sum("FlexTensor vLLM integration v2 state takeover complete" in line for line in log_lines) == 2
    execution_lines = [line for line in log_lines if "FlexTensor vLLM SHM execution" in line]
    assert len(execution_lines) == 2
    executed_dp_indexes = {
        match.group(1)
        for line in execution_lines
        if (match := re.search(r"\bdata_parallel_index=(\d+)", line)) is not None
    }
    assert executed_dp_indexes == {"0", "1"}
    assert {role for role in ("creator", "follower") if any(f"role={role}" in line for line in execution_lines)} == {
        "creator",
        "follower",
    }

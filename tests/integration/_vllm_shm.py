# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile setup and teardown checks shared by the vLLM SHM GPU suites."""

from pathlib import Path

import torch

from tests.integration._vllm_server import VllmCorrectnessCheck, VllmOffloadSmokeCase, run_vllm_server_test

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN_CORRECTNESS = VllmCorrectnessCheck(expected_substrings=("Paris",))
QWEN_ARGS = ("--max-model-len", "256", "--max-num-seqs", "1")


def assert_namespace_removed(namespace: str) -> None:
    shm_dir = Path("/dev/shm")  # noqa: S108 - read-only inspection of the Linux POSIX SHM namespace.
    assert shm_dir.is_dir(), "SHM teardown checks require Linux /dev/shm"
    leftovers = [str(path) for path in shm_dir.iterdir() if namespace in path.name]
    assert not leftovers, f"SHM artifacts survived server shutdown: {leftovers}"


def create_qwen_shm_profile(
    output_dir: Path,
    cache_dir: Path,
    *,
    cli_args: tuple[str, ...] = QWEN_ARGS,
    env_vars: tuple[tuple[str, str], ...] = (),
) -> Path:
    profile_dir = output_dir / "profile"
    profile_dir.mkdir()
    # Allow the roughly 2 GiB resident weights plus the >2 GiB allocation-block
    # working set observed on A100, while keeping most 7B weights offloaded.
    profile_gpu_fraction = (5 * 1024**3) / torch.cuda.get_device_properties(0).total_memory
    profile_case = (
        VllmOffloadSmokeCase(
            model_name=QWEN_MODEL,
            output_dir_name="qwen2_5_7b_dp2_shm_profile",
            cli_args=(*cli_args, "--gpu-memory-utilization", "0.9"),
        )
        .with_flextensor_offload()
        .with_env_vars(
            ("VLLM_CACHE_ROOT", str(cache_dir)),
            ("FT_MAX_GPU_MEM_FRACTION", str(profile_gpu_fraction)),
            ("FT_PROFILE_STORAGE_DIR", str(profile_dir)),
            ("FT_PROFILING_ITERS", "1"),
            *env_vars,
        )
    )
    _, _, profile_logs = run_vllm_server_test(
        profile_case,
        output_dir=output_dir / "profile_run",
        correctness_check=QWEN_CORRECTNESS,
        server_ready_timeout=1800,
    )
    assert (profile_dir / "profile.json").is_file()
    assert any("refreshed profile saved path=" in line for line in profile_logs)
    return profile_dir

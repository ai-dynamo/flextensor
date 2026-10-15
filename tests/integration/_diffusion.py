# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Wan parity/memory runner. Each mode owns a fresh CUDA process."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess  # noqa: S404 - isolated CUDA processes are part of the test
import sys
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tests.integration._diffusion_metrics import MemoryRecorder, assert_memory_limits, assert_video_parity

GIB = 1024**3
ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
PROMPTS = ("A small blue cube rotating on a plain background", "A red ball bouncing on a wooden floor")
NUM_FRAMES = 17
HEIGHT = WIDTH = 128
STEPS = 12
SEED = 42
DISCOVERY_ITERS = 1
PROFILING_ITERS = 2


def compile_block(module):
    return torch.compile(module, fullgraph=True, backend="inductor")


def _track_pipeline(memory, pipe, manager):
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            for tensor in (*component.parameters(), *component.buffers()):
                memory.storage.add(tensor, "weights")
    if manager is not None:
        tensor_manager = manager._tensor_manager
        for tensor in (*tensor_manager.tensors_map.values(), *tensor_manager._replan_source_data.values()):
            memory.storage.add(tensor, "weights")
    memory.sample()


def _compile_blocks(pipe):
    from flextensor.compiled_offload import bump_dynamo_limits_for_compiled_offload

    blocks = pipe.transformer.blocks
    bump_dynamo_limits_for_compiled_offload(len(blocks))
    for i, block in enumerate(blocks):
        blocks[i] = compile_block(block)


def run_worker(args):  # noqa: C901 - one linear model lifecycle, shared by all modes
    # Diffusers probes CUDA on import; configure the runtime before importing it.
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    import diffusers
    from diffusers import WanPipeline

    import flextensor as ft
    from flextensor.offload_manager import OffloadModelProxy, OffloadPhase

    # Keep copies/kernels asynchronous: CUDA_LAUNCH_BLOCKING would hide offload races.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available(), "diffusion integration requires CUDA"
    memory = MemoryRecorder()
    manager = None
    seen_phases = set()
    calls = 0
    results = {}
    pipe = None
    args.output.mkdir(parents=True, exist_ok=True)
    proxy_call = OffloadModelProxy.__call__

    def record_offload_call(proxy, *call_args, **kwargs):
        nonlocal calls
        calls += 1
        seen_phases.add(manager.phase.value)
        return proxy_call(proxy, *call_args, **kwargs)

    def record_baseline_call(_module, _inputs):
        nonlocal calls
        calls += 1

    def generate(prompt, artifact=None):
        final_latents = None

        def step_callback(pipeline, step, timestep, values):
            nonlocal final_latents
            if manager is not None:
                seen_phases.add(manager.phase.value)
            _track_pipeline(memory, pipeline, manager)
            if step == STEPS - 1:
                final_latents = values["latents"].detach().float().cpu().numpy().copy()
            return values

        frames = pipe(
            prompt=prompt,
            negative_prompt="",
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=STEPS,
            guidance_scale=5.0,
            max_sequence_length=64,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            output_type="np",
            callback_on_step_end=step_callback,
        ).frames[0]
        assert frames.shape == (NUM_FRAMES, HEIGHT, WIDTH, 3), frames.shape
        assert final_latents is not None and np.isfinite(final_latents).all(), "invalid final latents"
        assert_video_parity(frames, frames, min_psnr=48)
        if artifact is not None:
            np.savez(args.output / f"{artifact}.npz", frames=frames, latents=final_latents)

    try:
        with memory.allocations(), patch.object(OffloadModelProxy, "__call__", record_offload_call):
            with memory.phase("startup"):
                pipe = WanPipeline.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16)
                _track_pipeline(memory, pipe, manager)
                weight_bytes = sum(t.numel() * t.element_size() for t in pipe.transformer.parameters())
                results["transformer_weight_bytes"] = weight_bytes
                results["model_weight_bytes"] = memory.sample()["cpu_total_bytes"]
                if args.mode.startswith("baseline"):
                    pipe.transformer.register_forward_pre_hook(record_baseline_call)
                for name, component in pipe.components.items():
                    if isinstance(component, torch.nn.Module) and (
                        args.mode.startswith("baseline") or name != "transformer"
                    ):
                        component.to("cuda")
                if args.mode == "baseline-compiled":
                    _compile_blocks(pipe)
                if not args.mode.startswith("baseline"):
                    config = ft.OffloadConfig(
                        include_patterns=["blocks.*"],
                        discovery_iters=DISCOVERY_ITERS,
                        skip_discovery=args.mode == "compile-fn",
                        profiling_iters=PROFILING_ITERS,
                        min_blocks=2,
                        max_gpu_mem_fraction=0.6 * weight_bytes / torch.cuda.get_device_properties(0).total_memory,
                        external_compile=args.mode == "external-compile",
                    )
                    if args.mode == "profile":
                        pipe.transformer = ft.offload_from_profile(
                            pipe.transformer, str(args.profile_dir / "transformer"), config=config, name="transformer"
                        )
                    else:
                        pipe.transformer = ft.offload(
                            pipe.transformer,
                            config=config,
                            name="transformer",
                            compile_fn=compile_block if args.mode == "compile-fn" else None,
                        )
                    manager = ft.get_offload_manager("transformer")
                    seen_phases.add(manager.phase.value)
                    if args.mode == "profile":
                        assert manager.phase == OffloadPhase.INFERENCE, "profile unexpectedly needs warmup"

                for attempt in range(4):
                    generate(PROMPTS[0], f"transition-{attempt}")
                    if manager is None or manager.phase == OffloadPhase.INFERENCE:
                        break
                assert manager is None or manager.phase == OffloadPhase.INFERENCE, "warmup incomplete"
                if args.mode == "eager":
                    manager.save_profile(str(args.profile_dir / "transformer"))

                if args.mode == "external-compile":
                    _compile_blocks(pipe)
                    assert manager.request_strategy_replan() > 0, "replan was not armed"
                    for attempt in range(4):
                        generate(PROMPTS[0], f"replan-{attempt}")
                        if manager.strategy_replan_status.state == "done":
                            break
                    assert manager.strategy_replan_status.state == "done", manager.strategy_replan_status
                    assert manager.strategy_replan_status.applied is True, manager.strategy_replan_status
                # Warm both prompts before measuring A -> B -> A to exclude lazy caches.
                generate(PROMPTS[1])
                gc.collect()
                _track_pipeline(memory, pipe, manager)

            results["ft_gpu"] = asdict(manager.get_gpu_memory_usage()) if manager is not None else None
            results["live"] = []
            calls = 0
            with memory.phase("inference"):
                for index, prompt in enumerate((PROMPTS[0], PROMPTS[1], PROMPTS[0])):
                    generate(prompt, str(index))
                    gc.collect()
                    torch.cuda.synchronize()
                    results["live"].append({**memory.sample(), "gpu_allocated_bytes": torch.cuda.memory_allocated()})
            assert calls == 3 * STEPS * 2, f"missing denoising or classifier-free guidance calls: {calls}"
            if args.mode == "profile":
                assert seen_phases == {OffloadPhase.INFERENCE.value}
            elif manager is not None:
                expected = {OffloadPhase.PROFILING.value, OffloadPhase.INFERENCE.value}
                if args.mode != "compile-fn":
                    expected.add(OffloadPhase.DISCOVERY.value)
                assert expected <= seen_phases
            results.update(
                memory=memory.report,
                phases=sorted(seen_phases),
                forward_calls=calls,
                mode=args.mode,
                model_id=MODEL_ID,
                revision=REVISION,
                torch_version=torch.__version__,
                diffusers_version=diffusers.__version__,
                gpu=torch.cuda.get_device_name(),
                seed=SEED,
            )
    finally:
        if manager is not None:
            manager.release()
        results["memory"] = memory.report
        (args.output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


def run_mode(directory: Path, *, mode: str, profile_dir: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    env["PYTHONUNBUFFERED"] = "1"
    # Pin the tested path; unrelated shell tuning must not silently change coverage.
    for key in list(env):
        if key.startswith("FT_"):
            del env[key]
    command = [
        sys.executable,
        "-m",
        "tests.integration._diffusion",
        "--mode",
        mode,
        "--output",
        str(directory),
        "--profile-dir",
        str(profile_dir),
    ]
    log = directory / "run.log"
    print(f"Diffusion {mode}: {log}", flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(  # noqa: S603 - controlled test runner
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def terminate(_signum, _frame):
            raise SystemExit(128 + signal.SIGTERM)

        previous_handler = signal.signal(signal.SIGTERM, terminate)
        try:
            returncode = process.wait(timeout=int(os.environ.get("DIFFUSION_TIMEOUT", "3600")))
        finally:
            # Compiler children can outlive the worker, including after a clean exit.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            signal.signal(signal.SIGTERM, previous_handler)
    assert returncode == 0, f"{mode} failed; {log}\n" + "\n".join(log.read_text().splitlines()[-65:])


def check_run(directory: Path, reference_dir: Path, *, offloaded: bool) -> None:
    report = json.loads((directory / "metrics.json").read_text())
    reference = json.loads((reference_dir / "metrics.json").read_text())
    for key in (
        "model_id",
        "revision",
        "torch_version",
        "diffusers_version",
        "gpu",
        "seed",
        "transformer_weight_bytes",
    ):
        assert report[key] == reference[key], f"baseline mismatch: {key}"
    for index in range(3):
        with np.load(reference_dir / f"{index}.npz") as baseline, np.load(directory / f"{index}.npz") as candidate:
            minimum = assert_video_parity(baseline["frames"], candidate["frames"], min_psnr=48)
            print(f"{directory.name} request {index}: minimum frame PSNR = {minimum:.3f} dB")
            np.testing.assert_allclose(candidate["latents"], baseline["latents"], rtol=1e-3, atol=1e-3)
    with np.load(directory / "0.npz") as first, np.load(directory / "2.npz") as repeated:
        assert_video_parity(first["frames"], repeated["frames"], min_psnr=48)
        np.testing.assert_allclose(first["latents"], repeated["latents"], rtol=1e-3, atol=1e-3)
    assert (directory / "transition-0.npz").exists(), "missing first-request output"
    if report["mode"] == "external-compile":
        assert (directory / "replan-0.npz").exists(), "missing replan output"
    for path in (*directory.glob("transition-*.npz"), *directory.glob("replan-*.npz")):
        baseline_path = reference_dir / "0.npz"
        if path.name == "transition-0.npz" and report["mode"] in {"baseline-compiled", "compile-fn"}:
            baseline_path = reference_dir / "transition-0.npz"
        elif path.name.startswith("transition-") and report["mode"] == "external-compile":
            baseline_path = reference_dir.parent / "baseline" / "0.npz"
        with np.load(path) as warmup, np.load(baseline_path) as baseline:
            print(f"Checking lifecycle output: {path.name}")
            assert_video_parity(baseline["frames"], warmup["frames"], min_psnr=48)
            np.testing.assert_allclose(warmup["latents"], baseline["latents"], rtol=1e-3, atol=1e-3)

    # Initial conservative ceilings, fixed across runs (never learned from the candidate).
    gpu_limit, startup_ram, inference_ram = 22, 48, 32
    limits = {
        phase: {
            "gpu_allocated_peak_bytes": gpu_limit * GIB,
            "gpu_reserved_peak_bytes": (gpu_limit + 1) * GIB,
            "rss_peak_sampled_bytes": ram * GIB,
            "cpu_total_peak_bytes": 2.25 * reference["model_weight_bytes"],
        }
        for phase, ram in (("startup", startup_ram), ("inference", inference_ram))
    }
    limits["startup"]["rss_lifetime_peak_bytes"] = startup_ram * GIB
    limits["inference"]["cpu_total_peak_bytes"] = 1.1 * report["transformer_weight_bytes"] + 64 * 1024**2
    assert_memory_limits(report["memory"], limits)
    first, last = report["live"][0], report["live"][-1]
    for metric in ("gpu_allocated_bytes", "cpu_total_bytes", "gpu_total_bytes"):
        assert last[metric] <= first[metric] + 64 * 1024**2, f"live allocation growth: {metric}: {first} -> {last}"
    if offloaded:
        usage = report["ft_gpu"]
        assert usage["blocks_bytes"] > 0, "no transfer blocks allocated"
        assert usage["total_bytes"] <= 0.65 * report["transformer_weight_bytes"], "weight budget exceeded"
        assert report["memory"]["inference"]["cpu_ft_buffers_peak_bytes"] > 0
        assert (
            report["memory"]["inference"]["gpu_allocated_peak_bytes"]
            < reference["memory"]["inference"]["gpu_allocated_peak_bytes"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("baseline", "baseline-compiled", "eager", "profile", "compile-fn", "external-compile"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    run_worker(parser.parse_args())

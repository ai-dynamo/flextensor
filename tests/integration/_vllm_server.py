# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server orchestration helpers for vLLM integration tests."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess  # noqa: S404 - subprocess needed for test utilities
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import requests

from tests.integration._vllm_utils import (
    assert_vllm_cuda_platform_logged,
    collect_vllm_backend_evidence,
    query_runtime_gpu_backend_metadata,
    validate_latest_memory_transfer_stats_for_current_bus,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

DEFAULT_VLLM_PORT = 8000
FLEXTENSOR_OFFLOAD_WORKER_CLS = "flextensor.contrib.vllm.worker.FlexTensorOffloadWorker"
FLEXTENSOR_SNAPSHOT_WORKER_CLS = "flextensor.contrib.vllm.snapshot.FlexTensorSnapshotWorker"
FLEXTENSOR_WORKER_CLASSES = {
    FLEXTENSOR_OFFLOAD_WORKER_CLS,
    FLEXTENSOR_SNAPSHOT_WORKER_CLS,
}


@dataclass
class MemoryProfilingMetrics:
    """Memory profiling metrics extracted from vLLM server logs."""

    initial_free_memory_gib: float | None = None
    requested_memory_util: float | None = None
    requested_memory_gib: float | None = None
    free_memory_after_profiling_total_gib: float | None = None
    free_memory_after_profiling_within_requested_gib: float | None = None
    total_non_kv_cache_memory_gib: float | None = None
    torch_peak_memory_increase_gib: float | None = None
    non_torch_forward_increase_memory_gib: float | None = None
    weights_memory_gib: float | None = None
    available_kv_cache_memory_gib: float | None = None
    whole_model_budget_bytes: int | None = None
    managed_gpu_resident_bytes: int | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "initial_free_memory_gib": self.initial_free_memory_gib,
            "requested_memory_util": self.requested_memory_util,
            "requested_memory_gib": self.requested_memory_gib,
            "free_memory_after_profiling_total_gib": self.free_memory_after_profiling_total_gib,
            "free_memory_after_profiling_within_requested_gib": self.free_memory_after_profiling_within_requested_gib,
            "total_non_kv_cache_memory_gib": self.total_non_kv_cache_memory_gib,
            "torch_peak_memory_increase_gib": self.torch_peak_memory_increase_gib,
            "non_torch_forward_increase_memory_gib": self.non_torch_forward_increase_memory_gib,
            "weights_memory_gib": self.weights_memory_gib,
            "available_kv_cache_memory_gib": self.available_kv_cache_memory_gib,
            "whole_model_budget_bytes": self.whole_model_budget_bytes,
            "managed_gpu_resident_bytes": self.managed_gpu_resident_bytes,
        }


@dataclass(frozen=True)
class VllmCorrectnessCheck:
    """Deterministic chat request used to verify the served model returns expected content."""

    messages: tuple[dict[str, str], ...] = field(
        default_factory=lambda: ({"role": "user", "content": "What is the capital of France?"},)
    )
    expected_substrings: tuple[str, ...] = ()
    max_tokens: int = 20
    timeout: int = 60
    temperature: float = 0.0
    seed: int | None = 42
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    case_sensitive: bool = False
    request_count: int = 1
    repeat_count: int = 1  # Sequential batches, each with request_count concurrent requests.


@dataclass(frozen=True)
class VllmBenchmarkConfig:
    """Opt-in vLLM serving benchmark probe for performance evidence."""

    request_count: int = 5
    input_tokens: int = 128
    output_tokens: int = 16
    random_range_ratio: float = 0.0
    max_concurrency: int = 1
    seed: int = 42
    temperature: float = 0.0
    timeout: int = 300
    backend: str = "openai-chat"
    endpoint: str = "/v1/chat/completions"
    result_filename: str = "vllm_bench_serve.json"
    save_detailed: bool = True
    required_positive_metrics: tuple[str, ...] = (
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
    )
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class VllmOffloadSmokeCase:
    """Model-specific vLLM smoke-test launch configuration."""

    model_name: str
    output_dir_name: str
    cli_args: tuple[str, ...] = ()
    extra_env_vars: tuple[tuple[str, str], ...] = ()

    def with_flextensor_offload(self, worker_cls: str = FLEXTENSOR_OFFLOAD_WORKER_CLS) -> VllmOffloadSmokeCase:
        """Return this launch case with explicit FlexTensor offload env/worker args."""
        return replace(
            self,
            cli_args=_set_cli_arg(self.cli_args, "--worker-cls", worker_cls),
            extra_env_vars=_merge_env_vars(
                self.extra_env_vars,
                (
                    ("FT_ENABLED", "1"),
                    ("FT_ENABLE_DIAGNOSTICS", "1"),
                    ("FT_ENABLE_INSTRUMENTATION", "1"),
                ),
            ),
        )

    def with_env_vars(self, *env_vars: tuple[str, str]) -> VllmOffloadSmokeCase:
        """Return this launch case with additional environment variables."""
        return replace(self, extra_env_vars=_merge_env_vars(self.extra_env_vars, env_vars))


def _merge_env_vars(
    base_env_vars: tuple[tuple[str, str], ...],
    override_env_vars: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    merged = dict(base_env_vars)
    merged.update(dict(override_env_vars))
    return tuple(merged.items())


def _set_cli_arg(cli_args: tuple[str, ...], option: str, value: str) -> tuple[str, ...]:
    args = list(cli_args)
    for index, arg in enumerate(args):
        if arg == option:
            if index + 1 >= len(args):
                raise AssertionError(f"{option} must be followed by a value")
            args[index + 1] = value
            return tuple(args)
        if arg.startswith(f"{option}="):
            args[index] = f"{option}={value}"
            return tuple(args)
    args.extend((option, value))
    return tuple(args)


def resolve_hf_reasoning_parser(model_name: str, filename: str) -> str:
    """Download a model-provided vLLM reasoning parser and return its local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_name, filename=filename)


def with_hf_reasoning_parser(
    case: VllmOffloadSmokeCase,
    *,
    filename: str,
    parser_name: str,
) -> VllmOffloadSmokeCase:
    parser_path = resolve_hf_reasoning_parser(case.model_name, filename)
    return replace(
        case,
        cli_args=(
            *case.cli_args,
            "--reasoning-parser-plugin",
            str(parser_path),
            "--reasoning-parser",
            parser_name,
        ),
    )


def _cli_arg_value(cli_args: tuple[str, ...], option: str) -> str | None:
    for index, arg in enumerate(cli_args):
        if arg == option:
            if index + 1 >= len(cli_args):
                raise AssertionError(f"{option} must be followed by a value")
            return cli_args[index + 1]
        if arg.startswith(f"{option}="):
            return arg.split("=", 1)[1]
    return None


def case_port(case: VllmOffloadSmokeCase) -> int:
    """Return the vLLM server port declared in ``case.cli_args`` or vLLM's default."""
    port = _cli_arg_value(case.cli_args, "--port")
    return DEFAULT_VLLM_PORT if port is None else int(port)


def _is_truthy_env_value(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def case_uses_flextensor_offload(case: VllmOffloadSmokeCase) -> bool:
    """Return whether the launch case explicitly enables FlexTensor's vLLM worker."""
    env_enabled = _is_truthy_env_value(dict(case.extra_env_vars).get("FT_ENABLED"))
    worker_cls = _cli_arg_value(case.cli_args, "--worker-cls")
    return env_enabled and worker_cls in FLEXTENSOR_WORKER_CLASSES


def parse_memory_profiling_logs(log_lines: list[str]) -> MemoryProfilingMetrics:
    """Parse vLLM server logs to extract memory profiling metrics."""
    metrics = MemoryProfilingMetrics()
    log_text = "\n".join(log_lines)

    initial_memory_pattern = (
        r"Initial free memory:\s*([\d.]+)\s*GiB;\s*"
        r"Requested memory:\s*([\d.]+)\s*\(util\),\s*([\d.]+)\s*GiB"
    )
    match = re.search(initial_memory_pattern, log_text)
    if match:
        metrics.initial_free_memory_gib = float(match.group(1))
        metrics.requested_memory_util = float(match.group(2))
        metrics.requested_memory_gib = float(match.group(3))

    free_after_profiling_pattern = (
        r"Free memory after profiling:\s*([\d.]+)\s*GiB\s*\(total\),\s*"
        r"([\d.]+)\s*GiB\s*\(within requested\)"
    )
    match = re.search(free_after_profiling_pattern, log_text)
    if match:
        metrics.free_memory_after_profiling_total_gib = float(match.group(1))
        metrics.free_memory_after_profiling_within_requested_gib = float(match.group(2))

    memory_breakdown_pattern = (
        r"Total non KV cache memory:\s*([\d.]+)\s*GiB;\s*"
        r"torch peak memory increase:\s*([\d.]+)\s*GiB;\s*"
        r"non-torch forward increase memory:\s*([\d.]+)\s*GiB;\s*"
        r"weights memory:\s*([\d.]+)\s*GiB"
    )
    match = re.search(memory_breakdown_pattern, log_text)
    if match:
        metrics.total_non_kv_cache_memory_gib = float(match.group(1))
        metrics.torch_peak_memory_increase_gib = float(match.group(2))
        metrics.non_torch_forward_increase_memory_gib = float(match.group(3))
        metrics.weights_memory_gib = float(match.group(4))

    kv_cache_pattern = r"Available KV cache memory:\s*([\d.]+)\s*GiB"
    match = re.search(kv_cache_pattern, log_text)
    if match:
        metrics.available_kv_cache_memory_gib = float(match.group(1))

    budget_pattern = (
        r"FlexTensor v2 GPU budget resolved: "
        r"whole_model_budget_bytes=(\d+) managed_gpu_resident_bytes=(\d+)"
    )
    match = re.search(budget_pattern, log_text)
    if match:
        metrics.whole_model_budget_bytes = int(match.group(1))
        metrics.managed_gpu_resident_bytes = int(match.group(2))

    return metrics


def run_vllm_server_test(  # noqa: C901 - Keep server and optional sampler lifetimes together.
    case: VllmOffloadSmokeCase,
    output_dir: Path,
    chat_max_tokens: int = 20,
    chat_request_timeout: int = 60,
    correctness_check: VllmCorrectnessCheck | None = None,
    benchmark_config: VllmBenchmarkConfig | None = None,
    server_ready_timeout: int = 600,
    host_memory_sampler_factory: Callable[[int], Any] | None = None,
) -> tuple[MemoryProfilingMetrics, dict, list[str]]:
    """Run a single vLLM server test and return metrics.

    An optional factory receives the server PID and returns a running sampler
    with phase(), idle_samples(), and finish(output_path) methods.
    """
    process = None
    host_memory = None
    log_lines: list[str] = []
    offload_enabled = case_uses_flextensor_offload(case)
    worker_cls = _cli_arg_value(case.cli_args, "--worker-cls")
    offload_status = "with offloading" if offload_enabled else "without offloading (baseline)"
    result_metrics: dict = {
        "model_name": case.model_name,
        "offload_enabled": offload_enabled,
    }
    port = case_port(case)
    correctness = correctness_check or VllmCorrectnessCheck(max_tokens=chat_max_tokens, timeout=chat_request_timeout)

    success = False
    server_case = case
    try:
        instrumentation_dir = output_dir / "instrumentation"
        instrumentation_dir.mkdir(parents=True, exist_ok=True)

        default_env_vars = (
            ("VLLM_LOGGING_LEVEL", "DEBUG"),
            ("VLLM_NO_USAGE_STATS", "1"),
        )
        offload_output_env_vars = (
            (("FT_INSTRUMENTATION_OUTPUT_DIR", str(instrumentation_dir)),) if offload_enabled else ()
        )
        server_case = replace(
            case,
            extra_env_vars=_merge_env_vars(default_env_vars + offload_output_env_vars, case.extra_env_vars),
        )

        process, log_lines = start_vllm_server(server_case)
        if host_memory_sampler_factory is not None:
            host_memory = host_memory_sampler_factory(process.pid)
        ready = wait_for_server(port=port, timeout=server_ready_timeout, process=process)

        assert ready, f"Server failed to start within timeout ({offload_status})"

        start_time = time.time()
        chat_request_kwargs: dict[str, Any] = {
            "messages": list(correctness.messages),
            "model": case.model_name,
            "port": port,
            "max_tokens": correctness.max_tokens,
            "timeout": correctness.timeout,
            "temperature": correctness.temperature,
            "seed": correctness.seed,
        }
        if correctness.chat_template_kwargs:
            chat_request_kwargs["chat_template_kwargs"] = correctness.chat_template_kwargs
        if host_memory is not None:
            host_memory.phase("warmup")
        chat_responses = _run_correctness_requests(correctness, chat_request_kwargs)
        request_latency = time.time() - start_time

        chat_response = chat_responses[0]
        response_content = _validated_chat_response_content(chat_response)

        print(f"{offload_status} - Chat response: {response_content}")

        models_url = f"http://localhost:{port}/v1/models"
        models_response = requests.get(models_url, timeout=10)
        models_response.raise_for_status()

        models_data = models_response.json()
        assert "data" in models_data, "No data in models response"
        assert len(models_data["data"]) > 0, "No models returned"

        model_ids = [model["id"] for model in models_data["data"]]
        print(f"{offload_status} - Available models: {model_ids}")
        assert case.model_name in model_ids, f"Model {case.model_name} not found in {model_ids}"

        usage = chat_response.get("usage", {})
        correctness_metrics = {
            "expected_substrings": list(correctness.expected_substrings),
            "request_latency_seconds": request_latency,
            "response_content": response_content,
            "responses": chat_responses,
            "usage": usage,
        }
        if correctness.chat_template_kwargs:
            correctness_metrics["chat_template_kwargs"] = dict(correctness.chat_template_kwargs)
        result_metrics.update({
            "request_latency_seconds": request_latency,
            "usage": usage,
            "response_content": response_content,
            "correctness": correctness_metrics,
        })
        success = True

        if host_memory is not None:
            host_memory.idle_samples("warm")
            host_memory.phase("inference")
            probe_start = time.monotonic()
            probe_requests = 0
            # Keep correctness-checked requests in flight across several memory
            # samples. This excludes benchmark-client startup from the window.
            while time.monotonic() - probe_start < 5.0:
                probe_requests += len(_run_correctness_requests(correctness, chat_request_kwargs))
            host_memory.idle_samples("post_inference")
            result_metrics["host_memory_probe"] = {
                "completed_requests": probe_requests,
                "concurrency": correctness.request_count,
            }
            host_memory.phase("benchmark")

        if benchmark_config is not None:
            benchmark_metrics = run_vllm_benchmark(
                model=case.model_name,
                port=port,
                output_dir=output_dir / "benchmark",
                config=benchmark_config,
            )
            result_metrics["benchmark"] = benchmark_metrics

    finally:
        if process is not None:
            try:
                if host_memory is not None:
                    result_metrics["host_memory"] = host_memory.finish(output_dir / "host_memory.json")
            finally:
                stop_vllm_server(process)
            time.sleep(1)

            if success:
                assert_vllm_cuda_platform_logged(log_lines)

                log_text = "\n".join(log_lines)
                offload_config_found = "FlexTensor offloading enabled with config" in log_text
                offload_applied_found = "FlexTensor offloading applied" in log_text
                v2_takeover_found = "FlexTensor vLLM integration v2 state takeover complete" in log_text

                worker_selection = dict(server_case.extra_env_vars).get("FT_VLLM_USE_V2_WORKER", "1")
                if offload_enabled and worker_cls == FLEXTENSOR_OFFLOAD_WORKER_CLS:
                    if worker_selection == "0":
                        assert offload_config_found, "FlexTensor legacy offloading config not found"
                        assert offload_applied_found, "FlexTensor legacy offloading applied message not found"
                        assert not v2_takeover_found, "Unexpected v2 state takeover for legacy worker"
                    else:
                        assert v2_takeover_found, "FlexTensor v2 state takeover message not found"
                        assert not offload_config_found, "Unexpected legacy config marker for v2 worker"
                        assert not offload_applied_found, "Unexpected legacy applied marker for v2 worker"
                elif offload_enabled and worker_cls == FLEXTENSOR_SNAPSHOT_WORKER_CLS:
                    assert offload_config_found, "FlexTensor offloading config not found in logs"
                    assert offload_applied_found, "FlexTensor offloading applied message not found"
                else:
                    assert not offload_config_found, "Unexpected offloading config in baseline"
                    assert not offload_applied_found, "Unexpected offloading applied in baseline"
                    assert not v2_takeover_found, "Unexpected v2 state takeover in baseline"

    memory_metrics = parse_memory_profiling_logs(log_lines)

    print(f"\n{offload_status} - Memory profiling metrics from vLLM server:")
    print(f"  Initial free memory: {memory_metrics.initial_free_memory_gib} GiB")
    print(f"  Weights memory: {memory_metrics.weights_memory_gib} GiB")
    print(f"  Available KV cache memory: {memory_metrics.available_kv_cache_memory_gib} GiB")
    print(f"  Total non KV cache memory: {memory_metrics.total_non_kv_cache_memory_gib} GiB")

    result_metrics["memory_profiling"] = memory_metrics.to_dict()
    result_metrics["backend_evidence"] = collect_vllm_backend_evidence(
        log_lines,
        model_name=case.model_name,
        cli_args=list(server_case.cli_args),
        extra_env_vars=dict(server_case.extra_env_vars),
        runtime_gpu=query_runtime_gpu_backend_metadata(),
    )

    if offload_enabled and success and worker_cls == FLEXTENSOR_SNAPSHOT_WORKER_CLS:
        transfer_validation = validate_latest_memory_transfer_stats_for_current_bus(instrumentation_dir)
        print(f"\n{offload_status} - Memory transfer validation:")
        print(
            "  Bus: "
            f"PCIe Gen{transfer_validation['pcie_link_gen']} x{transfer_validation['pcie_link_width']} "
            f"({transfer_validation['pci_bus_id']}), CPU affinity={transfer_validation['cpu_affinity']}, "
            f"NUMA affinity={transfer_validation['numa_affinity']}"
        )
        print(
            "  Observed bandwidth: "
            f"{transfer_validation['observed_bandwidth_gbps']:.2f} GB/s "
            f"(expected {transfer_validation['min_expected_bandwidth_gbps']:.2f}-"
            f"{transfer_validation['max_expected_bandwidth_gbps']:.2f} GB/s)"
        )
        result_metrics["memory_transfer_validation"] = transfer_validation

    return memory_metrics, result_metrics, log_lines


def _run_correctness_requests(
    correctness: VllmCorrectnessCheck,
    request_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    assert correctness.repeat_count > 0, "repeat_count must be positive"
    responses = []
    for _ in range(correctness.repeat_count):
        if correctness.request_count == 1:
            batch = [make_chat_request(**request_kwargs)]
        else:
            with ThreadPoolExecutor(max_workers=correctness.request_count) as pool:
                pending = [pool.submit(make_chat_request, **request_kwargs) for _ in range(correctness.request_count)]
                batch = [request.result() for request in pending]
        for response in batch:
            _assert_correctness_content(_validated_chat_response_content(response), correctness)
            completion_tokens = response.get("usage", {}).get("completion_tokens")
            assert isinstance(completion_tokens, int) and completion_tokens > 0, (
                f"completion_tokens must be positive, got {completion_tokens!r}"
            )
        responses.extend(batch)
    return responses


def _validated_chat_response_content(chat_response: dict[str, Any]) -> str:
    assert "choices" in chat_response, "No choices in chat response"
    assert len(chat_response["choices"]) > 0, "Empty choices in chat response"
    assert "message" in chat_response["choices"][0], "No message in chat choice"
    message = chat_response["choices"][0]["message"]
    content = message.get("content")
    assert isinstance(content, str) and content, f"No text content in chat response: {chat_response!r}"
    return content


def _assert_correctness_content(response_content: str, correctness: VllmCorrectnessCheck) -> None:
    if correctness.case_sensitive:
        haystack = response_content
        missing = [expected for expected in correctness.expected_substrings if expected not in haystack]
    else:
        haystack = response_content.lower()
        missing = [expected for expected in correctness.expected_substrings if expected.lower() not in haystack]
    assert not missing, f"Chat response missing expected substring(s) {missing!r}: {response_content!r}"


def run_vllm_benchmark(
    *,
    model: str,
    port: int,
    output_dir: Path,
    config: VllmBenchmarkConfig,
) -> dict[str, Any]:
    """Run an opt-in ``vllm bench serve`` probe and return its saved JSON result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / config.result_filename
    cmd = _vllm_benchmark_command(model=model, port=port, output_dir=output_dir, config=config)

    print(f"\nRunning vLLM benchmark with command: {' '.join(cmd)}")
    completed = subprocess.run(  # noqa: S603 - command is built from controlled test config
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=config.timeout,
    )

    assert artifact_path.exists(), f"vLLM benchmark did not write result JSON: {artifact_path}"
    with artifact_path.open(encoding="utf-8") as f:
        benchmark_result = json.load(f)
    assert isinstance(benchmark_result, dict), f"vLLM benchmark result is not a JSON object: {artifact_path}"

    completed_requests = benchmark_result.get("completed")
    assert isinstance(completed_requests, int), f"vLLM benchmark did not report completed requests: {benchmark_result}"
    assert completed_requests >= config.request_count, (
        f"vLLM benchmark completed {completed_requests} requests, expected at least {config.request_count}"
    )

    for metric_name in config.required_positive_metrics:
        metric_value = benchmark_result.get(metric_name)
        assert isinstance(metric_value, int | float) and metric_value > 0, (
            f"vLLM benchmark metric {metric_name!r} must be positive, got {metric_value!r}. "
            f"Full result: {benchmark_result}"
        )

    benchmark_result["artifact_path"] = str(artifact_path)
    benchmark_result["stdout"] = completed.stdout
    benchmark_result["stderr"] = completed.stderr
    benchmark_result["command"] = cmd
    return benchmark_result


def _vllm_benchmark_command(
    *,
    model: str,
    port: int,
    output_dir: Path,
    config: VllmBenchmarkConfig,
) -> list[str]:
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        config.backend,
        "--base-url",
        f"http://localhost:{port}",
        "--endpoint",
        config.endpoint,
        "--model",
        model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(config.input_tokens),
        "--random-output-len",
        str(config.output_tokens),
        "--random-range-ratio",
        str(config.random_range_ratio),
        "--num-prompts",
        str(config.request_count),
        "--max-concurrency",
        str(config.max_concurrency),
        "--seed",
        str(config.seed),
        "--temperature",
        str(config.temperature),
        "--save-result",
    ]
    if config.save_detailed:
        cmd.append("--save-detailed")

    cmd.extend((
        "--result-dir",
        str(output_dir),
        "--result-filename",
        config.result_filename,
        *config.extra_args,
    ))
    return cmd


def _log_reader_thread(process: subprocess.Popen[bytes], log_lines: list[str], prefix: str = "[vLLM]") -> None:
    """Background thread to read and display server logs in real time."""
    if process.stdout is None:
        return

    for line in iter(process.stdout.readline, b""):
        if not line:
            break
        decoded_line = line.decode("utf-8", errors="replace").rstrip()
        if decoded_line:
            log_lines.append(decoded_line)
            print(f"{prefix} {decoded_line}", flush=True)

    remaining = process.stdout.read()
    if remaining:
        for line in remaining.decode("utf-8", errors="replace").splitlines():
            if line:
                log_lines.append(line)
                print(f"{prefix} {line}", flush=True)


def start_vllm_server(case: VllmOffloadSmokeCase) -> tuple[subprocess.Popen[bytes], list[str]]:
    """Start vLLM server in background with real-time log monitoring."""
    from tests.integration.conftest import enable_offline_if_cached

    enable_offline_if_cached(case.model_name)

    env_vars = dict(case.extra_env_vars)
    cmd = ["vllm", "serve", case.model_name, *case.cli_args]
    print(f"\nStarting vLLM server with command: {' '.join(cmd)}")
    print(f"Additional environment variables: {env_vars}")

    env = os.environ.copy()
    env.update(env_vars)

    process = subprocess.Popen(  # noqa: S603 - inputs are controlled by test parametrization
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=1,
        start_new_session=True,
    )

    log_lines: list[str] = []
    log_thread = threading.Thread(target=_log_reader_thread, args=(process, log_lines), daemon=True)
    log_thread.start()

    return process, log_lines


def wait_for_server(
    port: int = DEFAULT_VLLM_PORT,
    timeout: int = 600,
    check_interval: int = 5,
    process: subprocess.Popen[bytes] | None = None,
) -> bool:
    """Wait for vLLM server to be ready."""
    health_url = f"http://localhost:{port}/health"
    start_time = time.time()

    print(f"Waiting for server to be ready at {health_url}...")

    while time.time() - start_time < timeout:
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None:
                print(f"Server process exited with code {exit_code} before becoming ready")
                print("Check the server logs above for error details")
                return False

        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                print(f"Server is ready after {time.time() - start_time:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(max(check_interval - 2, 0.1))

    print(f"Server did not become ready within {timeout}s")
    return False


def stop_vllm_server(process: subprocess.Popen[bytes], timeout: int = 30) -> None:
    """Stop the vLLM server and every worker process it spawned."""
    if process.poll() is None:
        print("\nStopping vLLM server...")
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("Server did not stop gracefully, forcing...")
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        else:
            # The API server can exit before EngineCore. The dedicated process
            # group makes it safe to remove any descendants left after shutdown.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


def make_chat_request(
    messages: list[dict[str, str]],
    model: str,
    port: int = DEFAULT_VLLM_PORT,
    max_tokens: int = 100,
    timeout: int = 60,
    temperature: float = 0.0,
    seed: int | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make a chat completion request to vLLM API."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "return_token_ids": True,
    }
    if seed is not None:
        payload["seed"] = seed
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()

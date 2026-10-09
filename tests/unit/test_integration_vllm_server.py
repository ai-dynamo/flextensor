# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-style tests for vLLM integration server helpers."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tests.integration import _vllm_server
from tests.integration._vllm_server import (
    FLEXTENSOR_OFFLOAD_WORKER_CLS,
    FLEXTENSOR_SNAPSHOT_WORKER_CLS,
    MemoryProfilingMetrics,
    VllmBenchmarkConfig,
    VllmCorrectnessCheck,
    VllmOffloadSmokeCase,
    run_vllm_benchmark,
    run_vllm_server_test,
)


def _smoke_case(
    *,
    cli_args: tuple[str, ...] = (),
    extra_env_vars: tuple[tuple[str, str], ...] = (),
) -> VllmOffloadSmokeCase:
    return VllmOffloadSmokeCase(
        model_name="model",
        output_dir_name="case",
        cli_args=cli_args,
        extra_env_vars=extra_env_vars,
    )


def _patch_successful_vllm_server(
    monkeypatch: pytest.MonkeyPatch,
    case: VllmOffloadSmokeCase,
    *,
    response_content: str = "Paris",
    usage: dict[str, object] | None = None,
    log_lines: list[str] | None = None,
    observed_request: dict[str, object] | None = None,
) -> list[VllmOffloadSmokeCase]:
    started_cases: list[VllmOffloadSmokeCase] = []
    response_usage = {"completion_tokens": 1} if usage is None else usage
    server_logs = log_lines if log_lines is not None else ["Automatically detected platform cuda.", "server started"]

    def _fake_start_vllm_server(start_case):
        started_cases.append(start_case)
        return SimpleNamespace(poll=lambda: None), list(server_logs)

    def _fake_make_chat_request(**kwargs):
        if observed_request is not None:
            observed_request.update(kwargs)
        return {
            "choices": [{"message": {"content": response_content}}],
            "usage": response_usage,
        }

    monkeypatch.setattr(_vllm_server, "start_vllm_server", _fake_start_vllm_server)
    monkeypatch.setattr(_vllm_server, "wait_for_server", lambda **kwargs: True)
    monkeypatch.setattr(_vllm_server, "make_chat_request", _fake_make_chat_request)
    monkeypatch.setattr(
        _vllm_server.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"id": case.model_name}]},
        ),
    )
    monkeypatch.setattr(_vllm_server, "stop_vllm_server", lambda _process: None)
    monkeypatch.setattr(_vllm_server, "parse_memory_profiling_logs", lambda _lines: MemoryProfilingMetrics())
    monkeypatch.setattr(_vllm_server, "collect_vllm_backend_evidence", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(_vllm_server, "query_runtime_gpu_backend_metadata", dict)
    return started_cases


def test_case_with_flextensor_offload_adds_env_and_worker() -> None:
    case = VllmOffloadSmokeCase(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir_name="qwen_smoke",
        cli_args=("--enforce-eager",),
    ).with_flextensor_offload()

    env_vars = dict(case.extra_env_vars)
    assert env_vars["FT_ENABLED"] == "1"
    assert env_vars["FT_ENABLE_DIAGNOSTICS"] == "1"
    assert env_vars["FT_ENABLE_INSTRUMENTATION"] == "1"
    assert case.cli_args == ("--enforce-eager", "--worker-cls", FLEXTENSOR_OFFLOAD_WORKER_CLS)
    assert _vllm_server.case_uses_flextensor_offload(case)


def test_case_port_reads_cli_args() -> None:
    case = VllmOffloadSmokeCase(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir_name="qwen_smoke",
        cli_args=("--port", "8123"),
    )

    assert _vllm_server.case_port(case) == 8123


def test_empty_chat_content_error_includes_full_response() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": None, "reasoning_content": "Paris"},
                "token_ids": [1, 2, 3],
            }
        ],
        "usage": {"completion_tokens": 3},
    }

    with pytest.raises(AssertionError) as error:
        _vllm_server._validated_chat_response_content(response)

    assert str(error.value) == f"No text content in chat response: {response!r}"


def test_run_vllm_server_test_uses_case_cli_args_without_injecting_enforce_eager(monkeypatch, tmp_path) -> None:
    case = VllmOffloadSmokeCase(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir_name="baseline",
        cli_args=("--port", "8123", "--max-model-len", "64"),
    )
    started_cases = _patch_successful_vllm_server(monkeypatch, case)

    _memory, metrics, _logs = run_vllm_server_test(case, output_dir=tmp_path)

    assert started_cases[0].cli_args == case.cli_args
    assert "--enforce-eager" not in started_cases[0].cli_args
    assert dict(started_cases[0].extra_env_vars) == {
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "VLLM_NO_USAGE_STATS": "1",
    }
    assert metrics["offload_enabled"] is False


@pytest.mark.parametrize(
    ("worker_cls", "worker_env", "worker_logs"),
    [
        (
            FLEXTENSOR_OFFLOAD_WORKER_CLS,
            (("FT_VLLM_USE_V2_WORKER", "1"),),
            ["FlexTensor vLLM integration v2 state takeover complete"],
        ),
        (
            FLEXTENSOR_OFFLOAD_WORKER_CLS,
            (("FT_VLLM_USE_V2_WORKER", "0"),),
            ["FlexTensor offloading enabled with config", "FlexTensor offloading applied"],
        ),
        (
            FLEXTENSOR_SNAPSHOT_WORKER_CLS,
            (),
            ["FlexTensor offloading enabled with config", "FlexTensor offloading applied"],
        ),
    ],
)
def test_run_vllm_server_test_accepts_worker_specific_offload_logs(
    monkeypatch,
    tmp_path,
    worker_cls,
    worker_env,
    worker_logs,
) -> None:
    case = _smoke_case(cli_args=("--port", "8123")).with_flextensor_offload(worker_cls).with_env_vars(*worker_env)
    validated_instrumentation_dirs = []
    transfer_validation = {
        "pcie_link_gen": 4,
        "pcie_link_width": 16,
        "pci_bus_id": "00000000:01:00.0",
        "cpu_affinity": "0-31",
        "numa_affinity": "0",
        "observed_bandwidth_gbps": 20.0,
        "min_expected_bandwidth_gbps": 4.0,
        "max_expected_bandwidth_gbps": 40.0,
    }
    started_cases = _patch_successful_vllm_server(
        monkeypatch,
        case,
        log_lines=["Automatically detected platform cuda.", *worker_logs],
    )

    def validate_transfer(instrumentation_dir):
        validated_instrumentation_dirs.append(instrumentation_dir)
        return transfer_validation

    monkeypatch.setattr(
        _vllm_server,
        "validate_latest_memory_transfer_stats_for_current_bus",
        validate_transfer,
    )

    _memory, metrics, _logs = run_vllm_server_test(case, output_dir=tmp_path)

    env_vars = dict(started_cases[0].extra_env_vars)
    assert "FT_DEBUG_LOG_PATH" not in env_vars
    assert env_vars["FT_INSTRUMENTATION_OUTPUT_DIR"] == str(tmp_path / "instrumentation")
    assert env_vars["FT_ENABLE_DIAGNOSTICS"] == "1"
    assert env_vars["FT_ENABLE_INSTRUMENTATION"] == "1"
    assert metrics["offload_enabled"] is True
    if worker_cls == FLEXTENSOR_SNAPSHOT_WORKER_CLS:
        assert validated_instrumentation_dirs == [tmp_path / "instrumentation"]
        assert metrics["memory_transfer_validation"] == transfer_validation
    else:
        assert validated_instrumentation_dirs == []
        assert "memory_transfer_validation" not in metrics


def test_run_vllm_server_test_rejects_zero_completion_tokens(monkeypatch, tmp_path) -> None:
    case = _smoke_case(cli_args=("--port", "8123"))
    _patch_successful_vllm_server(monkeypatch, case, usage={"completion_tokens": 0})

    with pytest.raises(AssertionError, match="completion_tokens"):
        run_vllm_server_test(case, output_dir=tmp_path)


def test_run_vllm_server_test_rejects_v2_takeover_log_for_baseline(monkeypatch, tmp_path) -> None:
    case = _smoke_case(cli_args=("--port", "8123"))
    _patch_successful_vllm_server(
        monkeypatch,
        case,
        log_lines=[
            "Automatically detected platform cuda.",
            "FlexTensor vLLM integration v2 state takeover complete",
        ],
    )

    with pytest.raises(AssertionError, match="Unexpected v2 state takeover in baseline"):
        run_vllm_server_test(case, output_dir=tmp_path)


def test_run_vllm_server_test_records_backend_evidence_env_vars(monkeypatch, tmp_path) -> None:
    case = _smoke_case(
        cli_args=("--port", "8123"),
        extra_env_vars=(("FT_MAX_GPU_MEM_FRACTION", "0.75"),),
    )
    observed_backend_kwargs: dict[str, object] = {}
    _patch_successful_vllm_server(monkeypatch, case)

    def _fake_collect_backend_evidence(_log_lines, **kwargs):
        observed_backend_kwargs.update(kwargs)
        return {"extra_env_vars": kwargs["extra_env_vars"]}

    monkeypatch.setattr(_vllm_server, "collect_vllm_backend_evidence", _fake_collect_backend_evidence)

    _memory, metrics, _logs = run_vllm_server_test(case, output_dir=tmp_path)

    extra_env_vars = metrics["backend_evidence"]["extra_env_vars"]
    assert extra_env_vars["FT_MAX_GPU_MEM_FRACTION"] == "0.75"
    assert extra_env_vars["VLLM_LOGGING_LEVEL"] == "DEBUG"
    assert extra_env_vars["VLLM_NO_USAGE_STATS"] == "1"
    assert observed_backend_kwargs["extra_env_vars"] == extra_env_vars


def test_run_vllm_server_test_uses_configured_correctness_check(monkeypatch, tmp_path) -> None:
    case = _smoke_case(cli_args=("--port", "8123"))
    correctness = VllmCorrectnessCheck(
        messages=({"role": "user", "content": "Return only the word: blue"},),
        expected_substrings=("blue",),
        max_tokens=4,
        timeout=9,
        temperature=0.0,
        seed=123,
    )
    observed_request: dict[str, object] = {}

    _patch_successful_vllm_server(
        monkeypatch,
        case,
        response_content="blue",
        observed_request=observed_request,
    )

    _memory, metrics, _logs = run_vllm_server_test(case, output_dir=tmp_path, correctness_check=correctness)

    assert observed_request == {
        "messages": [{"role": "user", "content": "Return only the word: blue"}],
        "model": case.model_name,
        "port": 8123,
        "max_tokens": 4,
        "timeout": 9,
        "temperature": 0.0,
        "seed": 123,
    }
    assert metrics["correctness"] == {
        "expected_substrings": ["blue"],
        "request_latency_seconds": pytest.approx(metrics["request_latency_seconds"]),
        "response_content": "blue",
        "responses": [{"choices": [{"message": {"content": "blue"}}], "usage": {"completion_tokens": 1}}],
        "usage": {"completion_tokens": 1},
    }
    assert metrics["response_content"] == "blue"


def test_run_vllm_server_test_forwards_chat_template_kwargs(monkeypatch, tmp_path) -> None:
    case = _smoke_case(cli_args=("--port", "8123"))
    correctness = VllmCorrectnessCheck(
        max_tokens=1,
        timeout=180,
        chat_template_kwargs={"enable_thinking": False},
    )
    observed_request: dict[str, object] = {}

    _patch_successful_vllm_server(
        monkeypatch,
        case,
        response_content="A",
        observed_request=observed_request,
    )

    _memory, metrics, _logs = run_vllm_server_test(case, output_dir=tmp_path, correctness_check=correctness)

    assert observed_request["chat_template_kwargs"] == {"enable_thinking": False}
    assert metrics["correctness"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_run_vllm_server_test_rejects_response_without_expected_substring(monkeypatch, tmp_path) -> None:
    case = _smoke_case(cli_args=("--port", "8123"))
    correctness = VllmCorrectnessCheck(expected_substrings=("Paris",))

    _patch_successful_vllm_server(monkeypatch, case, response_content="London")

    with pytest.raises(AssertionError, match="expected substring"):
        run_vllm_server_test(case, output_dir=tmp_path, correctness_check=correctness)


def test_run_vllm_server_test_default_correctness_allows_any_non_empty_response(monkeypatch, tmp_path) -> None:
    case = _smoke_case(cli_args=("--port", "8123"))
    _patch_successful_vllm_server(monkeypatch, case, response_content="The")

    _memory, metrics, _logs = run_vllm_server_test(case, output_dir=tmp_path, chat_max_tokens=1)

    assert metrics["response_content"] == "The"
    assert metrics["correctness"]["expected_substrings"] == []


@pytest.mark.parametrize("wrong_response", [None, 0, 1])
def test_concurrent_correctness_checks_validate_every_response(monkeypatch, tmp_path, wrong_response) -> None:
    case = _smoke_case()
    _patch_successful_vllm_server(monkeypatch, case)
    both_requests_started = threading.Barrier(2)

    def respond(**_kwargs):
        index = both_requests_started.wait(timeout=5)
        return {
            "choices": [{"message": {"content": "London" if index == wrong_response else "Paris"}}],
            "usage": {"completion_tokens": 1},
        }

    monkeypatch.setattr(_vllm_server, "make_chat_request", respond)
    correctness = VllmCorrectnessCheck(expected_substrings=("Paris",), request_count=2)
    if wrong_response is not None:
        with pytest.raises(AssertionError, match="expected substring"):
            run_vllm_server_test(case, output_dir=tmp_path, correctness_check=correctness)
    else:
        _, metrics, _ = run_vllm_server_test(case, output_dir=tmp_path, correctness_check=correctness)
        assert len(metrics["correctness"]["responses"]) == 2


@pytest.mark.parametrize("wrong_response", [None, 4, 5])
def test_repeated_concurrent_correctness_checks_validate_later_batches(monkeypatch, tmp_path, wrong_response) -> None:
    case = _smoke_case()
    _patch_successful_vllm_server(monkeypatch, case)
    monkeypatch.setattr(_vllm_server.time, "sleep", lambda _seconds: None)
    both_requests_started = threading.Barrier(2)
    request_lock = threading.Lock()
    request_count = 0

    def respond(**_kwargs):
        nonlocal request_count
        both_requests_started.wait(timeout=5)
        with request_lock:
            index = request_count
            request_count += 1
        return {
            "choices": [{"message": {"content": "London" if index == wrong_response else "Paris"}}],
            "usage": {"completion_tokens": 8},
        }

    monkeypatch.setattr(_vllm_server, "make_chat_request", respond)
    correctness = VllmCorrectnessCheck(expected_substrings=("Paris",), request_count=2, repeat_count=3)
    if wrong_response is not None:
        with pytest.raises(AssertionError, match="expected substring"):
            run_vllm_server_test(case, output_dir=tmp_path, correctness_check=correctness)
    else:
        _, metrics, _ = run_vllm_server_test(case, output_dir=tmp_path, correctness_check=correctness)
        assert len(metrics["correctness"]["responses"]) == 6
    assert request_count == 6


def test_run_vllm_server_test_runs_optional_benchmark(monkeypatch, tmp_path) -> None:
    case = _smoke_case(cli_args=("--port", "8123"))
    benchmark = VllmBenchmarkConfig(request_count=3, input_tokens=16, output_tokens=4, max_concurrency=1, seed=123)
    observed: dict[str, object] = {}

    _patch_successful_vllm_server(monkeypatch, case)

    def _fake_run_vllm_benchmark(*, model, port, output_dir, config):
        observed.update({"model": model, "port": port, "output_dir": output_dir, "config": config})
        return {"completed": 3, "output_throughput": 12.5}

    monkeypatch.setattr(_vllm_server, "run_vllm_benchmark", _fake_run_vllm_benchmark)

    _memory, metrics, _logs = run_vllm_server_test(case, output_dir=tmp_path, benchmark_config=benchmark)

    assert observed == {
        "model": case.model_name,
        "port": 8123,
        "output_dir": tmp_path / "benchmark",
        "config": benchmark,
    }
    assert metrics["benchmark"] == {"completed": 3, "output_throughput": 12.5}


def test_run_vllm_benchmark_builds_vllm_bench_command_and_reads_result(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}
    config = VllmBenchmarkConfig(request_count=3, input_tokens=16, output_tokens=4, max_concurrency=1, seed=123)

    def _fake_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["kwargs"] = kwargs
        result_path = tmp_path / config.result_filename
        result_path.write_text(
            ('{"completed": 3, "request_throughput": 1.5, "output_throughput": 8.0, "total_token_throughput": 24.0}'),
            encoding="utf-8",
        )
        return SimpleNamespace(stdout="benchmark complete", stderr="", returncode=0)

    monkeypatch.setattr(_vllm_server.subprocess, "run", _fake_run)

    result = run_vllm_benchmark(model="model-id", port=8123, output_dir=tmp_path, config=config)

    assert observed["cmd"] == [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        "http://localhost:8123",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        "model-id",
        "--dataset-name",
        "random",
        "--random-input-len",
        "16",
        "--random-output-len",
        "4",
        "--random-range-ratio",
        "0.0",
        "--num-prompts",
        "3",
        "--max-concurrency",
        "1",
        "--seed",
        "123",
        "--temperature",
        "0.0",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(tmp_path),
        "--result-filename",
        config.result_filename,
    ]
    assert observed["kwargs"]["check"] is True
    assert observed["kwargs"]["capture_output"] is True
    assert result["completed"] == 3
    assert result["output_throughput"] == pytest.approx(8.0)
    assert result["artifact_path"] == str(tmp_path / config.result_filename)
    assert result["stdout"] == "benchmark complete"


def test_run_vllm_benchmark_rejects_non_positive_required_metric(monkeypatch, tmp_path) -> None:
    config = VllmBenchmarkConfig(request_count=1)

    def _fake_run(_cmd, **_kwargs):
        (tmp_path / config.result_filename).write_text(
            '{"completed": 1, "request_throughput": 1.0, "output_throughput": 0.0, "total_token_throughput": 1.0}',
            encoding="utf-8",
        )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(_vllm_server.subprocess, "run", _fake_run)

    with pytest.raises(AssertionError, match="output_throughput"):
        run_vllm_benchmark(model="model-id", port=8123, output_dir=tmp_path, config=config)


def test_case_with_flextensor_offload_replaces_existing_worker_arg() -> None:
    case = _smoke_case(
        cli_args=("--worker-cls=old.Worker", "--port", "8123"),
        extra_env_vars=(("FT_ENABLED", "0"), ("OTHER", "1")),
    ).with_flextensor_offload()

    assert case.cli_args == (f"--worker-cls={FLEXTENSOR_OFFLOAD_WORKER_CLS}", "--port", "8123")
    assert dict(case.extra_env_vars) == {
        "FT_ENABLED": "1",
        "OTHER": "1",
        "FT_ENABLE_DIAGNOSTICS": "1",
        "FT_ENABLE_INSTRUMENTATION": "1",
    }
    assert _vllm_server.case_uses_flextensor_offload(case)


def test_case_with_env_vars_overrides_existing_values() -> None:
    case = _smoke_case(extra_env_vars=(("A", "1"), ("B", "2"))).with_env_vars(("B", "override"), ("C", "3"))

    assert dict(case.extra_env_vars) == {"A": "1", "B": "override", "C": "3"}


def test_case_port_uses_default_when_cli_arg_missing() -> None:
    assert _vllm_server.case_port(_smoke_case()) == _vllm_server.DEFAULT_VLLM_PORT


def test_case_port_reads_equals_form() -> None:
    assert _vllm_server.case_port(_smoke_case(cli_args=("--port=8124",))) == 8124


def test_case_port_rejects_missing_value() -> None:
    with pytest.raises(AssertionError, match="--port must be followed"):
        _vllm_server.case_port(_smoke_case(cli_args=("--port",)))


def test_case_uses_flextensor_offload_rejects_truthy_env_with_unknown_worker() -> None:
    case = _smoke_case(
        cli_args=("--worker-cls", "custom.Worker"),
        extra_env_vars=(("FT_ENABLED", "true"),),
    )

    assert not _vllm_server.case_uses_flextensor_offload(case)


def test_memory_profiling_metrics_to_dict_includes_all_fields() -> None:
    metrics = MemoryProfilingMetrics(
        initial_free_memory_gib=10.0,
        requested_memory_util=0.9,
        requested_memory_gib=9.0,
        free_memory_after_profiling_total_gib=8.0,
        free_memory_after_profiling_within_requested_gib=7.0,
        total_non_kv_cache_memory_gib=6.0,
        torch_peak_memory_increase_gib=5.0,
        non_torch_forward_increase_memory_gib=4.0,
        weights_memory_gib=3.0,
        available_kv_cache_memory_gib=2.0,
        whole_model_budget_bytes=123456,
        managed_gpu_resident_bytes=120000,
    )

    assert metrics.to_dict() == {
        "initial_free_memory_gib": 10.0,
        "requested_memory_util": 0.9,
        "requested_memory_gib": 9.0,
        "free_memory_after_profiling_total_gib": 8.0,
        "free_memory_after_profiling_within_requested_gib": 7.0,
        "total_non_kv_cache_memory_gib": 6.0,
        "torch_peak_memory_increase_gib": 5.0,
        "non_torch_forward_increase_memory_gib": 4.0,
        "weights_memory_gib": 3.0,
        "available_kv_cache_memory_gib": 2.0,
        "whole_model_budget_bytes": 123456,
        "managed_gpu_resident_bytes": 120000,
    }


def test_parse_memory_profiling_logs_extracts_all_known_metrics() -> None:
    metrics = _vllm_server.parse_memory_profiling_logs([
        "Initial free memory: 79.11 GiB; Requested memory: 0.90 (util), 71.20 GiB",
        "Free memory after profiling: 62.50 GiB (total), 54.20 GiB (within requested)",
        (
            "Total non KV cache memory: 11.25 GiB; torch peak memory increase: 1.50 GiB; "
            "non-torch forward increase memory: 0.75 GiB; weights memory: 9.00 GiB"
        ),
        "Available KV cache memory: 42.75 GiB",
        "FlexTensor v2 GPU budget resolved: whole_model_budget_bytes=123456 managed_gpu_resident_bytes=120000",
    ])

    assert metrics == MemoryProfilingMetrics(
        initial_free_memory_gib=79.11,
        requested_memory_util=0.90,
        requested_memory_gib=71.20,
        free_memory_after_profiling_total_gib=62.50,
        free_memory_after_profiling_within_requested_gib=54.20,
        total_non_kv_cache_memory_gib=11.25,
        torch_peak_memory_increase_gib=1.50,
        non_torch_forward_increase_memory_gib=0.75,
        weights_memory_gib=9.00,
        available_kv_cache_memory_gib=42.75,
        whole_model_budget_bytes=123456,
        managed_gpu_resident_bytes=120000,
    )


def test_wait_for_server_returns_true_on_healthy_response(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def _fake_get(url, *, timeout):
        observed["url"] = url
        observed["timeout"] = timeout
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(_vllm_server.requests, "get", _fake_get)

    assert _vllm_server.wait_for_server(port=8123, timeout=1, check_interval=1)
    assert observed == {"url": "http://localhost:8123/health", "timeout": 2}


def test_wait_for_server_returns_false_when_process_exits(monkeypatch) -> None:
    monkeypatch.setattr(
        _vllm_server.requests,
        "get",
        lambda *_args, **_kwargs: pytest.fail("health endpoint should not be polled after process exit"),
    )

    assert not _vllm_server.wait_for_server(timeout=1, process=SimpleNamespace(poll=lambda: 7))


def test_start_vllm_server_builds_command_and_environment(monkeypatch) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "tests.integration.conftest.enable_offline_if_cached",
        lambda model_name: observed.setdefault("offline_model", model_name),
    )

    class _FakeProcess:
        stdout = None

    def _fake_popen(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["env"] = kwargs["env"]
        assert kwargs["stdout"] is _vllm_server.subprocess.PIPE
        assert kwargs["stderr"] is _vllm_server.subprocess.STDOUT
        assert kwargs["start_new_session"] is True
        return _FakeProcess()

    class _FakeThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            observed["thread_started"] = True

    monkeypatch.setattr(_vllm_server.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(_vllm_server.threading, "Thread", _FakeThread)
    case = VllmOffloadSmokeCase(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir_name="qwen",
        cli_args=("--port", "8123"),
        extra_env_vars=(("FT_ENABLED", "1"),),
    )

    process, log_lines = _vllm_server.start_vllm_server(case)

    assert isinstance(process, _FakeProcess)
    assert log_lines == []
    assert observed["offline_model"] == case.model_name
    assert observed["cmd"] == ["vllm", "serve", case.model_name, "--port", "8123"]
    assert observed["env"]["FT_ENABLED"] == "1"
    assert observed["thread_started"] is True


def test_stop_vllm_server_terminates_running_process_group(monkeypatch) -> None:
    events: list[object] = []

    class _Process:
        pid = 42

        def poll(self):
            return None

        def wait(self, *, timeout):
            events.append(("wait", timeout))

    monkeypatch.setattr(
        _vllm_server.os,
        "killpg",
        lambda pid, sig: events.append(("killpg", pid, sig)),
    )

    _vllm_server.stop_vllm_server(_Process(), timeout=5)

    assert events == [
        ("killpg", 42, _vllm_server.signal.SIGTERM),
        ("wait", 5),
        ("killpg", 42, _vllm_server.signal.SIGKILL),
    ]


def test_stop_vllm_server_kills_process_group_when_graceful_shutdown_times_out(monkeypatch) -> None:
    events: list[object] = []

    class _Process:
        pid = 43

        def poll(self):
            return None

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if ("killpg", 43, _vllm_server.signal.SIGKILL) not in events:
                raise _vllm_server.subprocess.TimeoutExpired(cmd="vllm", timeout=timeout)

    monkeypatch.setattr(
        _vllm_server.os,
        "killpg",
        lambda pid, sig: events.append(("killpg", pid, sig)),
    )

    _vllm_server.stop_vllm_server(_Process(), timeout=5)

    assert events == [
        ("killpg", 43, _vllm_server.signal.SIGTERM),
        ("wait", 5),
        ("killpg", 43, _vllm_server.signal.SIGKILL),
        ("wait", None),
    ]


@pytest.mark.parametrize("missing_signal", [_vllm_server.signal.SIGTERM, _vllm_server.signal.SIGKILL])
def test_stop_vllm_server_ignores_missing_process_group(monkeypatch, missing_signal) -> None:
    class _Process:
        pid = 44

        def poll(self):
            return None

        def wait(self, timeout=None):
            if timeout is not None and missing_signal is _vllm_server.signal.SIGKILL:
                raise _vllm_server.subprocess.TimeoutExpired(cmd="vllm", timeout=timeout)

    def killpg(_pid, sig):
        if sig is missing_signal:
            raise ProcessLookupError

    monkeypatch.setattr(_vllm_server.os, "killpg", killpg)

    _vllm_server.stop_vllm_server(_Process(), timeout=5)


def test_make_chat_request_posts_openai_payload(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def _fake_post(url, *, json, timeout):
        observed["url"] = url
        observed["json"] = json
        observed["timeout"] = timeout
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "OK"}}]},
        )

    monkeypatch.setattr(_vllm_server.requests, "post", _fake_post)

    response = _vllm_server.make_chat_request(
        messages=[{"role": "user", "content": "Say OK"}],
        model="model-id",
        port=8123,
        max_tokens=3,
        timeout=9,
        temperature=0.2,
        seed=123,
    )

    assert response["choices"][0]["message"]["content"] == "OK"
    assert observed == {
        "url": "http://localhost:8123/v1/chat/completions",
        "json": {
            "model": "model-id",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 3,
            "temperature": 0.2,
            "return_token_ids": True,
            "seed": 123,
        },
        "timeout": 9,
    }


def test_make_chat_request_posts_chat_template_kwargs_when_configured(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def _fake_post(url, *, json, timeout):
        observed["url"] = url
        observed["json"] = json
        observed["timeout"] = timeout
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "OK"}}]},
        )

    monkeypatch.setattr(_vllm_server.requests, "post", _fake_post)

    _vllm_server.make_chat_request(
        messages=[{"role": "user", "content": "Say OK"}],
        model="model-id",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert observed["json"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("ready", [True, False])
def test_host_memory_sampling_finishes_before_server_shutdown(monkeypatch, tmp_path, ready):
    from unittest.mock import Mock

    case = _smoke_case()
    _patch_successful_vllm_server(monkeypatch, case)
    original_start = _vllm_server.start_vllm_server

    def start(case):
        process, logs = original_start(case)
        process.pid = 123
        return process, logs

    sampler = Mock()
    sampler.finish.return_value = {"phases": {}, "failure": None}
    factory = Mock(return_value=sampler)
    monkeypatch.setattr(_vllm_server, "start_vllm_server", start)
    monkeypatch.setattr(_vllm_server, "wait_for_server", lambda **_kwargs: ready)
    monkeypatch.setattr(_vllm_server.time, "monotonic", Mock(side_effect=[0, 0, 6]))
    monkeypatch.setattr(_vllm_server.time, "sleep", lambda _seconds: None)

    def stop(_process):
        sampler.finish.assert_called_once_with(tmp_path / "host_memory.json")

    monkeypatch.setattr(_vllm_server, "stop_vllm_server", stop)
    if ready:
        _, metrics, _ = run_vllm_server_test(case, tmp_path, host_memory_sampler_factory=factory)
        assert metrics["host_memory"] is sampler.finish.return_value
        assert metrics["host_memory_probe"] == {"completed_requests": 1, "concurrency": 1}
        assert [call.args[0] for call in sampler.phase.call_args_list] == ["warmup", "inference", "benchmark"]
        assert [call.args[0] for call in sampler.idle_samples.call_args_list] == ["warm", "post_inference"]
    else:
        with pytest.raises(AssertionError, match="Server failed to start"):
            run_vllm_server_test(case, tmp_path, host_memory_sampler_factory=factory)
    factory.assert_called_once_with(123)
    sampler.finish.assert_called_once()

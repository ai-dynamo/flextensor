# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from flextensor.config import OffloadConfig
from flextensor.contrib.vllm._patterns import VLLM_DEFAULT_EXCLUDE_PATTERNS, VLLM_DEFAULT_INCLUDE_PATTERNS

from ._v2_worker_test_utils import _install_bootstrap_fakes, _worker


def _valid_shm_inputs(worker_module, profile_dir):
    vllm_config = worker_module.VllmConfig(
        compilation_config=SimpleNamespace(
            mode=worker_module.CompilationMode.NONE,
            cudagraph_mode=worker_module.CUDAGraphMode.NONE,
        ),
        load_config=SimpleNamespace(load_format="auto"),
        lora_config=None,
        model_config=SimpleNamespace(
            architecture="Qwen2ForCausalLM",
            architectures=["Qwen2ForCausalLM"],
            convert_type="none",
            quantization=None,
            runner_type="generate",
        ),
        parallel_config=SimpleNamespace(
            data_parallel_backend="mp",
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
            data_parallel_index=0,
            data_parallel_rank_local=0,
            data_parallel_size=1,
            data_parallel_size_local=1,
            decode_context_parallel_size=1,
            enable_elastic_ep=False,
            enable_ep_weight_filter=False,
            enable_eplb=False,
            enable_expert_parallel=False,
            nnodes=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            use_ubatching=False,
        ),
        speculative_config=None,
    )
    offload_config = OffloadConfig(
        profile_storage_dir=str(profile_dir),
        profile_read_only=True,
        transfer_mode="allocation_block_transfer",
        shm_enabled=True,
        shm_namespace="vllm-test-run",
        shm_wait_timeout=30,
        pinned_memory=False,
    )
    return vllm_config, offload_config


def _set_nested_attribute(root, path: str, value) -> None:
    owner = root
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)


def test_shm_worker_accepts_dense_local_dp_fields_after_vllm_resets_sizes(
    worker_module,
    tmp_path,
) -> None:
    vllm_config, offload_config = _valid_shm_inputs(worker_module, tmp_path)

    worker_module._validate_shm_worker(vllm_config, offload_config)


def test_shm_worker_accepts_other_model_architectures(
    worker_module,
    tmp_path,
) -> None:
    vllm_config, offload_config = _valid_shm_inputs(worker_module, tmp_path)
    vllm_config.model_config.architecture = "LlamaForCausalLM"
    vllm_config.model_config.architectures = ("LlamaForCausalLM",)

    worker_module._validate_shm_worker(vllm_config, offload_config)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("model_config.runner_type", "pooling", "unconverted generation"),
        ("model_config.convert_type", "embed", "unconverted generation"),
        ("model_config.quantization", "fp8", "quantization"),
        ("lora_config", object(), "LoRA"),
        ("speculative_config", object(), "speculative"),
        ("parallel_config.tensor_parallel_size", 2, "TP=1 and PP=1"),
        ("parallel_config.pipeline_parallel_size", 2, "TP=1 and PP=1"),
        ("parallel_config.prefill_context_parallel_size", 2, "PCP=1 and DCP=1"),
        ("parallel_config.decode_context_parallel_size", 2, "PCP=1 and DCP=1"),
        ("parallel_config.data_parallel_backend", "ray", "single-node"),
        ("parallel_config.nnodes", 2, "single-node"),
        ("parallel_config.data_parallel_external_lb", True, "external or hybrid"),
        ("parallel_config.data_parallel_hybrid_lb", True, "external or hybrid"),
        ("parallel_config.enable_expert_parallel", True, "expert parallelism"),
        ("load_config.load_format", "safetensors", "load_format='auto'"),
    ],
)
def test_shm_worker_rejects_unsupported_vllm_topology(
    worker_module,
    tmp_path,
    path,
    value,
    match,
) -> None:
    vllm_config, offload_config = _valid_shm_inputs(worker_module, tmp_path)
    _set_nested_attribute(vllm_config, path, value)

    with pytest.raises(worker_module.VllmFlexTensorV2Error, match=match):
        worker_module._validate_shm_worker(vllm_config, offload_config)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"profile_read_only": False}, "PROFILE_READ_ONLY"),
        ({"profile_storage_dir": None}, "PROFILE_STORAGE_DIR"),
    ],
)
def test_shm_worker_rejects_unsupported_flextensor_config(
    worker_module,
    tmp_path,
    updates,
    match,
) -> None:
    vllm_config, offload_config = _valid_shm_inputs(worker_module, tmp_path)
    offload_config = offload_config.model_copy(update=updates)

    with pytest.raises(worker_module.VllmFlexTensorV2Error, match=match):
        worker_module._validate_shm_worker(vllm_config, offload_config)


def test_timing_batch_env_is_registered_and_validated(worker_module, monkeypatch) -> None:
    from flextensor.config import _REGISTERED_ENV_VARS

    assert "FT_VLLM_TIMING_BATCH" in _REGISTERED_ENV_VARS
    monkeypatch.delenv("FT_VLLM_TIMING_BATCH", raising=False)
    assert worker_module.inference_profile.timing_batch_from_env() == "decode"
    monkeypatch.setenv("FT_VLLM_TIMING_BATCH", "")
    assert worker_module.inference_profile.timing_batch_from_env() is None
    monkeypatch.setenv("FT_VLLM_TIMING_BATCH", "decode")
    assert worker_module.inference_profile.timing_batch_from_env() == "decode"
    monkeypatch.setenv("FT_VLLM_TIMING_BATCH", "prefill")
    assert worker_module.inference_profile.timing_batch_from_env() == "prefill"
    monkeypatch.setenv("FT_VLLM_TIMING_BATCH", "prompt")
    with pytest.raises(RuntimeError, match=r"decode.*prefill"):
        worker_module.inference_profile.timing_batch_from_env()


def test_disabled_config_calls_only_vllm_load(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    monkeypatch.setattr(
        worker_module,
        "load_config",
        lambda **_kwargs: OffloadConfig(enabled=False, pinned_memory=False),
    )
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: pytest.fail("disabled worker checked vLLM version"))
    monkeypatch.setattr(worker_module, "_offloader_api", lambda: pytest.fail("disabled worker imported offloader API"))

    worker.load_model()

    assert events == ["vllm-load-model"]


def test_disabled_config_forwards_dummy_weight_loading(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    monkeypatch.setattr(
        worker_module,
        "load_config",
        lambda **_kwargs: OffloadConfig(enabled=False, pinned_memory=False),
    )

    worker.load_model(load_dummy_weights=True)

    assert events == ["vllm-load-model"]
    assert worker._load_dummy_weights is True


def test_offloader_api_runtime_annotation_resolves(worker_module) -> None:
    get_offloader, set_offloader = worker_module._offloader_api()

    assert callable(get_offloader)
    assert callable(set_offloader)


def test_worker_uses_vllm_namespaced_logger(worker_module) -> None:
    assert worker_module._test_initialized_logger_names.count("vllm.flextensor.v2.inference_profile") == 1
    assert worker_module._test_initialized_logger_names.count("vllm.flextensor.v2.worker") == 1
    assert worker_module.LOGGER.name == "vllm.flextensor.v2.worker"


@pytest.mark.parametrize("version", ["0.11.1", "0.16.5"])
def test_enabled_worker_rejects_old_vllm_before_bootstrap(worker_module, monkeypatch, version):
    events: list[str] = []
    worker = _worker(worker_module, events)
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: version)

    with pytest.raises(RuntimeError, match=r"requires vLLM >= 0\.17\.0"):
        worker.load_model()

    assert events == []


def test_stock_compile_rejected_before_bootstrap(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events, compilation_mode=worker_module.CompilationMode.STOCK_TORCH_COMPILE)
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")

    with pytest.raises(RuntimeError, match="STOCK_TORCH_COMPILE"):
        worker.load_model()

    assert events == []


def test_enabled_worker_reports_missing_offloader_api_before_bootstrap(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.17.0")
    monkeypatch.delattr(sys.modules["vllm.model_executor.offloader.base"], "get_offloader")

    with pytest.raises(RuntimeError, match="offloader singleton API is unavailable"):
        worker.load_model()

    assert events == []


@pytest.mark.parametrize(
    "speculative_config",
    [
        pytest.param(None, id="disabled"),
        pytest.param(SimpleNamespace(method="ngram"), id="ngram"),
        pytest.param(SimpleNamespace(method="ngram_gpu"), id="ngram-gpu"),
        pytest.param(SimpleNamespace(method="suffix"), id="suffix"),
        pytest.param(SimpleNamespace(method="custom_class"), id="custom-class"),
    ],
)
def test_model_free_speculative_config_is_accepted(worker_module, monkeypatch, speculative_config) -> None:
    worker = _worker(worker_module, [])
    worker.vllm_config.speculative_config = speculative_config
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")

    worker_module._validate_enabled_worker(
        worker.vllm_config,
        OffloadConfig(pinned_memory=False),
    )


@pytest.mark.parametrize(
    "speculative_config",
    [
        pytest.param(SimpleNamespace(), id="missing-method"),
        pytest.param(SimpleNamespace(method="draft_model"), id="draft-model"),
        pytest.param(SimpleNamespace(method="eagle"), id="eagle"),
        pytest.param(SimpleNamespace(method="mtp"), id="mtp"),
        pytest.param(SimpleNamespace(method="extract_hidden_states"), id="extract-hidden-states"),
        pytest.param(SimpleNamespace(method="future_method"), id="future-method"),
    ],
)
def test_model_backed_speculative_config_fails_before_bootstrap(
    worker_module,
    monkeypatch,
    speculative_config,
) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker.vllm_config.speculative_config = speculative_config
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    monkeypatch.setattr(
        worker_module,
        "_offloader_api",
        lambda: pytest.fail("unsupported speculative config reached the offloader API"),
    )

    with pytest.raises(
        worker_module.VllmFlexTensorV2Error,
        match=r"model-backed speculative loading.*single-root",
    ) as caught:
        worker.load_model()

    assert f"method={getattr(speculative_config, 'method', None)!r}" in str(caught.value)
    assert events == []


def _compile_config(mode=3, **updates):
    values = {
        "mode": mode,
        "use_inductor_graph_partition": False,
        "splitting_ops": ["vllm::attention", "vllm::unified_attention"],
        "_attention_ops": ["vllm::attention", "vllm::unified_attention"],
    }
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("cudagraph_mode", "expected_timing"),
    [
        pytest.param("NONE", "eager", id="without-cuda-graphs"),
        pytest.param("FULL_AND_PIECEWISE", "cuda_graph", id="with-cuda-graphs"),
    ],
)
def test_safe_vllm_compile_derives_flextensor_compile_and_timing_settings(
    worker_module,
    monkeypatch,
    cudagraph_mode,
    expected_timing,
):
    events: list[str] = []
    compile_mode = worker_module.CompilationMode.VLLM_COMPILE
    worker = _worker(worker_module, events, compilation_mode=compile_mode)
    worker.vllm_config.compilation_config = _compile_config(
        compile_mode,
        cudagraph_mode=worker_module.CUDAGraphMode[cudagraph_mode],
    )
    monkeypatch.setattr(
        worker_module,
        "load_config",
        lambda **_kwargs: OffloadConfig(
            external_compile=False,
            offload_timing="off",
            transfer_mode="allocation_block_transfer",
            pinned_memory=False,
        ),
    )
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    worker_module._test_external_compile = True
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    assert worker.model_runner.model is worker_module._test_proxy
    assert worker._offload_config.external_compile is True
    assert worker._offload_config.offload_timing == expected_timing
    warnings = [message for level, message in worker_module._test_logger_records if level == "warning"]
    assert warnings == [
        "worker v2 overrides explicit OffloadConfig.external_compile=False "
        "with True derived from vLLM compilation mode",
        f"worker v2 overrides explicit OffloadConfig.offload_timing='off' with {expected_timing!r} "
        "derived from vLLM CUDA-graph mode",
    ]


def test_eager_vllm_derives_eager_flextensor_settings(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    monkeypatch.setattr(
        worker_module,
        "load_config",
        lambda **_kwargs: OffloadConfig(
            external_compile=True,
            offload_timing="cuda_graph",
            pinned_memory=False,
        ),
    )
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    assert worker._offload_config.external_compile is False
    assert worker._offload_config.offload_timing == "eager"
    warnings = [message for level, message in worker_module._test_logger_records if level == "warning"]
    assert warnings == [
        "worker v2 overrides explicit OffloadConfig.external_compile=True "
        "with False derived from vLLM compilation mode",
        "worker v2 overrides explicit OffloadConfig.offload_timing='cuda_graph' with 'eager' "
        "derived from vLLM CUDA-graph mode",
    ]


def test_worker_v2_rejects_strategy_transfer_before_model_load(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    monkeypatch.setattr(
        worker_module,
        "load_config",
        lambda **_kwargs: OffloadConfig(
            offload_timing="off",
            transfer_mode="strategy",
            pinned_memory=False,
        ),
    )

    with pytest.raises(worker_module.VllmFlexTensorV2Error, match="requires a block transfer_mode"):
        worker.load_model()

    assert events == []


def test_worker_does_not_warn_for_matching_explicit_settings(worker_module, monkeypatch):
    events: list[str] = []
    compile_mode = worker_module.CompilationMode.VLLM_COMPILE
    worker = _worker(worker_module, events, compilation_mode=compile_mode)
    worker.vllm_config.compilation_config = _compile_config(
        compile_mode,
        cudagraph_mode=worker_module.CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    monkeypatch.setattr(
        worker_module,
        "load_config",
        lambda **_kwargs: OffloadConfig(
            external_compile=True,
            offload_timing="cuda_graph",
            transfer_mode="allocation_block_transfer",
            pinned_memory=False,
        ),
    )
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    worker_module._test_external_compile = True
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    assert not [record for record in worker_module._test_logger_records if record[0] == "warning"]


def test_vllm_compile_allows_other_model_architectures(worker_module, monkeypatch):
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    vllm_config = worker_module.VllmConfig(
        compilation_config=_compile_config(),
        model_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        parallel_config=SimpleNamespace(enable_elastic_ep=False, use_ubatching=False),
    )
    offload_config = OffloadConfig(
        external_compile=True,
        pinned_memory=False,
        transfer_mode="allocation_block_transfer",
    )

    worker_module._validate_enabled_worker(vllm_config, offload_config)


def test_vllm_compile_transfer_mode_error_lists_configured_and_allowed_modes(worker_module):
    offload_config = OffloadConfig.model_construct(transfer_mode="future_transfer_mode")

    with pytest.raises(RuntimeError) as error:
        worker_module._validate_vllm_compile_topology(
            worker_module.VllmConfig(compilation_config=_compile_config()),
            offload_config,
        )

    message = str(error.value)
    assert "future_transfer_mode" in message
    assert "allocation_block_transfer" in message
    assert "raw_block_transfer" in message


def test_enabled_worker_rejects_string_compilation_mode(worker_module, monkeypatch):
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    vllm_config = worker_module.VllmConfig(
        compilation_config=_compile_config(mode="VLLM_COMPILE"),
        model_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"]),
    )
    offload_config = OffloadConfig(
        external_compile=True,
        pinned_memory=False,
        transfer_mode="allocation_block_transfer",
    )

    with pytest.raises(RuntimeError, match="CompilationMode"):
        worker_module._validate_enabled_worker(vllm_config, offload_config)


@pytest.mark.parametrize(
    ("compilation_config", "transfer_mode", "external_compile", "message"),
    [
        (
            _compile_config(),
            "strategy",
            True,
            "requires a block transfer_mode",
        ),
        (
            _compile_config(use_inductor_graph_partition=True),
            "allocation_block_transfer",
            True,
            "use_inductor_graph_partition=False",
        ),
        (
            _compile_config(),
            "allocation_block_transfer",
            False,
            "requires OffloadConfig.external_compile=True",
        ),
        (
            SimpleNamespace(
                mode=3,
                splitting_ops=["vllm::attention"],
                _attention_ops=["vllm::attention"],
            ),
            "allocation_block_transfer",
            True,
            "must expose use_inductor_graph_partition=False",
        ),
        (
            _compile_config(splitting_ops=[]),
            "allocation_block_transfer",
            True,
            "non-empty resolved splitting_ops",
        ),
        (
            _compile_config(_attention_ops=[]),
            "allocation_block_transfer",
            True,
            "must expose a non-empty resolved _attention_ops",
        ),
        (
            _compile_config(splitting_ops=["vllm::attention"]),
            "allocation_block_transfer",
            True,
            "must include every vLLM attention op",
        ),
        (SimpleNamespace(mode=0), "allocation_block_transfer", True, "requires CompilationMode.VLLM_COMPILE"),
    ],
)
def test_unsafe_vllm_compile_topology_is_rejected(
    worker_module,
    monkeypatch,
    compilation_config,
    transfer_mode,
    external_compile,
    message,
):
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    config_type = OffloadConfig.model_construct if transfer_mode == "strategy" else OffloadConfig
    offload_config = config_type(external_compile=external_compile, pinned_memory=False, transfer_mode=transfer_mode)

    with pytest.raises(RuntimeError, match=message):
        worker_module._validate_enabled_worker(
            worker_module.VllmConfig(compilation_config=compilation_config),
            offload_config,
        )


def test_dynamo_trace_once_rejected_before_topology_validation(worker_module, monkeypatch):
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    offload_config = OffloadConfig(pinned_memory=False)

    with pytest.raises(RuntimeError, match=r"only supports CompilationMode.NONE or CompilationMode.VLLM_COMPILE"):
        worker_module._validate_enabled_worker(
            worker_module.VllmConfig(
                compilation_config=SimpleNamespace(mode=worker_module.CompilationMode.DYNAMO_TRACE_ONCE)
            ),
            offload_config,
        )


@pytest.mark.parametrize(
    ("config", "expected_includes", "expected_excludes"),
    [
        (
            OffloadConfig(enabled=True, pinned_memory=False, include_patterns=["model.layers.*"]),
            ["model.layers.*"],
            None,
        ),
        (
            OffloadConfig(enabled=True, pinned_memory=False, exclude_patterns=["model.layers.0"]),
            None,
            ["model.layers.0"],
        ),
        (
            OffloadConfig(enabled=True, pinned_memory=False, exclude_patterns=[]),
            None,
            [],
        ),
    ],
)
def test_enabled_worker_resolves_custom_selectors_before_bootstrap(
    worker_module,
    monkeypatch,
    config,
    expected_includes,
    expected_excludes,
):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    monkeypatch.setattr(worker_module.inference_profile, "load_saved_profile", lambda _config: None)
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    resolved = worker_module._test_takeover_config
    assert resolved.include_patterns == (
        VLLM_DEFAULT_INCLUDE_PATTERNS if expected_includes is None else expected_includes
    )
    assert resolved.exclude_patterns == (
        VLLM_DEFAULT_EXCLUDE_PATTERNS if expected_excludes is None else expected_excludes
    )


def test_enabled_worker_rejects_native_weight_transfer_before_bootstrap(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker.vllm_config.weight_transfer_config = object()
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    monkeypatch.setattr(
        worker_module,
        "_offloader_api",
        lambda: pytest.fail("weight transfer reached bootstrap setup"),
    )

    with pytest.raises(RuntimeError, match="weight_transfer_config"):
        worker.load_model()

    assert events == []


@pytest.mark.parametrize(("enable_dbo", "ubatch_size"), [(True, 0), (False, 2)])
def test_enabled_worker_rejects_ubatching_before_bootstrap(
    worker_module,
    monkeypatch,
    enable_dbo,
    ubatch_size,
):
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker.vllm_config.parallel_config = SimpleNamespace(
        enable_elastic_ep=False,
        enable_dbo=enable_dbo,
        ubatch_size=ubatch_size,
        use_ubatching=enable_dbo or ubatch_size > 1,
    )
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    monkeypatch.setattr(
        worker_module,
        "_offloader_api",
        lambda: pytest.fail("ubatching reached bootstrap setup"),
    )

    with pytest.raises(RuntimeError, match="use_ubatching"):
        worker.load_model()

    assert events == []


@pytest.mark.parametrize(
    ("load_dummy_weights", "enable_elastic_ep", "message"),
    [(True, False, "load_dummy_weights"), (False, True, "enable_elastic_ep")],
)
def test_enabled_worker_rejects_elastic_ep_loading_before_bootstrap(
    worker_module,
    monkeypatch,
    load_dummy_weights,
    enable_elastic_ep,
    message,
):
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker.vllm_config.parallel_config.enable_elastic_ep = enable_elastic_ep
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    monkeypatch.setattr(
        worker_module,
        "_offloader_api",
        lambda: pytest.fail("elastic EP reached bootstrap setup"),
    )

    with pytest.raises(RuntimeError, match=message):
        worker.load_model(load_dummy_weights=load_dummy_weights)

    assert events == []

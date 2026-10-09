# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from flextensor.config import OffloadConfig
from flextensor.contrib.vllm._patterns import VLLM_DEFAULT_EXCLUDE_PATTERNS, VLLM_DEFAULT_INCLUDE_PATTERNS
from flextensor.shared_offload import SharedOffloadSession

from ._v2_worker_test_utils import _install_bootstrap_fakes, _worker


def test_load_model_passes_explicit_logical_device_index(worker_module, monkeypatch) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker.device = torch.device("cuda:3")
    captured = {}

    def load_config(**kwargs):
        captured.update(kwargs)
        return OffloadConfig(enabled=False, pinned_memory=False)

    monkeypatch.setattr(worker_module, "load_config", load_config)
    worker.load_model()

    assert captured == {"gpu_device": 3}
    assert events == ["vllm-load-model"]


def test_load_model_rejects_unindexed_worker_device(worker_module) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker.device = torch.device("cuda")

    with pytest.raises(worker_module.VllmFlexTensorV2Error, match="explicit index"):
        worker.load_model()

    assert events == []


def test_load_saved_profile_uses_existing_profile_json(worker_module, monkeypatch, tmp_path) -> None:
    profile = object.__new__(worker_module.inference_profile.TensorManagerState)
    captured = []
    monkeypatch.setattr(
        worker_module.inference_profile.TensorManagerStateHandler,
        "load_from_file",
        lambda path: captured.append(path) or profile,
    )

    result = worker_module.inference_profile.load_saved_profile(
        OffloadConfig(profile_storage_dir=str(tmp_path), profile_read_only=True, pinned_memory=False)
    )

    assert result is profile
    assert captured == [tmp_path / "profile.json"]


def test_missing_saved_profile_uses_conservative_fallback(worker_module, tmp_path) -> None:
    result = worker_module.inference_profile.load_saved_profile(
        OffloadConfig(profile_storage_dir=str(tmp_path), pinned_memory=False)
    )

    assert result is None
    assert any(
        level == "warning" and message.startswith("saved profile not found")
        for level, message in worker_module._test_logger_records
    )


def test_malformed_saved_profile_shape_uses_conservative_fallback(worker_module, tmp_path) -> None:
    (tmp_path / "profile.json").write_text(
        '{"version": 3, "tensor_id_to_name_map": []}',
        encoding="utf-8",
    )

    result = worker_module.inference_profile.load_saved_profile(
        OffloadConfig(profile_storage_dir=str(tmp_path), pinned_memory=False)
    )

    assert result is None
    assert any(
        level == "warning" and message.startswith("saved profile at")
        for level, message in worker_module._test_logger_records
    )


def test_load_model_passes_saved_profile_to_takeover(worker_module, monkeypatch) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    config = OffloadConfig(
        profile_storage_dir="/profiles",
        profile_read_only=True,
        pinned_memory=False,
    )
    profile = object()
    loaded_configs = []
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(
        worker_module.inference_profile,
        "load_saved_profile",
        lambda actual_config: loaded_configs.append(actual_config) or profile,
    )
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    assert loaded_configs == [worker._offload_config]
    assert worker._offload_config.profile_storage_dir == config.profile_storage_dir
    assert worker_module._test_takeover_profile is profile
    assert worker._flextensor_bootstrap_offloader is worker_module._test_bootstrap


@pytest.mark.parametrize("offloader_name", ["UVAOffloader", "PrefetchOffloader", "BaseOffloader"])
def test_load_model_rejects_active_native_offloader_before_model_construction(
    worker_module,
    monkeypatch,
    offloader_name,
) -> None:
    from vllm.model_executor.offloader import base as offloader_base

    events: list[str] = []
    worker = _worker(worker_module, events)
    active_offloader = getattr(offloader_base, offloader_name)()
    set_calls = []
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: OffloadConfig(pinned_memory=False))
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    monkeypatch.setattr(worker_module.inference_profile, "load_saved_profile", lambda _config: None)
    monkeypatch.setattr(worker_module, "_offloader_api", lambda: (lambda: active_offloader, set_calls.append))
    monkeypatch.setattr(
        worker_module,
        "VllmBootstrapOffloader",
        lambda: pytest.fail("bootstrap construction must not run for a native offloader conflict"),
    )

    with pytest.raises(worker_module.VllmFlexTensorV2Error, match=offloader_name):
        worker.load_model()

    assert set_calls == []
    assert events == []


def test_load_model_enables_writable_refresh_with_derived_timing(worker_module, monkeypatch) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    config = OffloadConfig(
        profile_storage_dir="/profiles",
        offload_timing="off",
        transfer_mode="allocation_block_transfer",
        profiling_iters=3,
        pinned_memory=False,
    )
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(worker_module.inference_profile, "load_saved_profile", lambda _config: None)
    monkeypatch.setenv("FT_VLLM_TIMING_BATCH", "decode")
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    assert worker._flextensor_profile_refresh_enabled
    assert worker._flextensor_timing_batch == "decode"
    assert worker._flextensor_profile_sample_target == 3
    assert worker._flextensor_profile_sample_count == 0
    assert worker._offload_config.offload_timing == "eager"


@pytest.mark.parametrize(
    ("profile_storage_dir", "profile_read_only", "timing_batch"),
    [
        (None, False, "decode"),
        ("/profiles", True, "decode"),
        ("/profiles", False, ""),
    ],
)
def test_load_model_disables_refresh_without_every_required_setting(
    worker_module,
    monkeypatch,
    profile_storage_dir,
    profile_read_only,
    timing_batch,
) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    config = OffloadConfig(
        profile_storage_dir=profile_storage_dir,
        profile_read_only=profile_read_only,
        transfer_mode="allocation_block_transfer",
        pinned_memory=False,
    )
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(worker_module.inference_profile, "load_saved_profile", lambda _config: None)
    if timing_batch is None:
        monkeypatch.delenv("FT_VLLM_TIMING_BATCH", raising=False)
    else:
        monkeypatch.setenv("FT_VLLM_TIMING_BATCH", timing_batch)
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    assert not worker._flextensor_profile_refresh_enabled


def test_load_model_constructs_once_and_adopts_state_before_return(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)

    def publish(model_runner, raw_model, actual_proxy):
        assert model_runner is worker.model_runner
        assert raw_model is model_runner.model
        assert actual_proxy is worker_module._test_proxy
        events.append("publish-proxy")
        model_runner.model = actual_proxy

    monkeypatch.setattr(
        worker_module,
        "flextensor",
        SimpleNamespace(release=lambda: events.append("release-flextensor")),
        raising=False,
    )
    monkeypatch.setattr(worker_module, "__version__", "test-version")
    monkeypatch.setattr(worker_module, "_publish_model_to_runner", publish, raising=False)
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None), raising=False)

    worker.load_model()

    assert events == [
        "get-previous-offloader",
        "set-bootstrap-offloader",
        "vllm-load-model",
        "get-raw-model",
        "start-takeover-memory-profile",
        "bootstrap-post-init",
        "build-state",
        "offload-from-state",
        "stop-takeover-memory-profile",
        "publish-proxy",
    ]
    assert events.count("set-bootstrap-offloader") == 1
    assert worker_module._test_singleton.value is worker_module._test_bootstrap
    assert worker.model_runner.model is worker_module._test_proxy
    assert worker.model_runner.model_memory_usage == 323
    assert worker._offload_config.include_patterns == VLLM_DEFAULT_INCLUDE_PATTERNS
    assert worker._offload_config.exclude_patterns == VLLM_DEFAULT_EXCLUDE_PATTERNS
    assert worker._offload_config.external_compile is False
    assert (
        "info",
        f"FlexTensor test-version offloading enabled with config: {worker._offload_config}",
    ) in worker_module._test_logger_records
    assert ("info", "FlexTensor vLLM integration v2 state takeover complete") in worker_module._test_logger_records


def _configure_local_dp_worker(worker) -> None:
    worker.vllm_config.load_config = SimpleNamespace(load_format="auto")
    worker.vllm_config.lora_config = None
    worker.vllm_config.speculative_config = None
    worker.vllm_config.model_config.quantization = None
    worker.vllm_config.parallel_config = SimpleNamespace(
        data_parallel_backend="mp",
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
        data_parallel_index=0,
        data_parallel_rank_local=0,
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
    )


def _shm_offload_config(profile_dir) -> OffloadConfig:
    return OffloadConfig(
        profile_storage_dir=str(profile_dir),
        profile_read_only=True,
        transfer_mode="allocation_block_transfer",
        shm_enabled=True,
        shm_namespace="vllm-test-run",
        pinned_memory=False,
    )


@pytest.mark.parametrize("shm_role", [None, "creator", "follower"], ids=["regular", "shm-creator", "shm-follower"])
def test_load_model_joins_offloader_before_breakable_capture_ends(
    worker_module, monkeypatch, tmp_path, shm_role
) -> None:
    from vllm.model_executor.offloader import base as offloader_base

    events: list[str] = []
    worker = _worker(worker_module, events)
    worker.vllm_config.compilation_config.cudagraph_mode = worker_module.CUDAGraphMode.FULL_AND_PIECEWISE
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setattr(
        worker_module,
        "atexit",
        SimpleNamespace(register=lambda _callback: None, unregister=lambda _callback: None),
    )
    monkeypatch.setattr(worker_module.flextensor, "release", lambda: None)

    if shm_role is not None:
        _configure_local_dp_worker(worker)
        raw_model = worker.model_runner.model
        worker_module._test_proxy._get_model = lambda: raw_model
        config = _shm_offload_config(tmp_path)
        monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)

        @contextmanager
        def use_context(context):
            context.proxy = worker_module._test_proxy
            context.role = shm_role
            yield

        monkeypatch.setattr(
            worker_module, "_shared_loader_api", lambda: ("flextensor_shm", SimpleNamespace, use_context)
        )

    class BreakableCUDAGraphCapture:
        _capturing = True

        def add_eager(self, fn):
            events.append("capture-end")
            self._capturing = False
            result = fn()
            self._capturing = True
            events.append("capture-begin")
            return result

    def join_after_forward(offloader):
        assert offloader is worker_module._test_bootstrap
        assert worker.model_runner.model is worker_module._test_proxy
        events.append("offloader-join")

    breakable = ModuleType("vllm.compilation.breakable_cudagraph")
    breakable.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
    monkeypatch.setitem(sys.modules, breakable.__name__, breakable)
    monkeypatch.setattr(offloader_base, "get_offloader", lambda: worker_module._test_singleton.value)
    monkeypatch.setattr(worker_module.VllmBootstrapOffloader, "join_after_forward", join_after_forward, raising=False)

    worker.load_model()

    assert worker.model_runner.model is worker_module._test_proxy
    result = BreakableCUDAGraphCapture().add_eager(lambda: events.append("eager") or "attention-output")

    assert result == "attention-output"
    assert events[-4:] == ["offloader-join", "capture-end", "eager", "capture-begin"]
    worker.shutdown()


@pytest.mark.parametrize("compilation_mode", ["NONE", "VLLM_COMPILE"])
@pytest.mark.parametrize("cudagraph_mode", ["NONE", "FULL_AND_PIECEWISE"])
@pytest.mark.parametrize("shm_wait_timeout", [None, 30.0], ids=["default-timeout", "bounded-timeout"])
def test_shm_load_uses_registered_loader_adopts_session_and_session_owned_shutdown(
    worker_module,
    monkeypatch,
    tmp_path,
    compilation_mode,
    cudagraph_mode,
    shm_wait_timeout,
) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    _configure_local_dp_worker(worker)
    worker.vllm_config.compilation_config = SimpleNamespace(
        mode=worker_module.CompilationMode[compilation_mode],
        cudagraph_mode=worker_module.CUDAGraphMode[cudagraph_mode],
        use_inductor_graph_partition=False,
        splitting_ops=["vllm::attention"],
        _attention_ops=["vllm::attention"],
    )
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    config = _shm_offload_config(tmp_path)
    if shm_wait_timeout is not None:
        config = config.model_copy(update={"shm_wait_timeout": shm_wait_timeout})
    raw_model = worker.model_runner.model
    worker_module._test_proxy._get_model = lambda: raw_model

    class Session(SharedOffloadSession):
        def __init__(self, namespace, *, wait_timeout):
            super().__init__(namespace, wait_timeout=wait_timeout)
            assert (self.namespace, self.wait_timeout) == ("vllm-test-run", config.shm_wait_timeout)
            events.append("create-shared-session")

        def close(self, *, preserve_models=False):
            assert preserve_models is False
            super().close(preserve_models=preserve_models)
            events.append("close-shared-session")

    class Context(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(proxy=None, **kwargs)

    @contextmanager
    def use_context(context):
        assert context.session is worker_module._test_session
        assert context.profile_directory == str(tmp_path)
        assert context.materialize_device == torch.device("cuda:0")
        assert context.underlying_load_format == "auto"
        events.append("enter-shm-loader-context")
        context.proxy = worker_module._test_proxy
        context.role = "follower"
        try:
            yield
        finally:
            events.append("exit-shm-loader-context")

    registered_callbacks = []
    unregistered_callbacks = []

    def make_session(*args, **kwargs):
        session = Session(*args, **kwargs)
        worker_module._test_session = session
        return session

    def publish(model_runner, actual_raw, proxy):
        assert (model_runner, actual_raw, proxy) == (
            worker.model_runner,
            raw_model,
            worker_module._test_proxy,
        )
        events.append("publish-proxy")
        model_runner.model = proxy

    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(worker_module, "SharedOffloadSession", make_session)
    monkeypatch.setattr(
        worker_module,
        "_shared_loader_api",
        lambda: ("flextensor_shm", Context, use_context),
    )
    monkeypatch.setattr(
        worker_module.inference_profile,
        "load_saved_profile",
        lambda _config: pytest.fail("SHM worker must not load the profile before creator election"),
    )
    monkeypatch.setattr(worker_module, "_publish_model_to_runner", publish)
    monkeypatch.setattr(
        worker_module,
        "atexit",
        SimpleNamespace(
            register=lambda callback: registered_callbacks.append(callback),
            unregister=lambda callback: unregistered_callbacks.append(callback),
        ),
    )
    monkeypatch.setattr(
        worker_module.flextensor,
        "release",
        lambda: pytest.fail("SHM worker lifecycle must not use global release"),
    )

    worker.load_model()

    assert worker.vllm_config.load_config.load_format == "auto"
    assert worker.model_runner.model is worker_module._test_proxy
    assert worker.model_runner.model_memory_usage == 200
    assert worker._offload_config.external_compile is (compilation_mode == "VLLM_COMPILE")
    assert worker._offload_config.offload_timing == ("eager" if cudagraph_mode == "NONE" else "cuda_graph")
    assert registered_callbacks == [worker._flextensor_atexit_cleanup]
    assert events == [
        "get-previous-offloader",
        "create-shared-session",
        "set-bootstrap-offloader",
        "enter-shm-loader-context",
        "vllm-load-model",
        "exit-shm-loader-context",
        "get-raw-model",
        "bootstrap-post-init",
        "bind-shared-runtime",
        "publish-proxy",
    ]

    assert worker.compile_or_warm_up_model() == "compilation-times"
    assert events[-1] == "vllm-compile-or-warm-up"

    scheduled = SimpleNamespace(total_num_scheduled_tokens=1)
    worker.execute_model(scheduled)
    worker.execute_model(scheduled)
    assert (
        "info",
        "FlexTensor vLLM SHM execution data_parallel_index=0 role=follower",
    ) in worker_module._test_logger_records
    assert (
        sum("FlexTensor vLLM SHM execution" in message for _level, message in worker_module._test_logger_records) == 1
    )

    worker.shutdown()

    assert unregistered_callbacks == registered_callbacks
    assert not hasattr(worker, "_flextensor_shared_session")
    assert not hasattr(worker, "_flextensor_previous_offloader")
    assert events[-3:] == ["restore-previous-offloader", "close-shared-session", "vllm-shutdown"]


def test_shm_shutdown_retry_never_falls_back_to_global_release(worker_module, monkeypatch) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    previous = object()
    cleanup = object()
    unregister_attempts = 0

    class Session:
        def close(self, *, preserve_models=False):
            assert preserve_models is False
            events.append("close-shared-session")

    worker._flextensor_previous_offloader = previous
    worker._flextensor_shared_lifecycle = True
    worker._flextensor_shared_session = Session()
    worker._flextensor_atexit_cleanup = cleanup

    def set_offloader(offloader):
        assert offloader is previous
        events.append("restore-previous-offloader")

    def unregister(callback):
        nonlocal unregister_attempts
        assert callback is cleanup
        unregister_attempts += 1
        events.append("unregister-atexit")
        if unregister_attempts == 1:
            raise RuntimeError("unregister failed")

    monkeypatch.setattr(worker_module, "_offloader_api", lambda: (lambda: None, set_offloader))
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(unregister=unregister))
    monkeypatch.setattr(
        worker_module.flextensor,
        "release",
        lambda: pytest.fail("shared-session retry must never use global release"),
    )

    with pytest.raises(RuntimeError, match="unregister failed"):
        worker.shutdown()

    assert worker._flextensor_previous_offloader is previous
    assert worker._flextensor_shared_lifecycle is True
    assert isinstance(worker._flextensor_shared_session, Session)

    worker.shutdown()

    assert not hasattr(worker, "_flextensor_previous_offloader")
    assert not hasattr(worker, "_flextensor_shared_session")
    assert events == [
        "restore-previous-offloader",
        "close-shared-session",
        "unregister-atexit",
        "vllm-shutdown",
        "restore-previous-offloader",
        "close-shared-session",
        "unregister-atexit",
        "vllm-shutdown",
    ]


def test_shm_load_failure_restores_format_closes_session_and_restores_singleton(
    worker_module,
    monkeypatch,
    tmp_path,
) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker._failure = "vllm-load-model"
    _configure_local_dp_worker(worker)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)

    class Session:
        def __init__(self, *_args, **_kwargs):
            events.append("create-shared-session")

        def close(self, *, preserve_models=False):
            assert preserve_models is False
            events.append("close-shared-session")

    @contextmanager
    def use_context(_context):
        yield

    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: _shm_offload_config(tmp_path))
    monkeypatch.setattr(worker_module, "SharedOffloadSession", Session)
    monkeypatch.setattr(
        worker_module,
        "_shared_loader_api",
        lambda: ("flextensor_shm", SimpleNamespace, use_context),
    )
    monkeypatch.setattr(
        worker_module.flextensor,
        "release",
        lambda: pytest.fail("SHM worker rollback must not use global release"),
    )

    with pytest.raises(RuntimeError, match="vllm-load-model"):
        worker.load_model()

    assert worker.vllm_config.load_config.load_format == "auto"
    assert events[-2:] == ["close-shared-session", "restore-previous-offloader"]
    assert not hasattr(worker, "_flextensor_shared_session")


def test_shm_load_failure_retains_atexit_retry_when_session_close_fails(
    worker_module,
    monkeypatch,
    tmp_path,
) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker._failure = "vllm-load-model"
    _configure_local_dp_worker(worker)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    close_attempts = 0

    class Session:
        def __init__(self, *_args, **_kwargs):
            events.append("create-shared-session")

        def close(self, *, preserve_models=False):
            nonlocal close_attempts
            assert preserve_models is False
            close_attempts += 1
            events.append("close-shared-session")
            if close_attempts == 1:
                raise RuntimeError("close failed")

    @contextmanager
    def use_context(_context):
        yield

    registered_callbacks = []
    unregistered_callbacks = []
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: _shm_offload_config(tmp_path))
    monkeypatch.setattr(worker_module, "SharedOffloadSession", Session)
    monkeypatch.setattr(
        worker_module,
        "_shared_loader_api",
        lambda: ("flextensor_shm", SimpleNamespace, use_context),
    )
    monkeypatch.setattr(
        worker_module,
        "atexit",
        SimpleNamespace(
            register=lambda callback: registered_callbacks.append(callback),
            unregister=lambda callback: unregistered_callbacks.append(callback),
        ),
    )
    monkeypatch.setattr(
        worker_module.flextensor,
        "release",
        lambda: pytest.fail("SHM worker rollback must not use global release"),
    )

    with pytest.raises(RuntimeError, match="vllm-load-model") as caught:
        worker.load_model()

    notes = getattr(caught.value, "__notes__", ())
    assert not notes or any("retried at process exit" in note for note in notes)
    assert registered_callbacks == [worker._flextensor_atexit_cleanup]
    assert unregistered_callbacks == []
    registered_callbacks[0]()
    assert close_attempts == 2

    worker.shutdown()

    assert close_attempts == 3
    assert unregistered_callbacks == registered_callbacks
    assert not hasattr(worker, "_flextensor_shared_session")


RAW = nn.Module()
PROXY = nn.Module()


def _contains_identity(root: object, target: object) -> bool:
    queue = [root]
    seen: set[int] = set()
    while queue:
        owner = queue.pop(0)
        if owner is target:
            return True
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        for attribute in ("model", "runnable"):
            if hasattr(owner, attribute):
                queue.append(getattr(owner, attribute))
    return False


@pytest.mark.parametrize("shape", ["direct", "nested-model", "nested-runnable"])
def test_publish_replaces_known_runner_reference(worker_module, shape):
    nested = {
        "direct": RAW,
        "nested-model": SimpleNamespace(model=RAW),
        "nested-runnable": SimpleNamespace(runnable=RAW, model=RAW),
    }[shape]
    runner = SimpleNamespace(model=nested)
    worker_module._publish_model_to_runner(runner, RAW, PROXY)

    assert _contains_identity(runner, PROXY)
    assert not _contains_identity(runner, RAW)


def test_publish_accepts_vllm_v2_model_runner(worker_module):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = GPUModelRunner(model=RAW)

    worker_module._publish_model_to_runner(runner, RAW, PROXY)

    assert runner.model is PROXY


def test_publish_rejects_missing_raw_model_without_following_unknown_attributes(worker_module):
    runner = SimpleNamespace(model=SimpleNamespace(other=RAW))

    with pytest.raises(RuntimeError, match="could not publish OffloadModelProxy"):
        worker_module._publish_model_to_runner(runner, RAW, PROXY)

    assert runner.model.other is RAW


def test_publish_rejects_cycle_without_raw_model(worker_module):
    runner = SimpleNamespace()
    runner.model = runner

    with pytest.raises(RuntimeError, match="could not publish OffloadModelProxy"):
        worker_module._publish_model_to_runner(runner, RAW, PROXY)


def _fail_at(events: list[str], boundary: str):
    events.append(boundary)
    raise RuntimeError(boundary)


@pytest.mark.parametrize(
    "boundary",
    [
        "bootstrap-constructor",
        "set-bootstrap-offloader",
        "vllm-load-model",
        "get-raw-model",
        "bootstrap-post-init",
        "build-state",
        "offload-from-state",
    ],
)
def test_pre_takeover_failure_restores_singleton_without_releasing_manager(
    worker_module,
    monkeypatch,
    boundary,
):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    worker._failure = boundary
    worker_module._test_failure = boundary

    fake_flextensor = SimpleNamespace(release=lambda: events.append("release-flextensor"))
    monkeypatch.setattr(worker_module, "flextensor", fake_flextensor)
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    with pytest.raises(RuntimeError, match=boundary):
        worker.load_model()

    assert events[-1] == "restore-previous-offloader"
    assert "release-flextensor" not in events


@pytest.mark.parametrize(
    "boundary",
    ["publish-proxy", "register-atexit"],
)
def test_post_takeover_failure_releases_manager_before_restoring_singleton(
    worker_module,
    monkeypatch,
    boundary,
):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    active_owner = SimpleNamespace(value=None)
    real_publish = worker_module._publish_model_to_runner
    worker_module._test_failure = boundary

    def activate(_model, _config):
        active_owner.value = worker_module._test_proxy
        return worker_module._test_proxy

    worker_module._test_takeover_callback = activate

    def release():
        assert active_owner.value is worker_module._test_proxy
        events.append("release-flextensor")
        active_owner.value = None

    def publish(model_runner, raw_model, actual_proxy):
        events.append("publish-proxy")
        if boundary == "publish-proxy":
            raise RuntimeError(boundary)
        real_publish(model_runner, raw_model, actual_proxy)

    def register(_callback):
        events.append("register-atexit")
        if boundary == "register-atexit":
            raise RuntimeError(boundary)

    monkeypatch.setattr(
        worker_module,
        "flextensor",
        SimpleNamespace(release=release),
    )
    monkeypatch.setattr(worker_module, "_publish_model_to_runner", publish)
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=register))

    with pytest.raises(RuntimeError, match=boundary):
        worker.load_model()

    assert events[-2:] == ["release-flextensor", "restore-previous-offloader"]
    assert active_owner.value is None


def test_profiler_exit_failure_releases_takeover_before_restoring_singleton(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    active_owner = SimpleNamespace(value=None)

    class FailingDeviceMemoryProfiler:
        consumed_memory = 123

        def __enter__(self):
            events.append("start-takeover-memory-profile")
            return self

        def __exit__(self, _exc_type, _exc_val, _exc_tb):
            events.append("stop-takeover-memory-profile")
            raise RuntimeError("profiler-exit")

    def activate(_model, _config):
        active_owner.value = worker_module._test_proxy
        return worker_module._test_proxy

    worker_module._test_takeover_callback = activate

    def release():
        assert active_owner.value is worker_module._test_proxy
        events.append("release-flextensor")
        active_owner.value = None

    monkeypatch.setattr(worker_module, "DeviceMemoryProfiler", lambda _device: FailingDeviceMemoryProfiler())
    monkeypatch.setattr(
        worker_module,
        "flextensor",
        SimpleNamespace(release=release),
    )

    with pytest.raises(RuntimeError, match="profiler-exit"):
        worker.load_model()

    assert events[-2:] == ["release-flextensor", "restore-previous-offloader"]
    assert active_owner.value is None


def test_release_failure_still_restores_previous_singleton(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    worker_module._test_takeover_callback = lambda _model, _config: worker_module._test_proxy
    monkeypatch.setattr(
        worker_module,
        "flextensor",
        SimpleNamespace(
            release=lambda: _fail_at(events, "release-failed"),
        ),
    )
    monkeypatch.setattr(worker_module, "_publish_model_to_runner", lambda *args: _fail_at(events, "publish-failed"))

    with pytest.raises(RuntimeError, match="release-failed"):
        worker.load_model()

    assert events[-1] == "restore-previous-offloader"


def test_shutdown_restores_previous_singleton_before_release_and_vllm_shutdown(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    previous = object()
    worker._flextensor_previous_offloader = previous
    monkeypatch.setattr(
        worker_module,
        "_offloader_api",
        lambda: (
            lambda: pytest.fail("shutdown must not read the current singleton"),
            lambda offloader: (
                events.append("restore-previous-offloader")
                if offloader is previous
                else pytest.fail("wrong singleton restored")
            ),
        ),
    )
    monkeypatch.setattr(worker_module.flextensor, "release", lambda: events.append("release-flextensor"))

    worker.shutdown()

    assert events == ["restore-previous-offloader", "release-flextensor", "vllm-shutdown"]


def test_shutdown_keeps_previous_offloader_for_retry_when_release_fails(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    previous = object()
    worker._flextensor_previous_offloader = previous
    monkeypatch.setattr(
        worker_module,
        "_offloader_api",
        lambda: (
            lambda: pytest.fail("shutdown must not read the current singleton"),
            lambda offloader: (
                events.append("restore-previous-offloader")
                if offloader is previous
                else pytest.fail("wrong singleton restored")
            ),
        ),
    )
    release_attempts = 0

    def release():
        nonlocal release_attempts
        release_attempts += 1
        events.append("release-flextensor")
        if release_attempts == 1:
            raise RuntimeError("release failed")

    monkeypatch.setattr(worker_module.flextensor, "release", release)

    with pytest.raises(RuntimeError, match="release failed"):
        worker.shutdown()

    assert worker._flextensor_previous_offloader is previous

    worker.shutdown()

    assert not hasattr(worker, "_flextensor_previous_offloader")
    assert events == [
        "restore-previous-offloader",
        "release-flextensor",
        "vllm-shutdown",
        "restore-previous-offloader",
        "release-flextensor",
        "vllm-shutdown",
    ]


def test_shm_shutdown_preserves_restore_error_and_retries_failed_close(worker_module, monkeypatch) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    previous = object()
    restore_attempts = 0
    close_attempts = 0

    class Session:
        def close(self, *, preserve_models=False):
            nonlocal close_attempts
            assert preserve_models is False
            close_attempts += 1
            events.append("close-shared-session")
            if close_attempts == 1:
                raise RuntimeError("close failed")

    session = Session()
    cleanup = session.close
    worker._flextensor_previous_offloader = previous
    worker._flextensor_shared_lifecycle = True
    worker._flextensor_shared_session = session
    worker._flextensor_atexit_cleanup = cleanup

    def set_offloader(offloader):
        nonlocal restore_attempts
        assert offloader is previous
        restore_attempts += 1
        events.append("restore-previous-offloader")
        if restore_attempts == 1:
            raise RuntimeError("restore failed")

    unregistered_callbacks = []
    monkeypatch.setattr(worker_module, "_offloader_api", lambda: (lambda: None, set_offloader))
    monkeypatch.setattr(
        worker_module,
        "atexit",
        SimpleNamespace(unregister=lambda callback: unregistered_callbacks.append(callback)),
    )
    monkeypatch.setattr(
        worker_module.flextensor,
        "release",
        lambda: pytest.fail("shared-session retry must never use global release"),
    )

    with pytest.raises(RuntimeError, match="restore failed") as caught:
        worker.shutdown()

    notes = getattr(caught.value, "__notes__", ())
    assert not notes or any("closing the FlexTensor runtime" in note and "close failed" in note for note in notes)
    assert worker._flextensor_previous_offloader is previous
    assert worker._flextensor_shared_session is session
    assert unregistered_callbacks == []

    worker.shutdown()

    assert restore_attempts == 2
    assert close_attempts == 2
    assert unregistered_callbacks == [cleanup]
    assert not hasattr(worker, "_flextensor_previous_offloader")
    assert not hasattr(worker, "_flextensor_shared_session")


def test_shared_integration_helper_defaults_to_v2():
    from tests.integration._vllm_server import (
        FLEXTENSOR_OFFLOAD_WORKER_CLS,
        VllmOffloadSmokeCase,
    )

    case = VllmOffloadSmokeCase("model", "output").with_flextensor_offload()
    assert FLEXTENSOR_OFFLOAD_WORKER_CLS == ("flextensor.contrib.vllm.worker.FlexTensorOffloadWorker")
    assert FLEXTENSOR_OFFLOAD_WORKER_CLS in case.cli_args

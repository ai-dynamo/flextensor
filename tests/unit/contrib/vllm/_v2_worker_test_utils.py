# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


def _worker(worker_module, events: list[str], compilation_mode: object | None = None):
    if compilation_mode is None:
        compilation_mode = worker_module.CompilationMode.NONE
    worker = worker_module.FlexTensorOffloadWorker.__new__(worker_module.FlexTensorOffloadWorker)
    worker._events = events
    worker._failure = None
    worker.device = torch.device("cuda:0")
    worker.vllm_config = worker_module.VllmConfig(
        compilation_config=SimpleNamespace(
            mode=compilation_mode,
            cudagraph_mode=worker_module.CUDAGraphMode.NONE,
        ),
        model_config=SimpleNamespace(
            architecture="Qwen2ForCausalLM",
            architectures=["Qwen2ForCausalLM"],
            convert_type="none",
            runner_type="generate",
        ),
        parallel_config=SimpleNamespace(enable_elastic_ep=False, use_ubatching=False),
    )
    raw_model = nn.Module()
    model_runner = SimpleNamespace(model=raw_model, model_memory_usage=200)

    def get_model():
        events.append("get-raw-model")
        if worker._failure == "get-raw-model":
            raise RuntimeError("get-raw-model")
        return model_runner.model

    model_runner.get_model = get_model
    worker.model_runner = model_runner
    worker_module._test_events = events
    return worker


# ruff: ignore[noqa-comments] - compatibility with the pre-commit Ruff version.
def _install_bootstrap_fakes(  # noqa: C901
    worker_module,
    monkeypatch,
    events: list[str],
):
    from vllm.model_executor.offloader.base import NoopOffloader

    proxy = nn.Module()
    state = SimpleNamespace(
        stats=(
            SimpleNamespace(label="model.layers.0"),
            SimpleNamespace(label="model.layers.1"),
        )
    )
    previous = NoopOffloader()
    singleton = SimpleNamespace(value=previous)
    worker_module._test_failure = None

    def record(boundary: str) -> None:
        events.append(boundary)
        if worker_module._test_failure == boundary:
            raise RuntimeError(boundary)

    def fail_construction(boundary: str) -> None:
        if worker_module._test_failure == boundary:
            raise RuntimeError(boundary)

    class BootstrapOffloader:
        last_coordinate = (0, 1)

        def __init__(self) -> None:
            fail_construction("bootstrap-constructor")
            worker_module._test_bootstrap = self

        def takeover(self, model, config, device, profile=None):
            assert isinstance(model, nn.Module)
            assert config.enabled
            assert device == torch.device("cuda:0")
            assert config.external_compile is worker_module._test_external_compile
            worker_module._test_takeover_config = config
            worker_module._test_takeover_profile = profile
            record("bootstrap-post-init")
            record("build-state")
            record("offload-from-state")
            callback = worker_module._test_takeover_callback
            return proxy if callback is None else callback(model, config)

        def bind_shared_runtime(self, actual_proxy):
            assert actual_proxy is proxy
            record("bootstrap-post-init")
            record("bind-shared-runtime")

    class DeviceMemoryProfiler:
        consumed_memory = 123

        def __enter__(self):
            events.append("start-takeover-memory-profile")
            return self

        def __exit__(self, _exc_type, _exc_val, _exc_tb):
            events.append("stop-takeover-memory-profile")

    def get_offloader():
        events.append("get-previous-offloader")
        return previous

    def set_offloader(offloader):
        singleton.value = offloader
        if isinstance(offloader, BootstrapOffloader):
            record("set-bootstrap-offloader")
        elif offloader is previous:
            record("restore-previous-offloader")
        else:
            raise AssertionError(f"unexpected offloader {offloader!r}")

    monkeypatch.setattr(worker_module, "VllmBootstrapOffloader", BootstrapOffloader, raising=False)
    monkeypatch.setattr(worker_module, "_offloader_api", lambda: (get_offloader, set_offloader))
    monkeypatch.setattr(worker_module, "DeviceMemoryProfiler", lambda _device: DeviceMemoryProfiler())
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    worker_module._test_external_compile = False
    worker_module._test_takeover_callback = None
    worker_module._test_takeover_profile = None
    worker_module._test_takeover_config = None
    worker_module._test_singleton = singleton
    worker_module._test_proxy = proxy
    return state

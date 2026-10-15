# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum
import importlib
import logging
import sys
import types
from types import SimpleNamespace

import pytest

from flextensor.config import OffloadConfig


@pytest.fixture()
def bootstrap_module(monkeypatch):
    initialized_logger_names: list[str] = []

    def init_logger(name):
        initialized_logger_names.append(name)
        return logging.getLogger(name)

    base = types.ModuleType("vllm.model_executor.offloader.base")

    class BaseOffloader:
        def wrap_modules(self, modules_generator):
            return list(modules_generator)

        def post_init(self):
            return None

    base.BaseOffloader = BaseOffloader
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = init_logger
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.logger": vllm_logger,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.offloader": types.ModuleType("vllm.model_executor.offloader"),
        "vllm.model_executor.offloader.base": base,
    }
    previous = sys.modules.pop("flextensor.contrib.vllm.v2.offloader", None)
    previous_state_builder = sys.modules.pop("flextensor.contrib.vllm.v2.state_builder", None)
    v2_package = importlib.import_module("flextensor.contrib.vllm.v2")
    missing = object()
    previous_state_builder_attribute = getattr(v2_package, "state_builder", missing)
    if previous_state_builder_attribute is not missing:
        del v2_package.state_builder
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    try:
        module = importlib.import_module("flextensor.contrib.vllm.v2.offloader")
        monkeypatch.setattr(module, "benchmark_memory_transfers", lambda _stats, _device: {8: 0.1})
        module._test_initialized_logger_names = initialized_logger_names
        yield module
    finally:
        sys.modules.pop("flextensor.contrib.vllm.v2.offloader", None)
        sys.modules.pop("flextensor.contrib.vllm.v2.state_builder", None)
        if previous is not None:
            sys.modules["flextensor.contrib.vllm.v2.offloader"] = previous
        if previous_state_builder is not None:
            sys.modules["flextensor.contrib.vllm.v2.state_builder"] = previous_state_builder
        if previous_state_builder_attribute is missing:
            if hasattr(v2_package, "state_builder"):
                del v2_package.state_builder
        else:
            v2_package.state_builder = previous_state_builder_attribute


@pytest.fixture()
# ruff: ignore[noqa-comments] - compatibility with the pre-commit Ruff version.
def worker_module(monkeypatch):  # noqa: C901
    events: list[str] = []
    initialized_logger_names: list[str] = []
    logger_records: list[tuple[str, str]] = []
    singleton = SimpleNamespace(value=object())

    def record_log(level, message, *args):
        logger_records.append((level, message % args if args else message))

    logger = SimpleNamespace(
        name="vllm.flextensor.v2.worker",
        info=lambda message, *args: record_log("info", message, *args),
        warning=lambda message, *args: record_log("warning", message, *args),
        exception=lambda message, *args: record_log("exception", message, *args),
    )

    def init_logger(name):
        initialized_logger_names.append(name)
        return logger

    class BaseOffloader:
        def __init__(self) -> None:
            pass

    class NoopOffloader(BaseOffloader):
        pass

    class UVAOffloader(BaseOffloader):
        pass

    class PrefetchOffloader(BaseOffloader):
        pass

    def get_offloader():
        events.append("get-previous-offloader")
        return singleton.value

    def set_offloader(offloader):
        singleton.value = offloader

    class Worker:
        def load_model(self, *, load_dummy_weights: bool = False) -> None:
            self._load_dummy_weights = load_dummy_weights
            self._events.append("vllm-load-model")
            if self._failure == "vllm-load-model":
                raise RuntimeError("vllm-load-model")

        def compile_or_warm_up_model(self):
            self._events.append("vllm-compile-or-warm-up")
            return "compilation-times"

        def execute_model(self, scheduler_output):
            self._events.append("vllm-execute-model")
            return getattr(scheduler_output, "result", None)

        def shutdown(self) -> None:
            self._events.append("vllm-shutdown")

    class CompilationMode(enum.IntEnum):
        NONE = 0
        STOCK_TORCH_COMPILE = 1
        DYNAMO_TRACE_ONCE = 2
        VLLM_COMPILE = 3

    class CUDAGraphMode(enum.Enum):
        NONE = 0
        FULL_AND_PIECEWISE = (2, 1)

    class VllmConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            kwargs.setdefault(
                "parallel_config",
                SimpleNamespace(enable_elastic_ep=False, use_ubatching=False),
            )
            super().__init__(**kwargs)

    class GPUModelRunnerV2(SimpleNamespace):
        pass

    class DeviceMemoryProfiler:
        pass

    offloader_base = types.ModuleType("vllm.model_executor.offloader.base")
    offloader_base.BaseOffloader = BaseOffloader
    offloader_base.NoopOffloader = NoopOffloader
    offloader_base.UVAOffloader = UVAOffloader
    offloader_base.PrefetchOffloader = PrefetchOffloader
    offloader_base.get_offloader = get_offloader
    offloader_base.set_offloader = set_offloader
    gpu_worker = types.ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker.Worker = Worker
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = init_logger
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.CompilationMode = CompilationMode
    vllm_config.CUDAGraphMode = CUDAGraphMode
    vllm_config.VllmConfig = VllmConfig
    gpu_model_runner_v2 = types.ModuleType("vllm.v1.worker.gpu.model_runner")
    gpu_model_runner_v2.GPUModelRunner = GPUModelRunnerV2
    vllm_v1_utils = types.ModuleType("vllm.v1.utils")

    def compute_iteration_details(scheduler_output):
        context_request_ids = {request.req_id for request in scheduler_output.scheduled_new_reqs}
        context_request_ids.update(
            request_id
            for request_id in scheduler_output.num_scheduled_tokens
            if scheduler_output.scheduled_cached_reqs.is_context_phase(request_id)
        )
        context_tokens = sum(
            count
            for request_id, count in scheduler_output.num_scheduled_tokens.items()
            if request_id in context_request_ids
        )
        generation_tokens = sum(scheduler_output.num_scheduled_tokens.values()) - context_tokens
        return SimpleNamespace(
            num_ctx_requests=len(context_request_ids),
            num_ctx_tokens=context_tokens,
            num_generation_requests=len(scheduler_output.num_scheduled_tokens) - len(context_request_ids),
            num_generation_tokens=generation_tokens,
        )

    vllm_v1_utils.compute_iteration_details = compute_iteration_details
    vllm_utils = types.ModuleType("vllm.utils")
    mem_utils = types.ModuleType("vllm.utils.mem_utils")
    mem_utils.DeviceMemoryProfiler = DeviceMemoryProfiler
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.config": vllm_config,
        "vllm.logger": vllm_logger,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.utils": vllm_v1_utils,
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu": types.ModuleType("vllm.v1.worker.gpu"),
        "vllm.v1.worker.gpu.model_runner": gpu_model_runner_v2,
        "vllm.v1.worker.gpu_worker": gpu_worker,
        "vllm.utils": vllm_utils,
        "vllm.utils.mem_utils": mem_utils,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.offloader": types.ModuleType("vllm.model_executor.offloader"),
        "vllm.model_executor.offloader.base": offloader_base,
    }
    module_names = (
        "flextensor.contrib.vllm.v2.worker",
        "flextensor.contrib.vllm.v2.breakable_cudagraph",
        "flextensor.contrib.vllm.v2.offloader",
        "flextensor.contrib.vllm.v2.inference_profile",
        "flextensor.contrib.vllm.v2.state_builder",
    )
    removed = {name: sys.modules.pop(name) for name in module_names if name in sys.modules}
    v2_package = importlib.import_module("flextensor.contrib.vllm.v2")
    missing = object()
    previous_profile_attribute = getattr(v2_package, "inference_profile", missing)
    if previous_profile_attribute is not missing:
        del v2_package.inference_profile
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    try:
        module = importlib.import_module("flextensor.contrib.vllm.v2.worker")
        module._test_events = events
        module._test_initialized_logger_names = initialized_logger_names
        module._test_logger_records = logger_records
        monkeypatch.setattr(
            module,
            "load_config",
            lambda **_kwargs: OffloadConfig(enabled=True, pinned_memory=False),
        )
        yield module
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        if previous_profile_attribute is missing:
            if hasattr(v2_package, "inference_profile"):
                del v2_package.inference_profile
        else:
            v2_package.inference_profile = previous_profile_attribute
        sys.modules.update(removed)

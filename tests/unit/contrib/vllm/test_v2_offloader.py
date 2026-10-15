# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib
import logging
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from weakref import ref

import pytest
import torch
from torch import nn

from flextensor.collectors import LayerStatistics
from flextensor.config import OffloadConfig
from flextensor.state_handler import TensorManagerState

from ._v2_test_utils import (
    _meta_online_quant_unit,
    _module_with_tensor,
    _RecordingOffloadManager,
    _set_available_memory,
)


def test_v2_package_facade_does_not_import_vllm_eagerly() -> None:
    previous = sys.modules.pop("flextensor.contrib.vllm.v2", None)
    try:
        package = importlib.import_module("flextensor.contrib.vllm.v2")

        assert package.__all__ == ["FlexTensorOffloadWorker"]
        assert "flextensor.contrib.vllm.v2.offloader" not in sys.modules
    finally:
        sys.modules.pop("flextensor.contrib.vllm.v2", None)
        if previous is not None:
            sys.modules["flextensor.contrib.vllm.v2"] = previous


def test_v2_error_module_is_dependency_free() -> None:
    error_module = importlib.import_module("flextensor.contrib.vllm.v2.errors")

    assert issubclass(error_module.VllmFlexTensorV2Error, RuntimeError)


def test_offloader_uses_vllm_namespaced_logger(bootstrap_module) -> None:
    assert bootstrap_module._test_initialized_logger_names == [
        "vllm.flextensor.v2.state_builder",
        "vllm.flextensor.v2.offloader",
    ]
    assert bootstrap_module.LOGGER.name == "vllm.flextensor.v2.offloader"


def test_state_builder_uses_vllm_logger_without_annotation_aliases(bootstrap_module) -> None:
    state_builder = bootstrap_module.state_builder

    assert state_builder.LOGGER.name == "vllm.flextensor.v2.state_builder"
    assert not any(
        hasattr(state_builder, name)
        for name in ("_OFFLOAD_CONFIG_TYPE", "_LIVE_UNIT_TYPE", "_LOADED_MODEL_SCAN_TYPE", "_TORCH_DEVICE_TYPE")
    )


def test_bootstrap_stages_each_unit_before_requesting_the_next(bootstrap_module, monkeypatch) -> None:
    first = _module_with_tensor("weight")
    second = _module_with_tensor("weight")
    staged: list[int] = []

    monkeypatch.setattr(
        bootstrap_module,
        "_stage_parameter_storage_on_cpu",
        lambda parameter, staged_storages: staged.append(id(parameter)),
    )
    offloader = bootstrap_module.VllmBootstrapOffloader()

    def modules():
        yield first
        assert id(first.weight) in staged
        yield second

    assert offloader.wrap_modules(modules()) == [first, second]
    assert staged == [id(first.weight), id(second.weight)]
    assert offloader.last_coordinate == (0, 1)


def test_bootstrap_does_not_stage_buffers_meta_or_unsupported_layouts(bootstrap_module, monkeypatch) -> None:
    unit = nn.Module()
    unit.weight = nn.Parameter(torch.ones(2, 2))
    unit.meta_weight = nn.Parameter(torch.empty(2, 2, device="meta"))
    unit.sparse_weight = nn.Parameter(torch.ones(2, 2).to_sparse())
    unit.register_buffer("cache", torch.ones(2))
    staged: list[str] = []

    monkeypatch.setattr(
        bootstrap_module,
        "_stage_parameter_storage_on_cpu",
        lambda parameter, staged_storages: staged.append(
            next(name for name, value in unit.named_parameters() if value is parameter)
        ),
    )

    bootstrap_module.VllmBootstrapOffloader().wrap_modules(iter([unit]))

    assert staged == ["weight"]


def test_bootstrap_stages_shared_parameter_once(bootstrap_module, monkeypatch) -> None:
    shared = nn.Parameter(torch.empty(2, 2))
    first = nn.Module()
    first.weight = shared
    second = nn.Module()
    second.weight = shared
    staged: list[int] = []
    monkeypatch.setattr(
        bootstrap_module,
        "_stage_parameter_storage_on_cpu",
        lambda parameter, staged_storages: staged.append(id(parameter)),
    )
    offloader = bootstrap_module.VllmBootstrapOffloader()

    assert offloader.wrap_modules(iter([first, second])) == [first, second]

    assert staged == [id(shared)]


def test_bootstrap_preserves_distinct_parameter_views_of_shared_storage(bootstrap_module) -> None:
    backing = torch.arange(8, dtype=torch.float32)
    first = nn.Module()
    first.weight = nn.Parameter(backing[:4].view(2, 2))
    second = nn.Module()
    second.weight = nn.Parameter(backing[2:6])
    expected_first = first.weight.detach().clone()
    expected_second = second.weight.detach().clone()

    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader.wrap_modules(iter([first, second]))

    assert first.weight.device.type == second.weight.device.type == "cpu"
    assert first.weight.untyped_storage()._cdata == second.weight.untyped_storage()._cdata
    assert first.weight.storage_offset() == 0
    assert second.weight.storage_offset() == 2
    torch.testing.assert_close(first.weight, expected_first)
    torch.testing.assert_close(second.weight, expected_second)


def test_bootstrap_preserves_preexisting_buffer_view_during_weight_load(bootstrap_module) -> None:
    unit = nn.Module()
    unit.weight = nn.Parameter(torch.zeros(2, 4))
    unit.register_buffer("weight_view", unit.weight.view(4, 2), persistent=False)
    offloader = bootstrap_module.VllmBootstrapOffloader()

    offloader.wrap_modules(iter([unit]))
    unit.weight.data.copy_(torch.arange(8, dtype=torch.float32).view(2, 4))

    assert unit.weight.untyped_storage()._cdata == unit.weight_view.untyped_storage()._cdata
    torch.testing.assert_close(unit.weight_view, torch.arange(8, dtype=torch.float32).view(4, 2))


def test_bootstrap_preserves_tensor_alias_nested_in_quant_config_dataclasses(bootstrap_module) -> None:
    @dataclass
    class QuantDesc:
        scale: torch.Tensor

    @dataclass
    class QuantConfig:
        weight: QuantDesc

    unit = nn.Module()
    unit.weight = nn.Parameter(torch.arange(8, dtype=torch.float32).view(2, 4))
    cached_scale = unit.weight.view(4, 2)
    unit.quant_method = SimpleNamespace(moe_quant_config=QuantConfig(QuantDesc(cached_scale)))

    bootstrap_module.VllmBootstrapOffloader().wrap_modules(iter([unit]))

    assert cached_scale.untyped_storage()._cdata == unit.weight.untyped_storage()._cdata


def test_stage_parameter_storage_cache_releases_replaced_backing(bootstrap_module) -> None:
    parameter = nn.Parameter(torch.ones(2, 2))
    staged_storages = {}

    bootstrap_module._stage_parameter_storage_on_cpu(parameter, staged_storages)
    staged_storage = parameter.untyped_storage()
    staged_storage_ref = ref(staged_storage)

    parameter.data = parameter.detach().clone()
    del staged_storage
    gc.collect()

    assert staged_storage_ref() is None
    assert next(iter(staged_storages.values()))() is None


def test_bootstrap_does_not_alias_stale_reused_storage_identity(bootstrap_module, monkeypatch) -> None:
    monkeypatch.setattr(bootstrap_module, "_STORAGE_ID", lambda _storage: 1)
    first = _module_with_tensor("weight")
    first.weight.data.fill_(1)
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader.wrap_modules(iter([first]))

    second = _module_with_tensor("weight")
    second.weight.data.fill_(2)
    offloader.wrap_modules(iter([second]))

    assert first.weight.untyped_storage()._cdata != second.weight.untyped_storage()._cdata
    torch.testing.assert_close(first.weight, torch.ones(2, 2))
    torch.testing.assert_close(second.weight, torch.full((2, 2), 2.0))


def test_bootstrap_rejects_unit_before_partial_staging_when_host_is_full(
    bootstrap_module,
    monkeypatch,
) -> None:
    unit = _module_with_tensor("weight")
    staged: list[int] = []
    monkeypatch.setattr(
        bootstrap_module.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available=0),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "_stage_parameter_storage_on_cpu",
        lambda parameter, staged_storages: staged.append(id(parameter)),
    )

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="host budget"):
        bootstrap_module.VllmBootstrapOffloader().wrap_modules(iter([unit]))

    assert staged == []


def test_bootstrap_rechecks_host_budget_after_staged_storage_expires(
    bootstrap_module,
    monkeypatch,
) -> None:
    backing = torch.ones(4)
    first = nn.Parameter(backing[:2])
    second = nn.Module()
    second.weight = nn.Parameter(backing[2:])
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader._stage_parameter_on_cpu(first)
    first.data = first.detach().clone()
    gc.collect()
    _set_available_memory(monkeypatch, bootstrap_module, gpu=1024, host=0)

    with pytest.raises(
        bootstrap_module.VllmFlexTensorV2Error,
        match=r"required=16.*available=0",
    ):
        offloader._stage_concrete_parameters_on_cpu(second)


def test_online_quantization_cpu_placement_waits_for_final_parameter(bootstrap_module, monkeypatch) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    unit = _meta_online_quant_unit()
    callbacks = (
        unit.first.quant_method.process_weights_after_loading,
        unit.second.quant_method.process_weights_after_loading,
    )
    staged: list[nn.Parameter] = []
    monkeypatch.setattr(
        bootstrap_module,
        "_stage_parameter_storage_on_cpu",
        lambda parameter, staged_storages: staged.append(parameter),
    )

    offloader.wrap_modules(iter([unit]))
    assert unit.first.weight.is_meta
    assert unit.second.weight.is_meta
    assert unit.first.quant_method.process_weights_after_loading != callbacks[0]
    assert unit.second.quant_method.process_weights_after_loading != callbacks[1]

    unit.first.quant_method.process_weights_after_loading(unit.first)
    assert unit.first.weight.device.type == "cpu"
    assert unit.second.weight.is_meta
    assert staged == []

    unit.second.quant_method.process_weights_after_loading(unit.second)
    assert unit.first.weight.device.type == "cpu"
    assert unit.second.weight.device.type == "cpu"
    assert len(staged) == 4
    assert unit.first.quant_method.process_weights_after_loading == callbacks[0]
    assert unit.second.quant_method.process_weights_after_loading == callbacks[1]
    offloader.post_init()


def test_online_quantization_cpu_placement_logs_completion_once_with_final_bytes(
    bootstrap_module,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    unit = _meta_online_quant_unit()
    offloader.wrap_modules(iter([unit]))

    with caplog.at_level(logging.INFO, logger=bootstrap_module.LOGGER.name):
        unit.first.quant_method.process_weights_after_loading(unit.first)
        unit.second.quant_method.process_weights_after_loading(unit.second)
        unit.second.quant_method.process_weights_after_loading(unit.second)

    assert [
        record.getMessage()
        for record in caplog.records
        if "online quantization placement complete" in record.getMessage()
    ] == ["online quantization placement complete: coordinate=(0, 0) final_bytes=20"]


def test_post_init_rejects_pending_online_quantization_callback(bootstrap_module, monkeypatch) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    unit = _meta_online_quant_unit()
    offloader.wrap_modules(iter([unit]))

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="staging incomplete"):
        offloader.post_init()


def test_bootstrap_post_init_returns_after_first_success(bootstrap_module) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader.wrap_modules(iter([nn.Module()]))

    offloader.post_init()
    offloader.post_init()

    assert offloader._post_init_validated


def test_takeover_adopts_state_retains_manager_and_clears_bootstrap_storage(
    bootstrap_module,
    caplog,
    monkeypatch,
) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader.wrap_modules(iter([nn.Module()]))
    storage_key = (torch.device("cpu"), 1, 4)
    offloader._staged_storages[storage_key] = torch.ones(1)
    offloader._staged_storage_sources[storage_key] = object()
    state = TensorManagerState(
        loader_type="raw_block_transfer",
        tensor_id_to_name_map={},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={},
        release_strategy={},
        label_to_block_id={},
        stats=[LayerStatistics(label="model.layers.0", tensors=[], duration=1.0)],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )
    manager = _RecordingOffloadManager()
    proxy = nn.Module()
    proxy.offload_manager = manager
    captured = {}

    monkeypatch.setattr(offloader, "build_state", lambda model, config, device, profile=None: state)

    def adopt(model, actual_state, config, *, allow_strategy_replan):
        captured.update(
            model=model,
            state=actual_state,
            config=config,
            allow_strategy_replan=allow_strategy_replan,
        )
        return proxy

    monkeypatch.setattr(bootstrap_module.flextensor, "offload_from_state", adopt)
    model = nn.Module()
    config = OffloadConfig(enabled=True, external_compile=True, pinned_memory=False)

    with caplog.at_level(logging.INFO, logger=bootstrap_module.LOGGER.name):
        result = offloader.takeover(model, config, "cuda:0")
    offloader.sync_prev_onload()
    offloader.join_after_forward()

    assert result is proxy
    assert captured["model"] is model
    assert captured["state"] is state
    assert captured["config"].include_patterns == ["model.layers.0"]
    assert captured["config"].exclude_patterns == []
    assert captured["config"].external_compile is True
    assert captured["allow_strategy_replan"] is False
    assert offloader._runtime_manager is manager
    assert offloader.runtime_state is state
    assert offloader._staged_storages == {}
    assert offloader._staged_storage_sources == {}
    assert manager.runtime_calls == ["sync_prev_onload", "join_after_forward"]
    assert "FlexTensor v2 unit inventory: ['model.layers.0']" in caplog.text
    assert "state takeover installed loader_type=raw_block_transfer" in caplog.text


def test_bind_shared_runtime_adopts_live_state_and_clears_bootstrap_refs(
    bootstrap_module,
    caplog,
    monkeypatch,
) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader.wrap_modules(iter([nn.Module()]))
    storage_key = (torch.device("cpu"), 1, 4)
    offloader._staged_parameter_ids.add(1)
    offloader._staged_storages[storage_key] = torch.ones(1)
    offloader._staged_storage_sources[storage_key] = object()
    offloader._pending_deferred_callbacks[0, 0] = set()
    state = TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={},
        release_strategy={},
        label_to_block_id={},
        stats=[],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=[],
        shm_block_name_map={},
    )
    manager = _RecordingOffloadManager()
    monkeypatch.setattr(manager, "_export_runtime_state", lambda: state)
    proxy = nn.Module()
    proxy.offload_manager = manager

    with caplog.at_level(logging.INFO, logger=bootstrap_module.LOGGER.name):
        offloader.bind_shared_runtime(proxy)
    offloader.sync_prev_onload()
    offloader.join_after_forward()

    assert offloader._runtime_manager is manager
    assert offloader.runtime_state is state
    assert offloader._state_built
    assert offloader._live_units == []
    assert offloader._staged_parameter_ids == set()
    assert offloader._staged_storages == {}
    assert offloader._staged_storage_sources == {}
    assert offloader._pending_deferred_callbacks == {}
    assert manager.runtime_calls == ["sync_prev_onload", "join_after_forward"]
    assert "state takeover installed loader_type=allocation_block_transfer" in caplog.text


def test_bind_shared_runtime_requires_proxy_manager(bootstrap_module) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader.wrap_modules(iter([nn.Module()]))

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=r"shared offload proxy.*OffloadManager"):
        offloader.bind_shared_runtime(nn.Module())

    assert offloader._runtime_manager is None


def test_runtime_hooks_are_noops_before_takeover(bootstrap_module) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()

    offloader.sync_prev_onload()
    offloader.join_after_forward()


def test_runtime_state_rejects_access_before_takeover(bootstrap_module) -> None:
    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="before successful takeover"):
        _ = bootstrap_module.VllmBootstrapOffloader().runtime_state


@pytest.mark.parametrize("profile_mode", ["accepted", "incompatible", "absent"])
def test_build_state_benchmarks_transfers_for_every_profile_mode(
    bootstrap_module,
    monkeypatch,
    profile_mode,
) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader._post_init_validated = True
    device = torch.device("cuda:0")
    scan = SimpleNamespace(layer_stats=[LayerStatistics(label="layer", tensors=[], duration=1.0)])
    merged = SimpleNamespace(layer_stats=[LayerStatistics(label="layer", tensors=[], duration=2.0)])
    profile = None if profile_mode == "absent" else object.__new__(TensorManagerState)
    benchmark_calls = []
    benchmark_result = {4096: 0.25}
    build_calls = []
    expected_state = object.__new__(TensorManagerState)

    monkeypatch.setattr(bootstrap_module.state_builder, "resolve_cuda_device", lambda _device: device)
    monkeypatch.setattr(bootstrap_module.model_scan, "scan_loaded_model", lambda *_args, **_kwargs: scan)

    def merge_profile(_scan, _profile):
        if profile_mode == "incompatible":
            raise ValueError("incompatible")
        return merged

    monkeypatch.setattr(bootstrap_module.state_builder, "merge_profile_statistics", merge_profile)
    monkeypatch.setattr(
        bootstrap_module,
        "benchmark_memory_transfers",
        lambda layer_stats, actual_device: benchmark_calls.append((layer_stats, actual_device)) or benchmark_result,
        raising=False,
    )
    monkeypatch.setattr(
        bootstrap_module.state_builder,
        "build_conservative_state",
        lambda *args, **kwargs: build_calls.append((args, kwargs)) or expected_state,
    )

    assert (
        offloader.build_state(nn.Module(), OffloadConfig(pinned_memory=False), device, profile=profile)
        is expected_state
    )

    expected_layer_stats = merged.layer_stats if profile_mode == "accepted" else scan.layer_stats
    assert benchmark_calls == [(expected_layer_stats, device)]
    assert build_calls[0][1].get("memory_stats") is benchmark_result


def test_takeover_releases_adopted_default_manager_when_proxy_has_no_manager(
    bootstrap_module,
    monkeypatch,
) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader.wrap_modules(iter([nn.Module()]))
    state = SimpleNamespace(stats=[SimpleNamespace(label="model.layers.0")])
    releases = []
    monkeypatch.setattr(offloader, "build_state", lambda model, config, device, profile=None: state)
    monkeypatch.setattr(bootstrap_module.flextensor, "offload_from_state", lambda *args, **kwargs: object())
    monkeypatch.setattr(bootstrap_module.flextensor, "release", lambda: releases.append("release"))

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="OffloadManager"):
        offloader.takeover(
            nn.Module(),
            OffloadConfig(enabled=True, pinned_memory=False),
            "cuda:0",
        )

    assert releases == ["release"]
    assert offloader._runtime_manager is None


def test_wrap_modules_rejects_reuse_after_takeover(bootstrap_module) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader._runtime_manager = object()

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="runtime takeover"):
        offloader.wrap_modules(iter([nn.Module()]))


def test_online_quantization_final_callback_rejects_meta_parameter_before_moving(
    bootstrap_module,
    monkeypatch,
) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    unit = _meta_online_quant_unit()
    staged: list[nn.Parameter] = []
    monkeypatch.setattr(
        bootstrap_module,
        "_stage_parameter_storage_on_cpu",
        lambda parameter, staged_storages: staged.append(parameter),
    )
    offloader.wrap_modules(iter([unit]))
    unit.first.quant_method.process_weights_after_loading(unit.first)
    unit.remaining_meta = nn.Parameter(torch.empty(1, device="meta"))

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="remaining_meta"):
        unit.second.quant_method.process_weights_after_loading(unit.second)

    assert staged == []


def test_online_quantization_final_callback_rechecks_storage_bytes_and_restores_callbacks(
    bootstrap_module,
    monkeypatch,
) -> None:
    offloader = bootstrap_module.VllmBootstrapOffloader()
    unit = _meta_online_quant_unit()
    callbacks = (
        unit.first.quant_method.process_weights_after_loading,
        unit.second.quant_method.process_weights_after_loading,
    )
    staged: list[nn.Parameter] = []
    monkeypatch.setattr(
        bootstrap_module,
        "_stage_parameter_storage_on_cpu",
        lambda parameter, staged_storages: staged.append(parameter),
    )
    offloader.wrap_modules(iter([unit]))
    _set_available_memory(monkeypatch, bootstrap_module, gpu=1024, host=19)
    unit.first.quant_method.process_weights_after_loading(unit.first)

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=r"required=20.*available=19"):
        unit.second.quant_method.process_weights_after_loading(unit.second)

    assert staged == []
    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=r"pending=\{\(0, 0\): 1\}"):
        offloader.post_init()

    _set_available_memory(monkeypatch, bootstrap_module, gpu=1024, host=1024)
    unit.second.quant_method.process_weights_after_loading(unit.second)

    assert len(staged) == 4
    assert unit.first.quant_method.process_weights_after_loading == callbacks[0]
    assert unit.second.quant_method.process_weights_after_loading == callbacks[1]
    offloader.post_init()

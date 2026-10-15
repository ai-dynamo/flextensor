# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FlexTensor's vLLM model loader."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture()
def loader_module():  # noqa: C901 - local vLLM stub graph is intentionally compact
    """Import ``flextensor.contrib.vllm.loader`` with minimal vLLM stubs."""
    stubs: dict[str, types.ModuleType] = {}

    def _stub(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        stubs[name] = mod
        return mod

    class _DefaultModelLoader:
        def _prepare_weights(self, *args, **kwargs):
            return ("", [], False)

        def load_weights(self, model, model_config):
            raise AssertionError("test must patch load_weights")

    def _register_model_loader(_name: str):
        def _decorator(cls):
            return cls

        return _decorator

    @contextmanager
    def _set_default_torch_dtype(dtype):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(old_dtype)

    def _initialize_online_processing(module):
        module._vllm_online_processing_enabled = True

    _stub("vllm")
    _stub("vllm.config", ModelConfig=object, VllmConfig=object)
    _stub("vllm.logger", init_logger=lambda name: __import__("logging").getLogger(name))
    _stub("vllm.model_executor")
    _stub("vllm.model_executor.layers")
    _stub("vllm.model_executor.layers.attention_layer_base", AttentionLayerBase=type("AttentionLayerBase", (), {}))
    quantization_mod = _stub("vllm.model_executor.layers.quantization")
    quantization_mod.__path__ = []
    online_mod = _stub("vllm.model_executor.layers.quantization.online")
    online_mod.__path__ = []
    _stub(
        "vllm.model_executor.layers.quantization.fp8",
        initialize_online_processing=_initialize_online_processing,
        Fp8OnlineLinearMethod=_FakeLegacyFp8OnlineLinearMethod,
        Fp8LinearMethod=_FakeLegacyFp8LinearMethod,
        Fp8OnlineMoEMethod=_FakeLegacyFp8OnlineMoEMethod,
        Fp8MoEMethod=_FakeLegacyFp8MoEMethod,
    )
    _stub(
        "vllm.model_executor.layers.quantization.mxfp8",
        Mxfp8OnlineLinearMethod=_FakeLegacyMxfp8OnlineLinearMethod,
        Mxfp8OnlineMoEMethod=_FakeLegacyMxfp8OnlineMoEMethod,
    )
    _stub(
        "vllm.model_executor.layers.quantization.online.fp8",
        Fp8PerTensorOnlineLinearMethod=_FakeFp8PerTensorOnlineLinearMethod,
        Fp8PerBlockOnlineLinearMethod=_FakeFp8PerBlockOnlineLinearMethod,
        Fp8PerTensorOnlineMoEMethod=_FakeFp8PerTensorOnlineMoEMethod,
        Fp8PerBlockOnlineMoEMethod=_FakeFp8PerBlockOnlineMoEMethod,
    )
    _stub(
        "vllm.model_executor.layers.quantization.online.mxfp8",
        Mxfp8OnlineLinearMethod=_FakeMxfp8OnlineLinearMethod,
        Mxfp8OnlineMoEMethod=_FakeMxfp8OnlineMoEMethod,
    )
    _stub(
        "vllm.model_executor.model_loader",
        get_model_loader=lambda _config: pytest.fail("test must patch get_model_loader"),
        register_model_loader=_register_model_loader,
    )
    _stub("vllm.model_executor.model_loader.default_loader", DefaultModelLoader=_DefaultModelLoader)
    _stub(
        "vllm.model_executor.model_loader.utils",
        initialize_model=lambda **_kwargs: None,
    )
    _stub("vllm.platforms", current_platform=SimpleNamespace(device_type="cuda"))
    _stub("vllm.utils")
    _stub("vllm.utils.torch_utils", set_default_torch_dtype=_set_default_torch_dtype)

    previous = {name: sys.modules.get(name) for name in stubs}
    previous["flextensor.contrib.vllm.loader"] = sys.modules.pop("flextensor.contrib.vllm.loader", None)
    sys.modules.update(stubs)

    try:
        import flextensor.contrib.vllm.loader as loader

        yield loader
    finally:
        for name, mod in previous.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@pytest.fixture()
def v2_loader_module(loader_module):
    """Import the v2 SHM loader under the existing minimal vLLM stubs."""
    module_name = "flextensor.contrib.vllm.v2.loader"
    previous = sys.modules.pop(module_name, None)
    try:
        import flextensor.contrib.vllm.v2.loader as loader

        yield loader
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


class _FakeQuantMethod:
    def __init__(self):
        self.processed_devices: list[str] = []

    def process_weights_after_loading(self, module):
        device = getattr(module, "_fake_device", "cpu")
        if device == "cpu":
            raise AssertionError("FP8 online processing ran during CPU-first load")
        self.processed_devices.append(device)


class _FakeVllmParameter(nn.Parameter):
    pass


class _FakeLegacyFp8OnlineLinearMethod(_FakeQuantMethod):
    pass


class _FakeLegacyFp8LinearMethod(_FakeQuantMethod):
    pass


class _FakeLegacyFp8OnlineMoEMethod(_FakeQuantMethod):
    pass


class _FakeLegacyFp8MoEMethod(_FakeQuantMethod):
    pass


class _FakeLegacyMxfp8OnlineLinearMethod(_FakeQuantMethod):
    pass


class _FakeLegacyMxfp8OnlineMoEMethod(_FakeQuantMethod):
    pass


class _FakeFp8PerTensorOnlineLinearMethod(_FakeQuantMethod):
    pass


class _FakeFp8PerBlockOnlineLinearMethod(_FakeQuantMethod):
    pass


class _FakeFp8PerTensorOnlineMoEMethod(_FakeQuantMethod):
    pass


class _FakeFp8PerBlockOnlineMoEMethod(_FakeQuantMethod):
    pass


class _FakeMxfp8OnlineLinearMethod(_FakeQuantMethod):
    pass


class _FakeMxfp8OnlineMoEMethod(_FakeQuantMethod):
    pass


class _FakeFutureQuantMethod(_FakeQuantMethod):
    pass


class _FakePlainParameterRequiredMethod(_FakeFp8PerBlockOnlineLinearMethod):
    def process_weights_after_loading(self, module):
        assert type(module.weight) is nn.Parameter
        super().process_weights_after_loading(module)


class _FakeQuantLinear(nn.Module):
    def __init__(self, quant_method):
        super().__init__()
        self.quant_method = quant_method
        self._fake_device = "cpu"


class _FakeQuantLinearWithVllmParameter(_FakeQuantLinear):
    def __init__(self, quant_method):
        super().__init__(quant_method)
        self.weight = _FakeVllmParameter(torch.ones(2, 2), requires_grad=False)


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_linears = nn.ModuleList([
            _FakeQuantLinear(_FakeLegacyFp8OnlineLinearMethod()),
            _FakeQuantLinear(_FakeLegacyFp8LinearMethod()),
            _FakeQuantLinear(_FakeLegacyFp8OnlineMoEMethod()),
            _FakeQuantLinear(_FakeLegacyFp8MoEMethod()),
            _FakeQuantLinear(_FakeLegacyMxfp8OnlineLinearMethod()),
            _FakeQuantLinear(_FakeLegacyMxfp8OnlineMoEMethod()),
            _FakeQuantLinear(_FakeFp8PerTensorOnlineLinearMethod()),
            _FakeQuantLinear(_FakeFp8PerBlockOnlineLinearMethod()),
            _FakeQuantLinear(_FakeFp8PerTensorOnlineMoEMethod()),
            _FakeQuantLinear(_FakeFp8PerBlockOnlineMoEMethod()),
            _FakeQuantLinear(_FakeMxfp8OnlineLinearMethod()),
            _FakeQuantLinear(_FakeMxfp8OnlineMoEMethod()),
            _FakeQuantLinear(_FakeFutureQuantMethod()),
        ])
        self._fake_device = "cpu"

    def to(self, device):  # noqa: ANN001 - mirrors torch API
        device_name = str(device)
        self._fake_device = device_name
        for module in self.modules():
            module._fake_device = device_name
        return self


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeDecoderLayer()])


class TestFlexTensorModelLoader:
    def test_cpu_first_load_defers_vllm_online_quant_processing_until_gpu_phase(
        self,
        loader_module,
        monkeypatch,
    ):
        """vLLM online quant hooks must not process weights while FlexTensor loads on CPU."""
        model = _FakeModel()
        quant_linears = list(model.model.layers[0].quant_linears)

        def _initialize_model(*, vllm_config, model_config, **kwargs):
            # Simulate online quant create_weights() installing the
            # vLLM online-processing loader during model construction.
            for linear in quant_linears:
                sys.modules["vllm.model_executor.layers.quantization.fp8"].initialize_online_processing(
                    linear,
                )
            return model

        def _load_weights(self, loaded_model, model_config):
            assert loaded_model is model
            for linear in quant_linears:
                if getattr(linear, "_vllm_online_processing_enabled", False):
                    linear.quant_method.process_weights_after_loading(linear)

        monkeypatch.setattr(loader_module, "initialize_model", _initialize_model)
        monkeypatch.setattr(loader_module.FlexTensorModelLoader, "load_weights", _load_weights)

        loader = loader_module.FlexTensorModelLoader.__new__(loader_module.FlexTensorModelLoader)
        vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cuda", 0)))
        model_config = SimpleNamespace(dtype=torch.float32)

        loaded = loader.load_model(vllm_config, model_config)

        assert loaded is model
        assert [linear.quant_method.processed_devices for linear in quant_linears] == [["cuda:0"]] * len(quant_linears)

    def test_gpu_phase_materializes_vllm_parameter_subclasses_before_quant_processing(
        self,
        loader_module,
    ):
        """Deferred quantization must see plain Parameters, not vLLM loader wrappers."""
        model = _FakeModel()
        quant_method = _FakePlainParameterRequiredMethod()
        quant_linear = _FakeQuantLinearWithVllmParameter(quant_method)
        model.model.layers[0].quant_linears = nn.ModuleList([quant_linear])

        loader = loader_module.FlexTensorModelLoader.__new__(loader_module.FlexTensorModelLoader)
        model_config = SimpleNamespace(dtype=torch.float32)

        loader._process_weights_layer_by_layer(
            model,
            model_config,
            gpu_device=torch.device("cuda", 0),
            cpu_device=torch.device("cpu"),
            vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=512)),
        )

        assert type(quant_linear.weight) is nn.Parameter
        assert quant_method.processed_devices == ["cuda:0"]


def _shm_context(v2_loader_module, session):
    from flextensor.config import OffloadConfig

    return v2_loader_module.VllmShmLoadContext(
        session=session,
        offload_config=OffloadConfig(
            transfer_mode="allocation_block_transfer",
            profile_storage_dir="/profiles",
            profile_read_only=True,
            shm_enabled=True,
            shm_namespace="test-run",
            shm_wait_timeout=30,
            pinned_memory=False,
        ),
        profile_directory="/profiles",
        materialize_device=torch.device("cpu"),
        underlying_load_format="auto",
    )


class TestFlexTensorShmModelLoader:
    def test_creator_delegates_checkpoint_load_and_returns_concrete_model(
        self,
        v2_loader_module,
        monkeypatch,
    ) -> None:
        model = nn.Linear(2, 2)
        proxy = SimpleNamespace(_get_model=lambda: model)
        calls: list[str] = []
        logs: list[str] = []

        class Session(v2_loader_module.SharedOffloadSession):
            def __init__(self) -> None:
                super().__init__("test-run", wait_timeout=30)

            def offload_from_profile(self, profile_directory, **kwargs):
                assert profile_directory == "/profiles"
                assert kwargs["materialize_device"] == torch.device("cpu")
                calls.append("creator-factory")
                assert kwargs["creator_model_factory"]() is model
                return proxy

        class NativeLoader:
            def load_model(self, *, vllm_config, model_config, **kwargs):
                assert vllm_config is config
                assert model_config is model_config_arg
                assert kwargs == {"prefix": "model"}
                calls.append("checkpoint-load")
                return model

        loader = v2_loader_module.FlexTensorShmModelLoader.__new__(v2_loader_module.FlexTensorShmModelLoader)
        loader.load_config = SimpleNamespace(load_format=v2_loader_module._SHM_LOAD_FORMAT)
        context = _shm_context(v2_loader_module, Session())
        config = SimpleNamespace()
        model_config_arg = SimpleNamespace(dtype=torch.float32, quantization=None)
        monkeypatch.setattr(v2_loader_module, "get_model_loader", lambda load_config: NativeLoader())
        monkeypatch.setattr(v2_loader_module.logger, "info", lambda message, *args: logs.append(message % args))

        with v2_loader_module.use_vllm_shm_load(context):
            loaded = loader.load_model(config, model_config_arg, prefix="model")

        assert loaded is model
        assert context.proxy is proxy
        assert context.role == "creator"
        assert loader.load_config.load_format == v2_loader_module._SHM_LOAD_FORMAT
        assert calls == ["creator-factory", "checkpoint-load"]
        assert any("checkpoint_load=loaded" in message for message in logs)

    @pytest.mark.parametrize("materialize_device", [torch.device("cpu"), torch.device("meta")])
    def test_follower_preserves_shm_storage_while_finalizing_attention(
        self,
        v2_loader_module,
        monkeypatch,
        materialize_device,
    ) -> None:
        calls: list[str] = []
        logs: list[str] = []
        proxy = SimpleNamespace(_get_model=lambda: session.model)

        class Attention(nn.Module, v2_loader_module.AttentionLayerBase):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(2))
                self.register_buffer("scale", torch.empty(1))

            def process_weights_after_loading(self, dtype):
                assert dtype == torch.float32
                assert torch.get_default_device() == materialize_device
                self.scale.fill_(1)
                calls.append("post-load")

        class Topology(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attention = Attention()

        class Session(v2_loader_module.SharedOffloadSession):
            model: nn.Module

            def __init__(self) -> None:
                super().__init__("test-run", wait_timeout=30)

            def offload_from_profile(self, _profile_directory, **kwargs):
                assert kwargs["materialize_device"] == materialize_device
                calls.append("follower-factory")
                topology = kwargs["follower_model_factory"]()
                assert all(parameter.is_meta for parameter in topology.parameters())
                assert all(buffer.is_meta for buffer in topology.buffers())
                self.model = topology.to_empty(device=kwargs["materialize_device"])
                self.weight_storage = self.model.attention.weight.untyped_storage()
                self.scale_storage = self.model.attention.scale.untyped_storage()
                return proxy

        session = Session()
        loader = v2_loader_module.FlexTensorShmModelLoader.__new__(v2_loader_module.FlexTensorShmModelLoader)
        loader.load_config = SimpleNamespace(load_format=v2_loader_module._SHM_LOAD_FORMAT)
        context = _shm_context(v2_loader_module, session)
        context.materialize_device = materialize_device
        config = SimpleNamespace()
        model_config = SimpleNamespace(dtype=torch.float32, quantization=None)

        def initialize_model(**kwargs):
            assert kwargs == {"vllm_config": config, "model_config": model_config, "prefix": "model"}
            calls.append("meta-init")
            return Topology()

        monkeypatch.setattr(v2_loader_module, "initialize_model", initialize_model)
        monkeypatch.setattr(
            v2_loader_module,
            "get_model_loader",
            lambda _config: pytest.fail("follower must not construct a checkpoint loader"),
        )
        monkeypatch.setattr(v2_loader_module.logger, "info", lambda message, *args: logs.append(message % args))

        with v2_loader_module.use_vllm_shm_load(context):
            loaded = loader.load_model(config, model_config, prefix="model")

        assert loaded is session.model
        assert context.proxy is proxy
        assert context.role == "follower"
        assert calls == ["follower-factory", "meta-init", "post-load"]
        assert loaded.attention.weight.untyped_storage() is session.weight_storage
        assert loaded.attention.scale.untyped_storage() is session.scale_storage
        assert any("checkpoint_load=skipped" in message for message in logs)
        assert any("post-load storage identity verified tensors=2" in message for message in logs)

    def test_follower_rejects_post_load_storage_replacement(
        self,
        v2_loader_module,
        monkeypatch,
    ) -> None:
        class Attention(nn.Module, v2_loader_module.AttentionLayerBase):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def process_weights_after_loading(self, _dtype):
                self.weight.data = self.weight.data.clone()

        model = Attention()
        original_storage = model.weight.untyped_storage()
        proxy = SimpleNamespace(_get_model=lambda: model)

        class Session(v2_loader_module.SharedOffloadSession):
            def __init__(self) -> None:
                super().__init__("test-run", wait_timeout=30)

            def offload_from_profile(self, _profile_directory, **_kwargs):
                context.role = "follower"
                return proxy

        loader = v2_loader_module.FlexTensorShmModelLoader.__new__(v2_loader_module.FlexTensorShmModelLoader)
        loader.load_config = SimpleNamespace(load_format=v2_loader_module._SHM_LOAD_FORMAT)
        context = _shm_context(v2_loader_module, Session())
        context.materialize_device = torch.device("cuda:0")
        model_config = SimpleNamespace(dtype=torch.float32, quantization=None)

        with (
            pytest.raises(
                RuntimeError,
                match=r"must preserve registered tensor storage.*parameter 'weight' changed storage",
            ),
            v2_loader_module.use_vllm_shm_load(context),
        ):
            loader.load_model(SimpleNamespace(), model_config)

        assert model.weight.untyped_storage() is not original_storage
        assert context.proxy is None

    def test_creator_failure_restores_wrapped_format_and_context(
        self,
        v2_loader_module,
        monkeypatch,
    ) -> None:
        class Session(v2_loader_module.SharedOffloadSession):
            def __init__(self) -> None:
                super().__init__("test-run", wait_timeout=30)

            def offload_from_profile(self, _profile_directory, **kwargs):
                return kwargs["creator_model_factory"]()

        class FailingLoader:
            def load_model(self, **_kwargs):
                raise RuntimeError("checkpoint failed")

        loader = v2_loader_module.FlexTensorShmModelLoader.__new__(v2_loader_module.FlexTensorShmModelLoader)
        loader.load_config = SimpleNamespace(load_format=v2_loader_module._SHM_LOAD_FORMAT)
        context = _shm_context(v2_loader_module, Session())
        monkeypatch.setattr(v2_loader_module, "get_model_loader", lambda _config: FailingLoader())

        with pytest.raises(RuntimeError, match="checkpoint failed"), v2_loader_module.use_vllm_shm_load(context):
            loader.load_model(SimpleNamespace(), SimpleNamespace(dtype=torch.float32))

        assert loader.load_config.load_format == v2_loader_module._SHM_LOAD_FORMAT
        with pytest.raises(RuntimeError, match="active worker load context"):
            loader.load_model(SimpleNamespace(), SimpleNamespace(dtype=torch.float32))

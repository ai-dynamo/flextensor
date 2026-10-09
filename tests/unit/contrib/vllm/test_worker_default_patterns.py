# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM worker default include/exclude patterns."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


def _find_repo_root(path: Path) -> Path:
    """Find the repository root from a path inside the checkout."""
    for parent in path.resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"Could not find repository root from {path}")


@pytest.fixture()
def worker_module():
    """Load the real _legacy_worker.py with lightweight module stubs."""
    stubs: dict[str, types.ModuleType] = {}

    def _stub(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        stubs[name] = mod
        return mod

    class _Worker:  # noqa: N801 - mimic vLLM's class name
        pass

    class _PsutilError(Exception):
        pass

    _stub("psutil", Error=_PsutilError, virtual_memory=lambda: SimpleNamespace(available=0))

    flextensor = _stub("flextensor", get_offload_manager=lambda: None)
    flextensor.__path__ = []
    contrib = _stub("flextensor.contrib")
    contrib.__path__ = []
    vllm_pkg = _stub("flextensor.contrib.vllm")
    vllm_package_path = _find_repo_root(Path(__file__)) / "src" / "flextensor" / "contrib" / "vllm"
    vllm_pkg.__path__ = [str(vllm_package_path)]

    _stub("flextensor.compile", COMPILED_EAGER_PROFILE_FORWARDS=3)
    _stub("flextensor.config", load_config=lambda **_: SimpleNamespace(external_compile=False))
    _stub("flextensor.contrib.vllm.loader")
    _stub("flextensor.contrib.vllm._drafter_device", ensure_drafter_on_device=lambda *_: None)
    _stub("flextensor.contrib.vllm._logging", safely_install_flextensor_logging_bridge=lambda: None)
    _stub("flextensor.offload_manager", OffloadPhase=SimpleNamespace)
    _stub(
        "flextensor.utils",
        config_field_was_set=lambda config, field_name: field_name in getattr(config, "model_fields_set", set()),
    )

    _stub("vllm")
    _stub("vllm.logger", init_logger=logging.getLogger)
    _stub("vllm.utils")
    _stub("vllm.utils.mem_constants", GiB_bytes=1 << 30)
    _stub("vllm.v1")
    _stub("vllm.v1.worker")
    _stub("vllm.v1.worker.gpu_worker", Worker=_Worker)

    module_name = "flextensor.contrib.vllm._legacy_worker"
    patterns_module_name = "flextensor.contrib.vllm._patterns"
    previous = {name: sys.modules.get(name) for name in [*stubs, module_name, patterns_module_name]}
    sys.modules.update(stubs)
    sys.modules.pop(module_name, None)
    sys.modules.pop(patterns_module_name, None)

    worker_path = vllm_package_path / "_legacy_worker.py"
    spec = importlib.util.spec_from_file_location(module_name, worker_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, mod in previous.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_legacy_worker_warns_on_construction(worker_module) -> None:
    with pytest.warns(DeprecationWarning, match=r"legacy/v1.*worker v2.*FT_VLLM_USE_V2_WORKER.*v0\.6\.0"):
        worker_module.FlexTensorOffloadWorker()


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_worker_rejects_shm_before_model_loading(worker_module, monkeypatch, enabled) -> None:
    worker = worker_module.FlexTensorOffloadWorker()
    worker.device = SimpleNamespace(index=0)
    monkeypatch.setattr(
        worker_module,
        "load_config",
        lambda **_kwargs: SimpleNamespace(enabled=enabled, shm_enabled=True),
    )

    with pytest.raises(
        RuntimeError,
        match=r"legacy vLLM worker.*OffloadConfig\.shm_enabled=True.*FT_VLLM_USE_V2_WORKER=1",
    ):
        worker.load_model()


class TestVllmDefaultPatternUpdates:
    """vLLM worker defaults must be layer-focused and MoE-safe."""

    def test_omitted_max_gpu_mem_fraction_gets_vllm_fallback(self, worker_module) -> None:
        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["custom.layers.*"],
            exclude_patterns=[],
        )

        updates = worker_module._vllm_config_updates(config)

        assert updates["max_gpu_mem_fraction"] == pytest.approx(0.9)

    @pytest.mark.parametrize("value", [None, 0.75])
    def test_explicit_max_gpu_mem_fraction_is_preserved(self, worker_module, value) -> None:
        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["custom.layers.*"],
            exclude_patterns=[],
            max_gpu_mem_fraction=value,
            model_fields_set={"max_gpu_mem_fraction"},
        )

        updates = worker_module._vllm_config_updates(config)

        assert "max_gpu_mem_fraction" not in updates

    def test_wildcard_include_gets_vllm_layer_defaults(self, worker_module) -> None:
        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["*"],
            exclude_patterns=[],
        )

        updates = worker_module._vllm_config_updates(config)

        assert updates["include_patterns"] == worker_module.VLLM_DEFAULT_INCLUDE_PATTERNS
        assert updates["exclude_patterns"] == worker_module.VLLM_DEFAULT_EXCLUDE_PATTERNS
        assert updates["include_patterns"] == [
            "class:*DecoderLayer",
            "class:*DecoderBlock",
            "class:*TransformerBlock",
            "model.embed_tokens",
            "model.norm",
            "model.norm_f",
            "lm_head",
            "logits_processor",
            "language_model.model.embed_tokens",
            "language_model.model.norm",
            "language_model.lm_head",
            "language_model.logits_processor",
        ]
        assert updates["exclude_patterns"] == [
            "class:GateLinear",
            "model.layers.*.mixer.shared_experts",
            "model.layers.*.mixer.fc1_latent_proj",
            "model.layers.*.mixer.fc2_latent_proj",
            "model.layers.*.mlp.shared_expert",
            "language_model.model.layers.*.mlp.shared_expert",
            "language_model.model.layers.*.mlp.gate",
            "language_model.model.layers.*.mlp.shared_expert_gate",
            "language_model.model.layers.*.linear_attn.A_log",
            "language_model.model.layers.*.linear_attn.dt_bias",
        ]
        assert "*" not in updates["include_patterns"]
        assert "class:*Block" not in updates["include_patterns"]

    def test_pattern_resolver_returns_copies_of_defaults(self, worker_module) -> None:
        config = SimpleNamespace(include_patterns=["*"], exclude_patterns=[])

        includes, excludes = worker_module.resolve_vllm_patterns(config)
        includes.append("mutated")
        excludes.append("mutated")

        assert "mutated" not in worker_module.VLLM_DEFAULT_INCLUDE_PATTERNS
        assert "mutated" not in worker_module.VLLM_DEFAULT_EXCLUDE_PATTERNS

    def test_explicit_empty_exclude_patterns_are_preserved(self, worker_module) -> None:
        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["*"],
            exclude_patterns=[],
            model_fields_set={"exclude_patterns"},
        )

        updates = worker_module._vllm_config_updates(config)

        assert "exclude_patterns" not in updates

    def test_custom_include_and_exclude_patterns_are_preserved(self, worker_module) -> None:
        config = SimpleNamespace(
            discovery_iters=5,
            profiling_iters=4,
            include_patterns=["custom.layers.*"],
            exclude_patterns=["custom.layers.*.keep_on_gpu"],
        )

        updates = worker_module._vllm_config_updates(config)

        assert "include_patterns" not in updates
        assert "exclude_patterns" not in updates
        assert updates["discovery_iters"] == 5
        assert updates["profiling_iters"] == 4

    def test_external_compile_floors_profiling_iters_at_eager_seed(self, worker_module) -> None:
        from flextensor.compile import COMPILED_EAGER_PROFILE_FORWARDS

        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            external_compile=True,
            include_patterns=["custom.layers.*"],
            exclude_patterns=[],
        )

        updates = worker_module._vllm_config_updates(config)

        assert updates["profiling_iters"] == COMPILED_EAGER_PROFILE_FORWARDS


class TestVllmSkipDiscoveryPin:
    """vLLM worker must pin ``skip_discovery=False`` regardless of user config.

    Rationale: :meth:`FlexTensorOffloadWorker.warmup_and_profile_model` is
    hand-scheduled against the DISCOVERY→PROFILING transition landing between
    the first ``compile_or_warm_up_model()`` (2 forwards) and the explicit
    max-token profile loop. With ``skip_discovery=True`` the manager
    short-circuits directly to PROFILING inside ``offload()``, so vLLM's
    warmup forwards land in the profile budget — at ``profiling_iters=2``
    (the vLLM floor) the state machine reaches INFERENCE before the
    max-token loop even starts and zero max-token samples are profiled.
    """

    def test_skip_discovery_false_when_user_did_not_set(self, worker_module) -> None:
        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["*"],
            exclude_patterns=[],
            skip_discovery=True,  # caller opted in; worker must still veto
        )

        updates = worker_module._vllm_config_updates(config)

        assert updates["skip_discovery"] is False

    def test_skip_discovery_false_overrides_explicit_true(self, worker_module) -> None:
        """Even when the caller explicitly opts in, the vLLM worker must veto.

        The worker's warmup schedule is baked around a discovery→profile
        transition; honoring the caller's ``True`` here would silently
        collapse the profile budget for max-token forwards.
        """
        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["*"],
            exclude_patterns=[],
            skip_discovery=True,
            model_fields_set={"skip_discovery"},
        )

        updates = worker_module._vllm_config_updates(config)

        assert updates["skip_discovery"] is False, (
            "vLLM worker must veto skip_discovery=True; leaving it True would "
            "route compile_or_warm_up_model forwards into the profile budget "
            "(profiling_iters=2 floor → zero max-token samples profiled)"
        )

    def test_skip_discovery_false_preserved_when_user_already_false(self, worker_module) -> None:
        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["*"],
            exclude_patterns=[],
            skip_discovery=False,
        )

        updates = worker_module._vllm_config_updates(config)

        assert updates["skip_discovery"] is False


@dataclass
class _FakeCompileModule:
    """Stand-in for a ``@support_torch_compile`` module exposing ``do_not_compile``."""

    do_not_compile: bool
    compiled: bool = False


class _FakeRoot:
    def __init__(self, modules: list[object]) -> None:
        self._modules = modules

    def modules(self):
        return iter(self._modules)


class TestCompiledOffloadCompileGate:
    """The compiled-offload compile gate must defer the single compile to inference."""

    def _gate(self, worker_module):
        return worker_module.FlexTensorOffloadWorker._ft_set_compiled_offload_compile_enabled

    def test_noop_when_flag_disabled(self, worker_module, monkeypatch) -> None:
        monkeypatch.setattr(
            worker_module.flextensor,
            "get_offload_manager",
            lambda: SimpleNamespace(compiled_offload_active=False),
        )
        fake_self = SimpleNamespace()

        self._gate(worker_module)(fake_self, False)

    def test_disable_then_enable_round_trip(self, worker_module, monkeypatch) -> None:
        compilable_a = _FakeCompileModule(do_not_compile=False)
        compilable_b = _FakeCompileModule(do_not_compile=False)
        already_ignored = _FakeCompileModule(do_not_compile=True)
        plain = SimpleNamespace()  # no do_not_compile attribute
        root = _FakeRoot([compilable_a, already_ignored, plain, compilable_b])
        monkeypatch.setattr(
            worker_module.flextensor,
            "get_offload_manager",
            lambda: SimpleNamespace(model=root, compiled_offload_active=True),
        )
        fake_self = SimpleNamespace()
        gate = self._gate(worker_module)

        # Disable: only the two compilable modules get silenced; ignored/plain untouched.
        gate(fake_self, False)
        assert compilable_a.do_not_compile is True
        assert compilable_b.do_not_compile is True
        assert already_ignored.do_not_compile is True
        assert not hasattr(plain, "do_not_compile")
        assert fake_self._ft_compile_gated_modules == [compilable_a, compilable_b]

        # Simulate a stray compile having latched on one module.
        compilable_a.compiled = True

        # Enable: only the previously-gated modules are restored, and any latched
        # ``compiled`` flag is cleared so inference takes the first-compile path.
        gate(fake_self, True)
        assert compilable_a.do_not_compile is False
        assert compilable_b.do_not_compile is False
        assert compilable_a.compiled is False
        assert already_ignored.do_not_compile is True
        assert fake_self._ft_compile_gated_modules is None

    def test_duplicate_disable_preserves_gated_modules(self, worker_module, monkeypatch) -> None:
        compilable = _FakeCompileModule(do_not_compile=False)
        root = _FakeRoot([compilable])
        monkeypatch.setattr(
            worker_module.flextensor,
            "get_offload_manager",
            lambda: SimpleNamespace(model=root, compiled_offload_active=True),
        )
        fake_self = SimpleNamespace()
        gate = self._gate(worker_module)

        gate(fake_self, False)
        gate(fake_self, False)

        assert fake_self._ft_compile_gated_modules == [compilable]
        gate(fake_self, True)
        assert compilable.do_not_compile is False


class TestConfigureVllmCompileEnvForCompiledOffload:
    """Import-time vLLM compile env must not silently misconfigure AOT/cache."""

    def test_disables_cache_and_leaves_aot_unset(self, worker_module, monkeypatch, caplog) -> None:
        monkeypatch.delenv("VLLM_DISABLE_COMPILE_CACHE", raising=False)
        monkeypatch.delenv("VLLM_USE_AOT_COMPILE", raising=False)

        with caplog.at_level(logging.WARNING, logger=worker_module.LOGGER.name):
            worker_module._configure_vllm_compile_env_for_compiled_offload()

        assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
        assert "VLLM_USE_AOT_COMPILE" not in os.environ
        messages = [record.message for record in caplog.records]
        assert any("compiled-offload running under vLLM native fullgraph=True" in m for m in messages)
        assert not any("set directly" in m for m in messages)
        assert not any("correcting conflicting" in m for m in messages)
        assert not any("overriding explicit" in m for m in messages)

    def test_warns_before_overriding_explicit_cache_setting(self, worker_module, monkeypatch, caplog) -> None:
        monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
        monkeypatch.delenv("VLLM_USE_AOT_COMPILE", raising=False)

        with caplog.at_level(logging.WARNING, logger=worker_module.LOGGER.name):
            worker_module._configure_vllm_compile_env_for_compiled_offload()

        assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
        assert any("overriding explicit VLLM_DISABLE_COMPILE_CACHE=0" in r.message for r in caplog.records)

    def test_warns_and_corrects_explicit_aot_one(self, worker_module, monkeypatch, caplog) -> None:
        monkeypatch.delenv("VLLM_DISABLE_COMPILE_CACHE", raising=False)
        monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")

        with caplog.at_level(logging.WARNING, logger=worker_module.LOGGER.name):
            worker_module._configure_vllm_compile_env_for_compiled_offload()

        assert os.environ["VLLM_USE_AOT_COMPILE"] == "0"
        assert any("correcting conflicting VLLM_USE_AOT_COMPILE=1" in r.message for r in caplog.records)

    def test_leaves_explicit_aot_zero_alone(self, worker_module, monkeypatch, caplog) -> None:
        monkeypatch.delenv("VLLM_DISABLE_COMPILE_CACHE", raising=False)
        monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")

        with caplog.at_level(logging.WARNING, logger=worker_module.LOGGER.name):
            worker_module._configure_vllm_compile_env_for_compiled_offload()

        assert os.environ["VLLM_USE_AOT_COMPILE"] == "0"
        assert not any("correcting conflicting" in r.message for r in caplog.records)
        assert not any("VLLM_USE_AOT_COMPILE=0 set" in r.message for r in caplog.records)


class TestWarnVllmRuntimeRequirements:
    """Plain offload vs external_compile get different runtime warnings."""

    def test_plain_offload_warns_for_missing_enforce_eager(self, worker_module, caplog) -> None:
        offload_config = SimpleNamespace(external_compile=False)
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False))
        with caplog.at_level(logging.WARNING, logger=worker_module.LOGGER.name):
            worker_module._warn_vllm_runtime_requirements(offload_config, vllm_config)
        assert any("enforce-eager" in record.message for record in caplog.records)
        assert not any("cudagraph_mode" in record.message for record in caplog.records)

    def test_plain_offload_silent_when_enforce_eager(self, worker_module, caplog) -> None:
        offload_config = SimpleNamespace(external_compile=False)
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))
        with caplog.at_level(logging.WARNING, logger=worker_module.LOGGER.name):
            worker_module._warn_vllm_runtime_requirements(offload_config, vllm_config)
        assert not caplog.records

    def test_external_compile_warns_for_cudagraphs_not_enforce_eager(self, worker_module, caplog) -> None:
        offload_config = SimpleNamespace(external_compile=True)
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            compilation_config=SimpleNamespace(cudagraph_mode="PIECEWISE"),
        )
        with caplog.at_level(logging.WARNING, logger=worker_module.LOGGER.name):
            worker_module._warn_vllm_runtime_requirements(offload_config, vllm_config)
        assert any("cudagraph_mode" in record.message for record in caplog.records)
        assert not any("enforce-eager" in record.message for record in caplog.records)

    def test_external_compile_silent_when_cudagraph_mode_none(self, worker_module, caplog) -> None:
        offload_config = SimpleNamespace(external_compile=True)
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            compilation_config=SimpleNamespace(cudagraph_mode="NONE"),
        )
        with caplog.at_level(logging.WARNING, logger=worker_module.LOGGER.name):
            worker_module._warn_vllm_runtime_requirements(offload_config, vllm_config)
        assert not caplog.records

    def test_is_cudagraph_mode_none_accepts_enum_name(self, worker_module) -> None:
        assert worker_module._is_cudagraph_mode_none(SimpleNamespace(name="NONE")) is True
        assert worker_module._is_cudagraph_mode_none(SimpleNamespace(name="PIECEWISE")) is False
        assert worker_module._is_cudagraph_mode_none(0) is True

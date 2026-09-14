# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :meth:`flextensor.OffloadManager.request_strategy_replan`."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from flextensor.compile.lifecycle import COMPILED_WARMUP_FORWARDS
from flextensor.compile.warmup_tail import CompiledOffloadTailState
from flextensor.config import OffloadConfig
from flextensor.offload_manager import OffloadManager, OffloadPhase


@pytest.fixture
def fake_custom_ops():
    mod = types.ModuleType("flextensor.custom_ops")
    mod.install_active_loader = MagicMock(name="install_active_loader")
    mod.clear_active_loader = MagicMock(name="clear_active_loader")
    mod.enable_compiled_profiling = MagicMock(name="enable_compiled_profiling")
    mod.disable_compiled_profiling = MagicMock(name="disable_compiled_profiling")
    mod.finish_compiled_profiling = MagicMock(name="finish_compiled_profiling", return_value={})

    name = "flextensor.custom_ops"
    previous = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        yield mod
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


@pytest.fixture
def manager(monkeypatch):
    from flextensor.loaders import PreallocatedLoader

    class _StubLoader(PreallocatedLoader):
        def __init__(self) -> None:
            pass

        def enter(self, label: str) -> None:
            return None

        def exit(self, label: str) -> None:
            return None

        def preload(self) -> None:
            return None

        def prepare(self) -> None:
            return None

    tm = SimpleNamespace(
        tensor_layer_loader=_StubLoader(),
        replan_from_compiled_durations=MagicMock(name="replan_from_compiled_durations", return_value=True),
        replan_source_weights_available=True,
    )
    om = OffloadManager("default")
    om.set_config(OffloadConfig(profiling_iters=10))
    co = om._compiled
    co.compile_fn = None
    co.active = True
    co.replan_active = True
    om._tensor_manager = tm
    om._model = object()
    om._current_phase = OffloadPhase.INFERENCE
    layer0 = MagicMock(_ft_offload_name="l0")
    layer1 = MagicMock(_ft_offload_name="l1")
    om._patched_modules = [layer0, layer1]
    return SimpleNamespace(om=om, tm=tm)


def test_request_strategy_replan_returns_replan_iters(manager, fake_custom_ops):
    assert manager.om.request_strategy_replan() == COMPILED_WARMUP_FORWARDS + 10
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.WARMING
    assert manager.om._compiled.warm_seen == 0
    fake_custom_ops.enable_compiled_profiling.assert_not_called()


def test_request_strategy_replan_rides_callers_forwards(manager, fake_custom_ops):
    fake_custom_ops.finish_compiled_profiling.return_value = {"l0": [1.0, 2.0, 3.0], "l1": [4.0]}

    replan_iters = manager.om.request_strategy_replan()
    assert replan_iters == COMPILED_WARMUP_FORWARDS + 10

    for _ in range(replan_iters):
        manager.om.update_state()

    assert manager.om._compiled.tail_state == CompiledOffloadTailState.DONE
    durations_arg, model_arg = manager.tm.replan_from_compiled_durations.call_args.args
    assert durations_arg == {"l0": 2.0, "l1": 4.0}
    assert model_arg is manager.om._model
    fake_custom_ops.clear_active_loader.assert_called_once()
    fake_custom_ops.finish_compiled_profiling.assert_called_once()


def test_request_strategy_replan_finishes_when_profiling_iters_zero(manager, fake_custom_ops):
    """profiling_iters=0 must not leave the tail stuck in MEASURING."""
    manager.om.set_config(OffloadConfig(profiling_iters=0))
    fake_custom_ops.finish_compiled_profiling.return_value = {}

    replan_iters = manager.om.request_strategy_replan()
    assert replan_iters == COMPILED_WARMUP_FORWARDS

    for _ in range(replan_iters):
        manager.om.update_state()

    assert manager.om._compiled.tail_state == CompiledOffloadTailState.DONE
    fake_custom_ops.finish_compiled_profiling.assert_called_once()
    manager.tm.replan_from_compiled_durations.assert_not_called()


def test_arm_replan_tail_finishes_immediately_when_warmup_credited_and_measure_zero(manager, fake_custom_ops):
    manager.om.set_config(OffloadConfig(profiling_iters=0))
    fake_custom_ops.finish_compiled_profiling.return_value = {}

    remaining = manager.om._compiled.arm_replan_tail(compiled_warm_forwards=COMPILED_WARMUP_FORWARDS)
    assert remaining == 0
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.DONE
    fake_custom_ops.finish_compiled_profiling.assert_called_once()


def test_arm_replan_tail_enable_profiling_false_survives_warmup_transition(manager, fake_custom_ops):
    """enable_profiling=False must also gate WARMING → MEASURING, not only immediate arm."""
    finish = MagicMock(return_value=True)

    remaining = manager.om._compiled.arm_replan_tail(
        compiled_warm_forwards=0,
        enable_profiling=False,
        finish_replan=finish,
    )
    assert remaining == COMPILED_WARMUP_FORWARDS + 10
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.WARMING
    fake_custom_ops.enable_compiled_profiling.assert_not_called()

    for _ in range(remaining):
        manager.om._compiled.advance_tail(finish_replan=finish)

    assert manager.om._compiled.tail_state == CompiledOffloadTailState.DONE
    fake_custom_ops.enable_compiled_profiling.assert_not_called()
    finish.assert_called_once()


def test_request_strategy_replan_works_with_compile_fn_when_replan_armed(manager, fake_custom_ops):
    """compile_fn + non-view (replan_active) may request a post-compile rebuild."""
    manager.om._compiled.compile_fn = lambda m: m
    manager.om._compiled.replan_active = True

    assert manager.om.request_strategy_replan() == COMPILED_WARMUP_FORWARDS + 10
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.WARMING


def test_request_strategy_replan_noop_when_compile_fn_without_replan_arm(manager, fake_custom_ops, caplog):
    """Default compile_fn + view-profile leaves replan_active=False; do not arm a no-op tail."""
    manager.om._compiled.compile_fn = lambda m: m
    manager.om._compiled.replan_active = False

    with caplog.at_level("WARNING"):
        assert manager.om.request_strategy_replan() == 0
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.IDLE
    assert any("replan was not armed" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("offload", "replan"),
    [(False, True), (True, False), (False, False)],
    ids=["offload-off", "replan-off", "both-off"],
)
def test_request_strategy_replan_noop_when_flags_off(manager, monkeypatch, offload, replan, caplog):
    manager.om._compiled.active = offload
    manager.om._compiled.replan_active = replan
    with caplog.at_level("WARNING"):
        assert manager.om.request_strategy_replan() == 0
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.IDLE
    assert any("request_strategy_replan() ignored" in r.message for r in caplog.records)

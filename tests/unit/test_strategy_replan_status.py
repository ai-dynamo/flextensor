# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the passive external-compile / strategy-replan status API."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from flextensor.compile import StrategyReplanStatus
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
    mod.finish_compiled_profiling = MagicMock(name="finish_compiled_profiling", return_value={"l0": [1.0]})

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
def manager():
    """External-compile manager at INFERENCE with source weights retained."""
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
    om._patched_modules = [MagicMock(_ft_offload_name="l0")]
    return SimpleNamespace(om=om, tm=tm)


def _drive(om, iters):
    for _ in range(iters):
        om.update_state()


# -- external_compile_active -------------------------------------------------


def test_external_compile_active_for_live_external_compile_run(manager):
    assert manager.om.external_compile_active is True


def test_external_compile_active_false_with_compile_fn(manager):
    """FlexTensor-driven compile is not the external-compile contract."""
    manager.om._compiled.compile_fn = lambda m: m
    assert manager.om.external_compile_active is False


def test_external_compile_active_false_for_eager_offload(manager):
    manager.om._compiled.active = False
    assert manager.om.external_compile_active is False


def test_external_compile_active_ignores_config_mutation_after_offload(manager):
    """The property describes the resolved run, not the mutable config value."""
    manager.om._compiled.active = False
    manager.om.set_config(OffloadConfig(profiling_iters=10, external_compile=True))
    assert manager.om.config.external_compile is True
    assert manager.om.external_compile_active is False


# -- can_request -------------------------------------------------------------


def test_status_idle_and_requestable_before_first_request(manager):
    status = manager.om.strategy_replan_status
    assert status.state == "idle"
    assert status.can_request is True
    assert status.applied is None


def test_reading_status_does_not_arm_the_tail(manager, fake_custom_ops):
    for _ in range(3):
        assert manager.om.strategy_replan_status.state == "idle"
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.IDLE
    fake_custom_ops.enable_compiled_profiling.assert_not_called()


@pytest.mark.parametrize(
    "phase",
    [OffloadPhase.NOT_INITIALIZED, OffloadPhase.DISCOVERY, OffloadPhase.PROFILING],
    ids=["not-initialized", "discovery", "profiling"],
)
def test_cannot_request_before_inference(manager, phase):
    manager.om._current_phase = phase
    assert manager.om.strategy_replan_status.can_request is False


def test_cannot_request_when_replan_was_not_armed(manager):
    """Compiled view-profile keeps no source weights, so replan is unavailable."""
    manager.om._compiled.replan_active = False
    assert manager.om.strategy_replan_status.can_request is False


def test_cannot_request_when_compiled_offload_inactive(manager):
    manager.om._compiled.active = False
    assert manager.om.strategy_replan_status.can_request is False


def test_cannot_request_without_source_weights(manager):
    manager.tm.replan_source_weights_available = False
    assert manager.om.strategy_replan_status.can_request is False


def test_cannot_request_without_tensor_manager(manager):
    manager.om._tensor_manager = None
    assert manager.om.strategy_replan_status.can_request is False


# -- lifecycle transitions ---------------------------------------------------


def test_in_progress_while_tail_rides_forwards(manager, fake_custom_ops):
    replan_iters = manager.om.request_strategy_replan()
    assert replan_iters == COMPILED_WARMUP_FORWARDS + 10

    for remaining in range(replan_iters, 0, -1):
        status = manager.om.strategy_replan_status
        assert status.state == "in_progress", f"{remaining} forwards left"
        assert status.can_request is False
        assert status.applied is None
        manager.om.update_state()

    assert manager.om.strategy_replan_status.state == "done"


def test_done_applied_after_successful_rebuild(manager, fake_custom_ops):
    manager.tm.replan_from_compiled_durations.return_value = True

    _drive(manager.om, manager.om.request_strategy_replan())

    status = manager.om.strategy_replan_status
    assert status.state == "done"
    assert status.applied is True


def test_successful_rebuild_consumes_source_weights(manager, fake_custom_ops):
    """The real TensorManager drops the snapshot on success; no retry is offered."""

    def rebuild(*_args):
        manager.tm.replan_source_weights_available = False
        return True

    manager.tm.replan_from_compiled_durations.side_effect = rebuild

    _drive(manager.om, manager.om.request_strategy_replan())

    status = manager.om.strategy_replan_status
    assert status.applied is True
    assert status.can_request is False


def test_done_refused_stays_requestable_while_weights_are_retained(manager, fake_custom_ops):
    """A soft refusal keeps the snapshot, so the caller may retry."""
    manager.tm.replan_from_compiled_durations.return_value = False

    _drive(manager.om, manager.om.request_strategy_replan())

    status = manager.om.strategy_replan_status
    assert status.state == "done"
    assert status.applied is False
    assert status.can_request is True


def test_refused_replan_can_be_requested_again(manager, fake_custom_ops):
    manager.tm.replan_from_compiled_durations.return_value = False
    _drive(manager.om, manager.om.request_strategy_replan())
    assert manager.om.strategy_replan_status.applied is False

    manager.tm.replan_from_compiled_durations.return_value = True
    second_iters = manager.om.request_strategy_replan()
    assert second_iters == COMPILED_WARMUP_FORWARDS + 10

    status = manager.om.strategy_replan_status
    assert status.state == "in_progress"
    assert status.applied is None

    _drive(manager.om, second_iters)
    assert manager.om.strategy_replan_status.applied is True


def test_failed_rebuild_reports_failed_and_blocks_further_requests(manager, fake_custom_ops):
    manager.tm.replan_from_compiled_durations.side_effect = RuntimeError("rebuild exploded")

    replan_iters = manager.om.request_strategy_replan()
    with pytest.raises(RuntimeError, match="rebuild exploded"):
        _drive(manager.om, replan_iters)

    status = manager.om.strategy_replan_status
    assert status.state == "failed"
    assert status.applied is False
    assert status.can_request is False


def test_request_after_failure_is_refused_and_keeps_the_failure(manager, fake_custom_ops, caplog):
    """Re-arming would drop the failure and disarm the unsafe-inference guard."""
    manager.tm.replan_from_compiled_durations.side_effect = RuntimeError("rebuild exploded")
    with pytest.raises(RuntimeError, match="rebuild exploded"):
        _drive(manager.om, manager.om.request_strategy_replan())

    with caplog.at_level("WARNING"):
        assert manager.om.request_strategy_replan() == 0
    assert any("an earlier replan failed" in record.message for record in caplog.records)

    assert manager.om._compiled.tail_state == CompiledOffloadTailState.FAILED
    assert isinstance(manager.om._compiled.tail_failure, RuntimeError)
    assert manager.om.strategy_replan_status == StrategyReplanStatus(state="failed", can_request=False, applied=False)
    # The tail must still refuse to advance rather than measure a dead loader.
    with pytest.raises(RuntimeError, match="re-plan previously failed"):
        manager.om.update_state()


def test_request_while_tail_in_flight_is_refused_and_keeps_progress(manager, fake_custom_ops, caplog):
    """A second request must not silently restart the warmup counter."""
    first_iters = manager.om.request_strategy_replan()
    manager.om.update_state()
    assert manager.om._compiled.warm_seen == 1

    with caplog.at_level("WARNING"):
        assert manager.om.request_strategy_replan() == 0
    assert any("already in flight" in record.message for record in caplog.records)

    assert manager.om._compiled.warm_seen == 1
    assert manager.om.strategy_replan_status.state == "in_progress"

    # The original count still finishes the tail it armed.
    _drive(manager.om, first_iters - 1)
    assert manager.om.strategy_replan_status.applied is True


def test_request_before_inference_is_refused(manager, fake_custom_ops, caplog):
    manager.om._current_phase = OffloadPhase.PROFILING

    with caplog.at_level("WARNING"):
        assert manager.om.request_strategy_replan() == 0
    assert any("still in profiling" in record.message for record in caplog.records)
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.IDLE


def test_request_without_source_weights_is_refused(manager, fake_custom_ops, caplog):
    """A rebuild without the snapshot is refused anyway; do not spend forwards on it."""
    manager.tm.replan_source_weights_available = False

    with caplog.at_level("WARNING"):
        assert manager.om.request_strategy_replan() == 0
    assert any("no longer retains the source weights" in record.message for record in caplog.records)
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.IDLE


def test_view_profile_tail_done_without_request_reports_idle(manager, fake_custom_ops):
    """``setup_inference_no_replan`` closes the tail without any replan request."""
    manager.om._model = torch.nn.Module()
    manager.om._patched_modules = []
    manager.om._compiled.compile_fn = lambda m: m
    manager.om._compiled.replan_active = False
    manager.om._compiled.setup_inference_no_replan()
    assert manager.om._compiled.tail_state == CompiledOffloadTailState.DONE

    status = manager.om.strategy_replan_status
    assert status.state == "idle"
    assert status.applied is None
    assert status.can_request is False


def test_status_is_immutable(manager):
    status = manager.om.strategy_replan_status
    with pytest.raises(AttributeError):
        status.can_request = False

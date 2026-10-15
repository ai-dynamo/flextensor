# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CUDA-graph budget re-planning."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from flextensor.offload_manager import OffloadManager
from flextensor.offload_timing import (
    OffloadTimingCollector,
    OffloadTimingReport,
    OffloadTimingSnapshot,
    TrapTimingRecord,
    TrapTimingStats,
)


def _layer_stat(label: str, duration: float = 3.0) -> SimpleNamespace:
    return SimpleNamespace(
        label=label,
        duration=duration,
        model_copy=lambda **kw: SimpleNamespace(label=label, duration=kw.get("duration", duration)),
    )


class TestGraphAwareReplanMapping:
    def test_durations_mapped_by_trap_index(self) -> None:
        labels = ["model.layers.0", "model.layers.1"]
        report = OffloadTimingReport(
            per_trap=[
                TrapTimingStats(label="layers.0", compute_avg=0.45, compute_median=0.17),
                TrapTimingStats(label="layers.1", compute_avg=0.50, compute_median=0.21),
            ],
            passes=tuple(
                OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="layers.0", compute_ms=0.1),)) for _ in range(1)
            ),
        )
        durations = report.compute_budgets_by_profile_label(labels)
        assert durations == {
            "model.layers.0": 0.17,
            "model.layers.1": 0.21,
        }

    def test_skips_non_positive_budgets(self) -> None:
        report = OffloadTimingReport(
            per_trap=[TrapTimingStats(label="layers.0", compute_avg=0.0)],
            passes=tuple(
                OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="layers.0", compute_ms=0.1),)) for _ in range(1)
            ),
        )
        assert report.compute_budgets_by_profile_label(["model.layers.0"]) == {}

    def test_conservative_uses_compute_min(self) -> None:
        labels = ["model.layers.0", "model.layers.1"]
        report = OffloadTimingReport(
            per_trap=[
                TrapTimingStats(label="layers.0", compute_min=0.15, compute_median=0.17),
                TrapTimingStats(label="layers.1", compute_min=0.18, compute_median=0.20),
            ],
            passes=tuple(
                OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="layers.0", compute_ms=0.1),)) for _ in range(2)
            ),
        )
        assert report.compute_budgets_by_profile_label(labels, conservative=True) == {
            "model.layers.0": 0.15,
            "model.layers.1": 0.18,
        }

    def test_warns_on_timing_vs_profile_length_mismatch(self, caplog: pytest.LogCaptureFixture) -> None:
        report = OffloadTimingReport(
            per_trap=[
                TrapTimingStats(label="layers.0", compute_avg=0.45, compute_median=0.17),
            ],
            passes=tuple(
                OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="layers.0", compute_ms=0.1),)) for _ in range(1)
            ),
        )
        with caplog.at_level("WARNING"):
            durations = report.compute_budgets_by_profile_label(["model.layers.0", "model.layers.1"])
        assert durations == {"model.layers.0": 0.17}
        assert any("trap count (1) != profile label count (2)" in r.message for r in caplog.records)


class TestGraphAwareReplanRebuild:
    @pytest.fixture
    def manager(self) -> OffloadManager:
        om = OffloadManager("test_graph_replan")
        om._compiled.active = True  # noqa: SLF001
        om._compiled.replan_active = True  # noqa: SLF001
        om._tensor_manager = MagicMock()
        om._tensor_manager.stats = [_layer_stat("model.layers.0")]
        om._model = MagicMock()
        return om

    @staticmethod
    def _attach_collector(manager: OffloadManager, *, external_events: bool = True) -> OffloadTimingCollector:
        # Real collector so beartype accepts ``_offload_timing_collector``; force
        # ``_external_events`` so CUDA-graph gate tests do not depend on the probe.
        collector = OffloadTimingCollector(trap_labels=[], enabled=True, log_every=0)
        collector._external_events = external_events
        collector.finalize_replay_pass = MagicMock(return_value=True)  # type: ignore[method-assign]
        collector.disarm_replay_measure = MagicMock()  # type: ignore[method-assign]
        loader = MagicMock()
        loader.offload_timing_collector = collector
        manager._tensor_manager.tensor_layer_loader = loader
        return collector

    def test_manual_update_state_arms_then_rebuilds_on_last_update(self, manager: OffloadManager) -> None:
        """CUDA-graph path: arm → N x update_state → rebuild on last."""
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager._tensor_manager.stats = [_layer_stat("model.layers.0")]
        manager._tensor_manager.replan_from_compiled_durations.return_value = True
        manager._tensor_manager._drain_offload_timing_measure.return_value = OffloadTimingReport(
            per_trap=[TrapTimingStats(label="layers.0", compute_avg=0.18, compute_median=0.18)],
            passes=tuple(
                OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="layers.0", compute_ms=0.1),)) for _ in range(2)
            ),
        )
        manager.config = SimpleNamespace(
            offload_timing="cuda_graph",
            profiling_iters=2,
        )
        collector = self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing") as reset,
            patch.object(manager, "_arm_offload_timing_after_capture") as arm,
            patch.object(manager._compiled, "reinstall_compiled_loader") as reinstall,
        ):
            iters = manager.request_strategy_replan(manual_update_state=True)
            assert iters == 2
            assert manager._manual_update_state is True  # noqa: SLF001
            reset.assert_called_once_with()
            arm.assert_called_once_with()

            manager.update_state()
            assert manager._manual_update_state is True  # noqa: SLF001
            assert collector.finalize_replay_pass.call_count == 1

            manager.update_state()
            assert manager._manual_update_state is False  # noqa: SLF001
            assert collector.finalize_replay_pass.call_count == 2
            manager._tensor_manager.replan_from_compiled_durations.assert_called_once()
            reinstall.assert_called_once()
            assert manager._compiled._tail.replan_applied is True  # noqa: SLF001

    def test_manual_update_empty_measure_sets_replan_applied_false(self, manager: OffloadManager) -> None:
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager._tensor_manager._drain_offload_timing_measure.return_value = None
        manager.config = SimpleNamespace(
            offload_timing="cuda_graph",
            profiling_iters=1,
        )
        self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
            patch.object(manager._compiled, "reinstall_compiled_loader") as reinstall,
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 1
            manager.update_state()

        assert manager._compiled.tail_state.name == "DONE"
        assert manager._compiled._tail.replan_applied is False  # noqa: SLF001
        manager._tensor_manager.replan_from_compiled_durations.assert_not_called()
        reinstall.assert_not_called()

    @pytest.mark.parametrize(
        "empty_report",
        [
            # Isolate num_passes<=0 (per_trap present so only the passes guard fires).
            OffloadTimingReport(
                per_trap=(TrapTimingStats(label="layers.0", compute_min=0.11),),
                passes=(),
            ),
            # Isolate empty per_trap (passes present so only the per_trap guard fires).
            OffloadTimingReport(
                per_trap=(),
                passes=(OffloadTimingSnapshot(per_trap=()),),
            ),
        ],
        ids=["zero_passes", "passes_without_per_trap"],
    )
    def test_manual_update_structurally_empty_report_skips_rebuild(
        self,
        manager: OffloadManager,
        empty_report: OffloadTimingReport,
    ) -> None:
        """Zero passes or empty per_trap: keep loader, return False, still disarm."""
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager._tensor_manager._drain_offload_timing_measure.return_value = empty_report
        manager.config = SimpleNamespace(offload_timing="cuda_graph", profiling_iters=1)
        collector = self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
            patch.object(manager._compiled, "reinstall_compiled_loader") as reinstall,
            patch.object(manager, "_finish_replan_from_offload_timing") as finish_from_report,
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 1
            manager.update_state()

        assert manager._compiled.tail_state.name == "DONE"
        assert manager._compiled._tail.replan_applied is False  # noqa: SLF001
        finish_from_report.assert_not_called()
        manager._tensor_manager.replan_from_compiled_durations.assert_not_called()
        reinstall.assert_not_called()
        collector.disarm_replay_measure.assert_called()

    def test_update_offload_timing_forwards_replay_generation(self, manager: OffloadManager) -> None:
        manager.config = SimpleNamespace(offload_timing="eager")
        collector = self._attach_collector(manager)

        assert manager.update_offload_timing(replay_generation=7) is True
        collector.finalize_replay_pass.assert_called_once_with(replay_generation=7)

    def test_update_offload_timing_false_without_tensor_manager(self, manager: OffloadManager) -> None:
        manager._tensor_manager = None  # noqa: SLF001
        manager.config = SimpleNamespace(offload_timing="cuda_graph")
        assert manager.update_offload_timing() is False

    def test_update_offload_timing_true_when_timing_disabled(self, manager: OffloadManager) -> None:
        """Timing off is a deliberate no-op success — callers may still call after replay."""
        manager.config = SimpleNamespace(offload_timing="off")
        assert manager.update_offload_timing(replay_generation=3) is True

    def test_update_offload_timing_false_when_collector_missing(self, manager: OffloadManager) -> None:
        manager.config = SimpleNamespace(offload_timing="cuda_graph")
        manager._tensor_manager.tensor_layer_loader = None
        assert manager.update_offload_timing() is False

    def test_update_offload_timing_false_when_collector_disabled(self, manager: OffloadManager) -> None:
        manager.config = SimpleNamespace(offload_timing="cuda_graph")
        collector = self._attach_collector(manager)
        collector.enabled = False
        assert manager.update_offload_timing() is False
        collector.finalize_replay_pass.assert_not_called()

    def test_update_offload_timing_false_when_external_events_unavailable(
        self, manager: OffloadManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Public API must not publish after Event(external=True) soft-fallback."""
        manager.config = SimpleNamespace(offload_timing="cuda_graph")
        collector = OffloadTimingCollector(trap_labels=[], enabled=True, log_every=0)
        collector._external_events = False
        collector._pass_active = True
        collector._has_compute["L0"] = True
        loader = MagicMock()
        loader.offload_timing_collector = collector
        manager._tensor_manager.tensor_layer_loader = loader

        with (
            patch.object(collector, "_read_pass_snapshot") as read_snapshot,
            caplog.at_level("WARNING"),
        ):
            assert manager.update_offload_timing(replay_generation=1) is False

        read_snapshot.assert_not_called()
        assert collector._snapshots == []
        assert any("external CUDA timing events are unavailable" in r.message for r in caplog.records)

    def test_module_update_offload_timing_delegates(self) -> None:
        from flextensor import offload_manager as om_mod

        mock_om = MagicMock()
        mock_om.update_offload_timing.return_value = True
        with patch.object(om_mod, "get_offload_manager", return_value=mock_om) as get_om:
            assert om_mod.update_offload_timing("mgr", replay_generation=9) is True
        get_om.assert_called_once_with("mgr")
        mock_om.update_offload_timing.assert_called_once_with(replay_generation=9)

    def test_manual_update_finalize_failure_aborts_without_advance(self, manager: OffloadManager) -> None:
        from flextensor.compile.warmup_tail import CompiledOffloadTailState
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager.config = SimpleNamespace(
            offload_timing="cuda_graph",
            profiling_iters=2,
        )
        collector = self._attach_collector(manager)
        collector.finalize_replay_pass.side_effect = RuntimeError("cuda boom")

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
            patch.object(manager._compiled, "advance_tail") as advance,
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 2
            with pytest.raises(RuntimeError, match="update_offload_timing failed"):
                manager.update_state()

        advance.assert_not_called()
        assert manager._manual_update_state is False  # noqa: SLF001
        assert manager._compiled.tail_state == CompiledOffloadTailState.FAILED
        manager._tensor_manager.replan_from_compiled_durations.assert_not_called()
        collector.disarm_replay_measure.assert_called()

    def test_manual_update_finalize_noop_aborts_without_advance(self, manager: OffloadManager) -> None:
        from flextensor.compile.warmup_tail import CompiledOffloadTailState
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager.config = SimpleNamespace(
            offload_timing="cuda_graph",
            profiling_iters=2,
        )
        collector = self._attach_collector(manager)
        collector.finalize_replay_pass.return_value = False  # unexpected no-op

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
            patch.object(manager._compiled, "advance_tail") as advance,
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 2
            with pytest.raises(RuntimeError, match="update_offload_timing failed"):
                manager.update_state()

        advance.assert_not_called()
        assert manager._manual_update_state is False  # noqa: SLF001
        assert manager._compiled.tail_state == CompiledOffloadTailState.FAILED
        collector.disarm_replay_measure.assert_called()

    def test_manual_update_state_cleared_on_release_and_offload(self, manager: OffloadManager) -> None:
        """Aborting an armed CUDA-graph replan must not leak the flag across sessions."""
        import torch.nn as nn

        manager._manual_update_state = True  # noqa: SLF001
        manager.release()
        assert manager._manual_update_state is False  # noqa: SLF001

        manager._manual_update_state = True  # noqa: SLF001
        manager.config = SimpleNamespace(
            transfer_mode="allocation_block_transfer",
            profile_mode="view",
            external_compile=False,
            include_patterns=["*"],
            exclude_patterns=[],
        )
        with (
            patch.object(manager, "init"),
            patch.object(manager, "_offload_modules"),
            patch.object(manager, "_exclude_modules"),
            patch.object(manager, "_check_no_modules_patched"),
            patch.object(manager, "_transfer_hooks"),
            patch.object(manager, "_transition_to_warmup"),
            patch("flextensor.offload_manager.is_torch_compiled_module", return_value=False),
            patch.object(manager._compiled, "resolve_activation", return_value=False),
            patch.object(manager._compiled, "arm_non_destructive_first_loader"),
        ):
            manager.offload(nn.Linear(2, 2))
        assert manager._manual_update_state is False  # noqa: SLF001

    def test_manual_update_replan_refuses_without_offload_timing(
        self, manager: OffloadManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager.config = SimpleNamespace(offload_timing="off")
        with (
            patch.object(manager._compiled, "request_strategy_replan") as compiled_replan,
            caplog.at_level("WARNING"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 0
            compiled_replan.assert_not_called()
        assert manager._manual_update_state is False  # noqa: SLF001
        assert any("offload_timing='cuda_graph'" in r.message for r in caplog.records)

    def test_manual_update_replan_refuses_eager_timing(
        self, manager: OffloadManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager.config = SimpleNamespace(offload_timing="eager")
        with (
            patch.object(manager._compiled, "request_strategy_replan") as compiled_replan,
            caplog.at_level("WARNING"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 0
            compiled_replan.assert_not_called()
        assert manager._manual_update_state is False  # noqa: SLF001
        assert any("offload_timing='cuda_graph'" in r.message for r in caplog.records)

    def test_manual_update_replan_refuses_when_collector_lacks_external(
        self, manager: OffloadManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Config can request external events while the collector soft-fell back."""
        manager.config = SimpleNamespace(
            offload_timing="cuda_graph",
            profiling_iters=2,
        )
        self._attach_collector(manager, external_events=False)

        with (
            patch.object(manager, "reset_offload_timing") as reset,
            patch.object(manager, "_arm_offload_timing_after_capture") as arm,
            patch.object(manager._compiled, "arm_replan_tail") as arm_tail,
            caplog.at_level("WARNING"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 0
            reset.assert_not_called()
            arm.assert_not_called()
            arm_tail.assert_not_called()

        assert manager._manual_update_state is False  # noqa: SLF001
        assert any("fell back to internal events" in r.message for r in caplog.records)

    @pytest.mark.parametrize(
        ("active", "replan_active"),
        [
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_manual_update_replan_refuses_partial_compiled_state(
        self,
        manager: OffloadManager,
        active: bool,
        replan_active: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Both compiled.active and replan_active are required; otherwise no side effects."""
        manager._compiled.active = active  # noqa: SLF001
        manager._compiled.replan_active = replan_active  # noqa: SLF001
        manager.config = SimpleNamespace(offload_timing="cuda_graph", profiling_iters=2)
        self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing") as reset,
            patch.object(manager, "_arm_offload_timing_after_capture") as arm,
            patch.object(manager._compiled, "arm_replan_tail") as arm_tail,
            caplog.at_level("WARNING"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 0

        reset.assert_not_called()
        arm.assert_not_called()
        arm_tail.assert_not_called()
        assert manager._manual_update_state is False  # noqa: SLF001
        assert any("compiled replan was not armed" in r.message for r in caplog.records)

    def test_manual_update_replan_refuses_missing_collector(
        self, manager: OffloadManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager.config = SimpleNamespace(offload_timing="cuda_graph", profiling_iters=2)
        manager._tensor_manager.tensor_layer_loader = None

        with (
            patch.object(manager, "reset_offload_timing") as reset,
            patch.object(manager._compiled, "arm_replan_tail") as arm_tail,
            caplog.at_level("WARNING"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 0

        reset.assert_not_called()
        arm_tail.assert_not_called()
        assert manager._manual_update_state is False  # noqa: SLF001
        assert any("no enabled offload-timing collector" in r.message for r in caplog.records)

    def test_manual_update_replan_refuses_disabled_collector(
        self, manager: OffloadManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager.config = SimpleNamespace(offload_timing="cuda_graph", profiling_iters=2)
        collector = self._attach_collector(manager)
        collector.enabled = False

        with (
            patch.object(manager, "reset_offload_timing") as reset,
            patch.object(manager._compiled, "arm_replan_tail") as arm_tail,
            caplog.at_level("WARNING"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 0

        reset.assert_not_called()
        arm_tail.assert_not_called()
        assert manager._manual_update_state is False  # noqa: SLF001
        assert any("no enabled offload-timing collector" in r.message for r in caplog.records)

    def test_manual_update_replan_refuses_while_tail_in_flight(
        self, manager: OffloadManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A repeated manual request must not reset the in-flight measurement."""
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager.config = SimpleNamespace(offload_timing="cuda_graph", profiling_iters=2)
        self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 2
        manager._compiled.measure_seen = 1  # noqa: SLF001

        with (
            patch.object(manager, "reset_offload_timing") as reset,
            patch.object(manager._compiled, "arm_replan_tail") as arm_tail,
            caplog.at_level("WARNING"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 0

        reset.assert_not_called()
        arm_tail.assert_not_called()
        assert manager._compiled.measure_seen == 1  # noqa: SLF001
        assert any("already in flight" in r.message for r in caplog.records)

    def test_manual_update_replan_refuses_after_failure(
        self, manager: OffloadManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Re-arming would drop the failure that marks inference unsafe."""
        from flextensor.compile.warmup_tail import CompiledOffloadTailState
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager.config = SimpleNamespace(offload_timing="cuda_graph", profiling_iters=2)
        self._attach_collector(manager)
        failure = RuntimeError("rebuild exploded")
        manager._compiled.tail_state = CompiledOffloadTailState.FAILED  # noqa: SLF001
        manager._compiled.tail_failure = failure  # noqa: SLF001

        with (
            patch.object(manager, "reset_offload_timing") as reset,
            patch.object(manager._compiled, "arm_replan_tail") as arm_tail,
            caplog.at_level("WARNING"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 0

        reset.assert_not_called()
        arm_tail.assert_not_called()
        assert manager._manual_update_state is False  # noqa: SLF001
        assert manager._compiled.tail_state == CompiledOffloadTailState.FAILED  # noqa: SLF001
        assert manager._compiled.tail_failure is failure  # noqa: SLF001
        assert any("an earlier replan failed" in r.message for r in caplog.records)

    def test_manual_update_replan_arms_without_compiled_profiling(self, manager: OffloadManager) -> None:
        """CUDA-graph budgets come from offload timing — custom-op profiling stays off."""
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager.config = SimpleNamespace(offload_timing="cuda_graph", profiling_iters=2)
        self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
            patch.object(manager._compiled, "arm_replan_tail", return_value=2) as arm_tail,
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 2

        arm_tail.assert_called_once()
        kwargs = arm_tail.call_args.kwargs
        assert kwargs["enable_profiling"] is False
        assert kwargs["finish_replan"] == manager._finish_manual_update_replan

    def test_manual_update_replan_immediate_completion_when_measure_budget_zero(self, manager: OffloadManager) -> None:
        """profiling_iters=0 finishes inside arm_replan_tail via the completion callback."""
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager._tensor_manager.stats = [_layer_stat("model.layers.0")]
        manager._tensor_manager.replan_from_compiled_durations.return_value = True
        manager._tensor_manager._drain_offload_timing_measure.return_value = OffloadTimingReport(
            per_trap=[
                TrapTimingStats(
                    label="layers.0",
                    compute_min=0.11,
                    compute_median=0.22,
                    compute_avg=0.33,
                )
            ],
            passes=tuple(
                OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="layers.0", compute_ms=0.1),)) for _ in range(1)
            ),
        )
        manager.config = SimpleNamespace(offload_timing="cuda_graph", profiling_iters=0)
        self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
            patch.object(manager._compiled, "reinstall_compiled_loader") as reinstall,
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 0

        assert manager._manual_update_state is False  # noqa: SLF001
        assert manager._compiled._tail.enable_profiling is False  # noqa: SLF001
        manager._tensor_manager.replan_from_compiled_durations.assert_called_once_with(
            {"model.layers.0": 0.11},
            manager._model,
        )
        reinstall.assert_called_once()
        assert manager._compiled._tail.replan_applied is True  # noqa: SLF001

    def test_manual_update_replan_uses_conservative_compute_min(self, manager: OffloadManager) -> None:
        """Distinct min/median/avg — only the minimum must reach replan_from_compiled_durations."""
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager._tensor_manager.stats = [_layer_stat("model.layers.0")]
        manager._tensor_manager.replan_from_compiled_durations.return_value = True
        manager._tensor_manager._drain_offload_timing_measure.return_value = OffloadTimingReport(
            per_trap=[
                TrapTimingStats(
                    label="layers.0",
                    compute_min=0.11,
                    compute_median=0.22,
                    compute_avg=0.33,
                )
            ],
            passes=tuple(
                OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="layers.0", compute_ms=0.1),)) for _ in range(1)
            ),
        )
        manager.config = SimpleNamespace(offload_timing="cuda_graph", profiling_iters=1)
        self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
            patch.object(manager._compiled, "reinstall_compiled_loader"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 1
            manager.update_state()

        manager._tensor_manager.replan_from_compiled_durations.assert_called_once_with(
            {"model.layers.0": 0.11},
            manager._model,
        )

    def test_manual_update_finish_disarms_replay_measure(self, manager: OffloadManager) -> None:
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager._tensor_manager.stats = [_layer_stat("model.layers.0")]
        manager._tensor_manager.replan_from_compiled_durations.return_value = True
        manager._tensor_manager._drain_offload_timing_measure.return_value = OffloadTimingReport(
            per_trap=[TrapTimingStats(label="layers.0", compute_avg=0.18, compute_median=0.18)],
            passes=tuple(
                OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="layers.0", compute_ms=0.1),)) for _ in range(1)
            ),
        )
        manager.config = SimpleNamespace(
            offload_timing="cuda_graph",
            profiling_iters=1,
        )
        collector = self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
            patch.object(manager._compiled, "reinstall_compiled_loader"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 1
            manager.update_state()

        collector.disarm_replay_measure.assert_called()

    def test_manual_update_replan_warns_when_rebuild_fails(
        self, manager: OffloadManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        from flextensor.offload_manager import OffloadPhase

        manager._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        manager._tensor_manager.stats = [_layer_stat("model.layers.0")]
        manager._tensor_manager.replan_from_compiled_durations.return_value = False
        manager._tensor_manager._drain_offload_timing_measure.return_value = OffloadTimingReport(
            per_trap=[TrapTimingStats(label="layers.0", compute_avg=0.18, compute_median=0.18)],
            passes=tuple(
                OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="layers.0", compute_ms=0.1),)) for _ in range(1)
            ),
        )
        manager.config = SimpleNamespace(
            offload_timing="cuda_graph",
            profiling_iters=1,
        )
        self._attach_collector(manager)

        with (
            patch.object(manager, "reset_offload_timing"),
            patch.object(manager, "_arm_offload_timing_after_capture"),
            patch.object(manager._compiled, "reinstall_compiled_loader") as reinstall,
            caplog.at_level("WARNING"),
        ):
            assert manager.request_strategy_replan(manual_update_state=True) == 1
            manager.update_state()

        assert any("did not apply a new strategy" in r.message for r in caplog.records)
        reinstall.assert_not_called()
        manager._tensor_manager.replan_from_compiled_durations.assert_called_once()

    def test_collect_offload_timing_does_not_re_finalize_replay(self) -> None:
        """Durable collect must drain only — not finalize_replay_pass again."""
        import torch

        from flextensor.offload_timing import OffloadTimingSnapshot
        from flextensor.tensor_manager import TensorManager

        tm = TensorManager(
            device_gpu=torch.device("cpu"),
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
            _offload_timing="eager",
        )
        tm._offload_timing_measure.append(
            OffloadTimingSnapshot(
                per_trap=(
                    TrapTimingRecord(
                        label="L0",
                        wait_ms=1.0,
                        transfer_ms=2.0,
                        compute_ms=3.0,
                    ),
                ),
            )
        )
        collector = MagicMock()
        collector.enabled = True
        loader = MagicMock()
        loader.offload_timing_collector = collector
        tm.tensor_layer_loader = loader

        report = tm.collect_offload_timing()

        collector.finalize_replay_pass.assert_not_called()
        assert report is not None
        assert report.num_passes == 1
        assert len(tm._offload_timing_measure) == 0

    def test_offload_timing_measure_ring_buffer_cap(self) -> None:
        """Durable measure store drops oldest passes when maxlen is exceeded."""
        import torch

        from flextensor.offload_timing import OffloadTimingSnapshot
        from flextensor.tensor_manager import TensorManager

        tm = TensorManager(
            device_gpu=torch.device("cpu"),
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
            _offload_timing="eager",
            _offload_timing_measure_max_passes=3,
        )
        assert tm._offload_timing_measure.maxlen == 3

        for wait_ms in (1.0, 2.0, 3.0, 4.0, 5.0):
            tm._record_offload_timing_pass(
                OffloadTimingSnapshot(
                    per_trap=(TrapTimingRecord(label="L0", wait_ms=wait_ms),),
                )
            )

        assert len(tm._offload_timing_measure) == 3
        assert [s.total_wait_ms for s in tm._offload_timing_measure] == [3.0, 4.0, 5.0]

        report = tm.collect_offload_timing()
        assert report is not None
        assert report.num_passes == 3
        assert [s.total_wait_ms for s in report.passes] == [3.0, 4.0, 5.0]
        assert len(tm._offload_timing_measure) == 0

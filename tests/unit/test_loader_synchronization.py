# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for loader synchronization ordering.

Verifies that both PreallocatedBatchTransferTensorLoader and
PreallocatedBatchTransferTensorLoaderReordered issue CUDA synchronization
calls (wait_event, record_event) in the correct order relative to
schedule_transfer (data writes) and forward passes (data reads).

The tests use instrumented mocks that record every stream/event operation
into a shared call_log, allowing assertions on ordering without a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.core.transfer import CudaTransferRuntime
from flextensor.loaders import (
    PreallocatedBatchTransferTensorLoader,
    PreallocatedBatchTransferTensorLoaderReordered,
    RawBlockController,
)
from flextensor.offload_timing import OffloadTimingCollector

# ---------------------------------------------------------------------------
# Instrumented mock infrastructure
# ---------------------------------------------------------------------------


@dataclass
class MockEvent:
    """Mock CUDA event with completion tracking."""

    name: str
    completed: bool = False


@dataclass
class CallLogEntry:
    """Single operation in the call log."""

    op: str
    stream: str
    target: str
    stalled: bool = False

    def __repr__(self) -> str:
        stall = " STALL" if self.stalled else ""
        return f"{self.op}({self.stream}, {self.target}{stall})"


@dataclass
class SyncRecorder:
    """Records all stream/event operations in order."""

    log: list[CallLogEntry] = field(default_factory=list)
    _event_counter: int = 0

    def make_event(self, prefix: str = "ev", completed: bool = False) -> MockEvent:
        self._event_counter += 1
        return MockEvent(name=f"{prefix}_{self._event_counter}", completed=completed)

    def make_stream(self, name: str, events_completed: bool = True) -> MagicMock:
        """Create a mock stream.

        Args:
            name: Stream identifier for the log.
            events_completed: If False, record_event creates incomplete events
                (simulates slow transfers that haven't finished yet).
        """
        stream = MagicMock(name=name)
        recorder = self
        default_completed = events_completed

        def _wait_event(ev: MockEvent) -> None:
            stalled = not ev.completed
            recorder.log.append(CallLogEntry("wait_event", name, ev.name, stalled))
            ev.completed = True

        def _record_event() -> MockEvent:
            ev = recorder.make_event(prefix=f"{name}_ev", completed=default_completed)
            recorder.log.append(CallLogEntry("record_event", name, ev.name))
            return ev

        def _wait_stream(other: MagicMock) -> None:
            recorder.log.append(CallLogEntry("wait_stream", name, str(other.name)))

        stream.wait_event = _wait_event
        stream.record_event = _record_event
        stream.wait_stream = _wait_stream
        return stream

    def make_controller(self) -> MagicMock:
        controller = MagicMock(spec=RawBlockController)
        recorder = self

        def _schedule_transfer(label: str, non_blocking: bool = True) -> None:
            recorder.log.append(CallLogEntry("schedule_transfer", "transfer", label))

        controller.schedule_transfer = _schedule_transfer
        controller.get_gpu_memory_bytes = MagicMock(return_value=0)
        return controller

    def forward(self, label: str) -> None:
        """Simulate a layer's forward pass."""
        self.log.append(CallLogEntry("forward", "compute", label))

    def find_entries(self, op: str, target: str | None = None, stream: str | None = None) -> list[int]:
        """Find indices of log entries matching criteria."""
        results = []
        for i, entry in enumerate(self.log):
            if entry.op != op:
                continue
            if target is not None and entry.target != target:
                continue
            if stream is not None and entry.stream != stream:
                continue
            results.append(i)
        return results

    def find_first(self, op: str, target: str | None = None, stream: str | None = None) -> int:
        indices = self.find_entries(op, target, stream)
        if not indices:
            raise ValueError(f"No {op} entry found for target={target}, stream={stream}")
        return indices[0]

    def dump(self) -> str:
        return "\n".join(f"  [{i:3d}] {entry}" for i, entry in enumerate(self.log))


def assert_before(recorder: SyncRecorder, earlier_idx: int, later_idx: int, msg: str = "") -> None:
    """Assert that earlier_idx comes before later_idx in the log."""
    assert earlier_idx < later_idx, (
        f"Expected operation at [{earlier_idx}] before [{later_idx}]{' - ' + msg if msg else ''}\n"
        f"Log:\n{recorder.dump()}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_layer_stats(labels: list[str]) -> list[LayerStatistics]:
    """Create minimal LayerStatistics for testing."""
    tensor = TensorStatistics(tensor_id=1, name="t", size_bytes=1024, load_time_ms=0.1)
    return [LayerStatistics(label=label, tensors=[tensor], duration=1.0) for label in labels]


def _make_gap_layer_stats(labels: list[str], gap_labels: set[str]) -> list[LayerStatistics]:
    """Create LayerStatistics with gap layers (no tensors)."""
    tensor = TensorStatistics(tensor_id=1, name="t", size_bytes=1024, load_time_ms=0.1)
    stats = []
    for label in labels:
        if label in gap_labels:
            stats.append(LayerStatistics(label=label, tensors=[], duration=1.0))
        else:
            stats.append(LayerStatistics(label=label, tensors=[tensor], duration=1.0))
    return stats


def _build_pipeline_maps(labels: list[str], gap_labels: set[str]) -> tuple[dict[str, str], dict[str, int]]:
    """Derive transfer_to_compute_map and label_to_block_id from labels and gaps.

    Builds a standard 2-block alternating pipeline that bridges across gaps.
    embed is always the first transfer source.
    """
    offloaded = [lbl for lbl in labels if lbl not in gap_labels]
    transfer_to_compute_map = {}
    label_to_block_id = {}
    for i in range(len(offloaded) - 1):
        transfer_to_compute_map[offloaded[i]] = offloaded[i + 1]
        label_to_block_id[offloaded[i]] = i % 2
    return transfer_to_compute_map, label_to_block_id


@contextmanager
def _create_loader(
    loader_cls,
    layer_stats,
    label_to_block_id,
    transfer_to_compute_map,
    recorder,
    *,
    slow_transfer=False,
    offload_timing_collector=None,
):
    """Create a loader with fully mocked CUDA environment.

    Args:
        slow_transfer: If True, transfer stream events are created as incomplete,
            simulating slow PCIe transfers that haven't finished by the time
            the compute stream needs the data.
        offload_timing_collector: Optional collector (e.g. a spy); default no-op.
    """
    compute_stream = recorder.make_stream("compute", events_completed=True)
    transfer_stream = recorder.make_stream("transfer", events_completed=not slow_transfer)
    controller = recorder.make_controller()

    mock_device = torch.device("cuda")

    with (
        patch("torch.cuda.Stream", return_value=transfer_stream),
        patch("torch.cuda.current_stream", return_value=compute_stream),
        patch("torch.cuda.stream", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
    ):
        loader = loader_cls(
            layer_stats=layer_stats,
            device_gpu=mock_device,
            label_to_block_id=label_to_block_id,
            transfer_to_compute_map=transfer_to_compute_map,
            allocation_controller=controller,
            offload_timing_collector=offload_timing_collector,
        )
        # Clear the log from __init__/preload so tests start clean
        recorder.log.clear()
        recorder._event_counter = 0

        yield loader


def _run_inference(loader, labels: list[str], recorder: SyncRecorder) -> None:
    """Run one full enter/forward/exit cycle for all layers."""
    for label in labels:
        loader.enter(label)
        recorder.forward(label)
        loader.exit(label)


def _operation_trace(recorder: SyncRecorder) -> list[tuple[str, str]]:
    return [(entry.op, entry.stream) for entry in recorder.log]


@pytest.mark.parametrize(
    "loader_cls",
    [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
)
def test_preallocated_loader_delegates_to_core_runtime(loader_cls):
    labels = ["embed", "L0"]
    layer_stats = _make_layer_stats(labels)
    transfer_to_compute_map = {"embed": "L0"}
    label_to_block_id = {"embed": 0}
    recorder = SyncRecorder()

    with _create_loader(
        loader_cls,
        layer_stats,
        label_to_block_id,
        transfer_to_compute_map,
        recorder,
    ) as loader:
        assert isinstance(loader._transfer_runtime, CudaTransferRuntime)
        assert loader.transfer_stream is loader._transfer_runtime.transfer_stream


@pytest.mark.parametrize(
    "loader_cls",
    [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
)
def test_preallocated_loader_labels_transfer_ticket_with_schedule_and_consumer(loader_cls):
    labels = ["embed", "L0"]
    layer_stats = _make_layer_stats(labels)
    transfer_to_compute_map = {"embed": "L0"}
    label_to_block_id = {"embed": 0}
    recorder = SyncRecorder()

    with _create_loader(
        loader_cls,
        layer_stats,
        label_to_block_id,
        transfer_to_compute_map,
        recorder,
    ) as loader:
        loader.enter("embed")
        scheduled = next(iter(loader.scheduled_transfers.values()))
        ticket = scheduled[0] if isinstance(scheduled, list) else scheduled

        assert ticket.trace_label == "schedule=embed consume=L0"


@pytest.mark.parametrize(
    ("loader_cls", "expected"),
    [
        (
            PreallocatedBatchTransferTensorLoader,
            [
                ("record_event", "compute"),
                ("wait_event", "transfer"),
                ("schedule_transfer", "transfer"),
                ("record_event", "transfer"),
                ("forward", "compute"),
                ("wait_event", "compute"),
                ("record_event", "compute"),
                ("schedule_transfer", "transfer"),
                ("record_event", "transfer"),
                ("forward", "compute"),
                ("wait_event", "compute"),
                ("record_event", "compute"),
                ("wait_event", "transfer"),
                ("schedule_transfer", "transfer"),
                ("record_event", "transfer"),
                ("forward", "compute"),
                ("wait_event", "compute"),
                ("record_event", "compute"),
                ("forward", "compute"),
                ("record_event", "compute"),
                ("record_event", "transfer"),
                ("wait_event", "compute"),
            ],
        ),
        (
            PreallocatedBatchTransferTensorLoaderReordered,
            [
                ("record_event", "compute"),
                ("wait_event", "transfer"),
                ("schedule_transfer", "transfer"),
                ("record_event", "transfer"),
                ("forward", "compute"),
                ("record_event", "compute"),
                ("wait_event", "compute"),
                ("schedule_transfer", "transfer"),
                ("record_event", "transfer"),
                ("forward", "compute"),
                ("record_event", "compute"),
                ("wait_event", "compute"),
                ("wait_event", "transfer"),
                ("schedule_transfer", "transfer"),
                ("record_event", "transfer"),
                ("forward", "compute"),
                ("record_event", "compute"),
                ("wait_event", "compute"),
                ("forward", "compute"),
                ("record_event", "compute"),
                ("record_event", "transfer"),
                ("wait_event", "compute"),
            ],
        ),
    ],
)
def test_preallocated_loader_operation_trace_is_frozen(loader_cls, expected):
    labels = ["embed", "L0", "L1", "L2"]
    layer_stats = _make_layer_stats(labels)
    transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L2"}
    label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
    recorder = SyncRecorder()

    with _create_loader(
        loader_cls,
        layer_stats,
        label_to_block_id,
        transfer_to_compute_map,
        recorder,
    ) as loader:
        _run_inference(loader, labels, recorder)

    assert _operation_trace(recorder) == expected


@pytest.mark.parametrize(
    "loader_cls",
    [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
)
def test_sync_prev_onload_waits_compute_stream_on_transfer_stream(loader_cls):
    labels = ["embed", "L0"]
    layer_stats = _make_layer_stats(labels)
    transfer_to_compute_map, label_to_block_id = _build_pipeline_maps(labels, set())
    recorder = SyncRecorder()

    with _create_loader(
        loader_cls,
        layer_stats,
        label_to_block_id,
        transfer_to_compute_map,
        recorder,
    ) as loader:
        loader.sync_prev_onload()

    assert len(recorder.log) == 1
    assert recorder.log[0].op == "wait_stream"
    assert recorder.log[0].stream == "compute"


def test_reordered_preload_does_not_add_redundant_compute_wait():
    labels = ["embed", "L0", "L1"]
    layer_stats = _make_layer_stats(labels)
    transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "embed"}
    label_to_block_id = {"embed": 0, "L0": 1, "L1": 2}
    recorder = SyncRecorder()

    with _create_loader(
        PreallocatedBatchTransferTensorLoaderReordered,
        layer_stats,
        label_to_block_id,
        transfer_to_compute_map,
        recorder,
    ) as loader:
        loader.sync_prev_onload()
        recorder.log.clear()
        loader.enter("embed")

    assert recorder.find_entries("wait_event", stream="compute") == []


@pytest.mark.parametrize(
    "loader_cls",
    [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
)
def test_one_compute_label_releases_every_consumed_slot(loader_cls):
    labels = ["A", "B", "C", "tail"]
    layer_stats = _make_layer_stats(labels)
    transfer_to_compute_map = {"A": "C", "B": "C"}
    label_to_block_id = {"A": 0, "B": 1}
    recorder = SyncRecorder()

    with _create_loader(
        loader_cls,
        layer_stats,
        label_to_block_id,
        transfer_to_compute_map,
        recorder,
    ) as loader:
        for label in ["A", "B", "C"]:
            loader.enter(label)
            loader.exit(label)

        loader._transfer_runtime.submit(0, lambda: None)
        loader._transfer_runtime.submit(1, lambda: None)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestDefaultLoaderSimplePipeline:
    """Default loader: verify sync ordering in a simple pipeline."""

    def _setup(self):
        labels = ["embed", "L0", "L1", "L2"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L2"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        return labels, layer_stats, transfer_to_compute_map, label_to_block_id

    def test_sync_before_read(self):
        """Transfer wait_event must happen before the consuming layer's forward."""
        labels, layer_stats, t2c, l2b = self._setup()
        recorder = SyncRecorder()

        with _create_loader(PreallocatedBatchTransferTensorLoader, layer_stats, l2b, t2c, recorder) as loader:
            _run_inference(loader, labels, recorder)

        # Default loader: exit("embed") waits for scheduled_transfers["embed"]
        # which is the transfer that loaded L0's data.  This must be before L0's forward.
        for compute_label in ["L0", "L1", "L2"]:
            forward_idx = recorder.find_first("forward", target=compute_label)
            # Find the wait_event on compute stream that syncs the transfer for this layer.
            # In the default loader, the wait happens in exit() of the PREVIOUS layer,
            # which is before the current layer's forward.
            wait_indices = recorder.find_entries("wait_event", stream="compute")
            relevant_waits = [i for i in wait_indices if i < forward_idx]
            assert relevant_waits, (
                f"No wait_event on compute stream before forward({compute_label})\nLog:\n{recorder.dump()}"
            )


class TestReorderedLoaderSimplePipeline:
    """Reordered loader: verify sync ordering in a simple pipeline."""

    def _setup(self):
        labels = ["embed", "L0", "L1", "L2"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L2"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        return labels, layer_stats, transfer_to_compute_map, label_to_block_id

    def test_sync_before_read(self):
        """Transfer wait_event must happen in enter() before the layer's forward."""
        labels, layer_stats, t2c, l2b = self._setup()
        recorder = SyncRecorder()

        with _create_loader(PreallocatedBatchTransferTensorLoaderReordered, layer_stats, l2b, t2c, recorder) as loader:
            _run_inference(loader, labels, recorder)

        # Reordered loader: enter("L0") waits for scheduled_transfers["L0"]
        # (set by enter("embed")), which is before L0's forward.
        for compute_label in ["L0", "L1", "L2"]:
            forward_idx = recorder.find_first("forward", target=compute_label)
            wait_indices = recorder.find_entries("wait_event", stream="compute")
            relevant_waits = [i for i in wait_indices if i < forward_idx]
            assert relevant_waits, (
                f"No wait_event on compute stream before forward({compute_label})\nLog:\n{recorder.dump()}"
            )

    def test_sync_not_after_read(self):
        """Verify the wait is NOT only after the forward (the old bug)."""
        labels, layer_stats, t2c, l2b = self._setup()
        recorder = SyncRecorder()

        with _create_loader(PreallocatedBatchTransferTensorLoaderReordered, layer_stats, l2b, t2c, recorder) as loader:
            _run_inference(loader, labels, recorder)

        # For each compute target, the wait_event on compute stream must appear
        # BEFORE forward, not only after.
        for compute_label in ["L0", "L1", "L2"]:
            forward_idx = recorder.find_first("forward", target=compute_label)
            wait_indices = recorder.find_entries("wait_event", stream="compute")
            waits_before = [i for i in wait_indices if i < forward_idx]
            waits_after = [i for i in wait_indices if i > forward_idx]
            # There should be at least one wait BEFORE forward
            assert waits_before, (
                f"wait_event for {compute_label} only found after forward (old bug!)\n"
                f"  waits_after={waits_after}\n"
                f"Log:\n{recorder.dump()}"
            )


class TestReorderedLoaderGaps:
    """Reordered loader: verify sync with gap layers."""

    def test_single_gap(self):
        """Single gap layer (L2) between L1 and L3."""
        labels = ["embed", "L0", "L1", "L2", "L3"]
        gap_labels = {"L2"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        # L2 is a gap, so transfer skips from L1 to L3
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L3"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
        ) as loader:
            _run_inference(loader, labels, recorder)

        # L3's data was scheduled by enter("L1"), and the wait must happen
        # in enter("L3"), before L3's forward.
        forward_l3 = recorder.find_first("forward", target="L3")
        wait_indices = recorder.find_entries("wait_event", stream="compute")
        waits_before_l3 = [i for i in wait_indices if i < forward_l3]
        assert waits_before_l3, f"No wait_event before forward(L3) with gap at L2\nLog:\n{recorder.dump()}"

    def test_multiple_consecutive_gaps(self):
        """Multiple consecutive gap layers (L2, L3) between L1 and L4."""
        labels = ["embed", "L0", "L1", "L2", "L3", "L4"]
        gap_labels = {"L2", "L3"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L4"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
        ) as loader:
            _run_inference(loader, labels, recorder)

        forward_l4 = recorder.find_first("forward", target="L4")
        wait_indices = recorder.find_entries("wait_event", stream="compute")
        waits_before_l4 = [i for i in wait_indices if i < forward_l4]
        assert waits_before_l4, f"No wait_event before forward(L4) with gaps at L2, L3\nLog:\n{recorder.dump()}"

    def test_leading_gaps(self):
        """Leading gap layers (L0, L1, L2) with first real layer at L3."""
        labels = ["embed", "L0", "L1", "L2", "L3"]
        gap_labels = {"L0", "L1", "L2"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map = {"embed": "L3"}
        label_to_block_id = {"embed": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
        ) as loader:
            _run_inference(loader, labels, recorder)

        forward_l3 = recorder.find_first("forward", target="L3")
        wait_indices = recorder.find_entries("wait_event", stream="compute")
        waits_before_l3 = [i for i in wait_indices if i < forward_l3]
        assert waits_before_l3, f"No wait_event before forward(L3) with leading gaps\nLog:\n{recorder.dump()}"

    def test_gap_layer_no_spurious_sync(self):
        """Gap layers should not trigger data transfer waits."""
        labels = ["embed", "L0", "L1", "L2", "L3"]
        gap_labels = {"L2"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L3"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
        ) as loader:
            _run_inference(loader, labels, recorder)

        # L2 is not a compute target, so no data wait should occur in enter("L2")
        forward_l2 = recorder.find_first("forward", target="L2")
        # Find the enter("L2") region: between L1's exit and L2's forward
        forward_l1 = recorder.find_first("forward", target="L1")
        # Any wait_event on compute stream between L1's forward region and L2's forward
        # should NOT be a data wait for L2 (L2 is a gap, no data to wait for)
        enter_l2_waits = [
            i for i in recorder.find_entries("wait_event", stream="compute") if forward_l1 < i < forward_l2
        ]
        # The only compute-stream wait_event in this region could be from exit("L1")
        # for the default-style sync. For the reordered loader, there should be none
        # since L2 has no scheduled_transfers entry.
        # (This is a structural check - gap layers don't appear in scheduled_transfers)
        assert "L2" not in {recorder.log[i].target for i in enter_l2_waits if i < forward_l2}, (
            f"Spurious data wait for gap layer L2\nLog:\n{recorder.dump()}"
        )

    @pytest.mark.parametrize(
        "gap_labels",
        [
            pytest.param({"L2", "L5", "L6"}, id="single+double_gap"),
            pytest.param({"L2", "L3"}, id="consecutive_double"),
            pytest.param({"L3", "L4", "L5"}, id="consecutive_triple"),
        ],
    )
    def test_large_model_mixed_gaps(self, gap_labels):
        """10 layers with various gap patterns."""
        labels = ["embed", "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map, label_to_block_id = _build_pipeline_maps(labels, gap_labels)
        offloaded = [lbl for lbl in labels if lbl not in gap_labels]
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
        ) as loader:
            _run_inference(loader, labels, recorder)

        # Verify sync before every offloaded compute layer (except embed)
        for compute_label in offloaded[1:]:
            fwd_idx = recorder.find_first("forward", target=compute_label)
            waits_before = [i for i in recorder.find_entries("wait_event", stream="compute") if i < fwd_idx]
            assert waits_before, f"No wait_event before forward({compute_label})\nLog:\n{recorder.dump()}"

        # Verify no schedule_transfer for gap layers
        for gap_label in gap_labels:
            gap_schedules = recorder.find_entries("schedule_transfer", target=gap_label)
            assert not gap_schedules, f"Unexpected schedule_transfer for gap layer {gap_label}\nLog:\n{recorder.dump()}"


class TestDefaultLoaderGaps:
    """Default loader: verify correct behavior with gap layers."""

    def test_single_gap(self):
        """Single gap layer (L2) between L1 and L3."""
        labels = ["embed", "L0", "L1", "L2", "L3"]
        gap_labels = {"L2"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L3"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
        ) as loader:
            _run_inference(loader, labels, recorder)

        forward_l3 = recorder.find_first("forward", target="L3")
        assert forward_l3 is not None, f"No forward(L3)\nLog:\n{recorder.dump()}"

    def test_multiple_consecutive_gaps(self):
        """Multiple consecutive gap layers (L2, L3) between L1 and L4."""
        labels = ["embed", "L0", "L1", "L2", "L3", "L4"]
        gap_labels = {"L2", "L3"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L4"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
        ) as loader:
            _run_inference(loader, labels, recorder)

        forward_l4 = recorder.find_first("forward", target="L4")
        assert forward_l4 is not None, f"No forward(L4)\nLog:\n{recorder.dump()}"

    def test_leading_gaps(self):
        """Leading gap layers (L0, L1, L2) with first real layer at L3."""
        labels = ["embed", "L0", "L1", "L2", "L3"]
        gap_labels = {"L0", "L1", "L2"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map = {"embed": "L3"}
        label_to_block_id = {"embed": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
        ) as loader:
            _run_inference(loader, labels, recorder)

        forward_l3 = recorder.find_first("forward", target="L3")
        assert forward_l3 is not None, f"No forward(L3)\nLog:\n{recorder.dump()}"

    def test_gap_layer_no_spurious_sync(self):
        """Gap layers should not trigger schedule_transfer."""
        labels = ["embed", "L0", "L1", "L2", "L3"]
        gap_labels = {"L2"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L3"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
        ) as loader:
            _run_inference(loader, labels, recorder)

        forward_l1 = recorder.find_first("forward", target="L1")
        forward_l2 = recorder.find_first("forward", target="L2")
        gap_region_schedules = [i for i in recorder.find_entries("schedule_transfer") if forward_l1 < i < forward_l2]
        assert not gap_region_schedules, f"Spurious schedule_transfer in gap region (L1->L2)\nLog:\n{recorder.dump()}"

    def test_gap_compute_event_recorded(self):
        """Gap layers must still record compute events for block reuse safety."""
        labels = ["embed", "L0", "L1", "L2", "L3"]
        gap_labels = {"L2"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L3"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
        ) as loader:
            _run_inference(loader, labels, recorder)

        forward_l2 = recorder.find_first("forward", target="L2")
        forward_l3 = recorder.find_first("forward", target="L3")
        record_events_between = [
            i for i in recorder.find_entries("record_event", stream="compute") if forward_l2 < i < forward_l3
        ]
        assert record_events_between, f"No compute record_event after gap layer L2's forward\nLog:\n{recorder.dump()}"

    @pytest.mark.parametrize(
        "gap_labels",
        [
            pytest.param({"L2", "L5", "L6"}, id="single+double_gap"),
            pytest.param({"L2", "L3"}, id="consecutive_double"),
            pytest.param({"L3", "L4", "L5"}, id="consecutive_triple"),
        ],
    )
    def test_large_model_mixed_gaps(self, gap_labels):
        """10 layers with various gap patterns."""
        labels = ["embed", "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map, label_to_block_id = _build_pipeline_maps(labels, gap_labels)
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
        ) as loader:
            _run_inference(loader, labels, recorder)

        # Verify no schedule_transfer for gap layers
        for gap_label in gap_labels:
            gap_schedules = recorder.find_entries("schedule_transfer", target=gap_label)
            assert not gap_schedules, f"Unexpected schedule_transfer for gap layer {gap_label}\nLog:\n{recorder.dump()}"

        # Verify compute events are recorded for all layers (including gaps)
        all_record_events = recorder.find_entries("record_event", stream="compute")
        assert len(all_record_events) >= len(labels), (
            f"Expected at least {len(labels)} compute record_events, got {len(all_record_events)}\n"
            f"Log:\n{recorder.dump()}"
        )


class TestBlockReuseSafety:
    """Verify transfer stream waits for compute before overwriting a block."""

    def test_default_loader_block_reuse(self):
        """Default loader: transfer stream waits for compute before reusing block."""
        labels = ["embed", "L0", "L1", "L2"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L2"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}  # L1 reuses block 0
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
        ) as loader:
            _run_inference(loader, labels, recorder)

        # When enter("L1") wants to reuse block 0 (previously used by "embed"),
        # the transfer stream must wait for the runtime's last-use event for
        # block 0, recorded after L0 consumes embed's transfer.
        # This ensures L0's forward (which reads from block 0) has completed
        # before the transfer stream overwrites block 0 with L2's data.
        transfer_waits = recorder.find_entries("wait_event", stream="transfer")
        schedule_l1 = recorder.find_entries("schedule_transfer", target="L1")
        assert schedule_l1, f"No schedule_transfer for L1\nLog:\n{recorder.dump()}"

        # There must be a transfer-stream wait_event before schedule_transfer("L1")
        waits_before_l1_transfer = [i for i in transfer_waits if i < schedule_l1[0]]
        assert waits_before_l1_transfer, (
            f"Transfer stream did not wait before overwriting block 0 (schedule_transfer L1)\nLog:\n{recorder.dump()}"
        )

    def test_reordered_loader_block_reuse(self):
        """Reordered loader: transfer stream waits for compute before reusing block."""
        labels = ["embed", "L0", "L1", "L2"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L2"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
        ) as loader:
            _run_inference(loader, labels, recorder)

        transfer_waits = recorder.find_entries("wait_event", stream="transfer")
        schedule_l1 = recorder.find_entries("schedule_transfer", target="L1")
        assert schedule_l1, f"No schedule_transfer for L1\nLog:\n{recorder.dump()}"

        waits_before_l1_transfer = [i for i in transfer_waits if i < schedule_l1[0]]
        assert waits_before_l1_transfer, (
            f"Transfer stream did not wait before overwriting block 0 (schedule_transfer L1)\nLog:\n{recorder.dump()}"
        )


class TestFastTransfer:
    """Fast transfer: transfer completes before compute needs data."""

    def _setup_with_fast_events(self, loader_cls):
        labels = ["embed", "L0", "L1", "L2"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L2"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(loader_cls, layer_stats, label_to_block_id, transfer_to_compute_map, recorder) as loader:
            # For fast transfers, after each enter() we mark all pending events complete
            for label in labels:
                loader.enter(label)
                # Mark any transfer events as completed (simulates fast PCIe)
                for entry in recorder.log:
                    if entry.op == "record_event" and entry.stream == "transfer":
                        # Find the corresponding MockEvent and mark complete
                        pass
                recorder.forward(label)
                loader.exit(label)

        return recorder

    def test_default_loader_no_stall(self):
        """Default loader: fast transfers should not cause stalls."""
        recorder = self._setup_with_fast_events(PreallocatedBatchTransferTensorLoader)

        stall_entries = [e for e in recorder.log if e.op == "wait_event" and e.stalled]
        # With fast transfers (events created as completed=True by record_event),
        # wait_event should never stall.
        assert not stall_entries, (
            "Unexpected stalls with fast transfers:\n"
            + "\n".join(f"  {e}" for e in stall_entries)
            + f"\nLog:\n{recorder.dump()}"
        )

    def test_reordered_loader_no_stall(self):
        """Reordered loader: fast transfers should not cause stalls."""
        recorder = self._setup_with_fast_events(PreallocatedBatchTransferTensorLoaderReordered)

        stall_entries = [e for e in recorder.log if e.op == "wait_event" and e.stalled]
        assert not stall_entries, (
            "Unexpected stalls with fast transfers:\n"
            + "\n".join(f"  {e}" for e in stall_entries)
            + f"\nLog:\n{recorder.dump()}"
        )

    def test_default_loader_sync_still_present(self):
        """Even with fast transfers, sync points must exist before reads."""
        recorder = self._setup_with_fast_events(PreallocatedBatchTransferTensorLoader)

        for compute_label in ["L0", "L1", "L2"]:
            forward_idx = recorder.find_first("forward", target=compute_label)
            wait_indices = recorder.find_entries("wait_event", stream="compute")
            relevant_waits = [i for i in wait_indices if i < forward_idx]
            assert relevant_waits, (
                f"No sync before forward({compute_label}) even with fast transfers\nLog:\n{recorder.dump()}"
            )

    def test_reordered_loader_sync_still_present(self):
        """Even with fast transfers, sync points must exist before reads."""
        recorder = self._setup_with_fast_events(PreallocatedBatchTransferTensorLoaderReordered)

        for compute_label in ["L0", "L1", "L2"]:
            forward_idx = recorder.find_first("forward", target=compute_label)
            wait_indices = recorder.find_entries("wait_event", stream="compute")
            relevant_waits = [i for i in wait_indices if i < forward_idx]
            assert relevant_waits, (
                f"No sync before forward({compute_label}) even with fast transfers\nLog:\n{recorder.dump()}"
            )


class TestSlowTransfer:
    """Slow transfer: transfer NOT complete when compute needs data.

    This is the critical test that catches the original reordered loader bug
    where wait_event was in exit() instead of enter().
    """

    def test_default_loader_stall_before_read(self):
        """Default loader: stall happens in exit(L-1), before L's forward."""
        labels = ["embed", "L0", "L1", "L2"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L2"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
            slow_transfer=True,
        ) as loader:
            _run_inference(loader, labels, recorder)

        for compute_label in ["L0", "L1", "L2"]:
            forward_idx = recorder.find_first("forward", target=compute_label)
            stall_waits = [
                i
                for i, e in enumerate(recorder.log)
                if e.op == "wait_event" and e.stream == "compute" and e.stalled and i < forward_idx
            ]
            assert stall_waits, (
                f"Default loader: no stalling wait before forward({compute_label})\nLog:\n{recorder.dump()}"
            )

    def test_reordered_loader_stall_before_read(self):
        """Reordered loader: stall happens in enter(L), before L's forward."""
        labels = ["embed", "L0", "L1", "L2"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L2"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
            slow_transfer=True,
        ) as loader:
            _run_inference(loader, labels, recorder)

        for compute_label in ["L0", "L1", "L2"]:
            forward_idx = recorder.find_first("forward", target=compute_label)
            stall_waits = [
                i
                for i, e in enumerate(recorder.log)
                if e.op == "wait_event" and e.stream == "compute" and e.stalled and i < forward_idx
            ]
            assert stall_waits, (
                f"Reordered loader: no stalling wait before forward({compute_label})\n"
                f"This is the exact bug where wait was in exit() instead of enter()!\n"
                f"Log:\n{recorder.dump()}"
            )

    def test_reordered_loader_stall_not_only_after(self):
        """Verify the stall does NOT happen only after the forward (old bug pattern)."""
        labels = ["embed", "L0", "L1", "L2"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L2"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
            slow_transfer=True,
        ) as loader:
            _run_inference(loader, labels, recorder)

        for compute_label in ["L0", "L1", "L2"]:
            forward_idx = recorder.find_first("forward", target=compute_label)
            stalls_before = [
                i
                for i, e in enumerate(recorder.log)
                if e.op == "wait_event" and e.stream == "compute" and e.stalled and i < forward_idx
            ]
            stalls_after = [
                i
                for i, e in enumerate(recorder.log)
                if e.op == "wait_event" and e.stream == "compute" and e.stalled and i > forward_idx
            ]
            assert stalls_before, (
                f"Reordered loader: stall for {compute_label} found only AFTER forward "
                f"(stalls_after={stalls_after}). This means data was read before sync!\n"
                f"Log:\n{recorder.dump()}"
            )

    def test_slow_transfer_with_gaps(self):
        """Slow transfer with gap layers: stall must be before the gap-target's forward."""
        labels = ["embed", "L0", "L1", "L2", "L3"]
        gap_labels = {"L2"}
        layer_stats = _make_gap_layer_stats(labels, gap_labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "L3"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 0}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
            slow_transfer=True,
        ) as loader:
            _run_inference(loader, labels, recorder)

        forward_l3 = recorder.find_first("forward", target="L3")
        stalls_before_l3 = [
            i
            for i, e in enumerate(recorder.log)
            if e.op == "wait_event" and e.stream == "compute" and e.stalled and i < forward_l3
        ]
        assert stalls_before_l3, (
            f"Slow transfer with gaps: no stalling wait before forward(L3)\nLog:\n{recorder.dump()}"
        )


class TestCUDAGraphForkJoin:
    """CUDA graph fork/join: verify fork at start, join at end."""

    def test_default_loader_fork_at_first_layer(self):
        """Default loader: fork event at first layer's enter()."""
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
        ) as loader:
            _run_inference(loader, labels, recorder)

        # Fork: compute records event, transfer waits for it
        # This should be the first operations in the log
        assert recorder.log[0].op == "record_event" and recorder.log[0].stream == "compute", (
            f"First op should be compute record_event (fork), got {recorder.log[0]}\nLog:\n{recorder.dump()}"
        )
        assert recorder.log[1].op == "wait_event" and recorder.log[1].stream == "transfer", (
            f"Second op should be transfer wait_event (fork), got {recorder.log[1]}\nLog:\n{recorder.dump()}"
        )

    def test_default_loader_join_at_last_layer(self):
        """Default loader: join event at last layer's exit()."""
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoader, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
        ) as loader:
            _run_inference(loader, labels, recorder)

        # Join: transfer records event, compute waits for it (last 2 entries)
        last_two = recorder.log[-2:]
        assert last_two[0].op == "record_event" and last_two[0].stream == "transfer", (
            f"Second-to-last op should be transfer record_event (join), got {last_two[0]}\nLog:\n{recorder.dump()}"
        )
        assert last_two[1].op == "wait_event" and last_two[1].stream == "compute", (
            f"Last op should be compute wait_event (join), got {last_two[1]}\nLog:\n{recorder.dump()}"
        )

    def test_reordered_loader_fork_at_first_layer(self):
        """Reordered loader: fork event at first layer's enter()."""
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
        ) as loader:
            _run_inference(loader, labels, recorder)

        assert recorder.log[0].op == "record_event" and recorder.log[0].stream == "compute", (
            f"First op should be compute record_event (fork), got {recorder.log[0]}\nLog:\n{recorder.dump()}"
        )
        assert recorder.log[1].op == "wait_event" and recorder.log[1].stream == "transfer", (
            f"Second op should be transfer wait_event (fork), got {recorder.log[1]}\nLog:\n{recorder.dump()}"
        )

    def test_reordered_loader_join_at_last_layer(self):
        """Reordered loader: join event at last layer's exit()."""
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
        ) as loader:
            _run_inference(loader, labels, recorder)

        last_two = recorder.log[-2:]
        assert last_two[0].op == "record_event" and last_two[0].stream == "transfer", (
            f"Second-to-last op should be transfer record_event (join), got {last_two[0]}\nLog:\n{recorder.dump()}"
        )
        assert last_two[1].op == "wait_event" and last_two[1].stream == "compute", (
            f"Last op should be compute wait_event (join), got {last_two[1]}\nLog:\n{recorder.dump()}"
        )

    @staticmethod
    def _count_session_forks(recorder: SyncRecorder) -> int:
        """Count compute→transfer fork pairs (session / piece boundary forks)."""
        count = 0
        for i in range(len(recorder.log) - 1):
            e = recorder.log[i]
            e_next = recorder.log[i + 1]
            if (
                e.op == "record_event"
                and e.stream == "compute"
                and e_next.op == "wait_event"
                and e_next.stream == "transfer"
            ):
                count += 1
        return count

    @pytest.mark.parametrize(
        "loader_cls",
        [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
    )
    def test_single_session_fork_per_forward_without_piece_join(self, loader_cls):
        """FULL / eager: fork once at the first trap, not on every enter()."""
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(loader_cls, layer_stats, label_to_block_id, transfer_to_compute_map, recorder) as loader:
            _run_inference(loader, labels, recorder)

        assert self._count_session_forks(recorder) == 1, (
            f"{loader_cls.__name__}: expected one session fork per forward; log:\n{recorder.dump()}"
        )

    @pytest.mark.parametrize(
        "loader_cls",
        [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
    )
    def test_piece_join_allows_another_session_fork(self, loader_cls):
        """After join_after_forward, the next enter must fork again (PIECEWISE)."""
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(loader_cls, layer_stats, label_to_block_id, transfer_to_compute_map, recorder) as loader:
            loader.enter("embed")
            loader.exit("embed")
            loader.join_after_forward()
            forks_after_piece = self._count_session_forks(recorder)
            loader.enter("L0")
            forks_after_next_enter = self._count_session_forks(recorder)

        assert forks_after_piece == 1
        assert forks_after_next_enter == 2, (
            f"{loader_cls.__name__}: expected a second fork after piece join; log:\n{recorder.dump()}"
        )

    @pytest.mark.parametrize(
        "loader_cls",
        [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
    )
    def test_post_only_last_layer_exit_skips_cross_piece_transfer_join(self, loader_cls):
        labels = ["embed", "L0"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0"}
        label_to_block_id = {"embed": 0}
        recorder = SyncRecorder()

        with _create_loader(loader_cls, layer_stats, label_to_block_id, transfer_to_compute_map, recorder) as loader:
            loader.enter("L0")
            loader.join_after_forward()
            recorder.log.clear()

            policy = loader.piecewise_prefetch_policy
            policy.enabled = True
            policy.strict = True
            policy.on_wait("L0")
            policy.on_schedule("L0", "L0")

            loader.exit("L0")

            assert len(recorder.find_entries("record_event", stream="compute")) == 1, recorder.dump()
            assert recorder.find_entries("record_event", stream="transfer") == [], recorder.dump()
            assert recorder.find_entries("wait_event", stream="compute") == [], recorder.dump()
            assert policy.on_piece_join() == []
            assert loader.scheduled_transfers == {}

    def test_cross_iteration_fork_after_join(self):
        """Second iteration's fork must come after first iteration's join."""
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}

        for loader_cls in [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered]:
            recorder = SyncRecorder()
            with _create_loader(
                loader_cls, layer_stats, label_to_block_id, transfer_to_compute_map, recorder
            ) as loader:
                _run_inference(loader, labels, recorder)
                _run_inference(loader, labels, recorder)

            forwards = recorder.find_entries("forward")
            iter1_last_forward = forwards[len(labels) - 1]
            iter2_first_forward = forwards[len(labels)]

            # Join: transfer record_event followed by compute wait_event (after iter1)
            join_idx = None
            for i in range(iter1_last_forward, len(recorder.log) - 1):
                e = recorder.log[i]
                e_next = recorder.log[i + 1]
                if (
                    e.op == "record_event"
                    and e.stream == "transfer"
                    and e_next.op == "wait_event"
                    and e_next.stream == "compute"
                ):
                    join_idx = i
                    break

            # Fork: compute record_event followed by transfer wait_event (before iter2)
            fork_idx = None
            for i in range(iter1_last_forward, iter2_first_forward):
                e = recorder.log[i]
                if i + 1 < len(recorder.log):
                    e_next = recorder.log[i + 1]
                    if (
                        e.op == "record_event"
                        and e.stream == "compute"
                        and e_next.op == "wait_event"
                        and e_next.stream == "transfer"
                    ):
                        fork_idx = i
                        break

            assert join_idx is not None, (
                f"{loader_cls.__name__}: no join pattern after iteration 1\nLog:\n{recorder.dump()}"
            )
            assert fork_idx is not None, (
                f"{loader_cls.__name__}: no fork pattern before iteration 2\nLog:\n{recorder.dump()}"
            )
            assert_before(
                recorder, join_idx, fork_idx, f"{loader_cls.__name__}: join must come before fork across iterations"
            )


class TestEventMapClearing:
    """Cross-iteration event-map state hygiene.

    Pins last-layer exit() and first-layer enter() clears that prevent stale
    events from a prior iteration from leaking into the next iteration's
    waits, which under torch.compile / CUDA-graph capture surfaces as
    cudaErrorStreamCaptureIsolation.
    """

    @pytest.mark.parametrize(
        "loader_cls",
        [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
    )
    def test_last_layer_exit_clears_event_maps(self, loader_cls):
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(loader_cls, layer_stats, label_to_block_id, transfer_to_compute_map, recorder) as loader:
            _run_inference(loader, labels, recorder)
            assert loader._transfer_runtime._last_use_by_slot == {}
            assert loader.scheduled_transfers == {}, (
                f"scheduled_transfers should be empty after last-layer join; got {loader.scheduled_transfers}"
            )

    @pytest.mark.parametrize(
        "loader_cls",
        [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
    )
    def test_first_layer_enter_silent_when_maps_already_empty(self, loader_cls, caplog):
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(loader_cls, layer_stats, label_to_block_id, transfer_to_compute_map, recorder) as loader:
            assert not loader._transfer_runtime.is_active

            with caplog.at_level("DEBUG", logger="flextensor.loaders"):
                loader.enter("embed")

            assert all("stale" not in r.getMessage().lower() for r in caplog.records), (
                "no safety-net log should fire when maps are already empty"
            )


class TestScheduledTransferJoinThenClear:
    """Join-then-clear for ``scheduled_transfers`` (abort + multi-request)."""

    @pytest.mark.parametrize(
        "loader_cls",
        [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
    )
    def test_abort_recovery_joins_before_clearing_stale_scheduled(self, loader_cls, caplog):
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(loader_cls, layer_stats, label_to_block_id, transfer_to_compute_map, recorder) as loader:
            loader.enter("embed")
            stale = next(iter(loader.scheduled_transfers.values()))
            recorder.log.clear()

            with caplog.at_level("DEBUG", logger="flextensor.loaders"):
                loader.enter("embed")

            assert stale not in loader.scheduled_transfers.values()
            assert any("abort recovery" in r.getMessage().lower() for r in caplog.records)
            assert any(e.op == "record_event" and e.stream == "transfer" for e in recorder.log)
            assert any(e.op == "wait_event" and e.stream == "compute" for e in recorder.log)

    def test_reordered_preload_is_counted_as_consumed_by_piecewise_policy(self):
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1", "L1": "embed"}
        label_to_block_id = {"embed": 0, "L0": 1, "L1": 2}
        recorder = SyncRecorder()

        with _create_loader(
            PreallocatedBatchTransferTensorLoaderReordered,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
        ) as loader:
            loader.piecewise_prefetch_policy.enabled = True
            loader.piecewise_prefetch_policy.strict = True
            _run_inference(loader, labels, recorder)
            _run_inference(loader, labels, recorder)

    @pytest.mark.parametrize(
        "loader_cls",
        [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
    )
    @pytest.mark.parametrize("slow_transfer", [False, True])
    def test_multi_request_second_forward_does_not_wait_on_first_scheduled(self, loader_cls, slow_transfer: bool):
        """Two requests; request 2 must not wait request-1 transfer events."""
        labels = ["embed", "L0", "L1"]
        layer_stats = _make_layer_stats(labels)
        transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
        label_to_block_id = {"embed": 0, "L0": 1}
        recorder = SyncRecorder()

        with _create_loader(
            loader_cls,
            layer_stats,
            label_to_block_id,
            transfer_to_compute_map,
            recorder,
            slow_transfer=slow_transfer,
        ) as loader:
            _run_inference(loader, labels, recorder)
            assert loader.scheduled_transfers == {}
            req1_transfer_events = {e.target for e in recorder.log if e.op == "record_event" and e.stream == "transfer"}
            split = len(recorder.log)

            _run_inference(loader, labels, recorder)
            assert loader.scheduled_transfers == {}

            req2_waits = [
                e
                for e in recorder.log[split:]
                if e.op == "wait_event" and e.stream == "compute" and e.target in req1_transfer_events
            ]
            assert req2_waits == [], f"second request waited on first-request events: {req2_waits}\n{recorder.dump()}"
            if slow_transfer:
                assert any(
                    e.op == "wait_event" and e.stream == "compute" and e.stalled for e in recorder.log[:split]
                ), f"expected stalled wait under slow_transfer:\n{recorder.dump()}"


def _timing_hook_names(calls: list[tuple]) -> list[str]:
    return [c[0] for c in calls]


def _make_timing_hook_spy() -> tuple[OffloadTimingCollector, list[tuple]]:
    """Real collector (beartype) with hook methods replaced by call recorders."""
    collector = OffloadTimingCollector(trap_labels=[], enabled=True, log_every=0)
    calls: list[tuple] = []

    def _record(name: str):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))

        return _fn

    for name in (
        "on_pass_start",
        "record_transfer_start",
        "record_transfer_end",
        "record_compute_start",
        "record_compute_end",
        "record_wait_start",
        "record_wait_end",
    ):
        setattr(collector, name, _record(name))
    return collector, calls


def _run_with_timing_spy(loader_cls: type) -> tuple[list[tuple], object, set[str], list[str]]:
    labels = ["embed", "L0", "L1"]
    layer_stats = _make_layer_stats(labels)
    transfer_to_compute_map = {"embed": "L0", "L0": "L1"}
    label_to_block_id = {"embed": 0, "L0": 1}
    recorder = SyncRecorder()
    spy, calls = _make_timing_hook_spy()
    with _create_loader(
        loader_cls,
        layer_stats,
        label_to_block_id,
        transfer_to_compute_map,
        recorder,
        offload_timing_collector=spy,
    ) as loader:
        transfer_stream = loader.transfer_stream
        _run_inference(loader, labels, recorder)
    return calls, transfer_stream, set(label_to_block_id), labels


def _assert_matched_start_end(names: list[str], column: str) -> None:
    starts = [i for i, n in enumerate(names) if n == f"record_{column}_start"]
    ends = [i for i, n in enumerate(names) if n == f"record_{column}_end"]
    assert len(starts) == len(ends) > 0, f"{column}: starts={starts} ends={ends} calls={names}"
    for s, e in zip(starts, ends, strict=True):
        assert s < e, f"{column} start@{s} must precede end@{e}: {names}"


class TestLoaderTimingHookWiring:
    """Loader enter/exit must drive OffloadTimingCollector hooks (not only sync)."""

    @pytest.mark.parametrize(
        "loader_cls",
        [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
    )
    def test_hooks_labels_stream_and_paired_starts_ends(self, loader_cls: type) -> None:
        calls, transfer_stream, offloaded, labels = _run_with_timing_spy(loader_cls)
        names = _timing_hook_names(calls)

        assert names[0] == "on_pass_start"
        assert names.count("on_pass_start") == 1
        for column in ("transfer", "compute", "wait"):
            _assert_matched_start_end(names, column)

        for name, args, _kwargs in calls:
            if name in ("record_transfer_start", "record_transfer_end"):
                label, stream = args
                assert label in offloaded
                assert stream is transfer_stream

        assert {args[0] for name, args, _ in calls if name == "record_compute_start"} == set(labels)
        assert {args[0] for name, args, _ in calls if name == "record_compute_end"} == set(labels)
        wait_labels = [args[0] for name, args, _ in calls if name == "record_wait_start"]
        assert wait_labels == [args[0] for name, args, _ in calls if name == "record_wait_end"]
        assert wait_labels

    def test_default_loader_wait_follows_compute_end(self) -> None:
        calls, _stream, _offloaded, _labels = _run_with_timing_spy(PreallocatedBatchTransferTensorLoader)
        names = _timing_hook_names(calls)
        wait_labels = [args[0] for name, args, _ in calls if name == "record_wait_start"]
        for label in wait_labels:
            pair_ok = False
            for i, (name, args, _) in enumerate(calls):
                if name != "record_compute_end" or args[0] != label:
                    continue
                pair_ok = any(
                    calls[j][0] == "record_wait_start" and calls[j][1][0] == label for j in range(i + 1, len(calls))
                )
                break
            assert pair_ok, f"default loader: wait({label}) must follow compute_end({label}): {names}"

    def test_reordered_loader_wait_precedes_compute_start(self) -> None:
        calls, _stream, _offloaded, _labels = _run_with_timing_spy(PreallocatedBatchTransferTensorLoaderReordered)
        names = _timing_hook_names(calls)
        wait_labels = [args[0] for name, args, _ in calls if name == "record_wait_start"]
        for label in wait_labels:
            w_i = next(i for i, (n, a, _) in enumerate(calls) if n == "record_wait_start" and a[0] == label)
            c_i = next(i for i, (n, a, _) in enumerate(calls) if n == "record_compute_start" and a[0] == label)
            assert w_i < c_i, f"reordered: wait({label}) before compute_start({label}): {names}"


class TestLoaderDynamoDisable:
    """Inference loader enter/exit stay outside compiled subgraphs when traps run.

    Discovery's :class:`~flextensor.loaders.WarmupDirectTensorLoader` is not covered:
    it only runs under the Dynamo-disabled discovery ``patched_forward``, and
    compiled inference drives :class:`~flextensor.loaders.PreallocatedLoader`
    through custom ops rather than this loader.
    """

    def test_concrete_loader_enter_exit_are_dynamo_disabled(self) -> None:
        import inspect

        from flextensor import loaders as loaders_mod
        from flextensor.loaders import Loader, WarmupDirectTensorLoader

        marker = "_torchdynamo_disable"
        concrete = [
            obj
            for obj in vars(loaders_mod).values()
            if inspect.isclass(obj)
            and issubclass(obj, Loader)
            and obj is not Loader
            and obj is not WarmupDirectTensorLoader
            and not inspect.isabstract(obj)
        ]
        assert concrete, "expected at least one concrete Loader subclass"

        for cls in concrete:
            for name in ("enter", "exit"):
                method = cls.__dict__.get(name)
                if method is None:
                    # Inherited from a base class; checked on the defining class.
                    continue
                assert getattr(method, marker, False), (
                    f"{cls.__name__}.{name} must be wrapped with compiler_utils.disable so Dynamo runs it eagerly"
                )

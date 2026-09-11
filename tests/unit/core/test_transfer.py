# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone CUDA transfer runtime."""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

import flextensor.core as core
from flextensor.core.transfer import CudaTransferRuntime, TransferStreamSynchronizationError


@pytest.mark.parametrize(
    "name",
    ["CudaTransferRuntime", "TransferStreamSynchronizationError", "TransferTicket"],
)
def test_core_package_does_not_reexport_transfer_api(name: str) -> None:
    assert not hasattr(core, name)


@dataclass(frozen=True)
class FakeEvent:
    name: str


@dataclass
class CudaRecorder:
    log: list[tuple[Any, ...]] = field(default_factory=list)
    next_event: int = 0

    def event(self, stream: str) -> FakeEvent:
        self.next_event += 1
        return FakeEvent(f"{stream}-{self.next_event}")


class FakeStream:
    def __init__(self, name: str, recorder: CudaRecorder) -> None:
        self.name = name
        self.recorder = recorder

    def record_event(self) -> FakeEvent:
        event = self.recorder.event(self.name)
        self.recorder.log.append(("record_event", self.name, event))
        return event

    def wait_event(self, event: FakeEvent) -> None:
        self.recorder.log.append(("wait_event", self.name, event))

    def wait_stream(self, stream: "FakeStream") -> None:
        self.recorder.log.append(("wait_stream", self.name, stream.name))

    def synchronize(self) -> None:
        self.recorder.log.append(("synchronize", self.name))


@pytest.fixture
def cuda_runtime(monkeypatch):
    recorder = CudaRecorder()
    compute = FakeStream("compute", recorder)
    transfer = FakeStream("transfer", recorder)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: transfer)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    runtime = CudaTransferRuntime(torch.device("cuda:0"))
    return runtime, recorder


def test_runtime_uses_native_stream_priority(monkeypatch):
    stream = object()
    stream_kwargs = {}

    def make_stream(**kwargs):
        stream_kwargs.update(kwargs)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    device = torch.device("cuda:0")

    runtime = CudaTransferRuntime(device)

    assert runtime.transfer_stream is stream
    assert stream_kwargs == {"device": device}


def test_submit_wait_finish_and_same_slot_reuse(cuda_runtime):
    runtime, recorder = cuda_runtime

    runtime.begin()
    first = runtime.submit("slot-0", lambda: recorder.log.append(("copy", "first")))
    runtime.wait_ready(first)
    runtime.finish_use("slot-0")
    compute_done = next(entry[2] for entry in reversed(recorder.log) if entry[:2] == ("record_event", "compute"))
    runtime.submit("slot-0", lambda: recorder.log.append(("copy", "second")))

    wait_index = recorder.log.index(("wait_event", "transfer", compute_done))
    copy_index = recorder.log.index(("copy", "second"))
    assert wait_index < copy_index


def test_different_slots_have_no_cross_slot_wait(cuda_runtime):
    runtime, recorder = cuda_runtime
    runtime.begin()
    runtime.submit("slot-0", lambda: recorder.log.append(("copy", "first")))
    runtime.submit("slot-1", lambda: recorder.log.append(("copy", "second")))
    transfer_waits = [entry for entry in recorder.log if entry[:2] == ("wait_event", "transfer")]
    assert len(transfer_waits) == 1


def test_ticket_repr_exposes_trace_context_but_not_runtime_internals(cuda_runtime):
    runtime, _recorder = cuda_runtime
    runtime.begin()

    ticket = runtime.submit(
        "slot",
        lambda: None,
        trace_label="schedule=embed consume=L0",
    )

    assert repr(ticket) == "TransferTicket(slot='slot', trace_label='schedule=embed consume=L0')"


def test_busy_slot_is_rejected(cuda_runtime):
    runtime, _recorder = cuda_runtime
    runtime.begin()
    active = runtime.submit("slot", lambda: None, trace_label="schedule=embed consume=L0")
    with pytest.raises(RuntimeError, match="still in use") as caught:
        runtime.submit("slot", lambda: None)
    assert repr(active) in str(caught.value)


def test_foreign_ticket_is_rejected(cuda_runtime):
    first, _recorder = cuda_runtime
    second = CudaTransferRuntime(torch.device("cuda:0"))
    first.begin()
    second.begin()
    ticket = first.submit("slot", lambda: None)
    with pytest.raises(ValueError, match="different CudaTransferRuntime"):
        second.wait_ready(ticket)


def test_ticket_is_rejected_after_same_slot_reuse(cuda_runtime):
    runtime, _recorder = cuda_runtime
    runtime.begin()
    stale = runtime.submit("slot", lambda: None, trace_label="schedule=embed consume=L0")
    runtime.wait_ready(stale)
    runtime.finish_use("slot")
    current = runtime.submit("slot", lambda: None)
    runtime.wait_ready(current)

    with pytest.raises(ValueError, match="stale") as caught:
        runtime.wait_ready(stale)
    assert repr(stale) in str(caught.value)


def test_ticket_is_rejected_after_join(cuda_runtime):
    runtime, _recorder = cuda_runtime
    runtime.begin()
    stale = runtime.submit("slot", lambda: None)
    runtime.join()

    with pytest.raises(ValueError, match="stale"):
        runtime.wait_ready(stale)


def test_begin_rejects_nested_lifecycle(cuda_runtime):
    runtime, _recorder = cuda_runtime
    runtime.begin()
    runtime.submit("slot", lambda: None)

    with pytest.raises(RuntimeError, match="already active"):
        runtime.begin()


def test_is_active_tracks_lifecycle(cuda_runtime):
    runtime, _recorder = cuda_runtime
    assert not runtime.is_active

    runtime.begin()
    assert runtime.is_active

    runtime.join()
    assert not runtime.is_active


def test_close_is_idempotent_and_rejects_new_work(cuda_runtime):
    runtime, recorder = cuda_runtime
    runtime.close()
    runtime.close()
    assert [entry[:2] for entry in recorder.log].count(("synchronize", "transfer")) == 1
    with pytest.raises(RuntimeError, match="closed"):
        runtime.begin()


def test_transfer_callback_error_propagates(cuda_runtime):
    runtime, _recorder = cuda_runtime
    runtime.begin()

    def fail() -> None:
        raise LookupError("copy failed")

    with pytest.raises(LookupError, match="copy failed"):
        runtime.submit("slot", fail)


def test_close_wraps_cuda_synchronization_error(cuda_runtime, monkeypatch):
    runtime, _recorder = cuda_runtime
    error = RuntimeError("CUDA failed")

    def fail() -> None:
        raise error

    monkeypatch.setattr(runtime.transfer_stream, "synchronize", fail)
    with pytest.raises(TransferStreamSynchronizationError, match="CUDA failed") as caught:
        runtime.close()
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("operation", ["record_event", "wait_event"])
def test_join_wraps_cuda_synchronization_error(cuda_runtime, monkeypatch, operation):
    runtime, _recorder = cuda_runtime
    runtime.begin()
    error = RuntimeError(f"{operation} failed")

    def fail(*_args) -> None:
        raise error

    stream = runtime.transfer_stream if operation == "record_event" else torch.cuda.current_stream()
    monkeypatch.setattr(stream, operation, fail)

    with pytest.raises(TransferStreamSynchronizationError, match=f"{operation} failed") as caught:
        runtime.join()
    assert caught.value.__cause__ is error

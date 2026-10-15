# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Hashable  # noqa: TC003 - beartype resolves parameter annotations at runtime
from dataclasses import dataclass, field
from typing import Any

import torch


class TransferStreamSynchronizationError(RuntimeError):
    """Transfer stream could not be quiesced safely during shutdown."""


def synchronize_transfer_stream(stream: Any) -> None:
    """Synchronize ``stream`` and expose failures through the core error type."""
    try:
        stream.synchronize()
    except Exception as error:
        raise TransferStreamSynchronizationError(str(error)) from error


@dataclass(frozen=True, slots=True)
class TransferTicket:
    """Opaque readiness handle returned by :meth:`CudaTransferRuntime.submit`.

    A ticket is valid until its slot is submitted again or the runtime is
    joined or closed.
    """

    _runtime_token: object = field(repr=False)
    _event: Any = field(repr=False)
    slot: Hashable
    trace_label: str | None = None


class CudaTransferRuntime:
    """Order asynchronous CUDA transfers against slot consumers.

    The lifecycle is ``begin()`` → ``submit()`` → ``wait_ready()`` /
    ``finish_use()`` → ``join()``. One host thread must own an instance; this
    class is not thread-safe. GPU ordering is expressed only through CUDA
    streams and events, with no host mutex or condition-variable contract.
    """

    def __init__(self, device: torch.device) -> None:
        self.transfer_stream = torch.cuda.Stream(device=device)
        self._runtime_token = object()
        self._active_ticket_by_slot: dict[Hashable, TransferTicket] = {}
        self._last_use_by_slot: dict[Hashable, Any] = {}
        self._busy_slots: set[Hashable] = set()
        self._active = False
        self._closed = False

    @property
    def is_active(self) -> bool:
        """Whether a begun transfer lifecycle still needs an explicit join."""
        return self._active

    def begin(self) -> None:
        """Fork the current CUDA stream and start one transfer lifecycle."""
        self._ensure_open()
        if self._active:
            raise RuntimeError("CudaTransferRuntime is already active")
        self._last_use_by_slot.clear()
        self._busy_slots.clear()
        fork_event = torch.cuda.current_stream().record_event()
        self.transfer_stream.wait_event(fork_event)
        self._active = True

    def submit(
        self,
        slot: Hashable,
        transfer: Callable[[], None],
        *,
        trace_label: str | None = None,
    ) -> TransferTicket:
        """Enqueue ``transfer`` and return its readiness ticket.

        A busy slot must first be released with :meth:`finish_use`. Submitting
        a released slot invalidates its previous ticket.
        """
        self._ensure_open()
        if not self._active:
            raise RuntimeError("begin() must be called before submit()")
        if slot in self._busy_slots:
            raise RuntimeError(f"{self._active_ticket_by_slot[slot]} is still in use")
        previous_use = self._last_use_by_slot.get(slot)
        if previous_use is not None:
            self.transfer_stream.wait_event(previous_use)
        self._active_ticket_by_slot.pop(slot, None)
        with torch.cuda.stream(self.transfer_stream):
            transfer()
        event = self.transfer_stream.record_event()
        ticket = TransferTicket(self._runtime_token, event, slot, trace_label)
        self._active_ticket_by_slot[slot] = ticket
        self._busy_slots.add(slot)
        return ticket

    def wait_ready(self, ticket: TransferTicket) -> None:
        """Make the current CUDA stream wait for a live transfer ticket."""
        self._ensure_open()
        if ticket._runtime_token is not self._runtime_token:  # noqa: SLF001 - tickets are opaque to callers
            raise ValueError("ticket belongs to a different CudaTransferRuntime")
        if self._active_ticket_by_slot.get(ticket.slot) is not ticket:
            raise ValueError(f"stale {ticket}")
        torch.cuda.current_stream().wait_event(ticket._event)  # noqa: SLF001 - tickets are opaque to callers

    def finish_use(self, slot: Hashable) -> None:
        """Record when the current stream finishes consuming ``slot``."""
        self._ensure_open()
        if not self._active:
            raise RuntimeError("begin() must be called before finish_use()")
        self._last_use_by_slot[slot] = torch.cuda.current_stream().record_event()
        self._busy_slots.discard(slot)

    def join(self) -> None:
        """Join transfer work into the current stream and invalidate tickets."""
        self._ensure_open()
        if self._active:
            try:
                join_event = self.transfer_stream.record_event()
                torch.cuda.current_stream().wait_event(join_event)
            except Exception as error:
                raise TransferStreamSynchronizationError(str(error)) from error
            self._active = False
        self._active_ticket_by_slot.clear()
        self._last_use_by_slot.clear()
        self._busy_slots.clear()

    def close(self) -> None:
        """Synchronize and permanently close the runtime; repeated calls are safe."""
        if self._closed:
            return
        synchronize_transfer_stream(self.transfer_stream)
        self._active_ticket_by_slot.clear()
        self._last_use_by_slot.clear()
        self._busy_slots.clear()
        self._active = False
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CudaTransferRuntime is closed")

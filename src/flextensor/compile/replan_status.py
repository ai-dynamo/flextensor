# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public snapshot of the strategy-replan lifecycle."""

from dataclasses import dataclass
from typing import Literal

from flextensor.compile.warmup_tail import CompiledOffloadTailState

StrategyReplanState = Literal["idle", "in_progress", "done", "failed"]
"""Public replan lifecycle vocabulary (see :class:`StrategyReplanStatus`)."""


@dataclass(frozen=True, slots=True)
class StrategyReplanStatus:
    """Passive snapshot of
    :meth:`~flextensor.OffloadManager.request_strategy_replan` state.

    Read from
    :attr:`~flextensor.OffloadManager.strategy_replan_status`; reading never
    arms or advances the replan tail.

    Attributes:
        state: ``"idle"`` before any replan request, ``"in_progress"`` while the
            warm/measure tail rides the caller's forwards, ``"done"`` once it
            finished (applied or refused), ``"failed"`` when the rebuild raised
            — inference is unsafe after ``"failed"``.
        can_request: Whether calling ``request_strategy_replan()`` now is
            supported. A ``"done"`` replan that refused to rebuild can still be
            retried while FlexTensor holds the source weights;
            ``"in_progress"`` and ``"failed"`` cannot.
        applied: ``None`` before and during a request, ``True`` after a
            successful rebuild, ``False`` after a completed soft refusal
            (strategy kept) or a failure.
    """

    state: StrategyReplanState
    can_request: bool
    applied: bool | None


_PUBLIC_STATE_BY_TAIL: dict[CompiledOffloadTailState, StrategyReplanState] = {
    CompiledOffloadTailState.IDLE: "idle",
    CompiledOffloadTailState.WARMING: "in_progress",
    CompiledOffloadTailState.MEASURING: "in_progress",
    CompiledOffloadTailState.DONE: "done",
    CompiledOffloadTailState.FAILED: "failed",
}


def build_strategy_replan_status(
    tail_state: CompiledOffloadTailState,
    *,
    armed: bool,
    replan_applied: bool | None,
    can_request: bool,
) -> StrategyReplanStatus:
    """Project internal tail state onto :class:`StrategyReplanStatus`.

    Collapses the ``WARMING``/``MEASURING`` stages into ``"in_progress"`` so the
    public vocabulary survives changes to the tail. A ``DONE`` tail that was
    never armed reports ``"idle"``: compiled view-profile closes the tail
    without a replan request, and no request means no outcome to report.
    """
    if not armed and tail_state is CompiledOffloadTailState.DONE:
        return StrategyReplanStatus(state="idle", can_request=can_request, applied=None)
    state = _PUBLIC_STATE_BY_TAIL[tail_state]
    # Enforce the documented contract rather than mirroring ``replan_applied``:
    # an outcome exists only once a request has finished.
    applied = replan_applied if state in ("done", "failed") else None
    return StrategyReplanStatus(state=state, can_request=can_request, applied=applied)

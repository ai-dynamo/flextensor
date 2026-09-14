# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Passive warm → measure → done counter used by compiled-offload replan."""

from enum import Enum


class CompiledOffloadTailState(Enum):
    """State of the passive compiled-offload warm → measure → replan tail."""

    IDLE = "idle"
    WARMING = "warming"
    MEASURING = "measuring"
    DONE = "done"
    FAILED = "failed"


class WarmupTail:
    """Count unmeasured warmup forwards, then measured forwards, then finish.

    Used by the compiled replan tail (and similar fixed-warmup patterns).
    """

    def __init__(self, *, warmup_forwards: int, measure_forwards: int) -> None:
        self.warmup_forwards = max(warmup_forwards, 0)
        self.measure_forwards = max(measure_forwards, 0)
        self.state = CompiledOffloadTailState.IDLE
        self.warm_seen = 0
        self.measure_seen = 0
        self.failure: BaseException | None = None
        # Set when the tail reaches DONE: True if a new strategy was applied,
        # False if finish returned a soft keep-current (empty budgets / rebuild
        # refused). None while the tail has not finished.
        self.replan_applied: bool | None = None
        # Whether :meth:`arm` ran. ``mark_done`` also closes tails that were
        # never armed (compiled view-profile), which have no replan outcome to
        # report; see ``flextensor.compile.replan_status``.
        self.armed: bool = False
        # When False, skip enable_compiled_profiling on MEASURING entry (e.g.
        # CUDA-graph offload-timing replan supplies budgets elsewhere).
        self.enable_profiling: bool = True

    def reset(self) -> None:
        self.state = CompiledOffloadTailState.IDLE
        self.warm_seen = 0
        self.measure_seen = 0
        self.failure = None
        self.replan_applied = None
        self.enable_profiling = True
        self.armed = False

    def arm(self, *, credited_warm: int = 0, enable_profiling: bool = True) -> int:
        """Arm warm→measure; return remaining forwards until finish."""
        credited = min(max(credited_warm, 0), self.warmup_forwards)
        self.warm_seen = credited
        self.measure_seen = 0
        self.failure = None
        self.replan_applied = None
        self.enable_profiling = enable_profiling
        self.armed = True
        remaining_warm = self.warmup_forwards - credited
        if credited >= self.warmup_forwards:
            self.state = CompiledOffloadTailState.MEASURING
        else:
            self.state = CompiledOffloadTailState.WARMING
        return remaining_warm + self.measure_forwards

    def mark_done(self, *, applied: bool = True) -> None:
        """Mark the tail complete. ``applied=False`` means soft keep-current."""
        self.state = CompiledOffloadTailState.DONE
        self.replan_applied = applied

    def mark_failed(self, exc: BaseException) -> None:
        self.state = CompiledOffloadTailState.FAILED
        self.failure = exc
        self.replan_applied = False

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiled-offload lifecycle: per-unit ``compile_fn``, warm/measure/replan tail.

Owns the compile orchestration that previously lived on :class:`~flextensor.OffloadManager`.
Forward builders remain in :mod:`flextensor.compiled_offload`; residency custom ops in
:mod:`flextensor.custom_ops`.
"""

from flextensor.compile.lifecycle import (
    COMPILED_EAGER_PROFILE_FORWARDS,
    COMPILED_WARMUP_FORWARDS,
    CompiledOffload,
)
from flextensor.compile.replan_status import StrategyReplanState, StrategyReplanStatus

__all__ = [
    "COMPILED_EAGER_PROFILE_FORWARDS",
    "COMPILED_WARMUP_FORWARDS",
    "CompiledOffload",
    "StrategyReplanState",
    "StrategyReplanStatus",
]

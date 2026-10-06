# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join the vLLM offloader at every breakable CUDA-graph segment boundary.

``BreakableCUDAGraphCapture.add_eager`` ends the current capture segment in
the middle of a decoder layer, but vLLM only calls the offloader's
``join_after_forward`` before the *last* segment closes. Transfer-stream work
forked at a FlexTensor layer ``enter`` is then still unjoined at the eager
break (``cudaErrorStreamCaptureUnjoined``), and its readiness event belongs to
a capture that no longer exists.
"""

from __future__ import annotations

import functools
from typing import Any

from vllm.logger import init_logger

LOGGER = init_logger("vllm.flextensor.v2.breakable_cudagraph")

_JOIN_MARKER = "_ft_vllm_joins_offloader"


def install_breakable_segment_join() -> bool:
    """Patch ``add_eager`` to join the active offloader first; idempotent.

    Returns whether the hook is installed (False when this vLLM build has no
    breakable CUDA-graph capture).
    """
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        from vllm.model_executor.offloader.base import get_offloader
    except ImportError:
        return False

    add_eager = BreakableCUDAGraphCapture.add_eager
    if getattr(add_eager, _JOIN_MARKER, False):
        return True

    @functools.wraps(add_eager)
    def add_eager_after_offloader_join(capture: Any, fn: Any) -> Any:
        if getattr(capture, "_capturing", False):
            get_offloader().join_after_forward()
        return add_eager(capture, fn)

    setattr(add_eager_after_offloader_join, _JOIN_MARKER, True)
    BreakableCUDAGraphCapture.add_eager = add_eager_after_offloader_join  # type: ignore[method-assign, unused-ignore]
    LOGGER.info("FlexTensor joins the offloader before each breakable CUDA-graph eager segment")
    return True

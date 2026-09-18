# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch compiler helpers used at FlexTensor runtime boundaries."""

import functools
import gc
import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

import torch

_dynamo: Any | None
try:
    import torch._dynamo as _imported_dynamo
except ImportError:  # pragma: no cover - torch built without dynamo
    _dynamo = None
else:
    _dynamo = _imported_dynamo

F = TypeVar("F", bound=Callable[..., object])
_INDUCTOR_PATCH_WARNED = False
_MUTATION_READS_CACHE_TAG = "flextensor-mutation-reads-v1"
LOGGER = logging.getLogger(__name__)


def patch_inductor_mutation_reads() -> bool:
    """Correct mutation-renamed inputs in Inductor's tiling analysis, once per process.

    The faulty lookup exists in Torch 2.10 and 2.14; its numerical effect depends
    on the compiled graph and device. Match the exact unfixed loop so upstream
    fixes and other implementations are left alone. This changes compiler Python
    state, never installed files or model math. Remove when supported Torch builds
    all resolve mutation_real_name here.

    Related upstream reports (different passes, not fixes for this tiling lookup):
    - https://github.com/pytorch/pytorch/issues/196570 documents the same
      scheduler/LoopBody name mismatch in CPU local-buffer promotion;
      https://github.com/pytorch/pytorch/pull/196639 addresses that CPU path.
    - https://github.com/pytorch/pytorch/pull/153899 remaps mutation_real_name
      for graph-partition inputs/outputs, establishing the same naming distinction.
    """
    global _INDUCTOR_PATCH_WARNED
    try:
        import torch._inductor.tiling_utils as tiling
        from torch._dynamo.convert_frame import compile_lock
        from torch._inductor.codecache import FxGraphHashDetails
    except ImportError:  # Older Torch versions do not have this analysis.
        return False

    # Keep cache keys and tiling consistent with in-flight Dynamo compilations.
    with compile_lock:
        original = getattr(tiling, "extract_normalized_read_writes", None)
        if original is None:
            return False
        if getattr(original, "_ft_mutation_reads_fixed", False):
            return True
        try:
            source = inspect.getsource(original)
        except (OSError, TypeError):
            LOGGER.warning("Cannot inspect Inductor tiling analysis; mutation-read compatibility fix not applied")
            return False
        before = (
            "        for inp in inputs:\n"
            "            for expr in body.get_all_read_expr(inp):\n"
            "                n_reads[expr].add(inp)"
        )
        after = before.replace(
            "        for inp in inputs:\n",
            "        for inp in inputs:\n            inp = V.graph.scheduler.mutation_real_name.get(inp, inp)\n",
        )
        if source.count(before) != 1:
            if after not in source and not _INDUCTOR_PATCH_WARNED:
                LOGGER.warning(
                    "Unrecognized Inductor tiling analysis (torch %s); mutation-read compatibility fix not applied. "
                    "Verify compiled output parity on this build.",
                    torch.__version__,
                )
                _INDUCTOR_PATCH_WARNED = True
            return False
        namespace = dict(vars(tiling))
        # Execute only the installed dependency's source with the one guarded correction.
        exec(compile(source.replace(before, after), "<flextensor Inductor compatibility>", "exec"), namespace)  # noqa: S102
        corrected = namespace[original.__name__]
        functools.update_wrapper(corrected, original)
        corrected._ft_mutation_reads_fixed = True  # noqa: SLF001 - marker owned by FT
        original_hash_init = FxGraphHashDetails.__init__

        @functools.wraps(original_hash_init)
        def hash_init(details, *args, **kwargs):
            original_hash_init(details, *args, **kwargs)
            # Hash-time tagging survives caller config contexts; also covers AOTAutograd's subclass.
            if _MUTATION_READS_CACHE_TAG not in details.cache_key_tag.split("|"):
                details.cache_key_tag += "|" + _MUTATION_READS_CACHE_TAG

        FxGraphHashDetails.__init__ = hash_init
        tiling.extract_normalized_read_writes = corrected
        LOGGER.info("Applied Inductor mutation-read compatibility fix (torch %s)", torch.__version__)
        return True


def disable(func: F) -> F:
    """Disable torch compiler tracing for ``func`` when the public API exists."""
    compiler = getattr(torch, "compiler", None)
    compiler_disable = getattr(compiler, "disable", None) if compiler is not None else None
    if compiler_disable is not None:
        return cast("F", compiler_disable(func))
    dynamo_disable = getattr(_dynamo, "disable", None)
    if dynamo_disable is None:
        return func
    return cast("F", dynamo_disable(func))


def is_compiling() -> bool:
    """Return whether PyTorch is currently tracing/compiling user code."""
    compiler = getattr(torch, "compiler", None)
    compiler_is_compiling = getattr(compiler, "is_compiling", None) if compiler is not None else None
    if compiler_is_compiling is not None and compiler_is_compiling():
        return True
    dynamo_is_compiling = getattr(_dynamo, "is_compiling", None)
    return bool(dynamo_is_compiling is not None and dynamo_is_compiling())


def graph_break() -> None:
    """Insert a Dynamo graph break when Dynamo is available."""
    if _dynamo is None:
        return
    _dynamo.graph_break()


def is_torch_compiled_module(module: object) -> bool:
    """Return whether ``module`` is a ``torch.compile`` ``OptimizedModule`` wrapper.

    Detects the wrapper that ``torch.compile`` (and backends built on it, e.g.
    ``torch_tensorrt``) returns, including subclasses. FlexTensor cannot
    preprocess this wrapper: its self-referential ``__getattr__``/``__setattr__``
    proxy to ``_orig_mod`` makes module tree-walking recurse without termination,
    so ``offload()`` rejects it up front (compile *after* offload instead).

    Uses ``isinstance`` against Dynamo's ``OptimizedModule`` when available.
    Returns ``False`` when Dynamo is unavailable — there is no real compile
    wrapper to detect, and a class-name check would false-positive on unrelated
    types named ``OptimizedModule``.
    """
    if _dynamo is None:
        return False
    eval_frame = getattr(_dynamo, "eval_frame", None)
    optimized_cls = getattr(eval_frame, "OptimizedModule", None) if eval_frame is not None else None
    if optimized_cls is None:
        return False
    return isinstance(module, optimized_cls)


def find_torch_compile_wrapper(target: object) -> object | None:
    """Return a ``torch.compile`` OptimizedModule wrapper around ``target``."""
    if _dynamo is None:
        return None
    eval_frame = getattr(_dynamo, "eval_frame", None)
    optimized_cls = getattr(eval_frame, "OptimizedModule", None) if eval_frame is not None else None
    if optimized_cls is None:
        return None
    for obj in gc.get_objects():
        if type(obj) is not optimized_cls:
            continue
        try:
            orig = obj._orig_mod  # noqa: SLF001 - Dynamo exposes the wrapped module here
        except Exception:  # noqa: S112 - skip partially initialized Dynamo internals
            continue
        if orig is target:
            return cast("object", obj)
    return None

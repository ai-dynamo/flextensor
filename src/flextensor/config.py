# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes and utilities for FlexTensor.

This module provides configuration management for the offload manager,
including support for loading configuration from environment variables and files.
"""

import configparser
import difflib
import json
import logging
import os
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, Field, WithJsonSchema, field_validator, model_validator

from flextensor.host_pinning import PinnedMemoryMode
from flextensor.strategy import AdaptiveStrategy, Strategy
from flextensor.utils import CLASS_PATTERN_PREFIX, NAME_PATTERN_PREFIX

logger = logging.getLogger(__name__)

_REGISTERED_ENV_VARS: set[str] = set()

ProfileMode = Literal["torch_function", "getter", "view"]
OffloadTimingMode = Literal["off", "eager", "cuda_graph"]
PiecewisePrefetchMode = Literal["off", "warn", "error"]
BLOCK_TRANSFER_MODES = frozenset({"allocation_block_transfer", "raw_block_transfer"})
OFFLOAD_TRANSFER_MODES = frozenset({"strategy", *BLOCK_TRANSFER_MODES})

_REMOVED_FIELDS: dict[str, tuple[str, str]] = {
    "release_tensors": ("v0.2.0", "GPU tensors are now always released after layer execution."),
    "enable_direct_mode": ("v0.2.0", "Direct mode is now always enabled."),
    "enable_tracing": ("v0.2.0", "Use TensorManager(_use_trace_tensor=True) for tracing."),
    "rearrange_transfers": ("v0.2.0", "Transfer rearrangement is now auto-enabled when gap layers are detected."),
    "compute_transfer_gap": ("v0.2.0", "Use TensorManager(_compute_transfer_gap=N) to override."),
    "enable_untraced_tensor_discovery": (
        "v0.2.0",
        "Untraced tensor discovery is now always enabled."
        " Use TensorManager(_enable_untraced_tensor_discovery=False) to override.",
    ),
    "enable_module_tracker": ("v0.2.0", "ModuleTracker is now always enabled for manual traps."),
    "warmup_iters": ("v0.4.0", "Use discovery_iters instead."),
    "profile_iters": ("v0.4.0", "Use profiling_iters instead."),
}


class OffloadConfig(BaseModel):
    """Configuration for offload manager."""

    enabled: bool = Field(default=True)
    """Whether to enable offload."""

    gpu_device: int = Field(default=0, ge=0)
    """GPU device index to use."""

    pinned_memory: bool = Field(default=True)
    """Whether to use pinned (page-locked) memory for CPU tensors, enabling non-blocking GPU transfers."""

    pinned_memory_mode: PinnedMemoryMode = Field(default="torch")
    """How to pin CPU tensors when ``pinned_memory=True``.

    - ``"torch"`` (default) — :meth:`torch.Tensor.pin_memory`: copies the
      tensor into a fresh pinned allocation.
    - ``"host_register"`` — ``cudaHostRegister``: pins the existing allocation
      in place (no copy, lower peak host memory). Requires a loadable CUDA
      runtime.

    Has no effect when ``pinned_memory=False``.
    Can also be set via the ``FT_PINNED_MEMORY_MODE`` env var.

    .. note:: Only controls the non-SHM allocator path. When
        ``shm_enabled=True`` *and* the SHM segment is built with
        ``pinned_memory=True``, that segment is registered in place via
        ``cudaHostRegister`` regardless of this mode.

    .. note:: Records the requested value and never mutates. Two
        construction-time outcomes can diverge from a naive read of this
        field:

        - **CUDA available, cudart binding broken/missing** —
          :class:`~flextensor.tensor_manager.TensorManager` falls back from
          ``"host_register"`` to ``"torch"`` and emits a ``WARNING``.
        - **CUDA unavailable** — constructing
          :class:`~flextensor.tensor_manager.TensorManager` with
          ``pinned_memory=True`` raises ``RuntimeError``. Set
          ``pinned_memory=False`` on CPU-only hosts.
    """

    shm_enabled: bool = Field(default=False)
    """Enable cross-process weight sharing via POSIX shared memory.

    When true, model weights are stored in shared memory so multiple processes
    can share a single copy in CPU RAM. This selects SHM-backed allocation and
    requires a non-empty :attr:`shm_namespace` at configuration time. A
    multi-process application must also coordinate one creator with its
    followers. Enabling this flag alone does not elect a creator or build
    follower models on the meta device.
    Can also be set via the ``FT_SHM_ENABLED`` env var.
    """

    shm_namespace: str | None = Field(default=None)
    """Base namespace for shared memory blocks.

    Must be set explicitly when :attr:`shm_enabled` is true. Applications that
    have a stable model path may derive a value with
    :func:`flextensor.shm.compute_shm_namespace` before initializing a manager.
    Can also be set via FT_SHM_NAMESPACE env var.
    """

    shm_wait_timeout: float = Field(default=0.0, ge=0.0)
    """Timeout (seconds) for followers waiting on creator.

    0 means no overall limit; followers still use creator-heartbeat liveness
    detection. A positive value also bounds coordination-lock acquisition,
    though scheduler delays can make observed elapsed time slightly longer.
    Can also be set via FT_SHM_WAIT_TIMEOUT env var.
    """

    discovery_iters: int = Field(default=1, ge=0)
    """Number of discovery-phase iterations before profiling begins.

    Can also be set via the ``FT_DISCOVERY_ITERS`` env var.
    """

    skip_discovery: bool = Field(default=False)
    """Skip the discovery phase and jump directly to profiling.

    When ``True``, tensor-to-layer mappings are discovered statically from
    modules patched via ``include_patterns`` instead of being inferred from
    discovery-phase iterations. Falls back to the discovery phase if no
    modules are patched; a ``WARNING`` is logged and
    ``OffloadManager.skip_discovery_honored`` flips to ``False`` so the
    fallback is detectable without log scraping.

    Leave at the default ``False`` for manual ``offload_block()`` blocks.
    ``offload_block()`` raises when the skip is honored; the no-patched-
    modules fallback still allows manual blocks. Set ``True`` only on the
    auto-trap (``include_patterns``) path to cut startup time.

    .. note:: Minimum-iteration contract

        Because ``OffloadManager.update_state()`` runs as a post-forward hook,
        the PROFILING→INFERENCE transition can only fire *after* the first
        profile forward completes. Any transition — including
        ``profiling_iters=0`` — therefore always sees at least one profile
        forward's worth of collector data. Setting ``profiling_iters=0`` and
        ``profiling_iters=1`` produce the same runtime behavior; both are
        low-quality strategy inputs (``report_profiling_quality`` warns).
        The only way to reach INFERENCE without any profile forward is to
        restore a saved profile via ``load_profile`` / ``offload_from_profile``.

    The value is read once when the underlying ``TensorManager`` is first
    constructed (during the first ``offload()`` call). Calling ``set_config``
    with a different ``skip_discovery`` afterwards raises ``RuntimeError``;
    call ``release()`` and re-``offload()`` to pick up a new value.
    """

    profiling_iters: int = Field(default=10, ge=0)
    """Number of profiling-phase iterations to collect timing statistics.

    Can also be set via the ``FT_PROFILING_ITERS`` env var.
    """

    external_compile: bool = Field(default=False)
    """Prepare offloaded layers for *external* per-unit ``torch.compile`` after INFERENCE.

    Does **not** compile the model itself. It arms the custom-op / rolling-loader
    path so a caller (e.g. vLLM or a script) can ``torch.compile`` each offloaded
    unit after the INFERENCE transition. Pass ``compile_fn`` to
    :func:`~flextensor.offload` for the integrated path instead.

    Can also be set via ``FT_EXTERNAL_COMPILE``.

    Requires a block ``transfer_mode`` (``allocation_block_transfer`` or
    ``raw_block_transfer``); ``strategy`` is incompatible with the rolling-block
    custom ops used by compiled offload.

    With ``compile_fn`` and ``profile_mode='view'`` (default), FlexTensor
    auto-runs compiled view-profile so the strategy is built from compiled
    timings (no post-INFERENCE replan). Use
    :meth:`~flextensor.OffloadManager.request_strategy_replan` after a non-view
    profile or after external compile with ``external_compile=True``.
    """

    transfer_budget_scale: float = Field(default=1.0, gt=0.0)
    """Multiplier on the time budget for weight transfers.

    Values below 1.0 add a safety margin to absorb profiling measurement noise
    (e.g., 0.95 = 5% margin). Most relevant in latency mode
    (``max_gpu_mem_fraction=None``); in memory mode the strategy may override
    this value to meet the GPU memory target.
    """

    transfer_mode: str = Field(default="allocation_block_transfer")
    """Tensor loading mode. Block transfer loaders (``allocation_block_transfer``,
    ``raw_block_transfer``) do not support shared tensors between layers;
    use ``strategy`` for models with shared weights."""

    num_blocks: int = Field(default=4, ge=2)
    """Number of memory blocks for block transfer loaders.

    More blocks reserve more GPU memory but give the pipelining scheduler more
    room to overlap transfers with compute. Fewer blocks (e.g., 2) reduce GPU
    memory overhead and can work well for models whose layers have long compute
    times (such as diffusers), since the transfer easily finishes within a
    single layer's compute window.
    """

    min_blocks: int = Field(default=4, ge=2)
    """Minimum number of blocks for assignment optimization search range.

    The optimizer tries block counts from ``min_blocks`` to ``num_blocks`` and
    picks the best. Lower values give more freedom and reduce GPU memory, but
    may hurt pipelining throughput. Must be >= 2 (pipelined execution requires
    at least two blocks).
    Can also be set via the ``FT_MIN_BLOCKS`` env var.
    """

    max_gpu_mem_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    """Target maximum GPU memory as a fraction of total device memory (e.g. 0.9 = 90%).

    When set, the strategy switches to *memory mode* and keeps peak GPU usage within this
    budget. At runtime, the fractional budget is capped by actual available GPU memory so
    the strategy never targets more than what is free. When ``None``, the strategy operates
    in *latency mode* with no explicit memory constraint. Defaults to ``None``.
    Can also be set via the ``FT_MAX_GPU_MEM_FRACTION`` env var.
    """

    profile_storage_dir: str | None = Field(default=None)
    """Directory for profile persistence."""

    profile_read_only: bool = Field(default=False)
    """Only load profiles, saving is disabled."""

    load_strategy: Annotated[Strategy, WithJsonSchema({"type": "object"})] | None = Field(default=None)
    """Override automatic strategy selection."""

    enable_instrumentation: bool = Field(default=False)
    """Whether to capture component init args for debugging."""

    instrumentation_output_dir: str = Field(default=".flextensor/instrumentation")
    """Instrumentation output directory."""

    offload_timing: OffloadTimingMode = Field(default="off")
    """Inference offload-timing instrumentation mode.

    * ``"off"`` — disabled (default).
    * ``"eager"`` — record per-trap transfer / compute / wait with internal
      CUDA timing events (module forwards / collect via
      :meth:`OffloadManager.collect_offload_timing`).
    * ``"cuda_graph"`` — same measurement using
      ``torch.cuda.Event(..., external=True)`` so ``elapsed_time()`` is fresh
      after CUDA-graph replay (PyTorch >= 2.8). Required for
      :meth:`OffloadManager.request_strategy_replan` with
      ``manual_update_state=True``.

    Periodic log cadence and durable measure retention are internal defaults
    (not independently tunable). Can also be set via ``FT_OFFLOAD_TIMING``.

    Requires a block ``transfer_mode`` (``allocation_block_transfer`` or
    ``raw_block_transfer``). ``transfer_mode='strategy'`` has no
    ``PreallocatedLoader`` enter/exit timing hooks, so enabling timing with
    strategy is rejected at config construction.

    One-shot: consumed when the ``TensorManager`` is first constructed (first
    ``offload()``). A later ``set_config`` with a different value warns; call
    ``release()`` and re-``offload()`` to apply it.
    """

    piecewise_prefetch: PiecewisePrefetchMode = Field(default="warn")
    """Policy when a PIECEWISE join forces outstanding H2D onto the critical path.

    Under PIECEWISE capture, ``join_after_forward`` drains ``transfer_stream`` at
    the piece boundary. If schedule and wait straddle pieces (rearrange
    early-slot, or nested enter/exit such as parent ``1`` with graphs on
    ``1.1`` / ``1.2``), the transfer still completes but async overlap is lost.

    * ``"off"`` — disable the check.
    * ``"warn"`` — log a warning (default; lost overlap is not silent).
    * ``"error"`` — raise
      :class:`~flextensor.piecewise_prefetch_policy.PiecewisePrefetchPolicyError`.

    Integrations must call the block loader's ``join_after_forward()`` before
    each piece's ``capture_end``. There is no in-tree automatic piece hook yet;
    without that call only the last-trap join runs, so mid-piece boundaries are
    neither joined nor checked.

    Can also be set via ``FT_PIECEWISE_PREFETCH``.

    One-shot: consumed when the ``TensorManager`` is first constructed (first
    ``offload()``). A later ``set_config`` with a different value warns; call
    ``release()`` and re-``offload()`` to apply it.
    """

    include_patterns: list[str] = Field(default_factory=lambda: ["*"])
    """Module or parameter patterns to include for offloading (supports ``*`` and ``?`` wildcards).

    Each entry is one of three forms:

    * ``<glob>`` — name-based (module or parameter path), existing behaviour.
    * ``name:<glob>`` — name-based, explicit form.  Equivalent to ``<glob>``.
    * ``class:<glob>`` — class-based; matches modules whose class name matches
      the glob.  Each pattern is tested against both the short class name
      (``type(m).__name__``) and the fully-qualified class name
      (``f"{cls.__module__}.{cls.__qualname__}"``); a match on either wins.
      Useful for hybrid architectures where a single Python class (e.g.
      ``SharedExpertMLP``) appears at varying paths and renames across
      upstream versions.  FQCN globs such as ``class:torch.nn.*.Linear``
      provide an escape hatch when the same short name is used by multiple
      classes.

    Name patterns are matched against both module paths (from
    ``model.named_modules()``) and parameter paths (from
    ``model.named_parameters()``).

    **Module-level patterns** (e.g., ``layers.*``) patch matching modules and offload
    all their parameters.

    **Parameter-level patterns** (e.g., ``layers.*.weight``) automatically derive the
    module-level pattern (``layers.*``) for patching, but only offload parameters that
    match the full pattern. Non-matching parameters stay on GPU permanently.

    **Class patterns** patch every module of a matching class; all of that
    module's parameters are offloaded.  Class patterns are module-level only —
    ``class:*Expert*`` covers every descendant parameter of any matching module.

    Can also be set via the ``FT_INCLUDE_PATTERNS`` env var as a comma-separated list.

    .. note::
        Standalone ``*`` has different semantics depending on context:

        * **Module selection** — ``*`` matches exactly one path segment.
          ``layers.*`` matches ``layers.0`` but not ``layers.0.attn``.
        * **Parameter filtering** — ``*`` matches one or more segments.
          ``*.weight`` matches both ``linear.weight`` and ``layers.0.attn.weight``.
        * **Class matching** — ``*`` is a single-segment glob spanning the
          whole class-name string.  Short class names have no dots, so
          ``class:*Expert*`` behaves as a simple substring glob.  Patterns
          containing dots only match against the fully-qualified class name
          (e.g., ``class:*.SharedExpertMLP``).

    Validated at construction; assignment / in-place mutation bypasses validation.
    """

    exclude_patterns: list[str] = Field(default_factory=list)
    """Module/parameter patterns to exclude from offloading (supports wildcards).

    Accepts the same three forms as :attr:`include_patterns`: bare ``<glob>``,
    ``name:<glob>``, or ``class:<glob>``.  Class patterns match against both
    the short class name and the fully-qualified class name (see
    :attr:`include_patterns`).

    Applied after include_patterns: targets matching both include and exclude are NOT offloaded.
    Standalone ``*`` matches one or more segments in both module selection and parameter
    filtering, so ``foo.*`` excludes all descendants of ``foo``.  A ``class:`` match
    cascades to every descendant parameter of the matched module.

    Can also be set via the ``FT_EXCLUDE_PATTERNS`` env var as a comma-separated list.

    Example:
        Keeping Nemotron-H MoE routers and shared-expert MLPs on GPU while
        offloading the rest of each layer::

            exclude_patterns = ["class:SharedExpertMLP", "class:MoELayer", "*.norm"]

    Validated at construction; assignment / in-place mutation bypasses validation.
    """

    enable_diagnostics: bool = Field(default=False)
    """Whether to log diagnostic information (per-trap duration statistics, block assignment table)
       after strategy computation.
    """

    profile_mode: ProfileMode = Field(default="view")
    """Profile-phase mechanism (how the profile model is patched).

    * ``"view"`` — default. Pre-allocates substantial GPU memory during profile;
      pick ``"getter"`` if tight on memory.
    * ``"getter"`` — property getters route parameter access through the
      per-trap loader. Lower GPU footprint, attribute-getter overhead in
      per-trap durations.
    * ``"torch_function"`` — fallback for models that reject the patching used
      by ``"view"`` / ``"getter"``. Heavy per-op overhead; only valid with
      ``transfer_mode='strategy'``.

    ``"view"`` and ``"getter"`` are the two profiling variants of the
    model-patching ("direct") runtime and affect only the profile phase;
    ``"torch_function"`` selects the indirect runtime, which also changes
    warmup and inference.

    See ``profile_mode`` in ``docs/explanation/configuration.md`` for the
    timing model, memory formula, and tradeoffs. Can also be set via the
    ``FT_PROFILE_MODE`` env var.
    """

    model_config = {"arbitrary_types_allowed": True, "use_attribute_docstrings": True}

    @field_validator("include_patterns", "exclude_patterns", mode="before")
    @classmethod
    def _validate_pattern_entries(cls, v: Any) -> list[str] | Any:
        """Validate, normalize, and reject malformed pattern entries.

        Catches at construction (instead of at first ``offload()`` call):

        - Non-string entries: pydantic's default ``list[str]`` coercion would
          silently turn ``42`` into ``"42"``.
        - Empty / whitespace-only entries — whether the whole entry
          (``""``, ``"  "``) or the body after a ``class:`` / ``name:``
          prefix (``"class:"``, ``"name: "``).
        - Typo prefixes (``clas:Foo``, ``Class:Foo``): patterns containing
          ``:`` that don't start with a known prefix would otherwise route
          to ``name_bodies`` and silently never match.

        Whitespace around the entry and around the body is stripped so
        ``"class: Linear "`` is stored as ``"class:Linear"``; otherwise the
        padded body would never match ``cls.__name__``.

        ``str`` / ``bytes`` are excluded since they're sequences of characters,
        not pattern lists; non-sequence inputs fall through to pydantic.
        """
        if not isinstance(v, Sequence) or isinstance(v, (str, bytes)):
            return v
        normalized: list[str] = []
        for i, entry in enumerate(v):
            if not isinstance(entry, str):
                raise ValueError(f"pattern entries must be strings; got {type(entry).__name__} at index {i}: {entry!r}")
            entry = entry.strip()
            if not entry:
                raise ValueError(f"pattern entry at index {i} is empty or whitespace-only; remove it")
            for prefix in (CLASS_PATTERN_PREFIX, NAME_PATTERN_PREFIX):
                if entry.startswith(prefix):
                    body = entry[len(prefix) :].strip()
                    if not body:
                        raise ValueError(
                            f"pattern {entry!r} at index {i} has an empty body; use {prefix}<glob> or remove the entry"
                        )
                    entry = f"{prefix}{body}"
                    break
            else:
                if ":" in entry:
                    raise ValueError(
                        f"pattern {entry!r} at index {i} contains ':' but doesn't start "
                        f"with {CLASS_PATTERN_PREFIX!r} or {NAME_PATTERN_PREFIX!r}; "
                        f"did you mean {CLASS_PATTERN_PREFIX}<glob>?"
                    )
            normalized.append(entry)
        return normalized

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field, (version, msg) in _REMOVED_FIELDS.items():
                if field in data:
                    raise ValueError(
                        f"'{field}' was removed in {version}. {msg} Remove '{field}' from your OffloadConfig() call."
                    )
        return data

    @model_validator(mode="after")
    def _validate_block_range(self) -> Self:
        if self.num_blocks < self.min_blocks:
            raise ValueError(f"num_blocks ({self.num_blocks}) must be >= min_blocks ({self.min_blocks})")
        return self

    @model_validator(mode="after")
    def _validate_shm_namespace(self) -> Self:
        """Require a stable, visible namespace whenever SHM is enabled."""
        if self.shm_enabled and sys.platform == "win32":
            raise ValueError("Shared memory (SHM) is not supported on Windows")
        if self.shm_enabled and (self.shm_namespace is None or not self.shm_namespace.strip()):
            raise ValueError("shm_namespace must be non-empty when shm_enabled=True")
        return self

    @model_validator(mode="after")
    def _validate_profile_mode(self) -> Self:
        """Validate the publicly-observable ``profile_mode`` / ``transfer_mode``
        combinations at config construction so errors fail fast before the
        manager is built. ``TensorManager`` performs additional checks for
        internal flags (e.g. ``_use_trace_tensor``) that aren't expressible on
        ``OffloadConfig``."""
        block_loaders = BLOCK_TRANSFER_MODES
        if self.profile_mode == "torch_function" and self.transfer_mode in block_loaders:
            raise ValueError(
                f"profile_mode='torch_function' is incompatible with "
                f"transfer_mode={self.transfer_mode!r}. Use profile_mode='getter' "
                f"or 'view', or set transfer_mode='strategy'."
            )
        if self.external_compile and self.transfer_mode not in block_loaders:
            raise ValueError(
                f"external_compile=True requires a block transfer_mode "
                f"(allocation_block_transfer or raw_block_transfer); got "
                f"transfer_mode={self.transfer_mode!r}. The compiled-offload "
                f"path installs a PreallocatedLoader for pre_compute/post_compute "
                f"custom ops, which transfer_mode='strategy' does not provide."
            )
        if self.offload_timing != "off" and self.transfer_mode not in block_loaders:
            raise ValueError(
                f"offload_timing={self.offload_timing!r} requires a block transfer_mode "
                f"(allocation_block_transfer or raw_block_transfer); got "
                f"transfer_mode={self.transfer_mode!r}. Inference timing is recorded "
                f"by PreallocatedLoader enter/exit hooks, which "
                f"transfer_mode='strategy' does not install."
            )
        return self

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Return a validated copy while preserving explicit-field tracking."""
        if not update:
            return super().model_copy(deep=deep)

        validated_update = self._reject_removed_fields(dict(update))  # type: ignore[operator]
        copied = super().model_copy(update=validated_update, deep=deep)
        # Revalidate field values without serializing custom strategy objects.
        revalidated = type(self).model_validate(dict(copied))
        object.__setattr__(revalidated, "__pydantic_fields_set__", set(copied.model_fields_set))
        return revalidated


def resolve_load_strategy(config: OffloadConfig) -> Strategy:
    """Return the explicit strategy or a fresh adaptive default."""
    if config.load_strategy is not None:
        return config.load_strategy
    return AdaptiveStrategy(
        scale=config.transfer_budget_scale,
        loader_type=config.transfer_mode,
        n_blocks=config.num_blocks,
        min_blocks=config.min_blocks,
    )


def _resolve_union_field_type(annotation: object) -> type:
    """Pick the env-conversion type for a ``Union`` / PEP 604 annotation.

    Returns the first non-``None`` arm, mapped to one of
    ``bool``/``int``/``float``/``str``/``list``/``object`` so
    :func:`_convert_env_value` knows how to coerce the env-var string.
    """
    args = [a for a in get_args(annotation) if a is not type(None)]
    if not args:
        return str
    first_type = args[0]
    if get_origin(first_type) is Union:
        return object
    if get_origin(first_type) is list:
        return list
    if isinstance(first_type, type) and first_type in (bool, int, float, str):
        return first_type
    return object


def _resolve_field_type(annotation: object) -> type:
    """Pick the env-conversion type for a single OffloadConfig field annotation."""
    if annotation is None:
        return str

    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return _resolve_union_field_type(annotation)
    if origin is list:
        return list
    if origin is Literal:
        # Literal[...] members are always plain string/int values that
        # round-trip through the env-var pipeline as their underlying
        # type. Treat the annotation as the type of its first member so
        # _convert_env_value coerces correctly; Pydantic then validates
        # the coerced value against the Literal members.
        members = get_args(annotation)
        member_type = type(members[0]) if members else str
        return member_type if member_type in (bool, int, float, str) else str
    if isinstance(annotation, type) and annotation in (bool, int, float, str):
        return annotation
    return object


def _get_field_types() -> dict[str, type]:
    """Get field names and their base types from OffloadConfig.

    Returns:
        Dictionary mapping field names to their types.
        For Union/Optional types, returns the first non-None type.
        For list types (e.g., ``list[str]``), returns ``list``.
        For complex types (like strategies), returns object.
    """
    return {name: _resolve_field_type(field_info.annotation) for name, field_info in OffloadConfig.model_fields.items()}


def _parse_bool(value: str) -> bool:
    """Parse a string value to boolean with case-insensitive matching.

    Args:
        value: String value to parse

    Returns:
        Boolean value

    Raises:
        ValueError: If the value cannot be parsed as a boolean
    """
    lower_value = value.lower()
    if lower_value in ("true", "1", "yes", "y", "on"):
        return True
    if lower_value in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"Cannot parse '{value}' as boolean")


def _parse_none(value: str) -> None:
    """Parse a string value to None with case-insensitive matching.

    Args:
        value: String value to parse

    Returns:
        None if the value represents null/none, otherwise raises ValueError

    Raises:
        ValueError: If the value cannot be parsed as None
    """
    lower_value = value.lower()
    if lower_value in ("none", "null"):
        return None
    raise ValueError(f"Cannot parse '{value}' as None")


def _convert_env_value(value: str, field_type: type) -> Any:
    """Convert environment variable string to the appropriate type.

    Args:
        value: String value from environment variable
        field_type: Target type to convert to

    Returns:
        Converted value

    Raises:
        ValueError: If conversion fails
    """
    # Handle None/null values
    try:
        return _parse_none(value)
    except ValueError:
        pass  # Not a None value, continue with other conversions

    # Handle empty strings
    if value == "":
        return None

    # Type-specific conversions
    if field_type is bool:
        return _parse_bool(value)
    elif field_type is int:
        return int(value)
    elif field_type is float:
        return float(value)
    elif field_type is str:
        return value
    else:
        # For complex types (like strategies), return as string and let pydantic handle it
        return value


def _load_ini_file(config_path: Path, field_types: dict[str, type]) -> dict[str, Any]:
    """Load configuration from INI file.

    Args:
        config_path: Path to the INI file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If the file cannot be parsed or section is missing
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)

    section = "flextensor"
    if section not in parser:
        raise ValueError(f"INI file must contain [{section}] section")

    config_dict: dict[str, Any] = {}
    for field_name, field_type in field_types.items():
        if field_name in parser[section]:
            value = parser[section][field_name]
            if field_type is object:
                continue
            if field_type is list:
                config_dict[field_name] = [p.strip() for p in value.split(",") if p.strip()]
            else:
                config_dict[field_name] = _convert_env_value(value, field_type)

    for field in _REMOVED_FIELDS:
        if field in parser[section]:
            config_dict[field] = parser[section][field]

    return config_dict


def _process_data_dict(data: dict[str, Any], field_types: dict[str, type]) -> dict[str, Any]:
    """Process a data dictionary into configuration values.

    Common logic for JSON and YAML file loading that handles field filtering
    and type conversion.

    Args:
        data: Raw data dictionary from file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values
    """
    config_dict: dict[str, Any] = {}
    for field_name in data:
        if field_name in _REMOVED_FIELDS:
            config_dict[field_name] = data[field_name]
            continue

        if field_name not in field_types:
            continue
        field_type = field_types[field_name]
        if field_type is object:
            continue  # skip complex types (Strategy, etc.)
        value = data[field_name]
        if field_type is list:
            config_dict[field_name] = value  # JSON/YAML already has proper list
        elif isinstance(value, str) and field_type is not str:
            config_dict[field_name] = _convert_env_value(value, field_type)
        else:
            config_dict[field_name] = value

    return config_dict


def _load_json_file(config_path: Path, field_types: dict[str, type]) -> dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to the JSON file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values
    """
    with config_path.open() as f:
        data = json.load(f)

    return _process_data_dict(data, field_types)


def _load_yaml_file(config_path: Path, field_types: dict[str, type]) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values
    """
    with config_path.open() as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    return _process_data_dict(data, field_types)


def _load_from_file(config_path: Path, field_types: dict[str, type]) -> dict[str, Any]:
    """Load configuration from file based on extension.

    Args:
        config_path: Path to the configuration file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If the file format is not supported
    """
    suffix = config_path.suffix.lower()
    if suffix in (".conf", ".ini"):
        return _load_ini_file(config_path, field_types)
    elif suffix == ".json":
        return _load_json_file(config_path, field_types)
    elif suffix in (".yaml", ".yml"):
        return _load_yaml_file(config_path, field_types)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}. Use .conf, .ini, .json, .yaml, or .yml")


def _parse_env_list(field_name: str, env_value: str) -> list:
    """Parse a comma-separated env var into a typed list."""
    annotation = OffloadConfig.model_fields[field_name].annotation
    element_type = get_args(annotation)[0] if get_args(annotation) else str
    parts = [p.strip() for p in env_value.split(",") if p.strip()]
    if element_type is str:
        return parts
    return [_convert_env_value(p, element_type) for p in parts]


def _register_env_var(name: str) -> None:
    _REGISTERED_ENV_VARS.add(name)


def _known_env_vars(
    env_prefix: str,
    field_types: dict[str, type],
    config_file_env_var: str,
) -> set[str]:
    known = {f"{env_prefix}{field_name.upper()}" for field_name in field_types}
    known.update(f"{env_prefix}{field_name.upper()}" for field_name in _REMOVED_FIELDS)
    known.add(f"{env_prefix}CONFIG_FILE")
    if config_file_env_var and config_file_env_var.startswith(env_prefix):
        known.add(config_file_env_var)
    known.update(name for name in _REGISTERED_ENV_VARS if name.startswith(env_prefix))
    return known


def _validate_env_vars(
    env_prefix: str,
    field_types: dict[str, type],
    config_file_env_var: str,
) -> None:
    if not env_prefix:
        return

    known = _known_env_vars(env_prefix, field_types, config_file_env_var)
    errors = []
    for env_var_name in sorted(name for name in os.environ if name.startswith(env_prefix) and name not in known):
        message = f"Unrecognized environment variable with {env_prefix} prefix: {env_var_name}"
        suggestions = difflib.get_close_matches(env_var_name, sorted(known), n=1, cutoff=0.75)
        if suggestions:
            message += f". Did you mean: {suggestions[0]}?"
        errors.append(message)
    if errors:
        raise ValueError("\n".join(errors))


def _load_from_env(
    env_prefix: str,
    field_types: dict[str, type],
    config_file_env_var: str,
) -> dict[str, Any]:
    """Load configuration from environment variables.

    Args:
        env_prefix: Prefix for environment variable names
        field_types: Dictionary mapping field names to types
        config_file_env_var: Environment variable used to select the config file

    Returns:
        Dictionary of configuration values
    """
    config_dict: dict[str, Any] = {}
    _validate_env_vars(env_prefix, field_types, config_file_env_var)

    for field, (version, msg) in _REMOVED_FIELDS.items():
        env_var_name = f"{env_prefix}{field.upper()}"
        if os.environ.get(env_var_name) is not None:
            logger.warning(
                "Environment variable '%s' is ignored: '%s' was removed in %s. %s",
                env_var_name,
                field,
                version,
                msg,
            )

    for field_name, field_type in field_types.items():
        if field_type is object:
            continue

        env_var_name = f"{env_prefix}{field_name.upper()}"
        env_value = os.environ.get(env_var_name)
        if env_value is None:
            continue

        try:
            if field_type is list:
                config_dict[field_name] = _parse_env_list(field_name, env_value)
            else:
                config_dict[field_name] = _convert_env_value(env_value, field_type)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert {env_var_name}={env_value}: {e}") from e

    return config_dict


def load_config(
    config_path: str | Path | None = None,
    env_prefix: str = "FT_",
    use_env: bool = True,
    config_file_env_var: str = "FT_CONFIG_FILE",
    **kwargs: Any,
) -> OffloadConfig:
    """Load OffloadConfig with precedence: file < env vars < kwargs.

    This is the unified configuration loading function that combines file and
    environment variable loading with proper precedence.

    Precedence order (highest wins):

    1. kwargs (explicit arguments)
    2. Environment variables (if use_env=True)
    3. Config file values
    4. OffloadConfig defaults

    When loading from environment only (no file), `enabled` defaults to False.
    You must explicitly set FT_ENABLED=1 to enable offloading.

    Supported file formats (auto-detected by extension):

    - INI (.conf, .ini) — requires [flextensor] section
    - JSON (.json)
    - YAML (.yaml, .yml)

    Args:
        config_path: Path to configuration file. If None, checks config_file_env_var.
        env_prefix: Prefix for environment variable names (default: "FT_")
        use_env: Whether to load and override with environment variables (default: True)
        config_file_env_var: Environment variable for config file path (default: "FT_CONFIG_FILE")
        **kwargs: Additional keyword arguments to override all other values

    Returns:
        OffloadConfig instance

    Examples:
        >>> # Full integration: file + env + kwargs
        >>> config = load_config("flextensor.conf")
        >>>
        >>> # With env override: FT_GPU_DEVICE=2 overrides file value
        >>> config = load_config("flextensor.conf")
        >>>
        >>> # Env only (no file, enabled defaults to False)
        >>> config = load_config()
    """
    field_types = _get_field_types()
    config_dict: dict[str, Any] = {}

    # Load from file if path provided
    path_str = config_path or os.environ.get(config_file_env_var)
    if path_str is not None:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        config_dict = _load_from_file(path, field_types)
    elif use_env:
        # No file loaded and using env: default enabled to False
        config_dict["enabled"] = False

    # Override with environment variables
    if use_env:
        config_dict.update(_load_from_env(env_prefix, field_types, config_file_env_var))

    # Override with kwargs (highest precedence)
    config_dict.update(kwargs)

    return OffloadConfig(**config_dict)


def load_config_from_file(
    config_path: str | Path | None = None,
    env_var: str = "FT_CONFIG_FILE",
    **kwargs: Any,
) -> OffloadConfig:
    """Load OffloadConfig from a configuration file (no env var override).

    This is a convenience wrapper around load_config() that loads from file only,
    without environment variable overrides.

    Supported file formats (auto-detected by extension):

    - INI (.conf, .ini) — requires [flextensor] section
    - JSON (.json)
    - YAML (.yaml, .yml)

    Args:
        config_path: Path to configuration file. If None, uses env_var.
        env_var: Environment variable name for config file path (default: "FT_CONFIG_FILE")
        **kwargs: Additional keyword arguments to override file values

    Returns:
        OffloadConfig instance with values from file and kwargs

    Raises:
        FileNotFoundError: If the config file does not exist
        ValueError: If the file format is not supported or file is invalid

    Examples:
        >>> config = load_config_from_file("flextensor.conf")
        >>> config.enabled
        True
    """
    # If no config_path and no env var set, raise helpful error
    if config_path is None and os.environ.get(env_var) is None:
        raise FileNotFoundError(f"No config file specified. Provide config_path or set {env_var} environment variable.")

    return load_config(config_path=config_path, use_env=False, config_file_env_var=env_var, **kwargs)


def load_config_from_env(prefix: str = "FT_", **kwargs: Any) -> OffloadConfig:
    """Load OffloadConfig from environment variables (no file).

    This is a convenience wrapper around load_config() that loads from
    environment variables only, without file loading.

    Environment variables should be named as {prefix}{FIELD_NAME} in uppercase.
    For example, with the default prefix "FT_":

    - FT_ENABLED=1 (required to enable offloading)
    - FT_GPU_DEVICE=1
    - FT_DISCOVERY_ITERS=5

    Note: When loading from environment, `enabled` defaults to False.
    You must explicitly set FT_ENABLED=1 to enable offloading.

    Type conversions:

    - bool: Case-insensitive parsing of "true"/"false", "1"/"0", "yes"/"no", etc.
    - int: Direct integer conversion
    - float: Direct float conversion
    - str: Used as-is
    - None: Case-insensitive "none" or "null"

    Args:
        prefix: Prefix for environment variable names (default: "FT_")
        **kwargs: Additional keyword arguments to override environment variables

    Returns:
        OffloadConfig instance with values from environment variables and kwargs

    Examples:
        >>> import os
        >>> os.environ["FT_ENABLED"] = "1"
        >>> os.environ["FT_GPU_DEVICE"] = "1"
        >>> config = load_config_from_env()
        >>> config.enabled
        True
        >>> config.gpu_device
        1
    """
    return load_config(
        config_path=None,
        env_prefix=prefix,
        use_env=True,
        config_file_env_var="",  # Empty to prevent file loading
        **kwargs,
    )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import math
import pathlib
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import torch

logger = logging.getLogger(__name__)

# AllocationBlock's historical slot-packing policy. A 128-byte boundary keeps
# projected GPU suballocations conservatively aligned for CUDA memory access;
# it is not a CUDA correctness requirement. Changing it also changes planned
# block sizes, so the planner and allocator must share this value.
_DEFAULT_PACKED_TENSOR_ALIGNMENT_BYTES = 128

__all__ = [
    "any_path_matches_pattern",
    "atomic_write_json",
    "calculate_tensor_size",
    "clear_and_delete_tensor",
    "config_field_was_set",
    "delete_tensor",
    "get_class_matched_module_paths",
    "get_module_paths",
    "get_tensor_data",
    "is_dense_layout",
    "matches_any_class_pattern",
    "matches_any_pattern",
    "set_tensor_data",
]

# Prefix marker for class-based matching in include/exclude patterns.
# Example: ``class:SharedExpertMLP`` matches any module whose class name equals
# ``SharedExpertMLP``.  Patterns are tested against both the short class name
# (``type(m).__name__``) and the fully-qualified class name
# (``f"{cls.__module__}.{cls.__qualname__}"``) — a match on either succeeds.
# Glob wildcards (``*``, ``?``) are supported inside the class pattern body; a
# dot in the body effectively opts into FQCN matching since short class names
# never contain dots.
CLASS_PATTERN_PREFIX = "class:"

# Prefix marker for name-based matching.  Optional — a pattern without any prefix
# is also treated as a name pattern.  The explicit ``name:`` form exists for
# symmetry with ``class:`` and to disambiguate a literal module path that starts
# with ``class:`` (unlikely, but possible in principle).
NAME_PATTERN_PREFIX = "name:"


def get_tensor_data(tensor: torch.Tensor) -> torch.Tensor:
    """Read tensor data while bypassing subclass property overrides."""
    return cast("torch.Tensor", cast("Any", torch.Tensor.data).__get__(tensor, type(tensor)))


def set_tensor_data(tensor: torch.Tensor, replacement: torch.Tensor) -> None:
    """Replace tensor data while bypassing subclass property overrides."""
    cast("Any", torch.Tensor.data).__set__(tensor, replacement)


def _torch_dtype_name(dtype: torch.dtype) -> str:
    """Return the canonical JSON representation for a torch dtype."""
    rendered = str(dtype)
    if not rendered.startswith("torch."):
        raise ValueError(f"Unsupported torch dtype representation: {rendered!r}")
    return rendered.removeprefix("torch.")


def _parse_torch_dtype_name(value: Any) -> torch.dtype:
    """Parse a dtype name or PyTorch alias without evaluating arbitrary input."""
    if not isinstance(value, str) or not value:
        raise ValueError("dtype must be a non-empty string")
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported dtype {value!r}")
    return dtype


def _is_non_overlapping_dense(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    """Return whether a non-negative strided layout is packed without overlap or gaps."""
    if math.prod(shape) == 0:
        return True
    expected_stride = 1
    for size, step in sorted(
        ((size, step) for size, step in zip(shape, stride, strict=True) if size > 1),
        key=lambda item: item[1],
    ):
        if step != expected_stride:
            return False
        expected_stride *= size
    return True


def _parse_dense_tensor_layout(shape: Any, stride: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse JSON shape/stride lists describing a packed, non-overlapping tensor."""
    for field_name, value in (("shape", shape), ("stride", stride)):
        if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
            raise ValueError(f"{field_name} must contain non-negative integers")
    shape, stride = tuple(shape), tuple(stride)
    if len(shape) != len(stride):
        raise ValueError("shape/stride rank mismatch")
    if not _is_non_overlapping_dense(shape, stride):
        raise ValueError("tensor layout must be non-overlapping and dense")
    return shape, stride


def _compute_packed_byte_layout(
    sizes: Iterable[int],
    memory_alignment: int = _DEFAULT_PACKED_TENSOR_ALIGNMENT_BYTES,
) -> tuple[list[int], int]:
    """Return tensor offsets and total bytes for aligned allocation-block packing."""
    if type(memory_alignment) is not int or memory_alignment <= 0:
        raise ValueError(f"memory_alignment must be a positive integer, got {memory_alignment!r}")
    offsets: list[int] = []
    cursor = 0
    for size in sizes:
        offsets.append(cursor)
        cursor += size
        cursor = (cursor + memory_alignment - 1) // memory_alignment * memory_alignment
    return offsets, cursor


def config_field_was_set(config: Any, field_name: str) -> bool:
    """Return whether a Pydantic config field was explicitly provided.

    Args:
        config: Pydantic model instance, or compatible object, exposing
            ``model_fields_set`` or the deprecated ``__fields_set__``.
        field_name: Field name to check.

    Returns:
        ``True`` if the field was explicitly provided during model
        construction or copy/update; otherwise ``False``.
    """
    fields_set = getattr(config, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(config, "__fields_set__", None)
    return isinstance(fields_set, (set, frozenset)) and field_name in fields_set


def _compile_segment_regex(segment: str) -> re.Pattern[str]:
    """Compile a single wildcard pattern segment into a regex.

    ``*`` is mapped to ``.*`` and ``?`` to ``.`` so that they behave as
    shell-style glob wildcards within one dot-separated segment.
    """
    return re.compile(re.escape(segment).replace(r"\*", ".*").replace(r"\?", "."))


def get_module_paths(model: torch.nn.Module) -> list[str]:
    """Return dot-separated paths for all non-root modules in *model*."""
    return [p for p, _ in model.named_modules() if p]


def matches_any_pattern(path: str, patterns: list[str], *, recursive_star: bool = True) -> bool:
    """Check if a dot-separated path matches any wildcard pattern.

    Args:
        path: Dot-separated path (e.g., "layers.0.self_attn").
        patterns: List of patterns to match against.
        recursive_star: If True, standalone ``*`` matches 1+ segments.
            If False, standalone ``*`` matches exactly 1 segment.

    Returns:
        True if the path matches any of the patterns.
    """
    path_parts = path.split(".")
    regex_cache: dict[str, re.Pattern[str]] = {}
    for pattern in patterns:
        pattern_parts = pattern.split(".")
        for pp in pattern_parts:
            if pp not in regex_cache:
                regex_cache[pp] = _compile_segment_regex(pp)
        if _match_parts(path_parts, pattern_parts, regex_cache, recursive_star):
            return True
    return False


def any_path_matches_pattern(paths: list[str], pattern: str, *, recursive_star: bool = True) -> bool:
    """Check if *any* path in *paths* matches a single wildcard *pattern*.

    Unlike :func:`matches_any_pattern` (one path, many patterns), this function
    is optimised for the inverse: one pattern tested against many paths.  The
    pattern is compiled once and reused for every path.

    Args:
        paths: Dot-separated paths (e.g., module paths from ``named_modules``).
        pattern: Wildcard pattern to match against.
        recursive_star: If True, standalone ``*`` matches 1+ segments.
            If False, standalone ``*`` matches exactly 1 segment.

    Returns:
        ``True`` if at least one path matches the pattern.
    """
    pattern_parts = pattern.split(".")
    regex_cache: dict[str, re.Pattern[str]] = {}
    for pp in pattern_parts:
        if pp not in regex_cache:
            regex_cache[pp] = _compile_segment_regex(pp)
    return any(_match_parts(path.split("."), pattern_parts, regex_cache, recursive_star) for path in paths)


@dataclass(frozen=True, slots=True)
class PartitionedPatterns:
    """Result of splitting raw patterns by ``class:`` / ``name:`` prefix.

    ``*_bodies`` map raw pattern → body (prefix stripped); keys preserve
    the original pattern text.

    Supports tuple-style unpacking for back-compat:
    ``names, classes = partition_patterns(...)`` yields the body lists.
    """

    name_bodies: Mapping[str, str] = field(default_factory=dict)
    class_bodies: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``frozen=True`` only blocks attribute *rebinding*; without these
        # wraps a caller could still do ``p.name_bodies["x"] = "y"``.
        # Defensively copy first so external mutation of the source dict
        # can't leak through the proxy.
        object.__setattr__(self, "name_bodies", MappingProxyType(dict(self.name_bodies)))
        object.__setattr__(self, "class_bodies", MappingProxyType(dict(self.class_bodies)))

    @property
    def name_patterns(self) -> list[str]:
        return list(self.name_bodies.values())

    @property
    def class_patterns(self) -> list[str]:
        return list(self.class_bodies.values())

    def __iter__(self) -> Iterator[list[str]]:
        yield self.name_patterns
        yield self.class_patterns


def partition_patterns(patterns: list[str]) -> PartitionedPatterns:
    """Split *patterns* by ``class:`` / ``name:`` prefix.

    Internal helper; full pattern validation (typo prefixes, whitespace,
    type guards) lives at the ``OffloadConfig`` boundary. Empty bodies
    (``"class:"`` / ``"name:"``) raise :class:`ValueError` defensively here
    so direct test callers don't get silent zero-match patterns.

    Args:
        patterns: Raw patterns as stored in ``OffloadConfig.include_patterns`` /
            ``exclude_patterns``.

    Returns:
        :class:`PartitionedPatterns`. Tuple-unpackable as ``(name_patterns,
        class_patterns)`` of body strings for back-compat.

    Raises:
        ValueError: If any pattern uses the ``class:`` or ``name:`` prefix with
            an empty (or whitespace-only) body.

    Example:
        >>> names, classes = partition_patterns(["layers.*", "name:embed", "class:MoELayer"])
        >>> names, classes
        (['layers.*', 'embed'], ['MoELayer'])
    """
    name_bodies: dict[str, str] = {}
    class_bodies: dict[str, str] = {}
    for pattern in patterns:
        if pattern.startswith(CLASS_PATTERN_PREFIX):
            body = pattern[len(CLASS_PATTERN_PREFIX) :]
            if not body.strip():
                raise ValueError(
                    f"pattern {pattern!r} has an empty body; use {CLASS_PATTERN_PREFIX}<glob> or remove the entry"
                )
            class_bodies[pattern] = body
        elif pattern.startswith(NAME_PATTERN_PREFIX):
            body = pattern[len(NAME_PATTERN_PREFIX) :]
            if not body.strip():
                raise ValueError(
                    f"pattern {pattern!r} has an empty body; use {NAME_PATTERN_PREFIX}<glob> or remove the entry"
                )
            name_bodies[pattern] = body
        else:
            name_bodies[pattern] = pattern
    return PartitionedPatterns(name_bodies=name_bodies, class_bodies=class_bodies)


def _class_fqcn(cls: type) -> str:
    """Return ``f"{cls.__module__}.{cls.__qualname__}"`` (the fully-qualified class name).

    Used as a secondary haystack for ``class:`` patterns so that users can
    disambiguate classes with the same short name across packages, e.g.
    ``class:torch.nn.modules.linear.Linear`` vs. a user-defined ``Linear``.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def matches_any_class_pattern(cls: type, class_patterns: list[str]) -> bool:
    """Check if *cls* matches any glob pattern in *class_patterns*.

    Each pattern is tested against both the short class name
    (``cls.__name__``) *and* the fully-qualified class name
    (``f"{cls.__module__}.{cls.__qualname__}"``).  A match on either haystack
    is sufficient.  Class names are treated as atomic for the short-name
    check (no dot-split semantics); the FQCN check also treats the pattern as
    a single segment — so ``*`` spans dots.  This lets users write ergonomic
    patterns like::

        class:MoELayer            # short name
        class:*Expert*            # short-name glob
        class:torch.nn.*.Linear   # FQCN glob (disambiguates user ``Linear``)

    Since Python class short names never contain dots, a pattern containing
    a dot can only match via the FQCN haystack — in effect an opt-in escape
    hatch for name collisions without introducing new syntax.

    Args:
        cls: The class to test (typically ``type(module)``).
        class_patterns: Class patterns with the ``class:`` prefix already
            stripped (as produced by :func:`partition_patterns`).

    Returns:
        ``True`` if at least one pattern matches the short name or FQCN.
    """
    if not class_patterns:
        return False
    short_name = cls.__name__
    fqcn = _class_fqcn(cls)
    for pattern in class_patterns:
        regex = _compile_segment_regex(pattern)
        if regex.fullmatch(short_name) or regex.fullmatch(fqcn):
            return True
    return False


def get_class_matched_module_paths(model: Any, class_patterns: list[str]) -> set[str]:
    """Return dot-separated paths of modules whose class matches *class_patterns*.

    Walks ``model.named_modules()`` and collects paths whose owning module's
    class matches any of the patterns.  Matching uses
    :func:`matches_any_class_pattern`, which tests both the short class name
    and the fully-qualified class name.  The root module (empty path) is
    excluded for consistency with :func:`get_module_paths`.

    Args:
        model: A ``torch.nn.Module`` (or any object exposing
            ``named_modules()``).
        class_patterns: Class patterns with the ``class:`` prefix already
            stripped.

    Returns:
        Set of module paths whose class matches at least one pattern.  Empty
        set when ``class_patterns`` is empty.

    Raises:
        TypeError: If ``class_patterns`` is non-empty and ``model`` lacks a
            callable ``named_modules``.
    """
    if not class_patterns:
        return set()
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError(f"class: patterns require a model exposing named_modules(); got {type(model).__name__}.")
    matched: set[str] = set()
    for path, module in named_modules():
        if not path:
            continue
        if matches_any_class_pattern(type(module), class_patterns):
            matched.add(path)
    return matched


def _match_parts(
    path_parts: list[str],
    pattern_parts: list[str],
    regex_cache: dict[str, re.Pattern[str]],
    recursive_star: bool,
) -> bool:
    """Recursive segment matcher.

    Args:
        path_parts: Remaining path segments.
        pattern_parts: Remaining pattern segments.
        regex_cache: Map from pattern segment to pre-compiled regex.
        recursive_star: If True, standalone ``*`` matches 1+ segments.
    """
    if not pattern_parts:
        return not path_parts
    if not path_parts:
        return False

    pp = pattern_parts[0]
    if pp == "*" and recursive_star:
        # Standalone * matches 1 or more path segments
        return any(
            _match_parts(path_parts[i + 1 :], pattern_parts[1:], regex_cache, recursive_star)
            for i in range(len(path_parts))
        )
    else:
        if regex_cache[pp].fullmatch(path_parts[0]):
            return _match_parts(path_parts[1:], pattern_parts[1:], regex_cache, recursive_star)
        return False


def atomic_write_json(file_path: str | pathlib.Path, data: dict[str, Any]) -> None:
    """
    Write JSON data to a file atomically.

    The data is first written to a temporary file in the same directory,
    then atomically renamed to the target path. This ensures the file is
    never partially written in case of a crash.

    Args:
        file_path: Path to the file to write to.
        data: Dictionary to serialize as JSON.

    Example:
        >>> data = {"model": "gpt2", "layers": 12}
        >>> atomic_write_json("/tmp/config.json", data)
    """
    target_path = pathlib.Path(file_path)

    # Create parent directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in the same directory (ensures same filesystem for atomic rename)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=target_path.parent, prefix=f".{target_path.name}.", suffix=".tmp", delete=False
        ) as tmp_file:
            tmp_path = pathlib.Path(tmp_file.name)  # Capture path before potential json.dump error
            json.dump(data, tmp_file, indent=2)

        # Atomic rename (replaces existing file if present)
        tmp_path.replace(target_path)
        tmp_path = None  # Successfully renamed, no cleanup needed
    finally:
        # Clean up temp file if rename failed or exception occurred
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as e:
                logger.warning("Failed to clean up temp file %s: %s", tmp_path, e)


def is_dense_layout(tensor: torch.Tensor) -> bool:
    """Check if tensor elements occupy a contiguous memory block with no gaps.

    True for C-contiguous, Fortran-contiguous, and any permutation-contiguous
    layout where all ``numel`` elements are packed without holes.
    False for sliced/strided views with gaps between elements.

    Args:
        tensor: The tensor to check.

    Returns:
        True if the tensor's elements are densely packed in memory.

    Example:
        >>> import torch
        >>> is_dense_layout(torch.randn(3, 4))
        True
        >>> is_dense_layout(torch.randn(4, 3).t())
        True
        >>> is_dense_layout(torch.randn(4, 5)[:, ::2])
        False
    """
    if tensor.is_contiguous() or tensor.numel() <= 1:
        return True
    max_offset = sum((s - 1) * abs(st) for s, st in zip(tensor.shape, tensor.stride(), strict=False))
    return max_offset + 1 == tensor.numel()


def calculate_tensor_size(tensor: torch.Tensor) -> int:
    """
    Calculate the size of a tensor in bytes.

    Args:
        tensor: The tensor to calculate the size of

    Returns:
        The size of the tensor in bytes
    """
    numel = tensor.numel()
    element_size = tensor.element_size()
    return numel * element_size


def delete_tensor(tensor: torch.Tensor) -> None:
    """
    Delete a tensor from memory.

    Args:
        tensor: The tensor to be deleted from memory.
    """
    del tensor


def clear_and_delete_tensor(tensor: torch.Tensor) -> None:
    """
    Clear a tensor's storage and then delete it from memory.

    This function first resizes the tensor's underlying storage to 0,
    effectively clearing all data, then deletes the tensor reference.
    This can be useful for more thorough memory cleanup.

    Note: This provides more aggressive memory cleanup for cases where standard deletion
    is insufficient, particularly useful during memory profiling or debugging.

    Args:
        tensor: The tensor to be cleared and deleted from memory.
    """
    tensor.untyped_storage().resize_(0)
    del tensor

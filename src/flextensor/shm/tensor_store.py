# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resident tensor storage backed by shared memory.

The offload allocation blocks already live in shared memory.  This store covers
the complementary set: parameters that intentionally remain resident plus all
registered model buffers.  A creator publishes one compact, versioned block;
followers validate its manifest and copy tensors to their own target device.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable  # noqa: TC003 - beartype resolves public method hints at runtime.
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, TypeGuard

import torch
from torch import nn

from flextensor.shm.flexible_shm import FlexibleSharedMemory
from flextensor.utils import (
    _is_non_overlapping_dense,
    _parse_dense_tensor_layout,
    _parse_torch_dtype_name,
    _torch_dtype_name,
)

_MAGIC = b"FTTS"
_VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sIQQQ")
_DATA_ALIGNMENT = 64
_PAYLOAD_ALIGNMENT = 8
_REJECTED_STORE_ATTR = "_flextensor_rejected_shared_tensor_store"


def _align(value: int, alignment: int) -> int:
    """Round *value* up to the next *alignment* boundary."""
    return (value + alignment - 1) // alignment * alignment


def _tensor_store_name(namespace: str) -> str:
    """Return the compact shared-memory name for a resident tensor store."""
    return f"{namespace}_t"


def _is_plain_int(value: Any) -> TypeGuard[int]:
    """Return whether *value* is an integer but not a JSON boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _numel(shape: tuple[int, ...]) -> int:
    """Compute the logical element count for a tensor shape."""
    return math.prod(shape)


def _storage_elements(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    """Return storage elements needed for a zero-offset, non-negative strided view."""
    if _numel(shape) == 0:
        return 0
    return 1 + sum((size - 1) * step for size, step in zip(shape, stride, strict=True))


def _registered_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    """Resolve a registered parameter or buffer name to its owning module."""
    if "." not in qualified_name:
        return model, qualified_name
    module_name, leaf_name = qualified_name.rsplit(".", 1)
    return model.get_submodule(module_name), leaf_name


def _buffer_is_persistent(model: nn.Module, qualified_name: str) -> bool:
    """Return the persistence flag for a registered buffer alias."""
    parent, leaf_name = _registered_parent(model, qualified_name)
    return leaf_name not in parent._non_persistent_buffers_set  # noqa: SLF001


def _planned_tensor_layout(tensor: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Plan a normalized SHM layout without materializing payload bytes."""
    if tensor.layout is not torch.strided:
        raise ValueError(f"SharedTensorStore only supports strided tensors, got {tensor.layout}")

    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    if any(step < 0 for step in stride):
        raise ValueError("SharedTensorStore does not support negative tensor strides")

    if _numel(shape) == 0:
        return shape, stride, 0

    if not _is_non_overlapping_dense(shape, stride):
        next_stride = 1
        contiguous_stride = [0] * len(shape)
        for dimension in range(len(shape) - 1, -1, -1):
            contiguous_stride[dimension] = next_stride
            next_stride *= shape[dimension]
        stride = tuple(contiguous_stride)

    byte_length = _storage_elements(shape, stride) * tensor.element_size()
    return shape, stride, byte_length


@dataclass(frozen=True)
class _TensorDescriptor:
    """Validated metadata for one unique parameter or buffer object."""

    kind: str
    aliases: tuple[str, ...]
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    payload_offset: int
    byte_length: int
    requires_grad: bool
    persistent: tuple[bool, ...] | None

    def to_manifest(self) -> dict[str, Any]:
        """Convert the descriptor to its JSON-compatible representation."""
        return {
            "aliases": list(self.aliases),
            "dtype": _torch_dtype_name(self.dtype),
            "kind": self.kind,
            "payload_offset": self.payload_offset,
            "persistent": None if self.persistent is None else list(self.persistent),
            "requires_grad": self.requires_grad,
            "shape": list(self.shape),
            "stride": list(self.stride),
        }


@dataclass(frozen=True)
class _TensorPlan:
    """A descriptor paired with its source tensor for direct SHM publication."""

    descriptor: _TensorDescriptor
    source: torch.Tensor


def _collect_source_tensors(  # noqa: C901 - preserves registration and alias ordering while validating inputs.
    model: nn.Module,
    parameter_names: Iterable[str],
) -> list[tuple[str, tuple[str, ...], torch.Tensor, tuple[bool, ...] | None]]:
    """Select resident parameters and every non-None registered buffer."""
    parameters = dict(model.named_parameters(recurse=True, remove_duplicate=False))
    buffers = dict(model.named_buffers(recurse=True, remove_duplicate=False))
    requested = list(parameter_names)
    if any(not isinstance(name, str) or not name for name in requested):
        raise ValueError("SharedTensorStore parameter names must be non-empty strings")

    selected_parameter_objects: dict[int, torch.Tensor] = {}
    for name in requested:
        parameter = parameters.get(name)
        if parameter is None:
            object_kind = "buffer" if name in buffers else "missing"
            raise ValueError(
                f"SharedTensorStore requested parameter {name!r} is not a registered non-None parameter ({object_kind})"
            )
        tensor_id = id(parameter)
        if tensor_id not in selected_parameter_objects:
            selected_parameter_objects[tensor_id] = parameter

    parameter_aliases: dict[int, list[str]] = {tensor_id: [] for tensor_id in selected_parameter_objects}
    for name, parameter in parameters.items():
        if id(parameter) in parameter_aliases:
            parameter_aliases[id(parameter)].append(name)

    result: list[tuple[str, tuple[str, ...], torch.Tensor, tuple[bool, ...] | None]] = []
    for tensor_id, parameter in selected_parameter_objects.items():
        result.append((
            "parameter",
            tuple(parameter_aliases[tensor_id]),
            parameter,
            None,
        ))

    buffer_objects: dict[int, torch.Tensor] = {}
    buffer_aliases: dict[int, list[str]] = {}
    for name, buffer in buffers.items():
        tensor_id = id(buffer)
        if tensor_id not in buffer_objects:
            buffer_objects[tensor_id] = buffer
            buffer_aliases[tensor_id] = []
        buffer_aliases[tensor_id].append(name)

    for tensor_id, buffer in buffer_objects.items():
        if tensor_id in selected_parameter_objects:
            raise ValueError("SharedTensorStore cannot encode one object as both a parameter and a buffer")
        aliases = tuple(buffer_aliases[tensor_id])
        result.append((
            "buffer",
            aliases,
            buffer,
            tuple(_buffer_is_persistent(model, alias) for alias in aliases),
        ))
    return result


def _plan_store(
    model: nn.Module,
    parameter_names: Iterable[str],
) -> tuple[tuple[_TensorPlan, ...], bytes, int, int]:
    """Plan metadata and offsets without retaining any tensor payload copies."""
    plans: list[_TensorPlan] = []
    next_payload_offset = 0

    for kind, aliases, tensor, persistent in _collect_source_tensors(model, parameter_names):
        shape, stride, byte_length = _planned_tensor_layout(tensor)
        alignment = max(_PAYLOAD_ALIGNMENT, tensor.element_size())
        payload_offset = _align(next_payload_offset, alignment)
        plans.append(
            _TensorPlan(
                descriptor=_TensorDescriptor(
                    kind=kind,
                    aliases=aliases,
                    dtype=tensor.dtype,
                    shape=shape,
                    stride=stride,
                    payload_offset=payload_offset,
                    byte_length=byte_length,
                    requires_grad=bool(tensor.requires_grad) if kind == "parameter" else False,
                    persistent=persistent,
                ),
                source=tensor,
            )
        )
        next_payload_offset = payload_offset + byte_length

    manifest = {"tensors": [plan.descriptor.to_manifest() for plan in plans]}
    encoded_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    data_offset = _align(_HEADER_STRUCT.size + len(encoded_manifest), _DATA_ALIGNMENT)
    data_size = next_payload_offset
    return tuple(plans), encoded_manifest, data_offset, data_size


def _publish_store(
    shm: FlexibleSharedMemory,
    plans: tuple[_TensorPlan, ...],
    encoded_manifest: bytes,
    data_offset: int,
    data_size: int,
) -> None:
    """Publish metadata and stream each source tensor directly into its final SHM range."""
    block = shm.block
    if block is None or block.buf is None:
        raise RuntimeError("SharedTensorStore creator has no SHM block")
    # Own a child view so releasing it never invalidates SharedMemory's
    # process-local root ``buf`` view.
    buffer = memoryview(block.buf)
    try:
        _HEADER_STRUCT.pack_into(buffer, 0, _MAGIC, _VERSION, len(encoded_manifest), data_offset, data_size)
        buffer[_HEADER_STRUCT.size : _HEADER_STRUCT.size + len(encoded_manifest)] = encoded_manifest

        for plan in plans:
            descriptor = plan.descriptor
            if descriptor.byte_length == 0:
                continue
            byte_view = torch.frombuffer(
                buffer,
                dtype=torch.uint8,
                count=descriptor.byte_length,
                offset=data_offset + descriptor.payload_offset,
            )
            try:
                typed_storage = byte_view.view(descriptor.dtype)
                try:
                    destination = torch.as_strided(
                        typed_storage,
                        descriptor.shape,
                        descriptor.stride,
                        storage_offset=0,
                    )
                    try:
                        destination.copy_(plan.source.detach())
                    finally:
                        del destination
                finally:
                    del typed_storage
            finally:
                del byte_view
    finally:
        try:
            buffer.release()
        finally:
            del buffer


def _parse_descriptor(  # noqa: C901 - each branch validates one untrusted manifest field.
    value: Any,
    *,
    data_size: int,
    seen_aliases: set[str],
) -> _TensorDescriptor:
    """Validate one JSON tensor descriptor and its payload bounds."""
    if not isinstance(value, dict):
        raise RuntimeError("SharedTensorStore manifest tensor descriptor must be an object")

    kind = value.get("kind")
    if kind not in {"parameter", "buffer"}:
        raise RuntimeError(f"SharedTensorStore manifest contains invalid tensor kind {kind!r}")
    aliases_value = value.get("aliases")
    if (
        not isinstance(aliases_value, list)
        or not aliases_value
        or any(not isinstance(alias, str) or not alias for alias in aliases_value)
    ):
        raise RuntimeError("SharedTensorStore manifest aliases must be non-empty strings")
    aliases = tuple(aliases_value)
    if len(set(aliases)) != len(aliases):
        raise RuntimeError("SharedTensorStore manifest aliases must be unique")
    duplicate_aliases = seen_aliases.intersection(aliases)
    if duplicate_aliases:
        raise RuntimeError(f"SharedTensorStore manifest repeats aliases {sorted(duplicate_aliases)!r}")
    seen_aliases.update(aliases)

    try:
        dtype = _parse_torch_dtype_name(value.get("dtype"))
        shape, stride = _parse_dense_tensor_layout(value.get("shape"), value.get("stride"))
    except ValueError as error:
        raise RuntimeError(f"SharedTensorStore manifest: {error}") from error

    payload_offset = value.get("payload_offset")
    if not _is_plain_int(payload_offset) or payload_offset < 0:
        raise RuntimeError("SharedTensorStore manifest payload offset is invalid")
    element_size = torch.empty((), dtype=dtype).element_size()
    byte_length = _storage_elements(shape, stride) * element_size
    if payload_offset + byte_length > data_size:
        raise RuntimeError("SharedTensorStore payload bounds exceed the declared data region")
    if payload_offset % element_size:
        raise RuntimeError("SharedTensorStore payload offset is not dtype-aligned")

    requires_grad = value.get("requires_grad")
    if not isinstance(requires_grad, bool):
        raise RuntimeError("SharedTensorStore manifest requires_grad must be boolean")
    persistent_value = value.get("persistent")
    if kind == "parameter":
        if persistent_value is not None:
            raise RuntimeError("SharedTensorStore parameter descriptor cannot define buffer persistence")
        persistent = None
    else:
        if (
            not isinstance(persistent_value, list)
            or len(persistent_value) != len(aliases)
            or any(not isinstance(flag, bool) for flag in persistent_value)
        ):
            raise RuntimeError("SharedTensorStore buffer persistence must match its aliases")
        persistent = tuple(persistent_value)

    return _TensorDescriptor(
        kind=kind,
        aliases=aliases,
        dtype=dtype,
        shape=shape,
        stride=stride,
        payload_offset=payload_offset,
        byte_length=byte_length,
        requires_grad=requires_grad,
        persistent=persistent,
    )


def _read_store_manifest(
    buffer: memoryview,
) -> tuple[tuple[_TensorDescriptor, ...], int]:
    """Read and validate only the fixed header and JSON manifest."""
    segment_size = len(buffer)
    if segment_size < _HEADER_STRUCT.size:
        raise RuntimeError("SharedTensorStore header bounds exceed the SHM segment")
    magic, version, manifest_length, data_offset, data_size = _HEADER_STRUCT.unpack_from(buffer, 0)
    if magic != _MAGIC:
        raise RuntimeError(f"SharedTensorStore magic mismatch: expected {_MAGIC!r}, got {magic!r}")
    if version != _VERSION:
        raise RuntimeError(f"SharedTensorStore version mismatch: expected {_VERSION}, got {version}")

    manifest_start = _HEADER_STRUCT.size
    manifest_end = manifest_start + manifest_length
    data_end = data_offset + data_size
    # POSIX implementations may round a requested SHM allocation up to a
    # page. The header describes the logical payload extent, so trailing bytes
    # in the physical mapping are valid; a declared extent beyond it is not.
    if (
        manifest_length == 0
        or manifest_end > segment_size
        or data_offset < manifest_end
        or data_offset % _DATA_ALIGNMENT
        or data_end > segment_size
    ):
        raise RuntimeError("SharedTensorStore header bounds are invalid for the SHM segment")

    try:
        manifest = json.loads(bytes(buffer[manifest_start:manifest_end]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("SharedTensorStore manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"tensors"} or not isinstance(manifest["tensors"], list):
        raise RuntimeError("SharedTensorStore manifest must contain a tensor descriptor list")

    seen_aliases: set[str] = set()
    descriptors = tuple(
        _parse_descriptor(value, data_size=data_size, seen_aliases=seen_aliases) for value in manifest["tensors"]
    )
    occupied_ranges = sorted(
        (descriptor.payload_offset, descriptor.payload_offset + descriptor.byte_length)
        for descriptor in descriptors
        if descriptor.byte_length
    )
    for previous, current in pairwise(occupied_ranges):
        if current[0] < previous[1]:
            raise RuntimeError("SharedTensorStore payload bounds overlap between descriptors")
    return descriptors, data_offset


class SharedTensorStore:
    """Share resident model tensors between serving processes.

    Instances own one :class:`FlexibleSharedMemory` handle.  Materialization
    always copies out of that mapping, so closing the store cannot invalidate
    tensors already installed in a model.
    """

    def __init__(
        self,
        shm: FlexibleSharedMemory,
        descriptors: tuple[_TensorDescriptor, ...],
        data_offset: int,
    ) -> None:
        self._shm: FlexibleSharedMemory | None = shm
        self._descriptors = descriptors
        self._data_offset = data_offset

    @classmethod
    def create_from_model(
        cls,
        namespace: str,
        model: nn.Module,
        *,
        parameter_names: Iterable[str],
        shm_owner: FlexibleSharedMemory | None = None,
    ) -> SharedTensorStore:
        """Create and fully publish a resident tensor store from *model*."""
        plans, encoded_manifest, data_offset, data_size = _plan_store(model, parameter_names)
        descriptors = tuple(plan.descriptor for plan in plans)
        name = _tensor_store_name(namespace)

        def initialize(shm: FlexibleSharedMemory) -> None:
            _publish_store(shm, plans, encoded_manifest, data_offset, data_size)

        shm = FlexibleSharedMemory(
            name=name,
            shm_size=data_offset + data_size,
            _creator_initializer=initialize,
            _owner=shm_owner,
        )
        store = cls(shm, descriptors, data_offset)
        if not shm.shm_creator:
            error = FileExistsError(f"SharedTensorStore block {name!r} already exists")
            cls._close_rejected(store, error)
            raise error
        return store

    @classmethod
    def attach(cls, namespace: str, *, shm_owner: FlexibleSharedMemory | None = None) -> SharedTensorStore:
        """Attach to and validate an existing resident tensor store."""
        name = _tensor_store_name(namespace)
        shm = FlexibleSharedMemory(name=name, shm_size=0, _owner=shm_owner)
        try:
            block = shm.block
            if block is None or block.buf is None:
                raise RuntimeError("SharedTensorStore follower has no SHM block")
            buffer = memoryview(block.buf)
            try:
                descriptors, data_offset = _read_store_manifest(buffer)
            finally:
                buffer.release()
            return cls(shm, descriptors, data_offset)
        except BaseException as error:
            cls._close_rejected(cls(shm, (), 0), error)
            raise

    @classmethod
    def _close_rejected(cls, store: SharedTensorStore, error: BaseException) -> None:
        """Close a rejected handle or leave it on the primary error for its caller."""
        try:
            store.close()
        except BaseException as cleanup_error:
            vars(error)[_REJECTED_STORE_ATTR] = store
            if add_note := getattr(error, "add_note", None):
                add_note(f"SharedTensorStore cleanup also failed and was retained for retry: {cleanup_error!r}")

    @classmethod
    def _take_rejected(cls, error: BaseException) -> SharedTensorStore | None:
        """Transfer a rejected handle retained by :meth:`_close_rejected`."""
        store = vars(error).pop(_REJECTED_STORE_ATTR, None)
        return store if isinstance(store, cls) else None

    def _validate_target(
        self,
        model: nn.Module,
    ) -> None:
        """Validate every stored name before materializing any tensor."""
        parameters = dict(model.named_parameters(recurse=True, remove_duplicate=False))
        buffers = dict(model.named_buffers(recurse=True, remove_duplicate=False))

        for descriptor in self._descriptors:
            target_map = parameters if descriptor.kind == "parameter" else buffers
            wrong_kind_map = buffers if descriptor.kind == "parameter" else parameters
            for alias in descriptor.aliases:
                target = target_map.get(alias)
                if target is None:
                    detail = "wrong kind" if alias in wrong_kind_map else "missing"
                    raise ValueError(f"SharedTensorStore target {descriptor.kind} {alias!r} is {detail}")
                if tuple(target.shape) != descriptor.shape:
                    raise ValueError(
                        f"SharedTensorStore target {alias!r} shape mismatch: "
                        f"expected {descriptor.shape}, got {tuple(target.shape)}"
                    )
                # Meta skeletons describe topology, not necessarily checkpoint
                # storage dtypes. Builders commonly create every placeholder in
                # one runtime dtype even when the checkpoint retains selected
                # parameters or buffers in float32. The store replaces meta
                # targets below, so its manifest is authoritative there. Keep
                # concrete targets strict to catch genuinely incompatible
                # creator/follower models before any mutation.
                if not target.is_meta and target.dtype != descriptor.dtype:
                    raise ValueError(
                        f"SharedTensorStore target {alias!r} dtype mismatch: "
                        f"expected {descriptor.dtype}, got {target.dtype}"
                    )

    def _copy_tensor(
        self,
        buffer: memoryview,
        descriptor: _TensorDescriptor,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Copy one validated payload into independent target-device storage."""
        if descriptor.byte_length == 0:
            return torch.empty_strided(
                descriptor.shape,
                descriptor.stride,
                dtype=descriptor.dtype,
                device=device,
            )

        byte_view = torch.frombuffer(
            buffer,
            dtype=torch.uint8,
            count=descriptor.byte_length,
            offset=self._data_offset + descriptor.payload_offset,
        )
        try:
            typed_storage = byte_view.view(descriptor.dtype)
            try:
                shared_view = torch.as_strided(
                    typed_storage,
                    descriptor.shape,
                    descriptor.stride,
                    storage_offset=0,
                )
                try:
                    return shared_view.to(device=device, copy=True)
                finally:
                    del shared_view
            finally:
                del typed_storage
        finally:
            del byte_view

    def materialize_into(
        self,
        model: nn.Module,
        *,
        device: torch.device | str,
    ) -> None:
        """Validate, copy, and rebind stored tensors into *model*.

        All descriptors and all target registrations are validated before the
        first model attribute changes.
        """
        shm = self._shm
        if shm is None:
            raise RuntimeError("SharedTensorStore is closed")
        self._validate_target(model)

        block = shm.block
        if block is None or block.buf is None:
            raise RuntimeError("SharedTensorStore SHM block is not initialized")
        replacements: list[tuple[_TensorDescriptor, torch.Tensor]] = []
        buffer = memoryview(block.buf)
        try:
            for descriptor in self._descriptors:
                copied = self._copy_tensor(buffer, descriptor, device)
                if descriptor.kind == "parameter":
                    copied = nn.Parameter(copied, requires_grad=descriptor.requires_grad)
                replacements.append((descriptor, copied))
        finally:
            buffer.release()

        for descriptor, replacement in replacements:
            for alias_index, alias in enumerate(descriptor.aliases):
                parent, leaf_name = _registered_parent(model, alias)
                setattr(parent, leaf_name, replacement)
                if descriptor.kind == "buffer":
                    if descriptor.persistent is None:
                        raise RuntimeError("SharedTensorStore buffer descriptor lost its persistence metadata")
                    if descriptor.persistent[alias_index]:
                        parent._non_persistent_buffers_set.discard(leaf_name)  # noqa: SLF001
                    else:
                        parent._non_persistent_buffers_set.add(leaf_name)  # noqa: SLF001

    def close(self) -> None:
        """Close the FlexibleSharedMemory handle."""
        shm = self._shm
        if shm is None:
            return
        shm.close()
        self._shm = None
        self._descriptors = ()

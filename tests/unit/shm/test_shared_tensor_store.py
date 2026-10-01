# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable tests for the resident tensor shared-memory store."""

from __future__ import annotations

import json
import struct
import tracemalloc
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

import flextensor.shm.tensor_store as tensor_store_module
from flextensor.shm.flexible_shm import FlexibleSharedMemory
from flextensor.shm.tensor_store import SharedTensorStore


class _FakeFlexibleSharedMemory(FlexibleSharedMemory):
    """In-process stand-in that preserves FlexibleSharedMemory's create/attach contract."""

    _segments: ClassVar[dict[str, bytearray]] = {}
    _references: ClassVar[dict[str, int]] = {}

    def __init__(
        self,
        name: str,
        shm_size: int,
        *,
        pinned_memory: bool = False,
        _creator_initializer=None,
        **_kwargs,
    ) -> None:
        del pinned_memory
        self.name = name
        self._closed = False
        if shm_size == 0:
            if name not in self._segments:
                raise FileNotFoundError(name)
            self.shm_creator = False
        elif name in self._segments:
            self.shm_creator = False
        else:
            self._segments[name] = bytearray(shm_size)
            self._references[name] = 0
            self.shm_creator = True

        self._references[name] += 1
        self._block = SimpleNamespace(buf=memoryview(self._segments[name]))
        if self.shm_creator and _creator_initializer is not None:
            _creator_initializer(self)

    @property
    def block(self):
        return self._block

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._references[self.name] -= 1
        if self._references[self.name] == 0:
            del self._references[self.name]
            del self._segments[self.name]


class _DynamicBufferBlock:
    """Expose a fresh memoryview so the fake itself retains no exported view."""

    def __init__(self, segment: bytearray) -> None:
        self._segment = segment

    @property
    def buf(self) -> memoryview:
        return memoryview(self._segment)


class _CloseCheckingFlexibleSharedMemory(_FakeFlexibleSharedMemory):
    """Follower double whose final close rejects traceback-held tensor views."""

    def close(self) -> None:
        if self._closed:
            return
        self._block.buf.release()
        self._references[self.name] -= 1
        if self._references[self.name] == 0:
            self._segments[self.name].clear()
            del self._references[self.name]
            del self._segments[self.name]
        self._closed = True


class _RetryableCloseFlexibleSharedMemory(_FakeFlexibleSharedMemory):
    """Handle double that leaves ownership intact when close fails."""

    close_failures: ClassVar[int] = 0

    def close(self) -> None:
        if type(self).close_failures:
            type(self).close_failures -= 1
            raise RuntimeError("SHM cleanup failed")
        super().close()


class _RollbackCheckingFlexibleSharedMemory(FlexibleSharedMemory):
    """Constructor double whose rollback detects live exported buffer views."""

    rollback_closed_and_unlinked: ClassVar[bool] = False
    rollback_error: ClassVar[BufferError | None] = None

    def __init__(self, name: str, shm_size: int, *, _creator_initializer=None, **_kwargs) -> None:
        self.name = name
        self.shm_creator = True
        self._segment = bytearray(shm_size)
        try:
            assert _creator_initializer is not None
            _creator_initializer(self)
        except BaseException:
            try:
                self._segment.clear()
            except BufferError as error:
                type(self).rollback_error = error
            else:
                type(self).rollback_closed_and_unlinked = True
            raise

    @property
    def block(self):
        return _DynamicBufferBlock(self._segment)


@pytest.fixture(autouse=True)
def fake_flexible_shm(monkeypatch):
    """Keep store tests portable on hosts where POSIX shared memory is unavailable."""
    _FakeFlexibleSharedMemory._segments.clear()
    _FakeFlexibleSharedMemory._references.clear()
    _RetryableCloseFlexibleSharedMemory.close_failures = 0
    monkeypatch.setattr(
        "flextensor.shm.tensor_store.FlexibleSharedMemory",
        _FakeFlexibleSharedMemory,
        raising=False,
    )
    yield
    _FakeFlexibleSharedMemory._segments.clear()
    _FakeFlexibleSharedMemory._references.clear()
    _RetryableCloseFlexibleSharedMemory.close_failures = 0


class _MixedModel(nn.Module):
    def __init__(self, *, device: str = "cpu", values: bool = True) -> None:
        super().__init__()
        weight = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
        scalar = torch.tensor(7.5, dtype=torch.float64, device=device)
        counter = torch.tensor([3, 5], dtype=torch.int64, device=device)
        scratch = torch.tensor([1.25, 2.5], dtype=torch.float16, device=device)
        if not values and device != "meta":
            weight.zero_()
            scalar.zero_()
            counter.zero_()
            scratch.zero_()
        self.weight = nn.Parameter(weight)
        self.scalar = nn.Parameter(scalar)
        self.empty = nn.Parameter(torch.empty((0, 2), dtype=torch.float32, device=device))
        self.offloaded = nn.Parameter(torch.ones(2, dtype=torch.float32, device=device))
        self.register_buffer("counter", counter)
        self.register_buffer("scratch", scratch, persistent=False)
        self.register_buffer("optional", None)
        self.unregistered = torch.tensor([99], device=device)


class _AliasModel(nn.Module):
    def __init__(self, *, tied: bool = True, device: str = "cpu") -> None:
        super().__init__()
        left = nn.Parameter(torch.arange(4, dtype=torch.float32, device=device).reshape(2, 2))
        self.left = left
        self.right = left if tied else nn.Parameter(torch.zeros((2, 2), dtype=torch.float32, device=device))
        first = torch.tensor([4, 8], dtype=torch.int32, device=device)
        self.register_buffer("first", first)
        self.register_buffer(
            "second",
            first if tied else torch.zeros(2, dtype=torch.int32, device=device),
        )


_HEADER = tensor_store_module._HEADER_STRUCT  # noqa: SLF001


def _store_name(namespace: str) -> str:
    return tensor_store_module._tensor_store_name(namespace)  # noqa: SLF001


def _raw_segment(namespace: str) -> bytearray:
    return _FakeFlexibleSharedMemory._segments[_store_name(namespace)]


def _manifest(namespace: str) -> dict:
    raw = _raw_segment(namespace)
    _magic, _version, manifest_length, _data_offset, _data_size = _HEADER.unpack_from(raw)
    return json.loads(bytes(raw[_HEADER.size : _HEADER.size + manifest_length]))


def test_mixed_tensors_roundtrip_and_copy_out_of_store() -> None:
    """Named parameters and all registered buffers survive creator close after copy-out."""
    source = _MixedModel()
    creator = SharedTensorStore.create_from_model(
        "ft_mixed",
        source,
        parameter_names=["weight", "scalar", "empty"],
    )
    follower = SharedTensorStore.attach("ft_mixed")
    target = _MixedModel(device="meta")

    follower.materialize_into(target, device="cpu")
    follower.close()
    creator.close()

    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.scalar, source.scalar)
    assert target.empty.shape == (0, 2)
    torch.testing.assert_close(target.counter, source.counter)
    torch.testing.assert_close(target.scratch, source.scratch)
    assert "scratch" in target._non_persistent_buffers_set
    assert target.offloaded.device.type == "meta"
    assert target.unregistered.device.type == "meta"


def test_manifest_reader_accepts_platform_rounded_trailing_bytes() -> None:
    """The logical payload may end before an OS page-rounded SHM mapping."""
    creator = SharedTensorStore.create_from_model("ft_page_padding", _MixedModel(), parameter_names=["weight"])
    try:
        logical_segment = bytes(_raw_segment("ft_page_padding"))
        padded_segment = memoryview(logical_segment + bytes(4096))
        try:
            descriptors, data_offset = tensor_store_module._read_store_manifest(padded_segment)  # noqa: SLF001
        finally:
            padded_segment.release()

        assert data_offset < len(logical_segment)
        assert {descriptor.aliases[0] for descriptor in descriptors} == {"weight", "counter", "scratch"}
    finally:
        creator.close()


def test_manifest_dtype_replaces_provisional_meta_dtype() -> None:
    """Checkpoint metadata, not a meta skeleton's default dtype, controls materialization."""
    source = _MixedModel()
    creator = SharedTensorStore.create_from_model("ft_meta_dtype", source, parameter_names=["weight"])
    follower = SharedTensorStore.attach("ft_meta_dtype")
    target = _MixedModel(device="meta")
    target.weight = nn.Parameter(torch.empty(source.weight.shape, dtype=torch.bfloat16, device="meta"))

    follower.materialize_into(target, device="cpu")
    follower.close()
    creator.close()

    assert target.weight.dtype == torch.float32
    torch.testing.assert_close(target.weight, source.weight)


@pytest.mark.parametrize("dtype_name", ["float32", "float", "", "bogus", "Tensor", None, False, 42])
def test_manifest_dtype_validation_preserves_alias_compatibility(dtype_name: object) -> None:
    source = _MixedModel()
    creator = SharedTensorStore.create_from_model("ft_dtype", source, parameter_names=["weight"])
    raw = _raw_segment("ft_dtype")
    magic, version, _manifest_length, data_offset, data_size = _HEADER.unpack_from(raw)
    manifest = _manifest("ft_dtype")
    manifest["tensors"][0]["dtype"] = dtype_name
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert len(encoded) <= data_offset - _HEADER.size
    raw[_HEADER.size : _HEADER.size + len(encoded)] = encoded
    _HEADER.pack_into(raw, 0, magic, version, len(encoded), data_offset, data_size)

    try:
        if dtype_name in {"float32", "float"}:
            follower = SharedTensorStore.attach("ft_dtype")
            target = _MixedModel(device="meta")
            try:
                follower.materialize_into(target, device="cpu")
                torch.testing.assert_close(target.weight, source.weight)
            finally:
                follower.close()
        else:
            with pytest.raises(RuntimeError, match=r"manifest.*dtype"):
                SharedTensorStore.attach("ft_dtype")
        assert _FakeFlexibleSharedMemory._references[_store_name("ft_dtype")] == 1
    finally:
        creator.close()


def test_creator_streams_payload_without_full_blob_amplification() -> None:
    """Creator avoids full payload copies in both Python and native tensor allocations."""
    source = nn.Module()
    source.weight = nn.Parameter(torch.arange(1_000_000, dtype=torch.float32))
    payload_size = source.weight.numel() * source.weight.element_size()

    creator = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        tracemalloc.start()
        try:
            creator = SharedTensorStore.create_from_model("ft_stream", source, parameter_names=["weight"])
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
            if creator is not None:
                creator.close()

    assert peak < payload_size * 2
    for event in profile.events():
        assert event.cpu_memory_usage < payload_size, event.name


def test_failed_payload_copy_releases_views_before_constructor_rollback(monkeypatch) -> None:
    """A copy exception cannot retain SHM exports that block constructor rollback."""
    source = nn.Module()
    source.weight = nn.Parameter(torch.empty(8, dtype=torch.float32, device="meta"))
    _RollbackCheckingFlexibleSharedMemory.rollback_closed_and_unlinked = False
    _RollbackCheckingFlexibleSharedMemory.rollback_error = None
    monkeypatch.setattr(
        "flextensor.shm.tensor_store.FlexibleSharedMemory",
        _RollbackCheckingFlexibleSharedMemory,
    )

    with pytest.raises(NotImplementedError, match="meta tensor"):
        SharedTensorStore.create_from_model("ft_copy_failure", source, parameter_names=["weight"])

    assert _RollbackCheckingFlexibleSharedMemory.rollback_error is None
    assert _RollbackCheckingFlexibleSharedMemory.rollback_closed_and_unlinked is True


def test_failed_materialization_releases_views_before_immediate_store_cleanup(monkeypatch) -> None:
    """A failed device copy cannot leave traceback-held exports that block SHM cleanup."""
    monkeypatch.setattr(
        "flextensor.shm.tensor_store.FlexibleSharedMemory",
        _CloseCheckingFlexibleSharedMemory,
    )
    creator = SharedTensorStore.create_from_model("ft_copy_cleanup", _MixedModel(), parameter_names=["weight"])
    follower = SharedTensorStore.attach("ft_copy_cleanup")
    target = _MixedModel(device="meta")

    try:
        follower.materialize_into(target, device="not-a-real-device")
    except RuntimeError:
        # Cleanup occurs while the copy exception (and its traceback) is still active.
        follower.close()
        creator.close()
    else:
        pytest.fail("invalid target device unexpectedly materialized the shared tensor")

    assert _store_name("ft_copy_cleanup") not in _FakeFlexibleSharedMemory._segments


def test_dense_transpose_preserves_shape_stride_and_values() -> None:
    """A dense transposed parameter retains its non-contiguous layout."""
    source = nn.Module()
    source.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4).t())
    target = nn.Module()
    target.weight = nn.Parameter(
        torch.empty_strided(source.weight.shape, source.weight.stride(), dtype=source.weight.dtype, device="meta")
    )
    creator = SharedTensorStore.create_from_model("ft_transpose", source, parameter_names=["weight"])
    follower = SharedTensorStore.attach("ft_transpose")

    follower.materialize_into(target, device="cpu")

    torch.testing.assert_close(target.weight, source.weight)
    assert target.weight.stride() == source.weight.stride()
    follower.close()
    creator.close()


def test_gapped_layout_uses_contiguous_fallback() -> None:
    """A strided tensor with storage gaps roundtrips through a contiguous payload."""
    source = nn.Module()
    source.weight = nn.Parameter(torch.arange(8, dtype=torch.float32).as_strided((2, 3), (4, 1)))
    target = nn.Module()
    target.weight = nn.Parameter(torch.empty((2, 3), dtype=torch.float32, device="meta"))
    creator = SharedTensorStore.create_from_model("ft_gapped", source, parameter_names=["weight"])
    follower = SharedTensorStore.attach("ft_gapped")

    follower.materialize_into(target, device="cpu")

    torch.testing.assert_close(target.weight, source.weight)
    assert target.weight.is_contiguous()
    follower.close()
    creator.close()


def test_overlapping_layout_uses_contiguous_fallback() -> None:
    """An internally overlapping tensor is copied through a contiguous payload."""
    source = nn.Module()
    source.weight = nn.Parameter(torch.arange(9, dtype=torch.float32).as_strided((3, 3), (2, 2)))
    target = nn.Module()
    target.weight = nn.Parameter(torch.empty((3, 3), dtype=torch.float32, device="meta"))
    creator = SharedTensorStore.create_from_model("ft_overlap", source, parameter_names=["weight"])
    follower = SharedTensorStore.attach("ft_overlap")

    follower.materialize_into(target, device="cpu")

    torch.testing.assert_close(target.weight, source.weight)
    assert target.weight.is_contiguous()
    follower.close()
    creator.close()


def test_aliases_share_one_payload_and_are_rebound_to_one_object() -> None:
    """Tied parameters and aliased buffers are serialized and materialized once each."""
    source = _AliasModel()
    creator = SharedTensorStore.create_from_model("ft_alias", source, parameter_names=["left"])
    manifest = _manifest("ft_alias")
    follower = SharedTensorStore.attach("ft_alias")
    target = _AliasModel(tied=False, device="meta")

    follower.materialize_into(target, device="cpu")

    assert len(manifest["tensors"]) == 2
    assert target.left is target.right
    assert target.first is target.second
    torch.testing.assert_close(target.left, source.left)
    torch.testing.assert_close(target.first, source.first)
    follower.close()
    creator.close()


def test_store_restores_nonpersistent_buffer_membership() -> None:
    """Buffer persistence comes from the creator manifest, not the target skeleton."""
    source = _MixedModel()
    creator = SharedTensorStore.create_from_model("ft_persistence", source, parameter_names=[])
    follower = SharedTensorStore.attach("ft_persistence")
    target = _MixedModel(values=False)
    target._non_persistent_buffers_set.discard("scratch")

    follower.materialize_into(target, device="cpu")

    assert "scratch" in target._non_persistent_buffers_set
    torch.testing.assert_close(target.scratch, source.scratch)
    follower.close()
    creator.close()


@pytest.mark.parametrize("requested", [["missing"], ["counter"]])
def test_creator_rejects_missing_or_non_parameter_names(requested: list[str]) -> None:
    """The resident parameter allowlist cannot name absent objects or buffers."""
    with pytest.raises(ValueError, match=requested[0]):
        SharedTensorStore.create_from_model("ft_bad_name", _MixedModel(), parameter_names=requested)

    assert _store_name("ft_bad_name") not in _FakeFlexibleSharedMemory._segments


def test_creator_rejects_non_strided_tensors() -> None:
    """Sparse and other non-strided layouts are rejected explicitly."""
    model = nn.Module()
    sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (2,))
    model.weight = nn.Parameter(sparse)

    with pytest.raises(ValueError, match="strided"):
        SharedTensorStore.create_from_model("ft_sparse", model, parameter_names=["weight"])


def test_validation_finishes_before_any_target_mutation() -> None:
    """A late dtype mismatch leaves earlier parameters and buffers untouched."""
    source = _MixedModel()
    creator = SharedTensorStore.create_from_model(
        "ft_validate",
        source,
        parameter_names=["weight", "scalar"],
    )
    follower = SharedTensorStore.attach("ft_validate")
    target = _MixedModel(values=False)
    original_weight = target.weight
    original_counter = target.counter
    target.scalar = nn.Parameter(torch.zeros((), dtype=torch.float32))

    with pytest.raises(ValueError, match=r"scalar.*dtype"):
        follower.materialize_into(target, device="cpu")

    assert target.weight is original_weight
    assert target.counter is original_counter
    assert torch.count_nonzero(target.weight) == 0
    assert torch.count_nonzero(target.counter) == 0
    follower.close()
    creator.close()


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (0, b"NOPE", "magic"),
        (4, struct.pack("<I", 99), "version"),
    ],
)
def test_attach_rejects_corrupt_magic_or_version(offset: int, replacement: bytes, message: str) -> None:
    """Header identity fields gate attachment and a rejected handle is closed."""
    creator = SharedTensorStore.create_from_model("ft_header", _MixedModel(), parameter_names=["weight"])
    raw = _raw_segment("ft_header")
    raw[offset : offset + len(replacement)] = replacement

    with pytest.raises(RuntimeError, match=message):
        SharedTensorStore.attach("ft_header")

    assert _FakeFlexibleSharedMemory._references[_store_name("ft_header")] == 1
    creator.close()


def test_attach_rejects_invalid_manifest_json() -> None:
    """Malformed manifest bytes fail closed and do not leak the follower handle."""
    creator = SharedTensorStore.create_from_model("ft_manifest", _MixedModel(), parameter_names=["weight"])
    raw = _raw_segment("ft_manifest")
    raw[_HEADER.size] = ord("!")

    with pytest.raises(RuntimeError, match="manifest"):
        SharedTensorStore.attach("ft_manifest")

    assert _FakeFlexibleSharedMemory._references[_store_name("ft_manifest")] == 1
    creator.close()


def test_attach_preserves_validation_error_and_retryable_handle_when_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creator = SharedTensorStore.create_from_model("ft_attach_retry", _MixedModel(), parameter_names=["weight"])
    raw = _raw_segment("ft_attach_retry")
    raw[0:4] = b"NOPE"
    _RetryableCloseFlexibleSharedMemory.close_failures = 1
    monkeypatch.setattr(
        "flextensor.shm.tensor_store.FlexibleSharedMemory",
        _RetryableCloseFlexibleSharedMemory,
    )

    with pytest.raises(RuntimeError, match="magic") as exc_info:
        SharedTensorStore.attach("ft_attach_retry")

    rejected = SharedTensorStore._take_rejected(exc_info.value)  # noqa: SLF001
    assert rejected is not None
    assert _FakeFlexibleSharedMemory._references[_store_name("ft_attach_retry")] == 2
    rejected.close()
    assert _FakeFlexibleSharedMemory._references[_store_name("ft_attach_retry")] == 1
    creator.close()


@pytest.mark.parametrize("field", ["manifest_length", "data_offset", "data_size"])
def test_attach_rejects_header_bounds_outside_segment(field: str) -> None:
    """Every header range is checked before manifest or payload access."""
    creator = SharedTensorStore.create_from_model("ft_bounds", _MixedModel(), parameter_names=["weight"])
    raw = _raw_segment("ft_bounds")
    magic, version, manifest_length, data_offset, data_size = _HEADER.unpack_from(raw)
    if field == "manifest_length":
        manifest_length = len(raw)
    elif field == "data_offset":
        data_offset = _HEADER.size
    else:
        data_size = len(raw)
    _HEADER.pack_into(raw, 0, magic, version, manifest_length, data_offset, data_size)

    with pytest.raises(RuntimeError, match="bounds"):
        SharedTensorStore.attach("ft_bounds")

    creator.close()


def test_attach_rejects_payload_descriptor_outside_data_region() -> None:
    """Descriptor offsets and lengths are validated against the declared data region."""
    creator = SharedTensorStore.create_from_model("ft_payload", _MixedModel(), parameter_names=["weight"])
    raw = _raw_segment("ft_payload")
    magic, version, _manifest_length, data_offset, data_size = _HEADER.unpack_from(raw)
    manifest = _manifest("ft_payload")
    manifest["tensors"][0]["payload_offset"] = data_size
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert len(encoded) <= data_offset - _HEADER.size
    raw[_HEADER.size : _HEADER.size + len(encoded)] = encoded
    _HEADER.pack_into(raw, 0, magic, version, len(encoded), data_offset, data_size)

    with pytest.raises(RuntimeError, match=r"payload.*bounds"):
        SharedTensorStore.attach("ft_payload")

    creator.close()


@pytest.mark.parametrize(
    "layout",
    [
        {"shape": [2, 3], "stride": [1, 1]},
        {"shape": [3, 3], "stride": [2, 2]},
        {"shape": [2, 3], "stride": [4, 1]},
        {"shape": [2, 3], "stride": [3]},
        {"shape": [-2, 3], "stride": [3, 1]},
        {"shape": [2, 3], "stride": [3, -1]},
        {"shape": [True, 3], "stride": [3, 1]},
        {"shape": [2, 3], "stride": [3, True]},
        {"shape": [2.0, 3], "stride": [3, 1]},
        {"shape": None, "stride": [3, 1]},
        {"shape": [2, 3], "stride": None},
    ],
)
def test_attach_rejects_invalid_tensor_layout(layout) -> None:
    """A corrupt manifest cannot install an invalid or internally overlapping tensor view."""
    creator = SharedTensorStore.create_from_model("ft_stride_overlap", _MixedModel(), parameter_names=["weight"])
    raw = _raw_segment("ft_stride_overlap")
    magic, version, _manifest_length, data_offset, data_size = _HEADER.unpack_from(raw)
    manifest = _manifest("ft_stride_overlap")
    manifest["tensors"][0].update(layout)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert len(encoded) <= data_offset - _HEADER.size
    raw[_HEADER.size : _HEADER.size + len(encoded)] = encoded
    _HEADER.pack_into(raw, 0, magic, version, len(encoded), data_offset, data_size)

    with pytest.raises(RuntimeError, match=r"manifest.*(shape|stride|layout)"):
        SharedTensorStore.attach("ft_stride_overlap")

    assert _FakeFlexibleSharedMemory._references[_store_name("ft_stride_overlap")] == 1
    creator.close()


def test_creating_over_live_store_fails_without_damaging_creator() -> None:
    """A would-be second creator closes its attached handle and leaves the store usable."""
    source = _MixedModel()
    creator = SharedTensorStore.create_from_model("ft_duplicate", source, parameter_names=["weight"])

    with pytest.raises(FileExistsError, match=_store_name("ft_duplicate")):
        SharedTensorStore.create_from_model("ft_duplicate", source, parameter_names=["weight"])

    follower = SharedTensorStore.attach("ft_duplicate")
    target = _MixedModel(device="meta")
    follower.materialize_into(target, device="cpu")
    torch.testing.assert_close(target.weight, source.weight)
    follower.close()
    creator.close()


def test_duplicate_create_preserves_file_exists_and_retryable_handle_when_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flextensor.shm.tensor_store.FlexibleSharedMemory",
        _RetryableCloseFlexibleSharedMemory,
    )
    source = _MixedModel()
    creator = SharedTensorStore.create_from_model("ft_create_retry", source, parameter_names=["weight"])
    _RetryableCloseFlexibleSharedMemory.close_failures = 1

    with pytest.raises(FileExistsError, match=_store_name("ft_create_retry")) as exc_info:
        SharedTensorStore.create_from_model("ft_create_retry", source, parameter_names=["weight"])

    rejected = SharedTensorStore._take_rejected(exc_info.value)  # noqa: SLF001
    assert rejected is not None
    assert _FakeFlexibleSharedMemory._references[_store_name("ft_create_retry")] == 2
    rejected.close()
    assert _FakeFlexibleSharedMemory._references[_store_name("ft_create_retry")] == 1
    creator.close()


def test_attach_missing_store_fails() -> None:
    """Followers do not silently create an absent resident tensor store."""
    with pytest.raises(FileNotFoundError):
        SharedTensorStore.attach("ft_absent")


def test_close_is_idempotent() -> None:
    """Closing an already closed store is a no-op."""
    store = SharedTensorStore.create_from_model("ft_close", _MixedModel(), parameter_names=["weight"])

    store.close()
    store.close()

    assert _store_name("ft_close") not in _FakeFlexibleSharedMemory._segments

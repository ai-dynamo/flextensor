# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor recovery for failed destructive allocation-block construction."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from flextensor.allocation_block import AllocationBlock
from flextensor.collectors import TensorStatistics
from flextensor.utils import set_tensor_data

LOGGER = logging.getLogger(__name__)


@dataclass
class _ConstructionStorageSnapshot:
    original: StorageWeakRef
    restored: StorageWeakRef | None = None


@dataclass(frozen=True)
class _ConstructionTensorSnapshot:
    tensor: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage: _ConstructionStorageSnapshot
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    names: tuple[str | None, ...] | None
    is_conj: bool
    is_neg: bool


@dataclass(frozen=True)
class _ConstructionBlockRollback:
    block: AllocationBlock
    tensor_snapshots: tuple[_ConstructionTensorSnapshot, ...]


class ConstructionRollback:
    """Own tensor recovery until a transfer loader takes over the allocation blocks."""

    def __init__(self) -> None:
        self._tensor_rollbacks: list[_ConstructionBlockRollback] = []
        # Weak references preserve storage identity without retaining source allocations.
        self._storage_snapshots: dict[StorageWeakRef, _ConstructionStorageSnapshot] = {}

    def record(
        self,
        block: AllocationBlock,
        strategy: list[TensorStatistics],
        tensors_map: Mapping[int, torch.Tensor],
    ) -> None:
        """Capture metadata needed to undo AllocationBlock's storage release."""
        snapshots: list[_ConstructionTensorSnapshot] = []
        tensor_ids: set[int] = set()
        for tensor_info in strategy:
            if tensor_info.tensor_id in tensor_ids:
                continue
            tensor_ids.add(tensor_info.tensor_id)
            tensor = tensors_map[tensor_info.tensor_id]
            storage_ref = StorageWeakRef(tensor.untyped_storage())
            if storage_ref not in self._storage_snapshots:
                self._storage_snapshots[storage_ref] = _ConstructionStorageSnapshot(original=storage_ref)
            snapshots.append(
                _ConstructionTensorSnapshot(
                    tensor=tensor,
                    shape=tuple(tensor.shape),
                    stride=tensor.stride(),
                    storage_offset=int(tensor.storage_offset()),
                    storage=self._storage_snapshots[storage_ref],
                    dtype=tensor.dtype,
                    device=tensor.device,
                    layout=tensor.layout,
                    names=getattr(tensor, "names", None),
                    is_conj=tensor.is_conj(),
                    is_neg=tensor.is_neg(),
                )
            )
        self._tensor_rollbacks.append(_ConstructionBlockRollback(block=block, tensor_snapshots=tuple(snapshots)))

    def restore(self, *, attempts: int) -> list[AllocationBlock]:
        """Retry restoration and return blocks still needed by pending tensors."""
        for _attempt in range(attempts):
            if not self._tensor_rollbacks:
                break
            try:
                self._tensor_rollbacks = self._restore_released_tensors(self._tensor_rollbacks)
            except BaseException:
                # A custom tensor implementation may fail before the helper can
                # identify one block. Protect every still-pending block.
                LOGGER.warning("Tensor restoration cleanup failed", exc_info=True)
                break
        self._storage_snapshots.clear()
        return [rollback.block for rollback in self._tensor_rollbacks]

    def commit(self) -> None:
        """Discard recovery state after a successful ownership handoff."""
        self._tensor_rollbacks.clear()
        self._storage_snapshots.clear()

    @staticmethod
    def _tensor_matches_snapshot(snapshot: _ConstructionTensorSnapshot) -> bool:
        tensor = snapshot.tensor
        storage_ref = snapshot.storage.restored or snapshot.storage.original
        return (
            tuple(tensor.shape) == snapshot.shape
            and tensor.stride() == snapshot.stride
            and tensor.storage_offset() == snapshot.storage_offset
            and (storage_ref.expired() or tensor.untyped_storage()._cdata == storage_ref.cdata)  # noqa: SLF001
            and tensor.dtype == snapshot.dtype
            and tensor.device == snapshot.device
            and tensor.layout == snapshot.layout
            and getattr(tensor, "names", None) == snapshot.names
            and tensor.is_conj() == snapshot.is_conj
            and tensor.is_neg() == snapshot.is_neg
        )

    @classmethod
    def _restore_tensor_from_view(
        cls,
        snapshot: _ConstructionTensorSnapshot,
        view: torch.Tensor,
    ) -> None:
        if snapshot.layout != torch.strided or view.layout != torch.strided:
            raise RuntimeError("Allocation-block construction rollback only supports strided tensors")
        if tuple(view.shape) != snapshot.shape:
            raise RuntimeError(
                f"Rollback view shape {tuple(view.shape)!r} does not match source shape {snapshot.shape!r}"
            )
        if view.dtype != snapshot.dtype:
            raise RuntimeError(f"Rollback view dtype {view.dtype} does not match source dtype {snapshot.dtype}")

        storage: torch.UntypedStorage | None = None
        storage_ref = snapshot.storage.restored or snapshot.storage.original
        if not storage_ref.expired():
            storage = torch.UntypedStorage._new_with_weak_ptr(storage_ref.cdata)  # noqa: SLF001
        restored = torch.empty(0, dtype=snapshot.dtype, device=snapshot.device)
        # Reuse surviving backing storage or a sibling's restored storage.
        # set_ grows replacement storage for later disjoint views if needed.
        restored.set_(
            storage if storage is not None else restored.untyped_storage(),
            snapshot.storage_offset,
            snapshot.shape,
            snapshot.stride,
        )
        # Block views hold logical values; set these flags before copying so
        # copy_ writes the corresponding physical values into the new storage.
        torch._C._set_conj(restored, snapshot.is_conj)  # noqa: SLF001
        torch._C._set_neg(restored, snapshot.is_neg)  # noqa: SLF001
        with torch.no_grad():
            # Surviving original storage already contains every sibling's data;
            # replacement storage needs each released tensor's values copied.
            if snapshot.storage.original.expired():
                restored.copy_(view)
                snapshot.storage.restored = StorageWeakRef(restored.untyped_storage())
            set_tensor_data(snapshot.tensor, restored)
            rename = getattr(snapshot.tensor, "rename_", None)
            if snapshot.names is not None and callable(rename) and any(name is not None for name in snapshot.names):
                rename(*snapshot.names)
        if not cls._tensor_matches_snapshot(snapshot):
            raise RuntimeError("Allocation-block construction rollback did not preserve tensor metadata")

    def _restore_released_tensors(
        self,
        rollbacks: list[_ConstructionBlockRollback],
    ) -> list[_ConstructionBlockRollback]:
        """Restore released source storage, retaining blocks whose data is still needed."""
        pending: list[_ConstructionBlockRollback] = []
        for rollback in rollbacks:
            block_failed = False
            for index, snapshot in enumerate(rollback.tensor_snapshots):
                if self._tensor_matches_snapshot(snapshot):
                    continue
                try:
                    view = rollback.block.views[index]
                    self._restore_tensor_from_view(snapshot, view)
                except BaseException:
                    block_failed = True
                    LOGGER.warning("Failed to restore source tensor during construction rollback", exc_info=True)
            if block_failed:
                pending.append(rollback)
        return pending

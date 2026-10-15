# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Self

import torch

from flextensor.host_pinning import HostPinner
from flextensor.shm import FlexibleSharedMemory, ProcessFileLock
from flextensor.shm.flexible_shm import LOCK_ACQUIRE_TIMEOUT
from flextensor.shm.sync_primitives import BaseLock
from flextensor.utils import (
    _DEFAULT_PACKED_TENSOR_ALIGNMENT_BYTES,
    _compute_packed_byte_layout,
    is_dense_layout,
)

logger = logging.getLogger(__name__)

_PINNED_SHM_GIB_PER_SECOND = 1.0


class AllocationBlock:
    """
    A context manager for managing tensor allocations within a block.
    """

    def __init__(
        self,
        manager: Any = None,
        device: str = "cpu",
        block_size: int | None = None,
        shm_block_name: str | None = None,
        lock_class: type[BaseLock] | None = None,
        *,
        host_pinner: HostPinner,
        shm_owner: FlexibleSharedMemory | None = None,
        **config: Any,
    ) -> None:
        self.manager = manager
        self.tensors: list[torch.Tensor] = []
        self.views: list[torch.Tensor] = []
        self.tensor_offsets: list[int] = []
        self.device = device
        self.block_size = block_size
        self.shm_block_name = shm_block_name
        self.lock_class: type[BaseLock] = lock_class if lock_class is not None else ProcessFileLock
        self.shm_block: FlexibleSharedMemory | None = None
        self.shm_owner = shm_owner
        self.pinned_memory = config.get("pinned_memory", False)
        self.host_pinner: HostPinner = host_pinner
        self.memory_alignment = config.get("memory_alignment", _DEFAULT_PACKED_TENSOR_ALIGNMENT_BYTES)
        self.release_tensor_memory = config.get("release_tensor_memory", False)
        if type(self.memory_alignment) is not int or self.memory_alignment <= 0:
            raise ValueError(f"memory_alignment must be a positive integer, got {self.memory_alignment!r}")
        self.load_from_shm: bool = config.get("load_from_shm", False)
        self.shm_ptr: Any = None
        self.c_buf: Any = None
        self.is_memory_creator = True
        self.block: torch.Tensor | None = (
            self._make_base_block(block_size, device=device, pin_memory=self.pinned_memory)
            if block_size is not None
            else None
        )

    def add(self, tensor: torch.Tensor) -> None:
        self.tensors.append(tensor)

    def _require_block(self) -> torch.Tensor:
        if self.block is None:
            raise RuntimeError("Allocation block is not prepared")
        return self.block

    def allocate(self) -> list[torch.Tensor]:
        self._prepare_block()
        if self.is_memory_creator:
            self._tensors_to_memory_block()
        self.tensors.clear()
        return self.views

    def project_views(self, gpu_block: Self) -> tuple[list[torch.Tensor], torch.Tensor]:
        block = self._require_block()
        gpu_tensor = gpu_block._require_block()  # noqa: SLF001
        return (
            [
                self._view_from_block(
                    gpu_tensor,
                    view.dtype,
                    view.shape,
                    int(view.storage_offset() * view.element_size()),
                    view.stride(),
                )
                for view in self.views
            ],
            self._view_from_block(
                gpu_tensor,
                gpu_tensor.dtype,
                block.shape,
                int(block.storage_offset() * block.element_size()),
                block.stride(),
            ),
        )

    def copy_to(self, tensor: torch.Tensor, non_blocking: bool = False) -> None:
        non_blocking = non_blocking and tensor.device.type == "cuda"
        tensor.copy_(self._require_block(), non_blocking=non_blocking)

    def release(self) -> None:
        """Release shared memory resources."""
        if self.shm_block is not None:
            if self.shm_block_name is None:
                raise RuntimeError("Shared memory block has no name")
            sem_name = self.shm_block_name + "_rel_lock"
            lock = self.lock_class(sem_name, locked=False)
            with lock:
                self.shm_block.close()
                self.shm_block = None
            lock.close()
        self.block = None
        self.views = []
        self.tensors = []
        self.tensor_offsets = []

    def _prepare_block(self) -> None:
        self.tensor_offsets, tensor_offset = _compute_packed_byte_layout(
            (tensor.element_size() * tensor.numel() for tensor in self.tensors),
            self.memory_alignment,
        )

        self.block = self._make_base_block(tensor_offset, device=self.device, pin_memory=self.pinned_memory)
        self._prepare_views()

    def _make_base_block(self, nbytes: int, device: str = "cpu", pin_memory: bool = False) -> torch.Tensor:
        if self.shm_block_name is not None and device == "cpu":
            # Use shared memory
            # Assume at least 1 GiB/s host registration; tune this if slower hosts appear.
            lock_acquire_timeout = (
                max(LOCK_ACQUIRE_TIMEOUT, nbytes / (1024**3 * _PINNED_SHM_GIB_PER_SECOND))
                if pin_memory
                else LOCK_ACQUIRE_TIMEOUT
            )
            self.shm_block = FlexibleSharedMemory(
                name=self.shm_block_name,
                shm_size=nbytes,
                pinned_memory=pin_memory,
                lock_class=self.lock_class,
                lock_acquire_timeout=lock_acquire_timeout,
                _owner=self.shm_owner,
            )
            self.is_memory_creator = self.shm_block.shm_creator
            if self.is_memory_creator == self.load_from_shm:
                raise ValueError(
                    f"is_memory_creator ({self.is_memory_creator}) == load_from_shm ({self.load_from_shm})"
                )
            shm_block = self.shm_block.block
            if shm_block is None or shm_block.buf is None:
                raise RuntimeError("Shared memory block was closed during allocation")
            return torch.frombuffer(shm_block.buf, dtype=torch.uint8)

        # Use regular memory
        # Create tensor on specified device first
        tensor = torch.empty(nbytes, dtype=torch.uint8, device=device)
        if pin_memory and device == "cpu":
            tensor = self.host_pinner.pin(tensor)
        return tensor

    def _prepare_views(self) -> None:
        block = self._require_block()
        for tensor, tensor_offset in zip(self.tensors, self.tensor_offsets, strict=False):
            # Preserve the original stride pattern for dense layouts (e.g.
            # Fortran-contiguous weights created by ``weight.t()`` in modelopt).
            # Using the original strides ensures ``view.copy_(tensor)`` copies
            # bytes in the same memory order, so downstream kernels that rely on
            # a specific layout (column-major for CUTLASS FP8 GEMM) still work.
            stride = tensor.stride() if is_dense_layout(tensor) else None
            view = self._view_from_block(block, tensor.dtype, tensor.shape, tensor_offset, stride)
            self.views.append(view)

    def _tensors_to_memory_block(self) -> None:
        for view, tensor in zip(self.views, self.tensors, strict=False):
            view.copy_(tensor)
            if self.release_tensor_memory:
                # Release tensor memory immediately after copying to reduce peak memory
                # This replaces the underlying storage with an empty tensor while keeping
                # the same tensor object ID (important for downstream ID-based lookups)
                tensor.data = torch.empty(0, device="cpu", dtype=tensor.dtype)

    def _view_from_block(
        self,
        base_uint8: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.Size | tuple[int, ...],
        offset_bytes: int = 0,
        stride: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        ust = base_uint8.untyped_storage()  # shared untyped memory
        t = torch.empty(0, dtype=dtype, device=base_uint8.device)

        if offset_bytes % t.element_size() != 0:
            msg = f"offset_bytes ({offset_bytes}) must be a multiple of dtype element size {t.element_size()}"
            raise ValueError(msg)
        # storage_offset and stride are specified in ELEMENTS of target dtype (not bytes)
        storage_offset_elems = offset_bytes // t.element_size()

        # if stride=None -> use contiguous stride
        t.set_(ust, storage_offset_elems, shape, stride)  # type: ignore[arg-type]
        return t


class AllocationManager:
    """
    A manager for allocation blocks.
    """

    def __init__(
        self,
        shm_block_name_prefix: str | None = None,
        load_from_shm: bool = False,
        pinned_memory: bool = False,
        *,
        host_pinner: HostPinner,
        shm_owner: FlexibleSharedMemory | None = None,
        memory_alignment: int = _DEFAULT_PACKED_TENSOR_ALIGNMENT_BYTES,
        lock_class: type[BaseLock] | None = None,
        release_tensor_memory: bool = False,
    ) -> None:
        # Initialize any tracking structures or state here
        self.blocks: list[AllocationBlock] = []
        self.shm_block_name = shm_block_name_prefix
        self.block_index = 0
        self.load_from_shm = load_from_shm
        self.pinned_memory = pinned_memory
        self.host_pinner = host_pinner
        self.shm_owner = shm_owner
        self.memory_alignment = memory_alignment
        self.lock_class = lock_class if lock_class is not None else ProcessFileLock
        self.release_tensor_memory = release_tensor_memory

    def block(self, device: str = "cpu") -> AllocationBlock:
        """
        Create a new allocation block.
        """
        block_name = self.shm_block_name + "_" + str(self.block_index) if self.shm_block_name is not None else None
        self.block_index += 1
        block = AllocationBlock(
            self,
            device=device,
            shm_block_name=block_name if device == "cpu" else None,
            lock_class=self.lock_class,
            load_from_shm=self.load_from_shm,
            pinned_memory=self.pinned_memory,
            host_pinner=self.host_pinner,
            shm_owner=self.shm_owner,
            memory_alignment=self.memory_alignment,
            release_tensor_memory=self.release_tensor_memory,
        )
        self.blocks.append(block)
        return block

    def create_max_block(self, device: str = "cuda") -> AllocationBlock:
        """
        Create a block that can fit the largest block in the manager, based on byte size.
        """
        max_bytes = max(
            (block.block.numel() * block.block.element_size() for block in self.blocks if block.block is not None),
            default=0,
        )
        shm_name = self.shm_block_name if device == "cpu" else None
        return AllocationBlock(
            self,
            device=device,
            block_size=max_bytes,
            shm_block_name=shm_name,
            lock_class=self.lock_class,
            # Only propagate CPU-relevant flags when device is CPU
            load_from_shm=self.load_from_shm if device == "cpu" else False,
            pinned_memory=self.pinned_memory if device == "cpu" else False,
            host_pinner=self.host_pinner,
            shm_owner=self.shm_owner,
        )

    def release(self) -> None:
        for block in self.blocks:
            block.release()
        self.blocks = []
        self.block_index = 0


if __name__ == "__main__":
    manager = AllocationManager(host_pinner=HostPinner())

    b = manager.block()
    t1 = torch.ones(12, dtype=torch.float32)
    t2 = torch.ones(15, dtype=torch.float64) * 2
    t3 = torch.ones(13, dtype=torch.int32) * 3
    b.add(t1)
    b.add(t2)
    b.add(t3)
    views = b.allocate()

    b2 = manager.block()
    t4 = torch.ones(18, dtype=torch.float32)
    t5 = torch.ones(19, dtype=torch.float64) * 2
    b2.add(t4)
    b2.add(t5)
    views2 = b2.allocate()

    gpu_block = manager.create_max_block(device="cuda")
    gpu_views1, block_view1 = b.project_views(gpu_block)
    gpu_views2, block_view2 = b2.project_views(gpu_block)

    b.copy_to(block_view1)
    b2.copy_to(block_view2)

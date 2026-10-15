# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
import torch

from flextensor import allocation_block as allocation_block_module
from flextensor import host_pinning as host_pinning_module
from flextensor.allocation_block import AllocationBlock, AllocationManager
from flextensor.host_pinning import HostPinner, HostPinRegistry


class TestAllocationBlock:
    def setup_method(self):
        """Setup test fixtures before each test method."""
        # Create some tensors for testing
        # Note: Using float32 instead of float64 for MPS compatibility
        # Force CPU device for tests
        self.tensors = [
            torch.ones(12, dtype=torch.float32, device="cpu"),
            (torch.ones(15, dtype=torch.float32, device="cpu") * 2).to("cpu"),
            (torch.ones(13, dtype=torch.int32, device="cpu") * 3).to("cpu"),
        ]
        self.block = AllocationBlock(device="cpu", host_pinner=HostPinner())

    @pytest.mark.parametrize("memory_alignment", [0, -1, 1.5, True, "128"])
    def test_rejects_invalid_memory_alignment(self, memory_alignment: object):
        with pytest.raises(ValueError, match="memory_alignment must be a positive integer"):
            AllocationBlock(
                device="cpu",
                host_pinner=HostPinner(),
                memory_alignment=memory_alignment,
            )

    def test_allocate_and_allocate(self):
        # Allocate all tensors
        for tensor in self.tensors:
            self.block.add(tensor)
        # Commit and get the views
        views = self.block.allocate()
        # Check that the returned views have the same content as the original tensors
        assert len(views) == len(self.tensors)
        for orig, view in zip(self.tensors, views, strict=False):
            # Views should have same shape and dtype
            assert orig.shape == view.shape
            assert orig.dtype == view.dtype
            # Views should have same content
            if orig.dtype in [torch.float32, torch.float64]:
                assert torch.allclose(orig, view)
            else:
                assert torch.equal(orig, view)
        # Check that the data is correct
        assert torch.allclose(views[0], torch.ones(12, dtype=torch.float32))
        assert torch.allclose(views[1], torch.ones(15, dtype=torch.float32) * 2)
        assert torch.equal(views[2], torch.ones(13, dtype=torch.int32) * 3)

    def test_view_from_block(self):
        # Test the _view_from_block method for correct dtype and shape
        base = self.block._make_base_block(64, device="cpu")
        t_i32 = self.block._view_from_block(base, torch.int32, shape=(16,), offset_bytes=0)
        t_f64 = self.block._view_from_block(base, torch.float64, shape=(8,), offset_bytes=0)
        assert t_i32.shape == (16,)
        assert t_i32.dtype == torch.int32
        assert t_f64.shape == (8,)
        assert t_f64.dtype == torch.float64
        # Fill one view and check reinterpretation
        t_i32[:] = torch.arange(16, dtype=torch.int32)
        # The same memory, so t_f64 should see the same bytes
        assert t_f64.untyped_storage().data_ptr() == t_i32.untyped_storage().data_ptr()

    def test_offset_and_stride(self):
        # Test offset_bytes and stride
        base = self.block._make_base_block(32, device="cpu")
        # Offset by 16 bytes (4 float32 elements)
        t = self.block._view_from_block(base, torch.float32, shape=(2,), offset_bytes=16)
        assert t.storage_offset() == 4
        assert t.shape == (2,)
        assert t.dtype == torch.float32

    def test_copy_to_cpu_block(self):
        """Test copy_to with CPU target block."""
        # Create source block with one tensor
        source_block = AllocationBlock(device="cpu", host_pinner=HostPinner())
        test_tensor = torch.ones(10, dtype=torch.float32) * 5
        source_block.add(test_tensor)
        source_block.allocate()

        # Create target CPU block
        target_block = AllocationBlock(device="cpu", block_size=source_block.block.numel(), host_pinner=HostPinner())

        # Copy to target block
        result = source_block.copy_to(target_block.block, non_blocking=True)

        # Should return None for blocking copy
        assert result is None

        # Verify data was copied correctly
        target_view = source_block._view_from_block(target_block.block, torch.float32, (10,), 0)
        assert torch.allclose(target_view, test_tensor)

    def test_copy_to_cuda_block(self):
        """Test copy_to with CUDA target block."""
        # Create source block with one tensor
        source_block = AllocationBlock(device="cpu", host_pinner=HostPinner())
        test_tensor = torch.ones(8, dtype=torch.float32) * 3
        source_block.add(test_tensor)
        source_block.allocate()

        # Skip test if CUDA not available
        if not torch.cuda.is_available():
            print("CUDA not available, skipping CUDA copy test")
            return

        # Create target CUDA block
        target_block = AllocationBlock(device="cuda", block_size=source_block.block.numel(), host_pinner=HostPinner())

        # Copy to CUDA block with non-blocking
        gpu_views, gpu_block_view = source_block.project_views(target_block)
        source_block.copy_to(gpu_block_view, non_blocking=True)

        # Should return a CUDA event for non-blocking GPU copy
        event = torch.Event(device=gpu_views[0].device)
        event.record()

        # Synchronize before checking results
        event.synchronize()

        # Verify data was copied correctly to GPU
        assert torch.allclose(gpu_views[0].cpu(), test_tensor)


class TestMakeBaseBlockPinning:
    """Behavior of ``_make_base_block`` for the ``pin_memory=True`` path."""

    def _block_with_pinner(self, pinner: HostPinner) -> AllocationBlock:
        return AllocationBlock(device="cpu", host_pinner=pinner, pinned_memory=True)

    def test_large_pinned_shm_scales_lock_timeout(self, monkeypatch):
        """A follower gets enough lock time for the creator to pin a large block."""
        captured_constructor_args = {}

        class _SpyShm:
            def __init__(self, **kwargs):
                captured_constructor_args.update(kwargs)
                self.block = MagicMock()
                self.block.buf = bytearray(1)
                self.shm_creator = True

        monkeypatch.setattr(allocation_block_module, "FlexibleSharedMemory", _SpyShm)
        block = AllocationBlock(
            device="cpu",
            host_pinner=HostPinner(),
            pinned_memory=True,
            shm_block_name="large_pinned_shm",
        )

        block._make_base_block(6 * 1024**3, device="cpu", pin_memory=True)

        assert captured_constructor_args["lock_acquire_timeout"] == 6.0

    def test_host_register_failure_propagates(self):
        """A genuine cudaHostRegister failure must propagate as RuntimeError
        — silent fallback would mask RLIMIT_MEMLOCK / pinned-pool
        misconfigurations as a perf regression that is hard to diagnose
        from logs."""
        registry = MagicMock(spec=HostPinRegistry)
        registry.pin_in_place.side_effect = RuntimeError("CUDA runtime call cudaHostRegister failed with error code 2")
        pinner = HostPinner(registry)
        block = self._block_with_pinner(pinner)

        with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
            block._make_base_block(64, device="cpu", pin_memory=True)

        registry.pin_in_place.assert_called_once()

    def test_shm_segment_pinning_is_independent_of_pinned_memory_mode(self, monkeypatch):
        """Both ``OffloadConfig.pinned_memory_mode`` and the corresponding
        ``TensorManager.__init__`` docstring promise that SHM segments
        register in place via ``cudaHostRegister`` regardless of mode —
        because POSIX shared-memory buffers can't be re-allocated through
        PyTorch's pinned allocator (``tensor.pin_memory()``). The mode
        selector lives outside the SHM construction site entirely:
        ``AllocationBlock._make_base_block`` only sees the ``pin_memory``
        boolean and forwards it to ``FlexibleSharedMemory(pinned_memory=...)``.

        Pin the invariant directly: with ``shm_block_name`` set,
        ``FlexibleSharedMemory`` must be constructed with the same
        ``pinned_memory`` value regardless of which pinner is wired in
        (host_register vs torch). A future refactor that tries to "respect
        ``pinned_memory_mode`` inside SHM" would break the documented
        contract; this test catches it.

        Also verifies the negative invariant from the same docstrings:
        ``pinned_memory=False`` propagates through to
        ``FlexibleSharedMemory(pinned_memory=False)`` (no SHM-segment
        registration) regardless of mode.
        """
        captured_constructor_args: list[dict] = []

        class _SpyShm:
            def __init__(self, **kwargs):
                captured_constructor_args.append(kwargs)
                # Production code reads ``self.shm_block.block.buf`` and feeds it
                # to torch.frombuffer, so the spy needs a real bytes-like buffer.
                self._buf = bytearray(kwargs["shm_size"])
                self.block = MagicMock()
                self.block.buf = self._buf
                self.shm_creator = True

        monkeypatch.setattr(allocation_block_module, "FlexibleSharedMemory", _SpyShm)

        # Vary the pinner (proxy for "would have been built from
        # pinned_memory_mode='host_register' vs 'torch'") AND the
        # pinned_memory bool, and capture what FlexibleSharedMemory sees.
        cases = [
            ("host_register-with-pinning", HostPinner(MagicMock(spec=HostPinRegistry)), True),
            ("torch-with-pinning", HostPinner(), True),
            ("host_register-without-pinning", HostPinner(MagicMock(spec=HostPinRegistry)), False),
            ("torch-without-pinning", HostPinner(), False),
        ]

        for label, pinner, pinned_memory in cases:
            captured_constructor_args.clear()
            block = AllocationBlock(
                device="cpu",
                shm_block_name=f"test_shm_{label}",
                host_pinner=pinner,
                pinned_memory=pinned_memory,
                block_size=64,
            )

            assert len(captured_constructor_args) == 1, (
                f"{label}: expected exactly one FlexibleSharedMemory construction"
            )
            assert captured_constructor_args[0]["pinned_memory"] is pinned_memory, (
                f"{label}: AllocationBlock forwarded pinned_memory={captured_constructor_args[0]['pinned_memory']!r} "
                f"to FlexibleSharedMemory; expected {pinned_memory!r}. The mode of the "
                "host pinner must not influence whether the SHM segment is registered."
            )
            assert block.shm_block is not None

    def test_torch_mode_failure_propagates(self):
        """torch-mode pin_memory failures must propagate as RuntimeError from
        ``_make_base_block`` — silent fallback to pageable would mask
        RLIMIT_MEMLOCK / pinned-pool misconfigurations as a perf regression."""
        pinner = HostPinner()  # torch mode

        original_pin_memory = torch.Tensor.pin_memory
        try:
            torch.Tensor.pin_memory = lambda self: (_ for _ in ()).throw(  # type: ignore[assignment]
                RuntimeError("simulated pin_memory failure")
            )
            block = self._block_with_pinner(pinner)

            with pytest.raises(RuntimeError, match="simulated pin_memory failure"):
                block._make_base_block(32, device="cpu", pin_memory=True)
        finally:
            torch.Tensor.pin_memory = original_pin_memory  # type: ignore[assignment]


class _RecordingCudart:
    """Fake cudart that succeeds for register/unregister and records every call.

    Used by the registry-lifetime tests to assert that ``release()`` /
    ``shutdown()`` paths do *not* invoke ``cudaHostUnregister`` — the only
    legitimate unregister path is :meth:`HostPinner.release_all`.
    """

    def __init__(self) -> None:
        self.register_calls: list[int] = []
        self.unregister_calls: list[int] = []

    def cudaHostRegister(self, ptr, _size, _flags):  # noqa: N802
        self.register_calls.append(int(ptr))

        class _Ok:
            value = 0

        return _Ok()

    def cudaHostUnregister(self, ptr):  # noqa: N802
        self.unregister_calls.append(int(ptr))

        class _Ok:
            value = 0

        return _Ok()


class TestRegistryLifetimeAcrossBlockShutdown:
    """``AllocationBlock.release()`` / ``AllocationManager.release()`` must NOT
    call ``cudaHostUnregister``. The registry is global state owned by the
    :class:`~flextensor.tensor_manager.TensorManager`; the only legitimate
    unregister path is :meth:`HostPinner.release_all` (driven by
    ``TensorManager.shutdown``).

    Without this guard, a future "be tidy at shutdown" refactor that adds
    a release call here would either:

    - Free pinned storage while pages are still ``cudaHostRegister``-locked
      (kernel-level use-after-free), OR
    - Cause a double-unregister when ``TensorManager.shutdown`` also fires
      (spurious ``cudaErrorHostMemoryNotRegistered`` from the kernel).

    Both regress silently in CI today because no test asserts the contract.
    """

    def _make_pinner(self, monkeypatch):
        cudart = _RecordingCudart()
        monkeypatch.setattr(host_pinning_module, "_probe_cudart", lambda: cudart)
        registry = HostPinRegistry()
        return HostPinner(registry), registry, cudart

    def test_block_release_does_not_unregister_pins(self, monkeypatch):
        pinner, registry, cudart = self._make_pinner(monkeypatch)
        block = AllocationBlock(
            device="cpu",
            host_pinner=pinner,
            pinned_memory=True,
            block_size=128,
        )

        # The block's _make_base_block (driven by block_size) must have
        # routed through host_pinner.pin → cudaHostRegister.
        assert len(cudart.register_calls) == 1, "test precondition: block construction must have pinned its base buffer"
        assert len(registry) == 1

        block.release()

        # Critical: release() touched only block-local references; the
        # registry must still hold the entry and the kernel must not have
        # been told to unregister.
        assert len(registry) == 1, (
            "AllocationBlock.release() removed an entry from the host_pin_registry — "
            "this would free CPU storage while the kernel still holds the page locked. "
            "Only TensorManager.shutdown() may release pins."
        )
        assert cudart.unregister_calls == [], (
            "AllocationBlock.release() invoked cudaHostUnregister — "
            "the contract is that block release is a no-op for pins."
        )

        # And the legitimate path still works after the block is gone.
        pinner.release_all()
        assert len(registry) == 0
        assert len(cudart.unregister_calls) == 1

    def test_manager_release_does_not_unregister_pins(self, monkeypatch):
        pinner, registry, cudart = self._make_pinner(monkeypatch)
        manager = AllocationManager(host_pinner=pinner, pinned_memory=True)
        # AllocationManager.block() doesn't size eagerly; trigger the pinning
        # path explicitly per block so we have something for release() to walk.
        for nbytes in (64, 96, 128):
            block = manager.block()
            block.block = block._make_base_block(nbytes, device="cpu", pin_memory=True)

        assert len(cudart.register_calls) == 3
        assert len(registry) == 3

        manager.release()

        # Same invariant as above, applied across the manager's loop.
        assert len(registry) == 3, (
            "AllocationManager.release() removed entries from the host_pin_registry — "
            "the manager must defer all unregister calls to TensorManager.shutdown()."
        )
        assert cudart.unregister_calls == []

        pinner.release_all()
        assert len(registry) == 0
        assert len(cudart.unregister_calls) == 3

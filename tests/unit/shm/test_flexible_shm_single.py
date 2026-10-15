# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-process unit tests for FlexibleSharedMemory class."""

import contextlib
import os
import time
from unittest import mock

import pytest

from flextensor.shm import FlexibleSharedMemory, ProcessFileLock, SemaphoreLock
from flextensor.shm.flexible_shm import KeepAlive, _unlink_untracked_shm


def test_unlink_untracked_shm_balances_registration_if_peer_removed_name():
    """A second handle finding an absent name must undo its manual registration."""
    from multiprocessing.shared_memory import SharedMemory

    owner = SharedMemory(create=True, size=1)
    peer = SharedMemory(name=owner.name)
    try:
        owner.unlink()
        with (
            mock.patch("flextensor.shm.flexible_shm.resource_tracker_register") as register,
            mock.patch("flextensor.shm.flexible_shm.resource_tracker_unregister") as unregister,
        ):
            _unlink_untracked_shm(peer)

        register.assert_called_once_with(peer._name, "shared_memory")
        unregister.assert_called_once_with(peer._name, "shared_memory")
    finally:
        owner.close()
        peer.close()


def test_unlink_untracked_shm_balances_registration_before_reraising():
    """Unexpected unlink errors are reported after undoing the manual registration."""
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(create=True, size=1)
    try:
        with (
            mock.patch.object(SharedMemory, "unlink", side_effect=OSError("unlink failed")),
            mock.patch("flextensor.shm.flexible_shm.resource_tracker_register") as register,
            mock.patch("flextensor.shm.flexible_shm.resource_tracker_unregister") as unregister,
            pytest.raises(OSError, match="unlink failed"),
        ):
            _unlink_untracked_shm(shm)

        register.assert_called_once_with(shm._name, "shared_memory")
        unregister.assert_called_once_with(shm._name, "shared_memory")
    finally:
        shm.unlink()
        shm.close()


class TestFlexibleSharedMemorySingle:
    """Single-process tests for FlexibleSharedMemory class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "fsm_test"  # Short name to avoid POSIX name limits

        # Shared memory segments created by FlexibleSharedMemory
        # Must match the compact naming in FlexibleSharedMemory.__init__
        self.shm_names = [
            self.test_name,  # Main shared memory
            "sm_" + self.test_name + "_kd",  # Keep alive dict (SharedMemoryDict adds sm_ prefix)
            self.test_name + "_km",  # Keep alive timestamps array
        ]

        # Semaphores / lock names created by FlexibleSharedMemory
        self.semaphore_names = [
            self.test_name,  # Main lock
            self.test_name + "_ks",  # Keep alive lock
        ]

        self.test_size = 1024 * 4  # 4KB for tests
        self._cleanup_resources()

    def teardown_method(self):
        """Cleanup after each test method."""
        self._cleanup_resources()

    def _cleanup_resources(self):
        """Helper to cleanup test resources."""
        try:
            from multiprocessing.shared_memory import SharedMemory

            import posix_ipc

            # Clean up semaphores (POSIX)
            for sem_name in self.semaphore_names:
                try:
                    sem = posix_ipc.Semaphore(sem_name)
                    sem.close()
                except (posix_ipc.ExistentialError, FileNotFoundError):
                    pass
                try:
                    sem = posix_ipc.Semaphore(sem_name)
                    sem.unlink()
                except (posix_ipc.ExistentialError, FileNotFoundError):
                    pass

            # Clean up file locks
            import tempfile
            from pathlib import Path

            for lock_name in self.semaphore_names:
                lock_path = Path(tempfile.gettempdir()) / f"{lock_name.lstrip('/')}.lock"
                if lock_path.exists():
                    with contextlib.suppress(FileNotFoundError, OSError):
                        lock_path.unlink()

            # Clean up shared memory segments
            for shm_name in self.shm_names:
                try:
                    shm = SharedMemory(name=shm_name, create=False)
                    shm.close()
                    shm.unlink()
                except (FileNotFoundError, FileExistsError):
                    pass
                # Try to unlink even if we couldn't open it
                with contextlib.suppress(FileNotFoundError, FileExistsError):
                    SharedMemory(name=shm_name, create=False).unlink()
        except ImportError:
            pass

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_flexible_shared_memory_creation(self, lock_class):
        """Test FlexibleSharedMemory creation."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,  # Keep this basic test independent of CUDA.
            lock_class=lock_class,
        )

        assert fsm.name == self.test_name
        assert fsm.shm_size >= self.test_size  # Actual size may be rounded up
        assert fsm.pid == os.getpid()
        assert fsm.pinned_memory is False
        assert fsm.shm_creator is True  # Should be creator in single process
        assert fsm.shm is not None
        assert fsm.keep_alive is not None
        assert fsm.main_lock is not None

        fsm.close()

    def test_main_lock_acquisition_uses_custom_timeout_and_closes_on_failure(self):
        """A busy main lock reports the segment and honors its configured timeout."""
        with (
            mock.patch.object(
                ProcessFileLock,
                "acquire",
                autospec=True,
                side_effect=BlockingIOError,
            ) as acquire,
            mock.patch.object(
                ProcessFileLock,
                "close",
                autospec=True,
                side_effect=ProcessFileLock.close,
            ) as close,
            pytest.raises(BlockingIOError, match=rf"{self.test_name}.*17.0 seconds"),
        ):
            FlexibleSharedMemory(
                name=self.test_name,
                shm_size=self.test_size,
                pinned_memory=False,
                lock_class=ProcessFileLock,
                lock_acquire_timeout=17.0,
            )

        acquire.assert_called_once_with(mock.ANY, non_blocking=False, timeout=17.0)
        close.assert_called_once_with(mock.ANY)

    def test_capacity_failure_does_not_poison_retry(self):
        """A failed creator removes undersized auxiliary segments before retry."""
        with pytest.raises(RuntimeError):
            FlexibleSharedMemory(
                name=self.test_name,
                shm_size=self.test_size,
                keep_alive_dict_size=8,
                pinned_memory=False,
            )

        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_dict_size=128 * 1024,
            pinned_memory=False,
        )
        fsm.close()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_memory_block_access(self, lock_class):
        """Test accessing the memory block."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Should be able to access block property
        block = fsm.block
        assert block is not None
        assert block is fsm.shm
        # Note: block.size might not match self.test_size if connecting to existing memory
        # When shm_creator is False, the size parameter is ignored and existing size is used
        assert block.size >= self.test_size  # At least as large as requested

        # Should be able to write to memory
        test_data = b"Hello, FlexibleSharedMemory!"
        block.buf[: len(test_data)] = test_data

        # Should be able to read back
        read_data = bytes(block.buf[: len(test_data)])
        assert read_data == test_data

        fsm.close()

    def test_keep_alive_integration(self):
        """Test integration with KeepAlive functionality."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_seconds=2,
            pinned_memory=False,
        )

        # Keep alive should be running
        assert fsm.keep_alive is not None
        assert fsm.keep_alive.any_process_alive() is True

        # This handle's participant identity should be registered.
        with fsm.keep_alive.lock:
            assert fsm.participant_id in fsm.keep_alive.keep_alive_dict

        fsm.close()

    def test_same_process_handles_have_independent_participants(self):
        """Closing one handle must not withdraw another handle's heartbeat."""
        first = FlexibleSharedMemory(name=self.test_name, shm_size=self.test_size)
        second = FlexibleSharedMemory(name=self.test_name, shm_size=self.test_size)
        try:
            assert first.participant_id != second.participant_id
            first.close()
            assert second.keep_alive is not None
            with second.keep_alive.lock:
                assert first.participant_id not in second.keep_alive.keep_alive_dict
                assert second.participant_id in second.keep_alive.keep_alive_dict
        finally:
            if not first._closed:
                first.close()
            second.close()

    def test_pinned_mapping_uses_host_pin_registry(self, monkeypatch):
        """Pinned mappings are registered and released with the shared helper."""
        pin = mock.MagicMock()
        release = mock.MagicMock()
        monkeypatch.setattr("flextensor.shm.flexible_shm.HostPinRegistry.pin_in_place", pin)
        monkeypatch.setattr("flextensor.shm.flexible_shm.HostPinRegistry.release_all", release)

        fsm = FlexibleSharedMemory(name=self.test_name, shm_size=self.test_size, pinned_memory=True)
        pin.assert_called_once()

        fsm.close()
        release.assert_called_once_with()

    def test_creator_initializer_runs_only_for_creator(self):
        """Followers attach only after the creator has initialized its mapping."""
        creator_initializer = mock.MagicMock(side_effect=lambda shm: shm.block.buf.__setitem__(0, 42))
        follower_initializer = mock.MagicMock()
        creator = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            _creator_initializer=creator_initializer,
        )
        follower = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            _creator_initializer=follower_initializer,
        )
        try:
            creator_initializer.assert_called_once_with(creator)
            follower_initializer.assert_not_called()
            assert follower.block.buf[0] == 42
        finally:
            follower.close()
            creator.close()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_semaphore_protection(self, lock_class):
        """Test main semaphore protection."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Should be able to acquire and release main semaphore
        fsm.main_lock.acquire()

        # Do some work while holding semaphore
        test_data = b"Protected operation"
        fsm.block.buf[: len(test_data)] = test_data

        fsm.main_lock.release()

        # Verify data was written
        read_data = bytes(fsm.block.buf[: len(test_data)])
        assert read_data == test_data

        fsm.close()

    def test_pinned_memory_unavailable(self):
        """Unpinned mappings do not require CUDA host registration."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Should create successfully without pinned memory
        assert fsm.shm is not None
        assert fsm._host_pin_registry is None

        fsm.close()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_close_functionality(self, lock_class):
        """Test close functionality."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Should be functional before close
        assert fsm.keep_alive.any_process_alive()

        fsm.close()

        # Resources should be cleaned up
        assert fsm.main_lock is None
        assert fsm.shm is None

    def test_write_read_and_close(self):
        """Test that data can be written, read back, and the instance closed cleanly."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Write some data
        test_data = b"Shared data test"
        fsm.block.buf[: len(test_data)] = test_data

        # Verify we can read it back
        read_data = bytes(fsm.block.buf[: len(test_data)])
        assert read_data == test_data

        # Close the instance (single-process close unlinks the segment)
        fsm.close()

    def test_different_sizes(self):
        """Test with different memory sizes."""
        sizes = [1024, 4096]  # Reduced sizes for faster testing

        for size in sizes:
            test_name = f"{self.test_name}_{size}"
            # Clean up any leftover resources first
            self._cleanup_test_instance(test_name)
            time.sleep(0.1)  # Small delay to ensure cleanup completes

            fsm = FlexibleSharedMemory(
                name=test_name,
                shm_size=size,
                keep_alive_seconds=2,  # Short interval for faster test cleanup
                pinned_memory=False,
            )

            # Shared memory sizes are typically rounded up to page boundaries by the OS,
            # so we should verify it's at least as large as requested
            assert fsm.shm.size >= size
            assert fsm.shm_creator is True
            assert fsm.shm_size >= size  # Actual size may be rounded up

            # Should be able to use the memory
            test_pattern = b"x" * min(100, size)  # Don't fill entire memory for performance
            fsm.block.buf[: len(test_pattern)] = test_pattern
            read_pattern = bytes(fsm.block.buf[: len(test_pattern)])
            assert read_pattern == test_pattern

            fsm.close()
            time.sleep(0.1)  # Small delay before cleanup
            # Clean up after each iteration
            self._cleanup_test_instance(test_name)

    def _cleanup_test_instance(self, instance_name):
        """Helper to clean up a specific test instance."""
        try:
            from multiprocessing.shared_memory import SharedMemory

            import posix_ipc

            # Shared memory segments for this instance (must match FlexibleSharedMemory naming)
            shm_names = [
                instance_name,
                "sm_" + instance_name + "_kd",
                instance_name + "_km",
            ]

            # Semaphores / lock names for this instance
            sem_names = [
                instance_name,
                instance_name + "_ks",
            ]

            # Clean up semaphores
            for sem_name in sem_names:
                try:
                    sem = posix_ipc.Semaphore(sem_name)
                    sem.close()
                except (posix_ipc.ExistentialError, FileNotFoundError):
                    pass
                try:
                    sem = posix_ipc.Semaphore(sem_name)
                    sem.unlink()
                except (posix_ipc.ExistentialError, FileNotFoundError):
                    pass

            # Clean up shared memory segments
            for shm_name in shm_names:
                try:
                    shm = SharedMemory(name=shm_name, create=False)
                    shm.close()
                    shm.unlink()
                except (FileNotFoundError, FileExistsError):
                    pass
        except ImportError:
            pass

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_keep_alive_parameters(self, lock_class):
        """Test different keep alive parameters."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_dict_size=256 * 1024,  # 256KB
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        assert fsm.keep_alive.keep_alive_seconds == 2
        # Note: keep_alive_dict_size is used internally but not directly accessible

        fsm.close()

    def test_error_handling_invalid_size(self):
        """Test error handling with invalid sizes."""
        # Test with zero size
        with pytest.raises((ValueError, OSError)):
            FlexibleSharedMemory(
                name=self.test_name,
                shm_size=0,
                pinned_memory=False,
            )

        # Test with negative size
        with pytest.raises((ValueError, OSError)):
            FlexibleSharedMemory(
                name=self.test_name,
                shm_size=-1,
                pinned_memory=False,
            )

    def test_memory_isolation(self):
        """Test that different named memories are isolated."""
        name1 = self.test_name + "_1"
        name2 = self.test_name + "_2"

        # Clean up any leftover resources
        self._cleanup_test_instance(name1)
        self._cleanup_test_instance(name2)

        fsm1 = FlexibleSharedMemory(
            name=name1,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        fsm2 = FlexibleSharedMemory(
            name=name2,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Write different data to each
        data1 = b"Memory 1 data"
        data2 = b"Memory 2 data"

        fsm1.block.buf[: len(data1)] = data1
        fsm2.block.buf[: len(data2)] = data2

        # Each should retain its own data
        read_data1 = bytes(fsm1.block.buf[: len(data1)])
        read_data2 = bytes(fsm2.block.buf[: len(data2)])

        assert read_data1 == data1
        assert read_data2 == data2
        assert read_data1 != read_data2

        fsm2.close()
        fsm1.close()

        # Clean up after test
        self._cleanup_test_instance(name1)
        self._cleanup_test_instance(name2)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_init_failure_cleanup_creator_unlinks_segment(self, lock_class):
        """Test that when creator fails during init, the shared memory segment is unlinked.

        This prevents orphaned shared memory segments when initialization fails after
        the segment is created but before KeepAlive registration completes.
        """
        from multiprocessing.shared_memory import SharedMemory

        test_name = self.test_name + "_init_fail"
        self._cleanup_test_instance(test_name)

        # Mock KeepAlive to raise an exception after shared memory is created
        def failing_keepalive_init(self, *args, **kwargs):
            raise RuntimeError("Simulated KeepAlive initialization failure")

        with (
            mock.patch.object(KeepAlive, "__init__", failing_keepalive_init),
            pytest.raises(RuntimeError, match="Simulated KeepAlive initialization failure"),
        ):
            FlexibleSharedMemory(
                name=test_name,
                shm_size=self.test_size,
                lock_class=lock_class,
                pinned_memory=False,
            )

        # Verify the shared memory segment was cleaned up (unlinked)
        # Attempting to connect should fail with FileNotFoundError
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=test_name, create=False)

        # Clean up any remaining resources
        self._cleanup_test_instance(test_name)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_init_failure_cleanup_connector_preserves_segment(self, lock_class):
        """Test that when connector fails during init, the shared memory segment is NOT unlinked.

        This ensures that if a process connecting to existing shared memory fails,
        it doesn't destroy the segment that other processes are using.
        """
        from multiprocessing.shared_memory import SharedMemory

        test_name = self.test_name + "_conn_fail"
        self._cleanup_test_instance(test_name)

        # First, create the shared memory segment successfully
        fsm_creator = FlexibleSharedMemory(
            name=test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_seconds=30,
            pinned_memory=False,
        )

        # Write test data to verify segment integrity later
        test_data = b"Creator data"
        fsm_creator.block.buf[: len(test_data)] = test_data

        # Now try to connect with a connector that will fail during KeepAlive init
        # We use shm_size=0 to indicate we're connecting (not creating)
        # But we need to make KeepAlive fail for the connector

        original_init = KeepAlive.__init__

        call_count = [0]

        def failing_on_second_keepalive_init(self_ka, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:  # Fail on second call (connector)
                raise RuntimeError("Simulated connector KeepAlive failure")
            return original_init(self_ka, *args, **kwargs)

        # Reset call count and patch
        call_count[0] = 1  # Set to 1 so next call (connector) will fail

        with (
            mock.patch.object(KeepAlive, "__init__", failing_on_second_keepalive_init),
            pytest.raises(RuntimeError, match="Simulated connector KeepAlive failure"),
        ):
            # Try to connect to existing memory (shm_size > 0 but segment already exists)
            FlexibleSharedMemory(
                name=test_name,
                shm_size=self.test_size,  # Will connect since segment exists
                lock_class=lock_class,
                pinned_memory=False,
            )

        # Verify the original shared memory segment still exists and has correct data
        # The creator's segment should NOT have been unlinked
        assert fsm_creator.shm is not None
        read_data = bytes(fsm_creator.block.buf[: len(test_data)])
        assert read_data == test_data

        # Also verify we can still connect to the segment
        try:
            check_shm = SharedMemory(name=test_name, create=False)
            check_shm.close()
        except FileNotFoundError:
            pytest.fail("Shared memory segment was incorrectly unlinked by failed connector")

        # Clean up
        fsm_creator.close()
        self._cleanup_test_instance(test_name)

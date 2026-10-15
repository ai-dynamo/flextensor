# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-process unit tests for MultiprocessCondition class."""

import contextlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import posix_ipc
import pytest

import flextensor.shm.multiprocess_condition as condition_module
from flextensor.shm import MultiprocessCondition, ProcessFileLock, SemaphoreLock


class TestMultiprocessConditionSingle:
    """Single-process tests for MultiprocessCondition class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "mpc_s"  # Short name to avoid POSIX name limits
        # Clean up any leftovers before starting
        self._cleanup_resources(self.test_name)

    def teardown_method(self):
        """Cleanup after each test method."""
        # Cleanup any leftover resources
        self._cleanup_resources(self.test_name)

    def _cleanup_resources(self, name):
        """Helper to cleanup test resources."""
        try:
            import posix_ipc

            # Cleanup main semaphore (same name as the condition)
            try:
                sem = posix_ipc.Semaphore(name)
                sem.unlink()
                sem.close()
            except (posix_ipc.ExistentialError, FileNotFoundError):
                pass

            # Cleanup any process-specific semaphores
            pid = os.getpid()
            try:
                sem_name = f"/{name}_{pid}"
                sem = posix_ipc.Semaphore(sem_name)
                sem.unlink()
                sem.close()
            except (posix_ipc.ExistentialError, FileNotFoundError):
                pass
        except ImportError:
            pass

        try:
            from multiprocessing.shared_memory import SharedMemory

            # Cleanup shared memory for SharedMultiString (same name as the condition)
            try:
                shm = SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        except ImportError:
            pass

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_multiprocess_condition_creation(self, lock_class):
        """Test MultiprocessCondition creation."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        assert condition.name == self.test_name
        assert condition.notification_list is not None
        assert condition.lock is not None

        # Should initially not be ready
        assert not condition.notification_list.is_ready()

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_notify_ready_functionality(self, lock_class):
        """Test notify_ready functionality."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Initially not ready
        assert not condition.notification_list.is_ready()

        # Notify ready
        condition.notify_ready()

        # Should now be ready
        assert condition.notification_list.is_ready()

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_notify_ready_signals_every_registration_for_one_process(self, lock_class):
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)
        waiter = SemaphoreLock(f"{self.test_name}_{os.getpid()}", locked=True)
        try:
            with condition.lock:
                condition.notification_list.append(str(os.getpid()))
                condition.notification_list.append(str(os.getpid()))

            condition.notify_ready()

            waiter.acquire(non_blocking=True)
            waiter.acquire(non_blocking=True)
            with pytest.raises(posix_ipc.ExistentialError):
                posix_ipc.Semaphore(waiter.name).close()
        finally:
            waiter.unlink()
            waiter.close()
            condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_ready_waiter_leaves_semaphore_name_for_notifier(self, monkeypatch, lock_class):
        """A waiter may observe readiness before the notifier opens its semaphore."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)
        append = condition.notification_list.append

        def ready_after_registration(pid):
            append(pid)
            condition.notification_list.set_ready()

        monkeypatch.setattr(condition.notification_list, "append", ready_after_registration)
        try:
            condition.wait_for_ready()

            waiter_name = f"/{self.test_name}_{os.getpid()}"
            posix_ipc.Semaphore(waiter_name).close()
            condition.notify_ready()
            with pytest.raises(posix_ipc.ExistentialError):
                posix_ipc.Semaphore(waiter_name).close()
        finally:
            condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_same_process_waiters_register_and_wake_together(self, monkeypatch, lock_class):
        """Two local waiters must register without blocking the notifier's lock."""
        acquire = SemaphoreLock.acquire

        def bounded_acquire(self, non_blocking=False, timeout=None):
            return acquire(self, non_blocking=non_blocking, timeout=5.0 if timeout is None else timeout)

        # Bound real semaphore waits so a regression cannot hang the test runner.
        monkeypatch.setattr(SemaphoreLock, "acquire", bounded_acquire)
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)
        registered = threading.Event()
        append = condition.notification_list.append

        def record_registration(pid):
            append(pid)
            if len(condition.notification_list.get_list()) == 2:
                registered.set()

        monkeypatch.setattr(condition.notification_list, "append", record_registration)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                waiters = [executor.submit(condition.wait_for_ready) for _ in range(2)]
                assert registered.wait(timeout=5), "Both waiters must register before notification"

                condition.notify_ready()

                for waiter in waiters:
                    waiter.result(timeout=5)
            assert condition.notification_list.get_list() == [str(os.getpid())] * 2
        finally:
            condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_wait_for_ready_immediate(self, lock_class):
        """Test wait_for_ready when already ready."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Set ready first
        condition.notify_ready()

        # wait_for_ready should return immediately
        start_time = time.time()
        condition.wait_for_ready()
        wait_time = time.time() - start_time

        # Should be very fast since it's already ready
        assert wait_time < 0.1

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize(
        ("method_name", "initially_ready"),
        [("wait_for_ready", True), ("notify_ready", False)],
    )
    def test_ready_method_keeps_validated_resources_if_fields_are_cleared(self, method_name, initially_ready):
        """A concurrent close cannot replace resources after validation."""
        condition = MultiprocessCondition.__new__(MultiprocessCondition)
        condition.name = self.test_name

        class NotificationList:
            ready = initially_ready

            def is_ready(self):
                return self.ready

            def set_ready(self):
                self.ready = True

            def get_list(self):
                return []

        notification_list = NotificationList()
        condition.notification_list = notification_list

        class ClosingRaceLock:
            def __enter__(self):
                condition.lock = None
                condition.notification_list = None

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        condition.lock = ClosingRaceLock()

        getattr(condition, method_name)()

        assert notification_list.ready

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_semaphore_protection(self, lock_class):
        """Test that semaphore properly protects critical sections."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Test acquiring semaphore
        condition.lock.acquire()

        # Should be able to access notification_list while holding semaphore
        condition.notification_list.append("test_entry")
        strings = condition.notification_list.get_list()
        assert "test_entry" in strings

        condition.lock.release()

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_notification_list_interaction(self, lock_class):
        """Test interaction with underlying notification list."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Add process ID to notification list (simulating wait_for_ready behavior)
        with condition.lock:
            condition.notification_list.append(str(os.getpid()))
            strings = condition.notification_list.get_list()
            assert str(os.getpid()) in strings

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_connect_to_existing_condition(self, lock_class):
        """Test connecting to existing multiprocess condition."""
        # Create first condition
        condition1 = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)
        condition1.notification_list.append("test_data")

        # Connect to existing condition
        condition2 = MultiprocessCondition(name=self.test_name, is_creator=False, lock_class=lock_class)

        # Should see the same data
        strings = condition2.notification_list.get_list()
        assert "test_data" in strings

        condition2.close()
        condition1.close()
        condition1.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_close_functionality(self, lock_class):
        """Test close functionality."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Verify it's working
        condition.notification_list.append("test")
        assert "test" in condition.notification_list.get_list()

        # Close should clean up
        condition.close()

        # notification_list should be closed
        with pytest.raises((ValueError, AttributeError)):
            condition.notification_list.append("should_fail")

        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_unlink_functionality(self, lock_class):
        """Test unlink functionality."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # unlink() before close() should work and prevent reconnection
        condition.unlink()

        assert condition.notification_list is None
        assert condition.lock is None
        condition.unlink()

        with pytest.raises((FileNotFoundError, ValueError)):
            MultiprocessCondition(name=self.test_name, is_creator=False, lock_class=lock_class)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_unlink_names_keeps_local_handles_open(self, lock_class):
        """Global names disappear while the existing mapping and lock stay usable."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        try:
            condition.unlink_names()

            assert condition.notification_list is not None
            assert condition.lock is not None
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=self.test_name, create=False).close()
            if isinstance(condition.lock, SemaphoreLock):
                with pytest.raises(posix_ipc.ExistentialError):
                    posix_ipc.Semaphore(condition.lock.name).close()
            else:
                with pytest.raises(FileNotFoundError), Path(condition.lock.path).open("rb"):
                    pass

            condition.lock.acquire(non_blocking=True)
            condition.lock.release()
            condition.notification_list.set_ready()
            assert condition.notification_list.is_ready()
        finally:
            condition.unlink()

    def test_unlink_closes_semaphore_lock_handle(self):
        """Unlinking a condition must not leave its semaphore handle usable."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=SemaphoreLock)
        lock = condition.lock

        try:
            condition.unlink()

            with pytest.raises(posix_ipc.ExistentialError, match="closed"):
                lock.acquire(non_blocking=True)
        finally:
            with contextlib.suppress(posix_ipc.ExistentialError):
                lock.close()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_unlink_after_close_removes_names(self, lock_class):
        """Closing local handles must not prevent a later unlink."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)
        lock = condition.lock

        try:
            condition.close()
            condition.unlink()

            with pytest.raises(FileNotFoundError):
                SharedMemory(name=self.test_name, create=False).close()
            if isinstance(lock, SemaphoreLock):
                with pytest.raises(posix_ipc.ExistentialError):
                    posix_ipc.Semaphore(lock.name).close()
            else:
                with pytest.raises(FileNotFoundError), Path(lock.path).open("rb"):
                    pass
        finally:
            condition.unlink()

    def test_constructor_rolls_back_created_notification_list_when_lock_fails(self):
        """A failed lock constructor must not strand its newly created SHM name."""

        class FailingLock(ProcessFileLock):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("lock failed")

        with pytest.raises(RuntimeError, match="lock failed"):
            MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=FailingLock)

        with pytest.raises(FileNotFoundError):
            SharedMemory(name=self.test_name, create=False)

    def test_notify_ready_does_not_recreate_a_vanished_waiter(self):
        """Stale waiter entries are benign and cannot recreate semaphore names."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True)
        condition.notification_list.append("gone")

        try:
            condition.notify_ready()

            with pytest.raises(posix_ipc.ExistentialError):
                posix_ipc.Semaphore(f"/{self.test_name}_gone")
        finally:
            condition.unlink()

    @pytest.mark.parametrize("failure_stage", ["signal", "close"])
    def test_notify_ready_finishes_other_deliveries_before_raising(  # noqa: C901
        self,
        monkeypatch,
        failure_stage,
    ):
        """One failed waiter cannot prevent later waiters from being notified."""
        condition = MultiprocessCondition.__new__(MultiprocessCondition)
        condition.name = self.test_name
        events = []

        class NotificationList:
            ready = False

            def set_ready(self):
                self.ready = True

            def get_list(self):
                return ["bad", "good"]

        class Lock:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        class NotificationLock:
            def __init__(self, name, *, locked, create):
                assert locked is False
                assert create is False
                self.name = name
                events.append((name, "attach"))

            def signal(self):
                events.append((self.name, "signal"))
                if self.name.endswith("_bad") and failure_stage == "signal":
                    raise OSError("signal failed")

            def unlink_name(self):
                events.append((self.name, "unlink"))

            def close(self):
                events.append((self.name, "close"))
                if self.name.endswith("_bad") and failure_stage == "close":
                    raise OSError("close failed")

        condition.notification_list = NotificationList()
        condition.lock = Lock()
        monkeypatch.setattr(condition_module, "SemaphoreLock", NotificationLock)

        with pytest.raises(OSError, match=failure_stage):
            condition.notify_ready()

        assert condition.notification_list.ready
        good_name = f"{self.test_name}_good"
        assert (good_name, "signal") in events
        assert (good_name, "unlink") in events
        assert (good_name, "close") in events

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_close_lock_functionality(self, lock_class):
        """Test close_lock functionality."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        assert condition.lock is not None
        lock = condition.lock

        # Close semaphore specifically
        condition.close_lock()

        assert condition.lock is None

        # Should not be able to use semaphore after closing
        with pytest.raises(AttributeError):
            condition.lock.acquire()

        condition.close()
        condition.unlink()
        if isinstance(lock, SemaphoreLock):
            with pytest.raises(posix_ipc.ExistentialError):
                posix_ipc.Semaphore(lock.name)
        else:
            assert not Path(lock.path).exists()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    @pytest.mark.parametrize("create_flag", [True, False])
    def test_different_creation_modes(self, create_flag, lock_class):
        """Test different creation modes."""
        if create_flag:
            # Creating new condition
            condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)
            assert condition.notification_list is not None
            assert condition.lock is not None
            condition.close()
            condition.unlink()
        else:
            # First create a condition to connect to
            condition1 = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

            # Then connect without creating
            condition2 = MultiprocessCondition(name=self.test_name, is_creator=False, lock_class=lock_class)
            assert condition2.notification_list is not None
            assert condition2.lock is not None

            condition2.close()
            condition1.close()
            condition1.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_ready_state_persistence(self, lock_class):
        """Test that ready state persists across operations."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Set ready
        condition.notify_ready()
        assert condition.notification_list.is_ready()

        # Add more data - ready state should persist
        with condition.lock:
            condition.notification_list.append("additional_data")

        # Should still be ready
        assert condition.notification_list.is_ready()

        condition.close()
        condition.unlink()

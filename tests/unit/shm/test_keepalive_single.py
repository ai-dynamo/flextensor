# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-process unit tests for KeepAlive class."""

import time
from unittest.mock import patch

import pytest

from flextensor.shm import KeepAlive, ProcessFileLock, SemaphoreLock
from flextensor.shm.flexible_shm import KeepAliveDict


def test_prepare_slot_rejects_exhausted_timestamp_view():
    keep_alive_dict = object.__new__(KeepAliveDict)
    keep_alive_dict.dict = {111: 0}
    keep_alive_dict.shm_view = [0.0]
    keep_alive_dict.size = 8

    with pytest.raises(
        RuntimeError,
        match=r"timestamp slots exhausted: participant_id=222, slot=1, timestamp_capacity=1",
    ):
        keep_alive_dict.prepare_slot(222)

    assert keep_alive_dict.dict == {111: 0}
    assert keep_alive_dict.shm_view == [0.0]


class TestKeepAliveSingle:
    """Single-process tests for KeepAlive class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "ka_s"  # Short name to avoid POSIX name limits
        self.process_id = 12345  # Integer process ID for tests
        self.keep_alive_seconds = 2  # Short timeout for tests
        # Clean up any leftovers before starting
        self._cleanup()

    def teardown_method(self):
        """Cleanup after each test method."""
        self._cleanup()

    def _cleanup(self):
        """Cleanup after each test method.

        Use same names as KeepAlive: lock_name=f'{name}_s',
        dict_name=f'{name}_d', dict_meta_name=f'{name}_m'.
        """
        try:
            from multiprocessing.shared_memory import SharedMemory

            import posix_ipc

            # Cleanup semaphore (same name as KeepAlive lock_name)
            lock_name = f"{self.test_name}_s"
            sem_name = lock_name if lock_name.startswith("/") else f"/{lock_name}"
            try:
                sem = posix_ipc.Semaphore(sem_name)
                sem.close()
                sem.unlink()
            except (posix_ipc.ExistentialError, FileNotFoundError):
                pass

            # Cleanup KeepAliveDict shared memory (dict_name and meta_name)
            dict_name = f"{self.test_name}_d"
            meta_name = f"{self.test_name}_m"
            for shm_name in (dict_name, meta_name):
                try:
                    shm = SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass
        except ImportError:
            pass

    def test_real_dict_capacity_error_is_actionable(self):
        """Dictionary storage exhaustion must use the keep-alive capacity error."""
        keep_alive_dict = KeepAliveDict(
            name=f"{self.test_name}_d",
            size=8,
            is_creator=True,
            meta_name=f"{self.test_name}_m",
        )

        try:
            with pytest.raises(
                RuntimeError,
                match=r"mapping storage exhausted: participant_id=12345, slot=0, keep_alive_dict_size=8",
            ):
                keep_alive_dict.prepare_slot(self.process_id)

            assert list(keep_alive_dict.shm_view) == [0.0]
        finally:
            keep_alive_dict.unlink()

    def test_capacity_error_is_not_relabelled_as_lock_failure(self, monkeypatch):
        """KeepAlive initialization must preserve capacity diagnostics and close resources."""
        instances = []
        original_init = KeepAliveDict.__init__

        def capture_instance(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            instances.append(self)

        monkeypatch.setattr(KeepAliveDict, "__init__", capture_instance)

        try:
            with pytest.raises(
                RuntimeError,
                match=r"mapping storage exhausted: participant_id=12345, slot=0, keep_alive_dict_size=8",
            ):
                KeepAlive(
                    name=self.test_name,
                    process_id=self.process_id,
                    keep_alive_dict_size=8,
                    is_creator=True,
                )

            assert instances[0]._closed is True  # noqa: SLF001
        finally:
            if instances:
                instances[0].unlink()

    def test_lock_construction_failure_closes_keep_alive_dict(self, monkeypatch):
        """A lock construction error must close the dictionary opened earlier."""
        instances = []
        original_init = KeepAliveDict.__init__

        def capture_instance(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            instances.append(self)

        monkeypatch.setattr(KeepAliveDict, "__init__", capture_instance)

        try:
            with (
                patch.object(ProcessFileLock, "__init__", side_effect=OSError("lock construction failed")),
                pytest.raises(OSError, match="lock construction failed"),
            ):
                KeepAlive(
                    name=self.test_name,
                    process_id=self.process_id,
                    is_creator=True,
                )

            assert instances[0]._closed is True  # noqa: SLF001
        finally:
            if instances:
                instances[0].unlink()

    @pytest.mark.parametrize("keep_alive_seconds", [0, -1, float("nan"), float("inf")])
    def test_invalid_keep_alive_seconds_is_rejected(self, keep_alive_seconds):
        """An invalid interval must fail before starting a busy-loop heartbeat thread."""
        with (
            patch.object(KeepAlive, "_start_keep_alive_thread"),
            pytest.raises(ValueError, match="positive"),
        ):
            KeepAlive(
                name=self.test_name,
                process_id=self.process_id,
                keep_alive_seconds=keep_alive_seconds,
                is_creator=True,
            )

    def test_thread_start_failure_removes_registered_process(self):
        """A failed participant must not leave a live-looking heartbeat behind."""
        creator = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            is_creator=True,
        )
        failed_process_id = self.process_id + 1

        try:
            with (
                patch.object(KeepAlive, "_start_keep_alive_thread", side_effect=RuntimeError("thread start failed")),
                pytest.raises(RuntimeError, match="thread start failed"),
            ):
                KeepAlive(
                    name=self.test_name,
                    process_id=failed_process_id,
                    is_creator=False,
                )

            with creator.lock:
                assert failed_process_id not in creator.keep_alive_dict
        finally:
            creator.close(any_process_alive=False)

    def test_slot_failure_preserves_existing_process(self):
        """Failed slot allocation must not remove another live registration."""
        creator = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            is_creator=True,
        )

        try:
            with (
                patch.object(KeepAliveDict, "prepare_slot", side_effect=RuntimeError("capacity exhausted")),
                pytest.raises(RuntimeError, match="capacity exhausted"),
            ):
                KeepAlive(
                    name=self.test_name,
                    process_id=self.process_id,
                    is_creator=False,
                )

            with creator.lock:
                assert self.process_id in creator.keep_alive_dict
        finally:
            creator.close(any_process_alive=False)

    def test_thread_start_failure_preserves_existing_process(self):
        """Failed thread startup must preserve the existing process heartbeat."""
        with patch("flextensor.shm.flexible_shm.time.time", return_value=100):
            creator = KeepAlive(
                name=self.test_name,
                process_id=self.process_id,
                is_creator=True,
            )

            try:
                with (
                    patch.object(
                        KeepAlive,
                        "_start_keep_alive_thread",
                        side_effect=RuntimeError("thread start failed"),
                    ),
                    pytest.raises(RuntimeError, match="thread start failed"),
                ):
                    KeepAlive(
                        name=self.test_name,
                        process_id=self.process_id,
                        is_creator=False,
                    )

                with creator.lock:
                    creator.keep_alive_dict.set_by_slot(creator.slot, 200)
                with patch("flextensor.shm.flexible_shm.time.time", return_value=200):
                    assert creator.is_process_alive(self.process_id) is True
            finally:
                creator.close(any_process_alive=False)

    def test_thread_start_failure_does_not_refresh_stale_process(self):
        """Failed thread startup must not make a stale registration look alive."""
        with patch("flextensor.shm.flexible_shm.time.time", return_value=100):
            creator = KeepAlive(
                name=self.test_name,
                process_id=self.process_id,
                keep_alive_seconds=10,
                is_creator=True,
            )

        creator.stop_event.set()
        creator.keep_alive_thread.join(timeout=2)
        creator.keep_alive_thread = None

        try:
            with (
                patch("flextensor.shm.flexible_shm.time.time", return_value=200),
                patch.object(KeepAlive, "_start_keep_alive_thread", side_effect=RuntimeError("thread start failed")),
                pytest.raises(RuntimeError, match="thread start failed"),
            ):
                KeepAlive(
                    name=self.test_name,
                    process_id=self.process_id,
                    keep_alive_seconds=10,
                    is_creator=False,
                )

            with patch("flextensor.shm.flexible_shm.time.time", return_value=200):
                assert creator.is_process_alive(self.process_id) is False
        finally:
            creator.close(any_process_alive=False)

    @pytest.mark.parametrize("closed_index", [0, 1])
    def test_duplicate_process_registration_survives_one_close(self, closed_index):
        """One instance closing must not remove another instance's live registration."""
        instances = [
            KeepAlive(
                name=self.test_name,
                process_id=self.process_id,
                keep_alive_seconds=100,
                is_creator=True,
            ),
            KeepAlive(
                name=self.test_name,
                process_id=self.process_id,
                keep_alive_seconds=100,
                is_creator=False,
            ),
        ]
        closed = instances.pop(closed_index)
        remaining = instances[0]

        closed.close(any_process_alive=True)
        try:
            assert remaining.is_process_alive(self.process_id) is True
        finally:
            remaining.close(any_process_alive=False)

    def test_is_process_alive_checks_only_requested_pid(self):
        """A stale requested PID is dead even while another heartbeat is fresh."""
        keep_alive = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            keep_alive_seconds=100,
            is_creator=True,
        )

        try:
            with keep_alive.lock:
                other_slot = keep_alive.keep_alive_dict.prepare_slot(self.process_id + 1)
                keep_alive.keep_alive_dict.set_by_slot(keep_alive.slot, time.time() - 200)
                keep_alive.keep_alive_dict.set_by_slot(other_slot, time.time())

            assert keep_alive.is_process_alive(self.process_id) is False
            assert keep_alive.is_process_alive(self.process_id + 1) is True
        finally:
            keep_alive.close(any_process_alive=False)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_keepalive_initialization(self, lock_class):
        """Test KeepAlive initialization."""
        keep_alive = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            keep_alive_seconds=self.keep_alive_seconds,
            is_creator=True,
            lock_class=lock_class,
        )

        assert keep_alive.name == self.test_name
        assert keep_alive.process_id == self.process_id
        assert keep_alive.keep_alive_seconds == self.keep_alive_seconds
        assert keep_alive.keep_alive_dict is not None
        assert keep_alive.lock is not None
        assert keep_alive.keep_alive_thread is not None
        assert keep_alive.keep_alive_thread.is_alive()

        # Check that process is in the dictionary
        with keep_alive.lock:
            assert self.process_id in keep_alive.keep_alive_dict

        keep_alive.close(any_process_alive=False)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_keepalive_thread_updates_timestamp(self, lock_class):
        """Test that keep alive thread updates timestamp."""
        keep_alive = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            keep_alive_seconds=self.keep_alive_seconds,
            is_creator=True,
            lock_class=lock_class,
        )

        # Get initial timestamp via items() (pid, timestamp)
        with keep_alive.lock:
            ts_by_pid = dict(keep_alive.keep_alive_dict.items())
            initial_timestamp = ts_by_pid.get(self.process_id)
        assert initial_timestamp is not None

        # Wait for thread to update (half of keep_alive_seconds)
        time.sleep(self.keep_alive_seconds / 2 + 0.1)

        # Check timestamp was updated
        with keep_alive.lock:
            ts_by_pid = dict(keep_alive.keep_alive_dict.items())
            updated_timestamp = ts_by_pid.get(self.process_id)
        assert updated_timestamp is not None

        assert updated_timestamp > initial_timestamp

        keep_alive.close(any_process_alive=False)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_any_process_alive_true(self, lock_class):
        """Test any_process_alive returns True when processes are alive."""
        keep_alive = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            keep_alive_seconds=self.keep_alive_seconds,
            is_creator=True,
            lock_class=lock_class,
        )

        assert keep_alive.any_process_alive() is True

        keep_alive.close(any_process_alive=False)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_any_process_alive_false_after_timeout(self, lock_class):
        """Test any_process_alive returns False after timeout."""
        keep_alive = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            keep_alive_seconds=self.keep_alive_seconds,
            is_creator=True,
            lock_class=lock_class,
        )

        # Stop only the keep-alive thread (do not call stop() - it would unregister
        # the process). We need the process to stay in the dict with an old timestamp.
        keep_alive.stop_event.set()
        if keep_alive.keep_alive_thread is not None:
            keep_alive.keep_alive_thread.join(timeout=2)
            keep_alive.keep_alive_thread = None

        # Manually set an old timestamp via slot (process still in dict)
        current_time = int(time.time())
        old_timestamp = current_time - self.keep_alive_seconds - 1

        with keep_alive.lock:
            slot = keep_alive.keep_alive_dict.dict[self.process_id]
            keep_alive.keep_alive_dict.set_by_slot(slot, old_timestamp)

        # Should return False and clean up old process
        assert keep_alive.any_process_alive() is False

        # Process should be removed from dictionary
        with keep_alive.lock:
            assert self.process_id not in keep_alive.keep_alive_dict

        keep_alive.close(any_process_alive=False)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_stop_functionality(self, lock_class):
        """Test stop functionality."""
        keep_alive = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            keep_alive_seconds=self.keep_alive_seconds,
            is_creator=True,
            lock_class=lock_class,
        )

        assert keep_alive.keep_alive_thread.is_alive()

        keep_alive.stop()

        # Thread should be stopped and joined
        assert keep_alive.keep_alive_thread is None or not keep_alive.keep_alive_thread.is_alive()

        # Process should be removed from dictionary
        with keep_alive.lock:
            assert self.process_id not in keep_alive.keep_alive_dict

        keep_alive.close(any_process_alive=False)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_close_functionality(self, lock_class):
        """Test close functionality."""
        keep_alive = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            keep_alive_seconds=self.keep_alive_seconds,
            is_creator=True,
            lock_class=lock_class,
        )

        keep_alive.close(any_process_alive=False)

        # Should clean up resources
        assert keep_alive.keep_alive_dict is None
        assert keep_alive.lock is None

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_semaphore_locking(self, lock_class):
        """Test that semaphore properly protects shared resources."""
        keep_alive = KeepAlive(
            name=self.test_name,
            process_id=self.process_id,
            keep_alive_seconds=self.keep_alive_seconds,
            is_creator=True,
            lock_class=lock_class,
        )

        # Test acquiring and releasing semaphore
        keep_alive.lock.acquire()

        # Should be able to access dictionary while holding semaphore (via items())
        ts_by_pid = dict(keep_alive.keep_alive_dict.items())
        test_value = ts_by_pid.get(self.process_id)
        assert test_value is not None

        keep_alive.lock.release()

        keep_alive.close(any_process_alive=False)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_any_process_alive_near_second_boundary(self, lock_class):
        """Regression test for #90: any_process_alive() must not mark a freshly created
        process as stale just because initialization and the check straddle a second boundary.

        With integer timestamps, init at t=T.99 stores timestamp=T, and a check at
        t=(T+1).01 gives current_time=T+1, so ``T < (T+1) - 0.1 = T+0.9`` is True
        (incorrectly stale).  With float timestamps the comparison is
        ``T.99 < T+1.01 - 0.1 = T+0.91`` which is False (correctly alive).
        """
        t_init = 1_000_000_000.99  # just before a second boundary
        t_check = 1_000_000_001.01  # 20 ms later, just after the boundary

        with patch("flextensor.shm.flexible_shm.time.time") as mock_time:
            mock_time.return_value = t_init
            keep_alive = KeepAlive(
                name=self.test_name,
                process_id=self.process_id,
                keep_alive_seconds=0.1,
                is_creator=True,
                lock_class=lock_class,
            )

            # Stop the keepalive thread so it cannot refresh the timestamp
            keep_alive.stop_event.set()
            keep_alive.keep_alive_thread.join(timeout=2)
            keep_alive.keep_alive_thread = None

            # Only 20 ms have passed — the process must still be considered alive
            mock_time.return_value = t_check
            assert keep_alive.any_process_alive() is True

        keep_alive.close(any_process_alive=False)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_existing_semaphore_connection(self, lock_class):
        """Test connecting to existing semaphore."""
        # Create initial keep alive with semaphore
        keep_alive1 = KeepAlive(
            name=self.test_name,
            process_id=self.process_id + 1,
            keep_alive_seconds=self.keep_alive_seconds,
            is_creator=True,
            lock_class=lock_class,
        )

        # Connect to existing semaphore
        keep_alive2 = KeepAlive(
            name=self.test_name,
            process_id=self.process_id + 2,
            keep_alive_seconds=self.keep_alive_seconds,
            is_creator=False,
            lock_class=lock_class,
        )

        # Both should be able to access the same dictionary
        with keep_alive1.lock:
            assert len(keep_alive1.keep_alive_dict.dict.keys()) == 2

        with keep_alive2.lock:
            assert len(keep_alive2.keep_alive_dict.dict.keys()) == 2

        keep_alive2.close(any_process_alive=True)
        keep_alive1.close(any_process_alive=False)

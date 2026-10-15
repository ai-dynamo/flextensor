# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multithreaded unit tests for sync primitives.

These tests verify thread-safety of SemaphoreLock and ProcessFileLock under concurrent access.
Both lock implementations are thread-safe and support inter-thread synchronization.
"""

import contextlib
import tempfile
import threading
import time
from pathlib import Path

import posix_ipc
import pytest

from flextensor.shm import ProcessFileLock, SemaphoreLock


class TestSyncPrimitivesMultithread:
    """Multithreaded tests for thread-safe locks: SemaphoreLock and ProcessFileLock."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "sync_mt_test"
        self._cleanup()

    def teardown_method(self):
        """Cleanup after each test method."""
        self._cleanup()

    def _cleanup(self):
        """Cleanup any leftover resources."""
        # Cleanup semaphore - must unlink to remove from system
        try:
            sem = posix_ipc.Semaphore(f"/{self.test_name}")
            sem.unlink()
        except posix_ipc.ExistentialError:
            pass

        # Cleanup file lock
        lock_path = Path(tempfile.gettempdir()) / f"{self.test_name}.lock"
        with contextlib.suppress(FileNotFoundError, OSError):
            lock_path.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_concurrent_acquire_release(self, lock_class):
        """Test multiple threads contending for the same lock.

        Verifies that only one thread holds the lock at a time.
        """
        lock = lock_class(self.test_name, locked=False)
        results = []
        errors = []
        num_threads = 10
        iterations_per_thread = 5

        def worker(thread_id):
            try:
                for i in range(iterations_per_thread):
                    lock.acquire()
                    results.append(f"enter-{thread_id}-{i}")
                    time.sleep(0.001)  # Small delay to increase contention
                    results.append(f"exit-{thread_id}-{i}")
                    lock.release()
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Verify no errors occurred
        assert not errors, f"Errors occurred: {errors}"

        # Verify enter/exit pairs are properly nested (no interleaving within critical section)
        # Each "enter-X-Y" should be immediately followed by "exit-X-Y"
        for i in range(0, len(results), 2):
            enter = results[i]
            exit_ = results[i + 1]
            assert enter.startswith("enter-"), f"Expected enter at {i}, got {enter}"
            assert exit_.startswith("exit-"), f"Expected exit at {i + 1}, got {exit_}"
            # Extract thread_id and iteration from enter/exit
            enter_suffix = enter[6:]  # Remove "enter-"
            exit_suffix = exit_[5:]  # Remove "exit-"
            assert enter_suffix == exit_suffix, f"Mismatched enter/exit: {enter} vs {exit_}"

        lock.close()
        lock.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_context_manager_under_contention(self, lock_class):
        """Test context manager usage with multiple threads.

        Verifies that `with lock:` properly acquires and releases under contention.
        """
        lock = lock_class(self.test_name, locked=False)
        counter = {"value": 0}
        num_threads = 20
        increments_per_thread = 50

        def worker():
            for _ in range(increments_per_thread):
                with lock:
                    current = counter["value"]
                    time.sleep(0.0001)  # Small delay to increase race condition probability
                    counter["value"] = current + 1

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # If locks work correctly, counter should equal total increments
        expected = num_threads * increments_per_thread
        assert counter["value"] == expected, f"Race condition detected: {counter['value']} != {expected}"

        lock.close()
        lock.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_stress_no_deadlock(self, lock_class):
        """Stress test with high thread count and rapid acquire/release cycles.

        Verifies no deadlocks occur under heavy contention.
        """
        lock = lock_class(self.test_name, locked=False)
        completed = {"count": 0}
        completed_lock = threading.Lock()
        num_threads = 50
        cycles_per_thread = 20

        def worker():
            for _ in range(cycles_per_thread):
                lock.acquire()
                lock.release()
            with completed_lock:
                completed["count"] += 1

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        elapsed = time.time() - start_time

        # All threads should complete within timeout (no deadlock)
        assert completed["count"] == num_threads, (
            f"Deadlock suspected: only {completed['count']}/{num_threads} completed"
        )
        assert elapsed < 60, f"Test took too long: {elapsed}s (possible deadlock)"

        lock.close()
        lock.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_mutual_exclusion_correctness(self, lock_class):
        """Verify critical section protection with shared counter.

        Uses a non-atomic read-modify-write to verify mutual exclusion.
        """
        lock = lock_class(self.test_name, locked=False)
        shared_data = {"counter": 0, "max_concurrent": 0, "current_in_section": 0}
        data_lock = threading.Lock()  # Only for tracking max_concurrent
        num_threads = 15
        iterations = 30

        def worker():
            for _ in range(iterations):
                lock.acquire()
                try:
                    # Track concurrent access
                    with data_lock:
                        shared_data["current_in_section"] += 1
                        if shared_data["current_in_section"] > shared_data["max_concurrent"]:
                            shared_data["max_concurrent"] = shared_data["current_in_section"]

                    # Non-atomic increment (read, compute, write)
                    temp = shared_data["counter"]
                    time.sleep(0.0001)
                    shared_data["counter"] = temp + 1

                    with data_lock:
                        shared_data["current_in_section"] -= 1
                finally:
                    lock.release()

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # max_concurrent should be 1 (only one thread in critical section at a time)
        assert shared_data["max_concurrent"] == 1, (
            f"Mutual exclusion violated: {shared_data['max_concurrent']} threads in critical section"
        )

        # Counter should match expected value
        expected = num_threads * iterations
        assert shared_data["counter"] == expected, f"Race condition: {shared_data['counter']} != {expected}"

        lock.close()
        lock.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_silent_double_release(self, lock_class):
        """Verify release is idempotent (no error on double release)."""
        lock = lock_class(self.test_name, locked=False)

        # Acquire and release
        lock.acquire()
        lock.release()

        # Double release should not raise
        lock.release()
        lock.release()
        lock.release()

        # Should still work after double releases
        lock.acquire()
        lock.release()

        lock.close()
        lock.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_release_without_acquire(self, lock_class):
        """Verify release without prior acquire doesn't raise."""
        lock = lock_class(self.test_name, locked=False)

        # Release without acquire should not raise
        lock.release()
        lock.release()

        # Lock should still be usable
        lock.acquire()
        lock.release()

        lock.close()
        lock.unlink()

    def test_failed_process_file_lock_close_does_not_release_holder(self):
        """Closing a timed-out contender must not release another instance's thread lock."""
        holder = ProcessFileLock(self.test_name, locked=True)
        contender = ProcessFileLock(self.test_name, locked=False)

        try:
            with pytest.raises(BlockingIOError):
                contender.acquire(timeout=0.0)

            assert holder._thread_lock.locked()  # noqa: SLF001
            contender.close()
            assert holder._thread_lock.locked()  # noqa: SLF001
        finally:
            contender.close()
            holder.release()
            holder.unlink()

    def test_negative_timeout_is_rejected(self):
        """Negative values cannot silently select an indefinite wait."""
        lock = ProcessFileLock(self.test_name, locked=False)
        try:
            with pytest.raises(ValueError, match="non-negative"):
                lock.acquire(timeout=-1.0)
        finally:
            lock.unlink()

    def test_timeout_covers_thread_and_file_lock_waits(self, monkeypatch):
        """The timeout is one deadline across both acquisition phases."""
        lock = ProcessFileLock(self.test_name, locked=False)
        file_timeouts = []
        clock = iter([10.0, 10.12])
        monkeypatch.setattr(time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(lock, "_acquire_thread_lock", lambda *_: lock._thread_lock.acquire())
        monkeypatch.setattr(lock, "_acquire_file_lock", lambda _, timeout: file_timeouts.append(timeout))

        try:
            lock.acquire(timeout=0.2)
            assert file_timeouts == pytest.approx([0.08])
        finally:
            lock.release()
            lock.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_concurrent_context_manager_and_explicit(self, lock_class):
        """Test mixing context manager and explicit acquire/release."""
        lock = lock_class(self.test_name, locked=False)
        results = []
        num_threads = 10

        def context_manager_worker(thread_id):
            for i in range(5):
                with lock:
                    results.append(f"cm-enter-{thread_id}-{i}")
                    time.sleep(0.001)
                    results.append(f"cm-exit-{thread_id}-{i}")

        def explicit_worker(thread_id):
            for i in range(5):
                lock.acquire()
                results.append(f"ex-enter-{thread_id}-{i}")
                time.sleep(0.001)
                results.append(f"ex-exit-{thread_id}-{i}")
                lock.release()

        threads = []
        for i in range(num_threads // 2):
            threads.append(threading.Thread(target=context_manager_worker, args=(f"cm{i}",)))
            threads.append(threading.Thread(target=explicit_worker, args=(f"ex{i}",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Verify all enter/exit pairs are properly matched
        for i in range(0, len(results), 2):
            enter = results[i]
            exit_ = results[i + 1]
            assert "enter" in enter, f"Expected enter at {i}, got {enter}"
            assert "exit" in exit_, f"Expected exit at {i + 1}, got {exit_}"

        lock.close()
        lock.unlink()

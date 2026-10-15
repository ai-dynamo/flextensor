# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess unit tests for MultiprocessCondition class."""

from __future__ import annotations

import multiprocessing
import os
import time
from typing import TYPE_CHECKING

import posix_ipc
import pytest
from shm.multiprocess_utils import (
    EVENT_TIMEOUT,
    POLL_INTERVAL,
    assert_clean_exit,
    drain_results,
    format_failed_results,
    unlink_semaphore_if_present,
    unlink_shared_memory_if_present,
    wait_for_event,
)

from flextensor.shm import MultiprocessCondition, ProcessFileLock, SemaphoreLock

if TYPE_CHECKING:
    from collections.abc import Iterable
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as EventType
    from multiprocessing.synchronize import Lock as LockType


def wait_for_waiters_registered(
    name: str,
    waiter_pids: Iterable[int],
    lock_class: type,
    *,
    timeout: float = EVENT_TIMEOUT,
) -> None:
    """Block until ``waiter_pids`` have appended their PIDs to the notification list.

    Opens temporary handles to the existing notification list and lock, then
    closes them after observing the registrations.
    """
    deadline = time.monotonic() + timeout
    condition = MultiprocessCondition(name=name, is_creator=False, lock_class=lock_class)
    expected_pids = {str(pid) for pid in waiter_pids}
    registered_pids: set[str] = set()
    try:
        while time.monotonic() < deadline:
            with condition.lock:
                registered_pids = set(condition.notification_list.get_list())
            if expected_pids <= registered_pids:
                return
            time.sleep(POLL_INTERVAL)
    finally:
        condition.close()
        condition.close_lock()

    msg = (
        f"Timed out waiting for waiters to register: expected {sorted(expected_pids)}, "
        f"got {sorted(registered_pids)} after {timeout}s"
    )
    raise AssertionError(msg)


def wait_for_ready_process(name, result_queue, init_lock, lock_class):
    """Process that connects, waits for ready, returns observed state."""
    pid = os.getpid()
    try:
        with init_lock:
            condition = MultiprocessCondition(name=name, is_creator=False, lock_class=lock_class)

        start_time = time.time()
        condition.wait_for_ready()
        wait_time = time.time() - start_time

        is_ready = condition.notification_list.is_ready()
        pid_list = condition.notification_list.get_list()

        condition.close()
        condition.close_lock()

        result_queue.put(
            {
                "success": True,
                "wait_time": wait_time,
                "is_ready": is_ready,
                "pid_list": pid_list,
                "pid": pid,
            },
        )

    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "pid": pid,
            },
        )


def notify_ready_process(
    name: str,
    result_queue: Queue,
    init_lock: LockType,
    lock_class: type,
    ready_event: EventType | None = None,
    notify_event: EventType | None = None,
    close_event: EventType | None = None,
) -> None:
    """Process that creates a condition then signals readiness on demand."""
    pid = os.getpid()
    try:
        with init_lock:
            condition = MultiprocessCondition(name=name, is_creator=True, lock_class=lock_class)

        if ready_event is not None:
            ready_event.set()

        if notify_event is not None and not notify_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to allow notify_ready()")

        condition.notify_ready()

        if close_event is not None and not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to allow condition cleanup")

        condition.close()
        condition.unlink()
        condition.close_lock()

        result_queue.put(
            {
                "success": True,
                "action": "notify_ready",
                "pid": pid,
            },
        )

    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "action": "notify_ready",
                "pid": pid,
            },
        )


def concurrent_wait_and_modify_process(name, proc_id, result_queue, init_lock, lock_class):
    """Process that waits for ready then appends a marker."""
    try:
        with init_lock:
            condition = MultiprocessCondition(name=name, is_creator=False, lock_class=lock_class)

        start_time = time.time()
        condition.wait_for_ready()
        wait_time = time.time() - start_time

        with condition.lock:
            condition.notification_list.append(f"post_ready_{proc_id}")

        pid_list = condition.notification_list.get_list()

        condition.close()
        condition.close_lock()

        result_queue.put(
            {
                "success": True,
                "proc_id": proc_id,
                "wait_time": wait_time,
                "pid_list": pid_list,
                "pid": os.getpid(),
            },
        )

    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "proc_id": proc_id,
                "pid": os.getpid(),
            },
        )


def stress_test_process(name, proc_id, operations, result_queue, lock_class):
    """Stress the lock: repeatedly enter the critical section and append."""
    try:
        condition = MultiprocessCondition(name=name, is_creator=False, lock_class=lock_class)

        operations_completed = 0
        for i in range(operations):
            with condition.lock:
                condition.notification_list.get_list()
                condition.notification_list.append(f"proc{proc_id}_op{i}")
                time.sleep(0.001)
                operations_completed += 1

        condition.close()

        result_queue.put(
            {
                "success": True,
                "proc_id": proc_id,
                "operations_completed": operations_completed,
                "pid": os.getpid(),
            },
        )

    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "proc_id": proc_id,
                "pid": os.getpid(),
            },
        )


def cleanup_multiprocess_condition(name):
    """Remove the SharedMemory + Semaphore that MultiprocessCondition allocates."""
    unlink_semaphore_if_present(name)
    unlink_shared_memory_if_present(name)


def create_condition_process(name, result_queue, lock_class):
    """Create a condition with initial data for stress testing, then exit."""
    try:
        condition = MultiprocessCondition(name=name, is_creator=True, lock_class=lock_class)
        condition.notification_list.append("initial_data")
        condition.close()
        result_queue.put({"success": True, "action": "create"})
    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "action": "create"})


def quick_lifecycle_process(name, proc_id, result_queue, lock_class, creator_ready, notify_allowed):
    """Quick lifecycle: creator notifies after every follower has registered its wait."""
    try:
        if proc_id == 0:
            condition = MultiprocessCondition(name=name, is_creator=True, lock_class=lock_class)
            condition.notification_list.append(f"data_{proc_id}")
            creator_ready.set()
            if not notify_allowed.wait(timeout=EVENT_TIMEOUT):
                raise TimeoutError("Timed out waiting for follower registrations")
            condition.notify_ready()
            condition.close()
            condition.unlink()
        else:
            if not creator_ready.wait(timeout=EVENT_TIMEOUT):
                raise TimeoutError("Timed out waiting for condition creator")
            condition = MultiprocessCondition(name=name, is_creator=False, lock_class=lock_class)
            condition.wait_for_ready()
            condition.close()
            condition.close_lock()

        result_queue.put({"success": True, "proc_id": proc_id})
    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "proc_id": proc_id})


class TestMultiprocessConditionMultiprocess:
    """Multiprocess tests for MultiprocessCondition class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "mpc_mp"  # Short name to avoid POSIX name limits
        cleanup_multiprocess_condition(self.test_name)

    def teardown_method(self):
        """Cleanup after each test method."""
        cleanup_multiprocess_condition(self.test_name)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_basic_wait_and_notify(self, lock_class):
        """Test basic wait and notify across processes."""
        result_queue = multiprocessing.Queue()
        init_lock = multiprocessing.Lock()
        notifier_ready = multiprocessing.Event()
        notify_now = multiprocessing.Event()
        close_notifier = multiprocessing.Event()

        notifier_proc = multiprocessing.Process(
            target=notify_ready_process,
            args=(self.test_name, result_queue, init_lock, lock_class),
            kwargs={
                "ready_event": notifier_ready,
                "notify_event": notify_now,
                "close_event": close_notifier,
            },
        )

        waiter_proc = multiprocessing.Process(
            target=wait_for_ready_process,
            args=(self.test_name, result_queue, init_lock, lock_class),
        )

        try:
            notifier_proc.start()
            wait_for_event(notifier_ready, "notifier condition creation", proc=notifier_proc)

            waiter_proc.start()
            wait_for_waiters_registered(self.test_name, [waiter_proc.pid], lock_class)
            notify_now.set()

            waiter_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(waiter_proc, "waiter")
            close_notifier.set()
            notifier_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(notifier_proc, "notifier")

            results = drain_results(result_queue, expected_count=2)

            waiter_result = next(r for r in results if "wait_time" in r)
            notifier_result = next(r for r in results if r.get("action") == "notify_ready")

            assert waiter_result["success"] is True
            assert notifier_result["success"] is True

            # Waiter should return promptly once notify_now is set.
            assert waiter_result["wait_time"] <= 2.0
            assert waiter_result["is_ready"] is True
            assert str(waiter_result["pid"]) in waiter_result["pid_list"]

        finally:
            notify_now.set()
            close_notifier.set()
            for proc in [waiter_proc, notifier_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_multiple_waiters_single_notifier(self, lock_class):
        """Test multiple processes waiting for single notifier."""
        result_queue = multiprocessing.Queue()
        init_lock = multiprocessing.Lock()
        num_waiters = 4
        notifier_ready = multiprocessing.Event()
        notify_now = multiprocessing.Event()
        close_notifier = multiprocessing.Event()

        notifier_proc = multiprocessing.Process(
            target=notify_ready_process,
            args=(self.test_name, result_queue, init_lock, lock_class),
            kwargs={
                "ready_event": notifier_ready,
                "notify_event": notify_now,
                "close_event": close_notifier,
            },
        )

        waiter_procs = [
            multiprocessing.Process(
                target=wait_for_ready_process,
                args=(self.test_name, result_queue, init_lock, lock_class),
            )
            for _ in range(num_waiters)
        ]

        try:
            notifier_proc.start()
            wait_for_event(notifier_ready, "notifier condition creation", proc=notifier_proc)

            for proc in waiter_procs:
                proc.start()
            wait_for_waiters_registered(self.test_name, [proc.pid for proc in waiter_procs], lock_class)
            notify_now.set()

            for i, proc in enumerate(waiter_procs):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"waiter {i}")
            close_notifier.set()
            notifier_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(notifier_proc, "notifier")

            results = drain_results(result_queue, expected_count=num_waiters + 1)

            for result in results:
                assert result["success"] is True, f"Failed: {result.get('error', 'unknown')}"

            waiter_results = [r for r in results if "wait_time" in r]
            assert len(waiter_results) == num_waiters

            for waiter_result in waiter_results:
                assert waiter_result["wait_time"] <= 2.0
                assert waiter_result["is_ready"] is True

        finally:
            notify_now.set()
            close_notifier.set()
            for proc in [notifier_proc, *waiter_procs]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_concurrent_access_no_race_conditions(self, lock_class):
        """Test concurrent access to condition doesn't cause race conditions."""
        result_queue = multiprocessing.Queue()
        init_lock = multiprocessing.Lock()
        creator_ready = multiprocessing.Event()
        notify_now = multiprocessing.Event()
        close_creator = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=notify_ready_process,
            args=(self.test_name, result_queue, init_lock, lock_class),
            kwargs={
                "ready_event": creator_ready,
                "notify_event": notify_now,
                "close_event": close_creator,
            },
        )

        worker_procs = [
            multiprocessing.Process(
                target=concurrent_wait_and_modify_process,
                args=(self.test_name, i, result_queue, init_lock, lock_class),
            )
            for i in range(3)
        ]

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "creator condition creation", proc=creator_proc)

            for proc in worker_procs:
                proc.start()
            wait_for_waiters_registered(self.test_name, [proc.pid for proc in worker_procs], lock_class)
            notify_now.set()

            for i, proc in enumerate(worker_procs):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"worker {i}")
            close_creator.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=4)

            worker_results = [r for r in results if "proc_id" in r]
            assert len(worker_results) == 3
            for result in worker_results:
                assert result["success"] is True, f"Failed: {result.get('error', 'unknown')}"
                assert result["wait_time"] <= 2.0

            all_pid_list_items = set()
            for result in worker_results:
                all_pid_list_items.update(result["pid_list"])

            for i in range(3):
                expected_string = f"post_ready_{i}"
                assert any(expected_string in s for s in all_pid_list_items), (
                    f"Missing expected string: {expected_string}"
                )

        finally:
            notify_now.set()
            close_creator.set()
            for proc in [creator_proc, *worker_procs]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_semaphore_stress_test(self, lock_class):
        """Stress test semaphore protection under high concurrency."""
        result_queue = multiprocessing.Queue()
        num_processes = 4
        operations_per_process = 20

        creator_proc = multiprocessing.Process(
            target=create_condition_process,
            args=(self.test_name, result_queue, lock_class),
        )

        stress_procs = [
            multiprocessing.Process(
                target=stress_test_process,
                args=(self.test_name, i, operations_per_process, result_queue, lock_class),
            )
            for i in range(num_processes)
        ]

        try:
            creator_proc.start()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            for proc in stress_procs:
                proc.start()

            for i, proc in enumerate(stress_procs):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"stress {i}")

            results = drain_results(result_queue, expected_count=num_processes + 1)

            creator_results = [r for r in results if r.get("action") == "create"]
            assert len(creator_results) == 1
            assert creator_results[0]["success"] is True

            stress_results = [r for r in results if "operations_completed" in r]
            assert len(stress_results) == num_processes
            assert all(r["success"] for r in stress_results), (
                f"Stress workers failed: {format_failed_results(stress_results)}"
            )

            total_operations = sum(r["operations_completed"] for r in stress_results)
            expected_total = num_processes * operations_per_process
            assert total_operations == expected_total, f"Expected {expected_total} ops, got {total_operations}"

        finally:
            for proc in [creator_proc, *stress_procs]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

            try:
                condition = MultiprocessCondition(name=self.test_name, is_creator=False, lock_class=lock_class)
                condition.close()
                condition.unlink()
            except (posix_ipc.ExistentialError, FileNotFoundError, OSError):
                pass

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_cleanup_no_deadlock(self, lock_class):
        """Test that cleanup operations don't cause deadlocks."""
        result_queue = multiprocessing.Queue()
        num_workers = 4
        creator_ready = multiprocessing.Event()
        notify_allowed = multiprocessing.Event()

        processes = [
            multiprocessing.Process(
                target=quick_lifecycle_process,
                args=(self.test_name, i, result_queue, lock_class, creator_ready, notify_allowed),
            )
            for i in range(num_workers)
        ]

        try:
            for proc in processes:
                proc.start()

            wait_for_event(creator_ready, "condition creator startup", proc=processes[0])
            wait_for_waiters_registered(self.test_name, [proc.pid for proc in processes[1:]], lock_class)
            notify_allowed.set()

            for i, proc in enumerate(processes):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"process {i}")

            results = drain_results(result_queue, expected_count=num_workers)
            assert all(r["success"] for r in results), f"Failed: {format_failed_results(results)}"

        finally:
            notify_allowed.set()
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

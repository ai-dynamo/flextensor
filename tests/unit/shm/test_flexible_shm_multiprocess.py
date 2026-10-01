# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess unit tests for FlexibleSharedMemory class."""

from __future__ import annotations

import multiprocessing
import os
import time
from typing import TYPE_CHECKING

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

from flextensor.shm import FlexibleSharedMemory, ProcessFileLock, SemaphoreLock

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.process import BaseProcess
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as EventType


def create_and_write_process(
    name: str,
    shm_size: int,
    data_to_write: bytes | tuple[bytes, float],
    result_queue: Queue,
    lock_class: type,
    ready_event: EventType | None = None,
    close_event: EventType | None = None,
) -> None:
    """Create FlexibleSharedMemory and write data; stay alive until released."""
    keep_alive_time = 2.0
    if isinstance(data_to_write, tuple):
        data_to_write, keep_alive_time = data_to_write

    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=shm_size,
            pinned_memory=False,
            lock_class=lock_class,
        )

        fsm.block.buf[: len(data_to_write)] = data_to_write

        if ready_event is not None:
            ready_event.set()

        if close_event is not None:
            if not close_event.wait(timeout=EVENT_TIMEOUT):
                raise TimeoutError("Timed out waiting for parent to allow creator cleanup")
        else:
            time.sleep(keep_alive_time)

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "action": "create_and_write",
                "data_written": data_to_write,
                "shm_creator": fsm.shm_creator,
                "pid": os.getpid(),
            },
        )

    except (FileNotFoundError, FileExistsError, OSError, ValueError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "action": "create_and_write",
                "pid": os.getpid(),
            },
        )


def read_process(
    name: str,
    expected_data_len: int,
    result_queue: Queue,
    keep_alive_time: float,
    lock_class: type,
    ready_event: EventType | None = None,
    close_event: EventType | None = None,
) -> None:
    """Read data, optionally staying alive until released."""
    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=0,
            pinned_memory=False,
            lock_class=lock_class,
        )

        read_data = bytes(fsm.block.buf[:expected_data_len])

        if ready_event is not None:
            ready_event.set()

        if close_event is not None:
            if not close_event.wait(timeout=EVENT_TIMEOUT):
                raise TimeoutError("Timed out waiting for parent to allow reader cleanup")
        elif keep_alive_time > 0:
            time.sleep(keep_alive_time)

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "action": "read",
                "data_read": read_data,
                "shm_creator": fsm.shm_creator,
                "pid": os.getpid(),
            },
        )

    except (FileNotFoundError, FileExistsError, OSError, ValueError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "action": "read",
                "pid": os.getpid(),
            },
        )


def concurrent_access_process(
    name: str,
    proc_id: int,
    operations: int,
    result_queue: Queue,
    lock_class: type,
) -> None:
    """Increment a shared counter under the main lock."""
    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=0,
            pinned_memory=False,
            lock_class=lock_class,
        )

        successful_ops = 0
        for _i in range(operations):
            with fsm.main_lock:
                try:
                    counter = int.from_bytes(bytes(fsm.block.buf[:4]), "little")
                except (ValueError, TypeError):
                    counter = 0
                counter += 1
                fsm.block.buf[:4] = counter.to_bytes(4, "little")
                successful_ops += 1
                time.sleep(0.001)

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "action": "concurrent_access",
                "proc_id": proc_id,
                "successful_ops": successful_ops,
                "pid": os.getpid(),
            },
        )

    except (FileNotFoundError, FileExistsError, OSError, ValueError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "action": "concurrent_access",
                "proc_id": proc_id,
                "pid": os.getpid(),
            },
        )


def keep_alive_monitor_process(
    name: str,
    monitor_duration: float,
    result_queue: Queue,
    lock_class: type,
    ready_event: EventType | None = None,
) -> None:
    """Poll any_process_alive() while another process holds the slot."""
    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=0,
            pinned_memory=False,
            lock_class=lock_class,
        )

        if ready_event is not None:
            ready_event.set()

        alive_checks = []
        deadline = time.monotonic() + monitor_duration
        while time.monotonic() < deadline:
            alive = fsm.keep_alive.any_process_alive()
            alive_checks.append(alive)
            time.sleep(POLL_INTERVAL)

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "action": "keep_alive_monitor",
                "alive_checks": alive_checks,
                "monitor_duration": monitor_duration,
                "pid": os.getpid(),
            },
        )

    except (FileNotFoundError, FileExistsError, OSError, ValueError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "action": "keep_alive_monitor",
                "pid": os.getpid(),
            },
        )


def check_memory_size_process(
    name: str,
    expected_size: int,
    result_queue: Queue,
    create: bool,
    lock_class: type,
    ready_event: EventType | None = None,
    close_event: EventType | None = None,
) -> None:
    """Attach (creating or connecting) and report observed segment size."""
    try:
        if create:
            fsm = FlexibleSharedMemory(name=name, shm_size=expected_size, pinned_memory=False, lock_class=lock_class)
        else:
            fsm = FlexibleSharedMemory(name=name, shm_size=0, pinned_memory=False, lock_class=lock_class)

        actual_size = fsm.block.size

        if ready_event is not None:
            ready_event.set()

        if close_event is not None and not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to allow memory size process cleanup")

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "expected_size": expected_size,
                "actual_size": actual_size,
                "create": create,
                "shm_creator": fsm.shm_creator,
                "pid": os.getpid(),
            },
        )

    except (FileNotFoundError, FileExistsError, OSError, ValueError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "create": create,
                "pid": os.getpid(),
            },
        )


def coordinated_cleanup_process(
    name: str,
    proc_id: int,
    result_queue: Queue,
    lock_class: type,
    ready_event: EventType | None = None,
    close_event: EventType | None = None,
) -> None:
    """Attach, write a per-PID marker, then close on parent's signal."""
    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=4096,
            pinned_memory=False,
            lock_class=lock_class,
        )

        test_data = f"proc_{proc_id}_data".encode()
        fsm.block.buf[: len(test_data)] = test_data

        if ready_event is not None:
            ready_event.set()

        if close_event is None or not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to allow coordinated cleanup")

        others_alive = fsm.keep_alive.any_process_alive()

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "proc_id": proc_id,
                "others_alive_at_close": others_alive,
                "pid": os.getpid(),
            },
        )

    except (FileNotFoundError, FileExistsError, OSError, ValueError) as e:
        result_queue.put(
            {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "proc_id": proc_id,
                "pid": os.getpid(),
            },
        )


def run_coordinated_cleanup_processes(
    processes: Sequence[BaseProcess],
    ready_events: Sequence[EventType],
    close_events: Sequence[EventType],
) -> None:
    """Start cleanup workers, then release them in deterministic close order.

    Asserts each worker exits cleanly so a hung or crashed process is reported
    here rather than as a downstream "missing result" assertion.
    """
    for proc in processes:
        proc.start()

    for i, event in enumerate(ready_events):
        wait_for_event(event, f"cleanup worker {i} startup", proc=processes[i])

    close_events[0].set()
    processes[0].join(timeout=EVENT_TIMEOUT)
    assert_clean_exit(processes[0], "cleanup worker 0")

    for event in close_events[1:]:
        event.set()
    for i, proc in enumerate(processes[1:], start=1):
        proc.join(timeout=EVENT_TIMEOUT)
        assert_clean_exit(proc, f"cleanup worker {i}")


def cleanup_flexible_shared_memory(name: str) -> None:
    """Best-effort removal of FlexibleSharedMemory's resources."""
    # FlexibleSharedMemory naming:
    # - Main shm: name; main lock semaphore: name
    # - KeepAliveDict: sm_<name>_kd (shm), <name>_km (timestamps), <name>_ks (lock)
    for shm_name in (name, f"sm_{name}_kd", f"{name}_km"):
        unlink_shared_memory_if_present(shm_name)
    for sem_name in (name, f"{name}_ks"):
        unlink_semaphore_if_present(sem_name)


class TestFlexibleSharedMemoryMultiprocess:
    """Multiprocess tests for FlexibleSharedMemory class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "fsm_mp"  # Short name to avoid POSIX name limits
        self.test_size = 8192
        cleanup_flexible_shared_memory(self.test_name)

    def teardown_method(self):
        """Cleanup after each test method."""
        cleanup_flexible_shared_memory(self.test_name)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_basic_create_and_connect(self, lock_class):
        """Test basic creation and connection across processes."""
        result_queue = multiprocessing.Queue()
        test_data = b"Hello from creator process!"
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=create_and_write_process,
            args=(self.test_name, self.test_size, test_data, result_queue, lock_class),
            kwargs={"ready_event": creator_ready, "close_event": creator_close},
        )
        reader_proc = multiprocessing.Process(
            target=read_process,
            args=(self.test_name, len(test_data), result_queue, 0, lock_class),
        )

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "creator shared memory creation", proc=creator_proc)
            reader_proc.start()

            reader_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(reader_proc, "reader")
            creator_close.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=2)

            creator_result = next(r for r in results if r.get("action") == "create_and_write")
            reader_result = next(r for r in results if r.get("action") == "read")

            assert creator_result["success"] is True, f"Creator failed: {creator_result.get('error')}"
            assert reader_result["success"] is True, f"Reader failed: {reader_result.get('error')}"

            assert creator_result["shm_creator"] is True
            assert reader_result["shm_creator"] is False

            assert reader_result["data_read"] == test_data

        finally:
            creator_close.set()
            for proc in [creator_proc, reader_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_multiple_readers(self, lock_class):
        """Test one writer, multiple readers."""
        result_queue = multiprocessing.Queue()
        test_data = b"Shared data for multiple readers"
        num_readers = 3
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=create_and_write_process,
            args=(self.test_name, self.test_size, test_data, result_queue, lock_class),
            kwargs={"ready_event": creator_ready, "close_event": creator_close},
        )

        reader_procs = [
            multiprocessing.Process(
                target=read_process,
                args=(self.test_name, len(test_data), result_queue, 0, lock_class),
            )
            for _ in range(num_readers)
        ]

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "creator shared memory creation", proc=creator_proc)

            for proc in reader_procs:
                proc.start()

            for i, proc in enumerate(reader_procs):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"reader {i}")
            creator_close.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=num_readers + 1)

            for result in results:
                assert result["success"] is True, f"Failed: {result.get('error', 'unknown')}"

            reader_results = [r for r in results if r.get("action") == "read"]
            assert len(reader_results) == num_readers

            for reader_result in reader_results:
                assert reader_result["data_read"] == test_data

        finally:
            creator_close.set()
            for proc in [creator_proc, *reader_procs]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_concurrent_access_with_semaphore(self, lock_class):
        """Test concurrent access protection with semaphore."""
        result_queue = multiprocessing.Queue()
        num_processes = 4
        operations_per_process = 10

        init_data = (0).to_bytes(4, "little") + b"\x00" * (self.test_size - 4)
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=create_and_write_process,
            args=(self.test_name, self.test_size, init_data, result_queue, lock_class),
            kwargs={"ready_event": creator_ready, "close_event": creator_close},
        )

        worker_procs = [
            multiprocessing.Process(
                target=concurrent_access_process,
                args=(self.test_name, i, operations_per_process, result_queue, lock_class),
            )
            for i in range(num_processes)
        ]

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "creator shared memory creation", proc=creator_proc)

            for proc in worker_procs:
                proc.start()

            for i, proc in enumerate(worker_procs):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"worker {i}")
            creator_close.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=num_processes + 1)

            creator_results = [r for r in results if r.get("action") == "create_and_write"]
            assert len(creator_results) == 1
            assert creator_results[0]["success"] is True

            worker_results = [r for r in results if r.get("action") == "concurrent_access"]
            assert len(worker_results) == num_processes
            # All workers must succeed — assert_clean_exit above already proved
            # nothing crashed silently; surface any error message if a worker
            # reported failure via the queue.
            assert all(r["success"] for r in worker_results), (
                f"Worker failures: {format_failed_results(worker_results)}"
            )

            total_ops = sum(r["successful_ops"] for r in worker_results)
            expected_total = num_processes * operations_per_process
            assert total_ops == expected_total, f"Expected {expected_total} ops, got {total_ops}"

        finally:
            creator_close.set()
            for proc in [creator_proc, *worker_procs]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_keep_alive_functionality(self, lock_class):
        """Test keep alive functionality across processes."""
        result_queue = multiprocessing.Queue()
        long_running_ready = multiprocessing.Event()
        long_running_close = multiprocessing.Event()
        monitor_ready = multiprocessing.Event()

        long_running_proc = multiprocessing.Process(
            target=create_and_write_process,
            args=(
                self.test_name,
                self.test_size,
                b"keep_alive_test",
                result_queue,
                lock_class,
            ),
            kwargs={"ready_event": long_running_ready, "close_event": long_running_close},
        )

        monitor_proc = multiprocessing.Process(
            target=keep_alive_monitor_process,
            args=(self.test_name, 0.3, result_queue, lock_class),
            kwargs={"ready_event": monitor_ready},
        )

        try:
            long_running_proc.start()
            wait_for_event(long_running_ready, "long-running shared memory creation", proc=long_running_proc)
            monitor_proc.start()
            wait_for_event(monitor_ready, "keep-alive monitor startup", proc=monitor_proc)

            monitor_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(monitor_proc, "monitor")

            long_running_close.set()
            long_running_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(long_running_proc, "long-running")

            results = drain_results(result_queue, expected_count=2)

            for result in results:
                assert result["success"] is True, f"Failed: {result.get('error', 'unknown')}"

            monitor_results = [r for r in results if r.get("action") == "keep_alive_monitor"]
            assert len(monitor_results) == 1

            alive_checks = monitor_results[0]["alive_checks"]
            assert len(alive_checks) > 0
            assert any(alive_checks), "Should have detected alive processes"

        finally:
            long_running_close.set()
            for proc in [long_running_proc, monitor_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_memory_size_consistency(self, lock_class):
        """Test that memory size is consistent across processes."""
        result_queue = multiprocessing.Queue()
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=check_memory_size_process,
            args=(self.test_name, self.test_size, result_queue, True, lock_class),
            kwargs={"ready_event": creator_ready, "close_event": creator_close},
        )

        connector_procs = [
            multiprocessing.Process(
                target=check_memory_size_process,
                args=(self.test_name, self.test_size, result_queue, False, lock_class),
            )
            for _ in range(2)
        ]

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "creator shared memory creation", proc=creator_proc)

            for proc in connector_procs:
                proc.start()

            for i, proc in enumerate(connector_procs):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"connector {i}")

            creator_close.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=3)

            for result in results:
                assert result["success"] is True, f"Failed: {result.get('error', 'unknown')}"

            sizes = [r["actual_size"] for r in results]
            assert len(set(sizes)) == 1, f"All processes should see the same size, got {sizes}"
            assert sizes[0] >= self.test_size, f"Memory size should be at least {self.test_size}, got {sizes[0]}"

        finally:
            creator_close.set()
            for proc in [creator_proc, *connector_procs]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_cleanup_coordination(self, lock_class):
        """Test coordinated cleanup across processes."""
        result_queue = multiprocessing.Queue()
        num_workers = 4

        ready_events = [multiprocessing.Event() for _ in range(num_workers)]
        close_events = [multiprocessing.Event() for _ in range(num_workers)]

        processes = [
            multiprocessing.Process(
                target=coordinated_cleanup_process,
                args=(self.test_name, i, result_queue, lock_class),
                kwargs={"ready_event": ready_events[i], "close_event": close_events[i]},
            )
            for i in range(num_workers)
        ]

        try:
            run_coordinated_cleanup_processes(processes, ready_events, close_events)

            results = drain_results(result_queue, expected_count=num_workers)
            assert all(r["success"] for r in results), f"Failed: {format_failed_results(results)}"

            results_by_id = {r["proc_id"]: r for r in results}
            # P0 closes first while siblings are still attached.
            assert results_by_id[0]["others_alive_at_close"] is True

        finally:
            for event in close_events:
                event.set()
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

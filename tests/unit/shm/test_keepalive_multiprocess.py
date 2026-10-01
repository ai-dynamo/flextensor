# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess unit tests for KeepAlive class."""

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

from flextensor.shm import KeepAlive, ProcessFileLock, SemaphoreLock

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as EventType


def create_keepalive_process(
    name: str,
    label: str,
    keep_alive_seconds: float,
    create_semaphore: bool,
    result_queue: Queue,
    duration: float,
    lock_class: type,
    ready_event: EventType | None = None,
    stop_event: EventType | None = None,
) -> None:
    """Helper function to create KeepAlive in a separate process."""
    try:
        keep_alive = KeepAlive(
            name=name,
            process_id=os.getpid(),
            keep_alive_seconds=keep_alive_seconds,
            is_creator=create_semaphore,
            lock_class=lock_class,
        )

        if ready_event is not None:
            ready_event.set()

        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if stop_event is not None and stop_event.is_set():
                break
            time.sleep(POLL_INTERVAL)
            if not keep_alive.any_process_alive():
                break

        keep_alive.close(any_process_alive=True)
        result_queue.put({"success": True, "process_id": label})

    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "process_id": label})


def monitor_keepalive_process(
    name: str,
    label: str,
    keep_alive_seconds: float,
    result_queue: Queue,
    monitor_duration: float,
    lock_class: type,
    ready_event: EventType | None = None,
) -> None:
    """Monitor an existing KeepAlive in a separate process."""
    try:
        keep_alive = KeepAlive(
            name=name,
            process_id=os.getpid(),
            keep_alive_seconds=keep_alive_seconds,
            is_creator=False,
            lock_class=lock_class,
        )

        if ready_event is not None:
            ready_event.set()

        alive_checks = []
        deadline = time.monotonic() + monitor_duration
        while time.monotonic() < deadline:
            alive_checks.append(keep_alive.any_process_alive())
            time.sleep(POLL_INTERVAL)

        keep_alive.close(any_process_alive=True)
        result_queue.put(
            {
                "success": True,
                "process_id": label,
                "alive_checks": alive_checks,
            },
        )

    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "process_id": label})


def cleanup_resources(name: str) -> None:
    """Best-effort removal of KeepAlive's shared resources between tests.

    KeepAlive naming:
    - Main lock semaphore: ``name``;  internal-state semaphore: ``name``_s
    - SharedMemoryDict-backed state: ``<dict_name>``_d (data) + ``<dict_name>``_m (mutex),
      where ``<dict_name>`` is either ``name``_dict or ``name`` depending on which
      KeepAlive variant created the segment.

    Both ``unlink_*_if_present`` helpers retry internally with the same backoff,
    so we don't need an additional outer ``time.sleep`` — that was redundant.
    """
    for sem_name in (name, f"{name}_s"):
        unlink_semaphore_if_present(sem_name)

    for dict_name in (f"{name}_dict", name):
        unlink_shared_memory_if_present(f"{dict_name}_d")
        unlink_shared_memory_if_present(f"{dict_name}_m")


def quick_cleanup_process(
    name: str,
    label: str,
    result_queue: Queue,
    lock_class: type,
    is_creator: bool = False,
    ready_event: EventType | None = None,
    close_event: EventType | None = None,
) -> None:
    """Attach to KeepAlive, signal ready, wait for close signal, then close.

    Closes with ``any_process_alive=True`` — without an outer mutex, multiple
    workers calling ``any_process_alive=False`` race to ``unlink()`` the same
    POSIX shared resources; only the first wins, the rest hit ``FileNotFoundError``.
    Production code (``FlexibleSharedMemory._cleanup_with_coordination``) avoids
    that by serializing close() under a main_lock and recomputing
    ``any_process_alive()`` under the lock; the test's ``teardown_method`` does
    the eventual unlink instead.
    """
    try:
        keep_alive = KeepAlive(
            name=name,
            process_id=os.getpid(),
            keep_alive_seconds=1,
            is_creator=is_creator,
            lock_class=lock_class,
        )
        if ready_event is not None:
            ready_event.set()
        if close_event is not None:
            if not close_event.wait(timeout=EVENT_TIMEOUT):
                msg = f"[{label}] Timed out waiting for parent to allow close after {EVENT_TIMEOUT}s"
                raise TimeoutError(msg)
        else:
            time.sleep(POLL_INTERVAL)
        keep_alive.close(any_process_alive=True)
        result_queue.put({"success": True, "process_id": label})
    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "process_id": label})


def long_holding_process(
    name: str,
    label: str,
    result_queue: Queue,
    lock_class: type,
    hold_time: float = 0.2,
) -> None:
    """Process that holds the keep-alive lock for a while, then queries liveness."""
    try:
        keep_alive = KeepAlive(
            name=name,
            process_id=os.getpid(),
            keep_alive_seconds=1,
            is_creator=True,
            lock_class=lock_class,
        )

        with keep_alive.lock:
            time.sleep(hold_time)
            # Don't call any_process_alive() under the lock — it would re-enter
            # the same lock and deadlock.

        alive = keep_alive.any_process_alive()

        keep_alive.close(any_process_alive=False)
        result_queue.put({"success": True, "process_id": label, "alive": alive})
    except (posix_ipc.ExistentialError, posix_ipc.BusyError, FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "process_id": label})


class TestKeepAliveMultiprocess:
    """Multiprocess tests for KeepAlive class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "ka_mp"  # Short name to avoid POSIX name limits
        # Floor at 1.0s so the heartbeat thread (fires every keep_alive_seconds/2)
        # tolerates fork+import latency on shared CI runners — values below ~0.5s
        # can mark a freshly-spawned worker as stale before it has registered.
        self.keep_alive_seconds = 1.0
        cleanup_resources(self.test_name)

    def teardown_method(self):
        """Cleanup after each test method."""
        cleanup_resources(self.test_name)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_multiprocess_keepalive_basic(self, lock_class):
        """Test basic multiprocess KeepAlive functionality."""
        result_queue = multiprocessing.Queue()
        process1_ready = multiprocessing.Event()

        process1 = multiprocessing.Process(
            target=create_keepalive_process,
            args=(self.test_name, "proc1", self.keep_alive_seconds, True, result_queue, 0.4, lock_class),
            kwargs={"ready_event": process1_ready},
        )
        process2 = multiprocessing.Process(
            target=create_keepalive_process,
            args=(self.test_name, "proc2", self.keep_alive_seconds, False, result_queue, 0.4, lock_class),
        )

        try:
            process1.start()
            wait_for_event(process1_ready, "creator KeepAlive initialization", proc=process1)
            process2.start()

            process1.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(process1, "proc1")
            process2.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(process2, "proc2")

            results = drain_results(result_queue, expected_count=2)

            for result in results:
                assert result["success"] is True, (
                    f"Process {result['process_id']} failed: {result.get('error', 'Unknown error')}"
                )

        finally:
            for proc in [process1, process2]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_process_detection_and_cleanup(self, lock_class):
        """Test that processes can detect each other and cleanup dead processes."""
        result_queue = multiprocessing.Queue()
        creator_ready = multiprocessing.Event()
        monitor_ready = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=create_keepalive_process,
            args=(self.test_name, "creator", self.keep_alive_seconds, True, result_queue, 0.8, lock_class),
            kwargs={"ready_event": creator_ready},
        )

        monitor_proc = multiprocessing.Process(
            target=monitor_keepalive_process,
            args=(self.test_name, "monitor", self.keep_alive_seconds, result_queue, 0.3, lock_class),
            kwargs={"ready_event": monitor_ready},
        )

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "creator KeepAlive initialization", proc=creator_proc)
            monitor_proc.start()
            wait_for_event(monitor_ready, "monitor KeepAlive initialization", proc=monitor_proc)

            monitor_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(monitor_proc, "monitor")
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=2)
            results_by_id = {r["process_id"]: r for r in results}
            assert results_by_id["creator"]["success"] is True
            assert results_by_id["monitor"]["success"] is True

            alive_checks = results_by_id["monitor"]["alive_checks"]
            assert any(alive_checks), "Monitor should have detected other processes alive"

        finally:
            for proc in [creator_proc, monitor_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_race_condition_protection(self, lock_class):
        """Test that semaphores protect against race conditions when many processes
        race to create/connect to KeepAlive concurrently."""
        result_queue = multiprocessing.Queue()
        num_processes = 4
        # Workers must outlive the staleness threshold so a slow-to-register peer
        # isn't seen as stale by the others — derive from keep_alive_seconds so
        # raising the staleness floor in setup_method keeps this test meaningful.
        worker_duration = self.keep_alive_seconds
        processes = []

        for i in range(num_processes):
            create_semaphore = i == 0
            proc = multiprocessing.Process(
                target=create_keepalive_process,
                args=(
                    self.test_name,
                    f"proc_{i}",
                    self.keep_alive_seconds,
                    create_semaphore,
                    result_queue,
                    worker_duration,
                    lock_class,
                ),
            )
            processes.append(proc)

        try:
            for proc in processes:
                proc.start()
                time.sleep(0.02)

            for i, proc in enumerate(processes):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"proc_{i}")

            results = drain_results(result_queue, expected_count=num_processes)

            # Acceptable failure modes are race-induced IPC errors. Match by
            # exception type name (workers format errors as ``f"{type(e).__name__}: {e}"``)
            # rather than free-form message text, which can change with locale or
            # embedded paths and silently let unrelated failures slip through.
            allowed_error_prefixes = (
                "ExistentialError:",
                "FileNotFoundError:",
                "FileExistsError:",
            )
            for result in results:
                if not result["success"]:
                    error = result.get("error", "")
                    assert error.startswith(allowed_error_prefixes), f"Unexpected (non-race) error: {error}"

            # At minimum the creator must have succeeded; surface any unexpected
            # failure list with full detail.
            success_count = sum(1 for r in results if r["success"])
            assert success_count >= 1, f"No worker succeeded; failures: {format_failed_results(results)}"

        finally:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_no_deadlock_on_cleanup(self, lock_class):
        """Concurrent close() across three processes must not deadlock.

        All three workers attach (handshake via per-process ``ready_event``),
        then the parent fires every ``close_event`` simultaneously so the
        close()s actually overlap. Without the per-process ready handshake,
        the creator could finish its ``POLL_INTERVAL`` and ``unlink()`` the
        shared segments before later-spawned workers (which pay fork+import
        latency on busy CI runners) finish their ``KeepAlive(is_creator=False)``
        attach — surfaces as ``FileNotFoundError: '/ka_mp_m'``.
        """
        result_queue = multiprocessing.Queue()

        ready_events = [multiprocessing.Event() for _ in range(3)]
        close_events = [multiprocessing.Event() for _ in range(3)]
        processes = []
        for i in range(3):
            label = "creator" if i == 0 else f"proc_{i}"
            is_creator = i == 0
            proc = multiprocessing.Process(
                target=quick_cleanup_process,
                args=(self.test_name, label, result_queue, lock_class, is_creator),
                kwargs={"ready_event": ready_events[i], "close_event": close_events[i]},
            )
            processes.append(proc)

        try:
            # Creator must finish creating the resources before non-creators try
            # to attach — otherwise their ``is_creator=False`` connect races against
            # the missing shared segment.
            processes[0].start()
            wait_for_event(ready_events[0], "creator KeepAlive initialization", proc=processes[0])

            for proc in processes[1:]:
                proc.start()
            for i, ev in enumerate(ready_events[1:], start=1):
                wait_for_event(ev, f"process {i} KeepAlive initialization", proc=processes[i])

            # All three are now attached — release them concurrently to actually
            # exercise the no-deadlock-on-overlapping-close path.
            for ev in close_events:
                ev.set()

            for i, proc in enumerate(processes):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"process {i}")

            results = drain_results(result_queue, expected_count=len(processes))
            assert all(r["success"] for r in results), f"Failed: {format_failed_results(results)}"

        finally:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_long_lock_hold_does_not_block_liveness(self, lock_class):
        """Verify holding the keep-alive lock then releasing it leaves the
        liveness check working — i.e. a long critical section doesn't poison
        subsequent any_process_alive() calls."""
        result_queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=long_holding_process,
            args=(self.test_name, "holder", result_queue, lock_class),
        )

        try:
            process.start()
            process.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(process, "holder")

            results = drain_results(result_queue, expected_count=1)
            result = results[0]
            assert result["success"] is True, f"Failed: {result.get('error', 'unknown')}"
            assert result["alive"] is True

        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=EVENT_TIMEOUT)

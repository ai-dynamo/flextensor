# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess unit tests for KeepAlive class."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import subprocess  # noqa: S404 - isolate the resource tracker and capture its stderr
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

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
    from multiprocessing.synchronize import Barrier
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
    start_barrier: Barrier | None = None,
) -> None:
    """Helper function to create KeepAlive in a separate process."""
    if start_barrier is not None:
        start_barrier.wait()
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


def _initialization_race_process(name, lock_class, is_creator, initializing, resume, finish, result_queue):
    """Pause SHM sizing in the creator and report lock contention in the follower."""
    real_ftruncate = os.ftruncate

    def pause_before_sizing(fd, size):
        if is_creator:
            initializing.set()
            wait_for_event(resume, "creator initialization release", timeout=3 * EVENT_TIMEOUT)
        real_ftruncate(fd, size)

    class ObservedLock(lock_class):
        def acquire(self, non_blocking=False, timeout=None):
            if not is_creator and not getattr(self, "reported", False):
                self.reported = True
                try:
                    super().acquire(non_blocking=True)
                except (posix_ipc.BusyError, BlockingIOError):
                    result_queue.put(("blocked", os.getpid()))
                else:
                    raise AssertionError("follower acquired lock while creator was initializing")
            super().acquire(non_blocking=non_blocking, timeout=timeout)

    keep_alive = None
    try:
        with patch("multiprocessing.shared_memory.os.ftruncate", side_effect=pause_before_sizing):
            keep_alive = KeepAlive(
                name=name,
                process_id=os.getpid(),
                is_creator=is_creator,
                lock_class=ObservedLock,
            )
        result_queue.put(("ready", os.getpid()))
        wait_for_event(finish, "participant shutdown")
    except BaseException as error:
        result_queue.put(("error", f"{type(error).__name__}: {error}"))
        raise
    finally:
        if keep_alive is not None:
            keep_alive.close(any_process_alive=True)


def _check_initialization_race(lock_class):
    """A follower waits while the creator's SHM object exists with size zero."""
    context = multiprocessing.get_context("spawn")
    name = f"ka_init_{uuid.uuid4().hex[:8]}"
    initializing, resume, finish = (context.Event() for _ in range(3))
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_initialization_race_process,
            args=(name, lock_class, is_creator, initializing, resume, finish, result_queue),
        )
        for is_creator in (True, False)
    ]
    creator, follower = processes

    try:
        creator.start()
        wait_for_event(initializing, "creator paused before SHM sizing", proc=creator)
        follower.start()
        assert drain_results(result_queue, expected_count=1) == [("blocked", follower.pid)]

        resume.set()
        results = drain_results(result_queue, expected_count=2)
        assert set(results) == {("ready", creator.pid), ("ready", follower.pid)}
        finish.set()
        for proc in processes:
            proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(proc, "KeepAlive initialization participant")
    finally:
        resume.set()
        finish.set()
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=EVENT_TIMEOUT)
        for shm_name in (f"sm_{name}_d", f"{name}_m"):
            unlink_shared_memory_if_present(shm_name)
        with contextlib.suppress(posix_ipc.ExistentialError):
            posix_ipc.unlink_semaphore(f"/{name}_s")
        Path(ProcessFileLock.resolve_path(f"{name}_s")).unlink(missing_ok=True)
        result_queue.close()
        result_queue.join_thread()


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
        """Initialization blocks followers and leaves the resource tracker clean."""
        # A fresh interpreter owns its tracker, so capture includes tracker errors
        # through shutdown even when all worker exit codes are zero.
        script = (
            "from shm.test_keepalive_multiprocess import _check_initialization_race\n"
            f"from flextensor.shm import {lock_class.__name__}\n"
            f"_check_initialization_race({lock_class.__name__})\n"
        )
        result = subprocess.run(  # noqa: S603 - fixed test helper and parametrized lock classes
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=12 * EVENT_TIMEOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "resource_tracker" not in result.stderr, result.stderr

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

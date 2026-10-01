# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess lifecycle tests for FlexibleSharedMemory class.

Tests the complete lifecycle of shared memory across multiple processes:
- Creation and connection
- Read/write operations across processes
- Concurrent close ordering (P0 closes while siblings are alive)
- Sequential cleanup with the last surviving process tearing down resources
- Verification of no memory leaks
"""

from __future__ import annotations

import multiprocessing
import os
import traceback
import uuid
from typing import TYPE_CHECKING

import pytest
from shm.multiprocess_utils import (
    EVENT_TIMEOUT,
    assert_clean_exit,
    drain_results,
    format_failed_results,
    wait_for_event,
)

from flextensor.shm import FlexibleSharedMemory, ProcessFileLock, SemaphoreLock

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as EventType


def lifecycle_process(
    name: str,
    shm_size: int,
    process_id: int,
    data_to_write: bytes | None,
    result_queue: Queue,
    lock_class: type,
    ready_event: EventType,
    close_event: EventType,
) -> None:
    """Helper process for lifecycle testing.

    Args:
        name: Shared memory name
        shm_size: Size of shared memory (>0 to create/connect, 0 to connect only)
        process_id: Identifier for this process
        data_to_write: Data to write at this process's offset, or None to skip writing
        result_queue: Queue for sending results back
        lock_class: Lock class to use
        ready_event: Set after the process has attached and written its slot
        close_event: Wait for this before closing (lifecycle tests require
            deterministic close ordering, so the parent always supplies one)
    """
    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=shm_size,
            pinned_memory=False,
            lock_class=lock_class,
        )

        if data_to_write is not None:
            offset = process_id * 32
            fsm.block.buf[offset : offset + len(data_to_write)] = data_to_write

        data_from_p0 = bytes(fsm.block.buf[0:14])

        ready_event.set()

        if not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to allow lifecycle close")

        # Snapshot before close so the assertion observes the state we care about
        # (was anyone else alive while *I* was still attached).
        others_alive = fsm.keep_alive.any_process_alive()

        fsm.close()

        result_queue.put({
            "success": True,
            "process_id": process_id,
            "shm_creator": fsm.shm_creator,
            "data_from_p0": data_from_p0,
            "others_alive_at_close": others_alive,
            "pid": os.getpid(),
        })

    except (FileNotFoundError, FileExistsError, OSError, ValueError) as e:
        result_queue.put({
            "success": False,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "process_id": process_id,
            "pid": os.getpid(),
        })


@pytest.fixture(params=[SemaphoreLock, ProcessFileLock])
def lock_class(request: pytest.FixtureRequest) -> type:
    """Parameterize tests with different lock implementations."""
    return request.param


class TestFlexibleSharedMemoryLifecycle:
    """Test complete lifecycle of FlexibleSharedMemory across multiple processes."""

    def test_concurrent_then_sequential_close_three_processes(self, lock_class: type) -> None:
        """Three processes attach concurrently, then close in sequence.

        Scenario:
        1. P0 creates shared memory and writes its slot
        2. P1 and P2 connect concurrently while P0 is still alive (overlapping
           init exercises the multi-attach path)
        3. P0 is released first while P1 and P2 are still attached — verifies
           that mid-cleanup of one process doesn't disturb live siblings
        4. P1 then P2 close in turn; P2 (last out) tears down the shared memory
        5. Verify the segment is unlinked (a fresh connect raises)
        """
        ctx = multiprocessing.get_context("fork")
        shm_name = f"lc_{uuid.uuid4().hex[:8]}"
        shm_size = 1024
        result_queue = ctx.Queue()
        ready_events = [ctx.Event() for _ in range(3)]
        close_events = [ctx.Event() for _ in range(3)]

        processes_config = [
            {"process_id": 0, "shm_size": shm_size, "data": b"Hello from P0!"},
            {"process_id": 1, "shm_size": shm_size, "data": b"Hello from P1!"},
            {"process_id": 2, "shm_size": shm_size, "data": b"Hello from P2!"},
        ]

        processes = []
        for config in processes_config:
            proc = ctx.Process(
                target=lifecycle_process,
                args=(
                    shm_name,
                    config["shm_size"],
                    config["process_id"],
                    config["data"],
                    result_queue,
                    lock_class,
                ),
                kwargs={
                    "ready_event": ready_events[config["process_id"]],
                    "close_event": close_events[config["process_id"]],
                },
            )
            processes.append(proc)

        try:
            processes[0].start()
            wait_for_event(ready_events[0], "lifecycle process 0 startup", proc=processes[0])

            # Start P1 and P2 concurrently so their init races overlap with P0
            # still attached.
            for proc in processes[1:]:
                proc.start()
            for i, event in enumerate(ready_events[1:], start=1):
                wait_for_event(event, f"lifecycle process {i} startup", proc=processes[i])

            # Release P0 while P1 and P2 are still alive — exercises
            # cleanup-with-siblings-attached.
            close_events[0].set()
            processes[0].join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(processes[0], "P0")

            # Then sequential teardown of the rest.
            close_events[1].set()
            processes[1].join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(processes[1], "P1")
            close_events[2].set()
            processes[2].join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(processes[2], "P2")

            results = drain_results(result_queue, expected_count=3)
            assert all(r["success"] for r in results), f"Failed: {format_failed_results(results)}"

            results_by_id = {r["process_id"]: r for r in results}
            assert results_by_id[0]["shm_creator"] is True
            assert results_by_id[1]["shm_creator"] is False
            assert results_by_id[2]["shm_creator"] is False

            for result in results:
                assert result["data_from_p0"] == b"Hello from P0!", f"Process {result['process_id']} read wrong data"

            # P0 closes first while P1+P2 are alive — concurrent-close coverage.
            assert results_by_id[0]["others_alive_at_close"] is True, "P0 should have observed P1/P2 alive at its close"
            # P1 closes while P2 is alive — sibling-still-alive coverage.
            assert results_by_id[1]["others_alive_at_close"] is True, "P1 should have observed P2 alive at its close"

            # No leak: segment is gone.
            with pytest.raises(FileNotFoundError):
                FlexibleSharedMemory(
                    name=shm_name,
                    shm_size=0,
                    pinned_memory=False,
                    lock_class=lock_class,
                )

        finally:
            for event in close_events:
                event.set()
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

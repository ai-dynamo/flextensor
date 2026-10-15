# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple multiprocess test for SharedMultiString class."""

import multiprocessing
import os
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event as EventType

from shm.multiprocess_utils import (
    EVENT_TIMEOUT,
    assert_clean_exit,
    drain_results,
    unlink_shared_memory_if_present,
    wait_for_event,
)

from flextensor.shm import SharedMultiString


def writer_process(name: str, result_queue: Queue, ready_event: EventType, close_event: EventType) -> None:
    """Process that writes data to SharedMultiString."""
    pid = os.getpid()
    sms = SharedMultiString(name=name, create=True)
    sms.append("item1")
    sms.append("item2")
    sms.append("item3")

    sms.set_ready()

    ready_event.set()

    if not close_event.wait(timeout=EVENT_TIMEOUT):
        raise TimeoutError("Reader did not finish before writer close timeout")

    sms.close()

    result_queue.put(
        {
            "action": "write",
            "items_written": ["item1", "item2", "item3"],
            "pid": pid,
        },
    )


def reader_process(name: str, result_queue: Queue) -> None:
    """Process that reads data from SharedMultiString."""
    pid = os.getpid()
    sms = SharedMultiString(name=name, create=False)
    items = sms.get_list()
    is_ready = sms.is_ready()
    sms.close()

    result_queue.put(
        {
            "action": "read",
            "items_read": items,
            "is_ready": is_ready,
            "pid": pid,
        },
    )


def test_shared_multi_string_basic() -> None:
    """Test basic SharedMultiString data sharing between processes."""
    test_name = "sms_test"

    unlink_shared_memory_if_present(test_name)

    result_queue: Queue = multiprocessing.Queue()
    ready_event = multiprocessing.Event()
    close_event = multiprocessing.Event()

    writer_proc = multiprocessing.Process(
        target=writer_process,
        args=(test_name, result_queue, ready_event, close_event),
    )

    reader_proc = multiprocessing.Process(
        target=reader_process,
        args=(test_name, result_queue),
    )

    try:
        writer_proc.start()
        wait_for_event(ready_event, "writer SharedMultiString creation", proc=writer_proc)

        reader_proc.start()

        reader_proc.join(timeout=EVENT_TIMEOUT)
        assert_clean_exit(reader_proc, "reader")

        close_event.set()
        writer_proc.join(timeout=EVENT_TIMEOUT)
        assert_clean_exit(writer_proc, "writer")

        results = drain_results(result_queue, expected_count=2)

        writer_result = next((r for r in results if r.get("action") == "write"), None)
        reader_result = next((r for r in results if r.get("action") == "read"), None)

        assert writer_result is not None, "No writer result found"
        assert reader_result is not None, "No reader result found"

        expected_items = ["item1", "item2", "item3"]
        assert reader_result["items_read"] == expected_items, (
            f"Items mismatch: expected {expected_items}, got {reader_result['items_read']}"
        )
        assert reader_result["is_ready"] is True, "Ready flag should be True"

    finally:
        close_event.set()
        for proc in [writer_proc, reader_proc]:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=EVENT_TIMEOUT)

        unlink_shared_memory_if_present(test_name)


if __name__ == "__main__":
    test_shared_multi_string_basic()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess tests for ShmCoordinator creator/follower flow."""

from __future__ import annotations

import multiprocessing
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import posix_ipc
import pytest
from shm.multiprocess_utils import (
    EVENT_TIMEOUT,
    assert_clean_exit,
    drain_results,
    format_failed_results,
    wait_for_event,
)

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.shm.coordinator import ShmCoordinator, ShmCreatorDiedError, ShmCreatorFailedError
from flextensor.shm.flexible_shm import FlexibleSharedMemory
from flextensor.state_handler import TensorManagerState

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventType


def _make_minimal_state(loader_type: str = "allocation_block_transfer") -> TensorManagerState:
    tensor_stats = [
        TensorStatistics(tensor_id=1, name="layer.weight", size_bytes=1024, load_time_ms=0.1),
        TensorStatistics(tensor_id=2, name="layer.bias", size_bytes=1024, load_time_ms=0.1),
    ]
    return TensorManagerState(
        loader_type=loader_type,
        tensor_id_to_name_map={1: "layer.weight", 2: "layer.bias"},
        allocation_ordered={0: ["layer"]},
        label_to_size_map={"layer": 2048},
        block_sizes={0: 2048},
        load_strategy={"layer": tensor_stats},
        release_strategy={},
        label_to_block_id={"layer": 0},
        stats=[LayerStatistics(label="layer", tensors=tensor_stats, duration=1.0)],
        transfer_to_compute_map={"layer": "layer"},
        view_tensors_ids=[1, 2],
        view_tensors_names=["layer.weight", "layer.bias"],
        gpu_tensors_names=[],
        shm_block_name_map={"layer": "ft_test_w0"},
    )


def creator_process(
    namespace: str,
    result_queue: multiprocessing.Queue,
    ready_event: EventType,
    close_event: EventType,
) -> None:
    """Creator: write profile, notify, stay alive until ``close_event`` is set.

    Catches expected IPC errors and reports via the result queue. TimeoutError
    from event waits is *not* caught — it propagates to a non-zero exitcode so
    the parent's assert_clean_exit fires loudly instead of silently dropping
    the failure into a result-queue slot the parent might never drain.
    """
    try:
        coord = ShmCoordinator(namespace)
        state = _make_minimal_state()
        coord.write_profile(state)
        coord.notify_ready()

        ready_event.set()

        if not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to allow coordinator close")
        coord.close()
        result_queue.put({"success": True, "is_creator": True})
    except (FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "is_creator": True})


def follower_process(namespace: str, result_queue: multiprocessing.Queue) -> None:
    """Follower: wait for ready, read profile, verify data matches."""
    try:
        coord = ShmCoordinator(namespace)
        assert coord.is_creator is False, "Expected follower role"

        coord.wait_for_ready()
        state = coord.read_profile()

        coord.close()
        result_queue.put({
            "success": True,
            "is_creator": False,
            "loader_type": state.loader_type,
            "tensor_count": len(state.tensor_id_to_name_map),
            "block_count": len(state.allocation_ordered),
        })
    except (FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "is_creator": False})


def short_lived_creator_process(
    ns: str,
    rq: multiprocessing.Queue,
    ready_event,
    close_event,
    keep_alive_seconds: float = 30,
) -> None:
    """Creator that exits without unregistering after the checker has connected."""
    try:
        coord = ShmCoordinator(ns, keep_alive_seconds=keep_alive_seconds)
        coord.notify_ready()
        ready_event.set()
        if not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to release short-lived creator")
        rq.put({"success": True, "role": "creator"})
    except (FileNotFoundError, FileExistsError, OSError) as e:
        rq.put({"success": False, "error": f"{type(e).__name__}: {e}", "role": "creator"})


def liveness_checker_process(
    ns: str,
    rq: multiprocessing.Queue,
    connected_event: EventType,
    creator_closed_event: EventType,
    keep_alive_seconds: float = 30,
) -> None:
    """Connect, signal ``connected_event``, wait on ``creator_closed_event``, then check liveness.

    Two-event handshake: ``connected_event`` lets the parent release the creator
    only after this process has attached, and ``creator_closed_event`` ensures
    ``is_creator_alive()`` is called *after* the creator has exited.
    """
    try:
        coord = ShmCoordinator(ns, keep_alive_seconds=keep_alive_seconds)
        coord.wait_for_ready()
        connected_event.set()
        if not creator_closed_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for creator to close")
        time.sleep(keep_alive_seconds * 2)
        alive = coord.is_creator_alive()
        coord.close()
        rq.put({"success": True, "role": "checker", "creator_alive": alive})
    except (FileNotFoundError, FileExistsError, OSError) as e:
        rq.put({"success": False, "error": f"{type(e).__name__}: {e}", "role": "checker"})


def live_follower_process(
    ns: str,
    rq: multiprocessing.Queue,
    connected_event: EventType,
    close_event: EventType,
    keep_alive_seconds: float,
) -> None:
    """Keep another follower heartbeat fresh until the liveness check finishes."""
    try:
        coord = ShmCoordinator(ns, keep_alive_seconds=keep_alive_seconds)
        coord.wait_for_ready()
        connected_event.set()
        if not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to release live follower")
        coord.close()
        rq.put({"success": True, "role": "live_follower"})
    except (FileNotFoundError, FileExistsError, OSError) as e:
        rq.put({"success": False, "error": f"{type(e).__name__}: {e}", "role": "live_follower"})


def failing_creator_process(
    ns: str,
    rq: multiprocessing.Queue,
    created_event: EventType,
    fail_event: EventType,
    failed_event: EventType,
    close_event: EventType,
) -> None:
    """Publish FAILED on demand and remain attached until followers finish."""
    coord = None
    try:
        coord = ShmCoordinator(ns)
        assert coord.is_creator is True, "Expected creator role"
        created_event.set()
        if not fail_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to trigger creator failure")
        coord.mark_failed(ValueError("checkpoint\nload failed"))
        failed_event.set()
        if not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to allow failed creator close")
        coord.close()
        rq.put({"success": True, "role": "failed_creator"})
    except BaseException as exc:
        if coord is not None:
            coord.close()
        rq.put({"success": False, "error": f"{type(exc).__name__}: {exc}", "role": "failed_creator"})


def _wait_for_ready(coord: ShmCoordinator, waiting_event: EventType | None) -> None:
    """Signal only after INITIALIZING and creator liveness checks reach the polling sleep."""

    def signal_waiting_and_sleep(seconds: float) -> None:
        if waiting_event is not None:
            waiting_event.set()
        time.sleep(seconds)

    # Patch the module reference so background heartbeat sleeps cannot signal the event.
    with patch("flextensor.shm.coordinator.time", wraps=time) as coordinator_time:
        coordinator_time.sleep.side_effect = signal_waiting_and_sleep
        coord.wait_for_ready(timeout=EVENT_TIMEOUT)


def failed_follower_process(
    ns: str,
    rq: multiprocessing.Queue,
    waiting_event: EventType | None = None,
) -> None:
    """Attach to a failing generation and report its published diagnostic."""
    coord = None
    started = time.monotonic()
    try:
        coord = ShmCoordinator(ns)
        assert coord.is_creator is False, "Expected follower role"
        _wait_for_ready(coord, waiting_event)
        rq.put({"success": False, "error": "wait_for_ready unexpectedly returned", "role": "follower"})
    except ShmCreatorFailedError as exc:
        rq.put({
            "success": True,
            "role": "follower",
            "failure_message": exc.failure_message,
            "elapsed": time.monotonic() - started,
        })
    except BaseException as exc:
        rq.put({"success": False, "error": f"{type(exc).__name__}: {exc}", "role": "follower"})
    finally:
        if coord is not None:
            coord.close()


def initializing_creator_process(ns: str, created_event: EventType, keep_alive_seconds: float) -> None:
    """Remain INITIALIZING until forcibly terminated by the parent."""
    coord = ShmCoordinator(ns, keep_alive_seconds=keep_alive_seconds)
    assert coord.is_creator is True, "Expected creator role"
    coord.write_profile(_make_minimal_state())
    FlexibleSharedMemory(name=ns + "_data", shm_size=8, _owner=coord.memory_owner)
    created_event.set()
    while True:
        time.sleep(1)


def dead_creator_follower_process(
    ns: str,
    rq: multiprocessing.Queue,
    waiting_event: EventType,
    keep_alive_seconds: float,
) -> None:
    """Report heartbeat-aware failure after the initializing creator is killed."""
    coord = None
    started = time.monotonic()
    try:
        coord = ShmCoordinator(ns, keep_alive_seconds=keep_alive_seconds)
        assert coord.is_creator is False, "Expected follower role"
        _wait_for_ready(coord, waiting_event)
        rq.put({"success": False, "error": "wait_for_ready unexpectedly returned", "role": "follower"})
    except ShmCreatorDiedError as exc:
        rq.put({
            "success": True,
            "role": "follower",
            "creator_identity": exc.creator_identity,
            "elapsed": time.monotonic() - started,
        })
    except BaseException as exc:
        rq.put({"success": False, "error": f"{type(exc).__name__}: {exc}", "role": "follower"})
    finally:
        if coord is not None:
            coord.close()


class TestShmCoordinatorMultiprocess:
    """Multiprocess tests for ShmCoordinator."""

    @pytest.fixture
    def namespace(self):
        return f"ft_mp_{int(time.time() * 1000) % 100000}"

    def test_creator_follower_flow(self, namespace):
        """Full creator/follower flow across two OS processes."""
        result_queue = multiprocessing.Queue()
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=creator_process,
            args=(namespace, result_queue, creator_ready, creator_close),
        )
        follower_proc = multiprocessing.Process(target=follower_process, args=(namespace, result_queue))

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "coordinator creator startup", proc=creator_proc)
            follower_proc.start()

            follower_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(follower_proc, "follower")
            creator_close.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=2)

            for result in results:
                assert result["success"], f"Process failed: {result.get('error', 'unknown')}"

            follower_result = next(r for r in results if r["is_creator"] is False)
            assert follower_result["loader_type"] == "allocation_block_transfer"
            assert follower_result["tensor_count"] == 2
            assert follower_result["block_count"] == 1
        finally:
            creator_close.set()
            for proc in [creator_proc, follower_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    def test_multiple_followers(self, namespace):
        """One creator, multiple followers all read the same profile."""
        result_queue = multiprocessing.Queue()
        num_followers = 3
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=creator_process,
            args=(namespace, result_queue, creator_ready, creator_close),
        )
        follower_procs = [
            multiprocessing.Process(target=follower_process, args=(namespace, result_queue))
            for _ in range(num_followers)
        ]

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "coordinator creator startup", proc=creator_proc)

            for proc in follower_procs:
                proc.start()

            for i, proc in enumerate(follower_procs):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"follower {i}")
            creator_close.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=1 + num_followers)

            for result in results:
                assert result["success"], f"Process failed: {result.get('error', 'unknown')}"

            follower_results = [r for r in results if r["is_creator"] is False]
            assert len(follower_results) == num_followers
            for fr in follower_results:
                assert fr["loader_type"] == "allocation_block_transfer"
                assert fr["tensor_count"] == 2
        finally:
            creator_close.set()
            for proc in [creator_proc, *follower_procs]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    def test_creator_liveness_detection(self, namespace):
        """Follower detects a creator whose heartbeat has gone stale."""
        keep_alive_seconds = 0.5
        result_queue = multiprocessing.Queue()
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()
        creator_closed = multiprocessing.Event()
        checker_connected = multiprocessing.Event()
        live_follower_connected = multiprocessing.Event()
        live_follower_close = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=short_lived_creator_process,
            args=(namespace, result_queue, creator_ready, creator_close, keep_alive_seconds),
        )
        checker_proc = multiprocessing.Process(
            target=liveness_checker_process,
            args=(namespace, result_queue, checker_connected, creator_closed, keep_alive_seconds),
        )
        live_follower_proc = multiprocessing.Process(
            target=live_follower_process,
            args=(
                namespace,
                result_queue,
                live_follower_connected,
                live_follower_close,
                keep_alive_seconds,
            ),
        )

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "short-lived coordinator creator startup", proc=creator_proc)
            live_follower_proc.start()
            wait_for_event(live_follower_connected, "live coordinator follower connection", proc=live_follower_proc)
            checker_proc.start()
            wait_for_event(checker_connected, "coordinator liveness checker connection", proc=checker_proc)
            creator_close.set()

            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "short-lived creator")
            creator_closed.set()
            checker_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(checker_proc, "liveness checker")
            live_follower_close.set()
            live_follower_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(live_follower_proc, "live follower")

            results = drain_results(result_queue, expected_count=3)
            assert all(r["success"] for r in results), f"Failed: {format_failed_results(results)}"

            checker_result = next(r for r in results if r.get("role") == "checker")
            assert checker_result["creator_alive"] is False
        finally:
            creator_close.set()
            live_follower_close.set()
            for proc in [creator_proc, checker_proc, live_follower_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    def test_explicit_failure_reaches_waiting_and_late_followers(self, namespace):
        """FAILED reaches a waiting follower and is sticky for a later follower."""
        result_queue = multiprocessing.Queue()
        creator_created = multiprocessing.Event()
        trigger_failure = multiprocessing.Event()
        failure_published = multiprocessing.Event()
        creator_close = multiprocessing.Event()
        follower_waiting = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=failing_creator_process,
            args=(
                namespace,
                result_queue,
                creator_created,
                trigger_failure,
                failure_published,
                creator_close,
            ),
        )
        waiting_follower_proc = multiprocessing.Process(
            target=failed_follower_process,
            args=(namespace, result_queue, follower_waiting),
        )
        late_follower_proc = multiprocessing.Process(
            target=failed_follower_process,
            args=(namespace, result_queue),
        )

        try:
            creator_proc.start()
            wait_for_event(creator_created, "failing creator startup", proc=creator_proc)
            waiting_follower_proc.start()
            wait_for_event(
                follower_waiting,
                "failure follower to enter wait_for_ready",
                proc=waiting_follower_proc,
            )
            trigger_failure.set()
            wait_for_event(failure_published, "creator failure publication", proc=creator_proc)
            late_follower_proc.start()

            waiting_follower_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(waiting_follower_proc, "waiting failure follower")
            late_follower_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(late_follower_proc, "late failure follower")
            creator_close.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "failed creator")

            results = drain_results(result_queue, expected_count=3)
            assert all(r["success"] for r in results), f"Failed: {format_failed_results(results)}"
            follower_results = [r for r in results if r["role"] == "follower"]
            assert len(follower_results) == 2
            assert {r["failure_message"] for r in follower_results} == {"ValueError: checkpoint load failed"}
            assert all(r["elapsed"] < EVENT_TIMEOUT / 2 for r in follower_results)
        finally:
            trigger_failure.set()
            creator_close.set()
            for proc in [creator_proc, waiting_follower_proc, late_follower_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    def test_wait_aborts_when_initializing_creator_is_killed(self, namespace):
        """Follower observes creator heartbeat expiry before its overall timeout."""
        keep_alive_seconds = 0.25
        result_queue = multiprocessing.Queue()
        creator_created = multiprocessing.Event()
        follower_waiting = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=initializing_creator_process,
            args=(namespace, creator_created, keep_alive_seconds),
        )
        follower_proc = multiprocessing.Process(
            target=dead_creator_follower_process,
            args=(namespace, result_queue, follower_waiting, keep_alive_seconds),
        )

        try:
            creator_proc.start()
            wait_for_event(creator_created, "initializing creator startup", proc=creator_proc)
            follower_proc.start()
            wait_for_event(follower_waiting, "dead-creator follower to enter wait_for_ready", proc=follower_proc)

            creator_proc.terminate()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert not creator_proc.is_alive(), "initializing creator did not terminate"

            result = drain_results(result_queue, expected_count=1)[0]
            assert result["success"], result.get("error", "unknown follower failure")
            assert result["creator_identity"] is not None
            assert result["elapsed"] < EVENT_TIMEOUT / 2
            # The result is published before teardown, so the assertion above
            # measures heartbeat detection rather than last-owner unlink time.
            follower_proc.join(timeout=EVENT_TIMEOUT * 2)
            assert_clean_exit(follower_proc, "dead-creator follower")
            for suffix in ("_data", "_prof", "_crd_b", "_crd"):
                with pytest.raises(posix_ipc.ExistentialError):
                    unexpected = posix_ipc.SharedMemory("/" + namespace + suffix)
                    unexpected.close_fd()
        finally:
            for proc in [creator_proc, follower_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

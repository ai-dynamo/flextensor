# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for SHM multiprocess tests.

Centralizing these keeps timeout/cleanup behavior consistent across the suite
and lets us bake exitcode and child-death checks into a single place.

Event-wait timeouts default to 5s and are overridable per-run via
``FT_TEST_EVENT_TIMEOUT`` (e.g. ``FT_TEST_EVENT_TIMEOUT=15 pytest ...``) for
slow CI runners where fork+import latency can starve the default budget.
"""

from __future__ import annotations

import os
import queue as _queue
import time
import warnings
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Any

import posix_ipc

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as EventType

EVENT_TIMEOUT: float = float(os.environ.get("FT_TEST_EVENT_TIMEOUT", "5.0"))
POLL_INTERVAL: float = 0.05
UNLINK_RETRY_SLEEP: float = 0.1


def wait_for_event(
    event: EventType,
    description: str,
    *,
    proc: BaseProcess | None = None,
    timeout: float = EVENT_TIMEOUT,
) -> None:
    """Wait for a child-process readiness event.

    If the event isn't set within ``timeout``, raises ``AssertionError``. When ``proc``
    is supplied and has already exited, the message reports the exitcode so a
    crashed worker is visible rather than presented as an opaque "timed out".

    Uses ``if not event.wait(...): raise`` rather than ``assert`` so the check is
    not stripped under ``python -O``.
    """
    if not event.wait(timeout=timeout):
        if proc is not None and not proc.is_alive():
            msg = (
                f"{description}: process exited with exitcode={proc.exitcode} "
                f"before signaling readiness within {timeout}s"
            )
            raise AssertionError(msg)
        msg = f"Timed out waiting for {description} after {timeout}s"
        raise AssertionError(msg)


def assert_clean_exit(proc: BaseProcess, description: str) -> None:
    """Assert a worker process has joined and exited cleanly.

    Call after ``proc.join(timeout=...)``. A non-zero exitcode or still-alive
    process is reported with context so downstream "missing result" assertions
    don't mask the real failure.

    Uses ``raise AssertionError`` rather than ``assert`` so the check survives
    ``python -O``.
    """
    if proc.is_alive():
        msg = f"{description}: still alive after join (likely hung)"
        raise AssertionError(msg)
    if proc.exitcode != 0:
        msg = f"{description}: exited with code {proc.exitcode}"
        raise AssertionError(msg)


def drain_results(
    result_queue: Queue,
    expected_count: int,
    *,
    timeout: float = EVENT_TIMEOUT,
) -> list[Any]:
    """Drain exactly ``expected_count`` results from ``result_queue``.

    Replaces the ``while not q.empty(): q.get()`` pattern, which races slow
    workers (``Queue.empty()`` is unreliable cross-process). An ``Empty`` from a
    bounded ``get`` becomes a loud failure pointing at the missing result count.
    """
    results: list[Any] = []
    deadline = time.monotonic() + timeout
    for _ in range(expected_count):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            results.append(result_queue.get(timeout=remaining))
        except _queue.Empty as exc:
            msg = f"Expected {expected_count} results from queue, only got {len(results)} within {timeout}s"
            raise AssertionError(msg) from exc
    return results


def format_failed_results(results: list[dict[str, Any]]) -> str:
    """Format `success=False` payloads for an assertion message.

    Distinguishes a real worker failure (``success=False`` with an ``error``)
    from a malformed payload (no ``success`` key at all) — the latter is a
    test-bug, and rendering them identically as ``'unknown'`` hides that.
    """
    malformed = [r for r in results if "success" not in r]
    failed = [r for r in results if r.get("success") is False]
    parts: list[str] = []
    if failed:
        parts.append("failed: " + "; ".join(f"{r.get('error', 'unknown')!r}" for r in failed))
    if malformed:
        parts.append("malformed (no 'success' key): " + "; ".join(f"keys={sorted(r)}" for r in malformed))
    if not parts:
        return "no failed results"
    return " | ".join(parts)


def unlink_semaphore_if_present(sem_name: str) -> None:
    """Best-effort removal of a POSIX semaphore used by a test.

    Catches the specific IPC error types we expect; other exceptions propagate.
    Emits ``ResourceWarning`` if all three retries miss — silent failure here
    leaves a stale resource around that masquerades as a product bug in the
    next test's setup (the very silent-failure shape this suite avoids).
    """
    last_exc: BaseException | None = None
    for _ in range(3):
        try:
            sem = posix_ipc.Semaphore(sem_name)
            sem.close()
            sem.unlink()
            return
        except (posix_ipc.ExistentialError, FileNotFoundError):
            return
        except (posix_ipc.BusyError, OSError) as exc:
            last_exc = exc
            time.sleep(UNLINK_RETRY_SLEEP)
    warnings.warn(
        f"unlink_semaphore_if_present({sem_name!r}) gave up after 3 retries; last error: {last_exc!r}",
        ResourceWarning,
        stacklevel=2,
    )


def unlink_shared_memory_if_present(shm_name: str) -> None:
    """Best-effort removal of a shared-memory segment used by a test.

    See ``unlink_semaphore_if_present`` for retry/warning rationale.
    """
    last_exc: BaseException | None = None
    for _ in range(3):
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(UNLINK_RETRY_SLEEP)
    warnings.warn(
        f"unlink_shared_memory_if_present({shm_name!r}) gave up after 3 retries; last error: {last_exc!r}",
        ResourceWarning,
        stacklevel=2,
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess regressions for retryable FlexibleSharedMemory teardown."""

from __future__ import annotations

import multiprocessing
import tempfile
import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from shm.multiprocess_utils import (
    EVENT_TIMEOUT,
    assert_clean_exit,
    unlink_shared_memory_if_present,
    wait_for_event,
)

from flextensor.shm import FlexibleSharedMemory, ProcessFileLock

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventType


def _hold_main_lock(name: str, acquired: EventType, release: EventType) -> None:
    """Hold one real file-lock generation until the parent permits release."""
    lock = ProcessFileLock(name, locked=False)
    try:
        lock.acquire(non_blocking=False, timeout=EVENT_TIMEOUT)
        acquired.set()
        if not release.wait(timeout=max(2.0, EVENT_TIMEOUT * 2)):
            raise TimeoutError("Timed out waiting for parent to release teardown contention")
    finally:
        lock.release()
        lock.close()


def _hold_keep_alive_lock(name: str, acquired: EventType, release: EventType) -> None:
    """Hold the namespace's inner keep-alive lock."""
    _hold_main_lock(f"{name}_ks", acquired, release)


def _cleanup_namespace(name: str) -> None:
    """Best-effort cleanup for failures before the retry under test completes."""
    for shm_name in (name, f"sm_{name}_kd", f"{name}_km"):
        unlink_shared_memory_if_present(shm_name)

    # ProcessFileLock normally retains the stable main-lock path. This test uses
    # a unique namespace, so remove its test-only files after every handle exits.
    temp_dir = Path(tempfile.gettempdir())
    for lock_name in (name, f"{name}_ks"):
        (temp_dir / f"{lock_name}.lock").unlink(missing_ok=True)


def test_contended_close_is_visible_and_retry_removes_stale_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A contended close stays retryable, then removes the stale generation."""
    ctx = multiprocessing.get_context("fork")
    name = f"fsmcr_{uuid.uuid4().hex[:8]}"
    lock_acquired = ctx.Event()
    release_lock = ctx.Event()
    holder = ctx.Process(target=_hold_main_lock, args=(name, lock_acquired, release_lock))
    shm = FlexibleSharedMemory(
        name=name,
        shm_size=1024,
        keep_alive_seconds=2,
        pinned_memory=False,
        lock_class=ProcessFileLock,
    )
    replacement: FlexibleSharedMemory | None = None

    try:
        holder.start()
        wait_for_event(lock_acquired, "main-lock holder startup", proc=holder)

        # Keep this integration test fast while exercising the real lock.
        monkeypatch.setattr("flextensor.shm.flexible_shm.LOCK_ACQUIRE_TIMEOUT", 0.2)

        with pytest.raises(RuntimeError, match=r"could not acquire .*main lock.*teardown"):
            shm.close()

        assert shm._closed is False
        assert shm.shm is not None
        assert shm.main_lock is not None
        # The failed close retained both its live handles and the published
        # namespace rather than silently reporting local-only success.
        attached = SharedMemory(name=name, create=False)
        attached.close()

        release_lock.set()
        holder.join(timeout=EVENT_TIMEOUT)
        assert_clean_exit(holder, "main-lock holder")

        shm.close()
        assert shm._closed is True

        replacement = FlexibleSharedMemory(
            name=name,
            shm_size=1024,
            keep_alive_seconds=2,
            pinned_memory=False,
            lock_class=ProcessFileLock,
        )
        assert replacement.shm_creator is True
    finally:
        release_lock.set()
        if holder.pid is not None:
            holder.join(timeout=EVENT_TIMEOUT)
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=EVENT_TIMEOUT)
        if replacement is not None:
            replacement.close()
        if not shm._closed:
            shm.close()
        _cleanup_namespace(name)


def test_contended_keep_alive_lock_bounds_close_and_allows_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Close retains its outer anchor when the inner keep-alive lock times out."""
    ctx = multiprocessing.get_context("fork")
    name = f"fsmki_{uuid.uuid4().hex[:8]}"
    lock_acquired = ctx.Event()
    release_lock = ctx.Event()
    holder = ctx.Process(target=_hold_keep_alive_lock, args=(name, lock_acquired, release_lock))
    shm = FlexibleSharedMemory(
        name=name,
        shm_size=1024,
        keep_alive_seconds=2,
        pinned_memory=False,
        lock_class=ProcessFileLock,
    )

    try:
        holder.start()
        wait_for_event(lock_acquired, "keep-alive lock holder startup", proc=holder)
        monkeypatch.setattr("flextensor.shm.flexible_shm.LOCK_ACQUIRE_TIMEOUT", 0.2)

        started = time.monotonic()
        with pytest.raises(BlockingIOError):
            shm.close()
        assert time.monotonic() - started < 2.0
        assert shm._closed is False
        attached = SharedMemory(name=name, create=False)
        attached.close()

        release_lock.set()
        holder.join(timeout=EVENT_TIMEOUT)
        assert_clean_exit(holder, "keep-alive lock holder")

        shm.close()
        assert shm._closed is True
    finally:
        release_lock.set()
        if holder.pid is not None:
            holder.join(timeout=EVENT_TIMEOUT)
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=EVENT_TIMEOUT)
        if not shm._closed:
            shm.close()
        _cleanup_namespace(name)

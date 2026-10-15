# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable unit tests for synchronization primitive control flow."""

from unittest.mock import MagicMock, patch

import pytest

from flextensor.shm import ProcessFileLock


def _uninitialized_file_lock() -> ProcessFileLock:
    lock = ProcessFileLock.__new__(ProcessFileLock)
    lock.fd = MagicMock()
    lock.fd.fileno.return_value = 17
    lock._thread_lock = MagicMock()
    lock._init_non_blocking = False
    lock._held = False
    return lock


def test_process_file_lock_uses_one_deadline_across_both_lock_layers() -> None:
    """Time spent on the thread lock is deducted from the file-lock budget."""
    lock = _uninitialized_file_lock()
    clock = [10.0]

    def acquire_thread_lock(*, blocking: bool, timeout: float) -> bool:
        assert blocking is True
        assert timeout == pytest.approx(1.0)
        clock[0] += 0.75
        return True

    def advance_clock(seconds: float) -> None:
        clock[0] += seconds

    lock._thread_lock.acquire.side_effect = acquire_thread_lock

    with (
        patch("time.monotonic", side_effect=lambda: clock[0]),
        patch("time.time", side_effect=lambda: clock[0]),
        patch("time.sleep", side_effect=advance_clock),
        patch("flextensor.shm.sync_primitives.fcntl.flock", side_effect=BlockingIOError("busy")),
        pytest.raises(BlockingIOError),
    ):
        lock.acquire(timeout=1.0)

    assert clock[0] <= 11.01
    lock._thread_lock.release.assert_called_once_with()
    assert lock._held is False


def test_process_file_lock_release_then_close_does_not_release_another_owner() -> None:
    """Closing a released instance cannot unlock a shared lock now held elsewhere."""
    lock = _uninitialized_file_lock()
    lock._held = True

    with patch("flextensor.shm.sync_primitives.fcntl.flock") as flock:
        lock.release()
        # Another instance/thread may acquire the shared lock here. This
        # instance no longer owns it, so close must not release it again.
        lock.close()

    assert flock.call_count == 1
    lock._thread_lock.release.assert_called_once_with()
    assert lock.fd is None


def test_process_file_lock_unlink_name_keeps_local_handle_and_ownership(tmp_path) -> None:
    """Removing the stable name is distinct from closing or releasing this handle."""
    anchor = tmp_path / "anchor.lock"
    anchor.touch()
    lock = _uninitialized_file_lock()
    lock.path = str(anchor)
    lock._held = True

    lock.unlink_name()

    assert not anchor.exists()
    assert lock.fd is not None
    assert lock._held is True
    lock.fd.close.assert_not_called()
    lock._thread_lock.release.assert_not_called()

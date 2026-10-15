# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for multiprocess test cleanup helpers."""

import contextlib
import uuid

import posix_ipc
import pytest
from shm.multiprocess_utils import unlink_semaphore_if_present


@pytest.mark.parametrize("leading_slash", [False, True])
def test_unlink_semaphore_after_handles_close(leading_slash):
    """Cleanup removes the name even after every participant has closed it."""
    name = f"/sem_clean_{uuid.uuid4().hex[:8]}"
    semaphore = posix_ipc.Semaphore(name, flags=posix_ipc.O_CREX)
    semaphore.close()

    try:
        cleanup_name = name if leading_slash else name[1:]
        unlink_semaphore_if_present(cleanup_name)
        with pytest.raises(posix_ipc.ExistentialError):
            remaining = posix_ipc.Semaphore(name)
            remaining.close()
        # Removing an already absent semaphore must also succeed.
        unlink_semaphore_if_present(cleanup_name)
    finally:
        with contextlib.suppress(posix_ipc.ExistentialError):
            posix_ipc.unlink_semaphore(name)

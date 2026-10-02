# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess condition variable and shared string buffer.

This module provides synchronization primitives for coordinating multiple
processes using shared memory and semaphore-based signaling.
"""

import contextlib
import logging
import os
import struct
import sys
from collections import Counter
from multiprocessing.resource_tracker import register as resource_tracker_register
from multiprocessing.resource_tracker import unregister as resource_tracker_unregister
from multiprocessing.shared_memory import SharedMemory

from flextensor.shm.sync_primitives import BaseLock, ProcessFileLock, SemaphoreLock

if sys.platform != "win32":
    import posix_ipc

logger = logging.getLogger(__name__)

HEADER_SIZE = 5


def _unlink_shared_memory(shm: SharedMemory) -> None:
    """Unlink an untracked mapping without leaving a stale tracker entry."""
    tracker_name = shm._name  # noqa: SLF001
    resource_tracker_register(tracker_name, "shared_memory")
    try:
        shm.unlink()
    except BaseException:
        resource_tracker_unregister(tracker_name, "shared_memory")
        raise


class SharedMultiString:
    """
    Shared multi-string for multiprocess coordination.

    This class provides a shared memory buffer for storing a list of strings.
    The strings are stored in the shared memory buffer in a format that is easy to parse.
    The strings are separated by a pipe character.
    The first 4 bytes of the shared memory buffer are used to store the length of the list.
    The next byte is used to store a boolean value indicating if the list is ready.
    The rest of the shared memory buffer is used to store the strings.

    Note:
        This class is **not thread-safe**. If you need to use it from multiple threads,
        you must protect all accesses (reads and writes) with an external lock.
    """

    def __init__(self, name: str, size: int = 1024 * 128, create: bool = False) -> None:
        """Initialize the shared multi-string buffer.

        Args:
            name: Name for the shared memory segment.
            size: Size in bytes for the buffer.
            create: If True, create the shared memory; otherwise connect to existing.
        """
        # Respect the create parameter
        if sys.platform == "win32":
            raise NotImplementedError("Shared memory (SHM) is not supported on Windows")
        self.shm: SharedMemory | None = None
        self._created = False
        if create:
            try:
                self.shm = SharedMemory(name=name, size=size, create=True)
                self._created = True
                # Initialize the header for new shared memory
                self.shm.buf[0:4] = struct.pack("I", HEADER_SIZE)
                self.shm.buf[4:5] = struct.pack("?", False)
            except FileExistsError:
                # Already exists, connect to it
                self.shm = SharedMemory(name=name, create=False)
        else:
            # Connect to existing
            self.shm = SharedMemory(name=name, create=False)
        # Unregister from resource tracker to prevent spurious cleanup warnings in multiprocess scenarios
        # Use _name (with / prefix) to match what resource_tracker registered
        resource_tracker_unregister(self.shm._name, "shared_memory")  # noqa: SLF001
        self._shm_name = self.shm.name
        self._name_unlinked = False

    def append(self, string: str) -> None:
        """Append a string to the buffer."""
        if self.shm is None:
            msg = "SharedMultiString is closed"
            raise ValueError(msg)
        pos = struct.unpack("I", self.shm.buf[0:4])[0]
        if pos > HEADER_SIZE:
            string = "|" + string
        arr = string.encode("utf-8")
        if pos + len(arr) > self.shm.size:
            msg = "Shared memory is full"
            raise ValueError(msg)
        self.shm.buf[pos : pos + len(arr)] = arr
        pos += len(arr)
        self.shm.buf[0:4] = struct.pack("I", pos)

    def set_ready(self) -> None:
        """Mark the buffer as ready."""
        if self.shm is None:
            msg = "SharedMultiString is closed"
            raise ValueError(msg)
        self.shm.buf[4:5] = struct.pack("?", True)

    def is_ready(self) -> bool:
        """Check if the buffer has been marked as ready."""
        if self.shm is None:
            msg = "SharedMultiString is closed"
            raise ValueError(msg)
        return struct.unpack("?", self.shm.buf[4:5])[0]

    def get_list(self) -> list[str]:
        """Return all strings in the buffer as a list."""
        if self.shm is None:
            msg = "SharedMultiString is closed"
            raise ValueError(msg)
        # Read the current content length from shared memory header
        # (don't use self.pos as it may be stale in multiprocess scenarios)
        pos = struct.unpack("I", self.shm.buf[0:4])[0]
        # Filter out empty strings that can result from splitting
        s = bytes(self.shm.buf[HEADER_SIZE:pos]).decode("utf-8")
        return s.split("|") if s else []

    def close(self) -> None:
        """Close the shared memory without unlinking."""
        if self.shm is not None:
            self.shm.close()
            self.shm = None

    def unlink_name(self) -> None:
        """Remove the global SHM name without closing the local mapping."""
        if getattr(self, "_name_unlinked", False):
            return
        shm = self.shm
        close_after = False
        if shm is None:
            try:
                shm = SharedMemory(name=self._shm_name, create=False)
            except FileNotFoundError:
                self._name_unlinked = True
                return
            close_after = True
        try:
            with contextlib.suppress(FileNotFoundError):
                _unlink_shared_memory(shm)
        finally:
            if close_after:
                shm.close()
        self._name_unlinked = True

    def unlink(self) -> None:
        """Unlink and close the shared memory, removing it from the system."""
        self.unlink_name()
        self.close()


class MultiprocessCondition:
    """
    Multiprocess condition for multiprocess coordination.

    This class provides a mechanism for processes to wait for a condition to be met.
    The condition is set by one process and other processes can wait for it to be met.
    """

    def __init__(self, name: str, is_creator: bool = False, lock_class: type[BaseLock] = ProcessFileLock) -> None:
        """Initialize the multiprocess condition.

        Args:
            name: Name for shared resources.
            is_creator: If True, create shared resources; otherwise connect.
            lock_class: Lock class to use (default: ProcessFileLock).
        """
        self.name = name
        notification_list = SharedMultiString(name=name, create=is_creator)
        self.lock_class = lock_class
        try:
            lock = self.lock_class(name, locked=False)
        except BaseException as error:
            try:
                (notification_list.unlink if notification_list._created else notification_list.close)()  # noqa: SLF001
            except BaseException as cleanup_error:
                if add_note := getattr(error, "add_note", None):
                    add_note(f"Failed to roll back condition shared memory: {cleanup_error!r}")
            raise
        self.notification_list: SharedMultiString | None = notification_list
        self.lock: BaseLock | None = lock
        self._closed_lock: BaseLock | None = None

    def wait_for_ready(self) -> None:
        """Block until notify_ready() is called."""
        lock = self.lock
        notification_list = self.notification_list
        if lock is None or notification_list is None:
            raise RuntimeError("MultiprocessCondition is closed")
        pid = os.getpid()
        notification_lock = None
        with lock:
            if notification_list.is_ready():
                return
            lock_name = f"{self.name}_{pid}"
            # Other local waiters may already have consumed the semaphore token.
            # Acquire only after releasing the registration lock.
            notification_lock = SemaphoreLock(lock_name, locked=False)
            notification_list.append(f"{pid}")

        try:
            while not notification_list.is_ready():
                notification_lock.acquire(non_blocking=False)
        finally:
            if notification_lock is not None:
                # The notifier owns unlinking; other local waiters may still need the name.
                with contextlib.suppress(FileNotFoundError, AttributeError, OSError):
                    notification_lock.close()

    def notify_ready(self) -> None:
        """Signal every registered waiter that the condition is ready."""
        lock = self.lock
        notification_list = self.notification_list
        if lock is None or notification_list is None:
            raise RuntimeError("MultiprocessCondition is closed")
        with lock:
            notification_list.set_ready()
            pids = notification_list.get_list()
            delivery_errors: list[Exception] = []
            for waiter_pid, waiter_count in Counter(pids).items():
                if not waiter_pid:
                    continue
                lock_name = f"{self.name}_{waiter_pid}"
                try:
                    notification_lock = SemaphoreLock(lock_name, locked=False, create=False)
                except posix_ipc.ExistentialError:
                    continue
                except Exception as error:
                    logger.warning(
                        "Failed to attach to notification for process %s (lock_name=%s): %s",
                        waiter_pid,
                        lock_name,
                        error,
                    )
                    delivery_errors.append(error)
                    continue

                try:
                    for _ in range(waiter_count):
                        notification_lock.signal()
                    notification_lock.unlink_name()
                except Exception as error:
                    logger.warning(
                        "Failed to deliver notification to process %s (lock_name=%s): %s",
                        waiter_pid,
                        lock_name,
                        error,
                    )
                    delivery_errors.append(error)
                finally:
                    try:
                        notification_lock.close()
                    except Exception as error:
                        logger.warning(
                            "Failed to close notification for process %s (lock_name=%s): %s",
                            waiter_pid,
                            lock_name,
                            error,
                        )
                        delivery_errors.append(error)

            if delivery_errors:
                raise delivery_errors[0]

    def close(self) -> None:
        """Close the notification list without unlinking."""
        if self.notification_list is not None:
            self.notification_list.close()

    def unlink(self) -> None:
        """Unlink and close all shared resources."""
        self.unlink_names()
        if self.notification_list is not None:
            self.notification_list.close()
            self.notification_list = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None
        self._closed_lock = None

    def unlink_names(self) -> None:
        """Remove condition data/lock names without closing local handles."""
        if self.notification_list is not None:
            self.notification_list.unlink_name()
        lock = self.lock or getattr(self, "_closed_lock", None)
        if lock is not None:
            lock.unlink_name()

    def close_lock(self) -> None:
        """Close only the lock resource."""
        if self.lock is not None:
            self.lock.close()
            self._closed_lock = self.lock
            self.lock = None

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronization primitives for multiprocess coordination.

This module provides an abstraction layer for different locking mechanisms
to enable comparison and selection of the most reliable implementation.

Thread Safety:
    SemaphoreLock: Thread-safe. The underlying POSIX semaphores are managed by
    the kernel, which handles synchronization correctly across threads and processes.

    ProcessFileLock: Thread-safe. Uses a class-level registry that maps file paths
    to threading.Lock instances, ensuring all instances for the same path share
    the same thread lock. The underlying fcntl.flock provides inter-process
    synchronization, while the shared threading.Lock provides inter-thread
    synchronization.

    The release() method is idempotent when the lock is not held. Failures from
    an underlying unlock operation propagate and leave the lock logically held
    so callers can retry the release. An underlying "already released" result
    is reconciled as success.
"""

import _thread
import contextlib
import fcntl
import tempfile
import threading
import time
import weakref
from abc import ABC, abstractmethod
from typing import ClassVar

import posix_ipc


class BaseLock(ABC):
    """Abstract base class for lock implementations.

    Thread-safe base class that provides common context manager implementation.
    Subclasses must implement acquire(), release(), close(), and unlink().

    Args:
        name: Name/identifier for the lock (or path for file-based locks)
        locked: If True, acquire the lock immediately upon creation.
        non_blocking: Behavior when locked=True:
            - True: try to acquire non-blocking, raise exception if fails
            - False: blocking acquire (wait for lock)

    Attributes:
        _init_non_blocking: Stored non_blocking setting for context manager use.
    """

    _init_non_blocking: bool

    @abstractmethod
    def __init__(self, name: str, locked: bool = True, non_blocking: bool = False) -> None: ...

    def __enter__(self):
        """Enter context manager - acquire lock."""
        self.acquire(non_blocking=self._init_non_blocking)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - release lock."""
        self.release()
        return False

    @abstractmethod
    def acquire(self, non_blocking: bool = False, timeout: float | None = None):
        """Acquire the lock.

        Args:
            non_blocking: If True, try non-blocking acquire. If False, block until acquired.
            timeout: Maximum time in seconds to wait (only used if non_blocking=False).
                    None means wait indefinitely.

        Raises:
            BlockingIOError or posix_ipc.BusyError: If non_blocking=True and lock is busy,
                or if timeout expires.
        """

    @abstractmethod
    def release(self):
        """Release the lock.

        This method is idempotent - calling it when the lock is not held
        will silently succeed (no exception raised). An underlying release
        failure propagates and leaves the lock logically held for retry. An
        underlying "already released" result is reconciled as success.
        """

    @abstractmethod
    def close(self):
        """Close the lock resource."""

    @abstractmethod
    def unlink(self):
        """Remove the lock resource from the system."""

    @abstractmethod
    def unlink_name(self):
        """Remove only the global lock name, retaining this local handle."""


class SemaphoreLock(BaseLock):
    """Lock implementation using POSIX semaphores.

    Thread-safe lock that wraps posix_ipc.Semaphore to provide the BaseLock interface.
    The underlying POSIX semaphore is managed by the kernel for thread/process safety.
    """

    def __init__(
        self,
        name: str,
        locked: bool = True,
        non_blocking: bool = False,
        *,
        create: bool = True,
    ) -> None:
        """Initialize semaphore lock.

        Args:
            name: Name of the semaphore (must start with /)
            locked: If True, acquire lock immediately after creation.
            non_blocking: Only used when locked=True:
                - True: non-blocking acquire (raises error if busy)
                - False: blocking acquire (waits)
            create: Create the semaphore when it does not already exist.
        """
        self.name = name if name.startswith("/") else f"/{name}"
        self._init_non_blocking = non_blocking
        self._held = False
        self._closed = False

        flags = posix_ipc.O_CREAT if create else 0
        self.lock = posix_ipc.Semaphore(self.name, flags=flags, initial_value=1)

        # Acquire lock if requested
        if locked:
            self.acquire(non_blocking=non_blocking)

    def acquire(self, non_blocking: bool = False, timeout: float | None = None):
        """Acquire the lock.

        Args:
            non_blocking: If True, try non-blocking acquire. If False, block until acquired.
            timeout: Maximum time in seconds to wait (only used if non_blocking=False).
                    None means wait indefinitely. If timeout expires, raises posix_ipc.BusyError.

        Raises:
            posix_ipc.BusyError: If non_blocking=True and lock is busy, or if timeout expires.
        """
        if getattr(self, "_closed", False):
            raise posix_ipc.ExistentialError("SemaphoreLock is closed")
        if non_blocking:
            # Non-blocking acquire
            self.lock.acquire(timeout=0)
        elif timeout is not None:
            # Blocking acquire with timeout
            self.lock.acquire(timeout=timeout)
        else:
            # Blocking acquire, wait indefinitely
            self.lock.acquire()
        self._held = True

    def release(self):
        """Release the lock.

        This method is idempotent - calling it when the lock is not held
        will silently succeed (no-op if not held). An underlying semaphore
        release failure propagates and leaves the lock logically held for retry.
        """
        if getattr(self, "_closed", False) or not self._held:
            return

        # Mark the release in progress so a concurrent idempotent release is a
        # no-op. Restore ownership if the underlying release fails so the
        # caller can retry instead of silently losing track of a held lock.
        self._held = False
        try:
            self.lock.release()
        except ValueError:
            # POSIX reports overflow when the semaphore is already at its
            # maximum value.  The desired unlocked postcondition is already
            # true, so reconcile stale local ownership instead of poisoning
            # every later retry.
            return
        except BaseException:
            self._held = True
            raise

    def signal(self) -> None:
        """Increment the semaphore from another process (e.g. to wake a waiter).

        Use this when a different process needs to \"release\" the semaphore to
        unblock a waiter that is blocked on acquire(). Unlike release(), this
        does not check _held and always increments the underlying semaphore.
        """
        if getattr(self, "_closed", False):
            raise posix_ipc.ExistentialError("SemaphoreLock is closed")
        with contextlib.suppress(ValueError):
            self.lock.release()

    def close(self):
        """Close the semaphore."""
        if getattr(self, "_closed", False):
            return
        if self._held:
            self.release()
        self.lock.close()
        self._closed = True

    def unlink(self):
        """Remove the semaphore from the system."""
        self.unlink_name()

    def unlink_name(self):
        """Remove the semaphore name without closing this process's handle."""
        with contextlib.suppress(posix_ipc.ExistentialError):
            posix_ipc.unlink_semaphore(self.name)


class ProcessFileLock(BaseLock):
    """File-based lock using fcntl.flock (Linux/Unix) for inter-process and inter-thread synchronization.

    Thread Safety:
        This lock is thread-safe. It uses a class-level registry that maps file paths to
        threading.Lock instances. Multiple ProcessFileLock instances for the same path
        (even created separately) will share the same thread lock, ensuring proper
        synchronization between threads within the same process.

        The underlying fcntl.flock() provides inter-process synchronization, while the
        shared threading.Lock provides inter-thread synchronization.

    Args:
        path: Path to the lock file.
        locked: If True, acquire lock immediately in constructor.
        non_blocking: Only used when locked=True:
            - True: tries LOCK_EX | LOCK_NB (immediate exception if busy)
            - False: tries LOCK_EX (waits until available)
    """

    # Class-level registry for thread locks - maps normalized paths to threading.Lock instances
    # Note: Use _thread.LockType for type hints because threading.Lock is a factory function,
    # not a type class. This is needed for beartype compatibility with Python 3.12.
    _registry_lock: ClassVar[_thread.LockType] = threading.Lock()
    # Closed generation-specific locks must not accumulate in this registry.
    # Each open handle retains its thread lock strongly.
    _thread_locks: ClassVar[weakref.WeakValueDictionary[str, _thread.LockType]] = weakref.WeakValueDictionary()

    @staticmethod
    def resolve_path(path: str) -> str:
        """Resolve a lock name without opening or creating its file."""
        from pathlib import Path

        if Path(path).is_absolute():
            return path
        return str(Path(tempfile.gettempdir()) / f"{path.lstrip('/')}.lock")

    def __init__(self, path: str, locked: bool = True, non_blocking: bool = False) -> None:
        """Initialize file lock.

        Args:
            path: Path/name for the lock file
            locked: If True, acquire lock immediately
            non_blocking: Behavior when locked=True (see class docstring)
        """
        from pathlib import Path

        # Handle path construction
        self.path = self.resolve_path(path)

        # Ensure directory exists
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        # Get or create thread lock for this path from the class-level registry
        # Uses double-check locking pattern for thread safety
        self._thread_lock = self._get_or_create_thread_lock(self.path)

        self.fd: object | None = None
        self._init_non_blocking = non_blocking
        self._held = False

        # Open file (or create) - kept open for the lifetime of the lock
        self.fd = Path(self.path).open("a+")  # noqa: SIM115

        if locked:
            self.acquire(non_blocking=non_blocking)

    @classmethod
    def _get_or_create_thread_lock(cls, path: str) -> _thread.LockType:
        """Get or create a thread lock for the given path.

        Uses double-check locking pattern for thread-safe lazy initialization.

        Args:
            path: The normalized path to the lock file.

        Returns:
            The shared threading.Lock instance for this path.
        """
        # Fast path: lock already exists
        lock = cls._thread_locks.get(path)
        if lock is not None:
            return lock

        # Slow path: need to create lock
        with cls._registry_lock:
            # Double-check after acquiring registry lock
            lock = cls._thread_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                cls._thread_locks[path] = lock
            return lock

    def _acquire_thread_lock(self, non_blocking: bool, timeout: float | None) -> None:
        """Acquire the thread lock for inter-thread synchronization.

        Args:
            non_blocking: If True, try non-blocking acquire.
            timeout: Maximum time in seconds to wait (only if non_blocking=False).

        Raises:
            BlockingIOError: If lock cannot be acquired.
        """
        if non_blocking:
            if not self._thread_lock.acquire(blocking=False):
                raise BlockingIOError("Thread lock is busy")
        elif timeout is not None:
            if not self._thread_lock.acquire(blocking=True, timeout=timeout):
                raise BlockingIOError(f"Failed to acquire thread lock within {timeout} seconds")
        else:
            self._thread_lock.acquire()

    def _acquire_file_lock(self, non_blocking: bool, timeout: float | None) -> None:
        """Acquire the file lock for inter-process synchronization.

        Args:
            non_blocking: If True, try non-blocking acquire.
            timeout: Remaining time in seconds, or None to wait indefinitely.

        Raises:
            BlockingIOError: If lock cannot be acquired.
        """
        if non_blocking or timeout is None:
            flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
            fcntl.flock(self.fd.fileno(), flags)
        else:
            self._acquire_file_lock_until(time.monotonic() + timeout)

    def _acquire_file_lock_until(self, deadline: float) -> None:
        """Acquire the file lock before an absolute monotonic deadline.

        Args:
            deadline: Absolute ``time.monotonic()`` deadline.

        Raises:
            BlockingIOError: If timeout expires.
        """
        while True:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BlockingIOError("Failed to acquire file lock before deadline") from None
                time.sleep(min(0.01, remaining))

    def acquire(self, non_blocking: bool = False, timeout: float | None = None):
        """Acquire the lock.

        This method is thread-safe: it first acquires a shared thread lock for this path,
        then acquires the file lock for inter-process synchronization.

        Args:
            non_blocking:
                - False: blocking LOCK_EX (waits)
                - True: LOCK_EX | LOCK_NB (BlockingIOError if busy)
            timeout: Maximum time in seconds to wait. Only used if non_blocking=False.
                    Note: fcntl.flock doesn't support native timeout, so we use polling.
                    None means wait indefinitely.

        Raises:
            BlockingIOError: If non_blocking=True and lock is busy, or if timeout expires.
            ValueError: If the lock has been closed.
        """
        if self.fd is None:
            msg = "ProcessFileLock is closed"
            raise ValueError(msg)
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")

        deadline = None if timeout is None else time.monotonic() + timeout
        self._acquire_thread_lock(non_blocking, timeout)
        try:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            self._acquire_file_lock(non_blocking, remaining)
        except BaseException:
            self._thread_lock.release()
            raise
        self._held = True

    def release(self):
        """Release the lock without closing the file.

        This method releases both the file lock (inter-process) and the thread lock
        (inter-thread). It is idempotent - calling it when the lock is not held
        will silently succeed. An underlying release failure propagates and leaves
        the lock logically held so the caller can retry.
        """
        if self.fd is None or not self._held:
            return

        # Mark the release in progress so concurrent idempotent calls do not
        # attempt a second unlock. Restore ownership if either layer fails;
        # this also keeps the thread lock held when LOCK_UN itself fails.
        self._held = False
        try:
            # Release file lock first, then the thread lock so another thread
            # cannot enter while this process still owns the file lock.
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
        except BaseException:
            self._held = True
            raise

        try:
            self._thread_lock.release()
        except RuntimeError:
            # threading.Lock raises only when it is already unlocked.  The
            # release postcondition is therefore satisfied for both layers.
            return
        except BaseException:
            # A non-standard lock may expose another retryable failure. The
            # file layer is already unlocked; repeating LOCK_UN is harmless.
            self._held = True
            raise

    def close(self):
        """Close the lock file."""
        self.release()
        if self.fd is not None:
            self.fd.close()
            self.fd = None

    def unlink(self):
        """Remove the lock file from the system."""
        self.close()
        self.unlink_name()

    def unlink_name(self):
        """Remove the lock path without releasing or closing this handle."""
        from pathlib import Path

        path_obj = Path(self.path)
        if path_obj.exists():
            with contextlib.suppress(FileNotFoundError):
                path_obj.unlink()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flexible shared memory management with keep-alive tracking.

This module provides classes for managing shared memory segments across multiple
processes with automatic liveness tracking and coordinated cleanup.
"""

import contextlib
import json
import math
import os
import secrets
import sys
import threading
import time
from collections.abc import Callable, Iterator
from multiprocessing.resource_tracker import register as resource_tracker_register
from multiprocessing.resource_tracker import unregister as resource_tracker_unregister
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, ClassVar

import torch
from shared_memory_dict import SharedMemoryDict

from flextensor.host_pinning import HostPinRegistry
from flextensor.shm.multiprocess_condition import SharedMultiString
from flextensor.shm.sync_primitives import BaseLock, ProcessFileLock, SemaphoreLock

if sys.platform != "win32":
    import posix_ipc

LOCK_ACQUIRE_TIMEOUT = 5.0  # Must be float for beartype type checking
_PARTICIPANT_COUNTER_BITS = 32
_PARTICIPANT_COUNTER_LIMIT = 1 << _PARTICIPANT_COUNTER_BITS
_participant_counter = 0
_participant_counter_lock = threading.Lock()


def _unlink_untracked_shm(shm: SharedMemory) -> None:
    """Unlink a tracker-exempt SHM handle without leaving a stale registration."""
    resource_tracker_register(shm._name, "shared_memory")  # type: ignore[attr-defined]  # noqa: SLF001
    try:
        shm.unlink()
    except BaseException as error:
        resource_tracker_unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]  # noqa: SLF001
        if not isinstance(error, FileNotFoundError):
            raise


def _next_participant_id(process_id: int) -> int:
    """Return a collision-free per-process participant identifier."""
    global _participant_counter

    with _participant_counter_lock:
        sequence = _participant_counter
        if sequence >= _PARTICIPANT_COUNTER_LIMIT:
            raise RuntimeError("FlexibleSharedMemory participant counter exhausted")
        _participant_counter += 1
    return (process_id << _PARTICIPANT_COUNTER_BITS) | sequence


class KeepAliveDict:
    """Shared memory dictionary mapping participant IDs to timestamps for liveness tracking.

    Combines SharedMemoryDict for participant ID to slot mapping with a SharedMemory
    buffer for storing timestamps. Allows efficient liveness checks across processes.
    The ``process_id`` method parameters accept participant IDs; their names are
    retained for keyword compatibility.

    Note:
        This dictionary is **not thread-safe** or multi-process safe on its own.
        All modifications (e.g., add, remove, update) must be externally synchronized
        using appropriate locking to avoid races and corruption.
        The caller is responsible for using a multiprocessing lock or similar mechanism
        around all modifying operations.
    """

    def __init__(self, name: str, size: int, is_creator: bool, meta_name: str | None = None) -> None:
        """Initialize the keep-alive dictionary.

        Args:
            name: Name for the SharedMemoryDict.
            size: Size in bytes for both the dict and timestamp buffer.
            is_creator: If True, create the shared memory; otherwise connect to existing.
            meta_name: Optional separate name for the timestamp SharedMemory buffer.
        """
        self.name = name
        self.size = size

        # Allow separate name for metadata to control exact names (avoid length limits)
        if meta_name is None:
            meta_name = name + "_m"

        # SharedMemoryDict handles both creation and connection automatically
        # Note: SharedMemoryDict will add "sm_" prefix internally
        self.dict = SharedMemoryDict(name=name, size=size)
        # Use _name (with / prefix) to match what resource_tracker registered
        resource_tracker_unregister(self.dict.shm._name, "shared_memory")  # noqa: SLF001

        # For SharedMemory, respect is_creator but handle already-exists gracefully
        if is_creator:
            try:
                shm = SharedMemory(name=meta_name, create=True, size=size)
            except FileExistsError:
                # Creator expected to create, but it already exists - connect to it
                shm = SharedMemory(name=meta_name, create=False)
        else:
            # Not creator - just connect
            shm = SharedMemory(name=meta_name, create=False)
        self.shm: SharedMemory | None = shm
        # Use _name (with / prefix) to match what resource_tracker registered
        resource_tracker_unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]  # noqa: SLF001

        # macOS may round the POSIX mapping up to a page; expose only the
        # protocol-requested timestamp slots so capacity is platform-stable.
        shm_buffer = shm.buf
        if shm_buffer is None:
            raise RuntimeError("Shared memory buffer is unavailable")
        self.shm_view: memoryview | None = shm_buffer.cast("d")[: size // 8]  # type: ignore[assignment]
        self._closed = False
        self._meta_name_unlinked = False
        self._dict_name_unlinked = False

    def _view(self) -> Any:
        if self.shm_view is None:
            raise RuntimeError("Keep-alive dictionary is closed")
        return self.shm_view

    def prepare_slot(self, process_id: int) -> int:
        """Allocate a slot for a participant. It is not thread-safe or multi-process safe on its own."""
        if process_id in self.dict:
            return int(self.dict[process_id])
        shm_view = self._view()

        # Always compute the next available slot dynamically based on current dict state
        # This ensures correct behavior across multiple participants
        used_slots: set[int] = set()
        max_slot = -1
        for _, existing_slot in self.dict.items():
            used_slots.add(existing_slot)
            if existing_slot > max_slot:
                max_slot = existing_slot

        # Find a free slot (either reuse a gap or allocate new)
        slot: int | None = None
        for candidate in range(max_slot + 1):
            if candidate not in used_slots:
                slot = candidate
                break
        if slot is None:
            slot = max_slot + 1

        # POSIX may round tiny SHM allocations up to a page. Enforce the
        # requested protocol capacity rather than the platform allocation size.
        capacity = len(shm_view)
        if slot >= capacity:
            msg = (
                f"KeepAliveDict timestamp slots exhausted: participant_id={process_id}, slot={slot}, "
                f"timestamp_capacity={capacity}, keep_alive_dict_size={self.size}"
            )
            raise RuntimeError(msg)

        try:
            candidate_mapping = dict(self.dict.items())
            candidate_mapping[process_id] = slot
            serialized_mapping = self.dict._serializer.dumps(candidate_mapping)  # noqa: SLF001
            if len(serialized_mapping) > self.size:
                raise ValueError("exceeds configured storage")
            self.dict[process_id] = slot
        except ValueError as e:
            msg = (
                f"KeepAliveDict mapping storage exhausted: participant_id={process_id}, "
                f"slot={slot}, keep_alive_dict_size={self.size}"
            )
            raise RuntimeError(msg) from e
        shm_view[slot] = time.time()
        return slot

    def set_by_slot(self, slot: int, timestamp: int | float) -> None:
        """Set the timestamp for a given slot index."""
        shm_view = self._view()
        shm_view[slot] = float(timestamp)

    def pop(self, process_id: int) -> float:
        """Remove a participant from the dict. Must be called while holding the keep-alive lock."""
        slot = int(self.dict.pop(process_id))
        shm_view = self._view()
        val = float(shm_view[slot])
        # Clear the slot timestamp (slot will be reused by prepare_slot if needed)
        shm_view[slot] = 0.0
        return val

    def items(self) -> Iterator[tuple[int, float]]:
        """Yield (participant_id, timestamp) pairs for all registered participants."""
        shm_view = self._view()
        for participant_id, slot in self.dict.items():
            yield participant_id, float(shm_view[slot])

    def __contains__(self, process_id: int) -> bool:
        """Check if the participant ID passed as process_id is in the dictionary."""
        return process_id in self.dict

    def close(self) -> None:
        """Close shared memory resources without unlinking."""
        if self._closed:
            return

        # Release the memoryview first to allow proper cleanup
        if self.shm_view is not None:
            self.shm_view.release()
            self.shm_view = None

        # Close SharedMemoryDict's shared memory
        if self.dict is not None:
            self.dict.shm.close()

        # Close our shared memory
        if self.shm is not None:
            self.shm.close()

        self._closed = True

    def unlink_names(self) -> None:
        """Remove global data/meta names without closing local mappings."""
        if self.shm is not None and not self._meta_name_unlinked:
            _unlink_untracked_shm(self.shm)
            self._meta_name_unlinked = True
        if self.dict is not None and not self._dict_name_unlinked:
            _unlink_untracked_shm(self.dict.shm)
            self._dict_name_unlinked = True

    def unlink(self) -> None:
        """Unlink and close shared memory resources, removing them from the system."""
        self.unlink_names()
        self.close()
        self.shm = None
        self.dict = None


class KeepAlive:
    """Background thread that periodically updates a participant heartbeat.

    Registers a participant in a KeepAliveDict and spawns a daemon thread
    that periodically updates its timestamp. A process can host multiple participants.
    """

    _local_registrations: ClassVar[dict[tuple[str, int], int]] = {}
    _local_registrations_lock: ClassVar = threading.Lock()

    def __init__(
        self,
        name: str,
        process_id: int,
        keep_alive_dict_size: int = 128 * 1024,
        keep_alive_seconds: int | float = 30,
        is_creator: bool = False,
        lock_class: type[BaseLock] = ProcessFileLock,
        dict_name: str | None = None,
        dict_meta_name: str | None = None,
        lock_name: str | None = None,
        unlink_on_init_failure: bool = False,
        participant_id: int | None = None,
    ) -> None:
        """Initialize keep-alive tracking for a participant in the current process.

        Args:
            name: Base name for shared resources.
            process_id: Identifier for this process (typically PID).
            keep_alive_dict_size: Size in bytes for the keep-alive dictionary.
            keep_alive_seconds: Timeout after which a participant heartbeat is considered stale.
            is_creator: If True, create shared resources; otherwise connect.
            lock_class: Lock class to use (default: ProcessFileLock).
            dict_name: Optional custom name for the dictionary.
            dict_meta_name: Optional custom name for the metadata buffer.
            lock_name: Optional custom name for the lock.
            unlink_on_init_failure: Remove auxiliary resources if initialization fails.
            participant_id: Optional unique identity for this liveness participant.
                Defaults to process_id for direct-caller compatibility.
        """
        if keep_alive_seconds <= 0 or not math.isfinite(keep_alive_seconds):
            raise ValueError("keep_alive_seconds must be finite and positive")

        self.name = name

        # Allow caller to specify exact names to avoid excessive nesting
        dict_name = dict_name if dict_name is not None else f"{name}_d"
        dict_meta_name = dict_meta_name if dict_meta_name is not None else f"{name}_m"
        lock_name = lock_name if lock_name is not None else f"{name}_s"

        self.process_id = process_id
        self.participant_id = process_id if participant_id is None else participant_id
        self.keep_alive_seconds = keep_alive_seconds
        self._registration_key = (dict_name, self.participant_id)
        self._registered: bool = False
        self.keep_alive_dict: KeepAliveDict | None = KeepAliveDict(
            name=dict_name, size=keep_alive_dict_size, is_creator=is_creator, meta_name=dict_meta_name
        )
        self.create_lock = is_creator
        self._lock_name_unlinked = False
        self.stop_event = threading.Event()  # Event to signal thread to stop
        self.lock_class = lock_class
        self.keep_alive_thread: threading.Thread | None = None

        try:
            # Create or connect to lock, then acquire it for initialization
            self.lock: BaseLock | None = self.lock_class(lock_name, locked=False)
            self.lock.acquire(non_blocking=False, timeout=LOCK_ACQUIRE_TIMEOUT)
        except BaseException as e:
            self._cleanup_init_failure(unlink=unlink_on_init_failure)
            if isinstance(e, (posix_ipc.BusyError, RuntimeError, BlockingIOError)):
                msg = f"KeepAlive lock '{lock_name}' initialization failed (is_creator={is_creator})"
                raise RuntimeError(msg) from e
            raise

        registration_created = self.participant_id not in self.keep_alive_dict
        try:
            self.slot = self.keep_alive_dict.prepare_slot(self.participant_id)
            self._retain_registration()
            self._start_keep_alive_thread()
            if not registration_created:
                self.keep_alive_dict.set_by_slot(self.slot, time.time())
        except BaseException:
            self.stop_event.set()
            keep_alive_thread = getattr(self, "keep_alive_thread", None)
            if keep_alive_thread is not None and keep_alive_thread.is_alive():
                keep_alive_thread.join(timeout=2)
            self.keep_alive_thread = None
            last_registration = self._release_registration()
            with contextlib.suppress(Exception):
                if registration_created and last_registration and self.participant_id in self.keep_alive_dict:
                    self.keep_alive_dict.pop(self.participant_id)
            self._cleanup_init_failure(unlink=unlink_on_init_failure)
            raise
        finally:
            self.lock.release()

    def _cleanup_init_failure(self, *, unlink: bool) -> None:
        with contextlib.suppress(Exception):
            if self.keep_alive_dict is not None:
                if unlink:
                    self.keep_alive_dict.unlink()
                else:
                    self.keep_alive_dict.close()

        lock = getattr(self, "lock", None)
        if lock is not None:
            if unlink:
                with contextlib.suppress(Exception):
                    lock.unlink()
            with contextlib.suppress(Exception):
                lock.close()

    def _retain_registration(self) -> None:
        with self._local_registrations_lock:
            count = self._local_registrations.get(self._registration_key, 0)
            self._local_registrations[self._registration_key] = count + 1
            self._registered = True

    def _release_registration(self) -> bool:
        if not self._registered:
            return False
        with self._local_registrations_lock:
            count = self._local_registrations.get(self._registration_key, 1)
            if count > 1:
                self._local_registrations[self._registration_key] = count - 1
                last_registration = False
            else:
                self._local_registrations.pop(self._registration_key, None)
                last_registration = True
            self._registered = False
            return last_registration

    def _resources(self) -> tuple[BaseLock, KeepAliveDict]:
        if self.lock is None or self.keep_alive_dict is None:
            raise RuntimeError("Keep-alive resources are closed")
        return self.lock, self.keep_alive_dict

    def stop(self, lock_timeout: float | None = None) -> None:
        """Stop the keep-alive thread and unregister this participant."""
        self.stop_event.set()  # Signal thread to stop
        if self.keep_alive_thread is not None:
            self.keep_alive_thread.join(timeout=2)  # Wait max 2 seconds for thread to finish
            self.keep_alive_thread = None
        lock, keep_alive_dict = self._resources()
        lock.acquire(non_blocking=False, timeout=lock_timeout)
        try:
            if self._release_registration() and self.participant_id in keep_alive_dict:
                keep_alive_dict.pop(self.participant_id)
        finally:
            lock.release()

    def unlink_names(self) -> None:
        """Remove keepalive data/meta/lock names without closing handles."""
        if self.keep_alive_dict is not None:
            self.keep_alive_dict.unlink_names()
        if self.lock is not None and not getattr(self, "_lock_name_unlinked", False):
            self.lock.unlink_name()
            self._lock_name_unlinked = True

    def close_local_handles(self) -> None:
        """Close process-local keepalive handles without unlinking names."""
        if self.keep_alive_thread is not None:
            self.stop()
        if self.keep_alive_dict is not None:
            self.keep_alive_dict.close()
        if self.lock is not None:
            self.lock.close()
        self.keep_alive_dict = None
        self.lock = None

    def close(self, any_process_alive: bool = True) -> None:
        """Close keep-alive resources.

        Args:
            any_process_alive: If False, unlink shared resources; otherwise just close.
        """
        if self._registered or self.keep_alive_thread is not None:
            self.stop()
        if not any_process_alive:
            self.unlink_names()
        self.close_local_handles()

    def any_process_alive(self, lock_timeout: float | None = None) -> bool:
        """Check if any registered participant heartbeat is live and remove stale entries."""
        current_time = time.time()
        lock, keep_alive_dict = self._resources()
        lock.acquire(non_blocking=False, timeout=lock_timeout)
        try:
            alive_items = list(keep_alive_dict.items())
            alive = False
            for participant_id, timestamp in alive_items:
                if timestamp < current_time - self.keep_alive_seconds:
                    keep_alive_dict.pop(participant_id)
                else:
                    alive = True
            return alive
        finally:
            lock.release()

    def is_process_alive(self, process_id: int, lock_timeout: float | None = None) -> bool:
        """Check one registered participant heartbeat, removing it if stale.

        ``process_id`` accepts the full participant ID. The method and keyword
        names are retained for Python API compatibility.
        """
        current_time = time.time()
        lock, keep_alive_dict = self._resources()
        lock.acquire(non_blocking=False, timeout=lock_timeout)
        try:
            for participant_id, timestamp in keep_alive_dict.items():
                if participant_id != process_id:
                    continue
                if timestamp < current_time - self.keep_alive_seconds:
                    keep_alive_dict.pop(participant_id)
                    return False
                return True
            return False
        finally:
            lock.release()

    def _start_keep_alive_thread(self) -> None:
        self.keep_alive_thread = threading.Thread(target=self._keep_alive_thread)
        self.keep_alive_thread.daemon = True
        self.keep_alive_thread.start()

    def _keep_alive_thread(self) -> None:
        while not self.stop_event.is_set():
            # Use event.wait() instead of time.sleep() so we can be interrupted quickly
            if self.stop_event.wait(timeout=self.keep_alive_seconds / 2):
                break  # Event was set, stop immediately
            keep_alive_dict = self.keep_alive_dict
            if keep_alive_dict is None:
                break
            keep_alive_dict.set_by_slot(self.slot, time.time())


class FlexibleSharedMemory:
    """Manages a shared memory segment with keep-alive tracking.

    Provides coordinated access to shared memory across multiple processes with
    automatic liveness detection and cleanup of resources when the last participant leaves.
    """

    def __init__(
        self,
        name: str,
        shm_size: int,
        keep_alive_dict_size: int = 128 * 1024,
        keep_alive_seconds: int | float = 30,
        pinned_memory: bool = False,
        lock_class: type[BaseLock] = ProcessFileLock,
        lock_acquire_timeout: float | None = LOCK_ACQUIRE_TIMEOUT,
        _creator_initializer: Callable[["FlexibleSharedMemory"], None] | None = None,
        _owner: "FlexibleSharedMemory | None" = None,
    ) -> None:
        """Initialize flexible shared memory.

        Args:
            name: Name for the shared memory segment.
            shm_size: Size in bytes. If > 0, creates or connects; if 0, connects only.
            keep_alive_dict_size: Size for the keep-alive dictionary.
            keep_alive_seconds: Timeout for participant-heartbeat liveness detection.
            pinned_memory: If True, register memory with CUDA for faster transfers.
            lock_class: Lock class to use (default: ProcessFileLock).
            lock_acquire_timeout: Maximum seconds to wait for the main lock, or None to wait indefinitely.
            _creator_initializer: Internal callback invoked only by the process
                that creates the main SHM block, while the setup lock is held.
            _owner: Internal coordination handle that owns this block's lifetime.
                Owned blocks close locally; the last coordinator unlinks their names.

        """
        if sys.platform == "win32":
            raise NotImplementedError("Shared memory (SHM) is not supported on Windows")
        self.name = name
        self.shm_size = shm_size
        self.pid = os.getpid()
        self.participant_id = _next_participant_id(self.pid)
        self.pinned_memory = pinned_memory
        self.shm: SharedMemory | None = None
        self._host_pin_registry: HostPinRegistry | None = None
        self._pinned_shm_view: torch.Tensor | None = None
        self.main_lock: BaseLock | None = None
        self.keep_alive: KeepAlive | None = None
        self.shm_creator = False
        self.lock_class = lock_class
        self._closed = False
        self._teardown_anchor_held = False
        self._teardown_shm_unlinked = False
        self._teardown_main_lock_released = False
        self._owner = _owner
        self._rollback_pending = False
        self._owned_names: SharedMultiString | None = None
        self._children: set[FlexibleSharedMemory] = set()
        self._child_setup_condition = threading.Condition()
        self._child_setups = 0
        self._closing = False
        self._owns_child_teardown = False

        if shm_size < 0:
            msg = "shm_size cannot be negative"
            raise ValueError(msg)

        if _owner is None:
            self._initialize(keep_alive_dict_size, keep_alive_seconds, lock_acquire_timeout, _creator_initializer)
        else:
            with _owner._child_setup(self):  # noqa: SLF001
                self._initialize(keep_alive_dict_size, keep_alive_seconds, lock_acquire_timeout, _creator_initializer)

    def _initialize(
        self,
        keep_alive_dict_size: int,
        keep_alive_seconds: int | float,
        lock_acquire_timeout: float | None,
        creator_initializer: Callable[["FlexibleSharedMemory"], None] | None,
    ) -> None:
        """Open resources briefly under the owner lock; publish under the child lock."""
        create_auxiliary_resources = self.shm_size > 0
        try:
            if self._owner is None:
                self.main_lock = self.lock_class(self.name, locked=False)
            else:
                self._owner._open_child_lock(self)  # noqa: SLF001
            if self.main_lock is None:
                raise RuntimeError(f"Shared-memory block '{self.name}' has no setup lock")
            self.main_lock.acquire(non_blocking=False, timeout=lock_acquire_timeout)
        except BaseException as e:
            self._cleanup_on_init_failure()
            with contextlib.suppress(Exception):
                self._release_and_close_main_lock_after_init_failure()
            if isinstance(e, (posix_ipc.BusyError, BlockingIOError)):
                msg = (
                    f"Failed to acquire main lock for shared memory '{self.name}' within {lock_acquire_timeout} seconds"
                )
                raise BlockingIOError(msg) from e
            raise

        try:
            if self._owner is not None:
                self._owner._open_child_memory(self)  # noqa: SLF001
            else:
                self.shm_creator = self._create_memory_block()

            # Only the coordinator (or a standalone block) needs a heartbeat.
            if self._owner is None:
                self.keep_alive = KeepAlive(
                    name=self.name + "_k",  # Keep-alive base name
                    process_id=self.pid,
                    participant_id=self.participant_id,
                    keep_alive_dict_size=keep_alive_dict_size,
                    keep_alive_seconds=keep_alive_seconds,
                    is_creator=create_auxiliary_resources,
                    lock_class=self.lock_class,
                    # Compact suffixes fit macOS's 31-character POSIX name limit.
                    dict_name=self.name + "_kd",
                    dict_meta_name=self.name + "_km",
                    lock_name=self.name + "_ks",
                    unlink_on_init_failure=self.shm_creator,
                )

            if self.shm_creator and creator_initializer is not None:
                creator_initializer(self)

            # Register while construction is still exclusive. A creator that
            # cannot pin its mapping rolls back before any follower can attach.
            self._register_pinned_memory()
        except BaseException:
            # Setup failed while the main lock is still held. Roll back shared
            # resources, then release and close the constructor-only lock handle.
            self._cleanup_on_init_failure()
            self._release_and_close_main_lock_after_init_failure()
            raise

        try:
            self.main_lock.release()
        except BaseException:
            with contextlib.suppress(Exception):
                self._unregister_pinned_memory()
            self._cleanup_on_init_failure()
            self._release_and_close_main_lock_after_init_failure()
            raise

    def _release_and_close_main_lock_after_init_failure(self) -> None:
        """Best-effort release and close when setup fails with the lock held."""
        if self._owner is not None:
            # Owned failures use close(), retaining failed handles in the
            # owner's child set rather than dropping them here.
            return
        if self.main_lock is not None:
            with contextlib.suppress(Exception):
                self.main_lock.release()
            with contextlib.suppress(Exception):
                self.main_lock.close()
            self.main_lock = None

    def _enable_child_ownership(self, *, create: bool) -> None:
        """Open the coordinator's name registry after its protocol version is validated.

        Creation runs inside the coordinator initializer, under its setup lock.
        Followers already hold a heartbeat registration when they attach.
        """
        names = SharedMultiString(name=self.name + "_b", create=create)
        if create and not names._created:  # noqa: SLF001
            names.close()
            raise FileExistsError(f"Shared-memory ownership registry for '{self.name}' already exists")
        self._owned_names = names

    # Match ProfilingSuspender.suspended: Any avoids beartype's cross-version
    # rejection of contextmanager wrappers annotated as Iterator/Generator.
    @contextlib.contextmanager
    def _child_setup(self, child: "FlexibleSharedMemory") -> Any:
        """Keep this participant alive until its in-flight child setup finishes."""
        with self._child_setup_condition:
            if self._closing or self._closed:
                raise RuntimeError(f"Shared-memory owner '{self.name}' is closing")
            self._child_setups += 1
            self._children.add(child)
        try:
            yield
        finally:
            with self._child_setup_condition:
                self._child_setups -= 1
                self._child_setup_condition.notify_all()

    @contextlib.contextmanager
    def _owner_setup_lock(self) -> Any:
        """Fence resource opens against the last owner's teardown of this generation."""
        if self.main_lock is None or self._owned_names is None:
            raise RuntimeError(f"Shared-memory owner '{self.name}' is not initialized")
        if self._closed or self._teardown_anchor_held or self._teardown_main_lock_released:
            raise RuntimeError(f"Shared-memory owner '{self.name}' is closing")
        self.main_lock.acquire(non_blocking=False, timeout=LOCK_ACQUIRE_TIMEOUT)
        try:
            # The shared flag survives in old mappings after unlink, fencing
            # a paused participant even after a new generation uses the name.
            if self._owned_names.is_ready():
                raise RuntimeError(f"Shared-memory owner '{self.name}' is closing")
            yield self._owned_names
        finally:
            self.main_lock.release()

    @staticmethod
    def _resource_record(*fields: str) -> str:
        """Encode a resource without the shared string registry's delimiter."""
        return json.dumps(fields).replace("|", "\\u007c")

    def _open_child_lock(self, child: "FlexibleSharedMemory") -> None:
        """Journal and open a generation-specific lock without waiting under the owner lock."""
        if not child.name or "|" in child.name:
            raise ValueError(f"Invalid owned shared-memory name: {child.name!r}")
        if issubclass(child.lock_class, ProcessFileLock):
            kind = "file"
        elif issubclass(child.lock_class, SemaphoreLock):
            kind = "semaphore"
        else:
            raise TypeError("Owned shared memory requires ProcessFileLock or SemaphoreLock")
        with self._owner_setup_lock() as names:
            resources = [json.loads(record) for record in names.get_list()]
            existing = next((r for r in resources if r[0] != "shm" and r[1] == child.name), None)
            if existing is None:
                if not self.shm_creator or child.shm_size == 0:
                    raise FileNotFoundError(f"Shared-memory block '{child.name}' has not been published by its owner")
                lock_name = "ftl_" + secrets.token_hex(12)
                resource_name = ProcessFileLock.resolve_path(lock_name) if kind == "file" else "/" + lock_name
                names.append(self._resource_record(kind, child.name, resource_name))
            else:
                if existing[0] != kind:
                    raise ValueError(f"Shared-memory block '{child.name}' uses a different lock backend")
                resource_name = existing[2]
            child.main_lock = child.lock_class(resource_name, locked=False)

    def _open_child_memory(self, child: "FlexibleSharedMemory") -> None:
        """Open the mapping under the owner gate, after acquiring the child's lock."""
        with self._owner_setup_lock() as names:
            record = self._resource_record("shm", child.name)
            if record not in names.get_list():
                if not self.shm_creator or child.shm_size == 0:
                    raise FileNotFoundError(f"Shared-memory block '{child.name}' has not been published by its owner")
                # The child's setup lock is also held: never adopt a preexisting
                # block that belongs to a different lifetime.
                try:
                    existing = posix_ipc.SharedMemory("/" + child.name.lstrip("/"))
                except posix_ipc.ExistentialError:
                    pass
                else:
                    existing.close_fd()
                    raise FileExistsError(f"Shared-memory block '{child.name}' already exists")
                names.append(record)
            if not self.shm_creator:
                child.shm_size = 0  # Followers may only attach published blocks.
            child.shm_creator = child._create_memory_block()  # noqa: SLF001

    def _cleanup_on_init_failure(self) -> None:
        """Best-effort cleanup when __init__ fails after partial resource creation.

        Only unlinks resources if we created them (shm_creator=True), otherwise just closes.
        This prevents destroying shared memory segments that other processes are using.
        """
        # Use getattr for safe access since exception may occur before attributes are assigned
        if self._owner is not None:
            # Roll back only a newly created payload, keeping its journaled
            # lock for retries. Failed rollback/close stays in the owner's ledger.
            self._rollback_pending = self.shm_creator
            with contextlib.suppress(Exception):
                self.close()
            return

        if self._owned_names is not None:
            with contextlib.suppress(Exception):
                if self.shm_creator:
                    self._owned_names.unlink_name()
                self._owned_names.close()
            self._owned_names = None
        keep_alive = getattr(self, "keep_alive", None)
        shm = getattr(self, "shm", None)
        shm_creator = getattr(self, "shm_creator", False)

        # Clean up keep_alive if it was created
        if keep_alive is not None:
            with contextlib.suppress(Exception):
                keep_alive.stop()
            with contextlib.suppress(Exception):
                # Only unlink if no other process is alive (which shouldn't happen during init failure,
                # but be safe)
                keep_alive.close(any_process_alive=not shm_creator)
            self.keep_alive = None

        # Clean up shared memory - only unlink if WE created it
        if shm is not None:
            with contextlib.suppress(Exception):
                if shm_creator:
                    # We created this segment and failed before any other process could register,
                    # so it's safe to unlink
                    resource_tracker_register(shm._name, "shared_memory")  # noqa: SLF001
                    shm.unlink()
                shm.close()
            self.shm = None

    def _register_pinned_memory(self) -> None:
        """Register the SHM buffer with CUDA after constructor setup succeeds."""
        if not self.pinned_memory:
            return
        if self.shm is None:
            raise RuntimeError("Cannot register pinned shared memory before creating its SHM block")

        registry = HostPinRegistry()
        pinned_view = torch.frombuffer(self.shm.buf, dtype=torch.uint8)
        try:
            registry.pin_in_place(pinned_view)
        except Exception as exc:
            # Strip the traceback so constructor rollback can close the SHM
            # mapping without this frame retaining its exported buffer.
            raise exc.with_traceback(None) from None
        self._host_pin_registry = registry
        self._pinned_shm_view = pinned_view

    def _unregister_pinned_memory(self) -> None:
        """Unregister pinned memory if it was used."""
        registry = self._host_pin_registry
        if registry is None:
            self._pinned_shm_view = None
            return

        registry.release_all()
        remaining = len(registry)
        if remaining:
            raise RuntimeError(
                f"FlexibleSharedMemory '{self.name}' failed to unregister {remaining} pinned shared-memory "
                "region(s); the SHM mapping remains open for a later close retry"
            )

        # HostPinRegistry retains the tensor until every unregister succeeds.
        # Drop our references only afterwards so SharedMemory.close() sees no
        # exported torch view.
        self._host_pin_registry = None
        self._pinned_shm_view = None

    def _unlink_main_shm_name(self) -> None:
        """Remove only the main generation's SHM name, retaining its mapping."""
        if not self._teardown_shm_unlinked:
            if self.shm is not None:
                _unlink_untracked_shm(self.shm)
            self._teardown_shm_unlinked = True

    def _rollback_owned_creation(self) -> None:
        """Unlink failed publication under the owner gate while retaining the child lock."""
        owner = self._owner
        if owner is None or owner.main_lock is None or owner._owned_names is None:  # noqa: SLF001
            raise RuntimeError(f"Cannot roll back shared-memory block '{self.name}' without its owner")
        owner.main_lock.acquire(non_blocking=False, timeout=LOCK_ACQUIRE_TIMEOUT)
        try:
            # A retired owner's mapping may outlive its names. Never unlink
            # a replacement generation that now uses this payload's name.
            if not owner._owned_names.is_ready():  # noqa: SLF001
                self._unlink_main_shm_name()
            self._rollback_pending = False
        finally:
            owner.main_lock.release()

    def _close_local_resources(self) -> None:
        """Close local handles after anchor release, without unlinking any names."""
        if self._owned_names is not None:
            self._owned_names.close()
        if self.keep_alive is not None:
            self.keep_alive.close_local_handles()
        if self.shm is not None:
            self.shm.close()

    @staticmethod
    def _unlink_owned_resources(names: SharedMultiString) -> None:
        """Unlink journaled resources under the owner lock, retaining the journal on failure."""
        resources = [json.loads(record) for record in names.get_list()]
        for kind, name, *_ in resources:
            if kind == "shm":
                with contextlib.suppress(posix_ipc.ExistentialError):
                    posix_ipc.unlink_shared_memory("/" + name.lstrip("/"))
        for kind, _name, *resource in resources:
            if kind == "file":
                Path(resource[0]).unlink(missing_ok=True)
            elif kind == "semaphore":
                with contextlib.suppress(posix_ipc.ExistentialError):
                    posix_ipc.unlink_semaphore(resource[0])
        names.unlink_name()

    def _prepare_coordinated_teardown(self) -> None:
        """Decide ownership and unlink global names without releasing the anchor."""
        if self.keep_alive is not None:
            self.keep_alive.stop(lock_timeout=LOCK_ACQUIRE_TIMEOUT)
        if self._owned_names is not None and self._owned_names.is_ready() and not self._owns_child_teardown:
            # A different participant retired this generation while we were
            # stale. Never unlink resources belonging to its replacement.
            return
        if self.keep_alive is None or not self.keep_alive.any_process_alive(lock_timeout=LOCK_ACQUIRE_TIMEOUT):
            if self._owned_names is not None:
                self._owned_names.set_ready()  # This generation admits no more opens.
                self._owns_child_teardown = True
                self._unlink_owned_resources(self._owned_names)
            self._unlink_main_shm_name()
            if self.keep_alive is not None:
                self.keep_alive.unlink_names()

    def _acquire_teardown_coordination(self) -> None:
        """Acquire the main-lock anchor, retaining all state on failure."""
        if self._teardown_anchor_held:
            return
        lock = self.main_lock
        if lock is None:
            raise RuntimeError(
                f"FlexibleSharedMemory '{self.name}' cannot coordinate teardown because its main lock is missing; "
                "SHM handles remain available for a later close retry"
            )
        try:
            lock.acquire(non_blocking=False, timeout=LOCK_ACQUIRE_TIMEOUT)
        except (posix_ipc.BusyError, BlockingIOError) as exc:
            raise RuntimeError(
                f"FlexibleSharedMemory '{self.name}' could not acquire its main lock for teardown within "
                f"{LOCK_ACQUIRE_TIMEOUT} seconds; SHM handles remain available for a later close retry"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"FlexibleSharedMemory '{self.name}' failed to acquire its main lock for teardown; "
                "SHM handles remain available for a later close retry"
            ) from exc
        self._teardown_anchor_held = True

    def _release_after_successful_teardown(self) -> None:
        """Release coordination after shared-resource teardown."""
        if self.main_lock is None:
            raise RuntimeError(
                f"FlexibleSharedMemory '{self.name}' lost its main lock during teardown; "
                "SHM handles remain available for a later close retry"
            )
        self.main_lock.release()
        self._teardown_anchor_held = False
        self._teardown_main_lock_released = True

    def close(self) -> None:
        """Close and clean up all resources.

        A normal return guarantees that this handle no longer owns any SHM
        resources.  If teardown coordination fails, the error is raised and
        the coordination handles are retained so a later call can retry.
        """
        if self._closed:
            return

        with self._child_setup_condition:
            self._closing = True
            if not self._child_setup_condition.wait_for(lambda: self._child_setups == 0, timeout=LOCK_ACQUIRE_TIMEOUT):
                raise RuntimeError(f"Shared-memory owner '{self.name}' still has child setup in progress; retry close")
            children = tuple(self._children)
        for child in children:
            child.close()

        if self._rollback_pending:
            self._rollback_owned_creation()

        # Keep ourselves marked alive until we're done with shared resources.
        self._unregister_pinned_memory()

        if self._owner is None and not self._teardown_main_lock_released:
            self._acquire_teardown_coordination()
            self._prepare_coordinated_teardown()
            self._release_after_successful_teardown()

        self._close_local_resources()

        if self.main_lock is not None:
            self.main_lock.close()

        # Publish closed state only after every teardown stage has succeeded.
        self.keep_alive = None
        self.main_lock = None
        self.shm = None
        self._closed = True
        if self._owner is not None:
            with self._owner._child_setup_condition:  # noqa: SLF001
                self._owner._children.discard(self)  # noqa: SLF001
        self._owned_names = None

    def _create_memory_block(self) -> bool:
        """Create or connect to the shared memory block.

        Returns:
            True if this call created the memory, False if connected to existing.
        """
        # If shm_size is 0, we're connecting to existing memory, not creating
        # If shm_size > 0, we're creating (or trying to)
        is_created = False
        if self.shm_size > 0:
            # Try to create first
            try:
                self.shm = SharedMemory(name=self.name, size=self.shm_size, create=True)
                # We created it - update shm_creator to True
                is_created = True
            except FileExistsError:
                # Already exists, connect to it
                self.shm = SharedMemory(name=self.name, create=False)
                # We didn't create it
                is_created = False
        else:
            # Connecting to existing memory (shm_size == 0)
            self.shm = SharedMemory(name=self.name, create=False)
            is_created = False
        # Unregister from resource tracker to prevent spurious cleanup warnings in multiprocess scenarios
        # Use _name (with / prefix) to match what resource_tracker registered
        resource_tracker_unregister(self.shm._name, "shared_memory")  # type: ignore[attr-defined]  # noqa: SLF001
        self.shm_size = self.shm.size

        return is_created

    @property
    def block(self) -> SharedMemory | None:
        """The underlying SharedMemory object."""
        return self.shm

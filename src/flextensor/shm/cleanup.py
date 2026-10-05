# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit retirement of abandoned shared-offload generations."""

from __future__ import annotations

import contextlib
import math
import mmap
import time
from pathlib import Path
from typing import Any

import posix_ipc

from flextensor.shm.coord_block import CoordBlockHeader
from flextensor.shm.coordinator import DEFAULT_KEEP_ALIVE_SECONDS
from flextensor.shm.flexible_shm import FlexibleSharedMemory, KeepAliveDict
from flextensor.shm.multiprocess_condition import SharedMultiString
from flextensor.shm.namespace import SHM_PROTOCOL_VERSION, coord_block_name
from flextensor.shm.sync_primitives import ProcessFileLock


@contextlib.contextmanager
def _existing_mapping(name: str) -> Any:
    """Open metadata without creating SHM or registering with a resource tracker."""
    shm = posix_ipc.SharedMemory("/" + name.lstrip("/"))
    try:
        with mmap.mmap(shm.fd, shm.size) as mapping:
            yield mapping
    finally:
        shm.close_fd()


def _has_live_participants(name: str, deadline: float) -> bool:
    # Both mappings must exist before KeepAliveDict opens them: its dictionary
    # constructor otherwise creates missing storage. The coordinator lock keeps
    # setup/teardown excluded throughout these opens. No participant is added.
    with (
        _existing_mapping(f"sm_{name}_kd"),
        _existing_mapping(f"{name}_km") as timestamps,
        contextlib.closing(ProcessFileLock(f"{name}_ks", locked=False)) as lock,
    ):
        lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
        try:
            with contextlib.closing(
                KeepAliveDict(f"{name}_kd", size=len(timestamps), is_creator=False, meta_name=f"{name}_km")
            ) as participants:
                cutoff = time.time() - DEFAULT_KEEP_ALIVE_SECONDS
                values = list(participants.items())
                if any(not math.isfinite(timestamp) for _, timestamp in values):
                    raise RuntimeError(f"Invalid participant heartbeat in {name}")
                # PIDs from another container cannot be interpreted locally.
                return any(timestamp >= cutoff for _, timestamp in values)
        finally:
            lock.release()


def _unlink_metadata(name: str) -> None:
    for resource in (f"sm_{name}_kd", f"{name}_km"):
        with contextlib.suppress(posix_ipc.ExistentialError):
            posix_ipc.unlink_shared_memory("/" + resource)
    Path(ProcessFileLock.resolve_path(f"{name}_ks")).unlink(missing_ok=True)


def _try_cleanup(name: str, deadline: float) -> bool:
    with contextlib.ExitStack() as handles:
        try:
            mapping = handles.enter_context(_existing_mapping(name))
        except posix_ipc.ExistentialError:
            # Do not declare success if an interrupted older teardown left
            # metadata whose protocol/ownership can no longer be validated.
            for resource in (f"{name}_b", f"sm_{name}_kd", f"{name}_km"):
                try:
                    with _existing_mapping(resource):
                        raise RuntimeError(f"SHM metadata {resource} remains without coordinator {name}")
                except posix_ipc.ExistentialError:
                    pass
            return True

        header = CoordBlockHeader.read_from(mapping[:])
        header.validate(expected_version=header.flextensor_version, expected_protocol=SHM_PROTOCOL_VERSION)
        try:
            journal = handles.enter_context(contextlib.closing(SharedMultiString(name=f"{name}_b")))
        except FileNotFoundError:
            # Resume retirement after the journal and its children were
            # removed, before the coordinator anchor.
            journal = None

        if journal is not None and not journal.is_ready() and _has_live_participants(name, deadline):
            return False
        if journal is not None:
            FlexibleSharedMemory._unlink_owned_resources(journal)  # noqa: SLF001 - shared retirement primitive.
        _unlink_metadata(name)
        # Keep the coordinator until last so an interrupted cleanup is retryable.
        posix_ipc.unlink_shared_memory("/" + name)
        return True


def cleanup_manager_namespace(namespace: str, *, deadline: float) -> None:
    """Retire a known-protocol generation after its heartbeat leases expire.

    Call only after stopping its workers and preventing new launches with the
    same namespace. The caller must share the workers' IPC and lock directory.
    The stable setup-lock file intentionally survives, as in normal teardown.
    """
    name = coord_block_name(namespace)
    with contextlib.closing(ProcessFileLock(name, locked=False)) as lock:
        while True:
            try:
                lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
                try:
                    if _try_cleanup(name, deadline):
                        return
                finally:
                    lock.release()
            except BlockingIOError:
                if time.monotonic() < deadline:
                    continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"SHM cleanup timed out for {namespace}: participants are live or metadata is locked"
                )
            time.sleep(min(0.1, remaining))

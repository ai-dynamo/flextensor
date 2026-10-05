# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-free recovery of abandoned shared-offload generations."""

from __future__ import annotations

import contextlib
import multiprocessing
import signal
import struct
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import posix_ipc
import pytest

from flextensor.shared_offload import SharedOffloadSession
from flextensor.shm.coord_block import CoordBlockHeader
from flextensor.shm.coordinator import ShmCoordinator
from flextensor.shm.flexible_shm import FlexibleSharedMemory
from flextensor.shm.sync_primitives import ProcessFileLock


@pytest.fixture
def namespace():
    return "cleanup_" + uuid.uuid4().hex


def _manager_namespace(namespace):
    return SharedOffloadSession(namespace).namespace_for("m")


def _cleanup(namespace, timeout=0):
    SharedOffloadSession.cleanup_namespace(namespace, manager_names=["m"], timeout=timeout)


def _assert_absent(namespace):
    name = _manager_namespace(namespace)
    for resource in (name + "_data", name + "_crd", name + "_crd_b", name + "_crd_km", "sm_" + name + "_crd_kd"):
        with pytest.raises(posix_ipc.ExistentialError):
            shm = posix_ipc.SharedMemory("/" + resource)
            shm.close_fd()
    assert not Path(ProcessFileLock.resolve_path(name + "_crd_ks")).exists()


@pytest.mark.parametrize("old_state", ["initializing", "ready", "failed", "retiring", "journal_removed", "old_version"])
def test_cleanup_fences_old_handles_and_allows_fresh_generation(namespace, old_state):
    # Retained mappings must never unlink replacement resources when their
    # delayed close finally runs.
    name = _manager_namespace(namespace)
    old = ShmCoordinator(name)
    replacement = None
    FlexibleSharedMemory(name=name + "_data", shm_size=8, _owner=old.memory_owner)
    try:
        if old_state == "ready":
            old.notify_ready()
        elif old_state == "failed":
            old.mark_failed(RuntimeError("abandoned initialization"))
        elif old_state in {"retiring", "journal_removed"}:
            journal = old.memory_owner._owned_names
            journal.set_ready()
            if old_state == "journal_removed":
                old.memory_owner._unlink_owned_resources(journal)
        elif old_state == "old_version":
            header = CoordBlockHeader.read_from(old.memory_owner.block.buf)
            header.flextensor_version = "0.0.0+previous"
            header.write_to(old.memory_owner.block.buf)
        old.memory_owner.keep_alive.stop()
        _cleanup(namespace)
        _assert_absent(namespace)
        _cleanup(namespace)
        replacement = ShmCoordinator(name)
        assert replacement.is_creator
        child = FlexibleSharedMemory(name=name + "_data", shm_size=8, _owner=replacement.memory_owner)
        child.block.buf[:3] = b"new"
        old.close()
        attached = FlexibleSharedMemory(name=child.name, shm_size=0, _owner=replacement.memory_owner)
        assert bytes(attached.block.buf[:3]) == b"new"
        assert replacement.is_creator_alive()
    finally:
        old.close()
        if replacement is not None:
            replacement.close()


def test_live_namespace_and_unrelated_generation_survive(namespace):
    with contextlib.ExitStack() as stack:
        live = ShmCoordinator(_manager_namespace(namespace))
        stack.callback(live.close)
        other = ShmCoordinator(_manager_namespace(namespace + "_other"))
        stack.callback(other.close)
        before = dict(live.memory_owner.keep_alive.keep_alive_dict.items())
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="participants are live"):
            _cleanup(namespace, timeout=0.1)
        assert time.monotonic() - started < 2
        assert dict(live.memory_owner.keep_alive.keep_alive_dict.items()).keys() == before.keys()
        assert live.is_creator_alive()
        assert other.is_creator_alive()
        live.memory_owner.keep_alive.stop()
        _cleanup(namespace)
        _assert_absent(namespace)
        assert other.is_creator_alive()


@pytest.mark.parametrize("corruption", ["protocol", "header", "journal", "journal_length", "journal_ready"])
def test_cleanup_rejects_corrupt_metadata_without_unlinking_payload(namespace, corruption):
    name = _manager_namespace(namespace)
    owner = ShmCoordinator(name)
    child = FlexibleSharedMemory(name=name + "_data", shm_size=8, _owner=owner.memory_owner)
    owner.memory_owner.keep_alive.stop()
    header_buf = owner.memory_owner.block.buf
    journal = owner.memory_owner._owned_names
    original_header = bytes(header_buf)
    original_journal = bytes(journal.shm.buf)
    try:
        if corruption == "protocol":
            header = CoordBlockHeader.read_from(header_buf)
            header.protocol_version += 1
            header.write_to(header_buf)
        elif corruption == "header":
            header_buf[:4] = b"\xff" * 4
        elif corruption == "journal":
            journal.append('["unknown", "resource"]')
        elif corruption == "journal_length":
            journal.shm.buf[:4] = struct.pack("I", journal.shm.size + 1)
        else:
            journal.shm.buf[4] = 2
        before = bytes(journal.shm.buf)
        with pytest.raises(RuntimeError):
            _cleanup(namespace)
        assert bytes(journal.shm.buf) == before
        shm = posix_ipc.SharedMemory("/" + child.name)
        shm.close_fd()
    finally:
        header_buf[:] = original_header
        journal.shm.buf[:] = original_journal
        owner.close()


def test_cleanup_retry_after_interrupted_unlink(namespace, monkeypatch):
    name = _manager_namespace(namespace)
    owner = ShmCoordinator(name)
    child = FlexibleSharedMemory(name=name + "_data", shm_size=8, _owner=owner.memory_owner)
    owner.memory_owner.keep_alive.stop()
    unlink = posix_ipc.unlink_shared_memory

    def fail_payload(resource):
        if resource == "/" + child.name:
            raise OSError("injected unlink failure")
        unlink(resource)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(posix_ipc, "unlink_shared_memory", fail_payload)
            with pytest.raises(OSError, match="injected unlink failure"):
                _cleanup(namespace)
        _cleanup(namespace)
        _assert_absent(namespace)
    finally:
        owner.close()


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": -1},
        {"timeout": float("nan")},
        {"manager_names": []},
        {"manager_names": "m"},
        {"manager_names": ["m", "bad/name"]},
    ],
)
def test_cleanup_validates_all_options_before_removing_resources(namespace, options):
    owner = ShmCoordinator(_manager_namespace(namespace))
    owner.memory_owner.keep_alive.stop()
    try:
        with pytest.raises(ValueError):
            SharedOffloadSession.cleanup_namespace(namespace, **({"manager_names": ["m"]} | options))
        assert not owner.memory_owner._owned_names.is_ready()
    finally:
        owner.close()


def test_missing_namespace_is_noop_without_constructing_coordinator(namespace, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("cleanup must not initialize a coordinator")

    monkeypatch.setattr(ShmCoordinator, "__init__", unexpected)
    _cleanup(namespace)
    _cleanup(namespace)
    _assert_absent(namespace)


def test_cleanup_timeout_bounds_waiting_for_coordinator_lock(namespace):
    owner = ShmCoordinator(_manager_namespace(namespace))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with owner.memory_owner.main_lock:
                future = executor.submit(_cleanup, namespace, 0.1)
                with pytest.raises(TimeoutError, match="metadata is locked"):
                    future.result(timeout=2)
            assert owner.is_creator_alive()
    finally:
        owner.close()


def _abandoned_worker(namespace, ready):
    owner = ShmCoordinator(_manager_namespace(namespace))
    if owner.is_creator:
        FlexibleSharedMemory(name=_manager_namespace(namespace) + "_data", shm_size=8, _owner=owner.memory_owner)
        owner.notify_ready()
    ready.set()
    while True:
        signal.pause()


def test_cleanup_after_sigkill_of_creator_and_follower_waits_for_leases(namespace):
    context = multiprocessing.get_context("spawn")
    workers = []
    try:
        for _ in range(2):
            ready = context.Event()
            worker = context.Process(target=_abandoned_worker, args=(namespace, ready))
            workers.append(worker)
            worker.start()
            assert ready.wait(20), worker.exitcode
        for worker in workers:
            worker.kill()
            worker.join(timeout=10)
            assert worker.exitcode == -signal.SIGKILL
        # A foreign PID namespace could reuse these PIDs. Cleanup must honor
        # their real heartbeat leases, not decide they are dead via kill(pid, 0).
        with pytest.raises(TimeoutError):
            _cleanup(namespace)
        _cleanup(namespace, timeout=40)
        _assert_absent(namespace)
        _cleanup(namespace)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=10)
        _cleanup(namespace, timeout=40)

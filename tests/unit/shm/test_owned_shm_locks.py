# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned lock cleanup, parallel publication, and generation fencing."""

import multiprocessing
import threading
import time
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import posix_ipc
import pytest
import torch
from torch import nn

from flextensor.shm import FlexibleSharedMemory, ProcessFileLock, SemaphoreLock, ShmCoordinator
from flextensor.shm.tensor_store import SharedTensorStore


@pytest.fixture
def namespace():
    name = "ftol_" + uuid.uuid4().hex[:8]
    yield name
    # The coordinator's stable bootstrap lock is outside child ownership.
    Path(ProcessFileLock.resolve_path(name + "_crd")).unlink(missing_ok=True)


@pytest.fixture(params=[ProcessFileLock, SemaphoreLock])
def lock_class(request):
    return request.param


def _lock_resource(child):
    lock = child.main_lock
    if isinstance(lock, ProcessFileLock):
        return "file", lock.path
    return "semaphore", lock.name


def _assert_lock_exists(resource, expected):
    kind, name = resource
    if kind == "file":
        assert Path(name).exists() is expected
    elif expected:
        semaphore = posix_ipc.Semaphore(name)
        semaphore.close()
    else:
        with pytest.raises(posix_ipc.ExistentialError):
            semaphore = posix_ipc.Semaphore(name)
            semaphore.close()


def _assert_shm_absent(name):
    with pytest.raises(posix_ipc.ExistentialError):
        shm = posix_ipc.SharedMemory("/" + name)
        shm.close_fd()


def test_last_follower_removes_unopened_child_locks(namespace):
    """The registry preserves each backend even when the cleaner never opened a child."""
    creator = ShmCoordinator(namespace)
    follower = ShmCoordinator(namespace)
    children = []
    try:
        for index, backend in enumerate((ProcessFileLock, SemaphoreLock)):
            child = FlexibleSharedMemory(namespace + f"_d{index}", 8, _owner=creator.memory_owner, lock_class=backend)
            children.append((child.name, _lock_resource(child)))
        creator.close()
        for name, resource in children:
            _assert_lock_exists(resource, True)
            shm = posix_ipc.SharedMemory("/" + name)
            shm.close_fd()
        follower.close()
        for name, resource in children:
            _assert_lock_exists(resource, False)
            _assert_shm_absent(name)
    finally:
        creator.close()
        follower.close()


def test_failed_child_publication_rolls_back_before_owner_close(namespace, lock_class):
    creator = ShmCoordinator(namespace)
    resources = []

    def fail(child):
        resources.append(_lock_resource(child))
        raise ValueError("publication failed")

    try:
        with pytest.raises(ValueError, match="publication failed"):
            FlexibleSharedMemory(
                namespace + "_d", 8, _owner=creator.memory_owner, lock_class=lock_class, _creator_initializer=fail
            )
        _assert_shm_absent(namespace + "_d")
        _assert_lock_exists(resources[0], True)
        retry = FlexibleSharedMemory(namespace + "_d", 16, _owner=creator.memory_owner, lock_class=lock_class)
        assert retry.shm_creator
        assert _lock_resource(retry) == resources[0]
    finally:
        creator.close()
    _assert_lock_exists(resources[0], False)
    _assert_shm_absent(namespace + "_d")


def test_failed_owned_tensor_copy_rejects_attachment_and_allows_retry(namespace, monkeypatch):
    """Real owned SHM must not expose a valid manifest with an incomplete payload."""
    source = nn.Module()
    source.first = nn.Parameter(torch.tensor([3.0, 4.0]))
    source.second = nn.Parameter(torch.tensor([7.0, 8.0]))
    original_copy = torch.Tensor.copy_
    copies = 0

    def fail_second_copy(destination, source, *args, **kwargs):
        nonlocal copies
        copies += 1
        if copies == 2:
            # Do not retain a SHM view in the injected exception's traceback.
            del destination
            raise RuntimeError("second tensor copy failed")
        return original_copy(destination, source, *args, **kwargs)

    with ExitStack() as cleanup:
        creator = ShmCoordinator(namespace)
        cleanup.callback(creator.close)
        follower = ShmCoordinator(namespace)
        cleanup.callback(follower.close)
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "copy_", fail_second_copy)
            with pytest.raises(RuntimeError, match="second tensor copy failed"):
                SharedTensorStore.create_from_model(
                    namespace, source, parameter_names=["first", "second"], shm_owner=creator.memory_owner
                )
        with pytest.raises(FileNotFoundError):
            unexpected = SharedTensorStore.attach(namespace, shm_owner=follower.memory_owner)
            cleanup.callback(unexpected.close)
        store = SharedTensorStore.create_from_model(
            namespace, source, parameter_names=["first", "second"], shm_owner=creator.memory_owner
        )
        cleanup.callback(store.close)
        attached = SharedTensorStore.attach(namespace, shm_owner=follower.memory_owner)
        cleanup.callback(attached.close)
        target = nn.Module()
        target.first = nn.Parameter(torch.empty(2, device="meta"))
        target.second = nn.Parameter(torch.empty(2, device="meta"))
        attached.materialize_into(target, device="cpu")
        torch.testing.assert_close(target.first, source.first)
        torch.testing.assert_close(target.second, source.second)
    _assert_shm_absent(namespace + "_t")


def test_failed_publication_retains_lock_until_rollback_succeeds(namespace, lock_class, monkeypatch):
    """An unlink failure cannot release the setup lock and expose the partial payload."""
    with ExitStack() as cleanup:
        creator = ShmCoordinator(namespace)
        cleanup.callback(creator.close)
        follower = ShmCoordinator(namespace)
        cleanup.callback(follower.close)
        resources = []

        def fail_initialization(child):
            resources.append(_lock_resource(child))
            child.block.buf[:4] = b"bad!"
            raise ValueError("publication failed")

        def fail_unlink(_shm):
            raise OSError("rollback failed")

        with monkeypatch.context() as patch:
            patch.setattr("flextensor.shm.flexible_shm._unlink_untracked_shm", fail_unlink)
            with pytest.raises(ValueError, match="publication failed"):
                FlexibleSharedMemory(
                    namespace + "_d",
                    8,
                    _owner=creator.memory_owner,
                    lock_class=lock_class,
                    _creator_initializer=fail_initialization,
                )
            probe = lock_class(resources[0][1], locked=False)
            try:
                with pytest.raises((BlockingIOError, posix_ipc.BusyError)):
                    probe.acquire(non_blocking=True)
            finally:
                probe.close()
        creator.close()
        _assert_shm_absent(namespace + "_d")
        with pytest.raises(FileNotFoundError):
            FlexibleSharedMemory(namespace + "_d", 0, _owner=follower.memory_owner, lock_class=lock_class)


def test_failed_publication_cannot_unlink_replacement_generation(namespace, lock_class):
    """A publisher retired during initialization must only close its old mapping."""
    with ExitStack() as cleanup:
        creator = ShmCoordinator(namespace)
        cleanup.callback(creator.close)
        follower = ShmCoordinator(namespace)
        cleanup.callback(follower.close)
        replacements = []

        def retire_then_fail(_child):
            creator.memory_owner.keep_alive.stop()
            follower.close()
            replacement = ShmCoordinator(namespace)
            cleanup.callback(replacement.close)
            current = FlexibleSharedMemory(namespace + "_d", 8, _owner=replacement.memory_owner, lock_class=lock_class)
            current.block.buf[:4] = b"new!"
            replacements.append(replacement)
            raise ValueError("publication failed")

        with pytest.raises(ValueError, match="publication failed"):
            FlexibleSharedMemory(
                namespace + "_d",
                8,
                _owner=creator.memory_owner,
                lock_class=lock_class,
                _creator_initializer=retire_then_fail,
            )
        creator.close()
        attached = FlexibleSharedMemory(namespace + "_d", 0, _owner=replacements[0].memory_owner, lock_class=lock_class)
        assert bytes(attached.block.buf[:4]) == b"new!"


def test_closed_file_lock_releases_its_thread_lock(namespace):
    creator = ShmCoordinator(namespace)
    try:
        child = FlexibleSharedMemory(namespace + "_d", 8, _owner=creator.memory_owner)
        thread_lock = weakref.ref(child.main_lock._thread_lock)
    finally:
        creator.close()
    assert thread_lock() is None


def test_child_lock_unlink_failure_is_retryable(namespace, lock_class, monkeypatch):
    creator = ShmCoordinator(namespace)
    child = FlexibleSharedMemory(namespace + "_d", 8, _owner=creator.memory_owner, lock_class=lock_class)
    resource = _lock_resource(child)
    kind, name = resource
    target, attribute = (Path, "unlink") if kind == "file" else (posix_ipc, "unlink_semaphore")
    unlink = getattr(target, attribute)

    def fail(resource_name, *args, **kwargs):
        if str(resource_name) == name:
            raise OSError("child lock unlink failed")
        return unlink(resource_name, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(target, attribute, fail)
            with pytest.raises(OSError, match="child lock unlink failed"):
                creator.close()
            _assert_lock_exists(resource, True)
        creator.close()
        _assert_lock_exists(resource, False)
        _assert_shm_absent(child.name)
    finally:
        creator.close()


def test_different_blocks_publish_in_parallel_and_same_block_waits(namespace, lock_class):
    creator = ShmCoordinator(namespace)
    follower = ShmCoordinator(namespace)
    entered = threading.Event()
    release = threading.Event()

    def initialize(child):
        entered.set()
        assert release.wait(timeout=5)
        child.block.buf[:4] = b"done"

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(
                FlexibleSharedMemory,
                namespace + "_a",
                8,
                _owner=creator.memory_owner,
                lock_class=lock_class,
                _creator_initializer=initialize,
            )
            try:
                assert entered.wait(timeout=5)
                same = pool.submit(
                    FlexibleSharedMemory, namespace + "_a", 0, _owner=follower.memory_owner, lock_class=lock_class
                )
                other = pool.submit(
                    FlexibleSharedMemory, namespace + "_b", 8, _owner=creator.memory_owner, lock_class=lock_class
                )
                other.result(timeout=2)
                assert not same.done()
                # A blocked publisher must not prevent new participants joining.
                newcomer = ShmCoordinator(namespace)
                newcomer.close()
            finally:
                release.set()
            first.result(timeout=5)
            attached = same.result(timeout=5)
            assert bytes(attached.block.buf[:4]) == b"done"
    finally:
        release.set()
        follower.close()
        creator.close()


def test_close_drains_child_setup_before_unregistering(namespace, lock_class, monkeypatch):
    creator = ShmCoordinator(namespace)
    owner = creator.memory_owner
    entered = threading.Event()
    closing = threading.Event()
    release = threading.Event()
    resources = []
    wait_for = owner._child_setup_condition.wait_for

    def signal_wait(*args, **kwargs):
        closing.set()
        return wait_for(*args, **kwargs)

    def initialize(child):
        resources.append(_lock_resource(child))
        entered.set()
        assert release.wait(timeout=5)
        child.block.buf[:4] = b"done"

    monkeypatch.setattr(owner._child_setup_condition, "wait_for", signal_wait)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            setup = pool.submit(
                FlexibleSharedMemory,
                namespace + "_d",
                8,
                _owner=owner,
                lock_class=lock_class,
                _creator_initializer=initialize,
            )
            try:
                assert entered.wait(timeout=5)
                teardown = pool.submit(creator.close)
                assert closing.wait(timeout=5)
                assert not teardown.done()
                assert creator.is_creator_alive()
                with pytest.raises(RuntimeError, match="closing"):
                    FlexibleSharedMemory(namespace + "_new", 8, _owner=owner, lock_class=lock_class)
            finally:
                release.set()
            setup.result(timeout=5)
            teardown.result(timeout=5)
        _assert_lock_exists(resources[0], False)
        _assert_shm_absent(namespace + "_d")
    finally:
        release.set()
        creator.close()


def test_close_timeout_keeps_participant_alive_for_retry(namespace, monkeypatch):
    creator = ShmCoordinator(namespace)
    entered = threading.Event()
    release = threading.Event()

    def initialize(_child):
        entered.set()
        assert release.wait(timeout=5)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            setup = pool.submit(
                FlexibleSharedMemory,
                namespace + "_d",
                8,
                _owner=creator.memory_owner,
                _creator_initializer=initialize,
            )
            try:
                assert entered.wait(timeout=5)
                monkeypatch.setattr("flextensor.shm.flexible_shm.LOCK_ACQUIRE_TIMEOUT", 0.05)
                with pytest.raises(RuntimeError, match="child setup in progress"):
                    creator.close()
                assert creator.is_creator_alive()
            finally:
                release.set()
            setup.result(timeout=5)
        creator.close()
        _assert_shm_absent(namespace + "_d")
    finally:
        release.set()
        creator.close()


def test_retired_participant_cannot_reopen_or_unlink_replacement(namespace, lock_class):
    creator = ShmCoordinator(namespace)
    stale = ShmCoordinator(namespace)
    replacement = None
    try:
        old = FlexibleSharedMemory(namespace + "_d", 8, _owner=creator.memory_owner, lock_class=lock_class)
        old_resource = _lock_resource(old)
        # Model a participant declared dead while its old mappings still exist.
        stale.memory_owner.keep_alive.stop()
        creator.close()
        replacement = ShmCoordinator(namespace)
        current = FlexibleSharedMemory(namespace + "_d", 8, _owner=replacement.memory_owner, lock_class=lock_class)
        current.block.buf[:4] = b"new!"
        assert _lock_resource(current) != old_resource
        with pytest.raises(RuntimeError, match="closing"):
            FlexibleSharedMemory(namespace + "_d", 0, _owner=stale.memory_owner, lock_class=lock_class)
        stale.close()
        _assert_lock_exists(old_resource, False)
        _assert_lock_exists(_lock_resource(current), True)
        attached = FlexibleSharedMemory(namespace + "_d", 0, _owner=replacement.memory_owner, lock_class=lock_class)
        assert bytes(attached.block.buf[:4]) == b"new!"
    finally:
        stale.close()
        creator.close()
        if replacement is not None:
            replacement.close()


def _creator_waiting_to_be_killed(namespace, lock_class, queue):
    creator = ShmCoordinator(namespace, keep_alive_seconds=0.2)
    try:
        child = FlexibleSharedMemory(namespace + "_d", 8, _owner=creator.memory_owner, lock_class=lock_class)
        queue.put(_lock_resource(child))
        while True:
            time.sleep(1)
    finally:
        creator.close()


def test_survivor_unlinks_dead_creators_child_lock(namespace, lock_class):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_creator_waiting_to_be_killed, args=(namespace, lock_class, queue))
    follower = None
    process.start()
    try:
        resource = queue.get(timeout=10)
        follower = ShmCoordinator(namespace, keep_alive_seconds=0.2)
        process.terminate()
        process.join(timeout=5)
        assert not process.is_alive()
        deadline = time.monotonic() + 5
        while follower.is_creator_alive():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        follower.close()
        _assert_lock_exists(resource, False)
        _assert_shm_absent(namespace + "_d")
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if follower is not None:
            follower.close()
        queue.close()
        queue.join_thread()

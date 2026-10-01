# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ShmCoordinator — creator/follower orchestration."""

import time
from contextlib import ExitStack
from unittest.mock import MagicMock

import posix_ipc
import pytest
import torch

from flextensor.allocation_block import AllocationManager
from flextensor.host_pinning import HostPinner
from flextensor.shm.coord_block import CoordinatorState
from flextensor.shm.coordinator import ShmCoordinator, ShmCreatorDiedError, ShmCreatorFailedError
from flextensor.shm.flexible_shm import FlexibleSharedMemory
from flextensor.shm.namespace import SHM_PROTOCOL_VERSION
from flextensor.state_handler import TensorManagerState


@pytest.fixture
def namespace():
    return f"ft_test_{int(time.time() * 1000) % 100000}"


def _make_minimal_state() -> TensorManagerState:
    return TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={},
        release_strategy={},
        label_to_block_id={},
        stats=[],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=[],
        shm_block_name_map={},
    )


def _assert_unlinked(name: str) -> None:
    with pytest.raises(posix_ipc.ExistentialError):
        unexpected = posix_ipc.SharedMemory("/" + name.lstrip("/"))
        unexpected.close_fd()


def test_late_follower_attaches_profile_and_managed_weights_after_creator_close(namespace):
    with ExitStack() as cleanup:
        creator = ShmCoordinator(namespace)
        cleanup.callback(creator.close)
        follower = ShmCoordinator(namespace)
        cleanup.callback(follower.close)
        creator_blocks = AllocationManager(
            shm_block_name_prefix=namespace + "_w",
            host_pinner=HostPinner(),
            shm_owner=creator.memory_owner,
        )
        cleanup.callback(creator_blocks.release)
        block = creator_blocks.block()
        block.add(torch.arange(12, dtype=torch.float32).reshape(3, 4).t())
        block.allocate()
        assert block.shm_block.keep_alive is None
        block_name = block.shm_block.name
        creator.write_profile(_make_minimal_state())
        creator.notify_ready()
        creator_blocks.release()
        creator.close()

        follower.wait_for_ready(timeout=1.0)
        assert follower.read_profile().loader_type == "allocation_block_transfer"
        follower_blocks = AllocationManager(
            shm_block_name_prefix=namespace + "_w",
            host_pinner=HostPinner(),
            load_from_shm=True,
            shm_owner=follower.memory_owner,
        )
        cleanup.callback(follower_blocks.release)
        attached = follower_blocks.block()
        attached.add(torch.empty_strided((4, 3), (1, 4), device="meta"))
        (view,) = attached.allocate()
        torch.testing.assert_close(view, torch.arange(12, dtype=torch.float32).reshape(3, 4).t())
    for name in (block_name, namespace + "_prof", namespace + "_crd_b", namespace + "_crd"):
        _assert_unlinked(name)


def test_owner_does_not_adopt_an_existing_unrelated_block(namespace):
    with ExitStack() as cleanup:
        existing = FlexibleSharedMemory(name=namespace + "_data", shm_size=8)
        cleanup.callback(existing.close)
        creator = ShmCoordinator(namespace)
        cleanup.callback(creator.close)
        with pytest.raises(FileExistsError, match="already exists"):
            FlexibleSharedMemory(name=existing.name, shm_size=8, _owner=creator.memory_owner)
        creator.close()
        attached = FlexibleSharedMemory(name=existing.name, shm_size=0)
        cleanup.callback(attached.close)


def test_owner_retains_liveness_until_local_mapping_can_close(namespace):
    creator = ShmCoordinator(namespace)
    child = FlexibleSharedMemory(name=namespace + "_data", shm_size=8, _owner=creator.memory_owner)
    exported = memoryview(child.block.buf)
    try:
        with pytest.raises(BufferError):
            creator.close()
        assert creator.is_creator_alive()
    finally:
        exported.release()
        creator.close()
    _assert_unlinked(child.name)


def test_last_owner_retries_failed_name_cleanup(namespace, monkeypatch):
    creator = ShmCoordinator(namespace)
    child = FlexibleSharedMemory(name=namespace + "_data", shm_size=8, _owner=creator.memory_owner)
    unlink = posix_ipc.unlink_shared_memory
    failures = 1

    def fail_once(name):
        nonlocal failures
        if name == "/" + child.name and failures:
            failures -= 1
            raise OSError("injected unlink failure")
        unlink(name)

    monkeypatch.setattr(posix_ipc, "unlink_shared_memory", fail_once)
    try:
        with pytest.raises(OSError, match="injected unlink failure"):
            creator.close()
    finally:
        creator.close()
    _assert_unlinked(child.name)
    _assert_unlinked(namespace + "_crd_b")


def test_failed_child_initialization_remains_owned_for_cleanup(namespace):
    creator = ShmCoordinator(namespace)

    def fail_initialization(shm):
        shm.block.buf[:4] = b"test"
        raise ValueError("injected publication failure")

    try:
        with pytest.raises(ValueError, match="injected publication failure"):
            FlexibleSharedMemory(
                name=namespace + "_data",
                shm_size=8,
                _owner=creator.memory_owner,
                _creator_initializer=fail_initialization,
            )
    finally:
        creator.close()
    _assert_unlinked(namespace + "_data")


def test_follower_cannot_create_missing_owned_blocks(namespace):
    with ExitStack() as cleanup:
        creator = ShmCoordinator(namespace)
        cleanup.callback(creator.close)
        follower = ShmCoordinator(namespace)
        cleanup.callback(follower.close)
        with pytest.raises(FileNotFoundError, match="has not been published"):
            FlexibleSharedMemory(name=namespace + "_new", shm_size=8, _owner=follower.memory_owner)
        child = FlexibleSharedMemory(name=namespace + "_data", shm_size=8, _owner=creator.memory_owner)
        child.close()
        posix_ipc.unlink_shared_memory("/" + child.name)
        with pytest.raises(FileNotFoundError):
            FlexibleSharedMemory(name=child.name, shm_size=8, _owner=follower.memory_owner)


class TestShmCoordinatorCreator:
    """Tests for ShmCoordinator in creator role."""

    def test_first_process_becomes_creator(self, namespace):
        """First process to create coordinator becomes creator."""
        coord = ShmCoordinator(namespace)
        try:
            assert coord.is_creator is True
        finally:
            coord.close()

    def test_creator_can_write_and_notify(self, namespace):
        """Creator can write profile and notify followers."""
        coord = ShmCoordinator(namespace)
        try:
            coord.write_profile(_make_minimal_state())
            coord.notify_ready()
        finally:
            coord.close()


class TestShmCoordinatorFollower:
    """Tests for ShmCoordinator in follower role."""

    def test_follower_detects_existing_shm(self, namespace):
        """Second coordinator detects existing SHM and becomes follower."""
        creator = ShmCoordinator(namespace)
        try:
            creator.write_profile(_make_minimal_state())
            creator.notify_ready()
            follower = ShmCoordinator(namespace)
            try:
                assert follower.is_creator is False
            finally:
                follower.close()
        finally:
            creator.close()

    def test_follower_reads_profile(self, namespace):
        """Follower can read profile written by creator."""
        state = _make_minimal_state()
        creator = ShmCoordinator(namespace)
        try:
            creator.write_profile(state)
            creator.notify_ready()
            follower = ShmCoordinator(namespace)
            try:
                loaded = follower.read_profile()
                assert loaded.loader_type == state.loader_type
            finally:
                follower.close()
        finally:
            creator.close()

    @pytest.mark.parametrize("state", list(CoordinatorState), ids=lambda state: state.name)
    @pytest.mark.parametrize("timeout", [None, 1.0])
    def test_wait_rechecks_state_after_creator_close(self, namespace, monkeypatch, state, timeout):
        """Schedule publication and close between the initial header read and liveness check."""
        with ExitStack() as cleanup:
            creator = ShmCoordinator(namespace)
            cleanup.callback(creator.close)
            follower = ShmCoordinator(namespace)
            cleanup.callback(follower.close)
            read_header = follower._read_header

            def read_then_close_creator(lock_timeout=None):
                header = read_header(lock_timeout=lock_timeout)
                assert header.state is CoordinatorState.INITIALIZING
                monkeypatch.setattr(follower, "_read_header", read_header)
                if state is CoordinatorState.READY:
                    creator.notify_ready()
                elif state is CoordinatorState.FAILED:
                    creator.mark_failed(ValueError("checkpoint\nload failed"))
                creator.close()
                return header

            monkeypatch.setattr(follower, "_read_header", read_then_close_creator)
            if state is CoordinatorState.READY:
                follower.wait_for_ready(timeout=timeout)
            elif state is CoordinatorState.FAILED:
                with pytest.raises(ShmCreatorFailedError) as exc:
                    follower.wait_for_ready(timeout=timeout)
                assert exc.value.failure_message == "ValueError: checkpoint load failed"
                assert exc.value.namespace == namespace
            else:
                with pytest.raises(ShmCreatorDiedError):
                    follower.wait_for_ready(timeout=timeout)
            assert read_header().state is state
            assert not follower.is_creator_alive()

    @pytest.mark.parametrize("held_lock_name", ["header", "heartbeat", "header_recheck"])
    def test_wait_timeout_bounds_internal_lock_contention(self, namespace, monkeypatch, held_lock_name):
        """A held header or heartbeat lock cannot outlive the follower timeout."""
        creator = ShmCoordinator(namespace)
        follower = ShmCoordinator(namespace)
        coord_shm = follower._coord_shm
        assert coord_shm is not None
        assert coord_shm.main_lock is not None
        assert coord_shm.keep_alive is not None
        held_lock = coord_shm.keep_alive.lock if held_lock_name == "heartbeat" else coord_shm.main_lock
        if held_lock_name == "header_recheck":
            is_creator_alive = follower.is_creator_alive

            def close_creator_and_hold_header_lock(lock_timeout=None):
                creator.close()
                assert not is_creator_alive(lock_timeout=lock_timeout)
                held_lock.acquire(non_blocking=False, timeout=1.0)
                return False

            monkeypatch.setattr(follower, "is_creator_alive", close_creator_and_hold_header_lock)
        else:
            held_lock.acquire(non_blocking=False, timeout=1.0)
        try:
            started = time.monotonic()
            with pytest.raises(TimeoutError, match="timed out waiting") as exc:
                follower.wait_for_ready(timeout=0.05)
            assert isinstance(exc.value.__cause__, BlockingIOError)
            assert time.monotonic() - started < 1.0
        finally:
            held_lock.release()
            follower.close()
            creator.close()


class TestShmCoordinatorRoleGuards:
    """Tests for role enforcement on write_profile/read_profile."""

    def test_follower_cannot_write_profile(self, namespace):
        """write_profile raises RuntimeError on follower."""
        creator = ShmCoordinator(namespace)
        try:
            creator.write_profile(_make_minimal_state())
            creator.notify_ready()
            follower = ShmCoordinator(namespace)
            try:
                with pytest.raises(RuntimeError, match="follower"):
                    follower.write_profile(_make_minimal_state())
            finally:
                follower.close()
        finally:
            creator.close()

    def test_creator_cannot_read_profile(self, namespace):
        """read_profile raises RuntimeError on creator."""
        creator = ShmCoordinator(namespace)
        try:
            with pytest.raises(RuntimeError, match="creator"):
                creator.read_profile()
        finally:
            creator.close()


class TestShmCoordinatorVersionGate:
    """Tests for version gate enforcement."""

    @pytest.mark.parametrize("cleanup_error_type", [OSError, KeyboardInterrupt])
    def test_validation_error_survives_cleanup_failure(self, monkeypatch, caplog, cleanup_error_type):
        validation_error = RuntimeError("protocol version mismatch")
        cleanup_error = cleanup_error_type("close failed")
        close = MagicMock(side_effect=cleanup_error)

        class _FollowerShm(FlexibleSharedMemory):
            def __init__(self, *args, **kwargs):
                self.shm_creator = False
                self.close = close

        monkeypatch.setattr("flextensor.shm.coordinator.FlexibleSharedMemory", _FollowerShm)
        monkeypatch.setattr(ShmCoordinator, "_validate_version", MagicMock(side_effect=validation_error))

        with pytest.raises(RuntimeError, match="protocol version mismatch") as exc_info:
            ShmCoordinator("validation_cleanup_failure")

        assert exc_info.value is validation_error
        close.assert_called_once_with()
        assert any(
            record.getMessage() == "Failed to close SHM after follower validation failed"
            and record.exc_info is not None
            and record.exc_info[1] is cleanup_error
            for record in caplog.records
        )

    def test_v3_rejected_before_attaching_the_new_name_registry(self, namespace):
        from flextensor._version import __version__
        from flextensor.shm.coord_block import CoordBlockHeader

        def initialize(shm):
            CoordBlockHeader(flextensor_version=__version__, protocol_version=3).write_to(shm.block.buf)

        old_coord = FlexibleSharedMemory(name=namespace + "_crd", shm_size=256, _creator_initializer=initialize)
        try:
            with pytest.raises(RuntimeError, match="protocol version mismatch"):
                ShmCoordinator(namespace)
        finally:
            old_coord.close()

    def test_version_mismatch_raises(self, namespace):
        """Follower rejects SHM from a different FlexTensor version."""
        creator = ShmCoordinator(namespace)
        try:
            from flextensor.shm.coord_block import CoordBlockHeader

            fake_header = CoordBlockHeader(
                flextensor_version="0.0.0+fake",
                protocol_version=SHM_PROTOCOL_VERSION,
            )
            fake_header.write_to(creator._coord_shm.block.buf, offset=0)

            with pytest.raises(RuntimeError, match="FlexTensor version mismatch"):
                ShmCoordinator(namespace)
        finally:
            creator.close()

    def test_version_mismatch_cleans_up_shm(self, namespace):
        """Version mismatch closes SHM handle (no leak)."""
        creator = ShmCoordinator(namespace)
        try:
            from flextensor.shm.coord_block import CoordBlockHeader

            fake_header = CoordBlockHeader(
                flextensor_version="0.0.0+fake",
                protocol_version=SHM_PROTOCOL_VERSION,
            )
            fake_header.write_to(creator._coord_shm.block.buf, offset=0)

            with pytest.raises(RuntimeError):
                follower = ShmCoordinator(namespace)
                # Should not reach here, but if it does, clean up
                follower.close()  # pragma: no cover
        finally:
            creator.close()

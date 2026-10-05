# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHM coordinator for creator/follower weight sharing."""

from __future__ import annotations

import logging
import math
import sys
import time

from flextensor._version import __version__
from flextensor.shm.coord_block import CoordBlockHeader, CoordinatorState
from flextensor.shm.flexible_shm import FlexibleSharedMemory
from flextensor.shm.namespace import SHM_PROTOCOL_VERSION, coord_block_name, profile_block_name
from flextensor.state_handler import TensorManagerState, TensorManagerStateHandler

if sys.platform != "win32":
    import posix_ipc

logger = logging.getLogger(__name__)

# Coordination block stores only the CoordBlockHeader (248 B in protocol v4).
# KeepAlive uses separate SHM segments managed by FlexibleSharedMemory.
_COORD_BLOCK_SIZE = 256  # bytes — header + room for future fields
_LIVENESS_POLL_INTERVAL_SECONDS = 0.25
DEFAULT_KEEP_ALIVE_SECONDS = 30.0


class ShmCoordinatorError(RuntimeError):
    """Base class for coordinator-level shared initialization failures."""


class ShmCreatorFailedError(ShmCoordinatorError):
    """Raised in followers when the creator explicitly publishes a failure."""

    def __init__(self, namespace: str, failure_message: str) -> None:
        self.namespace = namespace
        self.failure_message = failure_message
        detail = failure_message or "creator initialization failed without details"
        super().__init__(f"Shared initialization for namespace '{namespace}' failed: {detail}")


class ShmCreatorDiedError(ShmCoordinatorError):
    """Raised in followers when the initializing creator heartbeat expires.

    ``creator_identity`` is the creator participant ID, not an OS PID.
    The constructor keyword and attribute name are retained for compatibility.
    """

    def __init__(self, namespace: str, creator_identity: int | None) -> None:
        self.namespace = namespace
        self.creator_identity = creator_identity
        detail = "unknown" if creator_identity is None else str(creator_identity)
        super().__init__(f"Creator participant {detail} died while initializing shared namespace '{namespace}'")


def _require_block(shm: FlexibleSharedMemory | None, label: str) -> FlexibleSharedMemory:
    """Return *shm* after verifying it and its block are not None."""
    if shm is None or shm.block is None:
        msg = f"ShmCoordinator: {label} SHM block is not initialised"
        raise RuntimeError(msg)
    return shm


def _require_buffer(shm: FlexibleSharedMemory | None, label: str) -> memoryview:
    """Return an open buffer for a required SHM block."""
    block = _require_block(shm, label).block
    if block is None or block.buf is None:
        raise RuntimeError(f"ShmCoordinator: {label} SHM buffer is not initialised")
    return block.buf


class ShmCoordinator:
    """Orchestrates creator/follower SHM sharing.

    On init, tries to connect to an existing coordination block:
    - If it exists: becomes follower, validates version gate.
    - If it doesn't exist: becomes creator, writes version header.

    Args:
        namespace: SHM namespace shared by the intended creator and followers.
        keep_alive_seconds: Timeout for participant-heartbeat liveness detection.
    """

    def __init__(self, namespace: str, keep_alive_seconds: int | float = DEFAULT_KEEP_ALIVE_SECONDS) -> None:
        self.namespace = namespace
        self._coord_shm: FlexibleSharedMemory | None = None
        self._profile_shm: FlexibleSharedMemory | None = None
        self.is_creator = False
        self._creator_participant_id: int | None = None

        crd_name = coord_block_name(namespace)

        try:
            coord_shm = FlexibleSharedMemory(
                name=crd_name,
                shm_size=0,
                keep_alive_seconds=keep_alive_seconds,
            )
        except FileNotFoundError:
            # The initializer executes only for the actual creator and while
            # FlexibleSharedMemory still owns its exclusive setup lock. A race
            # loser attaches after the header is fully published.
            coord_shm = FlexibleSharedMemory(
                name=crd_name,
                shm_size=_COORD_BLOCK_SIZE,
                keep_alive_seconds=keep_alive_seconds,
                _creator_initializer=self._write_version_header,
            )

        self._coord_shm = coord_shm
        self.is_creator = coord_shm.shm_creator
        if not self.is_creator:
            try:
                self._validate_version()
                coord_shm._enable_child_ownership(create=False)  # noqa: SLF001
            except BaseException:
                self._coord_shm = None
                try:
                    coord_shm.close()
                except BaseException:
                    logger.exception("Failed to close SHM after follower validation failed")
                raise
        logger.info("ShmCoordinator: %s initialized %s", "creator" if self.is_creator else "follower", namespace)

    def _write_version_header(self, shm: FlexibleSharedMemory) -> None:
        """Write version header to coordination block (creator only)."""
        shm._enable_child_ownership(create=True)  # noqa: SLF001
        creator_participant_id = shm.participant_id
        coord_shm = _require_block(shm, "coord")
        header = CoordBlockHeader(
            flextensor_version=__version__,
            protocol_version=SHM_PROTOCOL_VERSION,
            # Retain the field name for Python API compatibility. Protocol v4
            # stores the full participant ID in the existing 8-byte header field.
            creator_pid=creator_participant_id,
        )
        self._creator_participant_id = creator_participant_id
        # block guaranteed non-None by _require_block
        header.write_to(coord_shm.block.buf, offset=0)  # type: ignore[arg-type, union-attr]

    @property
    def memory_owner(self) -> FlexibleSharedMemory:
        """The shared lifetime for this manager's profile and tensor blocks."""
        return _require_block(self._coord_shm, "coord")

    def _validate_version(self) -> None:
        """Read and validate version header (follower only)."""
        header = self._read_header()
        self._creator_participant_id = header.creator_pid

    def _read_header(self, lock_timeout: float | None = None) -> CoordBlockHeader:
        """Read a coherent coordination header under the shared setup lock."""
        shm = _require_block(self._coord_shm, "coord")
        if shm.main_lock is None:
            msg = "ShmCoordinator: coord SHM lock is not initialised"
            raise RuntimeError(msg)
        shm.main_lock.acquire(non_blocking=False, timeout=lock_timeout)
        try:
            header = CoordBlockHeader.read_from(shm.block.buf, offset=0)  # type: ignore[arg-type, union-attr]
        finally:
            shm.main_lock.release()
        header.validate(expected_version=__version__, expected_protocol=SHM_PROTOCOL_VERSION)
        return header

    def _transition_creator_state(
        self,
        target: CoordinatorState,
        *,
        failure_message: str = "",
    ) -> None:
        """Atomically transition the creator from INITIALIZING to a terminal state."""
        if not self.is_creator:
            raise RuntimeError("Coordinator state can only be changed by the creator")
        shm = _require_block(self._coord_shm, "coord")
        if shm.main_lock is None:
            msg = "ShmCoordinator: coord SHM lock is not initialised"
            raise RuntimeError(msg)
        buffer = _require_buffer(shm, "coord")

        with shm.main_lock:
            header = CoordBlockHeader.read_from(buffer, offset=0)
            header.validate(expected_version=__version__, expected_protocol=SHM_PROTOCOL_VERSION)
            current = header.state
            if not isinstance(current, CoordinatorState):
                raise RuntimeError(f"Invalid coordinator state: {current}")
            if current == target:
                return
            if current is not CoordinatorState.INITIALIZING:
                raise RuntimeError(f"Cannot transition coordinator from {current.name} to {target.name}")
            header.state = target
            header.failure_message = failure_message
            header.write_to(buffer, offset=0)

    @staticmethod
    def _failure_summary(error: BaseException) -> str:
        """Build a bounded, single-line failure summary safe for shared metadata."""
        try:
            detail = str(error)
        except Exception:  # pragma: no cover - pathological exception implementation
            detail = "<unprintable exception>"
        summary = " ".join(f"{type(error).__name__}: {detail}".split())
        # CoordBlockHeader performs the final UTF-8 byte truncation.
        return summary

    def write_profile(self, state: TensorManagerState) -> None:
        """Write profile to SHM (creator only).

        Args:
            state: The TensorManagerState to share with followers.

        Raises:
            RuntimeError: If called on a follower coordinator.
            FileExistsError: If the profile block already exists.
        """
        if not self.is_creator:
            raise RuntimeError("write_profile called on follower coordinator")
        if self._profile_shm is not None:
            raise RuntimeError("write_profile called after this coordinator already published a profile")

        serialized = TensorManagerStateHandler.save_state_to_bytes(state)
        prof_name = profile_block_name(self.namespace)

        def initialize(shm: FlexibleSharedMemory) -> None:
            profile_shm = _require_block(shm, "profile")
            # The OS may round the physical mapping up to a page. Only the
            # serialized profile's logical capacity and bytes are authoritative.
            _require_buffer(profile_shm, "profile")[: len(serialized)] = serialized

        profile_shm = FlexibleSharedMemory(
            name=prof_name,
            shm_size=len(serialized),
            _creator_initializer=initialize,
            _owner=self.memory_owner,
        )
        if not profile_shm.shm_creator:
            profile_shm.close()
            raise FileExistsError(f"ShmCoordinator profile block '{prof_name}' already exists")
        self._profile_shm = profile_shm

    def read_profile(self) -> TensorManagerState:
        """Read profile from SHM (follower).

        Returns:
            The deserialized TensorManagerState.

        Raises:
            RuntimeError: If called on a creator coordinator or data is corrupt.
        """
        if self.is_creator:
            raise RuntimeError("read_profile called on creator coordinator")
        prof_name = profile_block_name(self.namespace)
        self._profile_shm = FlexibleSharedMemory(name=prof_name, shm_size=0, _owner=self.memory_owner)
        shm = _require_block(self._profile_shm, "profile")
        # Copy buffer to immutable bytes — prevents concurrent creator writes from
        # corrupting the JSON parse mid-read.
        data = bytes(shm.block.buf)  # type: ignore[arg-type, union-attr]
        try:
            return TensorManagerStateHandler.load_state_from_bytes(data)
        except Exception as exc:
            msg = f"ShmCoordinator: failed to deserialize profile from SHM block '{prof_name}'"
            raise RuntimeError(msg) from exc

    def _wait_timeout_error(self, timeout: float | None) -> TimeoutError:
        return TimeoutError(
            f"ShmCoordinator: timed out waiting for namespace '{self.namespace}' after {timeout} seconds"
        )

    def _remaining_wait(self, deadline: float | None, timeout: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._wait_timeout_error(timeout)
        return remaining

    def wait_for_ready(self, timeout: float | None = None) -> None:
        """Block until the creator publishes READY or a terminal failure.

        Args:
            timeout: Maximum seconds to wait, or ``None`` to wait indefinitely.

        Raises:
            TimeoutError: If the namespace is not ready before *timeout* expires.
            ShmCreatorFailedError: If the creator publishes FAILED.
            ShmCreatorDiedError: If its heartbeat expires while INITIALIZING.
        """
        _require_block(self._coord_shm, "coord")
        if self.is_creator:
            raise RuntimeError("wait_for_ready called on creator coordinator")
        if timeout is not None and (timeout < 0 or not math.isfinite(timeout)):
            raise ValueError("timeout must be finite and non-negative, or None")

        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                lock_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                header = self._read_header(lock_timeout=lock_timeout)
                if header.state is CoordinatorState.INITIALIZING and not self.is_creator_alive(
                    lock_timeout=self._remaining_wait(deadline, timeout)
                ):
                    # The creator may have published a terminal state before closing.
                    header = self._read_header(lock_timeout=self._remaining_wait(deadline, timeout))
                    if header.state is CoordinatorState.INITIALIZING:
                        raise ShmCreatorDiedError(self.namespace, self._creator_participant_id)
                if header.state is CoordinatorState.READY:
                    return
                if header.state is CoordinatorState.FAILED:
                    raise ShmCreatorFailedError(self.namespace, header.failure_message)

                remaining = self._remaining_wait(deadline, timeout)
                wait_timeout = (
                    _LIVENESS_POLL_INTERVAL_SECONDS
                    if remaining is None
                    else min(_LIVENESS_POLL_INTERVAL_SECONDS, remaining)
                )
                # Poll the authoritative header so creator liveness and state
                # transitions remain observable to late joiners.
                time.sleep(wait_timeout)
        except (BlockingIOError, posix_ipc.BusyError) as exc:
            if deadline is None:
                raise
            raise self._wait_timeout_error(timeout) from exc

    def notify_ready(self) -> None:
        """Publish READY. Followers observe the authoritative header by polling."""
        if self._coord_shm is None:
            msg = "ShmCoordinator: coord SHM is not initialised"
            raise RuntimeError(msg)
        self._transition_creator_state(CoordinatorState.READY)

    def mark_failed(self, error: BaseException) -> None:
        """Publish a creator initialization failure.

        Args:
            error: Original creator-side failure. Only its type and sanitized,
                bounded string representation are published.
        """
        if self._coord_shm is None:
            msg = "ShmCoordinator: coord SHM is not initialised"
            raise RuntimeError(msg)
        self._transition_creator_state(
            CoordinatorState.FAILED,
            failure_message=self._failure_summary(error),
        )

    def is_creator_alive(self, lock_timeout: float | None = None) -> bool:
        """Check whether the creator participant's heartbeat remains live."""
        if self._coord_shm is None or self._coord_shm.keep_alive is None or self._creator_participant_id is None:
            return False
        return self._coord_shm.keep_alive.is_process_alive(self._creator_participant_id, lock_timeout=lock_timeout)

    def close(self) -> None:
        """Release profile data before the control plane, retaining failed handles."""
        if self._profile_shm is not None:
            self._profile_shm.close()
            self._profile_shm = None
        if self._coord_shm is not None:
            self._coord_shm.close()
            self._coord_shm = None

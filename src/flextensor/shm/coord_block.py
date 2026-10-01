# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordination block header for SHM version gating."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

# Header layout: 4-byte version string length, up to 64 bytes version string,
# 4-byte protocol version, 8-byte creator participant ID, 4-byte coordinator state,
# 4-byte failure-string length, and up to 160 bytes of failure text.
_VERSION_MAX_LEN = 64
_FAILURE_MAX_LEN = 160
_COORD_HEADER_FORMAT = f"!I{_VERSION_MAX_LEN}sIQII{_FAILURE_MAX_LEN}s"
COORD_HEADER_SIZE = struct.calcsize(_COORD_HEADER_FORMAT)


class CoordinatorState(IntEnum):
    """Authoritative lifecycle state stored in the coordination block."""

    INITIALIZING = 1
    READY = 2
    FAILED = 3


@dataclass
class CoordBlockHeader:
    """Version and protocol header written at the start of the coordination block.

    Args:
        flextensor_version: FlexTensor version string (e.g., "0.3.3").
        protocol_version: SHM protocol version integer.
        creator_pid: In protocol v4, the full creator participant ID, not an OS PID.
            The field name is retained for Python API compatibility.
        state: Current creator initialization state.
        failure_message: Bounded, sanitized creator failure summary when state is FAILED.
    """

    flextensor_version: str
    protocol_version: int
    creator_pid: int = 0
    state: CoordinatorState | int = CoordinatorState.INITIALIZING
    failure_message: str = ""

    def write_to(self, buf: bytearray | memoryview, offset: int = 0) -> None:
        """Write header to buffer at offset.

        Args:
            buf: Target buffer (must have at least COORD_HEADER_SIZE bytes from offset).
            offset: Byte offset into the buffer.
        """
        version_bytes = self.flextensor_version.encode("utf-8")[:_VERSION_MAX_LEN]
        padded = version_bytes.ljust(_VERSION_MAX_LEN, b"\x00")
        failure_bytes = self.failure_message.encode("utf-8")[:_FAILURE_MAX_LEN]
        # Avoid publishing a partial UTF-8 code point when truncation lands in
        # the middle of a multi-byte character.
        failure_bytes = failure_bytes.decode("utf-8", errors="ignore").encode("utf-8")
        padded_failure = failure_bytes.ljust(_FAILURE_MAX_LEN, b"\x00")
        struct.pack_into(
            _COORD_HEADER_FORMAT,
            buf,
            offset,
            len(version_bytes),
            padded,
            self.protocol_version,
            self.creator_pid,
            int(self.state),
            len(failure_bytes),
            padded_failure,
        )

    @classmethod
    def read_from(cls, buf: bytes | bytearray | memoryview, offset: int = 0) -> CoordBlockHeader:
        """Read header from buffer at offset.

        Args:
            buf: Source buffer.
            offset: Byte offset into the buffer.

        Returns:
            Parsed CoordBlockHeader instance.
        """
        (
            version_len,
            version_bytes,
            protocol,
            creator_participant_id,
            state,
            failure_len,
            failure_bytes,
        ) = struct.unpack_from(_COORD_HEADER_FORMAT, buf, offset)
        if version_len > _VERSION_MAX_LEN:
            raise RuntimeError(f"Invalid coordination header version length: {version_len}")
        if failure_len > _FAILURE_MAX_LEN:
            raise RuntimeError(f"Invalid coordination header failure length: {failure_len}")
        version_str = version_bytes[:version_len].decode("utf-8")
        failure_message = failure_bytes[:failure_len].decode("utf-8")
        try:
            parsed_state: CoordinatorState | int = CoordinatorState(state)
        except ValueError:
            # Preserve unknown/legacy values so validate() can report a protocol
            # mismatch before checking fields introduced by a newer protocol.
            parsed_state = state
        return cls(
            flextensor_version=version_str,
            protocol_version=protocol,
            creator_pid=creator_participant_id,
            state=parsed_state,
            failure_message=failure_message,
        )

    def validate(self, expected_version: str, expected_protocol: int) -> None:
        """Validate header against expected values.

        Args:
            expected_version: Expected FlexTensor version string.
            expected_protocol: Expected SHM protocol version.

        Raises:
            RuntimeError: If version or protocol doesn't match.
        """
        if self.flextensor_version != expected_version:
            raise RuntimeError(
                f"FlexTensor version mismatch: SHM created by {self.flextensor_version}, "
                f"this process is {expected_version}. All processes must use the same version."
            )
        if self.protocol_version != expected_protocol:
            raise RuntimeError(
                f"SHM protocol version mismatch: SHM uses protocol {self.protocol_version}, "
                f"this process expects {expected_protocol}. All processes must use the same FlexTensor version."
            )
        if self.protocol_version >= 3 and self.creator_pid <= 0:
            raise RuntimeError("SHM protocol v3+ requires a positive creator identity.")
        if self.protocol_version >= 4 and not isinstance(self.state, CoordinatorState):
            raise RuntimeError(f"SHM protocol v4 has an invalid coordinator state: {self.state}")
        if self.state is not CoordinatorState.FAILED and self.failure_message:
            raise RuntimeError("Only a FAILED coordination header may contain a failure message.")

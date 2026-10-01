# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for coordination block header read/write/validate."""

import pytest

import flextensor
from flextensor.shm.coord_block import COORD_HEADER_SIZE, CoordBlockHeader, CoordinatorState


class TestCoordBlockHeader:
    """Tests for coordination block header read/write."""

    def test_header_roundtrip(self):
        """Write and read header produces same values."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=1,
            creator_pid=4_294_967_297,
        )
        buf = bytearray(COORD_HEADER_SIZE + 256)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        assert loaded.flextensor_version == flextensor.__version__
        assert loaded.protocol_version == 1
        assert loaded.creator_pid == 4_294_967_297
        assert loaded.state is CoordinatorState.INITIALIZING
        assert loaded.failure_message == ""

    def test_failed_header_roundtrip(self):
        """FAILED state and its bounded diagnostic survive serialization."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=4,
            creator_pid=4_294_967_297,
            state=CoordinatorState.FAILED,
            failure_message="RuntimeError: checkpoint load failed",
        )

        buf = bytearray(COORD_HEADER_SIZE)
        header.write_to(buf)

        assert CoordBlockHeader.read_from(buf) == header

    def test_failure_message_is_utf8_safely_truncated(self):
        """A long multi-byte diagnostic never leaves partial UTF-8 on the wire."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=4,
            creator_pid=1,
            state=CoordinatorState.FAILED,
            failure_message="ą" * 200,
        )

        buf = bytearray(COORD_HEADER_SIZE)
        header.write_to(buf)
        loaded = CoordBlockHeader.read_from(buf)

        assert loaded.failure_message
        assert len(loaded.failure_message.encode("utf-8")) <= 160
        assert set(loaded.failure_message) == {"ą"}

    def test_version_mismatch_raises(self):
        """Mismatched FlexTensor version raises clear error."""
        header = CoordBlockHeader(
            flextensor_version="0.0.0+fake",
            protocol_version=1,
        )
        buf = bytearray(COORD_HEADER_SIZE + 256)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        with pytest.raises(RuntimeError, match="FlexTensor version mismatch"):
            loaded.validate(
                expected_version=flextensor.__version__,
                expected_protocol=1,
            )

    def test_protocol_mismatch_raises(self):
        """Mismatched protocol version raises clear error."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=99,
        )
        buf = bytearray(COORD_HEADER_SIZE + 256)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        with pytest.raises(RuntimeError, match="SHM protocol version mismatch"):
            loaded.validate(
                expected_version=flextensor.__version__,
                expected_protocol=1,
            )

    def test_matching_version_passes(self):
        """Matching versions do not raise."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=1,
        )
        buf = bytearray(COORD_HEADER_SIZE + 256)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        # Should not raise
        loaded.validate(
            expected_version=flextensor.__version__,
            expected_protocol=1,
        )

    def test_protocol_v3_requires_creator_pid(self):
        """Protocol v3 rejects a header without creator identity."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=3,
        )

        with pytest.raises(RuntimeError, match="creator identity"):
            header.validate(
                expected_version=flextensor.__version__,
                expected_protocol=3,
            )

    def test_protocol_v4_rejects_unknown_state(self):
        """Protocol v4 rejects corrupt or unknown lifecycle values."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=4,
            creator_pid=1,
            state=99,
        )

        with pytest.raises(RuntimeError, match="invalid coordinator state"):
            header.validate(
                expected_version=flextensor.__version__,
                expected_protocol=4,
            )

    def test_non_failed_header_rejects_failure_message(self):
        """Failure diagnostics cannot be published without the FAILED state."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=4,
            creator_pid=1,
            state=CoordinatorState.READY,
            failure_message="should not be present",
        )

        with pytest.raises(RuntimeError, match="Only a FAILED"):
            header.validate(
                expected_version=flextensor.__version__,
                expected_protocol=4,
            )

    def test_header_at_nonzero_offset(self):
        """Header can be written and read at an arbitrary offset."""
        header = CoordBlockHeader(
            flextensor_version="1.2.3",
            protocol_version=2,
        )
        buf = bytearray(512)
        header.write_to(buf, offset=64)
        loaded = CoordBlockHeader.read_from(buf, offset=64)
        assert loaded.flextensor_version == "1.2.3"
        assert loaded.protocol_version == 2

    def test_short_version_string(self):
        """Short version strings are handled correctly."""
        header = CoordBlockHeader(
            flextensor_version="0.1",
            protocol_version=1,
        )
        buf = bytearray(COORD_HEADER_SIZE)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        assert loaded.flextensor_version == "0.1"

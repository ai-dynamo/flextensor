# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SHM serialization of TensorManagerState."""

import json
import struct
from multiprocessing.shared_memory import SharedMemory

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.state_handler import TensorManagerState, TensorManagerStateHandler


def _make_minimal_state() -> TensorManagerState:
    """Create a minimal valid TensorManagerState for testing."""
    tensor_stats = [
        TensorStatistics(tensor_id=0, name="layer0.weight", size_bytes=1024, load_time_ms=0.1),
        TensorStatistics(tensor_id=1, name="layer1.weight", size_bytes=2048, load_time_ms=0.2),
        TensorStatistics(tensor_id=2, name="layer2.weight", size_bytes=512, load_time_ms=0.05),
    ]
    return TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={0: "layer0.weight", 1: "layer1.weight", 2: "layer2.weight"},
        allocation_ordered={0: ["trap_0"], 1: ["trap_1"]},
        label_to_size_map={"trap_0": 3072, "trap_1": 512},
        block_sizes={0: 3072, 1: 512},
        load_strategy={"trap_0": tensor_stats[:2], "trap_1": tensor_stats[2:]},
        release_strategy={"trap_0": tensor_stats[:2], "trap_1": tensor_stats[2:]},
        label_to_block_id={"trap_0": 0, "trap_1": 1},
        stats=[
            LayerStatistics(label="trap_0", tensors=tensor_stats[:2], duration=1.0),
            LayerStatistics(label="trap_1", tensors=tensor_stats[2:], duration=0.5),
        ],
        transfer_to_compute_map={"trap_0": "trap_0", "trap_1": "trap_1"},
        view_tensors_ids=[0, 1, 2],
        view_tensors_names=["layer0.weight", "layer1.weight", "layer2.weight"],
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )


class TestStateHandlerShm:
    """Tests for SHM serialization of TensorManagerState."""

    def test_schema_v2_is_rejected_as_outdated(self):
        saved_state = _make_minimal_state().to_dict()
        saved_state["version"] = 2

        with pytest.raises(ValueError, match=r"schema version 2 is outdated \(current: 3\)"):
            TensorManagerState.from_dict(saved_state)

    def test_current_schema_v3_roundtrips(self):
        saved_state = _make_minimal_state().to_dict()

        assert saved_state["version"] == 3
        assert TensorManagerState.from_dict(saved_state).to_dict() == saved_state

    def test_legacy_v3_without_managed_tensor_metadata_still_loads(self):
        saved_state = _make_minimal_state().to_dict()
        saved_state.pop("managed_tensor_dtypes")
        saved_state.pop("managed_tensor_layouts")

        loaded = TensorManagerState.from_dict(saved_state)

        assert loaded.managed_tensor_dtypes == {}
        assert loaded.managed_tensor_layouts == {}

    @pytest.mark.parametrize(
        "first_layout",
        [
            {"shape": [16, 16], "stride": [1, 16]},
            {"shape": [16, 16], "stride": [16, 1]},
            {"shape": [1, 256], "stride": [0, 1]},
            {"shape": [4, 4, 16], "stride": [4, 1, 16]},
        ],
    )
    def test_mixed_managed_dtypes_roundtrip(self, first_layout):
        state = _make_minimal_state()
        state.managed_tensor_dtypes = {
            "layer0.weight": "float32",
            "layer1.weight": "bfloat16",
            "layer2.weight": "float8_e4m3fn",
        }
        state.managed_tensor_layouts = {
            "layer0.weight": first_layout,
            "layer1.weight": {"shape": [32, 32], "stride": [32, 1]},
            "layer2.weight": {"shape": [512], "stride": [1]},
        }

        loaded = TensorManagerStateHandler.load_state_from_bytes(TensorManagerStateHandler.save_state_to_bytes(state))

        assert loaded.managed_tensor_dtypes == state.managed_tensor_dtypes
        assert loaded.managed_tensor_layouts == state.managed_tensor_layouts

    def test_partial_managed_layout_map_is_rejected(self):
        saved_state = _make_minimal_state().to_dict()
        saved_state["managed_tensor_layouts"] = {"layer0.weight": {"shape": [256], "stride": [1]}}

        with pytest.raises(ValueError, match=r"managed_tensor_layouts.*cover every managed tensor"):
            TensorManagerState.from_dict(saved_state)

    @pytest.mark.parametrize(
        "layout",
        [
            {"shape": [256]},
            {"shape": [16, 16], "stride": [1]},
            {"shape": [-256], "stride": [1]},
            {"shape": [256], "stride": [-1]},
            {"shape": [256], "stride": [2]},
            {"shape": [256], "stride": [0]},
            {"shape": [True], "stride": [1]},
            {"shape": [256], "stride": [True]},
            {"shape": [3, 3], "stride": [2, 2]},
            {"shape": [1, 256], "stride": [2**64, 1]},
        ],
    )
    def test_invalid_managed_layout_is_rejected(self, layout):
        saved_state = _make_minimal_state().to_dict()
        saved_state["managed_tensor_layouts"] = {
            "layer0.weight": layout,
            "layer1.weight": {"shape": [512], "stride": [1]},
            "layer2.weight": {"shape": [128], "stride": [1]},
        }

        with pytest.raises(ValueError, match=r"invalid layout metadata"):
            TensorManagerState.from_dict(saved_state)

    def test_partial_managed_dtype_map_is_rejected(self):
        saved_state = _make_minimal_state().to_dict()
        saved_state["managed_tensor_dtypes"] = {"layer0.weight": "float32"}

        with pytest.raises(ValueError, match=r"managed_tensor_dtypes.*cover every managed tensor"):
            TensorManagerState.from_dict(saved_state)

    @pytest.mark.parametrize("dtype_name", ["not_a_torch_dtype", "Tensor", "", "float"])
    def test_invalid_managed_dtype_is_rejected(self, dtype_name):
        saved_state = _make_minimal_state().to_dict()
        saved_state["managed_tensor_dtypes"] = {
            "layer0.weight": "float32",
            "layer1.weight": dtype_name,
            "layer2.weight": "float32",
        }

        with pytest.raises(ValueError, match=r"invalid dtype metadata for 'layer1.weight'"):
            TensorManagerState.from_dict(saved_state)

    def test_roundtrip_to_bytes(self):
        """State survives serialization to bytes and back."""
        state = _make_minimal_state()
        buf = TensorManagerStateHandler.save_state_to_bytes(state)
        loaded = TensorManagerStateHandler.load_state_from_bytes(buf)
        assert loaded.loader_type == state.loader_type
        assert loaded.allocation_ordered == state.allocation_ordered
        assert loaded.label_to_block_id == state.label_to_block_id
        assert loaded.block_sizes == state.block_sizes
        assert loaded.transfer_to_compute_map == state.transfer_to_compute_map

    def test_save_to_bytes_format(self):
        """Serialized bytes start with 4-byte length prefix."""
        state = _make_minimal_state()
        buf = TensorManagerStateHandler.save_state_to_bytes(state)
        length = struct.unpack("!I", buf[:4])[0]
        assert length == len(buf) - 4  # length prefix excludes itself
        # Payload is valid JSON
        json.loads(buf[4 : 4 + length])

    def test_load_from_shm_buffer(self):
        """Can load state from a SharedMemory buffer."""
        state = _make_minimal_state()
        serialized = TensorManagerStateHandler.save_state_to_bytes(state)

        # Simulate SHM buffer
        shm = SharedMemory(create=True, size=len(serialized) + 1024)
        try:
            shm.buf[: len(serialized)] = serialized
            loaded = TensorManagerStateHandler.load_state_from_bytes(bytes(shm.buf[: len(serialized)]))
            assert loaded.allocation_ordered == state.allocation_ordered
            assert loaded.label_to_block_id == state.label_to_block_id
        finally:
            shm.close()
            shm.unlink()

    def test_roundtrip_preserves_tensor_stats(self):
        """Tensor statistics survive roundtrip serialization."""
        state = _make_minimal_state()
        buf = TensorManagerStateHandler.save_state_to_bytes(state)
        loaded = TensorManagerStateHandler.load_state_from_bytes(buf)
        # Check load_strategy tensor stats
        for label in state.load_strategy:
            assert len(loaded.load_strategy[label]) == len(state.load_strategy[label])
            for orig, restored in zip(state.load_strategy[label], loaded.load_strategy[label], strict=False):
                assert restored.name == orig.name
                assert restored.size_bytes == orig.size_bytes

    def test_roundtrip_preserves_layer_stats(self):
        """Layer statistics survive roundtrip serialization."""
        state = _make_minimal_state()
        buf = TensorManagerStateHandler.save_state_to_bytes(state)
        loaded = TensorManagerStateHandler.load_state_from_bytes(buf)
        assert len(loaded.stats) == len(state.stats)
        for orig, restored in zip(state.stats, loaded.stats, strict=False):
            assert restored.label == orig.label
            assert restored.duration == orig.duration

    def test_load_from_bytes_empty_buffer_raises(self):
        """Empty buffer raises ValueError with clear message."""
        with pytest.raises(ValueError, match=r"Buffer too short.*length prefix"):
            TensorManagerStateHandler.load_state_from_bytes(b"")

    def test_load_from_bytes_truncated_payload_raises(self):
        """Buffer with valid length prefix but truncated payload raises ValueError."""
        # Encode length prefix claiming 1000 bytes, but provide only 10
        buf = struct.pack("!I", 1000) + b"x" * 10
        with pytest.raises(ValueError, match=r"Buffer too short.*1000 bytes"):
            TensorManagerStateHandler.load_state_from_bytes(buf)

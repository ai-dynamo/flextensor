# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for remove_layers_compound function."""

from enum import IntEnum

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.strategy_operations import remove_layers_compound


class TestRemoveLayersCompound:
    """Test cases for remove_layers_compound function."""

    @pytest.fixture
    def sample_tensor_stats(self):
        """Create sample tensor statistics for testing."""
        return [
            TensorStatistics(tensor_id=1, name="tensor1", size_bytes=1024, load_time_ms=10.0),
            TensorStatistics(tensor_id=2, name="tensor2", size_bytes=2048, load_time_ms=20.0),
            TensorStatistics(tensor_id=3, name="tensor3", size_bytes=512, load_time_ms=5.0),
            TensorStatistics(tensor_id=4, name="tensor4", size_bytes=4096, load_time_ms=30.0),
            TensorStatistics(tensor_id=5, name="tensor5", size_bytes=256, load_time_ms=2.0),
        ]

    @pytest.fixture
    def sample_layer_stats(self, sample_tensor_stats):
        """Create sample layer statistics for testing."""
        return [
            LayerStatistics(
                label="layer1",
                tensors=[sample_tensor_stats[0], sample_tensor_stats[1]],  # 2 tensors, 3072 bytes, 30ms
                duration=30.0,
            ),
            LayerStatistics(
                label="layer2",
                tensors=[sample_tensor_stats[2]],  # 1 tensor, 512 bytes, 5ms
                duration=5.0,
            ),
            LayerStatistics(
                label="layer3",
                tensors=[sample_tensor_stats[3]],  # 1 tensor, 4096 bytes, 30ms
                duration=30.0,
            ),
            LayerStatistics(
                label="layer4",
                tensors=[sample_tensor_stats[4]],  # 1 tensor, 256 bytes, 2ms
                duration=2.0,
            ),
            LayerStatistics(
                label="layer5",
                tensors=[
                    sample_tensor_stats[0],
                    sample_tensor_stats[1],
                    sample_tensor_stats[2],
                ],  # 3 tensors, 3584 bytes, 35ms
                duration=35.0,
            ),
        ]

    @pytest.fixture
    def sample_strategy_map(self, sample_layer_stats):
        """Create sample strategy map for testing."""
        return {layer_stat.label: layer_stat.tensors for layer_stat in sample_layer_stats}

    def test_remove_largest_by_tensor_count(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers with largest tensor counts."""
        operations = [{"type": "largest", "n": 2}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # layer5 has 3 tensors (largest), layer1 has 2 tensors (second largest)
        expected_remaining = {"layer2", "layer3", "layer4"}
        assert set(result.keys()) == expected_remaining

    def test_remove_by_duration_ascending(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers with shortest duration."""
        operations = [{"type": "by_duration", "n": 2, "order": "asc"}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # layer4 has 2ms (shortest), layer2 has 5ms (second shortest)
        expected_remaining = {"layer1", "layer3", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_by_duration_descending(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers with longest duration."""
        operations = [{"type": "by_duration", "n": 2, "order": "desc"}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # layer5 has 35ms (longest), layer1 and layer3 both have 30ms (tie for second)
        # The function picks layer5 and one of the tied layers (layer1)
        expected_remaining = {"layer2", "layer3", "layer4"}
        assert set(result.keys()) == expected_remaining

    def test_remove_by_size_ascending(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers with smallest memory size."""
        operations = [{"type": "by_size", "n": 2, "order": "asc"}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        expected_remaining = {"layer1", "layer3", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_by_size_descending(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers with largest memory size."""
        operations = [{"type": "by_size", "n": 2, "order": "desc"}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        expected_remaining = {"layer1", "layer2", "layer4"}
        assert set(result.keys()) == expected_remaining

    def test_remove_first_n_layers(self, sample_strategy_map, sample_layer_stats):
        """Test removing first n layers."""
        operations = [{"type": "first_n", "n": 2}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove layer1 and layer2 (first 2 layers)
        expected_remaining = {"layer3", "layer4", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_last_n_layers(self, sample_strategy_map, sample_layer_stats):
        """Test removing last n layers."""
        operations = [{"type": "last_n", "n": 2}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove layer4 and layer5 (last 2 layers)
        expected_remaining = {"layer1", "layer2", "layer3"}
        assert set(result.keys()) == expected_remaining

    def test_remove_every_nth_layer(self, sample_strategy_map, sample_layer_stats):
        """Test removing every nth layer."""
        operations = [{"type": "every_nth", "n": 2, "offset": 0}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove every 2nd layer starting from offset 0: layer1, layer3, layer5
        expected_remaining = {"layer2", "layer4"}
        assert set(result.keys()) == expected_remaining

    def test_remove_every_nth_with_offset(self, sample_strategy_map, sample_layer_stats):
        """Test removing every nth layer with offset."""
        operations = [{"type": "every_nth", "n": 2, "offset": 1}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove every 2nd layer starting from offset 1: layer2, layer4
        expected_remaining = {"layer1", "layer3", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_consecutive_layers_with_range(self, sample_strategy_map, sample_layer_stats):
        """Test removing consecutive layers using range operation."""
        operations = [{"type": "range", "n": 3, "offset": 1}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove 3 consecutive layers starting from offset 1: layer2, layer3, layer4
        expected_remaining = {"layer1", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_range_layers(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers in a range."""
        operations = [{"type": "range", "n": 2, "offset": 1}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove 2 layers starting from offset 1: layer2, layer3
        expected_remaining = {"layer1", "layer4", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_single_layer(self, sample_strategy_map, sample_layer_stats):
        """Test removing a single layer by position."""
        operations = [{"type": "single", "n": 2}]  # n=2 means position 2 (layer3)

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove layer at position 2 (layer3)
        expected_remaining = {"layer1", "layer2", "layer4", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_specific_layers_by_name(self, sample_strategy_map, sample_layer_stats):
        """Test removing specific layers by name."""
        operations = [{"type": "names", "values": ["layer2", "layer4"]}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove layer2 and layer4
        expected_remaining = {"layer1", "layer3", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_specific_layers_by_index(self, sample_strategy_map, sample_layer_stats):
        """Test removing specific layers by index."""
        operations = [{"type": "indices", "values": [1, 3]}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove layers at indices 1 and 3 (layer2 and layer4)
        expected_remaining = {"layer1", "layer3", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_layers_accepts_integer_subclasses(self, sample_strategy_map, sample_layer_stats):
        """Integer-like indices follow Python's index protocol."""

        class LayerIndex(IntEnum):
            SECOND = 1

        result = remove_layers_compound(
            sample_strategy_map,
            sample_layer_stats,
            [{"type": "indices", "values": [LayerIndex.SECOND]}],
        )

        assert set(result) == {"layer1", "layer3", "layer4", "layer5"}

    def test_remove_layers_rejects_boolean_indices(self, sample_strategy_map, sample_layer_stats):
        """Booleans are flags, not layer indices."""
        with pytest.raises(ValueError, match="list of integer values"):
            remove_layers_compound(
                sample_strategy_map,
                sample_layer_stats,
                [{"type": "indices", "values": [True]}],
            )

    def test_remove_specific_layers_nonexistent(self, sample_strategy_map, sample_layer_stats):
        """Test removing specific layers that don't exist in strategy map."""
        operations = [{"type": "names", "values": ["nonexistent_layer", "layer2"]}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Only layer2 should be removed, nonexistent_layer should be ignored
        expected_remaining = {"layer1", "layer3", "layer4", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_specific_layers_invalid_index(self, sample_strategy_map, sample_layer_stats):
        """Test removing specific layers with invalid indices."""
        operations = [{"type": "indices", "values": [1, 10, -1]}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Only valid index 1 (layer2) should be removed
        expected_remaining = {"layer1", "layer3", "layer4", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_specific_operation(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers using the 'specific' operation type with names."""
        operations = [{"type": "names", "values": ["layer2", "layer4"]}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove layer2 and layer4
        expected_remaining = {"layer1", "layer3", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_layer_indices_operation(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers using the new 'layer_indices' operation type."""
        operations = [{"type": "indices", "values": [1, 3]}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove layers at indices 1 and 3 (layer2 and layer4)
        expected_remaining = {"layer1", "layer3", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_specific_nonexistent(self, sample_strategy_map, sample_layer_stats):
        """Test removing layer names that don't exist in strategy map."""
        operations = [{"type": "names", "values": ["nonexistent_layer", "layer2"]}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Only layer2 should be removed, nonexistent_layer should be ignored
        expected_remaining = {"layer1", "layer3", "layer4", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_remove_layer_indices_invalid(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers with invalid indices."""
        operations = [{"type": "indices", "values": [1, 10, -1]}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Only valid index 1 (layer2) should be removed
        expected_remaining = {"layer1", "layer3", "layer4", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_mixed_new_operation_types(self, sample_strategy_map, sample_layer_stats):
        """Test using both new operation types in separate operations."""
        operations = [
            {"type": "names", "values": ["layer2"]},
            {"type": "indices", "values": [3]},
        ]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove layer2 and layer4
        expected_remaining = {"layer1", "layer3", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_complex_compound_operations(self, sample_strategy_map, sample_layer_stats):
        """Test complex compound operations with multiple steps."""
        operations = [
            {"type": "every_nth", "n": 2, "offset": 0},  # Remove layer1, layer3, layer5
            {"type": "by_duration", "n": 1, "order": "asc"},  # Remove shortest remaining (layer4)
        ]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # After first operation: layer2, layer4 remain
        # After second operation: layer2 remains (layer4 has shorter duration: 2ms vs 5ms)
        expected_remaining = {"layer2"}
        assert set(result.keys()) == expected_remaining

    def test_multiple_operations_same_type(self, sample_strategy_map, sample_layer_stats):
        """Test multiple operations of the same type."""
        operations = [
            {"type": "largest", "n": 1},  # Remove layer5 (3 tensors)
            {"type": "largest", "n": 1},  # Remove layer1 (2 tensors)
        ]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        expected_remaining = {"layer2", "layer3", "layer4"}
        assert set(result.keys()) == expected_remaining

    def test_empty_operations_list(self, sample_strategy_map, sample_layer_stats):
        """Test with empty operations list."""
        operations = []

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Should return unchanged strategy map
        assert result == sample_strategy_map
        assert result is not sample_strategy_map  # Should be a copy

    def test_empty_strategy_map(self, sample_layer_stats):
        """Test with empty strategy map."""
        strategy_map = {}
        operations = [{"type": "largest", "n": 1}]

        result = remove_layers_compound(strategy_map, sample_layer_stats, operations)

        assert result == {}

    def test_nonexistent_layers_ignored(self, sample_strategy_map, sample_layer_stats):
        """Test that nonexistent layers are ignored silently."""
        # Add a layer to layer_stats that's not in strategy_map
        extra_layer = LayerStatistics(
            label="nonexistent_layer",
            tensors=[],
            duration=10.0,
        )
        extended_layer_stats = [*sample_layer_stats, extra_layer]

        operations = [{"type": "first_n", "n": 6}]  # Try to remove 6 layers (only 5 exist)

        result = remove_layers_compound(sample_strategy_map, extended_layer_stats, operations)

        # Should remove all existing layers
        assert result == {}

    def test_invalid_operation_type(self, sample_strategy_map, sample_layer_stats):
        """Test that invalid operation type raises ValueError."""
        operations = [{"type": "invalid_type", "n": 1}]

        with pytest.raises(ValueError, match="Invalid operation type: invalid_type"):
            remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

    def test_invalid_n_parameter(self, sample_strategy_map, sample_layer_stats):
        """Test that invalid n parameter raises ValueError."""
        operations = [{"type": "largest", "n": 0}]

        with pytest.raises(ValueError, match="Number of layers n must be at least 1"):
            remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

    def test_invalid_offset_parameter(self, sample_strategy_map, sample_layer_stats):
        """Test that invalid offset parameter raises ValueError."""
        operations = [{"type": "range", "n": 1, "offset": -1}]

        with pytest.raises(ValueError, match="Offset must be non-negative"):
            remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

    def test_remove_more_layers_than_exist(self, sample_strategy_map, sample_layer_stats):
        """Test removing more layers than exist."""
        operations = [{"type": "largest", "n": 10}]  # Try to remove 10 layers, only 5 exist

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Should remove all layers
        assert result == {}

    def test_remove_layers_out_of_range(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers with operations that go out of range."""
        operations = [{"type": "range", "n": 10, "offset": 3}]  # Try to remove 10 layers starting from position 3

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Should only remove layers 4 and 5 (positions 3 and 4)
        expected_remaining = {"layer1", "layer2", "layer3"}
        assert set(result.keys()) == expected_remaining

    def test_step_parameter_for_every_nth(self, sample_strategy_map, sample_layer_stats):
        """Test step parameter for every_nth operation."""
        operations = [{"type": "every_nth", "step": 3, "offset": 0}]  # Every 3rd layer

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Remove every 3rd layer: layer1, layer4
        expected_remaining = {"layer2", "layer3", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_default_order_parameter(self, sample_strategy_map, sample_layer_stats):
        """Test default order parameter for sorting operations."""
        operations = [{"type": "by_duration", "n": 2}]  # No order specified, should default to "desc"

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Should remove longest durations (layer5=35ms, layer1=30ms)
        expected_remaining = {"layer2", "layer3", "layer4"}
        assert set(result.keys()) == expected_remaining

    def test_smallest_tensor_count(self, sample_strategy_map, sample_layer_stats):
        """Test removing layers with smallest tensor counts."""
        operations = [{"type": "smallest", "n": 2}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # layer2, layer3, layer4 all have 1 tensor (smallest)
        # Should remove 2 of them (layer2 and layer3)
        expected_remaining = {"layer1", "layer4", "layer5"}
        assert set(result.keys()) == expected_remaining

    def test_preserve_original_strategy_map(self, sample_strategy_map, sample_layer_stats):
        """Test that original strategy map is not modified."""
        original_strategy_map = sample_strategy_map.copy()
        operations = [{"type": "largest", "n": 1}]

        result = remove_layers_compound(sample_strategy_map, sample_layer_stats, operations)

        # Original should be unchanged
        assert sample_strategy_map == original_strategy_map
        # Result should be different
        assert result != sample_strategy_map

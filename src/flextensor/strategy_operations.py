# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Strategy operations module for manipulating strategy maps and layer statistics.

This module provides various functions for removing layers, filling gaps,
and performing compound operations on strategy maps.
"""

import operator
from typing import Any

from flextensor.collectors import LayerStatistics, TensorStatistics


def get_layers_by_position(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    position: int,
) -> list[str]:
    """
    Get layers to remove by position in the layer sequence.

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order
        position: Position (0-based) of the layer to remove

    Returns:
        List of layer names to remove (empty if position is out of range or layer not in strategy_map)
    """
    if position >= len(layer_stats):
        return []  # Skip if out of range
    layer_name = layer_stats[position].label
    return [layer_name] if layer_name in strategy_map else []


def get_layers_by_names(strategy_map: dict[str, list[TensorStatistics]], names: list[str]) -> list[str]:
    """
    Get layers to remove by specific layer names.

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        names: List of layer names to remove

    Returns:
        List of layer names to remove (only those that exist in strategy_map)
    """
    layers_to_remove = []
    for layer_name in names:
        if layer_name in strategy_map:
            layers_to_remove.append(layer_name)
    return layers_to_remove


def get_layers_by_indices(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    indices: list[int],
) -> list[str]:
    """
    Get layers to remove by specific layer indices.

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order
        indices: List of layer indices to remove (0-based)

    Returns:
        List of layer names to remove (only those that exist in strategy_map)
    """
    layers_to_remove = []
    for idx in indices:
        if 0 <= idx < len(layer_stats):
            layer_name = layer_stats[idx].label
            if layer_name in strategy_map:
                layers_to_remove.append(layer_name)
    return layers_to_remove


def get_layers_by_every_nth(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    step: int,
    offset: int,
) -> list[str]:
    """
    Get layers to remove by selecting every nth layer starting from offset.

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order
        step: Step size (e.g., 2 for every 2nd layer)
        offset: Starting offset (0-based)

    Returns:
        List of layer names to remove
    """
    layers_to_remove = []
    for i in range(offset, len(layer_stats), step):
        layer_name = layer_stats[i].label
        if layer_name in strategy_map:
            layers_to_remove.append(layer_name)
    return layers_to_remove


def get_layers_by_range(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    count: int,
    offset: int,
) -> list[str]:
    """
    Get layers to remove in a specific range [offset, offset + count).

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order
        count: Number of layers to remove
        offset: Starting offset (0-based)

    Returns:
        List of layer names to remove
    """
    layers_to_remove = []
    for i in range(offset, min(offset + count, len(layer_stats))):
        layer_name = layer_stats[i].label
        if layer_name in strategy_map:
            layers_to_remove.append(layer_name)
    return layers_to_remove


def get_layers_by_first_n(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    count: int,
) -> list[str]:
    """
    Get the first n layers to remove.

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order
        count: Number of first layers to remove

    Returns:
        List of layer names to remove
    """
    layers_to_remove = []
    for i in range(min(count, len(layer_stats))):
        layer_name = layer_stats[i].label
        if layer_name in strategy_map:
            layers_to_remove.append(layer_name)
    return layers_to_remove


def get_layers_by_last_n(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    count: int,
) -> list[str]:
    """
    Get the last n layers to remove.

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order
        count: Number of last layers to remove

    Returns:
        List of layer names to remove
    """
    layers_to_remove = []
    start_idx = max(0, len(layer_stats) - count)
    for i in range(start_idx, len(layer_stats)):
        layer_name = layer_stats[i].label
        if layer_name in strategy_map:
            layers_to_remove.append(layer_name)
    return layers_to_remove


def get_layers_by_tensor_count(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    count: int,
    largest: bool = True,
) -> list[str]:
    """
    Get layers to remove based on tensor count (largest or smallest).

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order
        count: Number of layers to remove
        largest: If True, remove layers with largest tensor counts; if False, smallest

    Returns:
        List of layer names to remove
    """
    layer_tensor_counts = []
    for i, layer_stat in enumerate(layer_stats):
        layer_name = layer_stat.label
        if layer_name in strategy_map:
            tensor_count = len(layer_stat.tensors) if hasattr(layer_stat, "tensors") else 0
            layer_tensor_counts.append((tensor_count, i, layer_name))

    # Sort by tensor count (descending for largest, ascending for smallest)
    layer_tensor_counts.sort(key=lambda x: x[0], reverse=largest)

    # Take first n layers
    return [layer_name for _, _, layer_name in layer_tensor_counts[:count]]


def get_layers_by_duration(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    count: int,
    longest: bool = True,
) -> list[str]:
    """
    Get layers to remove based on duration (longest or shortest).

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order
        count: Number of layers to remove
        longest: If True, remove layers with longest duration; if False, shortest

    Returns:
        List of layer names to remove
    """
    layer_durations = []
    for i, layer_stat in enumerate(layer_stats):
        layer_name = layer_stat.label
        if layer_name in strategy_map:
            duration = getattr(layer_stat, "duration", 0)
            layer_durations.append((duration, i, layer_name))

    # Sort by duration
    layer_durations.sort(key=lambda x: x[0], reverse=longest)

    # Take first n layers
    return [layer_name for _, _, layer_name in layer_durations[:count]]


def get_layers_by_memory_size(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    count: int,
    largest: bool = True,
) -> list[str]:
    """
    Get layers to remove based on estimated memory size (largest or smallest).

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order
        count: Number of layers to remove
        largest: If True, remove layers with largest memory size; if False, smallest

    Returns:
        List of layer names to remove
    """
    layer_sizes = []
    for i, layer_stat in enumerate(layer_stats):
        layer_name = layer_stat.label
        if layer_name in strategy_map:
            # Calculate total memory size from tensors
            total_size = 0
            if hasattr(layer_stat, "tensors"):
                for tensor in layer_stat.tensors:
                    total_size += tensor.size_bytes
            layer_sizes.append((total_size, i, layer_name))

    # Sort by size
    layer_sizes.sort(key=lambda x: x[0], reverse=largest)

    # Take first n layers
    return [layer_name for _, _, layer_name in layer_sizes[:count]]


def remove_layers_compound(  # noqa: C901
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    operations: list[dict[str, Any]],
) -> dict[str, list[TensorStatistics]]:
    """
    Apply multiple removal operations in sequence to strategy_map.

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        layer_stats: List of LayerStatistics that determines the order for removal
        operations: List of operation dictionaries, each containing:
            - "type": "single", "names", "indices", "every_nth", "range", "first_n", "last_n",
                      "largest", "smallest", "by_duration", "by_size"
            - "n": Number of layers to remove
            - "offset": Starting offset (0-based, optional for some types)
            - "step": Step size for "every_nth" (optional, overrides n)
            - "order": "asc" or "desc" for sorting operations (optional)
            - "values": List of layer names to remove (for "by_names" type) or indices (for "by_indices" type)

    Returns:
        New strategy_map with all operations applied in sequence.
        Missing layers are ignored silently.

    Raises:
        ValueError: If parameters are invalid or operation type is not recognized
    """
    if not strategy_map:
        return strategy_map.copy()

    # Create a copy of the strategy_map
    new_strategy_map = strategy_map.copy()

    for op in operations:
        op_type = op.get("type", "single")
        n = op.get("n", 1)
        offset = op.get("offset", 0)
        step = op.get("step", n)
        order = op.get("order", "desc")  # "desc" for largest/longest, "asc" for smallest/shortest

        if op_type not in [
            "single",
            "names",
            "indices",
            "every_nth",
            "range",
            "first_n",
            "last_n",
            "largest",
            "smallest",
            "by_duration",
            "by_size",
        ]:
            msg = f"Invalid operation type: {op_type}."
            raise ValueError(
                msg,
            )

        if n < 1:
            msg = f"Number of layers n must be at least 1, got {n}"
            raise ValueError(msg)
        if offset < 0:
            msg = f"Offset must be non-negative, got {offset}"
            raise ValueError(msg)

        # Get layers to remove based on operation type
        if op_type == "single":
            layers_to_remove = get_layers_by_position(new_strategy_map, layer_stats, n)
        elif op_type == "names":
            names = op.get("values")
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise ValueError("names operation requires a list of string values")
            layers_to_remove = get_layers_by_names(new_strategy_map, names)
        elif op_type == "indices":
            indices = op.get("values")
            if not isinstance(indices, list) or any(isinstance(index, bool) for index in indices):
                raise ValueError("indices operation requires a list of integer values")
            try:
                indices = [operator.index(index) for index in indices]
            except TypeError as error:
                raise ValueError("indices operation requires a list of integer values") from error
            layers_to_remove = get_layers_by_indices(new_strategy_map, layer_stats, indices)
        elif op_type == "every_nth":
            layers_to_remove = get_layers_by_every_nth(new_strategy_map, layer_stats, step, offset)
        elif op_type == "range":
            layers_to_remove = get_layers_by_range(new_strategy_map, layer_stats, n, offset)
        elif op_type == "first_n":
            layers_to_remove = get_layers_by_first_n(new_strategy_map, layer_stats, n)
        elif op_type == "last_n":
            layers_to_remove = get_layers_by_last_n(new_strategy_map, layer_stats, n)
        elif op_type == "largest":
            layers_to_remove = get_layers_by_tensor_count(new_strategy_map, layer_stats, n, largest=True)
        elif op_type == "smallest":
            layers_to_remove = get_layers_by_tensor_count(new_strategy_map, layer_stats, n, largest=False)
        elif op_type == "by_duration":
            layers_to_remove = get_layers_by_duration(new_strategy_map, layer_stats, n, longest=(order == "desc"))
        elif op_type == "by_size":
            layers_to_remove = get_layers_by_memory_size(new_strategy_map, layer_stats, n, largest=(order == "desc"))

        # Remove all identified layers for this operation
        for layer_name in layers_to_remove:
            if layer_name in new_strategy_map:
                del new_strategy_map[layer_name]

    return new_strategy_map


def get_layer_name_to_position(layer_stats: list[LayerStatistics]) -> dict[str, int]:
    """
    Create a mapping from layer names to their positions in the layer sequence.

    Args:
        layer_stats: List of LayerStatistics that determines the layer order

    Returns:
        Dictionary mapping layer names to their positions (0-based index)
    """
    return {layer_stat.label: i for i, layer_stat in enumerate(layer_stats)}


def get_position_to_layer_name(layer_stats: list[LayerStatistics]) -> dict[int, str]:
    """
    Create a mapping from positions to layer names in the layer sequence.

    Args:
        layer_stats: List of LayerStatistics that determines the layer order

    Returns:
        Dictionary mapping positions (0-based index) to layer names
    """
    return {i: layer_stat.label for i, layer_stat in enumerate(layer_stats)}


def calculate_transfer_positions(
    transfer_to_compute_map: dict[str, str], layer_stats: list[LayerStatistics]
) -> tuple[set[int], set[int]]:
    """
    Calculate the positions of the transfers and the available positions.
    """
    layer_name_to_position = get_layer_name_to_position(layer_stats)
    transfer_positions = {
        layer_name_to_position[transfer] for transfer in transfer_to_compute_map if transfer in layer_name_to_position
    }
    available_positions = set(range(len(layer_stats))) - transfer_positions
    return transfer_positions, available_positions


def calculate_compute_position_by_block_id(
    label_to_block_id: dict[str, int], transfer_to_compute_map: dict[str, str]
) -> dict[int, list[tuple[str, str, int]]]:
    compute_by_block: dict[int, list[tuple[str, str, int]]] = {}
    for transfer_layer, compute_layer in transfer_to_compute_map.items():
        block_id = label_to_block_id.get(transfer_layer)
        if block_id is not None:
            if block_id not in compute_by_block:
                compute_by_block[block_id] = []
            compute_by_block[block_id].append((transfer_layer, compute_layer, block_id))

    return compute_by_block


def find_previous_compute_position_from_transfer(
    transfer_layer_name: str,
    block_id: int,
    transfer_to_compute_map: dict[str, str],
    label_to_block_id: dict[str, int],
    layer_name_to_position: dict[str, int],
) -> int:
    """
    Find the previous compute layer position for the given layer name from the transfer.
    If there is no previous compute layer, return -1.
    """
    compute_by_block = calculate_compute_position_by_block_id(label_to_block_id, transfer_to_compute_map)
    compute_by_current_block = compute_by_block[block_id]
    layer_positions = []
    for _transfer_layer, compute_layer, _block_id in compute_by_current_block:
        layer_positions.append(layer_name_to_position[compute_layer])

    transfer_layer_position = layer_name_to_position[transfer_layer_name]
    layer_positions.append(transfer_layer_position)
    layer_positions.sort()
    transfer_layer_position_index = layer_positions.index(transfer_layer_position)

    previous_index = -1
    if transfer_layer_position_index > 0:
        previous_index = layer_positions[transfer_layer_position_index - 1]
    return previous_index


def check_transfer_constraint(
    optimized_map: dict[str, str],
    optimized_block_map: dict[str, int],
    layer_stats: list[LayerStatistics],
) -> bool:
    """
    Check whether the transfer constraint is satisfied (silent version).

    The constraint is: transfers cannot be moved before the previous compute layer
    from the previous transfer within the same block ID.

    Args:
        optimized_map: Dictionary mapping transfer layer names to compute layer names
        optimized_block_map: Dictionary mapping transfer layer names to block IDs
        layer_stats: List of LayerStatistics that determines the layer order

    Returns:
        bool - True if constraint is satisfied, False if violated
    """

    layer_name_to_position = get_layer_name_to_position(layer_stats)
    violations = []

    for transfer_layer, compute_layer in optimized_map.items():
        # Find the position of the transfer layer
        transfer_pos = layer_name_to_position.get(transfer_layer, -1)
        if transfer_pos == -1:
            continue

        # Find the previous compute position for this transfer's block
        block_id = optimized_block_map.get(transfer_layer)
        if block_id is not None:
            prev_compute_pos = find_previous_compute_position_from_transfer(
                transfer_layer,
                block_id,
                optimized_map,
                optimized_block_map,
                layer_name_to_position,
            )

            if prev_compute_pos == -1:
                # No previous compute in this block, check against compute position
                compute_pos = layer_name_to_position.get(compute_layer, -1)
                if compute_pos == -1:
                    continue
                # No constraint violation if no previous compute in block
            # Check against previous compute position within the same block
            elif transfer_pos < prev_compute_pos:
                violations.append(True)

    return len(violations) == 0


def check_overlap_constraint(  # noqa: C901
    transfer_to_compute_map: dict[str, str],
    layer_stats: list[LayerStatistics],
    label_to_block_id: dict[str, int],
) -> bool:
    """
    Check for overlapping execution between compute and transfer operations within the same block_id (silent version).

    This constraint ensures that within each block_id:
    1. No transfer and compute operations occur at the same layer position
    2. No execution ranges of transfer-compute pairs overlap with each other

    Args:
        transfer_to_compute_map: Dictionary mapping transfer layer names to compute layer names
        layer_stats: List of LayerStatistics that determines the layer order
        label_to_block_id: Dictionary mapping layer names to block IDs

    Returns:
        bool - True if constraint is satisfied (no overlaps), False if violated
    """
    layer_name_to_position = get_layer_name_to_position(layer_stats)
    overlap_violations = []

    # Group transfers by block_id
    transfers_by_block: dict[int, list[tuple[str, str]]] = {}
    for transfer_layer, compute_layer in transfer_to_compute_map.items():
        block_id = label_to_block_id.get(transfer_layer)
        if block_id is not None:
            if block_id not in transfers_by_block:
                transfers_by_block[block_id] = []
            transfers_by_block[block_id].append((transfer_layer, compute_layer))
        else:
            continue  # Skip transfers without block ID

    # Check for overlaps within each block
    for _block_id, block_transfers in transfers_by_block.items():
        # Collect transfer-compute pairs with their positions
        transfer_compute_pairs = []

        for transfer_layer, compute_layer in block_transfers:
            transfer_pos = layer_name_to_position.get(transfer_layer, -1)
            compute_pos = layer_name_to_position.get(compute_layer, -1)

            if transfer_pos == -1 or compute_pos == -1:
                continue  # Skip pairs with missing positions

            transfer_compute_pairs.append((transfer_layer, compute_layer, transfer_pos, compute_pos))

        # Check for position overlaps (exact same position)
        all_positions: dict[int, list[tuple[str, str]]] = {}  # position -> list of (type, layer_name)

        for transfer_layer, compute_layer, transfer_pos, compute_pos in transfer_compute_pairs:
            # Track positions
            if transfer_pos not in all_positions:
                all_positions[transfer_pos] = []
            all_positions[transfer_pos].append(("transfer", transfer_layer))

            if compute_pos not in all_positions:
                all_positions[compute_pos] = []
            all_positions[compute_pos].append(("compute", compute_layer))

        # Check for position overlaps
        for _pos, operations in all_positions.items():
            transfers_at_pos = [op for op in operations if op[0] == "transfer"]
            computes_at_pos = [op for op in operations if op[0] == "compute"]

            if transfers_at_pos and computes_at_pos:
                # Position overlap found
                overlap_violations.append(True)

        # Check for range overlaps (execution ranges that overlap)
        for i, (_transfer1, _compute1, transfer_pos1, compute_pos1) in enumerate(transfer_compute_pairs):
            for j, (_transfer2, _compute2, transfer_pos2, compute_pos2) in enumerate(transfer_compute_pairs):
                if i >= j:  # Avoid checking the same pair twice
                    continue

                # Create execution ranges (transfer to compute position)
                # Handle cyclic cases where compute position < transfer position
                if compute_pos1 < transfer_pos1:
                    # Cyclic case: range wraps around from transfer_pos1 to end, then from start to compute_pos1
                    range1_positions = set(range(transfer_pos1, len(layer_stats))) | set(range(compute_pos1 + 1))
                else:
                    # Normal case: range from transfer_pos1 to compute_pos1
                    range1_positions = set(range(transfer_pos1, compute_pos1 + 1))

                if compute_pos2 < transfer_pos2:
                    # Cyclic case: range wraps around from transfer_pos2 to end, then from start to compute_pos2
                    range2_positions = set(range(transfer_pos2, len(layer_stats))) | set(range(compute_pos2 + 1))
                else:
                    # Normal case: range from transfer_pos2 to compute_pos2
                    range2_positions = set(range(transfer_pos2, compute_pos2 + 1))

                # Check if ranges overlap by checking for intersection
                if range1_positions & range2_positions:
                    overlap_violations.append(True)

    return len(overlap_violations) == 0


def rearrange_transfers(  # noqa: C901
    transfer_to_compute_map: dict[str, str],
    label_to_block_id: dict[str, int],
    layer_stats: list[LayerStatistics],
    min_compute_transfer_gap: int = 1,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """
    Move transfers as early as possible in the execution order to create more time between transfers and
    compute operations.

    This function finds available slots earlier in the execution sequence and moves transfer operations there,
    while making sure that data is still available when the compute operation needs it.

    Args:
        transfer_to_compute_map: Dictionary mapping transfer layer names to compute layer names
        label_to_block_id: Dictionary mapping layer names to block IDs
        layer_stats: List of LayerStatistics that determines the layer order
        min_compute_transfer_gap: Minimum gap between compute and transfer layers (default is 1)

    Returns:
        Tuple of (rearranged_transfer_map, rearranged_block_map, remapped_layers)
            where ``remapped_layers`` maps original transfer labels to their new
            labels after rearrangement.  Empty when no moves are made.
    """
    if not transfer_to_compute_map or not label_to_block_id or not layer_stats:
        return transfer_to_compute_map.copy(), label_to_block_id.copy(), {}

    remapped_layers = {}
    # Create position mapping for layers
    layer_name_to_position = get_layer_name_to_position(layer_stats)
    position_to_layer_name = get_position_to_layer_name(layer_stats)

    transfer_positions = {
        layer_name_to_position[transfer] for transfer in transfer_to_compute_map if transfer in layer_name_to_position
    }
    available_positions = set(range(len(layer_stats))) - transfer_positions

    transfers_by_block: dict[int, list[tuple[str, str, int]]] = {}
    for transfer_layer, compute_layer in transfer_to_compute_map.items():
        block_id = label_to_block_id.get(transfer_layer)
        if block_id is not None:
            if block_id not in transfers_by_block:
                transfers_by_block[block_id] = []
            transfers_by_block[block_id].append((transfer_layer, compute_layer, block_id))
    transfer_order_by_block = []
    for block_id in sorted(transfers_by_block.keys()):
        transfers = transfers_by_block[block_id]
        for transfer_layer, compute_layer, transfer_block_id in transfers:
            transfer_order_by_block.append((transfer_layer, compute_layer, transfer_block_id))

    transfer_by_order = []
    for transfer_layer, compute_layer in transfer_to_compute_map.items():
        block_id = label_to_block_id.get(transfer_layer)
        if block_id is not None:
            transfer_by_order.append((transfer_layer, compute_layer, block_id))

    transfer_order = transfer_by_order
    prev_compute_position = -1

    new_transfer_to_compute_map = transfer_to_compute_map.copy()
    new_label_to_block_id = label_to_block_id.copy()

    for transfer_layer, compute_layer, block_id in transfer_order:
        transfer_positions, available_positions = calculate_transfer_positions(new_transfer_to_compute_map, layer_stats)
        prev_compute_position = find_previous_compute_position_from_transfer(
            transfer_layer,
            block_id,
            new_transfer_to_compute_map,
            new_label_to_block_id,
            layer_name_to_position,
        )

        transfer_position = layer_name_to_position[transfer_layer]
        compute_position = layer_name_to_position[compute_layer]

        if prev_compute_position == -1:
            local_available_free_slots = available_positions & set(range(compute_position))
        else:
            lower_compute_position = prev_compute_position + min_compute_transfer_gap
            higher_compute_position = compute_position
            if higher_compute_position < lower_compute_position:  # cyclic case
                set_lower = set(range(higher_compute_position))
                set_higher = set(range(lower_compute_position, len(layer_stats)))
                set_range = set_lower | set_higher
                local_available_free_slots = available_positions & set_range
            else:  # normal case
                set_range = set(range(lower_compute_position, higher_compute_position))
                local_available_free_slots = available_positions & set_range
        # find which available position have largest gap from the compute position
        largest_gap = -1
        largest_gap_slot = -1
        local_available_free_slots.add(transfer_position)
        for slot in local_available_free_slots:
            if slot < compute_position:
                gap = compute_position - slot
                if gap > largest_gap:
                    largest_gap = gap
                    largest_gap_slot = slot
            else:
                gap = len(layer_stats) - slot + compute_position
                if gap > largest_gap:
                    largest_gap = gap
                    largest_gap_slot = slot
        slot_layer_name = position_to_layer_name.get(largest_gap_slot)

        if slot_layer_name is not None:
            del new_transfer_to_compute_map[transfer_layer]
            del new_label_to_block_id[transfer_layer]

            new_transfer_to_compute_map[slot_layer_name] = compute_layer
            new_label_to_block_id[slot_layer_name] = block_id
            remapped_layers[transfer_layer] = slot_layer_name

    # Validate constraints after rearrangement (silent validation)
    transfer_constraint_satisfied = check_transfer_constraint(
        new_transfer_to_compute_map,
        new_label_to_block_id,
        layer_stats,
    )

    overlap_constraint_satisfied = check_overlap_constraint(
        new_transfer_to_compute_map,
        layer_stats,
        new_label_to_block_id,
    )

    # Throw exceptions if constraints are violated
    if not transfer_constraint_satisfied:
        msg = "Transfer constraint violated after optimization."
        raise ValueError(msg)

    if not overlap_constraint_satisfied:
        msg = "Overlap constraint violated after optimization."
        raise ValueError(msg)

    return new_transfer_to_compute_map, new_label_to_block_id, remapped_layers


def find_transfers_for_preload(
    transfer_to_compute_map: dict[str, str],
    layer_stats: list[LayerStatistics],
) -> dict[str, str]:
    """
    Find transfers that need to be preloaded. This function identifies transfer operations that occur
    after their corresponding compute operations in the execution sequence, which require preloading
    to ensure data is available when needed.

    Args:
        transfer_to_compute_map: Dictionary mapping transfer layer names to compute layer names
        layer_stats: List of LayerStatistics that determines the layer order

    Returns:
        Dictionary containing only transfers that happen after their corresponding compute operations
    """
    if not transfer_to_compute_map or not layer_stats:
        return {}

    # Get layer name to position mapping
    layer_name_to_position = get_layer_name_to_position(layer_stats)

    filtered_transfers = {}

    for transfer_layer, compute_layer in transfer_to_compute_map.items():
        # Get positions for transfer and compute layers
        transfer_pos = layer_name_to_position.get(transfer_layer, -1)
        compute_pos = layer_name_to_position.get(compute_layer, -1)

        # Skip if either layer not found in layer_stats
        if transfer_pos == -1 or compute_pos == -1:
            continue

        # Check if transfer happens after compute
        if compute_pos < transfer_pos:
            filtered_transfers[transfer_layer] = compute_layer
    return filtered_transfers


def remap_strategy(
    strategy_map: dict[str, list[TensorStatistics]] | None,
    key_remap: dict[str, str],
) -> dict[str, list[TensorStatistics]]:
    """
    Remap keys in a strategy map using a provided key mapping dictionary.

    This function creates a new strategy map where the keys are remapped according to
    the provided key_remap dictionary. Keys that don't exist in the remap dictionary
    are kept unchanged.

    Args:
        strategy_map: Dictionary mapping layer names to lists of TensorStatistics
        key_remap: Dictionary mapping old keys to new keys

    Returns:
        New strategy map with remapped keys

    Example:
        Input strategy_map: {'layer.0': [tensor_stats], 'layer.1': [tensor_stats]}
        Input key_remap: {'layer.0': 'new_layer.0', 'layer.2': 'new_layer.2'}
        Output: {'new_layer.0': [tensor_stats], 'layer.1': [tensor_stats]}
    """
    if strategy_map is None or not strategy_map:
        return {} if strategy_map is None else strategy_map.copy()

    remapped_strategy = {}
    for old_key, tensor_stats in strategy_map.items():
        remapped_strategy[old_key] = tensor_stats

    for old_key, new_key in key_remap.items():
        if old_key in strategy_map:
            # Move the value from old key to new key
            remapped_strategy[new_key] = strategy_map[old_key]
            # Only remove the old key if it's different from the new key
            if old_key != new_key:
                del remapped_strategy[old_key]

    return remapped_strategy


def create_allocation_ordered(
    label_to_block_id: dict[str, int],
    layer_stats: list[LayerStatistics],
) -> dict[int, list[str]]:
    """
    Convert label_to_block_id mapping to allocation_ordered format.

    This function groups layer labels by their assigned block IDs, creating a dictionary
    where keys are block IDs and values are lists of layer names assigned to each block.
    The order of layer names within each block is determined by their position in the
    layer_stats sequence.

    Args:
        label_to_block_id: Dictionary mapping layer names to block IDs
        layer_stats: List of LayerStatistics that determines the layer order

    Returns:
        Dictionary mapping block IDs to lists of layer names assigned to each block,
        ordered by their position in layer_stats

    Example:
        Input: {'layer.0': 0, 'layer.1': 1, 'layer.2': 2, 'layer.3': 3, 'layer.4': 0, 'layer.5': 1}
        Output: {0: ['layer.0', 'layer.4'], 1: ['layer.1', 'layer.5'], 2: ['layer.2'], 3: ['layer.3']}
    """
    allocation_ordered: dict[int, list[str]] = {}

    # Group layers by block ID
    for layer_name, block_id in label_to_block_id.items():
        if block_id not in allocation_ordered:
            allocation_ordered[block_id] = []
        allocation_ordered[block_id].append(layer_name)

    # Get layer name to position mapping
    layer_name_to_position = get_layer_name_to_position(layer_stats)

    # Sort layers within each block by their position in layer_stats
    for block_id in allocation_ordered:
        allocation_ordered[block_id].sort(key=lambda layer_name: layer_name_to_position.get(layer_name, float("inf")))

    return allocation_ordered


def operations_to_csv_string(operations: list[dict[str, Any]]) -> str:
    """
    Convert a list of operations to a CSV-friendly string format.

    This function takes a list of operation dictionaries and converts them to a string
    that can be safely used in CSV files. Commas and other special characters are
    handled to ensure proper CSV formatting.

    Args:
        operations: List of operation dictionaries, each containing:
            - "type": Operation type (e.g., "every_nth", "single", "names", etc.)
            - "n": Number of layers to remove (optional)
            - "offset": Starting offset (optional)
            - "step": Step size for "every_nth" operations (optional)
            - "values": List of values for "names" or "indices" operations (optional)
            - "order": Order for sorting operations (optional)

    Returns:
        String representation of operations suitable for CSV output

    Example:
        Input: [{"type": "every_nth", "n": 2, "offset": 1}]
        Output: "every_nth:n=2:offset=1"

        Input: [{"type": "names", "values": ["layer1", "layer2"]}]
        Output: "names:values=layer1,layer2"
    """
    if not operations:
        return ""

    operation_strings = []

    for op in operations:
        op_type = op.get("type", "unknown")
        parts = [op_type]

        # Add parameters based on operation type
        if "n" in op:
            parts.append(f"n={op['n']}")

        if "offset" in op:
            parts.append(f"offset={op['offset']}")

        if "step" in op:
            parts.append(f"step={op['step']}")

        if "values" in op:
            values = op["values"]
            if isinstance(values, list):
                # Join values with commas and escape any commas in the values
                values_str = ",".join(str(v).replace(",", "\\,") for v in values)
                parts.append(f"values={values_str}")
            else:
                parts.append(f"values={values}")

        if "order" in op:
            parts.append(f"order={op['order']}")

        # Join parts with colons
        operation_string = ":".join(parts)
        operation_strings.append(operation_string)

    # Join multiple operations with semicolons
    return ";".join(operation_strings)

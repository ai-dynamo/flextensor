# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This module solves the problem of planning memory blocks that can fit elements from a map,
with the constraint that neighbouring layers cannot be placed in the same block.
This is essentially a graph coloring problem with size constraints.
"""

from collections import OrderedDict, defaultdict


class MemoryBlockPlanner:
    def __init__(self, layer_sizes: OrderedDict[str, int]):
        """
        Initialize the memory block planner with layer sizes.

        Args:
            layer_sizes: OrderedDict mapping layer names to their sizes in bytes
        """
        self.layer_sizes = layer_sizes
        self.layers = list(layer_sizes.keys())
        self.adjacency_graph = self._build_adjacency_graph()

    def _build_adjacency_graph(self) -> dict[str, set[str]]:
        """
        Build adjacency graph where layers are connected if they are neighbours.
        For sequential layers like layer_0, layer_1, layer_2, etc.
        Uses OrderedDict insertion order.
        """
        graph = defaultdict(set)

        # Use insertion order from OrderedDict
        sorted_layers = list(self.layer_sizes.keys())

        for i, layer in enumerate(sorted_layers):
            # Connect to previous layer if it exists
            if i > 0:
                prev_layer = sorted_layers[i - 1]
                graph[layer].add(prev_layer)
                graph[prev_layer].add(layer)

        return dict(graph)

    def find_minimum_blocks_sequential(self) -> int:
        """
        Find the minimum number of blocks needed for sequential access.
        This is equivalent to finding the chromatic number of the graph.

        Returns:
            Minimum number of blocks needed
        """
        if not self.layers:
            return 0

        # Use a greedy coloring algorithm to find the actual chromatic number
        # Sort layers by their order from OrderedDict
        sorted_layers = list(self.layer_sizes.keys())

        # Use greedy coloring: assign the smallest available color to each layer
        colors: dict[str, int] = {}  # layer -> color (block number)

        for layer in sorted_layers:
            # Find the smallest available color that doesn't conflict with neighbours
            used_colors = set()
            for neighbour in self.adjacency_graph.get(layer, set()):
                if neighbour in colors:
                    used_colors.add(colors[neighbour])

            # Find the smallest available color
            color = 0
            while color in used_colors:
                color += 1

            colors[layer] = color

        return max(colors.values()) + 1

    def optimize_block_distribution(self, num_blocks: int) -> dict[int, list[str]]:
        """
        Find the optimal distribution of all layers across n blocks to maximize performance.
        This distributes layers evenly across blocks while respecting adjacency constraints.

        Args:
            num_blocks: Number of blocks to distribute layers across

        Returns:
            Dictionary mapping block numbers to lists of layer names
        """
        if num_blocks < self.find_minimum_blocks_sequential():
            msg = f"Need at least {self.find_minimum_blocks_sequential()} blocks"
            raise ValueError(msg)

        # Sort layers by their order from OrderedDict
        sorted_layers = list(self.layer_sizes.keys())

        # Initialize blocks
        blocks: dict[int, list[str]] = {i: [] for i in range(num_blocks)}

        # Use a greedy approach: assign each layer to the block that has the fewest layers
        # while respecting adjacency constraints
        for layer in sorted_layers:
            # Find available blocks (those that don't contain neighbouring layers)
            available_blocks = []
            for block_num in range(num_blocks):
                can_place = True
                for existing_layer in blocks[block_num]:
                    if existing_layer in self.adjacency_graph.get(layer, set()):
                        can_place = False
                        break

                if can_place:
                    available_blocks.append(block_num)

            if not available_blocks:
                msg = f"Cannot place layer {layer} in any block"
                raise ValueError(msg)

            # Choose the block with the fewest layers (load balancing)
            best_block = min(available_blocks, key=lambda b: len(blocks[b]))
            blocks[best_block].append(layer)

        return {block_id: layers for block_id, layers in blocks.items() if layers}

    def allocate_blocks(self, num_blocks: int, block_capacity: int | None = None) -> dict[int, list[str]]:
        """
        Allocate layers to blocks using a greedy approach with backtracking.

        Args:
            num_blocks: Number of blocks to allocate
            block_capacity: Maximum capacity per block (if None, no size limit)

        Returns:
            Dictionary mapping block numbers to lists of layer names
        """
        if num_blocks <= 0:
            msg = "Number of blocks must be positive"
            raise ValueError(msg)

        # Initialize blocks
        blocks: dict[int, list[str]] = {i: [] for i in range(num_blocks)}
        block_sizes = dict.fromkeys(range(num_blocks), 0)

        # Try to allocate layers
        if self._allocate_recursive(blocks, block_sizes, block_capacity, 0):
            return blocks
        msg = f"Cannot allocate {len(self.layers)} layers to {num_blocks} blocks with given constraints"
        raise ValueError(msg)

    def _allocate_recursive(
        self,
        blocks: dict[int, list[str]],
        block_sizes: dict[int, int],
        block_capacity: int | None,
        layer_idx: int,
    ) -> bool:
        """
        Recursively try to allocate layers to blocks.

        Args:
            blocks: Current block allocation
            block_sizes: Current size of each block
            block_capacity: Maximum capacity per block
            layer_idx: Index of current layer to allocate

        Returns:
            True if allocation is possible, False otherwise
        """
        if layer_idx >= len(self.layers):
            return True

        current_layer = self.layers[layer_idx]
        current_size = self.layer_sizes[current_layer]

        # Try each block
        for block_num in range(len(blocks)):
            if self._can_place_in_block(current_layer, current_size, block_num, blocks, block_sizes, block_capacity):
                # Place layer in this block
                blocks[block_num].append(current_layer)
                block_sizes[block_num] += current_size

                # Recursively try to place remaining layers
                if self._allocate_recursive(blocks, block_sizes, block_capacity, layer_idx + 1):
                    return True

                # Backtrack: remove layer from this block
                blocks[block_num].remove(current_layer)
                block_sizes[block_num] -= current_size

        return False

    def _can_place_in_block(
        self,
        layer: str,
        size: int,
        block_num: int,
        blocks: dict[int, list[str]],
        block_sizes: dict[int, int],
        block_capacity: int | None,
    ) -> bool:
        """
        Check if a layer can be placed in a specific block.

        Args:
            layer: Layer name to check
            size: Size of the layer
            block_num: Block number to check
            blocks: Current block allocation
            block_sizes: Current size of each block
            block_capacity: Maximum capacity per block

        Returns:
            True if layer can be placed, False otherwise
        """
        # Check capacity constraint
        if block_capacity is not None and block_sizes[block_num] + size > block_capacity:
            return False

        # Check adjacency constraint
        for existing_layer in blocks[block_num]:  # noqa: SIM110
            if existing_layer in self.adjacency_graph.get(layer, set()):
                return False

        return True

    def find_minimum_blocks(self, block_capacity: int | None = None) -> tuple[int, dict[int, list[str]]]:
        """
        Find the minimum number of blocks needed to allocate all layers.

        Args:
            block_capacity: Maximum capacity per block (if None, no size limit)

        Returns:
            Tuple of (minimum_blocks, allocation)
        """
        if not self.layers:
            return 0, {}

        # Binary search for minimum number of blocks
        left = 1
        right = len(self.layers)  # Worst case: one layer per block

        while left < right:
            mid = (left + right) // 2
            try:
                allocation = self.allocate_blocks(mid, block_capacity)
                right = mid
            except ValueError:
                left = mid + 1

        # Final allocation
        allocation = self.allocate_blocks(left, block_capacity)
        return left, allocation

    def find_minimum_block_sizes(self, allocation: dict[int, list[str]]) -> dict[int, int]:
        """
        Find the minimum size required to store each layer group in the allocation.

        Args:
            allocation: Dictionary mapping block numbers to lists of layer names

        Returns:
            Dictionary mapping block numbers to their minimum required sizes in bytes

        Example:
            >>> layer_sizes = OrderedDict([
            ...     ("input", 8388608),
            ...     ("layer_0", 3145728000),
            ...     ("layer_1", 3221225472),
            ...     ("layer_2", 3019898880),
            ...     ("layer_3", 3145728000),
            ...     ("layer_4", 3221225472),
            ...     ("layer_5", 16777216),
            ... ])
            >>> planner = MemoryBlockPlanner(layer_sizes)
            >>> allocation = {0: ['input', 'layer_1', 'layer_3'],
            ...               1: ['layer_0', 'layer_2', 'layer_4'],
            ...               2: ['layer_5']}
            >>> min_sizes = planner.find_minimum_block_sizes(allocation)
            >>> print(min_sizes)
            {0: 3221225472, 1: 3221225472, 2: 16777216}
        """
        min_sizes = {}

        for block_num, layers in allocation.items():
            if not layers:
                min_sizes[block_num] = 0
                continue

            # Find the maximum size among all layers in this block
            # This represents the minimum size needed to store the largest layer
            max_layer_size = max(self.layer_sizes[layer] for layer in layers)
            min_sizes[block_num] = max_layer_size

        return min_sizes

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared memory management and synchronization primitives.

This package provides classes for managing shared memory, synchronization locks,
and multiprocess coordination.
"""

from flextensor.shm.coord_block import COORD_HEADER_SIZE, CoordBlockHeader
from flextensor.shm.coordinator import (
    ShmCoordinator,
    ShmCoordinatorError,
    ShmCreatorDiedError,
    ShmCreatorFailedError,
)
from flextensor.shm.flexible_shm import FlexibleSharedMemory, KeepAlive
from flextensor.shm.multiprocess_condition import (
    HEADER_SIZE,
    MultiprocessCondition,
    SharedMultiString,
)
from flextensor.shm.namespace import (
    SHM_PROTOCOL_VERSION,
    apply_rank_suffix,
    compute_shm_namespace,
    coord_block_name,
    profile_block_name,
    weight_block_name,
)
from flextensor.shm.sync_primitives import ProcessFileLock, SemaphoreLock

__all__ = [
    "COORD_HEADER_SIZE",
    "HEADER_SIZE",
    "SHM_PROTOCOL_VERSION",
    "CoordBlockHeader",
    "FlexibleSharedMemory",
    "KeepAlive",
    "MultiprocessCondition",
    "ProcessFileLock",
    "SemaphoreLock",
    "SharedMultiString",
    "ShmCoordinator",
    "ShmCoordinatorError",
    "ShmCreatorDiedError",
    "ShmCreatorFailedError",
    "apply_rank_suffix",
    "compute_shm_namespace",
    "coord_block_name",
    "profile_block_name",
    "weight_block_name",
]

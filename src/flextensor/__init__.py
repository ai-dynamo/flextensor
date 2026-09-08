# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FlexTensor: Tensor offloading and management library.

This module provides classes and functions for managing tensor offloading
strategies, collectors, loaders, and tensor management operations.
"""

import os

from flextensor._version import __version__

# Package-wide beartype integration with environment variable control
try:
    import beartype.claw

    if os.environ.get("DISABLE_BEARTYPE") not in ["1", "true", "True", "TRUE"]:
        beartype.claw.beartype_this_package()
except ImportError:
    # beartype not available, continue without runtime type checking
    pass

# New simplified public API
# Internal implementation (kept for backward compatibility and advanced usage)
from flextensor.benchmark_tensor_mode import BenchmarkReplace, PreloadToDevice, TensorBenchmarkMode
from flextensor.config import OffloadConfig, load_config, load_config_from_env, load_config_from_file
from flextensor.host_pinning import PinnedMemoryMode
from flextensor.lazy_model_init import load_model_from_profile
from flextensor.offload_manager import (
    DEFAULT_MANAGER_NAME,
    OffloadManager,
    clear_profiling_durations,
    collect_offload_timing,
    get_gpu_memory_usage,
    get_offload_manager,
    init,
    load_profile,
    offload,
    offload_block,
    offload_from_profile,
    offload_from_state,
    offload_units,
    pause_profiling,
    release,
    request_strategy_replan,
    reset_offload_timing,
    resident_units,
    resume_profiling,
    save_profile,
    set_config,
    suspend_profiling,
    update_offload_timing,
    update_state,
)
from flextensor.offload_timing import (
    OffloadTimingReport,
    OffloadTimingSnapshot,
    format_offload_timing_table,
)
from flextensor.strategy import (
    AdaptiveKnapsackStrategy,
    AdaptiveStrategy,
    BudgetFillGreedyStrategy,
    BudgetFillLayerDEStrategy,
    BudgetFillStrategy,
    BudgetFillTensorDEStrategy,
    GlobalOffloadStrategy,
    GlobalTensorSelectionStrategy,
    GreedyStrategy,
    KnapsackBlockStrategy,
    KnapsackStrategy,
    NthLayerStrategy,
    Strategy,
)
from flextensor.tensor_manager import TensorManager
from flextensor.types import GPUMemoryUsage

__all__ = [  # noqa: RUF022
    # Simplified API (recommended)
    "DEFAULT_MANAGER_NAME",
    "GPUMemoryUsage",
    "OffloadConfig",
    "OffloadManager",
    "PinnedMemoryMode",
    "clear_profiling_durations",
    "collect_offload_timing",
    "format_offload_timing_table",
    "get_gpu_memory_usage",
    "get_offload_manager",
    "init",
    "load_config",
    "load_config_from_env",
    "load_config_from_file",
    "offload",
    "offload_block",
    "offload_from_profile",
    "offload_from_state",
    "offload_units",
    "request_strategy_replan",
    "resident_units",
    "reset_offload_timing",
    "set_config",
    "save_profile",
    "load_profile",
    "pause_profiling",
    "release",
    "resume_profiling",
    "suspend_profiling",
    "update_state",
    "update_offload_timing",
    # Internal/Advanced API (for backward compatibility)
    "AdaptiveKnapsackStrategy",
    "AdaptiveStrategy",
    "BenchmarkReplace",
    "GlobalOffloadStrategy",
    "GlobalTensorSelectionStrategy",
    "BudgetFillStrategy",
    "BudgetFillGreedyStrategy",
    "BudgetFillTensorDEStrategy",
    "BudgetFillLayerDEStrategy",
    "GreedyStrategy",
    "KnapsackBlockStrategy",
    "KnapsackStrategy",
    "load_model_from_profile",
    "NthLayerStrategy",
    "PreloadToDevice",
    "Strategy",
    "TensorBenchmarkMode",
    "TensorManager",
    "OffloadTimingReport",
    "OffloadTimingSnapshot",
]

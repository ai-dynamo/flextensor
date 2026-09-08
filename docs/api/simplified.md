<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Simplified API

Module-level convenience functions — the recommended way to use FlexTensor.
These wrap the [`OffloadManager`](offload-manager.md) singleton for common use cases.

::: flextensor.offload_manager.DEFAULT_MANAGER_NAME
    options:
      show_root_full_path: false

::: flextensor.offload_manager.init
    options:
      show_root_full_path: false

::: flextensor.offload_manager.offload
    options:
      show_root_full_path: false

::: flextensor.offload_manager.get_offload_manager
    options:
      show_root_full_path: false

::: flextensor.offload_manager.set_config
    options:
      show_root_full_path: false

::: flextensor.offload_manager.offload_block
    options:
      show_root_full_path: false

::: flextensor.offload_manager.get_gpu_memory_usage
    options:
      show_root_full_path: false

::: flextensor.offload_manager.save_profile
    options:
      show_root_full_path: false

::: flextensor.offload_manager.load_profile
    options:
      show_root_full_path: false

::: flextensor.offload_manager.offload_from_profile
    options:
      show_root_full_path: false

::: flextensor.offload_manager.offload_from_state
    options:
      show_root_full_path: false

::: flextensor.offload_manager.release
    options:
      show_root_full_path: false

## Profiling Data Control

::: flextensor.offload_manager.clear_profiling_durations
    options:
      show_root_full_path: false

::: flextensor.offload_manager.suspend_profiling
    options:
      show_root_full_path: false

::: flextensor.offload_manager.resume_profiling
    options:
      show_root_full_path: false

::: flextensor.offload_manager.pause_profiling
    options:
      show_root_full_path: false

## Offload timing and strategy replan

Measure H2D / compute / wait during inference (`OffloadConfig.offload_timing`)
and rebuild the offload strategy after compile or CUDA-graph capture. After
INFERENCE, [`offload_units()`](#flextensor.offload_manager.offload_units) /
[`resident_units()`](#flextensor.offload_manager.resident_units) list the
modules to compile per unit. See
[Measure transfer overlap](../how-to/configure-for-common-scenarios.md#measure-transfer-overlap-during-inference)
and [torch.compile](../how-to/torch-compile.md).

::: flextensor.offload_manager.collect_offload_timing
    options:
      show_root_full_path: false

::: flextensor.offload_manager.reset_offload_timing
    options:
      show_root_full_path: false

::: flextensor.offload_manager.update_offload_timing
    options:
      show_root_full_path: false

::: flextensor.offload_manager.request_strategy_replan
    options:
      show_root_full_path: false

::: flextensor.offload_manager.offload_units
    options:
      show_root_full_path: false

::: flextensor.offload_manager.resident_units
    options:
      show_root_full_path: false

::: flextensor.offload_manager.update_state
    options:
      show_root_full_path: false

::: flextensor.offload_timing.format_offload_timing_table
    options:
      show_root_full_path: false

::: flextensor.offload_timing.OffloadTimingReport
    options:
      show_root_full_path: false
      members:
        - compute_budgets_by_profile_label
        - num_passes
        - total_compute_sum
        - total_transfer_sum
        - total_wait_sum
        - total_compute_avg
        - total_transfer_avg
        - total_wait_avg

::: flextensor.offload_timing.OffloadTimingSnapshot
    options:
      show_root_full_path: false
      members:
        - total_wait_ms
        - total_transfer_ms
        - total_compute_ms

::: flextensor.offload_timing.TrapTimingRecord
    options:
      show_root_full_path: false

::: flextensor.offload_timing.TrapTimingStats
    options:
      show_root_full_path: false
      members:
        - compute_budget_ms

## Lazy Model Initialization

::: flextensor.lazy_model_init.load_model_from_profile
    options:
      show_root_full_path: false

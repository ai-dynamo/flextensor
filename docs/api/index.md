<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# API Reference

Technical reference for FlexTensor's public API. All pages below are auto-generated from source docstrings.

## [Simplified API](simplified.md)

Module-level convenience functions — the recommended starting point for most users.

- [`DEFAULT_MANAGER_NAME`](simplified.md#flextensor.offload_manager.DEFAULT_MANAGER_NAME) — Name used by `get_offload_manager()` when no explicit name is provided (value: `"default"`)
- [`init()`](simplified.md#flextensor.offload_manager.init) — Pre-initialize the tensor manager (optional, called automatically by `offload()`)
- [`offload()`](simplified.md#flextensor.offload_manager.offload) — Patch a model for automatic tensor offloading
- [`get_offload_manager()`](simplified.md#flextensor.offload_manager.get_offload_manager) — Get or create an offload manager singleton
- [`set_config()`](simplified.md#flextensor.offload_manager.set_config) — Set configuration for an offload manager
- [`offload_block()`](simplified.md#flextensor.offload_manager.offload_block) — Context manager for manual offloading control
- [`get_gpu_memory_usage()`](simplified.md#flextensor.offload_manager.get_gpu_memory_usage) — Get GPU memory usage (inference mode only)
- [`save_profile()` / `load_profile()`](simplified.md#flextensor.offload_manager.save_profile) — Profile persistence
- [`offload_from_profile()`](simplified.md#flextensor.offload_manager.offload_from_profile) — Load a saved profile and offload in one step
- [`offload_from_state()`](simplified.md#flextensor.offload_manager.offload_from_state) — Adopt an in-memory saved state
- [`release()`](simplified.md#flextensor.offload_manager.release) — Release resources and restore model
- [`collect_offload_timing()`](simplified.md#flextensor.offload_manager.collect_offload_timing) / [`reset_offload_timing()`](simplified.md#flextensor.offload_manager.reset_offload_timing) / [`update_offload_timing()`](simplified.md#flextensor.offload_manager.update_offload_timing) — Inference offload-timing measure window
- [`request_strategy_replan()`](simplified.md#flextensor.offload_manager.request_strategy_replan) / [`update_state()`](simplified.md#flextensor.offload_manager.update_state) — Remeasure and rebuild strategy (compile / CUDA graphs)
- [`StrategyReplanStatus`](simplified.md#flextensor.compile.replan_status.StrategyReplanStatus) — Passive replan availability and outcome, read from `OffloadManager.strategy_replan_status`
- [`OffloadTimingReport`](simplified.md#flextensor.offload_timing.OffloadTimingReport) / [`OffloadTimingSnapshot`](simplified.md#flextensor.offload_timing.OffloadTimingSnapshot) — Timing aggregates and per-pass data
- [`load_model_from_profile()`](simplified.md#flextensor.lazy_model_init.load_model_from_profile) — Load a model from a saved profile with optimized weight loading

## [OffloadManager](offload-manager.md)

The `OffloadManager` class provides full control over the offloading lifecycle. Obtain an instance via `get_offload_manager()`.

## [Configuration](configuration.md)

- [`OffloadConfig`](configuration.md#flextensor.config.OffloadConfig) — All offloading parameters with sensible defaults
- [`load_config()`](configuration.md#flextensor.config.load_config) — Load config from files, env vars, and kwargs
- [`GPUMemoryUsage`](configuration.md#flextensor.types.GPUMemoryUsage) — GPU memory usage breakdown

## [Strategies](strategies.md)

Offloading strategies for `OffloadConfig.load_strategy`:

- [`Strategy`](strategies.md#flextensor.strategy.protocol.Strategy) — Protocol all strategies implement
- [`KnapsackStrategy`](strategies.md#flextensor.strategy.knapsack.KnapsackStrategy) — Dynamic programming optimization (default)
- [`GreedyStrategy`](strategies.md#flextensor.strategy.simple.GreedyStrategy) — Largest tensors first
- [`GlobalOffloadStrategy`](strategies.md#flextensor.strategy.global_strategy.GlobalOffloadStrategy) — Global optimization across layers
- And more...

## [Advanced](advanced.md)

Lower-level components for custom workflows:

- [`TensorManager`](advanced.md#flextensor.tensor_manager.TensorManager) — Direct control over discovery, profiling, and inference phases
- [`TensorBenchmarkMode`](advanced.md#flextensor.benchmark_tensor_mode.TensorBenchmarkMode) — Benchmark mode base class
- [`BenchmarkReplace`](advanced.md#flextensor.benchmark_tensor_mode.BenchmarkReplace) / [`PreloadToDevice`](advanced.md#flextensor.benchmark_tensor_mode.PreloadToDevice) — Benchmark implementations
- [Host pinning](advanced.md#host-pinning) — Lower-level pinned-memory helpers and modes

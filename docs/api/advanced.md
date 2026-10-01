<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Advanced API

Lower-level components for custom workflows. In most cases, use the
[Simplified API](simplified.md) or [`OffloadManager`](offload-manager.md) instead.

`TensorManager` provides direct control over discovery, profiling, and inference phases.
See [Internal Phases](../explanation/phases.md) for details on how these components work together.

::: flextensor.tensor_manager.TensorManager
    options:
      members:
        - set_model
        - initialize_warmup
        - initialize_profile
        - initialize_inference
        - prepare_warmup_mode
        - prepare_profile_mode
        - prepare_infer_mode
        - trap
        - release_memory
        - get_gpu_memory_usage
        - get_memory_transfer_stats
        - benchmark_context
        - save_profile
        - load_profile
        - load_state
        - restore_state
        - export_state
        - plan_state_adoption
        - execute_state_adoption
        - restore_adopted_state

## Benchmark Modes

These are used internally by `OffloadManager` and exposed for users who need custom benchmark instrumentation.

::: flextensor.benchmark_tensor_mode.TensorBenchmarkMode
    options:
      members:
        - get_results

::: flextensor.benchmark_tensor_mode.BenchmarkReplace
    options:
      members:
        - get_results

::: flextensor.benchmark_tensor_mode.PreloadToDevice
    options:
      members:
        - get_results

## Host Pinning

Lower-level helpers for `pinned_memory_mode` dispatch and `cudaHostRegister`
registrations. See [Configuration](../explanation/configuration.md#pinned-memory)
for user-facing semantics and
[Troubleshooting](../how-to/troubleshooting.md#step-3-switch-to-in-place-pinning)
for when to switch modes.

::: flextensor.host_pinning
    options:
      members:
        - PinnedMemoryMode
        - HostPinner
        - HostPinRegistry
        - NoOpHostPinner
        - is_available
        - make_host_pinner

## Shared Memory

[`SharedOffloadSession`][flextensor.shared_offload.SharedOffloadSession] provides
coordinated loading from profiles with application-supplied creator and follower
model factories. See
[Shared Memory](../explanation/configuration.md#shared-memory-cross-process-weight-sharing)
for session setup, restoration, and cleanup.

::: flextensor.shared_offload.SharedOffloadSession
    options:
      members:
        - validate_options
        - namespace_for
        - offload_from_profile
        - close

Coordination, shared ownership, and resident tensor storage for cross-process
weight sharing. See
[Low-level SHM protocol and ownership](../explanation/configuration.md#low-level-shm-protocol-and-ownership)
for publication, failure, and lifetime semantics.

::: flextensor.shm.coordinator.ShmCoordinator
    options:
      members:
        - memory_owner
        - wait_for_ready
        - notify_ready
        - mark_failed
        - close

::: flextensor.shm.coordinator.ShmCoordinatorError
    options:
      members: []

::: flextensor.shm.coordinator.ShmCreatorFailedError
    options:
      members: []

::: flextensor.shm.coordinator.ShmCreatorDiedError
    options:
      members: []

::: flextensor.allocation_block.AllocationManager
    options:
      members: []

::: flextensor.shm.tensor_store.SharedTensorStore
    options:
      members:
        - create_from_model
        - attach
        - materialize_into
        - close

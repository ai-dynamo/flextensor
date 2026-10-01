<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Understanding FlexTensor Configuration

This document explains the configuration system in FlexTensor—how it works, what each option controls, and how to make informed decisions about configuring tensor offloading for your workload.

## Overview

FlexTensor uses `OffloadConfig` to control all aspects of tensor offloading behavior. The configuration system is designed around three principles:

1. **Sensible defaults**: Works out-of-the-box for most use cases
2. **Progressive disclosure**: Simple cases need few options; advanced tuning is available when needed
3. **Multiple sources**: Configure via Python code, files, or environment variables

```python
from flextensor import OffloadConfig, offload

# Minimal configuration - uses sensible defaults
config = OffloadConfig()

# Tuned configuration for specific requirements
config = OffloadConfig(
    gpu_device=0,
    profiling_iters=10,
)

model = offload(model, config=config)
```

!!! warning "Single-thread only"
    FlexTensor is **not thread-safe**. The entire offloading lifecycle — setup, discovery, profiling, and inference — must run on one thread. Do not call `offload()`, run forward passes on a patched model, or access the same offload manager from multiple threads concurrently. If you need per-thread offloading, use a separate named manager and model per thread (see [Troubleshooting](../how-to/troubleshooting.md#fix-cross-thread-manager-access-errors)).

## Configuration Loading

FlexTensor supports three ways to load configuration, with clear precedence rules.

### Precedence Order

When multiple sources provide the same option, higher precedence wins:

1. **Explicit kwargs** (highest) — `OffloadConfig(gpu_device=1)`
2. **Environment variables** — `FT_GPU_DEVICE=1`
3. **Configuration file** — `gpu_device = 1` in file
4. **Default values** (lowest) — Built-in defaults

### Loading from Environment Variables

Environment variables use the `FT_` prefix by default:

```bash
export FT_ENABLED=1
export FT_GPU_DEVICE=0
export FT_DISCOVERY_ITERS=2
```

```python
from flextensor import load_config_from_env

config = load_config_from_env()  # Reads FT_* variables
```

!!! note "Environment-only Loading"
    When loading from environment variables without a config file, `enabled` defaults to `False`. You must explicitly set `FT_ENABLED=1` to enable offloading. This prevents accidental offloading when environment variables are partially set.

### Loading from Files

FlexTensor supports INI, JSON, and YAML formats:

=== "INI Format"

    ```ini
    ; flextensor.conf
    [flextensor]
    enabled = true
    gpu_device = 0
    discovery_iters = 1
    profiling_iters = 10
    ```

=== "YAML Format"

    ```yaml
    # flextensor.yaml
    enabled: true
    gpu_device: 0
    discovery_iters: 1
    profiling_iters: 10
    ```

=== "JSON Format"

    ```json
    {
      "enabled": true,
      "gpu_device": 0,
      "discovery_iters": 1,
      "profiling_iters": 10
    }
    ```

```python
from flextensor import load_config, load_config_from_file

# Auto-detect format from extension
config = load_config("flextensor.yaml")

# Or use the file-only loader (no env override)
config = load_config_from_file("flextensor.conf")
```

### Combined Loading

The `load_config()` function combines all sources with proper precedence:

```python
from flextensor import load_config

# File + env vars + kwargs (precedence: kwargs > env > file > defaults)
config = load_config(
    config_path="flextensor.yaml",  # Base configuration
    use_env=True,                     # Override with FT_* env vars
    gpu_device=1,                     # Override with explicit value
)
```

## Core Configuration Options

### Enabling Offloading

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `True` | Master switch for offloading |
| `gpu_device` | int | `0` | GPU device index to use |
| `include_patterns` | list[str] | `["*"]` | Patterns to include for offloading. Each entry is a `<glob>` (module / parameter path), an explicit `name:<glob>`, or a `class:<glob>` matching on the module's class. Default `["*"]` works for quick experimentation; use specific patterns (e.g., `class:*DecoderLayer` or `model.layers.*`) for transformer pipelining. |
| `exclude_patterns` | list[str] | `[]` | Patterns to exclude from offloading (same three forms as `include_patterns`). Applied after `include_patterns`. |

#### Include / Exclude Patterns

The `include_patterns` option specifies which modules in the model to offload. Each entry has one of three forms:

| Form | Selects on | Example |
|------|-----------|---------|
| `<glob>` | Module / parameter path (default) | `layers.*`, `*.weight` |
| `name:<glob>` | Module / parameter path (explicit) | `name:layers.*` |
| `class:<glob>` | Module's class (short name or FQCN) | `class:SharedExpertMLP`, `class:torch.nn.*.Linear` |

Bare patterns behave like `name:`. Globs support `*` (match any sequence) and `?` (match a single character).

```python
config = OffloadConfig(
    include_patterns=["layers.*", "embed", "head"],
)
model = offload(model, config=config)
```

Name patterns can also target individual parameters instead of entire modules. This is useful when you want to offload only specific weight tensors (e.g., large linear weights) while keeping others (e.g., small biases or normalization scales) on GPU:

```python
config = OffloadConfig(
    include_patterns=["*.weight"],
)

# Or target specific parameters within a module subtree
config = OffloadConfig(
    include_patterns=["layers.*.weight"],
)
```

The `class:` form matches on the module's Python class, so it is robust against path renames across upstream model revisions. Each pattern is tested against both `type(module).__name__` and the fully-qualified class name (`f"{cls.__module__}.{cls.__qualname__}"`); a match on either wins. Class patterns are module-level only — a `class:` match cascades to every parameter of the matched module.

```python
# Offload every SharedExpertMLP module regardless of where it sits in the tree
config = OffloadConfig(
    include_patterns=["class:SharedExpertMLP"],
)

# Disambiguate when multiple packages define a class with the same short name
config = OffloadConfig(
    include_patterns=["class:torch.nn.*.Linear"],
)
```

To exclude specific modules or parameters, use `exclude_patterns`. Exclude entries accept the same three forms:

```python
config = OffloadConfig(
    include_patterns=["layers.*", "embed", "head"],
    exclude_patterns=["head", "*.norm", "class:MoELayer"],
)
```

Patterns can also be set via environment variables as comma-separated lists. The `class:` / `name:` prefixes work in env vars too:

```bash
FT_INCLUDE_PATTERNS="layers.*,embed,head" FT_EXCLUDE_PATTERNS="class:MoELayer,*.norm" python my_script.py
```

See [Pattern Matching](pattern-matching.md) for the full matching semantics, including parameter-level vs. module-level matching, FQCN globs, and the dict-model caveats for `class:` patterns.

!!! tip "vLLM worker default patterns"
    When no custom `FT_INCLUDE_PATTERNS` / `FT_EXCLUDE_PATTERNS` are set, `FlexTensorOffloadWorker` installs vLLM-oriented defaults: decoder-layer class includes, common embedding/norm/head paths, and excludes for known MoE sidecars and tiny router/gating tensors that should stay GPU-resident.

    The class patterns give each transformer layer/block its own offload trap, enabling FlexTensor to overlap CPU→GPU transfers with GPU computation without depending on a specific `model.layers.*` path. The exact lists live in `VLLM_DEFAULT_INCLUDE_PATTERNS` and `VLLM_DEFAULT_EXCLUDE_PATTERNS` in `src/flextensor/contrib/vllm/worker.py`. For models with a different layout, inspect module names and classes with `model.named_modules()` and set `FT_INCLUDE_PATTERNS` / `FT_EXCLUDE_PATTERNS` accordingly.

The default `["*"]` matches all top-level child modules. It is a reasonable starting point for quick experimentation, but for transformer models in production, prefer specific patterns such as `class:*DecoderLayer` or `model.layers.*`. With `["*"]`, every top-level child is wrapped in a single coarse trap, which prevents per-trap pipelining. With layer-level class or name patterns, each matched module gets its own trap, and FlexTensor can overlap CPU→GPU transfers for trap N+1 while the GPU computes trap N.

When `enabled=False`, FlexTensor passes through to normal PyTorch execution with no overhead. This is useful for:

- Toggling offloading in production without code changes
- A/B testing offloading vs. baseline
- Disabling offloading on machines with sufficient GPU memory

### Memory Management

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pinned_memory` | bool | `True` | Use pinned memory for CPU-resident weights |
| `pinned_memory_mode` | `"torch" \| "host_register"` | `"torch"` | How to pin: `"torch"` (`tensor.pin_memory()`, copies into a fresh pinned buffer) or `"host_register"` (`cudaHostRegister`, pins existing storage in place — lower peak host RAM). Any other value is rejected at config-construction time. |
| `shm_enabled` | bool | `False` | Enable cross-process weight sharing via POSIX shared memory |
| `shm_namespace` | str \| None | `None` | Base namespace for SHM blocks (auto-derived if None) |
| `shm_wait_timeout` | float | `0.0` | Hard timeout (seconds) for followers waiting on creator |
| `max_gpu_mem_fraction` | float \| None | `None` | GPU memory budget as a fraction of total device memory (e.g. `0.9` = 90%). `None` = latency mode (no constraint). |

#### Pinned Memory

Pinned (page-locked) memory is required for non-blocking CPU→GPU transfers on a separate CUDA stream. This lets FlexTensor overlap weight transfers with GPU computation, hiding offloading latency. The trade-off is higher host memory pressure (pinned pages cannot be swapped). Disable when host memory is scarce:

```python
config = OffloadConfig(pinned_memory=False)
```

`pinned_memory_mode` controls *how* the pinning is done:

- `"torch"` (default) — `tensor.pin_memory()`. Copies the tensor into a fresh pinned allocation from PyTorch's caching pinned allocator, which rounds the allocation up to the next power of two and can substantially inflate host memory usage.
- `"host_register"` — `cudaHostRegister`. Pins the existing allocation in place; no copy, no power-of-two rounding. Requires CUDA; on a CUDA host with a broken `cudart` binding, falls back to `"torch"` with a warning. On a CPU-only host, `pinned_memory=True` raises `RuntimeError` at `OffloadManager`/`TensorManager` construction — set `pinned_memory=False` for CPU-only deployments. Use this when host RAM is the bottleneck.

```python
config = OffloadConfig(pinned_memory=True, pinned_memory_mode="host_register")
```

This option can also be set via the `FT_PINNED_MEMORY_MODE` environment variable:

```bash
FT_PINNED_MEMORY_MODE=host_register   # cudaHostRegister, lower peak host RAM
FT_PINNED_MEMORY_MODE=torch            # PyTorch's pinned allocator (default)
```

!!! note "Scope of `pinned_memory_mode`"
    `pinned_memory_mode` only controls how the **non-SHM allocator path** pins host memory. When `shm_enabled=True` and `pinned_memory=True`, the SHM segment itself is registered in place via `cudaHostRegister` regardless of the mode you choose — POSIX shared-memory buffers can't be re-allocated through PyTorch's pinned allocator. This is why `OffloadConfig(shm_enabled=True, pinned_memory=True, pinned_memory_mode="torch")` already gives you in-place pinning for the SHM segment.

    SHM mappings and the non-SHM `"host_register"` path use `cudaHostRegisterPortable` through
    `torch.cuda.cudart()`. The non-SHM `"torch"` path instead uses `tensor.pin_memory()` and copies
    into a PyTorch-managed pinned allocation. CUDA host registration is process-local, so every
    process that attaches to an SHM segment registers its own mapping and unregisters it during
    teardown. CuPy is not required.

!!! note "Resolved mode vs. requested mode"
    `OffloadConfig.pinned_memory_mode` always reports the value you set; it never mutates. Two construction-time outcomes can differ from a naive read of the field:

    - **CUDA available, cudart binding broken/missing** — `TensorManager` falls back from `"host_register"` to `"torch"` and emits a `WARNING` log naming the cause; pinning still happens via `torch.Tensor.pin_memory()`.
    - **CUDA unavailable** — `pinned_memory=True` raises `RuntimeError` at `OffloadManager`/`TensorManager` construction. Offloading without a GPU has no purpose, and silently degrading to pageable transfers would mask the misconfiguration as a perf regression. Set `pinned_memory=False` on intentional CPU-only deployments.

#### GPU Memory Limit

`max_gpu_mem_fraction` controls how much of the GPU's total memory FlexTensor may use. It accepts `None` or a `float` in the range `(0.0, 1.0]`, where `0.9` means "use at most 90% of total device memory."

The default is `None`: *latency mode*. FlexTensor applies no explicit GPU-memory
cap and chooses the strategy for minimum offloading latency based on
`transfer_budget_scale`:

```python
config = OffloadConfig()  # Latency-first default: no explicit memory cap
```

Set a fraction to opt into *memory mode*. The fraction is resolved to an
absolute byte count at runtime and then **capped by actual available GPU
memory**. If other consumers (CUDA context, KV cache, framework buffers) have
already used some GPU memory, the effective budget will be lower than
`total * fraction`. This ensures the strategy never targets more memory than is
actually free. The same config works portably across GPU SKUs with different
memory capacities.

```python
config = OffloadConfig(
    max_gpu_mem_fraction=0.9,  # Memory mode: cap FlexTensor at 90%
)
```

!!! note "Budget capping and minimum memory"
    If the budget is capped, a warning is logged with the adjusted value. If available
    GPU memory drops below 256 MiB, a `RuntimeError` is raised — see
    [Troubleshooting](../how-to/troubleshooting.md#step-4-check-for-insufficient-free-gpu-memory-error).

This option can also be set via the `FT_MAX_GPU_MEM_FRACTION` environment variable:

```bash
FT_MAX_GPU_MEM_FRACTION=0.8   # Use 80% of GPU memory (memory mode)
FT_MAX_GPU_MEM_FRACTION=none  # Use latency mode explicitly (also accepts "null" or "")
```

#### Shared Memory (Cross-Process Weight Sharing)

When `shm_enabled=True`, FlexTensor stores model weights in POSIX shared memory so
multiple processes (e.g., vLLM data-parallel replicas) can share a single copy of the
weights in CPU RAM. The first process to start becomes the *creator* and loads the model
normally; subsequent processes become *followers* and attach to existing shared memory
blocks without re-loading from disk.

```python
config = OffloadConfig(shm_enabled=True)  # Enable cross-process weight sharing
```

The `shm_namespace` is auto-derived from the model path, config fields, and manager
name, ensuring that different models, config variations, or named `OffloadManager`
instances get separate shared memory regions. Override with an explicit namespace when
needed:

```python
config = OffloadConfig(shm_enabled=True, shm_namespace="my_model_v1")
```

When `shm_wait_timeout` is `0.0` (the default), followers rely on heartbeat liveness
detection rather than a hard wall-clock timeout.

These options can also be set via `FT_SHM_ENABLED`, `FT_SHM_NAMESPACE`, and
`FT_SHM_WAIT_TIMEOUT` environment variables.

##### Low-level SHM protocol and ownership

SHM protocol v4 stores `INITIALIZING`, `READY`, or `FAILED` in the coordination
header. The creator calls
[`ShmCoordinator.notify_ready()`][flextensor.shm.coordinator.ShmCoordinator.notify_ready]
after publication, or [`mark_failed(error)`][flextensor.shm.coordinator.ShmCoordinator.mark_failed]
if initialization fails. Followers call
[`wait_for_ready(timeout=...)`][flextensor.shm.coordinator.ShmCoordinator.wait_for_ready]:
a published failure raises [`ShmCreatorFailedError`][flextensor.shm.coordinator.ShmCreatorFailedError],
and an expired creator heartbeat during initialization raises
[`ShmCreatorDiedError`][flextensor.shm.coordinator.ShmCreatorDiedError].
The timeout includes coordination-lock acquisition; `None` has no overall wait limit.
All participants must use the same FlexTensor and protocol versions. Restart them
together when upgrading from v3.

Heartbeat identities distinguish individual participants, including separate handles
in the same process. [`ShmCoordinator.memory_owner`][flextensor.shm.coordinator.ShmCoordinator.memory_owner]
lets profiles and tensor blocks share one heartbeat and lifetime. Pass it as `shm_owner` to
[`AllocationManager`][flextensor.allocation_block.AllocationManager],
[`SharedTensorStore.create_from_model()`][flextensor.shm.tensor_store.SharedTensorStore.create_from_model],
or [`SharedTensorStore.attach()`][flextensor.shm.tensor_store.SharedTensorStore.attach]. Blocks
remain attachable after the creator closes while another participant remains.
Failed cleanup retains the resources needed to retry
[`close()`][flextensor.shm.coordinator.ShmCoordinator.close].

Owned blocks support the built-in file and semaphore lock backends. The ownership
registry records their lock names as well as their SHM names, and the last participant
unlinks both. Each block initializes under its own lock; the coordinator lock only
protects registration and resource opening. Closing waits for local child setup to
finish and rejects new setup. The coordinator's own lock remains stable across namespace reuse.

[`SharedTensorStore`][flextensor.shm.tensor_store.SharedTensorStore] publishes resident parameters and
registered buffers separately from offloaded allocation blocks. Followers validate
the manifest's dtype, shape, stride, and byte ranges before copying tensors to
their target device. Tied registrations and buffer persistence are preserved.

### Profiling Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `skip_discovery` | bool | `False` | Skip DISCOVERY and derive tensor-to-layer mappings statically from patched modules. Leave at `False` when the model uses manual `offload_block()` blocks — `offload_block()` raises when this is `True`. |
| `discovery_iters` | int | `1` | Iterations to discover parameter-to-trap mappings. Only consulted when `skip_discovery=False`. |
| `profiling_iters` | int | `10` | Iterations to measure execution timing |

FlexTensor learns your model's behavior during initial iterations:

```
Under skip_discovery=False (default): DISCOVERY (discovery_iters) → PROFILING (profiling_iters) → INFERENCE
Under skip_discovery=True:            PROFILING (profiling_iters) → INFERENCE
Remaining iterations in both cases:  Optimized execution with learned strategy
```

Use `OffloadManager.iters_before_inference` for the exact path-aware
count (accounts for `skip_discovery`, compiled offload, and replan paths)
instead of summing the config fields by hand.

#### Tuning Iteration Counts

**`skip_discovery`**: Leave at `False` (default) so DISCOVERY runs —
required for manual `offload_block()` blocks. Set `True` for the
auto-trap path (offloading via `include_patterns`) to cut startup time.
When keeping discovery, tune `discovery_iters`:

- Dynamic control flow affecting parameter access patterns → 2-3 iterations
- Variable-length inputs that change which parameters are used → 2-3 iterations
- Static shapes → 1 iteration

**`profiling_iters`**: More iterations = more accurate timing estimates. Consider:

- **Noisy environments**: Increase to 20-50 for shared/cloud GPUs
- **Deterministic workloads**: 5-10 is often sufficient
- **Quick experimentation**: Reduce to 3-5 for faster iteration

```python
# Production deployment (accurate profiling)
config = OffloadConfig(profiling_iters=20)

# Development/debugging (fast iteration)
config = OffloadConfig(profiling_iters=3)
```

### Transfer Modes

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `transfer_mode` | str | `"allocation_block_transfer"` | Weight transfer strategy |
| `num_blocks` | int | `4` | Number of memory blocks for block transfer |
| `min_blocks` | int | `4` | Minimum blocks for assignment optimization |

FlexTensor supports different transfer strategies:

| Mode | Description | Best For |
|------|-------------|----------|
| `strategy` | Basic strategy-based loading | Simple models, debugging |
| `allocation_block_transfer` | Pre-allocated GPU blocks | Most production workloads |
| `raw_block_transfer` | Direct memory management | Maximum efficiency |

The default `allocation_block_transfer` provides a good balance of performance and memory efficiency by pre-allocating GPU memory blocks and reusing them across layers.

#### Minimum Blocks

`min_blocks` sets the lower bound of the block-count search range used by the assignment optimizer. Lower values give the optimizer more freedom and reduce GPU memory consumed by FlexTensor's memory blocks, but may hurt pipelining throughput. The value must be at least 2 (pipelined execution requires two blocks).

```python
config = OffloadConfig(
    num_blocks=4,
    min_blocks=2,  # Let optimizer try fewer blocks to save GPU memory
)
```

This option can also be set via the `FT_MIN_BLOCKS` environment variable.

#### Gap Traps and Transfer Windows

Gap traps are traps whose modules contain no offloadable weights. When a model has sequential gap traps between offloadable traps, the effective transfer window — the time available to pre-fetch the next trap's weights — is larger than a single module's execution time.

`GapAwareWindow` exploits this by computing transfer windows that span gap layers, allowing FlexTensor to start pre-fetching earlier. `strategy_has_transfer_gaps()` detects whether a computed strategy contains such gaps. When gap layers are detected during inference setup, transfer rearrangement is automatically enabled to take advantage of the extended windows.

### Strategy Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `transfer_budget_scale` | float | `1.0` | Multiplier on the time budget for weight transfers |
| `load_strategy` | Strategy | `None` | Override automatic strategy selection |

#### Understanding Strategies

FlexTensor includes several offloading strategies. The table below is a conceptual comparison of the most commonly used strategies; it is not a complete list. See the [Strategies reference page](../api/strategies.md) for the full API.

| Strategy | Algorithm | Best For |
|----------|-----------|----------|
| `KnapsackStrategy` | Dynamic programming optimization | General use |
| `GreedyStrategy` | Cumulative compute-budget heuristic | Quick approximation |
| `NthLayerStrategy` | Offload every Nth layer | Predictable patterns |
| `AdaptiveKnapsackStrategy` | Knapsack with runtime adaptation | Variable workloads |
| `AdaptiveStrategy` | Evaluates multiple candidates, selects best | Default automatic selection |
| `BudgetFillStrategy` | Facade: greedy + layer/tensor DE (keeps best min-offload plan) | Hard GPU budget with minimal offload |
| `BudgetFillGreedyStrategy` | Ranked-prefix scan under a hard peak budget | Same budget goal; fastest deterministic solver |
| `BudgetFillLayerDEStrategy` | DE over per-layer offload fractions (~n_layers vars) | Same budget goal; may find better layer mixes |
| `BudgetFillTensorDEStrategy` | DE over per-tensor binary selection | Fallback when best set is not a layer prefix |
| `GlobalOffloadStrategy` | Global optimizer across all layers | Maximizing overall memory reduction |
| `GlobalTensorSelectionStrategy` | Metaheuristic search (`"DE"` or `"SA"`) | Highest-quality solution when runtime allows |

When `load_strategy=None` (the default), FlexTensor runs `AdaptiveStrategy` internally. `AdaptiveStrategy` evaluates multiple candidate strategies and selects the one with the lowest estimated overhead that satisfies the memory constraint. When `max_gpu_mem_bytes` is set, `BudgetFillStrategy` is included with layer DE on and tensor DE off (Adaptive uses `GlobalTensorSelectionStrategy` under `extra_optimization` for expensive tensor search). Direct `BudgetFillStrategy()` defaults both DE solvers on and does not skip them solely because the peak already pinches the budget (a tight peak does not prove minimum offload):

```python
from flextensor import OffloadConfig, AdaptiveStrategy, BudgetFillStrategy

# Default Adaptive (includes BudgetFill when a GPU budget is set)
config = OffloadConfig(load_strategy=AdaptiveStrategy())

# Add GlobalTensorSelection
config = OffloadConfig(
    load_strategy=AdaptiveStrategy(extra_optimization=True)
)

# BudgetFill alone (layer + tensor DE on by default)
config = OffloadConfig(load_strategy=BudgetFillStrategy(n_blocks=4))
```

Override `load_strategy` only when you have specific requirements:

```python
from flextensor import OffloadConfig, GreedyStrategy

# Force greedy strategy for faster startup (less optimal offloading)
config = OffloadConfig(load_strategy=GreedyStrategy())
```

### Profile Phase Mode

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `profile_mode` | str | `"view"` | How the profile-phase model is wired to the per-trap loader |

`profile_mode` selects the mechanism used during the **profiling** phase. For `view` and `getter` — both variants of the model-patching runtime — the choice affects only the profile phase and has no effect on discovery or inference. `torch_function` additionally selects the indirect runtime (it forces `_direct_mode=False`), which also changes the warmup and inference trap classes.

| Mode | Description | Compatibility |
|------|-------------|---------------|
| `view` | Default. Profile model is patched with views into a single rotating GPU block (sized to the largest per-label exclusive footprint) plus a fixed prefix for tensors shared across labels. The timed region contains no property-getter indirection, so per-trap timings are cleaner than `getter`'s; for block-transfer loaders this also matches the access pattern used at inference. Both modes do the H2D copy before the timing window opens, so neither includes transfer cost in the per-trap duration. | All `transfer_mode` values |
| `getter` | Profile model uses Python property getters that route every parameter access through the per-trap loader. Lower GPU memory footprint during profile than `view`, at the cost of attribute-getter overhead in per-trap durations. | All `transfer_mode` values |
| `torch_function` | `TorchFunctionMode` traps that rewrite tensor arguments per call, without patching the model. Fallback for models that don't tolerate the patching used by `getter` or `view`. Significant per-op overhead; not torch.compile-compatible. | `transfer_mode='strategy'` only (block transfers also require patching the model) |

```python
config = OffloadConfig(profile_mode="view")
```

This option can also be set via the `FT_PROFILE_MODE` environment variable. Invalid combinations are rejected at config construction.

!!! note "Memory cost of `view`"
    `view` pre-allocates roughly *(bytes of the largest label's exclusive tensors) + (total bytes of tensors shared across labels)* of GPU memory for the duration of the profile phase (released before the inference loader is built). An equally-sized host-memory staging block is allocated alongside it. On models that already run close to the GPU memory ceiling, switch to `profile_mode="getter"` to keep the profile-phase footprint to one layer's tensors at a time.

!!! note "When to switch from `view`"
    Picking `getter` instead of the default is appropriate when (a) the profile phase OOMs under `view`, or (b) your model rejects `.data` reassignment on its parameters (for example, custom parameter classes with overridden ``__setattr__``). Picking `torch_function` is appropriate only when the model also rejects the synthetic-subclass + property-descriptor patching used by `getter`.

### Profile Storage

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `profile_storage_dir` | str | `None` | Directory for profile persistence |
| `profile_read_only` | bool | `False` | Only load profiles, don't save |

Profile storage enables skipping the discovery/profiling phases on subsequent runs by saving and loading offload profiles. Profiles are stored as JSON files in the specified directory.

```python
config = OffloadConfig(
    profile_storage_dir="/tmp/flextensor_profiles",
    profile_read_only=False,  # Allow saving profiles (default)
)

# First run: (discovery →) profiling → save → inference
om = flextensor.get_offload_manager()
model = om.offload(model, config=config)
for _ in range(om.iters_before_inference):
    model(sample_input)
om.save_profile()  # Saves to profile_storage_dir

# Subsequent runs: load profile → inference (faster startup)
om = flextensor.get_offload_manager()
om.set_config(config)
om.load_profile(model=model)  # Loads from profile_storage_dir
```

Set `profile_read_only=True` to prevent accidental profile overwrites in production deployments.

### Debugging Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_instrumentation` | bool | `False` | Capture component initialization args |
| `instrumentation_output_dir` | str | `".flextensor/instrumentation"` | Instrumentation output directory |
| `enable_diagnostics` | bool | `False` | Log memory transfer statistics, per-trap duration statistics, and block assignment table after strategy computation |
| `offload_timing` | `"off"` / `"eager"` / `"cuda_graph"` | `"off"` | Inference transfer / compute / wait timing; `"cuda_graph"` uses external CUDA events for replay readback (PyTorch ≥ 2.8). Requires a block `transfer_mode` (not `strategy`). |
| `piecewise_prefetch` | `"off"` / `"warn"` / `"error"` | `"warn"` | Policy when a PIECEWISE join forces outstanding H2D onto the critical path. Integrations must call loader `join_after_forward()` before each piece's `capture_end`; otherwise only the last-trap join runs and mid-piece boundaries are neither joined nor checked. |

These options help diagnose offloading behavior:

- **`enable_instrumentation`**: Captures the arguments passed to FlexTensor components at initialization time and writes them to `instrumentation_output_dir`. Useful for reproducing configuration state.
- **`enable_diagnostics`**: Logs three tables after strategy computation: a Memory Transfer Statistics table (tensor size → transfer time and bandwidth), a Trap Duration Statistics table (per-trap timing: min, max, median, avg, std, coefficient of variation), and the block assignment table (at NOTICE level 25). The Trap Duration Statistics table lists every offload trap created during profiling — it is the authoritative way to confirm which include patterns were applied as traps. Useful for diagnosing per-trap pipelining setup and inspecting why specific weights were assigned to specific pipeline blocks.
- **`offload_timing`**: Measures H2D overlap during inference (not just profiling). See [Measure transfer overlap during inference](../how-to/configure-for-common-scenarios.md#measure-transfer-overlap-during-inference).
- **`piecewise_prefetch`**: Warns (default) or errors when a PIECEWISE join breaks async H2D overlap. FlexTensor does not discover piece boundaries itself — the integration must call the block loader's `join_after_forward()` before each piece's `capture_end`. Without that, only the last-trap join runs, so mid-piece boundaries are neither joined nor checked by this policy.

The distinction between `enable_diagnostics` and `enable_instrumentation` is scope: `enable_diagnostics` reports strategy decisions (which tensors landed in which block and why); `enable_instrumentation` captures component initialization arguments (how each component was configured).

```python
# Enable instrumentation for debugging
config = OffloadConfig(enable_instrumentation=True)

# Inspect strategy decisions
config = OffloadConfig(enable_diagnostics=True)

# Measure transfer/compute/wait overlap during inference
config = OffloadConfig(offload_timing="eager")
```

!!! warning "Performance Impact"
    Instrumentation and offload timing add overhead. Use for measurement and debugging, not as a default production setting.

### Tensor Discovery

Tensor discovery ensures that all tensors are properly tracked, even those accessed by custom kernels that bypass PyTorch's dispatch system (such as Triton kernels used in FP8 inference). Both untraced tensor discovery and ModuleTracker are always enabled.

See [Tensor Discovery](tensor-discovery.md) for a detailed explanation of how the discovery mechanism works.

## Configuration Recipes

For ready-to-use starting configurations covering memory-constrained systems, performance-focused deployments, quick experimentation, and production, see [How to Configure FlexTensor for Common Scenarios](../how-to/configure-for-common-scenarios.md).

## Summary

| Category | Key Options | Typical Tuning |
|----------|-------------|----------------|
| **Core** | `enabled`, `gpu_device` | Set based on deployment |
| **Memory** | `pinned_memory`, `max_gpu_mem_fraction`, `shm_enabled` | Balance memory vs. performance |
| **Profiling** | `discovery_iters` (only when `skip_discovery=False`), `profiling_iters` | More iters = more accurate |
| **Transfer** | `transfer_mode`, `num_blocks`, `min_blocks` | Default works for most cases |
| **Debug** | `enable_diagnostics`, `enable_instrumentation`, `offload_timing` | Only when troubleshooting / measuring |

Start with defaults, measure performance, and tune based on your specific workload characteristics. The profiling system will adapt to your model's actual behavior, so explicit strategy configuration is rarely needed.

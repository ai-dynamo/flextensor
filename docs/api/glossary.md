<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Glossary

Canonical definitions for terms used throughout the FlexTensor documentation.
Terms marked with a dashed underline elsewhere in the docs show a tooltip
with their short definition on hover.

---

## Core Concepts

### Tensor

A PyTorch tensor — usually a model **parameter** (learnable weight) or **buffer**
(non-learnable state such as running statistics).

The documentation uses three related terms at different levels of specificity:

- **"weight"** or **"model weight"** — used in prose when describing what gets
  offloaded, transferred, loaded, or released. This is the most common term in
  user-facing text.
- **"parameter"** — used when the `torch.nn.Parameter` type matters (e.g.,
  parameter discovery, parameter-to-trap mapping).
- **"tensor"** — used in code identifiers (`TensorManager`, `tensor_ids`), named
  features (tensor discovery), and anywhere the referent could include buffers or
  other non-parameter tensors in the future.

### Inner Tensor Field

A tensor attached as a custom attribute to another tensor — for example,
`weight.scale` used for FP8 dequantization. Inner tensor fields are not standard
PyTorch parameters, so they require [tensor discovery](../explanation/tensor-discovery.md)
to be found and offloaded together with their parent tensor.

### Module

A `torch.nn.Module` instance in the model graph. Modules are the unit that
[include patterns](#include-pattern) select and [traps](#trap) wrap. In
documentation prose, "module" refers to what gets executed (the `forward`
call), while "trap" refers to the FlexTensor wrapper around that execution
(loading weights, timing, releasing memory).

### Layer

A layer in the model architecture — for example, a transformer layer
(`model.layers.0`). In documentation prose, "layer" is reserved for this
model-architecture meaning. FlexTensor-specific concepts use "trap" (the
wrapper) or "module" (the `nn.Module`).

!!! warning "\"Layer\" in code identifiers"
    Internal code uses "layer" in identifiers like `LayerStatistics` and
    `layer_stats` where "trap" is the precise meaning. See the
    [Trap vs layer note](#trap) for details. Issue #116 tracks renaming
    these identifiers.

### Include Pattern

A glob-style string in `OffloadConfig.include_patterns` that selects which modules
or parameters to include for offloading. Supports `*` (match any sequence) and
`?` (match a single character). Patterns are matched against both module paths
(from `named_modules()`) and parameter paths (from `named_parameters()`).

**Module-level patterns** (e.g., `"layers.*"`) patch matching modules and offload
all their parameters. **Parameter-level patterns** (e.g., `"layers.*.weight"`)
automatically derive the parent module pattern for patching, but only offload
parameters that match the full pattern.

See: [Configuration — Include Patterns](../explanation/configuration.md#include-exclude-patterns)

---

## Trap System

### Trap

A context manager that wraps a module's `forward` method to manage weight loading,
timing measurement, and memory release for that module. Each matched module gets its
own trap instance.

Internal trap classes by phase:

| Phase | Indirect mode | Direct mode |
|-------|---------------|-------------|
| Discovery | `WarmupTrap` | `WarmupTrapDirect` |
| Profiling | `Trap` | `TrapDirect` |
| Inference | `TrapInfer` | `TrapInferDirect` |

!!! note "Trap vs layer in code identifiers"
    Internal code uses "layer" in names like `LayerStatistics`, `layer_stats`,
    and `layer.label` — these all represent per-trap data, not model layers.
    A future refactor (see backlog) will rename these to `TrapStatistics`,
    `trap_stats`, etc. Until then, read "layer" in code identifiers as "trap."
    In documentation prose, "layer" is reserved for model architecture (e.g.,
    "transformer layer", `model.layers.*`).

### Forward Patching (Auto Trap)

The mechanism used by `flextensor.offload()` to automatically replace each matched
module's `forward` method with a trap wrapper. The patched module is marked with
`_ft_original_forward_func` and `_ft_offload_name` attributes.

"Forward patching" and "auto trap" are synonyms. Contrast with [manual trap](#manual-trap).

### Manual Trap

An `offload_block()` context manager that the user places explicitly around model
code. Used when forward patching is not feasible — for example, when custom logic
sits between module calls.

Requires `skip_discovery=False` on the `OffloadConfig` (the default);
`skip_discovery=True` short-circuits DISCOVERY and never captures
tensor mappings that only manual blocks can enumerate, so
`offload_block()` raises a `RuntimeError`.

```python
config = OffloadConfig(...)  # skip_discovery=False by default
model = flextensor.offload(model, config=config)
manager = flextensor.get_offload_manager()
with manager.offload_block("layer_0"):
    x = model.layer_0(x)
```

### Gap Trap

A trap whose module contains no offloadable weights. Gap traps extend the
[transfer window](#transfer-window) for neighboring traps, allowing earlier
pre-fetching. Sometimes called "gap layer" in code identifiers — see the
[Trap vs layer note](#trap) above.

### Direct Mode

Trap implementation that routes parameter access through materialized tensors
without per-operation dispatch interception. During discovery and profiling, raw
parameter storage can be temporarily rebound to the active materialized tensor;
during inference, the prepared model uses direct getters or block-backed tensor
views. Lower overhead than indirect mode.

### Indirect Mode

Trap implementation that intercepts PyTorch operations via `__torch_function__` to
replace CPU tensor references with GPU copies on the fly. More flexible than direct
mode (handles varying access patterns) but adds dispatch overhead.

---

## States

FlexTensor uses an internal state machine that progresses automatically:

```text
NOT_INITIALIZED → DISCOVERY → PROFILING → INFERENCE
```

See: [Internal Phases](../explanation/phases.md)

### Discovery Phase

First active phase. It is driven when `skip_discovery=False`, and also when
`skip_discovery=True` was requested but no patched modules were reachable —
the manager then falls back to a real discovery phase and
`OffloadManager.skip_discovery_honored` reads `False`. When the skip does
fire, no discovery *forwards* are consumed: `offload()` still enters
`DISCOVERY` briefly before transitioning to `PROFILING`, so a diagnostic dump
sampled between the two can legitimately observe that phase. When driven, runs
for `discovery_iters` iterations and
discovers which parameters belong to which traps via direct module
ownership, direct getter access, or `WarmupTrap` operation interception
depending on mode. Also referred to as the "parameter discovery phase."

### Profiling Phase

Second active phase. Runs for `profiling_iters` iterations. Measures per-trap
execution timing using CUDA events. The collected statistics feed into strategy
computation. "Profiling" is the process; the [offload profile](#offload-profile)
is the artifact it produces.

### Inference Phase

Third active phase (steady-state). Applies the computed offloading and release
strategies for production execution. No timing collection overhead.

---

## Strategies

### Offloading Strategy

An algorithm that decides which weights to keep on GPU and which to move to CPU,
based on profiling data. Configured via `OffloadConfig.load_strategy`.

Built-in strategies: `KnapsackStrategy`, `GreedyStrategy`, `NthLayerStrategy`,
`AdaptiveKnapsackStrategy`, `AdaptiveStrategy`, `GlobalOffloadStrategy`,
`GlobalTensorSelectionStrategy`.

See: [Strategies reference](strategies.md)

### Release Strategy

An algorithm that decides when to free GPU memory after a trap finishes executing
its module. Complementary to the offloading strategy — the offloading strategy
decides *what* to load; the release strategy decides *when* to unload.

### Assignment Strategy

An algorithm that maps weights to [memory blocks](#memory-block) for pipelined
transfer. Controls how weights are distributed across pre-allocated GPU blocks.

Built-in: `StrictRoundRobinAssignment`, `OptimizedRoundRobinAssignment`.

### Memory Mode

Strategy operating mode activated by setting `max_gpu_mem_fraction` to a float
in the range `(0.0, 1.0]`. The strategy keeps peak GPU usage within the
specified budget.

### Latency Mode

The default standalone strategy mode, activated by
`max_gpu_mem_fraction=None`. The strategy optimises for minimum offloading
latency based on `transfer_budget_scale`, with no explicit memory cap.

### Strategy Replan

Rebuilding the [offloading strategy](#offloading-strategy) after INFERENCE from
timings measured under the real runtime — `torch.compile`d units or captured
CUDA graphs — instead of the eager timings collected during the
[profiling phase](#profiling-phase). Requested with `request_strategy_replan()`,
which returns the number of forwards (or graph replays) to drive; FlexTensor
measures across them, then rebuilds the strategy and its loader.

A replan can only rebuild while FlexTensor still holds the **source weights**:
the original CPU weight copies retained before the first inference loader was
built. A successful rebuild consumes them, so it cannot be repeated.
`OffloadManager.strategy_replan_status` reports whether a replan is available
and whether it applied.

See: [torch.compile](../how-to/torch-compile.md)

---

## Transfer System

### Transfer Mode

The mechanism for physically moving weights between CPU and GPU. Configured via
`OffloadConfig.transfer_mode`.

| Value | Description |
|-------|-------------|
| `"strategy"` | Basic strategy-based loading |
| `"allocation_block_transfer"` | Pre-allocated GPU memory blocks with shared memory support (default) |
| `"raw_block_transfer"` | Pre-allocated GPU memory blocks, legacy — no shared memory support |

### Memory Block

A pre-allocated GPU memory region used by block-based transfer modes
(`allocation_block_transfer`, `raw_block_transfer`). The number of blocks is
controlled by `OffloadConfig.num_blocks` and `min_blocks`.

!!! note "Block vs offload_block()"
    "Block" in the transfer context means a GPU memory region. `offload_block()`
    is an unrelated API — it is a context manager for [manual traps](#manual-trap).

### Transfer Window

The time available to pre-fetch the next trap's weights while the current trap's
module executes on GPU. [Gap traps](#gap-trap) extend transfer windows.

### Pinned Memory

Page-locked CPU memory that enables asynchronous, non-blocking CPU-to-GPU
transfers via DMA (Direct Memory Access). Controlled by
`OffloadConfig.pinned_memory`.

---

## Profiles and Persistence

### Offload Profile

The serialized artifact produced by discovery and profiling: parameter-to-trap maps,
timing statistics, and the computed strategy. Saved via `save_profile()` and
reloaded via `load_profile()` or `offload_from_profile()` to skip those phases on
subsequent runs.

Not to be confused with the [profiling phase](#profiling-phase), which is the
phase that *produces* the profile.

---

## Debugging and Diagnostics

### Instrumentation

The `enable_instrumentation` feature that captures component initialization
arguments and writes them to disk. Answers "how was each component configured?"

### Diagnostics

The `enable_diagnostics` feature that logs strategy decisions after computation:
memory transfer statistics, per-trap duration statistics, and the block assignment
table. Answers "which weights landed in which block and why?"

### Tensor Discovery

The mechanism that finds tensors not traced through normal `__torch_function__`
interception — typically tensors accessed by custom CUDA/Triton kernels.
Uses three strategies in sequence: auto trap discovery, module tracker discovery,
and prefix matching.

See: [Tensor Discovery](../explanation/tensor-discovery.md)

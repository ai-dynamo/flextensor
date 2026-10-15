<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Use `torch.compile` with FlexTensor offload

FlexTensor supports `torch.compile` under one hard rule: **offload first, then
compile.** Calling `offload()` on an already-compiled model is rejected.

There are two paths, in order of preference:

| Path | When to use |
|---|---|
| **`compile_fn` on `offload()`** | Default — compiled offload with break-free custom ops |
| **`external_compile=True` + external compile** | Your pipeline already calls `torch.compile` outside FlexTensor |

Under offload, compile **one offloaded unit per graph**. Do not guess a
container such as `model.layers` — use [`offload_units()`](../api/offload-manager.md#flextensor.offload_manager.OffloadManager.offload_units)
(same set `compile_fn` wraps). GPU-resident islands are
[`resident_units()`](../api/offload-manager.md#flextensor.offload_manager.OffloadManager.resident_units).
Whole-model `torch.compile` is not supported for correct output: rolling GPU
weight slots can alias inside one monolithic graph. With `compile_fn`, FlexTensor
applies compile at the same granularity as offload automatically.

## Inductor compatibility

Some Inductor builds lose mutation-renamed reads from tiling scores, changing
compiled output. FlexTensor corrects the known source pattern when constructing
compiled-offload forwards, before profiling. The defect exists in Torch 2.10
and 2.14; numerical effects depend on the graph, device, shapes, and layouts.

The correction changes Inductor's Python function for the rest of the process.
It preserves installed files and model parameters. A suffix added when graph
cache keys are built prevents reuse of pre-fix graphs, even after caller config
contexts exit. The caller's global cache tag stays unchanged.
Plain imports and eager-only offload do not activate it.
Known-fixed code is left alone; unknown code is left alone with a warning.
Verify output parity on unrecognized builds. An upstream fix should eventually
replace this private-API workaround.

## Recommended: `compile_fn`

Pass any `module -> module` callable to `offload()`. With default
`profile_mode='view'`, FlexTensor runs **compiled view-profile** under
`compile_fn` (strategy timings are already compiled) and re-applies
`compile_fn` at INFERENCE for serving. No `request_strategy_replan()` is
needed on this path.

```python
import torch
import flextensor as ft

config = ft.OffloadConfig(include_patterns=["layers.*"], ...)
model = MyModel().to("cpu")

def compile_fn(module: torch.nn.Module) -> torch.nn.Module:
    return torch.compile(module, fullgraph=True)

model = ft.offload(model, config, compile_fn=compile_fn)
om = ft.get_offload_manager()

# discovery + compiled view-profile → INFERENCE (compile_fn applied)
for _ in range(om.iters_before_inference):
    model(x)
```

Runnable example: `examples/diffusers/compiled-offload/wan_t2v_compiled.py`
(compiled view-profile). Use `request_strategy_replan()` only when profile
was eager — `profile_mode='getter'`, or external compile after
`external_compile=True` (`wan_t2v_external_compile.py`).

`compile_fn` is backend-independent — Torch-TensorRT, a custom tuner, or
`lambda m: m` (identity compiler: no `torch.compile`, but still activates the
compiled-offload custom-op path; omit `compile_fn` entirely for plain eager offload).

## Advanced: external `torch.compile`

Use when compile must stay **outside** FlexTensor. Set
``OffloadConfig(external_compile=True)`` (or ``FT_EXTERNAL_COMPILE=1``), run the
eager lifecycle to INFERENCE, then call ``torch.compile`` yourself — **per
offloaded unit**, not on the whole model.

```python
import flextensor as ft
from flextensor.compiled_offload import bump_dynamo_limits_for_compiled_offload
from flextensor.offload_manager import get_offload_manager

config = ft.OffloadConfig(include_patterns=["layers.*"], external_compile=True)
model = ft.offload(model, config)
om = get_offload_manager()

for _ in range(om.iters_before_inference):
    model(x)  # discovery + profiling → INFERENCE (eager)

units = om.offload_units()
bump_dynamo_limits_for_compiled_offload(len(units))
for _name, module in units:
    module.compile(fullgraph=True)  # in-place; do not guess model.layers

for _name, module in om.resident_units():
    module.compile(fullgraph=True)  # GPU-resident; no replan required

for _ in range(om.request_strategy_replan()):
    model(x)  # warm→measure→replan tail
```

Runnable example: `examples/diffusers/compiled-offload/wan_t2v_external_compile.py`.

Prefer `compile_fn` unless your pipeline already owns the compile step.

### Preflight and verify the replan

`request_strategy_replan()` does not raise when replan is unavailable: it logs a
warning, returns `0`, and serving continues on the eager-profile strategy.
Pipelines that require the compiled strategy should therefore check availability
before the request and verify the outcome afterwards. Both properties below are
passive — reading them never arms or advances the tail:

```python
if not om.external_compile_active:
    raise RuntimeError("FlexTensor must use external_compile=True without compile_fn")

if not om.strategy_replan_status.can_request:
    raise RuntimeError("FlexTensor strategy replan is not available")

for _ in range(om.request_strategy_replan()):
    model(x)

status = om.strategy_replan_status
if status.state == "failed" or status.applied is not True:
    raise RuntimeError("FlexTensor did not apply the new strategy")
```

`can_request` is `True` only when a request would arm a tail: compiled offload
active with replan armed, the lifecycle at INFERENCE, no tail in flight or
`"failed"`, and the source weights still retained. A `"done"` replan that
refused to rebuild (`applied=False`, for example when no compiled timings were
captured) can be retried while FlexTensor still holds those weights; a
successful rebuild consumes them, and a `"failed"` tail is terminal (restart the
process).

Both `request_strategy_replan()` variants enforce these conditions: when
`can_request` is `False` they warn and return `0` without disturbing a tail that
is in flight or already failed. `can_request` is not sufficient for the
CUDA-graph variant below, which adds requirements it does not cover.

## CUDA graphs: ``request_strategy_replan(manual_update_state=True)``

After CUDA-graph capture, arm the same replan helper used for external
compile. Replay does not run module forward hooks, so pass
``manual_update_state=True`` and call ``update_state()`` after each replay.
**Recapture** CUDA graphs after rebuild (loader views change).

**Prerequisites** (otherwise ``request_strategy_replan(..., manual_update_state=True)``
returns ``0`` and performs no replan):

1. ``OffloadConfig(external_compile=True, offload_timing="cuda_graph", ...)``
   so compiled replan is armed and the collector uses ``external=True`` CUDA
   timing events (PyTorch >= 2.8).
2. A block ``transfer_mode`` compatible with compiled offload
   (``allocation_block_transfer`` / ``raw_block_transfer``).
3. Lifecycle already at INFERENCE (discovery + profiling done), with the
   rolling loader installed.

```python
import flextensor as ft

config = ft.OffloadConfig(
    include_patterns=["layers.*"],
    transfer_mode="allocation_block_transfer",
    external_compile=True,       # arms compiled replan; source weights survive first loader
    offload_timing="cuda_graph", # external timing events for post-replay readback
)
model = ft.offload(model, config)
om = ft.get_offload_manager()
for _ in range(om.iters_before_inference):
    model(x)  # discovery + profiling → INFERENCE

# Optional: compile per offloaded unit, then capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = model(static_x)

iters = ft.request_strategy_replan(manual_update_state=True)
for _ in range(iters):  # iters == 0 if any prerequisite above was missing
    graph.replay()
    ft.update_state()  # last call applies compute budgets + rebuilds

# Loader changed — rebuild CUDA graphs before serving.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = model(static_x)
```

``offload_timing="cuda_graph"`` clears prior durable measure when the manual
replan arms and enables post-capture timing readback.

**What replan consumes:** only the **compute** column (per-trap hiding
budget). The strategy rebuild still uses the original profiling
``memory_transfer_stats`` size→time curve for H2D cost — tensor sizes and
host↔device bandwidth are assumed unchanged under CUDA-graph replay.
Measured ``transfer_ms`` / ``wait_ms`` are diagnostic (``wait > 0`` means the
current schedule lost overlap); they are not planner inputs.

For a shutdown dump of serving timings (no replan), set
``offload_timing="cuda_graph"``, call ``reset_offload_timing`` before the
window you care about, after each ``graph.replay()`` call
``update_offload_timing`` (CUDA graphs freeze event readback until then), and
``collect_offload_timing`` at exit. ``update_state`` already invokes
``update_offload_timing`` when a graph replan was armed with
``manual_update_state=True``.

## Backends and modes

FlexTensor does not constrain which `backend=` / `mode=` you pass to
``torch.compile``. The hard constraints are ordering (offload first) and
granularity (one graph per offloaded unit under compiled offload).

## Limits

- **Don't offload an already-compiled model.** `offload()` detects
  ``OptimizedModule`` wrappers and raises. Use ``offload(..., compile_fn=...)``
  or ``external_compile=True`` with external compile instead.
- **Don't compile before discovery completes.** Calling `compiled(x)` while
  the manager is still in `DISCOVERY` can raise `Unhandled FakeTensor Device
  Propagation for aten.mm.default`. Warmup discovery uses real tensor IDs,
  which is incompatible with Dynamo's FakeTensor tracer.
- **Don't compile across a phase transition.** FlexTensor swaps trap behavior
  on discovery → profile → inference transitions. Compile after the transitions
  you care about (typically compile at or after INFERENCE for compiled offload).
- **Don't compile the whole model under offload.** Use per-block / per-unit
  compile so each graph only ever reads one rolling weight slot.

## Why discovery cannot run under `torch.compile`

Discovery uses warmup traps (`WarmupTrap` for indirect mode,
`WarmupTrapDirect` for normal direct mode). Both are `TorchFunctionMode`
based and use real tensor identity to build the tensor-to-trap mapping.
Direct warmup also materializes known raw parameter access by temporarily
binding original parameter storage to active GPU copies while the trap is
executing.

Simplified indirect-mode excerpt (see
`src/flextensor/trap_tensor_mode.py::WarmupTrap.__torch_function__` for
the full implementation):

```python
def __torch_function__(self, func, _types, args, kwargs=None):
    for arg in args:
        if self.tensor_manager.is_traced(arg) and arg.device != self.device_gpu:
            new_arg = arg.to(device=self.device_gpu, copy=True)
            torch.cuda.synchronize()
            ...
    return func(*new_args, **(new_kwargs or {}))
```

Under Dynamo tracing this fails for two compounding reasons:

1. `tensor_manager.is_traced(arg)` is `id(arg) in tracked_ids`. During
   tracing, `arg` is a `FakeTensor` with a different Python `id()` than
   the real parameter, so every lookup returns `False` and no substitution
   is recorded in the traced graph.
2. `FakeTensorProp` then sees `aten.mm.default(gpu_input_fake, cpu_weight_fake)`
   and raises `Unhandled FakeTensor Device Propagation for aten.mm.default,
   found two different devices cuda:0, cpu`.

## Related tests and examples

- `examples/diffusers/compiled-offload/` — `compile_fn` and external compile
- `tests/integration/L0_torch_compile/test_torch_compile.py` — compile + offload
  ordering, external compile, per-unit compile

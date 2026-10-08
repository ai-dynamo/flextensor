<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Serve a vLLM model with FlexTensor

This tutorial serves `Qwen/Qwen2.5-72B-Instruct` on one NVIDIA H100 with the
FlexTensor vLLM worker. Worker v2 is selected by default. It takes over the
model state after vLLM loads the model, places model weights according to a
FlexTensor offloading strategy, and can refresh an offload profile from serving
traffic.

The legacy/v1 worker (`FT_VLLM_USE_V2_WORKER=0`) is deprecated and will be
removed in v0.6.0. Unset `FT_VLLM_USE_V2_WORKER` or set it to `1` to use worker
v2 with the same public worker class path.

`serve.sh` forwards every argument after the model name to `vllm serve`. This
tutorial leaves model-specific vLLM settings at their defaults.

## Requirements

- An NVIDIA GPU supported by vLLM, with enough host RAM for offloaded weights.
- Docker with NVIDIA Container Toolkit, or an equivalent environment with FlexTensor and vLLM installed.
- vLLM 0.17.0 or later. This tutorial uses the `vllm/vllm-openai:v0.25.1` image.
- Run the commands from the FlexTensor repository root.

## Tutorial: build and reuse a decode profile

The first run uses conservative timing statistics because its profile directory
is empty. It then measures pure decode batches and writes an offload profile.
The second run loads those measured timings and computes a new strategy for the
current model, configuration, transfer benchmark, and GPU memory budget.

### 1. Create an empty profile directory

Use a different directory when changing the model, GPU topology, or the batch
type being measured.

```bash
mkdir -p "$PWD/.flextensor-profiles/qwen2.5-72b-h100-decode"
```

Make sure it does not already contain `profile.json` for this first run.

### 2. Start with conservative statistics and collect timings

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 --entrypoint bash \
  -e FT_MAX_GPU_MEM_FRACTION=0.7 \
  -e FT_PROFILING_ITERS=10 \
  -e FT_PROFILE_STORAGE_DIR=/profiles \
  -v "$PWD/.flextensor-profiles/qwen2.5-72b-h100-decode:/profiles" \
  -v "$PWD/examples/vllm:/workspace" \
  vllm/vllm-openai:v0.25.1 \
  -c '/workspace/install.sh && exec /workspace/serve.sh Qwen/Qwen2.5-72B-Instruct'
```

`FT_MAX_GPU_MEM_FRACTION` is the FlexTensor budget for model weights. It is
separate from vLLM's `--gpu-memory-utilization`, which also accounts for KV
cache and runtime allocations.

In another terminal, send requests until the server has collected ten pure
decode batches. The collector stops after the tenth matching iteration, so
extra requests are not sampled:

```bash
for _ in {1..10}; do
  bash examples/vllm/client.sh Qwen/Qwen2.5-72B-Instruct
done
```

Wait for this log message before stopping the server:

```text
refreshed profile saved path=/profiles/profile.json samples=10
```

The file on the host is
`.flextensor-profiles/qwen2.5-72b-h100-decode/profile.json`.

`FT_VLLM_TIMING_BATCH` defaults to `decode`, which accepts only pure decode
scheduler iterations. Set it to `prefill` to measure only pure prefill
iterations. Mixed prefill/decode iterations are ignored in both cases.

### 3. Restart with the measured profile

Stop the first server, then run:

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 --entrypoint bash \
  -e FT_MAX_GPU_MEM_FRACTION=0.7 \
  -e FT_PROFILE_STORAGE_DIR=/profiles \
  -e FT_PROFILE_READ_ONLY=1 \
  -v "$PWD/.flextensor-profiles/qwen2.5-72b-h100-decode:/profiles" \
  -v "$PWD/examples/vllm:/workspace" \
  vllm/vllm-openai:v0.25.1 \
  -c '/workspace/install.sh && exec /workspace/serve.sh Qwen/Qwen2.5-72B-Instruct'
```

Confirm both messages appear:

```text
saved profile loaded path=/profiles/profile.json
saved profile statistics accepted for bootstrap strategy recomputation
```

Worker v2 adopts compatible timing statistics from the saved offload profile,
but does not reuse its old strategy. It scans the current model and recomputes
the strategy from the current configuration, GPU budget, and transfer
benchmark. An active server does not replan or recapture CUDA graphs when the
profile is written; the restart is what applies the refreshed timings.

## Choose vLLM compilation settings for the model

The tutorial uses vLLM's compiled path and CUDA graphs. Worker v2 derives the
matching FlexTensor configuration from the resolved vLLM configuration:

- `OffloadConfig.external_compile=True` only for `VLLM_COMPILE`; it is false for eager vLLM
  execution.
- `OffloadConfig.offload_timing="cuda_graph"` when CUDA graphs are requested at model load;
  otherwise it is `"eager"`.

Worker v2 owns and overwrites those two settings. If you explicitly configure a
conflicting value, it logs a warning before overriding it. vLLM may resolve
attention-backend compatibility later and downgrade the requested CUDA-graph
mode. External timing events selected for a requested graph mode also support
eager execution, so each sampled `execute_model()` call uses actual replay
activity to choose CUDA-graph or eager finalization.

For `VLLM_COMPILE`, the default
`allocation_block_transfer` mode is supported. The resolved vLLM compilation
configuration must use attention-piecewise compilation, include every attention
operator in its splitting operators, and keep
`use_inductor_graph_partition=false`.

`CompilationMode.STOCK_TORCH_COMPILE` and whole-graph Inductor partitioning are
not supported. Worker v2 also rejects elastic expert parallelism, u-batching,
vLLM native weight transfer, and model-backed speculative loading. Model-free
speculative methods (`ngram`, `ngram_gpu`, `suffix`, and `custom_class`) are
supported by the worker validation.

## Use the helper with another model

```bash
bash examples/vllm/serve.sh MODEL_NAME [VLLM_ARGS...]
```

For example, eager mode is a caller choice, not a helper default:

```bash
bash examples/vllm/serve.sh MODEL_NAME --enforce-eager
```

The helper always enables FlexTensor and adds the stable worker class:

```text
flextensor.contrib.vllm.worker.FlexTensorOffloadWorker
```

## Scripts and configuration

- `install.sh` installs the versions in `requirements.txt`.
- `serve.sh` enables FlexTensor, selects worker v2 by default, adds the worker
  class, and forwards vLLM arguments.
- `client.sh` sends a chat completion request and lists available models. Its
  arguments are `MODEL_NAME [PORT]`, with port 8000 by default.

FlexTensor configuration uses `FT_*` environment variables. See the
[configuration reference](../../docs/explanation/configuration.md) and
[troubleshooting guide](../../docs/how-to/troubleshooting.md) for the complete
options and diagnostics.

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

## Share weights across local data-parallel workers

Worker v2 can bootstrap single-node local `mp` data-parallel workers from a
compatible profile through POSIX shared memory. One elected worker invokes
checkpoint loading. Followers build the same model topology on `meta`,
materialize their GPU-resident tensors, and attach to the restored FlexTensor
runtime and shared host backing. This does not share GPU memory between ranks.

Supported weight sharing is limited to DP workers within one `vllm serve`
launch. All workers in that launch must use the same `FT_SHM_NAMESPACE`;
each independent server launch must use a fresh, distinct namespace, even when
using the same model and profile. Sharing a namespace across independent
servers is unsupported. Creator election is namespace-based and does not
enforce a server-launch boundary, so reusing a namespace can cause separate
launches to join the same shared session.

The profile must be created first with SHM disabled, using the same model,
execution mode, and placement-affecting settings such as transfer mode, GPU budget,
and block count. The following source-checkout workflow uses eager execution
for both runs:

The bootstrap profile below is collected by one GPU worker. SHM startup restores
its saved allocation strategy in every DP worker; it does not re-benchmark H2D
bandwidth or replan while ranks transfer concurrently. Host-memory, NUMA, or
PCIe contention can therefore increase transfer waits and latency or reduce
throughput. Transfer completion is synchronized before a weight is consumed, so
this is a performance limitation rather than a correctness limitation. The
DP=2 L1 provides functional coverage only and claims no measured contention
slack. Qualify performance on the target node, topology, DP size, and workload;
representative concurrent profile collection is not yet supported by this SHM
workflow.

```bash
PROFILE_DIR="$PWD/.flextensor-profiles/qwen2.5-7b-dp2-eager"
mkdir -p "$PROFILE_DIR"

docker run --rm --gpus all --ipc=host -p 8000:8000 --entrypoint bash \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e FT_VLLM_USE_V2_WORKER=1 \
  -e FT_TRANSFER_MODE=allocation_block_transfer \
  -e FT_MAX_GPU_MEM_FRACTION=0.3 \
  -e FT_PROFILE_STORAGE_DIR=/profiles \
  -e FT_PROFILING_ITERS=1 \
  -e FT_VLLM_TIMING_BATCH=decode \
  -v "$PROFILE_DIR:/profiles" \
  -v "$PWD:/flextensor" \
  vllm/vllm-openai:v0.17.0 \
  -c 'pip install /flextensor && exec /flextensor/examples/vllm/serve.sh Qwen/Qwen2.5-7B-Instruct --enforce-eager --max-model-len 256 --max-num-seqs 1 --gpu-memory-utilization 0.9'
```

Send one request with `client.sh`, wait for the profile confirmation, and stop
the server:

```bash
bash examples/vllm/client.sh Qwen/Qwen2.5-7B-Instruct
```

```text
refreshed profile saved path=/profiles/profile.json samples=1
```

Then expose exactly two GPUs and start DP=2. Use a new
`FT_SHM_NAMESPACE` value for every server run:

```bash
SHM_NAMESPACE="qwen25-7b-dp2-$(date +%s)-$$"

docker run --rm --gpus all --ipc=host -p 8000:8000 --entrypoint bash \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e FT_VLLM_USE_V2_WORKER=1 \
  -e FT_TRANSFER_MODE=allocation_block_transfer \
  -e FT_MAX_GPU_MEM_FRACTION=0.3 \
  -e FT_PROFILE_STORAGE_DIR=/profiles \
  -e FT_PROFILE_READ_ONLY=1 \
  -e FT_SHM_ENABLED=1 \
  -e FT_SHM_NAMESPACE="$SHM_NAMESPACE" \
  -e FT_SHM_WAIT_TIMEOUT=1800 \
  -v "$PROFILE_DIR:/profiles" \
  -v "$PWD:/flextensor" \
  vllm/vllm-openai:v0.17.0 \
  -c 'pip install /flextensor && exec /flextensor/examples/vllm/serve.sh Qwen/Qwen2.5-7B-Instruct --enforce-eager --max-model-len 256 --max-num-seqs 1 --gpu-memory-utilization 0.9 --data-parallel-size 2 --data-parallel-backend mp'
```

The host `/dev/shm` must be large enough for the shared weights, which is why
the Docker examples use `--ipc=host`.

The SHM path does not restrict the model architecture. The current DP=2
acceptance test covers `Qwen2ForCausalLM`; other architectures are accepted but
have not been exercised by this test. Followers skip `load_weights()`, so a
model that depends on its non-tensor side effects may need architecture-specific
support.

This path requires vLLM 0.17.0 or later. The DP=2 acceptance test targets the
following configuration at the minimum version:

- worker v2 with a native, unconverted, unquantized generation model and `load_format=auto`;
- `allocation_block_transfer` and an existing read-only `profile.json`;
- one node with local `mp` data parallelism, `DP >= 2`, and TP, PP, PCP, and DCP all equal to 1;
- no external or hybrid DP load balancing, expert parallelism or EPLB, LoRA, or speculative decoding.

The L1 acceptance test targets DP=2 with `Qwen/Qwen2.5-7B-Instruct` and vLLM's
default compilation settings. Larger DP sizes are in the supported envelope
but have not yet been exercised by the L1 acceptance test.

## Choose vLLM compilation settings for the model

The single-worker profile-reuse tutorial uses vLLM's compiled path and CUDA
graphs. The SHM example above explicitly selects eager execution. Both paths
use the same compilation checks, and worker v2 derives the matching FlexTensor
configuration from the resolved vLLM configuration:

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

<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# FlexTensor

**Run larger models within your GPU memory budget.**

FlexTensor is a weight-offloading library for large-model inference, supporting PyTorch models and integration
with vLLM. It keeps selected model weights in CPU RAM and streams them to the GPU as needed, helping you run
models whose weights would otherwise exceed available GPU memory without reducing model size or precision.

FlexTensor profiles execution and prefetches upcoming weights to overlap transfers with GPU computation.
This can hide transfer overhead in compute-heavy workloads.
The achievable memory savings and inference speed depend on your model, workload and CPU–GPU bandwidth.

## Development

FlexTensor is actively developed at NVIDIA, with changes **upstreamed** from internal development to
[the public GitHub repository](https://github.com/ai-dynamo/flextensor). The public repository may therefore
lag work in progress, but community contributions are welcome: bug reports, workload results, documentation
improvements and pull requests all help shape the project.

[Open an issue](https://github.com/ai-dynamo/flextensor/issues) to share your use case or discuss a proposed
change and see the [contribution guide](CONTRIBUTING.md) to get started.

## Release Plan

- **16 December 2026**
- **3 February 2027**
- **11 March 2027**

These target dates may change; we will update this plan as needed based on feature readiness.

## Why FlexTensor

- **Use the GPUs you already have.** Run models that would otherwise exceed their available memory,
  provided the host has enough RAM and the GPU can hold the active weights and runtime working memory.
- **Free GPU memory occupied by weights.** Leave more room for activations, KV cache or other pipeline components.
- **Choose a memory–speed tradeoff.** Configure which modules to offload and the GPU memory budget;
  FlexTensor uses profiling to plan weight placement and transfers.

## Success stories

### LTX-2.3: 48% more video. Same GPU.

Generate longer videos on the hardware you already own. In our LTX-2.3 outpainting benchmark,
FlexTensor extended the maximum video length on one A100 80GB from **34.4 to 51 seconds**.
On an A100 SXM4 40GB, FlexTensor reached 20.71 seconds (497 frames), while native execution ran out of memory
on the same workload.

| GPU | Native execution, no offloading | With FlexTensor |
| --- | --- | --- |
| A100 80GB | 34.38 s maximum (825 frames) | **51.04 s maximum (1,225 frames)** — 48.5% longer |
| A100 40GB | Out of memory | **20.71 s maximum (497 frames)** |

The FlexTensor maximum-length measurements used A100 GPUs at 1280×704 resolution and 24 fps,
with one worker and `expandable_segments:True`. The reported durations refer to video length, not generation time.

**[Explore the LTX-2.3 outpainting example →](examples/ltx23/outpaint/)**

<details>
<summary><strong>Scale for faster responses or more videos per hour</strong></summary>

With a fixed fleet of eight A100 80GB GPUs, the same 361-frame FlexTensor workload can use more GPUs
per request to reduce latency or serve more independent replicas to increase total throughput.

| GPUs per request | Concurrent replicas | Median generation latency | Fleet throughput |
| --- | --- | --- | --- |
| 1 | 8 | 207 s | 134.4 clips/hour |
| 4 | 2 | 99 s | 72.1 clips/hour |
| 8 | 1 | 92 s | 39.0 clips/hour |

Eight GPUs per request reduced median latency by 55% relative to one, at the cost of lower aggregate
throughput. Four GPUs per request offered most of that latency benefit while retaining two replicas.
These measurements used an NVLink/NVSwitch system, five requests per replica and one request in flight
per replica. They compare context-parallel configurations, not FlexTensor against native execution.

</details>

### Cosmos-Transfer2.5: from out of memory to a finished video on a desktop GPU

The reference recipe calls for roughly **65 GB of GPU memory**. The FlexTensor example completed
video generation on a **32 GB RTX 5090**, opening this workload to a workstation GPU. The reference recipe
can also reduce GPU memory use by disabling guardrail models or offloading them to system memory.
The FlexTensor example keeps the full pipeline enabled and executes every stage on GPU by streaming
components through GPU memory as needed.

| GPU | Native Cosmos | FlexTensor example |
| --- | --- | --- |
| RTX 5090 32GB | Out of memory while allocating the text encoder | **Completed video; 30.5 GB peak GPU memory** |
| A100 40GB | Out of memory in the tokenizer/VAE stage | **Completed video; 38.2 GB peak GPU memory** |

Both runs used the full 35-step, 93-frame, 640×480 robot-edge workload. The example constructs the model
on CPU first, then streams roughly 26 GiB of pipeline weights through host RAM, including the 15.45 GiB
Qwen text encoder. The reported quality check found LPIPS ≈ 0.03 against native output across all frames,
indicating close perceptual similarity.

This uses the example's Cosmos compatibility patch for CPU-first loading. Host RAM supplies the weight
storage that no longer needs to stay on the GPU; plan for **at least 64 GB system RAM**, with **128 GB
recommended** for profiling and full-step experiments.

**[Run the Cosmos-Transfer2.5 example →](examples/cosmos-transfer25/)**

Results are specific to these workloads and configurations. They demonstrate capacity gains;
latency, peak memory and output quality should be measured for your deployment.

## Try it yourself: Wan2.2

**Two thirds less GPU memory, less than 1% generation-time overhead.** In our Wan2.2 benchmark on an
RTX PRO 6000 Blackwell, FlexTensor reduced peak allocated GPU memory from 78.2 to 25.9 GB.
The same workload also completed on a 32 GB GeForce RTX 5090, where native execution did not fit.

| GPU | Native peak allocated GPU memory | FlexTensor peak allocated GPU memory | Generation-time overhead |
| --- | --- | --- | --- |
| RTX PRO 6000 Blackwell (96 GB) | 78.2 GB | **25.9 GB (−67%)** | **< 1%** |
| GeForce RTX 5090 (32 GB) | Does not fit | **25.9 GB; completed** | — |

Measurements used `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, 81 frames at 832×480, 40 inference steps and
five runs per configuration. On the RTX PRO 6000, median generation time was 483.8 s with FlexTensor
versus 479.4 s native, excluding discovery and profiling. Memory figures measure peak allocated GPU
memory, not total device usage. The RTX 5090 host had 124 GiB of RAM; this is the tested configuration,
not a minimum host-RAM requirement.

**Turn a text prompt into a video with Wan2.2 and Diffusers.** The example keeps the two transformers' weights
in CPU RAM and streams them to the GPU during generation, reducing the GPU memory occupied by those weights.
The text encoder, VAE and runtime activations still need GPU memory.

The complete script loads the pipeline, configures offloading, runs an initial discovery/profiling generation
and then generates a video using the learned transfer schedule. The result is saved as `wan-t2v.mp4`.

**[Run the complete Wan2.2 example →](examples/diffusers/quickstart/)**

GPU and host RAM requirements and transfer overhead depend on the generation settings and hardware.

## Is FlexTensor right for your workload?

| Your workload | What to expect |
| --- | --- |
| Video generation and other compute-heavy diffusion workloads | Good candidates: longer GPU computation can provide time to overlap weight transfers. See the measured results and Wan example above. |
| LLM prefill or larger inference batches | Worth evaluating: more computation can improve transfer overlap. Measure with your actual input lengths and batch sizes. |
| Single-request LLM decode with strict per-token latency requirements | Transfers may dominate the short compute steps and add substantial latency. |
| GPU memory is mostly occupied by activations or KV cache | Limited help: offloading weights does not directly reduce these memory requirements. |
| The model already fits on GPU and lowest latency is the priority | Little reason to offload: do not expect a speedup over keeping all weights on GPU. |

You need a CUDA-capable GPU and enough host RAM for offloaded weights and loading/profiling overhead.
The GPU must still hold active and prefetched weights, non-offloaded components and runtime memory.
These examples cover inference, not training or backward passes.

Profile with a workload representative of production. A transfer schedule profiled with large batches may
perform poorly for single-token decode. See [performance considerations](docs/how-to/troubleshooting.md#when-to-expect-higher-overhead)
for more detail.

## Quickstart snippet

Install with `pip install flextensor`; see the [installation guide](docs/installation.md) for requirements.

This conceptual snippet shows how to wrap an existing model:

```python
import flextensor

model = YourModel()  # Initialize on CPU

model = flextensor.offload(
    model,
    config=flextensor.OffloadConfig(
        include_patterns=["layers.*"],
    ),
)

# Initial iterations discover tensor usage and profile execution.
for batch in dataloader:
    output = model(batch)
```

Replace `YourModel()` and `dataloader` with your own model and inputs. Match `include_patterns` to your
model's module names. Device placement for inputs and non-offloaded components depends on the model;
see the [Quick Start](docs/quick-start.md) and the complete Wan example above for setup details.
Measure inference speed after discovery and profiling finish. Saved profiles can be reused on later runs
with a compatible model and setup to avoid repeating those phases.

## Documentation

- [Quick Start](docs/quick-start.md) — setup, module selection and profile reuse.
- [Configuration](docs/explanation/configuration.md) — memory budgets and offloading options.
- [Troubleshooting](docs/how-to/troubleshooting.md) — memory errors and performance diagnosis.
- [API reference](docs/api/index.md) — detailed interfaces.

## License

FlexTensor is licensed under the [Apache License 2.0](LICENSE). See
[NOTICE](NOTICE) for the project notice, [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for
third-party dependency attributions and
[EXTERNAL_MATERIALS.md](EXTERNAL_MATERIALS.md) for external materials.

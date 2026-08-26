<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# LTX 2.3 outpaint — the optimizations, what each is worth, and whether it changes the output

`run_optimized.sbatch` turns all of these on. This file is the audit trail: every row was measured
on 8x A100-SXM4 NVSwitch with the harness in this directory, and every row states its parity against
the unoptimized path rather than asserting it is safe.

**Bit-exact** means identical output md5 and `cosine=1.000000`. Anything else is stated with its
measured cosine. The acceptance bar for this bf16 DiT is **cosine >= 0.995** — perturbing the eager
reference's input at bf16 rounding scale already moves cosine to ~0.9957, so max-abs error is not a
useful metric here.

## Where the time goes

The profile moves substantially as optimizations land, which is the main reason to re-measure rather
than reason from an old breakdown.

15 s clip, CP4xTP2, 8x A100-80GB (the 40 GB profile is the same shape; its residual move is 5.4 s):

| stage | DiT rebuilt per generate | DiT resident, `module.to` | DiT resident, flat move |
|---|---:|---:|---:|
| weight movement | — | 38.9 s | **3.7 s** |
| DiT stage 2 | 19.1 s | 18.2 s | 18.3 s |
| Gemma | 9.9 s | 9.3 s | 9.2 s |
| DiT stage 1 | 8.1 s | 7.8 s | 8.3 s |
| VAE decode | 8.1 s | 7.9 s | 6.3 s |
| VAE encode | 4.1 s | 4.0 s | 3.8 s |
| **E2E** | **131.4 s** | **95.6 s** | **55.5 s** |

DiT compute was 20% of the wall in the first column and is ~48% in the last. Backend choice matters
proportionally more once movement is fixed — which is why it is worth doing in that order.

## Pipeline: stop non-DiT stages running redundantly on all 8 ranks

A context-parallel server shards the DiT and leaves everything else replicated. These remove that.

| flag | what it does | measured | parity |
|---|---|---|---|
| `ENC_SHARD` | tile-shard the VAE reference encode | part of 38.6 -> 7.6 s (30 s clip) | bit-exact |
| `DECODE_SHARD` + `DECODE_BALANCE` | shard VAE decode tiles, cost-balanced (LPT) assignment | part of 51.5 -> 12.2 s | bit-exact |
| `GEMMA_BCAST` | load Gemma once on rank 0, broadcast contexts, instead of 8 redundant 24 GB loads | part of 15.7 -> 9.5 s | bit-exact |
| `CACHE_COND` | build the VAE encoder once and keep it; cache the reference conditioning | see note | bit-exact |
| `FAST_PREP` | reference assembly was `torch.cat` inside a 720-frame loop, i.e. O(n^2) | see note | bit-exact |
| `SHARD_PREP` | per-rank `av.seek` decode of the reference clip instead of 8 full decodes | see note | bit-exact |

Together, `b0 -> b4`: **1.36x** (15 s / 40 GB), **1.46x** (30 s / 40 GB), **1.61x** (30 s / 80 GB).

**Note on the last three.** Individually they are *not* separable at this sample size — each
configuration is one process launch, and the launch-to-launch spread is larger than the gap between
adjacent steps. `RESULTS.md` documents this. The group total is sound; per-flag attribution is not.

## Memory: keep full-sequence work off every rank

| flag | what it does | measured | parity |
|---|---|---|---|
| `SHARD_ADALN` | AdaLN timestep projections ran at full sequence length on every rank, before any CP sharding: ~10.9 GiB plus ~4.9 GiB for the cross-attention AdaLN at 30 s | enables 30 s on 40 GB | **bit-exact** |
| `SHARD_OUTPUT` | gather after `proj_out` instead of after the transformer blocks, so the per-token output tail runs on the shard | ~1.2 GiB; last blocker for 30 s on 40 GB | **not exact** — cosine 0.998949 / PSNR 36.0 dB |
| `SHARD_ON_CPU` | build and shard on the host, move only the per-rank shard | **required at 40 GB**; costs time at 80 GB | bit-exact |

Together on 80 GB at 30 s, where nothing forces them: peak **53.2 -> 35.9 GiB** and 1.16x wall.

`SHARD_OUTPUT` is the one non-exact change. Giving `proj_out`'s GEMM a shard rather than the full
sequence lets cuBLAS pick a different kernel and accumulate bf16 differently. It lands at the same
level context-parallel sharding itself shows against a single GPU, and the per-frame PSNR spread is
uniform (34.6-36.5 dB) rather than structured — accumulation noise, not an ordering bug.

## Weight lifecycle — the largest single win

| flag | what it does | measured | parity |
|---|---|---|---|
| `RESIDENT_DIT=1` + `RESIDENT_DIT_MODE=cpu` | build and shard each DiT stage once, park in host RAM, instead of rebuilding the 22B model on every generate | 131.4 -> ~95 s | bit-exact |
| `FLAT_MOVE=1` | one pinned host buffer per dtype, a single copy per dtype device-ward, and parking back is a pointer swap with no traffic at all | see below | **bit-exact** |

`FLAT_MOVE`, 15 s clip, interleaved repeat launches on each card:

| card | `module.to` | flat | E2E | move | peak |
|---|---:|---:|---:|---:|---:|
| A100-40GB | 95.29 / 97.35 s | **61.03 / 60.05 s** | **1.59x** | 38.3 -> 5.4 s | 29.7 GiB, unchanged |
| A100-80GB | 95.23 / 96.00 s | **55.40 / 55.59 s** | **1.72x** | 38.9 -> 3.7 s | 29.7 GiB, unchanged |

40 GB gets less than 80 GB because its residual move is 5.4 s rather than 3.7 s — slower
host-to-device bandwidth, not a different amount of work. Quote the figure for the card in question.

The device-side flat buffer is released when a stage parks. Holding it across the other stage's turn
would keep two buffers alive and add a full stage (~19 GiB) to peak — enough to not fit a 40 GB card.
Release is deliberately *without* `empty_cache()`: the caching allocator keeps the block and hands the
identically-sized buffer to the next stage, so peak returns to the `module.to` baseline at no cost.

A side effect worth having: removing ~35 s of pageable copying removed most of the run-to-run
variance. The flat arms land at 61.04/61.03 and 60.08/60.03 against 99.29/91.28 and 101.07/93.63 for
`module.to`, which makes future A/B comparisons on this pipeline much easier to resolve.

`nn.Module.to()` walks thousands of small parameter tensors in pageable host memory, and copies them
*back* to the host afterwards — pointless, since inference never modifies weights, so the host copy
is already authoritative. Halving the traffic and making the remaining half one large pinned copy is
worth **7.1x on the move at 40 GB** (10.7x at 80 GB), and the E2E figures above.

Both cards were measured with interleaved repeat launches (`module.to` / flat / `module.to` / flat).
The arms do not drift between rounds and the separation is an order of magnitude larger than the
within-arm spread.

Earlier notes on this codebase recorded CPU-resident as a *net loss* (host->device move 37.5 s
exceeding a 27 s rebuild). The move cost reproduces almost exactly, but the conclusion inverted,
because `SHARD_ON_CPU` now builds on the host anyway — so the resident path avoids re-reading the
model from disk rather than adding work.

## DiT backend

Measured as DiT stage time (stage 1 + stage 2), which is the signal; E2E is dominated by other
terms and its spread swamps the difference.

| backend | 2 s clip | 15 s clip | parity | status |
|---|---:|---:|---|---|
| eager | 9.56 s | 26.01 s | — | default |
| `torch.compile(inductor)` | **7.05 s (1.36x)** | **23.53 s (1.11x)** | cosine 0.9984 | opt-in via `COMPILE_DIT=inductor` |
| Torch-TensorRT, per block | **11.95 s (0.80x)** | build fails | cosine 0.9988 | not enabled |

**`torch.compile`'s benefit shrinks as the sequence grows** — 1.36x at 2 s, 1.11x at 15 s. At short
sequences the DiT is relatively more launch- and elementwise-bound, which is what inductor fuses
away; at video lengths it is GEMM-bound and cuBLAS is already doing the work. It is off by default
so the shipped configuration stays closest to eager numerics.

**Torch-TensorRT is slower than eager here**, and 1.7x slower than `torch.compile`. Two independent
reasons, both about the *configuration that is buildable*, not about TensorRT in general:

1. **No native TRT collectives** in torch-tensorrt 2.9.0, so every CP all-to-all and TP all-reduce
   inside a block is a graph break and each block lowers to several engines. The per-engine boundary
   cost outweighs TRT's kernel-selection advantage.
2. **At 15 s it does not build at all** — an OOM requesting 48.84 GiB. This is *not* the TRT builder
   workspace (capping `workspace_size` changes nothing); the traceback is in `proxy_tensor.py` /
   `torch/_ops.py`, i.e. Torch-TensorRT's lowering materialises a real tensor while tracing under
   `use_distributed_mode_trace`.

TRT's theoretical advantages over `torch.compile` — per-shape kernel autotuning and whole-attention
fusion — are precisely what the graph breaks prevent it from exploiting, and both target the
non-GEMM fraction that shrinks at video sequence lengths anyway.

**This does not settle the single-engine question.** A whole-stage single native engine previously
measured 2.07x against eager on this DiT. Reaching it needs native collectives so a block is one
engine, a lowering path that does not materialise ~49 GiB while tracing, and — for a CP x TP mesh —
the per-subgroup NCCL communicator work still in review upstream. Per-block multi-engine TRT should
not be quoted as the single-engine result.

**TensorRT is not enabled here, and the reasons are worth recording:**

1. It asked for a **48.84 GiB build workspace** for one DiT block at 361 frames, on top of the
   resident weights — it OOMed on an 80 GB card. TRT sizes its workspace from activation shapes, and
   this DiT at video sequence lengths is large. `TRT_WORKSPACE_MB` caps it, but the cap is not
   tuning: without it the build does not complete.
2. The available `torch-tensorrt` (2.9.0) has **no native TRT collectives**, so the CP/TP collectives
   become graph breaks and each block lowers to several engines. This is the weakest of the TRT
   configurations, not the whole-stage single engine.
3. A whole-stage single native engine for a **CP x TP** mesh additionally needs per-subgroup NCCL
   communicator support that is still in review upstream (Myelin !19723 and the TensorRT MRs that
   depend on it). A CP-only mesh is a single world group and can single-engine on a stock
   native-collectives build, but CP-only replicates the ~44 GiB DiT and therefore only exists on
   80 GB cards.

So "TensorRT vs `torch.compile`" is not one comparison but three, and only the weakest is currently
buildable here. Anything measured in that configuration should not be presented as the single-engine
number.

## Mesh

**CP4xTP2 on both cards.** The best mesh follows whichever cost dominates the wall, not the card:

| 30 s clip, 40 GB | CP4xTP2 | CP2xTP4 |
|---|---:|---:|
| DiT rebuilt per generate (movement dominates) | 192.1 s | **186.9 s** |
| resident DiT + flat movement (DiT dominates) | **117.6 s** | 127.1 s |

The ordering inverts. With the DiT rebuilt every generate, weight movement is ~38 s and CP2xTP4's
smaller per-rank shard moves faster. Once movement is ~4 s the DiT's own cost decides, and CP2xTP4
loses because 4-way tensor parallelism adds an all-reduce to every linear -- DiT 82.7 s vs 73.6 s.

This is a caution about carrying a mesh choice across a configuration change, not just across a
card. An earlier revision of this file recommended CP2xTP4 at 40 GB on the strength of the first
row.

CP8 does not fit either card: pure context parallelism replicates the ~44 GiB bf16 DiT on every
rank. Pure tensor parallelism (`cp1tp8`) reduces no rank's activation footprint and OOMs at 30 s on
40 GB. Pick the mesh for the target card; do not carry one across.

## What is left

- **Weight movement is now 3.7 s** of a 55 s wall, so further work there has little left to give.
- **VAE** is already tile-sharded and bit-exact; at 15 s it is ~10 s of the wall (encode 3.8 s,
  decode 6.3 s). Reducing it further needs a different decoder rather than more parallelism.
- **DiT compute is now ~48% of the wall**, so it is the remaining target — which is what makes the
  TensorRT question worth settling properly rather than abandoning.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# LTX 2.3 outpaint — E2E speedup results

**Hardware:** `luna-prod-48` (8x A100-SXM4-40GB) and `luna-prod-78-80gb` (8x A100-SXM4-80GB), both
NVSwitch (NV12 between all 8 GPUs). Each table states which.
**Software:** torch 2.9.1+cu128, LTX-2 `9377758`, driver 580-class.
**Workload:** 1280x704, `real_input_30s.mp4` (720 frames, 24 fps, md5 `c0495dd8c5aff4c9dee4edd196f5188f`,
built by `-stream_loop 4 -t 30` from the 8 s letterboxed source), seed 171198.
**Method:** `ITERS=3`, iteration 0 discarded, warm median of the remaining two. Every iteration is
listed. Reproduce with `run_all.sbatch` (see [`README.md`](README.md)).

**Known limitation:** each configuration is one process launch, and launch-to-launch variance is
larger than the gap between adjacent optimization steps. Endpoint comparisons are sound; per-step
attribution is not. See "The per-step attribution does not survive scrutiny" below.

Two ratios come out of this and they are **not** interchangeable. Both are given.

---

## 15 s clip (NF=361)

| config | warm median | all iterations | peak GPU |
|---|---:|---|---:|
| `1gpu_flextensor` | **299.12 s** | 305.44, 299.14, 299.10 | 25.1 GiB |
| `b0_dit_shard_only` | 179.27 s | 195.29, 179.66, 178.88 | 29.1 GiB |
| `b1_vae_gemma_decode` | 135.61 s | 149.70, 136.07, 135.14 | 29.1 GiB |
| `b2_cache_cond` | 135.36 s | 154.80, 136.07, 134.66 | 29.7 GiB |
| `b3_fast_prep` | 134.63 s | 151.74, 133.81, 135.45 | 29.7 GiB |
| `b4_shard_prep_MAXED` | **131.35 s** | 146.19, 132.46, 130.24 | 29.7 GiB |
| `b4_noadaln_nooutput` *(ablation)* | 140.47 s | 157.78, 140.87, 140.06 | **38.5 GiB** |
| `mesh_cp2tp4` | **129.49 s** | 144.20, 130.73, 128.25 | 25.9 GiB |
| `mesh_cp1tp8` | 136.70 s | 145.63, 137.79, 135.61 | 29.0 GiB |
| `mesh_cp8tp1` | **OOM** | — | — |

| ratio | value |
|---|---:|
| 1 GPU -> 8 GPU, maxed (`b4`) | **2.28x** |
| 1 GPU -> 8 GPU, best mesh (`cp2tp4`) | **2.31x** |
| 8 GPU DiT-shard-only -> maxed | **1.36x** |
| 8 GPU DiT-shard-only -> best mesh | **1.38x** |

## 30 s clip (NF=721), 8x A100-SXM4-**80GB** (`luna-prod-78-80gb`)

| config | warm median | all iterations | peak GPU |
|---|---:|---|---:|
| `1gpu_flextensor` | **700.83 s** | 709.18, 701.30, 700.36 | 42.5 GiB |
| `b0_dit_shard_only` | 309.42 s | 327.95, 310.25, 308.59 | 35.3 GiB |
| `b1_vae_gemma_decode` | 231.69 s | 242.00, 231.50, 231.88 | 35.3 GiB |
| `b2_cache_cond` | 204.73 s | 217.69, 205.66, 203.79 | 35.9 GiB |
| `b3_fast_prep` | 223.79 s | 233.60, 222.70, 224.87 | 35.9 GiB |
| `b4_shard_prep_MAXED` | **192.59 s** | 206.15, 193.08, 192.11 | 35.9 GiB |
| `b4_noadaln_nooutput` *(ablation)* | 223.16 s | 237.45, 227.48, 218.83 | **53.2 GiB** |
| `mesh_cp2tp4` | 210.90 s | 223.05, 212.76, 209.03 | 35.1 GiB |
| `mesh_cp1tp8` | 214.80 s | 224.67, 215.29, 214.30 | 45.6 GiB |
| `mesh_cp8tp1` | **OOM** | — | — |

| ratio | value |
|---|---:|
| 1 GPU -> 8 GPU, maxed | **3.64x** |
| 8 GPU DiT-shard-only -> maxed | **1.61x** |

**The 1-GPU denominator reproduces.** An earlier measurement on different hardware (since
decommissioned) recorded 722.80 s for this configuration; this run gives 700.83 s — within 3%, from
an sbatch script on a node anyone can allocate.

### The per-step attribution does not survive scrutiny

`b3_fast_prep` (223.79 s) is *slower* than `b2_cache_cond` (204.73 s), which cannot be a real effect:
fast-prep only replaces an O(n^2) `torch.cat` with an O(n) one. The DiT stage-1 timing explains it —
19.0 s, 29.2 s, 21.1 s in `b2`, `b3`, `b4` respectively, across configurations that do not touch the
DiT at all.

Each configuration's own iterations are tight (`b3`: 222.70, 224.87), so this is not sampling noise
within a run. It is variance **between process launches** — every configuration is a separate
`torch.distributed.run`, with its own NCCL setup and memory layout.

**Therefore: the `b1 -> b2 -> b3 -> b4` per-step numbers should not be quoted as an attribution of
individual optimizations.** The separation between them is smaller than the launch-to-launch spread.
What survives is the endpoint comparison, `b0 -> b4`, where the gap is several times the spread.
Resolving the individual steps needs multiple interleaved launches per configuration, which this
harness does not yet do.

## 30 s clip (NF=721), 8x A100-SXM4-40GB — the customer target

| config | warm median | all iterations | peak GPU |
|---|---:|---|---:|
| `1gpu_flextensor` | **OOM** | — | — |
| `b0_dit_shard_only` | 281.30 s | 290.01, 284.02, 278.57 | 35.3 GiB |
| `b1_vae_gemma_decode` | 200.56 s | 215.22, 202.87, 198.24 | 35.3 GiB |
| `b2_cache_cond` | 195.21 s | 216.50, 194.69, 195.74 | 35.9 GiB |
| `b3_fast_prep` | 194.00 s | 211.90, 195.73, 192.27 | 35.9 GiB |
| `b4_shard_prep_MAXED` | **192.11 s** | 206.90, 193.89, 190.34 | 35.9 GiB |
| `b4_noadaln_nooutput` *(ablation)* | **OOM** | — | — |
| `mesh_cp2tp4` | **186.89 s** | 199.71, 189.51, 184.26 | 35.1 GiB |
| `mesh_cp1tp8` | **OOM** | — | — |
| `mesh_cp8tp1` | **OOM** | — | — |

| ratio | value |
|---|---:|
| 8 GPU DiT-shard-only -> maxed | **1.46x** |
| 8 GPU DiT-shard-only -> best mesh | **1.51x** |
| 1 GPU -> 8 GPU | **not measurable** — see below |

## Independent reproduction on a second machine

Everything above was measured on `luna-prod-48`. To check that the numbers are a property of the
work rather than of that machine, the whole matrix was re-run on **`luna-dvt-43`** (8x A100-SXM4-40GB,
NVSwitch NV12) — a box that had never seen this project, provisioned from a bare OS, from a clean
clone of this branch, with **no conditioning footage supplied** (the synthetic clip described in
README.md).

| config | `luna-prod-48` | `luna-dvt-43` | delta |
|---|---:|---:|---:|
| `1gpu_flextensor` | OOM | OOM | matches |
| `b0_dit_shard_only` | 281.30 s | 280.85 s | -0.2% |
| `b1_vae_gemma_decode` | 200.56 s | 206.11 s | +2.8% |
| `b2_cache_cond` | 195.21 s | 193.44 s | -0.9% |
| `b3_fast_prep` | 194.00 s | 187.72 s | -3.2% |
| `b4_shard_prep_MAXED` | 192.11 s | 177.60 s | -7.6% |
| `b4_noadaln_nooutput` | OOM | OOM | matches |
| `mesh_cp2tp4` | 186.89 s | 180.61 s | -3.4% |
| `mesh_cp1tp8`, `mesh_cp8tp1` | OOM | OOM | matches |
| **everything on** (`PHASE=best`) | **117.6 s** | **118.05 s** | **+0.4%** |

Re-verified after the provisioning rework (`FLEXTENSOR_SRC` / editable install), because that
changed the path a new machine takes: the venv, the LTX-2 checkout, the staged example tree and the
inputs were deleted and rebuilt from a clean clone. Three independent `PHASE=best` measurements of
the same configuration now stand at **118.05 s**, **116.32 s** and **116.71 s** against a published
117.6 s, all at 35.9 GiB peak. (The weight cache was retained via `SEED_HF` in the later two runs,
so those exercise a fresh environment with seeded weights; the cold download was exercised in the
first.)

Peak memory agreed to 0.1 GiB on every row that ran, all parity gates passed, and each of the four
OOMs was confirmed to be a genuine CUDA OOM rather than a different failure wearing the same label —
so the configurations that are documented *not* to fit reproduce as faithfully as the ones that do.

Two honest caveats:

- **The mesh ordering inverted.** CP2xTP4 wins on `luna-prod-48` (186.89 vs 192.11); CP4xTP2 wins on
  `luna-dvt-43` (177.60 vs 180.61). The margin is ~1.7% either way, comparable to the run-to-run
  spread, so the supportable claim is that the two meshes are *equivalent* at 30 s on 40 GB — not
  that either reliably wins. Do not quote a preference between them from a single machine.
- These runs use the synthetic conditioning clip, so the output mp4s do not match the published ones
  visually. Timing is unaffected: the workload is fixed by canvas, frame count and frame rate, and
  parity is always measured against a reference produced from the same input in the same job.

## Headline: everything on, 30 s clip

Both rows of each pair measured in the **same job, same day, same code**.

| | 8x A100-80GB | 8x A100-40GB |
|---|---:|---:|
| 1 GPU (FlexTensor weight streaming) | 701.34 s | **OOM — does not run** |
| 8 GPU, all optimizations | **127.22 s** | **117.6 s** |
| **speedup** | **5.51x** | not expressible |

The 80 GB figure exceeds the 4.96x previously recorded for this workload, on a script anyone can
run. The 40 GB column has no ratio because a single 40 GB card cannot run a 30 s 1280x704 outpaint at
all, even with weight streaming — the honest statement there is *does not run on one card, runs on
eight in 117.6 s*.

At 15 s on 40 GB, where one card does manage it: **299.12 s -> 60.05 s = 4.98x**.

## Pipeline ladder — attribution, re-measured

The first attempt at this was retracted: launch-to-launch spread exceeded the gap between steps.
With flat weight movement the spread collapsed to well under a second, so the steps were re-run
interleaved across two rounds. 30 s clip, 40 GB, on top of resident DiT + flat movement:

| step | round 1 | round 2 | gain |
|---|---:|---:|---:|
| `b1` VAE tile-shard + Gemma broadcast + balanced decode | 130.30 s | 129.70 s | — |
| `b2` + cached reference conditioning | 120.35 s | 119.59 s | **-10.0 s** |
| `b3` + O(n) reference assembly | 118.68 s | 119.03 s | -1.1 s |
| `b4` + per-rank sharded reference decode | 118.01 s | 119.21 s | -0.25 s |

Only **`b2` is now resolvable** — 10 s, an order of magnitude above the ~1 s round-to-round spread.
`b3` is marginal and `b4` is inside the noise.

That is not a contradiction of the earlier claim that those two were worth tens of seconds: caching
the reference conditioning (`b2`) subsumes them. Both optimize decoding and assembling the reference
clip, which after `b2` happens once on the cold iteration — and the cold iteration is discarded. They
still matter for a server's first request; they do not matter for steady-state throughput.

## 30 s clip (NF=721), 8x A100-40GB — with the weight lifecycle fixed

The matrix above rebuilds the 22B DiT on every generate. Keeping it resident and moving it with
pinned flat buffers changes the picture substantially. Interleaved repeat launches:

| configuration | wall | weight move | peak |
|---|---:|---:|---:|
| maxed, DiT rebuilt per generate (`b4` above) | 192.11 s | — | 35.9 GiB |
| maxed, resident DiT, `module.to` | 156.78 / 155.08 s | 38.3 s | 35.9 GiB |
| **maxed, resident DiT, flat move** | **116.98 / 118.16 s** | **5.4 s** | 35.9 GiB |

Output byte-identical throughout (cosine 1.000000).

| ratio, 30 s / 40 GB | value |
|---|---:|
| DiT-shard-only -> best | **2.39x** |
| maxed-with-rebuild -> best | **1.63x** |
| 1 GPU -> best | still not measurable (1 GPU OOMs) |

## `SHARD_ON_CPU` on 80 GB — interleaved A/B, 30 s

| round | build | wall | peak |
|---|---|---:|---:|
| 1 | host | 214.56 s | 35.9 GiB |
| 1 | device | **169.97 s** | 43.0 GiB |
| 2 | host | 218.00 s | 35.9 GiB |
| 2 | device | **170.93 s** | 43.0 GiB |

Building on the host costs **1.27x** on 80 GB and saves 7.1 GiB. It is mandatory at 40 GB and should
be off at 80 GB — `run_optimized.sbatch` selects it from the card.

---

## Reading these numbers honestly

**There is no 1-GPU number at 30 s on 40 GB.** It OOMs even with FlexTensor weight streaming (a
4.86 GiB allocation fails). So the honest statement for the customer configuration is not a speedup
ratio at all: *a 30 s 1280x704 outpaint does not run on a single A100-40GB, and does run on eight in
192 s.* Quoting an 8-vs-1 ratio at 30 s on this hardware would require inventing a denominator.

**The "~1.3x" figure is the DiT-shard-only baseline.** `b0` is 8 GPUs with the DiT sharded and every
other stage still running redundantly on each rank — which is what a CP-only server does today. Going
from there to the maxed pipeline is **1.36x at 15 s and 1.46x at 30 s**. That is the number that
should be quoted when the question is "what did the pipeline work buy on top of context parallelism",
and it is not the same question as "how much faster is 8 GPUs than 1".

**Where the b0 -> b1 jump comes from** is visible in the stage breakdown at 30 s, and this one is
large enough to be safe from the launch variance discussed below: VAE encode 38.6 s -> 7.6 s, VAE
decode 51.5 s -> 12.2 s, Gemma 15.7 s -> 9.5 s. Tile-sharding the VAE and loading Gemma once instead
of eight times is the bulk of the improvement. The individual contributions of `b2`, `b3` and `b4`
are **not** separable at this sample size.

**Caveat on `b0`:** it runs with `SHARD_ADALN` and `SHARD_OUTPUT` enabled, because without them the
30 s case does not fit at all and there would be no `b0` row to compare against. So `b0` is
"DiT-sharded-only pipeline, on a memory footprint that fits" rather than a pristine upstream
configuration. The `b4_noadaln_nooutput` row is the clean ablation for those two changes.

**The memory work is worth different things at different sizes.** At 15 s on 40 GB it buys 1.07x and
8.8 GiB of headroom (38.5 -> 29.7 GiB). At 30 s on 40 GB it is the difference between running and not
running. At 30 s on 80 GB, where nothing is forced, it still removes **17.3 GiB** of peak (53.2 ->
35.9 GiB) and 1.16x of wall.

**Mesh preference inverts with card size.** CP4xTP2 wins on 80 GB (192.6 s vs 210.9 s for CP2xTP4);
CP2xTP4 wins on 40 GB (186.9 s vs 192.1 s). Pick per target, do not carry a mesh across.

**On 40 GB, CP2xTP4 is the best mesh** at both clip lengths (129.5 vs 131.4 s at 15 s; 186.9 vs
192.1 s at 30 s) and uses less memory; on 80 GB the ordering reverses (see above). CP8 never fits on
40 GB and OOMed on 80 GB as well: pure context parallelism replicates the ~44 GiB bf16 DiT on every
rank. Pure tensor parallelism (`cp1tp8`) fits at 15 s but is slower, and OOMs at 30 s on 40 GB
because it reduces no rank's activation footprint.

**Scaling is sub-linear and fixed costs matter.** 8 GPUs give 2.28x at 15 s on 40 GB and 3.64x at
30 s on 80 GB, not 8x. The pipeline rebuilds the 22B DiT on every generate, which no amount of
sharding removes; the longer the clip, the better that fixed cost amortizes, which is most of why
the 30 s ratio is higher than the 15 s one.

## Parity

Every configuration's output was compared against the maxed reference with `parity_gate.py`, which fails the job below the bar.
Configurations differing only in bit-exact optimizations return `cosine=1.000000 / PSNR 99 dB`.
`SHARD_OUTPUT` is the one change that is *not* bit-exact (cosine 0.998949 / PSNR 36.0 dB — see
[`README.md`](README.md) for why), which is above the 0.995 acceptance bar but should not be
described as exact.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# LTX 2.3 outpaint — reproducible E2E speedup benchmark

Everything here is designed so that someone who did not run it can check the numbers. One `sbatch`
per phase, on a named node, with the environment fingerprinted and every configuration
parity-checked against the reference output.

## What is measured

The benchmark reports **end-to-end wall time for one outpaint request**: reference decode, prompt
encoding, IC-LoRA conditioning, both diffusion stages, spatial upsampling, and video/audio decode.
Not denoise-loop-only.

Two independent numbers come out of it, and they should never be quoted interchangeably:

| ratio | denominator | what it answers |
|---|---|---|
| **8-GPU vs 1-GPU** | one GPU running the same clip | "how much does the extra hardware buy?" |
| **optimized vs unoptimized 8-GPU** | 8-GPU with only the DiT sharded | "how much did the pipeline work buy?" |

A DiT-sharded-only 8-GPU pipeline — every non-DiT stage still running redundantly on all ranks — is
the `b0_dit_shard_only` row. Comparing an optimized run against *that* is where a ~1.3x figure comes
from; comparing against a single GPU is a different and much larger number. Both are in the results
table so nobody has to guess which is which.

## Methodology

Held constant across every configuration:

- **Same input.** `real_input_30s.mp4`, built deterministically from `real_input_8s.mp4` by
  `-stream_loop 4 -t 30` (720 frames, 1280x704, 24 fps). Its md5 is recorded in
  `out/environment.txt`.
- **Same seed** (171198), same resolution, same frame count, same decode tiling.
- **`ITERS=3`, iteration 0 discarded.** The block-based pipeline rebuilds and frees the 22B DiT on
  every generate, so iter0 pays extra CUDA/`av` warmup. The reported number is the warm median, and
  **every individual iteration is printed** next to it so the spread is visible.
- **Peak GPU memory** reported per configuration.
- **Parity gate.** Every configuration's output mp4 is compared to the reference by
  `parity_gate.py`, which **exits non-zero below the threshold** and fails the job -- a parity
  regression cannot reach `..._DONE`. Changes claimed bit-exact are gated with `--exact`
  (`cosine=1.000000 / PSNR 99 dB`); everything else is gated at `--min-cosine` (default 0.995,
  override with `PARITY_MIN_COSINE`). The sharding meshes differ from a single-GPU run only by bf16
  collective reordering, which lands well inside that bar.

Recorded per run in `out/environment.txt`: date, host, Slurm job id, driver version, full
`nvidia-smi topo -m` (so a reviewer can confirm **NV12/NVSwitch** rather than PCIe — Ulysses
all-to-all over PCIe is roughly break-even and would invalidate the comparison), torch version,
LTX-2 commit, input md5 and script md5.

### Why cosine and not max-abs

For this 48-layer bf16 DiT, cosine is the meaningful acceptance metric. Perturbing the eager
reference's input by 0.4% — bf16 rounding scale — already moves video cosine to ~0.9957. A
sharded-vs-single-GPU cosine of ~0.997 is therefore *closer* to the reference than bf16 noise
itself, while max-abs relative error looks alarming for reasons that have nothing to do with the
sharding. Acceptance bar: **cosine >= 0.995**, and **1.000000** for anything claimed bit-exact.

## Running it

Everything needed is in this directory. There is no dependency on a particular cluster, a
pre-staged scratch tree, or files from anyone's home directory.

**Prerequisites**

- A Slurm partition with 8 GPUs on one node, NVSwitch-connected. `nvidia-smi topo -m` must show
  `NV#` between all eight; Ulysses all-to-all over PCIe is roughly break-even and would invalidate
  the comparison. The job records the topology so this is checkable after the fact.
- `HF_TOKEN` with access to `google/gemma-3-12b-it-qat-q4_0-unquantized`, which is gated. Not needed
  if you point `SEED_HF` at a cache that already holds it.
- A **conditioning clip** — but you do not have to supply one. The published numbers were measured
  on customer footage that cannot be committed, so by default the harness *synthesises* a conforming
  clip (`make_input.py`) and says so loudly in the log and in `environment.txt`
  (`synthetic_input=yes`). The workload is fixed by canvas, frame count and frame rate rather than
  by content, so the timing comparison reproduces exactly; only the appearance of the output
  differs. Parity is unaffected, because each configuration is compared against a reference
  generated from the same input in the same run — never against our published mp4s.

  To use real footage instead, put a letterboxed 1280x704 clip of at least 8 s at
  `inputs/real_input_8s.mp4`, with pure-black bars in the regions to outpaint. Build one with the
  helper that ships next to this example:
  ```bash
  mkdir -p inputs
  python ../letterbox.py --input <source>.mp4 --output inputs/real_input_8s.mp4 --width 1280 --height 704
  ```
  Set `SYNTH_INPUT=0` to make a missing clip a hard error rather than synthesising one.

  The 30 s clip used by the benchmark is derived from the 8 s one deterministically
  (`-stream_loop 4 -t 30 -c copy`) and verified to decode to exactly 720 frames on every run. The
  copy is a bitstream copy on purpose: re-encoding bleeds the inner content into the letterbox bars
  (measured peak 14 rather than 0), and the Outpaint IC-LoRA fills regions that are *exactly*
  RGB(0,0,0).
- Outbound network on the compute node for the first run: it clones LTX-2 from GitHub, installs the
  environment with `uv`, and pulls ~68 GB of weights from Hugging Face. Everything else is here.

**Run**

```bash
cd examples/ltx23/outpaint/benchmarks
export HF_TOKEN=...

# E2E speedup matrix, 30 s clip
sbatch -p <partition> --export=ALL,PHASE=bench,NF=721,ITERS=3 run_all.sbatch

# every optimization on, one run
sbatch -p <partition> --export=ALL,PHASE=best,NF=721,ITERS=3 run_all.sbatch

# DiT backends: eager vs torch.compile vs Torch-TensorRT
sbatch -p <partition> --export=ALL,PHASE=ditbe,NF=361,ITERS=3 run_all.sbatch

# bit-exactness ablation on a short clip
sbatch -p <partition> --export=ALL,PHASE=move,NF=49,ITERS=3 run_all.sbatch
```

Submit from this directory: `LTX_SRC` defaults to Slurm's submit directory, and the job fails
immediately with a clear message if it cannot find `integrated_e2e.py` next to it.

**Without Slurm.** Slurm is a convenience here, not a requirement — the scripts never call `srun`,
and every `SLURM_*` variable they read is defaulted. On a bare 8-GPU box, run them as ordinary shell
scripts and pass the settings as environment variables:

```bash
cd examples/ltx23/outpaint/benchmarks
export HF_TOKEN=...
PHASE=bench NF=721 ITERS=3 bash run_all.sbatch
```

`LTX_SRC` then falls back to `$PWD`, so run them from this directory or set it explicitly. Logs go to
`$LTX_LOGS/run_0` rather than `run_<jobid>`, since there is no job id to name them after.

**Knobs**

| variable | default | meaning |
|---|---|---|
| `LTX_SRC` | Slurm submit dir | directory holding these scripts |
| `LTX_WORK` | `/tmp/ltx_bench` | node-local scratch for venv, weights, outputs |
| `LTX_LOGS` | `$LTX_SRC/../_bench_logs` | where per-config logs are copied so they outlive the allocation |
| `FLEXTENSOR_SRC` | repository containing `LTX_SRC` | FlexTensor checkout installed into the LTX environment; set this explicitly when the benchmark scripts are copied elsewhere |
| `SEED_HF` | unset | a populated HF hub cache to rsync from instead of downloading |
| `NF` | `721` | frame count, must be `8k + 1`: `49` (2 s), `361` (15 s), `721` (30 s) |
| `ITERS` | `3` | iterations per configuration; iteration 0 is always discarded |
| `PARITY_MIN_COSINE` | `0.995` | acceptance bar for the parity gate |
| `LTX2_COMMIT` | `4f89057` | LTX-2 revision to build against |

`LTX2_COMMIT` is pinned on purpose. Cloning LTX-2's default branch made this benchmark silently
unreproducible: upstream replaced `ICLoraPipeline`'s keyword arguments with a `ModelPaths` object on
2026-08-11, so a clone taken after that date fails with `TypeError: ICLoraPipeline.__init__() got an
unexpected keyword argument 'distilled_checkpoint_path'`. `4f89057` (2026-08-03) is the last
revision with the signature this example targets, and the one the published numbers were measured
against. Point `LTX2_COMMIT` at a newer revision to test against current upstream.

The first run provisions `$LTX_WORK` (~30 min, mostly the weight download) and reuses it afterwards.
Compute nodes wipe `/tmp` between allocations, so provisioning happens **in the same job** as the
work rather than as a separate step — a separate provisioning job can leave a later benchmark
pointing at a venv that no longer exists, which fails as `rc=127` and reads like a benchmark failure.
Set `SEED_HF` to a shared cache to skip the download entirely.

**Outputs** land in `$LTX_WORK/out` and are copied to `$LTX_LOGS/run_<jobid>/`: one
`<phase>_<config>.log` and one mp4 per configuration, plus `environment_<jobid>.txt` with the
fingerprint. `PHASE=bench` additionally writes `progress.txt` and `results.jsonl`.

## The matrix

| row | what it isolates |
|---|---|
| `1gpu_flextensor` | the denominator. A 30 s clip does not fit one card natively (22B weights plus full-sequence stage-2 activations), so "one GPU" means FlexTensor weight streaming. |
| `b0_dit_shard_only` | 8 GPUs, DiT sharded, every other stage redundant per rank. |
| `b1_vae_gemma_decode` | + VAE tile-sharding, Gemma load-once-and-broadcast, cost-balanced decode. |
| `b2_cache_cond` | + VAE encoder built once and kept resident, reference conditioning cached. |
| `b3_fast_prep` | + O(n) reference assembly (the stock path is a `torch.cat` inside a 720-frame loop). |
| `b4_shard_prep_MAXED` | + per-rank seek-decode of the reference clip. |
| `b4_noadaln_nooutput` | ablation: maxed *without* the AdaLN / output sharding, to price those two. |
| `mesh_cp8tp1`, `mesh_cp2tp4`, `mesh_cp1tp8` | mesh sweep at maxed settings. |

Each `b*` row after `b1` differs from the one above it by exactly one flag. `b1` itself turns on
three at once (VAE tile-shard, Gemma broadcast, balanced decode), so it is a group rather than a
single-flag step.

Even so, the per-flag attribution beyond `b1` does **not** resolve at this sample size — see
`RESULTS.md`. The group totals are sound; the individual steps are not.

## Memory notes for 40 GB

The 22B DiT is ~44 GiB in bf16, so on a 40 GB card three things are load-bearing:

1. **`SHARD_ON_CPU=1`** — the stock build loads the full unsharded model *onto the GPU* and only
   then slices it, so the build spike is 44 GiB even with tensor parallelism. Building and sharding
   on the host and moving only the per-rank shard is what makes 40 GB viable.
2. **`SHARD_ADALN=1`** — the AdaLN timestep projections run before any context-parallel sharding,
   at full sequence length on every rank (~10.9 GiB plus ~4.9 GiB at 30 s).
3. **`SHARD_OUTPUT=1`** — gathers after `proj_out` rather than after the transformer blocks, so the
   per-token output tail runs on the shard.

`validate.sbatch` measures each one's effect on the output:

| flag | parity vs unmodified path | note |
|---|---|---|
| `SHARD_ON_CPU` | bit-exact | same slices, same hooks, just built on the host first |
| `SHARD_ADALN` | **bit-exact** (identical output md5, cosine 1.000000) | per-token op evaluated on fewer rows |
| `SHARD_OUTPUT` | **not bit-exact** -- cosine 0.998949 / PSNR 36.0 dB | see below |

`SHARD_OUTPUT` changes the token count that `proj_out`'s GEMM sees (a shard rather than the full
sequence), so cuBLAS can select a different kernel and accumulate bf16 in a different order. The
result sits at the same ~36 dB / cosine ~0.999 level as context-parallel sharding itself does
against a single-GPU run, comfortably over the 0.995 bar, and the per-frame PSNR spread is uniform
(34.6-36.5 dB) rather than structured -- which is what distinguishes accumulation noise from an
ordering bug. It should not be described as bit-exact.

It is nonetheless what makes 30 s fit: at NF=721 on 8x A100-40GB the run completes in 211.34 s at
35.9 GiB peak with it, and OOMs without it.

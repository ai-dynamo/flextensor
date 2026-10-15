#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=ltx_opt
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:8 --time=06:00:00
#
# LTX 2.3 outpaint, end to end, with every validated optimization enabled.
#
#   sbatch -p <partition> -w <node> --export=ALL,NF=721 run_optimized.sbatch
#
# One setting is chosen from the card:
#
#   SHARD_ON_CPU  build each DiT stage on the host and move only the per-rank shard to the device.
#                 Required at 40 GB -- the stock path spikes at the full ~44 GiB on device. Costs
#                 1.27x on 80 GB, where the card can take the direct build, so it is off there.
#
# The mesh is CP4xTP2 on both cards; it is not card-dependent (see the note below).
#
# Everything else is on unconditionally. See OPTIMIZATIONS.md for what each flag does, what it is
# worth, and whether it is bit-exact.
set -u
# Paths are resolved from the environment so this runs on any cluster:
#   LTX_SRC   directory holding these scripts (default: the Slurm submit directory)
#   LTX_WORK  node-local scratch for the venv, weights and outputs (default: /tmp/ltx_bench)
#   LTX_LOGS  where per-config logs are copied so they survive the allocation
#             (default: $LTX_SRC/../_bench_logs)
#   SEED_HF   optional pre-populated Hugging Face cache to rsync from instead of downloading
LTX_SRC=${LTX_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}
BASE=${LTX_WORK:-/tmp/ltx_bench}
LOGS=${LTX_LOGS:-$LTX_SRC/../_bench_logs}
mkdir -p "$LOGS"
if [ ! -f "$LTX_SRC/integrated_e2e.py" ]; then
  echo "LTX_SRC=$LTX_SRC does not contain integrated_e2e.py."
  echo "Submit from the benchmarks directory, or set LTX_SRC to it."
  exit 1
fi
OUT=$BASE/out
NF=${NF:-721}
ITERS=${ITERS:-3}
PY=$BASE/LTX-2/.venv/bin/python

# Provisioning lives in run_all.sbatch, which does it in-job. Compute nodes wipe /tmp between
# allocations, so run that once on this node first if the environment is missing.
[ -x "$PY" ] || {
  echo "Environment missing on $(hostname). Run run_all.sbatch on this node first."
  exit 1
}

export PATH="$BASE/bin:$PATH"
export UV_CACHE_DIR=$BASE/uvcache
export HF_HOME=$BASE/hf HF_HUB_CACHE=$BASE/hf/hub HF_HUB=$BASE/hf/hub HF_HUB_OFFLINE=1
export EX_DIR=$BASE/ex/ltx23/outpaint
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0 NCCL_DEBUG=WARN PG_TIMEOUT=3600
export OUTDIR=$OUT COND=$BASE/inputs/real_input_30s.mp4
mkdir -p $OUT

MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$MEM_MIB" -lt 60000 ]; then
  CARD=40GB
  export SHARD_ON_CPU=1
else
  CARD=80GB
  export SHARD_ON_CPU=0
fi
# CP4xTP2 on both cards. An earlier reading had CP2xTP4 ahead at 40 GB, but that was measured with
# the DiT rebuilt every generate, where weight movement dominated the wall. Once movement is ~4 s the
# DiT's own cost decides, and CP2xTP4 loses: 4-way tensor parallelism adds an all-reduce to every
# linear (30 s / 40 GB: DiT 82.7 s vs 73.6 s, wall 127.1 s vs 117.6 s). The best mesh follows which
# cost dominates, not the card.
export TP=${TP:-2}
echo "card=$CARD (${MEM_MIB} MiB) -> SHARD_ON_CPU=$SHARD_ON_CPU TP=$TP (CP$((8 / TP))xTP$TP)"

# --- pipeline: stop every non-DiT stage from running redundantly on all 8 ranks ----------------
export ENC_SHARD=1      # VAE reference-encode: tile-shard across ranks          (bit-exact)
export DECODE_SHARD=1   # VAE decode: shard temporal tiles                        (bit-exact)
export DECODE_BALANCE=1 # cost-balanced (LPT) tile assignment                     (bit-exact)
export GEMMA_BCAST=1    # load Gemma once on rank 0 and broadcast contexts        (bit-exact)
export CACHE_COND=1     # build the VAE encoder once; cache reference conditioning(bit-exact)
export FAST_PREP=1      # O(n) reference assembly instead of cat-in-loop          (bit-exact)
export SHARD_PREP=1     # per-rank seek-decode of the reference clip              (bit-exact)
export DEC_TILE_PX=512 DEC_TILE_FR=24

# --- memory: keep full-sequence work off every rank -------------------------------------------
export SHARD_ADALN=1  # AdaLN timestep projections on the local shard           (bit-exact)
export SHARD_OUTPUT=1 # gather after proj_out, not after the blocks    (cosine 0.9989, not exact)

# --- weight lifecycle: the single largest win --------------------------------------------------
export RESIDENT_DIT=1 RESIDENT_DIT_MODE=cpu
export FLAT_MOVE=1 # pinned flat host buffers, one copy per dtype            (bit-exact)

# --- DiT backend --------------------------------------------------------------------------------
# inductor measured ~1.11x on DiT stage time and is not bit-exact (cosine 0.9984). Off by default so
# the shipped configuration stays as close to eager numerics as possible; set COMPILE_DIT=inductor
# to enable. TensorRT is not enabled: see OPTIMIZATIONS.md.
export COMPILE_DIT=${COMPILE_DIT:-off}
export TRT_DIT=0

export NF ITERS BENCH_TAG=optimized
echo "=== running NF=$NF ITERS=$ITERS ==="
set -o pipefail
$PY -m torch.distributed.run --nproc_per_node=8 --master_port=$((29500 + RANDOM % 400)) \
  $BASE/integrated_e2e.py 2>&1 | tee $OUT/optimized_nf${NF}.log | grep -aE "^RESULT|^iter |\[adaln\]|\[shard_output\]|\[compile_dit\]"
RUN_RC=${PIPESTATUS[0]}
# grep exits non-zero when it matches nothing, so key off the run itself and additionally require a
# RESULT line -- otherwise a crashed run still reaches the completion message below.
if [ "$RUN_RC" -ne 0 ] || ! grep -aq "^RESULT " $OUT/optimized_nf${NF}.log; then
  echo "OPTIMIZED_FAILED rc=$RUN_RC (log: $OUT/optimized_nf${NF}.log)"
  exit 1
fi

KEEP=$LOGS/run_${SLURM_JOB_ID:-0}
mkdir -p $KEEP
cp $OUT/optimized_nf${NF}.log $KEEP/ 2>/dev/null
echo "OPTIMIZED_DONE  (log: $KEEP)"

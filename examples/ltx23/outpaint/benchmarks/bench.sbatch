#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=ltx_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#
# LTX-2.3 outpaint E2E speedup matrix -- reproducible benchmark.
#
#   sbatch -p <partition> bench.sbatch
#
# Every config uses the SAME input, the SAME seed, ITERS=3 with iter0 discarded, and reports the
# warm median plus every individual iteration. Each config's output mp4 is parity-checked against
# the reference config with parity_gate.py, which FAILS the job below the acceptance threshold.

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
PY=$BASE/LTX-2/.venv/bin/python
NF=${NF:-361}
ITERS=${ITERS:-3}
SEED=171198
COND=$BASE/inputs/real_input_30s.mp4

export PATH="$BASE/bin:$PATH"
export HF_HOME=$BASE/hf HF_HUB_CACHE=$BASE/hf/hub HF_HUB=$BASE/hf/hub HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0 NCCL_DEBUG=WARN PG_TIMEOUT=1800
export OUTDIR=$OUT COND=$COND
mkdir -p $OUT $BASE/ex
# EX_DIR must hold serve_infer.py, which integrated_e2e.py imports. That is the example directory
# one level above this one, so copy the whole examples/ltx23 tree node-local. The old form copied
# "$LTX_SRC/../../ltx23" -- which resolves to examples/ltx23/ltx23 and does not exist -- and hid the
# failure behind "|| true", so EX_DIR stayed empty and the run died with ModuleNotFoundError.
# Stage the whole examples/ltx23 tree, not just outpaint/: serve_infer.py puts its parent on
# sys.path and imports _common from it, so the example is not self-contained in its own directory.
# Resolve the path with cd/pwd first -- "$LTX_SRC/../.." ends in "..", and cp -r names the
# destination after the source's last component, so the raw form copied the *contents* of
# examples/ltx23 into ex/ instead of creating ex/ltx23.
EX_PARENT=$(cd "$LTX_SRC/../.." && pwd)
mkdir -p $BASE/ex && cp -r "$EX_PARENT" $BASE/ex/
export EX_DIR=$BASE/ex/ltx23/outpaint
if [ ! -f "$EX_DIR/serve_infer.py" ] || [ ! -f "$EX_DIR/../_common.py" ]; then
  echo "EX_DIR=$EX_DIR is missing serve_infer.py or ../_common.py -- example tree not staged."
  exit 1
fi
cp "$LTX_SRC"/*.py $BASE/ 2>/dev/null

# ---------------------------------------------------------------- environment fingerprint
FP=$OUT/environment.txt
{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "slurm_job=${SLURM_JOB_ID:-none}"
  echo "nf=$NF iters=$ITERS seed=$SEED"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  echo "--- topology (want NV12 between all 8):"
  nvidia-smi topo -m 2>/dev/null | head -10
  echo "--- torch:"
  $PY -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
  echo "--- LTX-2 commit:"
  git -C $BASE/LTX-2 rev-parse HEAD 2>/dev/null
  echo "--- input md5:"
  md5sum $COND $BASE/inputs/real_input_8s.mp4
  echo "--- integrated_e2e.py md5:"
  md5sum $BASE/integrated_e2e.py
} >$FP 2>&1
cat $FP

# ---------------------------------------------------------------- one run
# run_cfg <tag> <nproc> <TP> <env assignments...>
run_cfg() {
  local tag=$1
  shift
  local nproc=$1
  shift
  local tp=$1
  shift
  local log=$OUT/bench_${tag}.log
  echo "=== [$(date -u +%T)] $tag nproc=$nproc TP=$tp $* ===" | tee -a $OUT/progress.txt
  (
    export NF=$NF ITERS=$ITERS TP=$tp BENCH_TAG=$tag TRT_DIT=0 SEED=$SEED
    export DEC_TILE_PX=512 DEC_TILE_FR=24 SHARD_ON_CPU=1
    for kv in "$@"; do export "$kv"; done
    $PY -m torch.distributed.run --nproc_per_node=$nproc --master_port=$((29800 + RANDOM % 150)) \
      $BASE/integrated_e2e.py
  ) >$log 2>&1
  local rc=$?
  local res=$(grep -a "^RESULT " $log | tail -1)
  if [ -z "$res" ]; then
    echo "FAIL $tag rc=$rc :: $(grep -aoE 'OutOfMemoryError.{0,90}' $log | tail -1)" | tee -a $OUT/progress.txt
    echo "{\"tag\":\"$tag\",\"ok\":false,\"rc\":$rc}" >>$OUT/results.jsonl
  else
    echo "$res" | tee -a $OUT/progress.txt
    echo "{\"tag\":\"$tag\",\"ok\":true,\"result\":\"$(echo $res | sed 's/"/\\"/g')\"}" >>$OUT/results.jsonl
  fi
}

rm -f $OUT/results.jsonl $OUT/progress.txt
PARITY_RC=0

# ---------------------------------------------------------------- 1-GPU denominator
# 30s does not fit one 40GB card natively -> FlexTensor weight streaming is what "one GPU" means here.
run_cfg 1gpu_flextensor 1 1 OFFLOAD_MODE=cpu ENC_SHARD=0 GEMMA_BCAST=0 DECODE_SHARD=0 \
  CACHE_COND=0 FAST_PREP=0 SHARD_PREP=0 SHARD_ADALN=0 SHARD_OUTPUT=0

# ---------------------------------------------------------------- 8-GPU progression, mesh fixed at CP4xTP2
# Each step adds exactly one optimization so the attribution is checkable.
run_cfg b0_dit_shard_only 8 2 ENC_SHARD=0 GEMMA_BCAST=0 DECODE_SHARD=0 CACHE_COND=0 FAST_PREP=0 SHARD_PREP=0 SHARD_ADALN=1 SHARD_OUTPUT=1
run_cfg b1_vae_gemma_decode 8 2 ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=0 FAST_PREP=0 SHARD_PREP=0 SHARD_ADALN=1 SHARD_OUTPUT=1
run_cfg b2_cache_cond 8 2 ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=0 SHARD_PREP=0 SHARD_ADALN=1 SHARD_OUTPUT=1
run_cfg b3_fast_prep 8 2 ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=1 SHARD_PREP=0 SHARD_ADALN=1 SHARD_OUTPUT=1
run_cfg b4_shard_prep_MAXED 8 2 ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=1 SHARD_PREP=1 SHARD_ADALN=1 SHARD_OUTPUT=1
# ablation: maxed WITHOUT the new 40GB AdaLN/output sharding, to show what those two are worth
run_cfg b4_noadaln_nooutput 8 2 ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=1 SHARD_PREP=1 SHARD_ADALN=0 SHARD_OUTPUT=0

# ---------------------------------------------------------------- mesh sweep at maxed settings
for tp in 1 4 8; do
  run_cfg mesh_cp$((8 / tp))tp${tp} 8 $tp ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=1 SHARD_PREP=1 SHARD_ADALN=1 SHARD_OUTPUT=1
done

# ---------------------------------------------------------------- parity: every config vs the maxed reference
REF=$OUT/integ_cp4tp2_eager_b4_shard_prep_MAXED_nf${NF}.mp4
echo "=== PARITY vs $(basename $REF) ===" | tee -a $OUT/progress.txt
for f in $OUT/integ_*_nf${NF}.mp4; do
  [ "$f" = "$REF" ] && continue
  [ -f "$REF" ] || {
    echo "reference missing, skipping parity"
    break
  }
  $PY $BASE/parity_gate.py "$REF" "$f" --label "$(basename $f)" \
    --min-cosine "${PARITY_MIN_COSINE:-0.995}" 2>&1 | tee -a $OUT/progress.txt
  [ "${PIPESTATUS[0]}" -eq 0 ] || PARITY_RC=1
done

echo "=== SUMMARY ==="
cat $OUT/progress.txt
if [ "$PARITY_RC" -ne 0 ]; then
  echo "BENCH_FAILED_PARITY"
  exit 1
fi
echo "BENCH_DONE"

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=ltx_validate
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:8 --time=03:00:00
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
export PATH="$BASE/bin:$PATH"
export HF_HOME=$BASE/hf HF_HUB_CACHE=$BASE/hf/hub HF_HUB=$BASE/hf/hub HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0 NCCL_DEBUG=WARN PG_TIMEOUT=1800
export OUTDIR=$OUT COND=$BASE/inputs/real_input_30s.mp4
export DEC_TILE_PX=512 DEC_TILE_FR=24
export SHARD_ON_CPU=1
mkdir -p $OUT
PARITY_RC=0
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
cp "$LTX_SRC"/*.py $BASE/
run() {
  local tag=$1
  shift
  (
    export NF=49 ITERS=1 TP=2 BENCH_TAG=$tag TRT_DIT=0
    for kv in "$@"; do export "$kv"; done
    $PY -m torch.distributed.run --nproc_per_node=8 --master_port=$((29900 + RANDOM % 90)) $BASE/integrated_e2e.py
  ) >$OUT/val_$tag.log 2>&1
  echo "[$tag] rc=$? :: $(grep -a '^RESULT' $OUT/val_$tag.log | tail -1)"
}
# reference: all opts, no new 40GB sharding
run ref SHARD_ADALN=0 SHARD_OUTPUT=0
# each new fix in isolation, then together
run adaln SHARD_ADALN=1 SHARD_OUTPUT=0
run outp SHARD_ADALN=0 SHARD_OUTPUT=1
run both SHARD_ADALN=1 SHARD_OUTPUT=1
echo "=== PARITY vs ref (want cosine 1.000000) ==="
for t in adaln outp both; do
  # adaln is claimed bit-exact so it is gated exactly. outp -- and therefore both -- changes
  # proj_out's GEMM shape and lands near 0.9989, so it is gated at the acceptance bar instead.
  if [ "$t" = "adaln" ]; then GATE=(--exact); else GATE=(--min-cosine "${PARITY_MIN_COSINE:-0.995}"); fi
  $PY $BASE/parity_gate.py $OUT/integ_cp4tp2_eager_ref_nf49.mp4 \
    $OUT/integ_cp4tp2_eager_${t}_nf49.mp4 --label "$t" "${GATE[@]}" || PARITY_RC=1
done
md5sum $OUT/integ_cp4tp2_eager_{ref,adaln,outp,both}_nf49.mp4 2>/dev/null
if [ "$PARITY_RC" -ne 0 ]; then
  echo VALIDATE_FAILED_PARITY
  exit 1
fi
echo VALIDATE_DONE

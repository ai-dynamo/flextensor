#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=ltx_all
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:8 --time=11:00:00
#
# Self-contained: provisions node-local scratch if it is missing, then runs the requested phase.
# Compute nodes' /tmp is wiped between allocations, so provisioning must live in the same job as
# the work -- otherwise a benchmark can start against a venv that no longer exists.
#
#   sbatch -p <partition> -w <node> --export=ALL,PHASE=bench,NF=721,ITERS=3 run_all.sbatch
#   sbatch -p <partition> -w <node> --export=ALL,PHASE=ditbe,NF=361,ITERS=3 run_all.sbatch
set -u
# Paths are resolved from the environment so this runs on any cluster:
#   LTX_SRC   directory holding these scripts (default: the Slurm submit directory)
#   LTX_WORK  node-local scratch for the venv, weights and outputs (default: /tmp/ltx_bench)
#   LTX_LOGS  where per-config logs are copied so they survive the allocation
#             (default: $LTX_SRC/../_bench_logs)
#   SEED_HF   optional pre-populated Hugging Face cache to rsync from instead of downloading
#   FLEXTENSOR_SRC  FlexTensor checkout to install (default: repository containing LTX_SRC)
LTX_SRC=${LTX_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}
BASE=${LTX_WORK:-/tmp/ltx_bench}
LOGS=${LTX_LOGS:-$LTX_SRC/../_bench_logs}
mkdir -p "$LOGS"
if [ ! -f "$LTX_SRC/integrated_e2e.py" ]; then
  echo "LTX_SRC=$LTX_SRC does not contain integrated_e2e.py."
  echo "Submit from the benchmarks directory, or set LTX_SRC to it."
  exit 1
fi
FLEXTENSOR_SRC=${FLEXTENSOR_SRC:-$(cd "$LTX_SRC/../../../.." && pwd)}
if [ ! -f "$FLEXTENSOR_SRC/pyproject.toml" ]; then
  echo "FLEXTENSOR_SRC=$FLEXTENSOR_SRC does not contain pyproject.toml."
  echo "Set FLEXTENSOR_SRC to the FlexTensor checkout under test."
  exit 1
fi
if ! git -C "$FLEXTENSOR_SRC" rev-parse HEAD >/dev/null 2>&1; then
  echo "FLEXTENSOR_SRC=$FLEXTENSOR_SRC is not a Git checkout."
  echo "Use a checkout so the benchmark can record the exact FlexTensor revision."
  exit 1
fi
OUT=$BASE/out
PHASE=${PHASE:-bench}
NF=${NF:-721}
ITERS=${ITERS:-3}
# Acceptance bar for this bf16 DiT; see README.md for why cosine and not max-abs.
PARITY_MIN_COSINE=${PARITY_MIN_COSINE:-0.995}
PARITY_RC=0
RUN_RC=0
# Tags that produced a RESULT this run, so parity can check every configuration that was
# supposed to run rather than only the mp4s that happen to exist.
RUN_TAGS=""
# Configurations documented not to fit; see the 40 GB table in RESULTS.md. Only these may fail
# without failing the phase, and only when the log actually shows a CUDA OOM -- any other failure,
# or an OOM anywhere else, is a real failure.
EXPECTED_OOM_TAGS=" 1gpu_flextensor best_1gpu b4_noadaln_nooutput mesh_cp8tp1 mesh_cp1tp8 "
export PATH="$BASE/bin:$PATH"
# uv defaults its cache to $HOME/.cache/uv. Home here is a 5 GB quota, and unpacking the TensorRT
# libs (~2 GB) fails with ENOSPC -- which surfaces much later as "No module named tensorrt" and
# looks like a TensorRT problem rather than a disk one. Keep the cache node-local.
export UV_CACHE_DIR=$BASE/uvcache UV_LINK_MODE=copy
export HF_HOME=$BASE/hf HF_HUB_CACHE=$BASE/hf/hub HF_XET_CACHE=$BASE/hf/xet
export HF_TOKEN=${HF_TOKEN:-} # only needed when the shared cache cannot seed the gated Gemma repo
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" $BASE/inputs $OUT $BASE/ex
# $OUT is node-local and reused across invocations. Clear previous videos so parity can never
# compare against, or silently pass on, an artifact from an earlier run -- a failed rerun would
# otherwise leave the previous run's mp4 in place and look like it still worked. Logs are kept:
# they are copied out per job id and are what a failure is diagnosed from.
rm -f $OUT/integ_*.mp4
PY=$BASE/LTX-2/.venv/bin/python

hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8

# ------------------------------------------------------------------ provision (idempotent)
if ! $PY -c "import torch, ltx_pipelines, flextensor" 2>/dev/null; then
  echo "=== provisioning $BASE ==="
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$BASE/bin sh
  export PATH="$BASE/bin:$PATH"
  # LTX-2 is pinned. Cloning its default branch made the benchmark silently
  # unreproducible: upstream replaced ICLoraPipeline's keyword arguments with a ModelPaths object on
  # 2026-08-11 (commit 2362161), so a later clone fails with
  #   TypeError: ICLoraPipeline.__init__() got an unexpected keyword argument 'distilled_checkpoint_path'
  # 4f89057 (2026-08-03) is the last commit with the signature this example targets, and is what the
  # published numbers were measured against. Override with LTX2_COMMIT to test a newer upstream.
  LTX2_COMMIT=${LTX2_COMMIT:-4f8905737aac86a554637cac86c178877a39c744}
  [ -d $BASE/LTX-2 ] || {
    git clone -q --filter=blob:none https://github.com/Lightricks/LTX-2.git $BASE/LTX-2
    git -C $BASE/LTX-2 checkout -q "$LTX2_COMMIT"
  }
  (
    cd $BASE/LTX-2
    export UV_PROJECT_ENVIRONMENT=$BASE/LTX-2/.venv UV_LINK_MODE=copy UV_CACHE_DIR=$BASE/uvcache
    uv sync >$BASE/uv_sync.log 2>&1 || {
      tail -100 $BASE/uv_sync.log
      exit 1
    }
  ) || exit 1
fi
# Install the checkout that supplied this harness, not a potentially different PyPI release.
# Do this even when an existing node-local environment imports flextensor, otherwise a prior PyPI
# installation can silently survive and the benchmark will not exercise the revision under review.
uv pip install --python $PY -q -e "$FLEXTENSOR_SRC" huggingface_hub hf_transfer imageio-ffmpeg 2>&1 | tail -1 || exit 1
$PY -c "import torch,ltx_pipelines,flextensor;print('VENV_OK torch',torch.__version__,'gpus',torch.cuda.device_count())" || exit 1

# Seed from a shared cache when one exists -- ~69 GB over NFS beats re-downloading, and it means
# a teammate with the cache does not need the gated Gemma token at all. SEED_HF can point anywhere.
SEED_HF=${SEED_HF:-} # optional; set to a populated HF hub cache to skip downloading
if [ -d "$SEED_HF" ] && ! ls $BASE/hf/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-22b-distilled-1.1.safetensors >/dev/null 2>&1; then
  echo "=== seeding weights from $SEED_HF ==="
  mkdir -p $BASE/hf/hub
  rsync -aL --info=progress2 "$SEED_HF/" $BASE/hf/hub/ 2>&1 | tail -2
fi

if ! ls $BASE/hf/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-22b-distilled-1.1.safetensors >/dev/null 2>&1 ||
  ! ls $BASE/hf/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-spatial-upscaler-x2-1.1.safetensors >/dev/null 2>&1 ||
  ! ls -d $BASE/hf/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*/ >/dev/null 2>&1; then
  echo "=== fetching missing weights from Hugging Face ==="
  $PY - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download
c = os.environ["HF_HUB_CACHE"]
snapshot_download("google/gemma-3-12b-it-qat-q4_0-unquantized", cache_dir=c); print("GEMMA_OK", flush=True)
for r, f in [("oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint", "ltx-2.3-22b-ic-lora-outpaint.safetensors"),
             ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
             ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-1.1.safetensors")]:
    hf_hub_download(r, f, cache_dir=c); print("OK", r, f, flush=True)
print("WEIGHTS_READY")
PYEOF
fi

# Gemma is gated and is fetched as a snapshot rather than a single file, so the LTX file checks
# above do not cover it. Without this the run starts and fails much later, mid-pipeline.
if ! ls -d $BASE/hf/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*/ >/dev/null 2>&1; then
  echo "Gemma snapshot missing under $BASE/hf/hub -- set HF_TOKEN and re-run, or seed from SEED_HF."
  exit 1
fi

cp "$LTX_SRC"/*.py $BASE/

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
# The conditioning clip is deliberately not committed: the measurements were taken on customer
# footage. Any letterboxed source works -- the workload is fixed by the canvas, frame count and
# frame rate, not by the content. Produce one with the letterbox helper next to this example:
#
#   python ../letterbox.py --input <source>.mp4 --output inputs/real_input_8s.mp4 \
#          --width 1280 --height 704
#
# It must be at least 8 s so the 30 s clip can be built by looping it.
cp "$LTX_SRC"/inputs/*.mp4 $BASE/inputs/ 2>/dev/null || true
# With no clip supplied, synthesise a conforming one rather than refusing to run: the timing
# comparison depends on the canvas, frame count and frame rate, not the content, so the speedup can
# be verified with nothing but this repository. Set SYNTH_INPUT=0 to demand real footage instead.
SYNTH_INPUT=${SYNTH_INPUT:-1}
if [ ! -f "$BASE/inputs/real_input_8s.mp4" ] && [ "$SYNTH_INPUT" = "1" ]; then
  echo "############################################################"
  echo "# No inputs/real_input_8s.mp4 -- generating a SYNTHETIC clip."
  echo "# Timings are comparable; the output will not look like the"
  echo "# published mp4s. Supply real footage for a visual match."
  echo "############################################################"
  $PY $BASE/make_input.py --output $BASE/inputs/real_input_8s.mp4 || exit 1
  SYNTHETIC_INPUT=yes
fi
if [ ! -f "$BASE/inputs/real_input_8s.mp4" ]; then
  echo "Missing $LTX_SRC/inputs/real_input_8s.mp4 -- a letterboxed 1280x704 clip of at least 8 s."
  echo "Create one with: python $LTX_SRC/../letterbox.py --input <source>.mp4 \\"
  echo "                        --output $LTX_SRC/inputs/real_input_8s.mp4 --width 1280 --height 704"
  echo "Or leave SYNTH_INPUT=1 (the default) to generate a synthetic clip instead."
  exit 1
fi

# 30 s conditioning clip. No ffmpeg on the compute nodes; imageio-ffmpeg ships a static binary.
# Rebuilt (and verified) every job so a truncated file can never silently change the workload.
# Copy the bitstream rather than re-encoding. Re-encoding at crf 16 / yuv420p bled the inner content
# into the letterbox bars (measured: peak 14, 8112 non-black pixels in frame 0), which silently
# undoes the lossless padding letterbox.py produces -- and the Outpaint IC-LoRA fills regions that
# are *exactly* RGB(0,0,0). A stream copy is both exact and faster.
FF=$($PY -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null | tail -1)
[ -n "$FF" ] && $FF -y -loglevel error -stream_loop 4 -t 30 -i $BASE/inputs/real_input_8s.mp4 \
  -c copy $BASE/inputs/real_input_30s.mp4
$PY -c "
import av, sys
c = av.open('$BASE/inputs/real_input_30s.mp4'); v = c.streams.video[0]
n = sum(1 for _ in c.decode(video=0))
print('COND_30S frames', n, v.codec_context.width, 'x', v.codec_context.height, float(v.average_rate))
sys.exit(0 if n == 720 else 1)" || {
  echo "COND_30S_BAD"
  exit 1
}

# ------------------------------------------------------------------ common run env
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0 NCCL_DEBUG=WARN PG_TIMEOUT=3600
export OUTDIR=$OUT COND=$BASE/inputs/real_input_30s.mp4
export HF_HUB=$BASE/hf/hub HF_HUB_OFFLINE=1 # integrated_e2e.py resolves checkpoints from HF_HUB
export DEC_TILE_PX=512 DEC_TILE_FR=24 SHARD_ON_CPU=1
export TORCHINDUCTOR_CACHE_DIR=$BASE/inductor_cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR

{
  echo "date_utc=$(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID:-none} phase=$PHASE nf=$NF iters=$ITERS"
  # Records whether the conditioning clip was synthesised, so a reviewer never has to guess why an
  # output does not look like the published mp4s.
  echo "synthetic_input=${SYNTHETIC_INPUT:-no}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  nvidia-smi topo -m 2>/dev/null | head -10
  $PY -c "import torch;print('torch',torch.__version__,torch.version.cuda)"
  git -C $BASE/LTX-2 rev-parse HEAD
  git -C "$FLEXTENSOR_SRC" rev-parse HEAD
  git -C "$FLEXTENSOR_SRC" status --short
  md5sum $COND $BASE/integrated_e2e.py
} >$OUT/environment_${SLURM_JOB_ID:-0}.txt 2>&1
cat $OUT/environment_${SLURM_JOB_ID:-0}.txt

run() {
  local tag=$1
  shift
  echo "=== [$(date -u +%T)] $tag ==="
  (
    export NF=$NF ITERS=$ITERS TP=${TP:-2} BENCH_TAG=$tag TRT_DIT=0
    for kv in "$@"; do export "$kv"; done
    $PY -m torch.distributed.run --nproc_per_node=${NPROC:-8} --master_port=$((29500 + RANDOM % 400)) \
      $BASE/integrated_e2e.py
  ) >$OUT/${PHASE}_$tag.log 2>&1
  local rc=$?
  local res=$(grep -a "^RESULT " $OUT/${PHASE}_$tag.log | tail -1)
  if [ -z "$res" ]; then
    echo "FAIL $tag rc=$rc"
    grep -aE "^\[rank[0-9]+\]: [A-Za-z_.]*(Error|Exception)" $OUT/${PHASE}_$tag.log | head -2 | sed "s/^/    /"
    grep -aE "^\[rank0\]:.*(, in |File \")" $OUT/${PHASE}_$tag.log | tail -4 | sed "s/^/    /"
    # A configuration documented not to fit may OOM without failing the phase. Everything else
    # must, otherwise a broken configuration reaches PHASE_*_DONE and its missing output is
    # silently skipped by the parity step instead of being compared.
    if [ "${EXPECTED_OOM_TAGS#* $tag }" != "$EXPECTED_OOM_TAGS" ] &&
      grep -aq "CUDA out of memory" $OUT/${PHASE}_$tag.log; then
      echo "    EXPECTED_OOM: $tag is documented not to fit here; not failing the phase."
    else
      RUN_RC=1
    fi
  else
    echo "$res"
    grep -aE "\[compile_dit\]" $OUT/${PHASE}_$tag.log | head -1
    # integrated_e2e.py now emits RESULT only after a successful encode, so an mp4 must exist.
    if ! ls $OUT/integ_*_${tag}_nf${NF}.mp4 >/dev/null 2>&1; then
      echo "FAIL $tag: RESULT reported but no mp4 was written"
      RUN_RC=1
    else
      RUN_TAGS="$RUN_TAGS $tag"
    fi
  fi
}

MAXED="ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=1 SHARD_PREP=1 SHARD_ADALN=1 SHARD_OUTPUT=1"

if [ "$PHASE" = "ladder" ]; then
  # Re-run the pipeline ladder on top of the resident+flat weight path. The first attempt at this
  # attribution was retracted: launch-to-launch spread (~8 s) exceeded the gap between adjacent
  # steps. Flat movement collapsed that spread to ~0.05 s, so the steps may now be separable.
  # Interleaved across two rounds so a difference can be attributed to the flag, not the slot.
  R="RESIDENT_DIT=1 RESIDENT_DIT_MODE=cpu FLAT_MOVE=1 SHARD_ADALN=1 SHARD_OUTPUT=1"
  for round in 1 2; do
    run l${round}_b1 $R ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=0 FAST_PREP=0 SHARD_PREP=0
    run l${round}_b2 $R ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=0 SHARD_PREP=0
    run l${round}_b3 $R ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=1 SHARD_PREP=0
    run l${round}_b4 $R ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=1 SHARD_PREP=1
  done
elif [ "$PHASE" = "best" ]; then
  # Everything on: pipeline + memory + resident DiT + flat weight movement, with the host/device
  # build choice taken from the card. This is the configuration run_optimized.sbatch ships.
  MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ "$MEM" -lt 60000 ]; then SOC=1; else SOC=0; fi
  TPB=${TP:-2} # CP4xTP2 on both cards once weight movement is no longer the dominant cost
  echo "card=${MEM}MiB -> SHARD_ON_CPU=$SOC TP=$TPB"
  run best TP=$TPB $MAXED SHARD_ON_CPU=$SOC RESIDENT_DIT=1 RESIDENT_DIT_MODE=cpu FLAT_MOVE=1
  # 1-GPU denominator in the same job, same day, same code
  run best_1gpu NPROC=1 TP=1 OFFLOAD_MODE=cpu ENC_SHARD=0 GEMMA_BCAST=0 DECODE_SHARD=0 \
    CACHE_COND=0 FAST_PREP=0 SHARD_PREP=0 SHARD_ADALN=0 SHARD_OUTPUT=0
  BEST_REF=$OUT/integ_cp$((8 / TPB))tp${TPB}_eager_best_nf${NF}.mp4
  BEST_BASELINE=$OUT/integ_cp1tp1_eager_best_1gpu_nf${NF}.mp4
  echo "=== PARITY best vs 1-GPU baseline (bar: cosine >= $PARITY_MIN_COSINE) ==="
  if [ -f "$BEST_REF" ] && [ -f "$BEST_BASELINE" ]; then
    $PY $BASE/parity_gate.py "$BEST_BASELINE" "$BEST_REF" --label best-vs-1gpu \
      --min-cosine "$PARITY_MIN_COSINE" || PARITY_RC=1
  elif [ ! -f "$BEST_REF" ]; then
    # The configuration under test produced nothing -- always a failure.
    echo "best output missing: $BEST_REF"
    PARITY_RC=1
  elif grep -aq "CUDA out of memory" "$OUT/${PHASE}_best_1gpu.log" 2>/dev/null; then
    # A 30 s clip does not fit one 40 GB card even with weight streaming, so on that card there is
    # no 1-GPU denominator to compare against -- see the 40 GB table in RESULTS.md. That is the
    # documented outcome, not a regression, and it must not fail the job: otherwise PHASE=best can
    # never pass on the customer-target card. The 8-GPU result above still stands on its own; it
    # is parity-checked against a same-mesh reference by PHASE=bench.
    echo "best-vs-1gpu SKIPPED: the 1-GPU baseline ran out of memory on a ${MEM} MiB card."
    echo "  A 30 s 1280x704 outpaint does not fit one card of this size; there is no denominator"
    echo "  to compare against. Re-run on an 80 GB card, or at a shorter clip (NF=361), for a ratio."
  else
    echo "best parity inputs missing: $BEST_BASELINE (1-GPU run failed for a reason other than OOM)"
    PARITY_RC=1
  fi
elif [ "$PHASE" = "move" ]; then
  # Price the host<->device weight movement for the CPU-resident DiT, interleaved so the flag is
  # separable from the launch slot.
  R="RESIDENT_DIT=1 RESIDENT_DIT_MODE=cpu"
  for round in 1 2; do
    run mv${round}_moduleto $MAXED $R FLAT_MOVE=0
    run mv${round}_flat $MAXED $R FLAT_MOVE=1
  done
  echo "=== PARITY flat vs module.to (want cosine 1.000000) ==="
  A=$OUT/integ_cp4tp2_eager_mv1_moduleto_nf${NF}.mp4
  Bv=$OUT/integ_cp4tp2_eager_mv1_flat_nf${NF}.mp4
  if [ -f "$A" ] && [ -f "$Bv" ]; then
    # flat movement copies the weights verbatim, so this one must be exact.
    $PY $BASE/parity_gate.py "$A" "$Bv" --label flat-vs-module.to --exact || PARITY_RC=1
    md5sum "$A" "$Bv"
  else
    echo "flat/module.to outputs missing"
    PARITY_RC=1
  fi
elif [ "$PHASE" = "ab" ]; then
  # Interleaved repeat launches. Every configuration is its own torch.distributed.run, and the
  # launch-to-launch spread turned out to exceed the gap between adjacent optimization steps, so a
  # single pass in a fixed order cannot attribute a difference to the flag rather than to the slot
  # it ran in. Alternating and repeating separates the two: if A and B differ consistently across
  # both rounds, the flag is responsible; if they drift together, the order is.
  for round in 1 2; do
    run ab${round}_hostbuild $MAXED SHARD_ON_CPU=1
    run ab${round}_devbuild $MAXED SHARD_ON_CPU=0
  done
elif [ "$PHASE" = "hostbuild" ]; then
  # SHARD_ON_CPU builds each DiT stage on the host and moves only the per-rank shard to the device.
  # That is mandatory on 40 GB (the stock path spikes at the full ~44 GiB on device) but on 80 GB
  # the card can take the direct build, so the host round-trip is pure overhead. Price it.
  run maxed_hostbuild $MAXED SHARD_ON_CPU=1
  run maxed_devbuild $MAXED SHARD_ON_CPU=0
  run cp2tp4_devbuild TP=4 $MAXED SHARD_ON_CPU=0
elif [ "$PHASE" = "ditbe" ]; then
  # torch-tensorrt matched to the venv's torch (2.9.1+cu128)
  if ! $PY -c "import torch_tensorrt" 2>/dev/null; then
    uv pip install --python $PY -q --no-deps --index-url https://download.pytorch.org/whl/cu128 "torch-tensorrt==2.9.0" 2>&1 | tail -2
    # tensorrt-cu12 is only on NVIDIA's index; from PyPI alone uv falls back to an sdist and the
    # build fails, leaving torch-tensorrt unable to dlopen libnvinfer.
    uv pip install --python $PY -q --extra-index-url https://pypi.nvidia.com \
      "tensorrt-cu12==10.13.3.9.post1" "tensorrt-cu12-bindings==10.13.3.9.post1" \
      "tensorrt-cu12-libs==10.13.3.9.post1" dllist 2>&1 | tail -3
  fi
  # tensorrt-cu12 ships libnvinfer under site-packages/tensorrt_libs; torch-tensorrt dlopens it and
  # will not find it unless that directory is on the loader path.
  TRT_LIBS=$($PY -c "import os,tensorrt_libs;print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null)
  [ -n "$TRT_LIBS" ] && export LD_LIBRARY_PATH="$TRT_LIBS:${LD_LIBRARY_PATH:-}"
  echo "TRT_LIBS=$TRT_LIBS"
  $PY -c "
import torch; print('torch', torch.__version__)
try:
    import torch_tensorrt, tensorrt, torch_tensorrt._features as f
    print('torch_tensorrt', torch_tensorrt.__version__, 'tensorrt', tensorrt.__version__)
    print('native_trt_collectives', f.ENABLED_FEATURES.native_trt_collectives)
except Exception as e:
    print('TORCH_TRT_UNAVAILABLE', repr(e))" 2>&1 | tail -4
  # DiT kept resident so compilation is paid once, not per generate.
  R="RESIDENT_DIT=1 RESIDENT_DIT_MODE=cpu"
  run eager $MAXED $R COMPILE_DIT=off
  run inductor $MAXED $R COMPILE_DIT=inductor
  run trt $MAXED $R COMPILE_DIT=trt
  echo "=== PARITY vs eager (bar: cosine >= 0.995) ==="
  for t in inductor trt; do
    A=$OUT/integ_cp4tp2_eager_eager_nf${NF}.mp4
    Bv=$OUT/integ_cp4tp2_eager_${t}_nf${NF}.mp4
    if [ -f "$A" ] && [ -f "$Bv" ]; then
      $PY $BASE/parity_gate.py "$A" "$Bv" --label "$t" --min-cosine "$PARITY_MIN_COSINE" || PARITY_RC=1
    else
      echo "$t: output missing"
      PARITY_RC=1
    fi
  done
else
  run 1gpu_flextensor NPROC=1 TP=1 OFFLOAD_MODE=cpu ENC_SHARD=0 GEMMA_BCAST=0 DECODE_SHARD=0 \
    CACHE_COND=0 FAST_PREP=0 SHARD_PREP=0 SHARD_ADALN=0 SHARD_OUTPUT=0
  run b0_dit_shard_only ENC_SHARD=0 GEMMA_BCAST=0 DECODE_SHARD=0 CACHE_COND=0 FAST_PREP=0 SHARD_PREP=0 SHARD_ADALN=1 SHARD_OUTPUT=1
  run b1_vae_gemma_decode ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=0 FAST_PREP=0 SHARD_PREP=0 SHARD_ADALN=1 SHARD_OUTPUT=1
  run b2_cache_cond ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=0 SHARD_PREP=0 SHARD_ADALN=1 SHARD_OUTPUT=1
  run b3_fast_prep ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=1 SHARD_PREP=0 SHARD_ADALN=1 SHARD_OUTPUT=1
  run b4_shard_prep_MAXED $MAXED
  run b4_noadaln_nooutput ENC_SHARD=1 GEMMA_BCAST=1 DECODE_SHARD=1 CACHE_COND=1 FAST_PREP=1 SHARD_PREP=1 SHARD_ADALN=0 SHARD_OUTPUT=0
  for tp in 1 4 8; do run mesh_cp$((8 / tp))tp${tp} TP=$tp $MAXED; done
  REF=$OUT/integ_cp4tp2_eager_b4_shard_prep_MAXED_nf${NF}.mp4
  echo "=== PARITY vs maxed reference ==="
  if [ ! -f "$REF" ]; then
    # Previously this broke out of the loop silently, leaving PARITY_RC at 0 -- so a run whose
    # reference never materialised still reported success.
    echo "PARITY reference missing: $REF"
    PARITY_RC=1
  else
    # Iterate the tags that ran, not the mp4s that happen to be present. Globbing compared only
    # existing files, so a configuration that produced nothing was skipped rather than flagged.
    for t in $RUN_TAGS; do
      f=$(ls $OUT/integ_*_${t}_nf${NF}.mp4 2>/dev/null | head -1)
      [ "$f" = "$REF" ] && continue
      if [ -z "$f" ]; then
        echo "$t: expected output missing"
        PARITY_RC=1
        continue
      fi
      $PY $BASE/parity_gate.py "$REF" "$f" --label "$(basename $f)" \
        --min-cosine "$PARITY_MIN_COSINE" || PARITY_RC=1
    done
  fi
fi
# Node-local /tmp is wiped between allocations, so per-config logs must be copied out or a failure
# becomes undiagnosable once the node moves on.
KEEP=$LOGS/run_${SLURM_JOB_ID:-0}
mkdir -p $KEEP && cp $OUT/${PHASE}_*.log $OUT/environment_*.txt $KEEP/ 2>/dev/null
echo "logs preserved in $KEEP"
if [ "$RUN_RC" -ne 0 ]; then
  echo "PHASE_${PHASE}_FAILED_RUN"
  exit 1
fi
if [ "$PARITY_RC" -ne 0 ]; then
  echo "PHASE_${PHASE}_FAILED_PARITY"
  exit 1
fi
echo "PHASE_${PHASE}_DONE"

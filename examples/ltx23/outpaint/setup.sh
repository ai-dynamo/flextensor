#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
BASE="${BASE:-$WORKSPACE_DIR/outpaint}"
EX="${EX:-$SCRIPT_DIR}"
FLEXTENSOR_DIR="${FLEXTENSOR_DIR:-$EX/../../..}"
LTX_DIR="${LTX_DIR:-$BASE/LTX-2}"
LTX2_COMMIT="${LTX2_COMMIT:-4f8905737aac86a554637cac86c178877a39c744}"
VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/.venv}"
INPUT_DIR="${INPUT_DIR:-$BASE/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/outputs}"
HF_HOME="${HF_HOME:-$BASE/ltx2.3/huggingface_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/.cache}"
CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"
PY="$VENV_DIR/bin/python"
export HF_HOME HF_HUB_CACHE XDG_CACHE_HOME
export HF_XET_CACHE
OFFLOAD_TEXT="${OFFLOAD_TEXT:-1}"
TEXT_MEM_FRACTION="${TEXT_MEM_FRACTION:-0.05}"
MAX_GPU_MEM_FRACTION="${MAX_GPU_MEM_FRACTION:-0.15}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
CONTROL_PLANE_TIMEOUT_SECONDS="${CONTROL_PLANE_TIMEOUT_SECONDS:-30}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING

SNAP="${SNAP:-}"
DISTILLED="${DISTILLED:-}"
UPSAMPLER="${UPSAMPLER:-}"
GEMMA="${GEMMA:-}"
LORA="${LORA:-}"

DISTILLED_FILENAME="ltx-2.3-22b-distilled-1.1.safetensors"
UPSAMPLER_FILENAME="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LORA_FILENAME="ltx-2.3-22b-ic-lora-outpaint.safetensors"
GEMMA_REPO_ID="google/gemma-3-12b-it-qat-q4_0-unquantized"
LTX_LICENSE_URL="https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE"

DEFAULT_PROFILE_DIR="$OUTPUT_DIR/ltx23_outpaint_profile"
if [[ "$CONTEXT_PARALLEL_SIZE" != "1" ]]; then
  DEFAULT_PROFILE_DIR="${DEFAULT_PROFILE_DIR}_cp${CONTEXT_PARALLEL_SIZE}"
fi
PROFILE_DIR="${PROFILE_DIR:-$DEFAULT_PROFILE_DIR}"
LETTERBOXED_VIDEO="${LETTERBOXED_VIDEO:-$INPUT_DIR/letterboxed_720p.mp4}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-49}"
FRAME_RATE="${FRAME_RATE:-24}"

validate_num_frames() {
  if [[ ! "$NUM_FRAMES" =~ ^[0-9]+$ ]] ||
    ((NUM_FRAMES < 1 || (NUM_FRAMES - 1) % 8 != 0)); then
    echo "NUM_FRAMES must be a positive integer of the form 8*k + 1; got: $NUM_FRAMES" >&2
    exit 1
  fi
}

validate_context_parallel_size() {
  case "$CONTEXT_PARALLEL_SIZE" in
    1 | 2 | 4 | 8) ;;
    *)
      echo "CONTEXT_PARALLEL_SIZE must be one of 1, 2, 4, or 8; got: $CONTEXT_PARALLEL_SIZE" >&2
      exit 1
      ;;
  esac
}

require_hf_token_for_gemma() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required because $GEMMA_REPO_ID is a gated Hugging Face repo." >&2
    echo "Request access on Hugging Face, then export HF_TOKEN before running setup.sh." >&2
    exit 1
  fi
}

require_external_license_ack() {
  if [[ "${ACCEPT_EXTERNAL_LICENSES:-}" == "1" ]]; then
    return
  fi

  echo "ACCEPT_EXTERNAL_LICENSES=1 is required before downloading external Hugging Face artifacts." >&2
  echo "Review and comply with upstream model terms before downloading these artifacts." >&2
  echo "LTX terms: $LTX_LICENSE_URL" >&2
  exit 1
}

require_path() {
  local label="$1"
  local path="$2"

  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

resolve_existing_path() {
  local explicit_path="$1"
  local fallback_path="$2"
  shift 2

  if [[ -n "$explicit_path" ]]; then
    echo "$explicit_path"
    return
  fi

  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done

  echo "$fallback_path"
}

download_snapshot() {
  local repo_id="$1"
  local cache_dir="$2"

  "$PY" - "$repo_id" "$cache_dir" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
cache_dir = sys.argv[2]

stdout_fd = os.dup(1)
try:
    sys.stdout.flush()
    os.dup2(2, 1)
    path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
finally:
    sys.stdout.flush()
    os.dup2(stdout_fd, 1)
    os.close(stdout_fd)

print(path)
PY
}

validate_num_frames
validate_context_parallel_size
if [[ ! -f "$FLEXTENSOR_DIR/pyproject.toml" ]]; then
  echo "FlexTensor checkout not found at $FLEXTENSOR_DIR (missing pyproject.toml)." >&2
  echo "Set FLEXTENSOR_DIR to the FlexTensor repository root or EX to this example directory." >&2
  exit 1
fi
FLEXTENSOR_DIR="$(cd -- "$FLEXTENSOR_DIR" && pwd -P)"
mkdir -p "$BASE" "$INPUT_DIR" "$OUTPUT_DIR" "$WORKSPACE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$XDG_CACHE_HOME"

apt-get update
apt-get install -y nginx numactl

if [[ ! -d "$LTX_DIR/.git" ]]; then
  git clone https://github.com/Lightricks/LTX-2.git "$LTX_DIR"
else
  echo "Using existing LTX checkout at $LTX_DIR"
fi

if ! git -C "$LTX_DIR" cat-file -e "${LTX2_COMMIT}^{commit}"; then
  git -C "$LTX_DIR" fetch origin "$LTX2_COMMIT"
fi
git -C "$LTX_DIR" checkout --detach "$LTX2_COMMIT"

cd "$LTX_DIR"

if ! command -v uv >/dev/null 2>&1; then
  pip3 install uv
fi

export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
uv sync
uv pip install --python "$PY" -e "$FLEXTENSOR_DIR"

shopt -s nullglob
LTX23_HF_REPO="models--Lightricks--LTX-2.3"
LORA_HF_REPO="models--oumoumad--LTX-2.3-22b-IC-LoRA-Outpaint"
GEMMA_HF_REPO="models--google--gemma-3-12b-it-qat-q4_0-unquantized"

distilled_candidates=(
  "$WORKSPACE_DIR/models/ltx23/$DISTILLED_FILENAME"
  "$BASE/models/ltx23/$DISTILLED_FILENAME"
)
upsampler_candidates=(
  "$WORKSPACE_DIR/models/ltx23/$UPSAMPLER_FILENAME"
  "$BASE/models/ltx23/$UPSAMPLER_FILENAME"
)

if [[ -n "$SNAP" ]]; then
  distilled_candidates=("$SNAP/$DISTILLED_FILENAME" "${distilled_candidates[@]}")
  upsampler_candidates=("$SNAP/$UPSAMPLER_FILENAME" "${upsampler_candidates[@]}")
fi

distilled_candidates+=(
  "$BASE/ltx2.3/huggingface_cache/hub/$LTX23_HF_REPO"/snapshots/*/"$DISTILLED_FILENAME"
  "$WORKSPACE_DIR/huggingface_cache/hub/$LTX23_HF_REPO"/snapshots/*/"$DISTILLED_FILENAME"
  "$CACHE_DIR/$LTX23_HF_REPO"/snapshots/*/"$DISTILLED_FILENAME"
)
upsampler_candidates+=(
  "$BASE/ltx2.3/huggingface_cache/hub/$LTX23_HF_REPO"/snapshots/*/"$UPSAMPLER_FILENAME"
  "$WORKSPACE_DIR/huggingface_cache/hub/$LTX23_HF_REPO"/snapshots/*/"$UPSAMPLER_FILENAME"
  "$CACHE_DIR/$LTX23_HF_REPO"/snapshots/*/"$UPSAMPLER_FILENAME"
)
gemma_candidates=(
  "$WORKSPACE_DIR/models/gemma"
  "$BASE/gemma3"
  "$BASE/models/gemma"
  "$BASE/ltx2.3/huggingface_cache/hub/$GEMMA_HF_REPO"/snapshots/*
  "$WORKSPACE_DIR/huggingface_cache/hub/$GEMMA_HF_REPO"/snapshots/*
  "$CACHE_DIR/$GEMMA_HF_REPO"/snapshots/*
)
lora_candidates=(
  "$WORKSPACE_DIR/models/lora/$LORA_FILENAME"
  "$BASE/LTX-2.3-22b-IC-LoRA-Outpaint/$LORA_FILENAME"
  "$BASE/models/lora/$LORA_FILENAME"
  "$CACHE_DIR/$LORA_HF_REPO"/snapshots/*/"$LORA_FILENAME"
)

DISTILLED="$(resolve_existing_path "$DISTILLED" "" "${distilled_candidates[@]}")"
UPSAMPLER="$(resolve_existing_path "$UPSAMPLER" "" "${upsampler_candidates[@]}")"
GEMMA="$(resolve_existing_path "$GEMMA" "" "${gemma_candidates[@]}")"
LORA="$(resolve_existing_path "$LORA" "" "${lora_candidates[@]}")"

if [[ ! -f "$DISTILLED" || ! -f "$UPSAMPLER" ]]; then
  require_external_license_ack
  echo "Downloading LTX 2.3 snapshot into cache: $CACHE_DIR"
  LTX23_SNAPSHOT="$(download_snapshot "Lightricks/LTX-2.3" "$CACHE_DIR")"
  if [[ ! -f "$DISTILLED" ]]; then
    DISTILLED="$LTX23_SNAPSHOT/$DISTILLED_FILENAME"
  fi
  if [[ ! -f "$UPSAMPLER" ]]; then
    UPSAMPLER="$LTX23_SNAPSHOT/$UPSAMPLER_FILENAME"
  fi
fi

if [[ ! -d "$GEMMA" ]]; then
  require_hf_token_for_gemma
  require_external_license_ack
  echo "Downloading Gemma snapshot into cache: $CACHE_DIR"
  GEMMA_SNAPSHOT="$(download_snapshot "$GEMMA_REPO_ID" "$CACHE_DIR")"
  GEMMA="$GEMMA_SNAPSHOT"
  if [[ ! -e "$BASE/gemma3" ]]; then
    ln -s "$GEMMA_SNAPSHOT" "$BASE/gemma3"
  fi
fi

if [[ ! -f "$LORA" ]]; then
  require_external_license_ack
  echo "Downloading Outpaint LoRA snapshot into cache: $CACHE_DIR"
  LORA_SNAPSHOT="$(download_snapshot "oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint" "$CACHE_DIR")"
  LORA="$LORA_SNAPSHOT/$LORA_FILENAME"
fi
shopt -u nullglob

require_path "serve entrypoint" "$EX/serve_infer.py"

MODEL_ARGS=(--cache-dir "$CACHE_DIR")
if [[ -f "$DISTILLED" ]]; then
  MODEL_ARGS+=(--distilled-checkpoint-path "$DISTILLED")
  echo "Using local distilled checkpoint: $DISTILLED"
else
  echo "No local distilled checkpoint found; resolving $DISTILLED_FILENAME from Lightricks/LTX-2.3"
fi
if [[ -f "$UPSAMPLER" ]]; then
  MODEL_ARGS+=(--spatial-upsampler-path "$UPSAMPLER")
  echo "Using local spatial upsampler: $UPSAMPLER"
else
  echo "No local spatial upsampler found; resolving $UPSAMPLER_FILENAME from Lightricks/LTX-2.3"
fi
if [[ -d "$GEMMA" ]]; then
  MODEL_ARGS+=(--gemma-root "$GEMMA")
  echo "Using local Gemma root: $GEMMA"
else
  echo "No local Gemma root found; resolving $GEMMA_REPO_ID"
fi
if [[ -f "$LORA" ]]; then
  MODEL_ARGS+=(--lora-path "$LORA")
  echo "Using local LoRA checkpoint: $LORA"
else
  echo "No local LoRA checkpoint found; resolving $LORA_FILENAME from oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint"
fi
if [[ "$OFFLOAD_TEXT" != "0" ]]; then
  MODEL_ARGS+=(--offload-text --text-mem-fraction "$TEXT_MEM_FRACTION")
  echo "Using FlexTensor text-encoder offload with text mem fraction: $TEXT_MEM_FRACTION"
fi

PROFILE_COMMAND=("$PY")
if ((CONTEXT_PARALLEL_SIZE > 1)); then
  TORCHRUN="$VENV_DIR/bin/torchrun"
  require_path "torchrun executable" "$TORCHRUN"
  PROFILE_COMMAND=("$TORCHRUN" --standalone --nproc-per-node="$CONTEXT_PARALLEL_SIZE")
fi

"${PROFILE_COMMAND[@]}" "$EX/serve_infer.py" profile \
  --conditioning-video "$LETTERBOXED_VIDEO" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --frame-rate "$FRAME_RATE" \
  --context-parallel-size "$CONTEXT_PARALLEL_SIZE" \
  --control-plane-timeout-seconds "$CONTROL_PLANE_TIMEOUT_SECONDS" \
  "${MODEL_ARGS[@]}" \
  --max-gpu-mem-fraction "$MAX_GPU_MEM_FRACTION" \
  --profiling-iters 2 \
  --profile-dir "$PROFILE_DIR" \
  --output-path "$OUTPUT_DIR/profile_warmup.mp4"

echo "CP${CONTEXT_PARALLEL_SIZE} profiling complete: $PROFILE_DIR"
if ((CONTEXT_PARALLEL_SIZE == 1)); then
  echo "Run $EX/single_serve.sh or $EX/nginx_serve.sh to start serving."
else
  echo "Run CONTEXT_PARALLEL_SIZE=$CONTEXT_PARALLEL_SIZE $EX/nginx_serve.sh to start serving."
fi

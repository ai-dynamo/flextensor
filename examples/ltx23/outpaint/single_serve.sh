#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
BASE="${BASE:-$WORKSPACE_DIR/outpaint}"
EX="${EX:-/my_home/flex-tensor/examples/ltx23/outpaint}"
VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/.venv}"
INPUT_DIR="${INPUT_DIR:-$BASE/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/outputs}"
HF_HOME="${HF_HOME:-$BASE/ltx2.3/huggingface_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/.cache}"
CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8020}"
PY="$VENV_DIR/bin/python"
export HF_HOME HF_HUB_CACHE XDG_CACHE_HOME
export HF_XET_CACHE
OFFLOAD_TEXT="${OFFLOAD_TEXT:-1}"
TEXT_MEM_FRACTION="${TEXT_MEM_FRACTION:-0.05}"
MAX_GPU_MEM_FRACTION="${MAX_GPU_MEM_FRACTION:-0.15}"
# Reduce fragmentation before allocating large video workspaces, as in nginx_serve.sh.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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

PROFILE_DIR="${PROFILE_DIR:-$OUTPUT_DIR/ltx23_outpaint_profile}"
LETTERBOXED_VIDEO="${LETTERBOXED_VIDEO:-$INPUT_DIR/letterboxed_720p.mp4}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-49}"
FRAME_RATE="${FRAME_RATE:-24}"
SERVE_LOG="$OUTPUT_DIR/serve.log"
SERVER_PID=""
CLEANED_UP=0

validate_num_frames() {
  if [[ ! "$NUM_FRAMES" =~ ^[0-9]+$ ]] ||
    ((NUM_FRAMES < 1 || (NUM_FRAMES - 1) % 8 != 0)); then
    echo "NUM_FRAMES must be a positive integer of the form 8*k + 1; got: $NUM_FRAMES" >&2
    exit 1
  fi
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

cleanup_server() {
  set +e
  if ((CLEANED_UP)); then
    return
  fi
  CLEANED_UP=1
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping server process $SERVER_PID..."
    kill -INT "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

wait_for_server() {
  local server_pid="$1"
  local attempts="${SERVER_READY_ATTEMPTS:-3600}"
  local ready_line="Serving on http://$HOST:$PORT"
  local attempt

  echo "Waiting up to $attempts seconds for server warmup; logs: $SERVE_LOG"
  for attempt in $(seq 1 "$attempts"); do
    if [[ -f "$SERVE_LOG" ]] &&
      grep -Fq "$ready_line" "$SERVE_LOG" &&
      curl -fsS "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
      echo "Server is ready at http://$HOST:$PORT"
      return
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before becoming ready; see $SERVE_LOG" >&2
      exit 1
    fi
    if ((attempt % 30 == 0)) && [[ -f "$SERVE_LOG" ]]; then
      echo "Still waiting; latest log line:"
      sed -n '$p' "$SERVE_LOG"
    fi
    sleep 1
  done

  echo "Server did not become ready at http://$HOST:$PORT; see $SERVE_LOG" >&2
  exit 1
}

validate_num_frames
mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$XDG_CACHE_HOME"
trap cleanup_server EXIT
trap 'trap - EXIT; cleanup_server; exit 130' HUP INT TERM

require_path "Python venv interpreter" "$PY"
require_path "serve entrypoint" "$EX/serve_infer.py"
require_path "letterboxed conditioning video" "$LETTERBOXED_VIDEO"
require_path "FlexTensor profile directory" "$PROFILE_DIR"

shopt -s nullglob
LTX23_HF_REPO="models--Lightricks--LTX-2.3"
LORA_HF_REPO="models--oumoumad--LTX-2.3-22b-IC-LoRA-Outpaint"

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
)
lora_candidates=(
  "$WORKSPACE_DIR/models/lora/$LORA_FILENAME"
  "$BASE/LTX-2.3-22b-IC-LoRA-Outpaint/$LORA_FILENAME"
  "$BASE/models/lora/$LORA_FILENAME"
  "$CACHE_DIR/$LORA_HF_REPO"/snapshots/*/"$LORA_FILENAME"
)

DISTILLED="$(resolve_existing_path "$DISTILLED" "$WORKSPACE_DIR/models/ltx23/$DISTILLED_FILENAME" "${distilled_candidates[@]}")"
UPSAMPLER="$(resolve_existing_path "$UPSAMPLER" "$WORKSPACE_DIR/models/ltx23/$UPSAMPLER_FILENAME" "${upsampler_candidates[@]}")"
GEMMA="$(resolve_existing_path "$GEMMA" "$WORKSPACE_DIR/models/gemma" "${gemma_candidates[@]}")"
LORA="$(resolve_existing_path "$LORA" "$WORKSPACE_DIR/models/lora/$LORA_FILENAME" "${lora_candidates[@]}")"
shopt -u nullglob

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
  echo "No local Gemma root found; resolving google/gemma-3-12b-it-qat-q4_0-unquantized"
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

nohup "$PY" "$EX/serve_infer.py" serve \
  --conditioning-video "$LETTERBOXED_VIDEO" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --frame-rate "$FRAME_RATE" \
  --control-plane-timeout-seconds "$CONTROL_PLANE_TIMEOUT_SECONDS" \
  "${MODEL_ARGS[@]}" \
  --max-gpu-mem-fraction "$MAX_GPU_MEM_FRACTION" \
  --profile-dir "$PROFILE_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --warmup-output-path "$OUTPUT_DIR/server_warmup.mp4" \
  --output-path "$OUTPUT_DIR/server_default.mp4" \
  >"$SERVE_LOG" 2>&1 &
SERVER_PID="$!"

wait_for_server "$SERVER_PID"
trap - EXIT HUP INT TERM

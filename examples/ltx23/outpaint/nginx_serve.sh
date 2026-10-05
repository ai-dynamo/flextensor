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
WORKER_HOST="${WORKER_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8020}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"
# NUM_WORKERS remains the number of GPUs for backward compatibility. The
# number of HTTP replicas is NUM_WORKERS / CONTEXT_PARALLEL_SIZE.
NUM_WORKERS="${NUM_WORKERS:-8}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
NUMA_INTERLEAVE="${NUMA_INTERLEAVE:-1}"
CONTROL_PLANE_TIMEOUT_SECONDS="${CONTROL_PLANE_TIMEOUT_SECONDS:-30}"
TORCHRUN_MAX_RESTARTS="${TORCHRUN_MAX_RESTARTS:-}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
PY="$VENV_DIR/bin/python"
export HF_HOME HF_HUB_CACHE XDG_CACHE_HOME
export HF_XET_CACHE PYTORCH_CUDA_ALLOC_CONF TORCH_NCCL_ASYNC_ERROR_HANDLING

SNAP="${SNAP:-}"
DISTILLED="${DISTILLED:-}"
UPSAMPLER="${UPSAMPLER:-}"
GEMMA="${GEMMA:-}"
LORA="${LORA:-}"

DISTILLED_FILENAME="ltx-2.3-22b-distilled-1.1.safetensors"
UPSAMPLER_FILENAME="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LORA_FILENAME="ltx-2.3-22b-ic-lora-outpaint.safetensors"

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
MAX_GPU_MEM_FRACTION="${MAX_GPU_MEM_FRACTION:-0.15}"
SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-3600}"
OFFLOAD_TEXT="${OFFLOAD_TEXT:-1}"
TEXT_MEM_FRACTION="${TEXT_MEM_FRACTION:-0.05}"
SHM_NAMESPACE="${SHM_NAMESPACE:-}"
SHM_WAIT_TIMEOUT="${SHM_WAIT_TIMEOUT:-3600}"
SHM_NOFILE_SOFT_LIMIT="${SHM_NOFILE_SOFT_LIMIT:-4096}"
WORKER_TERM_GRACE="${WORKER_TERM_GRACE:-600}"
NGINX_CONF="${NGINX_CONF:-/tmp/ltx23-outpaint-nginx-${USER:-user}.conf}"
NGINX_ERROR_LOG="${NGINX_ERROR_LOG:-$OUTPUT_DIR/nginx_error.log}"
NGINX_ACCESS_LOG="${NGINX_ACCESS_LOG:-$OUTPUT_DIR/nginx_access.log}"

WORKER_PIDS=()
WORKER_PGIDS=()
WORKER_LOGS=()
WORKER_RANK_DIR=""
NGINX_PID=""
CLEANED_UP=0
REPLICA_COUNT=0
SHM_ARGS=()
NUMA_PREFIX=()

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

validate_offload_text() {
  if [[ "$OFFLOAD_TEXT" != "0" && "$OFFLOAD_TEXT" != "1" ]]; then
    echo "OFFLOAD_TEXT must be 0 or 1; got: $OFFLOAD_TEXT" >&2
    exit 1
  fi
}

validate_worker_term_grace() {
  if [[ ! "$WORKER_TERM_GRACE" =~ ^[0-9]+$ ]] || ((WORKER_TERM_GRACE < 1)); then
    echo "WORKER_TERM_GRACE must be a positive integer; got: $WORKER_TERM_GRACE" >&2
    exit 1
  fi
}

configure_numa() {
  case "$NUMA_INTERLEAVE" in
    0)
      echo "NUMA interleaving disabled; workers retain the inherited memory policy."
      return
      ;;
    1) ;;
    *)
      echo "NUMA_INTERLEAVE must be 0 or 1; got: $NUMA_INTERLEAVE" >&2
      exit 1
      ;;
  esac

  local numactl_path
  if ! numactl_path="$(command -v numactl)"; then
    echo "NUMA_INTERLEAVE=1 requires numactl. Run setup.sh or install numactl, or set NUMA_INTERLEAVE=0." >&2
    exit 1
  fi
  NUMA_PREFIX=("$numactl_path" --interleave=all)
  if ! "${NUMA_PREFIX[@]}" true; then
    echo "Cannot apply numactl --interleave=all. For Docker, add --cap-add SYS_NICE before the image name," >&2
    echo "or set NUMA_INTERLEAVE=0 to retain the inherited memory policy." >&2
    exit 1
  fi
  echo "NUMA interleaving enabled across all allowed memory nodes (numactl --interleave=all)."
}

validate_torchrun_max_restarts() {
  if [[ ! "$TORCHRUN_MAX_RESTARTS" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "TORCHRUN_MAX_RESTARTS must be a non-negative integer without leading zeros;" >&2
    echo "got: $TORCHRUN_MAX_RESTARTS" >&2
    exit 1
  fi
}

requires_zero_torchrun_restarts() {
  ((CONTEXT_PARALLEL_SIZE > 1))
}

configure_torchrun_restarts() {
  if [[ -z "$TORCHRUN_MAX_RESTARTS" ]]; then
    if requires_zero_torchrun_restarts; then
      TORCHRUN_MAX_RESTARTS=0
    else
      TORCHRUN_MAX_RESTARTS=3
    fi
  fi
  validate_torchrun_max_restarts
  if requires_zero_torchrun_restarts && [[ "$TORCHRUN_MAX_RESTARTS" != "0" ]]; then
    echo "TORCHRUN_MAX_RESTARTS must be 0 when CONTEXT_PARALLEL_SIZE>1;" >&2
    echo "coordinated SHM startup cannot safely restart after partial creator/follower initialization." >&2
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

require_profile_file() {
  local label="$1"
  local path="$2"

  if [[ ! -f "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

validate_profile_parallelism() {
  local metadata_path="$1"

  "$PY" - "$metadata_path" "$CONTEXT_PARALLEL_SIZE" <<'PY'
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
requested_size = int(sys.argv[2])
try:
    metadata = json.loads(metadata_path.read_text())
    profile_size = metadata["context_parallel_size"]
    if not isinstance(profile_size, int) or isinstance(profile_size, bool):
        raise ValueError(f"context_parallel_size must be an integer, got {profile_size!r}")
except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
    raise SystemExit(f"Invalid context-parallel profile metadata {metadata_path}: {exc}") from exc

if profile_size != requested_size:
    raise SystemExit(
        f"FlexTensor profile {metadata_path.parent} was created for CP{profile_size}, "
        f"but this launcher requested CP{requested_size}."
    )
PY
}

validate_profile_files() {
  local metadata_path="$PROFILE_DIR/parallelism.json"

  require_profile_file "stage 1 FlexTensor profile" "$PROFILE_DIR/stage1/profile.json"
  require_profile_file "stage 2 FlexTensor profile" "$PROFILE_DIR/stage2/profile.json"
  if [[ "$OFFLOAD_TEXT" == "1" ]]; then
    require_profile_file "text-encoder FlexTensor profile" "$PROFILE_DIR/text/profile.json"
  fi
  # serve_infer.py accepts a legacy CP1 profile without metadata, but every
  # context-parallel profile must declare the CP size it was created with.
  if ((CONTEXT_PARALLEL_SIZE > 1)) || [[ -e "$metadata_path" ]]; then
    require_profile_file "context-parallel profile metadata" "$metadata_path"
    validate_profile_parallelism "$metadata_path"
  fi
  echo "Validated required FlexTensor profile files in: $PROFILE_DIR"
}

require_command() {
  local command_name="$1"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
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

parse_gpu_ids() {
  local raw_gpu_ids="${GPU_IDS:-}"

  if [[ -n "$raw_gpu_ids" ]]; then
    # Accept either "0 1 2" or "0,1,2" for convenience.
    read -r -a GPU_ID_ARRAY <<<"${raw_gpu_ids//,/ }"
    NUM_WORKERS="${#GPU_ID_ARRAY[@]}"
    return
  fi

  GPU_ID_ARRAY=()
  local worker_index
  for worker_index in $(seq 0 $((NUM_WORKERS - 1))); do
    GPU_ID_ARRAY+=("$worker_index")
  done
}

configure_shm() {
  if [[ -z "$SHM_NAMESPACE" ]]; then
    local SHM_USER="${USER:-user}"
    SHM_USER="${SHM_USER//[^A-Za-z0-9_]/_}"
    SHM_USER="${SHM_USER:0:24}"
    SHM_USER="${SHM_USER:-user}"
    # Container launches can repeat PID 1 while host SHM survives a killed run.
    local run_id
    run_id="$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')"
    SHM_NAMESPACE="ltx23_outpaint_${SHM_USER}_${FRONTEND_PORT}_$$_${run_id}"
  fi
  if [[ ! "$SHM_NAMESPACE" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "SHM_NAMESPACE must contain only letters, digits, and underscores; got: $SHM_NAMESPACE" >&2
    exit 1
  fi
  if [[ ! "$SHM_WAIT_TIMEOUT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
    [[ "$SHM_WAIT_TIMEOUT" =~ ^0*([.]0*)?$ ]]; then
    echo "SHM_WAIT_TIMEOUT must be a positive number; got: $SHM_WAIT_TIMEOUT" >&2
    exit 1
  fi
  SHM_ARGS=(
    --shm-namespace "$SHM_NAMESPACE"
    --shm-wait-timeout "$SHM_WAIT_TIMEOUT"
  )
  echo "Shared-memory weight sharing enabled with run namespace: $SHM_NAMESPACE"
}

configure_shm_nofile_limit() {
  # Shared allocations and coordinator heartbeats keep descriptors open; workers inherit this limit.
  if ! ulimit -Sn "$SHM_NOFILE_SOFT_LIMIT"; then
    echo "Unable to set the open-file soft limit to $SHM_NOFILE_SOFT_LIMIT." >&2
    echo "For Docker, use --ulimit nofile=$SHM_NOFILE_SOFT_LIMIT:$SHM_NOFILE_SOFT_LIMIT." >&2
    exit 1
  fi
}

replica_group_has_live_processes() {
  local pgid="$1"
  if [[ ! "$pgid" =~ ^[1-9][0-9]*$ ]] || ((pgid <= 1)); then
    return 1
  fi

  local state
  # setsid makes the session ID equal to the process-group ID. Numeric `ps -g`
  # selects that session on Linux and is also supported by BSD/macOS ps.
  while read -r state; do
    if [[ -n "$state" && "$state" != Z* ]]; then
      return 0
    fi
  done < <(ps -o stat= -g "$pgid" 2>/dev/null)
  return 1
}

# shellcheck disable=SC2329  # Invoked only from the trap-driven cleanup path.
signal_replica_group() {
  local signal_name="$1"
  local pgid="$2"
  if [[ ! "$pgid" =~ ^[1-9][0-9]*$ ]] || ((pgid <= 1)); then
    echo "Refusing to signal invalid replica process group: $pgid" >&2
    return 1
  fi
  if replica_group_has_live_processes "$pgid"; then
    kill -s "$signal_name" -- "-$pgid" 2>/dev/null || true
  fi
}

wait_for_replica_group_start() {
  local worker_pid="$1"
  local worker_pgid="$2"
  local serve_log="$3"
  local attempt
  for attempt in {1..100}; do
    if ! kill -0 "$worker_pid" 2>/dev/null; then
      echo "Replica launcher process $worker_pid exited before setsid established process group $worker_pgid." >&2
      print_log_context "replica launcher" "$serve_log"
      return 1
    fi
    if replica_group_has_live_processes "$worker_pgid"; then
      return 0
    fi
    sleep 0.05
  done
  echo "Replica launcher process $worker_pid did not establish process group $worker_pgid." >&2
  print_log_context "replica launcher" "$serve_log"
  return 1
}

# shellcheck disable=SC2329  # Invoked through trap handlers below.
cleanup() {
  set +e
  if ((CLEANED_UP)); then
    return
  fi
  CLEANED_UP=1
  echo "Stopping nginx and worker processes..."
  if [[ -n "$WORKER_RANK_DIR" && -d "$WORKER_RANK_DIR/active" ]]; then
    # Close registration before collecting ranks. Wrappers that start later
    # must exit without loading a model, even if their torchrun agent is gone.
    mv "$WORKER_RANK_DIR/active" "$WORKER_RANK_DIR/closing"
    local rank_file
    for rank_file in "$WORKER_RANK_DIR/closing/"*; do
      [[ -f "$rank_file" ]] || continue
      WORKER_PGIDS+=("${rank_file##*/}")
    done
  fi
  if [[ -n "$NGINX_PID" ]] && kill -0 "$NGINX_PID" 2>/dev/null; then
    kill -TERM "$NGINX_PID" 2>/dev/null || true
  fi
  local worker_index
  for worker_index in "${!WORKER_PGIDS[@]}"; do
    signal_replica_group TERM "${WORKER_PGIDS[$worker_index]}" || true
  done

  local deadline=$((SECONDS + WORKER_TERM_GRACE))
  local any_alive
  while ((SECONDS < deadline)); do
    any_alive=0
    if [[ -n "$NGINX_PID" ]] && kill -0 "$NGINX_PID" 2>/dev/null; then
      any_alive=1
    fi
    for worker_index in "${!WORKER_PGIDS[@]}"; do
      if replica_group_has_live_processes "${WORKER_PGIDS[$worker_index]}"; then
        any_alive=1
        break
      fi
    done
    if ((!any_alive)); then
      break
    fi
    sleep 1
  done

  if [[ -n "$NGINX_PID" ]] && kill -0 "$NGINX_PID" 2>/dev/null; then
    kill -KILL "$NGINX_PID" 2>/dev/null || true
  fi
  for worker_index in "${!WORKER_PGIDS[@]}"; do
    if replica_group_has_live_processes "${WORKER_PGIDS[$worker_index]}"; then
      echo "WARNING: SIGKILL required for worker/rank group ${WORKER_PGIDS[$worker_index]}; SHM namespace: $SHM_NAMESPACE" >&2
      signal_replica_group KILL "${WORKER_PGIDS[$worker_index]}" || true
    fi
  done

  # SIGKILL delivery is asynchronous. Never unlink SHM while recorded ranks
  # still run, and do not wait indefinitely for an uninterruptible process.
  deadline=$((SECONDS + 5))
  while true; do
    any_alive=0
    for worker_index in "${!WORKER_PGIDS[@]}"; do
      if replica_group_has_live_processes "${WORKER_PGIDS[$worker_index]}"; then
        any_alive=1
        break
      fi
    done
    if [[ -n "$NGINX_PID" ]] && kill -0 "$NGINX_PID" 2>/dev/null; then
      any_alive=1
    fi
    if ((!any_alive)); then
      break
    fi
    if ((SECONDS >= deadline)); then
      echo "Processes remain alive; skipping SHM cleanup for namespace: $SHM_NAMESPACE" >&2
      return 1
    fi
    sleep 0.1
  done

  if [[ -n "$NGINX_PID" ]]; then
    wait "$NGINX_PID" 2>/dev/null || true
  fi
  local worker_pid
  for worker_pid in "${WORKER_PIDS[@]}"; do
    wait "$worker_pid" 2>/dev/null || true
  done
  if [[ -n "$WORKER_RANK_DIR" ]]; then
    rm -rf -- "$WORKER_RANK_DIR"
  fi
  echo "Worker processes stopped."
  if ((${#WORKER_PIDS[@]})); then
    if ! "$PY" "$EX/cleanup_shm.py" --shm-namespace "$SHM_NAMESPACE"; then
      echo "SHM cleanup incomplete for namespace: $SHM_NAMESPACE. Retry cleanup_shm.py after stopping all workers." >&2
      return 1
    fi
  fi
  return 0
}

# shellcheck disable=SC2329  # Invoked by EXIT trap.
finish() {
  local status="$1"
  trap - EXIT HUP INT TERM
  if ! cleanup && ((status == 0)); then
    status=1
  fi
  exit "$status"
}

check_all_workers_alive() {
  local worker_index
  for worker_index in "${!WORKER_PIDS[@]}"; do
    local worker_pid="${WORKER_PIDS[$worker_index]}"
    local worker_pgid="${WORKER_PGIDS[$worker_index]}"
    if ! kill -0 "$worker_pid" 2>/dev/null; then
      local worker_port=$((BASE_PORT + worker_index))
      local worker_log="${WORKER_LOGS[$worker_index]}"
      echo "Replica $worker_index launcher process $worker_pid on port $worker_port exited during startup." >&2
      print_log_context "worker $worker_index" "$worker_log"
      return 1
    fi
    if ! replica_group_has_live_processes "$worker_pgid"; then
      local worker_port=$((BASE_PORT + worker_index))
      local worker_log="${WORKER_LOGS[$worker_index]}"
      echo "Replica $worker_index process group $worker_pgid on port $worker_port has no live processes." >&2
      print_log_context "worker $worker_index" "$worker_log"
      return 1
    fi
  done
  return 0
}

wait_for_worker() {
  local worker_pid="$1"
  local worker_port="$2"
  local serve_log="$3"
  local ready_line="Serving on http://$WORKER_HOST:$worker_port"
  local attempt

  echo "Waiting up to $SERVER_READY_ATTEMPTS seconds for worker $worker_pid on $WORKER_HOST:$worker_port; log: $serve_log"
  for attempt in $(seq 1 "$SERVER_READY_ATTEMPTS"); do
    if ! check_all_workers_alive; then
      echo "Worker startup aborted while waiting for port $worker_port; see worker logs above." >&2
      exit 1
    fi
    if [[ -f "$serve_log" ]] && grep -Fq "$ready_line" "$serve_log" &&
      curl -fsS "http://$WORKER_HOST:$worker_port/healthz" >/dev/null 2>&1; then
      echo "Worker ready at http://$WORKER_HOST:$worker_port"
      return
    fi
    if ((attempt % 30 == 0)) && [[ -f "$serve_log" ]]; then
      echo "Still waiting for port $worker_port; latest log line:"
      sed -n '$p' "$serve_log"
    fi
    sleep 1
  done

  echo "Worker did not become ready at http://$WORKER_HOST:$worker_port; see $serve_log" >&2
  exit 1
}

print_log_context() {
  local label="$1"
  local log_path="$2"
  local max_lines="${3:-40}"

  if [[ ! -f "$log_path" ]]; then
    echo "No $label log found at $log_path" >&2
    return
  fi
  if [[ ! -s "$log_path" ]]; then
    echo "$label log is empty: $log_path" >&2
    return
  fi

  local line_count
  local start_line
  line_count="$(wc -l <"$log_path")"
  start_line=$((line_count > max_lines ? line_count - max_lines + 1 : 1))

  echo "Last $max_lines lines from $label log ($log_path):" >&2
  sed -n "${start_line},${line_count}p" "$log_path" >&2
}

report_exited_processes() {
  local wait_status="$1"
  local reported=0

  echo "A worker or NGINX process exited; wait status: $wait_status" >&2
  if [[ -n "$NGINX_PID" ]] && ! kill -0 "$NGINX_PID" 2>/dev/null; then
    echo "NGINX process $NGINX_PID has exited." >&2
    print_log_context "NGINX error" "$NGINX_ERROR_LOG"
    print_log_context "NGINX access" "$NGINX_ACCESS_LOG"
    reported=1
  fi

  local worker_index
  for worker_index in "${!WORKER_PIDS[@]}"; do
    local worker_pid="${WORKER_PIDS[$worker_index]}"
    if ! kill -0 "$worker_pid" 2>/dev/null; then
      local worker_port=$((BASE_PORT + worker_index))
      echo "Worker $worker_index process $worker_pid on port $worker_port has exited." >&2
      print_log_context "worker $worker_index" "${WORKER_LOGS[$worker_index]}"
      reported=1
    fi
  done

  if ((!reported)); then
    echo "No exited process could be identified; recent logs follow." >&2
    print_log_context "NGINX error" "$NGINX_ERROR_LOG"
    print_log_context "NGINX access" "$NGINX_ACCESS_LOG"
    for worker_index in "${!WORKER_LOGS[@]}"; do
      print_log_context "worker $worker_index" "${WORKER_LOGS[$worker_index]}"
    done
  fi
}

write_nginx_config() {
  local servers=""
  local replica_index

  for replica_index in $(seq 0 $((REPLICA_COUNT - 1))); do
    local worker_port=$((BASE_PORT + replica_index))
    servers+="        server $WORKER_HOST:$worker_port max_fails=1 fail_timeout=30s;
"
  done

  cat >"$NGINX_CONF" <<EOF
worker_processes 1;
error_log $NGINX_ERROR_LOG warn;
pid /tmp/ltx23-outpaint-nginx-${USER:-user}-${FRONTEND_PORT}.pid;

events {
    worker_connections 4096;
}

http {
    log_format upstreamlog '\$remote_addr [\$time_local] "\$request" '
                           'status=\$status upstream=\$upstream_addr '
                           'request_time=\$request_time upstream_time=\$upstream_response_time';
    access_log $NGINX_ACCESS_LOG upstreamlog;
    client_max_body_size 1m;
    client_body_temp_path /tmp/ltx23-outpaint-client-body;
    proxy_temp_path /tmp/ltx23-outpaint-proxy;
    fastcgi_temp_path /tmp/ltx23-outpaint-fastcgi;
    uwsgi_temp_path /tmp/ltx23-outpaint-uwsgi;
    scgi_temp_path /tmp/ltx23-outpaint-scgi;

    upstream ltx_outpaint_workers {
$servers    }

    server {
        listen $FRONTEND_HOST:$FRONTEND_PORT;

        location / {
            proxy_pass http://ltx_outpaint_workers;
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_connect_timeout 10m;
            proxy_send_timeout 10m;
            proxy_read_timeout 10m;
            proxy_next_upstream off;
        }
    }
}
EOF
}

validate_num_frames
validate_context_parallel_size
validate_offload_text
validate_worker_term_grace
configure_numa
require_path "Python venv interpreter" "$PY"
configure_shm
configure_shm_nofile_limit
configure_torchrun_restarts
mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$XDG_CACHE_HOME"
trap 'finish "$?"' EXIT
trap 'exit 130' HUP INT TERM

require_command curl
require_command nginx
require_command ps
require_command setsid
require_path "serve entrypoint" "$EX/serve_infer.py"
require_path "SHM cleanup entrypoint" "$EX/cleanup_shm.py"
require_path "letterboxed conditioning video" "$LETTERBOXED_VIDEO"
require_path "FlexTensor profile directory" "$PROFILE_DIR"
validate_profile_files
parse_gpu_ids

if ((NUM_WORKERS < 1)); then
  echo "NUM_WORKERS must be at least 1" >&2
  exit 1
fi
if ((${#GPU_ID_ARRAY[@]} % CONTEXT_PARALLEL_SIZE != 0)); then
  echo "GPU count (${#GPU_ID_ARRAY[@]}) must be divisible by CONTEXT_PARALLEL_SIZE ($CONTEXT_PARALLEL_SIZE)" >&2
  exit 1
fi
REPLICA_COUNT=$((${#GPU_ID_ARRAY[@]} / CONTEXT_PARALLEL_SIZE))
if ((CONTEXT_PARALLEL_SIZE > 1)); then
  require_path "torchrun executable" "$VENV_DIR/bin/torchrun"
fi

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
if [[ "$OFFLOAD_TEXT" == "1" ]]; then
  MODEL_ARGS+=(--offload-text --text-mem-fraction "$TEXT_MEM_FRACTION")
  echo "Using FlexTensor text-encoder offload with text mem fraction: $TEXT_MEM_FRACTION"
fi

if ((CONTEXT_PARALLEL_SIZE > 1)); then
  WORKER_RANK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ltx23-outpaint-ranks.XXXXXXXX")"
  mkdir "$WORKER_RANK_DIR/active"
fi

for replica_index in $(seq 0 $((REPLICA_COUNT - 1))); do
  group_start=$((replica_index * CONTEXT_PARALLEL_SIZE))
  replica_gpu_ids=("${GPU_ID_ARRAY[@]:group_start:CONTEXT_PARALLEL_SIZE}")
  visible_devices="$(
    IFS=,
    echo "${replica_gpu_ids[*]}"
  )"
  WORKER_PORT=$((BASE_PORT + replica_index))
  SERVE_LOG="$OUTPUT_DIR/serve_cp${CONTEXT_PARALLEL_SIZE}_replica${replica_index}_port${WORKER_PORT}.log"
  WORKER_LOGS+=("$SERVE_LOG")
  echo "Starting CP${CONTEXT_PARALLEL_SIZE} replica $replica_index on GPUs $visible_devices at http://$WORKER_HOST:$WORKER_PORT"

  LAUNCH_COMMAND=("$PY")
  if ((CONTEXT_PARALLEL_SIZE > 1)); then
    # shellcheck disable=SC2016  # Expand registration variables in each rank, not in the launcher.
    LAUNCH_COMMAND=(
      "$VENV_DIR/bin/torchrun"
      --standalone
      --nproc-per-node="$CONTEXT_PARALLEL_SIZE"
      --max-restarts="$TORCHRUN_MAX_RESTARTS"
      # torchrun gives each rank its own session. Register its group before
      # exec so cleanup can signal ranks independently of the agent.
      --no-python "$BASH" -c '
        # Use BASHPID because torchrun substitutes $$ with a literal dollar.
        : > "$OUTPAINT_RANK_DIR/$BASHPID" || exit 1
        # Registration may have closed while the marker was being created.
        [[ -d "$OUTPAINT_RANK_DIR" ]] || exit 1
        exec "$@"
      ' outpaint-rank "$PY" -u
    )
  fi

  CUDA_VISIBLE_DEVICES="$visible_devices" OUTPAINT_RANK_DIR="$WORKER_RANK_DIR/active" \
    nohup setsid "${NUMA_PREFIX[@]}" "${LAUNCH_COMMAND[@]}" "$EX/serve_infer.py" serve \
    --conditioning-video "$LETTERBOXED_VIDEO" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$NUM_FRAMES" \
    --frame-rate "$FRAME_RATE" \
    --context-parallel-size "$CONTEXT_PARALLEL_SIZE" \
    --control-plane-timeout-seconds "$CONTROL_PLANE_TIMEOUT_SECONDS" \
    "${MODEL_ARGS[@]}" \
    --max-gpu-mem-fraction "$MAX_GPU_MEM_FRACTION" \
    --profile-dir "$PROFILE_DIR" \
    "${SHM_ARGS[@]}" \
    --host "$WORKER_HOST" \
    --port "$WORKER_PORT" \
    --warmup-output-path "$OUTPUT_DIR/server_warmup_cp${CONTEXT_PARALLEL_SIZE}_replica${replica_index}.mp4" \
    --output-path "$OUTPUT_DIR/server_default_cp${CONTEXT_PARALLEL_SIZE}_replica${replica_index}.mp4" \
    >"$SERVE_LOG" 2>&1 &
  worker_pid="$!"
  WORKER_PIDS+=("$worker_pid")
  # util-linux setsid execs the command in a new session whose process-group ID
  # is the launcher PID. Detached torchrun rank groups register separately.
  WORKER_PGIDS+=("$worker_pid")
  if ! wait_for_replica_group_start "$worker_pid" "$worker_pid" "$SERVE_LOG"; then
    exit 1
  fi
done

for worker_index in "${!WORKER_PIDS[@]}"; do
  wait_for_worker "${WORKER_PIDS[$worker_index]}" "$((BASE_PORT + worker_index))" "${WORKER_LOGS[$worker_index]}"
done

check_all_workers_alive
write_nginx_config
echo "Generated nginx config:"
sed -n '1,200p' "$NGINX_CONF"
nginx -t -c "$NGINX_CONF"
nginx -c "$NGINX_CONF" -g "daemon off;" &
NGINX_PID="$!"

echo "Frontend ready at http://$FRONTEND_HOST:$FRONTEND_PORT"
echo "Worker logs:"
printf '  %s\n' "${WORKER_LOGS[@]}"
wait_status=0
wait -n || wait_status=$?
report_exited_processes "$wait_status"
exit "$wait_status"

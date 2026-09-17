#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Integration tests for offload strategies using generated DeepSeek R1-like data
#
# Usage: ./test.sh [--diagnostics] [--gpu-mem GB] [--target-gpu-mem GB] [--num-layers N]
#                  [--seed N] [--transfer-ratio R ...]
#
# This script runs all offload strategies with synthetic data and validates:
# - No pipeline violations (consecutive layers using same block)
# - Peak GPU memory within specified limit
#
# Exit code 0 = all strategies pass validation
# Exit code 1 = at least one strategy failed validation

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default parameters
GPU_MEM=50
TARGET_GPU_MEM=""
NUM_LAYERS=63
SEED=42
TRANSFER_RATIO="1.0 0.1"
DIAGNOSTICS_FLAG="--diagnostics"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --diagnostics)
      DIAGNOSTICS_FLAG="--diagnostics"
      shift
      ;;
    --no-diagnostics)
      DIAGNOSTICS_FLAG=""
      shift
      ;;
    --gpu-mem)
      GPU_MEM="$2"
      shift 2
      ;;
    --target-gpu-mem)
      TARGET_GPU_MEM="$2"
      shift 2
      ;;
    --num-layers)
      NUM_LAYERS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --transfer-ratio)
      TRANSFER_RATIO="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--diagnostics] [--no-diagnostics] [--gpu-mem GB] [--target-gpu-mem GB] [--num-layers N] [--seed N] [--transfer-ratio 'R ...']"
      exit 1
      ;;
  esac
done

echo ""
echo "========================================"
echo "OFFLOAD STRATEGY INTEGRATION TESTS"
echo "========================================"
echo ""
echo "Configuration:"
echo "  GPU Memory Limit: ${GPU_MEM} GB"
if [[ -n "$TARGET_GPU_MEM" ]]; then
  echo "  GPU Memory Target: ${TARGET_GPU_MEM} GB"
fi
echo "  Number of Layers: ${NUM_LAYERS}"
echo "  Random Seed: ${SEED}"
echo "  Transfer Ratio: ${TRANSFER_RATIO}"
echo ""

# Build optional arguments
TARGET_FLAG=""
if [[ -n "$TARGET_GPU_MEM" ]]; then
  TARGET_FLAG="--target-gpu-mem $TARGET_GPU_MEM"
fi

# Test 1: Standard run (no gaps)
echo "----------------------------------------"
echo "Test 1: Standard layers (no gaps)"
echo "----------------------------------------"
python "$SCRIPT_DIR/analyze_offload_strategy.py" \
  --strategy recommended \
  --gpu-mem "$GPU_MEM" \
  $TARGET_FLAG \
  --generate \
  --num-layers "$NUM_LAYERS" \
  --seed "$SEED" \
  --transfer-ratio $TRANSFER_RATIO \
  --validate \
  $DIAGNOSTICS_FLAG

# Test 2: Run with gap layers to exercise gap-aware strategies
# Gaps at layers 5, 15,16, 30,31,32 simulate non-offloadable layers in the model
echo ""
echo "----------------------------------------"
echo "Test 2: With gap layers"
echo "----------------------------------------"
python "$SCRIPT_DIR/analyze_offload_strategy.py" \
  --strategy recommended \
  --gpu-mem "$GPU_MEM" \
  $TARGET_FLAG \
  --generate \
  --num-layers "$NUM_LAYERS" \
  --seed "$SEED" \
  --transfer-ratio $TRANSFER_RATIO \
  --gap-layers 5 15 16 30 31 32 \
  --validate \
  $DIAGNOSTICS_FLAG

echo ""
echo "========================================"
echo "ALL TESTS PASSED"
echo "========================================"

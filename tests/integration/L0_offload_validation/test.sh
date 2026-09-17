#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run multiple tests by calling run_single_test.py multiple times
# Each test saves its results to a separate file
# Usage: ./test.sh [--seed SEED] [--no-seed]
#   --seed SEED : Use specific seed for reproducibility (default: 42)
#   --no-seed   : Disable seed (random initialization)

set -xeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default seed for reproducibility
DEFAULT_SEED=42
USE_SEED=true
SEED_VALUE=$DEFAULT_SEED

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed)
      SEED_VALUE="$2"
      USE_SEED=true
      shift 2
      ;;
    --no-seed)
      USE_SEED=false
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--seed SEED] [--no-seed]"
      echo "  --seed SEED : Use specific seed (default: $DEFAULT_SEED)"
      echo "  --no-seed   : Disable seed (random initialization)"
      exit 1
      ;;
  esac
done

# Set seed argument for python script
if [ "$USE_SEED" = true ]; then
  SEED_ARG="--seed $SEED_VALUE"
  echo "🎲 Using seed: $SEED_VALUE (for reproducibility)"
else
  SEED_ARG=""
  echo "⚠️  No seed set - results will vary between runs"
fi

# Install test-specific requirements if not in CI (CI installs with flextensor for proper dependency resolution)
if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
  pip list
fi

# Create results directory
RESULTS_DIR="$SCRIPT_DIR/test_results"
mkdir -p "$RESULTS_DIR"

# Clear old results
echo "🧹 Cleaning old results..."
rm -f "$RESULTS_DIR"/*.json

echo ""
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo "RUNNING VALIDATION TESTS"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo ""

# Define test configurations
# Format: "model_preset test_preset"
# Note: iterations are now part of the model preset, not test preset
# Note: offload-adaptive (AdaptiveStrategy) internally evaluates Knapsack, Global, and
# GlobalTensorSelection strategies, so individual strategy tests are commented out to
# reduce CI time. Uncomment them to exercise each strategy in isolation if needed.
TESTS=(
  #"basic-small baseline"
  #"basic-small offload-strategy"
  #"basic-small offload-raw-block-transfer"
  #"basic-small offload-allocation-block-transfer"
  #"expert-small baseline"
  #"expert-small offload-strategy"
  #"expert-small offload-raw-block-transfer"
  #"expert-small offload-allocation-block-transfer"
  "basic-full baseline"
  #"basic-full offload-strategy"
  #"basic-full offload-raw-block-transfer"
  #"basic-full offload-allocation-block-transfer"
  #"basic-full offload-global"
  #"basic-full offload-tensor-selection"
  "basic-full offload-adaptive"
  "expert-8layer-8expert baseline"
  #"expert-8layer-8expert offload-strategy"
  #"expert-8layer-8expert offload-raw-block-transfer"
  #"expert-8layer-8expert offload-allocation-block-transfer"
  #"expert-8layer-8expert offload-global"
  #"expert-8layer-8expert offload-tensor-selection"
  "expert-8layer-8expert offload-adaptive"
  "non-uniform-16layer baseline"
  #"non-uniform-16layer offload-strategy"
  #"non-uniform-16layer offload-raw-block-transfer"
  #"non-uniform-16layer offload-allocation-block-transfer"
  #"non-uniform-16layer offload-global"
  #"non-uniform-16layer offload-tensor-selection"
  "non-uniform-16layer offload-adaptive"

  # ========== High-Level API (OffloadManager + AdaptiveStrategy) ==========
  "expert-8layer-8expert om-baseline"
  "expert-8layer-8expert om-offload-adaptive"
  "non-uniform-16layer om-baseline"
  "non-uniform-16layer om-offload-adaptive"
)

# Run each test
TEST_COUNT=${#TESTS[@]}
CURRENT=0
FAILED_TESTS=()

for test_config in "${TESTS[@]}"; do
  CURRENT=$((CURRENT + 1))

  # Parse model and test preset
  MODEL=$(echo "$test_config" | cut -d' ' -f1)
  TEST=$(echo "$test_config" | cut -d' ' -f2)

  echo "[$CURRENT/$TEST_COUNT] Running: $MODEL + $TEST"

  # Run test and save results (with optional seed)
  # Use 'if cmd; then' form so set -e does not exit before the else branch
  if python "$SCRIPT_DIR/run_single_test.py" "$MODEL" "$TEST" --save-results --results-dir "$RESULTS_DIR" $SEED_ARG; then
    echo "  ✅ PASS"
  else
    echo "  ❌ FAIL"
    FAILED_TESTS+=("$MODEL + $TEST")
    # Stop on first failure for faster debugging
    echo ""
    echo "❌ Test failed! Stopping test run."
    echo "Failed test: $MODEL + $TEST"
    echo ""
    echo "To continue running remaining tests, comment out the 'exit 1' in test.sh"
    exit 1
  fi

  echo ""
done

echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo "All tests completed!"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Generating summary..."
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo ""

# Generate summary from all result files
python "$SCRIPT_DIR/summarize_results.py" "$RESULTS_DIR"

# View-mode profile integration tests (pytest-style; not driven by run_single_test.py).
echo ""
echo "Running profile-view-mode integration tests..."
PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
[ -n "${PYTEST_MARKER:-}" ] && PYTEST_ARGS+=(-m "$PYTEST_MARKER")
timeout "${TIMEOUT:-1800}" python3 -m pytest "$SCRIPT_DIR/test_profile_view_mode_integration.py" "${PYTEST_ARGS[@]}"

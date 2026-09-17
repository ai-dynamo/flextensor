#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run Torch-TensorRT + FlexTensor compatibility tests
# Validates that FlexTensor offloading works correctly with torch_tensorrt.compile.

set -xeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Verify torch_tensorrt is available
python -c "import torch_tensorrt; print(f'torch_tensorrt {torch_tensorrt.__version__}')" || {
  echo "ERROR: torch_tensorrt not installed. Use an NGC PyTorch container or install manually."
  exit 1
}

# Install test-specific requirements if not in CI
if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
fi

PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
[ -n "${PYTEST_MARKER:-}" ] && PYTEST_ARGS+=(-m "$PYTEST_MARKER")

timeout "${TIMEOUT:-1800}" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"

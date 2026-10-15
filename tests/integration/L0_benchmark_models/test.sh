#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -xeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install test-specific requirements if not in CI (CI installs with flextensor for proper dependency resolution)
if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
  pip list
fi

TIMEOUT=${TIMEOUT:-1800}
echo "Running HuggingFace model benchmarking integration tests..."

# Build pytest command with optional marker filter
PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
if [ -n "${PYTEST_MARKER:-}" ]; then
  echo "Filtering tests with marker: $PYTEST_MARKER"
  PYTEST_ARGS+=(-m "$PYTEST_MARKER")
fi

timeout "$TIMEOUT" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"

echo "Integration tests completed successfully!"

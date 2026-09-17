#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run CUDA Graphs + FlexTensor compatibility tests
# Validates that FlexTensor offloading works correctly when model forward
# passes are captured and replayed as CUDA graphs.

set -xeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install test-specific requirements if not in CI
if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
fi

PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
[ -n "${PYTEST_MARKER:-}" ] && PYTEST_ARGS+=(-m "$PYTEST_MARKER")

timeout "${TIMEOUT:-1800}" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"

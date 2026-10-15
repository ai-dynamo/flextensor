#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -xeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_install_requirements.sh
source "$SCRIPT_DIR/../_install_requirements.sh"

# Install test-specific requirements if not in CI (CI installs with flextensor for proper dependency resolution)
if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  install_requirements_preserving_vllm "$SCRIPT_DIR/requirements.txt"
  pip list
fi

TIMEOUT=${TIMEOUT:-3600}
echo "Running Nemotron vLLM integration tests with timeout ${TIMEOUT}s..."

# Build pytest command with optional marker filter
PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
if [ -n "${PYTEST_MARKER:-}" ]; then
  echo "Filtering tests with marker: $PYTEST_MARKER"
  PYTEST_ARGS+=(-m "$PYTEST_MARKER")
fi

timeout "$TIMEOUT" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"

echo "Integration tests completed successfully!"

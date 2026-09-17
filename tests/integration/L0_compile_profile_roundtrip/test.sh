#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run torch.compile + CUDA graphs + FlexTensor profile round-trip tests.
#
# Tests cover:
#   A  Profile save/restore correctness with CUDA graphs and torch.compile
#   B  torch.compile(proxy) wrapping after eager lifecycle
#   C  Dynamo graph structure inspection (subgraph counts, fullgraph failure)
#   D  Four-way CPU/GPU compile performance comparison (informational)

set -xeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install test-specific requirements if not in CI.
if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
fi

PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
[ -n "${PYTEST_MARKER:-}" ] && PYTEST_ARGS+=(-m "$PYTEST_MARKER")

timeout "${TIMEOUT:-1800}" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"

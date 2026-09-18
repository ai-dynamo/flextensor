#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -xeo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${CI:-}" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
fi
PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10 --basetemp="$SCRIPT_DIR/test_results")
[ -n "${PYTEST_MARKER:-}" ] && PYTEST_ARGS+=(-m "$PYTEST_MARKER")
timeout "${TIMEOUT:-7200}" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}" "$@"

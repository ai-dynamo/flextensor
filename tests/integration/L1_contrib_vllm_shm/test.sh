#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_install_requirements.sh
source "$SCRIPT_DIR/../_install_requirements.sh"

if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  install_requirements_preserving_vllm "$SCRIPT_DIR/requirements.txt"
  pip list
fi

TIMEOUT=${TIMEOUT:-7200}
PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=1 --durations=10)
if [ -n "${PYTEST_MARKER:-}" ]; then
  PYTEST_ARGS+=(-m "$PYTEST_MARKER")
fi

# Send one interrupt to pytest so its finally blocks stop the dedicated vLLM process group.
# --foreground prevents a second group signal from interrupting that cleanup.
# Allow in-flight requests and the server's bounded shutdown to finish before killing pytest.
timeout --foreground --signal=INT --kill-after=120s "$TIMEOUT" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"

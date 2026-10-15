<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Tests

## Integration Tests

Structure: `integration/L{0,1,2}_<name>/` — each directory contains `.dockerimages`, `requirements.txt`, `test.sh`.

**GPU runner markers**: use `gpu_vram_*`, `gpu_vram_min_*`, `gpu_sm_*`, or `gpu_sm_min_*`; CI maps them to available runners.

**requirements.txt**: for non-obvious version floors, add a brief comment with the model/doc/job reference that justifies the pin.

### test.sh Pattern

Leave `CUDA_VISIBLE_DEVICES` unchanged, including when it is unset. GPU selection belongs to the runner/CI or local caller; for a local single-GPU run, use `CUDA_VISIBLE_DEVICES=0 ./test.sh`.

GPU checks and diagnostics run once in `tests/integration/conftest.py`.
Do not duplicate them in shell wrappers.

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ] && pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
[ -n "${PYTEST_MARKER:-}" ] && PYTEST_ARGS+=(-m "$PYTEST_MARKER")
timeout "${TIMEOUT:-1800}" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"
```

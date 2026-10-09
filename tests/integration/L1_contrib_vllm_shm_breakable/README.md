<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SHM with breakable CUDA graphs

This Linux GPU suite uses `vllm/vllm-openai:v0.25.1` and requires two GPUs with
at least 80 GiB each. Run `./tests/integration/L1_contrib_vllm_shm_breakable/test.sh`
from an environment containing the checkout and its test dependencies.

The test generates a Qwen2.5-7B offloading profile with a 5 GiB GPU weight budget,
then starts DP=2 with one SHM creator and one follower. Both launches set
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` and `cudagraph_mode=PIECEWISE` so attention eager
breaks exercise the transfer-stream join. The DP workers use a fresh cache root,
separate from profile generation. Prefix caching is disabled, and three batches
of two concurrent requests repeat prefill and decode. Every response must contain
`Paris` and generate more than one token.

Acceptance requires vLLM's successful `Captured breakable cudagraph` debug log
with multiple graph segments and at least one eager break from each worker PID.
Those PIDs must match the SHM execution markers for DP ranks 0 and 1, with one
creator and one follower. Hook-installation messages alone do not qualify.
The test also checks checkpoint sharing and namespace cleanup after shutdown.
It sets `--shutdown-timeout 15` to allow worker cleanup; vLLM 0.25's default
zero-second abort can kill the workers before they release their SHM resources.

Results under `test_results/<test-name>/` include the generated profile,
`server.log`, all correctness responses in `metrics.json`, and per-rank capture
evidence in `breakable_captures.json`. Startup failures are visible in pytest's
captured server output. This suite does not repeat the DP=1 host-memory baseline;
`L1_contrib_vllm_shm` retains that acceptance check on vLLM 0.17.0.

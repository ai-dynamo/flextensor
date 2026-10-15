<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared host-weight memory acceptance

This Linux test requires two GPUs with at least 80 GiB each and vLLM >= 0.17.
It profiles Qwen2.5-7B with a 5 GiB GPU weight budget, then runs DP=1 and DP=2
against that same read-only profile. Both runs send the same two concurrent
correctness requests, including repeated batches for at least five seconds.
The profile must contain offloaded weights.
The budget accommodates roughly 2 GiB of resident weights and more than 2 GiB
of allocation-block working memory observed on A100; a 4 GiB budget failed
during initial strategy construction.
Both measured runs explicitly use multiprocessing workers and one API server.
Otherwise vLLM's DP=1 default combines engine and worker in one process, while
DP=2 adds separate workers and extra API processes, invalidating the baseline
overhead comparison.

`test_vllm_shm.py` configures and runs the experiment. The `_host_memory.py`
module in this folder implements sampling and memory acceptance. The test
supplies a sampler factory to the shared `_vllm_server.py` helper, which starts
it with the server PID, marks phases, and saves its report before shutdown.

The opt-in server sampler reads `/proc/PID/smaps` for the server and all its
current descendants, then waits 0.5 seconds before the next background collection.
The actual interval includes collection time. It also collects explicit idle
samples. It saves:

- `dp1_shm/host_memory.json` and `dp2_shm/host_memory.json`: per-phase peak
  process-tree RSS and PSS, the running statistics needed for acceptance,
  sample counts, and the first collection error per phase. Incomplete samples
  are excluded from peaks and counted separately through the total and complete
  sample counts. The first failing steady-state sample, if any, is included
  under `failure` with its reasons and per-process details.
- `host_memory_acceptance.json`: profile-derived CPU block sizes, unique
  offloaded tensor payload size, baseline range, measured budgets, observed
  maxima, and failures.

Each sample is checked as it arrives, updates running counts and extrema, and
is discarded unless it is the first failing sample. DP=2 uses the completed
DP=1 summary to check memory budgets during sampling. No sample history is
retained or saved, and `finish()` writes the accumulated report directly.
Peak RSS and PSS are each the maximum of the process-tree sum in a complete
sample, not a sum of individual process peaks.

Startup (including follower attachment), warmup, warmed idle, concurrent
inference, and post-inference idle have separate phase labels. Startup peaks
are **sampled lower bounds**, reported separately and excluded from steady-state
budgets. This is an acceptance check, not a peak-memory or throughput benchmark.

## Accounting and acceptance

Weight files are matched by namespace and device/inode. Each worker must map
all the same offload blocks and resident-tensor store. CPU block sizes are
computed from the profile's tensor payloads and allocation alignment; reusable
GPU block capacities do not describe total CPU backing. Allocated file bytes
are counted once, and the workers' summed weight PSS must account for them
within one page per file per process. All steady-state samples must be complete,
unswapped, and consistent across both runs, with at least three per phase.

RSS is diagnostic only: summing it double-counts shared pages. PSS, private
pages, and backing bytes overlap and are never added together. Linux can label
singly mapped shared pages private, so matching weight mappings are subtracted
from both PSS and private totals before checking overhead.

For each steady-state phase, let `B` be the maximum DP=1 process-tree non-weight
PSS. The tolerance is its observed max-minus-min range plus one page per weight
file per baseline process. DP=2 non-weight PSS and private memory must each fit
`2 * B + tolerance` for the entire tree and `B + tolerance` for each worker.
Using the whole baseline tree conservatively includes another copy of server
and helper overhead. The tree allowance must also be **smaller than one unique
offloaded-weight payload**. Otherwise qualification fails as inconclusive,
rather than accepting a threshold that could conceal a complete private copy.
Smaller duplicates below the allowance are not ruled out by this check.

The budget is measured on the runner; it is not a universal percentage or a
calibrated performance claim. Summaries are saved before server shutdown,
including collection errors. A failed sensitivity gate needs investigation
of the fixture or overhead, not an arbitrary larger tolerance. CPU unit tests
exercise parsing, accounting, injected private copies, and failure handling;
actual host-RAM savings require a successful run on the GPU runner.

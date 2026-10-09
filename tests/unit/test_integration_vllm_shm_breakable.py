# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that GPU acceptance cannot pass without capture evidence from both workers."""

import pytest

from tests.integration.L1_contrib_vllm_shm_breakable.test_vllm_shm_breakable import (
    _assert_breakable_capture_on_both_ranks,
)


@pytest.mark.parametrize("missing_evidence", [None, "rank", "capture", "breaks", "graphs", "pid", "role"])
def test_breakable_capture_requires_both_executing_workers(missing_evidence):
    lines = [
        "(Worker_DP0 pid=100) FlexTensor vLLM SHM execution data_parallel_index=0 role=creator",
        "(Worker_DP1 pid=200) FlexTensor vLLM SHM execution data_parallel_index=1 role=follower",
    ]
    capture = "Captured breakable cudagraph for BatchDescriptor(num_tokens=1): "
    capture += "BreakableCUDAGraphCapture(graphs=29, eager_breaks=28)"
    # Multiple captures from rank 0 must never stand in for rank 1.
    lines.extend([f"(Worker_DP0 pid=100) {capture}"] * 2)
    follower = f"(Worker_DP1 pid=200) {capture}"
    if missing_evidence == "rank":
        lines.pop(1)
    elif missing_evidence == "capture":
        follower = "(Worker_DP1 pid=200) FlexTensor joins the offloader before each breakable CUDA-graph eager segment"
    elif missing_evidence == "breaks":
        follower = follower.replace("eager_breaks=28", "eager_breaks=0")
    elif missing_evidence == "graphs":
        follower = follower.replace("graphs=29", "graphs=1")
    elif missing_evidence == "pid":
        follower = follower.replace("pid=200", "pid=300")
    elif missing_evidence == "role":
        lines[1] = lines[1].replace("role=follower", "role=creator")
    lines.append(follower)

    if missing_evidence is not None:
        with pytest.raises(AssertionError):
            _assert_breakable_capture_on_both_ranks(lines)
    else:
        evidence = _assert_breakable_capture_on_both_ranks(lines)
        assert set(evidence) == {"0", "1"}
        assert evidence["1"] == [follower]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise memory accounting without requiring Linux, vLLM, or GPUs."""

import json
import os
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.integration.L1_contrib_vllm_shm import _host_memory
from tests.integration.L1_contrib_vllm_shm._host_memory import (
    HostMemorySampler,
    parse_smaps,
    qualify_shared_host_memory,
)
from tests.integration.L1_contrib_vllm_shm.test_vllm_shm import _profile_offload_backing

_MIB = 1024**2
_OFFLOAD = 100 * _MIB
_BACKING = _OFFLOAD + _MIB
_IDS = [(1, 100), (1, 101)]


def _evidence(workers):
    samples = []
    for phase in ("warm", "inference", "post_inference"):
        for _ in range(3):
            processes = [
                {
                    "pid": worker,
                    "rss_bytes": _BACKING + _MIB,
                    "pss_bytes": _BACKING // workers + _MIB,
                    "private_bytes": (_BACKING if workers == 1 else 0) + _MIB,
                    "swap_bytes": 0,
                    "weight_pss_bytes": _BACKING // workers,
                    "weight_private_bytes": _BACKING if workers == 1 else 0,
                    "weight_ids": _IDS,
                }
                for worker in range(workers)
            ]
            samples.append({
                "phase": phase,
                "errors": [],
                "processes": processes,
                "backing": [
                    {"suffix": "_w0_0", "identity": _IDS[0], "size_bytes": _OFFLOAD, "allocated_bytes": _OFFLOAD},
                    {"suffix": "_t", "identity": _IDS[1], "size_bytes": _MIB, "allocated_bytes": _MIB},
                ],
            })
    return {"page_size_bytes": 4096, "samples": samples}


def _summarize(evidence, workers, baseline=None):
    report = {
        "page_size_bytes": evidence["page_size_bytes"],
        "workers": workers,
        "phases": {},
        "backing_sizes": None,
        "failure": None,
    }
    for sample in evidence["samples"]:
        _host_memory._record_sample(report, sample, {"_w0_0": _OFFLOAD}, baseline)
    return report


def _qualify(one, two):
    baseline = _summarize(one, 1)
    return qualify_shared_host_memory(baseline, _summarize(two, 2, baseline), _OFFLOAD)


def test_smaps_identifies_shared_backing_even_when_reported_as_private():
    smaps = """1000-2000 rw-s 00000000 00:01 100 /dev/shm/model_w0_0
Rss: 8 kB
Pss: 8 kB
Private_Clean: 0 kB
Private_Dirty: 8 kB
Swap: 0 kB
2000-3000 rw-p 00000000 00:01 101 /other/file
Rss: 4 kB
Pss: 4 kB
Private_Clean: 1 kB
Private_Dirty: 3 kB
Swap: 0 kB
"""
    identity = (os.makedev(0, 1), 100)
    result = parse_smaps(smaps, {identity})
    assert result["pss_bytes"] == result["private_bytes"] == 12 * 1024
    assert result["weight_pss_bytes"] == result["weight_private_bytes"] == 8 * 1024
    assert result["weight_ids"] == [identity]
    with pytest.raises(AssertionError, match=r"incomplete|Incomplete"):
        parse_smaps(smaps.removesuffix("Swap: 0 kB\n"), {identity})


def test_shared_pages_count_once_despite_doubled_rss_and_single_mapping_private():
    one, two = _evidence(1), _evidence(2)
    result = _qualify(one, two)
    assert not result["failures"]
    assert result["phases"]["warm"]["shared_tree_non_weight_pss_max_bytes"] == 2 * _MIB
    # Startup allocations are recorded separately, never used to loosen budgets.
    peak = deepcopy(one["samples"][0])
    peak["phase"] = "startup"
    peak["processes"][0]["pss_bytes"] += 1000 * _MIB
    one["samples"].append(peak)
    assert _qualify(one, two) == result


@pytest.mark.parametrize("in_helper", [False, True])
def test_full_private_copy_is_rejected_even_outside_weight_holding_worker(in_helper):
    one, two = _evidence(1), _evidence(2)
    processes = two["samples"][0]["processes"]
    if in_helper:
        helper = dict(processes[0], pid=1000, weight_ids=[], weight_pss_bytes=0, weight_private_bytes=0)
        processes.append(helper)
    processes[-1]["pss_bytes"] += _OFFLOAD
    processes[-1]["private_bytes"] += _OFFLOAD
    assert any("process-tree non-weight memory exceeds" in failure for failure in _qualify(one, two)["failures"])


@pytest.mark.parametrize("problem", ["missing", "error", "mapping", "backing", "population", "swap", "pss"])
def test_incomplete_or_inconsistent_evidence_cannot_qualify(problem):
    one, two = _evidence(1), _evidence(2)
    sample = two["samples"][0]
    if problem == "missing":
        two["samples"].pop(0)
    elif problem == "error":
        sample["errors"].append("permission denied")
    elif problem == "mapping":
        sample["processes"][1]["weight_ids"] = []
    elif problem == "backing":
        sample["backing"][0]["size_bytes"] -= 4096
    elif problem == "population":
        sample["backing"][0]["allocated_bytes"] -= 4096
    elif problem == "swap":
        sample["processes"][0]["swap_bytes"] = 4096
    elif problem == "pss":
        sample["processes"][0]["weight_pss_bytes"] //= 2
    assert _qualify(one, two)["failures"]


def test_noisy_baseline_cannot_hide_a_full_copy():
    one, two = _evidence(1), _evidence(2)
    for sample in one["samples"]:
        sample["processes"][0]["pss_bytes"] += _OFFLOAD
    assert any("qualification is inconclusive" in failure for failure in _qualify(one, two)["failures"])


def test_sampler_preserves_first_failure_in_its_only_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(_host_memory.Path, "is_file", lambda _path: True)

    def denied(*_args):
        raise PermissionError("smaps")

    monkeypatch.setattr(_host_memory, "sample_process_tree", denied)
    sampler = HostMemorySampler(123, "namespace", {"_w0_0": _OFFLOAD}, 1)
    sampler.phase("warm")
    sampler._sample()
    report = sampler.finish(tmp_path / "host_memory.json")
    assert "samples" not in report
    assert "PermissionError" in report["failure"]["sample"]["errors"][0]
    assert report["phases"]["warm"].get("peak_pss_bytes") is None
    assert report["phases"]["warm"]["complete_sample_count"] == 0
    assert json.loads((tmp_path / "host_memory.json").read_text()) == report
    assert list(tmp_path.iterdir()) == [tmp_path / "host_memory.json"]
    assert not sampler._thread.is_alive()


def test_running_statistics_keep_peaks_without_sample_history():
    raw = _evidence(2)
    # Different workers peak in different samples; never sum their individual peaks.
    raw["samples"][0]["processes"][0]["pss_bytes"] += _MIB
    raw["samples"][1]["processes"][1]["pss_bytes"] += 2 * _MIB
    report = _summarize(raw, 2)
    assert "samples" not in report
    assert report["failure"] is None
    warm = report["phases"]["warm"]
    assert warm["peak_pss_bytes"] == _BACKING + 4 * _MIB
    assert warm["peak_rss_bytes"] == 2 * (_BACKING + _MIB)
    assert warm["peak_shared_backing_bytes"] == _BACKING
    assert warm["non_weight_pss_min_bytes"] == 2 * _MIB
    assert report["phases"]["inference"]["peak_pss_bytes"] == _BACKING + 2 * _MIB

    initial_size = len(json.dumps(report))
    for _ in range(1000):
        _host_memory._record_sample(report, raw["samples"][0], {"_w0_0": _OFFLOAD})
    assert warm["sample_count"] == warm["complete_sample_count"] == 1003
    assert len(json.dumps(report)) < initial_size + 20


def test_online_budget_failure_keeps_one_snapshot_and_continues_counting():
    baseline = _summarize(_evidence(1), 1)
    report = _summarize(_evidence(2), 2, baseline)
    sample = deepcopy(_evidence(2)["samples"][0])
    sample["processes"][0]["pss_bytes"] += _OFFLOAD
    _host_memory._record_sample(report, sample, {"_w0_0": _OFFLOAD}, baseline)
    failure = report["failure"]
    assert any("process-tree" in reason for reason in failure["reasons"])
    assert failure["sample"] == sample
    subsequent = deepcopy(sample)
    subsequent["processes"][0]["pss_bytes"] += _OFFLOAD
    _host_memory._record_sample(report, subsequent, {"_w0_0": _OFFLOAD}, baseline)
    assert report["failure"] is failure
    assert report["phases"]["warm"]["sample_count"] == 5
    assert report["phases"]["warm"]["peak_non_weight_pss_bytes"] == 2 * _MIB + 2 * _OFFLOAD


def test_worker_budget_still_applies_when_tree_memory_fits():
    one, two = _evidence(1), _evidence(2)
    processes = two["samples"][0]["processes"]
    processes[0]["pss_bytes"] += _MIB
    processes[1]["pss_bytes"] -= _MIB
    failures = _qualify(one, two)["failures"]
    assert any("worker non-weight memory exceeds" in failure for failure in failures)
    assert not any("process-tree" in failure for failure in failures)


def test_profile_uses_all_cpu_blocks_and_deduplicates_tensor_payload():
    tensor = SimpleNamespace(tensor_id=1, size_bytes=1024)
    other = SimpleNamespace(tensor_id=2, size_bytes=2048)
    state = SimpleNamespace(
        allocation_ordered={5: ["first", "empty", "second"], 9: ["third"]},
        load_strategy={"first": [tensor, tensor], "empty": [], "second": [other], "third": [tensor]},
    )
    sizes, duplicate_bytes = _profile_offload_backing(state)
    assert sizes == {"_w0_0": 1024, "_w0_1": 2048, "_w1_0": 1024}
    assert duplicate_bytes == 3072


def test_sampler_covers_descendants_and_counts_backing_by_identity(monkeypatch):
    paths = [
        SimpleNamespace(
            name=name,
            stat=lambda: SimpleNamespace(st_dev=1, st_ino=100, st_size=4096, st_blocks=8),
        )
        for name in ("weights_w0_0", "weights_w0_1", "weights_t", "weights_other", "other_w0_0")
    ]
    child = SimpleNamespace(pid=20, create_time=lambda: 1, is_running=lambda: True, name=lambda: "worker")
    root = SimpleNamespace(
        pid=10,
        create_time=lambda: 1,
        is_running=lambda: True,
        name=lambda: "server",
        children=lambda *, recursive: [child] if recursive else [],
    )
    reads = []
    monkeypatch.setattr(_host_memory.Path, "iterdir", lambda _path: iter(paths))
    monkeypatch.setattr(_host_memory.Path, "read_text", lambda path: reads.append(str(path)) or "smaps")
    monkeypatch.setattr(_host_memory.psutil, "Process", lambda _pid: root)
    monkeypatch.setattr(_host_memory, "parse_smaps", lambda _text, identities: {"weight_ids": sorted(identities)})
    sample = _host_memory.sample_process_tree(10, "weights")
    assert not sample["errors"]
    assert reads == ["/proc/10/smaps", "/proc/20/smaps"]
    assert [p["pid"] for p in sample["processes"]] == [10, 20]
    assert len(sample["backing"]) == 1
    assert sample["backing"][0]["suffix"] == "_t"
    assert sample["backing"][0]["allocated_bytes"] == 4096

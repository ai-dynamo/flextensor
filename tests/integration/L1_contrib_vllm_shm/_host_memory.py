# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linux host-memory evidence for the SHM acceptance test (no GPU dependency)."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import psutil

_SMAPS_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s+\S+\s+\S+\s+([0-9a-f]+):([0-9a-f]+)\s+(\d+)")
_FIELDS = ("Rss", "Pss", "Private_Clean", "Private_Dirty", "Swap")
_STEADY_PHASES = ("warm", "inference", "post_inference")


def parse_smaps(text: str, weight_ids: set[tuple[int, int]]) -> dict[str, Any]:
    """Sum mappings, distinguishing weight backing by device/inode, not page labels.

    Linux may report singly mapped MAP_SHARED pages as Private_Dirty. Subtract
    only the matching weight mappings when calculating non-weight overhead.
    """
    total = dict.fromkeys(_FIELDS, 0)
    weights = dict.fromkeys(_FIELDS, 0)
    mapped_weights: set[tuple[int, int]] = set()
    is_weight = False
    mapping_fields: set[str] = set()
    mappings = 0
    for line in text.splitlines():
        if header := _SMAPS_HEADER.match(line):
            if mappings:
                assert set(_FIELDS) <= mapping_fields, "Incomplete smaps mapping"
            identity = (os.makedev(int(header[1], 16), int(header[2], 16)), int(header[3]))
            is_weight = identity in weight_ids
            if is_weight:
                mapped_weights.add(identity)
            mapping_fields = set()
            mappings += 1
        else:
            key, separator, value = line.partition(":")
            if separator and key in total:
                amount, unit = value.split()
                assert unit == "kB", f"Unexpected smaps unit: {unit}"
                amount_bytes = int(amount) * 1024
                total[key] += amount_bytes
                if is_weight:
                    weights[key] += amount_bytes
                mapping_fields.add(key)
    assert mappings and set(_FIELDS) <= mapping_fields, "Empty or incomplete smaps"
    return {
        "rss_bytes": total["Rss"],
        "pss_bytes": total["Pss"],
        "private_bytes": total["Private_Clean"] + total["Private_Dirty"],
        "swap_bytes": total["Swap"],
        "weight_pss_bytes": weights["Pss"],
        "weight_private_bytes": weights["Private_Clean"] + weights["Private_Dirty"],
        "weight_ids": sorted(mapped_weights),
    }


def sample_process_tree(root_pid: int, namespace: str) -> dict[str, Any]:
    """Read all current descendants and count each named weight backing once."""
    weight_pattern = re.compile(re.escape(namespace) + r"(_w\d+_\d+|_t)\Z")
    backing: dict[tuple[int, int], dict[str, Any]] = {}
    errors: list[str] = []
    for path in Path("/dev/shm").iterdir():  # noqa: S108 - Linux POSIX SHM inspection.
        if match := weight_pattern.fullmatch(path.name):
            try:
                stat = path.stat()
            except FileNotFoundError:
                errors.append(f"SHM backing disappeared: {path.name}")
                continue
            identity = (stat.st_dev, stat.st_ino)
            backing[identity] = {
                "suffix": match[1],
                "identity": identity,
                "size_bytes": stat.st_size,
                "allocated_bytes": stat.st_blocks * 512,
            }
    root = psutil.Process(root_pid)
    processes = []
    for process in [root, *root.children(recursive=True)]:
        try:
            started = process.create_time()
            memory = parse_smaps(Path(f"/proc/{process.pid}/smaps").read_text(), set(backing))
            assert process.is_running() and process.create_time() == started, "Process changed during sampling"
            processes.append({"pid": process.pid, "started": started, "name": process.name(), **memory})
        except (OSError, psutil.Error, AssertionError) as error:
            errors.append(f"pid={process.pid}: {type(error).__name__}: {error}")
    return {"processes": processes, "backing": list(backing.values()), "errors": errors}


def _phase_budget(phase: dict[str, Any]) -> dict[str, int]:
    low, high = phase["non_weight_pss_min_bytes"], phase["peak_non_weight_pss_bytes"]
    tolerance = high - low + phase["peak_rounding_bytes"]
    return {
        "baseline_non_weight_pss_min_bytes": low,
        "baseline_non_weight_pss_max_bytes": high,
        "tolerance_bytes": tolerance,
        "tree_non_weight_budget_bytes": 2 * high + tolerance,
        "worker_non_weight_budget_bytes": high + tolerance,
    }


def _check_memory_budget(phase: dict[str, Any], budget: dict[str, int]) -> list[str]:
    failures = []
    if (
        max(phase["peak_non_weight_pss_bytes"], phase["peak_non_weight_private_bytes"])
        > budget["tree_non_weight_budget_bytes"]
    ):
        failures.append(f"process-tree non-weight memory exceeds {budget['tree_non_weight_budget_bytes']} bytes")
    if phase["peak_worker_non_weight_bytes"] > budget["worker_non_weight_budget_bytes"]:
        failures.append(f"worker non-weight memory exceeds {budget['worker_non_weight_budget_bytes']} bytes")
    return failures


def _record_sample(
    report: dict[str, Any],
    sample: dict[str, Any],
    expected_offload_backing: dict[str, int],
    baseline: dict[str, Any] | None = None,
) -> None:
    """Validate one sample and retain only running statistics and the first failure."""
    phase = sample["phase"]
    stats = report["phases"].setdefault(phase, {"sample_count": 0, "complete_sample_count": 0})
    stats["sample_count"] += 1
    if sample["errors"]:
        stats.setdefault("first_collection_error", sample["errors"][0])
    else:
        stats["complete_sample_count"] += 1
        processes = sample["processes"]
        non_weight_pss = sum(p["pss_bytes"] - p["weight_pss_bytes"] for p in processes)
        peaks = {
            "peak_rss_bytes": sum(p["rss_bytes"] for p in processes),
            "peak_pss_bytes": sum(p["pss_bytes"] for p in processes),
            "peak_non_weight_pss_bytes": non_weight_pss,
            "peak_non_weight_private_bytes": sum(p["private_bytes"] - p["weight_private_bytes"] for p in processes),
            "peak_worker_non_weight_bytes": max(
                (
                    max(p["pss_bytes"] - p["weight_pss_bytes"], p["private_bytes"] - p["weight_private_bytes"])
                    for p in processes
                    if p["weight_ids"]
                ),
                default=0,
            ),
            "peak_rounding_bytes": report["page_size_bytes"] * len(sample["backing"]) * len(processes),
            "peak_shared_backing_bytes": sum(f["allocated_bytes"] for f in sample["backing"]),
        }
        for key, value in peaks.items():
            stats[key] = max(stats.get(key, value), value)
        stats["non_weight_pss_min_bytes"] = min(stats.get("non_weight_pss_min_bytes", non_weight_pss), non_weight_pss)

    if phase not in _STEADY_PHASES:
        return
    failures = _validate_sample(sample, report, expected_offload_backing)
    sizes = {f["suffix"]: f["size_bytes"] for f in sample["backing"]}
    if report["backing_sizes"] is None:
        report["backing_sizes"] = sizes
    if sizes != report["backing_sizes"] or (baseline is not None and sizes != baseline["backing_sizes"]):
        failures.append("weight backing changed between samples or DP sizes")
    if baseline is not None and not sample["errors"]:
        base_phase = baseline["phases"].get(phase, {})
        if base_phase.get("complete_sample_count", 0):
            failures.extend(_check_memory_budget(stats, _phase_budget(base_phase)))
    if failures and report["failure"] is None:
        report["failure"] = {"phase": phase, "reasons": failures, "sample": sample}


class HostMemorySampler:
    """Sample and validate online; memory use does not grow with the sample count."""

    interval_seconds = 0.5

    def __init__(
        self,
        root_pid: int,
        namespace: str,
        expected_offload_backing: dict[str, int],
        workers: int,
        *,
        baseline: dict[str, Any] | None = None,
    ) -> None:
        if not Path("/proc/self/smaps").is_file():
            raise RuntimeError("Host-memory acceptance requires Linux /proc/PID/smaps")
        self.root_pid = root_pid
        self.namespace = namespace
        self.expected_offload_backing = expected_offload_backing
        self.baseline = baseline
        self.report: dict[str, Any] = {
            "namespace": namespace,
            "interval_seconds": self.interval_seconds,
            "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
            "workers": workers,
            "backing_sizes": None,
            "phases": {},
            "failure": None,
        }
        self._phase = "startup"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def phase(self, name: str) -> None:
        with self._lock:
            self._phase = name

    def _sample(self) -> None:
        with self._lock:
            try:
                sample = sample_process_tree(self.root_pid, self.namespace)
            except (OSError, psutil.Error, AssertionError) as error:
                sample = {"processes": [], "backing": [], "errors": [f"{type(error).__name__}: {error}"]}
            _record_sample(self.report, {"phase": self._phase, **sample}, self.expected_offload_backing, self.baseline)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def idle_samples(self, phase: str) -> None:
        self.phase(phase)
        for _ in range(3):
            self._sample()
            self._stop.wait(self.interval_seconds)

    def finish(self, output_path: Path) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.report, indent=2) + "\n")
        return self.report


def _validate_sample(
    sample: dict[str, Any],
    evidence: dict[str, Any],
    expected_offload_backing: dict[str, int],
) -> list[str]:
    failures: list[str] = []
    if sample["errors"]:
        failures.append(f"incomplete sample: {sample['errors']}")
        return failures
    files = sample["backing"]
    sizes = {f["suffix"]: f["size_bytes"] for f in files}
    if {k: v for k, v in sizes.items() if k != "_t"} != expected_offload_backing or sizes.get("_t", 0) <= 0:
        failures.append(f"SHM backing does not match the profile: {sizes}")
    identities = {tuple(f["identity"]) for f in files}
    participants = [p for p in sample["processes"] if p["weight_ids"]]
    workers = evidence["workers"]
    if len(participants) != workers or any(set(map(tuple, p["weight_ids"])) != identities for p in participants):
        failures.append(f"expected {workers} workers mapping the same complete weight backing")
    if any(p["swap_bytes"] for p in sample["processes"]):
        failures.append("swapping makes host-RAM qualification inconclusive")
    rounding = evidence["page_size_bytes"] * max(1, len(files)) * len(sample["processes"])
    allocated = sum(f["allocated_bytes"] for f in files)
    pss = sum(p["weight_pss_bytes"] for p in sample["processes"])
    if allocated < sum(f["size_bytes"] for f in files):
        failures.append("weight backing is not fully populated")
    if abs(allocated - pss) > rounding:
        failures.append(f"mapped weight PSS {pss} does not account for backing {allocated}")
    return failures


def qualify_shared_host_memory(
    baseline: dict[str, Any],
    shared: dict[str, Any],
    duplicate_weight_bytes: int,
) -> dict[str, Any]:
    """Apply the same full-copy-sensitive budgets to the accumulated statistics.

    PSS, private pages, and SHM backing overlap and are never added together.
    Startup peaks are sampled lower bounds and never contribute to the budgets.
    """
    failures: list[str] = []
    report: dict[str, Any] = {"failures": failures, "duplicate_weight_bytes": duplicate_weight_bytes, "phases": {}}
    if duplicate_weight_bytes <= 0:
        failures.append("Profile must offload weights to qualify host-memory sharing")
        return report
    for label, evidence in (("baseline", baseline), ("shared", shared)):
        if evidence["failure"] is not None:
            failure = evidence["failure"]
            failures.extend(f"{label}/{failure['phase']}: {reason}" for reason in failure["reasons"])
        for phase in _STEADY_PHASES:
            if evidence["phases"].get(phase, {}).get("complete_sample_count", 0) < 3:
                failures.append(f"{label}: fewer than three complete {phase} samples")
    if failures:
        return report
    if shared["backing_sizes"] != baseline["backing_sizes"]:
        failures.append("weight backing changed between DP sizes")
    for phase in _STEADY_PHASES:
        budget = _phase_budget(baseline["phases"][phase])
        observed = shared["phases"][phase]
        report["phases"][phase] = {
            **budget,
            "shared_tree_non_weight_pss_max_bytes": observed["peak_non_weight_pss_bytes"],
            "shared_tree_non_weight_private_max_bytes": observed["peak_non_weight_private_bytes"],
        }
        if budget["tree_non_weight_budget_bytes"] >= duplicate_weight_bytes:
            failures.append(
                f"{phase}: baseline/noise budget could hide a full weight copy; qualification is inconclusive"
            )
        failures.extend(f"{phase}: {reason}" for reason in _check_memory_budget(observed, budget))
    return report

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for integration tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # noqa: S404 - subprocess needed for test diagnostics
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

# Files that must all be present in the local HF cache before we trust it enough
# to enable offline mode. A partial cache left by an HTTP 429-interrupted
# download often contains ``config.json`` but not the tokenizer files, which
# later surfaces deep inside the tokenizer as ``vocab_file=None``. Probing all
# three avoids that false-positive. Safe for every LLM this suite currently
# exercises (all have a fast ``tokenizer.json``).
_REQUIRED_CACHE_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)
_WEIGHT_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

PCIE_QUERY_ARGS = (
    "nvidia-smi",
    (
        "--query-gpu=index,name,uuid,pci.bus_id,pcie.link.gen.gpucurrent,pcie.link.gen.max,"
        "pcie.link.gen.gpumax,pcie.link.gen.hostmax,pcie.link.width.current,pcie.link.width.max,"
        "driver_version"
    ),
    "--format=csv",
)


def _run_nvidia_smi(args: tuple[str, ...], *, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - controlled integration-test diagnostic command
        list(args),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _print_command_output(title: str, args: tuple[str, ...], *, timeout_s: int) -> subprocess.CompletedProcess[str]:
    print(f"=== {title} ===")
    result = _run_nvidia_smi(args, timeout_s=timeout_s)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    return result


def _model_weights_are_cached(model_name: str, load_from_cache: Callable[[str, str], object]) -> bool:
    for index_name in _WEIGHT_INDEX_FILES:
        index_path = load_from_cache(model_name, index_name)
        if not isinstance(index_path, str):
            continue

        try:
            index = json.loads(Path(index_path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            continue

        shards = list(index["weight_map"].values())
        if not shards or not all(isinstance(shard, str) for shard in shards):
            continue
        if all(isinstance(load_from_cache(model_name, shard), str) for shard in set(shards)):
            return True

    return any(isinstance(load_from_cache(model_name, filename), str) for filename in _WEIGHT_FILES)


@pytest.fixture(scope="session", autouse=True)
def require_integration_nvidia_gpu() -> None:
    """Require an NVIDIA GPU once per integration pytest session."""
    if shutil.which("nvidia-smi") is None:
        pytest.fail("nvidia-smi not found. Integration tests require an NVIDIA GPU.")

    summary = _print_command_output("nvidia-smi", ("nvidia-smi",), timeout_s=15)
    if summary.returncode != 0:
        pytest.fail("NVIDIA GPU not detected. Integration tests require an NVIDIA GPU.")

    pcie = _print_command_output("nvidia-smi PCIe details", PCIE_QUERY_ARGS, timeout_s=15)
    if pcie.returncode != 0:
        print("WARNING: failed to query detailed NVIDIA GPU PCIe information.")

    topo = _print_command_output("nvidia-smi topo -m", ("nvidia-smi", "topo", "-m"), timeout_s=15)
    if topo.returncode != 0:
        print("WARNING: failed to query NVIDIA GPU topology.")


def enable_offline_if_cached(model_name: str) -> None:
    """Set ``HF_HUB_OFFLINE=1`` when *model_name* is fully present in the local HF cache.

    Prevents 429 burst rate-limit crashes from HF API calls for files not
    present in the repo (e.g. quantization configs).  Safe to call for
    different models in one process: each call re-evaluates the current model.

    Each metadata file and every indexed model-weight shard must resolve to a
    cached path; otherwise online mode stays on so the download can finish.
    """
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        from huggingface_hub._snapshot_download import read_tree_cache
    except ImportError:
        read_tree_cache = None

    try:
        snapshot_path = Path(snapshot_download(model_name, local_files_only=True))
    except LocalEntryNotFoundError:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        print(f"[HF cache] {model_name} not fully cached (incomplete snapshot) — HF API calls enabled")
        return

    tree_entries = (
        None if read_tree_cache is None else read_tree_cache(str(snapshot_path.parent.parent), snapshot_path.name)
    )
    if tree_entries is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        print(f"[HF cache] {model_name} cannot verify snapshot completeness — HF API calls enabled")
        return

    missing = [name for name in _REQUIRED_CACHE_FILES if not isinstance(try_to_load_from_cache(model_name, name), str)]
    if not missing and not _model_weights_are_cached(model_name, try_to_load_from_cache):
        missing.append("model weights")
    if missing:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        print(f"[HF cache] {model_name} not fully cached (missing: {', '.join(missing)}) — HF API calls enabled")
        return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"[HF cache] {model_name} fully cached — set HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")

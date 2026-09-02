# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``enable_offline_if_cached`` in ``tests/integration/conftest.py``."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

hf_snapshot = pytest.importorskip("huggingface_hub._snapshot_download")
try:
    TreeCacheEntry = hf_snapshot.TreeCacheEntry
    write_tree_cache = hf_snapshot.write_tree_cache
except AttributeError:
    pytest.skip("huggingface_hub does not expose tree-cache support", allow_module_level=True)

from tests.integration.conftest import _REQUIRED_CACHE_FILES, enable_offline_if_cached  # noqa: E402

_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(autouse=True)
def _restore_hf_env() -> None:
    before = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    yield
    for k, v in before.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def _complete_local_snapshot(tmp_path: Path) -> None:
    storage = tmp_path / "models--Qwen--Qwen2.5-7B-Instruct"
    snapshot = storage / "snapshots" / "commit"
    snapshot.mkdir(parents=True)
    write_tree_cache(str(storage), "commit", {"config.json": TreeCacheEntry(size=1, blob_id="blob")})
    with mock.patch("huggingface_hub.snapshot_download", return_value=str(snapshot)):
        yield


def test_enables_offline_when_all_required_files_are_cached(capsys: pytest.CaptureFixture[str]) -> None:
    def fake_load(_model: str, filename: str) -> str | None:
        if filename in _REQUIRED_CACHE_FILES or filename == "model.safetensors":
            return "/fake/path"
        return None

    with mock.patch("huggingface_hub.try_to_load_from_cache", side_effect=fake_load) as m:
        enable_offline_if_cached(_MODEL)

    requested_files = {c.args[1] for c in m.call_args_list}
    assert set(_REQUIRED_CACHE_FILES) < requested_files
    assert "model.safetensors" in requested_files
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert "fully cached" in capsys.readouterr().out


def test_stays_online_when_huggingface_reports_incomplete_snapshot(capsys: pytest.CaptureFixture[str]) -> None:
    from huggingface_hub.errors import IncompleteSnapshotError

    with (
        mock.patch(
            "huggingface_hub.snapshot_download",
            side_effect=IncompleteSnapshotError("incomplete snapshot", "/fake/snapshot"),
        ),
        mock.patch("huggingface_hub.try_to_load_from_cache", return_value="/fake/path"),
    ):
        enable_offline_if_cached(_MODEL)

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "incomplete snapshot" in capsys.readouterr().out


def test_stays_online_when_snapshot_completeness_cannot_be_verified(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        mock.patch("huggingface_hub.snapshot_download", return_value="/fake/snapshots/commit"),
        mock.patch("huggingface_hub.try_to_load_from_cache", return_value="/fake/path"),
    ):
        enable_offline_if_cached(_MODEL)

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "cannot verify snapshot completeness" in capsys.readouterr().out


def test_stays_online_when_snapshot_tree_manifest_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    storage = tmp_path / "models--Qwen--Qwen2.5-7B-Instruct"
    snapshot = storage / "snapshots" / "invalid-commit"
    snapshot.mkdir(parents=True)
    tree = storage / "trees" / "invalid-commit.json"
    tree.write_text("{}")

    with (
        mock.patch("huggingface_hub.snapshot_download", return_value=str(snapshot)),
        mock.patch("huggingface_hub.try_to_load_from_cache", return_value="/fake/path"),
    ):
        enable_offline_if_cached(_MODEL)

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "cannot verify snapshot completeness" in capsys.readouterr().out


def test_stays_online_when_any_required_file_is_missing(capsys: pytest.CaptureFixture[str]) -> None:
    missing_file = "tokenizer.json"

    def fake_load(_model: str, filename: str) -> str | None:
        return None if filename == missing_file else "/fake/path"

    with mock.patch("huggingface_hub.try_to_load_from_cache", side_effect=fake_load):
        enable_offline_if_cached(_MODEL)

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    out = capsys.readouterr().out
    assert "not fully cached" in out
    assert missing_file in out


def test_stays_online_when_a_weight_shard_is_missing(tmp_path: Path) -> None:
    index_file = tmp_path / "model.safetensors.index.json"
    index_file.write_text(json.dumps({"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}))

    def fake_load(_model: str, filename: str) -> str | None:
        if filename in _REQUIRED_CACHE_FILES:
            return "/fake/path"
        if filename == index_file.name:
            return str(index_file)
        if filename == "model-00001.safetensors":
            return "/fake/shard"
        return None

    with mock.patch("huggingface_hub.try_to_load_from_cache", side_effect=fake_load):
        enable_offline_if_cached(_MODEL)

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_enables_offline_when_single_weight_file_exists_and_index_is_incomplete(tmp_path: Path) -> None:
    index_file = tmp_path / "pytorch_model.bin.index.json"
    index_file.write_text(json.dumps({"weight_map": {"a": "pytorch_model-00001.bin"}}))

    def fake_load(_model: str, filename: str) -> str | None:
        if filename in _REQUIRED_CACHE_FILES:
            return "/fake/path"
        if filename == "model.safetensors":
            return "/fake/model.safetensors"
        if filename == index_file.name:
            return str(index_file)
        return None

    with mock.patch("huggingface_hub.try_to_load_from_cache", side_effect=fake_load):
        enable_offline_if_cached(_MODEL)

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_reenables_online_for_a_different_uncached_model() -> None:
    cached_model = "Qwen/Qwen3.6-35B-A3B"
    uncached_model = "Qwen/Qwen3-30B-A3B-FP8"

    def fake_load(model: str, _filename: str) -> str | None:
        if model != cached_model or _filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            return None
        return "/fake/path"

    with mock.patch("huggingface_hub.try_to_load_from_cache", side_effect=fake_load):
        enable_offline_if_cached(cached_model)
        enable_offline_if_cached(uncached_model)

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHM namespace computation for cross-process weight sharing."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from flextensor.config import OffloadConfig  # noqa: TC001
from flextensor.gpu_budget import resolve_gpu_mem_bytes
from flextensor.offload_manager import DEFAULT_MANAGER_NAME

SHM_PROTOCOL_VERSION: int = 4
"""Bumped when SHM data layout changes in incompatible ways.

History:
    v4: coordinated standalone serving adds stable namespaces, per-handle participant
        identities, exact managed-tensor dtypes, a resident-tensor store, and an
        INITIALIZING/READY/FAILED coordinator state machine. Profile and tensor
        blocks share the coordinator's lifetime and heartbeat; a shared name
        registry permits cleanup even after the creator dies. A live-participant
        check gates adoption of existing generations.
    v3: coordination header stores the creator PID for exact liveness checks.
    v2: namespace hash keys changed (module_patterns → include_patterns, added exclude_patterns).
    v1: initial release.
"""


def compute_shm_namespace(
    model_path: str,
    config: OffloadConfig,
    extra_keys: dict[str, Any] | None = None,
    manager_name: str = DEFAULT_MANAGER_NAME,
) -> str:
    """Compute a deterministic SHM namespace from model identity and config.

    Args:
        model_path: Path to the model (will be resolved to canonical form).
        config: FlexTensor offload config.
        extra_keys: Additional keys affecting tensor layout (e.g., vLLM config fields).
            Values must be JSON-serializable.
        manager_name: OffloadManager name. Included in the hash so that multiple
            named managers sharing the same model/config get distinct namespaces.

    Returns:
        Base namespace string like "ft_a1b2c3d4" (8 hex chars).
        If config.shm_namespace is set, returns that value directly.
    """
    if config.shm_namespace is not None:
        return config.shm_namespace

    resolved_bytes = resolve_gpu_mem_bytes(config, context="computing SHM namespace")

    hash_input: dict[str, Any] = {
        "model_path": str(pathlib.Path(model_path).resolve()),
        "include_patterns": sorted(config.include_patterns),
        "exclude_patterns": sorted(config.exclude_patterns),
        "max_gpu_mem_bytes": resolved_bytes,
        "manager_name": manager_name,
    }
    if config.load_strategy is not None:
        hash_input["strategy"] = type(config.load_strategy).__name__

    if extra_keys:
        hash_input["extra"] = {k: str(v) for k, v in sorted(extra_keys.items())}

    canonical = json.dumps(hash_input, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:8]
    return f"ft_{digest}"


def apply_rank_suffix(
    base_namespace: str,
    tp_rank: int,
    pp_rank: int,
    ep_rank: int | None = None,
) -> str:
    """Append parallelism rank suffix to base namespace.

    Args:
        base_namespace: Base namespace from compute_shm_namespace().
        tp_rank: Tensor parallelism rank.
        pp_rank: Pipeline parallelism rank.
        ep_rank: Expert parallelism rank (omitted if None).

    Returns:
        Rank-scoped namespace like "ft_a1b2c3d4_tp0_pp0" or "ft_a1b2c3d4_tp0_pp0_ep2".
    """
    suffix = f"_tp{tp_rank}_pp{pp_rank}"
    if ep_rank is not None:
        suffix += f"_ep{ep_rank}"
    return base_namespace + suffix


def weight_block_name(namespace: str, block_index: int) -> str:
    """Return the SHM name for a weight block.

    Args:
        namespace: Rank-scoped namespace (e.g., "ft_abc123_tp0_pp0").
        block_index: Allocation block index.

    Returns:
        SHM block name like "ft_abc123_tp0_pp0_w0".
    """
    return f"{namespace}_w{block_index}"


def profile_block_name(namespace: str) -> str:
    """Return the SHM name for the profile metadata block.

    Args:
        namespace: Rank-scoped namespace.

    Returns:
        SHM block name like "ft_abc123_tp0_pp0_prof".
    """
    return f"{namespace}_prof"


def coord_block_name(namespace: str) -> str:
    """Return the SHM name for the coordination block.

    Args:
        namespace: Rank-scoped namespace.

    Returns:
        SHM block name like "ft_abc123_tp0_pp0_crd".
    """
    return f"{namespace}_crd"

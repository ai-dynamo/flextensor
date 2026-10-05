# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build FlexTensor runtime state from a fully loaded vLLM model."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import cast

import psutil
import torch
from vllm.logger import init_logger

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.config import OFFLOAD_TRANSFER_MODES, OffloadConfig, resolve_load_strategy
from flextensor.contrib.vllm.v2.errors import VllmFlexTensorV2Error

# ruff: ignore[noqa-comments] - compatibility with the pre-commit Ruff version.
from flextensor.contrib.vllm.v2.model_scan import (  # noqa: TC001 — beartype resolves annotations at runtime.
    LoadedModelScan,
)
from flextensor.gpu_budget import CUDAMemorySnapshot
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator
from flextensor.state_handler import TensorManagerState
from flextensor.strategy import BlockStrategyData, Strategy, StrategyResult, evaluate_strategy_result

LOGGER = init_logger("vllm.flextensor.v2.state_builder")


def resolve_cuda_device(device_gpu: torch.device | str | int) -> torch.device:
    try:
        device = torch.device(device_gpu)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise VllmFlexTensorV2Error(f"invalid CUDA device {device_gpu!r}: {exc}") from exc
    if device.type != "cuda":
        raise VllmFlexTensorV2Error(f"expected a CUDA device, got {device}")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())  # type: ignore[unreachable]
    return device


# ruff: ignore[noqa-comments] - compatibility with the pre-commit Ruff version.
def _validated_block_data(  # noqa: C901 - validates one block-data contract.
    block_data: BlockStrategyData,
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    *,
    loader_type: str,
    strategy_name: str,
) -> BlockStrategyData:
    """Validate strategy output before evaluation and state construction.

    This deliberately overlaps ``TensorManagerState.validate_internal()``:
    strategy-owned data must fail at its boundary before downstream code
    assumes its mappings are well formed.
    """
    try:
        if not isinstance(block_data.label_to_size_map, dict) or any(
            not isinstance(label, str) or type(size) is not int or size < 0
            for label, size in block_data.label_to_size_map.items()
        ):
            raise TypeError("label_to_size_map must be dict[str, non-negative int]")
        if not isinstance(block_data.allocation_ordered, dict) or any(
            type(block_id) is not int
            or block_id < 0
            or not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
            for block_id, labels in block_data.allocation_ordered.items()
        ):
            raise TypeError("allocation_ordered must be dict[non-negative int, list[str]]")
        if not isinstance(block_data.label_to_block_id, dict) or any(
            not isinstance(label, str) or type(block_id) is not int or block_id < 0
            for label, block_id in block_data.label_to_block_id.items()
        ):
            raise TypeError("label_to_block_id must be dict[str, non-negative int]")
        if not isinstance(block_data.transfer_to_compute_map, dict) or any(
            not isinstance(transfer, str) or not isinstance(compute, str)
            for transfer, compute in block_data.transfer_to_compute_map.items()
        ):
            raise TypeError("transfer_to_compute_map must be dict[str, str]")
        if isinstance(block_data.block_sizes, dict):
            block_sizes = dict(block_data.block_sizes)
        elif isinstance(block_data.block_sizes, list):
            block_sizes = dict(enumerate(block_data.block_sizes))
        else:
            raise TypeError("block_sizes must be dict[int, int] or list[int]")
        if any(
            type(block_id) is not int or block_id < 0 or type(size) is not int or size < 0
            for block_id, size in block_sizes.items()
        ):
            raise TypeError("block_sizes must contain non-negative integer IDs and sizes")
    except (AttributeError, TypeError, ValueError) as exc:
        raise VllmFlexTensorV2Error(
            f"conservative state strategy returned malformed block_data: strategy={strategy_name} error={exc}"
        ) from exc

    allocation_ordered = {block_id: list(labels) for block_id, labels in block_data.allocation_ordered.items()}
    label_to_size_map = dict(block_data.label_to_size_map)
    label_to_block_id = dict(block_data.label_to_block_id)
    transfer_to_compute_map = dict(block_data.transfer_to_compute_map)
    known_labels = {layer.label for layer in layer_stats}
    block_labels = {
        *label_to_size_map,
        *(label for labels in allocation_ordered.values() for label in labels),
        *label_to_block_id,
        *transfer_to_compute_map,
        *transfer_to_compute_map.values(),
    }
    unknown_labels = sorted(block_labels - known_labels)
    if unknown_labels:
        raise VllmFlexTensorV2Error(
            f"conservative state strategy returned unknown block labels: {unknown_labels} strategy={strategy_name}"
        )

    load_stats = {label: statistics for label, statistics in strategy_map.items() if statistics}
    load_labels = set(load_stats)
    allocated_labels = [label for labels in allocation_ordered.values() for label in labels]
    allocation_counts = Counter(allocated_labels)
    missing = sorted(load_labels - allocation_counts.keys())
    duplicate = sorted(label for label, count in allocation_counts.items() if count != 1)
    unexpected = sorted(allocation_counts.keys() - load_labels)
    empty_blocks = sorted(block_id for block_id, labels in allocation_ordered.items() if not labels)
    if missing or duplicate or unexpected or empty_blocks:
        raise VllmFlexTensorV2Error(
            "conservative state block_data allocation_ordered is inconsistent: "
            f"missing={missing} duplicate={duplicate} unexpected={unexpected} empty_blocks={empty_blocks} "
            f"strategy={strategy_name}"
        )

    expected_label_to_block_id = {
        label: block_id for block_id, labels in allocation_ordered.items() for label in labels
    }
    if label_to_block_id != expected_label_to_block_id:
        raise VllmFlexTensorV2Error(
            "conservative state block_data label_to_block_id disagrees with allocation_ordered: "
            f"strategy={strategy_name}"
        )

    logical_label_sizes = {
        label: sum(statistic.size_bytes for statistic in statistics) for label, statistics in load_stats.items()
    }
    expected_label_sizes = logical_label_sizes if loader_type == "raw_block_transfer" or label_to_size_map else {}
    if label_to_size_map != expected_label_sizes:
        raise VllmFlexTensorV2Error(
            "conservative state block_data label_to_size_map disagrees with selected tensor sizes: "
            f"strategy={strategy_name}"
        )

    expected_block_sizes = {
        block_id: max(logical_label_sizes[label] for label in labels) for block_id, labels in allocation_ordered.items()
    }
    if block_sizes != expected_block_sizes:
        raise VllmFlexTensorV2Error(
            f"conservative state block_data block_sizes disagrees with allocation capacities: strategy={strategy_name}"
        )

    consuming_label_by_id = {statistic.tensor_id: layer.label for layer in layer_stats for statistic in layer.tensors}
    expected_transfer_to_compute_map = {}
    for transfer_label, statistics in load_stats.items():
        compute_labels = {consuming_label_by_id[statistic.tensor_id] for statistic in statistics}
        if len(compute_labels) != 1:
            raise VllmFlexTensorV2Error(
                "conservative state block_data transfer label selects tensors from multiple compute labels: "
                f"transfer_label={transfer_label!r} strategy={strategy_name}"
            )
        expected_transfer_to_compute_map[transfer_label] = compute_labels.pop()
    if transfer_to_compute_map != expected_transfer_to_compute_map:
        raise VllmFlexTensorV2Error(
            "conservative state block_data transfer_to_compute_map disagrees with selected tensor consumers: "
            f"strategy={strategy_name}"
        )

    return BlockStrategyData(
        label_to_size_map=label_to_size_map,
        allocation_ordered=allocation_ordered,
        block_sizes=block_sizes,
        label_to_block_id=label_to_block_id,
        transfer_to_compute_map=transfer_to_compute_map,
    )


def merge_profile_statistics(
    scan_result: LoadedModelScan,
    profile: TensorManagerState,
) -> LoadedModelScan:
    """Rebind compatible saved timings to a fresh model scan."""
    profile.validate_internal()
    fresh_labels = [layer.label for layer in scan_result.layer_stats]
    saved_labels = [layer.label for layer in profile.stats]
    if saved_labels != fresh_labels:
        raise VllmFlexTensorV2Error(f"saved profile layer order mismatch: saved={saved_labels} current={fresh_labels}")

    merged_layers: list[LayerStatistics] = []
    for fresh_layer, saved_layer in zip(scan_result.layer_stats, profile.stats, strict=True):
        if not math.isfinite(saved_layer.duration) or saved_layer.duration <= 0:
            raise VllmFlexTensorV2Error(
                f"saved profile has invalid duration for layer {fresh_layer.label!r}: {saved_layer.duration!r}"
            )
        fresh_by_name = {tensor.name: tensor for tensor in fresh_layer.tensors}
        saved_by_name = {tensor.name: tensor for tensor in saved_layer.tensors}
        if set(saved_by_name) != set(fresh_by_name):
            raise VllmFlexTensorV2Error(f"saved profile tensor inventory mismatch for layer {fresh_layer.label!r}")
        for name, fresh_tensor in fresh_by_name.items():
            if saved_by_name[name].size_bytes != fresh_tensor.size_bytes:
                raise VllmFlexTensorV2Error(
                    f"saved profile tensor size mismatch for {name!r}: "
                    f"saved={saved_by_name[name].size_bytes} current={fresh_tensor.size_bytes}"
                )
        merged_layers.append(
            LayerStatistics(
                label=fresh_layer.label,
                tensors=list(fresh_layer.tensors),
                duration=saved_layer.duration,
            )
        )
    return replace(scan_result, layer_stats=merged_layers)


# ruff: ignore[noqa-comments] - compatibility with the pre-commit Ruff version.
def _compute_strategy_result(  # noqa: C901 - validates one strategy result.
    strategy: Strategy,
    layer_stats: list[LayerStatistics],
    memory_stats: dict[int, float],
    strategy_budget: int,
    *,
    loader_type: str,
    canonical_by_id: dict[int, TensorStatistics],
) -> tuple[StrategyResult, set[int]]:
    strategy_name = type(strategy).__name__
    result = strategy.compute(layer_stats, memory_stats, strategy_budget)
    if not isinstance(result, StrategyResult) or not isinstance(result.strategy_map, dict):
        raise VllmFlexTensorV2Error(
            "conservative state strategy returned malformed result: "
            f"expected=StrategyResult actual={type(result).__name__} strategy={strategy_name}"
        )
    if any(not isinstance(label, str) for label in result.strategy_map):
        raise VllmFlexTensorV2Error(f"conservative state strategy returned non-string labels: strategy={strategy_name}")
    known_labels = {layer.label for layer in layer_stats}
    unknown_labels = sorted(set(result.strategy_map) - known_labels)
    if unknown_labels:
        raise VllmFlexTensorV2Error(
            f"conservative state strategy returned unknown labels: {unknown_labels} strategy={strategy_name}"
        )

    normalized_strategy: dict[str, list[TensorStatistics]] = {}
    selected_ids: set[int] = set()
    for label, returned_statistics in result.strategy_map.items():
        if not isinstance(returned_statistics, list) or any(
            not isinstance(statistic, TensorStatistics) for statistic in returned_statistics
        ):
            raise VllmFlexTensorV2Error(
                "conservative state strategy_map entries must be list[TensorStatistics]: "
                f"label={label!r} strategy={strategy_name}"
            )
        normalized_statistics = []
        for returned in returned_statistics:
            canonical = canonical_by_id.get(returned.tensor_id)
            if canonical is None:
                raise VllmFlexTensorV2Error(
                    "conservative state strategy selected absent or ineligible tensor: "
                    f"tensor_id={returned.tensor_id} strategy={strategy_name}"
                )
            if returned != canonical:
                raise VllmFlexTensorV2Error(
                    "conservative state strategy returned conflicting tensor metadata: "
                    f"tensor_id={returned.tensor_id} expected={canonical} "
                    f"actual={returned} strategy={strategy_name}"
                )
            if returned.tensor_id in selected_ids:
                raise VllmFlexTensorV2Error(
                    "conservative state strategy selected a tensor more than once: "
                    f"tensor_id={returned.tensor_id} strategy={strategy_name}"
                )
            selected_ids.add(returned.tensor_id)
            normalized_statistics.append(canonical)
        normalized_strategy[label] = normalized_statistics

    block_data = result.block_data
    if loader_type != "strategy" and not isinstance(block_data, BlockStrategyData):
        raise VllmFlexTensorV2Error(
            f"conservative state block loader requires block_data: loader_type={loader_type} strategy={strategy_name}"
        )
    if loader_type != "strategy":
        block_data = _validated_block_data(
            cast("BlockStrategyData", block_data),
            normalized_strategy,
            layer_stats,
            loader_type=loader_type,
            strategy_name=strategy_name,
        )
    normalized_result = StrategyResult(strategy_map=normalized_strategy, block_data=block_data)
    try:
        score = evaluate_strategy_result(
            normalized_result,
            layer_stats,
            strategy_name=strategy_name,
            interpolator=MemoryTransferInterpolator(memory_stats) if memory_stats else None,
            max_gpu_mem_bytes=strategy_budget,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise VllmFlexTensorV2Error(
            f"conservative state strategy result cannot be evaluated: strategy={strategy_name} error={exc}"
        ) from exc
    if not score.is_valid:
        raise VllmFlexTensorV2Error(
            "conservative state strategy result is invalid: "
            f"peak={score.peak_memory_bytes} budget={strategy_budget} "
            f"block_violations={score.consecutive_violations} strategy={strategy_name}"
        )
    return normalized_result, selected_ids


# ruff: ignore[noqa-comments] - compatibility with the pre-commit Ruff version.
def build_conservative_state(  # noqa: C901 - validates and materializes one strategy result.
    scan_result: LoadedModelScan,
    config: OffloadConfig,
    device_gpu: torch.device,
    *,
    memory_stats: dict[int, float],
) -> TensorManagerState:
    loader_type = config.transfer_mode
    if loader_type not in OFFLOAD_TRANSFER_MODES:
        raise VllmFlexTensorV2Error(f"unsupported conservative state loader type: {loader_type!r}")

    if not memory_stats or any(
        type(size_bytes) is not int
        or size_bytes <= 0
        or not isinstance(duration_ms, (int, float))
        or not math.isfinite(duration_ms)
        or duration_ms <= 0
        for size_bytes, duration_ms in memory_stats.items()
    ):
        raise VllmFlexTensorV2Error(f"invalid memory transfer benchmark: {memory_stats!r}")
    try:
        interpolator = MemoryTransferInterpolator(memory_stats)
        layer_stats = [
            layer.model_copy(
                update={
                    "tensors": [
                        tensor.model_copy(update={"load_time_ms": interpolator.bytes_to_duration(tensor.size_bytes)})
                        for tensor in layer.tensors
                    ]
                }
            )
            for layer in scan_result.layer_stats
        ]
    except (TypeError, ValueError) as exc:
        raise VllmFlexTensorV2Error(f"invalid memory transfer benchmark: {exc}") from exc

    name_by_tensor_id = scan_result.name_by_tensor_id
    storage_by_tensor_id = scan_result.storage_by_tensor_id
    gpu_constant_ids = scan_result.gpu_constant_ids
    canonical_by_id: dict[int, TensorStatistics] = {}
    for layer in layer_stats:
        for statistic in layer.tensors:
            if statistic.tensor_id in canonical_by_id:
                raise VllmFlexTensorV2Error(
                    f"transformed eligible tensor appears in multiple runtime layers: tensor_id={statistic.tensor_id}"
                )
            canonical_by_id[statistic.tensor_id] = statistic

    snapshot = CUDAMemorySnapshot.capture(device_gpu)
    target_device = device_gpu
    current_gpu_storage = {
        inspection.key: inspection.nbytes
        for inspection in storage_by_tensor_id.values()
        if inspection.key.device == target_device
    }
    whole_model_budget = snapshot.available_bytes + sum(current_gpu_storage.values())
    if config.max_gpu_mem_fraction is not None:
        whole_model_budget = min(
            whole_model_budget,
            int(snapshot.total_bytes * config.max_gpu_mem_fraction),
        )
    resident_storage = {
        storage_by_tensor_id[tensor_id].key: storage_by_tensor_id[tensor_id].nbytes for tensor_id in gpu_constant_ids
    }
    resident_bytes = sum(resident_storage.values())
    strategy_budget = whole_model_budget - resident_bytes
    strategy = resolve_load_strategy(config)
    strategy_name = type(strategy).__name__
    if strategy_budget < 0:
        raise VllmFlexTensorV2Error(
            "conservative state resident tensors exceed GPU budget: "
            f"required={resident_bytes} available={whole_model_budget} strategy={strategy_name}"
        )

    normalized_result, selected_ids = _compute_strategy_result(
        strategy,
        layer_stats,
        memory_stats,
        strategy_budget,
        loader_type=loader_type,
        canonical_by_id=canonical_by_id,
    )
    normalized_strategy = normalized_result.strategy_map
    block_data = normalized_result.block_data

    incremental_host_storage = {
        storage_by_tensor_id[tensor_id].key: storage_by_tensor_id[tensor_id].nbytes
        for tensor_id in selected_ids
        if storage_by_tensor_id[tensor_id].key.device.type != "cpu"
    }
    incremental_host_bytes = sum(incremental_host_storage.values())
    host_available_bytes = psutil.virtual_memory().available
    if incremental_host_bytes > host_available_bytes:
        raise VllmFlexTensorV2Error(
            "conservative state selection exceeds incremental host budget: "
            f"required={incremental_host_bytes} available={host_available_bytes} strategy={strategy_name}"
        )

    release_strategy: dict[str, list[TensorStatistics]] = {}
    if loader_type == "strategy":
        for layer in layer_stats:
            selected = [statistic for statistic in layer.tensors if statistic.tensor_id in selected_ids]
            if selected:
                release_strategy[layer.label] = selected
        allocation_ordered: dict[int, list[str]] = {}
        label_to_size_map: dict[str, int] = {}
        block_sizes: dict[int, int] = {}
        label_to_block_id: dict[str, int] = {}
        transfer_to_compute_map: dict[str, str] = {}
        view_tensors_ids: list[int] = []
    else:
        block_data = cast("BlockStrategyData", block_data)
        allocation_ordered = {block_id: list(labels) for block_id, labels in block_data.allocation_ordered.items()}
        label_to_size_map = dict(block_data.label_to_size_map)
        block_sizes = (
            dict(block_data.block_sizes)
            if isinstance(block_data.block_sizes, dict)
            else dict(enumerate(block_data.block_sizes))
        )
        label_to_block_id = dict(block_data.label_to_block_id)
        transfer_to_compute_map = dict(block_data.transfer_to_compute_map)
        view_tensors_ids = [
            statistic.tensor_id
            for layer in layer_stats
            for statistic in layer.tensors
            if statistic.tensor_id in selected_ids
        ]

    managed_gpu_resident_storage = {
        inspection.key: inspection.nbytes
        for tensor_id, inspection in storage_by_tensor_id.items()
        if tensor_id not in selected_ids
    }
    managed_gpu_resident_bytes = sum(managed_gpu_resident_storage.values())
    LOGGER.info(
        "FlexTensor v2 GPU budget resolved: whole_model_budget_bytes=%d managed_gpu_resident_bytes=%d",
        whole_model_budget,
        managed_gpu_resident_bytes,
    )

    view_tensors_names = [name_by_tensor_id[tensor_id] for tensor_id in view_tensors_ids]
    gpu_tensors_names = [name for tensor_id, name in name_by_tensor_id.items() if tensor_id not in selected_ids]
    return TensorManagerState(
        loader_type=loader_type,
        tensor_id_to_name_map=dict(name_by_tensor_id),
        allocation_ordered=allocation_ordered,
        label_to_size_map=label_to_size_map,
        block_sizes=block_sizes,
        load_strategy=normalized_strategy,
        release_strategy=release_strategy,
        label_to_block_id=label_to_block_id,
        stats=layer_stats,
        transfer_to_compute_map=transfer_to_compute_map,
        view_tensors_ids=view_tensors_ids,
        view_tensors_names=view_tensors_names,
        gpu_tensors_names=gpu_tensors_names,
        shm_block_name_map=None,
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from flextensor.state_transition import (
    HostAllocation,
    MemoryCapacity,
    ModelPlacementState,
    StorageState,
    TensorState,
    TransitionSpec,
    plan_state_transition,
)


def _state(
    *, first_device: str = "cpu", second_device: str = "cuda:0", first_pinned: bool = False
) -> ModelPlacementState:
    return ModelPlacementState(
        tensors=(
            TensorState(
                id="tensor:0",
                names=("left.weight", "right.weight"),
                storage_id="storage:0",
                logical_bytes=8,
                kind="parameter",
            ),
            TensorState(
                id="tensor:1",
                names=("view",),
                storage_id="storage:0",
                logical_bytes=4,
                kind="parameter",
            ),
            TensorState(
                id="tensor:2",
                names=(),
                storage_id="storage:1",
                logical_bytes=8,
                kind="tensor",
            ),
        ),
        storages=(
            StorageState(id="storage:0", device=first_device, nbytes=16, pinned=first_pinned),
            StorageState(id="storage:1", device=second_device, nbytes=8, pinned=False),
        ),
    )


def test_model_placement_state_round_trips_through_json() -> None:
    state = _state()

    restored = ModelPlacementState.from_dict(json.loads(json.dumps(state.to_dict())))

    assert restored == state


def test_model_placement_state_rejects_string_instead_of_name_list() -> None:
    serialized = _state().to_dict()
    serialized["tensors"][0]["names"] = "ab"

    with pytest.raises(ValueError, match="names"):
        ModelPlacementState.from_dict(serialized)


def test_plan_groups_aliases_by_storage_and_orders_for_capacity() -> None:
    current = _state()
    target = _state(first_device="cuda:0", second_device="cpu")

    plan = plan_state_transition(
        current,
        target,
        transition=TransitionSpec(),
        capacity=MemoryCapacity(host_bytes=8, gpu_bytes=8),
    )

    assert [migration.storage_id for migration in plan.migrations] == ["storage:1", "storage:0"]
    assert plan.migrations[1].names == ("left.weight", "right.weight", "view")
    assert plan.peak_host_bytes == 8
    assert plan.peak_gpu_bytes == 8


def test_plan_accepts_expanded_tensor_larger_than_storage() -> None:
    expanded = ModelPlacementState(
        tensors=(
            TensorState(
                id="tensor:0",
                names=("scale",),
                storage_id="storage:0",
                logical_bytes=16,
                kind="buffer",
            ),
        ),
        storages=(StorageState(id="storage:0", device="cpu", nbytes=4, pinned=False),),
    )

    plan = plan_state_transition(
        expanded,
        expanded,
        transition=TransitionSpec(),
        capacity=MemoryCapacity(host_bytes=0, gpu_bytes=0),
    )

    assert plan.migrations == ()


def test_plan_prefers_promotions_when_both_directions_fit() -> None:
    current = _state()
    target = _state(first_device="cuda:0", second_device="cpu")

    plan = plan_state_transition(
        current,
        target,
        transition=TransitionSpec(),
        capacity=MemoryCapacity(host_bytes=1024, gpu_bytes=1024),
    )

    assert [migration.storage_id for migration in plan.migrations] == ["storage:0", "storage:1"]


def test_plan_counts_pinning_copy_and_ordered_host_allocations() -> None:
    current = _state(first_device="cpu", second_device="cpu")
    target = ModelPlacementState(
        tensors=current.tensors,
        storages=(
            StorageState(id="storage:0", device="cpu", nbytes=16, pinned=True),
            current.storages[1],
        ),
    )

    plan = plan_state_transition(
        current,
        target,
        transition=TransitionSpec(
            pinning_copy_storage_ids=("storage:0",),
            host_allocations=(HostAllocation(nbytes=128, temporary_copy_bytes=128),),
        ),
        capacity=MemoryCapacity(host_bytes=256, gpu_bytes=0),
    )

    assert plan.pinning_groups == (("left.weight", "right.weight", "view"),)
    assert plan.peak_host_bytes == 256


def test_plan_allows_pinned_cpu_storage_to_move_to_cuda() -> None:
    current = _state(first_device="cpu", second_device="cpu", first_pinned=True)
    target = _state(first_device="cuda:0", second_device="cpu")

    plan = plan_state_transition(
        current,
        target,
        transition=TransitionSpec(),
        capacity=MemoryCapacity(host_bytes=0, gpu_bytes=16),
    )

    assert [migration.storage_id for migration in plan.migrations] == ["storage:0"]
    assert plan.peak_gpu_bytes == 16


def test_plan_rejects_target_inventory_drift() -> None:
    current = _state()
    target = ModelPlacementState(
        tensors=(
            TensorState(
                id="tensor:0",
                names=current.tensors[0].names,
                storage_id="storage:0",
                logical_bytes=7,
                kind="parameter",
            ),
            *current.tensors[1:],
        ),
        storages=current.storages,
    )

    with pytest.raises(ValueError, match="tensor inventory"):
        plan_state_transition(
            current,
            target,
            transition=TransitionSpec(),
            capacity=MemoryCapacity(host_bytes=1024, gpu_bytes=1024),
        )


def test_plan_rejects_unmodeled_cross_device_cuda_migration() -> None:
    current = _state(first_device="cuda:0", second_device="cpu")
    target = _state(first_device="cuda:1", second_device="cpu")

    with pytest.raises(ValueError, match="not modeled by this planner"):
        plan_state_transition(
            current,
            target,
            transition=TransitionSpec(),
            capacity=MemoryCapacity(host_bytes=1024, gpu_bytes=1024),
        )


def test_plan_rejects_capacity_accounting_across_cuda_devices() -> None:
    current = _state(first_device="cuda:1", second_device="cpu")
    target = _state(first_device="cpu", second_device="cuda:0")

    with pytest.raises(ValueError, match="multiple CUDA devices"):
        plan_state_transition(
            current,
            target,
            transition=TransitionSpec(),
            capacity=MemoryCapacity(host_bytes=16, gpu_bytes=0),
        )


def test_plan_rejects_insufficient_transition_capacity() -> None:
    current = _state(first_device="cpu", second_device="cpu")

    with pytest.raises(RuntimeError, match="Insufficient capacity"):
        plan_state_transition(
            current,
            current,
            transition=TransitionSpec(host_allocations=(HostAllocation(nbytes=129),)),
            capacity=MemoryCapacity(host_bytes=128, gpu_bytes=0),
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serializable model placement state and read-only transition planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SCHEMA_VERSION = 1
_SEARCH_STATE_LIMIT = 10_000


@dataclass(frozen=True, slots=True)
class TensorState:
    """One tensor object and the backing storage it uses.

    ``id`` is snapshot-local; ``names`` holds its module paths, including aliases,
    while ``storage_id`` groups distinct tensor views backed by one allocation.
    """

    id: str
    names: tuple[str, ...]
    storage_id: str
    logical_bytes: int
    kind: str


@dataclass(frozen=True, slots=True)
class StorageState:
    """Placement of one logical backing allocation.

    ``id`` stays stable between current and target states. ``nbytes`` is the
    full backing allocation size, not the logical size of one tensor view.
    """

    id: str
    device: str
    nbytes: int
    pinned: bool


@dataclass(frozen=True, slots=True)
class ModelPlacementState:
    """JSON-serializable tensor and storage inventory."""

    tensors: tuple[TensorState, ...]
    storages: tuple[StorageState, ...]
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.schema_version,
            "tensors": [
                {
                    "id": tensor.id,
                    "names": list(tensor.names),
                    "storage_id": tensor.storage_id,
                    "logical_bytes": tensor.logical_bytes,
                    "kind": tensor.kind,
                }
                for tensor in self.tensors
            ],
            "storages": [
                {
                    "id": storage.id,
                    "device": storage.device,
                    "nbytes": storage.nbytes,
                    "pinned": storage.pinned,
                }
                for storage in self.storages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPlacementState:
        """Deserialize and validate an untrusted placement-state dictionary."""
        if not isinstance(data, dict):
            raise ValueError("Model placement state must be a dictionary")
        if data.get("version") != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported model placement schema version {data.get('version')!r}; expected {_SCHEMA_VERSION}"
            )
        tensor_rows = data.get("tensors")
        storage_rows = data.get("storages")
        if not isinstance(tensor_rows, list) or not isinstance(storage_rows, list):
            raise ValueError("Model placement state tensors and storages must be lists")
        if any(not isinstance(row, dict) or not isinstance(row.get("names"), list) for row in tensor_rows):
            raise ValueError("Each tensor entry must be a dictionary with a names list")
        if any(not isinstance(row, dict) for row in storage_rows):
            raise ValueError("Each storage entry must be a dictionary")

        try:
            tensors = tuple(
                TensorState(
                    id=row["id"],
                    names=tuple(row["names"]),
                    storage_id=row["storage_id"],
                    logical_bytes=row["logical_bytes"],
                    kind=row["kind"],
                )
                for row in tensor_rows
            )
            storages = tuple(
                StorageState(
                    id=row["id"],
                    device=row["device"],
                    nbytes=row["nbytes"],
                    pinned=row["pinned"],
                )
                for row in storage_rows
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid model placement state entry: {error}") from error

        state = cls(tensors=tensors, storages=storages)
        _validate_model_state(state)
        return state


@dataclass(frozen=True, slots=True)
class HostAllocation:
    """One retained host allocation and its temporary copy peak."""

    nbytes: int
    temporary_copy_bytes: int = 0


@dataclass(frozen=True, slots=True)
class TransitionSpec:
    """Memory outside model-owned storages required while adopting a target."""

    extra_gpu_bytes: int = 0
    pinning_copy_storage_ids: tuple[str, ...] = ()
    host_allocations: tuple[HostAllocation, ...] = ()
    release_host_storage_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryCapacity:
    """Extra host and GPU bytes available above the current model state."""

    host_bytes: int
    gpu_bytes: int


@dataclass(frozen=True, slots=True)
class StorageMigration:
    """One logical backing allocation to copy between host and GPU.

    ``storage_id`` identifies the same allocation in both states; the source
    and destination devices describe its placement before and after the copy.
    """

    storage_id: str
    names: tuple[str, ...]
    source_device: str
    destination_device: str
    nbytes: int


@dataclass(frozen=True, slots=True)
class StateTransitionPlan:
    """Read-only, capacity-safe storage migration plan."""

    migrations: tuple[StorageMigration, ...]
    pinning_groups: tuple[tuple[str, ...], ...]
    peak_host_bytes: int
    peak_gpu_bytes: int


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_device(device: object) -> None:
    valid = device == "cpu" or (
        isinstance(device, str) and device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
    )
    if not valid:
        raise ValueError(f"Unsupported device {device!r}; expected 'cpu' or an explicit CUDA device")


def _validate_model_state(state: ModelPlacementState) -> None:  # noqa: C901
    if state.schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model placement schema version {state.schema_version!r}; expected {_SCHEMA_VERSION}"
        )

    tensor_ids: set[str] = set()
    names: set[str] = set()
    referenced_storage_ids: set[str] = set()
    for tensor in state.tensors:
        if not isinstance(tensor.id, str) or not tensor.id or tensor.id in tensor_ids:
            raise ValueError(f"Tensor IDs must be unique non-empty strings: {tensor.id!r}")
        tensor_ids.add(tensor.id)
        if not isinstance(tensor.names, tuple) or any(not isinstance(name, str) or not name for name in tensor.names):
            raise ValueError(f"Tensor {tensor.id!r} names must be non-empty strings")
        duplicate_names = names.intersection(tensor.names)
        if len(set(tensor.names)) != len(tensor.names) or duplicate_names:
            raise ValueError(f"Tensor names must be unique: {sorted(duplicate_names or set(tensor.names))}")
        names.update(tensor.names)
        if not isinstance(tensor.storage_id, str) or not tensor.storage_id:
            raise ValueError(f"Tensor {tensor.id!r} has an invalid storage ID")
        referenced_storage_ids.add(tensor.storage_id)
        if not _is_non_negative_int(tensor.logical_bytes):
            raise ValueError(f"Tensor {tensor.id!r} logical_bytes must be a non-negative integer")
        if not isinstance(tensor.kind, str) or not tensor.kind:
            raise ValueError(f"Tensor {tensor.id!r} kind must be a non-empty string")

    storage_ids: set[str] = set()
    for storage in state.storages:
        if not isinstance(storage.id, str) or not storage.id or storage.id in storage_ids:
            raise ValueError(f"Storage IDs must be unique non-empty strings: {storage.id!r}")
        storage_ids.add(storage.id)
        _validate_device(storage.device)
        if not _is_non_negative_int(storage.nbytes):
            raise ValueError(f"Storage {storage.id!r} nbytes must be a non-negative integer")
        if type(storage.pinned) is not bool:
            raise ValueError(f"Storage {storage.id!r} pinned must be a boolean")
        if storage.pinned and storage.device != "cpu":
            raise ValueError(f"Storage {storage.id!r} cannot be pinned on {storage.device}")

    if referenced_storage_ids != storage_ids:
        raise ValueError(
            "Storage inventory must exactly match tensor references: "
            f"missing={sorted(referenced_storage_ids - storage_ids)}, "
            f"unreferenced={sorted(storage_ids - referenced_storage_ids)}"
        )


def _validate_transition_inputs(  # noqa: C901
    current: ModelPlacementState,
    target: ModelPlacementState,
    transition: TransitionSpec,
    capacity: MemoryCapacity,
) -> tuple[dict[str, StorageState], dict[str, StorageState]]:
    _validate_model_state(current)
    _validate_model_state(target)
    current_tensors_by_id = {tensor.id: tensor for tensor in current.tensors}
    target_tensors_by_id = {tensor.id: tensor for tensor in target.tensors}
    if current_tensors_by_id != target_tensors_by_id:
        raise ValueError("Current and target tensor inventory must match; only storage placement may change")

    current_storages_by_id = {storage.id: storage for storage in current.storages}
    target_storages_by_id = {storage.id: storage for storage in target.storages}
    if current_storages_by_id.keys() != target_storages_by_id.keys():
        raise ValueError("Current and target storage inventory must match")
    for storage_id, current_storage in current_storages_by_id.items():
        target_storage = target_storages_by_id[storage_id]
        if current_storage.nbytes != target_storage.nbytes:
            raise ValueError(f"Storage {storage_id!r} size differs between current and target state")
        if current_storage.pinned and current_storage.device == target_storage.device and not target_storage.pinned:
            raise ValueError(f"Unpinning storage {storage_id!r} is unsupported")
        if (
            current_storage.device != target_storage.device
            and current_storage.device.startswith("cuda:")
            and target_storage.device.startswith("cuda:")
        ):
            raise ValueError(
                "Cross-device CUDA migration is not modeled by this planner: "
                f"{current_storage.device} -> {target_storage.device}"
            )

    cuda_devices = {
        storage.device for state in (current, target) for storage in state.storages if storage.device != "cpu"
    }
    if len(cuda_devices) > 1:
        raise ValueError(f"Capacity accounting across multiple CUDA devices is unsupported: {sorted(cuda_devices)}")

    byte_counts = [capacity.host_bytes, capacity.gpu_bytes, transition.extra_gpu_bytes]
    byte_counts.extend(allocation.nbytes for allocation in transition.host_allocations)
    byte_counts.extend(allocation.temporary_copy_bytes for allocation in transition.host_allocations)
    if not all(_is_non_negative_int(value) for value in byte_counts):
        raise ValueError("Capacity and transition byte counts must be non-negative integers")

    pinning_ids = set(transition.pinning_copy_storage_ids)
    if len(pinning_ids) != len(transition.pinning_copy_storage_ids):
        raise ValueError("pinning_copy_storage_ids must be unique")
    newly_pinned_ids = {
        storage_id
        for storage_id, storage in current_storages_by_id.items()
        if not storage.pinned and target_storages_by_id[storage_id].pinned
    }
    if not pinning_ids <= newly_pinned_ids:
        raise ValueError("Pinning-copy storage IDs must refer to newly pinned target storages")

    release_ids = set(transition.release_host_storage_ids)
    if len(release_ids) != len(transition.release_host_storage_ids) or not release_ids <= target_storages_by_id.keys():
        raise ValueError("release_host_storage_ids must contain unique known storage IDs")
    if any(target_storages_by_id[storage_id].device != "cpu" for storage_id in release_ids):
        raise ValueError("Only target host storages can be released during loader construction")
    return current_storages_by_id, target_storages_by_id


def _tensor_names_by_storage_id(state: ModelPlacementState) -> dict[str, tuple[str, ...]]:
    names_by_storage_id: dict[str, list[str]] = {storage.id: [] for storage in state.storages}
    for tensor in state.tensors:
        names_by_storage_id[tensor.storage_id].extend(tensor.names)
    return {storage_id: tuple(sorted(tensor_names)) for storage_id, tensor_names in names_by_storage_id.items()}


def _find_migration_order(
    migrations: tuple[StorageMigration, ...],
    *,
    host_capacity: int,
    gpu_capacity: int,
    extra_gpu_bytes: int,
) -> list[StorageMigration] | None:
    # The remaining-set mask determines the cumulative memory deltas, regardless of migration order.
    full_mask = (1 << len(migrations)) - 1
    frontier: list[tuple[int, int, int, tuple[int, ...]]] = [(full_mask, 0, 0, ())]
    visited = {full_mask}
    while frontier:
        remaining, host_bytes, gpu_bytes, order = frontier.pop()
        if remaining == 0:
            if gpu_bytes + extra_gpu_bytes <= gpu_capacity:
                return [migrations[index] for index in order]
            continue
        if len(visited) >= _SEARCH_STATE_LIMIT:
            raise RuntimeError(f"Capacity-safe migration ordering search exceeded {_SEARCH_STATE_LIMIT} states")
        for index, migration in enumerate(migrations):
            bit = 1 << index
            if not remaining & bit:
                continue
            if migration.destination_device.startswith("cuda:"):
                next_host = host_bytes - migration.nbytes
                next_gpu = gpu_bytes + migration.nbytes
            else:
                next_host = host_bytes + migration.nbytes
                next_gpu = gpu_bytes - migration.nbytes
            if next_host > host_capacity or next_gpu > gpu_capacity:
                continue
            next_remaining = remaining ^ bit
            if next_remaining in visited:
                continue
            visited.add(next_remaining)
            frontier.append((next_remaining, next_host, next_gpu, (*order, index)))
    return None


def _capacity_error(required_host: int, required_gpu: int, capacity: MemoryCapacity) -> RuntimeError:
    return RuntimeError(
        "Insufficient capacity for state transition: "
        f"host(required={max(0, required_host)}, available={capacity.host_bytes}); "
        f"gpu(required={max(0, required_gpu)}, available={capacity.gpu_bytes})"
    )


def _order_migrations(
    migrations: tuple[StorageMigration, ...],
    *,
    transition: TransitionSpec,
    capacity: MemoryCapacity,
) -> list[StorageMigration]:
    pending = list(migrations)
    ordered: list[StorageMigration] = []
    # These are deltas from the current placement; freeing a source can make a delta negative.
    host_bytes = gpu_bytes = 0
    while pending:
        fitting_demotions = [
            migration
            for migration in pending
            if migration.destination_device == "cpu" and host_bytes + migration.nbytes <= capacity.host_bytes
        ]
        fitting_promotions = [
            migration
            for migration in pending
            if migration.destination_device.startswith("cuda:") and gpu_bytes + migration.nbytes <= capacity.gpu_bytes
        ]
        if not fitting_demotions and not fitting_promotions:
            # Greedy choices can dead-end, so exhaust bounded orderings before reporting insufficient capacity.
            fallback = _find_migration_order(
                migrations,
                host_capacity=capacity.host_bytes,
                gpu_capacity=capacity.gpu_bytes,
                extra_gpu_bytes=transition.extra_gpu_bytes,
            )
            if fallback is not None:
                return fallback
            final_host = sum(
                migration.nbytes if migration.destination_device == "cpu" else -migration.nbytes
                for migration in migrations
            )
            raise _capacity_error(final_host, -final_host + transition.extra_gpu_bytes, capacity)

        pending_promotions = sum(
            migration.nbytes for migration in pending if migration.destination_device.startswith("cuda:")
        )
        pending_demotions = sum(migration.nbytes for migration in pending if migration.destination_device == "cpu")
        gpu_needs_freeing = gpu_bytes + pending_promotions + transition.extra_gpu_bytes > capacity.gpu_bytes
        host_needs_freeing = host_bytes + pending_demotions > capacity.host_bytes
        if gpu_needs_freeing and fitting_demotions:
            candidates = fitting_demotions
        elif host_needs_freeing and fitting_promotions:
            candidates = fitting_promotions
        else:
            candidates = fitting_promotions or fitting_demotions
        migration = max(candidates, key=lambda item: (item.nbytes, item.names))
        pending.remove(migration)
        ordered.append(migration)
        if migration.destination_device.startswith("cuda:"):
            host_bytes -= migration.nbytes
            gpu_bytes += migration.nbytes
        else:
            host_bytes += migration.nbytes
            gpu_bytes -= migration.nbytes
    return ordered


def plan_state_transition(
    current: ModelPlacementState,
    target: ModelPlacementState,
    *,
    transition: TransitionSpec,
    capacity: MemoryCapacity,
) -> StateTransitionPlan:
    """Plan placement changes for one logical tensor/storage inventory."""
    current_storages, target_storages = _validate_transition_inputs(current, target, transition, capacity)
    names_by_storage = _tensor_names_by_storage_id(current)
    migrations = tuple(
        StorageMigration(
            storage_id=storage.id,
            names=names_by_storage[storage.id],
            source_device=storage.device,
            destination_device=target_storages[storage.id].device,
            nbytes=storage.nbytes,
        )
        for storage in current.storages
        if storage.device != target_storages[storage.id].device
    )
    ordered = _order_migrations(migrations, transition=transition, capacity=capacity)

    # Peak values measure additional bytes above the current placement, not total process memory.
    host_bytes = gpu_bytes = 0
    peak_host_bytes = peak_gpu_bytes = 0
    for migration in ordered:
        if migration.destination_device.startswith("cuda:"):
            host_bytes -= migration.nbytes
            gpu_bytes += migration.nbytes
        else:
            host_bytes += migration.nbytes
            gpu_bytes -= migration.nbytes
        peak_host_bytes = max(peak_host_bytes, host_bytes)
        peak_gpu_bytes = max(peak_gpu_bytes, gpu_bytes)

    # Loader construction follows migration: GPU blocks, copy peaks, retained host blocks, then released homes.
    gpu_bytes += transition.extra_gpu_bytes
    peak_gpu_bytes = max(peak_gpu_bytes, gpu_bytes)
    if gpu_bytes > capacity.gpu_bytes:
        raise _capacity_error(host_bytes, gpu_bytes, capacity)

    for storage_id in transition.pinning_copy_storage_ids:
        peak_host_bytes = max(peak_host_bytes, host_bytes + target_storages[storage_id].nbytes)
    for allocation in transition.host_allocations:
        host_bytes += allocation.nbytes
        peak_host_bytes = max(peak_host_bytes, host_bytes + allocation.temporary_copy_bytes)
    host_bytes -= sum(target_storages[storage_id].nbytes for storage_id in transition.release_host_storage_ids)
    if peak_host_bytes > capacity.host_bytes or host_bytes > capacity.host_bytes:
        raise _capacity_error(max(peak_host_bytes, host_bytes), gpu_bytes, capacity)

    newly_pinned_ids = {
        storage_id
        for storage_id, storage in current_storages.items()
        if not storage.pinned and target_storages[storage_id].pinned
    }
    pinning_groups = tuple(sorted(names_by_storage[storage_id] for storage_id in newly_pinned_ids))
    return StateTransitionPlan(
        migrations=tuple(ordered),
        pinning_groups=pinning_groups,
        peak_host_bytes=peak_host_bytes,
        peak_gpu_bytes=peak_gpu_bytes,
    )

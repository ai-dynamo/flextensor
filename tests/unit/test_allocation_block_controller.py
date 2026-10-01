# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from flextensor.allocation_block import AllocationBlock, AllocationManager
from flextensor.collectors import TensorStatistics
from flextensor.construction_rollback import ConstructionRollback
from flextensor.host_pinning import NoOpHostPinner
from flextensor.loaders import AllocationBlockController
from flextensor.state_handler import LoaderInputData
from flextensor.tensor_manager import TensorManager
from flextensor.utils import get_tensor_data, set_tensor_data


def _tensor_stats(tensor_id: int, size_bytes: int) -> TensorStatistics:
    return TensorStatistics(tensor_id=tensor_id, name=f"t{tensor_id}", size_bytes=size_bytes, load_time_ms=0.0)


def _unnamed_tensor_data(tensor: torch.Tensor) -> torch.Tensor:
    data = get_tensor_data(tensor)
    rename = getattr(data, "rename", None)
    return rename(None) if callable(rename) else data


def test_allocation_block_controller_has_no_post_copy_release_hook() -> None:
    assert not hasattr(AllocationBlockController, "release_memory")


@pytest.mark.parametrize("commit_construction", [False, True])
def test_allocation_block_controller_releases_tensor_storage_during_block_allocation(commit_construction: bool) -> None:
    label = "layer.0.weight"
    tensor = torch.arange(6, dtype=torch.float32)
    expected = tensor.clone()

    class _Controller(AllocationBlockController):
        def release_memory(self, tensors_list: list[torch.Tensor]) -> None:
            raise AssertionError("AllocationBlockController must not use a post-copy release hook")

    controller = _Controller(
        allocation_ordered={0: [label]},
        device_gpu=torch.device("cpu"),
        tensors_map={1: tensor},
        strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=tensor.numel() * tensor.element_size())]},
        label_to_block_id={label: 0},
        host_pinner=NoOpHostPinner(),
    )

    assert tensor.numel() == 0
    assert tensor.untyped_storage().nbytes() == 0
    controller.schedule_transfer(label, non_blocking=False)
    assert torch.equal(controller.get_tensor_id_to_view_mapping()[1], expected)

    if commit_construction:
        controller.commit_construction()
    controller.shutdown()

    if commit_construction:
        assert tensor.numel() == 0
    else:
        torch.testing.assert_close(tensor, expected)


class _ConstructionError(RuntimeError):
    pass


class _RejectDataAssignmentParameter(torch.nn.Parameter):
    @property
    def data(self) -> torch.Tensor:
        return get_tensor_data(self)

    @data.setter
    def data(self, _value: torch.Tensor) -> None:
        raise RuntimeError("custom parameter rejects .data assignment")


class _FakeBlock(AllocationBlock):
    def __init__(
        self,
        *,
        allocate_error: BaseException | None = None,
        project_error: BaseException | None = None,
        release_errors_remaining: int = 0,
        release_source_storage: bool = False,
    ) -> None:
        self.allocate_error = allocate_error
        self.project_error = project_error
        self.release_errors_remaining = release_errors_remaining
        self.release_source_storage = release_source_storage
        self.release_calls = 0
        self.tensors: list[torch.Tensor] = []
        self.views = [torch.empty(1)]

    def add(self, tensor: torch.Tensor) -> None:
        self.tensors.append(tensor)

    def allocate(self) -> list[torch.Tensor]:
        if self.allocate_error is not None:
            raise self.allocate_error
        if self.release_source_storage:
            # Rollback must preserve source metadata even if block views only carry its values.
            self.views = [
                _unnamed_tensor_data(tensor).clone(memory_format=torch.preserve_format) for tensor in self.tensors
            ]
            for tensor in self.tensors:
                set_tensor_data(tensor, torch.empty(0, dtype=tensor.dtype, device=tensor.device))
            self.tensors.clear()
        return self.views

    def project_views(self, gpu_block: "_FakeBlock") -> tuple[list[torch.Tensor], torch.Tensor]:
        del gpu_block
        if self.project_error is not None:
            raise self.project_error
        return self.views, torch.empty(1)

    def release(self) -> None:
        self.release_calls += 1
        if self.release_errors_remaining:
            self.release_errors_remaining -= 1
            raise RuntimeError("transient release failure")
        self.views = []


class _FakeAllocationManager:
    instances: ClassVar[list["_FakeAllocationManager"]] = []
    fail_allocate_number: int | None = None
    fail_gpu_creation = False
    fail_projection_number: int | None = None
    fail_release_number: int | None = None
    release_failures = 1
    release_source_storage = False
    _cpu_block_number = 0

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.fail_allocate_number = None
        cls.fail_gpu_creation = False
        cls.fail_projection_number = None
        cls.fail_release_number = None
        cls.release_failures = 1
        cls.release_source_storage = False
        cls._cpu_block_number = 0

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.blocks: list[_FakeBlock] = []
        self.gpu_blocks: list[_FakeBlock] = []
        type(self).instances.append(self)

    def block(self) -> _FakeBlock:
        cls = type(self)
        cls._cpu_block_number += 1
        error = (
            _ConstructionError("second block allocation failed")
            if cls.fail_allocate_number == cls._cpu_block_number
            else None
        )
        project_error = (
            _ConstructionError("view projection failed")
            if cls.fail_projection_number == cls._cpu_block_number
            else None
        )
        block = _FakeBlock(
            allocate_error=error,
            project_error=project_error,
            release_errors_remaining=(cls.release_failures if cls.fail_release_number == cls._cpu_block_number else 0),
            release_source_storage=cls.release_source_storage,
        )
        self.blocks.append(block)
        return block

    def create_max_block(self, device: str) -> _FakeBlock:
        del device
        cls = type(self)
        if cls.fail_gpu_creation:
            raise _ConstructionError("gpu block creation failed")
        block = _FakeBlock()
        self.gpu_blocks.append(block)
        return block


def _all_fake_blocks() -> list[_FakeBlock]:
    return [block for manager in _FakeAllocationManager.instances for block in [*manager.blocks, *manager.gpu_blocks]]


def _assert_failed_controller_was_rolled_back(controller: AllocationBlockController) -> None:
    assert all(block.release_calls == 1 for block in _all_fake_blocks())
    assert controller.block_map_cpu == {}
    assert controller.block_map_gpu == {}
    assert controller.label_to_cpu_tensor_id_map == {}
    assert controller.label_to_gpu_block == {}
    assert controller.label_to_tensor_views_map == {}
    assert controller.gpu_block_view_map == {}


def test_constructor_rolls_back_prior_and_current_blocks_when_second_allocate_fails(monkeypatch) -> None:
    _FakeAllocationManager.reset()
    _FakeAllocationManager.fail_allocate_number = 2
    controller = AllocationBlockController.__new__(AllocationBlockController)
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)

    with pytest.raises(_ConstructionError, match="second block allocation failed") as raised:
        AllocationBlockController.__init__(
            controller,
            allocation_ordered={0: ["left", "right"]},
            device_gpu=torch.device("cpu"),
            tensors_map={1: torch.ones(1)},
            strategy_map={"left": [_tensor_stats(1, 4)], "right": [_tensor_stats(1, 4)]},
            label_to_block_id={"left": 0, "right": 0},
            use_shm=True,
            block_name_fn=lambda index: f"test_block_{index}",
            host_pinner=NoOpHostPinner(),
        )

    assert str(raised.value) == "second block allocation failed"
    _assert_failed_controller_was_rolled_back(controller)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("strategy", "missing"),
        ("gpu", "gpu block creation failed"),
        ("projection", "view projection failed"),
    ],
)
def test_constructor_rolls_back_every_resource_without_masking_primary_failure(monkeypatch, failure, message) -> None:
    _FakeAllocationManager.reset()
    labels = ["left"]
    strategy_map = {"left": [_tensor_stats(1, 4)]}
    if failure == "strategy":
        labels.append("missing")
    elif failure == "gpu":
        _FakeAllocationManager.fail_gpu_creation = True
    else:
        labels.append("right")
        strategy_map["right"] = [_tensor_stats(1, 4)]
        _FakeAllocationManager.fail_projection_number = 1

    controller = AllocationBlockController.__new__(AllocationBlockController)
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)
    with pytest.raises((KeyError, _ConstructionError), match=message):
        AllocationBlockController.__init__(
            controller,
            allocation_ordered={0: labels},
            device_gpu=torch.device("cpu"),
            tensors_map={1: torch.ones(1)},
            strategy_map=strategy_map,
            label_to_block_id=dict.fromkeys(labels, 0),
            use_shm=True,
            shm_block_name_map={},
            block_name_fn=lambda index: f"test_block_{index}",
            host_pinner=NoOpHostPinner(),
        )

    _assert_failed_controller_was_rolled_back(controller)


def test_failed_construction_does_not_publish_names_and_retry_reconstructs(monkeypatch) -> None:
    _FakeAllocationManager.reset()
    _FakeAllocationManager.fail_allocate_number = 2
    published_names = {"existing": "existing_block"}
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)

    with pytest.raises(_ConstructionError, match="second block allocation failed"):
        AllocationBlockController(
            allocation_ordered={0: ["left", "right"]},
            device_gpu=torch.device("cpu"),
            tensors_map={1: torch.ones(1)},
            strategy_map={"left": [_tensor_stats(1, 4)], "right": [_tensor_stats(1, 4)]},
            label_to_block_id={"left": 0, "right": 0},
            use_shm=True,
            shm_block_name_map=published_names,
            block_name_fn=lambda index: f"test_block_{index}",
            host_pinner=NoOpHostPinner(),
        )

    assert published_names == {"existing": "existing_block"}

    _FakeAllocationManager.reset()
    retry = AllocationBlockController(
        allocation_ordered={0: ["left", "right"]},
        device_gpu=torch.device("cpu"),
        tensors_map={1: torch.ones(1)},
        strategy_map={"left": [_tensor_stats(1, 4)], "right": [_tensor_stats(1, 4)]},
        label_to_block_id={"left": 0, "right": 0},
        use_shm=True,
        shm_block_name_map=published_names,
        block_name_fn=lambda index: f"test_block_{index}",
        host_pinner=NoOpHostPinner(),
    )

    assert retry.shm_block_name_map == {
        "left": "test_block_0",
        "right": "test_block_0",
    }


def test_shm_names_cover_only_nonempty_load_labels(monkeypatch) -> None:
    _FakeAllocationManager.reset()
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)

    controller = AllocationBlockController(
        allocation_ordered={0: ["empty", "active"]},
        device_gpu=torch.device("cpu"),
        tensors_map={1: torch.ones(1)},
        strategy_map={"empty": [], "active": [_tensor_stats(1, 4)]},
        label_to_block_id={"empty": 0, "active": 0},
        use_shm=True,
        block_name_fn=lambda index: f"test_block_{index}",
        host_pinner=NoOpHostPinner(),
    )

    assert controller.shm_block_name_map == {"active": "test_block_0"}


def test_shm_follower_uses_published_name_when_first_group_label_is_empty(monkeypatch) -> None:
    _FakeAllocationManager.reset()
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)

    controller = AllocationBlockController(
        allocation_ordered={0: ["empty", "active"]},
        device_gpu=torch.device("cpu"),
        tensors_map={1: torch.ones(1)},
        strategy_map={"empty": [], "active": [_tensor_stats(1, 4)]},
        label_to_block_id={"empty": 0, "active": 0},
        use_shm=True,
        shm_block_name_map={"active": "published_block"},
        block_name_fn=lambda index: f"unexpected_block_{index}",
        load_from_shm=True,
        host_pinner=NoOpHostPinner(),
    )

    assert controller.shm_block_name_map == {"active": "published_block"}
    assert _FakeAllocationManager.instances[0].kwargs["shm_block_name_prefix"] == "published_block"


@pytest.mark.parametrize("placement", ["same_label", "other_block", "unmanaged"])
def test_constructor_rejects_overlapping_storage_before_releasing_any_tensor(
    monkeypatch: pytest.MonkeyPatch, placement: str
) -> None:
    _FakeAllocationManager.reset()
    _FakeAllocationManager.release_source_storage = True
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)
    backing = torch.arange(8.0)
    first = torch.nn.Parameter(backing[:4], requires_grad=False)
    second = torch.nn.Parameter(backing[2:6], requires_grad=False)
    tensors = {1: torch.arange(3.0), 2: first, 3: second}
    expected = {key: tensor.clone() for key, tensor in tensors.items()}
    storage_ids = {key: tensor.untyped_storage()._cdata for key, tensor in tensors.items()}
    allocation_ordered = {0: ["independent", "left"]}
    strategy = {"independent": [_tensor_stats(1, 12)], "left": [_tensor_stats(2, 16)]}
    label_to_block_id = {"independent": 0, "left": 0}
    if placement == "same_label":
        strategy["left"].append(_tensor_stats(3, 16))
    elif placement == "other_block":
        allocation_ordered[1] = ["right"]
        strategy["right"] = [_tensor_stats(3, 16)]
        label_to_block_id["right"] = 1

    with pytest.raises(ValueError, match="distinct tensor views sharing storage"):
        AllocationBlockController(
            allocation_ordered=allocation_ordered,
            device_gpu=torch.device("cpu"),
            tensors_map=tensors,
            strategy_map=strategy,
            label_to_block_id=label_to_block_id,
            host_pinner=NoOpHostPinner(),
        )

    assert not _FakeAllocationManager.instances
    assert first.storage_offset() == 0
    assert second.storage_offset() == 2
    for key, tensor in tensors.items():
        assert torch.equal(tensor, expected[key])
        assert tensor.untyped_storage()._cdata == storage_ids[key]


@pytest.mark.parametrize("keep_backing", [False, True])
def test_failed_repacking_restores_disjoint_views_into_one_storage_and_allows_retry(keep_backing: bool) -> None:
    backing = torch.arange(16.0)
    first = torch.nn.Parameter(backing[2:8].reshape(2, 3).t(), requires_grad=False)
    second = torch.nn.Parameter(backing[8:12].view(torch.uint8), requires_grad=False)
    tensors = {1: first, 2: second}
    expected = {key: tensor.clone() for key, tensor in tensors.items()}
    offsets = {key: tensor.storage_offset() for key, tensor in tensors.items()}
    strides = {key: tensor.stride() for key, tensor in tensors.items()}
    source_storage = StorageWeakRef(first.untyped_storage())
    if not keep_backing:
        del backing
    kwargs = {
        "allocation_ordered": {0: ["left"], 1: ["right"]},
        "device_gpu": torch.device("cpu"),
        "tensors_map": tensors,
        "strategy_map": {"left": [_tensor_stats(1, 24)], "right": [_tensor_stats(2, 16)]},
        "label_to_block_id": {"left": 0, "right": 1},
        "host_pinner": NoOpHostPinner(),
    }

    def fail_projection(*args, **kwargs):
        del args, kwargs
        assert first.numel() == second.numel() == 0
        assert source_storage.expired() is not keep_backing
        raise _ConstructionError("view projection failed")

    with (
        patch.object(AllocationBlock, "project_views", side_effect=fail_projection),
        pytest.raises(_ConstructionError, match="view projection failed"),
    ):
        AllocationBlockController(**kwargs)

    assert first.untyped_storage()._cdata == second.untyped_storage()._cdata
    if keep_backing:
        assert first.untyped_storage()._cdata == source_storage.cdata
    for key, tensor in tensors.items():
        torch.testing.assert_close(tensor, expected[key])
        assert tensor.storage_offset() == offsets[key]
        assert tensor.stride() == strides[key]

    retry = AllocationBlockController(**kwargs)
    try:
        retry.commit_construction()
        for label in ("left", "right"):
            retry.schedule_transfer(label, non_blocking=False)
        for key, view in retry.tensor_id_to_view_map.items():
            torch.testing.assert_close(view, expected[key])
    finally:
        retry.shutdown()


@pytest.mark.parametrize(
    "source_kind", ["private_slice", "external_parameter", "external_view", "external_expanded_view"]
)
def test_failed_construction_restores_offset_and_surviving_backing_storage(source_kind: str) -> None:
    backing = torch.arange(16.0)
    weight = backing[2:5].expand(4, 3) if source_kind == "external_expanded_view" else backing[2:14].view(3, 4).t()
    if source_kind in {"private_slice", "external_parameter"}:
        weight = torch.nn.Parameter(weight, requires_grad=False)
    external = torch.nn.Parameter(backing, requires_grad=False) if source_kind == "external_parameter" else None
    del backing
    source_storage = StorageWeakRef(weight.untyped_storage())
    expected = weight.clone(memory_format=torch.preserve_format)
    expected_stride = weight.stride()
    tensor_id = id(weight)

    def fail_gpu_creation(*args, **kwargs):
        del args, kwargs
        assert weight.numel() == 0
        # Rollback metadata must not retain a private source allocation.
        assert source_storage.expired() == (source_kind == "private_slice")
        raise _ConstructionError("gpu block creation failed")

    with (
        patch.object(AllocationManager, "create_max_block", side_effect=fail_gpu_creation),
        pytest.raises(_ConstructionError, match="gpu block creation failed"),
    ):
        AllocationBlockController(
            allocation_ordered={0: ["weight"]},
            device_gpu=torch.device("cpu"),
            # Repeated references to one tensor are supported; overlapping
            # distinct tensors are rejected by the separate preflight test.
            tensors_map={tensor_id: weight, -1: weight},
            strategy_map={"weight": [_tensor_stats(tensor_id, weight.numel() * weight.element_size())]},
            label_to_block_id={"weight": 0},
            host_pinner=NoOpHostPinner(),
        )

    assert id(weight) == tensor_id
    assert weight.storage_offset() == 2
    assert weight.stride() == expected_stride
    assert torch.equal(weight, expected)
    if source_kind != "private_slice":
        assert weight.untyped_storage()._cdata == source_storage.cdata
        with torch.no_grad():
            weight[0, 0] = 99
        alias = external if external is not None else weight._base
        assert alias[2].item() == 99


@pytest.mark.parametrize("tensor_kind", ["plain", "subclass", "named", "conjugate", "negative", "conjugate_negative"])
def test_failed_construction_restores_same_tensors_and_object_aliases_before_retry(
    monkeypatch: pytest.MonkeyPatch, tensor_kind: str
) -> None:
    _FakeAllocationManager.reset()
    _FakeAllocationManager.fail_allocate_number = 2
    _FakeAllocationManager.release_source_storage = True
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)

    first = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    if tensor_kind == "subclass":
        first = _RejectDataAssignmentParameter(first, requires_grad=False)
    elif tensor_kind == "named":
        if not callable(getattr(torch.Tensor, "refine_names", None)):
            pytest.skip("PyTorch named tensors are unavailable")
        first = first.refine_names("column", "row")
    elif tensor_kind in {"conjugate", "negative", "conjugate_negative"}:
        first = torch.complex(first, first + 1).conj()
        if tensor_kind == "negative":
            first = first.imag
        elif tensor_kind == "conjugate_negative":
            first = torch._neg_view(first)
    first_alias = first
    expected_stride = first.stride()
    expected_offset = first.storage_offset()
    expected_names = getattr(first, "names", None)
    expected_conj, expected_neg = first.is_conj(), first.is_neg()
    second = torch.tensor([21.0, 22.0])
    expected_first = _unnamed_tensor_data(first).clone(memory_format=torch.preserve_format)
    expected_second = second.clone()
    tensors = {1: first, 2: second}
    strategy = {"left": [_tensor_stats(1, first.numel() * first.element_size())], "right": [_tensor_stats(2, 8)]}

    with pytest.raises(_ConstructionError, match="second block allocation failed"):
        AllocationBlockController(
            allocation_ordered={0: ["left", "right"]},
            device_gpu=torch.device("cpu"),
            tensors_map=tensors,
            strategy_map=strategy,
            label_to_block_id={"left": 0, "right": 0},
            use_shm=True,
            block_name_fn=lambda index: f"test_block_{index}",
            host_pinner=NoOpHostPinner(),
        )

    assert first_alias is first
    assert first.shape == expected_first.shape
    assert first.stride() == expected_stride
    assert first.storage_offset() == expected_offset
    assert first.dtype == expected_first.dtype
    assert getattr(first, "names", None) == expected_names
    assert first.is_conj() == expected_conj
    assert first.is_neg() == expected_neg
    assert torch.equal(_unnamed_tensor_data(first), expected_first)
    assert torch.equal(second, expected_second)
    assert all(block.release_calls == 1 for block in _all_fake_blocks())

    _FakeAllocationManager.reset()
    _FakeAllocationManager.release_source_storage = True
    retry = AllocationBlockController(
        allocation_ordered={0: ["left", "right"]},
        device_gpu=torch.device("cpu"),
        tensors_map=tensors,
        strategy_map=strategy,
        label_to_block_id={"left": 0, "right": 0},
        use_shm=True,
        block_name_fn=lambda index: f"test_block_{index}",
        host_pinner=NoOpHostPinner(),
    )

    restored_views = retry.get_tensor_id_to_view_mapping()
    assert restored_views[1].stride() == expected_first.stride()
    assert torch.equal(restored_views[1], expected_first)
    assert torch.equal(restored_views[2], expected_second)


@pytest.mark.skipif(
    not callable(getattr(torch.Tensor, "refine_names", None)), reason="PyTorch named tensors are unavailable"
)
def test_failed_name_restoration_retains_block_until_cleanup_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAllocationManager.reset()
    _FakeAllocationManager.fail_allocate_number = 2
    _FakeAllocationManager.release_source_storage = True
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)
    tensor = torch.arange(6.0).reshape(2, 3).refine_names("row", "column")
    expected = tensor.rename(None).clone()

    def fail_rename(_tensor: torch.Tensor, *_names: str | None) -> None:
        raise RuntimeError("name restoration failed")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "rename_", fail_rename)
        with pytest.raises(_ConstructionError, match="second block allocation failed") as raised:
            AllocationBlockController(
                allocation_ordered={0: ["left", "right"]},
                device_gpu=torch.device("cpu"),
                tensors_map={1: tensor, 2: torch.ones(1)},
                strategy_map={"left": [_tensor_stats(1, 24)], "right": [_tensor_stats(2, 4)]},
                label_to_block_id={"left": 0, "right": 0},
                host_pinner=NoOpHostPinner(),
            )

    cleanup = raised.value._flextensor_cleanup_controller
    assert _all_fake_blocks()[0].release_calls == 0
    assert tensor.names == (None, None)

    cleanup.shutdown()

    assert tensor.names == ("row", "column")
    assert torch.equal(tensor.rename(None), expected)
    assert cleanup._owned_blocks == []
    assert all(block.release_calls == 1 for block in _all_fake_blocks())


def test_constructor_preserves_primary_error_and_retains_staged_cleanup_for_retry(monkeypatch) -> None:
    _FakeAllocationManager.reset()
    _FakeAllocationManager.fail_allocate_number = 2
    _FakeAllocationManager.fail_release_number = 1
    _FakeAllocationManager.release_failures = 2
    controller = AllocationBlockController.__new__(AllocationBlockController)
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)

    with pytest.raises(_ConstructionError, match="second block allocation failed") as raised:
        AllocationBlockController.__init__(
            controller,
            allocation_ordered={0: ["left", "right"]},
            device_gpu=torch.device("cpu"),
            tensors_map={1: torch.ones(1)},
            strategy_map={"left": [_tensor_stats(1, 4)], "right": [_tensor_stats(1, 4)]},
            label_to_block_id={"left": 0, "right": 0},
            use_shm=True,
            block_name_fn=lambda index: f"test_block_{index}",
            host_pinner=NoOpHostPinner(),
        )

    first, second = _all_fake_blocks()
    assert first.release_calls == 2
    assert second.release_calls == 1
    cleanup_controller = raised.value._flextensor_cleanup_controller
    assert cleanup_controller is controller

    cleanup_controller.shutdown()

    assert first.release_calls == 3
    assert second.release_calls == 1
    assert controller._owned_blocks == []


def test_constructor_attaches_handles_when_tensor_restoration_cannot_finish(monkeypatch) -> None:
    _FakeAllocationManager.reset()
    _FakeAllocationManager.fail_allocate_number = 2
    _FakeAllocationManager.release_source_storage = True
    controller = AllocationBlockController.__new__(AllocationBlockController)
    monkeypatch.setattr("flextensor.loaders.AllocationManager", _FakeAllocationManager)
    monkeypatch.setattr(
        ConstructionRollback,
        "_restore_released_tensors",
        lambda _self, _records: (_ for _ in ()).throw(RuntimeError("restore failed")),
    )

    with pytest.raises(_ConstructionError, match="second block allocation failed") as raised:
        AllocationBlockController.__init__(
            controller,
            allocation_ordered={0: ["left", "right"]},
            device_gpu=torch.device("cpu"),
            tensors_map={1: torch.ones(1)},
            strategy_map={"left": [_tensor_stats(1, 4)], "right": [_tensor_stats(1, 4)]},
            label_to_block_id={"left": 0, "right": 0},
            use_shm=True,
            block_name_fn=lambda index: f"test_block_{index}",
            host_pinner=NoOpHostPinner(),
        )

    cleanup_controller = raised.value._flextensor_cleanup_controller
    assert cleanup_controller is controller
    assert cleanup_controller._owned_blocks
    assert _all_fake_blocks()[0].release_calls == 0


class _WrapperConstructionError(RuntimeError):
    pass


@pytest.mark.parametrize("rearrange_transfers", [False, True])
def test_wrapper_construction_failure_aborts_controller_and_same_tensors_retry(
    rearrange_transfers: bool,
) -> None:
    manager = TensorManager.__new__(TensorManager)
    manager.rearrange_transfers = rearrange_transfers
    manager.shm_namespace = None
    manager.use_shm = False
    manager.load_from_shm = False
    manager.shm_owner = None
    manager.device_gpu = torch.device("cpu")
    manager.host_pinner = NoOpHostPinner()
    manager.stats = []
    manager.transfer_stream_priority = 0
    manager.loader_type = "allocation_block_transfer"
    manager.enable_offload_timing = False
    manager.enable_piecewise_prefetch_check = False

    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    weight_alias = weight
    expected = weight.clone(memory_format=torch.preserve_format)
    tensor_id = id(weight)
    manager.tensors_map = {tensor_id: weight}
    manager.load_strategy = {"layer": [_tensor_stats(tensor_id, weight.numel() * weight.element_size())]}
    data = LoaderInputData(
        allocation_ordered={0: ["layer"]},
        label_to_block_id={"layer": 0},
        transfer_to_compute_map={"layer": "layer"},
    )

    controllers: list[AllocationBlockController] = []
    blocks_at_handoff = []
    primary_error = _WrapperConstructionError("wrapper construction failed")
    loader_target = (
        "flextensor.tensor_manager.PreallocatedBatchTransferTensorLoaderReordered"
        if rearrange_transfers
        else "flextensor.tensor_manager.PreallocatedBatchTransferTensorLoader"
    )

    def construct_controller(*args, **kwargs):
        controller = AllocationBlockController(*args, **kwargs)
        controllers.append(controller)
        return controller

    def fail_wrapper(*args, allocation_controller, **kwargs):
        del args, kwargs
        blocks_at_handoff.extend(allocation_controller._owned_blocks)
        assert weight.numel() == 0
        raise primary_error

    with (
        patch("flextensor.tensor_manager.AllocationBlockController", side_effect=construct_controller),
        patch(loader_target, side_effect=fail_wrapper),
        pytest.raises(_WrapperConstructionError) as raised,
    ):
        manager._setup_allocation_block_loader(data, prepare_state=False)

    assert raised.value is primary_error
    assert weight_alias is weight
    assert weight.shape == expected.shape
    assert weight.stride() == expected.stride()
    assert weight.dtype == expected.dtype
    assert torch.equal(weight, expected)
    assert controllers[0]._owned_blocks == []
    assert controllers[0].block_map_cpu == {}
    assert all(block.block is None and block.views == [] for block in blocks_at_handoff)
    assert not hasattr(manager, "tensor_layer_loader")

    wrapper = SimpleNamespace()

    def finish_wrapper(*args, allocation_controller, **kwargs):
        del args, kwargs
        wrapper.allocation_controller = allocation_controller
        return wrapper

    with (
        patch("flextensor.tensor_manager.AllocationBlockController", side_effect=construct_controller),
        patch(loader_target, side_effect=finish_wrapper),
    ):
        manager._setup_allocation_block_loader(data, prepare_state=False)

    retry_controller = controllers[1]
    assert manager.tensor_layer_loader is wrapper
    retry_controller.schedule_transfer("layer", non_blocking=False)
    assert retry_controller.get_tensor_id_to_view_mapping()[tensor_id].stride() == expected.stride()
    assert torch.equal(retry_controller.get_tensor_id_to_view_mapping()[tensor_id], expected)
    retry_controller.shutdown()
    assert weight.numel() == 0

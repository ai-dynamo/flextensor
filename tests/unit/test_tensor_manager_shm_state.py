# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for creator/follower TensorManager state restoration."""

from __future__ import annotations

import copy
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor.allocation_block import AllocationBlock
from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.helpers import NoOpTensorManager
from flextensor.host_pinning import HostPinner, NoOpHostPinner
from flextensor.loaders import AllocationBlockController
from flextensor.offload_manager import TensorManagerProtocol
from flextensor.shm.flexible_shm import FlexibleSharedMemory
from flextensor.state_handler import LoaderInputData, TensorManagerState, TensorManagerStateHandler
from flextensor.tensor_manager import TensorManager


class _RestoreModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.ones(2))
        self.right = nn.Parameter(torch.ones(3))


def _stat(tensor_id: int, name: str, *, size_bytes: int = 8) -> TensorStatistics:
    return TensorStatistics(tensor_id=tensor_id, name=name, size_bytes=size_bytes, load_time_ms=0.1)


def _state(
    *,
    loader_type: str = "allocation_block_transfer",
    shm_block_name_map: dict[str, str] | None = None,
    managed_tensor_dtypes: dict[str, str] | None = None,
) -> TensorManagerState:
    left = _stat(101, "left")
    right = _stat(102, "right", size_bytes=12)
    return TensorManagerState(
        loader_type=loader_type,
        tensor_id_to_name_map={101: "left", 102: "right"},
        allocation_ordered={0: ["left"], 1: ["right"]},
        label_to_size_map={},
        block_sizes={0: 8, 1: 12},
        load_strategy={"left": [left], "right": [right]},
        release_strategy={},
        label_to_block_id={"left": 0, "right": 1},
        stats=[
            LayerStatistics(label="left", tensors=[left], duration=1.0),
            LayerStatistics(label="right", tensors=[right], duration=1.0),
        ],
        transfer_to_compute_map={"left": "left", "right": "right"},
        view_tensors_ids=[101, 102],
        view_tensors_names=["left", "right"],
        gpu_tensors_names=[],
        shm_block_name_map=shm_block_name_map,
        managed_tensor_dtypes=(
            {"left": "float32", "right": "float32"} if managed_tensor_dtypes is None else managed_tensor_dtypes
        ),
        managed_tensor_layouts={
            "left": {"shape": [2], "stride": [1]},
            "right": {"shape": [3], "stride": [1]},
        },
    )


def _manager(*, loader_type: str = "allocation_block_transfer", use_shm: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        loader_type=loader_type,
        use_shm=use_shm,
        use_trace_tensor=True,
        set_model=MagicMock(),
        tensor_manager_state=None,
    )


@pytest.mark.parametrize(
    "saved_map",
    [
        None,
        {},
        {"left": "stale_left"},
        {"left": "", "right": "stale_right"},
        {"left": "stale_left", "right": "stale_right"},
    ],
)
def test_creator_restore_clears_saved_block_names_without_mutating_caller(
    saved_map: dict[str, str] | None,
) -> None:
    state = _state(shm_block_name_map=saved_map)
    caller_snapshot = copy.deepcopy(state)
    manager = _manager()

    TensorManagerStateHandler(manager).restore_state(_RestoreModel(), state, load_from_shm=False)

    assert state == caller_snapshot
    assert manager.tensor_manager_state is not state
    assert manager.tensor_manager_state.shm_block_name_map is None
    assert manager.load_from_shm is False


@pytest.mark.parametrize(
    "saved_map",
    [
        None,
        {},
        {"left": "block_0"},
    ],
)
def test_follower_restore_rejects_missing_or_incomplete_block_names(
    saved_map: dict[str, str] | None,
) -> None:
    manager = _manager()

    with pytest.raises(ValueError, match="shm_block_name_map"):
        TensorManagerStateHandler(manager).restore_state(
            _RestoreModel(),
            _state(shm_block_name_map=saved_map),
            load_from_shm=True,
        )

    assert manager.tensor_manager_state is None


def test_follower_restore_accepts_complete_block_names_without_aliasing_caller() -> None:
    state = _state(shm_block_name_map={"left": "block_0", "right": "block_1"})
    manager = _manager()

    TensorManagerStateHandler(manager).restore_state(_RestoreModel(), state, load_from_shm=True)

    assert manager.tensor_manager_state.shm_block_name_map == state.shm_block_name_map
    assert manager.tensor_manager_state.shm_block_name_map is not state.shm_block_name_map
    assert manager.load_from_shm is True


@pytest.mark.parametrize("load_from_shm", [False, True])
@pytest.mark.parametrize("right_kind", ["managed", "resident_parameter", "resident_buffer"])
def test_shared_restore_rejects_storage_aliases_before_mutation(load_from_shm: bool, right_kind: str) -> None:
    storage = torch.arange(4, dtype=torch.float32)
    model = nn.Module()
    model.left = nn.Parameter(storage[:2])
    if right_kind == "resident_buffer":
        model.register_buffer("right", storage[1:])
    else:
        model.right = nn.Parameter(storage[1:])
    state = _state(shm_block_name_map={"left": "block_0", "right": "block_1"})
    if right_kind != "managed":
        state.allocation_ordered.pop(1)
        state.block_sizes.pop(1)
        state.load_strategy.pop("right")
        state.label_to_block_id.pop("right")
        state.transfer_to_compute_map.pop("right")
        state.stats = state.stats[:1]
        state.view_tensors_ids = [101]
        state.view_tensors_names = ["left"]
        state.gpu_tensors_names = ["right"]
        state.shm_block_name_map.pop("right")
        state.managed_tensor_dtypes.pop("right")
        state.managed_tensor_layouts.pop("right")
    manager = _manager()
    original_state = copy.deepcopy(state)
    expected_error = "sharing storage" if right_kind == "managed" else "Contradictory alias destinations"

    with pytest.raises(ValueError, match=expected_error):
        TensorManagerStateHandler(manager).restore_state(model, state, load_from_shm=load_from_shm)

    assert state == original_state
    assert manager.tensor_manager_state is None
    manager.set_model.assert_not_called()
    assert model.left.requires_grad
    assert model.left.untyped_storage()._cdata == model.right.untyped_storage()._cdata
    torch.testing.assert_close(storage, torch.arange(4, dtype=torch.float32))


@pytest.mark.parametrize("empty_load_label", [False, True])
def test_resident_only_follower_accepts_empty_managed_metadata(empty_load_label: bool) -> None:
    model = _RestoreModel()
    state = TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={101: "left", 102: "right"},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={"left": []} if empty_load_label else {},
        release_strategy={},
        label_to_block_id={},
        stats=[],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=["left", "right"],
        shm_block_name_map={},
    )
    caller_snapshot = copy.deepcopy(state)
    manager = _manager()

    result = TensorManagerStateHandler(manager).restore_state(model, state, load_from_shm=True)

    assert not result
    assert state == caller_snapshot
    assert manager.tensor_manager_state is not state
    assert manager.tensor_manager_state.gpu_tensors_names == ["left", "right"]
    assert manager.tensor_manager_state.managed_tensor_dtypes == {}
    assert manager.tensor_manager_state.managed_tensor_layouts == {}
    assert manager.tensor_manager_state.to_loader_input_data().shm_block_name_map == {}
    assert manager.tensors_map == {}
    assert manager.traced_tensors == set()
    assert manager.load_from_shm is True
    manager.set_model.assert_called_once_with(model)


def test_follower_restore_requires_published_managed_dtypes() -> None:
    state = _state(
        shm_block_name_map={"left": "block_0", "right": "block_1"},
        managed_tensor_dtypes={},
    )

    with pytest.raises(ValueError, match=r"dtype metadata"):
        TensorManagerStateHandler(_manager()).restore_state(
            _RestoreModel(),
            state,
            load_from_shm=True,
        )


def test_follower_restore_requires_published_managed_layouts() -> None:
    state = _state(shm_block_name_map={"left": "block_0", "right": "block_1"})
    state.managed_tensor_layouts = {}

    with pytest.raises(ValueError, match=r"layout metadata"):
        TensorManagerStateHandler(_manager()).restore_state(_RestoreModel(), state, load_from_shm=True)


@pytest.mark.parametrize("follower_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "source",
    [
        pytest.param(torch.arange(6, dtype=torch.float32).reshape(3, 2).t(), id="transposed"),
        pytest.param(torch.arange(12, dtype=torch.float32).reshape(2, 6)[:, ::2], id="sliced"),
        pytest.param(torch.arange(3, dtype=torch.float32).expand(2, 3), id="expanded"),
    ],
)
def test_follower_reads_published_managed_layout_from_shm(source: torch.Tensor, follower_dtype: torch.dtype) -> None:
    source = source.detach()
    expected = source.clone()
    tensor_id = id(source)
    block_name = "layout_" + uuid.uuid4().hex[:8]
    creator = AllocationBlock(shm_block_name=block_name, host_pinner=NoOpHostPinner(), release_tensor_memory=True)
    follower = AllocationBlock(shm_block_name=block_name, host_pinner=NoOpHostPinner(), load_from_shm=True)
    try:
        creator.add(source)
        creator_view = creator.allocate()[0]
        stat = _stat(tensor_id, "weight", size_bytes=expected.numel() * expected.element_size())
        manager = SimpleNamespace(
            loader_type="allocation_block_transfer",
            tensor_id_to_name_map={tensor_id: "weight"},
            tensor_layer_loader=SimpleNamespace(
                allocation_controller=SimpleNamespace(get_tensor_id_to_view_mapping=lambda: {tensor_id: creator_view})
            ),
        )
        published = TensorManagerStateHandler(manager).prepare_state(
            loader_type="allocation_block_transfer",
            allocation_ordered={0: ["linear"]},
            label_to_size_map={},
            block_sizes={0: stat.size_bytes},
            load_strategy={"linear": [stat]},
            release_strategy={},
            label_to_block_id={"linear": 0},
            stats=[LayerStatistics(label="linear", tensors=[stat], duration=1.0)],
            transfer_to_compute_map={"linear": "linear"},
            shm_block_name_map={"linear": block_name},
        )
        restored = TensorManagerStateHandler.load_state_from_bytes(
            TensorManagerStateHandler.save_state_to_bytes(published)
        )
        model = nn.Linear(3, 2, bias=False, device="meta", dtype=follower_dtype)
        original_weight = model.weight
        TensorManagerStateHandler(_manager()).restore_state(model, restored, load_from_shm=True)

        follower.add(model.weight)
        follower_view = follower.allocate()[0]

        assert model.weight is original_weight
        assert follower_view.dtype == creator_view.dtype
        assert follower_view.stride() == creator_view.stride()
        torch.testing.assert_close(follower_view, expected)
        inputs = torch.arange(3, dtype=torch.float32)
        torch.testing.assert_close(nn.functional.linear(inputs, follower_view), nn.functional.linear(inputs, expected))
    finally:
        follower.release()
        creator.release()


def test_follower_shape_preflight_does_not_mutate_meta_model() -> None:
    model = _RestoreModel()
    model.left = nn.Parameter(torch.empty_strided((2,), (2,), dtype=torch.bfloat16, device="meta"))
    model.right = nn.Parameter(torch.empty((1, 3), dtype=torch.bfloat16, device="meta"))
    state = _state(shm_block_name_map={"left": "block_0", "right": "block_1"})

    # The right parameter has the expected byte count, but a different shape.
    with pytest.raises(ValueError, match=r"shape drift for right"):
        TensorManagerStateHandler(_manager()).restore_state(model, state, load_from_shm=True)

    assert model.left.dtype is torch.bfloat16
    assert model.left.stride() == (2,)


def test_concrete_follower_rejects_incompatible_packed_stride() -> None:
    model = _RestoreModel()
    model.left = nn.Parameter(torch.ones(2, 3))
    state = _state(shm_block_name_map={"left": "block_0", "right": "block_1"})
    left_stat = _stat(101, "left", size_bytes=24)
    state.load_strategy["left"] = [left_stat]
    state.stats[0] = state.stats[0].model_copy(update={"tensors": [left_stat]})
    state.block_sizes[0] = 24
    state.managed_tensor_layouts["left"] = {"shape": [2, 3], "stride": [1, 2]}

    with pytest.raises(ValueError, match=r"stride drift for left"):
        TensorManagerStateHandler(_manager()).restore_state(model, state, load_from_shm=True)

    torch.testing.assert_close(model.left, torch.ones(2, 3))


def test_follower_retypes_meta_placeholders_before_size_validation() -> None:
    class _MetaRestoreModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Parameter(torch.empty(2, dtype=torch.bfloat16, device="meta"), requires_grad=False)
            self.right = nn.Parameter(torch.empty(3, dtype=torch.bfloat16, device="meta"), requires_grad=False)

    model = _MetaRestoreModel()
    original_left_id = id(model.left)
    state = _state(
        shm_block_name_map={"left": "block_0", "right": "block_1"},
        managed_tensor_dtypes={"left": "float32", "right": "bfloat16"},
    )
    right_stat = state.load_strategy["right"][0].model_copy(update={"size_bytes": 6})
    state.load_strategy["right"] = [right_stat]
    state.stats[1] = state.stats[1].model_copy(update={"tensors": [right_stat]})
    state.block_sizes[1] = 6

    TensorManagerStateHandler(_manager()).restore_state(model, state, load_from_shm=True)

    assert id(model.left) == original_left_id
    assert model.left.dtype is torch.float32
    assert model.left.requires_grad is False
    assert model.right.dtype is torch.bfloat16


def test_follower_dtype_preflight_does_not_mutate_meta_model_on_size_drift() -> None:
    class _MetaRestoreModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Parameter(torch.empty(2, dtype=torch.bfloat16, device="meta"), requires_grad=False)
            self.right = nn.Parameter(torch.empty(3, dtype=torch.bfloat16, device="meta"), requires_grad=False)

    model = _MetaRestoreModel()
    state = _state(
        shm_block_name_map={"left": "block_0", "right": "block_1"},
        managed_tensor_dtypes={"left": "float32", "right": "bfloat16"},
    )
    invalid_right_stat = state.load_strategy["right"][0].model_copy(update={"size_bytes": 7})
    state.load_strategy["right"] = [invalid_right_stat]
    state.stats[1] = state.stats[1].model_copy(update={"tensors": [invalid_right_stat]})
    state.block_sizes[1] = 7

    with pytest.raises(ValueError, match=r"size_bytes drift for right"):
        TensorManagerStateHandler(_manager()).restore_state(model, state, load_from_shm=True)

    assert model.left.dtype is torch.bfloat16
    assert model.right.dtype is torch.bfloat16


def test_follower_rejects_adoption_mode_before_retyping_meta_model() -> None:
    model = nn.Module()
    model.register_parameter(
        "left",
        nn.Parameter(torch.empty(2, dtype=torch.bfloat16, device="meta"), requires_grad=False),
    )
    model.register_parameter(
        "right",
        nn.Parameter(torch.empty(3, dtype=torch.bfloat16, device="meta"), requires_grad=False),
    )
    state = _state(
        shm_block_name_map={"left": "block_0", "right": "block_1"},
        managed_tensor_dtypes={"left": "float32", "right": "bfloat16"},
    )

    with pytest.raises(ValueError, match=r"incompatible with preprocess_model_state=False"):
        TensorManagerStateHandler(_manager()).restore_state(
            model,
            state,
            load_from_shm=True,
            preprocess_model_state=False,
        )

    assert model.left.dtype is torch.bfloat16


def test_concrete_managed_dtype_drift_is_rejected_even_when_itemsize_matches() -> None:
    state = _state(managed_tensor_dtypes={"left": "int32", "right": "float32"})

    with pytest.raises(ValueError, match=r"dtype drift for left"):
        TensorManagerStateHandler(_manager()).restore_state(_RestoreModel(), state, load_from_shm=False)


def test_follower_restore_rejects_different_prefixes_within_one_allocation_group() -> None:
    state = _state(shm_block_name_map={"left": "block_0", "right": "block_1"})
    state.allocation_ordered = {0: ["left", "right"]}
    state.label_to_block_id = {"left": 0, "right": 0}
    state.block_sizes = {0: 12}
    manager = _manager()

    with pytest.raises(ValueError, match=r"shm_block_name_map.*one shared prefix"):
        TensorManagerStateHandler(manager).restore_state(
            _RestoreModel(),
            state,
            load_from_shm=True,
        )


@pytest.mark.parametrize("invalid_name", ["", "   "])
def test_follower_restore_rejects_blank_block_names(invalid_name: str) -> None:
    manager = _manager()

    with pytest.raises(ValueError, match="shm_block_name_map"):
        TensorManagerStateHandler(manager).restore_state(
            _RestoreModel(),
            _state(shm_block_name_map={"left": invalid_name, "right": "block_1"}),
            load_from_shm=True,
        )


def test_follower_restore_requires_shm_enabled_manager() -> None:
    manager = _manager(use_shm=False)

    with pytest.raises(ValueError, match="use_shm"):
        TensorManagerStateHandler(manager).restore_state(
            _RestoreModel(),
            _state(shm_block_name_map={"left": "block_0", "right": "block_1"}),
            load_from_shm=True,
        )


@pytest.mark.parametrize("loader_type", ["strategy", "raw_block_transfer"])
@pytest.mark.parametrize("load_from_shm", [False, True])
def test_shm_restore_rejects_non_allocation_loaders(loader_type: str, load_from_shm: bool) -> None:
    manager = _manager(loader_type=loader_type)

    with pytest.raises(ValueError, match="allocation_block_transfer"):
        TensorManagerStateHandler(manager).restore_state(
            _RestoreModel(),
            _state(
                loader_type=loader_type,
                shm_block_name_map={"left": "block_0", "right": "block_1"},
            ),
            load_from_shm=load_from_shm,
        )


def _export_manager() -> tuple[TensorManager, AllocationBlockController, TensorManagerState]:
    manager = TensorManager.__new__(TensorManager)
    restored_state = _state(shm_block_name_map=None)
    # Simulate the existing on-disk v3 profile used by the creator. It has no
    # tensor metadata; export must enrich the live state from controller views.
    restored_state.managed_tensor_dtypes = {}
    restored_state.managed_tensor_layouts = {}

    offloaded = torch.ones(2)
    viewed = torch.ones(3, dtype=torch.bfloat16)
    resident_b = torch.ones(4)
    resident_a = torch.ones(5)
    manager.loader_type = "allocation_block_transfer"
    manager.tensor_manager_state = restored_state
    manager.tensor_id_to_name_map = {
        id(offloaded): "left.current",
        id(viewed): "right.current",
        id(resident_b): "b.resident",
        id(resident_a): "a.resident",
    }
    live_left = _stat(id(offloaded), "stale.left")
    live_right = _stat(id(viewed), "stale.right", size_bytes=6)
    manager.load_strategy = {"left": [live_left], "right": [live_right]}
    manager.stats = [
        LayerStatistics(label="left", tensors=[live_left], duration=2.0),
        LayerStatistics(label="right", tensors=[live_right], duration=3.0),
    ]

    controller = AllocationBlockController.__new__(AllocationBlockController)
    controller.shm_block_name_map = {"left": "live_block_0", "right": "live_block_1"}
    controller.tensor_id_to_view_map = {
        id(offloaded): torch.empty_strided((2,), (1,)),
        id(viewed): torch.empty_strided((3,), (1,), dtype=torch.bfloat16),
    }
    restored_state.block_sizes[1] = 6
    manager.tensor_layer_loader = SimpleNamespace(allocation_controller=controller)
    return manager, controller, restored_state


def test_export_state_rebuilds_live_ids_names_views_and_block_names() -> None:
    manager, controller, restored_state = _export_manager()

    exported = manager.export_state()

    assert exported is not restored_state
    assert exported.tensor_id_to_name_map == manager.tensor_id_to_name_map
    assert exported.load_strategy["left"][0].tensor_id == next(iter(manager.load_strategy["left"])).tensor_id
    assert exported.load_strategy["left"][0].name == "left.current"
    assert exported.stats[0].tensors[0].name == "left.current"
    assert exported.view_tensors_ids == list(controller.tensor_id_to_view_map)
    assert exported.view_tensors_names == ["left.current", "right.current"]
    assert exported.shm_block_name_map == controller.shm_block_name_map
    assert exported.gpu_tensors_names == ["a.resident", "b.resident"]
    assert exported.managed_tensor_dtypes == {
        "left.current": "float32",
        "right.current": "bfloat16",
    }
    assert exported.managed_tensor_layouts == {
        "left.current": {"shape": [2], "stride": [1]},
        "right.current": {"shape": [3], "stride": [1]},
    }


def test_export_state_is_deeply_isolated_from_live_manager_and_restored_state() -> None:
    manager, controller, restored_state = _export_manager()
    exported = manager.export_state()

    exported.allocation_ordered[0].append("mutated")
    exported.tensor_id_to_name_map[-1] = "mutated"
    assert exported.shm_block_name_map is not None
    exported.shm_block_name_map["left"] = "mutated"
    exported.load_strategy["left"][0] = _stat(-1, "mutated")

    assert restored_state.allocation_ordered[0] == ["left"]
    assert -1 not in manager.tensor_id_to_name_map
    assert controller.shm_block_name_map["left"] == "live_block_0"
    assert manager.load_strategy["left"][0].name == "stale.left"


def test_export_state_requires_a_live_state() -> None:
    manager = TensorManager.__new__(TensorManager)
    manager.tensor_manager_state = None
    manager.loader_type = "allocation_block_transfer"

    with pytest.raises(RuntimeError, match="state"):
        manager.export_state()


def test_export_state_requires_allocation_block_loader() -> None:
    manager = TensorManager.__new__(TensorManager)
    manager.tensor_manager_state = _state(loader_type="strategy")
    manager.loader_type = "strategy"
    manager.tensor_layer_loader = SimpleNamespace(allocation_controller=object())

    with pytest.raises(RuntimeError, match="allocation_block_transfer"):
        manager.export_state()


@pytest.mark.parametrize(
    "loader",
    [None, SimpleNamespace(allocation_controller=object())],
)
def test_export_state_requires_a_live_allocation_controller(loader: object | None) -> None:
    manager = TensorManager.__new__(TensorManager)
    manager.tensor_manager_state = _state()
    manager.loader_type = "allocation_block_transfer"
    manager.tensor_layer_loader = loader

    with pytest.raises(RuntimeError, match="AllocationBlockController"):
        manager.export_state()


def test_noop_export_state_has_protocol_parity_and_fails_clearly() -> None:
    assert "export_state" in TensorManagerProtocol.__dict__

    with pytest.raises(RuntimeError, match="export"):
        NoOpTensorManager(device_gpu="cpu").export_state()


def test_restore_state_preserves_live_buffer_ids_for_export_after_model_clear() -> None:
    class _ModelWithBuffer(_RestoreModel):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("resident_cache", torch.ones(4))

    model = _ModelWithBuffer()
    manager = TensorManager.__new__(TensorManager)
    manager.tensors_map = {}
    manager.tensor_id_to_name_map = {}

    with patch.object(
        TensorManagerStateHandler,
        "restore_state",
        side_effect=lambda *_args, **_kwargs: manager.set_model(model),
    ):
        manager.restore_state(model, _state())

    buffer = model.get_buffer("resident_cache")
    assert manager.tensor_id_to_name_map[id(buffer)] == "resident_cache"

    # The restored inference path clears ``TensorManager.model`` before the
    # creator publishes its live state; the retained map remains complete.
    manager.model = None
    assert "resident_cache" in manager.tensor_id_to_name_map.values()


def test_meta_replacement_refreshes_old_id_value_for_block_teardown_restore() -> None:
    class _MetaWeight(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(3, device="meta"))

    model = _MetaWeight()
    source = model.weight
    source_id = id(source)
    cpu_source_view = torch.tensor([1.0, 2.0, 3.0])
    projected_view = torch.zeros(3)

    manager = TensorManager.__new__(TensorManager)
    manager.device_gpu = torch.device("cpu")
    manager.tensors_map = {source_id: source}
    manager.tensor_id_to_name_map = {source_id: "weight"}

    manager._prepare_view_model_from_id_to_view_map(model, {source_id: projected_view})

    assert source_id in manager.tensors_map
    assert manager.tensors_map[source_id] is model.weight
    assert manager.tensors_map[source_id] is not source
    assert model.weight.untyped_storage().data_ptr() == projected_view.untyped_storage().data_ptr()

    controller = AllocationBlockController.__new__(AllocationBlockController)
    controller.block_map_cpu = {
        "layer": SimpleNamespace(views=[cpu_source_view], shm_block=object()),
    }
    controller.label_to_cpu_tensor_id_map = {"layer": [source_id]}
    controller.tensor_id_to_view_map = {source_id: projected_view}
    manager.tensor_layer_loader = SimpleNamespace(allocation_controller=controller)

    manager._restore_block_loader_tensor_data()

    assert torch.equal(model.weight, cpu_source_view)
    assert model.weight.untyped_storage().data_ptr() != cpu_source_view.untyped_storage().data_ptr()


@pytest.mark.parametrize("load_from_shm", [False, True])
def test_tensor_manager_forwards_explicit_restore_role_to_controller(load_from_shm: bool) -> None:
    manager = TensorManager.__new__(TensorManager)
    manager.rearrange_transfers = False
    manager.shm_owner = MagicMock(spec=FlexibleSharedMemory)
    manager.shm_namespace = "ft_stable"
    manager.use_shm = True
    manager.load_from_shm = load_from_shm
    manager.device_gpu = torch.device("cpu")
    manager.tensors_map = {}
    manager.load_strategy = {}
    manager.host_pinner = HostPinner()
    manager.stats = []
    manager.transfer_stream_priority = 1
    manager.enable_offload_timing = False
    manager.enable_piecewise_prefetch_check = False

    with (
        patch("flextensor.tensor_manager.AllocationBlockController") as controller_type,
        patch("flextensor.tensor_manager.PreallocatedBatchTransferTensorLoader"),
    ):
        manager._setup_allocation_block_loader(LoaderInputData(), prepare_state=False)

    assert controller_type.call_args.kwargs["load_from_shm"] is load_from_shm
    assert controller_type.call_args.kwargs["shm_owner"] is manager.shm_owner
    assert manager.load_from_shm is False


@pytest.mark.parametrize(
    ("use_shm", "expected_map"),
    [(False, None), (True, {"layer": "live_block"})],
)
def test_prepared_profile_persists_block_names_only_for_shm(
    use_shm: bool,
    expected_map: dict[str, str] | None,
) -> None:
    manager = TensorManager.__new__(TensorManager)
    manager.rearrange_transfers = False
    manager.shm_owner = None
    manager.shm_namespace = "ft_stable" if use_shm else None
    manager.use_shm = use_shm
    manager.load_from_shm = False
    manager.device_gpu = torch.device("cpu")
    manager.tensors_map = {}
    manager.load_strategy = {}
    manager.host_pinner = HostPinner()
    manager.stats = []
    manager.transfer_stream_priority = 1
    manager.enable_offload_timing = False
    manager.enable_piecewise_prefetch_check = False
    manager.loader_type = "allocation_block_transfer"
    manager.prepare_tensor_manager_state = MagicMock(return_value=object())
    controller = MagicMock(shm_block_name_map={"layer": "live_block"})

    with (
        patch("flextensor.tensor_manager.AllocationBlockController", return_value=controller),
        patch("flextensor.tensor_manager.PreallocatedBatchTransferTensorLoader"),
    ):
        manager._setup_allocation_block_loader(LoaderInputData(), prepare_state=True)

    assert manager.prepare_tensor_manager_state.call_args.kwargs["shm_block_name_map"] == expected_map


def test_successful_follower_build_consumes_role_before_a_later_fresh_build() -> None:
    manager = TensorManager.__new__(TensorManager)
    manager.rearrange_transfers = False
    manager.shm_owner = None
    manager.shm_namespace = "ft_stable"
    manager.use_shm = True
    manager.load_from_shm = True
    manager.device_gpu = torch.device("cpu")
    manager.tensors_map = {}
    manager.load_strategy = {}
    manager.host_pinner = HostPinner()
    manager.stats = []
    manager.transfer_stream_priority = 1
    manager.enable_offload_timing = False
    manager.enable_piecewise_prefetch_check = False

    with (
        patch("flextensor.tensor_manager.AllocationBlockController") as controller_type,
        patch("flextensor.tensor_manager.PreallocatedBatchTransferTensorLoader"),
    ):
        manager._setup_allocation_block_loader(LoaderInputData(), prepare_state=False)
        manager._setup_allocation_block_loader(LoaderInputData(), prepare_state=False)

    assert [call.kwargs["load_from_shm"] for call in controller_type.call_args_list] == [True, False]


def test_failed_follower_build_retains_role_for_retry() -> None:
    manager = TensorManager.__new__(TensorManager)
    manager.rearrange_transfers = False
    manager.shm_owner = None
    manager.shm_namespace = "ft_stable"
    manager.use_shm = True
    manager.load_from_shm = True
    manager.device_gpu = torch.device("cpu")
    manager.tensors_map = {}
    manager.load_strategy = {}
    manager.host_pinner = HostPinner()
    manager.stats = []
    manager.transfer_stream_priority = 1
    manager.enable_offload_timing = False
    manager.enable_piecewise_prefetch_check = False
    controller = MagicMock()

    with (
        patch("flextensor.tensor_manager.AllocationBlockController", return_value=controller),
        patch(
            "flextensor.tensor_manager.PreallocatedBatchTransferTensorLoader",
            side_effect=RuntimeError("wrapper failed"),
        ),
        pytest.raises(RuntimeError, match="wrapper failed"),
    ):
        manager._setup_allocation_block_loader(LoaderInputData(), prepare_state=False)

    assert manager.load_from_shm is True
    controller.abort_construction.assert_called_once()


class _WrapperConstructionError(RuntimeError):
    pass


@pytest.mark.parametrize("rearrange_transfers", [False, True])
def test_wrapper_construction_failure_aborts_controller_and_same_tensors_retry(
    rearrange_transfers: bool,
) -> None:
    manager = TensorManager.__new__(TensorManager)
    manager.rearrange_transfers = rearrange_transfers
    manager.shm_owner = None
    manager.shm_namespace = None
    manager.use_shm = False
    manager.load_from_shm = False
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
    manager.load_strategy = {"layer": [_stat(tensor_id, "layer")]}
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

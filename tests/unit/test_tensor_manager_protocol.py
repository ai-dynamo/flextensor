# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the structural TensorManager contract used by OffloadManager."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from flextensor.allocation_block import AllocationBlock
from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.helpers import NoOpTensorManager
from flextensor.loaders import AllocationBlockController
from flextensor.offload_manager import TensorManagerProtocol
from flextensor.state_handler import TensorManagerState
from flextensor.tensor_manager import TensorManager
from flextensor.utils import set_tensor_data


def _make_tensor_manager(*, loader_type: str = "allocation_block_transfer") -> TensorManager:
    """Create a fresh TensorManager without requiring CUDA hardware."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.Event", return_value=MagicMock()),
    ):
        return TensorManager(
            device_gpu="cpu",
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
            loader_type=loader_type,
        )


def _make_state_for_model(model: torch.nn.Module, loader_type: str) -> TensorManagerState:
    tensor_stats = [
        TensorStatistics(
            tensor_id=index,
            name=name,
            size_bytes=tensor.nelement() * tensor.element_size(),
            load_time_ms=0.0,
        )
        for index, (name, tensor) in enumerate(model.named_parameters())
    ]
    block_loader = loader_type in {"allocation_block_transfer", "raw_block_transfer"}
    logical_bytes = sum(stat.size_bytes for stat in tensor_stats)
    return TensorManagerState(
        loader_type=loader_type,
        tensor_id_to_name_map={stat.tensor_id: stat.name for stat in tensor_stats},
        allocation_ordered={0: ["layer_0"]} if block_loader else {},
        label_to_size_map={"layer_0": logical_bytes} if loader_type == "raw_block_transfer" else {},
        block_sizes={0: logical_bytes} if block_loader else {},
        load_strategy={"layer_0": tensor_stats},
        release_strategy={} if block_loader else {"layer_0": tensor_stats},
        label_to_block_id={"layer_0": 0} if block_loader else {},
        stats=[LayerStatistics(label="layer_0", tensors=tensor_stats, duration=1.0)],
        transfer_to_compute_map={"layer_0": "layer_0"} if block_loader else {},
        view_tensors_ids=[stat.tensor_id for stat in tensor_stats] if block_loader else [],
        view_tensors_names=[stat.name for stat in tensor_stats] if block_loader else [],
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )


def _stub_restored_phase_setup(manager: TensorManager) -> None:
    manager.prepare_infer_load_mode = MagicMock()
    manager.prepare_final_model = MagicMock(side_effect=lambda model: model)


def test_tensor_manager_satisfies_offload_manager_protocol() -> None:
    """The concrete TensorManager must expose the OffloadManager surface."""
    assert isinstance(_make_tensor_manager(), TensorManagerProtocol)


def test_noop_tensor_manager_satisfies_offload_manager_protocol() -> None:
    """The disabled-offload manager must stay in lockstep with TensorManager."""
    assert isinstance(NoOpTensorManager(device_gpu="cpu"), TensorManagerProtocol)


def test_tensor_manager_profile_restore_paths_associate_model() -> None:
    """Profile restore/load associate the model used by later phase calls."""
    restore_manager = _make_tensor_manager()
    restored_model = torch.nn.Linear(2, 2)
    _stub_restored_phase_setup(restore_manager)

    restore_manager.restore_state(
        restored_model,
        _make_state_for_model(restored_model, loader_type=restore_manager.loader_type),
    )
    assert restore_manager.initialize_warmup() is restored_model
    assert restore_manager.initialize_profile() is restored_model
    assert restore_manager.initialize_inference() is restored_model

    load_manager = _make_tensor_manager()
    loaded_model = torch.nn.Linear(2, 2)
    _stub_restored_phase_setup(load_manager)
    with patch.object(
        load_manager,
        "_load_state_from_file",
        return_value=_make_state_for_model(loaded_model, loader_type=load_manager.loader_type),
    ) as load_state:
        load_manager.load_profile("unused", loaded_model)

    load_state.assert_called_once()
    assert load_manager.initialize_warmup() is loaded_model
    assert load_manager.initialize_profile() is loaded_model
    assert load_manager.initialize_inference() is loaded_model


def test_noop_tensor_manager_profile_restore_paths_associate_model() -> None:
    """Disabled offload preserves the profile-load phase sequence contract."""
    manager = NoOpTensorManager(device_gpu="cpu")
    restored_model = object()
    loaded_model = object()

    manager.restore_state(restored_model, object())
    assert manager.initialize_warmup() is restored_model
    assert manager.initialize_profile() is restored_model
    assert manager.initialize_inference() is restored_model

    manager.load_profile("unused", loaded_model)
    assert manager.initialize_warmup() is loaded_model
    assert manager.initialize_profile() is loaded_model
    assert manager.initialize_inference() is loaded_model


class _SharedModuleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = torch.nn.Linear(2, 2)
        self.left = shared
        self.right = shared
        self.register_buffer("constant", torch.ones(2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(inputs))


class _RejectClassRestoreModel(_SharedModuleModel):
    def __init__(self) -> None:
        super().__init__()
        self.reject_class_restore = False

    def __setattr__(self, name, value):
        if name == "__class__" and self.__dict__.get("reject_class_restore", False):
            raise RuntimeError("class restoration failed")
        super().__setattr__(name, value)


def test_prepare_final_model_in_place_preserves_identity_and_shutdown_restores_classes() -> None:
    manager = _make_tensor_manager(loader_type="strategy")
    model = _SharedModuleModel()
    modules_before = {name: id(module) for name, module in model.named_modules()}
    classes_before = {id(module): type(module) for module in model.modules()}
    parameters_before = {name: id(parameter) for name, parameter in model.named_parameters()}

    final_model = manager.prepare_final_model(model, in_place=True)

    assert final_model is model
    assert {name: id(module) for name, module in model.named_modules()} == modules_before
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameters_before
    assert model.left is model.right
    assert type(model.left) is not classes_before[id(model.left)]
    assert isinstance(model.left, classes_before[id(model.left)])

    manager.shutdown()
    assert all(type(module) is classes_before[id(module)] for module in model.modules())


def test_shutdown_restores_in_place_classes_even_when_loader_shutdown_fails() -> None:
    manager = _make_tensor_manager(loader_type="strategy")
    model = _SharedModuleModel()
    classes_before = {id(module): type(module) for module in model.modules()}
    manager.prepare_final_model(model, in_place=True)
    manager.tensor_layer_loader = MagicMock()
    manager.tensor_layer_loader.shutdown.side_effect = RuntimeError("loader teardown failed")

    with pytest.raises(RuntimeError, match="loader teardown failed"):
        manager.shutdown()

    assert all(type(module) is classes_before[id(module)] for module in model.modules())


def test_shutdown_restores_classes_before_loader_teardown() -> None:
    manager = _make_tensor_manager(loader_type="strategy")
    model = _SharedModuleModel()
    classes_before = {id(module): type(module) for module in model.modules()}
    manager.prepare_final_model(model, in_place=True)
    manager.tensor_layer_loader = MagicMock()

    def assert_classes_restored():
        assert all(type(module) is classes_before[id(module)] for module in model.modules())

    manager.tensor_layer_loader.shutdown.side_effect = assert_classes_restored

    manager.shutdown()

    manager.tensor_layer_loader.shutdown.assert_called_once()


def test_failed_class_restoration_remains_tracked_and_aborts_loader_teardown() -> None:
    manager = _make_tensor_manager(loader_type="strategy")
    model = _RejectClassRestoreModel()
    original_class = type(model)
    manager.prepare_final_model(model, in_place=True)
    manager.tensor_layer_loader = MagicMock()
    model.reject_class_restore = True

    with pytest.raises(RuntimeError, match="class restoration failed"):
        manager.shutdown()

    manager.tensor_layer_loader.shutdown.assert_not_called()
    assert manager._in_place_original_classes == {model: original_class}

    model.reject_class_restore = False
    manager.shutdown()
    assert manager._in_place_original_classes == {}
    manager.tensor_layer_loader.shutdown.assert_called_once()


def _block_restore_case() -> tuple[
    TensorManager,
    torch.nn.Parameter,
    torch.Tensor,
    AllocationBlock,
    int,
    AllocationBlockController,
]:
    manager = _make_tensor_manager()
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
    expected = parameter.detach().clone()
    block = AllocationBlock(
        device="cpu",
        host_pinner=manager.host_pinner,
        release_tensor_memory=True,
    )
    block.add(parameter)
    source_view = block.allocate()[0]
    source_storage = source_view.untyped_storage()._cdata  # noqa: SLF001

    runtime_view = expected.clone()
    set_tensor_data(parameter, runtime_view)
    controller = AllocationBlockController(
        allocation_ordered={},
        device_gpu=torch.device("cpu"),
        tensors_map={},
        strategy_map={},
        label_to_block_id={},
        host_pinner=manager.host_pinner,
    )
    controller.block_map_cpu = {"layer": block}
    controller.label_to_cpu_tensor_id_map = {"layer": [id(parameter)]}
    controller.tensor_id_to_view_map = {id(parameter): runtime_view}
    controller._owned_blocks = [block]  # noqa: SLF001 - construct focused controller fixture
    manager.tensors_map = {id(parameter): parameter}
    manager.tensor_layer_loader = MagicMock(
        allocation_controller=controller,
        shutdown=controller.shutdown,
    )
    return manager, parameter, expected, block, source_storage, controller


def test_shutdown_transfers_regular_cpu_block_storage_back_to_model() -> None:
    manager, parameter, expected, _block, source_storage, controller = _block_restore_case()

    manager.shutdown()

    torch.testing.assert_close(parameter, expected)
    assert parameter.untyped_storage()._cdata == source_storage  # noqa: SLF001
    assert controller.block_map_cpu == {}


def test_reoffload_after_shutdown_accepts_disjoint_weights_in_retained_cpu_block() -> None:
    model = torch.nn.Linear(3, 2).requires_grad_(False)
    inputs = torch.arange(3.0)
    expected = model(inputs).clone()
    for _cycle in range(2):
        manager = _make_tensor_manager()
        manager.tensors_map = {id(tensor): tensor for tensor in model.parameters()}
        strategy = [
            TensorStatistics(
                tensor_id=id(tensor), name=name, size_bytes=tensor.numel() * tensor.element_size(), load_time_ms=0.0
            )
            for name, tensor in model.named_parameters()
        ]
        controller = AllocationBlockController(
            allocation_ordered={0: ["layer"]},
            device_gpu=torch.device("cpu"),
            tensors_map=manager.tensors_map,
            strategy_map={"layer": strategy},
            label_to_block_id={"layer": 0},
            host_pinner=manager.host_pinner,
        )
        cpu_storage_id = controller.block_map_cpu["layer"].block.untyped_storage()._cdata
        for tensor_id, view in controller.tensor_id_to_view_map.items():
            set_tensor_data(manager.tensors_map[tensor_id], view)
        controller.commit_construction()
        controller.schedule_transfer("layer", non_blocking=False)
        torch.testing.assert_close(model(inputs), expected)
        manager.tensor_layer_loader = MagicMock(allocation_controller=controller, shutdown=controller.shutdown)

        manager.shutdown()

        torch.testing.assert_close(model(inputs), expected)
        assert model.weight.untyped_storage()._cdata == cpu_storage_id
        assert model.bias.untyped_storage()._cdata == cpu_storage_id


def test_shutdown_copies_shm_block_storage_before_unmapping() -> None:
    manager, parameter, expected, block, source_storage, controller = _block_restore_case()
    runtime_storage = parameter.untyped_storage()._cdata  # noqa: SLF001
    block.shm_block = MagicMock()
    block.shm_block_name = "test"
    block.lock_class = MagicMock(return_value=MagicMock())

    manager.shutdown()

    torch.testing.assert_close(parameter, expected)
    assert parameter.untyped_storage()._cdata not in {source_storage, runtime_storage}  # noqa: SLF001
    assert controller.block_map_cpu == {}


def test_shutdown_can_discard_shm_backed_model_without_copying_weights() -> None:
    manager, parameter, _expected, block, _source_storage, controller = _block_restore_case()
    runtime_storage = parameter.untyped_storage()._cdata  # noqa: SLF001
    shm_block = MagicMock()
    block.shm_block = shm_block
    block.shm_block_name = "test"
    block.lock_class = MagicMock(return_value=MagicMock())

    with patch.object(manager, "_restore_block_loader_tensor_data") as restore_tensor_data:
        manager.shutdown(preserve_model=False)

    restore_tensor_data.assert_not_called()
    assert parameter.numel() == 0
    assert parameter.untyped_storage()._cdata != runtime_storage  # noqa: SLF001
    shm_block.close.assert_called_once_with()
    assert controller.block_map_cpu == {}


def test_noop_prepare_final_model_accepts_in_place() -> None:
    model = _SharedModuleModel()
    assert NoOpTensorManager(device_gpu="cpu").prepare_final_model(model, in_place=True) is model

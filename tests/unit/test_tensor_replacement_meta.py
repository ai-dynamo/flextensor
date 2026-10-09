# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for restoring meta tensors from allocation-block views."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from flextensor.loaders import AllocationBlockController
from flextensor.tensor_manager import TensorManager
from flextensor.tensor_processors import TensorReplacementProcessor


class _RejectDataAssignmentParameter(nn.Parameter):
    @property
    def data(self):
        return torch.Tensor.data.__get__(self, type(self))

    @data.setter
    def data(self, _value):
        raise RuntimeError("custom parameter rejects .data assignment")


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("parameter_type", [nn.Parameter, _RejectDataAssignmentParameter])
@pytest.mark.parametrize("mapped_is_parameter", [False, True])
def test_mapped_meta_parameter_is_rebound_with_grad_stride_and_storage(
    requires_grad: bool, parameter_type: type[nn.Parameter], mapped_is_parameter: bool
) -> None:
    source = parameter_type(
        torch.empty_strided((2, 3), (1, 2), device="meta"),
        requires_grad=requires_grad,
    )
    mapped = torch.empty_strided((2, 3), (1, 2))
    if mapped_is_parameter:
        mapped = nn.Parameter(mapped, requires_grad=not requires_grad)

    result = TensorReplacementProcessor({id(source): mapped}).process(source)

    assert type(result) is parameter_type
    assert result is not source
    assert source.is_meta
    assert result.requires_grad is requires_grad
    assert result.device.type == "cpu"
    assert result.stride() == mapped.stride()
    assert result.storage_offset() == mapped.storage_offset()
    assert result.untyped_storage().data_ptr() == mapped.untyped_storage().data_ptr()
    if parameter_type is _RejectDataAssignmentParameter:
        with pytest.raises(RuntimeError, match="custom parameter rejects"):
            result.data = mapped


def test_mapped_meta_plain_tensor_is_replaced_directly() -> None:
    source = torch.empty_strided((2, 3), (1, 2), device="meta")
    mapped = torch.empty_strided((2, 3), (1, 2))

    result = TensorReplacementProcessor({id(source): mapped}).process(source)

    assert result is mapped


def test_unmapped_meta_tensor_remains_meta_and_preserves_identity() -> None:
    source = nn.Parameter(torch.empty(4, device="meta"))

    result = TensorReplacementProcessor({}).process(source)

    assert result is source
    assert result.is_meta


@pytest.mark.parametrize("parameter_type", [nn.Parameter, _RejectDataAssignmentParameter])
def test_mapped_meta_parameter_tied_aliases_share_new_parameter_identity(parameter_type: type[nn.Parameter]) -> None:
    class _Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = parameter_type(torch.empty_strided((2, 3), (1, 2), device="meta"))
            self.left = shared
            self.right = shared

    model = _Tied()
    source = model.left
    mapped = torch.empty_strided((2, 3), (1, 2))

    TensorReplacementProcessor({id(source): mapped}).apply(model)

    assert model.left is model.right
    assert model.left is not source
    assert type(model.left) is parameter_type
    assert model.left.untyped_storage().data_ptr() == mapped.untyped_storage().data_ptr()


@pytest.mark.parametrize("mapped_is_parameter", [False, True])
def test_mapped_meta_parameter_factory_preserves_custom_constructor_and_tied_aliases(mapped_is_parameter: bool) -> None:
    class _TaggedParameter(nn.Parameter):
        def __new__(cls, data, *, tag):
            instance = super().__new__(cls, data, requires_grad=False)
            instance._tag = tag
            return instance

        @property
        def tag(self):
            return self._tag

    source = _TaggedParameter(torch.empty(3, device="meta"), tag="weight")
    model = nn.Module()
    model.left = model.right = source
    mapped = torch.arange(3.0)
    if mapped_is_parameter:
        mapped = nn.Parameter(mapped)
    factory = Mock(side_effect=lambda original, data: _TaggedParameter(data, tag=original.tag))

    TensorReplacementProcessor({id(source): mapped}, parameter_factory=factory).apply(model)

    factory.assert_called_once()
    assert factory.call_args.args[0] is source
    assert type(model.left) is _TaggedParameter
    assert model.left is model.right
    assert model.left is not source
    assert source.is_meta
    assert model.left.tag == "weight"
    assert not model.left.requires_grad
    assert torch.equal(model.left, mapped)
    assert model.left.untyped_storage().data_ptr() == mapped.untyped_storage().data_ptr()


def test_non_meta_parameter_keeps_existing_in_place_behavior() -> None:
    source = nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    mapped = torch.empty_strided((2, 3), (1, 2))

    result = TensorReplacementProcessor({id(source): mapped}).process(source)

    assert result is source
    assert result.requires_grad is False
    assert result.stride() == mapped.stride()
    assert result.untyped_storage().data_ptr() == mapped.untyped_storage().data_ptr()


def test_manager_materializes_parameter_with_required_constructor_metadata() -> None:
    class _ShardedParameter(nn.Parameter):
        def __new__(cls, data, **kwargs):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, input_dim, *, data, weight_loader):
            self._input_dim = input_dim
            self._weight_loader = weight_loader

        @property
        def input_dim(self):
            return self._input_dim

        @property
        def weight_loader(self):
            return self._weight_loader

    loader = Mock()
    source = _ShardedParameter(data=torch.empty((2, 3), device="meta"), input_dim=1, weight_loader=loader)
    model = nn.Module()
    model.left = model.right = source
    mapped = torch.arange(12.0).as_strided((2, 3), (1, 2), storage_offset=3)
    manager = TensorManager.__new__(TensorManager)
    manager.device_gpu = torch.device("cpu")
    manager.tensors_map = {id(source): source}
    manager.tensor_id_to_name_map = {id(source): "left"}

    manager._prepare_view_model_from_id_to_view_map(model, {id(source): mapped})

    result = model.left
    assert type(result) is _ShardedParameter
    assert result is model.right
    assert result is manager.tensors_map[id(source)]
    assert result is not source
    assert source.is_meta
    assert not result.requires_grad
    assert result.input_dim == 1
    assert result.weight_loader is loader
    assert result.stride() == mapped.stride()
    assert result.storage_offset() == mapped.storage_offset()
    assert result.untyped_storage().data_ptr() == mapped.untyped_storage().data_ptr()
    assert torch.equal(result, mapped)


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

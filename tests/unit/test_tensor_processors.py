# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tensor_processors module."""

import collections
import dataclasses

import pytest
import torch
import torch.nn as nn

from flextensor.tensor_manager import ModelDict
from flextensor.tensor_processors import (
    DisableRequiresGradTensorProcessor,
    LegacySetTypeHandler,
    MoveBuffersToGPUTensorProcessor,
    MoveUnmappedTensorsToGPUProcessor,
    ProcessingContext,
    ReachableTensorMapProcessor,
    TensorMappingProcessor,
    TensorProcessor,
    compute_reachable_tensor_ids,
    compute_reachable_tensor_map,
    create_model_with_shared_tensors,
    preserve_parameter_type,
)


def test_disable_requires_grad_skips_non_leaf_views() -> None:
    model = nn.Module()
    model.base = torch.arange(4.0, requires_grad=True)
    model.view = model.base[1:]

    DisableRequiresGradTensorProcessor().apply(model)

    assert model.base.requires_grad is False
    assert model.view.requires_grad is True


class TestTensorMappingProcessor:
    """Test cases for TensorMappingProcessor to ensure only nn.Parameters are tracked."""

    def test_cpu_parameter_is_tracked(self):
        """Test that CPU nn.Parameter objects are tracked for offload."""
        processor = TensorMappingProcessor()

        # Create a CPU parameter
        param = nn.Parameter(torch.randn(10, 10))

        # Process the parameter
        result = processor.process(param)

        # Assert parameter was tracked
        assert id(param) in processor.tensors_map
        assert processor.tensors_map[id(param)] is param
        assert result is param

    def test_buffer_is_tracked(self):
        """Test that buffers (non-Parameter tensors) ARE tracked."""
        processor = TensorMappingProcessor()

        # Create a regular tensor (buffer)
        buffer = torch.randn(10, 10)

        # Process the buffer
        result = processor.process(buffer)

        # Assert buffer was tracked
        assert id(buffer) in processor.tensors_map
        assert processor.tensors_map[id(buffer)] is buffer
        assert result is buffer

    def test_regular_tensor_is_tracked(self):
        """Test that regular tensors (not Parameters) ARE tracked."""
        processor = TensorMappingProcessor()

        # Create a regular tensor
        regular_tensor = torch.randn(5, 5)

        # Process the tensor
        result = processor.process(regular_tensor)

        # Assert tensor was tracked
        assert id(regular_tensor) in processor.tensors_map
        assert processor.tensors_map[id(regular_tensor)] is regular_tensor
        assert result is regular_tensor

    def test_gpu_parameter_is_not_tracked(self):
        """Test that GPU parameters are NOT tracked (only CPU parameters should be offloaded)."""
        if not torch.cuda.is_available():
            # Skip test if CUDA is not available
            return

        processor = TensorMappingProcessor()

        # Create a GPU parameter
        param_gpu = nn.Parameter(torch.randn(10, 10, device="cuda"))

        # Process the GPU parameter
        result = processor.process(param_gpu)

        # Assert GPU parameter was NOT tracked
        assert id(param_gpu) not in processor.tensors_map
        assert len(processor.tensors_map) == 0
        assert result is param_gpu

    def test_meta_parameter_is_not_tracked(self):
        """Test that meta tensors are NOT tracked."""
        processor = TensorMappingProcessor()

        # Create a meta parameter
        param_meta = nn.Parameter(torch.empty(10, 10, device="meta"))

        # Process the meta parameter
        result = processor.process(param_meta)

        # Assert meta parameter was NOT tracked
        assert id(param_meta) not in processor.tensors_map
        assert len(processor.tensors_map) == 0
        assert result is param_meta

    def test_non_tensor_returns_unchanged(self):
        """Test that non-tensor objects are returned unchanged."""
        processor = TensorMappingProcessor()

        # Test with various non-tensor objects
        test_objects = [
            42,
            "string",
            [1, 2, 3],
            {"key": "value"},
            None,
        ]

        for obj in test_objects:
            result = processor.process(obj)
            assert result is obj
            assert len(processor.tensors_map) == 0

    def test_multiple_parameters_tracked(self):
        """Test that multiple CPU parameters are all tracked correctly."""
        processor = TensorMappingProcessor()

        # Create multiple parameters
        param1 = nn.Parameter(torch.randn(5, 5))
        param2 = nn.Parameter(torch.randn(10, 10))
        param3 = nn.Parameter(torch.randn(3, 3))

        # Process all parameters
        processor.process(param1)
        processor.process(param2)
        processor.process(param3)

        # Assert all parameters were tracked
        assert len(processor.tensors_map) == 3
        assert id(param1) in processor.tensors_map
        assert id(param2) in processor.tensors_map
        assert id(param3) in processor.tensors_map

    def test_mixed_tensors_all_tracked(self):
        """Test that all tensors are tracked when processing mixed tensor types."""
        processor = TensorMappingProcessor()

        # Create mixed tensor types
        param1 = nn.Parameter(torch.randn(5, 5))
        buffer = torch.randn(5, 5)
        regular_tensor = torch.randn(5, 5)
        param2 = nn.Parameter(torch.randn(10, 10))

        # Process all tensors
        processor.process(param1)
        processor.process(buffer)
        processor.process(regular_tensor)
        processor.process(param2)

        # Assert all tensors were tracked
        assert len(processor.tensors_map) == 4
        assert id(param1) in processor.tensors_map
        assert id(param2) in processor.tensors_map
        assert id(buffer) in processor.tensors_map
        assert id(regular_tensor) in processor.tensors_map

    def test_module_with_parameters_and_buffers(self):
        """Test processing a module with both parameters and buffers."""
        processor = TensorMappingProcessor()

        # Create a simple module
        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 10))
                self.register_buffer("buffer", torch.randn(10, 10))

        module = SimpleModule()

        # Apply processor to the module
        processor.apply(module)

        # Assert both the parameter and buffer were tracked
        assert len(processor.tensors_map) == 2
        assert id(module.weight) in processor.tensors_map
        assert id(module.buffer) in processor.tensors_map

    def test_parameter_with_requires_grad_false(self):
        """Test that parameters with requires_grad=False are still tracked."""
        processor = TensorMappingProcessor()

        # Create a parameter with requires_grad=False
        param = nn.Parameter(torch.randn(10, 10), requires_grad=False)

        # Process the parameter
        result = processor.process(param)

        # Assert parameter was tracked regardless of requires_grad
        assert id(param) in processor.tensors_map
        assert processor.tensors_map[id(param)] is param
        assert result is param


class TestReachableTensorMapProcessor:
    """Contract for ``ReachableTensorMapProcessor`` and its convenience wrappers.

    The processor walks an ``nn.Module`` (parameters, buffers, submodules,
    attributes) or a ``dict`` model, plus tensor inner fields (e.g.
    ``weight.scale`` on a quantised parameter), and records every non-meta
    tensor visited.

    The narrowing input it produces is consumed by
    :class:`flextensor.loaders.TensorStrategyLoader` to scope the
    untimed-traced rescue — see
    ``tests/unit/test_untimed_traps_runtime.py::TestStrategyLoaderUntimedRescue``
    for the rescue-side assertions.
    """

    def test_returns_module_param_and_buffer_ids(self) -> None:
        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.zeros(4))
                self.register_buffer("b", torch.zeros(2))

        m = _M()
        ids = compute_reachable_tensor_ids(m)

        assert id(m.w) in ids
        assert id(m.b) in ids

    def test_returns_reachable_tensor_map(self) -> None:
        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.zeros(4))
                self.register_buffer("b", torch.zeros(2))

        m = _M()
        tensors = compute_reachable_tensor_map(m)

        assert tensors[id(m.w)] is m.w
        assert tensors[id(m.b)] is m.b

    def test_returns_tensors_nested_in_custom_tensor_containers(self) -> None:
        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.arange(8.0), requires_grad=False)

        m = _M()
        dict_view = m.w[1:4]
        set_view = m.w[4:7]
        m.w.views = {"nested": {"slice": dict_view}, "set": {set_view}}

        ids = compute_reachable_tensor_ids(m)

        assert id(dict_view) in ids
        assert id(set_view) in ids

    def test_returns_tensors_nested_in_list_and_tuple_attributes(self) -> None:
        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.arange(8.0), requires_grad=False)

        m = _M()
        list_view = m.w[1:4]
        tuple_view = m.w[4:7]
        m.views = [list_view, (tuple_view,)]

        ids = compute_reachable_tensor_ids(m)

        assert id(list_view) in ids
        assert id(tuple_view) in ids

    def test_returns_dict_model_tensor_ids(self) -> None:
        t1 = torch.zeros(4)
        t2 = torch.zeros(8)
        model = {"a": t1, "b": t2, "non_tensor": "ignored"}

        ids = compute_reachable_tensor_ids(model)

        assert ids == {id(t1), id(t2)}

    def test_returns_model_dict_tensor_ids(self) -> None:
        t1 = torch.zeros(4)
        t2 = torch.zeros(8)
        model = ModelDict(model={"a": t1, "b": t2, "non_tensor": "ignored"})

        ids = compute_reachable_tensor_ids(model)

        assert ids == {id(t1), id(t2)}

    def test_module_with_nonstandard_missing_items_getattr(self) -> None:
        """Diffusers modules can raise ``KeyError`` for missing config attrs."""

        class _DiffusersLikeModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._internal_dict: dict[str, object] = {}
                self.w = nn.Parameter(torch.zeros(4))

            def __getattr__(self, name: str) -> object:
                if name == "items":
                    return self._internal_dict[name]
                return super().__getattr__(name)

        m = _DiffusersLikeModule()
        ids = compute_reachable_tensor_ids(m)

        assert id(m.w) in ids

    def test_returns_empty_for_none(self) -> None:
        assert compute_reachable_tensor_ids(None) == set()

    def test_discovers_inner_field_tensors(self) -> None:
        """Inner-field tensors (e.g. ``weight.scale``) are reachable too.

        This is the case a hand-rolled
        ``named_parameters``/``named_buffers`` walk would miss: a
        quantised parameter that carries a sibling tensor as an instance
        attribute. The :class:`TensorProcessor` traversal — inherited by
        :class:`ReachableTensorMapProcessor` — discovers these via the standard
        ``map_inner_fields`` walk, so they participate in the rescue's
        narrowing instead of being silently excluded.
        """

        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.zeros(4))
                self.w.scale = torch.zeros(1)

        m = _M()
        ids = compute_reachable_tensor_ids(m)

        assert id(m.w) in ids
        assert id(m.w.scale) in ids

    def test_recurses_into_submodules(self) -> None:
        """Submodule params and buffers are part of the reachable set."""

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.zeros(2))

        class _Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = _Inner()

        m = _Outer()
        ids = compute_reachable_tensor_ids(m)

        assert id(m.inner.w) in ids

    def test_skips_meta_tensors(self) -> None:
        """Meta tensors have no allocation backing their ``id()``.

        Excluding them keeps the narrowing input aligned with what
        :class:`TensorMappingProcessor` actually records into
        ``tensors_map`` (CPU, non-meta tensors).
        """

        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.zeros(4))
                self.register_buffer("meta_buf", torch.zeros(2, device="meta"))

        m = _M()
        ids = compute_reachable_tensor_ids(m)

        assert id(m.w) in ids
        assert id(m.meta_buf) not in ids

    def test_id_wrapper_does_not_mutate_attributes(self) -> None:
        """Read-only walk: ``update_attributes=False`` keeps the model intact."""

        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.zeros(4))

        m = _M()
        original_w = m.w
        ids = compute_reachable_tensor_ids(m)

        assert m.w is original_w
        assert id(original_w) in ids

    def test_map_processor_does_not_mutate_attributes(self) -> None:
        class _M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.zeros(4))

        m = _M()
        original_w = m.w
        proc = ReachableTensorMapProcessor()
        proc.apply(m)

        assert m.w is original_w
        assert proc.get_results()[id(original_w)] is original_w

    def test_map_processor_revisits_plain_objects_on_each_apply(self) -> None:
        @dataclasses.dataclass
        class Holder:
            tensor: torch.Tensor

        m = nn.Module()
        m.holder = Holder(torch.zeros(1))
        proc = ReachableTensorMapProcessor()
        proc.apply(m)

        replacement = torch.ones(1)
        m.holder.tensor = replacement
        proc.apply(m)

        assert proc.get_results()[id(replacement)] is replacement


class TestMoveBuffersToGPUTensorProcessor:
    """Test cases for MoveBuffersToGPUTensorProcessor to ensure buffers are moved to GPU."""

    def test_rebinds_every_name_for_an_aliased_buffer(self):
        module = nn.Module()
        shared = torch.ones(1)
        module.register_buffer("first", shared)
        module.register_buffer("second", shared)

        MoveBuffersToGPUTensorProcessor(torch.device("meta")).apply(module)

        assert module.first is module.second
        assert module.first.device.type == "meta"

    def test_simple_module_with_buffer(self):
        """Test moving buffers in a simple module with one buffer."""
        if not torch.cuda.is_available():
            # Skip test if CUDA is not available
            return

        # Create a simple module with a buffer
        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.randn(10, 10))

        module = SimpleModule()
        device_gpu = torch.device("cuda:0")

        # Assert buffer is initially on CPU
        assert module.buffer.device.type == "cpu"

        # Apply processor
        processor = MoveBuffersToGPUTensorProcessor(device_gpu)
        processor.apply(module)

        # Assert buffer is now on GPU
        assert module.buffer.device.type == "cuda"

    def test_nested_module_with_buffers(self):
        """Test moving buffers in a nested module structure."""
        if not torch.cuda.is_available():
            # Skip test if CUDA is not available
            return

        # Create nested modules with buffers
        class ChildModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("child_buffer", torch.randn(5, 5))

        class ParentModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("parent_buffer", torch.randn(10, 10))
                self.child = ChildModule()

        module = ParentModule()
        device_gpu = torch.device("cuda:0")

        # Assert buffers are initially on CPU
        assert module.parent_buffer.device.type == "cpu"
        assert module.child.child_buffer.device.type == "cpu"

        # Apply processor
        processor = MoveBuffersToGPUTensorProcessor(device_gpu)
        processor.apply(module)

        # Assert all buffers are now on GPU
        assert module.parent_buffer.device.type == "cuda"
        assert module.child.child_buffer.device.type == "cuda"

    def test_parameters_not_moved(self):
        """Test that parameters are NOT moved, only buffers."""
        if not torch.cuda.is_available():
            # Skip test if CUDA is not available
            return

        # Create a module with both parameter and buffer
        class MixedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 10))
                self.register_buffer("buffer", torch.randn(10, 10))

        module = MixedModule()
        device_gpu = torch.device("cuda:0")

        # Assert both are initially on CPU
        assert module.weight.device.type == "cpu"
        assert module.buffer.device.type == "cpu"

        # Apply processor
        processor = MoveBuffersToGPUTensorProcessor(device_gpu)
        processor.apply(module)

        # Assert only buffer is on GPU, parameter remains on CPU
        assert module.weight.device.type == "cpu"
        assert module.buffer.device.type == "cuda"

    def test_multiple_buffers_in_module(self):
        """Test moving multiple buffers in a single module."""
        if not torch.cuda.is_available():
            # Skip test if CUDA is not available
            return

        # Create a module with multiple buffers
        class MultiBufferModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.randn(5, 5))
                self.register_buffer("buffer2", torch.randn(10, 10))
                self.register_buffer("buffer3", torch.randn(3, 3))

        module = MultiBufferModule()
        device_gpu = torch.device("cuda:0")

        # Assert all buffers are initially on CPU
        assert module.buffer1.device.type == "cpu"
        assert module.buffer2.device.type == "cpu"
        assert module.buffer3.device.type == "cpu"

        # Apply processor
        processor = MoveBuffersToGPUTensorProcessor(device_gpu)
        processor.apply(module)

        # Assert all buffers are now on GPU
        assert module.buffer1.device.type == "cuda"
        assert module.buffer2.device.type == "cuda"
        assert module.buffer3.device.type == "cuda"

    def test_module_without_buffers(self):
        """Test that processor works correctly on modules without buffers."""
        if not torch.cuda.is_available():
            # Skip test if CUDA is not available
            return

        # Create a module with only parameters
        class ParameterOnlyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 10))

        module = ParameterOnlyModule()
        device_gpu = torch.device("cuda:0")

        # Apply processor (should not raise any errors)
        processor = MoveBuffersToGPUTensorProcessor(device_gpu)
        processor.apply(module)

        # Assert parameter is still on CPU
        assert module.weight.device.type == "cpu"


class TestMoveUnmappedTensorsToGPUProcessor:
    """Test cases for MoveUnmappedTensorsToGPUProcessor handling tensors with read-only properties."""

    def test_tensor_with_readonly_properties(self):
        """
        Test that processor handles tensors with read-only properties gracefully.
        This simulates vLLM's ModelWeightParameter which has read-only input_dim and output_dim properties
        and preserves its type through __torch_function__.
        """
        if not torch.cuda.is_available():
            # Skip test if CUDA is not available
            return

        # Create a mock Parameter class with read-only properties (simulating vLLM's ModelWeightParameter)
        class MockParameterWithReadonlyProps(nn.Parameter):
            """Simulates vLLM's ModelWeightParameter with read-only properties."""

            def __new__(cls, data, input_dim=1, output_dim=0):
                return super().__new__(cls, data=data, requires_grad=False)

            def __init__(self, data, input_dim=1, output_dim=0):
                self._input_dim = input_dim
                self._output_dim = output_dim

            @property
            def input_dim(self):
                """Read-only property that cannot be set directly."""
                return self._input_dim

            @property
            def output_dim(self):
                """Read-only property that cannot be set directly."""
                return self._output_dim

            def to(self, *args, **kwargs):
                """
                Override to preserve type through .to() operations.
                This simulates vLLM's behavior where ModelWeightParameter type is preserved.
                """
                # Call parent's to() which returns a new tensor
                new_data = super().to(*args, **kwargs)
                # Wrap it back in our class to preserve the type and properties
                result = MockParameterWithReadonlyProps.__new__(
                    MockParameterWithReadonlyProps, new_data, input_dim=self._input_dim, output_dim=self._output_dim
                )
                result._input_dim = self._input_dim
                result._output_dim = self._output_dim
                return result

        # Create a tensor with read-only properties
        tensor_data = torch.randn(10, 10)
        mock_param = MockParameterWithReadonlyProps(tensor_data, input_dim=1, output_dim=0)

        # Move to CPU to ensure it starts there
        mock_param = mock_param.cpu()

        # Create processor
        device_gpu = torch.device("cuda:0")
        tensor_id_mapping = {}  # Empty mapping means all tensors are unmapped
        processor = MoveUnmappedTensorsToGPUProcessor(device_gpu, tensor_id_mapping)

        # This should not raise AttributeError when trying to set read-only properties
        result = processor.process(mock_param)

        # Assert tensor was moved to GPU
        assert result.device.type == "cuda"
        # Assert read-only properties are preserved in result if possible
        if hasattr(result, "input_dim"):
            assert result.input_dim == 1
        if hasattr(result, "output_dim"):
            assert result.output_dim == 0


class TestCreateModelWithSharedTensors:
    """Test cases for create_model_with_shared_tensors to ensure proper module copying."""

    def test_none_parameter_is_preserved(self):
        """Test that parameters registered with None value are preserved.

        This reproduces a bug where vLLM's QKVParallelLinear fails because
        bias=None parameters are not copied by create_model_with_shared_tensors.
        The issue is that named_parameters() skips None parameters, so they
        are not copied to the new model.

        Bug: AttributeError: 'QKVParallelLinear' object has no attribute 'bias'
        """

        class LinearWithOptionalBias(nn.Module):
            """Simulates vLLM's linear layers that register bias as None when not used."""

            def __init__(self, in_features: int, out_features: int, bias: bool = False):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                # This is how vLLM registers bias=None - using register_parameter
                if bias:
                    self.bias = nn.Parameter(torch.zeros(out_features))
                else:
                    self.register_parameter("bias", None)
                self.skip_bias_add = True  # vLLM-style attribute

            def forward(self, x):
                # This pattern mirrors vLLM's linear.py:561 that fails when bias is missing
                bias = self.bias if not self.skip_bias_add else None
                return torch.nn.functional.linear(x, self.weight, bias)

        # Create module with bias=None (as vLLM does)
        original = LinearWithOptionalBias(10, 20, bias=False)

        # Verify original has bias attribute (as None)
        assert hasattr(original, "bias"), "Original module should have bias attribute"
        assert original.bias is None, "Original bias should be None"
        assert "bias" in original._parameters, "bias should be in _parameters"

        # Create a copy using the function under test
        copied = create_model_with_shared_tensors(original)

        # This assertion will FAIL with the current bug:
        # The copied model should have the 'bias' attribute
        assert hasattr(copied, "bias"), "Copied module should have bias attribute"
        assert "bias" in copied._parameters, "bias should be in copied _parameters"
        assert copied.bias is None, "Copied bias should be None"

        # Additional verification: forward should work without AttributeError
        x = torch.randn(5, 10)
        try:
            _ = copied(x)
        except AttributeError as e:
            raise AssertionError(f"Forward pass failed due to missing attribute: {e}") from e

    def test_none_buffer_is_preserved(self):
        """Test that buffers registered with None value are preserved."""

        class ModuleWithOptionalBuffer(nn.Module):
            """Module that registers a buffer as None."""

            def __init__(self, use_buffer: bool = False):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 10))
                if use_buffer:
                    self.register_buffer("optional_buffer", torch.zeros(10))
                else:
                    self.register_buffer("optional_buffer", None)

        # Create module with buffer=None
        original = ModuleWithOptionalBuffer(use_buffer=False)

        # Verify original has buffer attribute (as None)
        assert hasattr(original, "optional_buffer"), "Original should have optional_buffer attribute"
        assert original.optional_buffer is None, "Original buffer should be None"
        assert "optional_buffer" in original._buffers, "optional_buffer should be in _buffers"

        # Create a copy
        copied = create_model_with_shared_tensors(original)

        # Verify copied has buffer attribute (as None)
        assert hasattr(copied, "optional_buffer"), "Copied should have optional_buffer attribute"
        assert "optional_buffer" in copied._buffers, "optional_buffer should be in copied _buffers"
        assert copied.optional_buffer is None, "Copied buffer should be None"

    def test_nested_module_with_none_parameters(self):
        """Test that None parameters are preserved in nested modules."""

        class InnerLinear(nn.Module):
            def __init__(self, in_features: int, out_features: int, bias: bool = False):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                if bias:
                    self.bias = nn.Parameter(torch.zeros(out_features))
                else:
                    self.register_parameter("bias", None)

        class OuterModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = InnerLinear(10, 20, bias=False)
                self.linear2 = InnerLinear(20, 10, bias=True)

        original = OuterModule()

        # Verify original structure
        assert original.linear1.bias is None
        assert original.linear2.bias is not None
        assert "bias" in original.linear1._parameters
        assert "bias" in original.linear2._parameters

        # Create a copy
        copied = create_model_with_shared_tensors(original)

        # Verify both modules have their bias attributes preserved
        assert hasattr(copied.linear1, "bias"), "Copied linear1 should have bias attribute"
        assert "bias" in copied.linear1._parameters, "bias should be in linear1._parameters"
        assert copied.linear1.bias is None, "Copied linear1 bias should be None"

        assert hasattr(copied.linear2, "bias"), "Copied linear2 should have bias attribute"
        assert copied.linear2.bias is not None, "Copied linear2 bias should not be None"

    def test_unregistered_tensor_attribute_is_preserved(self) -> None:
        """Unregistered tensor attributes must survive copy."""

        class LinearWithWorkspace(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))
                self.workspace = torch.empty(8)

        original = nn.Sequential(LinearWithWorkspace())

        copied = create_model_with_shared_tensors(original)

        assert hasattr(copied[0], "workspace")
        assert copied[0].workspace is original[0].workspace


class FrozenDict(collections.OrderedDict):
    """Minimal FrozenDict for testing — mirrors diffusers' FrozenDict behaviour.

    Stores every key as an object attribute so ``d.key`` works, then freezes
    the instance to prevent further mutation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            setattr(self, key, value)
        self.__frozen = True

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise AttributeError(f"Cannot set attribute on frozen {self.__class__.__name__}")
        super().__setattr__(name, value)


class _IdentityProcessor(TensorProcessor):
    """TensorProcessor that returns tensors unchanged — nothing should be mutated."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, src):
        return src


class _CloneProcessor(TensorProcessor):
    """TensorProcessor that clones every tensor — simulates a real move/copy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, src):
        if isinstance(src, torch.Tensor):
            return src.clone()
        return src


class TestDuplicateModuleTraversalLogging:
    """Duplicate module traversal should be summarized once, not logged per skip."""

    @staticmethod
    def _model_with_shared_child(shared: nn.Module) -> nn.Module:
        model = nn.Module()
        model.left = nn.Module()
        model.right = nn.Module()
        model.extra = nn.Module()
        model.left.shared = shared
        model.right.shared = shared
        model.extra.shared = shared
        return model

    def test_tensor_processor_logs_duplicate_module_summary(self, caplog):
        shared = nn.Linear(2, 2)
        model = self._model_with_shared_child(shared)
        processor = _IdentityProcessor()

        with caplog.at_level("DEBUG", logger="flextensor.tensor_processors"):
            processor.apply(model)

        messages = [record.getMessage() for record in caplog.records]
        summary = next(message for message in messages if "already-visited module visit" in message)
        assert "_IdentityProcessor: skipped 2 already-visited module visit(s)" in summary
        assert f"shared (id={id(shared)}, skipped=2)" in summary

    def test_move_buffers_processor_logs_duplicate_module_summary(self, caplog):
        shared = nn.Module()
        shared.register_buffer("buffer", torch.ones(1))
        model = self._model_with_shared_child(shared)
        processor = MoveBuffersToGPUTensorProcessor(torch.device("cpu"))

        with caplog.at_level("DEBUG", logger="flextensor.tensor_processors"):
            processor.apply(model)

        messages = [record.getMessage() for record in caplog.records]
        summary = next(message for message in messages if "already-visited module visit" in message)
        assert "MoveBuffersToGPUTensorProcessor: skipped 2 already-visited module visit(s)" in summary
        assert f"shared (id={id(shared)}, skipped=2)" in summary


class TestFrozenDictPreservation:
    """Test that _apply_on preserves OrderedDict subclass types (e.g. FrozenDict)."""

    def test_frozendict_preserved_when_no_values_change(self):
        """FrozenDict must survive _apply_on when none of its values are tensors."""

        class ModelWithFrozenConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self._internal_dict = FrozenDict({"num_layers": 4, "hidden_size": 128})
                self.linear = nn.Linear(10, 10)

            @property
            def config(self):
                return self._internal_dict

        model = ModelWithFrozenConfig()

        # Apply a processor that does not change any values
        processor = _IdentityProcessor()
        processor.apply(model)

        # FrozenDict type must be preserved
        assert type(model._internal_dict).__name__ == "FrozenDict", (
            f"Expected FrozenDict, got {type(model._internal_dict).__name__}"
        )
        # Attribute access must still work
        assert model.config.num_layers == 4
        assert model.config.hidden_size == 128

    def test_frozendict_replaced_when_tensor_value_changes(self):
        """When a FrozenDict contains a tensor that gets processed, a new OrderedDict is created."""

        class ModelWithTensorConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self._internal_dict = FrozenDict({"scale": torch.tensor(1.0), "name": "test"})
                self.linear = nn.Linear(10, 10)

        model = ModelWithTensorConfig()

        # Apply a processor that clones tensors (changes identity)
        processor = _CloneProcessor()
        processor.apply(model)

        # Original FrozenDict is immutable, so a new OrderedDict should replace it
        assert isinstance(model._internal_dict, collections.OrderedDict)
        assert "scale" in model._internal_dict
        assert "name" in model._internal_dict

    def test_plain_ordered_dict_preserved_when_no_values_change(self):
        """Plain OrderedDict with only non-tensor values should be preserved as-is."""

        class ModelWithOrderedDict(nn.Module):
            def __init__(self):
                super().__init__()
                self.metadata = collections.OrderedDict({"version": 1, "mode": "train"})
                self.linear = nn.Linear(10, 10)

        model = ModelWithOrderedDict()
        original_dict = model.metadata

        processor = _IdentityProcessor()
        processor.apply(model)

        # Same object should be preserved
        assert model.metadata is original_dict

    def test_plain_dict_preserved_when_no_values_change(self):
        """Plain dict with only non-tensor values should be preserved as-is."""

        class ModelWithDict(nn.Module):
            def __init__(self):
                super().__init__()
                self.settings = {"lr": 0.01, "momentum": 0.9}
                self.linear = nn.Linear(10, 10)

        model = ModelWithDict()
        original_dict = model.settings

        processor = _IdentityProcessor()
        processor.apply(model)

        assert model.settings is original_dict


class TestPreserveParameterType:
    """Test that _apply_on preserves nn.Parameter types and object identity."""

    def test_preserve_parameter_type_returns_same_object(self):
        """preserve_parameter_type should return the same Parameter object with updated .data."""
        original = nn.Parameter(torch.randn(4, 4))
        new_data = torch.randn(4, 4)
        original_id = id(original)

        result = preserve_parameter_type(original, new_data)

        assert isinstance(result, nn.Parameter), "Result should be nn.Parameter"
        assert id(result) == original_id, "Should return the same object (identity preserved)"
        assert torch.equal(result.data, new_data), "Data should be updated"

    def test_preserve_parameter_type_keeps_requires_grad(self):
        """requires_grad should be preserved from the original Parameter."""
        param_grad = nn.Parameter(torch.randn(3, 3), requires_grad=True)
        param_no_grad = nn.Parameter(torch.randn(3, 3), requires_grad=False)
        new_data = torch.randn(3, 3)

        result_grad = preserve_parameter_type(param_grad, new_data.clone())
        result_no_grad = preserve_parameter_type(param_no_grad, new_data.clone())

        assert result_grad.requires_grad is True
        assert result_no_grad.requires_grad is False

    def test_preserve_parameter_type_noop_for_plain_tensor(self):
        """When original is a plain tensor (not Parameter), return new_value unchanged."""
        original = torch.randn(4, 4)
        new_data = torch.randn(4, 4)

        result = preserve_parameter_type(original, new_data)

        assert result is new_data, "Should return new_value as-is for non-Parameter"

    def test_preserve_parameter_type_noop_when_already_parameter(self):
        """When new_value is already an nn.Parameter, return it unchanged."""
        original = nn.Parameter(torch.randn(4, 4))
        new_param = nn.Parameter(torch.randn(4, 4))

        result = preserve_parameter_type(original, new_param)

        assert result is new_param, "Should return new_value as-is when already a Parameter"

    def test_parameters_dict_preserved_through_apply(self):
        """nn.Parameters in _parameters dict must remain Parameters after _apply_on."""

        class ModelWithParameter(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale_shift_table = nn.Parameter(torch.randn(6, 64))
                self.weight = nn.Parameter(torch.randn(10, 10))

        model = ModelWithParameter()
        original_sst_id = id(model.scale_shift_table)
        original_weight_id = id(model.weight)

        # Clone processor changes tensor data but returns plain Tensor
        processor = _CloneProcessor()
        processor.apply(model)

        # Parameters must still be nn.Parameter
        assert isinstance(model.scale_shift_table, nn.Parameter), (
            f"Expected nn.Parameter, got {type(model.scale_shift_table).__name__}"
        )
        assert isinstance(model.weight, nn.Parameter), f"Expected nn.Parameter, got {type(model.weight).__name__}"

        # Object identity must be preserved (in-place .data update)
        assert id(model.scale_shift_table) == original_sst_id, "scale_shift_table identity should be preserved"
        assert id(model.weight) == original_weight_id, "weight identity should be preserved"

    def test_parameter_identity_preserved_for_id_tracking(self):
        """Verify that id()-based tracking still works after processing.

        This is critical because tensors_map, traced_tensors, and processor
        caches all use id(tensor) as keys.
        """

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(5, 5))

        model = SimpleModel()

        # Record the parameter id before processing
        param_id_before = id(model.param)

        # Process with a processor that modifies tensor data
        processor = _CloneProcessor()
        processor.apply(model)

        # The id must be the same — this is what tensors_map relies on
        param_id_after = id(model.param)
        assert param_id_before == param_id_after, (
            f"Parameter id changed from {param_id_before} to {param_id_after}; "
            "this would break id()-based tensor tracking"
        )

    def test_force_update_creates_new_parameter(self):
        """When force_update_nn_parameters=True, a new nn.Parameter is created.

        This intentionally breaks identity — use when building an independent
        model copy where parameters should not be shared.
        """
        original = nn.Parameter(torch.randn(4, 4))
        new_data = torch.randn(4, 4)
        original_id = id(original)

        result = preserve_parameter_type(original, new_data, force_update=True)

        assert isinstance(result, nn.Parameter), "Result should be nn.Parameter"
        assert id(result) != original_id, "Should create a new Parameter object"
        assert torch.equal(result.data, new_data), "Data should match new_data"

    def test_force_update_preserves_requires_grad(self):
        """force_update_nn_parameters=True should still preserve requires_grad."""

        param_grad = nn.Parameter(torch.randn(3, 3), requires_grad=True)
        param_no_grad = nn.Parameter(torch.randn(3, 3), requires_grad=False)
        new_data = torch.randn(3, 3)

        result_grad = preserve_parameter_type(param_grad, new_data.clone(), force_update=True)
        result_no_grad = preserve_parameter_type(param_no_grad, new_data.clone(), force_update=True)

        assert result_grad.requires_grad is True
        assert result_no_grad.requires_grad is False

    def test_force_update_through_apply_creates_new_parameters(self):
        """End-to-end: force_update_nn_parameters=True creates new Parameters via _apply_on."""

        class ModelWithParameter(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(5, 5))

        model = ModelWithParameter()
        original_id = id(model.param)

        # Clone processor with force_update — should create new Parameter objects
        processor = _CloneProcessor(force_update_nn_parameters=True)
        processor.apply(model)

        assert isinstance(model.param, nn.Parameter), "Should still be nn.Parameter"
        assert id(model.param) != original_id, "Should be a new Parameter object"


class TestUnwrapCompiledModule:
    """``_unwrap_compiled_module`` must follow Dynamo's OptimizedModule contract."""

    def test_unwrap_real_torch_compile_wrapper(self):
        from flextensor.tensor_processors import _unwrap_compiled_module

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale_shift_table = nn.Parameter(torch.randn(2, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.scale_shift_table.sum()

        inner = Block()
        compiled = torch.compile(inner)
        assert _unwrap_compiled_module(compiled) is inner
        assert _unwrap_compiled_module(inner) is inner

    def test_unwrap_raises_on_broken_wrapper(self, monkeypatch):
        import flextensor.tensor_processors as tp

        class BrokenOptimizedModule(nn.Module):
            def __init__(self):
                super().__init__()

        monkeypatch.setattr(
            tp,
            "is_torch_compiled_module",
            lambda module: isinstance(module, BrokenOptimizedModule),
        )
        broken = BrokenOptimizedModule()
        with pytest.raises(RuntimeError, match="missing _orig_mod"):
            tp._unwrap_compiled_module(broken)

    def test_apply_on_compiled_preserves_direct_parameters(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale_shift_table = nn.Parameter(torch.randn(2, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.scale_shift_table.sum()

        inner = Block()
        original_id = id(inner.scale_shift_table)
        compiled = torch.compile(inner)
        _CloneProcessor().apply(compiled)
        assert isinstance(inner.scale_shift_table, nn.Parameter)
        assert id(inner.scale_shift_table) == original_id


class TestParameterFactory:
    """Test the parameter_factory hook in preserve_parameter_type and process_and_preserve."""

    def test_default_behavior_unchanged_without_factory(self):
        """Without a factory, force_update uses the default type(original)(...) path."""
        original = nn.Parameter(torch.randn(4, 4))
        new_data = torch.randn(4, 4)

        result = preserve_parameter_type(original, new_data, force_update=True)

        assert isinstance(result, nn.Parameter)
        assert id(result) != id(original)
        assert torch.equal(result.data, new_data)

    def test_factory_called_on_force_update(self):
        """When force_update=True and a factory is provided, the factory constructs the parameter."""

        class CustomParam(nn.Parameter):
            def __new__(cls, data, requires_grad=True, label="default"):
                instance = super().__new__(cls, data, requires_grad=requires_grad)
                instance.label = label
                return instance

        original = CustomParam(torch.randn(3, 3), label="custom")
        new_data = torch.randn(3, 3)

        def factory(orig: nn.Parameter, data: torch.Tensor) -> nn.Parameter:
            return CustomParam(data, requires_grad=orig.requires_grad, label=orig.label)

        result = preserve_parameter_type(original, new_data, force_update=True, parameter_factory=factory)

        assert isinstance(result, CustomParam), "Factory should produce a CustomParam"
        assert result.label == "custom", "Factory should preserve custom attributes"
        assert torch.equal(result.data, new_data)
        assert id(result) != id(original)

    def test_factory_ignored_when_force_update_false(self):
        """When force_update=False, the factory is never called — in-place update is used."""
        original = nn.Parameter(torch.randn(4, 4))
        new_data = torch.randn(4, 4)
        original_id = id(original)

        def factory(orig, data):
            raise AssertionError("Factory should not be called when force_update=False")

        result = preserve_parameter_type(original, new_data, force_update=False, parameter_factory=factory)

        assert id(result) == original_id, "Should update in-place, preserving identity"
        assert torch.equal(result.data, new_data)

    def test_factory_via_process_and_preserve(self):
        """End-to-end: process_and_preserve forwards the factory to preserve_parameter_type."""

        class CustomParam(nn.Parameter):
            def __new__(cls, data, requires_grad=True, tag=""):
                instance = super().__new__(cls, data, requires_grad=requires_grad)
                instance.tag = tag
                return instance

        factory_calls = []

        def factory(orig: nn.Parameter, data: torch.Tensor) -> nn.Parameter:
            factory_calls.append(orig)
            return CustomParam(data, requires_grad=orig.requires_grad, tag=orig.tag)

        original = CustomParam(torch.randn(3, 3), tag="test")

        processor = _CloneProcessor(force_update_nn_parameters=True)
        ctx = ProcessingContext(processor)

        result = ctx.process_and_preserve(original, parameter_factory=factory)

        assert len(factory_calls) == 1, "Factory should have been called once"
        assert isinstance(result, CustomParam), "Result should be CustomParam"
        assert result.tag == "test", "Factory should preserve the tag"
        assert id(result) != id(original), "Should be a new object (force_update=True)"

    def test_factory_not_called_via_process_and_preserve_inplace(self):
        """process_and_preserve with force_update=False should not invoke the factory."""

        def factory(orig, data):
            raise AssertionError("Factory should not be called for in-place update")

        original = nn.Parameter(torch.randn(3, 3))
        original_id = id(original)

        processor = _CloneProcessor(force_update_nn_parameters=False)
        ctx = ProcessingContext(processor)

        result = ctx.process_and_preserve(original, parameter_factory=factory)

        assert id(result) == original_id, "Should update in-place"


class TestTypeHandlerSystem:
    """Test the type handler dispatch system for TensorProcessor."""

    def setup_method(self):
        """Clean up global handlers before each test."""
        TensorProcessor.clear_global_type_handlers()

    def teardown_method(self):
        """Clean up global handlers after each test."""
        TensorProcessor.clear_global_type_handlers()

    # --- Basic handler registration ---

    def test_instance_handler_is_called(self):
        """A handler registered on an instance should process matching attributes."""

        class _DoubleHandler:
            """Doubles all float tensors (for testing)."""

            def can_handle(self, value):
                return isinstance(value, torch.Tensor) and value.dtype == torch.float32

            def process_attribute(self, value, ctx):
                return value * 2

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(3, 3))

        model = SimpleModel()
        processor = _IdentityProcessor()
        processor.register_type_handler(_DoubleHandler())
        processor.apply(model)

        # The custom handler should have doubled the weight data
        assert torch.allclose(model.weight.data, torch.ones(3, 3) * 2)

    def test_global_handler_is_called(self):
        """A globally registered handler should affect all processor instances."""

        call_count = 0

        class _CountingHandler:
            def can_handle(self, value):
                return isinstance(value, torch.Tensor)

            def process_attribute(self, value, ctx):
                nonlocal call_count
                call_count += 1
                return value

        TensorProcessor.register_global_type_handler(_CountingHandler())

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 3))

        model = SimpleModel()
        processor = _IdentityProcessor()
        processor.apply(model)

        # The global handler should have been called for the tensor attribute
        assert call_count > 0

    def test_instance_handler_takes_priority_over_global(self):
        """Instance-level handlers should be checked before global handlers."""

        order = []

        class _GlobalHandler:
            def can_handle(self, value):
                if isinstance(value, torch.Tensor):
                    order.append("global")
                    return True
                return False

            def process_attribute(self, value, ctx):
                return value

        class _InstanceHandler:
            def can_handle(self, value):
                if isinstance(value, torch.Tensor):
                    order.append("instance")
                    return True
                return False

            def process_attribute(self, value, ctx):
                return value

        TensorProcessor.register_global_type_handler(_GlobalHandler())

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2, 2))

        model = SimpleModel()
        processor = _IdentityProcessor()
        processor.register_type_handler(_InstanceHandler())
        processor.apply(model)

        # Instance handler should be checked first and win
        # For the tensor attribute in __dict__, the first "instance" match stops iteration
        assert order[0] == "instance"

    def test_custom_handler_takes_priority_over_builtin(self):
        """Custom handlers should be checked before built-in tensor/dict/set handlers."""

        class _SkipTensorHandler:
            """Returns tensors unchanged without going through process()."""

            def can_handle(self, value):
                return isinstance(value, torch.Tensor)

            def process_attribute(self, value, ctx):
                # Return a marker tensor to prove this handler was used
                return torch.zeros_like(value)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(3, 3))

        model = SimpleModel()
        # CloneProcessor would clone; custom handler zeroes out
        processor = _CloneProcessor()
        processor.register_type_handler(_SkipTensorHandler())
        processor.apply(model)

        # Custom handler should have zeroed the weight, not cloned it
        assert torch.allclose(model.weight.data, torch.zeros(3, 3))

    def test_handler_via_constructor(self):
        """Handlers can be passed via the type_handlers constructor parameter."""

        class _ZeroHandler:
            def can_handle(self, value):
                return isinstance(value, torch.Tensor)

            def process_attribute(self, value, ctx):
                return torch.zeros_like(value)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4, 4))

        model = SimpleModel()
        processor = _IdentityProcessor(type_handlers=[_ZeroHandler()])
        processor.apply(model)

        assert torch.allclose(model.weight.data, torch.zeros(4, 4))

    def test_clear_global_handlers(self):
        """clear_global_type_handlers should remove all global handlers."""

        class _NeverHandler:
            def can_handle(self, value):
                raise AssertionError("Should not be called after clearing")

            def process_attribute(self, value, ctx):
                raise AssertionError("Should not be called after clearing")

        TensorProcessor.register_global_type_handler(_NeverHandler())
        TensorProcessor.clear_global_type_handlers()

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2, 2))

        model = SimpleModel()
        processor = _IdentityProcessor()
        # Should not raise — _NeverHandler was cleared
        processor.apply(model)

    # --- Custom type handling for exotic types ---

    def test_custom_handler_for_exotic_parameter_subclass(self):
        """Simulate handling a SharedWeightParameter-like type with custom handler.

        SharedWeightParameter raises ValueError on .data access and has
        custom attributes (partitions, local_tensors) that need special
        processing.
        """

        class ExoticParameter(nn.Parameter):
            """Simulates vLLM's SharedWeightParameter."""

            def __new__(cls, **kwargs):
                return super().__new__(cls, data=torch.empty(0), requires_grad=False)

            def __init__(self, **kwargs):
                self.partitions = {}
                self.local_tensors = set()

            @property
            def data(self):
                raise ValueError("Cannot access .data on ExoticParameter")

            @data.setter
            def data(self, value):
                # Allow PyTorch internals to set data during __new__
                torch.Tensor.data.fset(self, value)

        class ExoticHandler:
            """Custom handler that processes ExoticParameter's internals."""

            def can_handle(self, value):
                return isinstance(value, ExoticParameter)

            def process_attribute(self, value, ctx):
                # Process partitions
                for key, param in value.partitions.items():
                    new_data = ctx.process(param)
                    value.partitions[key] = new_data
                # Process local_tensors
                value.local_tensors = {ctx.process(t) for t in value.local_tensors}
                return value

        class ModelWithExotic(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                self.exotic = ExoticParameter()
                self.exotic.partitions[0] = nn.Parameter(torch.randn(5, 5))
                self.exotic.local_tensors.add(torch.randn(3, 3))

        model = ModelWithExotic()

        processor = _CloneProcessor()
        processor.register_type_handler(ExoticHandler())
        processor.apply(model)

        # ExoticParameter itself should still be the same object
        assert isinstance(model.exotic, ExoticParameter)
        # Partitions should have been processed (cloned)
        assert model.exotic.partitions[0] is not None
        # local_tensors should still exist
        assert len(model.exotic.local_tensors) == 1

    def test_custom_handler_for_dict_values(self):
        """Custom handlers should be checked for values inside dicts too.

        This is critical for _parameters dicts containing custom Parameter
        subclasses that need special processing.
        """

        class SpecialTensor(torch.Tensor):
            """A tensor subclass that should be handled specially."""

            _marker = "special"

        handled_values = []

        class SpecialTensorHandler:
            def can_handle(self, value):
                return isinstance(value, SpecialTensor)

            def process_attribute(self, value, ctx):
                handled_values.append(value)
                return value  # Return as-is

        class ModelWithDictContainingSpecial(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                # Store a special tensor in a dict attribute
                special = SpecialTensor(torch.randn(3, 3))
                self.custom_dict = {"normal_key": "normal_value", "special": special}

        model = ModelWithDictContainingSpecial()
        processor = _IdentityProcessor()
        processor.register_type_handler(SpecialTensorHandler())
        processor.apply(model)

        # The custom handler should have been called for the SpecialTensor inside the dict
        assert len(handled_values) == 1
        assert isinstance(handled_values[0], SpecialTensor)

    def test_custom_handler_for_set_values(self):
        """Custom handlers should be checked for values inside sets too."""

        class TaggedTensor(torch.Tensor):
            """Tensor subclass with a tag."""

        handled_count = 0

        class TaggedTensorHandler:
            def can_handle(self, value):
                return isinstance(value, TaggedTensor)

            def process_attribute(self, value, ctx):
                nonlocal handled_count
                handled_count += 1
                return value

        class ModelWithTaggedSet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.tensor_set = {TaggedTensor(torch.randn(2, 2)), TaggedTensor(torch.randn(3, 3))}

        model = ModelWithTaggedSet()
        processor = _IdentityProcessor()
        processor.register_type_handler(TaggedTensorHandler())
        processor.apply(model)

        assert handled_count == 2, f"Expected 2 tagged tensors handled, got {handled_count}"

    # --- ProcessingContext API ---

    def test_processing_context_process_delegates(self):
        """ProcessingContext.process() should delegate to processor.process()."""
        processor = _CloneProcessor()
        ctx = ProcessingContext(processor)

        original = torch.randn(3, 3)
        result = ctx.process(original)

        # CloneProcessor.process clones tensors
        assert result is not original
        assert torch.equal(result, original)

    def test_processing_context_process_and_preserve(self):
        """ProcessingContext.process_and_preserve() should preserve nn.Parameter."""
        processor = _CloneProcessor()
        ctx = ProcessingContext(processor)

        original = nn.Parameter(torch.randn(3, 3))
        original_id = id(original)
        result = ctx.process_and_preserve(original)

        # Should preserve the original Parameter object (in-place .data update)
        assert isinstance(result, nn.Parameter)
        assert id(result) == original_id

    def test_processing_context_dispatch_checks_custom_handlers(self):
        """ProcessingContext.dispatch() should check custom handlers."""

        class _MarkerHandler:
            def can_handle(self, value):
                return isinstance(value, str) and value == "MARKER"

            def process_attribute(self, value, ctx):
                return "PROCESSED"

        processor = _IdentityProcessor()
        processor.register_type_handler(_MarkerHandler())
        ctx = ProcessingContext(processor)

        assert ctx.dispatch("MARKER") == "PROCESSED"
        assert ctx.dispatch("other") == "other"  # Not handled, returned as-is

    # --- Backward compatibility ---

    def test_builtin_handlers_preserve_frozendict(self):
        """Existing FrozenDict preservation behavior must still work."""

        class ModelWithFrozenConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self._internal_dict = FrozenDict({"num_layers": 4, "hidden_size": 128})
                self.linear = nn.Linear(10, 10)

        model = ModelWithFrozenConfig()
        processor = _IdentityProcessor()
        processor.apply(model)

        assert type(model._internal_dict).__name__ == "FrozenDict"
        assert model._internal_dict["num_layers"] == 4

    def test_builtin_handlers_preserve_parameter_identity(self):
        """Existing Parameter identity preservation must still work."""

        class ModelWithParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

        model = ModelWithParam()
        original_id = id(model.weight)

        processor = _CloneProcessor()
        processor.apply(model)

        assert isinstance(model.weight, nn.Parameter)
        assert id(model.weight) == original_id

    def test_no_handler_leaves_attribute_unchanged(self):
        """Attributes that match no handler should be left unchanged."""

        class ModelWithMisc(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.name = "test_model"
                self.count = 42
                self.flag = True

        model = ModelWithMisc()
        processor = _IdentityProcessor()
        processor.apply(model)

        assert model.name == "test_model"
        assert model.count == 42
        assert model.flag is True

    def test_multiple_handlers_first_match_wins(self):
        """When multiple handlers can_handle the same value, first match wins."""

        call_log = []

        class _HandlerA:
            def can_handle(self, value):
                return isinstance(value, torch.Tensor)

            def process_attribute(self, value, ctx):
                call_log.append("A")
                return value

        class _HandlerB:
            def can_handle(self, value):
                return isinstance(value, torch.Tensor)

            def process_attribute(self, value, ctx):
                call_log.append("B")
                return value

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2, 2))

        model = SimpleModel()
        # Register B first, then A — A should be checked first (inserted at 0)
        processor = _IdentityProcessor()
        processor.register_type_handler(_HandlerB())
        processor.register_type_handler(_HandlerA())
        processor.apply(model)

        # A was registered last → inserted at position 0 → checked first
        assert "A" in call_log
        assert "B" not in call_log


class TestLegacySetTypeHandler:
    """Test that LegacySetTypeHandler reproduces original pre-ADR-0003 set semantics."""

    def setup_method(self):
        TensorProcessor.clear_global_type_handlers()

    def teardown_method(self):
        TensorProcessor.clear_global_type_handlers()

    def test_parameter_wrapper_not_preserved_in_set(self):
        """Original set handling called self.process() directly — Parameter wrapper was stripped.

        With the default SetTypeHandler, Parameter preservation kicks in.
        With LegacySetTypeHandler, the old stripping behavior is reproduced.
        """

        class ModelWithParamSet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.tensor_set = {nn.Parameter(torch.randn(2, 2), requires_grad=False)}

        model = ModelWithParamSet()
        processor = _CloneProcessor()
        processor.register_type_handler(LegacySetTypeHandler())
        processor.apply(model)

        # With the legacy handler, process() returns a plain clone (not a Parameter)
        for elem in model.tensor_set:
            assert isinstance(elem, torch.Tensor)
            assert not isinstance(elem, nn.Parameter), "LegacySetTypeHandler should NOT preserve nn.Parameter wrapper"

    def test_default_set_handler_preserves_parameter(self):
        """Contrast: the default SetTypeHandler DOES preserve nn.Parameter."""

        class ModelWithParamSet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.tensor_set = {nn.Parameter(torch.randn(2, 2), requires_grad=False)}

        model = ModelWithParamSet()
        original_param = next(iter(model.tensor_set))
        original_id = id(original_param)

        processor = _CloneProcessor()
        # No LegacySetTypeHandler — use the default built-in handler
        processor.apply(model)

        for elem in model.tensor_set:
            assert isinstance(elem, nn.Parameter), "Default SetTypeHandler should preserve nn.Parameter wrapper"
            assert id(elem) == original_id, "Default handler preserves identity via in-place .data update"

    def test_legacy_handler_skips_custom_handler_dispatch(self):
        """LegacySetTypeHandler uses ctx.process() directly — no handler dispatch for elements."""

        class TaggedTensor(torch.Tensor):
            pass

        handler_called = False

        class TaggedHandler:
            def can_handle(self, value):
                return isinstance(value, TaggedTensor)

            def process_attribute(self, value, ctx):
                nonlocal handler_called
                handler_called = True
                return value

        class ModelWithTaggedSet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.tensor_set = {TaggedTensor(torch.randn(2, 2))}

        model = ModelWithTaggedSet()
        processor = _IdentityProcessor()
        processor.register_type_handler(TaggedHandler())
        processor.register_type_handler(LegacySetTypeHandler())
        processor.apply(model)

        # LegacySetTypeHandler calls ctx.process() directly, so TaggedHandler is never checked
        assert not handler_called, "LegacySetTypeHandler should bypass handler dispatch for set elements"

    def test_legacy_handler_processes_plain_tensors(self):
        """LegacySetTypeHandler should still process plain tensors through process()."""

        class ModelWithTensorSet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                t = torch.randn(3, 3)
                self.tensor_set = {t}

        model = ModelWithTensorSet()
        original_elem = next(iter(model.tensor_set))

        processor = _CloneProcessor()
        processor.register_type_handler(LegacySetTypeHandler())
        processor.apply(model)

        new_elem = next(iter(model.tensor_set))
        # CloneProcessor.process() clones the tensor — should be a different object
        assert new_elem is not original_elem
        assert torch.equal(new_elem, original_elem)

    def test_legacy_handler_overrides_builtin_via_priority(self):
        """LegacySetTypeHandler registered on an instance takes priority over built-in SetTypeHandler."""

        class ModelWithSet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.tensor_set = {torch.randn(2, 2)}

        model = ModelWithSet()
        processor = _IdentityProcessor()
        processor.register_type_handler(LegacySetTypeHandler())

        # Should work without errors — legacy handler takes priority over built-in
        processor.apply(model)
        assert len(model.tensor_set) == 1


class TestNestedContainerDispatch:
    """Verify that dispatch() dispatches to built-in handlers for nested containers.

    These tests prove that nested dicts, sets, and deeply nested structures are
    correctly processed — disproving the concern that dispatch() silently
    skips built-in handlers.
    """

    def test_dict_value_that_is_itself_a_dict_with_tensors(self):
        """Tensors inside a nested dict (dict of dict) must be processed."""

        class ModelWithNestedDict(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.nested = {
                    "inner": {
                        "tensor": torch.randn(3, 3),
                        "plain": "hello",
                    },
                    "top_tensor": torch.randn(2, 2),
                }

        model = ModelWithNestedDict()
        inner_tensor_orig = model.nested["inner"]["tensor"]
        top_tensor_orig = model.nested["top_tensor"]

        processor = _CloneProcessor()
        processor.apply(model)

        # Top-level tensor in the dict should have been cloned
        assert model.nested["top_tensor"] is not top_tensor_orig
        assert torch.equal(model.nested["top_tensor"], top_tensor_orig)

        # Inner nested dict tensor should ALSO have been cloned
        assert model.nested["inner"]["tensor"] is not inner_tensor_orig
        assert torch.equal(model.nested["inner"]["tensor"], inner_tensor_orig)

        # Non-tensor value should be unchanged
        assert model.nested["inner"]["plain"] == "hello"

    def test_dict_value_that_is_a_set_with_tensors(self):
        """Tensors inside a set nested within a dict must be processed."""

        class ModelWithDictOfSet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.data = {
                    "tensor_set": {torch.randn(2, 2), torch.randn(3, 3)},
                    "plain": "value",
                }

        model = ModelWithDictOfSet()
        original_tensors = {id(t) for t in model.data["tensor_set"]}

        processor = _CloneProcessor()
        processor.apply(model)

        # Each tensor in the set should be a new clone (different identity)
        for elem in model.data["tensor_set"]:
            assert isinstance(elem, torch.Tensor)
            assert id(elem) not in original_tensors

        # Non-tensor value should be unchanged
        assert model.data["plain"] == "value"

    def test_deeply_nested_dict_no_infinite_recursion(self):
        """Deeply nested dicts (5 levels) should process correctly without infinite recursion."""

        inner = {"tensor": torch.randn(2, 2)}
        nested: dict = inner
        for _ in range(4):
            nested = {"child": nested}

        class ModelWithDeepNesting(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.deep = nested

        model = ModelWithDeepNesting()
        original_tensor = inner["tensor"]

        processor = _CloneProcessor()
        processor.apply(model)

        # Navigate to the deepest tensor
        node = model.deep
        for _ in range(4):
            node = node["child"]
        deep_tensor = node["tensor"]

        # Tensor at the bottom should have been cloned
        assert deep_tensor is not original_tensor
        assert torch.equal(deep_tensor, original_tensor)

    def test_set_inside_dict_inside_dict(self):
        """Three-level nesting: dict -> dict -> set with tensors."""

        class ModelWithTripleNesting(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.config = {
                    "group": {
                        "tensors": {torch.randn(2, 2)},
                    },
                }

        model = ModelWithTripleNesting()
        original_elem = next(iter(model.config["group"]["tensors"]))

        processor = _CloneProcessor()
        processor.apply(model)

        new_elem = next(iter(model.config["group"]["tensors"]))
        assert new_elem is not original_elem
        assert torch.equal(new_elem, original_elem)

    def test_dispatch_and_apply_on_use_same_handler_chain(self):
        """Both _apply_on and dispatch must use the identical handler chain.

        This verifies they cannot drift out of sync.
        """

        class _SpyHandler:
            """Handler that records the handler list identity it's part of."""

            def can_handle(self, value):
                return False  # Never matches; we just want to observe

            def process_attribute(self, value, ctx):
                return value

        processor = _CloneProcessor()
        spy = _SpyHandler()
        processor.register_type_handler(spy)

        # Capture the handler chains used by each path
        apply_on_chain = processor.get_type_handlers()

        ctx = ProcessingContext(processor)
        dispatch_chain = ctx._processor.get_type_handlers()

        # Both should be equal (same handlers in same order)
        assert len(apply_on_chain) == len(dispatch_chain)
        for h_apply, h_pv in zip(apply_on_chain, dispatch_chain, strict=False):
            assert h_apply is h_pv, f"Handler mismatch: _apply_on uses {h_apply!r} but dispatch uses {h_pv!r}"

    def test_custom_handler_reached_inside_nested_dict(self):
        """A custom handler should be dispatched for values inside nested dicts.

        This is the most direct test of the review claim: if dispatch()
        skipped built-in handlers, the inner dict would not be recursed into,
        and the custom handler would never see the value inside it.
        """

        custom_handled = []

        @dataclasses.dataclass
        class _MarkerObj:
            """Non-tensor, non-dict, non-set object that needs custom handling."""

            tag: str

        class _MarkerHandler:
            def can_handle(self, value):
                return isinstance(value, _MarkerObj)

            def process_attribute(self, value, ctx):
                custom_handled.append(value.tag)
                return value

        class ModelWithNestedCustom(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.data = {
                    "level1": {
                        "marker": _MarkerObj("deep"),
                    },
                }

        model = ModelWithNestedCustom()
        processor = _IdentityProcessor()
        processor.register_type_handler(_MarkerHandler())
        processor.apply(model)

        # If dispatch() skipped built-in dict handler, "deep" would never appear
        assert "deep" in custom_handled, (
            "Custom handler was not dispatched for object inside nested dict — "
            "dispatch() may not be checking built-in handlers"
        )

    def test_nn_parameter_inside_nested_dict_preserves_type(self):
        """nn.Parameter inside a nested dict should be preserved through dispatch()."""

        class ModelWithNestedParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.extra = {
                    "nested": {
                        "param": nn.Parameter(torch.randn(4, 4)),
                    },
                }

        model = ModelWithNestedParam()
        original_param = model.extra["nested"]["param"]
        original_id = id(original_param)

        processor = _CloneProcessor()
        processor.apply(model)

        result = model.extra["nested"]["param"]
        # Parameter type and identity should be preserved (in-place .data update)
        assert isinstance(result, nn.Parameter), f"Expected nn.Parameter, got {type(result).__name__}"
        assert id(result) == original_id, "Parameter identity should be preserved"

    def test_empty_nested_containers_no_crash(self):
        """Empty nested dicts and sets should be handled gracefully."""

        class ModelWithEmptyNesting(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.data = {
                    "empty_dict": {},
                    "empty_set_in_dict": {"s": set()},
                    "dict_in_dict": {"inner": {}},
                }

        model = ModelWithEmptyNesting()
        processor = _CloneProcessor()
        # Should not raise any errors
        processor.apply(model)

        assert model.data["empty_dict"] == {}
        assert model.data["empty_set_in_dict"]["s"] == set()
        assert model.data["dict_in_dict"]["inner"] == {}


class TestDocstringExamples:
    """Tests for the examples shown in ProcessingContext docstrings.

    These ensure the documented examples actually work as advertised.
    """

    def setup_method(self):
        TensorProcessor.clear_global_type_handlers()

    def teardown_method(self):
        TensorProcessor.clear_global_type_handlers()

    # --- ctx.process() example: TensorCacheHandler ---

    def test_process_example_tensor_cache_handler(self):
        """The TensorCacheHandler docstring example processes plain tensors without Parameter wrapping."""

        @dataclasses.dataclass
        class TensorCache:
            keys: torch.Tensor
            values: torch.Tensor

        class TensorCacheHandler:
            def can_handle(self, value):
                return isinstance(value, TensorCache)

            def process_attribute(self, value, ctx):
                value.keys = ctx.process(value.keys)
                value.values = ctx.process(value.values)
                return value

        keys = torch.randn(4, 4)
        values = torch.randn(4, 4)
        cache = TensorCache(keys, values)

        class ModelWithCache(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.cache = None

        model = ModelWithCache()
        model.cache = cache

        processor = _CloneProcessor()
        processor.register_type_handler(TensorCacheHandler())
        processor.apply(model)

        assert isinstance(model.cache, TensorCache), "Container type should be preserved"
        assert model.cache.keys is not keys, "Keys should be a new tensor object (cloned)"
        assert torch.equal(model.cache.keys, keys), "Cloned data should match original"

    # --- ctx.process_and_preserve() example: simple handler ---

    def test_process_and_preserve_example_simple_handler(self):
        """The simple process_and_preserve docstring example preserves nn.Parameter."""

        class MyTensorHandler:
            def can_handle(self, value):
                return isinstance(value, torch.Tensor)

            def process_attribute(self, value, ctx):
                return ctx.process_and_preserve(value)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 3))

        model = SimpleModel()
        original_id = id(model.weight)

        processor = _CloneProcessor()
        processor.register_type_handler(MyTensorHandler())
        processor.apply(model)

        assert isinstance(model.weight, nn.Parameter), "Parameter type should be preserved"
        assert id(model.weight) == original_id, "Parameter identity should be preserved (in-place update)"

    # --- ctx.process_and_preserve(parameter_factory=...) example ---

    def test_process_and_preserve_example_with_factory(self):
        """The factory docstring example reconstructs a Parameter subclass via custom factory."""

        class CustomParam(nn.Parameter):
            def __new__(cls, data, requires_grad=True, partitions=None):
                instance = super().__new__(cls, data, requires_grad=requires_grad)
                instance.partitions = partitions or {}
                return instance

        class CustomParamHandler:
            def can_handle(self, value):
                return isinstance(value, CustomParam)

            def process_attribute(self, value, ctx):
                return ctx.process_and_preserve(value, parameter_factory=self._create)

            def _create(self, original, new_data):
                return CustomParam(new_data, requires_grad=original.requires_grad, partitions=original.partitions)

        class ModelWithCustomParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = CustomParam(torch.randn(3, 3), partitions={"a": 1, "b": 2})

        model = ModelWithCustomParam()

        processor = _CloneProcessor(force_update_nn_parameters=True)
        processor.register_type_handler(CustomParamHandler())
        processor.apply(model)

        assert isinstance(model.weight, CustomParam), "Subclass type should be preserved"
        assert model.weight.partitions == {"a": 1, "b": 2}, "Custom attributes should be preserved"

    # --- ctx.dispatch() example: NamedTupleHandler ---

    def test_dispatch_example_named_tuple_handler(self):
        """The NamedTupleHandler docstring example dispatches each field through the handler chain."""
        from collections import namedtuple

        LayerCache = namedtuple("LayerCache", ["key_cache", "value_cache"])

        class NamedTupleHandler:
            def can_handle(self, value):
                return isinstance(value, tuple) and hasattr(value, "_fields")

            def process_attribute(self, value, ctx):
                processed = {field: ctx.dispatch(getattr(value, field)) for field in value._fields}
                return type(value)(**processed)

        key = torch.randn(4, 4)
        val = torch.randn(4, 4)
        cache = LayerCache(key_cache=key, value_cache=val)

        class ModelWithNamedTuple(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.cache = None

        model = ModelWithNamedTuple()
        model.cache = cache

        processor = _CloneProcessor()
        processor.register_type_handler(NamedTupleHandler())
        processor.apply(model)

        assert isinstance(model.cache, LayerCache), "NamedTuple type should be preserved"
        assert model.cache.key_cache is not key, "Tensors should be new objects (cloned)"
        assert torch.equal(model.cache.key_cache, key), "Cloned data should match original"

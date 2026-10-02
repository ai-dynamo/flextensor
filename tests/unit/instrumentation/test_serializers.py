# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for instrumentation serializers."""

import json
import types

import torch
from pydantic import BaseModel

from flextensor.instrumentation.serializers import CIRCULAR_REF_MARKER, serialize_args, serialize_value


class SamplePydanticModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class SampleClass:
    """Sample class for testing object serialization."""

    def __init__(self, x: int, y: str):
        self.x = x
        self.y = y
        self._private = "hidden"


class CircularRefA:
    """Class that can have circular references for testing."""

    def __init__(self, name: str):
        self.name = name
        self.ref = None  # Will be set to create circular reference


class CircularRefB:
    """Class that references back to CircularRefA."""

    def __init__(self, value: int):
        self.value = value
        self.back_ref = None  # Will point back to CircularRefA


class TestSerializeValue:
    """Tests for serialize_value function."""

    def test_serialize_none(self):
        """Test serializing None."""
        assert serialize_value(None) is None

    def test_serialize_primitives(self):
        """Test serializing primitive types."""
        assert serialize_value(True) is True
        assert serialize_value(False) is False
        assert serialize_value(42) == 42
        assert serialize_value(3.14) == 3.14
        assert serialize_value("hello") == "hello"

    def test_serialize_torch_device(self):
        """Test serializing torch.device."""
        device = torch.device("cuda:0")
        result = serialize_value(device)
        assert result == "cuda:0"

        device_cpu = torch.device("cpu")
        result_cpu = serialize_value(device_cpu)
        assert result_cpu == "cpu"

    def test_serialize_torch_dtype(self):
        """Test serializing torch.dtype."""
        result = serialize_value(torch.float32)
        assert "float32" in result

        result_half = serialize_value(torch.float16)
        assert "float16" in result_half

    def test_serialize_pydantic_model(self):
        """Test serializing Pydantic BaseModel."""
        model = SamplePydanticModel(name="test", value=42)
        result = serialize_value(model)

        assert result["_type"] == "SamplePydanticModel"
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_serialize_list(self):
        """Test serializing lists."""
        result = serialize_value([1, 2, 3])
        assert result == [1, 2, 3]

        # Nested list with torch device
        device = torch.device("cpu")
        result_nested = serialize_value([1, device, "test"])
        assert result_nested == [1, "cpu", "test"]

    def test_serialize_tuple(self):
        """Test serializing tuples."""
        result = serialize_value((1, 2, 3))
        assert result == [1, 2, 3]  # Converted to list

    def test_serialize_dict(self):
        """Test serializing dictionaries."""
        data = {"a": 1, "b": "test", "c": None}
        result = serialize_value(data)
        assert result == {"a": 1, "b": "test", "c": None}

    def test_serialize_nested_dict(self):
        """Test serializing nested dictionaries."""
        device = torch.device("cuda:0")
        data = {"device": device, "config": {"value": 42}}
        result = serialize_value(data)

        assert result["device"] == "cuda:0"
        assert result["config"]["value"] == 42

    def test_serialize_set(self):
        """Test serializing sets (converted to sorted list)."""
        result = serialize_value({3, 1, 2})
        assert result == [1, 2, 3]  # Sorted

    def test_serialize_callable(self):
        """Test serializing callables."""

        def sample_func():
            pass

        result = serialize_value(sample_func)
        assert "sample_func" in result

    def test_serialize_type(self):
        """Test serializing type objects."""
        result = serialize_value(SampleClass)
        assert "SampleClass" in result

    def test_serialize_custom_object(self):
        """Test serializing custom objects with __dict__."""
        obj = SampleClass(10, "hello")
        result = serialize_value(obj)

        assert result["_type"] == "SampleClass"
        assert result["x"] == 10
        assert result["y"] == "hello"
        # Private attributes should not be included
        assert "_private" not in result

    def test_serialize_tensor_basic(self):
        """Test serializing a basic torch.Tensor."""
        tensor = torch.zeros(3, 4, dtype=torch.float32)
        result = serialize_value(tensor)

        assert result["_type"] == "torch.Tensor"
        assert result["shape"] == [3, 4]
        assert "float32" in result["dtype"]
        assert result["device"] == "cpu"
        assert result["requires_grad"] is False
        assert result["numel"] == 12
        assert result["element_size"] == 4
        assert result["nbytes"] == 48  # 3 * 4 * 4 bytes (float32)
        assert result["is_contiguous"] is True
        assert "strided" in result["layout"]

    def test_serialize_tensor_with_grad(self):
        """Test serializing a tensor with requires_grad=True."""
        tensor = torch.randn(2, 2, requires_grad=True)
        result = serialize_value(tensor)

        assert result["_type"] == "torch.Tensor"
        assert result["requires_grad"] is True

    def test_serialize_tensor_non_contiguous(self):
        """Test serializing a non-contiguous tensor."""
        tensor = torch.zeros(4, 4)
        non_contig = tensor.t()  # Transpose creates non-contiguous tensor
        result = serialize_value(non_contig)

        assert result["_type"] == "torch.Tensor"
        assert result["is_contiguous"] is False

    def test_serialize_tensor_different_dtypes(self):
        """Test serializing tensors with different dtypes."""
        tensor_int = torch.zeros(2, dtype=torch.int64)
        result_int = serialize_value(tensor_int)
        assert "int64" in result_int["dtype"]

        tensor_half = torch.zeros(2, dtype=torch.float16)
        result_half = serialize_value(tensor_half)
        assert "float16" in result_half["dtype"]

        tensor_bool = torch.zeros(2, dtype=torch.bool)
        result_bool = serialize_value(tensor_bool)
        assert "bool" in result_bool["dtype"]

    def test_serialize_tensor_in_list(self):
        """Test serializing a list containing tensors."""
        tensor = torch.ones(2, 3)
        result = serialize_value([1, tensor, "test"])

        assert result[0] == 1
        assert result[1]["_type"] == "torch.Tensor"
        assert result[1]["shape"] == [2, 3]
        assert result[2] == "test"

    def test_serialize_tensor_in_dict(self):
        """Test serializing a dict containing tensors."""
        tensor = torch.zeros(5)
        result = serialize_value({"weight": tensor, "name": "layer1"})

        assert result["weight"]["_type"] == "torch.Tensor"
        assert result["weight"]["shape"] == [5]
        assert result["weight"]["numel"] == 5
        assert result["name"] == "layer1"

    def test_serialize_mappingproxy(self):
        """MappingProxyType is not a dict subclass; must still serialize as a mapping.

        Regression for #141: the serializer fell back to ``repr(value)`` and emitted
        ``"mappingproxy({..: Parameter containing: tensor(...)})"`` strings — non-JSON,
        rejected by downstream JSON consumers (JET's ES flattened-field indexer hit
        this first, but any `json.loads` round-trip on the payload would break).
        """
        param = torch.nn.Parameter(torch.tensor([-4.3030e-03, -1.7822e-02], dtype=torch.bfloat16))
        frozen = types.MappingProxyType({"layer.0.weight": param, "n_blocks": 8})

        result = serialize_value(frozen)

        assert isinstance(result, dict)
        assert result["layer.0.weight"]["_type"] == "torch.Tensor"
        assert result["layer.0.weight"]["dtype"] == "torch.bfloat16"
        assert result["n_blocks"] == 8

        payload = json.dumps(result)
        assert "mappingproxy" not in payload
        assert "Parameter containing" not in payload
        assert "tensor(" not in payload

    def test_serialize_args_with_mappingproxy(self):
        """End-to-end: serialize_args sanitises frozen ``{id(tensor): Parameter}`` maps.

        Mirrors the real-world shape ``TensorManager.tensors_map`` takes at
        ``tensor_manager.py:435`` (``self.tensors_map = types.MappingProxyType(...)``)
        and is then passed to downstream ``@instrumentable`` components via their
        ``__init__`` kwargs.
        """
        param = torch.nn.Parameter(torch.randn(4, dtype=torch.bfloat16))
        tensors_map = types.MappingProxyType({id(param): param})

        result = serialize_args({"tensors_map": tensors_map})

        payload = json.dumps(result)
        assert "mappingproxy" not in payload
        assert "Parameter containing" not in payload
        assert result["tensors_map"][str(id(param))]["_type"] == "torch.Tensor"

    def test_serialize_mappingproxy_with_none_values(self):
        """MappingProxyType values can be None — matches ``_parameters.items()`` shape.

        See tensor_processors.py: ``_parameters.items()`` yields ``None`` for unset
        parameters, which is why the codebase deliberately avoids ``named_parameters()``.
        The serializer must not choke on that.
        """
        frozen = types.MappingProxyType({"weight": None, "count": 5})

        result = serialize_value(frozen)

        assert result == {"weight": None, "count": 5}

    def test_serialize_nested_mappingproxy(self):
        """Nested MappingProxyType inside a dict and a list must both recurse, not repr."""
        outer = {
            "frozen": types.MappingProxyType({"inner": 1}),
            "also": [types.MappingProxyType({"leaf": 2})],
        }

        result = serialize_value(outer)

        assert result == {"frozen": {"inner": 1}, "also": [{"leaf": 2}]}

    def test_serialize_unknown_type_returns_sentinel(self, caplog):
        """Unknown non-JSON types must serialize as a discoverable sentinel + warning.

        The pre-existing ``repr(value)`` fallback is the silent-failure shape that
        originally produced FT #141 — a raw repr string masquerading as valid data.
        Replace with a structured ``_type: "_unserialized"`` envelope and a logged
        warning so future unknown types are observable in logs AND indexable downstream.
        """
        import logging

        def gen() -> object:
            yield 1

        generator = gen()

        with caplog.at_level(logging.WARNING, logger="flextensor.instrumentation.serializers"):
            result = serialize_value(generator)

        assert isinstance(result, dict)
        assert result["_type"] == "_unserialized"
        assert "generator" in result["_class"]
        assert "generator object" in result["_repr"]
        # A warning must be logged so ops can spot the fallback in CI/production.
        assert any("_unserialized" in r.message or "no handler" in r.message for r in caplog.records)

    def test_serialize_unknown_type_with_broken_repr(self):
        """The sentinel fallback must tolerate objects whose ``__repr__`` raises.

        Partially-initialised objects (common during ML model construction) often
        have ``__repr__`` methods that depend on attrs set later in ``__init__``.
        The sentinel branch is meant to *be* the graceful-degradation path — it
        must not itself raise when the value's ``__repr__`` is broken.

        Uses ``__slots__ = ()`` so the instance has no ``__dict__`` and falls
        through the object-introspection branch into the fallback.
        """

        class Broken:
            __slots__ = ()

            def __repr__(self) -> str:
                raise RuntimeError("repr blew up")

        result = serialize_value(Broken())

        assert isinstance(result, dict)
        assert result["_type"] == "_unserialized"
        assert "Broken" in result["_class"]
        assert "repr failed" in result["_repr"]
        assert "RuntimeError" in result["_repr"]

    def test_serialize_cyclic_mappingproxy(self):
        """Cycles through MappingProxyType must emit ``<circular reference>``, not recurse.

        Pre-fix, MappingProxyType hit the ``repr(value)`` fallback which handled cycles
        accidentally. Post-fix the Mapping branch recurses via ``_serialize_dict``, so
        dict/MappingProxyType cycle tracking is required to avoid ``RecursionError``.
        """
        inner: dict[str, object] = {}
        frozen = types.MappingProxyType(inner)
        inner["self"] = frozen

        # Must not raise RecursionError.
        result = serialize_value(frozen)

        assert result == {"self": CIRCULAR_REF_MARKER}

    def test_serialize_shared_mapping_is_not_a_cycle(self):
        """Same Mapping instance reused as siblings must serialize both times, not collapse.

        Cycle detection keyed on ``id(value)`` must only match *on the ancestor path*,
        not on every ancestor-ever-visited. Sibling-reuse (``[proxy, proxy]``,
        ``{"a": proxy, "b": proxy}``) has no cycle — both occurrences must render in full.
        A seen-set that's never pruned conflates "seen on the current path" with
        "seen anywhere" and wrongly stamps the second sibling ``<circular reference>``.
        """
        shared = types.MappingProxyType({"k": 1})
        result = serialize_value([shared, shared])
        assert result == [{"k": 1}, {"k": 1}]

    def test_serialize_shared_dict_is_not_a_cycle(self):
        """Plain dicts also must not collapse on sibling reuse.

        The widening to ``Mapping`` routed plain ``dict`` through the same cycle-tracked
        helper as ``MappingProxyType``. Before the fix, ``dict`` sibling-reuse worked by
        accident (no cycle tracking). Post-fix, same invariant must hold.
        """
        shared = {"k": 1}
        result = serialize_value({"a": shared, "b": shared})
        assert result == {"a": {"k": 1}, "b": {"k": 1}}

    def test_serialize_shared_object_is_not_a_cycle(self):
        """Same invariant for attribute-introspected objects."""

        class Holder:
            def __init__(self) -> None:
                self.v = 1

        shared = Holder()
        result = serialize_value([shared, shared])
        # Both entries must render; neither is a circular reference.
        assert len(result) == 2
        assert result[0] == result[1]
        assert result[0].get("v") == 1
        assert CIRCULAR_REF_MARKER not in result


class TestSerializeArgs:
    """Tests for serialize_args function."""

    def test_serialize_args_basic(self):
        """Test serializing a dictionary of arguments."""
        args = {
            "device": torch.device("cuda:0"),
            "batch_size": 32,
            "name": "test_model",
        }
        result = serialize_args(args)

        assert result["device"] == "cuda:0"
        assert result["batch_size"] == 32
        assert result["name"] == "test_model"

    def test_serialize_args_complex(self):
        """Test serializing complex nested arguments."""
        model = SamplePydanticModel(name="nested", value=100)
        args = {
            "model": model,
            "options": {"flag": True, "count": 5},
            "values": [1, 2, 3],
        }
        result = serialize_args(args)

        assert result["model"]["_type"] == "SamplePydanticModel"
        assert result["options"]["flag"] is True
        assert result["values"] == [1, 2, 3]

    def test_serialize_args_with_tensor(self):
        """Test serializing arguments containing tensors."""
        tensor = torch.randn(10, 20)
        args = {
            "weight": tensor,
            "bias": torch.zeros(20),
            "name": "linear",
        }
        result = serialize_args(args)

        assert result["weight"]["_type"] == "torch.Tensor"
        assert result["weight"]["shape"] == [10, 20]
        assert result["bias"]["_type"] == "torch.Tensor"
        assert result["bias"]["shape"] == [20]
        assert result["name"] == "linear"


class TestCircularReferences:
    """Tests for handling circular references during serialization."""

    def test_serialize_self_referencing_list(self) -> None:
        """A list's back-reference becomes the same marker used for objects."""
        values: list[object] = ["value"]
        values.append(values)

        result = serialize_args({"values": values})

        assert json.loads(json.dumps(result)) == {"values": ["value", CIRCULAR_REF_MARKER]}

    def test_serialize_tuple_list_cycle(self) -> None:
        """Tuples participate in ancestor tracking when a list points back to them."""
        values: list[object] = []
        outer = (values,)
        values.append(outer)

        assert serialize_value(outer) == [[CIRCULAR_REF_MARKER]]

    def test_serialize_list_mapping_cycle(self) -> None:
        """A mixed cycle stops at the first repeated collection on the path."""
        values: list[object] = []
        values.append({"owner": values})

        assert serialize_value(values) == [{"owner": CIRCULAR_REF_MARKER}]

    def test_serialize_shared_list_is_not_a_cycle(self) -> None:
        """Reusing one list as siblings still serializes each occurrence in full."""
        shared = [1, [2]]

        assert serialize_value([shared, shared]) == [[1, [2]], [1, [2]]]

    def test_serialize_self_referencing_object(self):
        """Test serializing an object that references itself."""
        obj = CircularRefA("self_ref")
        obj.ref = obj  # Create self-reference

        # Should not raise RecursionError
        result = serialize_value(obj)

        assert result["_type"] == "CircularRefA"
        assert result["name"] == "self_ref"
        assert result["ref"] == "<circular reference>"

    def test_serialize_mutual_circular_reference(self):
        """Test serializing objects with mutual circular references (A -> B -> A)."""
        obj_a = CircularRefA("object_a")
        obj_b = CircularRefB(42)
        obj_a.ref = obj_b
        obj_b.back_ref = obj_a  # Create circular reference

        # Should not raise RecursionError
        result = serialize_value(obj_a)

        assert result["_type"] == "CircularRefA"
        assert result["name"] == "object_a"
        assert result["ref"]["_type"] == "CircularRefB"
        assert result["ref"]["value"] == 42
        assert result["ref"]["back_ref"] == "<circular reference>"

    def test_serialize_circular_reference_in_list(self):
        """Test serializing a list containing objects with circular references."""
        obj = CircularRefA("in_list")
        obj.ref = obj  # Self-reference

        # Should not raise RecursionError
        result = serialize_value([obj, "other"])

        assert result[0]["_type"] == "CircularRefA"
        assert result[0]["ref"] == "<circular reference>"
        assert result[1] == "other"

    def test_serialize_circular_reference_in_dict(self):
        """Test serializing a dict containing objects with circular references."""
        obj = CircularRefA("in_dict")
        obj.ref = obj  # Self-reference

        # Should not raise RecursionError
        result = serialize_value({"obj": obj, "key": "value"})

        assert result["obj"]["_type"] == "CircularRefA"
        assert result["obj"]["ref"] == "<circular reference>"
        assert result["key"] == "value"

    def test_serialize_deep_circular_reference(self):
        """Test serializing objects with deep circular references (A -> B -> C -> A)."""

        class CircularRefC:
            def __init__(self):
                self.next = None

        obj_a = CircularRefA("deep_a")
        obj_b = CircularRefB(100)
        obj_c = CircularRefC()

        obj_a.ref = obj_b
        obj_b.back_ref = obj_c
        obj_c.next = obj_a  # Complete the cycle

        # Should not raise RecursionError
        result = serialize_value(obj_a)

        assert result["_type"] == "CircularRefA"
        assert result["ref"]["_type"] == "CircularRefB"
        assert result["ref"]["back_ref"]["_type"] == "CircularRefC"
        assert result["ref"]["back_ref"]["next"] == "<circular reference>"

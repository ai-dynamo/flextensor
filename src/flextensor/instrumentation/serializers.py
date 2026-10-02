# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON serializers for complex types used in FlexTensor components.

This module provides serialization helpers for types that are not natively
JSON-serializable, such as torch.device, Pydantic models, and strategy classes.
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

# Sentinel value for circular references
CIRCULAR_REF_MARKER = "<circular reference>"

# Envelope for values that no serializer branch handled. A logged warning accompanies
# every emission so unknown types are observable in ops pipelines rather than silently
# bleeding a raw repr() into downstream JSON consumers (the FT #141 silent-failure shape).
UNSERIALIZED_TYPE = "_unserialized"


def _serialize_pydantic_model(value: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic BaseModel."""
    return {
        "_type": value.__class__.__name__,
        "_module": value.__class__.__module__,
        **value.model_dump(mode="json"),
    }


def _serialize_collection(value: list | tuple | set, seen: set[int]) -> list[Any] | str:
    """Serialize a collection, tracking only the current ancestor path."""
    obj_id = id(value)
    if obj_id in seen:
        return CIRCULAR_REF_MARKER
    seen.add(obj_id)
    try:
        if isinstance(value, set):
            return [_serialize_value_impl(item, seen) for item in sorted(value, key=lambda x: str(x))]
        return [_serialize_value_impl(item, seen) for item in value]
    finally:
        seen.discard(obj_id)


def _serialize_callable(value: Any) -> str:
    """Serialize a callable (function, method, or class)."""
    if hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def _serialize_object(value: Any, seen: set[int]) -> dict[str, Any]:
    """Serialize an object with __dict__ (e.g., strategy classes)."""
    class_info: dict[str, Any] = {
        "_type": value.__class__.__name__,
        "_module": value.__class__.__module__,
    }
    # Serialize public attributes
    for attr_name, attr_value in vars(value).items():
        if not attr_name.startswith("_"):
            class_info[attr_name] = _serialize_value_impl(attr_value, seen)
    return class_info


def _serialize_dict(value: Mapping[Any, Any], seen: set[int]) -> dict[str, Any]:
    """Serialize any Mapping (dict, MappingProxyType, OrderedDict, etc.)."""
    return {str(k): _serialize_value_impl(v, seen) for k, v in value.items()}


def _serialize_object_with_cycle_check(value: Any, seen: set[int]) -> Any:
    """Serialize an object, checking for circular references first.

    ``seen`` tracks the current *ancestor path*, not every id ever visited — so a
    shared value reused as siblings (``[x, x]``) is not mis-labelled as a cycle.
    """
    obj_id = id(value)
    if obj_id in seen:
        return CIRCULAR_REF_MARKER
    seen.add(obj_id)
    try:
        return _serialize_object(value, seen)
    finally:
        seen.discard(obj_id)


def _serialize_dict_with_cycle_check(value: Mapping[Any, Any], seen: set[int]) -> Any:
    """Serialize a Mapping, checking for circular references first.

    Needed because Mappings (dict, MappingProxyType, …) can form direct cycles
    (``d['self'] = d``) that would otherwise recurse forever. ``seen`` tracks the
    current ancestor path only — pruned on return so sibling reuse (``[m, m]``)
    isn't conflated with a true cycle.
    """
    obj_id = id(value)
    if obj_id in seen:
        return CIRCULAR_REF_MARKER
    seen.add(obj_id)
    try:
        return _serialize_dict(value, seen)
    finally:
        seen.discard(obj_id)


def _serialize_tensor(value: torch.Tensor) -> dict[str, Any]:
    """Serialize a torch.Tensor to metadata dict."""
    return {
        "_type": "torch.Tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": value.requires_grad,
        "numel": value.numel(),
        "element_size": value.element_size(),
        "nbytes": value.nbytes,
        "is_contiguous": value.is_contiguous(),
        "layout": str(value.layout),
    }


def _serialize_unknown(value: Any) -> dict[str, str]:
    """Emit the ``_unserialized`` envelope + warning for values no branch handled.

    ``repr()`` is called inside a guard because a broken ``__repr__`` (common on
    partially-initialised objects) must not itself break the fallback path — the
    whole point of this branch is graceful degradation.
    """
    cls = type(value)
    class_path = f"{cls.__module__}.{cls.__qualname__}"
    LOGGER.warning("serialize_value: no handler for %s; emitting %s envelope", class_path, UNSERIALIZED_TYPE)
    try:
        repr_str = repr(value)
    except Exception as exc:
        repr_str = f"<repr failed: {type(exc).__name__}>"
    return {"_type": UNSERIALIZED_TYPE, "_class": class_path, "_repr": repr_str}


def _serialize_value_impl(value: Any, seen: set[int]) -> Any:
    """Internal implementation of serialize_value with circular reference tracking.

    Args:
        value: The value to serialize.
        seen: Set of object IDs already visited (for circular reference detection).

    Returns:
        JSON-serializable representation of the value.
    """
    # Handle None and primitives
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Handle torch types
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)

    # Handle torch.Tensor
    if isinstance(value, torch.Tensor):
        return _serialize_tensor(value)

    # Handle Pydantic models
    if isinstance(value, BaseModel):
        return _serialize_pydantic_model(value)

    # Handle collections
    if isinstance(value, (list, tuple, set)):
        return _serialize_collection(value, seen)

    # Handle dictionaries and other Mappings (e.g. MappingProxyType, OrderedDict).
    # MappingProxyType is not a dict subclass, so we widen to the Mapping ABC here —
    # otherwise frozen maps like TensorManager.tensors_map fall through to repr() and
    # emit non-JSON "mappingproxy(Parameter containing: tensor(...))" strings (FT #141).
    if isinstance(value, Mapping):
        return _serialize_dict_with_cycle_check(value, seen)

    # Handle type objects (classes)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"

    # Handle callables (functions, methods)
    if callable(value):
        return _serialize_callable(value)

    # Handle objects with __dict__ (e.g., strategy classes)
    if hasattr(value, "__class__") and hasattr(value, "__dict__"):
        return _serialize_object_with_cycle_check(value, seen)

    # Fallback: emit a structured sentinel + log a warning. A bare repr() would be
    # JSON-valid-looking but semantically opaque and often malformed (see FT #141).
    return _serialize_unknown(value)


def serialize_value(value: Any) -> Any:
    """Serialize a value to a JSON-compatible representation.

    Handles special types:
    - torch.device: Serialized as string (e.g., "cuda:0")
    - torch.dtype: Serialized as string (e.g., "torch.float32")
    - torch.Tensor: Serialized as metadata dict (shape, dtype, device, etc.)
    - Pydantic BaseModel: Serialized via model_dump()
    - Classes with __dict__: Serialized as class name + attributes
    - Callables: Serialized as qualified name or repr
    - Circular references: Serialized as "<circular reference>"

    Args:
        value: The value to serialize.

    Returns:
        JSON-serializable representation of the value.
    """
    return _serialize_value_impl(value, set())


def serialize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Serialize a dictionary of initialization arguments.

    Args:
        args: Dictionary mapping argument names to values.

    Returns:
        Dictionary with all values serialized to JSON-compatible types.
    """
    return {key: serialize_value(value) for key, value in args.items()}

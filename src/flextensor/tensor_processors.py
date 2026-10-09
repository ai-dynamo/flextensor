# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import collections
import inspect
import logging
import statistics
import types
from collections.abc import Callable
from typing import Any, ClassVar, Protocol, TypedDict, cast, runtime_checkable

import torch

from flextensor.collectors import TensorStatistics
from flextensor.compiler_utils import is_torch_compiled_module
from flextensor.host_pinning import HostPinner
from flextensor.instrumentation import instrumentable
from flextensor.utils import calculate_tensor_size

LOGGER = logging.getLogger(__name__)
_UNMAPPED_GPU_MOVE_SAFETY_BYTES = 16 * 1024**2

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


ParameterFactory = Callable[[torch.nn.Parameter, torch.Tensor], torch.nn.Parameter]
"""Callable that creates a new ``nn.Parameter`` (or subclass) from the original and new data.

Used by :func:`preserve_parameter_type` and
:meth:`ProcessingContext.process_and_preserve` to let custom
:class:`TypeHandler` implementations control how ``nn.Parameter`` subclasses
with exotic constructors are reconstructed during ``force_update``.

The callable receives:

- **original** — the original ``nn.Parameter`` (or subclass) before processing.
- **new_data** — the processed tensor data to wrap.

It must return a new ``nn.Parameter`` (or subclass) instance.
"""


def preserve_parameter_type(
    original_value: object,
    new_value: object,
    force_update: bool = False,
    parameter_factory: ParameterFactory | None = None,
) -> object:
    """Preserve ``nn.Parameter`` wrapper when tensor processing strips it.

    Operations like ``tensor.to()`` or ``tensor.pin_memory()`` return a plain
    ``torch.Tensor``, stripping the ``nn.Parameter`` wrapper.

    By default the existing Parameter's ``.data`` is updated in-place,
    preserving object identity — critical for id()-based tracking in
    ``tensors_map``, ``traced_tensors``, and processor caches.

    When *force_update* is ``True``, a new ``nn.Parameter`` is created instead.
    Use this when building an independent copy of a model where identity should
    intentionally differ (e.g. creating a separate profile model).

    When *force_update* is ``True`` and *parameter_factory* is provided, the
    factory is used to construct the new parameter instead of the default
    ``type(original)(new_data, requires_grad=...)``.  This allows
    ``nn.Parameter`` subclasses with incompatible constructors to be
    reconstructed correctly.

    Args:
        original_value: The original value before processing.
        new_value: The processed value.
        force_update: If True, create a new Parameter instead of updating in-place.
        parameter_factory: Optional callable ``(original, new_data) -> nn.Parameter``
            used to construct the new parameter when *force_update* is True.
            Ignored when *force_update* is False.

    Returns:
        The value with ``Parameter`` wrapper preserved if applicable.
    """
    if (
        isinstance(original_value, torch.nn.Parameter)
        and isinstance(new_value, torch.Tensor)
        and not isinstance(new_value, torch.nn.Parameter)
    ):
        if force_update:
            if parameter_factory is not None:
                return parameter_factory(original_value, new_value)
            param_type = type(original_value)
            return param_type(new_value, requires_grad=original_value.requires_grad)
        original_value.data = new_value
        return original_value
    return new_value


# ---------------------------------------------------------------------------
# Type handler system
# ---------------------------------------------------------------------------


@runtime_checkable
class TypeHandler(Protocol):
    """Protocol for custom type handlers in TensorProcessor.

    Implement this to add support for custom attribute types during tensor
    processing.  Handlers are checked in priority order: instance-level custom
    handlers first, then global handlers, then built-in handlers.  The first
    handler whose ``can_handle()`` returns ``True`` processes the attribute.

    Example:
        >>> class SharedWeightHandler:
        ...     def can_handle(self, value: object) -> bool:
        ...         return hasattr(value, 'partitions') and hasattr(value, 'local_tensors')
        ...
        ...     def process_attribute(self, value: object, ctx: ProcessingContext) -> object:
        ...         for key, param in value.partitions.items():
        ...             new_data = ctx.process(param.data)
        ...             value.partitions[key] = torch.nn.Parameter(data=new_data, requires_grad=False)
        ...         value.local_tensors = {ctx.process(t) for t in value.local_tensors}
        ...         return value
        ...
        >>> TensorProcessor.register_global_type_handler(SharedWeightHandler())
    """

    def can_handle(self, value: Any) -> bool:
        """Return True if this handler should process the given attribute value.

        Args:
            value: An attribute value from a module's ``__dict__``.

        Returns:
            True if this handler can process this value type.
        """
        ...

    def process_attribute(self, value: Any, ctx: "ProcessingContext") -> Any:
        """Process the attribute value and return the result.

        Args:
            value: The attribute value to process (type guaranteed by ``can_handle``).
            ctx: Processing context with utility methods for tensor processing.

        Returns:
            The processed value to set on the model attribute.
        """
        ...


class ProcessingContext:
    """Context provided to :meth:`TypeHandler.process_attribute`.

    Wraps the owning :class:`TensorProcessor` and exposes convenient helpers
    so that handler implementations never need to depend on processor internals.
    """

    def __init__(self, processor: "TensorProcessor") -> None:
        self._processor = processor

    def process(self, value: object) -> object:
        """Process a value through the processor's :meth:`~TensorProcessor.process` method.

        Applies the raw tensor transformation (e.g. ``.to(device)``,
        ``.pin_memory()``) **without** ``nn.Parameter`` re-wrapping.  Use this
        when you manage parameter construction yourself.

        For the common case where Parameter preservation is needed, use
        :meth:`process_and_preserve` instead.

        Args:
            value: The value to process.

        Returns:
            The processed value.

        Example:
            A handler for a metadata container that holds plain tensors
            (not ``nn.Parameter``) where re-wrapping is not needed::

                class TensorCacheHandler:
                    def can_handle(self, value):
                        return isinstance(value, TensorCache)

                    def process_attribute(self, value, ctx):
                        value.keys = ctx.process(value.keys)
                        value.values = ctx.process(value.values)
                        return value
        """
        return self._processor.process(value)

    def process_and_preserve(
        self,
        value: object,
        parameter_factory: ParameterFactory | None = None,
    ) -> object:
        """Process a value and preserve ``nn.Parameter`` type.

        Convenience method combining :meth:`process` and
        :func:`preserve_parameter_type`.  This is the most common operation
        for tensor attributes and tensor values inside dicts.

        Args:
            value: The value to process (typically ``torch.Tensor`` or ``nn.Parameter``).
            parameter_factory: Optional callable ``(original, new_data) -> nn.Parameter``
                used to construct the new parameter when ``force_update`` is enabled
                on the processor.  Allows ``nn.Parameter`` subclasses with incompatible
                constructors to be reconstructed correctly.  Ignored when the processor
                updates parameters in-place.

        Returns:
            The processed value with ``Parameter`` wrapper preserved.

        Examples:
            Most handlers simply call this without a factory — Parameter
            preservation is handled automatically::

                class MyTensorHandler:
                    def can_handle(self, value):
                        return isinstance(value, torch.Tensor)

                    def process_attribute(self, value, ctx):
                        return ctx.process_and_preserve(value)

            A handler for an ``nn.Parameter`` subclass with an exotic
            constructor can pass a factory to control reconstruction::

                class SharedWeightHandler:
                    def can_handle(self, value):
                        return hasattr(value, "partitions")

                    def process_attribute(self, value, ctx):
                        return ctx.process_and_preserve(
                            value, parameter_factory=self._create
                        )

                    def _create(self, original, new_data):
                        return type(original)(
                            new_data, partitions=original.partitions
                        )
        """
        new_value = self._processor.process(value)
        return preserve_parameter_type(value, new_value, self._processor.force_update_nn_parameters, parameter_factory)

    def dispatch(self, value: object) -> object:
        """Process a value through the full handler chain.

        Dispatches through instance-level, global, and built-in type handlers
        (via :meth:`TensorProcessor.get_type_handlers`).  Values without a
        matching handler are returned unchanged.

        Use this inside container handlers (dict, set) to process their
        elements with full handler support, including recursion into nested
        containers.

        Args:
            value: The value to process.

        Returns:
            The processed value.

        Example:
            A handler for a custom container that holds mixed values —
            some may be tensors, others may be dicts or custom types with
            their own handlers::

                class NamedTupleHandler:
                    def can_handle(self, value):
                        return isinstance(value, tuple) and hasattr(value, "_fields")

                    def process_attribute(self, value, ctx):
                        processed = {
                            field: ctx.dispatch(getattr(value, field))
                            for field in value._fields
                        }
                        return type(value)(**processed)
        """
        for handler in self._processor.get_type_handlers():
            if handler.can_handle(value):
                return handler.process_attribute(value, self)
        return value


# --- Built-in type handlers ------------------------------------------------


class TensorTypeHandler:
    """Built-in handler for ``torch.Tensor`` attributes (including ``nn.Parameter``)."""

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, torch.Tensor)

    def process_attribute(self, value: torch.Tensor, ctx: ProcessingContext) -> object:
        return ctx.process_and_preserve(value)


class _ReachableContainerTypeHandler:
    """Read-only traversal that skips values which cannot contain tensors.

    A container already being traversed higher up the stack (a dict or list that contains itself) is returned
    as-is instead of recursing forever.
    """

    _LEAF_TYPES = (types.NoneType, bool, int, float, complex, str, bytes)

    def __init__(self) -> None:
        self._in_progress: set[int] = set()

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, (dict, set)) or type(value) in (list, tuple)

    def process_attribute(self, value: dict | list | set | tuple, ctx: ProcessingContext) -> object:
        value_id = id(value)
        if value_id in self._in_progress:
            return value
        self._in_progress.add(value_id)
        try:
            values = value.values() if isinstance(value, dict) else value
            for item in values:
                if type(item) not in self._LEAF_TYPES:
                    ctx.dispatch(item)
        finally:
            self._in_progress.discard(value_id)
        return value


class _ObjectAttributesTypeHandler:
    """Read-only traversal of tensor-bearing attributes on plain objects.

    Python module objects are namespaces, not model state, and are skipped: an engine object holding a module
    reference (e.g. SGLang's ``GroupCoordinator.device_module`` set to ``torch.cuda``) would otherwise lead the
    walk into ``sys.modules`` and every loaded module.
    """

    def __init__(self) -> None:
        self._visited: set[int] = set()

    def reset(self) -> None:
        """Forget objects visited during the previous traversal."""
        self._visited.clear()

    def can_handle(self, value: Any) -> bool:
        # CPython type metadata avoids slow static reflection on every dispatched value.
        return (
            not isinstance(value, (type, types.ModuleType, torch.Tensor, torch.nn.Module))
            and getattr(type(value), "__dictoffset__", 0) != 0
        )

    def process_attribute(self, value: object, ctx: ProcessingContext) -> object:
        value_id = id(value)
        if value_id in self._visited:
            return value
        self._visited.add(value_id)
        for field in vars(value).values():
            ctx.dispatch(field)
        return value


class SetTypeHandler:
    """Built-in handler for non-empty ``set`` attributes."""

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, set) and len(value) > 0

    def process_attribute(self, value: set[Any], ctx: ProcessingContext) -> set[Any]:
        return {ctx.dispatch(v) for v in value}


class LegacySetTypeHandler:
    """Set handler matching pre-ADR-0003 behaviour: ``process()`` only, no Parameter preservation.

    The default built-in ``SetTypeHandler`` processes set elements through the
    full handler dispatch chain (``ctx.dispatch``), which also applies
    ``nn.Parameter`` preservation.  The original hardcoded set logic called
    ``self.process(value)`` directly — no handler dispatch, no Parameter
    re-wrapping.

    Register this handler on a processor instance if the improved default
    causes issues with a specific model:

    Example:
        >>> processor = MoveToGPUTensorProcessor(device)
        >>> processor.register_type_handler(LegacySetTypeHandler())
    """

    def can_handle(self, value: Any) -> bool:
        """Return True for non-empty sets."""
        return isinstance(value, set) and len(value) > 0

    def process_attribute(self, value: set[Any], ctx: ProcessingContext) -> set[Any]:
        """Process each element via ``ctx.process()`` — no handler dispatch or Parameter preservation."""
        return {ctx.process(v) for v in value}


class DictTypeHandler:
    """Built-in handler for ``dict`` attributes containing tensor values.

    Preserves ``OrderedDict`` type.  When no values change, preserves the
    original dict object (maintaining immutable dict types like ``FrozenDict``).
    """

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, dict)

    def process_attribute(self, value: dict[Any, Any], ctx: ProcessingContext) -> dict[Any, Any]:
        new_dict = collections.OrderedDict() if isinstance(value, collections.OrderedDict) else {}
        any_changed = False
        for key, v in value.items():
            new_v = ctx.dispatch(v)
            new_dict[key] = new_v
            if new_v is not v:
                any_changed = True
        return new_dict if any_changed else value


def _unwrap_compiled_module(module: object) -> object:
    """Return the real module behind a ``torch.compile`` ``OptimizedModule``.

    ``OptimizedModule.__setattr__`` forwards every attribute *write* whose name
    is not in its small exempt set -- including ``_parameters``, ``_buffers`` and
    ``_modules`` -- to its wrapped ``_orig_mod``. A :class:`TensorProcessor`'s
    ``_apply_on`` loop ends with ``setattr(module, key, value)`` for each
    ``__dict__`` entry, so running it on the wrapper writes the wrapper's *empty*
    ``_parameters``/``_buffers`` onto the real module, silently dropping its
    direct ``nn.Parameter`` attributes (e.g. a transformer block's
    ``scale_shift_table``). This bites whenever a module is compiled *before*
    being (re-)processed -- notably the compiled-offload re-plan, which
    re-prepares the model after per-block ``torch.compile``. Always process the
    underlying module instead; a no-op for uncompiled modules.

    Broken wrappers (missing ``_orig_mod``, or a cycle) raise rather than being
    processed unsafely.
    """
    while is_torch_compiled_module(module):
        modules = getattr(module, "_modules", None)
        if not isinstance(modules, dict) or "_orig_mod" not in modules:
            raise RuntimeError(
                "FlexTensor: torch.compile OptimizedModule is missing _orig_mod; refusing to process the wrapper."
            )
        module = modules["_orig_mod"]
    return module


class TensorProcessor:
    _global_type_handlers: ClassVar[list[TypeHandler]] = []
    _DEFAULT_TYPE_HANDLERS: ClassVar[list[TypeHandler]] = [
        TensorTypeHandler(),
        SetTypeHandler(),
        DictTypeHandler(),
    ]

    def __init__(
        self,
        update_attributes: bool = True,
        recursive: bool = True,
        process_dict: bool = True,
        force_update_nn_parameters: bool = False,
        type_handlers: list[TypeHandler] | None = None,
    ):
        self.update_attributes = update_attributes
        self.recursive = recursive
        self.process_dict = process_dict
        self.force_update_nn_parameters = force_update_nn_parameters
        self._custom_type_handlers: list[TypeHandler] = list(type_handlers or [])
        # Cycle/dup guard: OptimizedModule._orig_mod re-exposes the original
        # model as a child, producing self-references during recursion.  Also
        # initialised here so direct ``_process_module`` callers (subclasses
        # / tests) don't hit AttributeError before ``apply()`` runs.
        self._visited: set[int] = set()
        self._skipped_visited_modules: list[tuple[str, int]] = []

    def apply(self, model: object) -> None:
        # Reset between applications so each call starts with a clean slate.
        self._visited = set()
        self._skipped_visited_modules = []
        if isinstance(model, dict):
            self._process_dict(model, "model")
        else:
            self._process_module(model, "model")
        self._log_skipped_visited_modules()
        self.cleanup()

    def process(self, _src: object) -> Any:
        pass

    def cleanup(self) -> None:
        pass

    @classmethod
    def register_global_type_handler(cls, handler: TypeHandler) -> None:
        """Register a type handler that applies to **all** TensorProcessor instances.

        Global handlers are checked after instance-level handlers but before
        built-in handlers.  Later registrations take higher priority (checked
        first).

        Args:
            handler: The type handler to register.
        """
        cls._global_type_handlers.insert(0, handler)

    @classmethod
    def clear_global_type_handlers(cls) -> None:
        """Remove all globally registered type handlers."""
        cls._global_type_handlers.clear()

    def register_type_handler(self, handler: TypeHandler) -> None:
        """Register a custom type handler for this processor instance.

        Instance handlers are checked first, before global and built-in
        handlers.  Later registrations take higher priority (checked first).

        Args:
            handler: The type handler to register.
        """
        self._custom_type_handlers.insert(0, handler)

    def get_type_handlers(self) -> list[TypeHandler]:
        """Get the full handler chain: instance → global → built-in."""
        return [*self._custom_type_handlers, *self._global_type_handlers, *self._DEFAULT_TYPE_HANDLERS]

    def _record_skipped_visited_module(self, module: torch.nn.Module, name: str) -> None:
        self._skipped_visited_modules.append((name, id(module)))

    def _mark_module_visited(self, module: torch.nn.Module, name: str) -> bool:
        # Identity-keyed so weight-shared modules reachable via two parents are processed once.
        module_id = id(module)
        if module_id in self._visited:
            self._record_skipped_visited_module(module, name)
            return False
        self._visited.add(module_id)
        return True

    def _log_skipped_visited_modules(self) -> None:
        if not self._skipped_visited_modules:
            return

        skipped_counts = collections.Counter(self._skipped_visited_modules)
        total_skips = sum(skipped_counts.values())
        details = ", ".join(
            f"{name} (id={module_id}, skipped={count})" for (name, module_id), count in skipped_counts.items()
        )
        LOGGER.debug(
            "%s: skipped %d already-visited module visit(s): %s",
            self.__class__.__name__,
            total_skips,
            details,
        )

    def _process_inner_fields(self, src: object, new_tensor: object) -> None:
        ref_tensor_fields = set(dir(src))
        # Use class-level attributes to detect instance-specific fields
        # This ensures we find custom attributes like weight.scale even when new_tensor = src
        new_tensor_fields = set(dir(new_tensor.__class__))
        missing_fields_set = ref_tensor_fields - new_tensor_fields
        missing_fields = [name for name in missing_fields_set if not name.startswith("_")]
        for missing_field in missing_fields:
            field = getattr(src, missing_field)
            if isinstance(field, torch.Tensor):
                new_tensor_field = self.process(field)
                setattr(new_tensor, missing_field, new_tensor_field)
            else:
                setattr(new_tensor, missing_field, field)

    def _process_dict(self, dict_model: dict[object, object], _name: str) -> None:
        if not self.process_dict:
            return
        updated_attributes = {}
        for attr_name, attr_value in dict_model.items():
            updated_attributes[attr_name] = self.process(attr_value)
        if self.update_attributes:
            for attr_name, attr_value in updated_attributes.items():
                dict_model[attr_name] = attr_value

    def _process_module(self, module: Any, _name: str) -> Any:
        # torch.compile wraps modules in an OptimizedModule that forwards
        # attribute writes to its _orig_mod; processing the wrapper would wipe
        # the real module's _parameters/_buffers (see _unwrap_compiled_module).
        module = _unwrap_compiled_module(module)
        if not self._mark_module_visited(module, _name):
            return module

        # Extend the current module
        self._apply_on(module)

        # Recursively process all child modules
        if self.recursive:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Module):
                    # Replace the child module with its extended version
                    self._apply_on(self._process_module(child_module, child_name))

    def _apply_on(self, model: object) -> None:
        updated_attributes = {}
        if hasattr(model, "__dict__"):
            ctx = ProcessingContext(self)
            for attr_name, attr_value in model.__dict__.items():
                new_value = ctx.dispatch(attr_value)
                # Reassigning unchanged values can register deliberately hidden nn.Modules.
                if new_value is not attr_value:
                    updated_attributes[attr_name] = new_value

        if self.update_attributes:
            for key, value in updated_attributes.items():
                setattr(model, key, value)


@instrumentable
class MoveToPinMemoryTensorProcessor(TensorProcessor):
    """Pin CPU tensors via the configured :class:`HostPinner`.

    Dispatches to ``tensor.pin_memory()`` (torch mode) or ``cudaHostRegister``
    (host_register mode) based on the pinner's mode.

    The cache is keyed by ``id(src)`` and skips ``pin`` on repeats of the same
    Python object. Distinct objects that share storage (e.g. a tensor and its
    view) miss the cache; in host_register mode the second ``pin`` call finds
    the storage pointer already in the registry and returns without calling
    ``cudaHostRegister`` again.
    """

    def __init__(self, host_pinner: HostPinner) -> None:
        super().__init__()
        self.host_pinner = host_pinner
        self.cache: dict[int, torch.Tensor] = {}

    def cleanup(self) -> None:
        self.cache = {}

    def process(self, src: object) -> object:
        if not isinstance(src, torch.Tensor) or src.is_meta:
            return src
        if src.device != torch.device("cpu"):
            return src
        src_tensor_id = id(src)
        if src_tensor_id in self.cache:
            pinned_tensor = self.cache[src_tensor_id]
        else:
            pinned_tensor = self.host_pinner.pin(src)
            self.cache[src_tensor_id] = pinned_tensor
        self._process_inner_fields(src, pinned_tensor)
        return pinned_tensor


@instrumentable
class DisableRequiresGradTensorProcessor(TensorProcessor):
    def __init__(self) -> None:
        super().__init__(update_attributes=False)

    def process(self, src: Any) -> Any:
        if isinstance(src, torch.Tensor) and src.is_leaf:
            src.requires_grad_(False)
        return src


@instrumentable
class MoveToGPUTensorProcessor(TensorProcessor):
    def __init__(
        self,
        device_gpu: torch.device,
        process_inner_fields: bool = True,
        recursive: bool = True,
        process_dict: bool = True,
    ) -> None:
        super().__init__(recursive=recursive, process_dict=process_dict)
        self.device_gpu = device_gpu
        self.cache: dict[int, torch.Tensor] = {}
        self.process_inner_fields = process_inner_fields

    def cleanup(self) -> None:
        self.cache = {}

    def process(self, src: object) -> object:
        if not isinstance(src, torch.Tensor) or src.is_meta:
            return src
        src_tensor_id = id(src)
        if src_tensor_id in self.cache:
            new_tensor = self.cache[src_tensor_id]
        else:
            new_tensor = src.to(device=self.device_gpu)
            self.cache[src_tensor_id] = new_tensor
        if self.process_inner_fields:
            self._process_inner_fields(src, new_tensor)

        return new_tensor


@instrumentable
class MoveUnmappedTensorsToGPUProcessor(TensorProcessor):
    def __init__(self, device_gpu: torch.device, tensor_id_mapping: dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.tensor_id_mapping = tensor_id_mapping
        self.device_gpu = torch.device(device_gpu)
        self.move_to_gpu = MoveToGPUTensorProcessor(device_gpu, process_inner_fields=False)
        self.cache: dict[int, torch.Tensor] = {}
        self.unmapped_gpu_bytes = 0

    def apply(self, model: object) -> None:
        self.unmapped_gpu_bytes = 0  # Reset before processing to ensure fresh count on reuse
        super().apply(model)

    def cleanup(self) -> None:
        self.cache = {}
        self.move_to_gpu.cleanup()

    def _guard_unmapped_cuda_move(self, src: torch.Tensor) -> None:
        if self.device_gpu.type != "cuda":
            return
        if src.device.type == "cuda" and src.device == self.device_gpu:
            return

        tensor_bytes = src.numel() * src.element_size()
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_gpu)
        required_bytes = tensor_bytes + _UNMAPPED_GPU_MOVE_SAFETY_BYTES
        if free_bytes >= required_bytes:
            return

        raise RuntimeError(
            "Insufficient GPU memory for unmapped tensor move: "
            f"tensor={tensor_bytes} bytes, free={free_bytes} bytes, "
            f"total={total_bytes} bytes, safety_margin={_UNMAPPED_GPU_MOVE_SAFETY_BYTES} bytes, "
            f"device={self.device_gpu}. "
            "FlexTensor cannot finalize this model without exceeding available CUDA memory."
        )

    def process(self, src: object) -> object:
        if not isinstance(src, torch.Tensor) or src.is_meta:
            return src
        src_tensor_id = id(src)
        if src_tensor_id in self.cache:
            return self.cache[src_tensor_id]

        # Determine the new tensor: use mapped version if available, otherwise move to GPU
        if src_tensor_id in self.tensor_id_mapping:
            new_tensor = self.tensor_id_mapping[src_tensor_id]
        else:
            self._guard_unmapped_cuda_move(src)
            new_tensor = cast("torch.Tensor", self.move_to_gpu.process(src))
            # Track GPU memory used by unmapped tensors
            if new_tensor.device.type == "cuda":
                self.unmapped_gpu_bytes += new_tensor.numel() * new_tensor.element_size()
        self.cache[src_tensor_id] = new_tensor
        self._process_inner_fields(src, new_tensor)

        return new_tensor


@instrumentable
class MoveBuffersToGPUTensorProcessor(TensorProcessor):
    def __init__(self, device_gpu: torch.device) -> None:
        # Buffers only exist in modules, not in dictionaries, so process_dict=False
        super().__init__(update_attributes=False, process_dict=False)
        self.device_gpu = device_gpu
        self.move_to_gpu = MoveToGPUTensorProcessor(
            device_gpu, process_inner_fields=True, recursive=False, process_dict=False
        )

    def cleanup(self) -> None:
        self.move_to_gpu.cleanup()

    def _process_module(self, module: Any, _name: str) -> Any:
        module = _unwrap_compiled_module(module)
        if not self._mark_module_visited(module, _name):
            return module

        # Move all buffers in this module to GPU
        for buffer_name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            new_buffer = self.move_to_gpu.process(buffer)
            module.register_buffer(buffer_name, cast("torch.Tensor", new_buffer))

        # Recursively process child modules
        if self.recursive:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Module):
                    self._process_module(child_module, child_name)

        return module


@instrumentable
class TensorReplacementProcessor(TensorProcessor):
    """Replace mapped tensors, preserving meta Parameter subclasses.

    Meta parameters are rebound without rerunning subclass constructors, with
    their instance attributes preserved. ``parameter_factory`` can override
    this for subclasses requiring custom reconstruction.
    """

    def __init__(
        self,
        tensor_id_mapping: dict[int, torch.Tensor],
        update_nn_parameters: bool = True,
        *,
        parameter_factory: ParameterFactory | None = None,
    ) -> None:
        super().__init__()
        self.tensor_id_mapping = tensor_id_mapping
        self.update_nn_parameters = update_nn_parameters
        self.parameter_factory = parameter_factory
        self.cache: dict[int, torch.Tensor] = {}

    def cleanup(self) -> None:
        self.cache = {}

    def process(self, src: object) -> object:
        if not isinstance(src, torch.Tensor):
            return src
        src_tensor_id = id(src)
        if src_tensor_id in self.cache:
            return self.cache[src_tensor_id]
        if src_tensor_id not in self.tensor_id_mapping:
            if not src.is_meta:
                self._process_inner_fields(src, src)
            return src
        new_tensor = self.tensor_id_mapping[src_tensor_id]
        if src.is_meta:
            if isinstance(src, torch.nn.parameter.Parameter):
                # Unwrap a mapped Parameter so the source subtype and grad flag win.
                new_data = new_tensor.detach() if isinstance(new_tensor, torch.nn.Parameter) else new_tensor
                if self.parameter_factory is not None:
                    new_tensor = self.parameter_factory(src, new_data)
                else:
                    # Constructors such as vLLM's require metadata beyond data.
                    new_tensor = torch.nn.Parameter.__new__(type(src), new_data, requires_grad=src.requires_grad)
                    new_tensor.__dict__.update(src.__dict__)
        elif isinstance(src, torch.nn.parameter.Parameter):
            if self.update_nn_parameters:
                src.data = new_tensor
                new_tensor = src
            else:
                new_tensor = torch.nn.parameter.Parameter(data=new_tensor, requires_grad=src.requires_grad)

        self._process_inner_fields(src, new_tensor)

        self.cache[src_tensor_id] = new_tensor
        return new_tensor


@instrumentable
class TensorMappingProcessor(TensorProcessor):
    def __init__(
        self,
    ) -> None:
        super().__init__(update_attributes=False)
        self.tensors_map: dict[int, torch.Tensor] = {}

    def get_results(self) -> dict[int, torch.Tensor]:
        """
        Get tensor mapping

        Returns:
            - tensors_map: Mapping of tensor IDs to tensors
        """
        return self.tensors_map

    def map_inner_fields(self, src: object) -> None:
        ref_tensor_fields = set(dir(src))
        new_tensor_fields = set(dir(torch.Tensor))
        missing_fields_set = ref_tensor_fields - new_tensor_fields
        missing_fields = [name for name in missing_fields_set if not name.startswith("_")]
        for missing_field in missing_fields:
            field = getattr(src, missing_field)
            if isinstance(field, torch.Tensor):
                self.process(field)

    def process(self, src: object) -> object:
        # Only process tensor results, skip non-tensor returns
        # Skip processing meta tensors (they have no data to transfer)
        if not isinstance(src, torch.Tensor) or src.is_meta:
            return src

        if src.device.type != "cpu":
            return src

        tensor_id = id(src)
        self.tensors_map[tensor_id] = src
        self.map_inner_fields(src)
        return src


class BenchmarkResults(TypedDict):
    """Return type for :meth:`BenchmarkTensorProcessor.get_results`."""

    tensor_statistics_map: dict[int, TensorStatistics]
    tensors_map: dict[int, torch.Tensor]


@instrumentable
class ReachableTensorMapProcessor(TensorProcessor):
    """Collect every tensor reachable from a model, keyed by ``id(tensor)``.

    Read-only walk over an ``nn.Module`` (parameters, buffers, attributes,
    submodules) or an ``items()``-based model container, plus tensor inner
    fields (e.g. the ``scale`` attribute on a quantised ``nn.Parameter``). Skips meta
    tensors — their ``id()`` doesn't correspond to a real allocation, so
    they aren't useful as a narrowing input.

    Adds list, tuple, and plain-object attribute traversal to the standard
    :class:`TensorProcessor` walk. The convenience wrapper
    :func:`compute_reachable_tensor_map` is the typical entry point for callers
    that need the tensor objects.
    """

    def __init__(self) -> None:
        self._object_attributes_handler = _ObjectAttributesTypeHandler()
        super().__init__(
            update_attributes=False,
            type_handlers=[_ReachableContainerTypeHandler(), self._object_attributes_handler],
        )
        self.reachable_tensors: dict[int, torch.Tensor] = {}

    def apply(self, model: object) -> None:
        self._object_attributes_handler.reset()
        # ``ModelDict`` is intentionally not imported here: tensor_manager
        # imports this module, so support the wrapper structurally via items().
        self._visited = set()
        self._skipped_visited_modules = []
        if _has_items(model) and not isinstance(model, torch.nn.Module):
            self._process_items_container(model)
        else:
            self._process_module(model, "model")
        self._log_skipped_visited_modules()
        self.cleanup()

    def _process_items_container(self, model: object) -> None:
        for _key, value in model.items():  # type: ignore[attr-defined]
            self.process(value)

    def map_inner_fields(self, src: torch.Tensor) -> None:
        self.process(getattr(src, "_base", None))
        ref_tensor_fields = set(dir(src))
        new_tensor_fields = set(dir(torch.Tensor))
        ctx = ProcessingContext(self)
        for missing_field in ref_tensor_fields - new_tensor_fields:
            if not missing_field.startswith("_"):
                ctx.dispatch(getattr(src, missing_field))

    def process(self, src: object) -> object:
        if not isinstance(src, torch.Tensor) or src.is_meta:
            return src
        tensor_id = id(src)
        if tensor_id in self.reachable_tensors:
            return src
        self.reachable_tensors[tensor_id] = src
        self.map_inner_fields(src)
        return src

    def get_results(self) -> dict[int, torch.Tensor]:
        return self.reachable_tensors


def compute_reachable_tensor_map(model: object | None) -> dict[int, torch.Tensor]:
    """Convenience wrapper around :class:`ReachableTensorMapProcessor`.

    Returns the tensors reachable from ``model``, keyed by ``id(tensor)``.
    ``None`` and unsupported model containers yield an empty mapping.
    """
    if model is None or not _is_reachable_tensor_container(model):
        return {}
    collector = ReachableTensorMapProcessor()
    collector.apply(model)
    return collector.get_results()


def compute_reachable_tensor_ids(model: object | None) -> set[int]:
    """Return IDs for tensors reachable from ``model``.

    ``None`` and unsupported model containers yield an empty set so callers
    can use this as a narrowing input to
    :class:`~flextensor.loaders.TensorStrategyLoader` without an extra check.
    """
    return set(compute_reachable_tensor_map(model))


def _has_items(model: object) -> bool:
    return callable(inspect.getattr_static(model, "items", None))


def _is_reachable_tensor_container(model: object) -> bool:
    return isinstance(model, torch.nn.Module) or _has_items(model)


@instrumentable
class BenchmarkTensorProcessor(TensorProcessor):
    def __init__(
        self,
        device_gpu: torch.device,
        iterations: int = 10,
    ) -> None:
        super().__init__(update_attributes=False)
        self.iterations = iterations
        self.warmup_iterations = 5
        self.device_gpu = device_gpu
        self.tensor_statistics_map: dict[int, TensorStatistics] = {}
        self.tensors_map: dict[int, torch.Tensor] = {}
        self.cache_results: dict[str, float] = {}

    def get_results(self) -> BenchmarkResults:
        """
        Get benchmark results.

        Returns:
            Dictionary containing benchmark results with keys:
            - tensor_statistics_map: Mapping of tensor IDs to statistics
            - tensors_map: Mapping of tensor IDs to tensors
        """
        return {
            "tensor_statistics_map": self.tensor_statistics_map,
            "tensors_map": self.tensors_map,
        }

    def benchmark_inner_fields(self, src: object) -> None:
        ref_tensor_fields = set(dir(src))
        new_tensor_fields = set(dir(torch.Tensor))
        missing_fields_set = ref_tensor_fields - new_tensor_fields
        missing_fields = [name for name in missing_fields_set if not name.startswith("_")]
        for missing_field in missing_fields:
            field = getattr(src, missing_field)
            if isinstance(field, torch.Tensor):
                self.process(field)

    def process(self, src: object) -> object:
        # Only process tensor results, skip non-tensor returns
        # Skip processing meta tensors (they have no data to transfer)
        if not isinstance(src, torch.Tensor) or src.is_meta:
            return src
        tensor_size = calculate_tensor_size(src)
        key = str(src.size())
        load_time_ms = None
        if key in self.cache_results:
            load_time_ms = self.cache_results[key]

        result_tensor = src

        if result_tensor.device.type != "cpu":
            return src

        if load_time_ms is None:
            transfer_times = []
            for _ in range(self.warmup_iterations):
                result_tensor.to(device=self.device_gpu)
                torch.cuda.synchronize()
            for _ in range(self.iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = result_tensor.to(device=self.device_gpu)
                end_event.record()
                torch.cuda.synchronize()
                duration = start_event.elapsed_time(end_event)
                transfer_times.append(duration)
            load_time_ms = float(statistics.median(transfer_times))
            self.cache_results[key] = load_time_ms

        new_tensor = src
        tensor_id = id(new_tensor)
        self.tensor_statistics_map[tensor_id] = TensorStatistics(
            tensor_id=tensor_id,
            name="",
            size_bytes=tensor_size,
            load_time_ms=load_time_ms,
        )
        self.tensors_map[tensor_id] = new_tensor
        self.benchmark_inner_fields(new_tensor)
        return new_tensor


def preprocess_model(
    model: object,
    tensor_manager: Any,  # TensorManager; Any avoids circular import
    device_gpu: torch.device,
    disable_gradient: bool = True,
    pin_memory: bool = False,
    *,
    host_pinner: HostPinner,
    move_top_level_buffers_to_gpu: bool = True,
) -> None:
    if pin_memory:
        move_to_pinned_memory = MoveToPinMemoryTensorProcessor(host_pinner)
        move_to_pinned_memory.apply(model)
    if disable_gradient:
        disable_grad = DisableRequiresGradTensorProcessor()
        disable_grad.apply(model)
    tensor_mapping = TensorMappingProcessor()
    tensor_mapping.apply(model)
    tensor_manager.tensors_map = tensor_mapping.get_results()
    if move_top_level_buffers_to_gpu:
        move_to_gpu = MoveToGPUTensorProcessor(
            device_gpu, process_inner_fields=True, recursive=False, process_dict=False
        )
        move_to_gpu.apply(model)

    # Move all buffers to GPU across all modules
    move_all_buffers = MoveBuffersToGPUTensorProcessor(device_gpu)
    move_all_buffers.apply(model)

    for tensor_id, _tensor in tensor_manager.tensors_map.items():
        tensor_manager.traced_tensors.add(tensor_id)


# clone_model_with_shared_weights
def create_model_with_shared_tensors(source_model: torch.nn.Module) -> torch.nn.Module:
    """
    Create a model copy with shared tensors.

    This approach builds the new model incrementally without deep copying.
    """
    # Get the model class
    model_class = source_model.__class__

    # Create new instance without calling __init__
    new_model = model_class.__new__(model_class)

    # Copy plain attributes. Parameters/modules are handled by the recursive
    # copy below; plain tensor attributes are operational state and should
    # remain shared with the source model.
    for name, value in source_model.__dict__.items():
        if not isinstance(value, torch.nn.Parameter | torch.nn.Module):
            setattr(new_model, name, value)

    # Initialize the module properly
    torch.nn.Module.__init__(new_model)

    # Copy modules recursively with shared tensors
    _copy_modules_with_shared_tensors(source_model, new_model)

    return new_model


def _copy_modules_with_shared_tensors(source_module: torch.nn.Module, target_module: torch.nn.Module) -> None:
    """Recursively copy modules with shared tensors."""

    # Copy plain attributes, rebinding bound methods to target module.
    # Parameters/modules are handled separately below; unregistered runtime
    # tensor attributes must be preserved.
    # Skip _modules as it should be managed separately (child modules are handled below)

    skipped_attrs = {"_modules", "_parameters", "_buffers"}  # PyTorch internal dicts
    for name, value in source_module.__dict__.items():
        if name in skipped_attrs:
            # These are managed by PyTorch's __setattr__ when we set children/params/buffers
            continue
        if not isinstance(value, torch.nn.Parameter | torch.nn.Module):
            # Rebind bound methods that are bound to the source module
            if isinstance(value, types.MethodType) and value.__self__ is source_module:
                value = types.MethodType(value.__func__, target_module)
            setattr(target_module, name, value)

    # Copy parameters (including None parameters)
    # NOTE: We must use _parameters.items() instead of named_parameters() because
    # named_parameters() skips None values. This is critical for modules like vLLM's
    # QKVParallelLinear that register bias=None via register_parameter("bias", None).
    for name, param in source_module._parameters.items():  # noqa: SLF001
        target_module.register_parameter(name, param)

    # Copy buffers (including None buffers)
    # NOTE: We must use _buffers.items() instead of named_buffers() because
    # named_buffers() skips None values.
    for name, buffer in source_module._buffers.items():  # noqa: SLF001
        target_module.register_buffer(name, buffer)

    # Copy child modules recursively
    for name, child_module in source_module.named_children():
        if isinstance(child_module, torch.nn.Module):
            # Create new child module
            new_child = child_module.__class__.__new__(child_module.__class__)
            torch.nn.Module.__init__(new_child)
            setattr(target_module, name, new_child)

            # Recursively copy with shared tensors
            _copy_modules_with_shared_tensors(child_module, new_child)

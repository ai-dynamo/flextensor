# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

from flextensor.tensor_manager import TensorManager  # noqa: TC001

__all__ = ["load_model_from_profile"]

# ============================================================================
# Utility: Selective Tensor Loading
# ============================================================================


def load_specific_keys(
    file_path: str, keys: list[str], device: str = "cpu", *, strict: bool = False
) -> dict[str, torch.Tensor]:
    """
    Load only specified keys from safetensors file.

    This is memory efficient - only the requested keys are loaded into memory!

    Args:
        file_path: Path to the safetensors file
        keys: List of weight keys to load
        device: Target device for loaded weights (default: "cpu")
        strict: If True, raise KeyError when any requested key is missing
            from the safetensors file. Default False for backward compatibility.

    Returns:
        Dictionary mapping key names to loaded weights

    Raises:
        KeyError: If strict=True and any requested keys are not found in the file
    """
    from safetensors import safe_open

    tensors = {}
    with safe_open(file_path, framework="pt", device=device) as f:
        available_keys = set(f.keys())

        if strict:
            missing_keys = [key for key in keys if key not in available_keys]
            if missing_keys:
                raise KeyError(f"Missing keys in safetensors file '{file_path}': {missing_keys}")

        for key in keys:
            if key in available_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


"""
Intercepting model creation to control parameter/buffer device placement.

This module demonstrates how to automatically place parameters on meta device
and buffers on GPU during model initialization, without modifying model code.

## Problem Statement

When working with large models, you often want to:
1. Keep parameters on meta device (zero memory footprint)
2. Keep buffers on GPU (for fast computation)
3. Preserve nested parameter attributes (e.g., weight.scale)

**Automatic nested parameter tracking (COMPLETE SOLUTION)
- Handles nested parameter attributes (e.g., weight.scale -> scale)
- Creates reference model to discover intended structure
- Automatically restores links after initialization and load_state_dict()
- API: `create_split_device_model_with_tracking()`

## Key Challenge: Nested Parameters

Nested parameters like `weight.scale` are Python attributes that:
- Don't appear in state_dict (only registered parameters do)
- Get lost when creating new Parameter objects (patching breaks identity)
- Need explicit restoration after load_state_dict()

Our solution:
1. Create reference model on meta device (no patching) to discover structure
2. Track which nested attributes point to which registered parameters
3. After patching/loading, restore these links automatically

## Usage Example

```python
# Create model with automatic nested parameter handling
model, tracker = create_split_device_model_with_tracking(
    MyModel,
    model_args=(config,),
    param_device='meta',
    buffer_device='cuda'
)

# Nested links are already restored
assert model.weight.scale is model.scale  # ✓

# After loading state dict, restore links
model.load_state_dict(state_dict, assign=True)
tracker.restore_links(model)
assert model.weight.scale is model.scale  # ✓
```
"""


class ModulePatcher:
    """
    Patches nn.Module methods to control device placement.
    Works with any existing model without modification!

    Warning:
        This class uses global monkey-patching of ``nn.Module.register_buffer`` and
        ``nn.Module.register_parameter``. It is NOT thread-safe: concurrent model
        construction on multiple threads will observe the same patched behavior,
        causing race conditions. Use only in single-threaded contexts.
    """

    _original_register_buffer: Callable[..., Any] | None = None
    _original_register_parameter: Callable[..., Any] | None = None
    _param_device: str | None = None
    _buffer_device: str | None = None

    @classmethod
    def patch(cls, param_device: str = "meta", buffer_device: str = "cuda") -> None:
        """Enable patching with specified devices"""
        cls._param_device = param_device
        cls._buffer_device = buffer_device

        # Save original methods
        if cls._original_register_buffer is None:
            cls._original_register_buffer = nn.Module.register_buffer
            cls._original_register_parameter = nn.Module.register_parameter

        # Patch methods
        nn.Module.register_buffer = cls._patched_register_buffer  # type: ignore[method-assign]
        nn.Module.register_parameter = cls._patched_register_parameter  # type: ignore[method-assign]

    @classmethod
    def unpatch(cls) -> None:
        """Restore original methods"""
        if cls._original_register_buffer:
            nn.Module.register_buffer = cls._original_register_buffer  # type: ignore[method-assign]
            nn.Module.register_parameter = cls._original_register_parameter  # type: ignore[method-assign,assignment]
            cls._original_register_buffer = None
            cls._original_register_parameter = None

    @staticmethod
    def _patched_register_buffer(
        self: nn.Module, name: str, tensor: torch.Tensor | None, persistent: bool = True
    ) -> None:
        """Patched version that moves buffers to buffer device"""
        if tensor is not None and ModulePatcher._buffer_device:  # noqa: SIM102
            if tensor.device.type != ModulePatcher._buffer_device:
                tensor = tensor.to(ModulePatcher._buffer_device)
        original_register_buffer = ModulePatcher._original_register_buffer
        if original_register_buffer is None:
            raise RuntimeError("ModulePatcher original register_buffer callback is missing")
        original_register_buffer(self, name, tensor, persistent=persistent)

    @staticmethod
    def _patched_register_parameter(self: nn.Module, name: str, param: nn.Parameter | None) -> None:
        """Patched version that moves parameters to param device"""
        if param is not None and ModulePatcher._param_device:  # noqa: SIM102
            if param.device.type != ModulePatcher._param_device:
                param = nn.Parameter(
                    param.data.to(ModulePatcher._param_device),
                    requires_grad=param.requires_grad,
                )
        original_register_parameter = ModulePatcher._original_register_parameter
        if original_register_parameter is None:
            raise RuntimeError("ModulePatcher original register_parameter callback is missing")
        original_register_parameter(self, name, param)


def create_split_device_model(
    model_class: type[nn.Module],
    model_args: tuple = (),
    model_kwargs: dict | None = None,
    param_device: str = "meta",
    buffer_device: str = "cuda",
) -> nn.Module:
    """
    Factory function that creates a model with parameters on one device and buffers on another.

    This function temporarily patches ``nn.Module.register_buffer`` and
    ``nn.Module.register_parameter`` to intercept device placement during model
    construction. The patch is applied globally and restored in a try/finally block.

    Warning:
        **Not thread-safe.** This function uses global monkey-patching of nn.Module
        methods. Concurrent model construction on multiple threads will observe the
        same patched behavior, causing race conditions. Only use in single-threaded
        contexts or protect calls with external synchronization.

    Args:
        model_class: The model class to instantiate
        model_args: Positional arguments for model constructor
        model_kwargs: Keyword arguments for model constructor
        param_device: Device for parameters (default: 'meta')
        buffer_device: Device for buffers (default: 'cuda')

    Returns:
        Model instance with split device placement

    Example:
        model = create_split_device_model(
            MyModel,
            model_args=(config,),
            param_device='meta',
            buffer_device='cuda'
        )
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Use monkey patching approach
    ModulePatcher.patch(param_device=param_device, buffer_device=buffer_device)

    try:
        # Create model - will automatically use patched methods
        model = model_class(*model_args, **model_kwargs)
    finally:
        # Always unpatch, even if there's an error
        ModulePatcher.unpatch()

    return model


class NestedParameterTracker:
    """
    Tracks and restores nested parameter attributes after load_state_dict.

    Nested parameters like `layer.weight.scale` are Python attributes that
    don't get saved/restored by PyTorch's state dict. This class tracks them
    during initialization and restores them after loading.

    Supported Pattern:
        This tracker detects nested parameters assigned as attributes directly on
        ``nn.Parameter`` objects that expose a ``__dict__``. The common idiom is::

            self.weight.scale = self.scale = nn.Parameter(...)

        This pattern is used in DeepSeek-V3 model implementation
        and tested in ``tests/unit/test_tensor_inner_fields.py``.

    Limitations:
        - Nested parameters stored only as module attributes (not on the Parameter's
          ``__dict__``) are **not** discovered.
        - Parameter or Tensor subclasses that use ``__slots__`` or otherwise disallow
          arbitrary attribute assignment are **not** supported.
        - Extend this tracker if those cases must be supported.
    """

    def __init__(self) -> None:
        self.nested_links: dict[tuple[str, str], str] = {}  # {(parent_path, attr_name): target_param_name}
        self.tracking = False
        self._root_module: nn.Module | None = None  # Store root module during scanning

    def start_tracking(self, module: nn.Module, verbose: bool = False) -> None:
        """Start tracking nested parameter assignments

        Args:
            module: The module to scan for nested parameters
            verbose: If True, print detailed tracking information
        """
        self.nested_links.clear()
        self.tracking = True
        self._verbose = verbose
        self._root_module = module  # Store root for path resolution
        self._scan_module(module, "")
        self.tracking = False
        self._root_module = None

    def _scan_module(self, module: nn.Module, prefix: str) -> None:
        """Recursively scan module for nested parameters

        Args:
            module: Current module being scanned (changes during recursion)
            prefix: Dotted path from root to current module
        """
        # Check all parameters in this module
        for param_name, param in module._parameters.items():  # noqa: SLF001
            if param is None:
                continue

            param_full_path = f"{prefix}.{param_name}" if prefix else param_name

            # Check the parameter's __dict__ for custom attributes (not inherited methods/properties)
            if hasattr(param, "__dict__"):
                for attr_name, attr_value in param.__dict__.items():
                    if attr_name.startswith("_"):
                        continue

                    if isinstance(attr_value, nn.Parameter):
                        # Found a nested parameter! Find which registered param it is
                        # Get parent module using root module and prefix
                        root_module = cast("nn.Module", self._root_module)
                        parent_module = self._get_module_by_path(root_module, prefix)
                        target_name = self._find_registered_param_relative(
                            root_module,
                            parent_module,
                            prefix,
                            attr_value,
                        )
                        if target_name:
                            self.nested_links[param_full_path, attr_name] = target_name

        # Recurse into child modules
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._scan_module(child_module, child_prefix)

    def _get_module_by_path(self, root_module: nn.Module, path: str) -> nn.Module:
        """Get a submodule by dotted path"""
        if not path:
            return root_module

        module = root_module
        for part in path.split("."):
            module = getattr(module, part)
        return module

    def _find_registered_param_relative(
        self, root_module: nn.Module, parent_module: nn.Module, parent_path: str, param: nn.Parameter
    ) -> str | None:
        """Find the registered name of a parameter, searching relative to parent first

        For a parameter at 'layers.0.attn.wq_a.weight' with nested attribute 'scale',
        first check if 'layers.0.attn.wq_a.scale' exists before checking globally.
        """
        # First, search in the parent module's parameters
        for param_name, p in parent_module.named_parameters():
            if p is param:
                # Construct full path
                if parent_path:
                    return f"{parent_path}.{param_name}"
                return param_name

        # If not found locally, search globally (for shared parameters)
        for name, p in root_module.named_parameters():
            if p is param:
                return name

        return None

    def diagnose_failures(self, module: nn.Module) -> None:
        """Diagnose why nested parameter restoration might fail

        Checks if tracked target parameters actually exist in the model.
        """
        print("\n=== Diagnosing Nested Parameter Tracker ===")  # noqa: T201
        print(f"Total tracked links: {len(self.nested_links)}")  # noqa: T201

        # Get all registered parameters
        all_params = dict(module.named_parameters())
        print(f"Total registered parameters in model: {len(all_params)}")  # noqa: T201

        # Check each tracked link
        missing_targets = []
        for (parent_path, attr_name), target_name in self.nested_links.items():
            if target_name not in all_params:
                missing_targets.append((parent_path, attr_name, target_name))

        if missing_targets:
            print(f"\n❌ Found {len(missing_targets)} links with missing target parameters:")  # noqa: T201
            for parent_path, attr_name, target_name in missing_targets[:10]:  # Show first 10
                print(f"  {parent_path}.{attr_name} -> {target_name} (NOT FOUND)")  # noqa: T201
                # Try to find similar names
                similar = [name for name in all_params if target_name.split(".")[-1] in name][:3]
                if similar:
                    print(f"    Similar parameters: {similar}")  # noqa: T201
        else:
            print("\n✅ All target parameters exist in the model")  # noqa: T201

        print("=" * 50)  # noqa: T201

    def restore_links(self, module: nn.Module, verbose: bool = False) -> None:  # noqa: C901
        """Restore nested parameter links after load_state_dict

        Args:
            module: The module to restore links in
            verbose: If True, print detailed restoration information
        """
        if not self.nested_links:
            return

        restored_count = 0
        failed = []

        for (parent_path, attr_name), target_name in self.nested_links.items():
            try:
                # Get parent parameter
                parent_parts = parent_path.split(".")
                parent = module
                for part in parent_parts:
                    parent = getattr(parent, part)

                # Get target parameter
                target_parts = target_name.split(".")
                target = module
                for i, part in enumerate(target_parts):
                    try:
                        target = getattr(target, part)
                    except AttributeError:
                        if verbose:
                            print(f"  Debug: Failed at part {i} ('{part}') of path '{target_name}'")  # noqa: T201
                            print(f"         Current object type: {type(target).__name__}")  # noqa: T201
                        raise

                # Restore the link using object.__setattr__ to avoid Parameter's restrictions
                object.__setattr__(parent, attr_name, target)
                restored_count += 1
                if verbose:
                    print(f"  Restored: {parent_path}.{attr_name} -> {target_name}")  # noqa: T201
            except (AttributeError, RuntimeError) as e:
                failed.append((parent_path, attr_name, target_name, str(e)))
                if verbose or len(failed) <= 5:  # Show first 5 failures always
                    print(f"  Warning: Failed to restore {parent_path}.{attr_name} -> {target_name}: {e}")  # noqa: T201

        if restored_count > 0:
            print(f"Restored {restored_count} nested parameter link(s)")  # noqa: T201
        if failed and not verbose:
            print(f"Failed to restore {len(failed)} link(s) (set verbose=True for details)")  # noqa: T201


def create_split_device_model_with_tracking(
    model_class: type[nn.Module],
    model_args: tuple = (),
    model_kwargs: dict | None = None,
    param_device: str = "meta",
    buffer_device: str = "cuda",
) -> tuple[nn.Module, NestedParameterTracker]:
    """
    Create a model with parameter tracking for nested parameters.

    Strategy:
    1. Create reference model on meta device to discover intended structure
    2. Track nested parameter links from reference
    3. Create actual split device model with patching
    4. Restore links based on reference mappings

    This ensures nested parameter attributes (like weight.scale) are preserved
    both during initialization and after load_state_dict().

    Warning:
        **Not thread-safe.** Internally calls :func:`create_split_device_model` which
        uses global monkey-patching. See that function's documentation for details.

    Returns:
        Tuple of (model, tracker) where tracker can restore nested links after load_state_dict

    Example:
        >>> model, tracker = create_split_device_model_with_tracking(
        ...     MyModel, model_args=(config,), param_device='meta', buffer_device='cuda'
        ... )
        >>> # Nested links are already restored
        >>> assert model.weight.scale is model.scale
        >>>
        >>> # After loading state dict, restore links again
        >>> model.load_state_dict(state_dict, assign=True)
        >>> tracker.restore_links(model)
        >>> assert model.weight.scale is model.scale
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Step 1: Create reference model on meta device (no patching) to discover structure
    with torch.device("meta"):
        reference_model = model_class(*model_args, **model_kwargs)

    # Step 2: Track nested parameters from reference model
    tracker = NestedParameterTracker()
    tracker.start_tracking(reference_model)

    # Step 3: Create actual split device model with patching
    model = create_split_device_model(model_class, model_args, model_kwargs, param_device, buffer_device)

    # Step 4: Restore links immediately based on reference mappings
    tracker.restore_links(model)

    return model, tracker


# ============================================================================
# High-Level Model Loading API
# ============================================================================


def load_model_from_profile(
    tensor_manager: TensorManager,
    profile_directory: str,
    model_class: type[nn.Module],
    model_args: tuple = (),
    safetensor_path: str | None = None,
    model_kwargs: dict | None = None,
    buffer_device: str | None = None,
    diagnose: bool = False,
    strict: bool = False,
) -> nn.Module:
    """
    Load a model from a saved profile state with optimized weight loading.

    This function creates a model with parameters on meta device and buffers on GPU,
    then loads only the necessary weights from safetensors based on the profile state.
    This is more memory-efficient than loading the full model.

    Args:
        tensor_manager: The TensorManager instance to use for state restoration
        profile_directory: Directory containing the profile (profile.json)
        model_class: The model class to instantiate (e.g., Transformer)
        model_args: Positional arguments for the model constructor
        safetensor_path: Path to the safetensors file. Required.
        model_kwargs: Keyword arguments for the model constructor
        buffer_device: Device for model buffers (default: uses tensor manager's device)
        diagnose: If True, run diagnostics on nested parameter tracking
        strict: If True, raise KeyError when any weight key from the profile
            is missing in the safetensors file. Default False.

    Returns:
        The loaded model in eval mode, ready for use with tensor_manager

    Raises:
        KeyError: If strict=True and any profile weight keys are missing from safetensors

    Example:
        >>> from flextensor import TensorManager
        >>> from flextensor.lazy_model_init import load_model_from_profile
        >>>
        >>> tensor_manager = TensorManager(device_gpu, ...)
        >>> model = load_model_from_profile(
        ...     tensor_manager,
        ...     "./profile",
        ...     Transformer,
        ...     model_args=(model_args,),
        ...     safetensor_path="/path/to/model.safetensors",
        ... )
        >>> tensor_manager.set_model(model)
        >>> model.set_tensor_manager(tensor_manager)
        >>> model = tensor_manager.initialize_warmup()
    """
    if model_kwargs is None:
        model_kwargs = {}

    if safetensor_path is None:
        msg = "safetensor_path is required"
        raise ValueError(msg)

    if buffer_device is None:
        buffer_device = str(tensor_manager.device_gpu)

    # Load state from profile directory
    state = tensor_manager.load_state(profile_directory)

    # For block transfer loaders, we need to load weights selectively
    if tensor_manager.loader_type in ["allocation_block_transfer", "raw_block_transfer"]:
        # Create split model with buffers on GPU and weights on meta (will be replaced later)
        model, tracker = create_split_device_model_with_tracking(
            model_class,
            model_args=model_args,
            model_kwargs=model_kwargs,
            param_device="meta",
            buffer_device=buffer_device,
        )
        model = model.eval()

        # Load only specific weights based on profile state
        tensors_gpu = load_specific_keys(
            str(safetensor_path), state.gpu_tensors_names, device=buffer_device, strict=strict
        )
        view_tensors_cpu = load_specific_keys(
            str(safetensor_path), state.view_tensors_names, device="cpu", strict=strict
        )

        # Diagnose nested parameter tracking if requested
        if diagnose:
            tracker.diagnose_failures(model)

        # Replace model weights with loaded tensors
        model.load_state_dict(tensors_gpu, strict=False, assign=True)
        model.load_state_dict(view_tensors_cpu, strict=False, assign=True)

        # Restore links again after load_state_dict
        tracker.restore_links(model)

    else:
        # For strategy loader, create model normally and load full weights
        from safetensors.torch import load_model

        with torch.device("cpu"):
            model = model_class(*model_args, **model_kwargs)
        model = model.eval()
        load_model(model, safetensor_path)

    # Restore tensor manager state from profile
    tensor_manager.restore_state(model, state)

    return model

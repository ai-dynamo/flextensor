# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled-offload helpers: shared ``pre_compute/post_compute`` boundary scheduling.

The compiled-offload lifecycle (patch, loader install, strategy) lives on
:class:`~flextensor.compile.CompiledOffload`, driven by thin hooks from
:class:`~flextensor.OffloadManager`. At INFERENCE, auto-patched forwards call
:func:`_compiled_trap_enter` / :func:`_compiled_trap_exit` directly (Dynamo
cannot trace trap construction). Manual ``offload_block`` stays on the eager
trap path. With ``offload(..., compile_fn=...)`` and default view-profile,
strategy timings are already compiled during PROFILE; call
:meth:`~flextensor.OffloadManager.request_strategy_replan` only after
direct-profile or external ``torch.compile``. Teardown goes through
:func:`~flextensor.release`.
"""

import functools
from collections.abc import Callable
from typing import Any, no_type_check

import torch


def bump_dynamo_limits_for_compiled_offload(n_units: int) -> None:
    """Raise Dynamo recompile/cache limits to fit ``n_units`` offloaded units.

    Each unit's compiled forward closes over a distinct ``offload_unit_name``, so
    Dynamo specializes per unit and N units need a recompile limit >= N (default is 8).
    Only ever raises limits, never lowers them.
    """
    if n_units <= 0:
        return
    try:
        import torch._dynamo as _dynamo
    except Exception:  # pragma: no cover - torch always present in practice
        return
    needed = n_units * 2 + 16
    for attr in (
        "recompile_limit",
        "cache_size_limit",
        "accumulated_recompile_limit",
        "accumulated_cache_size_limit",
    ):
        if hasattr(_dynamo.config, attr):
            current = getattr(_dynamo.config, attr) or 0
            if needed > current:
                setattr(_dynamo.config, attr, needed)


@no_type_check
def _carrier_tensor_from(value: object) -> torch.Tensor | None:
    """Extract a data-dependency carrier tensor from a forward arg or return."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
        return None
    carrier_attr_names = ("x", "hidden_states", "latent", "inputs_embeds", "sample", "tokens")
    for name in carrier_attr_names:
        attr = getattr(value, name, None)
        if isinstance(attr, torch.Tensor):
            return attr
    return None


@no_type_check
def _resolve_compiled_offload_dep_input(args: tuple, kwargs: dict) -> torch.Tensor | None:
    """Pick an input tensor as the data-dependency carrier for ``pre_compute``."""
    for arg in args:
        carrier = _carrier_tensor_from(arg)
        if carrier is not None:
            return carrier
    for key in ("hidden_states", "input_ids", "x", "inputs_embeds"):
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            return value
    for value in kwargs.values():
        carrier = _carrier_tensor_from(value)
        if carrier is not None:
            return carrier
    return None


@no_type_check
def _resolve_compiled_offload_dep_output(output: object) -> torch.Tensor | None:
    """Pick an output tensor as the data-dependency carrier for ``post_compute``."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            carrier = _carrier_tensor_from(item)
            if carrier is not None:
                return carrier
        return None
    return _carrier_tensor_from(output)


@no_type_check
def _compiled_trap_enter(
    offload_unit_name: str,
    manager_id: int,
    input_carrier: torch.Tensor | None,
    *,
    args: tuple = (),
    kwargs: dict | None = None,
) -> None:
    """Drive ``pre_compute``; used by auto-patched compiled forwards."""
    from flextensor import custom_ops as _ft_custom_ops

    if not _ft_custom_ops.has_active_loader(manager_id):
        if input_carrier is not None and offload_unit_name:
            torch.ops.flextensor.pre_compute(input_carrier, offload_unit_name, manager_id)
        return

    if input_carrier is None:
        _raise_compiled_offload_missing_input(offload_unit_name, args, kwargs or {})
    torch.ops.flextensor.pre_compute(input_carrier, offload_unit_name, manager_id)


@no_type_check
def _loader_exit_for_unit(offload_unit_name: str, manager_id: int) -> None:
    """Best-effort ``loader.exit`` when no output carrier is available."""
    from flextensor import custom_ops as _ft_custom_ops

    loader = _ft_custom_ops.get_active_loader(manager_id)
    if loader is None or not offload_unit_name:
        return
    loader.exit(offload_unit_name)


@no_type_check
def _compiled_trap_exit(
    offload_unit_name: str,
    manager_id: int,
    output_carrier: torch.Tensor | None,
    *,
    output: object = None,
    best_effort: bool = False,
) -> None:
    """Drive ``post_compute``; used by auto-patched compiled forwards."""
    from flextensor import custom_ops as _ft_custom_ops

    if not _ft_custom_ops.has_active_loader(manager_id):
        if output_carrier is not None and offload_unit_name:
            torch.ops.flextensor.post_compute(output_carrier, offload_unit_name, manager_id)
        return

    if output_carrier is None:
        if best_effort:
            _loader_exit_for_unit(offload_unit_name, manager_id)
        else:
            _raise_compiled_offload_missing_output(offload_unit_name, output)
        return
    torch.ops.flextensor.post_compute(output_carrier, offload_unit_name, manager_id)


@no_type_check
def _summarize_carrier_candidate(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, (tuple, list)):
        inner = ", ".join(_summarize_carrier_candidate(item) for item in value)
        return f"{type(value).__name__}[{inner}]"
    return type(value).__name__


@no_type_check
def _raise_compiled_offload_missing_input(offload_unit_name: str, args: tuple, kwargs: dict) -> None:
    observed_args = ", ".join(_summarize_carrier_candidate(arg) for arg in args)
    observed_kwargs = ", ".join(f"{key}={_summarize_carrier_candidate(value)}" for key, value in kwargs.items())
    raise RuntimeError(
        f"FlexTensor compiled-offload: unit {offload_unit_name!r} resolved no input "
        f"carrier tensor while the loader is active — H2D for this unit will not run. "
        f"Expected a tensor in args/kwargs (e.g. hidden_states). "
        f"Observed args=({observed_args}) kwargs=({observed_kwargs}). "
        f"Use eager offload (FT_EXTERNAL_COMPILE=0) or extend "
        f"_resolve_compiled_offload_dep_input for this signature."
    )


@no_type_check
def _raise_compiled_offload_missing_output(offload_unit_name: str, output: object) -> None:
    raise RuntimeError(
        f"FlexTensor compiled-offload: unit {offload_unit_name!r} resolved no output "
        f"carrier tensor while the loader is active — slot release for this unit will not run. "
        f"Expected a tensor return (e.g. hidden_states). "
        f"Observed return: {_summarize_carrier_candidate(output)}. "
        f"Use eager offload (FT_EXTERNAL_COMPILE=0) or extend "
        f"_resolve_compiled_offload_dep_output for this return type."
    )


def build_compiled_offload_forward(
    original_forward_func: Callable[..., Any],
    offload_name: str,
) -> Callable[..., Any]:
    """Build the compile-transparent inference forward installed at INFERENCE.

    Wraps the real forward with ``pre_compute`` / ``post_compute`` so residency
    boundaries stay inside the compiled graph and the unit body can compile as
    one subgraph. Manual ``offload_block`` stays on the eager trap path and must
    not wrap these units. Discovery/profiling keep the default eager
    ``patched_forward``.

    ``offload_name`` is closed over and passed directly to the custom ops as the
    stable offload-unit name.
    """
    from flextensor import custom_ops as _ft_custom_ops
    from flextensor.compiler_utils import patch_inductor_mutation_reads

    patch_inductor_mutation_reads()

    def compiled_offload_forward(self_module: Any, *args: Any, **kwargs: Any) -> Any:
        dep_input = _resolve_compiled_offload_dep_input(args, kwargs)
        manager_id = getattr(self_module, "_ft_manager_id", 0)
        if dep_input is None and _ft_custom_ops.has_active_loader(manager_id):
            _raise_compiled_offload_missing_input(
                offload_name,
                args,
                kwargs,
            )
        entered = dep_input is not None and bool(offload_name)
        _compiled_trap_enter(
            offload_name,
            manager_id,
            dep_input,
        )
        output = None
        try:
            output = original_forward_func(self_module, *args, **kwargs)
            dep_output = _resolve_compiled_offload_dep_output(output)
            if dep_output is None and _ft_custom_ops.has_active_loader(manager_id):
                _raise_compiled_offload_missing_output(
                    offload_name,
                    output,
                )
            return output
        finally:
            if entered:
                dep_output = _resolve_compiled_offload_dep_output(output) if output is not None else None
                _compiled_trap_exit(
                    offload_name,
                    manager_id,
                    dep_output,
                    output=output,
                    best_effort=dep_output is None,
                )

    return functools.wraps(original_forward_func)(compiled_offload_forward)


@no_type_check
def build_profile_compile_forward(
    original_forward_func: Callable[..., Any],
    offload_name: str,
    *,
    get_tensor_manager: Callable[[], Any],
    should_record_duration: Callable[[], bool],
) -> Callable[..., Any]:
    """Build a traceable profile forward for view-mode compiled profiling.

    Inlines ``ProfileBlockController.enter/exit`` and CUDA event timing (same
    window as :class:`~flextensor.trap_tensor_mode.TrapProfileView`) so the
    inner compute can be wrapped with ``compile_fn`` without going through the
    default eager discovery/profile ``patched_forward`` trap wrapper.
    """
    from flextensor.profile_block_controller import ProfileBlockController

    def profile_compiled_forward(self_module, *args, **kwargs):
        tm = get_tensor_manager()
        if tm is None:
            raise RuntimeError("FlexTensor compiled-profile: tensor manager missing during view profile forward.")
        loader = getattr(tm, "tensor_layer_loader", None)
        if not isinstance(loader, ProfileBlockController):
            raise RuntimeError(
                "FlexTensor compiled-profile: expected ProfileBlockController during view profile, "
                f"got {type(loader).__name__}."
            )
        loader.enter(offload_name)
        start_event = tm.trap_start_event
        end_event = tm.trap_end_event
        start_event.record()
        output = None
        try:
            inner = getattr(self_module, "_ft_profile_compiled_inner", None)
            output = original_forward_func(self_module, *args, **kwargs) if inner is None else inner(*args, **kwargs)
        finally:
            end_event.record()
            end_event.synchronize()
            if should_record_duration():
                tm.record_duration(offload_name, start_event.elapsed_time(end_event))
            loader.exit(offload_name)
        return output

    return functools.wraps(original_forward_func)(profile_compiled_forward)

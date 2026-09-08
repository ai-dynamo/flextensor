# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve outermost offload / resident units and build in-place submodule setters."""

from collections.abc import Callable, Iterable

from torch import nn

from flextensor.tensor_discovery import (
    _SUBMODULE_PROBE_EXCEPTIONS,
    get_offload_name,
    has_patched_ancestor,
    is_offload_patched_module,
)


def make_submodule_setter(model: nn.Module, qualified_name: str) -> Callable[[nn.Module], None]:
    """Build a setter that replaces the submodule at ``qualified_name`` in-place."""
    parent_path, _, child = qualified_name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model

    def _setter(new_module: nn.Module) -> None:
        if isinstance(parent, nn.ModuleList) and child.isdigit():
            parent[int(child)] = new_module
        else:
            setattr(parent, child, new_module)

    return _setter


def resolve_offload_units(
    model: nn.Module,
    patched_modules: Iterable[nn.Module],
) -> list[tuple[str, nn.Module]]:
    """Return ``(qualified_name, module)`` pairs for outermost offload units.

    Units follow ``patched_modules`` order, skip nested patched modules (the
    ancestor is the unit), and drop duplicate identities (first path wins).
    Names prefer the stamped offload-unit name, then the live ``named_modules``
    path. The module is always the **live** child at that path on ``model`` so
    a phase swap cannot leak a detached copy from ``patched_modules``.
    """
    if not patched_modules:
        return []
    name_by_module = {module: name for name, module in model.named_modules()}
    seen: set[int] = set()
    units: list[tuple[str, nn.Module]] = []
    for module in patched_modules:
        qualified_name = get_offload_name(module) or name_by_module.get(module)
        if not qualified_name:
            continue
        try:
            live = model.get_submodule(qualified_name)
        except _SUBMODULE_PROBE_EXCEPTIONS:
            # Same missing-path pair as ``has_patched_ancestor``: PyTorch
            # ``get_submodule`` raises ``AttributeError``; vLLM
            # ``StageMissingLayer.__getattr__`` raises ``KeyError``. Skip the
            # unit. ``TypeError`` / ``RuntimeError`` still propagate.
            continue
        live_id = id(live)
        if live_id in seen:
            continue
        if has_patched_ancestor(model, qualified_name):
            continue
        seen.add(live_id)
        units.append((qualified_name, live))
    return units


def _subtree_has_offload_unit(
    module: nn.Module,
    offload_ids: set[int],
    cache: dict[int, bool],
) -> bool:
    """Return whether ``module`` is an offload unit or has one in its child tree.

    Identity-based so a shared unit registered under a second parent
    (``inner`` and ``outer.inner``) still counts as contained. Cycle-safe:
    in-progress nodes are treated as not-yet-found.
    """
    module_id = id(module)
    cached = cache.get(module_id)
    if cached is not None:
        return cached
    if module_id in offload_ids:
        cache[module_id] = True
        return True
    cache[module_id] = False
    found = any(_subtree_has_offload_unit(child, offload_ids, cache) for child in module.children())
    cache[module_id] = found
    return found


def _lacks_executable_forward(module: nn.Module) -> bool:
    """Return whether ``module`` still uses ``nn.Module.forward``.

    ``ModuleList`` / ``ModuleDict`` (and a bare ``nn.Module`` namespace) do not
    run their children; the parent iterates or indexes them. Listing those
    containers as compile units would leave the executed children eager.
    ``Sequential`` and custom modules override ``forward`` and stay islands.
    """
    return type(module).forward is nn.Module.forward


def resolve_resident_units(
    model: nn.Module,
    patched_modules: Iterable[nn.Module] = (),
) -> list[tuple[str, nn.Module]]:
    """Return outermost resident islands as ``(qualified_name, module)`` pairs.

    An island is a module that is not an offload unit and contains none. The
    root is never returned: if it contains offload units, walk into children;
    if it does not, each direct child is an island. Children of offload units
    are omitted. Shared modules are identity-deduplicated (first path wins).
    Registry containers without an executable ``forward`` (``ModuleList``,
    ``ModuleDict``) are not islands: their callable children are listed
    instead.

    Boundaries come from :func:`resolve_offload_units` live **identities**, not
    from patch markers or the canonical path alone. ``compile_fn`` may replace
    a unit with an unmarked wrapper; that object stays an offload unit. A
    shared unit reachable as ``inner`` and ``outer.inner`` is contained under
    both parents (``named_modules`` first-path wins does not hide ``outer``).
    """
    offload_ids = {id(module) for _name, module in resolve_offload_units(model, patched_modules)}
    contains_cache: dict[int, bool] = {}
    units: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()

    def visit(prefix: str, module: nn.Module, *, is_root: bool) -> None:
        if not is_root and id(module) in offload_ids:
            return
        if _subtree_has_offload_unit(module, offload_ids, contains_cache):
            for child_name, child in module.named_children():
                path = f"{prefix}.{child_name}" if prefix else child_name
                visit(path, child, is_root=False)
            return
        if is_root or _lacks_executable_forward(module):
            for child_name, child in module.named_children():
                path = f"{prefix}.{child_name}" if prefix else child_name
                visit(path, child, is_root=False)
            return
        module_id = id(module)
        if module_id in seen:
            return
        seen.add(module_id)
        units.append((prefix, module))

    visit("", model, is_root=True)
    return units


def resolve_compile_targets(
    model: nn.Module,
    patched_modules: Iterable[nn.Module],
) -> list[tuple[Callable[[nn.Module], None], nn.Module]]:
    """Return ``(setter, module)`` pairs — one per outermost offloaded unit.

    Only outermost patched modules are targeted: if a patched module has a patched
    ancestor, that ancestor is the offload/compile unit and the nested one is skipped.
    """
    return [
        (make_submodule_setter(model, name), module) for name, module in resolve_offload_units(model, patched_modules)
    ]


def resolve_compile_targets_on_model(
    model: nn.Module,
) -> list[tuple[Callable[[nn.Module], None], nn.Module]]:
    """Like :func:`resolve_compile_targets` but discovers patched units by walking ``model``."""
    targets: list[tuple[Callable[[nn.Module], None], nn.Module]] = []
    for qualified_name, module in model.named_modules():
        if not qualified_name:
            continue
        if not is_offload_patched_module(module):
            continue
        if has_patched_ancestor(model, qualified_name):
            continue
        targets.append((make_submodule_setter(model, qualified_name), module))
    return targets

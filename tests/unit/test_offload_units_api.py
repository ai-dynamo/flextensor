# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :meth:`~flextensor.OffloadManager.offload_units` and
:meth:`~flextensor.OffloadManager.resident_units`.

CPU-only: patching is exercised via ``_offload_modules`` so TensorManager / GPU
init is not required.
"""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from torch import nn

from flextensor.compile.module_swap import resolve_compile_targets, resolve_offload_units, resolve_resident_units
from flextensor.offload_manager import OffloadManager, get_offload_manager, offload_units, resident_units


class _Block(nn.Module):
    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):  # noqa: D102
        return self.lin(x)


class _Stack(nn.Module):
    """Repeated blocks plus a resident head, matching typical transformer layout."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([_Block() for _ in range(n)])
        self.head = nn.Linear(4, 4)

    def forward(self, x):  # noqa: D102
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


@pytest.fixture
def manager() -> Iterator[OffloadManager]:
    mgr = OffloadManager(f"test-units-{uuid4()}")
    try:
        yield mgr
    finally:
        mgr.release()


def _patch(manager: OffloadManager, model: nn.Module, patterns: list[str]) -> nn.Module:
    manager._model = model  # noqa: SLF001
    manager._offload_modules(model, patterns)
    return model


def test_offload_units_empty_without_model(manager: OffloadManager) -> None:
    assert manager.offload_units() == ()
    assert manager.resident_units() == ()


def test_offload_units_empty_when_nothing_patched(manager: OffloadManager) -> None:
    model = _patch(manager, _Stack(), [])
    assert manager.offload_units() == ()
    resident = manager.resident_units()
    # ModuleList is not a compile unit: the parent iterates ``blocks[i](x)``.
    assert resident == (
        ("embed", model.embed),
        ("blocks.0", model.blocks[0]),
        ("blocks.1", model.blocks[1]),
        ("blocks.2", model.blocks[2]),
        ("head", model.head),
    )


def test_offload_units_are_outermost_in_patch_order(manager: OffloadManager) -> None:
    model = _patch(manager, _Stack(), ["blocks.*"])
    units = manager.offload_units()
    assert units == (
        ("blocks.0", model.blocks[0]),
        ("blocks.1", model.blocks[1]),
        ("blocks.2", model.blocks[2]),
    )
    assert isinstance(units, tuple)


def test_offload_units_skip_nested_patched_module(manager: OffloadManager) -> None:
    model = _Stack(n=2)
    manager._model = model  # noqa: SLF001
    outer = model.blocks[0]
    outer._ft_original_forward_func = outer.forward  # noqa: SLF001
    outer._ft_offload_name = "blocks.0"  # noqa: SLF001
    inner = outer.lin
    inner._ft_original_forward_func = inner.forward  # noqa: SLF001
    inner._ft_offload_name = "blocks.0.lin"  # noqa: SLF001
    sibling = model.blocks[1]
    sibling._ft_original_forward_func = sibling.forward  # noqa: SLF001
    sibling._ft_offload_name = "blocks.1"  # noqa: SLF001
    manager._patched_modules = [outer, inner, sibling]  # noqa: SLF001

    assert manager.offload_units() == (("blocks.0", outer), ("blocks.1", sibling))


def test_offload_units_identity_deduplicates(manager: OffloadManager) -> None:
    model = _patch(manager, _Stack(n=1), ["blocks.0"])
    unit = model.blocks[0]
    manager._patched_modules.append(unit)  # noqa: SLF001
    assert manager.offload_units() == (("blocks.0", unit),)


def test_offload_units_returns_live_submodule_after_tree_swap(manager: OffloadManager) -> None:
    """Phase swaps replace ``_model`` but leave ``_patched_modules`` pointing at the old tree."""
    stale_model = _patch(manager, _Stack(n=1), ["blocks.0"])
    stale = stale_model.blocks[0]
    live_model = _Stack(n=1)
    live = live_model.blocks[0]
    live._ft_offload_name = "blocks.0"  # noqa: SLF001
    live._ft_original_forward_func = live.forward  # noqa: SLF001
    manager._model = live_model  # noqa: SLF001

    units = manager.offload_units()
    assert units == (("blocks.0", live),)
    assert units[0][1] is not stale

    setter, compiled_target = resolve_compile_targets(live_model, manager._patched_modules)[0]  # noqa: SLF001
    assert compiled_target is live
    replacement = nn.Identity()
    setter(replacement)
    assert live_model.blocks[0] is replacement


def test_offload_units_match_compile_fn_targets(manager: OffloadManager) -> None:
    model = _patch(manager, _Stack(), ["blocks.0", "blocks.2"])
    listed = manager.offload_units()
    compiled = resolve_compile_targets(model, manager._patched_modules)  # noqa: SLF001
    assert [module for _name, module in listed] == [module for _setter, module in compiled]


@pytest.mark.parametrize(
    "exc",
    [AttributeError("wrapper hides this path"), KeyError("module")],
    ids=lambda e: type(e).__name__,
)
def test_offload_units_skips_unresolved_live_path(manager: OffloadManager, exc: Exception) -> None:
    """vLLM ``StageMissingLayer`` raises ``KeyError`` from ``get_submodule``, not ``AttributeError``."""
    from unittest.mock import MagicMock

    stale = nn.Linear(4, 4)
    stale._ft_offload_name = "layers.0"  # noqa: SLF001
    stale._ft_original_forward_func = stale.forward  # noqa: SLF001
    model = MagicMock(spec=nn.Module)
    model.named_modules.return_value = []
    model.get_submodule.side_effect = exc
    manager._model = model  # noqa: SLF001
    manager._patched_modules = [stale]  # noqa: SLF001
    assert manager.offload_units() == ()
    model.get_submodule.assert_called_once_with("layers.0")


@pytest.mark.parametrize(
    "exc",
    [TypeError("descriptor misbehaved"), RuntimeError("CUDA error: no CUDA GPUs are available")],
    ids=lambda e: type(e).__name__,
)
def test_offload_units_propagates_unexpected_get_submodule_error(manager: OffloadManager, exc: Exception) -> None:
    from unittest.mock import MagicMock

    stale = nn.Linear(4, 4)
    stale._ft_offload_name = "layers.0"  # noqa: SLF001
    stale._ft_original_forward_func = stale.forward  # noqa: SLF001
    model = MagicMock(spec=nn.Module)
    model.named_modules.return_value = []
    model.get_submodule.side_effect = exc
    manager._model = model  # noqa: SLF001
    manager._patched_modules = [stale]  # noqa: SLF001
    with pytest.raises(type(exc), match=str(exc)):
        manager.offload_units()


def test_resident_units_omit_unmarked_compile_fn_replacement(manager: OffloadManager) -> None:
    """``apply_compile_fn`` parent-swaps whatever ``compile_fn`` returns; markers are not copied."""
    model = _patch(manager, _Stack(n=2), ["blocks.0"])
    replacement = nn.Identity()
    model.blocks[0] = replacement
    assert manager.offload_units() == (("blocks.0", replacement),)
    assert manager.resident_units() == (
        ("embed", model.embed),
        ("blocks.1", model.blocks[1]),
        ("head", model.head),
    )


def test_resident_units_do_not_collapse_replaced_offload_children(manager: OffloadManager) -> None:
    """If every list child is an unmarked offload replacement, ``blocks`` is not a resident island."""
    model = _patch(manager, _Stack(n=2), ["blocks.*"])
    model.blocks[0] = nn.Identity()
    model.blocks[1] = nn.Identity()
    assert [name for name, _module in manager.offload_units()] == ["blocks.0", "blocks.1"]
    assert manager.resident_units() == (("embed", model.embed), ("head", model.head))


def test_resident_units_are_outermost_islands(manager: OffloadManager) -> None:
    model = _patch(manager, _Stack(), ["blocks.0", "blocks.2"])
    resident = manager.resident_units()
    assert resident == (
        ("embed", model.embed),
        ("blocks.1", model.blocks[1]),
        ("head", model.head),
    )
    names = {name for name, _module in resident}
    assert "blocks" not in names
    assert "blocks.0" not in names
    assert "blocks.0.lin" not in names
    assert "blocks.1.lin" not in names
    assert "blocks.2" not in names


def test_resident_units_exclude_children_of_offload_units(manager: OffloadManager) -> None:
    _patch(manager, _Stack(n=1), ["blocks.0"])
    names = [name for name, _module in manager.resident_units()]
    assert "blocks.0.lin" not in names
    assert names == ["embed", "head"]


def test_resident_units_only_head_when_all_blocks_offloaded(manager: OffloadManager) -> None:
    model = _patch(manager, _Stack(), ["blocks.*"])
    assert manager.resident_units() == (("embed", model.embed), ("head", model.head))


def test_resident_units_omit_nested_parent_of_shared_offload_unit(manager: OffloadManager) -> None:
    """Same multi-parent shape as ``TestDetectCrossModuleReads.test_multi_parent_module_is_detected``.

    ``offload_units`` keeps the first path (``inner``). ``outer`` still holds
    that object as ``outer.inner`` and must not be a resident compile island.
    """

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, x):  # noqa: D102
            return self.lin(x)

    class Outer(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner
            self.gate = nn.Linear(2, 2)

        def forward(self, x):  # noqa: D102
            return self.inner(x) + self.gate(x)

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = Inner()
            self.outer = Outer(self.inner)

        def forward(self, x):  # noqa: D102
            return self.outer(x)

    model = _patch(manager, Root(), ["inner"])
    assert manager.offload_units() == (("inner", model.inner),)
    assert model.outer.inner is model.inner
    assert manager.resident_units() == (("outer.gate", model.outer.gate),)


def test_resident_units_descend_registry_containers_not_sequential(manager: OffloadManager) -> None:
    """``ModuleList`` / ``ModuleDict`` are not called; ``Sequential`` is one graph."""

    class _Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            self.tower = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
            self.heads = nn.ModuleDict({"a": nn.Linear(4, 4), "b": nn.Linear(4, 4)})

        def forward(self, x):  # noqa: D102
            x = self.blocks[0](x)
            x = self.tower(x)
            return self.heads["a"](x)

    model = _Root()
    manager._model = model  # noqa: SLF001
    assert manager.resident_units() == (
        ("blocks.0", model.blocks[0]),
        ("blocks.1", model.blocks[1]),
        ("tower", model.tower),
        ("heads.a", model.heads["a"]),
        ("heads.b", model.heads["b"]),
    )


def test_resident_units_identity_deduplicates_shared_module(manager: OffloadManager) -> None:
    shared = nn.Linear(4, 4)
    model = nn.Module()
    model.add_module("a", shared)
    model.add_module("b", shared)
    manager._model = model  # noqa: SLF001
    assert manager.resident_units() == (("a", shared),)


def test_listing_helpers_agree_with_manager(manager: OffloadManager) -> None:
    model = _Stack()
    assert resolve_offload_units(model, []) == []
    _patch(manager, model, ["head"])
    assert resolve_offload_units(model, manager._patched_modules) == [("head", model.head)]  # noqa: SLF001
    assert resolve_resident_units(model, manager._patched_modules) == [  # noqa: SLF001
        ("embed", model.embed),
        ("blocks.0", model.blocks[0]),
        ("blocks.1", model.blocks[1]),
        ("blocks.2", model.blocks[2]),
    ]


def test_package_level_wrappers_match_manager() -> None:
    name = f"test-units-pkg-{uuid4()}"
    om = get_offload_manager(name)
    try:
        model = _Stack()
        om._model = model  # noqa: SLF001
        om._offload_modules(model, ["blocks.1"])
        assert offload_units(name) == om.offload_units() == (("blocks.1", model.blocks[1]),)
        assert resident_units(name) == om.resident_units()
        assert [n for n, _m in resident_units(name)] == ["embed", "blocks.0", "blocks.2", "head"]
    finally:
        om.release()

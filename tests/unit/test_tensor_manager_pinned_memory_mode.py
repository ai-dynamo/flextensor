# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TensorManager's pinned_memory_mode option."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from flextensor import host_pinning
from flextensor import tensor_manager as tm_module
from flextensor.collectors import TensorStatistics
from flextensor.core.transfer import TransferStreamSynchronizationError
from flextensor.host_pinning import HostPinRegistry
from flextensor.loaders import (
    PreallocatedBatchTransferTensorLoader,
    PreallocatedBatchTransferTensorLoaderReordered,
    PreallocatedLoader,
    TensorStrategyLoader,
)
from flextensor.state_handler import LoaderInputData
from flextensor.strategy import GreedyStrategy
from flextensor.tensor_manager import TensorManager


@pytest.fixture
def _fake_cudart_available(monkeypatch):
    """Make host_pinning.is_available() report True without touching CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fake = MagicMock()
    fake.cudaHostRegister.return_value = 0
    fake.cudaHostUnregister.return_value = 0
    monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: fake)
    return fake


def _make_manager(**kwargs) -> TensorManager:
    device = torch.device("cpu")
    return TensorManager(device_gpu=device, tensor_manager_load_strategy=GreedyStrategy(), **kwargs)


def test_default_mode_is_torch_with_no_registry(_fake_cudart_available) -> None:
    """Default config (``pinned_memory_mode='torch'``) wires up a torch-mode
    pinner: no :class:`HostPinRegistry` is attached, so ``cudaHostRegister``
    is never invoked."""
    tm = _make_manager()
    assert tm.host_pin_registry is None


def test_host_register_mode_creates_registry(_fake_cudart_available) -> None:
    """``pinned_memory_mode='host_register'`` attaches a
    :class:`HostPinRegistry` — that's the observable signal that pin calls
    will dispatch to ``cudaHostRegister`` instead of ``tensor.pin_memory()``."""
    tm = _make_manager(pinned_memory_mode="host_register")
    assert isinstance(tm.host_pin_registry, HostPinRegistry)


def test_invalid_mode_raises() -> None:
    """An invalid ``pinned_memory_mode`` is rejected at the function
    boundary by @beartype's runtime Literal check before any
    :class:`HostPinner` is constructed."""
    from beartype.roar import BeartypeCallHintParamViolation

    with pytest.raises(BeartypeCallHintParamViolation, match="not_a_mode"):
        _make_manager(pinned_memory_mode="not_a_mode")  # type: ignore[arg-type]


def test_construction_on_cpu_only_host_raises(monkeypatch) -> None:
    """When CUDA is unavailable on the host, neither ``host_register`` nor
    ``torch`` mode can pin. ``pinned_memory=True`` is a misconfiguration in
    that case — offloading without a GPU has no purpose — so
    :class:`TensorManager` construction surfaces it as a ``RuntimeError``
    rather than silently degrading to pageable transfers."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: None)

    with pytest.raises(RuntimeError, match="pinned_memory=True requires a CUDA host"):
        _make_manager(pinned_memory_mode="host_register")


def test_host_register_disabled_when_pinned_memory_false(_fake_cudart_available) -> None:
    """``pinned_memory=False`` overrides ``pinned_memory_mode`` — no registry
    is created because pinning itself is disabled."""
    tm = _make_manager(pinned_memory=False, pinned_memory_mode="host_register")
    assert tm.host_pin_registry is None


class TestShouldPinInPreprocess:
    """``TensorManager.should_pin_in_preprocess()`` is the single source of
    truth for the pin_memory bool passed to ``preprocess_model`` from both
    the warmup path (``initialize_warmup``) and the discovery-restore path
    (``state_handler.TensorManagerStateHandler.restore_state``).

    Block-loader paths manage their own per-block pinning, so preprocess
    must not pin during model preparation regardless of ``pinned_memory``.
    Strategy loader pins during preprocess and inherits the user's
    ``pinned_memory`` flag verbatim.
    """

    def test_strategy_loader_inherits_pinned_memory_flag(self, _fake_cudart_available) -> None:
        tm = _make_manager(loader_type="strategy", pinned_memory=True)
        assert tm.should_pin_in_preprocess() is True

    def test_strategy_loader_respects_pinned_memory_false(self) -> None:
        tm = _make_manager(loader_type="strategy", pinned_memory=False)
        assert tm.should_pin_in_preprocess() is False

    @pytest.mark.parametrize("loader_type", ["allocation_block_transfer", "raw_block_transfer"])
    def test_block_loaders_force_false_even_when_pinned_memory_true(self, _fake_cudart_available, loader_type) -> None:
        tm = _make_manager(loader_type=loader_type, pinned_memory=True)
        assert tm.should_pin_in_preprocess() is False, (
            f"{loader_type} pins its own per-block buffers later; preprocess must not pin"
        )


def test_shutdown_releases_host_pin_registry(_fake_cudart_available) -> None:
    tm = _make_manager(pinned_memory_mode="host_register")
    registry = tm.host_pin_registry
    assert registry is not None

    tensor = torch.zeros(16, dtype=torch.uint8)
    registry.pin_in_place(tensor)
    assert len(registry) == 1

    tm.shutdown()
    assert len(registry) == 0


def test_initialize_warmup_routes_host_pinner_through_to_pin_processor(_fake_cudart_available, monkeypatch) -> None:
    """Wiring guard: a refactor that drops ``host_pinner=self.host_pinner``
    from the ``preprocess_model`` call site in ``initialize_warmup`` would
    silently disable host_register pinning for offloaded parameters and every
    existing unit test would still pass. Pin the contract end-to-end:

    - ``preprocess_model`` must be called with ``pin_memory=True`` and the
      manager's own ``host_pinner``,
    - the resulting ``MoveToPinMemoryTensorProcessor`` must actually register
      CPU parameters with the registry,
    - which must dispatch to ``cudaHostRegister`` via the cudart binding.
    """
    captured: dict = {}
    real_preprocess = tm_module.preprocess_model

    def spy(model, tensor_manager, device_gpu, **kwargs):
        captured["called"] = True
        captured["host_pinner"] = kwargs.get("host_pinner")
        captured["pin_memory"] = kwargs.get("pin_memory")
        return real_preprocess(model, tensor_manager, device_gpu, **kwargs)

    monkeypatch.setattr(tm_module, "preprocess_model", spy)

    # loader_type="strategy" is the path that pins per-tensor via the
    # MoveToPinMemoryTensorProcessor (the block loaders pin their own large
    # blocks instead and force pin_memory=False here).
    tm = _make_manager(
        pinned_memory=True,
        pinned_memory_mode="host_register",
        loader_type="strategy",
    )
    assert tm.host_pin_registry is not None
    tm.set_model(torch.nn.Linear(4, 4))

    # Stub lifecycle pieces we don't need: they freeze tensors_map (which
    # complicates assertions) and require real CUDA in places.
    monkeypatch.setattr(tm, "_move_non_offloaded_tensors_to_gpu", lambda *a, **kw: None)
    monkeypatch.setattr(tm, "prepare_model_ids", lambda *a, **kw: None)
    monkeypatch.setattr(tm, "prepare_warmup_mode", lambda: None)

    tm.initialize_warmup()

    assert captured.get("called"), "preprocess_model was never invoked"
    assert captured["pin_memory"] is True, "pin_memory kwarg lost on the way through"
    assert captured["host_pinner"] is tm.host_pinner, "host_pinner kwarg lost on the way through"

    # Behavior-level: pinning must have actually happened.
    assert len(tm.host_pin_registry) > 0, "MoveToPinMemoryTensorProcessor never reached the registry"
    assert _fake_cudart_available.cudaHostRegister.call_count > 0, "registry did not dispatch to cudaHostRegister"

    pre_unregisters = _fake_cudart_available.cudaHostUnregister.call_count
    tm.shutdown()
    assert len(tm.host_pin_registry) == 0
    assert _fake_cudart_available.cudaHostUnregister.call_count > pre_unregisters


def test_setup_allocation_block_loader_routes_host_pinner_through_to_blocks(
    _fake_cudart_available, monkeypatch
) -> None:
    """Wiring guard for the default ``loader_type='allocation_block_transfer'``
    path. Pinning is forwarded through three sites:

    - ``TensorManager._setup_allocation_block_loader`` →
      :class:`AllocationBlockController` (``host_pinner=self.host_pinner``)
    - :class:`AllocationBlockController` → :class:`AllocationManager`
    - :class:`AllocationManager` → :class:`AllocationBlock` →
      ``host_pinner.pin``

    A regression that drops the kwarg at any of those sites produces a
    default :class:`HostPinner` (torch mode) inside the chain and
    ``cudaHostRegister`` is never invoked — but every existing test still
    passes. Pin the contract end-to-end by asserting the registry is
    populated and ``cudaHostRegister`` was actually called.
    """
    monkeypatch.setattr(tm_module, "PreallocatedBatchTransferTensorLoader", MagicMock())

    tm = _make_manager(pinned_memory=True, pinned_memory_mode="host_register")
    assert tm.loader_type == "allocation_block_transfer"
    assert tm.host_pin_registry is not None

    weight = torch.zeros(64, dtype=torch.uint8)
    tm.tensors_map = {id(weight): weight}
    tm.stats = []
    tm.load_strategy = {
        "label_a": [
            TensorStatistics(
                tensor_id=id(weight),
                name="label_a",
                size_bytes=weight.numel() * weight.element_size(),
                load_time_ms=0.0,
            )
        ]
    }

    data = LoaderInputData(
        allocation_ordered={0: ["label_a"]},
        label_to_block_id={"label_a": 0},
        transfer_to_compute_map={"label_a": "label_a"},
    )

    pre_register_calls = _fake_cudart_available.cudaHostRegister.call_count
    tm._setup_allocation_block_loader(data, prepare_state=False)

    assert _fake_cudart_available.cudaHostRegister.call_count > pre_register_calls, (
        "AllocationBlockController did not dispatch to cudaHostRegister — "
        "host_pinner kwarg dropped somewhere in the forwarding chain"
    )
    assert len(tm.host_pin_registry) > 0, (
        "host_pin_registry is empty — host_pinner was not threaded through "
        "TensorManager → AllocationBlockController → AllocationManager → AllocationBlock"
    )


def test_setup_raw_block_loader_routes_host_pinner_through_to_blocks(_fake_cudart_available, monkeypatch) -> None:
    """Wiring guard for ``loader_type='raw_block_transfer'``. Pinning is
    forwarded through two sites:

    - ``TensorManager._setup_raw_block_loader`` → :class:`RawBlockController`
      (``host_pinner=self.host_pinner``)
    - :class:`RawBlockController` → ``host_pinner.pin`` (per-layer block)

    Same regression class as the allocation_block_transfer guard above:
    dropping the kwarg at either site silently disables host_register pinning
    for raw-block loads, but every existing functional test still passes
    because the loader's behavior on a CPU-only test rig is indistinguishable.
    Pin the contract end-to-end here too.
    """
    monkeypatch.setattr(tm_module, "PreallocatedBatchTransferTensorLoader", MagicMock())

    tm = _make_manager(
        pinned_memory=True,
        pinned_memory_mode="host_register",
        loader_type="raw_block_transfer",
    )
    assert tm.loader_type == "raw_block_transfer"
    assert tm.host_pin_registry is not None

    weight = torch.zeros(64, dtype=torch.uint8)
    tm.tensors_map = {id(weight): weight}
    tm.stats = []
    tm.load_strategy = {
        "label_a": [
            TensorStatistics(
                tensor_id=id(weight),
                name="label_a",
                size_bytes=weight.numel() * weight.element_size(),
                load_time_ms=0.0,
            )
        ]
    }

    data = LoaderInputData(
        allocation_ordered={0: ["label_a"]},
        label_to_block_id={"label_a": 0},
        transfer_to_compute_map={"label_a": "label_a"},
        label_to_size_map={"label_a": weight.numel() * weight.element_size()},
        block_sizes={0: weight.numel() * weight.element_size()},
    )

    pre_register_calls = _fake_cudart_available.cudaHostRegister.call_count
    tm._setup_raw_block_loader(data, prepare_state=False)

    assert _fake_cudart_available.cudaHostRegister.call_count > pre_register_calls, (
        "RawBlockController did not dispatch to cudaHostRegister — host_pinner "
        "kwarg dropped somewhere in the forwarding chain"
    )
    assert len(tm.host_pin_registry) > 0, (
        "host_pin_registry is empty — host_pinner was not threaded through "
        "TensorManager → RawBlockController → host_pinner.pin"
    )


def test_shutdown_releases_pins_when_loader_shutdown_raises(_fake_cudart_available) -> None:
    # Regression: TensorManager.shutdown() previously skipped host_pinner.release_all()
    # when the loader's shutdown raised (e.g. SHM teardown / file-lock release failures),
    # leaking every cudaHostRegister pin until process exit. The CHANGELOG promises
    # "released on shutdown()", so the unregister path must be guaranteed.
    tm = _make_manager(pinned_memory_mode="host_register")
    registry = tm.host_pin_registry
    assert registry is not None

    tensor = torch.zeros(16, dtype=torch.uint8)
    registry.pin_in_place(tensor)
    assert len(registry) == 1

    failing_loader = MagicMock()
    failing_loader.shutdown.side_effect = RuntimeError("simulated SHM teardown failure")
    tm.tensor_layer_loader = failing_loader

    with pytest.raises(RuntimeError, match="simulated SHM teardown failure"):
        tm.shutdown()

    failing_loader.shutdown.assert_called_once()
    assert len(registry) == 0


def test_shutdown_propagates_release_all_failure_when_loader_succeeds(_fake_cudart_available) -> None:
    tm = _make_manager(pinned_memory_mode="host_register")
    tm.tensor_layer_loader = MagicMock()
    tm.host_pinner = MagicMock()
    tm.host_pinner.release_all.side_effect = RuntimeError("pin release failed")

    with pytest.raises(RuntimeError, match="pin release failed"):
        tm.shutdown()


def test_preallocated_base_shutdown_does_not_require_transfer_stream() -> None:
    events = []
    loader = SimpleNamespace(allocation_controller=SimpleNamespace(shutdown=lambda: events.append("shutdown")))

    PreallocatedLoader.shutdown(loader)

    assert events == ["shutdown"]


@pytest.mark.parametrize(
    "loader_type",
    [PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered],
)
def test_shutdown_synchronizes_preallocated_loader_once(loader_type) -> None:
    events = []
    tm = _make_manager(pinned_memory=False)
    loader = loader_type.__new__(loader_type)
    loader._transfer_runtime = SimpleNamespace(close=lambda: events.append("synchronize"))
    loader.allocation_controller = SimpleNamespace(shutdown=lambda: events.append("shutdown"))
    tm.tensor_layer_loader = loader

    tm.shutdown()

    assert events == ["synchronize", "shutdown"]


@pytest.mark.parametrize("loader_type", [TensorStrategyLoader, PreallocatedBatchTransferTensorLoader])
def test_shutdown_transfer_sync_failure_retains_pins(_fake_cudart_available, loader_type) -> None:
    tm = _make_manager(pinned_memory_mode="host_register")
    registry = tm.host_pin_registry
    assert registry is not None
    registry.pin_in_place(torch.zeros(16, dtype=torch.uint8))

    loader = loader_type.__new__(loader_type)
    if loader_type is PreallocatedBatchTransferTensorLoader:
        loader._transfer_runtime = MagicMock()
        loader._transfer_runtime.close.side_effect = TransferStreamSynchronizationError("transfer sync failed")
    else:
        loader.transfer_stream = MagicMock()
        loader.transfer_stream.synchronize.side_effect = RuntimeError("transfer sync failed")
    tm.tensor_layer_loader = loader

    with pytest.raises(RuntimeError, match="transfer sync failed"):
        tm.shutdown()

    assert len(registry) == 1


def test_shutdown_loader_exception_not_masked_when_release_all_also_raises(_fake_cudart_available, caplog) -> None:
    """If both ``loader.shutdown()`` and ``host_pinner.release_all()`` raise
    inside ``TensorManager.shutdown()``, the original loader exception must
    surface to the caller — masking it under a cleanup error makes the
    operator chase the wrong root cause. The release_all failure must
    instead be logged via ``logger.exception`` so the traceback is still
    accessible."""

    tm = _make_manager(pinned_memory_mode="host_register")

    failing_loader = MagicMock()
    failing_loader.shutdown.side_effect = RuntimeError("ROOT CAUSE: SHM teardown failure")
    tm.tensor_layer_loader = failing_loader

    failing_pinner = MagicMock()
    failing_pinner.release_all.side_effect = RuntimeError("CLEANUP NOISE: cudart already torn down")
    tm.host_pinner = failing_pinner

    with caplog.at_level(logging.ERROR, logger=tm_module.__name__), pytest.raises(RuntimeError, match="ROOT CAUSE"):
        tm.shutdown()

    failing_loader.shutdown.assert_called_once()
    failing_pinner.release_all.assert_called_once()

    # The release_all failure must be logged as an ERROR with traceback
    # so it isn't silently dropped, but it must not be the exception the
    # caller sees.
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "release_all failure during shutdown must be logged at ERROR"
    assert any("release_all" in r.message for r in errors)
    assert any(r.exc_info is not None for r in errors), (
        "logger.exception must attach the release_all traceback for diagnosis"
    )


@pytest.mark.parametrize("requested", ["torch", "host_register"])
def test_offload_manager_forwards_pinned_memory_mode_to_tensor_manager(requested) -> None:
    """Wiring guard for the primary entry path: ``OffloadConfig.pinned_memory_mode``
    must reach :class:`TensorManager.__init__` unchanged. A typo or dropped kwarg
    in :meth:`OffloadManager._initialize_tensor_manager` would silently ignore
    the user's explicit ``"host_register"`` request and fall back to the default
    ``"torch"`` mode — every existing functional test would still pass because
    the difference is only observable via the construction-time WARNING log
    and the presence of a :class:`HostPinRegistry`.

    Patches :class:`TensorManager` at its source module (the call site
    lazy-imports it inside ``_initialize_tensor_manager`` to break a circular
    import) and asserts the kwarg arrives with the configured value.
    """
    from unittest.mock import patch

    from flextensor.offload_manager import OffloadConfig, OffloadManager

    config = OffloadConfig(enabled=True, pinned_memory_mode=requested)
    om = OffloadManager("test_pinned_mode_forwarding")
    om.set_config(config)

    with (
        patch("flextensor.offload_manager.resolve_load_strategy"),
        patch("flextensor.tensor_manager.TensorManager") as mock_tm,
    ):
        om._initialize_tensor_manager()

    mock_tm.assert_called_once()
    _, kwargs = mock_tm.call_args
    assert kwargs["pinned_memory_mode"] == requested, (
        f"OffloadManager dropped pinned_memory_mode={requested!r} on its way to "
        f"TensorManager — got {kwargs.get('pinned_memory_mode')!r}"
    )
    # pinned_memory must also flow through; the mode is meaningless without it.
    assert kwargs["pinned_memory"] is True

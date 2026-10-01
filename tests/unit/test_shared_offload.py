# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import flextensor as ft
import flextensor.shared_offload as shared_offload
from flextensor.config import OffloadConfig
from flextensor.offload_manager import OffloadManager
from flextensor.shared_offload import SharedOffloadSession
from flextensor.shm import ShmCoordinatorError, ShmCreatorDiedError, ShmCreatorFailedError
from flextensor.shm.coordinator import ShmCoordinator
from flextensor.shm.flexible_shm import FlexibleSharedMemory
from flextensor.shm.tensor_store import SharedTensorStore
from flextensor.strategy import NthLayerStrategy

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeModel(nn.Module):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label
        self.resident = nn.Linear(2, 2, bias=False)
        self.register_buffer("rotary_cache", torch.ones(2))


class FakeProxy(nn.Module):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label


class FakeManager(OffloadManager):
    def __init__(self, name: str, events: list[tuple[Any, ...]]) -> None:
        self.name = name
        self.events = events
        self._model = FakeModel("final-model")
        self.disk_state = SimpleNamespace(gpu_tensors_names=["resident.weight"])
        self.live_state = SimpleNamespace(gpu_tensors_names=["resident.weight", "rotary_cache"])
        self.config: OffloadConfig | None = None
        self.reservation: object | None = None
        self.release_calls: list[bool] = []
        self.release_failures = 0

    def _prepare_shared_offload(self, config: OffloadConfig, *, owner_token: object) -> None:
        if self.reservation is not None:
            raise RuntimeError("manager is already reserved")
        self.events.append(("prepare_manager", self.name, config.shm_namespace, config.shm_wait_timeout))
        self.config = config
        self.reservation = owner_token

    def _restore_shared_offload(
        self,
        model: nn.Module,
        state: object,
        namespace: str,
        *,
        load_from_shm: bool,
        owner_token: object | None = None,
        shm_owner: FlexibleSharedMemory | None = None,
        compile_fn: object = None,
        before_compile: Callable[[], None] | None = None,
    ) -> nn.Module:
        assert self.reservation is owner_token
        self.shm_owner = shm_owner
        if load_from_shm:
            self.events.append(("initialize_follower", self.name, state, model, namespace, compile_fn))
            self._model = model
            return FakeProxy("follower-proxy")
        self.events.append(("initialize_creator", self.name, model, state, namespace, compile_fn))
        if before_compile is not None:
            before_compile()
        return FakeProxy("creator-proxy")

    def _export_runtime_state(self) -> object:
        self.events.append(("export_state", self.name))
        return self.live_state

    def _release_shared_offload(
        self,
        *,
        owner_token: object,
        preserve_model: bool,
    ) -> None:
        if self.reservation is None or self.reservation is not owner_token:
            raise RuntimeError("Shared offload session no longer owns its manager reservation")
        self.release(preserve_model=preserve_model)
        self.reservation = None

    def release(self, *, preserve_model: bool = True) -> None:
        self.release_calls.append(preserve_model)
        self.events.append(("release_manager", self.name, preserve_model))
        if self.release_failures:
            self.release_failures -= 1
            raise RuntimeError("manager cleanup failed")


class FakeCoordinator(ShmCoordinator):
    instances: ClassVar[list[FakeCoordinator]] = []

    def __init__(self, namespace: str, is_creator: bool, events: list[tuple[Any, ...]]) -> None:
        self.namespace = namespace
        self.is_creator = is_creator
        self._coord_shm = MagicMock(spec=FlexibleSharedMemory)
        self.events = events
        self.state = SimpleNamespace(gpu_tensors_names=["resident.weight", "rotary_cache"])
        self.close_failures = 0
        self.instances.append(self)
        self.events.append(("create_coordinator", namespace, is_creator))

    def wait_for_ready(self, timeout: float | None = None) -> None:
        self.events.append(("wait_for_ready", timeout))

    def read_profile(self) -> object:
        self.events.append(("read_profile",))
        return self.state

    def write_profile(self, state: object) -> None:
        self.events.append(("write_profile", state))

    def notify_ready(self) -> None:
        self.events.append(("notify_ready",))

    def mark_failed(self, error: BaseException) -> None:
        self.events.append(("mark_failed", type(error).__name__, str(error)))

    def close(self) -> None:
        self.events.append(("close_coordinator", self.namespace))
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("coordinator cleanup failed")


class FakeStore(SharedTensorStore):
    events: ClassVar[list[tuple[Any, ...]]] = []
    instances: ClassVar[list[FakeStore]] = []

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.close_failures = 0

    @classmethod
    def reset(cls, events: list[tuple[Any, ...]]) -> None:
        cls.events = events
        cls.instances = []

    @classmethod
    def create_from_model(
        cls,
        namespace: str,
        model: nn.Module,
        *,
        parameter_names: list[str],
        shm_owner: FlexibleSharedMemory | None = None,
    ) -> FakeStore:
        cls.events.append(("create_store", namespace, model, parameter_names))
        store = cls(namespace)
        store.shm_owner = shm_owner
        cls.instances.append(store)
        return store

    @classmethod
    def attach(cls, namespace: str, *, shm_owner: FlexibleSharedMemory | None = None) -> FakeStore:
        cls.events.append(("attach_store", namespace))
        store = cls(namespace)
        store.shm_owner = shm_owner
        cls.instances.append(store)
        return store

    def materialize_into(
        self,
        model: nn.Module,
        *,
        device: torch.device | str,
    ) -> None:
        self.events.append(("materialize", model, device))

    def close(self) -> None:
        self.events.append(("close_store", self.namespace))
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("store cleanup failed")


def config() -> OffloadConfig:
    return OffloadConfig(pinned_memory=False, transfer_mode="allocation_block_transfer")


def manager_namespace(name: str, *, namespace: str = "run") -> str:
    return SharedOffloadSession(namespace).namespace_for(name)


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[tuple[Any, ...]],
    roles: dict[str, bool],
) -> dict[str, FakeManager]:
    managers: dict[str, FakeManager] = {}
    FakeCoordinator.instances = []
    FakeStore.reset(events)

    def get_manager(name: str) -> FakeManager:
        events.append(("get_manager", name))
        return managers.setdefault(name, FakeManager(name, events))

    def load_profile(path: Path) -> object:
        events.append(("load_profile", path))
        return managers[path.parent.name].disk_state

    monkeypatch.setattr(shared_offload.TensorManagerStateHandler, "load_from_file", load_profile)
    monkeypatch.setattr(shared_offload, "get_offload_manager", get_manager)
    monkeypatch.setattr(
        shared_offload,
        "ShmCoordinator",
        lambda namespace: FakeCoordinator(namespace, roles[namespace], events),
    )
    monkeypatch.setattr(shared_offload, "SharedTensorStore", FakeStore)
    return managers


def load(
    session: SharedOffloadSession,
    name: str,
    events: list[tuple[Any, ...]],
    *,
    materialize_device: torch.device | str = "cpu",
) -> nn.Module:
    return session.offload_from_profile(
        f"/profiles/{name}",
        config=config(),
        name=name,
        creator_model_factory=lambda: (events.append(("creator_factory", name)), FakeModel("full"))[1],
        follower_model_factory=lambda: (events.append(("follower_factory", name)), FakeModel("meta"))[1],
        materialize_device=materialize_device,
    )


def test_public_facade_exports_shared_offload_api() -> None:
    assert ft.SharedOffloadSession is SharedOffloadSession
    assert "SharedOffloadSession" in ft.__all__
    assert ft.ShmCoordinatorError is ShmCoordinatorError
    assert ft.ShmCreatorDiedError is ShmCreatorDiedError
    assert ft.ShmCreatorFailedError is ShmCreatorFailedError


def test_session_config_preserves_strategy_and_does_not_mutate_input(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    strategy = NthLayerStrategy(nth_layer=2)
    original = OffloadConfig(
        pinned_memory=False,
        transfer_mode="allocation_block_transfer",
        load_strategy=strategy,
        shm_wait_timeout=5.0,
    )

    with SharedOffloadSession("run", wait_timeout=17.0) as session:
        session.offload_from_profile(
            "/profiles/model",
            config=original,
            name="model",
            creator_model_factory=lambda: FakeModel("full"),
            follower_model_factory=lambda: pytest.fail("creator path"),
        )
        cloned = managers["model"].config
        assert cloned is not None and cloned is not original
        assert cloned.load_strategy is strategy
        assert cloned.shm_enabled is True
        assert cloned.shm_namespace == session.namespace_for("model")
        assert cloned.shm_wait_timeout == 17.0

    assert original.load_strategy is strategy
    assert original.shm_enabled is False
    assert original.shm_namespace is None
    assert original.shm_wait_timeout == 5.0


def test_manager_namespaces_are_compact_deterministic_and_unambiguous() -> None:
    manager = SharedOffloadSession("x" * 96).namespace_for("y" * 64)

    assert manager == SharedOffloadSession("x" * 96).namespace_for("y" * 64)
    assert len(manager) == 15
    assert manager.startswith("ft_")
    assert SharedOffloadSession("a_b").namespace_for("c") != SharedOffloadSession("a").namespace_for("b_c")


def test_creator_publishes_then_session_owns_resources_until_close(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    session = SharedOffloadSession("run", wait_timeout=27.5)

    proxy = load(session, "model", events)

    assert isinstance(proxy, FakeProxy) and proxy.label == "creator-proxy"
    assert [event[0] for event in events] == [
        "get_manager",
        "prepare_manager",
        "create_coordinator",
        "load_profile",
        "creator_factory",
        "initialize_creator",
        "export_state",
        "create_store",
        "write_profile",
        "notify_ready",
    ]
    assert events[3] == ("load_profile", Path("/profiles/model/profile.json"))
    assert events[5][3] is managers["model"].disk_state
    assert managers["model"].reservation is not None
    assert managers["model"].shm_owner is FakeCoordinator.instances[0].memory_owner
    assert FakeStore.instances[0].shm_owner is managers["model"].shm_owner

    session.close()

    assert [event[0] for event in events[-3:]] == ["release_manager", "close_store", "close_coordinator"]
    assert managers["model"].reservation is None


@pytest.mark.parametrize("wait_timeout, coordinator_timeout", [(0.0, None), (9, 9.0)])
def test_follower_waits_then_materializes_before_restore(
    monkeypatch: pytest.MonkeyPatch,
    wait_timeout: float,
    coordinator_timeout: float | None,
) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): False})
    SharedOffloadSession.validate_options("run", wait_timeout=wait_timeout)
    session = SharedOffloadSession("run", wait_timeout=wait_timeout)

    proxy = load(session, "model", events, materialize_device="cuda:3")

    assert isinstance(proxy, FakeProxy) and proxy.label == "follower-proxy"
    assert managers["model"].config.shm_wait_timeout == wait_timeout
    assert ("wait_for_ready", coordinator_timeout) in events
    assert managers["model"].shm_owner is FakeCoordinator.instances[0].memory_owner
    assert FakeStore.instances[0].shm_owner is managers["model"].shm_owner
    assert [event[0] for event in events] == [
        "get_manager",
        "prepare_manager",
        "create_coordinator",
        "wait_for_ready",
        "read_profile",
        "follower_factory",
        "attach_store",
        "materialize",
        "initialize_follower",
    ]
    assert events[-2][2:] == ("cuda:3",)
    session.close()


def test_profile_read_failure_publishes_failure_and_releases_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    primary_error = OSError("profile unreadable")
    load_profile = MagicMock(side_effect=primary_error)
    monkeypatch.setattr(shared_offload.TensorManagerStateHandler, "load_from_file", load_profile)
    session = SharedOffloadSession("run")

    with pytest.raises(OSError) as exc_info:
        load(session, "model", events)

    assert exc_info.value is primary_error
    load_profile.assert_called_once_with(Path("/profiles/model/profile.json"))
    assert [event[0] for event in events] == [
        "get_manager",
        "prepare_manager",
        "create_coordinator",
        "mark_failed",
        "release_manager",
        "close_coordinator",
    ]
    assert managers["model"].reservation is None
    session.close()


def test_active_manager_rejected_before_election_or_profile_read(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = OffloadManager("model")
    manager._tensor_manager = MagicMock()  # noqa: SLF001 - simulate an active runtime.
    monkeypatch.setattr(shared_offload, "get_offload_manager", lambda name: manager)
    coordinator = MagicMock()
    load_profile = MagicMock()
    monkeypatch.setattr(shared_offload, "ShmCoordinator", coordinator)
    monkeypatch.setattr(shared_offload.TensorManagerStateHandler, "load_from_file", load_profile)
    events: list[tuple[Any, ...]] = []
    session = SharedOffloadSession("run")

    with pytest.raises(RuntimeError, match="existing TensorManager"):
        load(session, "model", events)

    coordinator.assert_not_called()
    load_profile.assert_not_called()
    assert events == []
    session.close()


def test_session_retries_a_rejected_store_handle_without_replacing_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): False})
    primary_error = ValueError("invalid resident store")

    class RejectingStore(FakeStore):
        @classmethod
        def attach(cls, namespace: str, *, shm_owner: FlexibleSharedMemory | None = None) -> RejectingStore:
            cls.events.append(("attach_store", namespace))
            store = cls(namespace)
            cls.instances.append(store)
            store.close_failures = 2
            cls._close_rejected(store, primary_error)  # noqa: SLF001
            raise primary_error

    monkeypatch.setattr(shared_offload, "SharedTensorStore", RejectingStore)
    session = SharedOffloadSession("run")

    with pytest.raises(ValueError) as exc_info:
        load(session, "model", events)

    assert exc_info.value is primary_error
    assert [event[0] for event in events[-3:]] == ["close_store", "release_manager", "close_store"]
    session.close()
    assert [event[0] for event in events[-2:]] == ["close_store", "close_coordinator"]


def test_named_managers_retry_in_reverse_load_order(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(
        monkeypatch,
        events=events,
        roles={
            manager_namespace("text"): True,
            manager_namespace("stage1"): False,
            manager_namespace("stage2"): True,
        },
    )
    session = SharedOffloadSession("run")
    for name in ("text", "stage1", "stage2"):
        load(session, name, events)
        managers[name].release_failures = 1

    events.clear()
    with pytest.raises(RuntimeError, match="manager cleanup failed"):
        session.close()
    assert events == [
        ("release_manager", "stage2", False),
        ("release_manager", "stage1", False),
        ("release_manager", "text", False),
    ]

    events.clear()
    session.close()
    assert events == [
        ("release_manager", "stage2", False),
        ("close_store", manager_namespace("stage2")),
        ("close_coordinator", manager_namespace("stage2")),
        ("release_manager", "stage1", False),
        ("close_store", manager_namespace("stage1")),
        ("close_coordinator", manager_namespace("stage1")),
        ("release_manager", "text", False),
        ("close_store", manager_namespace("text")),
        ("close_coordinator", manager_namespace("text")),
    ]
    events.clear()
    session.close()
    assert events == []


def test_creator_failure_is_published_before_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[Any, ...]] = []
    install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    session = SharedOffloadSession("run", wait_timeout=13)

    with pytest.raises(LookupError, match="checkpoint failed"):
        session.offload_from_profile(
            "/profiles/model",
            config=config(),
            name="model",
            creator_model_factory=lambda: (_ for _ in ()).throw(LookupError("checkpoint failed")),
            follower_model_factory=lambda: pytest.fail("creator path"),
        )

    assert [event[0] for event in events[-3:]] == ["mark_failed", "release_manager", "close_coordinator"]
    assert events[-3] == ("mark_failed", "LookupError", "checkpoint failed")


def test_failed_initialization_cleanup_is_retained_for_close(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    session = SharedOffloadSession("run")

    def fail_factory() -> nn.Module:
        managers["model"].release_failures = 1
        raise LookupError("checkpoint failed")

    with pytest.raises(LookupError, match="checkpoint failed"):
        session.offload_from_profile(
            "/profiles/model",
            config=config(),
            name="model",
            creator_model_factory=fail_factory,
            follower_model_factory=lambda: pytest.fail("creator path"),
        )

    assert [event[0] for event in events[-2:]] == ["mark_failed", "release_manager"]
    session.close()
    assert [event[0] for event in events[-2:]] == ["release_manager", "close_coordinator"]


def test_publication_failure_is_a_note_on_the_primary_error(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    publication_error = RuntimeError("failed publication")
    primary_error = LookupError("checkpoint failed")

    class FailingCoordinator(FakeCoordinator):
        def mark_failed(self, error: BaseException) -> None:
            super().mark_failed(error)
            raise publication_error

    monkeypatch.setattr(
        shared_offload,
        "ShmCoordinator",
        lambda namespace: FailingCoordinator(namespace, True, events),
    )

    with pytest.raises(LookupError) as exc_info:
        SharedOffloadSession("run").offload_from_profile(
            "/profiles/model",
            config=config(),
            name="model",
            creator_model_factory=lambda: (_ for _ in ()).throw(primary_error),
            follower_model_factory=lambda: pytest.fail("creator path"),
        )

    assert exc_info.value is primary_error
    if hasattr(BaseException, "add_note"):
        assert any("could not publish" in note for note in getattr(exc_info.value, "__notes__", ()))
    assert managers["model"].reservation is None


@pytest.mark.parametrize("failed_resource", ["manager", "store", "coordinator"])
@pytest.mark.parametrize("preserve_models", [False, True])
def test_close_retries_from_the_failed_resource(
    monkeypatch: pytest.MonkeyPatch,
    failed_resource: str,
    preserve_models: bool,
) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    session = SharedOffloadSession("run")
    load(session, "model", events)
    if failed_resource == "manager":
        managers["model"].release_failures = 1
    elif failed_resource == "store":
        FakeStore.instances[0].close_failures = 1
    else:
        FakeCoordinator.instances[0].close_failures = 1
    events.clear()

    with pytest.raises(RuntimeError, match=f"{failed_resource} cleanup failed"):
        session.close(preserve_models=preserve_models)
    first_attempt = list(events)
    session.close(preserve_models=not preserve_models)

    expected_release_calls = 2 if failed_resource == "manager" else 1
    assert managers["model"].release_calls == [preserve_models] * expected_release_calls

    if failed_resource == "manager":
        assert [event[0] for event in first_attempt] == ["release_manager"]
        assert [event[0] for event in events] == [
            "release_manager",
            "release_manager",
            "close_store",
            "close_coordinator",
        ]
    elif failed_resource == "store":
        assert [event[0] for event in first_attempt] == ["release_manager", "close_store"]
        assert [event[0] for event in events] == [
            "release_manager",
            "close_store",
            "close_store",
            "close_coordinator",
        ]
    else:
        assert [event[0] for event in first_attempt] == ["release_manager", "close_store", "close_coordinator"]
        assert [event[0] for event in events] == [
            "release_manager",
            "close_store",
            "close_coordinator",
            "close_coordinator",
        ]


@pytest.mark.parametrize("replacement_token", [None, object()])
def test_close_retains_resources_when_manager_reservation_no_longer_matches(
    monkeypatch: pytest.MonkeyPatch,
    replacement_token: object | None,
) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    session = SharedOffloadSession("run")
    load(session, "model", events)
    owner_token = managers["model"].reservation
    managers["model"].reservation = replacement_token
    events.clear()

    with pytest.raises(RuntimeError, match="no longer owns"):
        session.close()

    assert events == []
    managers["model"].reservation = owner_token
    session.close()
    assert [event[0] for event in events] == ["release_manager", "close_store", "close_coordinator"]


def test_context_manager_propagates_cleanup_error_after_successful_body(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    session = SharedOffloadSession("run")
    load(session, "model", events)
    managers["model"].release_failures = 1

    with pytest.raises(RuntimeError, match="manager cleanup failed"), session:
        pass

    assert managers["model"].release_calls == [False]
    session.close()
    assert managers["model"].release_calls == [False, False]


def test_context_manager_preserves_body_error_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[tuple[Any, ...]] = []
    managers = install_fakes(monkeypatch, events=events, roles={manager_namespace("model"): True})
    session = SharedOffloadSession("run")
    load(session, "model", events)
    managers["model"].release_failures = 1
    primary_error = LookupError("body failed")

    with pytest.raises(LookupError) as exc_info, session:
        raise primary_error

    assert exc_info.value is primary_error
    if hasattr(BaseException, "add_note"):
        assert any("cleanup also failed" in note for note in getattr(exc_info.value, "__notes__", ()))
    cleanup_log = next(record for record in caplog.records if "in-flight exception" in record.getMessage())
    assert cleanup_log.levelname == "ERROR"
    assert cleanup_log.exc_info is not None
    assert isinstance(cleanup_log.exc_info[1], RuntimeError)
    assert str(cleanup_log.exc_info[1]) == "manager cleanup failed"
    assert cleanup_log.exc_info[2] is not None
    session.close()


@pytest.mark.parametrize("namespace", ["", "has space", "has/slash", ".leading", "x" * 97])
def test_session_rejects_unsafe_namespaces(namespace: str) -> None:
    with pytest.raises(ValueError, match="namespace"):
        SharedOffloadSession(namespace)


def test_validator_rejects_non_string_namespace() -> None:
    with pytest.raises(ValueError, match="namespace"):
        SharedOffloadSession.validate_options(Path("safe"))


@pytest.mark.parametrize("timeout", [None, True, False, -1, float("inf"), float("nan"), "bad"])
def test_session_requires_a_non_negative_finite_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="non-negative finite"):
        SharedOffloadSession("safe", wait_timeout=timeout)


@pytest.mark.parametrize(
    "config_update, error_match",
    [
        ({"enabled": False}, "enabled=True"),
        ({"transfer_mode": "per_tensor_transfer"}, "allocation_block_transfer"),
        ({"transfer_mode": "strategy"}, "allocation_block_transfer"),
        ({"transfer_mode": "raw_block_transfer"}, "allocation_block_transfer"),
    ],
)
def test_shared_offload_requires_enabled_allocation_block_config(
    monkeypatch: pytest.MonkeyPatch,
    config_update: dict[str, Any],
    error_match: str,
) -> None:
    session = SharedOffloadSession("run")
    load_profile = MagicMock()
    factory = MagicMock()
    monkeypatch.setattr(shared_offload.TensorManagerStateHandler, "load_from_file", load_profile)
    kwargs = {
        "profile_directory": "/profiles/model",
        "name": "model",
        "creator_model_factory": factory,
        "follower_model_factory": factory,
    }
    with pytest.raises(ValueError, match=error_match):
        session.offload_from_profile(config=config().model_copy(update=config_update), **kwargs)
    load_profile.assert_not_called()
    factory.assert_not_called()


def test_resident_parameter_names_validate_the_published_inventory() -> None:
    model = FakeModel("model")

    assert shared_offload._resident_parameter_names(  # noqa: SLF001
        model,
        ["resident.weight", "rotary_cache"],
    ) == ["resident.weight"]
    with pytest.raises(RuntimeError, match="duplicates"):
        shared_offload._resident_parameter_names(model, ["resident.weight", "resident.weight"])  # noqa: SLF001
    with pytest.raises(RuntimeError, match="not registered"):
        shared_offload._resident_parameter_names(model, ["missing"])  # noqa: SLF001


def test_session_rejects_cross_thread_and_post_fork_access(monkeypatch: pytest.MonkeyPatch) -> None:
    session = SharedOffloadSession("run")
    errors: list[BaseException] = []
    thread = threading.Thread(target=lambda: _capture_error(session.close, errors))
    thread.start()
    thread.join(timeout=2)

    assert len(errors) == 1
    assert "belongs to thread" in str(errors[0])

    monkeypatch.setattr(shared_offload.os, "getpid", lambda: session._owner_pid + 1)  # noqa: SLF001
    with pytest.raises(RuntimeError, match="after fork"):
        session.close()


def _capture_error(call: Any, errors: list[BaseException]) -> None:
    try:
        call()
    except BaseException as error:
        errors.append(error)

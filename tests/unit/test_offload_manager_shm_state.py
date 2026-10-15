# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for creator/follower initialization from restored SHM state."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from torch import nn

import flextensor.shared_offload as shared_offload
from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.config import OffloadConfig
from flextensor.loaders import PreallocatedLoader
from flextensor.offload_manager import OffloadManager, OffloadModelProxy, OffloadPhase
from flextensor.shared_offload import SharedOffloadSession
from flextensor.state_handler import TensorManagerState, TensorManagerStateHandler


def _state() -> TensorManagerState:
    weight = TensorStatistics(tensor_id=101, name="weight", size_bytes=16, load_time_ms=0.1)
    bias = TensorStatistics(tensor_id=102, name="bias", size_bytes=8, load_time_ms=0.1)
    return TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={101: "weight", 102: "bias"},
        allocation_ordered={0: ["layer"]},
        label_to_size_map={},
        block_sizes={0: 24},
        load_strategy={"layer": [weight, bias]},
        release_strategy={},
        label_to_block_id={"layer": 0},
        stats=[LayerStatistics(label="layer", tensors=[weight, bias], duration=1.0)],
        transfer_to_compute_map={"layer": "layer"},
        view_tensors_ids=[101, 102],
        view_tensors_names=["weight", "bias"],
        gpu_tensors_names=[],
        shm_block_name_map={"layer": "block_0"},
    )


def _configured_manager(name: str = "shm_restore") -> OffloadManager:
    manager = OffloadManager(name)
    manager.set_config(
        OffloadConfig(
            shm_enabled=True,
            shm_namespace="ft_stable",
            transfer_mode="allocation_block_transfer",
        )
    )
    return manager


class _FakeTensorManager:
    def __init__(self, phase_models: tuple[nn.Module, nn.Module, nn.Module]) -> None:
        self.use_shm = True
        self.shm_namespace: str | None = None
        self.events: list[object] = []
        self.phase_models = phase_models
        self.exported_state = _state()
        self.release_memory = MagicMock()
        self.shutdown = MagicMock()
        self.arm_non_destructive_first_loader = MagicMock()
        self.clear_replan_state = MagicMock()

    def restore_state(
        self,
        model: nn.Module,
        state: TensorManagerState,
        *,
        load_from_shm: bool = False,
    ) -> None:
        self.events.append(("restore", load_from_shm, self.shm_namespace, model, state))

    def initialize_warmup(self) -> nn.Module:
        self.events.append("warmup")
        return self.phase_models[0]

    def initialize_profile(self) -> nn.Module:
        self.events.append("profile")
        return self.phase_models[1]

    def initialize_inference(self) -> nn.Module:
        self.events.append("inference")
        return self.phase_models[2]

    def export_state(self) -> TensorManagerState:
        self.events.append("export")
        return self.exported_state


@pytest.mark.parametrize("load_from_shm", [False, True])
def test_restored_initializer_uses_explicit_role_and_common_phase_sequence(load_from_shm: bool) -> None:
    manager = _configured_manager()
    input_model = nn.Linear(2, 2)
    warmup_model = nn.Linear(2, 2)
    profile_model = nn.Linear(2, 2)
    inference_model = nn.Linear(2, 2)
    tensor_manager = _FakeTensorManager((warmup_model, profile_model, inference_model))
    manager._initialize_tensor_manager = MagicMock(
        side_effect=lambda: setattr(manager, "_tensor_manager", tensor_manager)
    )
    manager._iteration_count = 17

    with (
        patch.object(manager, "_offload_modules") as offload_modules,
        patch.object(manager, "_exclude_modules") as exclude_modules,
        patch.object(manager, "_check_no_modules_patched") as check_modules,
    ):
        state = _state()
        proxy = manager._restore_shared_offload(input_model, state, "ft_stable", load_from_shm=load_from_shm)

    assert isinstance(proxy, OffloadModelProxy)
    assert manager._model is inference_model
    assert proxy.__subject__ is inference_model
    assert manager._current_phase is OffloadPhase.INFERENCE
    assert manager._iteration_count == 0
    assert manager._tensor_manager is tensor_manager
    assert tensor_manager.events == [
        ("restore", load_from_shm, "ft_stable", input_model, state),
        "warmup",
        "profile",
        "inference",
    ]
    offload_modules.assert_called_once_with(inference_model, ["layer"])
    exclude_modules.assert_called_once_with(inference_model, manager.config.exclude_patterns)
    check_modules.assert_called_once_with()
    assert manager._state_hook_handle is not None
    tagged_hooks = [
        hook for hook in inference_model._forward_hooks.values() if getattr(hook, "_ft_state_update_hook", False)
    ]
    assert len(tagged_hooks) == 1
    tensor_manager.release_memory.assert_not_called()
    tensor_manager.shutdown.assert_not_called()


@pytest.mark.parametrize("is_creator", [False, True])
def test_compiled_shm_restore_only_patches_profiled_units(is_creator: bool) -> None:
    manager = _configured_manager("profiled_units")
    manager.set_config(manager.config.model_copy(update={"external_compile": True, "include_patterns": ["*"]}))
    model = nn.Sequential()
    for name in ("embed", "layer", "norm", "head"):
        model.add_module(name, nn.Linear(2, 2))
    inputs = torch.ones(1, 2)
    expected = model(inputs)
    tensor_manager = _FakeTensorManager((model, model, model))
    loader = MagicMock(spec=PreallocatedLoader)
    tensor_manager.tensor_layer_loader = loader
    manager._initialize_tensor_manager = MagicMock(
        side_effect=lambda: setattr(manager, "_tensor_manager", tensor_manager)
    )

    try:
        proxy = manager._restore_shared_offload(model, _state(), "ft_stable", load_from_shm=not is_creator)

        torch.testing.assert_close(proxy(inputs), expected)
        # Resident modules must not start transfer scopes outside the saved
        # first/last layer boundaries, including before a captured piece.
        assert manager.get_layer_label_by_idx() == ["layer"]
        assert loader.enter.call_args_list == [call("layer")]
        assert loader.exit.call_args_list == [call("layer")]
    finally:
        manager.release()


def test_compiled_session_publishes_original_names_for_follower(monkeypatch: pytest.MonkeyPatch) -> None:
    creator_model = nn.Sequential()
    creator_model.add_module("layer", nn.Linear(2, 2))
    creator_model.layer.register_buffer("cache", torch.tensor([3.0, 4.0]))
    follower_model = nn.Sequential()
    follower_model.add_module("layer", nn.Linear(2, 2, device="meta"))
    follower_model.layer.register_buffer("cache", torch.empty(2, device="meta"))
    state = TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={101: "layer.weight", 102: "layer.bias", 103: "layer.cache"},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={},
        release_strategy={},
        label_to_block_id={},
        stats=[LayerStatistics(label="layer", tensors=[], duration=1.0)],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=["layer.weight", "layer.bias", "layer.cache"],
        shm_block_name_map={},
    )
    managers = []
    for role, model in (("creator", creator_model), ("follower", follower_model)):
        manager = OffloadManager(f"compiled_session_{role}")
        tensor_manager = _FakeTensorManager((model, model, model))
        tensor_manager.exported_state = state
        tensor_manager.tensor_layer_loader = MagicMock(spec=PreallocatedLoader)
        monkeypatch.setattr(
            manager,
            "_initialize_tensor_manager",
            lambda manager=manager, tensor_manager=tensor_manager: setattr(manager, "_tensor_manager", tensor_manager),
        )
        managers.append(manager)
    manager_iter = iter(managers)
    monkeypatch.setattr(shared_offload, "get_offload_manager", lambda name: next(manager_iter))
    monkeypatch.setattr(TensorManagerStateHandler, "load_from_file", lambda path: state)
    namespace = f"compile_{uuid.uuid4().hex[:12]}"
    config = OffloadConfig(pinned_memory=False, transfer_mode="allocation_block_transfer")
    inputs = torch.ones(1, 2)
    expected = creator_model(inputs).detach()

    with SharedOffloadSession(namespace) as creator, SharedOffloadSession(namespace) as follower:

        def compile_fn(module: nn.Module) -> nn.Module:
            return torch.compile(module, backend="eager")

        creator_proxy = creator.offload_from_profile(
            "/profiles/model",
            config=config,
            creator_model_factory=lambda: creator_model,
            follower_model_factory=lambda: pytest.fail("expected creator"),
            materialize_device="cpu",
            compile_fn=compile_fn,
        )
        follower_proxy = follower.offload_from_profile(
            "/profiles/model",
            config=config,
            creator_model_factory=lambda: pytest.fail("expected follower"),
            follower_model_factory=lambda: follower_model,
            materialize_device="cpu",
            compile_fn=compile_fn,
        )

        for proxy in (creator_proxy, follower_proxy):
            assert "layer._orig_mod.cache" in dict(proxy.__subject__.named_buffers())
            torch.testing.assert_close(proxy.layer.cache, torch.tensor([3.0, 4.0]))
            torch.testing.assert_close(proxy(inputs), expected)


def test_compiled_shm_follower_disables_replan_but_keeps_compiled_activation(caplog) -> None:
    manager = _configured_manager("compiled_follower")
    manager.set_config(manager.config.model_copy(update={"external_compile": True}))
    model = nn.Linear(2, 2)
    tensor_manager = _FakeTensorManager((model, model, model))
    manager._initialize_tensor_manager = MagicMock(
        side_effect=lambda: setattr(manager, "_tensor_manager", tensor_manager)
    )

    with (
        patch.object(manager, "_offload_modules"),
        patch.object(manager, "_exclude_modules"),
        patch.object(manager, "_check_no_modules_patched"),
        patch.object(manager._compiled, "on_enter_inference"),
        caplog.at_level("WARNING"),
    ):
        manager._restore_shared_offload(model, _state(), "ft_stable", load_from_shm=True)

    assert manager._compiled.active is True
    assert manager._compiled.replan_active is False
    tensor_manager.arm_non_destructive_first_loader.assert_not_called()
    assert "strategy replan is disabled for coordinated SHM loads" in caplog.text


def test_compiled_shm_creator_retains_replan_and_source_snapshot_arming() -> None:
    manager = _configured_manager("compiled_creator")
    manager.set_config(manager.config.model_copy(update={"external_compile": True}))
    model = nn.Linear(2, 2)
    tensor_manager = _FakeTensorManager((model, model, model))
    manager._initialize_tensor_manager = MagicMock(
        side_effect=lambda: setattr(manager, "_tensor_manager", tensor_manager)
    )

    with (
        patch.object(manager, "_offload_modules"),
        patch.object(manager, "_exclude_modules"),
        patch.object(manager, "_check_no_modules_patched"),
        patch.object(manager._compiled, "on_enter_inference"),
    ):
        manager._restore_shared_offload(model, _state(), "ft_stable", load_from_shm=False)

    assert manager._compiled.active is True
    assert manager._compiled.replan_active is True
    tensor_manager.arm_non_destructive_first_loader.assert_called_once_with()


@pytest.mark.parametrize("load_from_shm", [False, True])
def test_restored_initializer_rejects_namespace_mismatch_before_manager_creation(load_from_shm: bool) -> None:
    manager = _configured_manager()
    manager._initialize_tensor_manager = MagicMock()

    with pytest.raises(ValueError, match="namespace"):
        manager._restore_shared_offload(nn.Linear(2, 2), _state(), "ft_other", load_from_shm=load_from_shm)

    manager._initialize_tensor_manager.assert_not_called()


def test_restored_initializer_rejects_preinitialized_tensor_manager() -> None:
    manager = _configured_manager("reuse")
    model = nn.Linear(2, 2)
    tensor_manager = _FakeTensorManager((model, model, model))
    tensor_manager.shm_namespace = "ft_stable"
    manager._tensor_manager = tensor_manager
    manager._initialize_tensor_manager = MagicMock()

    with pytest.raises(RuntimeError, match=r"active.*release\(\)"):
        manager._restore_shared_offload(model, _state(), "ft_stable", load_from_shm=False)

    manager._initialize_tensor_manager.assert_not_called()
    assert manager._tensor_manager is tensor_manager
    tensor_manager.release_memory.assert_not_called()
    tensor_manager.shutdown.assert_not_called()


def test_restored_initializer_rejects_active_runtime_without_mutating_old_state() -> None:
    manager = _configured_manager("active_rejected")
    old_model = nn.Linear(2, 2)
    tensor_manager = _FakeTensorManager((nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)))
    tensor_manager.shm_namespace = "ft_stable"
    manager._tensor_manager = tensor_manager
    manager._model = old_model
    manager._model_proxy = OffloadModelProxy(old_model, manager)
    manager._current_phase = OffloadPhase.INFERENCE
    manager._iteration_count = 11
    manager._initialize_tensor_manager = MagicMock()

    def user_hook(_module: nn.Module, _inputs: object, output: object) -> object:
        return output

    old_model.register_forward_hook(user_hook)

    with pytest.raises(RuntimeError, match=r"active.*release\(\)"):
        manager._restore_shared_offload(old_model, _state(), "ft_stable", load_from_shm=False)

    assert manager._tensor_manager is tensor_manager
    assert manager._model is old_model
    assert manager._model_proxy.__subject__ is old_model
    assert manager._current_phase is OffloadPhase.INFERENCE
    assert manager._iteration_count == 11
    assert user_hook in old_model._forward_hooks.values()
    assert tensor_manager.events == []
    manager._initialize_tensor_manager.assert_not_called()
    tensor_manager.release_memory.assert_not_called()
    tensor_manager.shutdown.assert_not_called()


def test_restored_initializer_cleans_fresh_candidate_after_loader_initialization_failure() -> None:
    manager = _configured_manager("loader_failure")
    model = nn.Linear(2, 2)
    tensor_manager = _FakeTensorManager((model, model, model))
    tensor_manager.initialize_warmup = MagicMock(side_effect=RuntimeError("loader initialization failed"))
    manager._initialize_tensor_manager = MagicMock(
        side_effect=lambda: setattr(manager, "_tensor_manager", tensor_manager)
    )

    with pytest.raises(RuntimeError, match="loader initialization failed"):
        manager._restore_shared_offload(model, _state(), "ft_stable", load_from_shm=False)

    tensor_manager.release_memory.assert_not_called()
    tensor_manager.shutdown.assert_called_once_with(preserve_model=False)
    assert manager._tensor_manager is None
    assert manager._model is None
    assert manager._model_proxy is None
    assert manager._current_phase is OffloadPhase.NOT_INITIALIZED


def test_restored_initializer_unpatches_partial_modules_and_cleans_candidate() -> None:
    class _PatchModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

    manager = _configured_manager("patch_failure")
    model = _PatchModel()
    tensor_manager = _FakeTensorManager((model, model, model))
    manager._initialize_tensor_manager = MagicMock(
        side_effect=lambda: setattr(manager, "_tensor_manager", tensor_manager)
    )

    def patch_then_fail(_model: nn.Module, _patterns: list[str]) -> None:
        manager._patch_module_forward(model.first, "first")
        raise RuntimeError("patch failed")

    manager._offload_modules = MagicMock(side_effect=patch_then_fail)

    with pytest.raises(RuntimeError, match="patch failed"):
        manager._restore_shared_offload(model, _state(), "ft_stable", load_from_shm=False)

    assert "_ft_original_forward_func" not in model.first.__dict__
    assert manager._patched_modules == []
    tensor_manager.release_memory.assert_not_called()
    tensor_manager.shutdown.assert_called_once_with(preserve_model=False)
    assert manager._tensor_manager is None


def test_restored_initializer_rolls_back_partial_swap_and_removes_new_hook() -> None:
    manager = _configured_manager("swap_failure")
    model = nn.Linear(2, 2)
    tensor_manager = _FakeTensorManager((model, model, model))
    manager._initialize_tensor_manager = MagicMock(
        side_effect=lambda: setattr(manager, "_tensor_manager", tensor_manager)
    )

    def existing_tagged_hook(_module: nn.Module, _inputs: object, output: object) -> object:
        return output

    existing_tagged_hook._ft_state_update_hook = True  # type: ignore[attr-defined]  # noqa: SLF001
    model.register_forward_hook(existing_tagged_hook)

    def partial_swap_then_fail(new_model: nn.Module, _phase: OffloadPhase) -> None:
        manager._model = new_model
        assert manager._model_proxy is not None
        manager._model_proxy.__subject__ = new_model
        manager._current_phase = OffloadPhase.INFERENCE
        manager._install_state_update_hook()
        raise RuntimeError("swap failed")

    manager._swap_to_new_model = MagicMock(side_effect=partial_swap_then_fail)

    with (
        patch.object(manager, "_offload_modules"),
        patch.object(manager, "_exclude_modules"),
        patch.object(manager, "_check_no_modules_patched"),
        pytest.raises(RuntimeError, match="swap failed"),
    ):
        manager._restore_shared_offload(model, _state(), "ft_stable", load_from_shm=False)

    tagged_hooks = [hook for hook in model._forward_hooks.values() if getattr(hook, "_ft_state_update_hook", False)]
    assert tagged_hooks == [existing_tagged_hook]
    assert manager._state_hook_handle is None
    assert manager._tensor_manager is None
    assert manager._model is None
    assert manager._model_proxy is None
    assert manager._current_phase is OffloadPhase.NOT_INITIALIZED
    tensor_manager.release_memory.assert_not_called()
    tensor_manager.shutdown.assert_called_once_with(preserve_model=False)


@pytest.mark.parametrize("reuse_method", ["init", "offload", "offload_from_state"])
def test_restored_initializer_cleanup_preserves_primary_error_and_blocks_reuse(reuse_method: str) -> None:
    class _FatalRestoreError(BaseException):
        pass

    manager = _configured_manager("cleanup_failure")
    model = nn.Linear(2, 2)
    tensor_manager = _FakeTensorManager((model, model, model))
    tensor_manager.initialize_warmup = MagicMock(side_effect=_FatalRestoreError("primary failure"))
    tensor_manager.shutdown.side_effect = [
        RuntimeError("rollback shutdown cleanup failed"),
        RuntimeError("retry shutdown cleanup failed"),
        None,
    ]
    manager._initialize_tensor_manager = MagicMock(
        side_effect=lambda: setattr(manager, "_tensor_manager", tensor_manager)
    )

    with pytest.raises(_FatalRestoreError, match="primary failure"):
        manager._restore_shared_offload(model, _state(), "ft_stable", load_from_shm=False)

    tensor_manager.release_memory.assert_not_called()
    tensor_manager.shutdown.assert_called_once_with(preserve_model=False)
    assert manager._tensor_manager is None
    assert manager._failed_restored_tensor_manager is tensor_manager

    original_config = manager.config
    next_config = original_config.model_copy(update={"shm_namespace": "next_run"})
    reuse_args = {"init": (), "offload": (model,), "offload_from_state": (model, _state())}[reuse_method]
    with pytest.raises(RuntimeError, match=r"candidate cleanup is pending.*release\(\)"):
        getattr(manager, reuse_method)(*reuse_args, config=next_config)

    with pytest.raises(RuntimeError, match="cleanup remains pending"):
        manager.release()

    with pytest.raises(RuntimeError, match=r"candidate cleanup is pending.*release\(\)"):
        getattr(manager, reuse_method)(*reuse_args, config=next_config)

    assert manager.config is original_config
    assert manager._tensor_manager is None
    assert manager._failed_restored_tensor_manager is tensor_manager
    manager._initialize_tensor_manager.assert_called_once_with()

    manager.release()

    assert tensor_manager.shutdown.call_args_list == [call(preserve_model=False)] * 3
    assert manager._failed_restored_tensor_manager is None

    replacement = _FakeTensorManager((model, model, model))
    manager._initialize_tensor_manager.side_effect = lambda: setattr(manager, "_tensor_manager", replacement)
    manager.init(next_config)

    assert manager._initialize_tensor_manager.call_count == 2
    assert manager._tensor_manager is replacement
    assert manager.config is next_config
    manager.release()


def test_session_reservation_blocks_public_release_until_owned_cleanup() -> None:
    manager = _configured_manager("session_reservation")
    token = object()
    manager._prepare_shared_offload(manager.config, owner_token=token)

    with pytest.raises(RuntimeError, match="active SharedOffloadSession"):
        manager.release()

    assert manager._shared_offload_owner_token is token
    assert manager._release_shared_offload(owner_token=token, preserve_model=False) is None
    assert manager._shared_offload_owner_token is None
    manager.release()


@pytest.mark.parametrize(
    "reservation, owner_token",
    [
        pytest.param(object(), object(), id="wrong-token"),
        pytest.param({}, {}, id="equal-but-not-identical"),
        pytest.param(object(), None, id="missing-caller-token"),
        pytest.param(None, object(), id="no-reservation"),
        pytest.param(None, None, id="no-reservation-or-token"),
    ],
)
def test_shared_release_rejects_invalid_ownership_without_releasing(
    reservation: object | None,
    owner_token: object | None,
) -> None:
    manager = _configured_manager("invalid_owner")
    if reservation is not None:
        manager._prepare_shared_offload(manager.config, owner_token=reservation)
    manager.release = MagicMock()

    with pytest.raises(RuntimeError, match="no longer owns"):
        manager._release_shared_offload(owner_token=owner_token, preserve_model=False)

    manager.release.assert_not_called()
    assert manager._shared_offload_owner_token is reservation


def test_matching_session_owner_can_retry_transient_manager_cleanup() -> None:
    manager = _configured_manager("session_cleanup_retry")
    token = object()
    manager._prepare_shared_offload(manager.config, owner_token=token)
    tensor_manager = MagicMock()
    tensor_manager.release_memory.side_effect = [RuntimeError("transient release failure"), None]
    manager._tensor_manager = tensor_manager

    with pytest.raises(RuntimeError, match="transient release failure"):
        manager._release_shared_offload(owner_token=token, preserve_model=False)

    assert manager._shared_offload_owner_token is token
    assert manager._cleanup_blocked is True

    assert manager._release_shared_offload(owner_token=token, preserve_model=False) is None
    assert manager._shared_offload_owner_token is None
    assert manager._tensor_manager is None
    assert manager._cleanup_blocked is False


def test_export_runtime_state_delegates_to_live_tensor_manager_export() -> None:
    manager = _configured_manager("export")
    saved_attribute = _state()
    exported = _state()
    exported.shm_block_name_map = {"layer": "live_block"}
    tensor_manager = MagicMock()
    tensor_manager.tensor_manager_state = saved_attribute
    tensor_manager.export_state.return_value = exported
    manager._tensor_manager = tensor_manager

    result = manager._export_runtime_state()

    assert result is exported
    assert result is not saved_attribute
    tensor_manager.export_state.assert_called_once_with()


def test_export_runtime_state_requires_initialized_tensor_manager() -> None:
    manager = _configured_manager("export_missing")

    with pytest.raises(RuntimeError, match="TensorManager"):
        manager._export_runtime_state()


class _Coordinator:
    def __init__(self, state: TensorManagerState) -> None:
        self.namespace = "ft_stable"
        self.is_creator = False
        self.wait_for_ready = MagicMock()
        self.read_profile = MagicMock(return_value=state)


def test_coordinator_initializer_is_thin_wait_read_delegate() -> None:
    manager = _configured_manager("coordinator")
    manager.set_config(manager.config.model_copy(update={"shm_wait_timeout": 12.0}))
    state = _state()
    model = nn.Linear(2, 2)
    coordinator = _Coordinator(state)
    proxy = OffloadModelProxy(model, manager)
    manager._restore_shared_offload = MagicMock(return_value=proxy)

    result = manager._initialize_from_shm(coordinator, model)

    assert result is proxy
    coordinator.wait_for_ready.assert_called_once_with(timeout=12.0)
    coordinator.read_profile.assert_called_once_with()
    manager._restore_shared_offload.assert_called_once_with(model, state, "ft_stable", load_from_shm=True)


def test_restored_initializer_names_every_remaining_meta_parameter_and_buffer() -> None:
    class _MetaModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(2, 2, device="meta")
            self.register_buffer("cache", torch.empty(2, device="meta"))

    manager = _configured_manager("meta_failure")
    model = _MetaModel()
    tensor_manager = _FakeTensorManager((model, model, model))
    manager._initialize_tensor_manager = MagicMock(
        side_effect=lambda: setattr(manager, "_tensor_manager", tensor_manager)
    )

    with pytest.raises(RuntimeError) as error:
        manager._restore_shared_offload(model, _state(), "ft_stable", load_from_shm=True)

    message = str(error.value)
    assert "layer.weight" in message
    assert "layer.bias" in message
    assert "cache" in message
    assert manager._model_proxy is None
    assert manager._state_hook_handle is None
    assert manager._current_phase is OffloadPhase.NOT_INITIALIZED

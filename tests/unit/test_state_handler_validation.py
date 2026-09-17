# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TensorManagerStateHandler validation functionality.

This test suite validates the state compatibility checking between saved profiles
and current models. It verifies:

1. Validation detection: Correctly identifies missing and unexpected tensor names
2. Strict mode: Raises ValueError when tensors are missing from the model
3. Non-strict mode: Logs warnings and skips missing tensors
4. restore_state integration: Validation is properly called during restore
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from flextensor import state_handler as state_handler_module
from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.host_pinning import HostPinner
from flextensor.loaders import _compute_preload
from flextensor.state_handler import (
    StateValidationResult,
    TensorManagerState,
    TensorManagerStateHandler,
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing with known parameter names."""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 20)
        self.layer2 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


def create_tensor_statistics(names: list[str], size_by_name: dict[str, int] | None = None) -> list[TensorStatistics]:
    """Create a list of TensorStatistics with given names."""
    size_by_name = size_by_name or {}
    return [
        TensorStatistics(tensor_id=i, name=name, size_bytes=size_by_name.get(name, 1000), load_time_ms=0.1)
        for i, name in enumerate(names)
    ]


def create_layer_statistics(names: list[str], size_by_name: dict[str, int] | None = None) -> list[LayerStatistics]:
    """Create a LayerStatistics with given tensor names."""
    tensors = create_tensor_statistics(names, size_by_name)
    return [LayerStatistics(label="layer_0", tensors=tensors, duration=1.0)]


def create_mock_state(
    tensor_names: list[str],
    view_tensor_names: list[str] | None = None,
    model: torch.nn.Module | dict | None = None,
) -> TensorManagerState:
    """Create a mock TensorManagerState with the given tensor names."""
    if view_tensor_names is None:
        view_tensor_names = []
    tensor_names = list(tensor_names)
    inventory_names = [*tensor_names, *(name for name in view_tensor_names if name not in tensor_names)]
    tensor_id_by_name = {name: tensor_id for tensor_id, name in enumerate(inventory_names)}
    if isinstance(model, torch.nn.Module):
        live_tensors = dict(model.named_parameters(remove_duplicate=False))
        live_tensors.update(model.named_buffers(remove_duplicate=False))
    else:
        live_tensors = model or {}
    size_by_name = {
        name: tensor.numel() * tensor.element_size()
        for name, tensor in live_tensors.items()
        if isinstance(tensor, torch.Tensor)
    }

    return TensorManagerState(
        loader_type="strategy",
        tensor_id_to_name_map=dict(enumerate(inventory_names)),
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={"layer_0": create_tensor_statistics(tensor_names, size_by_name)},
        release_strategy={"layer_0": create_tensor_statistics(tensor_names, size_by_name)},
        label_to_block_id={},
        stats=create_layer_statistics(tensor_names, size_by_name),
        transfer_to_compute_map={},
        view_tensors_ids=[tensor_id_by_name[name] for name in view_tensor_names],
        view_tensors_names=view_tensor_names,
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )


def test_from_dict_rejects_duplicate_inventory_names() -> None:
    data = create_mock_state(["first", "second"]).to_dict()
    data["tensor_id_to_name_map"]["1"] = "first"

    with pytest.raises(ValueError, match=r"inventory names.*unique"):
        TensorManagerState.from_dict(data)


def test_from_dict_rejects_unknown_loader_type() -> None:
    data = create_mock_state(["weight"]).to_dict()
    data["loader_type"] = "unknown"

    with pytest.raises(ValueError, match="Unknown loader type"):
        TensorManagerState.from_dict(data)


def test_from_dict_rejects_statistic_identity_mismatch() -> None:
    data = create_mock_state(["weight"]).to_dict()
    data["load_strategy"]["layer_0"][0]["tensor_id"] = -1

    with pytest.raises(ValueError, match=r"TensorStatistics.*tensor_id.*name"):
        TensorManagerState.from_dict(data)


class TestStateValidationResult:
    """Tests for the StateValidationResult dataclass."""

    def test_empty_result_is_falsy(self):
        """An empty result (no missing/unexpected) should be falsy."""
        result = StateValidationResult()
        assert not result
        assert result.missing_keys == []
        assert result.unexpected_keys == []

    def test_result_with_missing_keys_is_truthy(self):
        """A result with missing keys should be truthy."""
        result = StateValidationResult(missing_keys=["layer.weight"])
        assert result
        assert result.missing_keys == ["layer.weight"]
        assert result.unexpected_keys == []

    def test_result_with_unexpected_keys_is_truthy(self):
        """A result with unexpected keys should be truthy."""
        result = StateValidationResult(unexpected_keys=["new_layer.weight"])
        assert result
        assert result.missing_keys == []
        assert result.unexpected_keys == ["new_layer.weight"]

    def test_result_with_both_is_truthy(self):
        """A result with both missing and unexpected keys should be truthy."""
        result = StateValidationResult(
            missing_keys=["old.weight"],
            unexpected_keys=["new.weight"],
        )
        assert result


class TestValidateStateCompatibility:
    """Tests for the validate_state_compatibility method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        # Model has: layer1.weight, layer1.bias, layer2.weight, layer2.bias
        self.model_tensor_names = {name for name, _ in self.model.named_parameters()}

        # Create mock tensor manager
        self.mock_tm = MagicMock()
        self.handler = TensorManagerStateHandler(self.mock_tm)

    def test_exact_match_no_issues(self):
        """When state tensors exactly match model tensors, no issues reported."""
        state = create_mock_state(list(self.model_tensor_names))

        result = self.handler.validate_state_compatibility(self.model, state)

        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert not result

    def test_missing_keys_detected(self):
        """Tensors in state but not in model are reported as missing."""
        # Add a tensor name that doesn't exist in the model
        tensor_names = [*self.model_tensor_names, "nonexistent.weight", "another.bias"]
        state = create_mock_state(tensor_names)

        result = self.handler.validate_state_compatibility(self.model, state)

        assert "nonexistent.weight" in result.missing_keys
        assert "another.bias" in result.missing_keys
        assert len(result.missing_keys) == 2

    def test_unexpected_keys_detected(self):
        """Tensors in model but not in state are reported as unexpected."""
        # Only include some of the model's tensors in the state
        partial_names = ["layer1.weight", "layer1.bias"]
        state = create_mock_state(partial_names)

        result = self.handler.validate_state_compatibility(self.model, state)

        assert "layer2.weight" in result.unexpected_keys
        assert "layer2.bias" in result.unexpected_keys
        assert len(result.unexpected_keys) == 2

    def test_both_missing_and_unexpected(self):
        """Both missing and unexpected keys can be detected simultaneously."""
        # State has some tensors model doesn't, model has some tensors state doesn't
        state_names = ["layer1.weight", "old_layer.weight"]
        state = create_mock_state(state_names)

        result = self.handler.validate_state_compatibility(self.model, state)

        assert "old_layer.weight" in result.missing_keys
        assert "layer1.bias" in result.unexpected_keys
        assert "layer2.weight" in result.unexpected_keys
        assert "layer2.bias" in result.unexpected_keys

    def test_view_tensors_included_in_validation(self):
        """View tensor names from state are also validated."""
        # Model tensors plus a view tensor that doesn't exist
        state = create_mock_state(
            list(self.model_tensor_names),
            view_tensor_names=["missing_view.weight"],
        )

        result = self.handler.validate_state_compatibility(self.model, state)

        assert "missing_view.weight" in result.missing_keys

    def test_dict_model_supported(self):
        """Validation works with dict models (state_dict style)."""
        model_dict = {name: torch.randn(10) for name in ["a.weight", "b.weight"]}
        state = create_mock_state(["a.weight", "c.weight"])

        result = self.handler.validate_state_compatibility(model_dict, state)

        assert "c.weight" in result.missing_keys
        assert "b.weight" in result.unexpected_keys

    def test_keys_are_sorted(self):
        """Missing and unexpected keys are returned sorted for consistency."""
        state_names = ["z.weight", "a.weight", "m.weight"]
        state = create_mock_state(state_names)

        result = self.handler.validate_state_compatibility(self.model, state)

        # Missing keys should be sorted
        assert result.missing_keys == sorted(result.missing_keys)
        # Unexpected keys should be sorted
        assert result.unexpected_keys == sorted(result.unexpected_keys)


class TestCheckValidationResult:
    """Tests for the _check_validation_result static method."""

    def test_no_issues_does_not_raise(self):
        """When there are no issues, no exception is raised."""
        result = StateValidationResult()
        # Should not raise
        TensorManagerStateHandler._check_validation_result(result, strict=True)

    def test_missing_keys_strict_raises_value_error(self):
        """In strict mode, missing keys raise ValueError."""
        result = StateValidationResult(missing_keys=["layer.weight", "layer.bias"])

        with pytest.raises(ValueError) as exc_info:
            TensorManagerStateHandler._check_validation_result(result, strict=True)

        error_msg = str(exc_info.value)
        assert "2 tensor(s)" in error_msg
        assert "layer.weight" in error_msg
        assert "strict=False" in error_msg

    def test_missing_keys_non_strict_logs_warning(self, caplog):
        """In non-strict mode, missing keys log a warning but don't raise."""
        result = StateValidationResult(missing_keys=["layer.weight"])

        with caplog.at_level(logging.WARNING):
            # Should not raise
            TensorManagerStateHandler._check_validation_result(result, strict=False)

        assert "1 tensor(s)" in caplog.text
        assert "layer.weight" in caplog.text
        assert "Skipping" in caplog.text

    def test_unexpected_keys_logs_info(self, caplog):
        """Unexpected keys log an info message (both strict and non-strict)."""
        result = StateValidationResult(unexpected_keys=["new.weight", "new.bias"])

        with caplog.at_level(logging.INFO):
            TensorManagerStateHandler._check_validation_result(result, strict=True)

        assert "2 tensor(s)" in caplog.text
        assert "new.weight" in caplog.text

    def test_many_missing_keys_shows_sample_with_ellipsis(self):
        """When there are many missing keys, only a sample is shown."""
        many_keys = [f"layer{i}.weight" for i in range(10)]
        result = StateValidationResult(missing_keys=many_keys)

        with pytest.raises(ValueError) as exc_info:
            TensorManagerStateHandler._check_validation_result(result, strict=True)

        error_msg = str(exc_info.value)
        assert "10 tensor(s)" in error_msg
        assert "..." in error_msg  # Ellipsis indicates truncation


class TestLoaderTypeMismatch:
    """Tests for loader_type mismatch detection in restore_state."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model_tensor_names = [name for name, _ in self.model.named_parameters()]

    def test_restore_state_raises_on_loader_type_mismatch(self):
        """restore_state raises ValueError when loader_type doesn't match."""
        # Create mock tensor manager with strategy loader type
        mock_tm = MagicMock()
        mock_tm.use_trace_tensor = True
        mock_tm.loader_type = "strategy"

        handler = TensorManagerStateHandler(mock_tm)

        # Create state with different loader_type (allocation_block_transfer)
        state = create_mock_state([])
        state.loader_type = "allocation_block_transfer"

        with pytest.raises(ValueError) as exc_info:
            handler.restore_state(self.model, state)

        error_msg = str(exc_info.value)
        assert "Saved profile uses loader_type='allocation_block_transfer'" in error_msg
        assert "TensorManager is configured with loader_type='strategy'" in error_msg
        assert "incompatible tensor mappings" in error_msg
        assert "re-profile" in error_msg

    def test_restore_state_succeeds_on_matching_loader_type(self):
        """restore_state succeeds when loader_type matches."""
        mock_tm = MagicMock()
        mock_tm.use_trace_tensor = True
        mock_tm.loader_type = "allocation_block_transfer"

        handler = TensorManagerStateHandler(mock_tm)

        state = create_mock_state([])
        state.loader_type = "allocation_block_transfer"

        # Should not raise
        result = handler.restore_state(self.model, state, strict=True)
        assert isinstance(result, StateValidationResult)

    def test_restore_state_loader_type_mismatch_all_combinations(self):
        """Test loader_type mismatch detection for various loader type combinations."""
        loader_types = ["strategy", "allocation_block_transfer", "raw_block_transfer"]

        for manager_loader_type in loader_types:
            for state_loader_type in loader_types:
                mock_tm = MagicMock()
                mock_tm.use_trace_tensor = True
                mock_tm.loader_type = manager_loader_type

                handler = TensorManagerStateHandler(mock_tm)

                state = create_mock_state([])
                state.loader_type = state_loader_type

                if manager_loader_type == state_loader_type:
                    # Should not raise
                    handler.restore_state(self.model, state, strict=True)
                else:
                    # Should raise
                    with pytest.raises(ValueError, match="Saved profile uses loader_type="):
                        handler.restore_state(self.model, state, strict=True)


class TestRestoreStateWithValidation:
    """Integration tests for restore_state with validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model_tensor_names = [name for name, _ in self.model.named_parameters()]

        # Create mock tensor manager with required attributes
        self.mock_tm = MagicMock()
        self.mock_tm.use_trace_tensor = True  # Skip preprocess_model
        self.mock_tm.loader_type = "strategy"

        self.handler = TensorManagerStateHandler(self.mock_tm)

    @pytest.mark.parametrize("as_dict", [False, True])
    def test_restore_state_keeps_permanent_parameters_available_for_preload(self, as_dict):
        """The strategy loader must receive the parameters its preload plan references."""
        state = create_mock_state(self.model_tensor_names, model=self.model)
        permanent_names = {"layer1.weight", "layer1.bias"}
        for strategy in (state.load_strategy, state.release_strategy):
            strategy["layer_0"] = [stat for stat in strategy["layer_0"] if stat.name not in permanent_names]
        state.gpu_tensors_names = sorted(permanent_names)
        state.validate_internal()
        parameters = dict(self.model.named_parameters())

        self.handler.restore_state(parameters if as_dict else self.model, state)

        restored = self.mock_tm.tensor_manager_state
        preload_ids = _compute_preload(restored.stats, restored.load_strategy)
        assert preload_ids == {id(parameters[name]) for name in permanent_names}
        for name in permanent_names:
            tensor_id = id(parameters[name])
            assert self.mock_tm.tensors_map[tensor_id] is parameters[name]
            assert tensor_id in self.mock_tm.traced_tensors

    def test_restore_state_strict_raises_on_missing(self):
        """restore_state with strict=True raises on missing tensors."""
        state = create_mock_state([*self.model_tensor_names, "missing.weight"], model=self.model)

        with pytest.raises(ValueError, match=r"missing\.weight"):
            self.handler.restore_state(self.model, state, strict=True)

    def test_restore_state_non_strict_skips_missing(self, caplog):
        """restore_state with strict=False skips missing tensors."""
        extra_tensor = "missing.weight"
        state = create_mock_state([*self.model_tensor_names, extra_tensor], model=self.model)

        with caplog.at_level(logging.WARNING):
            result = self.handler.restore_state(self.model, state, strict=False)

        # Should have logged a warning
        assert "missing.weight" in caplog.text

        # Result should indicate what was missing
        assert extra_tensor in result.missing_keys

        # The restored state (deep-copied inside restore_state) should exclude the missing tensor
        restored_state = self.mock_tm.tensor_manager_state
        for s_stats in restored_state.load_strategy.values():
            tensor_names = [s.name for s in s_stats]
            assert extra_tensor not in tensor_names

    def test_restore_state_returns_validation_result(self):
        """restore_state returns StateValidationResult for inspection."""
        # Use exact match - should have no missing but maybe unexpected
        partial_names = ["layer1.weight", "layer1.bias"]
        state = create_mock_state(partial_names, model=self.model)

        result = self.handler.restore_state(self.model, state, strict=True)

        assert isinstance(result, StateValidationResult)
        assert result.missing_keys == []
        assert "layer2.weight" in result.unexpected_keys

    def test_restore_state_default_is_strict(self):
        """restore_state defaults to strict=True."""
        state = create_mock_state([*self.model_tensor_names, "missing.weight"], model=self.model)

        # Should raise without explicit strict parameter
        with pytest.raises(ValueError):
            self.handler.restore_state(self.model, state)

    def test_restore_state_rejects_renamed_persisted_buffer(self):
        self.model.register_buffer("renamed_constant", torch.ones(1))
        state = create_mock_state(self.model_tensor_names, model=self.model)
        state.tensor_id_to_name_map[-1] = "constant"
        state.gpu_tensors_names = ["constant"]
        persisted_state = TensorManagerState.from_dict(state.to_dict())

        with pytest.raises(ValueError, match="constant"):
            self.handler.restore_state(self.model, persisted_state, strict=True)

    def test_restore_state_rejects_managed_buffer(self):
        self.model.register_buffer("constant", torch.ones(1))
        names = [*self.model_tensor_names, "constant"]
        state = create_mock_state(names, model=self.model)

        with pytest.raises(ValueError, match="buffer"):
            self.handler.restore_state(self.model, state, strict=True)

    def test_restore_state_rejects_empty_shm_map_when_shm_is_enabled(self):
        state = create_mock_state(self.model_tensor_names, model=self.model)
        state.loader_type = "allocation_block_transfer"
        state.release_strategy = {}
        state.allocation_ordered = {0: ["layer_0"]}
        state.block_sizes = {0: sum(stat.size_bytes for stat in state.load_strategy["layer_0"])}
        state.label_to_block_id = {"layer_0": 0}
        state.transfer_to_compute_map = {"layer_0": "layer_0"}
        state.view_tensors_ids = list(state.tensor_id_to_name_map)
        state.view_tensors_names = self.model_tensor_names
        state.shm_block_name_map = {}
        self.mock_tm.loader_type = "allocation_block_transfer"
        self.mock_tm.use_shm = True

        with pytest.raises(ValueError, match="shm_block_name_map"):
            self.handler.restore_state(self.model, state, strict=True)

    def test_restore_state_rejects_shared_storage_for_block_loader(self):
        storage = torch.arange(8, dtype=torch.float32)
        model = torch.nn.Module()
        model.register_parameter("first", torch.nn.Parameter(storage[:4], requires_grad=False))
        model.register_parameter("second", torch.nn.Parameter(storage[2:6], requires_grad=False))
        state = create_mock_state(["first", "second"], model=model)
        state.loader_type = "allocation_block_transfer"
        state.allocation_ordered = {0: ["layer_0"]}
        state.block_sizes = {0: sum(stat.size_bytes for stat in state.load_strategy["layer_0"])}
        state.label_to_block_id = {"layer_0": 0}
        state.transfer_to_compute_map = {"layer_0": "layer_0"}
        state.view_tensors_ids = [0, 1]
        state.view_tensors_names = ["first", "second"]
        self.mock_tm.loader_type = "allocation_block_transfer"

        with pytest.raises(ValueError, match="sharing storage"):
            self.handler.restore_state(model, state, strict=True)

    def test_restore_state_rejects_shared_storage_split_between_destinations(self):
        storage = torch.arange(8, dtype=torch.float32)
        model = torch.nn.Module()
        model.register_parameter("managed", torch.nn.Parameter(storage[:4], requires_grad=False))
        model.register_parameter("gpu", torch.nn.Parameter(storage[2:6], requires_grad=False))
        state = create_mock_state(["managed", "gpu"], model=model)
        state.load_strategy["layer_0"] = [state.load_strategy["layer_0"][0]]
        state.release_strategy["layer_0"] = [state.release_strategy["layer_0"][0]]
        state.stats[0] = state.stats[0].model_copy(update={"tensors": [state.stats[0].tensors[0]]})
        state.gpu_tensors_names = ["gpu"]

        with pytest.raises(ValueError, match="Contradictory alias destinations"):
            self.handler.restore_state(model, state, strict=True)

    def test_restore_state_rejects_meta_tensor_before_mutation(self):
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.empty(1, device="meta"), requires_grad=False))
        state = create_mock_state(["weight"], model=model)

        with pytest.raises(ValueError, match="meta tensor"):
            self.handler.restore_state(model, state, strict=True)

    def test_restore_state_filters_view_tensors(self):
        """restore_state filters out missing view tensor names."""
        state = create_mock_state(
            self.model_tensor_names,
            view_tensor_names=["valid.view", "missing.view"],
            model=self.model,
        )
        # Add valid.view to model for partial match
        # (In this case, both will be filtered since SimpleModel doesn't have them)

        _result = self.handler.restore_state(self.model, state, strict=False)

        # The restored state (deep-copied inside restore_state) should have filtered view tensors
        # Since neither exists in SimpleModel, both should be removed
        restored_state = self.mock_tm.tensor_manager_state
        assert "missing.view" not in restored_state.view_tensors_names
        assert "valid.view" not in restored_state.view_tensors_names


class TestRestoreStateForwardsHostPinner:
    """restore_state must forward the manager's HostPinner into preprocess_model.

    Regression guard: the HostPinner registry is owned by TensorManager and must
    be reused across the system. If restore_state ever constructs a fresh
    HostPinner() instead of forwarding ``tm.host_pinner``, host_register-mode
    pinnings would not be tracked by the manager's registry.
    """

    def setup_method(self):
        self.model = SimpleModel()
        self.model_tensor_names = [name for name, _ in self.model.named_parameters()]

        self.host_pinner = HostPinner()
        self.mock_tm = MagicMock()
        self.mock_tm.use_trace_tensor = False  # exercise the preprocess_model branch
        self.mock_tm.loader_type = "strategy"
        self.mock_tm.pinned_memory = True
        self.mock_tm.host_pinner = self.host_pinner
        self.mock_tm.device_gpu = torch.device("cpu")
        self.mock_tm.move_top_level_buffers_to_gpu = False

        self.handler = TensorManagerStateHandler(self.mock_tm)

    def _patch_preprocess_model(self, monkeypatch):
        captured: dict = {}

        def fake_preprocess_model(model, tensor_manager, device_gpu, **kwargs):
            captured["model"] = model
            captured["tensor_manager"] = tensor_manager
            captured["device_gpu"] = device_gpu
            captured.update(kwargs)

        monkeypatch.setattr(state_handler_module, "preprocess_model", fake_preprocess_model)
        return captured

    def test_restore_state_forwards_manager_host_pinner(self, monkeypatch):
        captured = self._patch_preprocess_model(monkeypatch)
        state = create_mock_state(self.model_tensor_names, model=self.model)

        self.handler.restore_state(self.model, state, strict=True)

        assert "host_pinner" in captured, "restore_state must pass host_pinner to preprocess_model"
        assert captured["host_pinner"] is self.host_pinner, (
            "restore_state must forward the same HostPinner instance owned by the manager, "
            "not a freshly constructed default."
        )

    def test_restore_state_validates_before_preprocessing(self, monkeypatch):
        captured = self._patch_preprocess_model(monkeypatch)
        state = create_mock_state(self.model_tensor_names, model=self.model)
        state.tensor_id_to_name_map[-1] = self.model_tensor_names[0]

        with pytest.raises(ValueError, match=r"inventory names.*unique"):
            self.handler.restore_state(self.model, state, strict=True)

        assert captured == {}

    def test_restore_state_rejects_size_drift_before_preprocessing(self, monkeypatch):
        captured = self._patch_preprocess_model(monkeypatch)
        state = create_mock_state(self.model_tensor_names, model=self.model)
        stat = state.load_strategy["layer_0"][0]
        state.load_strategy["layer_0"][0] = stat.model_copy(update={"size_bytes": stat.size_bytes + 1})

        with pytest.raises(ValueError, match="size_bytes drift"):
            self.handler.restore_state(self.model, state, strict=True)

        assert captured == {}

    def test_restore_state_consults_should_pin_in_preprocess(self, monkeypatch):
        """The pin_memory value forwarded to ``preprocess_model`` must come
        from :meth:`TensorManager.should_pin_in_preprocess` — not be
        re-derived inline. Pins the contract that block-loader semantics
        live in one place.

        The per-loader-type rule itself is covered by
        :class:`TestShouldPinInPreprocess` in
        ``test_tensor_manager_pinned_memory_mode.py``.
        """
        sentinel = object()
        self.mock_tm.should_pin_in_preprocess.return_value = sentinel
        captured = self._patch_preprocess_model(monkeypatch)
        state = create_mock_state(self.model_tensor_names, model=self.model)

        self.handler.restore_state(self.model, state, strict=True)

        self.mock_tm.should_pin_in_preprocess.assert_called_once_with()
        assert captured["pin_memory"] is sentinel, (
            "state_handler must forward should_pin_in_preprocess()'s "
            "return value verbatim, not re-derive the rule from tm.pinned_memory + "
            "tm.loader_type"
        )
        assert captured["host_pinner"] is self.host_pinner


class TestPinnedMemoryModeNotInSavedState:
    """``pinned_memory_mode`` is a host-side resource policy, not part of
    the discovered loader plan, and must not be carried by
    :class:`TensorManagerState`. Pinning this contract prevents a silent
    mismatch where a profile saved under one mode is later restored into a
    manager with a different mode and the JSON looks authoritative."""

    def test_pinned_memory_mode_not_a_field_of_state(self):
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(TensorManagerState)}
        assert "pinned_memory_mode" not in field_names, (
            "pinned_memory_mode must not be a TensorManagerState field — see TensorManagerState docstring for rationale"
        )

    def test_to_dict_round_trip_does_not_carry_pinned_memory_mode(self):
        state = create_mock_state(["layer.weight"])

        as_dict = state.to_dict()

        assert "pinned_memory_mode" not in as_dict, (
            "to_dict() must not surface pinned_memory_mode — restored "
            "managers must take their mode from the live constructor "
            "argument, not from a serialized profile"
        )

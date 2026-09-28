# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadManager and TensorManager save_profile/load_profile functionality.

This test suite validates profile saving and loading operations:

1. save_profile: Saves tensor manager state to a directory as profile.json
2. load_profile: Loads tensor manager state from a directory and restores it to a model

Tests use pytest's tmp_path fixture to create temporary directories that are
automatically cleaned up after each test.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import flextensor
from flextensor import custom_ops
from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.loaders import PreallocatedLoader
from flextensor.offload_manager import OffloadConfig, OffloadManager, OffloadModelProxy, OffloadPhase
from flextensor.state_handler import TensorManagerState
from flextensor.state_transition import StateTransitionPlan
from flextensor.tensor_manager import TensorManager


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


class _RecordingPreallocatedLoader(PreallocatedLoader):
    def __init__(self) -> None:
        self.boundary_calls: list[str] = []

    def enter(self, label: str) -> None:
        pass

    def exit(self, label: str) -> None:
        pass

    def preload(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def sync_prev_onload(self) -> None:
        self.boundary_calls.append("sync_prev_onload")

    def join_after_forward(self) -> None:
        self.boundary_calls.append("join_after_forward")


class _RejectDataAssignmentParameter(torch.nn.Parameter):
    def __new__(cls, data: torch.Tensor) -> "_RejectDataAssignmentParameter":
        return super().__new__(cls, data, requires_grad=False)

    @property
    def data(self) -> torch.Tensor:
        return torch.Tensor.data.__get__(self, type(self))

    @data.setter
    def data(self, _value: torch.Tensor) -> None:
        raise RuntimeError("custom parameter rejects .data assignment")


class MockTensorManagerState:
    """Mock TensorManagerState for testing."""

    def __init__(self):
        self.load_strategy = {}
        self.stats = []
        self.release_strategy = {}

    def to_dict(self):
        return {
            "loader_type": "strategy",
            "tensor_id_to_name_map": {},
            "allocation_ordered": {},
            "label_to_size_map": {},
            "block_sizes": {},
            "load_strategy": {},
            "release_strategy": {},
            "label_to_block_id": {},
            "stats": [],
            "transfer_to_compute_map": {},
            "view_tensors_ids": [],
            "view_tensors_names": [],
            "gpu_tensors_names": [],
            "shm_block_name_map": None,
        }


def _state_for_model(model: torch.nn.Module) -> TensorManagerState:
    tensors = list(model.named_parameters())
    stats = [
        TensorStatistics(
            tensor_id=index,
            name=name,
            size_bytes=tensor.numel() * tensor.element_size(),
            load_time_ms=0.0,
        )
        for index, (name, tensor) in enumerate(tensors)
    ]
    return TensorManagerState(
        loader_type="strategy",
        tensor_id_to_name_map={stat.tensor_id: stat.name for stat in stats},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={"linear": stats},
        release_strategy={"linear": stats},
        label_to_block_id={},
        stats=[LayerStatistics(label="linear", tensors=stats, duration=1.0)],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )


def _takeover_state_for_model(model: torch.nn.Module, loader_type: str) -> TensorManagerState:
    state = _state_for_model(model)
    state.loader_type = loader_type
    if loader_type in {"allocation_block_transfer", "raw_block_transfer"}:
        stats = state.load_strategy["linear"]
        state.release_strategy = {}
        state.view_tensors_ids = [stat.tensor_id for stat in stats]
        state.view_tensors_names = [stat.name for stat in stats]
        state.allocation_ordered = {0: ["linear"]}
        state.label_to_size_map = {"linear": sum(stat.size_bytes for stat in stats)}
        state.block_sizes = {0: sum(stat.size_bytes for stat in stats)}
        state.label_to_block_id = {"linear": 0}
        state.transfer_to_compute_map = {"linear": "linear"}
    return state


def _takeover_tensor_manager(loader_type: str) -> TensorManager:
    with patch("torch.cuda.Event", return_value=MagicMock()):
        return TensorManager(
            device_gpu=torch.device("cpu"),
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
            loader_type=loader_type,
            include_patterns=["linear1"],
        )


def test_runtime_boundary_hooks_delegate_to_preallocated_loader():
    loader = _RecordingPreallocatedLoader()
    manager = OffloadManager("runtime-boundaries")
    manager._compiled = SimpleNamespace(active=True)
    manager._tensor_manager = _takeover_tensor_manager("strategy")
    manager._tensor_manager.tensor_layer_loader = loader

    assert manager._tensor_manager.sync_prev_onload() is None
    assert manager._tensor_manager.join_after_forward() is None
    assert manager.sync_prev_onload() is None
    assert manager.join_after_forward() is None

    assert loader.boundary_calls == [
        "sync_prev_onload",
        "join_after_forward",
        "sync_prev_onload",
        "join_after_forward",
    ]


@pytest.mark.parametrize("compiled_active", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("loader", [None, object()], ids=["missing", "wrong-type"])
def test_runtime_boundary_hooks_are_noops_without_preallocated_loader(compiled_active, loader):
    manager = OffloadManager("runtime-boundary-noop")
    manager._compiled = SimpleNamespace(active=compiled_active)
    manager._tensor_manager = _takeover_tensor_manager("strategy")
    manager._tensor_manager.tensor_layer_loader = loader

    assert manager._tensor_manager.sync_prev_onload() is None
    assert manager._tensor_manager.join_after_forward() is None
    assert manager.sync_prev_onload() is None
    assert manager.join_after_forward() is None


def test_runtime_boundary_hooks_are_noops_without_tensor_manager():
    manager = OffloadManager("runtime-boundaries-without-manager")

    assert manager.sync_prev_onload() is None
    assert manager.join_after_forward() is None


class TestOffloadManagerSaveProfile:
    """Test cases for OffloadManager.save_profile()."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_calls_tensor_manager_with_directory(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that save_profile passes the directory path to tensor_manager.save_profile()."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, include_patterns=["linear1"], skip_discovery=False)
        om.offload(self.model, config=config)

        profile_dir = tmp_path / "new_profile_dir"

        om.save_profile(str(profile_dir))

        # Verify tensor_manager.save_profile was called with the directory path
        mock_tensor_manager.save_profile.assert_called_once_with(str(profile_dir))

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_delegates_to_tensor_manager(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that save_profile delegates to tensor_manager.save_profile()."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, include_patterns=["linear1"], skip_discovery=False)
        om.offload(self.model, config=config)

        profile_dir = str(tmp_path / "profile")
        om.save_profile(profile_dir)

        mock_tensor_manager.save_profile.assert_called_once_with(profile_dir)

    def test_save_profile_raises_if_not_initialized(self, tmp_path):
        """Test that save_profile raises RuntimeError if tensor_manager is not initialized."""
        om = OffloadManager("test")

        with pytest.raises(RuntimeError, match="Tensor manager not initialized"):
            om.save_profile(str(tmp_path / "profile"))


class TestOffloadManagerLoadProfile:
    """Test cases for OffloadManager.load_profile()."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_load_profile_delegates_to_tensor_manager(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that load_profile delegates to tensor_manager.load_profile()."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, include_patterns=["linear1"], skip_discovery=False)
        om.offload(self.model, config=config)

        profile_dir = str(tmp_path / "profile")
        om.load_profile(profile_dir, self.model)

        mock_tensor_manager.load_profile.assert_called_once_with(profile_dir, self.model)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_load_profile_uses_current_model_if_not_provided(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that load_profile uses the current model if model is not provided."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, include_patterns=["linear1"], skip_discovery=False)
        om.offload(self.model, config=config)

        profile_dir = str(tmp_path / "profile")
        om.load_profile(profile_dir)

        # Should use the internal model
        mock_tensor_manager.load_profile.assert_called_once()
        call_args = mock_tensor_manager.load_profile.call_args
        assert call_args[0][0] == profile_dir
        # The model passed should be the internal model
        assert call_args[0][1] is not None

    def test_load_profile_raises_if_no_model(self, tmp_path):
        """Test that load_profile raises RuntimeError if no model is available."""
        om = OffloadManager("test")
        # Initialize tensor manager without a model
        om._initialize_tensor_manager = MagicMock()

        with pytest.raises(RuntimeError, match="No model provided"):
            om.load_profile(str(tmp_path / "profile"))


class TestTensorManagerSaveProfile:
    """Test cases for TensorManager.save_profile()."""

    def test_save_profile_creates_directory_and_file(self, tmp_path):
        """Test that save_profile creates the directory and profile.json file."""
        from flextensor.state_handler import TensorManagerState
        from flextensor.tensor_manager import TensorManager

        # Create a minimal TensorManager with mocked dependencies
        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)
            tm.tensor_manager_state = MagicMock(spec=TensorManagerState)
            tm.tensor_manager_state.to_dict.return_value = {
                "loader_type": "strategy",
                "tensor_id_to_name_map": {},
                "allocation_ordered": {},
                "label_to_size_map": {},
                "block_sizes": {},
                "load_strategy": {},
                "release_strategy": {},
                "label_to_block_id": {},
                "stats": [],
                "transfer_to_compute_map": {},
                "view_tensors_ids": [],
                "view_tensors_names": [],
                "gpu_tensors_names": [],
                "shm_block_name_map": None,
            }

            profile_dir = tmp_path / "profile"
            assert not profile_dir.exists()

            tm.save_profile(str(profile_dir))

            assert profile_dir.exists()
            profile_file = profile_dir / "profile.json"
            assert profile_file.exists()

            # Verify JSON content is valid
            with profile_file.open() as f:
                data = json.load(f)
            assert "loader_type" in data

    def test_save_profile_overwrites_existing(self, tmp_path):
        """Test that save_profile overwrites existing profile.json."""
        from flextensor.state_handler import TensorManagerState
        from flextensor.tensor_manager import TensorManager

        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        profile_file = profile_dir / "profile.json"

        # Write initial content
        profile_file.write_text('{"old": "data"}')

        # Create TensorManager and save new profile
        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)
            tm.tensor_manager_state = MagicMock(spec=TensorManagerState)
            tm.tensor_manager_state.to_dict.return_value = {
                "loader_type": "new_strategy",
                "tensor_id_to_name_map": {},
                "allocation_ordered": {},
                "label_to_size_map": {},
                "block_sizes": {},
                "load_strategy": {},
                "release_strategy": {},
                "label_to_block_id": {},
                "stats": [],
                "transfer_to_compute_map": {},
                "view_tensors_ids": [],
                "view_tensors_names": [],
                "gpu_tensors_names": [],
                "shm_block_name_map": None,
            }

            tm.save_profile(str(profile_dir))

            # Verify new content
            with profile_file.open() as f:
                data = json.load(f)
            assert data["loader_type"] == "new_strategy"
            assert "old" not in data


class TestTensorManagerLoadProfile:
    """Test cases for TensorManager.load_profile()."""

    def test_load_profile_reads_from_directory(self, tmp_path):
        """Test that load_profile reads profile.json from the specified directory."""
        from flextensor.tensor_manager import TensorManager

        # Create a valid profile
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        profile_file = profile_dir / "profile.json"

        from flextensor.state_handler import TensorManagerState

        profile_data = {
            "version": TensorManagerState.SCHEMA_VERSION,
            "loader_type": "strategy",
            "tensor_id_to_name_map": {"123": "layer.weight"},
            "allocation_ordered": {},
            "label_to_size_map": {},
            "block_sizes": {},
            "load_strategy": {},
            "release_strategy": {},
            "label_to_block_id": {},
            "stats": [],
            "transfer_to_compute_map": {},
            "view_tensors_ids": [123],
            "view_tensors_names": ["layer.weight"],
            "gpu_tensors_names": [],
            "shm_block_name_map": None,
        }
        profile_file.write_text(json.dumps(profile_data))

        # Create TensorManager with mocked restore_state
        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)
            tm.restore_state = MagicMock()

            model = SimpleModel()
            tm.load_profile(str(profile_dir), model)

            # Verify restore_state was called with the loaded state
            tm.restore_state.assert_called_once()
            call_args = tm.restore_state.call_args
            assert call_args[0][0] is model
            loaded_state = call_args[0][1]
            assert loaded_state.loader_type == "strategy"

    def test_load_profile_raises_if_file_not_found(self, tmp_path):
        """Test that load_profile raises error if profile.json doesn't exist."""
        from flextensor.tensor_manager import TensorManager

        profile_dir = tmp_path / "nonexistent"

        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)

            model = SimpleModel()
            with pytest.raises(FileNotFoundError):
                tm.load_profile(str(profile_dir), model)


class TestProfileRoundTrip:
    """Test cases for save/load profile round-trip functionality."""

    def test_profile_json_structure(self, tmp_path):
        """Test that saved profile has correct JSON structure."""
        from flextensor.state_handler import TensorManagerState
        from flextensor.tensor_manager import TensorManager

        profile_dir = tmp_path / "profile"

        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)
            tm.tensor_manager_state = MagicMock(spec=TensorManagerState)
            tm.tensor_manager_state.to_dict.return_value = {
                "loader_type": "allocation_block_transfer",
                "tensor_id_to_name_map": {"1": "linear1.weight", "2": "linear1.bias"},
                "allocation_ordered": {"0": ["layer1", "layer2"]},
                "label_to_size_map": {"layer1": 1024, "layer2": 512},
                "block_sizes": {"0": 2048},
                "load_strategy": {"layer1": []},
                "release_strategy": {"layer1": []},
                "label_to_block_id": {"layer1": 0},
                "stats": [{"label": "layer1", "tensors": [], "duration": 1.5}],
                "transfer_to_compute_map": {"layer1": "layer1"},
                "view_tensors_ids": [1, 2],
                "view_tensors_names": ["linear1.weight", "linear1.bias"],
                "gpu_tensors_names": ["linear2.weight"],
                "shm_block_name_map": {"block_0": "shm_block_0"},
            }

            tm.save_profile(str(profile_dir))

            profile_file = profile_dir / "profile.json"
            with profile_file.open() as f:
                data = json.load(f)

            # Verify all required fields are present
            assert data["loader_type"] == "allocation_block_transfer"
            assert "1" in data["tensor_id_to_name_map"]
            assert data["allocation_ordered"]["0"] == ["layer1", "layer2"]
            assert data["view_tensors_names"] == ["linear1.weight", "linear1.bias"]
            assert data["shm_block_name_map"]["block_0"] == "shm_block_0"


class TestProfileDirectoryFromConfig:
    """Test cases for profile_directory falling back to config.profile_storage_dir."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()

    def test_resolve_profile_directory_uses_argument(self):
        """Test that explicit argument takes precedence over config."""
        om = OffloadManager("test")
        om.set_config(OffloadConfig(profile_storage_dir="/config/path"))

        resolved = om._resolve_profile_directory("/explicit/path")
        assert resolved == "/explicit/path"

    def test_resolve_profile_directory_falls_back_to_config(self):
        """Test that config.profile_storage_dir is used when argument is None."""
        om = OffloadManager("test")
        om.set_config(OffloadConfig(profile_storage_dir="/config/path"))

        resolved = om._resolve_profile_directory(None)
        assert resolved == "/config/path"

    def test_resolve_profile_directory_raises_when_neither_set(self):
        """Test that ValueError is raised when no directory is available."""
        om = OffloadManager("test")
        om.set_config(OffloadConfig(profile_storage_dir=None))

        with pytest.raises(ValueError, match="No profile directory specified"):
            om._resolve_profile_directory(None)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_uses_config_directory(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that save_profile() without arguments uses config.profile_storage_dir."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        profile_dir = str(tmp_path / "config_profile")
        config = OffloadConfig(
            enabled=True, include_patterns=["linear1"], profile_storage_dir=profile_dir, skip_discovery=False
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        om.save_profile()

        mock_tensor_manager.save_profile.assert_called_once_with(profile_dir)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_argument_overrides_config(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that explicit argument overrides config.profile_storage_dir."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        config = OffloadConfig(
            enabled=True, include_patterns=["linear1"], profile_storage_dir="/config/path", skip_discovery=False
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        explicit_dir = str(tmp_path / "explicit_profile")
        om.save_profile(explicit_dir)

        mock_tensor_manager.save_profile.assert_called_once_with(explicit_dir)

    def test_save_profile_raises_when_no_directory(self):
        """Test that save_profile raises ValueError when no directory is available."""
        om = OffloadManager("test")
        om._tensor_manager = MagicMock()

        with pytest.raises(ValueError, match="No profile directory specified"):
            om.save_profile()

    def test_save_profile_raises_when_read_only(self):
        """Test that save_profile raises ValueError when profile_read_only is True."""
        om = OffloadManager("test")
        om.set_config(OffloadConfig(profile_read_only=True, profile_storage_dir="/some/path"))
        om._tensor_manager = MagicMock()

        with pytest.raises(ValueError, match="profile_read_only is True"):
            om.save_profile()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_succeeds_when_not_read_only(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test that save_profile works when profile_read_only is False."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        config = OffloadConfig(
            enabled=True,
            include_patterns=["linear1"],
            profile_storage_dir="/some/path",
            skip_discovery=False,
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        om.save_profile()

        mock_tensor_manager.save_profile.assert_called_once_with("/some/path")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_load_profile_uses_config_directory(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that load_profile() without directory uses config.profile_storage_dir."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        profile_dir = str(tmp_path / "config_profile")
        config = OffloadConfig(
            enabled=True, include_patterns=["linear1"], profile_storage_dir=profile_dir, skip_discovery=False
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        om.load_profile(model=self.model)

        mock_tensor_manager.load_profile.assert_called_once_with(profile_dir, self.model)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_load_profile_argument_overrides_config(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that explicit argument overrides config.profile_storage_dir for load."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        config = OffloadConfig(
            enabled=True, include_patterns=["linear1"], profile_storage_dir="/config/path", skip_discovery=False
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        explicit_dir = str(tmp_path / "explicit_profile")
        om.load_profile(explicit_dir, self.model)

        mock_tensor_manager.load_profile.assert_called_once_with(explicit_dir, self.model)

    def test_load_profile_raises_when_no_directory(self):
        """Test that load_profile raises ValueError when no directory is available."""
        om = OffloadManager("test")
        om._tensor_manager = MagicMock()
        om._model = self.model

        with pytest.raises(ValueError, match="No profile directory specified"):
            om.load_profile()


class TestModuleLevelProfileFunctions:
    """Test cases for module-level save_profile and load_profile functions."""

    @patch("flextensor.offload_manager.get_offload_manager")
    def test_module_save_profile(self, mock_get_om, tmp_path):
        """Test that module-level save_profile delegates to OffloadManager."""
        from flextensor.offload_manager import save_profile

        mock_om = MagicMock()
        mock_get_om.return_value = mock_om

        profile_dir = str(tmp_path / "profile")
        save_profile(profile_dir)

        mock_om.save_profile.assert_called_once_with(profile_dir)

    @patch("flextensor.offload_manager.get_offload_manager")
    def test_module_load_profile(self, mock_get_om, tmp_path):
        """Test that module-level load_profile delegates to OffloadManager."""
        from flextensor.offload_manager import load_profile

        mock_om = MagicMock()
        mock_get_om.return_value = mock_om

        model = SimpleModel()
        profile_dir = str(tmp_path / "profile")
        load_profile(profile_dir, model)

        mock_om.load_profile.assert_called_once_with(profile_dir, model)

    @patch("flextensor.offload_manager.get_offload_manager")
    def test_module_load_profile_without_model(self, mock_get_om, tmp_path):
        """Test that module-level load_profile works without model argument."""
        from flextensor.offload_manager import load_profile

        mock_om = MagicMock()
        mock_get_om.return_value = mock_om

        profile_dir = str(tmp_path / "profile")
        load_profile(profile_dir)

        mock_om.load_profile.assert_called_once_with(profile_dir, None)


class TestOffloadFromState:
    @pytest.mark.parametrize("phase", [OffloadPhase.NOT_INITIALIZED, OffloadPhase.PROFILING])
    def test_instrumentation_dump_is_skipped_before_inference(self, phase):
        manager = OffloadManager("instrumentation")
        manager.config = OffloadConfig(enable_instrumentation=True, instrumentation_output_dir="instrumentation")
        manager._tensor_manager = MagicMock()
        manager._current_phase = phase

        with patch("flextensor.offload_manager.dump_to_directory", side_effect=AssertionError("unexpected dump")):
            manager._dump_instrumentation()

    def test_instrumentation_dump_failure_does_not_escape(self, caplog):
        manager = OffloadManager("instrumentation")
        manager.config = OffloadConfig(enable_instrumentation=True, instrumentation_output_dir="instrumentation")
        manager._tensor_manager = MagicMock()
        manager._current_phase = OffloadPhase.INFERENCE

        with (
            patch("flextensor.offload_manager.dump_to_directory", side_effect=OSError("disk full")),
            caplog.at_level("WARNING", logger="flextensor.offload_manager"),
        ):
            manager._dump_instrumentation()

        assert "instrumentation dump failed" in caplog.text

    @patch("flextensor.offload_manager.get_offload_manager")
    def test_module_level_api_delegates_to_named_manager(self, mock_get_om):
        model = SimpleModel()
        state = _state_for_model(model)
        config = OffloadConfig(include_patterns=["linear1"])
        final_model = SimpleModel()
        manager = MagicMock()
        manager.offload_from_state.return_value = final_model
        mock_get_om.return_value = manager

        result = flextensor.offload_from_state(model, state, config=config, name="adopted")

        mock_get_om.assert_called_once_with("adopted")
        manager.offload_from_state.assert_called_once_with(
            model,
            state,
            config=config,
            allow_strategy_replan=True,
        )
        assert result is final_model

    def test_manager_runs_takeover_in_order_and_rejects_a_second_active_call(self):
        model = SimpleModel()
        state = _state_for_model(model)
        config = OffloadConfig(
            include_patterns=["linear1"],
            exclude_patterns=["linear2"],
            enable_instrumentation=True,
            instrumentation_output_dir="instrumentation",
        )
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        events: list[str] = []
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.side_effect = lambda *_: (events.append("plan"), plan)[1]
        tensor_manager.execute_state_adoption.side_effect = lambda *_: events.append("execute")
        tensor_manager.restore_adopted_state.side_effect = lambda *_: events.append("restore")
        tensor_manager.prepare_infer_load_mode.side_effect = lambda: events.append("loader")
        tensor_manager.prepare_final_model.side_effect = lambda candidate, **_: (
            events.append("finalize"),
            candidate,
        )[1]
        tensor_manager.set_model.side_effect = lambda *_: events.append("set_model")
        final_model = SimpleModel()
        tensor_manager.initialize_inference.side_effect = lambda: (events.append("publish"), final_model)[1]
        tensor_manager.get_memory_transfer_stats.return_value = {1024: 0.5}
        manager = OffloadManager("adopted")
        manager._tensor_manager = tensor_manager
        manager._offload_modules = MagicMock(side_effect=lambda *_: events.append("patch"))
        manager._exclude_modules = MagicMock(side_effect=lambda *_: events.append("exclude"))
        manager._check_no_modules_patched = MagicMock()
        resolve_activation = manager._compiled.resolve_activation
        manager._compiled.resolve_activation = MagicMock(
            side_effect=lambda *args: (events.append("resolve"), resolve_activation(*args))[1]
        )
        manager._compiled.arm_non_destructive_first_loader = MagicMock(side_effect=lambda: events.append("arm"))
        manager._install_compiled_forwards = MagicMock(side_effect=lambda: events.append("install-compiled"))
        manager._compiled.on_enter_inference = MagicMock(side_effect=lambda: events.append("enter-compiled"))
        init = manager.init
        manager.init = MagicMock(side_effect=lambda *args, **kwargs: (events.append("init"), init(*args, **kwargs))[1])

        with patch(
            "flextensor.offload_manager.dump_to_directory",
            side_effect=lambda *_args, **_kwargs: events.append("dump"),
        ) as dump:
            result = manager.offload_from_state(model, state, config=config)

        assert isinstance(result, OffloadModelProxy)
        assert result.__subject__ is final_model
        assert result.offload_manager is manager
        assert events == [
            "resolve",
            "init",
            "arm",
            "plan",
            "execute",
            "restore",
            "loader",
            "patch",
            "exclude",
            "finalize",
            "set_model",
            "publish",
            "install-compiled",
            "enter-compiled",
            "dump",
        ]
        dump.assert_called_once_with("instrumentation", extra={"memory_transfer_stats": {1024: 0.5}})
        tensor_manager.execute_state_adoption.assert_called_once_with(model, plan)
        tensor_manager.prepare_final_model.assert_called_once_with(model, in_place=True)
        assert manager.model is final_model
        assert manager._current_phase is OffloadPhase.INFERENCE

        with pytest.raises(RuntimeError, match="already active"):
            manager.offload_from_state(model, state, config=config)
        tensor_manager.plan_state_adoption.assert_called_once()

    def test_external_compile_without_strategy_replan_installs_the_adopted_loader(self):
        model = SimpleModel()
        state = _state_for_model(model)
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        loader = MagicMock(spec=PreallocatedLoader)
        tensor_manager = MagicMock()
        tensor_manager.tensor_layer_loader = loader
        tensor_manager.plan_state_adoption.return_value = plan
        tensor_manager.prepare_final_model.return_value = model
        tensor_manager.initialize_inference.return_value = model
        manager = OffloadManager("adopted-compiled")
        manager._tensor_manager = tensor_manager

        try:
            manager.offload_from_state(
                model,
                state,
                config=OffloadConfig(
                    external_compile=True,
                    transfer_mode="allocation_block_transfer",
                ),
                allow_strategy_replan=False,
            )

            assert manager._compiled.active
            assert not manager.compiled_replan_active
            tensor_manager.arm_non_destructive_first_loader.assert_not_called()
            assert custom_ops.get_active_loader(manager.compiled_offload_manager_id) is loader
        finally:
            manager.release()

    def test_failed_external_compile_takeover_clears_compiled_lifecycle(self):
        model = SimpleModel()
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.side_effect = RuntimeError("plan failed")
        manager = OffloadManager("adopted-compiled-failure")
        manager._tensor_manager = tensor_manager

        with pytest.raises(RuntimeError, match="plan failed"):
            manager.offload_from_state(
                model,
                _state_for_model(model),
                config=OffloadConfig(
                    external_compile=True,
                    transfer_mode="allocation_block_transfer",
                ),
            )

        assert not manager._compiled.active
        assert custom_ops.get_active_loader(manager.compiled_offload_manager_id) is None

    def test_compiled_model_is_rejected_before_state_planning(self):
        model = SimpleModel()
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.return_value = StateTransitionPlan(
            migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0
        )
        tensor_manager.prepare_final_model.return_value = model
        tensor_manager.initialize_inference.return_value = model
        manager = OffloadManager("adopted-compiled-model")
        manager._tensor_manager = tensor_manager

        with (
            patch("flextensor.offload_manager.is_torch_compiled_module", return_value=True),
            pytest.raises(RuntimeError, match="eager model"),
        ):
            manager.offload_from_state(model, _state_for_model(model))

        tensor_manager.plan_state_adoption.assert_not_called()

    def test_takeover_init_failure_tears_down_resolved_compiled_activation(self):
        model = SimpleModel()
        state = _state_for_model(model)
        manager = OffloadManager("compiled-init-failure")
        manager._compiled.resolve_activation = MagicMock(return_value=True)
        manager._compiled.teardown = MagicMock()
        manager.init = MagicMock(side_effect=RuntimeError("init failed"))
        config = OffloadConfig(
            external_compile=True,
            transfer_mode="allocation_block_transfer",
            pinned_memory=False,
        )

        with pytest.raises(RuntimeError, match="init failed"):
            manager.offload_from_state(model, state, config=config)

        manager._compiled.resolve_activation.assert_called_once_with(config, None)
        manager._compiled.teardown.assert_called_once_with()

    def test_init_failure_cleans_partially_initialized_tensor_manager_before_retry(self):
        tensor_manager = MagicMock()
        tensor_manager.set_skip_discovery.side_effect = RuntimeError("skip-discovery failed")
        manager = OffloadManager("partial-init-failure")
        config = OffloadConfig(pinned_memory=False)

        with patch("flextensor.tensor_manager.TensorManager", return_value=tensor_manager) as tensor_manager_cls:
            with pytest.raises(RuntimeError, match="skip-discovery failed"):
                manager.init(config)

            tensor_manager.shutdown.assert_called_once_with()
            assert manager._tensor_manager is None

            tensor_manager.set_skip_discovery.side_effect = None
            manager.init(config)

        assert tensor_manager_cls.call_count == 2
        assert manager._tensor_manager is tensor_manager

    def test_takeover_failure_after_activation_tears_down_compiled_state(self):
        model = SimpleModel()
        state = _state_for_model(model)
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.side_effect = RuntimeError("plan failed")
        manager = OffloadManager("compiled-plan-failure")
        manager._tensor_manager = tensor_manager
        manager._compiled.resolve_activation = MagicMock(return_value=True)
        manager._compiled.teardown = MagicMock()
        config = OffloadConfig(
            external_compile=True,
            transfer_mode="allocation_block_transfer",
            pinned_memory=False,
        )

        with pytest.raises(RuntimeError, match="plan failed"):
            manager.offload_from_state(model, state, config=config)

        manager._compiled.resolve_activation.assert_called_once_with(config, None)
        manager._compiled.teardown.assert_called_once_with()

    def test_postmigration_failure_restores_owned_parameter_data_and_closes_manager(self):
        model = SimpleModel()
        state = _state_for_model(model)
        migrated = {parameter: torch.full_like(parameter, 7) for parameter in model.parameters()}
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.return_value = plan

        def migrate(*_):
            for parameter, data in migrated.items():
                parameter.data = data

        def fail_restore(*_):
            for parameter in model.parameters():
                parameter.data = torch.empty(0, dtype=parameter.dtype)
            raise RuntimeError("restore failed")

        tensor_manager.execute_state_adoption.side_effect = migrate
        tensor_manager.restore_adopted_state.side_effect = fail_restore
        manager = OffloadManager("failed-adoption")
        manager._tensor_manager = tensor_manager

        with pytest.raises(RuntimeError, match="restore failed"):
            manager.offload_from_state(model, state, config=OffloadConfig(pinned_memory=False))

        for parameter, expected in migrated.items():
            torch.testing.assert_close(parameter, expected)
        tensor_manager.release_memory.assert_called_once()
        tensor_manager.shutdown.assert_called_once()
        assert manager._tensor_manager is None
        assert manager._current_phase is OffloadPhase.NOT_INITIALIZED
        assert not manager._patched_modules

    def test_cleanup_bypasses_parameter_subclass_data_setter(self):
        model = torch.nn.Module()
        parameter = _RejectDataAssignmentParameter(torch.arange(4.0))
        model.register_parameter("weight", parameter)
        state = _state_for_model(model)
        retained = torch.full_like(parameter, 13)
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.return_value = plan
        tensor_manager.execute_state_adoption.side_effect = lambda *_: torch.Tensor.data.__set__(parameter, retained)

        def fail_restore(*_):
            torch.Tensor.data.__set__(parameter, torch.empty(0))
            raise RuntimeError("restore failed")

        tensor_manager.restore_adopted_state.side_effect = fail_restore
        manager = OffloadManager("subclass-cleanup")
        manager._tensor_manager = tensor_manager

        with pytest.raises(RuntimeError, match="restore failed"):
            manager.offload_from_state(model, state, config=OffloadConfig(pinned_memory=False))

        torch.testing.assert_close(parameter, retained)
        tensor_manager.shutdown.assert_called_once()
        assert manager._tensor_manager is None

    def test_failed_parameter_restoration_retains_storage_owner(self, monkeypatch):
        model = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.arange(4.0), requires_grad=False)
        model.register_parameter("weight", parameter)
        state = _state_for_model(model)
        retained = parameter.detach().clone()
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.return_value = plan
        tensor_manager.execute_state_adoption = MagicMock()

        def fail_restore(*_):
            torch.Tensor.data.__set__(parameter, torch.empty(0))
            raise RuntimeError("restore failed")

        tensor_manager.restore_adopted_state.side_effect = fail_restore
        manager = OffloadManager("retained-owner")
        manager._tensor_manager = tensor_manager
        monkeypatch.setattr(
            "flextensor.offload_manager.set_tensor_data",
            MagicMock(side_effect=RuntimeError("rebind failed")),
            raising=False,
        )

        with pytest.raises(RuntimeError, match="restore failed"):
            manager.offload_from_state(model, state, config=OffloadConfig(pinned_memory=False))

        assert parameter.numel() == 0
        assert retained.numel() == 4
        tensor_manager.release_memory.assert_not_called()
        tensor_manager.shutdown.assert_not_called()
        assert manager._tensor_manager is tensor_manager
        assert manager._cleanup_blocked
        with pytest.raises(RuntimeError, match="cleanup is incomplete"):
            manager.offload_from_state(model, state, config=OffloadConfig(pinned_memory=False))

    def test_postmigration_snapshot_failure_cleans_inactive_manager(self):
        model = SimpleModel()
        state = _state_for_model(model)
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.return_value = plan
        manager = OffloadManager("snapshot-failure")
        manager._tensor_manager = tensor_manager
        model.parameters = MagicMock(side_effect=RuntimeError("snapshot failed"))

        with pytest.raises(RuntimeError, match="snapshot failed"):
            manager.offload_from_state(model, state, config=OffloadConfig(pinned_memory=False))

        tensor_manager.execute_state_adoption.assert_called_once_with(model, plan)
        tensor_manager.release_memory.assert_called_once()
        tensor_manager.shutdown.assert_called_once()
        assert manager._tensor_manager is None
        assert not manager._cleanup_blocked
        assert manager._state_hook_handle is None
        assert manager._patched_modules == []
        manager._initialize_tensor_manager = MagicMock()
        manager.init()
        manager._initialize_tensor_manager.assert_called_once()

    def test_snapshot_cleanup_shutdown_failure_retains_explicit_owner(self):
        model = SimpleModel()
        state = _state_for_model(model)
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.return_value = plan
        tensor_manager.shutdown.side_effect = RuntimeError("shutdown failed")
        manager = OffloadManager("snapshot-cleanup-failure")
        manager._tensor_manager = tensor_manager
        model.parameters = MagicMock(side_effect=RuntimeError("snapshot failed"))

        with pytest.raises(RuntimeError, match="snapshot failed"):
            manager.offload_from_state(model, state, config=OffloadConfig(pinned_memory=False))

        assert manager._tensor_manager is tensor_manager
        assert manager._model is model
        assert manager._cleanup_blocked
        with pytest.raises(RuntimeError, match="cleanup is incomplete"):
            manager.offload_from_state(model, state)

    @pytest.mark.parametrize("failure_stage", ["release", "shutdown"])
    def test_takeover_cleanup_resource_failure_retains_manager(self, failure_stage):
        model = SimpleModel()
        state = _state_for_model(model)
        retained = {parameter: parameter.detach().clone() for parameter in model.parameters()}
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.return_value = plan
        tensor_manager.restore_adopted_state.side_effect = RuntimeError("restore failed")
        if failure_stage == "release":
            tensor_manager.release_memory.side_effect = RuntimeError("release failed")
        else:
            tensor_manager.shutdown.side_effect = RuntimeError("shutdown failed")
        manager = OffloadManager(f"cleanup-{failure_stage}-failure")
        manager._tensor_manager = tensor_manager

        with pytest.raises(RuntimeError, match="restore failed"):
            manager.offload_from_state(model, state, config=OffloadConfig(pinned_memory=False))

        for parameter, expected in retained.items():
            torch.testing.assert_close(parameter, expected)
        if failure_stage == "release":
            tensor_manager.shutdown.assert_not_called()
        else:
            tensor_manager.shutdown.assert_called_once()
        assert manager._tensor_manager is tensor_manager
        assert manager._model is model
        assert manager._cleanup_blocked
        assert manager._failed_state_takeover_parameter_data

    def test_takeover_cleanup_forward_restore_failure_retains_manager(self):
        model = SimpleModel()
        state = _state_for_model(model)
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.return_value = plan
        tensor_manager.prepare_final_model.side_effect = RuntimeError("finalize failed")
        manager = OffloadManager("cleanup-unpatch-failure")
        manager._tensor_manager = tensor_manager
        manager._restore_module_forward = MagicMock(side_effect=RuntimeError("unpatch failed"))

        with pytest.raises(RuntimeError, match="finalize failed"):
            manager.offload_from_state(
                model,
                state,
                config=OffloadConfig(include_patterns=["linear1"], pinned_memory=False),
            )

        tensor_manager.release_memory.assert_not_called()
        tensor_manager.shutdown.assert_not_called()
        assert manager._tensor_manager is tensor_manager
        assert manager._patched_modules == [model.linear1]
        assert manager._cleanup_blocked

    @pytest.mark.parametrize("failure_stage", ["release", "shutdown"])
    def test_release_failure_preserves_patches_and_manager_ownership(self, failure_stage):
        model = SimpleModel()
        tensor_manager = MagicMock()
        if failure_stage == "release":
            tensor_manager.release_memory.side_effect = RuntimeError("release failed")
        else:
            tensor_manager.shutdown.side_effect = RuntimeError("shutdown failed")
        manager = OffloadManager(f"active-{failure_stage}-failure")
        manager._tensor_manager = tensor_manager
        manager._model = model
        manager._current_phase = OffloadPhase.INFERENCE
        manager._state_takeover_active = True
        manager._patch_module_forward(model.linear1, "linear1")

        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            manager.release()

        assert manager._tensor_manager is tensor_manager
        assert manager._model is model
        assert manager._current_phase is OffloadPhase.INFERENCE
        assert manager._state_takeover_active
        assert manager._patched_modules == [model.linear1]
        assert "_ft_original_forward_func" in model.linear1.__dict__
        assert manager._cleanup_blocked
        tensor_manager.shutdown.assert_called_once()

    def test_release_preserves_first_error_when_shutdown_also_fails(self):
        model = SimpleModel()
        tensor_manager = MagicMock()
        tensor_manager.release_memory.side_effect = RuntimeError("first release failure")
        tensor_manager.shutdown.side_effect = RuntimeError("second shutdown failure")
        manager = OffloadManager("double-release-failure")
        manager._tensor_manager = tensor_manager
        manager._model = model
        manager._current_phase = OffloadPhase.INFERENCE
        manager._patch_module_forward(model.linear1, "linear1")

        with pytest.raises(RuntimeError, match="first release failure"):
            manager.release()

        tensor_manager.shutdown.assert_called_once()
        assert manager._tensor_manager is tensor_manager
        assert manager._patched_modules == [model.linear1]
        assert manager._cleanup_blocked

    def test_release_tears_down_tensor_manager_before_unpatching(self):
        model = SimpleModel()
        events = []
        tensor_manager = MagicMock()
        tensor_manager.release_memory.side_effect = lambda: events.append("release")
        tensor_manager.shutdown.side_effect = lambda: events.append("shutdown")
        manager = OffloadManager("release-order")
        manager._tensor_manager = tensor_manager
        manager._patch_module_forward(model.linear1, "linear1")
        restore_forward = manager._restore_module_forward

        def record_restore(module):
            events.append("unpatch")
            restore_forward(module)

        manager._restore_module_forward = record_restore

        manager.release()

        assert events == ["release", "shutdown", "unpatch"]

    def test_release_can_discard_model_without_restoring_tensor_data(self):
        tensor_manager = MagicMock()
        manager = OffloadManager("release-discard-model")
        manager._tensor_manager = tensor_manager

        manager.release(preserve_model=False)

        tensor_manager.release_memory.assert_called_once_with()
        tensor_manager.shutdown.assert_called_once_with(preserve_model=False)
        assert manager._tensor_manager is None

    @pytest.mark.parametrize("loader_type", ["allocation_block_transfer", "raw_block_transfer"])
    def test_release_restores_block_loader_parameter_data(self, loader_type):
        model = SimpleModel().eval()
        state = _takeover_state_for_model(model, loader_type)
        values_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        tensor_manager = _takeover_tensor_manager(loader_type)
        tensor_manager.plan_state_adoption = MagicMock(
            return_value=StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        )
        tensor_manager.execute_state_adoption = MagicMock()
        manager = OffloadManager(f"release-restores-{loader_type}")
        manager._tensor_manager = tensor_manager

        stream_context = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        with (
            patch("flextensor.state_handler.TensorManagerStateHandler._validate_adopted_placement"),
            patch("torch.cuda.Stream", return_value=MagicMock()),
            patch("torch.cuda.stream", return_value=stream_context),
            patch("torch.cuda.current_stream", return_value=MagicMock()),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.device_count", return_value=0),
        ):
            manager.offload_from_state(
                model,
                state,
                config=OffloadConfig(
                    include_patterns=["linear1"],
                    transfer_mode=loader_type,
                    pinned_memory=False,
                ),
            )
            rolling_views = list(
                tensor_manager.tensor_layer_loader.allocation_controller.tensor_id_to_view_map.values()
            )
            manager.release()

        for view in rolling_views:
            view.zero_()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, values_before[name])

    @pytest.mark.parametrize("failure_stage", ["plan", "migration"])
    def test_pre_setup_failure_closes_inactive_manager_without_rollback(self, failure_stage):
        model = SimpleModel()
        state = _state_for_model(model)
        original = next(model.parameters())
        migrated = torch.full_like(original, 11)
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        tensor_manager = MagicMock()
        tensor_manager.plan_state_adoption.return_value = plan
        if failure_stage == "plan":
            tensor_manager.plan_state_adoption.side_effect = RuntimeError("plan failed")
        else:

            def fail_migration(*_):
                original.data = migrated
                raise RuntimeError("migration failed")

            tensor_manager.execute_state_adoption.side_effect = fail_migration
        manager = OffloadManager(f"failed-{failure_stage}")
        manager._tensor_manager = tensor_manager

        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            manager.offload_from_state(model, state, config=OffloadConfig(pinned_memory=False))

        if failure_stage == "plan":
            tensor_manager.execute_state_adoption.assert_not_called()
        else:
            torch.testing.assert_close(original, migrated)
        tensor_manager.shutdown.assert_called_once()
        assert manager._tensor_manager is None
        assert not manager._state_takeover_active

    @pytest.mark.parametrize("loader_type", ["strategy", "allocation_block_transfer", "raw_block_transfer"])
    @pytest.mark.parametrize("failure_stage", ["restore", "loader", "patch", "finalize", "inference", "swap"])
    def test_setup_failure_unpatches_and_releases_owned_loader_state(self, loader_type, failure_stage):  # noqa: C901
        model = SimpleModel().eval()
        state = _takeover_state_for_model(model, loader_type)
        values_before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
        data_ptrs_before = {name: tensor.data_ptr() for name, tensor in model.named_parameters()}
        classes_before = {id(module): type(module) for module in model.modules()}
        plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
        tensor_manager = _takeover_tensor_manager(loader_type)
        tensor_manager.plan_state_adoption = MagicMock(return_value=plan)
        tensor_manager.execute_state_adoption = MagicMock()
        loader_holder = []

        restore_adopted_state = tensor_manager.restore_adopted_state

        def restore(*args):
            result = restore_adopted_state(*args)
            if failure_stage == "restore":
                raise RuntimeError("restore failed")
            return result

        tensor_manager.restore_adopted_state = restore
        prepare_infer_load_mode = tensor_manager.prepare_infer_load_mode

        def prepare_loader():
            result = prepare_infer_load_mode()
            loader_holder.append(tensor_manager.tensor_layer_loader)
            if failure_stage == "loader":
                raise RuntimeError("loader failed")
            return result

        tensor_manager.prepare_infer_load_mode = prepare_loader
        prepare_final_model = tensor_manager.prepare_final_model

        def finalize(*args, **kwargs):
            result = prepare_final_model(*args, **kwargs)
            if failure_stage == "finalize":
                raise RuntimeError("finalize failed")
            return result

        tensor_manager.prepare_final_model = finalize
        initialize_inference = tensor_manager.initialize_inference

        def inference():
            result = initialize_inference()
            if failure_stage == "inference":
                raise RuntimeError("inference failed")
            return result

        tensor_manager.initialize_inference = inference
        manager = OffloadManager(f"failed-{loader_type}-{failure_stage}")
        manager._tensor_manager = tensor_manager
        config = OffloadConfig(
            include_patterns=["linear1"],
            transfer_mode=loader_type,
            pinned_memory=False,
        )
        if failure_stage == "patch":
            offload_modules = manager._offload_modules

            def fail_after_patch(*args):
                offload_modules(*args)
                raise RuntimeError("patch failed")

            manager._offload_modules = fail_after_patch
        if failure_stage == "swap":
            swap_to_new_model = manager._swap_to_new_model

            def fail_after_swap(*args):
                swap_to_new_model(*args)
                raise RuntimeError("swap failed")

            manager._swap_to_new_model = fail_after_swap

        stream_context = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        with (
            patch("flextensor.state_handler.TensorManagerStateHandler._validate_adopted_placement"),
            patch("torch.cuda.Stream", return_value=MagicMock()),
            patch("torch.cuda.stream", return_value=stream_context),
            patch("torch.cuda.current_stream", return_value=MagicMock()),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.device_count", return_value=0),
            pytest.raises(RuntimeError, match=f"{failure_stage} failed"),
        ):
            manager.offload_from_state(model, state, config=config)

        assert manager._tensor_manager is None
        assert manager._model is None
        assert manager._current_phase is OffloadPhase.NOT_INITIALIZED
        assert manager._state_hook_handle is None
        assert not manager._state_takeover_active
        assert not any("_ft_original_forward_func" in module.__dict__ for module in model.modules())
        assert all(type(module) is classes_before[id(module)] for module in model.modules())
        assert not any(
            getattr(hook, "_ft_state_update_hook", False)
            for module in model.modules()
            for hook in module._forward_hooks.values()
        )
        for loader in loader_holder:
            if loader_type in {"allocation_block_transfer", "raw_block_transfer"}:
                assert loader.allocation_controller.block_map_cpu == {}
                assert loader.allocation_controller.block_map_gpu == {}
            else:
                assert loader.cpu_to_gpu_map == {}
        for name, tensor in model.named_parameters():
            torch.testing.assert_close(tensor, values_before[name])
            assert tensor.data_ptr() == data_ptrs_before[name]


class TestOffloadFromProfile:
    """Test cases for the offload_from_profile convenience function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_calls_init_load_profile_offload_in_order(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that offload_from_profile calls init, load_profile, and offload in correct order."""
        from flextensor.offload_manager import offload_from_profile

        mock_offload.return_value = self.model
        config = OffloadConfig(include_patterns=["linear1"])
        profile_dir = str(tmp_path / "profile")

        call_order = []
        mock_init.side_effect = lambda **kw: call_order.append("init")
        mock_load.side_effect = lambda *a, **kw: call_order.append("load_profile")
        mock_offload.side_effect = lambda *a, **kw: (call_order.append("offload"), self.model)[1]

        offload_from_profile(self.model, profile_dir, config=config, name="test")

        assert call_order == ["init", "load_profile", "offload"]

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_passes_correct_arguments(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that offload_from_profile passes the correct arguments to each function."""
        from flextensor.offload_manager import offload_from_profile

        mock_offload.return_value = self.model
        config = OffloadConfig(include_patterns=["linear1"])
        profile_dir = str(tmp_path / "profile")

        offload_from_profile(self.model, profile_dir, config=config, name="mymodel")

        mock_init.assert_called_once_with(config=config, name="mymodel")
        mock_load.assert_called_once_with(profile_dir, model=self.model, name="mymodel")
        mock_offload.assert_called_once_with(self.model, config=config, name="mymodel", compile_fn=None)

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_returns_offload_result(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that offload_from_profile returns the result from offload()."""
        from flextensor.offload_manager import offload_from_profile

        proxy_model = SimpleModel()
        mock_offload.return_value = proxy_model
        config = OffloadConfig(include_patterns=["linear1"])
        profile_dir = str(tmp_path / "profile")

        result = offload_from_profile(self.model, profile_dir, config=config, name="test")

        assert result is proxy_model

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_uses_default_name(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that offload_from_profile uses DEFAULT_MANAGER_NAME when name is not provided."""
        from flextensor.offload_manager import DEFAULT_MANAGER_NAME, offload_from_profile

        mock_offload.return_value = self.model
        profile_dir = str(tmp_path / "profile")

        offload_from_profile(self.model, profile_dir)

        mock_init.assert_called_once_with(config=None, name=DEFAULT_MANAGER_NAME)
        mock_load.assert_called_once_with(profile_dir, model=self.model, name=DEFAULT_MANAGER_NAME)
        mock_offload.assert_called_once_with(self.model, config=None, name=DEFAULT_MANAGER_NAME, compile_fn=None)

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_forwards_compile_fn(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that compile_fn is forwarded to offload()."""
        from flextensor.offload_manager import offload_from_profile

        mock_offload.return_value = self.model
        config = OffloadConfig(include_patterns=["linear1"])
        profile_dir = str(tmp_path / "profile")

        def compile_fn(module: torch.nn.Module) -> torch.nn.Module:
            return module

        offload_from_profile(self.model, profile_dir, config=config, name="test", compile_fn=compile_fn)

        mock_offload.assert_called_once_with(self.model, config=config, name="test", compile_fn=compile_fn)

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_propagates_load_profile_error(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that errors from load_profile propagate without calling offload."""
        from flextensor.offload_manager import offload_from_profile

        mock_load.side_effect = FileNotFoundError("profile.json not found")
        profile_dir = str(tmp_path / "nonexistent")

        with pytest.raises(FileNotFoundError, match=r"profile\.json not found"):
            offload_from_profile(self.model, profile_dir)

        mock_init.assert_called_once()
        mock_offload.assert_not_called()

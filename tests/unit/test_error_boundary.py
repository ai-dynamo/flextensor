# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for error boundary observability.

Tests the diagnostic snapshot helper, error boundary decorator, and
patched forward error boundary. All GPU/host resources are mocked.
"""

import logging
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from flextensor.offload_manager import OffloadManager, OffloadPhase, _error_boundary, _GiB, _log_diagnostic_snapshot


class TestLogDiagnosticSnapshot:
    """Tests for _log_diagnostic_snapshot helper."""

    def test_logs_all_fields(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify snapshot logs phase, iteration, manager name, GPU info, host info, pid."""
        om = OffloadManager("test_mgr")
        om._current_phase = OffloadPhase.PROFILING
        om._iteration_count = 5

        mock_props = SimpleNamespace(total_memory=80 * _GiB)
        mock_vm = SimpleNamespace(
            used=32 * _GiB,
            available=48 * _GiB,
            total=80 * _GiB,
        )

        with (
            patch("flextensor.offload_manager.torch.cuda.device_count", return_value=1),
            patch("flextensor.offload_manager.torch.cuda.memory_allocated", return_value=10 * _GiB),
            patch("flextensor.offload_manager.torch.cuda.memory_reserved", return_value=20 * _GiB),
            patch("flextensor.offload_manager.torch.cuda.get_device_properties", return_value=mock_props),
            patch("flextensor.offload_manager.psutil.virtual_memory", return_value=mock_vm),
            caplog.at_level(logging.ERROR, logger="flextensor.offload_manager"),
        ):
            try:
                raise ValueError("test error")
            except ValueError:
                _log_diagnostic_snapshot("test_label", om)

        assert "FlexTensor error in test_label" in caplog.text
        assert "phase=profiling" in caplog.text
        assert "iteration=5" in caplog.text
        assert "manager='test_mgr'" in caplog.text
        assert "gpu0:" in caplog.text
        assert "alloc=10.00GiB" in caplog.text
        assert "reserved=20.00GiB" in caplog.text
        assert "total=80.00GiB" in caplog.text
        assert "host:" in caplog.text
        assert "avail=48.00GiB" in caplog.text
        assert f"pid={os.getpid()}" in caplog.text

    def test_diagnostic_failure_does_not_mask_original(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify GPU/host query failures degrade to '<unavailable>' without masking the error."""
        om = OffloadManager("test_degrade")
        om._current_phase = OffloadPhase.DISCOVERY
        om._iteration_count = 2

        with (
            patch("flextensor.offload_manager.torch.cuda.device_count", side_effect=RuntimeError("cuda broken")),
            patch("flextensor.offload_manager.psutil.virtual_memory", side_effect=OSError("psutil broken")),
            caplog.at_level(logging.ERROR, logger="flextensor.offload_manager"),
        ):
            try:
                raise ValueError("original error")
            except ValueError:
                _log_diagnostic_snapshot("test_degrade_label", om)

        assert "FlexTensor error in test_degrade_label" in caplog.text
        assert "phase=discovery" in caplog.text
        assert "gpu: <unavailable>" in caplog.text
        assert "host: <unavailable>" in caplog.text

    def test_snapshot_with_none_om(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify snapshot works when om is None (no FlexTensor state available)."""
        with (
            patch("flextensor.offload_manager.torch.cuda.device_count", return_value=0),
            patch("flextensor.offload_manager.psutil.virtual_memory", side_effect=OSError("no psutil")),
            caplog.at_level(logging.ERROR, logger="flextensor.offload_manager"),
        ):
            try:
                raise RuntimeError("some error")
            except RuntimeError:
                _log_diagnostic_snapshot("test_none_om", None)

        assert "FlexTensor error in test_none_om" in caplog.text
        assert "phase=" not in caplog.text  # No OM phase when om is None
        assert f"pid={os.getpid()}" in caplog.text


class TestErrorBoundary:
    """Tests for @_error_boundary decorator."""

    def test_reraises_original_exception(self) -> None:
        """Verify the decorator re-raises the original exception unchanged."""
        om = OffloadManager("test_reraise")

        @_error_boundary
        def _failing_method(self: OffloadManager) -> None:
            raise ValueError("original message")

        with pytest.raises(ValueError, match="original message"):
            _failing_method(om)

    def test_no_overhead_on_success(self) -> None:
        """Verify the decorator is transparent on the happy path."""
        om = OffloadManager("test_success")

        @_error_boundary
        def _succeeding_method(self: OffloadManager) -> str:
            return "result"

        assert _succeeding_method(om) == "result"

    def test_logs_diagnostics_on_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify the decorator triggers diagnostic logging on exception."""
        om = OffloadManager("test_diag")
        om._current_phase = OffloadPhase.INFERENCE
        om._iteration_count = 42

        @_error_boundary
        def _failing_method(self: OffloadManager) -> None:
            raise RuntimeError("boom")

        with (
            patch("flextensor.offload_manager.torch.cuda.device_count", return_value=0),
            patch("flextensor.offload_manager.psutil.virtual_memory", side_effect=OSError),
            caplog.at_level(logging.ERROR, logger="flextensor.offload_manager"),
            pytest.raises(RuntimeError, match="boom"),
        ):
            _failing_method(om)

        assert "phase=inference" in caplog.text
        assert "iteration=42" in caplog.text

    def test_transition_methods_are_decorated(self) -> None:
        """Verify all 4 transition methods have the error boundary applied."""
        for method_name in (
            "_transition_to_warmup",
            "_transition_to_profile",
            "_transition_to_inference",
            "_initialize_from_shm",
        ):
            method = getattr(OffloadManager, method_name)
            assert hasattr(method, "__wrapped__"), f"{method_name} is not decorated with @_error_boundary"

    def test_initialize_from_shm_error_boundary(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify _initialize_from_shm logs diagnostics on failure and re-raises."""
        om = OffloadManager("test_shm")
        om._current_phase = OffloadPhase.NOT_INITIALIZED

        # Use a concrete class that satisfies _ShmCoordinatorLike (beartype checks structural compliance).
        class _FakeCoordinator:
            namespace: str = "test-ns"
            is_creator: bool = False

            def wait_for_ready(self, timeout: float | None = None) -> None:
                raise RuntimeError("shm timeout")

            def read_profile(self):  # type: ignore[return]
                ...

        mock_coordinator = _FakeCoordinator()

        with (
            patch("flextensor.offload_manager.torch.cuda.device_count", return_value=0),
            patch("flextensor.offload_manager.psutil.virtual_memory", side_effect=OSError),
            caplog.at_level(logging.ERROR, logger="flextensor.offload_manager"),
            pytest.raises(RuntimeError, match="shm timeout"),
        ):
            om._initialize_from_shm(mock_coordinator, nn.Linear(2, 2))

        assert "OffloadManager._initialize_from_shm" in caplog.text
        assert "phase=not_initialized" in caplog.text


class TestPatchedForwardErrorBoundary:
    """Tests for error boundary in patched_forward closure."""

    def test_patched_forward_logs_diagnostics_on_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify patched forward logs diagnostic snapshot with offload_name context."""

        class FailingModule(nn.Module):
            def forward(self, x: object) -> object:
                raise RuntimeError("forward exploded")

        module = FailingModule()
        om = OffloadManager("test_fwd")
        om._current_phase = OffloadPhase.DISCOVERY
        om._iteration_count = 1

        # Patched forward calls _tensor_manager.trap directly (bypasses
        # offload_block to avoid the user-facing skip_discovery guard).
        @contextmanager
        def noop_block(name: str):
            yield

        om._tensor_manager = MagicMock()
        om._tensor_manager.trap = noop_block
        om._patch_module_forward(module, "model.layers.5")

        with (
            patch("flextensor.offload_manager.torch.cuda.device_count", return_value=0),
            patch("flextensor.offload_manager.psutil.virtual_memory", side_effect=OSError),
            caplog.at_level(logging.ERROR, logger="flextensor.offload_manager"),
            pytest.raises(RuntimeError, match="forward exploded"),
        ):
            module.forward(None)

        assert "FlexTensor error in forward(model.layers.5)" in caplog.text
        assert "phase=discovery" in caplog.text

    def test_patched_forward_reraises_original(self) -> None:
        """Verify original exception propagates unchanged through patched forward."""

        class FailingModule(nn.Module):
            def forward(self, x: object) -> object:
                raise ValueError("specific error")

        module = FailingModule()
        om = OffloadManager("test_fwd_reraise")

        @contextmanager
        def noop_block(name: str):
            yield

        om._tensor_manager = MagicMock()
        om._tensor_manager.trap = noop_block
        om._patch_module_forward(module, "test_block")

        with (
            patch("flextensor.offload_manager.torch.cuda.device_count", return_value=0),
            patch("flextensor.offload_manager.psutil.virtual_memory", side_effect=OSError),
            pytest.raises(ValueError, match="specific error"),
        ):
            module.forward(None)

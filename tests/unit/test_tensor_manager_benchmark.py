# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TensorManager benchmark integration functionality."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from flextensor.benchmark_tensor_mode import BenchmarkReplace, NoOpBenchmark, PreloadToDevice, TensorBenchmarkMode
from flextensor.collectors import IterativeLayerStatistics, LayerStatistics, TensorStatistics
from flextensor.host_pinning import HostPinner
from flextensor.profile_block_controller import ProfileBlockController
from flextensor.strategy import BlockStrategyData, KnapsackStrategy, StrategyResult
from flextensor.tensor import TraceTensor
from flextensor.tensor_manager import TensorManager


@pytest.fixture(autouse=True)
def _fake_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend CUDA is available so ``TensorManager(pinned_memory=True)``
    construction doesn't raise on CPU-only CI hosts. ``make_host_pinner``
    only inspects ``torch.cuda.is_available()`` for the torch-mode path
    used here; no real cudart calls occur.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


class TestTensorManagerBenchmarkIntegration:
    """Test cases for TensorManager benchmark integration."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.device_gpu = torch.device("cuda:0")
        self.device_cpu = torch.device("cpu")
        self.strategy = KnapsackStrategy(scale=0.8)

    def test_tensor_manager_default_benchmark_cls_is_noop(self) -> None:
        """Default (non-tracing) TensorManager uses NoOpBenchmark."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )

        assert tensor_manager._benchmark_cls == NoOpBenchmark

    def test_tensor_manager_tracing_uses_benchmark_replace(self) -> None:
        """With _use_trace_tensor=True, TensorManager uses BenchmarkReplace."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            _use_trace_tensor=True,
            profile_mode="getter",
        )

        assert tensor_manager._benchmark_cls == BenchmarkReplace

    def test_trace_tensor_rebinds_is_traced(self) -> None:
        """_use_trace_tensor=True rebinds is_traced to is_traced_trace_tensor."""
        tm_default = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tm_tracing = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            _use_trace_tensor=True,
            profile_mode="getter",
        )

        assert tm_default.is_traced == tm_default.__class__.is_traced.__get__(tm_default)
        assert tm_tracing.is_traced == tm_tracing.is_traced_trace_tensor

    def test_default_is_traced_uses_id_set(self) -> None:
        """Default is_traced returns True only for tensors whose id is in traced_tensors."""
        tm = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor = torch.zeros(4)
        assert tm.is_traced(tensor) is False

        tm.traced_tensors.add(id(tensor))
        assert tm.is_traced(tensor) is True

        assert tm.is_traced("not a tensor") is False

    def test_trace_tensor_is_traced_uses_isinstance(self) -> None:
        """With _use_trace_tensor=True, is_traced checks isinstance(TraceTensor)."""
        tm = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            _use_trace_tensor=True,
            profile_mode="getter",
        )
        plain = torch.zeros(4)
        traced = TraceTensor(torch.zeros(4))

        assert tm.is_traced(plain) is False
        assert tm.is_traced(traced) is True

        tm.traced_tensors.add(id(plain))
        assert tm.is_traced(plain) is False

    def test_benchmark_context_method_exists(self) -> None:
        """Test that benchmark_context method exists and is callable."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )

        assert hasattr(tensor_manager, "benchmark_context")
        assert callable(tensor_manager.benchmark_context)

    def test_benchmark_context_forwards_host_pinner(self) -> None:
        """Wiring guard: :meth:`TensorManager.benchmark_context` must pass
        ``host_pinner=self.host_pinner`` to ``self._benchmark_cls(...)``. A
        regression that drops the kwarg leaves the benchmark with a default
        :class:`HostPinner` (torch mode), so profiling stats are collected
        with a different pinner than the one used at inference time —
        :attr:`TensorStatistics.load_time_ms` then no longer represents
        production transfer cost. CI stays green because every other
        benchmark test only checks ``pinned_memory`` and ``iterations``.

        Same regression class as the allocation_block_transfer / raw_block_transfer
        / OffloadManager forwarding guards in
        ``test_tensor_manager_pinned_memory_mode.py``.
        """
        captured: dict = {}

        class CapturingBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations, host_pinner=None):
                captured["host_pinner"] = host_pinner
                self.results = {"tensor_statistics_map": {}, "tensors_map": {}}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = CapturingBenchmark

        with tensor_manager.benchmark_context(iterations=1):
            pass

        assert captured["host_pinner"] is tensor_manager.host_pinner, (
            "benchmark_context dropped host_pinner=self.host_pinner — profiling "
            "would silently use a default HostPinner (torch mode) instead of the "
            "configured pinner, so stats no longer reflect inference behavior."
        )

    def test_benchmark_context_creates_benchmark_instance(self) -> None:
        """Test that benchmark_context creates benchmark instance with correct parameters."""

        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations, host_pinner=None):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {"tensor_statistics_map": {}, "tensors_map": {}}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        with tensor_manager.benchmark_context(iterations=5) as benchmark:
            assert isinstance(benchmark, TestMockBenchmark)
            assert benchmark.device_gpu == self.device_gpu
            assert benchmark.pinned_memory is True
            assert benchmark.iterations == 5

    def test_benchmark_context_default_iterations(self) -> None:
        """Test that benchmark_context uses default iterations parameter."""

        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations, host_pinner=None):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {"tensor_statistics_map": {}, "tensors_map": {}}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        with tensor_manager.benchmark_context() as benchmark:
            assert isinstance(benchmark, TestMockBenchmark)
            assert benchmark.device_gpu == self.device_gpu

    def test_benchmark_context_automatic_stats_integration(self) -> None:
        """Test that benchmark results are automatically integrated into TensorManager."""

        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations, host_pinner=None):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {
                    "tensor_statistics_map": {123: "stats1", 456: "stats2"},
                    "tensors_map": {123: "tensor_obj1", 456: "tensor_obj2"},
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        assert tensor_manager.tensor_statistics_map == {}
        assert tensor_manager.tensors_map == {}

        with tensor_manager.benchmark_context():
            pass

        assert tensor_manager.tensor_statistics_map == {123: "stats1", 456: "stats2"}
        assert tensor_manager.tensors_map == {123: "tensor_obj1", 456: "tensor_obj2"}

    def test_benchmark_context_stats_replacement(self) -> None:
        """Test that benchmark results replace existing TensorManager stats."""

        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations, host_pinner=None):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {
                    "tensor_statistics_map": {123: "new_stats"},
                    "tensors_map": {123: "new_tensor_obj"},
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        tensor_manager.tensor_statistics_map = {111: "existing_stats"}
        tensor_manager.tensors_map = {111: "existing_tensor_obj"}

        with tensor_manager.benchmark_context():
            pass

        assert tensor_manager.tensor_statistics_map == {123: "new_stats"}
        assert tensor_manager.tensors_map == {123: "new_tensor_obj"}

    def test_benchmark_context_exception_handling(self) -> None:
        """Test that stats are still integrated even if exception occurs in context."""

        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations, host_pinner=None):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {
                    "tensor_statistics_map": {123: "stats1"},
                    "tensors_map": {123: "tensor_obj1"},
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        with pytest.raises(ValueError), tensor_manager.benchmark_context():
            raise ValueError("Test exception")

        assert tensor_manager.tensor_statistics_map == {123: "stats1"}
        assert tensor_manager.tensors_map == {123: "tensor_obj1"}

    def test_hardcoded_internal_fields(self) -> None:
        """Test that release_tensors and direct_enabled are hardcoded, pinned_memory is configurable."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )

        assert tensor_manager.pinned_memory is True
        assert tensor_manager.release_tensors is True
        assert tensor_manager.direct_enabled is True
        assert tensor_manager.tensor_manager_load_strategy == self.strategy
        assert hasattr(tensor_manager, "prepare_warmup_mode")
        assert hasattr(tensor_manager, "prepare_profile_mode")
        assert hasattr(tensor_manager, "prepare_infer_mode")
        assert hasattr(tensor_manager, "trap")
        assert hasattr(tensor_manager, "release_memory")
        assert hasattr(tensor_manager, "is_traced")

    def test_pinned_memory_configurable(self) -> None:
        """Test that pinned_memory can be set to False."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            pinned_memory=False,
        )

        assert tensor_manager.pinned_memory is False

    @pytest.mark.parametrize("loader_type", ["allocation_block_transfer", "raw_block_transfer"])
    def test_block_transfer_requires_direct_mode(self, loader_type: str) -> None:
        """Block transfer loaders must reject ``profile_mode='torch_function'``."""
        with pytest.raises(ValueError, match=r"torch_function.*incompatible"):
            TensorManager(
                device_gpu=self.device_gpu,
                tensor_manager_load_strategy=self.strategy,
                loader_type=loader_type,
                profile_mode="torch_function",
            )

    def test_strategy_loader_allows_torch_function_profile_mode(self) -> None:
        """Strategy loader accepts ``profile_mode='torch_function'``."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            loader_type="strategy",
            profile_mode="torch_function",
        )

        assert tensor_manager.direct_enabled is False
        assert tensor_manager.profile_mode == "torch_function"

    def test_direct_mode_flag_decoupled_from_profile_mode(self) -> None:
        """``_direct_mode`` is the runtime-family axis, independent of the
        profile-phase variant. ``_direct_mode=False`` selects the indirect
        runtime even with a ``view``/``getter`` ``profile_mode`` (the variant is
        then ignored), and ``profile_mode='torch_function'`` forces it ``False``.
        """
        # Default: direct family.
        tm_default = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        assert tm_default._direct_mode is True
        assert tm_default.direct_enabled is True

        # Explicit indirect family with a view profile_mode -> view is ignored.
        tm_indirect = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            loader_type="strategy",
            profile_mode="view",
            _direct_mode=False,
        )
        assert tm_indirect.direct_enabled is False
        assert tm_indirect._profile_uses_views is False

        # profile_mode='torch_function' forces the indirect family.
        tm_tf = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            loader_type="strategy",
            profile_mode="torch_function",
        )
        assert tm_tf._direct_mode is False

    def test_indirect_mode_rejects_block_loader(self) -> None:
        """``_direct_mode=False`` is incompatible with block-transfer loaders,
        same constraint as ``profile_mode='torch_function'``.
        """
        with pytest.raises(ValueError, match=r"[Ii]ndirect mode.*incompatible"):
            TensorManager(
                device_gpu=self.device_gpu,
                tensor_manager_load_strategy=self.strategy,
                loader_type="allocation_block_transfer",
                _direct_mode=False,
            )

    def test_profile_mode_view_accepts_strategy_loader(self) -> None:
        """``profile_mode='view'`` is accepted with ``loader_type='strategy'``.

        The view-mode profile controller is self-contained and torn down before
        the inference loader is built, so the profile mechanism and inference
        loader are independent.
        """
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            loader_type="strategy",
            profile_mode="view",
        )

        assert tensor_manager.profile_mode == "view"
        assert tensor_manager.loader_type == "strategy"

    def test_profile_mode_view_rejects_trace_tensor(self) -> None:
        """``profile_mode='view'`` is incompatible with trace-tensor benchmarking."""
        with pytest.raises(ValueError, match=r"view.*_use_trace_tensor"):
            TensorManager(
                device_gpu=self.device_gpu,
                tensor_manager_load_strategy=self.strategy,
                loader_type="allocation_block_transfer",
                profile_mode="view",
                _use_trace_tensor=True,
            )

    def test_profile_mode_unknown_rejected(self) -> None:
        """Unknown ``profile_mode`` value raises immediately.

        ``profile_mode`` carries a Literal type hint, so beartype rejects unknown
        values before our internal validator runs. Either error is acceptable;
        this asserts the name surfaces in the message.
        """
        with pytest.raises(Exception, match="profile_mode"):
            TensorManager(
                device_gpu=self.device_gpu,
                tensor_manager_load_strategy=self.strategy,
                profile_mode="bogus",  # type: ignore[arg-type]
            )


class MockBenchmarkReplace(TensorBenchmarkMode):
    """Mock benchmark class for testing."""

    def __init__(self, device_gpu, pinned_memory, iterations, host_pinner=None):
        self.device_gpu = device_gpu
        self.pinned_memory = pinned_memory
        self.iterations = iterations
        self.results = {
            "tensor_statistics_map": {999: "mock_stats"},
            "tensors_map": {999: "mock_tensor_obj"},
        }

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        return None

    def get_results(self):
        return self.results


class TestTensorManagerBenchmarkParametrization:
    """Test cases for TensorManager benchmark class parametrization."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.device_gpu = torch.device("cuda:0")
        self.strategy = KnapsackStrategy(scale=0.8)

    def test_custom_benchmark_class_usage(self) -> None:
        """Test that a custom benchmark class can be injected and used."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = MockBenchmarkReplace

        with tensor_manager.benchmark_context(iterations=5) as benchmark:
            assert isinstance(benchmark, MockBenchmarkReplace)
            assert benchmark.device_gpu == self.device_gpu
            assert benchmark.pinned_memory
            assert benchmark.iterations == 5

        assert tensor_manager.tensor_statistics_map == {999: "mock_stats"}
        assert tensor_manager.tensors_map == {999: "mock_tensor_obj"}

    def test_abstract_base_class_interface(self) -> None:
        """Test that both BenchmarkReplace and PreloadToDevice implement TensorBenchmarkMode interface."""
        assert issubclass(BenchmarkReplace, TensorBenchmarkMode)
        assert issubclass(PreloadToDevice, TensorBenchmarkMode)

        benchmark_replace = BenchmarkReplace(
            self.device_gpu, pinned_memory=True, iterations=10, host_pinner=HostPinner()
        )
        preload_to_device = PreloadToDevice(self.device_gpu, pinned_memory=True, iterations=10)

        assert hasattr(benchmark_replace, "get_results")
        assert hasattr(preload_to_device, "get_results")
        assert callable(benchmark_replace.get_results)
        assert callable(preload_to_device.get_results)

        br_results = benchmark_replace.get_results()
        ptd_results = preload_to_device.get_results()

        expected_keys = {"tensor_statistics_map", "tensors_map"}
        assert set(br_results.keys()) == expected_keys
        assert set(ptd_results.keys()) == expected_keys


def _make_layer(label: str, n_tensors: int = 1, duration: float = 10.0) -> LayerStatistics:
    """Helper to create a LayerStatistics with dummy tensors."""
    tensors = [
        TensorStatistics(tensor_id=i, name=f"{label}_t{i}", size_bytes=1024, load_time_ms=0.1) for i in range(n_tensors)
    ]
    return LayerStatistics(label=label, tensors=tensors, duration=duration)


def _make_gap_layer(label: str) -> LayerStatistics:
    """Helper to create a gap layer (no tensors)."""
    return LayerStatistics(label=label, tensors=[], duration=5.0)


def _make_tm(**kwargs) -> TensorManager:
    """Create a TensorManager with minimal required args."""
    defaults = {
        "device_gpu": torch.device("cuda:0"),
        "tensor_manager_load_strategy": KnapsackStrategy(scale=0.8),
    }
    defaults.update(kwargs)
    return TensorManager(**defaults)


def test_set_model_accepts_dict_metadata() -> None:
    tm = _make_tm()
    model = {"weight": torch.zeros(1), "metadata": "value"}

    tm.set_model(model)

    assert tm.model is model


@patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=None)
def test_block_loader_normalizes_list_block_sizes(_budget) -> None:
    tm = _make_tm(loader_type="raw_block_transfer")
    tm.stats = []
    tm.memory_transfer_stats = {}
    tm.model = None
    tm.tensors_map = {}
    tm._create_loader = MagicMock()
    tm._log_inference_diagnostics = MagicMock()
    tm.tensor_manager_load_strategy = MagicMock()
    tm.tensor_manager_load_strategy.compute.return_value = StrategyResult(
        strategy_map={},
        block_data=BlockStrategyData(
            allocation_ordered={},
            label_to_block_id={},
            transfer_to_compute_map={},
            label_to_size_map={},
            block_sizes=[128, 256],
        ),
    )

    tm._compute_strategy_and_build_loader()

    loader_data = tm._create_loader.call_args.args[0]
    assert loader_data.block_sizes == {0: 128, 1: 256}


class TestUntracedTensorDiscoveryBranch:
    """Verify enable_untraced_tensor_discovery gates discover_untraced_tensors_for_layers."""

    def _prepare_tm(self, *, discovery_enabled: bool) -> TensorManager:
        # ``profile_mode='getter'`` here because these tests drive
        # ``prepare_profile_direct_mode`` directly without going through
        # ``prepare_profile_direct_mode_model``, so the view path's
        # pre-built ``ProfileBlockController`` is unavailable.
        tm = _make_tm(
            _enable_untraced_tensor_discovery=discovery_enabled,
            profile_mode="getter",
        )
        tm.layer_statistics_collector = MagicMock()
        # ``UntimedTrapRescuer`` type-checks ``list[IterativeLayerStatistics]``;
        # the collector produces the same type in production.
        tm.layer_statistics_collector.get_layer_stats.return_value = [
            IterativeLayerStatistics(label="layer_0", tensor_ids=set(), duration=10.0),
        ]
        tm.model = MagicMock()
        tm.tensor_id_to_name_map = {}
        tm.module_tracker = None
        return tm

    @patch("flextensor.tensor_manager.TensorLayerLoader")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.discover_untraced_tensors_for_layers")
    def test_discovery_called_when_enabled(self, mock_discover, mock_filter, _mock_loader) -> None:
        mock_filter.return_value.filter_by_tensor_ids.return_value = [
            IterativeLayerStatistics(label="layer_0", tensor_ids=set(), duration=10.0),
        ]
        mock_discover.return_value = [
            IterativeLayerStatistics(label="layer_0", tensor_ids=set(), duration=10.0),
        ]
        tm = self._prepare_tm(discovery_enabled=True)
        tm.prepare_profile_direct_mode()
        mock_discover.assert_called_once()

    @patch("flextensor.tensor_manager.TensorLayerLoader")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.discover_untraced_tensors_for_layers")
    def test_discovery_skipped_when_disabled(self, mock_discover, mock_filter, _mock_loader) -> None:
        mock_filter.return_value.filter_by_tensor_ids.return_value = [
            IterativeLayerStatistics(label="layer_0", tensor_ids=set(), duration=10.0),
        ]
        tm = self._prepare_tm(discovery_enabled=False)
        tm.prepare_profile_direct_mode()
        mock_discover.assert_not_called()


class TestPrepareProfileDirectModeOrdering:
    """Verify view-mode requires ``prepare_profile_direct_mode_model`` first.

    For ``profile_mode='view'`` the setup is split across two methods:
    ``prepare_profile_direct_mode_model`` builds the
    ``ProfileBlockController`` and patches the model with views, then
    ``prepare_profile_direct_mode`` wires ``TrapProfileView`` against it.
    Reversing or skipping the first call must raise loudly so the bug
    surfaces at setup time, not as silent garbage profile measurements later.
    """

    def test_raises_when_called_before_prepare_model_in_view_mode(self) -> None:
        tm = _make_tm(profile_mode="view")
        assert not isinstance(tm.tensor_layer_loader, ProfileBlockController)

        with pytest.raises(RuntimeError, match="view-mode controller missing"):
            tm.prepare_profile_direct_mode()


class TestInitializeProfileViewModeFailureCleanup:
    """``initialize_profile`` must drop the view-mode controller if patching
    raises after ``ProfileBlockController.__init__`` has allocated the GPU
    block. Otherwise the multi-GiB block leaks until process exit, since the
    only path that calls ``_teardown_profile_block_controller`` in normal flow
    is ``prepare_infer_mode``, which never runs after a failed profile setup.
    """

    def test_view_mode_setup_failure_releases_controller(self) -> None:
        tm = _make_tm(profile_mode="view")
        tm.model = MagicMock()
        tm._move_non_offloaded_tensors_to_gpu = MagicMock()

        # Stand-in controller: ``spec=ProfileBlockController`` makes the mock
        # pass ``isinstance`` so the teardown path actually fires.
        fake_controller = MagicMock(spec=ProfileBlockController)

        def _fake_prepare_model(model):
            tm.tensor_layer_loader = fake_controller
            raise RuntimeError("simulated patching failure")

        tm.prepare_profile_direct_mode_model = _fake_prepare_model

        with pytest.raises(RuntimeError, match="simulated patching failure"):
            tm.initialize_profile()

        fake_controller.teardown.assert_called_once_with(tm.model, tm.tensors_map)
        assert tm.tensor_layer_loader is None


class TestShutdownTearsDownProfileController:
    """``TensorManager.shutdown()`` must restore ``.data`` before releasing the
    GPU block when a view-mode profile is still in flight. Otherwise patched
    parameters end up aliasing freed storage -> silent garbage on any
    subsequent use of the model.
    """

    def test_shutdown_runs_controller_teardown_first(self) -> None:
        tm = _make_tm(profile_mode="view")
        tm.model = MagicMock()
        tm.host_pinner = MagicMock()

        controller = MagicMock(spec=ProfileBlockController)
        tm.tensor_layer_loader = controller

        tm.shutdown()

        # ``.data`` restoration runs (via ``_teardown_profile_block_controller``)
        # and the loader ref is cleared, so the subsequent ``loader.shutdown()``
        # branch is a no-op rather than freeing a still-referenced block.
        controller.teardown.assert_called_once_with(tm.model, tm.tensors_map)
        controller.shutdown.assert_not_called()
        assert tm.tensor_layer_loader is None
        tm.host_pinner.release_all.assert_called_once()


class TestTeardownProfileBlockControllerClearsRefs:
    """``_teardown_profile_block_controller`` must clear
    ``tensor_layer_loader`` even when ``controller.teardown`` raises.
    Otherwise the manager keeps a dangling ref into a shut-down controller,
    breaking the next ``release_memory()`` (which routes through the loader)
    and any subsequent profile setup that branches on the loader's type.
    """

    def test_loader_cleared_when_teardown_raises(self) -> None:
        tm = _make_tm(profile_mode="view")
        tm.model = MagicMock()

        controller = MagicMock(spec=ProfileBlockController)
        controller.teardown.side_effect = RuntimeError("teardown failed")
        tm.tensor_layer_loader = controller

        with pytest.raises(RuntimeError, match="teardown failed"):
            tm._teardown_profile_block_controller()

        assert tm.tensor_layer_loader is None


class TestPrepareProfileDirectModeModelDictDispatch:
    """View-mode + dict-shaped models (vLLM-style flows).

    ``prepare_profile_direct_mode_model`` branches on ``isinstance(model, dict)``
    to choose between ``copy.copy(model)`` and ``create_model_with_shared_tensors``.
    Now that ``profile_mode='view'`` is the default, dict-using callers exercise
    this branch in the hot path; the integration test only covers ``nn.Module``,
    so this unit pins that the dict input is *copied* (not aliased) and routed
    through the view-profile-model construction.
    """

    def test_view_mode_with_dict_model_routes_through_view_path(self) -> None:
        tm = _make_tm(profile_mode="view")
        prepared_dict = {"sentinel": object()}
        tm._prepare_view_profile_model = MagicMock(return_value=prepared_dict)

        original = {"weight": torch.zeros(4, 4)}
        result = tm.prepare_profile_direct_mode_model(original)

        # Routed through the view path with a *copy* of the input dict.
        tm._prepare_view_profile_model.assert_called_once()
        passed_model = tm._prepare_view_profile_model.call_args.args[0]
        assert isinstance(passed_model, dict)
        assert passed_model is not original
        assert passed_model == original
        assert result is prepared_dict


class TestAutoEnableRearrangeTransfers:
    """Verify rearrange_transfers is auto-enabled when gap layers are detected.

    Tests exercise prepare_infer_mode with heavy patching to isolate the
    auto-enable branch that checks for permanent gap layers.
    """

    def _prepare_tm_for_infer(self, *, rearrange: bool = False) -> TensorManager:
        tm = _make_tm(loader_type="strategy", _rearrange_transfers=rearrange)
        tm.layer_statistics_collector = MagicMock()
        tm.layer_statistics_collector.get_layer_stats.return_value = []
        tm.model = MagicMock(spec=[])
        tm.tensor_id_to_name_map = {}
        tm.tensors_map = {}
        return tm

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps", return_value=True)
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.report_profiling_quality", return_value=None)
    def test_auto_enables_on_gap_layers(
        self, _report_quality, _filter, mock_compute_stats, _bench, _mem, _budget, _remove, mock_has_gaps, _loader
    ) -> None:
        stats_with_gaps = [_make_layer("layer_0"), _make_gap_layer("gap"), _make_layer("layer_2")]
        mock_compute_stats.return_value = stats_with_gaps

        tm = self._prepare_tm_for_infer(rearrange=False)
        result = MagicMock()
        result.strategy_map = {"layer_0": [], "gap": [], "layer_2": []}
        result.block_data = None
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = result

        tm.prepare_infer_mode()

        assert tm.rearrange_transfers is True
        mock_has_gaps.assert_called_once()

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps", return_value=False)
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.report_profiling_quality", return_value=None)
    def test_no_auto_enable_without_transfer_gaps(
        self, _report_quality, _filter, mock_compute_stats, _bench, _mem, _budget, _remove, mock_has_gaps, _loader
    ) -> None:
        stats_with_gaps = [_make_layer("layer_0"), _make_gap_layer("gap"), _make_layer("layer_2")]
        mock_compute_stats.return_value = stats_with_gaps

        tm = self._prepare_tm_for_infer(rearrange=False)
        result = MagicMock()
        result.strategy_map = {"layer_0": [], "gap": [], "layer_2": []}
        result.block_data = None
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = result

        tm.prepare_infer_mode()

        assert tm.rearrange_transfers is False

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps")
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.report_profiling_quality", return_value=None)
    def test_no_auto_enable_without_gap_layers(
        self, _report_quality, _filter, mock_compute_stats, _bench, _mem, _budget, _remove, mock_has_gaps, _loader
    ) -> None:
        stats_no_gaps = [_make_layer("layer_0"), _make_layer("layer_1"), _make_layer("layer_2")]
        mock_compute_stats.return_value = stats_no_gaps

        tm = self._prepare_tm_for_infer(rearrange=False)
        result = MagicMock()
        result.strategy_map = {"layer_0": [], "layer_1": [], "layer_2": []}
        result.block_data = None
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = result

        tm.prepare_infer_mode()

        assert tm.rearrange_transfers is False
        mock_has_gaps.assert_not_called()

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps")
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.report_profiling_quality", return_value=None)
    def test_already_enabled_skips_auto_detection(
        self, _report_quality, _filter, mock_compute_stats, _bench, _mem, _budget, _remove, mock_has_gaps, _loader
    ) -> None:
        stats_with_gaps = [_make_layer("layer_0"), _make_gap_layer("gap"), _make_layer("layer_2")]
        mock_compute_stats.return_value = stats_with_gaps

        tm = self._prepare_tm_for_infer(rearrange=True)
        result = MagicMock()
        result.strategy_map = {"layer_0": [], "gap": [], "layer_2": []}
        result.block_data = None
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = result

        tm.prepare_infer_mode()

        assert tm.rearrange_transfers is True
        mock_has_gaps.assert_not_called()


class TestInitializeProfileSavedStateShortCircuit:
    """``profile_mode='view'`` is the new default. The saved-profile workflow
    short-circuits ``initialize_profile`` (state already populated), so the
    view controller is *never* built. Pin that the short-circuit returns the
    untouched model and leaves no controller behind -- otherwise a future
    refactor that moves view-mode setup above the short-circuit would
    silently allocate (and leak) the rotating block.
    """

    def test_view_mode_with_saved_state_skips_controller_setup(self) -> None:
        tm = _make_tm(profile_mode="view")
        tm.model = MagicMock(name="user_model")
        # The short-circuit keys off the *externally restored* marker, not on
        # ``tensor_manager_state`` alone — a live cycle reaching INFERENCE also
        # populates that attribute and must not short-circuit.
        tm.tensor_manager_state = MagicMock(name="loaded_state")
        tm._state_restored_from_profile = True
        tm._move_non_offloaded_tensors_to_gpu = MagicMock()
        tm.prepare_profile_direct_mode_model = MagicMock()
        tm.prepare_profile_direct_mode = MagicMock()
        tm.prepare_profile_mode = MagicMock()

        result = tm.initialize_profile()

        assert result is tm.model
        assert not isinstance(tm.tensor_layer_loader, ProfileBlockController)
        tm._move_non_offloaded_tensors_to_gpu.assert_not_called()
        tm.prepare_profile_direct_mode_model.assert_not_called()
        tm.prepare_profile_direct_mode.assert_not_called()
        tm.prepare_profile_mode.assert_not_called()


class TestPrepareWarmupModeResetsLayerStats:
    """``prepare_warmup_mode`` must clear ``_layer_stats_computed`` so a fresh
    discovery cycle (warmup -> profile -> ... -> warmup -> profile) recomputes
    layer stats instead of silently reusing the previous cycle's. Drop the
    reset and cycle 2's profile picks up cycle 1's stale stats.
    """

    def test_layer_stats_flag_is_reset_on_each_warmup_entry(self) -> None:
        tm = _make_tm()
        tm.model = None  # avoid module-tracker registration in this minimal harness

        tm.prepare_warmup_mode()
        assert tm._layer_stats_computed is False

        tm._layer_stats_computed = True
        tm.prepare_warmup_mode()
        assert tm._layer_stats_computed is False

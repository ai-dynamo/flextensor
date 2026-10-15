# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: synthetic DiT + compiled offload + per-block compile backends.

Uses the lightweight :class:`~tests.integration._synthetic_dit.SyntheticDiT` model
(from the diffusers ``synthetic_offload_check`` harness) to validate:

1. **Correctness** — offload (eager and compiled) matches a no-offload GPU baseline.
2. **Memory** — under a capped GPU budget, FlexTensor keeps less weight memory
   resident than the full baseline model.
"""

import copy
import gc
import uuid

import pytest
import torch

from flextensor import OffloadConfig, StrategyReplanStatus, get_offload_manager, offload_from_profile
from flextensor.compile.lifecycle import COMPILED_WARMUP_FORWARDS, PROFILE_COMPILE_WARMUP_FORWARDS
from flextensor.compile.warmup_tail import CompiledOffloadTailState
from flextensor.custom_ops import get_active_loader
from flextensor.offload_manager import OffloadPhase
from tests.integration._compile_helpers import compile_transformer_blocks, set_seed
from tests.integration._synthetic_dit import make_synthetic_dit, make_synthetic_input

try:
    import torch_tensorrt  # noqa: F401

    HAS_TORCH_TRT = True
except ImportError:
    HAS_TORCH_TRT = False

pytestmark = pytest.mark.gpu_vram_24g

# Memory test: large enough to stream when GPU budget is capped below weight size.
MEMORY_LAYERS = 8
MEMORY_DIM = 512
# Compile correctness: smaller model (stable numerics under Inductor per-block).
COMPILE_LAYERS = 4
COMPILE_DIM = 256
HEADS = 8
SEQ = 64
BATCH = 1
DISCOVERY_ITERS = 1
PROFILING_ITERS = 3
INFERENCE_ITERS = 2
WARMUP_ITERS = 2
TIMING_ITERS = 3
NUM_BLOCKS = 2
SEED = 0

# Cap GPU budget to this fraction of full model weight bytes (forces streaming).
WEIGHT_BUDGET_RATIO = 0.75
# Offload weight footprint must stay below this fraction of the baseline weight size.
MAX_OFFLOAD_WEIGHT_FRACTION = 0.85

RTOL = 2e-2
ATOL = 2e-2

COMPILE_BACKENDS = [
    pytest.param("inductor", id="inductor"),
    pytest.param(
        "torch_tensorrt",
        marks=pytest.mark.skipif(not HAS_TORCH_TRT, reason="torch_tensorrt not installed"),
        id="torch_tensorrt",
    ),
]


def _model_weight_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _memory_pressure_fraction(model: torch.nn.Module, *, budget_ratio: float) -> float:
    """``OffloadConfig.max_gpu_mem_fraction`` that caps budget to ``budget_ratio * weight size``."""
    weight_bytes = _model_weight_bytes(model)
    total = torch.cuda.get_device_properties(0).total_memory
    return (weight_bytes * budget_ratio) / total


def _peak_allocated_bytes(forward_fn, *, warmup: int = WARMUP_ITERS, iters: int = TIMING_ITERS) -> int:
    """Peak CUDA bytes during ``iters`` timed forwards (after ``warmup`` untimed runs)."""
    with torch.no_grad():
        for _ in range(warmup):
            forward_fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(iters):
            forward_fn()
        torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def _synthetic_offload_config(
    model: torch.nn.Module,
    *,
    memory_pressure: bool = False,
) -> OffloadConfig:
    kwargs: dict = {
        "discovery_iters": DISCOVERY_ITERS,
        "profiling_iters": PROFILING_ITERS,
        "num_blocks": NUM_BLOCKS,
        "min_blocks": NUM_BLOCKS,
        "include_patterns": ["transformer_blocks.*"],
        "external_compile": True,
    }
    if memory_pressure:
        kwargs["max_gpu_mem_fraction"] = _memory_pressure_fraction(model, budget_ratio=WEIGHT_BUDGET_RATIO)
    return OffloadConfig(**kwargs)


def _synthetic_compile_fn_config(
    model: torch.nn.Module,
    *,
    memory_pressure: bool = False,
    profile_mode: str = "view",
) -> OffloadConfig:
    # ``skip_discovery=False`` here so the compile-path arithmetic
    # ``iters_before_inference == discovery_iters + PROFILE_COMPILE_WARMUP_FORWARDS
    # + profiling_iters`` this suite pins holds. With ``skip_discovery=True``
    # the discovery component drops to zero — that equivalence is covered
    # by unit tests in ``tests/unit/test_offload_manager_phase.py``.
    kwargs: dict = {
        "discovery_iters": DISCOVERY_ITERS,
        "profiling_iters": PROFILING_ITERS,
        "num_blocks": NUM_BLOCKS,
        "min_blocks": NUM_BLOCKS,
        "include_patterns": ["transformer_blocks.*"],
        "profile_mode": profile_mode,
        "skip_discovery": False,
    }
    if memory_pressure:
        kwargs["max_gpu_mem_fraction"] = _memory_pressure_fraction(model, budget_ratio=WEIGHT_BUDGET_RATIO)
    return OffloadConfig(**kwargs)


def _drive_to_inference(proxy: torch.nn.Module, x: torch.Tensor, om) -> None:
    with torch.no_grad():
        for _ in range(om.iters_before_inference):
            proxy(x)


def _recording_loader_hooks(om) -> tuple[list[str], list[str]]:
    """Wrap the active rolling loader so enter/exit calls are recorded."""
    loader = get_active_loader(om.compiled_offload_manager_id)
    assert loader is not None
    entered: list[str] = []
    exited: list[str] = []
    orig_enter = loader.enter
    orig_exit = loader.exit

    def enter(label: str) -> None:
        entered.append(label)
        orig_enter(label)

    def record_exit(label: str) -> None:
        exited.append(label)
        orig_exit(label)

    loader.enter = enter  # type: ignore[method-assign]
    loader.exit = record_exit  # type: ignore[method-assign]
    return entered, exited


def _assert_matches_baseline(
    output: torch.Tensor,
    baseline: torch.Tensor,
    *,
    msg: str,
) -> None:
    torch.testing.assert_close(output.float(), baseline.float(), rtol=RTOL, atol=ATOL, msg=msg)


def _baseline_output(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    ref_model = copy.deepcopy(model).to(device).eval()
    with torch.no_grad():
        return ref_model(x).detach().clone()


@pytest.fixture()
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestSyntheticCompiledOffload:
    def test_baseline_offload_memory_and_correctness(self, device: torch.device) -> None:
        """Offload eager output matches baseline; weight footprint is below full model size."""
        set_seed(SEED)
        dtype = torch.bfloat16

        model = make_synthetic_dit(
            layers=MEMORY_LAYERS,
            dim=MEMORY_DIM,
            heads=HEADS,
            dtype=dtype,
            device="cpu",
            seed=SEED,
        )
        x = make_synthetic_input(batch=BATCH, seq=SEQ, dim=MEMORY_DIM, dtype=dtype, device=device, seed=SEED + 1)
        baseline_weight_mb = _model_weight_bytes(model) / (1024 * 1024)

        ref_model = copy.deepcopy(model).to(device).eval()

        def baseline_forward() -> None:
            with torch.no_grad():
                ref_model(x)

        with torch.no_grad():
            baseline_out = ref_model(x).detach().clone()
        baseline_peak_mb = _peak_allocated_bytes(baseline_forward) / (1024 * 1024)

        ref_model = None
        gc.collect()
        torch.cuda.empty_cache()

        manager_name = f"test_synth_mem_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _synthetic_offload_config(model, memory_pressure=True)

        try:
            proxy = om.offload(model, config=config)
            _drive_to_inference(proxy, x, om)

            assert om._current_phase == OffloadPhase.INFERENCE  # noqa: SLF001
            assert get_active_loader(om.compiled_offload_manager_id) is not None

            with torch.no_grad():
                offload_out = proxy(x).detach().clone()
            _assert_matches_baseline(
                offload_out,
                baseline_out,
                msg="offload eager output diverges from no-offload baseline",
            )

            ft_usage = om.get_gpu_memory_usage()
            offload_peak_mb = _peak_allocated_bytes(lambda: proxy(x)) / (1024 * 1024)

            assert ft_usage.total_mb < baseline_weight_mb * MAX_OFFLOAD_WEIGHT_FRACTION, (
                f"FlexTensor weight footprint {ft_usage.total_mb:.1f}MB should stay below "
                f"{MAX_OFFLOAD_WEIGHT_FRACTION:.0%} of baseline weight size "
                f"({baseline_weight_mb:.1f}MB); blocks={ft_usage.blocks_mb:.1f}MB "
                f"unmapped={ft_usage.unmapped_tensors_mb:.1f}MB"
            )
            assert offload_peak_mb < baseline_peak_mb, (
                f"offload inference peak {offload_peak_mb:.1f}MB should be below "
                f"no-offload baseline peak {baseline_peak_mb:.1f}MB"
            )
        finally:
            om.release()

    @pytest.mark.parametrize("compile_backend", COMPILE_BACKENDS)
    def test_offload_compile_matches_baseline(self, device: torch.device, compile_backend: str) -> None:
        """Per-block compile (Inductor / TRT) matches baseline after offload reaches INFERENCE."""
        set_seed(SEED)
        dtype = torch.float32 if compile_backend == "torch_tensorrt" else torch.bfloat16

        model = make_synthetic_dit(
            layers=COMPILE_LAYERS,
            dim=COMPILE_DIM,
            heads=HEADS,
            dtype=dtype,
            device="cpu",
            seed=SEED,
        )
        x = make_synthetic_input(batch=BATCH, seq=SEQ, dim=COMPILE_DIM, dtype=dtype, device=device, seed=SEED + 1)
        baseline_out = _baseline_output(model, x, device)

        manager_name = f"test_synth_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _synthetic_offload_config(model, memory_pressure=False)

        try:
            proxy = om.offload(model, config=config)
            _drive_to_inference(proxy, x, om)

            with torch.no_grad():
                eager_offload = proxy(x).detach().clone()
            _assert_matches_baseline(
                eager_offload,
                baseline_out,
                msg="offload eager diverges from baseline (pre-compile)",
            )

            compile_transformer_blocks(
                proxy,
                backend=compile_backend,
                mode="default",
                trt_enabled_precisions={dtype} if compile_backend == "torch_tensorrt" else None,
            )

            torch._dynamo.reset()
            with torch.no_grad():
                for _ in range(INFERENCE_ITERS):
                    proxy(x)
                compiled_out = proxy(x).detach().clone()

            _assert_matches_baseline(
                compiled_out,
                baseline_out,
                msg=f"offload+{compile_backend} diverges from baseline",
            )
            _assert_matches_baseline(
                compiled_out,
                eager_offload,
                msg=f"offload+{compile_backend} diverges from offload eager",
            )
        finally:
            om.release()
            torch._dynamo.reset()

    @pytest.mark.parametrize("compile_backend", COMPILE_BACKENDS)
    def test_offload_compile_fullgraph_matches_baseline(self, device: torch.device, compile_backend: str) -> None:
        """Per-block ``fullgraph=True`` compile matches baseline (slot-alias-safe mode)."""
        set_seed(SEED)
        dtype = torch.float32 if compile_backend == "torch_tensorrt" else torch.bfloat16

        model = make_synthetic_dit(
            layers=COMPILE_LAYERS,
            dim=COMPILE_DIM,
            heads=HEADS,
            dtype=dtype,
            device="cpu",
            seed=SEED,
        )
        x = make_synthetic_input(batch=BATCH, seq=SEQ, dim=COMPILE_DIM, dtype=dtype, device=device, seed=SEED + 1)
        baseline_out = _baseline_output(model, x, device)

        manager_name = f"test_synth_fg_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _synthetic_offload_config(model, memory_pressure=False)

        try:
            proxy = om.offload(model, config=config)
            _drive_to_inference(proxy, x, om)

            compile_transformer_blocks(
                proxy,
                backend=compile_backend,
                mode="default",
                fullgraph=True,
                trt_enabled_precisions={dtype} if compile_backend == "torch_tensorrt" else None,
            )

            torch._dynamo.reset()
            with torch.no_grad():
                for _ in range(INFERENCE_ITERS):
                    proxy(x)
                compiled_out = proxy(x).detach().clone()

            _assert_matches_baseline(
                compiled_out,
                baseline_out,
                msg=f"offload+{compile_backend}+fullgraph diverges from baseline",
            )
        finally:
            om.release()
            torch._dynamo.reset()

    def test_compile_fn_fullgraph_matches_baseline(self, device: torch.device) -> None:
        """``compile_fn`` + getter profile: post-compile replan rebuilds, then matches baseline."""
        set_seed(SEED)
        dtype = torch.bfloat16

        model = make_synthetic_dit(
            layers=COMPILE_LAYERS,
            dim=COMPILE_DIM,
            heads=HEADS,
            dtype=dtype,
            device="cpu",
            seed=SEED,
        )
        x = make_synthetic_input(batch=BATCH, seq=SEQ, dim=COMPILE_DIM, dtype=dtype, device=device, seed=SEED + 1)
        baseline_out = _baseline_output(model, x, device)

        def compile_fn(module: torch.nn.Module) -> torch.nn.Module:
            return torch.compile(module, fullgraph=True, backend="inductor")

        manager_name = f"test_synth_cfn_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        # Non-view profile → replan_active; source weights retained for rebuild.
        config = _synthetic_compile_fn_config(model, memory_pressure=False, profile_mode="getter")

        try:
            proxy = om.offload(model, config=config, compile_fn=compile_fn)
            assert om.compiled_replan_active is True
            _drive_to_inference(proxy, x, om)
            assert om._current_phase == OffloadPhase.INFERENCE  # noqa: SLF001

            tm = om._tensor_manager
            assert tm is not None
            assert tm._replan_source_data, (  # noqa: SLF001
                "Supported replan path must retain source weights before the first loader build"
            )
            loader_before = tm.tensor_layer_loader
            assert loader_before is not None

            replan_iters = om.request_strategy_replan()
            assert replan_iters > 0
            with torch.no_grad():
                for _ in range(replan_iters):
                    proxy(x)
            assert om._compiled.tail_state == CompiledOffloadTailState.DONE  # noqa: SLF001
            assert tm.tensor_layer_loader is not loader_before, "replan must rebuild the inference loader"
            assert not tm._replan_source_data, "successful rebuild clears the source-weight snapshot"  # noqa: SLF001
            assert get_active_loader(om.compiled_offload_manager_id) is tm.tensor_layer_loader

            torch._dynamo.reset()
            with torch.no_grad():
                compiled_out = proxy(x).detach().clone()

            _assert_matches_baseline(
                compiled_out,
                baseline_out,
                msg="compile_fn+getter+fullgraph diverges from baseline after replan rebuild",
            )
        finally:
            om.release()
            torch._dynamo.reset()

    def test_compile_fn_compiled_profile_no_replan_matches_baseline(self, device: torch.device) -> None:
        """``compile_fn`` + view profiles compiled (no replan tail)."""
        set_seed(SEED)
        dtype = torch.bfloat16

        model = make_synthetic_dit(
            layers=COMPILE_LAYERS,
            dim=COMPILE_DIM,
            heads=HEADS,
            dtype=dtype,
            device="cpu",
            seed=SEED,
        )
        x = make_synthetic_input(batch=BATCH, seq=SEQ, dim=COMPILE_DIM, dtype=dtype, device=device, seed=SEED + 1)
        baseline_out = _baseline_output(model, x, device)

        def compile_fn(module: torch.nn.Module) -> torch.nn.Module:
            return torch.compile(module, fullgraph=True, backend="inductor")

        manager_name = f"test_synth_cprofile_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _synthetic_compile_fn_config(model, memory_pressure=False)

        try:
            proxy = om.offload(model, config=config, compile_fn=compile_fn)
            assert om.compiled_profile_active is True
            assert om.compiled_replan_active is False
            assert om.iters_before_inference == (
                config.discovery_iters + PROFILE_COMPILE_WARMUP_FORWARDS + config.profiling_iters
            )
            _drive_to_inference(proxy, x, om)
            assert om._current_phase == OffloadPhase.INFERENCE  # noqa: SLF001
            assert om._compiled.tail_state == CompiledOffloadTailState.DONE  # noqa: SLF001
            # compile_fn is FlexTensor-driven and view-profile needs no replan.
            assert not om.external_compile_active
            assert not om.strategy_replan_status.can_request
            assert om.request_strategy_replan() == 0
            assert om._compiled.tail_state == CompiledOffloadTailState.DONE  # noqa: SLF001
            assert om.strategy_replan_status == StrategyReplanStatus(state="idle", can_request=False, applied=None)

            torch._dynamo.reset()
            with torch.no_grad():
                compiled_out = proxy(x).detach().clone()

            _assert_matches_baseline(
                compiled_out,
                baseline_out,
                msg="compile_fn compiled-profile path diverges from baseline",
            )
        finally:
            om.release()
            torch._dynamo.reset()

    def test_compiled_profile_offload_from_profile_skips_reprofile(self, device: torch.device, tmp_path) -> None:
        """Restored compiled-profile must land in INFERENCE without view-profile re-arm."""
        set_seed(SEED)
        dtype = torch.bfloat16

        model = make_synthetic_dit(
            layers=COMPILE_LAYERS,
            dim=COMPILE_DIM,
            heads=HEADS,
            dtype=dtype,
            device="cpu",
            seed=SEED,
        )
        x = make_synthetic_input(batch=BATCH, seq=SEQ, dim=COMPILE_DIM, dtype=dtype, device=device, seed=SEED + 1)

        def compile_fn(module: torch.nn.Module) -> torch.nn.Module:
            return torch.compile(module, fullgraph=True, backend="inductor")

        config = _synthetic_compile_fn_config(model, memory_pressure=False)
        profile_dir = tmp_path / "compiled_profile"

        manager_name_save = f"test_synth_cprofile_save_{uuid.uuid4().hex[:8]}"
        om_save = get_offload_manager(manager_name_save)
        try:
            proxy = om_save.offload(model, config=config, compile_fn=compile_fn)
            _drive_to_inference(proxy, x, om_save)
            assert om_save._current_phase == OffloadPhase.INFERENCE  # noqa: SLF001
            om_save.save_profile(str(profile_dir))
        finally:
            om_save.release()
            torch._dynamo.reset()

        model_restore = make_synthetic_dit(
            layers=COMPILE_LAYERS,
            dim=COMPILE_DIM,
            heads=HEADS,
            dtype=dtype,
            device="cpu",
            seed=SEED,
        )
        manager_name_restore = f"test_synth_cprofile_restore_{uuid.uuid4().hex[:8]}"
        om_restore = get_offload_manager(manager_name_restore)
        try:
            proxy = offload_from_profile(
                model_restore,
                str(profile_dir),
                config=config,
                name=manager_name_restore,
                compile_fn=compile_fn,
            )
            assert om_restore._current_phase == OffloadPhase.INFERENCE  # noqa: SLF001
            assert om_restore.compiled_profile_active is True
            torch._dynamo.reset()
            with torch.no_grad():
                proxy(x)
        finally:
            om_restore.release()
            torch._dynamo.reset()

    def test_loader_installed_and_forwards_traceable_at_inference(self, device: torch.device) -> None:
        """Smoke: INFERENCE leaves loader registered and forwards are not Dynamo-disabled."""
        model = make_synthetic_dit(
            layers=COMPILE_LAYERS,
            dim=COMPILE_DIM,
            heads=HEADS,
            dtype=torch.bfloat16,
            device="cpu",
        )
        x = make_synthetic_input(batch=BATCH, seq=SEQ, dim=COMPILE_DIM, dtype=torch.bfloat16, device=device)

        manager_name = f"test_synth_smoke_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _synthetic_offload_config(model, memory_pressure=False)

        try:
            proxy = om.offload(model, config=config)
            _drive_to_inference(proxy, x, om)

            assert get_active_loader(om.compiled_offload_manager_id) is not None
            for module in om._patched_modules:  # noqa: SLF001
                bound = module.forward
                func = getattr(bound, "__func__", bound)
                assert not getattr(func, "_torchdynamo_disable", False), (
                    f"{module} forward must stay traceable under compiled_offload"
                )
        finally:
            om.release()


class TestExternalCompiledOffloadReplan:
    def test_external_per_block_compile_replan_matches_baseline(self, device: torch.device) -> None:
        """External compile + ``request_strategy_replan()`` e2e."""
        set_seed(SEED)
        dtype = torch.bfloat16

        model = make_synthetic_dit(
            layers=COMPILE_LAYERS,
            dim=COMPILE_DIM,
            heads=HEADS,
            dtype=dtype,
            device="cpu",
            seed=SEED,
        )
        x = make_synthetic_input(batch=BATCH, seq=SEQ, dim=COMPILE_DIM, dtype=dtype, device=device, seed=SEED + 1)
        baseline_out = _baseline_output(model, x, device)

        manager_name = f"test_synth_ext_replan_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _synthetic_offload_config(model, memory_pressure=False)

        try:
            proxy = om.offload(model, config=config)
            _drive_to_inference(proxy, x, om)
            assert om._current_phase == OffloadPhase.INFERENCE  # noqa: SLF001
            assert get_active_loader(om.compiled_offload_manager_id) is not None

            compile_transformer_blocks(proxy, backend="inductor", fullgraph=True)

            tm = om._tensor_manager
            assert tm is not None
            assert tm._replan_source_data, (  # noqa: SLF001
                "external_compile must retain source weights before the first loader build"
            )
            loader_before = tm.tensor_layer_loader
            assert loader_before is not None

            assert om.external_compile_active
            preflight = om.strategy_replan_status
            assert preflight.state == "idle"
            assert preflight.can_request
            assert preflight.applied is None

            replan_iters = om.request_strategy_replan()
            assert replan_iters == COMPILED_WARMUP_FORWARDS + PROFILING_ITERS
            assert om.strategy_replan_status.state == "in_progress"

            torch._dynamo.reset()
            with torch.no_grad():
                for _ in range(replan_iters):
                    proxy(x)
            assert om._compiled.tail_state == CompiledOffloadTailState.DONE  # noqa: SLF001
            assert tm.tensor_layer_loader is not loader_before, "replan must rebuild the inference loader"
            assert not tm._replan_source_data, "successful rebuild clears the source-weight snapshot"  # noqa: SLF001
            applied_status = om.strategy_replan_status
            assert applied_status.state == "done"
            assert applied_status.applied is True
            # The rebuild consumed the snapshot, so a second replan is refused.
            assert not applied_status.can_request
            assert get_active_loader(om.compiled_offload_manager_id) is tm.tensor_layer_loader

            with torch.no_grad():
                for _ in range(INFERENCE_ITERS):
                    proxy(x)
                compiled_out = proxy(x).detach().clone()

            _assert_matches_baseline(
                compiled_out,
                baseline_out,
                msg="external per-block compile + replan diverges from baseline",
            )
        finally:
            om.release()
            torch._dynamo.reset()

    def test_pre_post_compute_ops_fire_under_cuda_inductor_compile(self, device: torch.device) -> None:
        """``pre_compute/post_compute`` dispatch runs inside a real CUDA Inductor compiled forward."""
        set_seed(SEED)
        dtype = torch.bfloat16

        model = make_synthetic_dit(
            layers=COMPILE_LAYERS,
            dim=COMPILE_DIM,
            heads=HEADS,
            dtype=dtype,
            device="cpu",
            seed=SEED,
        )
        x = make_synthetic_input(batch=BATCH, seq=SEQ, dim=COMPILE_DIM, dtype=dtype, device=device, seed=SEED + 1)
        baseline_out = _baseline_output(model, x, device)

        manager_name = f"test_synth_loader_cuda_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _synthetic_offload_config(model, memory_pressure=False)

        try:
            proxy = om.offload(model, config=config)
            _drive_to_inference(proxy, x, om)
            entered, exited = _recording_loader_hooks(om)

            compile_transformer_blocks(proxy, backend="inductor", fullgraph=True)

            torch._dynamo.reset()
            with torch.no_grad():
                for _ in range(INFERENCE_ITERS):
                    proxy(x)
                compiled_out = proxy(x).detach().clone()

            assert entered, "compiled forward never called loader.enter"
            assert exited, "compiled forward never called loader.exit"
            assert len(entered) == len(exited), (
                f"loader enter/exit mismatch: {len(entered)} enters vs {len(exited)} exits"
            )
            _assert_matches_baseline(
                compiled_out,
                baseline_out,
                msg="CUDA Inductor compile with loader dispatch diverges from baseline",
            )
        finally:
            om.release()
            torch._dynamo.reset()

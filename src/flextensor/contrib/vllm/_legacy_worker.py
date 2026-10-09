# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlexTensor-enabled vLLM Worker for offloading support.

Usage:
    FT_ENABLED=1 vllm serve model \\
        --worker-cls flextensor.contrib.vllm.worker.FlexTensorOffloadWorker

See flextensor.config.OffloadConfig for configuration options (FT_* env vars).
"""

import atexit
import importlib
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, cast

import psutil
from typing_extensions import deprecated
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker.gpu_worker import Worker

import flextensor
from flextensor.compile import COMPILED_EAGER_PROFILE_FORWARDS
from flextensor.config import load_config
from flextensor.contrib.vllm import _patterns
from flextensor.contrib.vllm._drafter_device import ensure_drafter_on_device
from flextensor.contrib.vllm._logging import safely_install_flextensor_logging_bridge
from flextensor.offload_manager import OffloadPhase
from flextensor.utils import config_field_was_set

# vLLM's V2 runner initializes attention metadata after FlexTensor's load-time profiling.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

safely_install_flextensor_logging_bridge()
# Register FlexTensor's vLLM load_format side effect.
importlib.import_module("flextensor.contrib.vllm.loader")

LOGGER = logging.getLogger("flextensor.contrib.vllm.worker")

VLLM_DEFAULT_INCLUDE_PATTERNS = _patterns.VLLM_DEFAULT_INCLUDE_PATTERNS
VLLM_DEFAULT_EXCLUDE_PATTERNS = _patterns.VLLM_DEFAULT_EXCLUDE_PATTERNS
resolve_vllm_patterns = _patterns.resolve_vllm_patterns

_VLLM_DISABLE_COMPILE_CACHE = "VLLM_DISABLE_COMPILE_CACHE"
_VLLM_USE_AOT_COMPILE = "VLLM_USE_AOT_COMPILE"
# FlexTensor applies this fraction to model weights only. It is not vLLM's
# gpu_memory_utilization, which also covers KV cache and runtime allocations.
_VLLM_DEFAULT_MAX_GPU_MEM_FRACTION = 0.9


def _configure_vllm_compile_env_for_compiled_offload() -> None:
    """Align vLLM compile env with FlexTensor's deferred compile path.

    Compilation is deferred until inference (see
    ``_ft_set_compiled_offload_compile_enabled``), so the on-disk compile cache
    must be disabled. Supported vLLM versions already default AOT off when the
    cache is disabled, so leave ``VLLM_USE_AOT_COMPILE`` unset unless the user
    set an explicit conflicting value — then warn and correct it. Likewise warn
    before overriding an explicit cache setting.
    """
    existing_cache = os.environ.get(_VLLM_DISABLE_COMPILE_CACHE)
    if existing_cache is not None and existing_cache != "1":
        LOGGER.warning(
            "FT-COMPILE-PATH: overriding explicit %s=%s -> 1 for compiled-offload "
            "(on-disk compile cache must stay disabled until inference)",
            _VLLM_DISABLE_COMPILE_CACHE,
            existing_cache,
        )
    os.environ[_VLLM_DISABLE_COMPILE_CACHE] = "1"

    existing_aot = os.environ.get(_VLLM_USE_AOT_COMPILE)
    if existing_aot == "1":
        LOGGER.warning(
            "FT-COMPILE-PATH: correcting conflicting %s=1 -> 0 for compiled-offload "
            "(AOT would run too early; disabling the compile cache already defaults "
            "AOT off)",
            _VLLM_USE_AOT_COMPILE,
        )
        os.environ[_VLLM_USE_AOT_COMPILE] = "0"

    LOGGER.warning("FT-COMPILE-PATH: compiled-offload running under vLLM native fullgraph=True")


# Compiled-offload: configure vLLM compile env at import time, before any
# ``@support_torch_compile`` model class is instantiated. Use the same
# ``load_config()`` path as runtime so env spellings and ``FT_CONFIG_FILE``
# agree with OffloadManager activation.
if load_config().external_compile:
    _configure_vllm_compile_env_for_compiled_offload()

VLLM_COMPILE_WARMUP_FORWARD_COUNT = 2
VLLM_DISCOVERY_ITER_FLOOR = VLLM_COMPILE_WARMUP_FORWARD_COUNT + 1
VLLM_PROFILING_ITER_FLOOR = 2


def _GiB(b: int) -> float:  # noqa: N802
    return b / float(GiB_bytes)


def _is_cudagraph_mode_none(mode: Any) -> bool:
    """Return True when ``cudagraph_mode`` means CUDA graphs are disabled."""
    if mode is None:
        return False
    if isinstance(mode, (int, float)) and int(mode) == 0:
        return True
    name = getattr(mode, "name", None)
    if name is not None:
        return str(name).upper() == "NONE"
    text = str(mode).upper()
    return text == "NONE" or text.endswith(".NONE")


def _cudagraphs_disabled(vllm_config: Any) -> bool:
    """Best-effort check that vLLM CUDA graphs are already off."""
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return False
    return _is_cudagraph_mode_none(getattr(compilation_config, "cudagraph_mode", None))


def _warn_vllm_runtime_requirements(offload_config: Any, vllm_config: Any) -> None:
    """Warn about unsupported vLLM runtime knobs for the active FlexTensor path.

    * Plain offload: needs ``--enforce-eager`` (CUDA graphs + compile off).
    * ``external_compile``: needs vLLM native compile (not eager), but CUDA
      graphs are not supported yet — ask for ``cudagraph_mode: NONE``.
    """
    if getattr(offload_config, "external_compile", False):
        if not _cudagraphs_disabled(vllm_config):
            LOGGER.warning(
                "FlexTensor compiled offload does not support CUDA graphs yet. "
                'Pass --compilation-config \'{"cudagraph_mode": "NONE"}\'.'
            )
        return
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is not None and not getattr(model_config, "enforce_eager", False):
        LOGGER.warning("FlexTensor offloading requires eager mode. Add --enforce-eager flag.")


_CUDA_GRAPH_WRAPPER_LOGGED = False


def _resolve_cuda_graph_wrapper() -> type[Any] | None:
    """Locate vLLM's ``CUDAGraphWrapper`` class.

    The primary import path logs at INFO the first time it succeeds.  The
    fallback import path logs at DEBUG so normal startup stays quiet while
    debug logs still show which wrapper was selected.  If both imports fail,
    return ``None`` and warn so callers can treat CUDA-graph wrapping as
    unavailable.
    """
    global _CUDA_GRAPH_WRAPPER_LOGGED
    primary_exc: ImportError | None = None
    try:
        from vllm.compilation.wrapper import CUDAGraphWrapper

        if not _CUDA_GRAPH_WRAPPER_LOGGED:
            LOGGER.info("FlexTensor: using CUDAGraphWrapper from vllm.compilation.wrapper (primary path)")
            _CUDA_GRAPH_WRAPPER_LOGGED = True
        return cast("type[Any]", CUDAGraphWrapper)
    except ImportError as exc:
        primary_exc = exc

    try:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        if not _CUDA_GRAPH_WRAPPER_LOGGED:
            LOGGER.debug(
                "FlexTensor: using CUDAGraphWrapper from vllm.compilation.cuda_graph (fallback path; "
                "primary import failed: %s)",
                primary_exc,
            )
            _CUDA_GRAPH_WRAPPER_LOGGED = True
        return cast("type[Any]", CUDAGraphWrapper)
    except ImportError as exc:
        LOGGER.warning(
            "FlexTensor: CUDAGraphWrapper not importable from either vllm.compilation.wrapper "
            "(%s) or vllm.compilation.cuda_graph (%s); CUDA-graph wrapping is unavailable in "
            "this vLLM version",
            primary_exc,
            exc,
        )
        return None


def _vllm_config_updates(offload_config: Any) -> dict[str, Any]:
    """Return vLLM-specific OffloadConfig updates.

    Args:
        offload_config: OffloadConfig-like object with discovery/profiling
            iteration counts and include/exclude pattern lists.

    Returns:
        Config update dictionary with minimum vLLM warmup iteration counts,
        a 0.9 model-weight GPU memory fraction when the user omitted that setting,
        ``skip_discovery=False`` (vLLM manages its own warmup schedule and
        relies on the DISCOVERY→PROFILING transition landing between
        ``compile_or_warm_up_model()`` and the explicit max-token loop —
        see :meth:`FlexTensorOffloadWorker.warmup_and_profile_model`),
        default non-wildcard include patterns when the user did not customize
        includes, and MoE sidecar excludes when the user did not customize
        excludes.
    """
    # vLLM's first ``compile_or_warm_up_model()`` runs two forwards that count
    # as FlexTensor discovery iterations. Keep one additional explicit small-token
    # discovery pass, while using max-token profiling passes to bound startup
    # cost. On the compiled-offload + re-plan path the eager seed is a fixed
    # ``COMPILED_EAGER_PROFILE_FORWARDS`` count (independent of the measure
    # window sized by ``profiling_iters``), so floor profiling_iters there too.
    #
    # ``skip_discovery=False`` is required here regardless of the OffloadConfig
    # default: with ``skip_discovery=True`` the manager short-circuits to
    # PROFILING inside ``offload()``, so the two vLLM warmup forwards land in
    # the profile budget instead of DISCOVERY. At ``profiling_iters=2`` (the
    # vLLM floor) the state machine reaches INFERENCE before the explicit
    # max-token loop even starts, so zero max-token forwards are profiled.
    profiling_floor = VLLM_PROFILING_ITER_FLOOR
    # ``offload_config`` already comes from ``load_config()`` (env + optional file).
    if getattr(offload_config, "external_compile", False):
        profiling_floor = max(profiling_floor, COMPILED_EAGER_PROFILE_FORWARDS)
    config_updates: dict[str, Any] = {
        "discovery_iters": max(offload_config.discovery_iters, VLLM_DISCOVERY_ITER_FLOOR),
        "profiling_iters": max(offload_config.profiling_iters, profiling_floor),
        "skip_discovery": False,
    }
    if not config_field_was_set(offload_config, "max_gpu_mem_fraction"):
        config_updates["max_gpu_mem_fraction"] = _VLLM_DEFAULT_MAX_GPU_MEM_FRACTION
    include_patterns, exclude_patterns = resolve_vllm_patterns(offload_config)
    if include_patterns != offload_config.include_patterns:
        config_updates["include_patterns"] = include_patterns
    if exclude_patterns != offload_config.exclude_patterns:
        config_updates["exclude_patterns"] = exclude_patterns
    return config_updates


# Return type is `Any` for beartype + @contextmanager compatibility.
@contextmanager
def vllm_model_context(model_runner: Any) -> Any:
    """Context manager for unwrapping/re-wrapping vLLM model wrappers.

    Extracts actual model from UBatchWrapper/CUDAGraphWrapper on entry.
    Updates wrapper with modified model on exit.

    Args:
        model_runner: The model runner containing the model (wrapped or unwrapped).

    Yields:
        A single-element list containing the unwrapped model.
        Modify model_container[0] to update the wrapper on exit.

    Example:
        >>> with vllm_model_context(self.model_runner) as model_container:
        ...     model_container[0] = flextensor.offload(model_container[0], config)
    """
    model_or_wrapper = model_runner.model
    actual_model = model_or_wrapper
    wrapper_attrs: list[str] = []  # Tracks which wrapper attributes to update

    try:
        from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

        if isinstance(model_or_wrapper, UBatchWrapper):
            actual_model = model_or_wrapper.runnable
            wrapper_attrs.append("runnable")
            # Also track 'model' attribute if it points to the same object
            # vLLM's UBatchWrapper may use both attributes
            if hasattr(model_or_wrapper, "model") and model_or_wrapper.model is model_or_wrapper.runnable:
                wrapper_attrs.append("model")
    except ImportError:
        pass

    CUDAGraphWrapper = _resolve_cuda_graph_wrapper()  # noqa: N806 — class returned from helper, used as isinstance target

    if CUDAGraphWrapper is not None and isinstance(model_or_wrapper, CUDAGraphWrapper):
        actual_model = model_or_wrapper.model
        wrapper_attrs.append("model")

    model_container = [actual_model]
    try:
        yield model_container
    finally:
        if wrapper_attrs:
            for attr in wrapper_attrs:
                setattr(model_or_wrapper, attr, model_container[0])
        else:
            model_runner.model = model_container[0]


@deprecated(
    "The legacy/v1 FlexTensor vLLM worker is deprecated. "
    "Use worker v2 instead by unsetting FT_VLLM_USE_V2_WORKER or setting it to '1'. "
    "Will be removed in v0.6.0."
)
class FlexTensorOffloadWorker(Worker):
    """vLLM Worker with FlexTensor offloading support.

    Applies tensor offloading after model loading when FT_ENABLED=1.

    .. deprecated:: v0.5.0
        Use worker v2 by unsetting ``FT_VLLM_USE_V2_WORKER`` or setting it to
        ``1``. The legacy/v1 worker will be removed in v0.6.0.
    """

    _ft_compile_gated_modules: list[Any] | None

    def load_model(self) -> None:
        """Load model with optional FlexTensor offloading."""
        # Create offload config using the device set by init_device()
        offload_config = load_config(gpu_device=self.device.index)
        if offload_config.shm_enabled:
            raise RuntimeError(
                "The legacy vLLM worker does not support OffloadConfig.shm_enabled=True; "
                "use worker v2 (FT_VLLM_USE_V2_WORKER=1)."
            )
        config_updates = _vllm_config_updates(offload_config)
        self._offload_config = offload_config.model_copy(update=config_updates)
        if self._offload_config.enabled:
            LOGGER.info("FlexTensor offloading enabled with config: %s", self._offload_config)

        if self._offload_config.enabled:
            old_device = self.vllm_config.load_config.device or self.vllm_config.device_config.device
            old_load_format = self.vllm_config.load_config.load_format

            # Use FlexTensor's custom loader: loads on CPU, processes CUDA-only ops
            # (FP8 quant, MLA attention) on GPU layer-by-layer, returns model on CPU
            self.vllm_config.load_config.load_format = "flextensor"
            self.vllm_config.load_config.device = "cpu"

            LOGGER.info(
                "FlexTensor: Loading model with CPU-first strategy (was: %s); load_format: %s -> flextensor",
                old_device,
                old_load_format,
            )

        super().load_model()

        if not self._offload_config.enabled:
            return

        _warn_vllm_runtime_requirements(self._offload_config, self.vllm_config)

        # Speculative-decoding drafter (e.g. MTP / Eagle) lives outside the
        # main model and is not reached by flextensor.offload(). FT's
        # CPU-first loader leaves its weights on CPU; warmup then crashes
        # when vLLM's @torch.compile layernorm helper sees CPU weights
        # against CUDA inputs (flex-tensor #140). Push the drafter to GPU
        # BEFORE flextensor.offload() walks the main model — some drafter
        # submodules (e.g. embed_tokens) can be identity-shared with the
        # main model. If offload runs first, FT installs forward patches
        # against the still-CPU tensor IDs; the later .to(cuda) swaps
        # those tensors out, and the first drafter forward then hits a
        # KeyError in FT's cpu_to_gpu_map on trap exit.
        ensure_drafter_on_device(self.model_runner, self.device)

        atexit.register(flextensor.release)
        # Extract, offload, and update model using context manager
        with vllm_model_context(self.model_runner) as model_container:
            model_container[0] = flextensor.offload(model_container[0], self._offload_config)

        self.warmup_and_profile_model()
        self.model_runner.model_memory_usage = flextensor.get_gpu_memory_usage().total_bytes
        LOGGER.info(
            "FlexTensor offloading applied (GPU usage: %.2f GiB)",
            _GiB(self.model_runner.model_memory_usage),
        )

    def _ft_warn_host_memory_before_inference(self) -> None:
        """Log a low-memory warning before the inference transition when probeable."""
        try:
            vm = psutil.virtual_memory()
            free_gib = vm.available / GiB_bytes
            if free_gib < 2.0:
                LOGGER.warning(
                    "FlexTensor: Low host memory (%.1f GiB free) — inference transition may OOM",
                    free_gib,
                )
        except (OSError, AttributeError, psutil.Error) as exc:
            # First probe failure goes out at WARNING so an operator on a
            # locked-down container learns FlexTensor cannot pre-check OOM
            # risk; subsequent failures are demoted to DEBUG to avoid noise.
            if not getattr(self, "_ft_host_mem_probe_warned", False):
                LOGGER.warning(
                    "FlexTensor: host-memory pre-check failed (%s) — unable to warn about OOM risk "
                    "before inference transition; further probe failures will be logged at DEBUG",
                    exc,
                )
                self._ft_host_mem_probe_warned = True
            else:
                LOGGER.debug("FlexTensor: host-memory pre-check failed again (%s)", exc)

    def _ft_run_compiled_offload_inference_tail(self, run_forwards: Callable[[int], None]) -> None:
        """Run post-INFERENCE compiled-offload replan or compile warmup forwards."""
        om = flextensor.get_offload_manager()
        if om.compiled_offload_active and om.compiled_replan_active:
            replan_iters = om.request_strategy_replan()
            if replan_iters:
                run_forwards(replan_iters)
        elif om.compiled_offload_active:
            run_forwards(VLLM_COMPILE_WARMUP_FORWARD_COUNT)

    def warmup_and_profile_model(self) -> None:
        """Run discovery and profiling iterations for FlexTensor offloading.

        Executes discovery iterations to map parameters to traps, then runs
        profiling iterations at max batch size to collect layer statistics
        for the offloading strategy. Finally switches to inference mode.
        """
        max_num_tokens = min(self.model_runner.max_model_len, self.vllm_config.scheduler_config.max_num_batched_tokens)

        def _run_forwards(n: int) -> None:
            for _ in range(n):
                self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001

        # vLLM-only: defer @support_torch_compile until after the INFERENCE transition
        # (compile-transparent forwards + rolling loader are installed by OffloadManager).
        self._ft_set_compiled_offload_compile_enabled(False)
        self.compile_or_warm_up_model()
        # vLLM's first compile_or_warm_up_model() runs 2 forward passes through
        # the offloaded model; each one counts as a discovery iteration in the
        # OffloadManager, so the loop below only tops up to discovery_iters.
        compile_warm_iters = VLLM_COMPILE_WARMUP_FORWARD_COUNT

        discovery_iters = self._offload_config.discovery_iters - compile_warm_iters
        for i in range(discovery_iters):
            LOGGER.info("FlexTensor: Discovery iteration %d/%d (num_tokens=1)", i + 1, discovery_iters)
            self.model_runner._dummy_run(1, skip_eplb=True)  # noqa: SLF001

        # Pause profiling around vLLM's mixed-batch warmup: durations are suppressed
        # and the PROFILING counter is frozen, so only the explicit iterations below
        # feed the profile. The context manager also guarantees resume on exception.
        with flextensor.pause_profiling():
            self.compile_or_warm_up_model()
        om = flextensor.get_offload_manager()
        eager_profiling_iters = om.eager_profiling_iters
        for i in range(eager_profiling_iters):
            LOGGER.info(
                "FlexTensor: Profiling iteration %d/%d (max_num_tokens=%d)",
                i + 1,
                eager_profiling_iters,
                max_num_tokens,
            )
            self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001

        LOGGER.info("FlexTensor: Switching to inference mode")
        self._ft_warn_host_memory_before_inference()
        if om.phase != OffloadPhase.INFERENCE:
            raise RuntimeError(
                "FlexTensor compiled-offload: warmup ended before PROFILING→INFERENCE "
                f"(phase={om.phase.value}). Run {eager_profiling_iters} explicit profiling "
                "forwards for the eager seed when the compiled replan path is active."
            )
        # Last profiling forward triggered INFERENCE: compile-transparent forwards
        # and the rolling loader are installed by OffloadManager.
        self._ft_set_compiled_offload_compile_enabled(True)
        self._ft_run_compiled_offload_inference_tail(_run_forwards)

    def _ft_set_compiled_offload_compile_enabled(self, enabled: bool) -> None:
        """Flip ``do_not_compile`` on vLLM's compile wrappers.

        vLLM compiles once and keeps that graph. Turn compile off for discovery/profiling,
        then back on at INFERENCE once offload forwards and the rolling loader are ready.
        """
        om = flextensor.get_offload_manager()
        if not om.compiled_offload_active:
            return
        root = getattr(om, "model", None)
        if root is None:
            return
        gated: list[Any] | None
        if not enabled:
            existing = getattr(self, "_ft_compile_gated_modules", None)
            if existing is not None:
                LOGGER.warning(
                    "FlexTensor compiled-offload: compile gate already suppressed on %d module(s); "
                    "ignoring duplicate disable (likely a retry after partial warmup)",
                    len(existing),
                )
                return
            # Disable: remember the modules we silence so we only re-enable those
            # (modules already do_not_compile, e.g. NONE/ignored, stay untouched).
            gated = [module for module in root.modules() if getattr(module, "do_not_compile", None) is False]
            for module in gated:
                module.do_not_compile = True
            self._ft_compile_gated_modules = gated
            LOGGER.info(
                "FlexTensor compiled-offload: suppressed torch.compile on %d module(s) "
                "for discovery/profiling; first compile deferred to inference",
                len(gated),
            )
            return
        gated = getattr(self, "_ft_compile_gated_modules", None)
        if gated is None:
            LOGGER.warning(
                "FlexTensor compiled-offload: enable called without a prior disable gate; "
                "leaving module compile flags unchanged",
            )
            return
        for module in gated:
            module.do_not_compile = False
            # Defensive: the wrapper latches this after its first compile; ensure a
            # fresh first-compile path at inference even if a stray forward slipped
            # through with compilation enabled.
            if getattr(module, "compiled", False):
                module.compiled = False
        self._ft_compile_gated_modules = None
        LOGGER.info(
            "FlexTensor compiled-offload: re-enabled torch.compile on %d module(s); "
            "inference warmup will trigger the compile-transparent graph compile",
            len(gated),
        )

    def shutdown(self) -> None:
        super().shutdown()
        flextensor.release()

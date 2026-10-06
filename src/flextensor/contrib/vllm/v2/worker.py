# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrate FlexTensor bootstrap and runtime takeover with vLLM integration v2."""

import atexit
import importlib.metadata
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from packaging.version import Version
from torch import nn
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.worker.gpu_worker import Worker

import flextensor
import flextensor.contrib.vllm.v2.inference_profile as inference_profile
from flextensor._version import __version__
from flextensor.config import BLOCK_TRANSFER_MODES, OffloadConfig, load_config
from flextensor.contrib.vllm._patterns import (
    resolve_vllm_patterns,
)
from flextensor.contrib.vllm.v2.breakable_cudagraph import install_breakable_segment_join
from flextensor.contrib.vllm.v2.errors import VllmFlexTensorV2Error
from flextensor.contrib.vllm.v2.offloader import VllmBootstrapOffloader
from flextensor.offload_timing import OFFLOAD_TIMING_MEASURE_MAX_PASSES
from flextensor.utils import config_field_was_set

LOGGER = init_logger("vllm.flextensor.v2.worker")
MIN_VLLM_VERSION = "0.17.0"
_MODEL_FREE_SPECULATIVE_METHODS = frozenset({"custom_class", "ngram", "ngram_gpu", "suffix"})


@runtime_checkable
class _ModelRunner(Protocol):
    model: object


def _vllm_version() -> str:
    return importlib.metadata.version("vllm")


def _validate_speculative_loading(vllm_config: VllmConfig) -> None:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return
    method = getattr(speculative_config, "method", None)
    if method not in _MODEL_FREE_SPECULATIVE_METHODS:
        raise VllmFlexTensorV2Error(
            "does not support model-backed speculative loading in the current "
            f"single-root implementation: method={method!r}"
        )


def _validate_enabled_worker(vllm_config: VllmConfig, offload_config: OffloadConfig) -> None:
    if Version(_vllm_version()) < Version(MIN_VLLM_VERSION):
        raise VllmFlexTensorV2Error(f"requires vLLM >= {MIN_VLLM_VERSION} (got {_vllm_version()})")
    parallel_config = vllm_config.parallel_config
    if getattr(parallel_config, "enable_elastic_ep", False):
        raise VllmFlexTensorV2Error("does not support enable_elastic_ep")
    if getattr(parallel_config, "use_ubatching", False):
        raise VllmFlexTensorV2Error("does not support use_ubatching")
    if getattr(vllm_config, "weight_transfer_config", None) is not None:
        raise VllmFlexTensorV2Error("does not support weight_transfer_config")
    _validate_speculative_loading(vllm_config)
    mode = vllm_config.compilation_config.mode
    if mode == CompilationMode.NONE:
        if offload_config.external_compile:
            raise VllmFlexTensorV2Error("OffloadConfig.external_compile=True requires CompilationMode.VLLM_COMPILE")
        return
    if mode == CompilationMode.STOCK_TORCH_COMPILE:
        raise VllmFlexTensorV2Error(
            "CompilationMode.STOCK_TORCH_COMPILE is not supported due to compilation happening before state takeover"
        )
    if mode != CompilationMode.VLLM_COMPILE:
        raise VllmFlexTensorV2Error(f"only supports CompilationMode.NONE or CompilationMode.VLLM_COMPILE; got {mode!r}")
    if not offload_config.external_compile:
        raise VllmFlexTensorV2Error("CompilationMode.VLLM_COMPILE requires OffloadConfig.external_compile=True")
    _validate_vllm_compile_topology(vllm_config, offload_config)


def _validate_vllm_compile_topology(vllm_config: VllmConfig, offload_config: OffloadConfig) -> None:
    """Require the validated piecewise Inductor path for rolling weight slots.

    FlexTensor supports Inductor within vLLM's compiled pieces, but not vLLM's
    whole-graph Inductor partitioning. The checks below select the required
    attention split; they do not validate CUDA-graph mode or derive the actual
    graph topology.
    """

    if offload_config.transfer_mode not in BLOCK_TRANSFER_MODES:
        raise VllmFlexTensorV2Error(
            "VLLM_COMPILE requires a block transfer_mode in "
            f"{sorted(BLOCK_TRANSFER_MODES)}; got {offload_config.transfer_mode!r}"
        )

    compilation = getattr(vllm_config, "compilation_config", None)
    if compilation is None or not hasattr(compilation, "use_inductor_graph_partition"):
        raise VllmFlexTensorV2Error(
            "VLLM_COMPILE requires a resolved compilation config that must expose use_inductor_graph_partition=False"
        )
    if compilation.use_inductor_graph_partition is not False:
        raise VllmFlexTensorV2Error(
            "VLLM_COMPILE requires use_inductor_graph_partition=False; "
            "whole-graph Inductor optimization is unsafe for rolling weight slots"
        )

    splitting_ops = getattr(compilation, "splitting_ops", None)
    if not isinstance(splitting_ops, (list, tuple)) or not splitting_ops:
        raise VllmFlexTensorV2Error(
            "VLLM_COMPILE requires non-empty resolved splitting_ops for attention-piecewise compilation"
        )
    attention_ops = getattr(compilation, "_attention_ops", None)
    if not isinstance(attention_ops, (list, tuple)) or not attention_ops:
        raise VllmFlexTensorV2Error(
            "VLLM_COMPILE compilation config must expose a non-empty resolved _attention_ops list"
        )
    if any(not isinstance(op, str) for op in (*splitting_ops, *attention_ops)):
        raise VllmFlexTensorV2Error("VLLM_COMPILE partition operator names must be strings")
    missing_attention_ops = sorted(set(attention_ops) - set(splitting_ops))
    if missing_attention_ops:
        raise VllmFlexTensorV2Error(
            f"VLLM_COMPILE splitting_ops must include every vLLM attention op; missing={missing_attention_ops}"
        )


def _offloader_api() -> tuple[Callable[[], Any], Callable[[Any], None]]:
    try:
        from vllm.model_executor.offloader.base import get_offloader, set_offloader
    except ImportError as exc:
        raise VllmFlexTensorV2Error("offloader singleton API is unavailable") from exc
    return get_offloader, set_offloader


def _validate_native_offloader(offloader: Any) -> None:
    try:
        from vllm.model_executor.offloader.base import NoopOffloader
    except ImportError as exc:
        raise VllmFlexTensorV2Error("native offloader type API is unavailable") from exc
    if not isinstance(offloader, NoopOffloader):
        raise VllmFlexTensorV2Error(
            "native vLLM offloader conflict: "
            f"active={type(offloader).__name__} requested={VllmBootstrapOffloader.__name__}"
        )


def _publish_model_to_runner(model_runner: _ModelRunner, raw_model: nn.Module, proxy: nn.Module) -> None:
    queue = [model_runner]
    seen: set[int] = set()
    replaced = False
    while queue:
        owner = queue.pop(0)
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        for attribute in ("model", "runnable"):
            if not hasattr(owner, attribute):
                continue
            value = getattr(owner, attribute)
            if value is raw_model:
                setattr(owner, attribute, proxy)
                replaced = True
            elif hasattr(value, "model") or hasattr(value, "runnable"):
                queue.append(value)
    if not replaced:
        raise VllmFlexTensorV2Error("could not publish OffloadModelProxy through model runner")


class FlexTensorOffloadWorker(Worker):
    """State-takeover worker for FlexTensor vLLM integration v2."""

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        if self.device.index is None:
            raise VllmFlexTensorV2Error(f"worker CUDA device must have an explicit index, got {self.device}")
        offload_config = load_config(gpu_device=self.device.index)
        self._offload_config = offload_config
        if not offload_config.enabled:
            super().load_model(*args, **kwargs)
            return
        if offload_config.transfer_mode not in BLOCK_TRANSFER_MODES:
            raise VllmFlexTensorV2Error(
                "worker v2 requires a block transfer_mode for production profiling; "
                f"got {offload_config.transfer_mode!r}"
            )

        if kwargs.get("load_dummy_weights"):
            raise VllmFlexTensorV2Error(
                "does not support load_dummy_weights=True as parameter to "
                "Worker.load_model (is this used with enable_elastic_ep?)"
            )

        include_patterns, exclude_patterns = resolve_vllm_patterns(offload_config)
        compilation_config = self.vllm_config.compilation_config
        external_compile = compilation_config.mode == CompilationMode.VLLM_COMPILE
        offload_timing = "eager" if compilation_config.cudagraph_mode == CUDAGraphMode.NONE else "cuda_graph"
        if (
            config_field_was_set(offload_config, "external_compile")
            and offload_config.external_compile != external_compile
        ):
            LOGGER.warning(
                "worker v2 overrides explicit OffloadConfig.external_compile=%r "
                "with %r derived from vLLM compilation mode",
                offload_config.external_compile,
                external_compile,
            )
        if config_field_was_set(offload_config, "offload_timing") and offload_config.offload_timing != offload_timing:
            LOGGER.warning(
                "worker v2 overrides explicit OffloadConfig.offload_timing=%r "
                "with %r derived from vLLM CUDA-graph mode",
                offload_config.offload_timing,
                offload_timing,
            )
        offload_config = offload_config.model_copy(
            update={
                "include_patterns": include_patterns,
                "exclude_patterns": exclude_patterns,
                "external_compile": external_compile,
                "offload_timing": offload_timing,
            }
        )
        self._offload_config = offload_config
        LOGGER.info("FlexTensor %s offloading enabled with config: %s", __version__, offload_config)
        _validate_enabled_worker(self.vllm_config, offload_config)
        timing_batch = inference_profile.timing_batch_from_env()
        profile_sample_target = max(1, offload_config.profiling_iters)
        profile_refresh_enabled = (
            offload_config.profile_storage_dir is not None
            and not offload_config.profile_read_only
            and offload_config.offload_timing in {"eager", "cuda_graph"}
            and timing_batch is not None
        )
        if profile_refresh_enabled and profile_sample_target > OFFLOAD_TIMING_MEASURE_MAX_PASSES:
            raise VllmFlexTensorV2Error(
                "profiling_iters for writable CUDA-graph profile refresh must not exceed "
                f"the durable timing retention limit {OFFLOAD_TIMING_MEASURE_MAX_PASSES}; "
                f"got {offload_config.profiling_iters}"
            )
        saved_profile = inference_profile.load_saved_profile(offload_config)

        get_offloader, set_offloader = _offloader_api()
        previous = get_offloader()
        _validate_native_offloader(previous)
        manager_active = False
        try:
            bootstrap_offloader = VllmBootstrapOffloader()
            set_offloader(bootstrap_offloader)
            super().load_model()

            raw_model = self.model_runner.get_model()
            with DeviceMemoryProfiler(self.device) as takeover_memory:
                proxy = bootstrap_offloader.takeover(
                    raw_model,
                    offload_config,
                    self.device,
                    profile=saved_profile,
                )
                manager_active = True
            _publish_model_to_runner(self.model_runner, raw_model, proxy)
            install_breakable_segment_join()
            self.model_runner.model_memory_usage += takeover_memory.consumed_memory
            atexit.register(flextensor.release)
            self._flextensor_previous_offloader = previous
            self._flextensor_bootstrap_offloader = bootstrap_offloader
            self._flextensor_timing_batch = timing_batch
            self._flextensor_profile_sample_count = 0
            self._flextensor_profile_sample_target = profile_sample_target
            self._flextensor_profile_refresh_enabled = profile_refresh_enabled
            self._flextensor_replay_patch: inference_profile.ReplayPatch | None = None
            LOGGER.info("FlexTensor vLLM integration v2 state takeover complete")
        except Exception:
            try:
                if manager_active:
                    flextensor.release()
            finally:
                set_offloader(previous)
            LOGGER.exception(
                "FlexTensor vLLM integration v2 load failed: last_coordinate=%s",
                getattr(locals().get("bootstrap_offloader"), "last_coordinate", None),
            )
            raise

    def compile_or_warm_up_model(self) -> Any:
        profile_refresh_enabled = getattr(self, "_flextensor_profile_refresh_enabled", False)
        self._flextensor_profile_refresh_enabled = False
        try:
            result = super().compile_or_warm_up_model()
        finally:
            self._flextensor_profile_refresh_enabled = profile_refresh_enabled
        if not profile_refresh_enabled:
            return result
        self._flextensor_replay_patch = inference_profile.patch_cudagraph_replay_counter()
        try:
            self._flextensor_bootstrap_offloader.reset_offload_timing_sampling()
        except Exception as exc:
            self._disable_profile_refresh(f"cannot reset offload timing: {exc}")
            return result
        self._flextensor_profile_sample_count = 0
        LOGGER.info(
            "production offload-timing sampling started batch=%s samples=%d",
            self._flextensor_timing_batch,
            self._flextensor_profile_sample_target,
        )
        return result

    def execute_model(self, scheduler_output: Any) -> Any:
        if not getattr(self, "_flextensor_profile_refresh_enabled", False):
            return super().execute_model(scheduler_output)
        batch = inference_profile.classify_timing_batch(scheduler_output)
        if batch != self._flextensor_timing_batch:
            return super().execute_model(scheduler_output)

        generation_before = inference_profile.current_cudagraph_replay_generation()
        try:
            self._flextensor_bootstrap_offloader.begin_offload_timing_sample()
        except Exception as exc:
            self._disable_profile_refresh(f"offload timing preparation failed: {exc}")
            return super().execute_model(scheduler_output)
        try:
            result = super().execute_model(scheduler_output)
        except BaseException:
            self._flextensor_bootstrap_offloader.cancel_offload_timing_sample()
            raise
        generation_after = inference_profile.current_cudagraph_replay_generation()
        replay_generation = generation_after if generation_after > generation_before else None
        try:
            published = self._flextensor_bootstrap_offloader.finish_offload_timing_sample(
                replay_generation=replay_generation
            )
        except Exception as exc:
            self._disable_profile_refresh(f"offload timing finalization failed: {exc}")
            return result
        if not published:
            self._disable_profile_refresh("offload timing finalization returned no sample")
            return result

        self._flextensor_profile_sample_count += 1
        if self._flextensor_profile_sample_count == self._flextensor_profile_sample_target:
            self._save_refreshed_profile()
        return result

    def _disable_profile_refresh(self, reason: str) -> None:
        if getattr(self, "_flextensor_profile_refresh_enabled", False):
            LOGGER.warning("profile refresh disabled for this run: %s", reason)
        self._stop_profile_refresh()

    def _stop_profile_refresh(self) -> None:
        self._flextensor_profile_refresh_enabled = False
        bootstrap_offloader = getattr(self, "_flextensor_bootstrap_offloader", None)
        if bootstrap_offloader is not None:
            cancel = getattr(bootstrap_offloader, "cancel_offload_timing_sample", None)
            if callable(cancel):
                cancel()
        replay_patch = getattr(self, "_flextensor_replay_patch", None)
        if replay_patch is not None:
            inference_profile.restore_cudagraph_replay(replay_patch)
            self._flextensor_replay_patch = None

    def _save_refreshed_profile(self) -> None:
        self._stop_profile_refresh()
        # TODO: Rename bootstrap_offloader to offloader and consider
        # renaming its class to FlexTensorOffloader.
        inference_profile.save_refreshed_profile(
            config=self._offload_config,
            state=self._flextensor_bootstrap_offloader.runtime_state,
        )

    def shutdown(self) -> None:
        replay_patch = getattr(self, "_flextensor_replay_patch", None)
        if replay_patch is not None:
            inference_profile.restore_cudagraph_replay(replay_patch)
            self._flextensor_replay_patch = None
        try:
            previous = self._flextensor_previous_offloader
        except AttributeError:
            super().shutdown()
            return
        _get_offloader, set_offloader = _offloader_api()
        try:
            try:
                set_offloader(previous)
            finally:
                flextensor.release()
            del self._flextensor_previous_offloader
        finally:
            super().shutdown()

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
from flextensor.shared_offload import SharedOffloadSession
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


def _load_format_name(load_format: Any) -> str:
    value = getattr(load_format, "value", load_format)
    return value if isinstance(value, str) else str(value)


def _validate_shm_worker(  # noqa: C901 - each branch rejects one unsupported SHM topology.
    vllm_config: VllmConfig,
    offload_config: OffloadConfig,
) -> None:
    """Fail closed around the first validated vLLM shared-memory topology."""
    if offload_config.profile_storage_dir is None or not offload_config.profile_read_only:
        raise VllmFlexTensorV2Error("vLLM SHM integration requires FT_PROFILE_STORAGE_DIR and FT_PROFILE_READ_ONLY=1")

    model_config = vllm_config.model_config
    if (
        getattr(model_config, "runner_type", None) != "generate"
        or getattr(model_config, "convert_type", None) != "none"
    ):
        raise VllmFlexTensorV2Error("vLLM SHM integration requires the native, unconverted generation model")
    if getattr(model_config, "quantization", None) is not None:
        raise VllmFlexTensorV2Error("vLLM SHM integration does not support quantization")
    if getattr(vllm_config, "lora_config", None) is not None:
        raise VllmFlexTensorV2Error("vLLM SHM integration does not support LoRA")
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise VllmFlexTensorV2Error("vLLM SHM integration does not support speculative decoding")

    parallel = vllm_config.parallel_config
    if getattr(parallel, "tensor_parallel_size", 1) != 1 or getattr(parallel, "pipeline_parallel_size", 1) != 1:
        raise VllmFlexTensorV2Error("vLLM SHM integration currently requires TP=1 and PP=1")
    if (
        getattr(parallel, "prefill_context_parallel_size", 1) != 1
        or getattr(parallel, "decode_context_parallel_size", 1) != 1
    ):
        raise VllmFlexTensorV2Error("vLLM SHM integration currently requires PCP=1 and DCP=1")
    if getattr(parallel, "data_parallel_backend", None) != "mp" or getattr(parallel, "nnodes", 1) != 1:
        raise VllmFlexTensorV2Error("vLLM SHM integration currently requires single-node data_parallel_backend='mp'")
    if getattr(parallel, "data_parallel_external_lb", False) or getattr(parallel, "data_parallel_hybrid_lb", False):
        raise VllmFlexTensorV2Error("vLLM SHM integration does not support external or hybrid DP load balancing")
    unsupported_ep = any(
        getattr(parallel, field, False)
        for field in ("enable_expert_parallel", "enable_eplb", "enable_ep_weight_filter")
    )
    if unsupported_ep:
        raise VllmFlexTensorV2Error("vLLM SHM integration does not support expert parallelism or EPLB")

    load_config = getattr(vllm_config, "load_config", None)
    load_format = getattr(load_config, "load_format", None)
    if _load_format_name(load_format) != "auto":
        raise VllmFlexTensorV2Error(f"vLLM SHM integration currently requires load_format='auto'; got {load_format!r}")


def _shared_loader_api() -> tuple[str, type[Any], Callable[[Any], Any]]:
    try:
        from flextensor.contrib.vllm.v2.loader import (
            _SHM_LOAD_FORMAT,
            VllmShmLoadContext,
            use_vllm_shm_load,
        )
    except ImportError as exc:
        raise VllmFlexTensorV2Error("vLLM SHM model-loader API is unavailable") from exc
    return _SHM_LOAD_FORMAT, VllmShmLoadContext, use_vllm_shm_load


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

    def load_model(  # noqa: C901 - regular and SHM bootstraps have distinct lifecycles.
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
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
        if offload_config.shm_enabled:
            _validate_shm_worker(self.vllm_config, offload_config)
            self._load_model_from_shared_profile(offload_config, *args, **kwargs)
            return

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
            self._activate_runtime(raw_model, proxy)
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

    def _activate_runtime(self, raw_model: nn.Module, proxy: nn.Module) -> None:
        """Publish the proxy and install the offloader join at graph segment boundaries."""
        _publish_model_to_runner(self.model_runner, raw_model, proxy)
        install_breakable_segment_join()

    def _load_model_from_shared_profile(  # noqa: C901 - transactional load preserves the primary failure.
        self,
        offload_config: OffloadConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        namespace = offload_config.shm_namespace
        profile_directory = offload_config.profile_storage_dir
        if namespace is None or profile_directory is None:
            raise VllmFlexTensorV2Error("vLLM SHM integration requires an explicit namespace and profile directory")

        get_offloader, set_offloader = _offloader_api()
        previous = get_offloader()
        _validate_native_offloader(previous)
        session = SharedOffloadSession(namespace, wait_timeout=offload_config.shm_wait_timeout)
        self._flextensor_shared_lifecycle = True
        cleanup = session.close
        cleanup_registered = False
        cleanup_failed = False
        try:
            atexit.register(cleanup)
            cleanup_registered = True
            shm_load_format, context_type, use_shm_load = _shared_loader_api()
            load_config = self.vllm_config.load_config
            original_load_format = load_config.load_format
            context = context_type(
                session=session,
                offload_config=offload_config,
                profile_directory=profile_directory,
                materialize_device=self.device,
                underlying_load_format=original_load_format,
            )
            bootstrap_offloader = VllmBootstrapOffloader()
            set_offloader(bootstrap_offloader)
            load_config.load_format = shm_load_format
            try:
                with use_shm_load(context):
                    super().load_model(*args, **kwargs)
            finally:
                load_config.load_format = original_load_format

            proxy = context.proxy
            if proxy is None:
                raise VllmFlexTensorV2Error("vLLM SHM loader returned without publishing its model proxy")
            raw_model = self.model_runner.get_model()
            if proxy._get_model() is not raw_model:  # noqa: SLF001 - package-owned proxy handoff boundary.
                raise VllmFlexTensorV2Error("vLLM SHM loader returned a different model to the model runner")
            bootstrap_offloader.bind_shared_runtime(proxy)
            self._activate_runtime(raw_model, proxy)

            self._flextensor_previous_offloader = previous
            self._flextensor_bootstrap_offloader = bootstrap_offloader
            self._flextensor_shared_session = session
            self._flextensor_atexit_cleanup = cleanup
            self._flextensor_shm_role = context.role
            self._flextensor_shm_execution_logged = False
            self._flextensor_profile_refresh_enabled = False
            self._flextensor_replay_patch = None
            LOGGER.info("FlexTensor vLLM integration v2 state takeover complete")
        except BaseException as primary_error:
            try:
                session.close()
            except BaseException as cleanup_error:
                cleanup_failed = True
                self._flextensor_previous_offloader = previous
                self._flextensor_shared_session = session
                if cleanup_registered:
                    self._flextensor_atexit_cleanup = cleanup
                if add_note := getattr(primary_error, "add_note", None):
                    add_note(f"FlexTensor SHM cleanup failed and will be retried at process exit: {cleanup_error!r}")
                LOGGER.exception("FlexTensor vLLM SHM cleanup failed; atexit retry retained")
            else:
                if cleanup_registered:
                    try:
                        atexit.unregister(cleanup)
                    except BaseException as unregister_error:
                        cleanup_failed = True
                        self._flextensor_previous_offloader = previous
                        self._flextensor_shared_session = session
                        self._flextensor_atexit_cleanup = cleanup
                        if add_note := getattr(primary_error, "add_note", None):
                            add_note(f"FlexTensor SHM atexit cleanup could not be unregistered: {unregister_error!r}")
                        LOGGER.exception("FlexTensor vLLM SHM atexit cleanup unregister failed")
            try:
                set_offloader(previous)
            except BaseException as restore_error:
                self._flextensor_previous_offloader = previous
                if add_note := getattr(primary_error, "add_note", None):
                    add_note(f"FlexTensor could not restore the native vLLM offloader: {restore_error!r}")
                LOGGER.exception("FlexTensor could not restore the native vLLM offloader after SHM load failure")
            LOGGER.exception(
                "FlexTensor vLLM integration v2 SHM load failed: last_coordinate=%s cleanup_failed=%s",
                getattr(locals().get("bootstrap_offloader"), "last_coordinate", None),
                cleanup_failed,
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
            result = super().execute_model(scheduler_output)
            if (
                getattr(self, "_flextensor_shared_lifecycle", False)
                and not getattr(self, "_flextensor_shm_execution_logged", False)
                and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0
            ):
                self._flextensor_shm_execution_logged = True
                LOGGER.info(
                    "FlexTensor vLLM SHM execution data_parallel_index=%s role=%s",
                    getattr(self.vllm_config.parallel_config, "data_parallel_index", None),
                    getattr(self, "_flextensor_shm_role", None),
                )
            return result
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

    def shutdown(self) -> None:  # noqa: C901 - all teardown stages run while retaining the first failure.
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
        shared_lifecycle = getattr(self, "_flextensor_shared_lifecycle", False)
        shared_session = getattr(self, "_flextensor_shared_session", None)
        primary_error: BaseException | None = None

        def record_error(error: BaseException, action: str) -> None:
            nonlocal primary_error
            if primary_error is None:
                primary_error = error
            elif add_note := getattr(primary_error, "add_note", None):
                add_note(f"FlexTensor shutdown also failed while {action}: {error!r}")

        try:
            set_offloader(previous)
        except BaseException as error:
            record_error(error, "restoring the native vLLM offloader")

        cleanup_succeeded = False
        try:
            if shared_lifecycle:
                if shared_session is not None:
                    shared_session.close(preserve_models=False)
            else:
                flextensor.release()
        except BaseException as error:
            record_error(error, "closing the FlexTensor runtime")
        else:
            cleanup_succeeded = True

        if shared_lifecycle and cleanup_succeeded:
            try:
                cleanup = getattr(self, "_flextensor_atexit_cleanup", None)
                if cleanup is not None:
                    atexit.unregister(cleanup)
            except BaseException as error:
                record_error(error, "unregistering the SHM atexit cleanup")
            else:
                if hasattr(self, "_flextensor_shared_session"):
                    del self._flextensor_shared_session
                if hasattr(self, "_flextensor_atexit_cleanup"):
                    del self._flextensor_atexit_cleanup

        if primary_error is None:
            del self._flextensor_previous_offloader

        try:
            super().shutdown()
        except BaseException as error:
            record_error(error, "shutting down the vLLM worker")

        if primary_error is not None:
            raise primary_error

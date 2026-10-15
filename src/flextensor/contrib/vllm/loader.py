# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlexTensor-enabled vLLM Model Loader.

Loads models on CPU, then processes weights on GPU layer-by-layer (for CUDA-only ops
like FP8 quantization or MLA attention) before moving them back to CPU for FlexTensor
management. Automatically used by FlexTensorOffloadWorker.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import torch
from torch import nn
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from flextensor.contrib.vllm._logging import safely_install_flextensor_logging_bridge

safely_install_flextensor_logging_bridge()

logger = logging.getLogger(__name__)


def _iter_model_weight_processing_methods(model: nn.Module) -> Iterator[Any]:
    """Yield unique model quantization methods that process weights after loading."""
    seen: set[int] = set()
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None or not hasattr(quant_method, "process_weights_after_loading"):
            continue

        method_id = id(quant_method)
        if method_id in seen:
            continue
        seen.add(method_id)
        yield quant_method


def _defer_vllm_cuda_weight_processing_impl(model: nn.Module) -> Iterator[None]:
    """Defer vLLM weight processing while FlexTensor loads on CPU.

    vLLM can call ``process_weights_after_loading`` as checkpoint shards finish
    loading. FlexTensor intentionally performs Phase 1 on CPU, so processing
    waits until Phase 2, where this loader moves one layer at a time to the
    target GPU.
    """
    patched_methods: list[tuple[Any, bool, Any]] = []

    def _deferred_process_weights_after_loading(_layer: Any) -> None:
        pass

    for quant_method in _iter_model_weight_processing_methods(model):
        method_attrs = getattr(quant_method, "__dict__", {})
        had_instance_method = "process_weights_after_loading" in method_attrs
        original_method = method_attrs.get("process_weights_after_loading")
        patched_methods.append((quant_method, had_instance_method, original_method))
        quant_method.process_weights_after_loading = _deferred_process_weights_after_loading

    if patched_methods:
        logger.info(
            "FlexTensorModelLoader: Deferring vLLM weight processing for %d quantization method "
            "instance(s) until Phase 2; matching modules are processed layer-by-layer on GPU",
            len(patched_methods),
        )

    try:
        yield
    finally:
        for quant_method, had_instance_method, original_method in patched_methods:
            if had_instance_method:
                quant_method.process_weights_after_loading = original_method
            else:
                del quant_method.process_weights_after_loading


defer_vllm_cuda_weight_processing = contextmanager(_defer_vllm_cuda_weight_processing_impl)


@register_model_loader("flextensor")
class FlexTensorModelLoader(DefaultModelLoader):
    """Model loader for FlexTensor offloading with 2-phase loading strategy."""

    _UNDERLYING_FORMAT = "auto"

    def _prepare_weights(self, *args: object, **kwargs: object) -> tuple[str, list[str], bool]:
        """Prepare weights, mapping flextensor format to auto."""
        original_format = self.load_config.load_format
        if original_format == "flextensor":
            self.load_config.load_format = self._UNDERLYING_FORMAT

        try:
            return super()._prepare_weights(*args, **kwargs)  # type: ignore[no-any-return]  # vllm parent is untyped
        finally:
            self.load_config.load_format = original_format

    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, **kwargs: object) -> nn.Module:
        """Load model with 2-phase strategy: CPU init/load -> GPU weight processing (if needed)."""
        cpu_device = torch.device("cpu")
        gpu_device = vllm_config.device_config.device

        logger.info("FlexTensorModelLoader: Phase 1 - Init and load weights on CPU")

        # Override default device to CPU since current_platform.device_type
        # is sometimes used as default device for tensor creation
        original_default_device = None
        original_platform_device_type = current_platform.device_type
        try:
            current_platform.device_type = "cpu"
            with suppress(Exception):
                original_default_device = torch.get_default_device()
                torch.set_default_device("cpu")
            with set_default_torch_dtype(model_config.dtype):
                model = initialize_model(vllm_config=vllm_config, model_config=model_config, **kwargs)
                with defer_vllm_cuda_weight_processing(model):
                    self.load_weights(model, model_config)
        finally:
            current_platform.device_type = original_platform_device_type
            with suppress(Exception):
                torch.set_default_device(original_default_device)

        logger.info("FlexTensorModelLoader: Phase 2 - Process weights on GPU (if needed)")
        self._process_weights_layer_by_layer(model, model_config, gpu_device, cpu_device, vllm_config)

        return model.eval()  # type: ignore[no-any-return]

    def _find_layers_needing_gpu_processing(self, model: nn.Module) -> dict[str, nn.Module]:
        """Find decoder layers containing modules that need GPU weight processing."""
        layers_to_process: dict[str, nn.Module] = {}

        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            has_quant = quant_method is not None and hasattr(quant_method, "process_weights_after_loading")
            has_attn_processing = isinstance(module, AttentionLayerBase) and hasattr(
                module, "process_weights_after_loading"
            )

            if not (has_quant or has_attn_processing):
                continue

            # Find the parent "decoder layer" to process as a unit
            # This ensures dependencies like kv_b_proj are included
            layer_name = self._find_processing_unit(name)
            if layer_name is None:
                continue
            if layer_name not in layers_to_process:
                parent = model
                for part in layer_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                layers_to_process[layer_name] = parent

        return layers_to_process

    def _process_weights_layer_by_layer(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        gpu_device: torch.device,
        cpu_device: torch.device,
        vllm_config: VllmConfig,
    ) -> None:
        """Process weights layer-by-layer on GPU to avoid OOM."""
        layers_to_process = self._find_layers_needing_gpu_processing(model)

        if not layers_to_process:
            logger.info("FlexTensorModelLoader: No layers need GPU processing")
            return

        warmup_deep_gemm = self._deep_gemm_warmup_enabled()
        logger.info(
            "FlexTensorModelLoader: Processing %d layers on GPU (DeepGEMM warmup: %s)",
            len(layers_to_process),
            "on" if warmup_deep_gemm else "off",
        )
        processed_count = 0
        for layer_name, layer_module in layers_to_process.items():
            try:
                layer_module.to(gpu_device)
                self._materialize_parameter_subclasses(layer_module)
                self._process_layer_weights(layer_module, model_config, gpu_device)
                if warmup_deep_gemm:
                    self._warmup_layer_deep_gemm(layer_module, vllm_config)
                layer_module.to(cpu_device)
                torch.cuda.empty_cache()
                processed_count += 1
            except Exception as e:
                logger.error("Failed to process layer %s: %s", layer_name, e)
                with suppress(Exception):
                    layer_module.to(cpu_device)
                raise

        logger.info("FlexTensorModelLoader: Processed %d layers", processed_count)
        if warmup_deep_gemm:
            self._log_deep_gemm_warmup_summary()

    @staticmethod
    def _deep_gemm_warmup_enabled() -> bool:
        """Whether to warm DeepGEMM kernels during load (mirrors vLLM's gate)."""
        try:
            import vllm.envs as envs
            from vllm.utils.deep_gemm import is_deep_gemm_supported
        except Exception:
            return False
        try:
            return bool(envs.VLLM_DEEP_GEMM_WARMUP != "skip" and is_deep_gemm_supported())
        except Exception:
            logger.debug(
                "FlexTensorModelLoader: DeepGEMM support probe failed; disabling warmup",
                exc_info=True,
            )
            return False

    @staticmethod
    def _warmup_layer_deep_gemm(layer: nn.Module, vllm_config: VllmConfig) -> None:
        """JIT DeepGEMM while this layer is transiently on GPU.

        Under offload, weights return to CPU after load, so we warm each layer
        during its transient GPU pass instead of vLLM's post-load warmup.
        Failures are logged and loading continues; DeepGEMM compiles on first
        real use instead.
        """
        try:
            from vllm.model_executor.warmup.deep_gemm_warmup import (
                deepgemm_fp8_gemm_nt_warmup,
                deepgemm_grouped_fp8_gemm_nt_contiguous_warmup,
            )

            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            deepgemm_fp8_gemm_nt_warmup(layer, max_tokens)
            deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(layer, max_tokens)
        except Exception as exc:
            logger.warning(
                "FlexTensorModelLoader: per-layer DeepGEMM warmup failed (%s); "
                "kernels will JIT lazily on first use instead.",
                exc,
            )

    @staticmethod
    def _log_deep_gemm_warmup_summary() -> None:
        """Report how many DeepGEMM kernel shapes were JIT'd during load."""
        try:
            from vllm.model_executor.warmup.deep_gemm_warmup import (
                FP8_GEMM_NT_WARMUP_CACHE,
                GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE,
            )
        except Exception:
            return
        logger.info(
            "FlexTensorModelLoader: DeepGEMM warmup JIT'd %d fp8-linear + %d grouped-MoE kernel shape(s) during load",
            len(FP8_GEMM_NT_WARMUP_CACHE),
            len(GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE),
        )

    def _materialize_parameter_subclasses(self, layer: nn.Module) -> None:
        """Replace loaded vLLM parameter wrappers with plain Parameters.

        vLLM's online quantization loaders register ``BasevLLMParameter``
        subclasses while weights are being loaded. After loading, their loader
        metadata is no longer needed. Keeping the subclasses through deferred
        GPU processing can make compiled quantization helpers recurse through
        ``__torch_function__`` on some online schemes.
        """
        for module in layer.modules():
            for name, param in list(module.named_parameters(recurse=False)):
                if type(param) is nn.Parameter:
                    continue
                setattr(module, name, nn.Parameter(param.detach(), requires_grad=param.requires_grad))

    def _find_processing_unit(self, module_name: str) -> str | None:
        """Find parent decoder layer for module (e.g., 'model.layers.0' for 'model.layers.0.attn')."""
        parts = module_name.split(".")

        # Look for "layers.N" pattern (common in transformer models)
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                # Return "model.layers.N" as the processing unit
                return ".".join(parts[: i + 2])
        return None

    def _process_layer_weights(
        self,
        layer: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
    ) -> None:
        """Process weights for a single layer on GPU."""
        for _name, module in layer.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and hasattr(quant_method, "process_weights_after_loading"):
                quant_method.process_weights_after_loading(module)

            if isinstance(module, AttentionLayerBase) and hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading(model_config.dtype)

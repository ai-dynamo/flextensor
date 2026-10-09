# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-backed SHM model loader for the vLLM integration v2 worker."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype

from flextensor.config import OffloadConfig
from flextensor.contrib.vllm._logging import safely_install_flextensor_logging_bridge
from flextensor.offload_manager import OffloadModelProxy
from flextensor.shared_offload import SharedOffloadSession

safely_install_flextensor_logging_bridge()

logger = logging.getLogger(__name__)
_SHM_LOAD_FORMAT = "flextensor_shm"


@dataclass(slots=True)
class VllmShmLoadContext:
    """Process-local inputs for one profile-backed vLLM SHM load."""

    session: SharedOffloadSession
    offload_config: OffloadConfig
    profile_directory: str
    materialize_device: torch.device
    underlying_load_format: Any
    role: str | None = None
    proxy: OffloadModelProxy | None = None


_ACTIVE_SHM_LOAD: ContextVar[VllmShmLoadContext | None] = ContextVar(
    "flextensor_vllm_shm_load",
    default=None,
)


def _use_vllm_shm_load_impl(context: VllmShmLoadContext) -> Iterator[None]:
    """Expose one worker-owned shared session to the registered loader."""
    if _ACTIVE_SHM_LOAD.get() is not None:
        raise RuntimeError("a vLLM SHM model load is already active in this context")
    token = _ACTIVE_SHM_LOAD.set(context)
    try:
        yield
    finally:
        _ACTIVE_SHM_LOAD.reset(token)


use_vllm_shm_load = contextmanager(_use_vllm_shm_load_impl)


def _snapshot_registered_tensor_storages(
    model: nn.Module,
) -> dict[tuple[str, str], tuple[torch.Tensor, Any]]:
    """Keep live handles to every registered tensor and its backing storage."""
    snapshot: dict[tuple[str, str], tuple[torch.Tensor, Any]] = {
        ("parameter", name): (parameter, parameter.untyped_storage())
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    snapshot.update({
        ("buffer", name): (buffer, buffer.untyped_storage())
        for name, buffer in model.named_buffers(remove_duplicate=False)
    })
    return snapshot


def _validate_registered_tensor_storages_unchanged(
    model: nn.Module,
    snapshot: dict[tuple[str, str], tuple[torch.Tensor, Any]],
) -> None:
    """Reject post-load hooks that detach tensors from FlexTensor-owned storage."""
    current: dict[tuple[str, str], torch.Tensor] = {
        ("parameter", name): parameter for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    current.update({("buffer", name): buffer for name, buffer in model.named_buffers(remove_duplicate=False)})
    changed = []
    for (kind, name), (expected_tensor, expected_storage) in snapshot.items():
        tensor = current.get((kind, name))
        if tensor is None:
            changed.append(f"{kind} {name!r} was removed")
        elif tensor is not expected_tensor:
            changed.append(f"{kind} {name!r} was replaced")
        elif tensor.untyped_storage() is not expected_storage:
            changed.append(f"{kind} {name!r} changed storage")
    if changed:
        raise RuntimeError(
            "vLLM SHM follower post-load processing must preserve registered tensor storage; "
            f"changed tensors: {changed[:10]!r}"
        )


def _process_shm_follower_weights_after_loading(model: nn.Module, model_config: ModelConfig) -> None:
    """Run attention post-load hooks without vLLM's CPU transfer context."""
    for module in model.modules():
        if isinstance(module, AttentionLayerBase) and hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading(model_config.dtype)


@register_model_loader(_SHM_LOAD_FORMAT)
class FlexTensorShmModelLoader(DefaultModelLoader):
    """Restore a profiled vLLM model through a worker-owned shared session."""

    def _load_checkpoint_model(
        self,
        context: VllmShmLoadContext,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        **kwargs: object,
    ) -> nn.Module:
        if context.underlying_load_format == _SHM_LOAD_FORMAT:
            raise RuntimeError("vLLM SHM loader cannot wrap itself")
        shared_format = self.load_config.load_format
        self.load_config.load_format = context.underlying_load_format
        try:
            loader = get_model_loader(self.load_config)
            model = cast("nn.Module", loader.load_model(vllm_config=vllm_config, model_config=model_config, **kwargs))
        finally:
            self.load_config.load_format = shared_format
        context.role = "creator"
        logger.info("saved profile loaded path=%s", Path(context.profile_directory) / "profile.json")
        logger.info("FlexTensor vLLM SHM bootstrap role=creator checkpoint_load=loaded")
        return model

    def _build_meta_model(
        self,
        context: VllmShmLoadContext,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        **kwargs: object,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype), torch.device("meta"):
            model = cast(
                "nn.Module",
                initialize_model(vllm_config=vllm_config, model_config=model_config, **kwargs),
            )
        concrete_parameters = [name for name, parameter in model.named_parameters() if not parameter.is_meta]
        if concrete_parameters:
            raise RuntimeError(
                f"vLLM SHM follower topology constructed concrete parameters: {concrete_parameters[:10]!r}"
            )
        concrete_buffers = [name for name, buffer in model.named_buffers() if not buffer.is_meta]
        if concrete_buffers:
            raise RuntimeError(f"vLLM SHM follower topology constructed concrete buffers: {concrete_buffers[:10]!r}")
        context.role = "follower"
        logger.info("FlexTensor vLLM SHM bootstrap role=follower checkpoint_load=skipped")
        return model.eval()

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        **kwargs: object,
    ) -> nn.Module:
        context = _ACTIVE_SHM_LOAD.get()
        if context is None:
            raise RuntimeError("vLLM SHM loader requires an active worker load context")
        if context.proxy is not None:
            raise RuntimeError("vLLM SHM loader context has already restored a model")

        proxy = cast(
            "OffloadModelProxy",
            context.session.offload_from_profile(
                context.profile_directory,
                config=context.offload_config,
                creator_model_factory=lambda: self._load_checkpoint_model(
                    context,
                    vllm_config,
                    model_config,
                    **kwargs,
                ),
                follower_model_factory=lambda: self._build_meta_model(
                    context,
                    vllm_config,
                    model_config,
                    **kwargs,
                ),
                materialize_device=context.materialize_device,
            ),
        )
        model = proxy._get_model()  # noqa: SLF001 - package-owned proxy handoff to vLLM.
        if context.role == "follower":
            storages = _snapshot_registered_tensor_storages(model)
            with context.materialize_device:
                _process_shm_follower_weights_after_loading(model, model_config)
            _validate_registered_tensor_storages_unchanged(model, storages)
            logger.info(
                "FlexTensor vLLM SHM follower post-load storage identity verified tensors=%d",
                len(storages),
            )
        elif context.role != "creator":
            raise RuntimeError("vLLM SHM session returned without selecting a creator or follower role")
        context.proxy = proxy
        return model

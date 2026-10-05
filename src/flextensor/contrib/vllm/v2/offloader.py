# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implement CPU staging and FlexTensor runtime takeover through the vLLM offloader API."""

from operator import attrgetter
from typing import Any, TypeAlias, cast
from weakref import ReferenceType, WeakMethod, ref

import psutil
import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader as _VllmBaseOffloader

import flextensor
from flextensor.config import OffloadConfig
from flextensor.contrib.vllm.v2 import model_scan, state_builder
from flextensor.contrib.vllm.v2.errors import VllmFlexTensorV2Error
from flextensor.memory_transfer_benchmark import benchmark_memory_transfers
from flextensor.offload_manager import OffloadManager
from flextensor.state_handler import TensorManagerState
from flextensor.tensor_processors import compute_reachable_tensor_map

LOGGER = init_logger("vllm.flextensor.v2.offloader")

_STORAGE_ID = attrgetter("_cdata")

# (source device, storage identity, storage bytes)
StageStorageKey: TypeAlias = tuple[torch.device, int, int]


def _group_aliases(named_tensors: Any) -> tuple[tuple[torch.Tensor, tuple[str, ...]], ...]:
    tensors: dict[int, torch.Tensor] = {}
    aliases: dict[int, list[str]] = {}
    for name, tensor in named_tensors:
        object_id = id(tensor)
        tensors.setdefault(object_id, tensor)
        aliases.setdefault(object_id, []).append(name)
    return tuple((tensors[object_id], tuple(names)) for object_id, names in aliases.items())


def _stage_storage_key(tensor: torch.Tensor) -> StageStorageKey:
    storage = tensor.untyped_storage()
    return tensor.device, int(_STORAGE_ID(storage)), storage.nbytes()


def _stage_parameter_storage_on_cpu(
    parameter: nn.Parameter,
    staged_storages: dict[StageStorageKey, ReferenceType[torch.UntypedStorage]],
) -> None:
    source_storage = parameter.untyped_storage()
    storage_key = _stage_storage_key(parameter)
    cpu_storage_ref = staged_storages.get(storage_key)
    cpu_storage = cpu_storage_ref() if cpu_storage_ref is not None else None
    created = cpu_storage is None
    if cpu_storage is None:
        cpu_backing = torch.empty(source_storage.nbytes(), dtype=torch.uint8, device="cpu")
        cpu_storage = cpu_backing.untyped_storage()
        staged_storages[storage_key] = ref(cpu_storage)

    staged = torch.empty(0, dtype=parameter.dtype, device="cpu")
    try:
        staged.set_(
            cpu_storage,
            parameter.storage_offset(),
            parameter.size(),
            parameter.stride(),
        )
        staged.copy_(parameter.detach())
        parameter.data = staged
    except (RuntimeError, TypeError, ValueError) as exc:
        if created:
            staged_storages.pop(storage_key, None)
        raise VllmFlexTensorV2Error(
            f"cannot preserve parameter storage while staging: shape={tuple(parameter.shape)} "
            f"stride={tuple(parameter.stride())} offset={parameter.storage_offset()}"
        ) from exc


def _move_storage_alias_to_cpu(tensor: torch.Tensor, cpu_storage: torch.UntypedStorage) -> None:
    staged = torch.empty(0, dtype=tensor.dtype, device="cpu")
    try:
        staged.set_(cpu_storage, tensor.storage_offset(), tensor.size(), tensor.stride())
        staged.copy_(tensor.detach())
        tensor.data = staged
    except (RuntimeError, TypeError, ValueError) as exc:
        raise VllmFlexTensorV2Error(
            f"cannot preserve tensor storage alias while staging: shape={tuple(tensor.shape)} "
            f"stride={tuple(tensor.stride())} offset={tensor.storage_offset()}"
        ) from exc


def _storage_aliases_by_key(
    module: nn.Module,
    storage_keys: set[StageStorageKey],
) -> dict[StageStorageKey, list[torch.Tensor]]:
    aliases_by_storage: dict[StageStorageKey, list[torch.Tensor]] = {storage_key: [] for storage_key in storage_keys}
    for tensor in compute_reachable_tensor_map(module).values():
        if tensor.is_meta or tensor.layout != torch.strided:
            continue
        aliases = aliases_by_storage.get(_stage_storage_key(tensor))
        if aliases is not None:
            aliases.append(tensor)
    return aliases_by_storage


class VllmBootstrapOffloader(_VllmBaseOffloader):
    def __init__(self) -> None:
        super().__init__()
        self._wrap_call_index = 0
        self._live_units: list[tuple[int, int, nn.Module]] = []
        self._staged_parameter_ids: set[int] = set()
        self._staged_storages: dict[StageStorageKey, ReferenceType[torch.UntypedStorage]] = {}
        self._staged_storage_sources: dict[StageStorageKey, ReferenceType[torch.UntypedStorage]] = {}
        self._pending_deferred_callbacks: dict[tuple[int, int], set[int]] = {}
        self._post_init_validated = False
        self._runtime_manager: OffloadManager | None = None
        self._runtime_state: TensorManagerState | None = None
        self._state_built = False
        self.last_coordinate: tuple[int, int] | None = None

    def wrap_modules(self, modules_generator: Any) -> list[nn.Module]:
        if self._runtime_manager is not None:
            raise VllmFlexTensorV2Error("wrap_modules cannot run after runtime takeover")
        modules: list[nn.Module] = []
        wrap_call_index = self._wrap_call_index
        for module_index, module in enumerate(modules_generator):
            coordinate = (wrap_call_index, module_index)
            self._live_units.append((*coordinate, module))
            # one completed unit must fit on GPU; stream parameters inside a unit only if measurements require it.
            self._stage_concrete_parameters_on_cpu(module)
            self._install_supported_deferred_callbacks(module, coordinate)
            modules.append(module)
            self.last_coordinate = coordinate
        self._wrap_call_index += 1
        return modules

    def _stage_parameter_on_cpu(self, parameter: nn.Parameter) -> None:
        parameter_id = id(parameter)
        if parameter_id in self._staged_parameter_ids:
            return
        source_storage = parameter.untyped_storage()
        storage_key = _stage_storage_key(parameter)
        source_ref = self._staged_storage_sources.get(storage_key)
        if source_ref is None or source_ref() is not source_storage:
            self._staged_storages.pop(storage_key, None)
            self._staged_storage_sources.pop(storage_key, None)
        _stage_parameter_storage_on_cpu(parameter, self._staged_storages)
        self._staged_storage_sources[storage_key] = ref(source_storage)
        self._staged_parameter_ids.add(parameter_id)

    def _stage_concrete_parameters_on_cpu(self, module: nn.Module) -> None:
        parameters = [
            parameter
            for parameter in module.parameters()
            if not parameter.is_meta and parameter.layout == torch.strided
        ]
        parameter_storage_keys = {
            _stage_storage_key(parameter) for parameter in parameters if id(parameter) not in self._staged_parameter_ids
        }
        new_storage_keys = set()
        for parameter in parameters:
            storage_key = _stage_storage_key(parameter)
            staged_ref = self._staged_storages.get(storage_key)
            source_ref = self._staged_storage_sources.get(storage_key)
            if id(parameter) not in self._staged_parameter_ids and (
                staged_ref is None
                or staged_ref() is None
                or source_ref is None
                or source_ref() is not parameter.untyped_storage()
            ):
                new_storage_keys.add(storage_key)
        required_bytes = sum(storage_key[-1] for storage_key in new_storage_keys)
        available_bytes = psutil.virtual_memory().available
        if required_bytes > available_bytes:
            raise VllmFlexTensorV2Error(
                f"bootstrap staging exceeds current host budget: required={required_bytes} available={available_bytes}"
            )
        aliases_by_storage = _storage_aliases_by_key(module, parameter_storage_keys)
        for parameter in parameters:
            if id(parameter) in self._staged_parameter_ids:
                continue
            storage_key = _stage_storage_key(parameter)
            self._stage_parameter_on_cpu(parameter)
            cpu_storage = parameter.untyped_storage()
            for alias in aliases_by_storage.get(storage_key, ()):
                if alias is parameter:
                    continue
                _move_storage_alias_to_cpu(alias, cpu_storage)
                if isinstance(alias, nn.Parameter):
                    self._staged_parameter_ids.add(id(alias))

    def _install_supported_deferred_callbacks(
        self,
        unit: nn.Module,
        coordinate: tuple[int, int],
    ) -> None:
        callback_modules = tuple(
            module
            for module in unit.modules()
            if getattr(getattr(module, "quant_method", None), "uses_meta_device", False)
        )
        if not callback_modules:
            return

        callback_methods = tuple(cast("Any", module.quant_method) for module in callback_modules)
        original_callbacks = tuple((method, method.process_weights_after_loading) for method in callback_methods)
        pending_callbacks = {id(module) for module in callback_modules}
        complete_deferred_staging = WeakMethod(self._complete_deferred_staging)
        unit_ref = ref(unit)
        self._pending_deferred_callbacks[coordinate] = pending_callbacks
        for owner, (quant_method, original) in zip(callback_modules, original_callbacks, strict=True):

            def process_weights_after_loading(
                module: nn.Module,
                *args: Any,
                _original: Any = original,
                _callback_module_id: int = id(owner),
                **kwargs: Any,
            ) -> Any:
                result = _original(module, *args, **kwargs)
                if _callback_module_id not in pending_callbacks:
                    return result
                if len(pending_callbacks) > 1:
                    pending_callbacks.remove(_callback_module_id)
                    return result

                complete = complete_deferred_staging()
                deferred_unit = unit_ref()
                if complete is None or deferred_unit is None:
                    raise VllmFlexTensorV2Error(
                        f"online quantization placement owner unavailable: coordinate={coordinate}"
                    )
                complete(deferred_unit, coordinate)

                for method, callback in original_callbacks:
                    method.process_weights_after_loading = callback
                pending_callbacks.remove(_callback_module_id)
                return result

            quant_method.process_weights_after_loading = process_weights_after_loading

    def _complete_deferred_staging(self, unit: nn.Module, coordinate: tuple[int, int]) -> None:
        parameters = _group_aliases(unit.named_parameters(remove_duplicate=False))
        meta_names = tuple(name for parameter, names in parameters if parameter.is_meta for name in names)
        if not parameters or meta_names:
            raise VllmFlexTensorV2Error(
                f"online quantization final parameters missing or meta: coordinate={coordinate} names={meta_names}"
            )
        self._stage_concrete_parameters_on_cpu(unit)
        LOGGER.info(
            "online quantization placement complete: coordinate=%s final_bytes=%d",
            coordinate,
            sum(parameter.untyped_storage().nbytes() for parameter, _names in parameters),
        )

    def build_state(
        self,
        model: nn.Module,
        config: OffloadConfig,
        device_gpu: torch.device | str | int,
        profile: TensorManagerState | None = None,
    ) -> TensorManagerState:
        if not self._post_init_validated:
            raise VllmFlexTensorV2Error("state construction requires successful bootstrap post_init")
        if self._state_built:
            raise VllmFlexTensorV2Error("state already built")

        live_units = tuple(self._live_units)
        resolved_device = state_builder.resolve_cuda_device(device_gpu)
        scan_result = model_scan.scan_loaded_model(
            model,
            live_units,
            resolved_device,
            include_patterns=config.include_patterns,
            exclude_patterns=config.exclude_patterns,
        )
        if profile is not None:
            try:
                scan_result = state_builder.merge_profile_statistics(scan_result, profile)
            except (ValueError, VllmFlexTensorV2Error) as exc:
                LOGGER.warning("saved profile is incompatible; using conservative statistics: %s", exc)
            else:
                LOGGER.info("saved profile statistics accepted for bootstrap strategy recomputation")
        memory_stats = benchmark_memory_transfers(scan_result.layer_stats, resolved_device)
        self._state_built = True
        self._live_units.clear()
        return state_builder.build_conservative_state(
            scan_result,
            config,
            resolved_device,
            memory_stats=memory_stats,
        )

    def takeover(
        self,
        model: nn.Module,
        config: OffloadConfig,
        device_gpu: torch.device | str | int,
        profile: TensorManagerState | None = None,
    ) -> nn.Module:
        if self._runtime_manager is not None:
            raise VllmFlexTensorV2Error("runtime takeover already completed")
        # vLLM's post_init has none of the model/config inputs needed for worker-owned takeover.
        self.post_init()
        state = self.build_state(model, config, device_gpu, profile=profile)
        labels = [statistic.label for statistic in state.stats]
        if not labels or any(not isinstance(label, str) or not label for label in labels):
            raise VllmFlexTensorV2Error("bootstrap state must provide non-empty runtime labels")
        if len(labels) != len(set(labels)):
            raise VllmFlexTensorV2Error("bootstrap state runtime labels must be unique")
        LOGGER.info("FlexTensor v2 unit inventory: %s", labels)
        takeover_config = config.model_copy(
            update={
                "include_patterns": labels,
                "exclude_patterns": [],
            }
        )

        proxy: nn.Module | None = None
        runtime_manager: OffloadManager | None = None
        try:
            proxy = flextensor.offload_from_state(
                model,
                state,
                takeover_config,
                allow_strategy_replan=False,
            )
            candidate = getattr(proxy, "offload_manager", None)
            if not isinstance(candidate, OffloadManager):
                raise VllmFlexTensorV2Error("state takeover proxy did not expose an OffloadManager")
            runtime_manager = candidate
            self._runtime_manager = runtime_manager
            self._runtime_state = state
            LOGGER.info("FlexTensor vLLM integration v2 state takeover installed loader_type=%s", state.loader_type)
            # Runtime ownership is established; unused bootstrap CPU backing can now be reclaimed.
            self._staged_storages.clear()
            self._staged_storage_sources.clear()
            return proxy
        except Exception:
            self._runtime_manager = None
            # Prefer the recovered manager; global release is only a fallback when the proxy did not expose it.
            if runtime_manager is not None:
                runtime_manager.release()
            elif proxy is not None:
                flextensor.release()
            raise

    @property
    def runtime_state(self) -> TensorManagerState:
        if self._runtime_state is None:
            raise VllmFlexTensorV2Error("runtime state is unavailable before successful takeover")
        return self._runtime_state

    def reset_offload_timing_sampling(self) -> None:
        if self._runtime_manager is not None:
            self._runtime_manager.reset_offload_timing_sampling()

    def begin_offload_timing_sample(self) -> None:
        if self._runtime_manager is not None:
            self._runtime_manager.begin_offload_timing_sample()

    def finish_offload_timing_sample(self, *, replay_generation: int | None) -> bool:
        return self._runtime_manager is not None and self._runtime_manager.finish_offload_timing_sample(
            replay_generation=replay_generation
        )

    def cancel_offload_timing_sample(self) -> None:
        if self._runtime_manager is not None:
            self._runtime_manager.cancel_offload_timing_sample()

    def sync_prev_onload(self) -> None:
        # vLLM may call this method earlier than we expect it to be called
        if self._runtime_manager is not None:
            self._runtime_manager.sync_prev_onload()

    def join_after_forward(self) -> None:
        # vLLM may call this method earlier than we expect it to be called
        if self._runtime_manager is not None:
            self._runtime_manager.join_after_forward()

    def post_init(self) -> None:
        if self._post_init_validated:
            return
        pending = {coordinate: len(ids) for coordinate, ids in self._pending_deferred_callbacks.items() if ids}
        if not self._live_units or pending:
            raise VllmFlexTensorV2Error(
                f"bootstrap staging incomplete: observed_units={len(self._live_units)} pending={pending}"
            )
        self._post_init_validated = True

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU memory snapshot collection for vLLM workers.

Provides MemorySnapshotMixin for capturing GPU memory state at key lifecycle
points, and two worker classes that apply it to vanilla vLLM and FlexTensor
workers.

Usage (vanilla vLLM):
    vllm serve <model> --worker-cls flextensor.contrib.vllm.snapshot.SnapshotWorker

Usage (FlexTensor):
    FT_ENABLED=1 vllm serve <model> \\
        --worker-cls flextensor.contrib.vllm.snapshot.FlexTensorSnapshotWorker
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from flextensor.config import _register_env_var

_SNAPSHOT_OUTPUT_DIR_ENV_VAR = "FT_VLLM_SNAPSHOT_OUTPUT_DIR"
_register_env_var(_SNAPSHOT_OUTPUT_DIR_ENV_VAR)

try:
    from vllm.utils.mem_utils import MemorySnapshot
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu_worker import Worker
except ModuleNotFoundError:  # vllm not installed (e.g. unit-test environment)
    MemorySnapshot = None
    KVCacheConfig = None
    Worker = None
else:
    # In the else clause so a rename of flextensor.contrib.vllm._logging is
    # not misclassified as "vLLM not installed".
    from flextensor.contrib.vllm._logging import safely_install_flextensor_logging_bridge

    safely_install_flextensor_logging_bridge()

logger = logging.getLogger(__name__)

try:
    from flextensor.contrib.vllm._legacy_worker import FlexTensorOffloadWorker
except (ModuleNotFoundError, ImportError):
    FlexTensorOffloadWorker = None  # type: ignore[assignment,misc]

# E402: must sit below the vLLM/bridge try/except/else so absent vLLM doesn't mask this import.
from flextensor.instrumentation.host_resources import capture_host_resources  # noqa: E402

_GIB = 1024**3

_SNAPSHOT_FIELDS = (
    "torch_peak",
    "free_memory",
    "total_memory",
    "cuda_memory",
    "torch_memory",
    "non_torch_memory",
    "timestamp",
)


def _serialize_module_parameters(params: Any) -> dict[str, dict[str, Any]]:
    """Serialize a module's ``_parameters`` to metadata-only dicts.

    Extracts only shape, dtype, numel, and size_bytes — never raw tensor
    values — so the output is safe for JSON/ElasticSearch ingestion.

    Args:
        params: A module's ``_parameters`` mapping (name → Parameter|None).

    Returns:
        Dict mapping parameter names to their metadata.
        None-valued parameters (e.g., optional bias) are skipped.
    """
    result: dict[str, dict[str, Any]] = {}
    for name, p in params.items():
        if p is None:
            continue
        numel = p.numel()
        result[name] = {
            "shape": list(p.shape),
            "dtype": str(p.dtype),
            "numel": numel,
            "size_bytes": numel * p.element_size(),
        }
    return result


def _collect_model_modules(model: Any) -> list[dict[str, Any]]:
    """Collect parameter metadata for all modules in a model.

    Iterates over ``model.named_modules()`` and serializes each module's
    parameters as metadata-only dicts (no raw tensor values).

    Args:
        model: A PyTorch ``nn.Module`` (or vLLM wrapper with ``named_modules``).

    Returns:
        List of dicts, each with ``name`` and ``tensors_map`` keys.
        Only modules with at least one non-None parameter are included.
    """
    modules: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not hasattr(module, "_parameters") or not module._parameters:  # noqa: SLF001
            continue
        tensors_map = _serialize_module_parameters(module._parameters)  # noqa: SLF001
        if tensors_map:
            modules.append({"name": name, "tensors_map": tensors_map})
    return modules


class MemorySnapshotMixin:
    """Mixin that captures GPU memory snapshots at worker lifecycle points.

    Provides _take_snapshot(label) to record a MemorySnapshot and
    _dump_snapshots() to write all collected snapshots to a JSON file.

    Expects the host class to provide: rank, local_rank, device, vllm_config.
    """

    _snapshots: list[dict]

    def _take_snapshot(self, label: str) -> None:
        """Capture current GPU memory and host resource state and append to snapshot list.

        Args:
            label: Descriptive label for this snapshot (e.g., "after_load_model").
        """
        if not hasattr(self, "_snapshots"):
            self._snapshots = []

        snapshot = MemorySnapshot()
        gpu_memory = {field: getattr(snapshot, field) for field in _SNAPSHOT_FIELDS}
        entry: dict = {
            "label": label,
            "gpu_memory": gpu_memory,
            "host_memory": capture_host_resources(),
        }
        self._snapshots.append(entry)
        logger.info(
            "Memory snapshot [%s]: free=%.2f GiB, cuda=%.2f GiB",
            label,
            gpu_memory["free_memory"] / _GIB,
            gpu_memory["cuda_memory"] / _GIB,
        )

    def _dump_snapshots(self) -> None:
        """Write collected snapshots to a JSON file.

        Writes to the directory specified by ``FT_VLLM_SNAPSHOT_OUTPUT_DIR``.
        If the environment variable is unset or empty, the dump is skipped.
        """
        output_dir_str = os.environ.get(_SNAPSHOT_OUTPUT_DIR_ENV_VAR, "")
        if not output_dir_str:
            logger.debug("%s not set, skipping snapshot dump", _SNAPSHOT_OUTPUT_DIR_ENV_VAR)
            return

        if not hasattr(self, "_snapshots"):
            self._snapshots = []

        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"gpu_snapshots_rank{self.rank}_device{self.local_rank}_{ts}.json"  # type: ignore[attr-defined]
        output_path = output_dir / filename

        data: dict[str, Any] = {
            "worker_type": type(self).__name__,
            "model": self.vllm_config.model_config.model,  # type: ignore[attr-defined]
            "rank": self.rank,  # type: ignore[attr-defined]
            "local_rank": self.local_rank,  # type: ignore[attr-defined]
            "device": str(self.device),  # type: ignore[attr-defined]
            "snapshots": self._snapshots,
        }

        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model = getattr(model_runner, "model", None)
            if model is not None:
                try:
                    data["modules"] = _collect_model_modules(model)
                except (AttributeError, TypeError, RuntimeError):
                    logger.warning("Failed to collect module data for %s", type(model).__name__, exc_info=True)

        output_path.write_text(json.dumps(data, indent=2))
        logger.info("GPU memory snapshots written to %s", output_path)


if Worker is not None:

    class SnapshotWorker(MemorySnapshotMixin, Worker):
        """vLLM Worker with GPU memory snapshot collection.

        Captures memory snapshots at each lifecycle point and dumps them to JSON
        after the final warmup step.

        Usage:
            vllm serve <model> --worker-cls flextensor.contrib.vllm.snapshot.SnapshotWorker
        """

        def init_device(self) -> None:
            """Initialize device and capture post-init memory snapshot."""
            super().init_device()
            self._take_snapshot("after_init_device")

        def load_model(self) -> None:
            """Load model weights and capture post-load memory snapshot."""
            super().load_model()
            self._take_snapshot("after_load_model")

        def determine_available_memory(self) -> int:
            """Determine available GPU memory and capture snapshot.

            Returns:
                Available GPU memory in bytes.
            """
            result = super().determine_available_memory()
            self._take_snapshot("after_determine_available_memory")
            return cast("int", result)

        def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
            """Initialize KV cache from config and capture post-init snapshot.

            Args:
                kv_cache_config: KV cache configuration from the scheduler.
            """
            super().initialize_from_config(kv_cache_config)
            self._take_snapshot("after_kv_cache_init")

        def compile_or_warm_up_model(self) -> Any:
            """Warm up model, capture final snapshot, and dump all snapshots to disk."""
            result = super().compile_or_warm_up_model()
            self._take_snapshot("after_compile_warmup")
            self._dump_snapshots()
            return result


if FlexTensorOffloadWorker is not None:

    class FlexTensorSnapshotWorker(MemorySnapshotMixin, FlexTensorOffloadWorker):
        """FlexTensor Worker with GPU memory snapshot collection.

        Captures memory snapshots at each lifecycle point and dumps them to JSON
        after the final warmup step.

        Usage:
            FT_ENABLED=1 vllm serve <model> \\
                --worker-cls flextensor.contrib.vllm.snapshot.FlexTensorSnapshotWorker
        """

        def init_device(self) -> None:
            """Initialize device and capture post-init memory snapshot."""
            super().init_device()
            self._take_snapshot("after_init_device")

        def load_model(self) -> None:
            """Load model weights and capture post-load memory snapshot."""
            super().load_model()
            self._take_snapshot("after_load_model")

        def determine_available_memory(self) -> int:
            """Determine available GPU memory and capture snapshot.

            Returns:
                Available GPU memory in bytes.
            """
            result = super().determine_available_memory()
            self._take_snapshot("after_determine_available_memory")
            return cast("int", result)

        def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
            """Initialize KV cache from config and capture post-init snapshot.

            Args:
                kv_cache_config: KV cache configuration from the scheduler.
            """
            super().initialize_from_config(kv_cache_config)
            self._take_snapshot("after_kv_cache_init")

        def warmup_and_profile_model(self) -> None:
            """Run FlexTensor discovery/profiling without triggering snapshot dump.

            Sets an internal flag so compile_or_warm_up_model() calls during
            profiling do not take snapshots or dump to disk. The snapshot and
            dump happen only on the final vLLM-orchestrated call.
            """
            self._in_ft_warmup = True
            try:
                super().warmup_and_profile_model()
            finally:
                self._in_ft_warmup = False

        def compile_or_warm_up_model(self) -> Any:
            """Warm up model; capture snapshot and dump only on the final vLLM call.

            When called from warmup_and_profile_model() (internal profiling),
            _in_ft_warmup is True and the snapshot/dump are suppressed.
            The snapshot and dump happen only on the external lifecycle call.
            """
            result = super().compile_or_warm_up_model()
            if not getattr(self, "_in_ft_warmup", False):
                self._take_snapshot("after_compile_warmup")
                self._dump_snapshots()
            return result

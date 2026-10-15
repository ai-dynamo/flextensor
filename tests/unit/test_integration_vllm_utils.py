# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-style tests for the integration-test helpers in vllm_utils.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests.integration import _vllm_utils
from tests.integration._vllm_utils import (
    CpuGpuBusInfo,
    assert_moe_backend_evidence,
    assert_moe_backend_selection,
    assert_moe_backend_selection_in,
    assert_no_triton_cpu_pointer_failure,
    assert_rejected_moe_backend_reason,
    assert_vllm_cuda_platform_logged,
    collect_vllm_backend_evidence,
    load_gpu_memory_snapshots,
    load_latest_instrumentation_dump,
    load_latest_memory_transfer_stats,
    parse_block_assignment_layers,
    parse_log_level,
    parse_nvidia_smi_pcie_query,
    parse_nvidia_smi_topo_matrix,
    sanitize_test_name,
    validate_latest_memory_transfer_stats_for_current_bus,
    validate_memory_transfer_stats_for_bus,
)

if TYPE_CHECKING:
    from pathlib import Path


def _bus_info(
    *,
    device_index: int = 0,
    name: str = "GPU",
    pci_bus_id: str = "00000000:01:00.0",
    pcie_link_gen: int = 4,
    pcie_link_width: int = 16,
    cpu_affinity: str | None = None,
    numa_affinity: str | None = None,
) -> CpuGpuBusInfo:
    return CpuGpuBusInfo(
        device_index=device_index,
        name=name,
        pci_bus_id=pci_bus_id,
        pcie_link_gen=pcie_link_gen,
        pcie_link_width=pcie_link_width,
        cpu_affinity=cpu_affinity,
        numa_affinity=numa_affinity,
    )


def _write_components_dump(root: Path, timestamp: str, filename: str, payload: str) -> Path:
    dump_dir = root / timestamp
    dump_dir.mkdir(parents=True)
    dump_path = dump_dir / filename
    dump_path.write_text(payload)
    return dump_path


def test_sanitize_test_name_replaces_unsafe_path_characters() -> None:
    assert sanitize_test_name('test_model[param/with:"unsafe"]') == "test_model_param_with_unsafe"


def test_parse_log_level_info() -> None:
    assert parse_log_level("(EngineCore pid=123) INFO 04-18 22:28:47 [worker.py:131] hello") == "INFO"


def test_parse_log_level_warning() -> None:
    assert parse_log_level("(EngineCore pid=123) WARNING 04-18 22:28:47 [x.py:1] hi") == "WARNING"


def test_parse_log_level_without_prefix() -> None:
    assert parse_log_level("INFO 04-18 22:28:47 [x.py:1] hi") == "INFO"


def test_parse_log_level_with_vllm_tag() -> None:
    assert parse_log_level("[vLLM] (EngineCore pid=123) INFO foo") == "INFO"


def test_parse_log_level_none_on_plain_text() -> None:
    assert parse_log_level("(EngineCore pid=123) some message without a level") is None


_BLOCK_ASSIGNMENT_FIXTURE = """
some unrelated log line

====================
BLOCK ASSIGNMENT: KnapsackStrategy
====================
Layer        Layer Size    Offload   Transfer | C.Blk T.Blk   Blk Size | Pipeline                       Compute
--------------------------------------------------------------------------------
model.layers.0    123.45MB    -   45.00MB |     0     1    150.00MB | fill blk 1 (1st transfer)       4.20ms
model.layers.1    123.45MB  45.00MB   50.00MB |     1     2    150.00MB | read blk 1, fill blk 2       4.10ms
model.layers.2    123.45MB  50.00MB       -   |     2     -          - | read blk 2                    4.30ms
--------------------------------------------------------------------------------
Total: layer_size=370.34MB, offload=95.00MB, compute=12.60ms
Compute: min=4.10ms, max=4.30ms, avg=4.20ms
Block Sizes:
  Block 0:   150.00MB  (transfers=1: model.layers.0 | computes=1: model.layers.0)
  Block 1:   150.00MB  (transfers=1: model.layers.1 | computes=1: model.layers.1)

other log stuff after the table
"""


def test_parse_block_assignment_layers_returns_labels_in_order() -> None:
    layers = parse_block_assignment_layers(_BLOCK_ASSIGNMENT_FIXTURE.splitlines())
    assert layers == ["model.layers.0", "model.layers.1", "model.layers.2"]


def test_parse_block_assignment_layers_empty_when_no_table() -> None:
    assert parse_block_assignment_layers(["no table here", "just noise"]) == []


def test_parse_block_assignment_layers_tolerates_vllm_pid_prefix() -> None:
    prefixed = [f"(EngineCore pid=123) {line}" for line in _BLOCK_ASSIGNMENT_FIXTURE.splitlines()]
    assert parse_block_assignment_layers(prefixed) == ["model.layers.0", "model.layers.1", "model.layers.2"]


def test_parse_block_assignment_layers_stops_at_trailing_content() -> None:
    # "other log stuff after the table" must not be captured as a layer.
    layers = parse_block_assignment_layers(_BLOCK_ASSIGNMENT_FIXTURE.splitlines())
    assert "other" not in layers


def test_parse_nvidia_smi_pcie_query_returns_gpu_bus_details() -> None:
    output = """
0, NVIDIA RTX PRO 6000 Blackwell, 00000000:01:00.0, 3, 16, 5, 16
1, NVIDIA RTX PRO 6000 Blackwell, 00000000:41:00.0, 4, 8, 5, 16
"""

    bus = parse_nvidia_smi_pcie_query(output, device_index=0)

    assert bus == CpuGpuBusInfo(
        device_index=0,
        name="NVIDIA RTX PRO 6000 Blackwell",
        pci_bus_id="00000000:01:00.0",
        pcie_link_gen=5,
        pcie_link_width=16,
        pcie_link_gen_current=3,
        pcie_link_width_current=16,
        cpu_affinity=None,
        numa_affinity=None,
    )


def test_parse_nvidia_smi_pcie_query_matches_gpu_uuid_when_visible_devices_uses_uuid() -> None:
    output = """
0, NVIDIA RTX PRO 6000 Blackwell, GPU-uuid-0, 00000000:01:00.0, 3, 16, 5, 16
1, NVIDIA RTX PRO 6000 Blackwell, GPU-uuid-1, 00000000:41:00.0, 4, 8, 5, 16
"""

    bus = parse_nvidia_smi_pcie_query(output, device_uuid="GPU-uuid-1")

    assert bus == CpuGpuBusInfo(
        device_index=1,
        name="NVIDIA RTX PRO 6000 Blackwell",
        pci_bus_id="00000000:41:00.0",
        pcie_link_gen=5,
        pcie_link_width=16,
        pcie_link_gen_current=4,
        pcie_link_width_current=8,
        cpu_affinity=None,
        numa_affinity=None,
    )


def test_parse_nvidia_smi_topo_matrix_adds_cpu_affinity() -> None:
    bus = _bus_info(name="NVIDIA RTX PRO 6000 Blackwell", pcie_link_gen=5)
    topo_output = """
\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\tX\t0-31\t0\tN/A
"""

    updated = parse_nvidia_smi_topo_matrix(topo_output, bus)

    assert updated.cpu_affinity == "0-31"
    assert updated.numa_affinity == "0"


def test_parse_nvidia_smi_topo_matrix_tolerates_non_gpu_columns() -> None:
    bus = _bus_info(pcie_link_gen=5)
    topo_output = """
\tGPU0\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\tX\tPIX\t0-31,64-95\t0\tN/A
"""

    updated = parse_nvidia_smi_topo_matrix(topo_output, bus)

    assert updated.cpu_affinity == "0-31,64-95"
    assert updated.numa_affinity == "0"


def test_parse_nvidia_smi_topo_matrix_uses_affinity_column_names() -> None:
    bus = _bus_info(pcie_link_gen=5)
    topo_output = """
\tGPU0\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\tExtra Column
GPU0\tX\tPIX\t0-31,64-95\t0\tN/A\tignored
"""

    updated = parse_nvidia_smi_topo_matrix(topo_output, bus)

    assert updated.cpu_affinity == "0-31,64-95"
    assert updated.numa_affinity == "0"


def test_query_cpu_gpu_bus_info_maps_uuid_visible_device(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-uuid-1")

    def _fake_run(cmd, **_kwargs):
        if cmd[:1] == ["nvidia-smi"] and "--query-gpu=" in cmd[1]:
            assert "uuid" in cmd[1]
            return SimpleNamespace(
                stdout=(
                    "0, NVIDIA RTX PRO 6000 Blackwell, GPU-uuid-0, 00000000:01:00.0, 3, 16, 5, 16\n"
                    "1, NVIDIA RTX PRO 6000 Blackwell, GPU-uuid-1, 00000000:41:00.0, 4, 8, 5, 16\n"
                ),
                returncode=0,
            )
        if cmd == ["nvidia-smi", "topo", "-m"]:
            return SimpleNamespace(stdout="", returncode=1)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(_vllm_utils.subprocess, "run", _fake_run)

    bus = _vllm_utils.query_cpu_gpu_bus_info()

    assert bus.device_index == 1
    assert bus.pci_bus_id == "00000000:41:00.0"


def test_validate_memory_transfer_stats_accepts_bandwidth_in_pcie_range() -> None:
    bus = _bus_info(cpu_affinity="0-31", numa_affinity="0")

    summary = validate_memory_transfer_stats_for_bus({"1073741824": 45.0}, bus)

    assert summary["sample_size_bytes"] == 1073741824
    assert summary["observed_bandwidth_gbps"] == pytest.approx(23.86, rel=0.01)
    assert summary["theoretical_bandwidth_gbps"] == pytest.approx(31.51, rel=0.01)


def test_validate_memory_transfer_stats_rejects_out_of_range_bandwidth() -> None:
    with pytest.raises(AssertionError, match="outside expected"):
        validate_memory_transfer_stats_for_bus({"1073741824": 400.0}, _bus_info())


def test_load_latest_memory_transfer_stats_reads_instrumentation_dump(tmp_path) -> None:
    older_dir = tmp_path / "20260101_010101"
    newer_dir = tmp_path / "20260101_020202"
    older_dir.mkdir()
    newer_dir.mkdir()
    (older_dir / "components.old.json").write_text('{"memory_transfer_stats": {"1024": 0.5}}')
    (newer_dir / "components.new.json").write_text('{"memory_transfer_stats": {"2048": 0.25}}')

    stats, dump_path = load_latest_memory_transfer_stats(tmp_path)

    assert stats == {"2048": 0.25}
    assert dump_path.name == "components.new.json"


def test_assert_vllm_cuda_platform_logged_accepts_vllm_platform_message() -> None:
    assert_vllm_cuda_platform_logged(["INFO 05-21 12:00:00 [__init__.py:1] Automatically detected platform cuda."])


def test_assert_vllm_cuda_platform_logged_accepts_gpu_kv_cache_message() -> None:
    assert_vllm_cuda_platform_logged([
        "INFO 05-21 12:00:00 [kv_cache_utils.py:1319] GPU KV cache size: 1,771,984 tokens"
    ])


def test_assert_vllm_cuda_platform_logged_rejects_missing_platform_message() -> None:
    with pytest.raises(AssertionError, match="CUDA platform"):
        assert_vllm_cuda_platform_logged(["INFO server started without platform line"])


def test_collect_vllm_backend_evidence_records_moe_attention_and_run_config() -> None:
    log_lines = [
        "[vLLM] (APIServer pid=139) INFO 05-21 12:28:45 [entrypoints/utils.py:233] "
        "non-default args: {'model_tag': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4', "
        "'model': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4', 'enforce_eager': True}",
        "[vLLM] (EngineCore pid=273) INFO 05-21 12:29:03 [v1/engine/core.py:109] "
        "Initializing a V1 LLM engine (v0.20.2) with config: "
        "model='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4', dtype=torch.bfloat16, "
        "quantization=modelopt_fp4, device_config=cuda, "
        "kernel_config=KernelConfig(enable_flashinfer_autotune=True, moe_backend='auto')",
        "[vLLM] (EngineCore pid=273) DEBUG 05-21 12:29:05 "
        "[model_executor/.../oracle/nvfp4.py:283] NvFp4 MoE backend 'FLASHINFER_TRTLLM' "
        "does not support the deployment configuration since kernel does not support current device cuda.",
        "[vLLM] (EngineCore pid=273) INFO 05-21 12:29:05 "
        "[model_executor/.../oracle/nvfp4.py:280] Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend "
        "out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTLASS'].",
        "[vLLM] (EngineCore pid=273) INFO 05-21 12:29:05 [platforms/cuda.py:368] "
        "Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN'].",
    ]

    evidence = collect_vllm_backend_evidence(
        log_lines,
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        cli_args=["--enforce-eager", "--trust-remote-code"],
        extra_env_vars={"FT_MAX_GPU_MEM_FRACTION": "0.75"},
        runtime_gpu={"gpu_name": "NVIDIA RTX PRO 6000 Blackwell", "sm_capability": "sm_120"},
    )

    assert evidence["model_name"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    assert evidence["checkpoint"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    assert evidence["vllm_args"] == ["--enforce-eager", "--trust-remote-code"]
    assert evidence["extra_env_vars"]["FT_MAX_GPU_MEM_FRACTION"] == "0.75"
    assert evidence["runtime_gpu"]["gpu_name"] == "NVIDIA RTX PRO 6000 Blackwell"
    assert evidence["runtime_gpu"]["sm_capability"] == "sm_120"
    assert evidence["quantization"] == "modelopt_fp4"
    assert evidence["device_config"] == "cuda"
    assert evidence["kernel_config_moe_backend"] == "auto"
    assert evidence["selected_moe_backend"] == "FLASHINFER_CUTLASS"
    assert evidence["selected_moe_backend_family"] == "NvFp4"
    assert evidence["potential_moe_backends"] == ["FLASHINFER_TRTLLM", "FLASHINFER_CUTEDSL", "FLASHINFER_CUTLASS"]
    assert evidence["rejected_moe_backends"] == [
        {
            "backend": "FLASHINFER_TRTLLM",
            "family": "NvFp4",
            "reason": "kernel does not support current device cuda",
        }
    ]
    assert evidence["selected_attention_backend"] == "FLASHINFER"
    assert evidence["potential_attention_backends"] == ["FLASHINFER", "TRITON_ATTN"]


def test_collect_vllm_backend_evidence_records_unquoted_unquantized_rejections() -> None:
    log_lines = [
        "[vLLM] (EngineCore pid=242) INFO 05-21 15:26:01 [v1/engine/core.py:109] "
        "Initializing a V1 LLM engine (v0.20.2) with config: model='Qwen/Qwen3.6-35B-A3B', "
        "quantization=None, device_config=cuda, "
        "kernel_config=KernelConfig(enable_flashinfer_autotune=True, moe_backend='auto')",
        "[vLLM] (EngineCore pid=242) DEBUG 05-21 15:26:07 "
        "[model_executor/.../oracle/unquantized.py:289] Unquantized MoE backend FlashInfer TRTLLM "
        "does not support the deployment configuration since kernel does not support current device cuda.",
        "[vLLM] (EngineCore pid=242) DEBUG 05-21 15:26:07 "
        "[model_executor/.../oracle/unquantized.py:289] Unquantized MoE backend FlashInfer CUTLASS "
        "does not support the deployment configuration since kernel does not support current device cuda.",
        "[vLLM] (EngineCore pid=242) INFO 05-21 15:26:07 "
        "[model_executor/.../oracle/unquantized.py:286] Using TRITON Unquantized MoE backend "
        "out of potential backends: ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON'].",
    ]

    evidence = collect_vllm_backend_evidence(log_lines, model_name="Qwen/Qwen3.6-35B-A3B")

    assert_moe_backend_selection(
        evidence,
        expected_backend="TRITON",
        expected_family="Unquantized",
        expected_potential_backends=("FlashInfer TRTLLM", "FlashInfer CUTLASS", "TRITON"),
    )
    assert_rejected_moe_backend_reason(
        evidence,
        backend="FlashInfer TRTLLM",
        reason_contains="kernel does not support current device cuda",
    )
    assert_rejected_moe_backend_reason(
        evidence,
        backend="FlashInfer CUTLASS",
        reason_contains="kernel does not support current device cuda",
    )


def test_collect_vllm_backend_evidence_records_multiword_unquoted_selection() -> None:
    log_lines = [
        "[vLLM] (EngineCore pid=299) INFO 05-21 16:12:14 "
        "[model_executor/.../oracle/unquantized.py:286] Using FlashInfer CUTLASS Unquantized MoE backend "
        "out of potential backends: ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON'].",
    ]

    evidence = collect_vllm_backend_evidence(log_lines, model_name="Qwen/Qwen3.6-35B-A3B")

    assert_moe_backend_selection_in(
        evidence,
        expected_backends=("FlashInfer CUTLASS",),
        expected_family="Unquantized",
        expected_potential_backends=("FlashInfer TRTLLM", "FlashInfer CUTLASS", "TRITON"),
    )


@pytest.mark.parametrize(
    ("selection_log", "expected_backend"),
    [
        ("Using TRITON backend for Unquantized MoE", "TRITON"),
        ("Using FlashInfer CUTLASS backend for Unquantized MoE", "FlashInfer CUTLASS"),
    ],
)
def test_collect_vllm_backend_evidence_accepts_nvidia_unquantized_word_order(
    selection_log: str,
    expected_backend: str,
) -> None:
    evidence = collect_vllm_backend_evidence([selection_log], model_name="Qwen/Qwen3.6-35B-A3B")

    assert evidence["selected_moe_backend"] == expected_backend
    assert evidence["selected_moe_backend_family"] == "Unquantized"


def test_nvidia_selection_clears_stale_potential_backends() -> None:
    evidence = collect_vllm_backend_evidence(
        [
            "Using FlashInfer CUTLASS Unquantized MoE backend out of potential backends: ['FlashInfer CUTLASS'].",
            "Using TRITON backend for Unquantized MoE",
        ],
        model_name="Qwen/Qwen3.6-35B-A3B",
    )

    assert evidence["selected_moe_backend"] == "TRITON"
    assert evidence["potential_moe_backends"] == []


def test_collect_vllm_backend_evidence_records_quantized_selection_without_potential_backends() -> None:
    log_lines = [
        "[vLLM] (EngineCore pid=881) INFO 05-21 17:42:14 "
        "[model_executor/.../oracle/mxfp8.py:88] Using 'MARLIN' MxFp8 MoE backend.",
    ]

    evidence = collect_vllm_backend_evidence(log_lines, model_name="Qwen/Qwen3.6-35B-A3B")

    assert_moe_backend_selection(
        evidence,
        expected_backend="MARLIN",
        expected_family="MxFp8",
    )
    assert evidence["potential_moe_backends"] == []


def test_assert_moe_backend_selection_in_accepts_one_of_expected_backends() -> None:
    evidence = {
        "selected_moe_backend": "FlashInfer CUTLASS",
        "selected_moe_backend_family": "Unquantized",
        "potential_moe_backends": ["FlashInfer TRTLLM", "FlashInfer CUTLASS", "TRITON"],
    }

    assert_moe_backend_selection_in(
        evidence,
        expected_backends=("FlashInfer TRTLLM", "FlashInfer CUTLASS", "TRITON"),
        expected_family="Unquantized",
        expected_potential_backends=("FlashInfer TRTLLM", "FlashInfer CUTLASS", "TRITON"),
    )


def test_assert_moe_backend_selection_in_rejects_unexpected_backend() -> None:
    evidence = {
        "selected_moe_backend": "BATCHED_TRITON",
        "selected_moe_backend_family": "Unquantized",
        "potential_moe_backends": ["FlashInfer CUTLASS", "TRITON", "BATCHED_TRITON"],
    }

    with pytest.raises(AssertionError, match="Expected one of"):
        assert_moe_backend_selection_in(
            evidence,
            expected_backends=("FlashInfer CUTLASS", "TRITON"),
            expected_family="Unquantized",
            expected_potential_backends=("FlashInfer CUTLASS", "TRITON"),
        )


def test_assert_moe_backend_evidence_rejects_logs_without_selected_backend() -> None:
    evidence = collect_vllm_backend_evidence(["INFO server started"], model_name="Qwen/Qwen3.6-35B-A3B")

    with pytest.raises(AssertionError, match="selected MoE backend"):
        assert_moe_backend_evidence(evidence)


def test_assert_no_triton_cpu_pointer_failure_rejects_old_failure_signature() -> None:
    with pytest.raises(AssertionError, match="Pointer argument"):
        assert_no_triton_cpu_pointer_failure([
            "ValueError: Pointer argument (at 1) cannot be accessed from Triton (cpu tensor?)"
        ])


def test_parse_nvidia_smi_pcie_query_rejects_missing_selected_gpu() -> None:
    output = "0, NVIDIA A30, 00000000:01:00.0, 4, 16, 4, 16\n"

    with pytest.raises(ValueError, match="GPU index 1"):
        parse_nvidia_smi_pcie_query(output, device_index=1)


def test_parse_nvidia_smi_pcie_query_rejects_unreported_link_field() -> None:
    output = "0, NVIDIA A30, 00000000:01:00.0, N/A, 16, 4, 16\n"

    with pytest.raises(ValueError, match="current PCIe link generation"):
        parse_nvidia_smi_pcie_query(output, device_index=0)


def test_query_cpu_gpu_bus_info_maps_numeric_visible_device(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    def _fake_run(cmd, **_kwargs):
        if cmd[:1] == ["nvidia-smi"] and "--query-gpu=" in cmd[1]:
            return SimpleNamespace(stdout="3, NVIDIA A30, GPU-uuid-3, 00000000:81:00.0, 4, 16, 4, 16\n")
        if cmd == ["nvidia-smi", "topo", "-m"]:
            return SimpleNamespace(stdout="", returncode=1)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(_vllm_utils.subprocess, "run", _fake_run)

    bus = _vllm_utils.query_cpu_gpu_bus_info()

    assert bus.device_index == 3
    assert bus.name == "NVIDIA A30"
    assert bus.pci_bus_id == "00000000:81:00.0"


def test_load_latest_memory_transfer_stats_skips_dumps_without_stats(tmp_path) -> None:
    _write_components_dump(
        tmp_path,
        "20260101_010101",
        "components.old.json",
        '{"memory_transfer_stats": {"4096": 0.5}}',
    )
    _write_components_dump(tmp_path, "20260101_020202", "components.new.json", '{"components": []}')

    stats, dump_path = load_latest_memory_transfer_stats(tmp_path)

    assert stats == {"4096": 0.5}
    assert dump_path.name == "components.old.json"


def test_load_latest_memory_transfer_stats_rejects_missing_stats(tmp_path) -> None:
    _write_components_dump(tmp_path, "20260101_010101", "components.json", '{"components": []}')

    with pytest.raises(AssertionError, match="No FlexTensor memory transfer stats"):
        load_latest_memory_transfer_stats(tmp_path)


def test_load_latest_instrumentation_dump_reads_newest_payload(tmp_path) -> None:
    instrumentation_dir = tmp_path / "instrumentation"
    _write_components_dump(instrumentation_dir, "20260101_010101", "components.old.json", '{"old": true}')
    _write_components_dump(instrumentation_dir, "20260101_020202", "components.new.json", '{"new": true}')

    dump_path, payload = load_latest_instrumentation_dump(tmp_path)

    assert dump_path.name == "components.new.json"
    assert payload == {"new": True}


def test_load_latest_instrumentation_dump_rejects_missing_dump(tmp_path) -> None:
    with pytest.raises(AssertionError, match="No FlexTensor instrumentation dump"):
        load_latest_instrumentation_dump(tmp_path)


def test_load_gpu_memory_snapshots_reads_rank_zero_file(tmp_path) -> None:
    (tmp_path / "gpu_snapshots_rank0_device0.json").write_text('{"snapshots": [{"label": "after_load_model"}]}')
    (tmp_path / "gpu_snapshots_rank1_device0.json").write_text('{"snapshots": [{"label": "other_rank"}]}')

    assert load_gpu_memory_snapshots(tmp_path) == [{"label": "after_load_model"}]


def test_load_gpu_memory_snapshots_returns_empty_when_missing(tmp_path) -> None:
    assert load_gpu_memory_snapshots(tmp_path) == []


def test_validate_latest_memory_transfer_stats_for_current_bus_adds_dump_path(monkeypatch, tmp_path) -> None:
    dump_path = _write_components_dump(
        tmp_path,
        "20260101_010101",
        "components.stats.json",
        '{"memory_transfer_stats": {"1073741824": 45.0}}',
    )
    monkeypatch.setattr(_vllm_utils, "query_cpu_gpu_bus_info", _bus_info)

    summary = validate_latest_memory_transfer_stats_for_current_bus(tmp_path)

    assert summary["instrumentation_dump"] == str(dump_path)
    assert summary["gpu_name"] == "GPU"


def test_validate_memory_transfer_stats_rejects_unknown_pcie_generation() -> None:
    with pytest.raises(AssertionError, match="Unsupported PCIe link generation"):
        validate_memory_transfer_stats_for_bus({"1024": 1.0}, _bus_info(pcie_link_gen=99))


def test_assert_moe_backend_selection_in_rejects_missing_potential_backend() -> None:
    evidence = {
        "selected_moe_backend": "TRITON",
        "selected_moe_backend_family": "Fp8",
        "potential_moe_backends": ["TRITON"],
    }

    with pytest.raises(AssertionError, match="Expected potential MoE backends"):
        assert_moe_backend_selection_in(
            evidence,
            expected_backends=("TRITON",),
            expected_family="Fp8",
            expected_potential_backends=("TRITON", "FlashInfer CUTLASS"),
        )

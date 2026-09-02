# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for vLLM integration tests with FlexTensor."""

import csv
import json
import os
import re
import subprocess  # noqa: S404 - subprocess needed for test utilities
from ast import literal_eval
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CpuGpuBusInfo:
    """CPU-to-GPU bus details collected by the pytest process."""

    device_index: int
    name: str
    pci_bus_id: str
    pcie_link_gen: int
    pcie_link_width: int
    pcie_link_gen_current: int | None = None
    pcie_link_width_current: int | None = None
    cpu_affinity: str | None = None
    numa_affinity: str | None = None


def sanitize_test_name(test_name: str) -> str:
    """Sanitize a pytest test name so it can be used as a directory name."""
    sanitized = re.sub(r'[<>:"/\\|?*\[\]]', "_", test_name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def assert_vllm_cuda_platform_logged(log_lines: list[str]) -> None:
    """Assert that vLLM selected the CUDA platform for this server run."""
    log_text = "\n".join(log_lines).lower()
    cuda_platform_patterns = (
        "automatically detected platform cuda",
        "platform cuda",
        "cuda platform",
        "device_config=cuda",
        "gpu kv cache",
        "gpu usage",
    )

    assert any(pattern in log_text for pattern in cuda_platform_patterns), (
        "vLLM did not log CUDA platform usage. Expected a log line like 'Automatically detected platform cuda.'"
    )


def _parse_backend_list(raw_list: str) -> list[str]:
    try:
        parsed = literal_eval(raw_list)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [part.strip().strip("'\"") for part in raw_list.strip("[]").split(",") if part.strip()]


def _first_config_value(log_text: str, pattern: str) -> str | None:
    match = re.search(pattern, log_text)
    return match.group(1) if match else None


def query_runtime_gpu_backend_metadata() -> dict[str, Any]:
    """Return best-effort local GPU metadata for backend-coverage artifacts."""
    try:
        import torch
    except ImportError:
        return {}

    try:
        if not torch.cuda.is_available():
            return {}
        capability = torch.cuda.get_device_capability(0)
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(capability),
            "sm_capability": f"sm_{capability[0]}{capability[1]}",
        }
    except (AssertionError, RuntimeError):
        return {}


def collect_vllm_backend_evidence(
    log_lines: list[str],
    *,
    model_name: str,
    cli_args: list[str] | tuple[str, ...] | None = None,
    extra_env_vars: dict[str, str] | None = None,
    runtime_gpu: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect vLLM backend-selection evidence for integration artifacts."""
    log_text = "\n".join(log_lines)
    evidence: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint": _first_config_value(log_text, r"model='([^']+)'") or model_name,
        "vllm_args": list(cli_args or ()),
        "extra_env_vars": dict(extra_env_vars or {}),
        "runtime_gpu": dict(runtime_gpu or {}),
        "vllm_version": _first_config_value(log_text, r"LLM engine \(v([^)]+)\)"),
        "non_default_args": _first_config_value(log_text, r"non-default args:\s*(\{.*\})"),
        "quantization": _first_config_value(log_text, r"quantization=([^,\s)]+)"),
        "device_config": _first_config_value(log_text, r"device_config=([^,\s)]+)"),
        "kernel_config_moe_backend": _first_config_value(log_text, r"moe_backend='([^']+)'"),
        "selected_moe_backend": None,
        "selected_moe_backend_family": None,
        "potential_moe_backends": [],
        "rejected_moe_backends": [],
        "selected_attention_backend": None,
        "potential_attention_backends": [],
        "attention_rejection_summary": None,
    }

    selected_moe_re = re.compile(
        r"Using\s+(?:'(?P<quoted_backend>[^']+)'|(?P<plain_backend>.+?))\s+"
        r"(?P<family>\w+)\s+MoE backend"
        r"(?:\s+out of potential backends:\s+(?P<potential>\[[^\]]*\]))?"
    )
    nvidia_selected_moe_re = re.compile(r"Using\s+(?P<backend>.+?)\s+backend for\s+(?P<family>\w+)\s+MoE")
    rejected_moe_re = re.compile(
        r"(?P<family>\w+)\s+MoE backend\s+"
        r"(?:'(?P<quoted_backend>[^']+)'|(?P<plain_backend>.+?))\s+does not support "
        r"the deployment configuration since (?P<reason>.*?)(?:\.)?$"
    )
    selected_attention_re = re.compile(
        r"Using\s+(?P<backend>\S+)\s+attention backend out of potential backends:\s+"
        r"(?P<potential>\[[^\]]*\])"
    )

    rejected_moe: list[dict[str, str]] = []
    for raw_line in log_lines:
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line)

        selected_moe = selected_moe_re.search(line)
        if selected_moe:
            evidence["selected_moe_backend"] = (
                selected_moe.group("quoted_backend") or selected_moe.group("plain_backend")
            ).strip()
            evidence["selected_moe_backend_family"] = selected_moe.group("family")
            evidence["potential_moe_backends"] = _parse_backend_list(selected_moe.group("potential") or "[]")
            continue

        nvidia_selected_moe = nvidia_selected_moe_re.search(line)
        if nvidia_selected_moe:
            evidence["selected_moe_backend"] = nvidia_selected_moe.group("backend").strip()
            evidence["selected_moe_backend_family"] = nvidia_selected_moe.group("family")
            evidence["potential_moe_backends"] = []
            continue

        rejected = rejected_moe_re.search(line)
        if rejected:
            rejected_moe.append({
                "backend": (rejected.group("quoted_backend") or rejected.group("plain_backend")).strip(),
                "family": rejected.group("family"),
                "reason": rejected.group("reason"),
            })
            continue

        selected_attention = selected_attention_re.search(line)
        if selected_attention:
            evidence["selected_attention_backend"] = selected_attention.group("backend")
            evidence["potential_attention_backends"] = _parse_backend_list(selected_attention.group("potential"))
            continue

        if "Some attention backends are not valid" in line:
            evidence["attention_rejection_summary"] = line

    evidence["rejected_moe_backends"] = rejected_moe
    return evidence


def assert_moe_backend_evidence(evidence: dict[str, object]) -> None:
    """Assert that a MoE smoke run captured vLLM's selected MoE backend."""
    assert evidence.get("selected_moe_backend"), (
        "vLLM logs did not include a selected MoE backend. Expected a line like "
        "'Using TRITON Unquantized MoE backend' or "
        "'Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend' or "
        "'Using 'MARLIN' MxFp8 MoE backend'."
    )


def assert_moe_backend_selection(
    evidence: dict[str, object],
    *,
    expected_backend: str,
    expected_family: str | None = None,
    expected_potential_backends: tuple[str, ...] = (),
) -> None:
    """Assert the selected MoE backend and optional oracle metadata."""
    assert_moe_backend_selection_in(
        evidence,
        expected_backends=(expected_backend,),
        expected_family=expected_family,
        expected_potential_backends=expected_potential_backends,
    )


def assert_moe_backend_selection_in(
    evidence: dict[str, object],
    *,
    expected_backends: tuple[str, ...],
    expected_family: str | None = None,
    expected_potential_backends: tuple[str, ...] = (),
) -> None:
    """Assert the selected MoE backend is one of the expected oracle choices."""
    assert_moe_backend_evidence(evidence)
    actual_backend = evidence.get("selected_moe_backend")
    assert actual_backend in expected_backends, (
        f"Expected one of vLLM MoE backends {expected_backends!r}, got {actual_backend!r}. "
        f"Full backend evidence: {evidence}"
    )

    if expected_family is not None:
        actual_family = evidence.get("selected_moe_backend_family")
        assert actual_family == expected_family, (
            f"Expected vLLM MoE backend family {expected_family!r}, got {actual_family!r}. "
            f"Full backend evidence: {evidence}"
        )

    potential_backends = evidence.get("potential_moe_backends") or []
    missing = [backend for backend in expected_potential_backends if backend not in potential_backends]
    assert not missing, (
        f"Expected potential MoE backends {missing!r} in vLLM oracle evidence. Full backend evidence: {evidence}"
    )


def assert_moe_backend_family(evidence: dict[str, object], *, expected_family: str) -> None:
    """Assert a quantized MoE run recorded the expected oracle family."""
    assert_moe_backend_evidence(evidence)
    actual_family = evidence.get("selected_moe_backend_family")
    assert actual_family == expected_family, (
        f"Expected vLLM MoE backend family {expected_family!r}, got {actual_family!r}. "
        f"Full backend evidence: {evidence}"
    )


def assert_rejected_moe_backend_reason(
    evidence: dict[str, object],
    *,
    backend: str,
    reason_contains: str,
) -> None:
    """Assert vLLM logged an oracle rejection reason for a MoE backend."""
    rejections = evidence.get("rejected_moe_backends")
    assert isinstance(rejections, list), f"Backend evidence did not contain rejection records: {evidence}"

    for rejection in rejections:
        if not isinstance(rejection, dict):
            continue
        reason = str(rejection.get("reason", ""))
        if rejection.get("backend") == backend and reason_contains in reason:
            return

    raise AssertionError(
        f"Expected rejection for MoE backend {backend!r} containing {reason_contains!r}. "
        f"Full backend evidence: {evidence}"
    )


def assert_no_triton_cpu_pointer_failure(log_lines: list[str]) -> None:
    """Assert the old Triton CPU-pointer failure did not occur."""
    log_text = "\n".join(log_lines)
    failure = re.search(r"Pointer argument .* cannot be accessed from Triton", log_text)
    assert failure is None, f"Detected Triton CPU-pointer failure in vLLM logs: {failure.group(0)}"


def _parse_int_field(value: str, field_name: str) -> int:
    value = value.strip()
    if not value or value.upper() == "N/A":
        raise ValueError(f"nvidia-smi did not report {field_name}: {value!r}")
    return int(value)


def parse_nvidia_smi_pcie_query(
    output: str,
    *,
    device_index: int = 0,
    device_uuid: str | None = None,
) -> CpuGpuBusInfo:
    """Parse ``nvidia-smi --query-gpu`` PCIe output for one GPU."""
    for row in csv.reader(StringIO(output.strip())):
        if not row:
            continue
        fields = [field.strip() for field in row]
        if len(fields) < 5:
            continue
        try:
            row_index = int(fields[0])
        except ValueError:
            continue

        has_uuid = len(fields) >= 8 and not re.match(r"^[0-9A-Fa-f]{8}:", fields[2])
        row_uuid = fields[2] if has_uuid else None
        if device_uuid is not None:
            if row_uuid != device_uuid:
                continue
        elif row_index != device_index:
            continue

        pci_bus_idx = 3 if has_uuid else 2
        pcie_current_gen_idx = pci_bus_idx + 1
        pcie_current_width_idx = pci_bus_idx + 2
        pcie_max_gen_idx = pci_bus_idx + 3
        pcie_max_width_idx = pci_bus_idx + 4

        pcie_link_gen_current = _parse_int_field(fields[pcie_current_gen_idx], "current PCIe link generation")
        pcie_link_width_current = _parse_int_field(fields[pcie_current_width_idx], "current PCIe link width")
        pcie_link_gen = (
            _parse_int_field(fields[pcie_max_gen_idx], "maximum PCIe link generation")
            if len(fields) > pcie_max_gen_idx
            else pcie_link_gen_current
        )
        pcie_link_width = (
            _parse_int_field(fields[pcie_max_width_idx], "maximum PCIe link width")
            if len(fields) > pcie_max_width_idx
            else pcie_link_width_current
        )
        return CpuGpuBusInfo(
            device_index=row_index,
            name=fields[1],
            pci_bus_id=fields[pci_bus_idx],
            pcie_link_gen=pcie_link_gen,
            pcie_link_width=pcie_link_width,
            pcie_link_gen_current=pcie_link_gen_current,
            pcie_link_width_current=pcie_link_width_current,
        )

    selector = f"GPU UUID {device_uuid}" if device_uuid is not None else f"GPU index {device_index}"
    raise ValueError(f"nvidia-smi PCIe query output did not include {selector}:\n{output}")


def _split_nvidia_smi_topo_line(raw_line: str) -> list[str]:
    if "\t" in raw_line:
        return [token.strip() for token in raw_line.strip().split("\t") if token.strip()]
    return raw_line.split()


def _normalize_nvidia_smi_topo_header(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(tokens):
        if idx + 1 < len(tokens) and tokens[idx : idx + 2] == ["CPU", "Affinity"]:
            normalized.append("CPU Affinity")
            idx += 2
        elif idx + 1 < len(tokens) and tokens[idx : idx + 2] == ["NUMA", "Affinity"]:
            normalized.append("NUMA Affinity")
            idx += 2
        elif idx + 2 < len(tokens) and tokens[idx : idx + 3] == ["GPU", "NUMA", "ID"]:
            normalized.append("GPU NUMA ID")
            idx += 3
        else:
            normalized.append(tokens[idx])
            idx += 1
    return normalized


def _topo_row_value(header_tokens: list[str], row_tokens: list[str], column_name: str) -> str | None:
    try:
        header_idx = header_tokens.index(column_name)
    except ValueError:
        return None
    row_idx = header_idx + 1
    return row_tokens[row_idx] if len(row_tokens) > row_idx else None


def parse_nvidia_smi_topo_matrix(output: str, bus: CpuGpuBusInfo) -> CpuGpuBusInfo:
    """Add CPU affinity details from ``nvidia-smi topo -m`` output."""
    header_tokens: list[str] | None = None
    gpu_label = f"GPU{bus.device_index}"

    for raw_line in output.splitlines():
        tokens = _split_nvidia_smi_topo_line(raw_line)
        if not tokens:
            continue
        normalized_header = _normalize_nvidia_smi_topo_header(tokens)
        if "CPU Affinity" in normalized_header and any(token.startswith("GPU") for token in normalized_header):
            header_tokens = normalized_header
            continue
        if tokens[0] != gpu_label or header_tokens is None:
            continue

        cpu_affinity = _topo_row_value(header_tokens, tokens, "CPU Affinity")
        numa_affinity = _topo_row_value(header_tokens, tokens, "NUMA Affinity")
        return CpuGpuBusInfo(
            device_index=bus.device_index,
            name=bus.name,
            pci_bus_id=bus.pci_bus_id,
            pcie_link_gen=bus.pcie_link_gen,
            pcie_link_width=bus.pcie_link_width,
            pcie_link_gen_current=bus.pcie_link_gen_current,
            pcie_link_width_current=bus.pcie_link_width_current,
            cpu_affinity=cpu_affinity,
            numa_affinity=numa_affinity,
        )

    return bus


def _nvidia_smi_device_selector(device_index: int = 0) -> tuple[int, str | None]:
    """Map CUDA-visible local device index to nvidia-smi index when possible."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return device_index, None
    visible = [part.strip() for part in visible_devices.split(",") if part.strip()]
    if len(visible) <= device_index:
        return device_index, None
    selected_device = visible[device_index]
    if selected_device.isdigit():
        return int(selected_device), None
    return device_index, selected_device


def _nvidia_smi_device_index(device_index: int = 0) -> int:
    """Map CUDA-visible local device index to nvidia-smi index when possible."""
    nvidia_smi_index, _ = _nvidia_smi_device_selector(device_index)
    return nvidia_smi_index


def query_cpu_gpu_bus_info(*, device_index: int = 0, timeout_s: int = 15) -> CpuGpuBusInfo:
    """Collect CPU/GPU bus details from subprocess calls in the pytest process."""
    nvidia_smi_index, nvidia_smi_uuid = _nvidia_smi_device_selector(device_index)
    query_cmd = [
        "nvidia-smi",
        (
            "--query-gpu=index,name,uuid,pci.bus_id,pcie.link.gen.gpucurrent,pcie.link.width.current,"
            "pcie.link.gen.max,pcie.link.width.max"
        ),
        "--format=csv,noheader,nounits",
    ]
    query = subprocess.run(  # noqa: S603 - controlled test utility command
        query_cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    bus = parse_nvidia_smi_pcie_query(query.stdout, device_index=nvidia_smi_index, device_uuid=nvidia_smi_uuid)

    topo_cmd = ["nvidia-smi", "topo", "-m"]
    topo = subprocess.run(  # noqa: S603 - controlled test utility command
        topo_cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if topo.returncode == 0:
        bus = parse_nvidia_smi_topo_matrix(topo.stdout, bus)
    return bus


def load_latest_memory_transfer_stats(instrumentation_dir: Path) -> tuple[dict[str, float], Path]:
    """Load memory-transfer stats from the latest FlexTensor instrumentation dump."""
    dumps = sorted(instrumentation_dir.rglob("components.*.json"))
    for dump_path in reversed(dumps):
        with dump_path.open() as f:
            payload = json.load(f)
        stats = payload.get("memory_transfer_stats")
        if isinstance(stats, dict) and stats:
            return {str(size): float(time_ms) for size, time_ms in stats.items()}, dump_path

    raise AssertionError(f"No FlexTensor memory transfer stats found under {instrumentation_dir}")


def load_latest_instrumentation_dump(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Return the newest FlexTensor instrumentation JSON payload from a vLLM smoke run."""
    dumps = sorted((output_dir / "instrumentation").glob("*/components.*.json"))
    assert dumps, f"No FlexTensor instrumentation dump found under {output_dir / 'instrumentation'}"

    dump_path = dumps[-1]
    with dump_path.open() as f:
        payload = json.load(f)
    assert isinstance(payload, dict), f"Instrumentation dump payload is not a JSON object: {dump_path}"
    return dump_path, payload


def _pcie_theoretical_bandwidth_gbps(bus: CpuGpuBusInfo) -> float:
    # Effective unidirectional payload bandwidth per lane, decimal GB/s.
    per_lane_gbps = {
        1: 0.250,
        2: 0.500,
        3: 0.985,
        4: 1.969,
        5: 3.938,
        6: 7.563,
    }
    try:
        return per_lane_gbps[bus.pcie_link_gen] * bus.pcie_link_width
    except KeyError as exc:
        raise AssertionError(f"Unsupported PCIe link generation for transfer validation: {bus.pcie_link_gen}") from exc


def validate_memory_transfer_stats_for_bus(
    memory_transfer_stats: dict[str, float],
    bus: CpuGpuBusInfo,
    *,
    lower_fraction: float = 0.15,
    upper_fraction: float = 1.25,
) -> dict[str, float | int | str | None]:
    """Validate FlexTensor transfer stats against the detected CPU-to-GPU PCIe bus."""
    samples: list[tuple[int, float]] = []
    for size_bytes, time_ms in memory_transfer_stats.items():
        size = int(size_bytes)
        duration_ms = float(time_ms)
        if size > 0 and duration_ms > 0:
            samples.append((size, duration_ms))

    assert samples, "Instrumentation dump is missing FlexTensor memory transfer stats"

    sample_size_bytes, sample_time_ms = max(samples, key=lambda item: item[0])
    observed_gbps = sample_size_bytes / (sample_time_ms / 1000.0) / 1e9
    theoretical_gbps = _pcie_theoretical_bandwidth_gbps(bus)
    min_expected_gbps = theoretical_gbps * lower_fraction
    max_expected_gbps = theoretical_gbps * upper_fraction

    assert min_expected_gbps <= observed_gbps <= max_expected_gbps, (
        "FlexTensor memory transfer bandwidth outside expected CPU-GPU bus range: "
        f"observed={observed_gbps:.2f} GB/s for {sample_size_bytes} bytes in {sample_time_ms:.3f} ms, "
        f"expected={min_expected_gbps:.2f}-{max_expected_gbps:.2f} GB/s, "
        f"bus=PCIe Gen{bus.pcie_link_gen} x{bus.pcie_link_width} ({bus.pci_bus_id}), "
        f"cpu_affinity={bus.cpu_affinity}, numa_affinity={bus.numa_affinity}"
    )

    return {
        "device_index": bus.device_index,
        "gpu_name": bus.name,
        "pci_bus_id": bus.pci_bus_id,
        "pcie_link_gen": bus.pcie_link_gen,
        "pcie_link_width": bus.pcie_link_width,
        "pcie_link_gen_current": bus.pcie_link_gen_current,
        "pcie_link_width_current": bus.pcie_link_width_current,
        "cpu_affinity": bus.cpu_affinity,
        "numa_affinity": bus.numa_affinity,
        "sample_size_bytes": sample_size_bytes,
        "sample_transfer_ms": sample_time_ms,
        "observed_bandwidth_gbps": observed_gbps,
        "theoretical_bandwidth_gbps": theoretical_gbps,
        "min_expected_bandwidth_gbps": min_expected_gbps,
        "max_expected_bandwidth_gbps": max_expected_gbps,
    }


def validate_latest_memory_transfer_stats_for_current_bus(
    instrumentation_dir: Path,
) -> dict[str, float | int | str | None]:
    """Validate latest instrumentation transfer stats against the local GPU bus."""
    bus = query_cpu_gpu_bus_info()
    stats, dump_path = load_latest_memory_transfer_stats(instrumentation_dir)
    summary = validate_memory_transfer_stats_for_bus(stats, bus)
    summary["instrumentation_dump"] = str(dump_path)
    return summary


def parse_layer_duration_stats(log_lines: list[str]) -> dict[str, dict[str, float | int]]:
    """Parse the Layer Duration Statistics table from FlexTensor log output.

    The table is emitted only on the WARNING path of
    ``LayerStatisticsAnalyzer.check_measurement_consistency`` when any layer has
    insufficient samples or high unexplained CV. Consistent-measurement runs
    produce no table — use :func:`parse_block_assignment_layers` for trap
    enumeration when presence must be guaranteed.

    Args:
        log_lines: List of log lines captured from vLLM server

    Returns:
        Dict mapping layer_name to timing stats dict with keys:
        min, max, median, avg, std (all in ms), count (int), and cv — the coefficient
        of variation as a percentage value (e.g., 25.0 means 25%). Note: this differs
        from LayerDurationStatistics.coefficient_of_variation, which is a ratio (e.g., 0.25).
        Returns empty dict if the table is not found in the logs.

    Example:
        >>> stats = parse_layer_duration_stats(log_lines)
        >>> any(".layers." in k or re.search(r"\\.\\d+$", k) for k in stats)
        True
        >>> "LlamaForCausalLM.model" in stats  # coarse trap — bad
        False
    """
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    # Row: LayerName  float float float float float  float%  int
    row_re = re.compile(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)$")
    # vLLM formats records as: "LEVEL MM-DD HH:MM:SS [module/file.py:LINE] <message>"
    vllm_prefix_re = re.compile(
        r"^(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[[^\]]+\]\s*"
    )
    in_table = False
    result: dict[str, dict[str, float | int]] = {}

    for raw_line in log_lines:
        # Strip ANSI codes first
        line = ansi_escape.sub("", raw_line)
        # log_lines stores the raw vLLM subprocess output (the test print adds "[vLLM]" to
        # the screen only). Each line starts with "(ProcessName pid=N) " from vLLM's logger.
        # Strip "[vLLM] " tag (present if line came from a nested test print) then the
        # "(ProcessName pid=N) " prefix that vLLM always emits.
        line = re.sub(r"^\[vLLM\]\s*", "", line)
        line = re.sub(r"^\(\S+ pid=\d+\)\s*", "", line)
        # Strip optional FlexTensor timestamp: [2026-01-01 ...] LEVEL module.py:N:
        line = re.sub(r"^\[\d{4}-\d{2}-\d{2}.*?\]\s*\w+\s+\S+:\s*", "", line)
        # Strip optional vLLM log-record prefix left on records that propagated through
        # the vLLM bridge: "LEVEL MM-DD HH:MM:SS [file.py:LINE] "
        line = vllm_prefix_re.sub("", line).strip()

        if "Layer Duration Statistics" in line:
            in_table = True
            continue

        if in_table:
            m = row_re.match(line)
            if m:
                result[m.group(1)] = {
                    "min": float(m.group(2)),
                    "max": float(m.group(3)),
                    "median": float(m.group(4)),
                    "avg": float(m.group(5)),
                    "std": float(m.group(6)),
                    "cv": float(m.group(7)),
                    "count": int(m.group(8)),
                }
            elif line.startswith(("=", "-", "Layer")):
                continue  # separator or column header
            elif line:
                in_table = False  # non-data line ends the table

    return result


def parse_block_assignment_layers(log_lines: list[str]) -> list[str]:
    """Parse layer labels from the BLOCK ASSIGNMENT table in FlexTensor log output.

    The BLOCK ASSIGNMENT table always emits when ``FT_ENABLE_DIAGNOSTICS=1``
    (unlike Layer Duration Statistics, which only emits on the WARNING path
    for inconsistent measurements), so it is the reliable source for trap
    enumeration regardless of measurement quality.

    Rows have the layer label as the first whitespace-delimited token; the
    header and separator rows are skipped.

    Args:
        log_lines: List of log lines captured from the vLLM server.

    Returns:
        Ordered list of layer labels (e.g. ``["model.layers.0", "model.layers.1", ...]``)
        or ``[]`` if the table is not found.
    """
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    vllm_prefix_re = re.compile(
        r"^(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[[^\]]+\]\s*"
    )
    # Data row: label followed by a size token like "12.34MB" or "1.23GB" or "-".
    row_re = re.compile(r"^(\S+)\s+(?:\d+(?:\.\d+)?\s*[KMGT]?B|-)\s")

    in_table = False
    layers: list[str] = []

    for raw_line in log_lines:
        line = ansi_escape.sub("", raw_line)
        line = re.sub(r"^\[vLLM\]\s*", "", line)
        line = re.sub(r"^\(\S+ pid=\d+\)\s*", "", line)
        line = re.sub(r"^\[\d{4}-\d{2}-\d{2}.*?\]\s*\w+\s+\S+:\s*", "", line)
        line = vllm_prefix_re.sub("", line).strip()

        if line.startswith("BLOCK ASSIGNMENT:"):
            in_table = True
            continue

        if not in_table:
            continue

        if line.startswith(("=", "-", "Layer ", "Total:", "Compute:", "Block Sizes:", "  Block ")):
            continue

        m = row_re.match(line)
        if m:
            layers.append(m.group(1))
        elif line == "":
            continue
        else:
            # Non-row, non-separator content ends the table section.
            in_table = False

    return layers


def load_gpu_memory_snapshots(snapshot_dir: Path) -> list[dict]:
    """Load GPU memory snapshots from a directory written by MemorySnapshotMixin.

    Reads all JSON files matching ``gpu_snapshots_rank*_device*.json`` in the
    given directory and returns the snapshots list from the first (and expected
    only) file found. Suitable for rank-0 single-device validation.

    Args:
        snapshot_dir: Directory passed as FT_VLLM_SNAPSHOT_OUTPUT_DIR.

    Returns:
        List of snapshot dicts, each with ``label``, ``gpu_memory``, and
        ``host_memory`` keys. Returns an empty list if no file is found.

    Example:
        >>> snapshots = load_gpu_memory_snapshots(Path("/tmp/gpu_snapshots"))
        >>> labels = [s["label"] for s in snapshots]
        >>> assert "after_load_model" in labels
    """
    json_files = sorted(snapshot_dir.glob("gpu_snapshots_rank0_device*.json"))
    if not json_files:
        return []
    data = json.loads(json_files[0].read_text())
    return data.get("snapshots", [])


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_LEVEL_LINE_RE = re.compile(
    r"^(?:\(\S+ pid=\d+\)\s*)?"  # optional '(ProcessName pid=N) '
    r"(?:\[\d{4}-\d{2}-\d{2}.*?\]\s*)?"  # optional FT timestamp
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\b"
)


def parse_log_level(line: str) -> str | None:
    """Return the level token from a vLLM-formatted log line, or None if absent.

    Strips ANSI escapes and the optional '[vLLM] ' test prefix, then matches the
    ``(ProcessName pid=N) LEVEL …`` pattern that vLLM uses in its subprocess
    output.

    Args:
        line: Raw log line captured from vLLM subprocess stdout/stderr.

    Returns:
        The level name (e.g. ``'INFO'``) or ``None`` if the line does not carry
        a recognisable level token.
    """
    stripped = _ANSI_RE.sub("", line)
    stripped = re.sub(r"^\[vLLM\]\s*", "", stripped)
    m = _LEVEL_LINE_RE.match(stripped)
    return m.group("level") if m else None

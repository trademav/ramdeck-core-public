"""Fail-closed validation for the intended Tiny/Mac split diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hmac
import re
from typing import Sequence


class SplitDiagnosticError(RuntimeError):
    """Raised when a split diagnostic safety gate fails."""


@dataclass(frozen=True)
class SplitLimits:
    tiny_fit_target_mib: int = 2048
    mac_fit_target_mib: int = 1536
    context_size: int = 2048
    parallel_sequences: int = 1
    total_layers: int = 49
    max_rpc_layers: int = 20
    min_local_layers: int = 29
    max_mac_aggregate_mib: float = 4096.0
    max_total_kv_mib: float = 512.0
    min_tiny_available_mib: int = 12288
    min_mac_available_mib: int = 10240
    min_metal_working_set_mib: int = 12124


@dataclass(frozen=True)
class CapacitySnapshot:
    tiny_available_mib: int
    mac_available_mib: int
    metal_working_set_mib: int
    mac_memory_pressure: str
    existing_tiny_head: bool = False
    existing_mac_rpc: bool = False
    pending_commands: int = 0
    auto_rebalance_enabled: bool = False


@dataclass(frozen=True)
class PlacementReport:
    rpc_endpoint_count: int
    rpc_layers: tuple[int, ...]
    local_layers: tuple[int, ...]
    mac_model_mib: float
    mac_kv_mib: float
    mac_graph_mib: float
    total_kv_mib: float
    context_size: int | None
    parallel_sequences: int | None
    fatal_markers: tuple[str, ...]

    @property
    def mac_aggregate_mib(self) -> float:
        return self.mac_model_mib + self.mac_kv_mib + self.mac_graph_mib

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["mac_aggregate_mib"] = self.mac_aggregate_mib
        return payload


_LAYER_RE = re.compile(r"load_tensors:\s+layer\s+(\d+) assigned to device\s+(RPC0|CPU)\b")
_BUFFER_RE = re.compile(
    r"(?P<device>RPC0(?:\[[^\]]+\])?|CPU(?:_Mapped)?)\s+"
    r"(?P<kind>model|KV|compute) buffer size\s*=\s*(?P<size>\d+(?:\.\d+)?)\s*MiB",
    re.IGNORECASE,
)
_CONTEXT_PATTERNS = (
    re.compile(r"\bn_ctx\s*=\s*(\d+)\b"),
    re.compile(r"\bn_ctx_train\s*=\s*(\d+)\b"),
    re.compile(r"\bcontext(?: size)?\s*=\s*(\d+)\b", re.IGNORECASE),
)
_PARALLEL_RE = re.compile(r"\bn_parallel\s*=\s*(\d+)\b")
_FATAL_SIGNATURES = (
    "out of memory",
    "allocation failed",
    "failed to allocate",
    "remote rpc server crashed",
    "malformed response",
    "recv failed",
    "sigabrt",
    "fatal error",
    "watchdog",
)


def build_tiny_arguments(
    *,
    binary: str,
    model: str,
    rpc_endpoint: str,
    port: int,
    limits: SplitLimits | None = None,
) -> list[str]:
    limits = limits or SplitLimits()
    return [
        binary,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-m", model,
        "--rpc", rpc_endpoint,
        "--fit-target", f"{limits.tiny_fit_target_mib},{limits.mac_fit_target_mib}",
        "--ctx-size", str(limits.context_size),
        "--parallel", str(limits.parallel_sequences),
        "--flash-attn", "off",
        "--no-warmup",
        "-v",
    ]


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    indexes = [index for index, value in enumerate(arguments) if value == option]
    if len(indexes) != 1:
        return None
    index = indexes[0]
    if index + 1 >= len(arguments):
        return None
    return arguments[index + 1]


def validate_tiny_arguments(
    arguments: Sequence[str],
    *,
    rpc_endpoint: str,
    limits: SplitLimits | None = None,
) -> None:
    limits = limits or SplitLimits()
    forbidden = {"--fit", "--tensor-split", "--gpu-layers", "-ngl", "--n-gpu-layers"}
    present_forbidden = sorted(forbidden.intersection(arguments))
    if present_forbidden:
        raise SplitDiagnosticError(f"forbidden Tiny arguments: {', '.join(present_forbidden)}")

    expected = {
        "--rpc": rpc_endpoint,
        "--fit-target": f"{limits.tiny_fit_target_mib},{limits.mac_fit_target_mib}",
        "--ctx-size": str(limits.context_size),
        "--parallel": str(limits.parallel_sequences),
        "--flash-attn": "off",
    }
    for option, expected_value in expected.items():
        actual = _option_value(arguments, option)
        if actual != expected_value:
            raise SplitDiagnosticError(f"{option} must be {expected_value!r}, got {actual!r}")
    if arguments.count("--no-warmup") != 1:
        raise SplitDiagnosticError("Tiny arguments must contain exactly one --no-warmup")


def validate_capacity(snapshot: CapacitySnapshot, limits: SplitLimits | None = None) -> None:
    limits = limits or SplitLimits()
    failures: list[str] = []
    if snapshot.tiny_available_mib < limits.min_tiny_available_mib:
        failures.append(
            f"Tiny available memory {snapshot.tiny_available_mib} MiB is below {limits.min_tiny_available_mib} MiB"
        )
    if snapshot.mac_available_mib < limits.min_mac_available_mib:
        failures.append(
            f"Mac available memory {snapshot.mac_available_mib} MiB is below {limits.min_mac_available_mib} MiB"
        )
    if snapshot.metal_working_set_mib < limits.min_metal_working_set_mib:
        failures.append(
            f"Metal working set {snapshot.metal_working_set_mib} MiB is below {limits.min_metal_working_set_mib} MiB"
        )
    if snapshot.mac_memory_pressure.strip().lower() != "normal":
        failures.append(f"Mac memory pressure is {snapshot.mac_memory_pressure!r}, expected 'normal'")
    if snapshot.existing_tiny_head:
        failures.append("an existing temporary Tiny head was detected")
    if snapshot.existing_mac_rpc:
        failures.append("an existing temporary Mac RPC process was detected")
    if snapshot.pending_commands != 0:
        failures.append(f"coordinator has {snapshot.pending_commands} pending commands")
    if snapshot.auto_rebalance_enabled:
        failures.append("automatic rebalance is enabled")
    if failures:
        raise SplitDiagnosticError("capacity gate failed: " + "; ".join(failures))


def parse_placement_log(text: str, rpc_endpoint: str) -> PlacementReport:
    rpc_layers: set[int] = set()
    local_layers: set[int] = set()
    conflicting_layers: set[int] = set()
    for match in _LAYER_RE.finditer(text):
        layer = int(match.group(1))
        destination = match.group(2)
        target = rpc_layers if destination == "RPC0" else local_layers
        other = local_layers if destination == "RPC0" else rpc_layers
        if layer in other:
            conflicting_layers.add(layer)
        target.add(layer)
    if conflicting_layers:
        layers = ",".join(str(layer) for layer in sorted(conflicting_layers))
        raise SplitDiagnosticError(f"conflicting placement records for layers: {layers}")

    mac_model_mib = 0.0
    mac_kv_mib = 0.0
    mac_graph_mib = 0.0
    total_kv_mib = 0.0
    for match in _BUFFER_RE.finditer(text):
        device = match.group("device")
        kind = match.group("kind").lower()
        size = float(match.group("size"))
        if kind == "kv":
            total_kv_mib += size
        if not device.upper().startswith("RPC0"):
            continue
        if kind == "model":
            mac_model_mib += size
        elif kind == "kv":
            mac_kv_mib += size
        elif kind == "compute":
            mac_graph_mib += size

    context_size = None
    for pattern in _CONTEXT_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            context_size = int(matches[-1])
            break
    parallel_matches = _PARALLEL_RE.findall(text)
    parallel_sequences = int(parallel_matches[-1]) if parallel_matches else None
    fatal_markers = tuple(signature for signature in _FATAL_SIGNATURES if signature in text.lower())
    endpoint_pattern = re.compile(rf"RPC0\s*:\s*{re.escape(rpc_endpoint)}\b")

    return PlacementReport(
        rpc_endpoint_count=len(endpoint_pattern.findall(text)),
        rpc_layers=tuple(sorted(rpc_layers)),
        local_layers=tuple(sorted(local_layers)),
        mac_model_mib=mac_model_mib,
        mac_kv_mib=mac_kv_mib,
        mac_graph_mib=mac_graph_mib,
        total_kv_mib=total_kv_mib,
        context_size=context_size,
        parallel_sequences=parallel_sequences,
        fatal_markers=fatal_markers,
    )


def validate_placement(report: PlacementReport, limits: SplitLimits | None = None) -> None:
    limits = limits or SplitLimits()
    failures: list[str] = []
    rpc_count = len(report.rpc_layers)
    local_count = len(report.local_layers)
    placed_layers = set(report.rpc_layers) | set(report.local_layers)
    expected_layers = set(range(limits.total_layers))

    if report.rpc_endpoint_count != 1:
        failures.append(f"RPC0 endpoint appeared {report.rpc_endpoint_count} times, expected once")
    if not 1 <= rpc_count <= limits.max_rpc_layers:
        failures.append(f"RPC0 layer count {rpc_count} is outside 1..{limits.max_rpc_layers}")
    if not limits.min_local_layers <= local_count < limits.total_layers:
        failures.append(
            f"Tiny-local layer count {local_count} is outside {limits.min_local_layers}..{limits.total_layers - 1}"
        )
    if placed_layers != expected_layers:
        missing = sorted(expected_layers - placed_layers)
        extra = sorted(placed_layers - expected_layers)
        failures.append(f"layer set mismatch: missing={missing}, extra={extra}")
    if report.mac_model_mib <= 0 or report.mac_kv_mib <= 0 or report.mac_graph_mib <= 0:
        failures.append("Mac model, KV, and compute/graph buffer measurements are all required")
    if report.mac_aggregate_mib > limits.max_mac_aggregate_mib:
        failures.append(
            f"Mac aggregate {report.mac_aggregate_mib:.2f} MiB exceeds {limits.max_mac_aggregate_mib:.2f} MiB"
        )
    if report.total_kv_mib <= 0 or report.total_kv_mib > limits.max_total_kv_mib:
        failures.append(f"total KV {report.total_kv_mib:.2f} MiB is outside 0..{limits.max_total_kv_mib:.2f} MiB")
    if report.context_size != limits.context_size:
        failures.append(f"context {report.context_size!r} does not equal {limits.context_size}")
    if report.parallel_sequences != limits.parallel_sequences:
        failures.append(f"parallel sequences {report.parallel_sequences!r} does not equal {limits.parallel_sequences}")
    if report.fatal_markers:
        failures.append(f"fatal log markers present: {', '.join(report.fatal_markers)}")
    if failures:
        raise SplitDiagnosticError("placement gate failed: " + "; ".join(failures))


def require_execution_approval(*, execute: bool, supplied_token: str, expected_token: str) -> None:
    if not execute:
        raise SplitDiagnosticError("execution is disabled; pass --execute only after separate approval")
    if not expected_token:
        raise SplitDiagnosticError("RAMDECK_SPLIT_APPROVAL is not configured")
    if not supplied_token or not hmac.compare_digest(supplied_token, expected_token):
        raise SplitDiagnosticError("approval token mismatch")
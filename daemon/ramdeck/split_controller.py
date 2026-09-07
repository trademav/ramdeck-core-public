"""Two-stage controller for the intended Tiny-primary/Mac-worker diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Protocol

from .split_diagnostic import (
    CapacitySnapshot,
    PlacementReport,
    SplitDiagnosticError,
    SplitLimits,
    build_tiny_arguments,
    parse_placement_log,
    require_execution_approval,
    validate_capacity,
    validate_placement,
    validate_tiny_arguments,
)
from .split_lifecycle import CleanupReceipt, ProcessIdentity


@dataclass(frozen=True)
class SplitRunConfig:
    tiny_binary: str
    tiny_model: str
    tiny_port: int
    tiny_ssh_host: str
    tiny_python: str
    tiny_stdout_log: str
    tiny_stderr_log: str
    tiny_binary_sha256: str
    mac_rpc_binary: str
    mac_rpc_port: int
    mac_stdout_log: str
    mac_stderr_log: str
    mac_binary_sha256: str
    output_dir: str
    model_id: str = "Qwen2.5-14B-Instruct-Q4_K_M"
    placement_timeout_sec: float = 120.0
    request_timeout_sec: float = 60.0

    @property
    def rpc_endpoint(self) -> str:
        return f"127.0.0.1:{self.mac_rpc_port}"

    def fingerprint(self, limits: SplitLimits) -> str:
        payload = {
            "config": asdict(self),
            "limits": asdict(limits),
            "rpc_endpoint": self.rpc_endpoint,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class LocalOwner(Protocol):
    def start(self) -> ProcessIdentity: ...
    def stop(self, graceful_timeout: float = 5.0, kill_timeout: float = 5.0) -> CleanupReceipt: ...


class RemoteOwner(Protocol):
    def start(self, timeout: float = 10.0) -> ProcessIdentity: ...
    def stop(self, timeout: float = 15.0) -> CleanupReceipt: ...
    def read_event(self, timeout: float) -> tuple[str, Any]: ...
    def request_once(self, *, url: str, body: dict[str, Any], timeout: float) -> None: ...


class SplitDiagnosticController:
    def __init__(
        self,
        config: SplitRunConfig,
        *,
        capacity_probe: Callable[[], CapacitySnapshot],
        local_owner_factory: Callable[[], LocalOwner],
        remote_owner_factory: Callable[[list[str]], RemoteOwner],
        wait_mac_ready: Callable[[float], None],
        limits: SplitLimits | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.capacity_probe = capacity_probe
        self.local_owner_factory = local_owner_factory
        self.remote_owner_factory = remote_owner_factory
        self.wait_mac_ready = wait_mac_ready
        self.limits = limits or SplitLimits()
        self.monotonic = monotonic
        self.wall_time = wall_time

    def render_tiny_arguments(self) -> list[str]:
        arguments = build_tiny_arguments(
            binary=self.config.tiny_binary,
            model=self.config.tiny_model,
            rpc_endpoint=self.config.rpc_endpoint,
            port=self.config.tiny_port,
            limits=self.limits,
        )
        validate_tiny_arguments(arguments, rpc_endpoint=self.config.rpc_endpoint, limits=self.limits)
        return arguments

    def run(
        self,
        *,
        stage: str,
        execute: bool,
        supplied_approval_token: str,
        expected_approval_token: str,
        stage1_report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if stage not in {"placement", "sample"}:
            raise SplitDiagnosticError(f"unsupported stage {stage!r}")
        require_execution_approval(
            execute=execute,
            supplied_token=supplied_approval_token,
            expected_token=expected_approval_token,
        )
        arguments = self.render_tiny_arguments()
        fingerprint = self.config.fingerprint(self.limits)
        if stage == "sample":
            self._validate_stage1_report(stage1_report_path, fingerprint)

        capacity = self.capacity_probe()
        validate_capacity(capacity, self.limits)
        started_at = self.wall_time()
        report: dict[str, Any] = {
            "schema_version": 1,
            "stage": stage,
            "passed": False,
            "started_at": started_at,
            "config_fingerprint": fingerprint,
            "tiny_arguments": arguments,
            "capacity": asdict(capacity),
            "request_count": 0,
        }
        local_owner = self.local_owner_factory()
        remote_owner = self.remote_owner_factory(arguments)
        local_started = False
        remote_started = False
        primary_error: Exception | None = None
        placement: PlacementReport | None = None
        try:
            report["mac_identity"] = asdict(local_owner.start())
            local_started = True
            self.wait_mac_ready(10.0)
            report["tiny_identity"] = asdict(remote_owner.start())
            remote_started = True
            placement = self._wait_for_placement(remote_owner)
            report["placement"] = placement.to_dict()
            if stage == "sample":
                response = self._send_one_request(remote_owner)
                report["request_count"] = 1
                report["response"] = response
            report["passed"] = True
        except Exception as exc:
            primary_error = exc
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup_errors: list[str] = []
            if remote_started:
                try:
                    report["tiny_cleanup"] = asdict(remote_owner.stop())
                except Exception as exc:
                    cleanup_errors.append(f"Tiny cleanup: {type(exc).__name__}: {exc}")
            if local_started:
                try:
                    report["mac_cleanup"] = asdict(local_owner.stop())
                except Exception as exc:
                    cleanup_errors.append(f"Mac cleanup: {type(exc).__name__}: {exc}")
            if cleanup_errors:
                report["passed"] = False
                report["cleanup_errors"] = cleanup_errors
            report["finished_at"] = self.wall_time()
            report_path = self._write_report(report)
            report["report_path"] = str(report_path)

        if report.get("cleanup_errors"):
            raise SplitDiagnosticError("; ".join(report["cleanup_errors"]))
        if primary_error is not None:
            raise primary_error
        return report

    def _wait_for_placement(self, remote_owner: RemoteOwner) -> PlacementReport:
        deadline = self.monotonic() + self.config.placement_timeout_sec
        lines: list[str] = []
        while self.monotonic() < deadline:
            event, payload = remote_owner.read_event(max(0.1, deadline - self.monotonic()))
            if event != "log":
                if event == "response":
                    raise SplitDiagnosticError("request response arrived before the placement gate")
                continue
            lines.append(str(payload))
            report = parse_placement_log("\n".join(lines), self.config.rpc_endpoint)
            self._reject_early(report)
            try:
                validate_placement(report, self.limits)
                return report
            except SplitDiagnosticError:
                continue
        raise SplitDiagnosticError("placement gate timed out before all required measurements arrived")

    def _reject_early(self, report: PlacementReport) -> None:
        placed_count = len(set(report.rpc_layers) | set(report.local_layers))
        rpc_count = len(report.rpc_layers)
        if report.fatal_markers:
            raise SplitDiagnosticError(f"fatal log markers present: {', '.join(report.fatal_markers)}")
        if report.mac_aggregate_mib > self.limits.max_mac_aggregate_mib:
            raise SplitDiagnosticError(
                f"Mac aggregate {report.mac_aggregate_mib:.2f} MiB exceeds {self.limits.max_mac_aggregate_mib:.2f} MiB"
            )
        if rpc_count > self.limits.max_rpc_layers:
            raise SplitDiagnosticError(
                f"RPC0 layer count {rpc_count} is outside 1..{self.limits.max_rpc_layers}"
            )
        if placed_count == self.limits.total_layers and rpc_count == self.limits.total_layers:
            raise SplitDiagnosticError(
                f"RPC0 layer count {rpc_count} is outside 1..{self.limits.max_rpc_layers}"
            )

    def _send_one_request(self, remote_owner: RemoteOwner) -> dict[str, Any]:
        remote_owner.request_once(
            url=f"http://127.0.0.1:{self.config.tiny_port}/v1/chat/completions",
            body={
                "model": self.config.model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": False,
            },
            timeout=self.config.request_timeout_sec,
        )
        deadline = self.monotonic() + self.config.request_timeout_sec + 5.0
        while self.monotonic() < deadline:
            event, payload = remote_owner.read_event(max(0.1, deadline - self.monotonic()))
            if event != "response":
                continue
            if payload.get("request_count") != 1:
                raise SplitDiagnosticError("remote helper did not confirm exactly one request")
            if payload.get("status") != 200:
                raise SplitDiagnosticError(f"sample request failed: {payload}")
            try:
                body = json.loads(payload.get("body") or "")
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise SplitDiagnosticError("sample response did not contain chat content") from exc
            if not isinstance(content, str) or not content.strip():
                raise SplitDiagnosticError("sample response content was empty")
            return {"status": 200, "content": content, "request_count": 1}
        raise SplitDiagnosticError("sample request timed out")

    def _validate_stage1_report(self, path: str | Path | None, fingerprint: str) -> None:
        if path is None:
            raise SplitDiagnosticError("sample stage requires --stage1-report")
        report_path = Path(path)
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitDiagnosticError(f"could not read Stage 1 report: {exc}") from exc
        if report.get("stage") != "placement" or report.get("passed") is not True:
            raise SplitDiagnosticError("Stage 1 report is not a passing placement report")
        if report.get("config_fingerprint") != fingerprint:
            raise SplitDiagnosticError("Stage 1 report configuration does not match this run")
        if not report.get("tiny_cleanup", {}).get("identity_absent"):
            raise SplitDiagnosticError("Stage 1 report does not prove Tiny cleanup")
        if not report.get("mac_cleanup", {}).get("identity_absent"):
            raise SplitDiagnosticError("Stage 1 report does not prove Mac cleanup")

    def _write_report(self, report: dict[str, Any]) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(report["started_at"] * 1000)
        path = output_dir / f"{report['stage']}-{timestamp}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path
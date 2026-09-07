"""Supervise the llama-server process owned by a RAMDeck node agent."""

from __future__ import annotations

import copy
import hashlib
import getpass
from pathlib import Path
import os
import socket
import subprocess
import time
from typing import Any, Callable, TextIO

import httpx

try:
    import psutil
except Exception:
    psutil = None


REASON_MODEL_MISSING = "model_missing"
REASON_PROCESS_EXITED = "process_exited"
REASON_PORT_BIND_FAILED = "port_bind_failed"
REASON_RPC_PEER_UNREACHABLE = "rpc_peer_unreachable"
REASON_CUDA_OR_BACKEND_INIT_FAILED = "cuda_or_backend_init_failed"
REASON_MODEL_LOAD_FAILED = "model_load_failed"
REASON_MEMORY_OR_KV_CACHE_FAILED = "memory_or_kv_cache_failed"
REASON_TIMEOUT_UNKNOWN = "readiness_timeout_unknown"
REASON_STANDALONE_FALLBACK_INFEASIBLE = "standalone_fallback_infeasible"

LOG_TAIL_MAX_CHARS = 12000
LOG_TAIL_MAX_LINES = 120

RPC_FALLBACK_ENV = "RAMDECK_RPC_STANDALONE_FALLBACK"
RPC_FALLBACK_SAFETY_MARGIN_ENV = "RAMDECK_RPC_FALLBACK_SAFETY_MARGIN"
DEFAULT_RPC_FALLBACK_SAFETY_MARGIN = 0.85
PORT_RELEASE_TIMEOUT_SEC = 15.0


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _tail_text_file(path: str | None, max_lines: int = LOG_TAIL_MAX_LINES, max_chars: int = LOG_TAIL_MAX_CHARS) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.is_file():
        return []
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    lines = text.splitlines()[-max_lines:]
    if not lines:
        return []
    total = "\n".join(lines)
    if len(total) > max_chars:
        total = total[-max_chars:]
        lines = total.splitlines()
    return lines


def _classify_terminal_reason(*, log_lines: list[str], exited: bool, model_missing: bool) -> str:
    if model_missing:
        return REASON_MODEL_MISSING

    lowered = "\n".join(log_lines).lower()

    if "address already in use" in lowered or ("bind" in lowered and "fail" in lowered):
        return REASON_PORT_BIND_FAILED
    if (
        "rpc" in lowered
        and (
            "unreachable" in lowered
            or "connection refused" in lowered
            or "timed out" in lowered
            or "send failed" in lowered
            or "remote rpc server crashed" in lowered
        )
    ):
        return REASON_RPC_PEER_UNREACHABLE
    if (
        "cuda" in lowered
        or "cublas" in lowered
        or "hipblas" in lowered
        or "vulkan" in lowered
        or "metal" in lowered
    ) and ("fail" in lowered or "error" in lowered or "not found" in lowered):
        return REASON_CUDA_OR_BACKEND_INIT_FAILED
    if (
        "model load" in lowered
        or "load_model" in lowered
        or "llama_model_load" in lowered
    ) and ("fail" in lowered or "error" in lowered):
        return REASON_MODEL_LOAD_FAILED
    if "out of memory" in lowered or "kv cache" in lowered or "insufficient" in lowered:
        return REASON_MEMORY_OR_KV_CACHE_FAILED
    if exited:
        return REASON_PROCESS_EXITED
    return REASON_TIMEOUT_UNKNOWN


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _parse_safety_margin(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0.0:
        return default
    if value > 1.0:
        return 1.0
    return value


class EngineSupervisor:
    def __init__(
        self,
        binary: Path,
        log_handle: TextIO | None = None,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        health_timeout_sec: float = 180.0,
    ):
        self.binary = binary
        self.log_handle = log_handle
        self.popen_factory = popen_factory
        self.health_timeout_sec = health_timeout_sec
        self.process: subprocess.Popen | None = None
        self.loaded_model: str | None = None
        self.port = 8080
        self.current_launch_id: str | None = None
        self.last_launch_diagnostic: dict[str, Any] | None = None

    def report(self) -> dict[str, Any]:
        running = self.process is not None and self.process.poll() is None
        return {
            "engine_state": "primary" if running else "idle_worker",
            "loaded_model": self.loaded_model if running else None,
            "launch_id": self.current_launch_id if running else None,
        }

    def execute(self, command: dict[str, Any]) -> dict[str, Any | None]:
        command_id = str(command.get("command_id") or "")
        launch_diag: dict[str, Any] | None = None
        try:
            action = command.get("action")
            if action == "start":
                launch_diag = self.start(
                    model_path=str(command["model_path"]),
                    port=int(command.get("port", 8080)),
                    rpc_peers=[str(peer) for peer in command.get("rpc_peers") or []],
                    fit_target_mib=[int(value) for value in command.get("fit_target_mib") or []],
                    ctx_size=int(command["ctx_size"]) if command.get("ctx_size") is not None else None,
                    parallel=int(command["parallel"]) if command.get("parallel") is not None else None,
                    gpu_layers=int(command["gpu_layers"]) if command.get("gpu_layers") is not None else None,
                    threads=int(command["threads"]) if command.get("threads") is not None else None,
                    batch_threads=int(command["batch_threads"]) if command.get("batch_threads") is not None else None,
                    batch_size=int(command["batch_size"]) if command.get("batch_size") is not None else None,
                    ubatch_size=int(command["ubatch_size"]) if command.get("ubatch_size") is not None else None,
                    timeout_sec=float(command["timeout"]) if command.get("timeout") is not None else None,
                )
            elif action == "stop":
                self.stop()
            else:
                raise ValueError(f"unsupported engine command: {action}")
            response: dict[str, Any | None] = {
                "command_id": command_id,
                "command_status": "ok",
                "command_error": None,
            }
            if launch_diag is not None:
                response.update(
                    {
                        "command_result_reason": "ready",
                        "command_result_launch_id": launch_diag.get("launch_id"),
                        "command_result_phase": launch_diag.get("phase"),
                        "command_result_elapsed_sec": launch_diag.get("elapsed_sec"),
                        "command_result_diagnostics": launch_diag,
                    }
                )
            return response
        except Exception as exc:
            failure_diag = self.last_launch_diagnostic or {}
            return {
                "command_id": command_id,
                "command_status": "error",
                "command_error": str(exc),
                "command_result_reason": failure_diag.get("terminal_reason") or REASON_TIMEOUT_UNKNOWN,
                "command_result_launch_id": failure_diag.get("launch_id"),
                "command_result_phase": failure_diag.get("phase"),
                "command_result_elapsed_sec": failure_diag.get("elapsed_sec"),
                "command_result_diagnostics": failure_diag or None,
            }

    def start(
        self,
        model_path: str,
        port: int,
        rpc_peers: list[str],
        fit_target_mib: list[int] | None = None,
        ctx_size: int | None = None,
        parallel: int | None = None,
        gpu_layers: int | None = None,
        threads: int | None = None,
        batch_threads: int | None = None,
        batch_size: int | None = None,
        ubatch_size: int | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        model = Path(model_path)
        launch_id = f"launch-{int(time.time() * 1000)}"
        model_exists = model.is_file()
        model_readable = os.access(model, os.R_OK) if model_exists else False
        requested_fit_target_mib = list(fit_target_mib or [])
        fallback_enabled = bool(rpc_peers) and _env_flag_enabled(RPC_FALLBACK_ENV, default=True)
        fallback_safety_margin = _parse_safety_margin(
            os.environ.get(RPC_FALLBACK_SAFETY_MARGIN_ENV),
            DEFAULT_RPC_FALLBACK_SAFETY_MARGIN,
        )

        launch_diag = {
            "launch_id": launch_id,
            "phase": "preflight",
            "requested_role": "primary",
            "started_at": time.time(),
            "node_role_hint": "head",
            "resolved_model_path": str(model.resolve(strict=False)),
            "model_exists": model_exists,
            "model_readable": model_readable,
            "rpc_peers": list(rpc_peers),
            "fit_target_mib": requested_fit_target_mib,
            "ctx_size": ctx_size,
            "parallel": parallel,
            "gpu_layers": gpu_layers,
            "threads": threads,
            "batch_threads": batch_threads,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "port": int(port),
            "health_timeout_sec": float(self.health_timeout_sec),
            "fallback_policy": "retry_without_rpc_once" if fallback_enabled else "disabled",
            "service_identity": self._service_identity(),
            "environment": self._environment_fingerprint(),
        }

        if not model_exists or not model_readable:
            launch_diag.update(
                {
                    "phase": "failed_preflight",
                    "terminal_reason": REASON_MODEL_MISSING,
                    "terminal_detail": f"model not found or unreadable: {model}",
                    "elapsed_sec": 0.0,
                    "log_tail": [],
                }
            )
            self.last_launch_diagnostic = launch_diag
            raise FileNotFoundError(f"model not found: {model}")

        if not self.binary.is_file():
            raise FileNotFoundError(f"llama-server not found: {self.binary}")

        self.stop()
        self._wait_for_port_free(port, timeout_sec=PORT_RELEASE_TIMEOUT_SEC)
        launch_diag["executable_path"] = str(self.binary.resolve(strict=False))
        launch_diag["executable_sha256"] = _sha256_file(self.binary)
        launch_diag["port_release_timeout_sec"] = float(PORT_RELEASE_TIMEOUT_SEC)
        try:
            self._launch_once(
                model=model,
                port=port,
                rpc_peers=rpc_peers,
                fit_target_mib=requested_fit_target_mib,
                ctx_size=ctx_size,
                parallel=parallel,
                gpu_layers=gpu_layers,
                threads=threads,
                batch_threads=batch_threads,
                batch_size=batch_size,
                ubatch_size=ubatch_size,
                launch_diag=launch_diag,
                attempt="rpc_or_local",
            )
            self._wait_until_healthy(launch_diag, timeout_sec=timeout_sec)
            self.last_launch_diagnostic = launch_diag
            return launch_diag
        except Exception as exc:
            failure_snapshot = copy.deepcopy(self.last_launch_diagnostic or launch_diag)
            failure_snapshot.setdefault("phase", "failed")
            failure_snapshot.setdefault("terminal_reason", REASON_TIMEOUT_UNKNOWN)
            failure_snapshot.setdefault("terminal_detail", str(exc))
            failure_snapshot.setdefault(
                "elapsed_sec",
                round(time.time() - float(failure_snapshot.get("started_at", time.time())), 3),
            )
            failure_snapshot.setdefault("log_tail", self._log_tail())
            self.last_launch_diagnostic = failure_snapshot
            failure_reason = str(failure_snapshot.get("terminal_reason") or "")
            should_fallback = fallback_enabled and failure_reason in {
                REASON_RPC_PEER_UNREACHABLE,
                REASON_TIMEOUT_UNKNOWN,
                REASON_PROCESS_EXITED,
            }
            if not should_fallback:
                raise

            total_model_footprint_mib = sum(max(0, int(value)) for value in requested_fit_target_mib)
            primary_available_mib = self._live_available_capacity_mib()
            launch_diag["fallback_applied"] = True
            launch_diag["fallback_reason"] = failure_reason or REASON_TIMEOUT_UNKNOWN
            launch_diag["fallback_from"] = failure_snapshot
            launch_diag["fallback_message"] = "RPC launch failed readiness checks; retrying without RPC peers"
            launch_diag["fallback_requested_total_fit_target_mib"] = total_model_footprint_mib
            launch_diag["fallback_primary_available_mib"] = primary_available_mib
            launch_diag["fallback_safety_margin"] = fallback_safety_margin

            if (
                total_model_footprint_mib > 0
                and primary_available_mib is not None
                and total_model_footprint_mib > int(primary_available_mib * fallback_safety_margin)
            ):
                detail = (
                    f"Peer failure ({failure_reason or REASON_TIMEOUT_UNKNOWN}) would require standalone "
                    f"load of {total_model_footprint_mib} MiB, but primary only has "
                    f"{primary_available_mib} MiB available (safety margin={fallback_safety_margin:.2f}). "
                    "Aborting instead of attempting an infeasible standalone load."
                )
                launch_diag.update(
                    {
                        "phase": "failed",
                        "terminal_reason": REASON_STANDALONE_FALLBACK_INFEASIBLE,
                        "terminal_detail": detail,
                        "elapsed_sec": round(
                            time.time() - float(launch_diag.get("started_at", time.time())),
                            3,
                        ),
                        "log_tail": self._log_tail(),
                    }
                )
                self.last_launch_diagnostic = launch_diag
                raise RuntimeError(detail)

            self.stop()

            for reduced_peers, reduced_fit_target in self._reduced_topology_candidates(
                rpc_peers,
                requested_fit_target_mib,
                failure_snapshot,
            ):
                try:
                    self._launch_once(
                        model=model,
                        port=port,
                        rpc_peers=reduced_peers,
                        fit_target_mib=reduced_fit_target,
                        ctx_size=ctx_size,
                        parallel=parallel,
                        gpu_layers=gpu_layers,
                        threads=threads,
                        batch_threads=batch_threads,
                        batch_size=batch_size,
                        ubatch_size=ubatch_size,
                        launch_diag=launch_diag,
                        attempt="reduced_topology_fallback",
                    )
                    self.last_launch_diagnostic = launch_diag
                    return launch_diag
                except Exception:
                    failure_snapshot = copy.deepcopy(self.last_launch_diagnostic or launch_diag)
                    launch_diag["fallback_from"] = failure_snapshot

            fallback_fit_target_mib = [total_model_footprint_mib] if total_model_footprint_mib > 0 else []

            self._launch_once(
                model=model,
                port=port,
                rpc_peers=[],
                fit_target_mib=fallback_fit_target_mib,
                ctx_size=ctx_size,
                parallel=parallel,
                gpu_layers=gpu_layers,
                threads=threads,
                batch_threads=batch_threads,
                batch_size=batch_size,
                ubatch_size=ubatch_size,
                launch_diag=launch_diag,
                attempt="standalone_fallback",
            )
            self.last_launch_diagnostic = launch_diag
            return launch_diag

    def _launch_once(
        self,
        *,
        model: Path,
        port: int,
        rpc_peers: list[str],
        fit_target_mib: list[int],
        ctx_size: int | None,
        parallel: int | None,
        gpu_layers: int | None = None,
        threads: int | None = None,
        batch_threads: int | None = None,
        batch_size: int | None = None,
        ubatch_size: int | None = None,
        launch_diag: dict[str, Any] = None,
        attempt: str,
    ) -> None:
        command = [
            str(self.binary),
            "--host", "0.0.0.0",
            "--port", str(port),
            "-m", str(model),
        ]
        if rpc_peers:
            command.extend(["--rpc", ",".join(rpc_peers)])
        if fit_target_mib:
            command.extend(["--fit-target", ",".join(str(value) for value in fit_target_mib)])
        if ctx_size is not None:
            command.extend(["--ctx-size", str(ctx_size)])
        if parallel is not None:
            command.extend(["--parallel", str(parallel)])
        if gpu_layers is not None:
            command.extend(["-ngl", str(gpu_layers)])
        if threads is not None:
            command.extend(["-t", str(threads)])
        if batch_threads is not None:
            command.extend(["-tb", str(batch_threads)])
        if batch_size is not None:
            command.extend(["--batch-size", str(batch_size)])
        if ubatch_size is not None:
            command.extend(["--ubatch-size", str(ubatch_size)])

        launch_diag["attempt"] = attempt
        launch_diag["rpc_peers_effective"] = list(rpc_peers)
        launch_diag["fit_target_mib_effective"] = list(fit_target_mib)
        launch_diag["command"] = list(command)

        env = os.environ.copy()
        env["GGML_RPC_NO_COMM"] = "1"
        kwargs: dict[str, Any] = {
            "cwd": str(self.binary.parent),
            "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            "env": env,
        }
        if self.log_handle is not None:
            kwargs["stdout"] = self.log_handle
            kwargs["stderr"] = subprocess.STDOUT
        launch_diag["working_directory"] = kwargs.get("cwd")

        self.process = self.popen_factory(command, **kwargs)
        self.loaded_model = str(model)
        self.port = port
        self.current_launch_id = str(launch_diag.get("launch_id") or "") or None
        launch_diag["pid"] = int(self.process.pid)
        launch_diag["parent_pid"] = int(os.getpid())
        self._wait_until_healthy(launch_diag)

    def stop(self) -> None:
        process = self.process
        active_port = self.port
        self.process = None
        self.loaded_model = None
        self.current_launch_id = None
        if process is None or process.poll() is not None:
            return
        self._terminate_process_tree(process)
        self._wait_for_port_free(active_port, timeout_sec=PORT_RELEASE_TIMEOUT_SEC)

    def _wait_until_healthy(self, launch_diag: dict[str, Any], timeout_sec: float | None = None) -> None:
        effective_timeout = timeout_sec if timeout_sec is not None else self.health_timeout_sec
        deadline = time.monotonic() + effective_timeout
        url = f"http://127.0.0.1:{self.port}/health"
        last_health_status: int | None = None
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                log_tail = self._log_tail()
                reason = _classify_terminal_reason(log_lines=log_tail, exited=True, model_missing=False)
                elapsed = round(time.time() - float(launch_diag.get("started_at", time.time())), 3)
                launch_diag.update(
                    {
                        "phase": "failed",
                        "elapsed_sec": elapsed,
                        "last_health_status": last_health_status,
                        "terminal_reason": reason,
                        "terminal_detail": "llama-server exited before becoming healthy",
                        "log_tail": log_tail,
                    }
                )
                self.last_launch_diagnostic = launch_diag
                raise RuntimeError("llama-server exited before becoming healthy")
            try:
                response = httpx.get(url, timeout=1.0)
                last_health_status = int(response.status_code)
                if response.status_code == 200 and response.json().get("status") == "ok":
                    elapsed = round(time.time() - float(launch_diag.get("started_at", time.time())), 3)
                    launch_diag.update(
                        {
                            "phase": "ready",
                            "elapsed_sec": elapsed,
                            "last_health_status": last_health_status,
                            "terminal_reason": "ready",
                            "terminal_detail": "engine reported healthy",
                            "log_tail": self._log_tail(),
                        }
                    )
                    return
            except Exception:
                pass
            time.sleep(0.25)

        log_tail = self._log_tail()
        reason = _classify_terminal_reason(log_lines=log_tail, exited=False, model_missing=False)
        elapsed = round(time.time() - float(launch_diag.get("started_at", time.time())), 3)
        launch_diag.update(
            {
                "phase": "failed",
                "elapsed_sec": elapsed,
                "last_health_status": last_health_status,
                "terminal_reason": reason,
                "terminal_detail": f"llama-server did not become healthy within {effective_timeout:g}s",
                "log_tail": log_tail,
            }
        )
        self.last_launch_diagnostic = launch_diag
        self.stop()
        raise TimeoutError(f"llama-server did not become healthy within {effective_timeout:g}s")

    def _log_tail(self) -> list[str]:
        path: str | None = None
        if self.log_handle is not None:
            path = getattr(self.log_handle, "name", None)
        return _tail_text_file(path)

    def _live_available_capacity_mib(self) -> int | None:
        if psutil is None:
            return None
        try:
            available = int(psutil.virtual_memory().available // (1024 * 1024))
            return max(0, available)
        except Exception:
            return None

    def _reduced_topology_candidates(
        self,
        rpc_peers: list[str],
        requested_fit_target_mib: list[int],
        failure_snapshot: dict[str, Any],
    ) -> list[tuple[list[str], list[int]]]:
        if len(rpc_peers) <= 1:
            return []
        if len(requested_fit_target_mib) != len(rpc_peers) + 1:
            return []

        primary_target = int(requested_fit_target_mib[0])
        peer_targets = [int(value) for value in requested_fit_target_mib[1:]]
        failed_peer = self._detect_failed_peer(rpc_peers, failure_snapshot)

        ordered_indexes = list(range(len(rpc_peers)))
        if failed_peer in rpc_peers:
            failed_index = rpc_peers.index(failed_peer)
            ordered_indexes.remove(failed_index)
            ordered_indexes.insert(0, failed_index)

        candidates: list[tuple[list[str], list[int]]] = []
        for drop_index in ordered_indexes:
            reduced_peers = [peer for index, peer in enumerate(rpc_peers) if index != drop_index]
            if not reduced_peers:
                continue
            reduced_fit_target = [
                primary_target,
                *[target for index, target in enumerate(peer_targets) if index != drop_index],
            ]
            candidates.append((reduced_peers, reduced_fit_target))
        return candidates

    def _detect_failed_peer(self, rpc_peers: list[str], failure_snapshot: dict[str, Any]) -> str | None:
        detail = str(failure_snapshot.get("terminal_detail") or "")
        for peer in rpc_peers:
            if peer in detail:
                return peer

        for line in _tail_text_file(getattr(self.log_handle, "name", None)):
            lowered = line.lower()
            if "rpc" not in lowered:
                continue
            for peer in rpc_peers:
                if peer in line:
                    return peer
        return None

    def _service_identity(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "user": getpass.getuser(),
            "platform": os.name,
        }
        if hasattr(os, "getuid"):
            payload["uid"] = os.getuid()
        if hasattr(os, "geteuid"):
            payload["euid"] = os.geteuid()
        return payload

    def _environment_fingerprint(self) -> dict[str, Any]:
        backend_keys = [
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "ONEAPI_DEVICE_SELECTOR",
            "RAMDECK_LLAMA_SERVER_BASE_URL",
            "RAMDECK_LLAMA_SERVER_PORT",
            "RAMDECK_LLAMA_SERVER_HOST",
            "RAMDECK_LLAMA_RPC_SERVERS",
        ]
        path_value = os.environ.get("PATH", "")
        safe_backend = {key: os.environ.get(key) for key in backend_keys if os.environ.get(key)}
        return {
            "path_entries": len([entry for entry in path_value.split(os.pathsep) if entry]),
            "path_sha256": hashlib.sha256(path_value.encode("utf-8", errors="ignore")).hexdigest() if path_value else None,
            "backend_vars": safe_backend,
        }

    def _terminate_process_tree(self, process: subprocess.Popen, grace_sec: float = 10.0) -> None:
        if process.poll() is not None:
            return
        if psutil is None:
            process.terminate()
            try:
                process.wait(timeout=grace_sec)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            return

        try:
            root = psutil.Process(process.pid)
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=grace_sec)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            return

        children = root.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except Exception:
                pass
        try:
            root.terminate()
        except Exception:
            pass

        _, alive = psutil.wait_procs(children + [root], timeout=grace_sec)
        for proc in alive:
            try:
                proc.kill()
            except Exception:
                pass
        if alive:
            try:
                psutil.wait_procs(alive, timeout=5.0)
            except Exception:
                pass

    def _wait_for_port_free(self, port: int, timeout_sec: float = PORT_RELEASE_TIMEOUT_SEC, interval_sec: float = 0.1) -> None:
        if not self._is_port_open(port):
            return
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while time.monotonic() < deadline:
            if not self._is_port_open(port):
                return
            time.sleep(max(0.05, float(interval_sec)))
        raise TimeoutError(f"llama-server port {port} did not become free within {timeout_sec:g}s")

    def _is_port_open(self, port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        try:
            return sock.connect_ex(("127.0.0.1", int(port))) == 0
        except Exception:
            return False
        finally:
            sock.close()

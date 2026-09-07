"""RAMDeck contributor agent.

Runs on a contributing PC/Mac/Linux device, launches the local llama.cpp RPC
worker, registers capacity with a RAMDeck coordinator, and sends heartbeats.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import platform
import queue
import re
import shutil
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import tempfile
import uuid
import webbrowser
from typing import Any, Callable, TextIO
import psutil
import httpx

try:
    from .discovery import DiscoveredCoordinator, discover_coordinators
    from .compatibility import parse_gguf_metadata
    from .engine_supervisor import EngineSupervisor
except ImportError:  # PyInstaller can execute this file as a script entrypoint.
    from ramdeck.discovery import DiscoveredCoordinator, discover_coordinators
    from ramdeck.compatibility import parse_gguf_metadata
    from ramdeck.engine_supervisor import EngineSupervisor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [node-agent] %(message)s")
logger = logging.getLogger("ramdeck.node_agent")

DEFAULT_RPC_PORT = 50052
DEFAULT_CPU_RPC_PORT = 50053
DEFAULT_COORDINATOR_PORT = 8420
DEFAULT_COORDINATOR_IP = "127.0.0.1"
DEFAULT_HEARTBEAT_INTERVAL_SEC = 4
DEFAULT_RETRY_BACKOFF_SEC = 4
DEFAULT_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SEC = 10.0
DEFAULT_MODEL_DOWNLOAD_READ_TIMEOUT_SEC = 120.0
DEFAULT_MODEL_DOWNLOAD_WRITE_TIMEOUT_SEC = 30.0
DEFAULT_MODEL_DOWNLOAD_POOL_TIMEOUT_SEC = 10.0
DEFAULT_MODEL_DOWNLOAD_IDLE_TIMEOUT_SEC = 60.0
DEFAULT_MODEL_DOWNLOAD_MIN_THROUGHPUT_BPS = 1024.0
DEFAULT_MODEL_DOWNLOAD_MIN_THROUGHPUT_WINDOW_SEC = 120.0
DEFAULT_MODEL_DOWNLOAD_MAX_RETRIES = 3
DEFAULT_MODEL_DOWNLOAD_BACKOFF_SEC = 5.0
DEFAULT_RPC_STARTUP_TIMEOUT_SEC = 30
DEFAULT_RPC_RESTART_BACKOFF_SEC = 2
DEFAULT_RPC_MAX_RESTARTS = 8
DEFAULT_RPC_HEALTH_FAILURE_THRESHOLD = 360
DEFAULT_RPC_HEALTH_TIMEOUT_FAILURE_MULTIPLIER = 0
DEFAULT_RPC_SHUTDOWN_TIMEOUT_SEC = 5.0
UNKNOWN_NODE_STATUS = 404
FIREWALL_RULE_RPC_DISPLAY_NAME = "RAMDeck llama.cpp RPC 50052"
FIREWALL_RULE_CPU_RPC_DISPLAY_NAME = "RAMDeck RPC (CPU secondary)"
FIREWALL_RULE_LLAMA_SERVER_DISPLAY_NAME = "RAMDeck Llama Server (8080)"
DEFAULT_LLAMA_SERVER_PORT = 8080
WINDOWS_APP_NAME = "RAMDeck"
WINDOWS_INSTALLED_EXE = "RAMDeck-Node.exe"
WINDOWS_STARTUP_SHORTCUT = "RAMDeck Node.lnk"
WINDOWS_SCHEDULED_TASK_NAME = "RAMDeck Node"
WINDOWS_TRAY_TASK_NAME = "RAMDeck Node Tray"
WINDOWS_SCHEDULED_TASK_PATH = "\\RAMDeck\\"
WINDOWS_SERVICE_NAME = "RAMDeckNode"
WINDOWS_SERVICE_DISPLAY_NAME = "RAMDeck Node Service"
WINDOWS_CONFIG_NAME = "config.json"
WINDOWS_BIN_DIR_NAME = "bin"
WINDOWS_LOG_DIR_NAME = "logs"
AGENT_BUILD_LABEL = "0.2.0-core"
ANDROID_CRITICAL_AVAILABLE_MB = 2200
ANDROID_HIGH_AVAILABLE_MB = 2800


class ConfigRequiredError(RuntimeError):
    pass


class DownloadTransferError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        bytes_downloaded: int = 0,
        bytes_total: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.bytes_downloaded = bytes_downloaded
        self.bytes_total = bytes_total
        self.retryable = retryable


def _download_client_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=float(os.environ.get("RAMDECK_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SEC", str(DEFAULT_MODEL_DOWNLOAD_CONNECT_TIMEOUT_SEC))),
        read=float(os.environ.get("RAMDECK_MODEL_DOWNLOAD_READ_TIMEOUT_SEC", str(DEFAULT_MODEL_DOWNLOAD_READ_TIMEOUT_SEC))),
        write=float(os.environ.get("RAMDECK_MODEL_DOWNLOAD_WRITE_TIMEOUT_SEC", str(DEFAULT_MODEL_DOWNLOAD_WRITE_TIMEOUT_SEC))),
        pool=float(os.environ.get("RAMDECK_MODEL_DOWNLOAD_POOL_TIMEOUT_SEC", str(DEFAULT_MODEL_DOWNLOAD_POOL_TIMEOUT_SEC))),
    )


def _download_retry_budget() -> int:
    return max(1, int(os.environ.get("RAMDECK_MODEL_DOWNLOAD_MAX_RETRIES", str(DEFAULT_MODEL_DOWNLOAD_MAX_RETRIES))))


def _download_backoff_seconds(attempt: int) -> float:
    base = float(os.environ.get("RAMDECK_MODEL_DOWNLOAD_BACKOFF_SEC", str(DEFAULT_MODEL_DOWNLOAD_BACKOFF_SEC)))
    return base * (2 ** max(0, attempt - 1))


def _format_download_progress(bytes_downloaded: int, bytes_total: int | None) -> str:
    if bytes_total:
        return f"{bytes_downloaded} / {bytes_total} bytes"
    return f"{bytes_downloaded} bytes"


def _download_is_retryable_http_status(status_code: int) -> bool:
    return status_code >= 500


def _download_model_once(
    url: str,
    filename: str,
    hf_token: str | None,
    models_dir: Path,
) -> dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    target = (models_dir / filename).resolve()
    if models_dir.resolve() not in target.parents and target != models_dir.resolve():
        raise RuntimeError("target path escapes models directory")

    tmp_path = target.with_suffix(target.suffix + ".part")
    tmp_path.unlink(missing_ok=True)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    idle_timeout = float(os.environ.get("RAMDECK_MODEL_DOWNLOAD_IDLE_TIMEOUT_SEC", str(DEFAULT_MODEL_DOWNLOAD_IDLE_TIMEOUT_SEC)))
    throughput_window_sec = float(os.environ.get("RAMDECK_MODEL_DOWNLOAD_MIN_THROUGHPUT_WINDOW_SEC", str(DEFAULT_MODEL_DOWNLOAD_MIN_THROUGHPUT_WINDOW_SEC)))
    min_throughput_bps = float(os.environ.get("RAMDECK_MODEL_DOWNLOAD_MIN_THROUGHPUT_BPS", str(DEFAULT_MODEL_DOWNLOAD_MIN_THROUGHPUT_BPS)))
    bytes_total: int | None = None
    bytes_downloaded = 0
    last_progress_time = time.monotonic()
    throughput_samples: list[tuple[float, int]] = [(last_progress_time, 0)]

    try:
        with httpx.Client(follow_redirects=True, timeout=_download_client_timeout()) as client:
            with client.stream("GET", url, headers=headers) as response:
                status_code = response.status_code
                if 400 <= status_code < 500:
                    raise DownloadTransferError(
                        f"download rejected with HTTP {status_code}",
                        bytes_downloaded=0,
                        bytes_total=None,
                        retryable=False,
                    )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise DownloadTransferError(
                        f"download failed with HTTP {status_code}",
                        bytes_downloaded=bytes_downloaded,
                        bytes_total=bytes_total,
                        retryable=_download_is_retryable_http_status(status_code),
                    ) from exc

                content_length = response.headers.get("Content-Length") or response.headers.get("content-length")
                if content_length:
                    try:
                        bytes_total = int(content_length)
                    except ValueError:
                        bytes_total = None

                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 256):
                        now = time.monotonic()
                        if now - last_progress_time > idle_timeout:
                            raise DownloadTransferError(
                                f"download stalled for {now - last_progress_time:.0f}s without receiving bytes",
                                bytes_downloaded=bytes_downloaded,
                                bytes_total=bytes_total,
                                retryable=True,
                            )
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bytes_downloaded += len(chunk)
                        last_progress_time = now
                        throughput_samples.append((now, bytes_downloaded))
                        cutoff = now - throughput_window_sec
                        while len(throughput_samples) > 1 and throughput_samples[0][0] < cutoff:
                            throughput_samples.pop(0)
                        if len(throughput_samples) > 1:
                            window_elapsed = throughput_samples[-1][0] - throughput_samples[0][0]
                            if window_elapsed >= throughput_window_sec:
                                window_bytes = throughput_samples[-1][1] - throughput_samples[0][1]
                                window_bps = window_bytes / window_elapsed if window_elapsed > 0 else 0.0
                                if window_bps < min_throughput_bps:
                                    raise DownloadTransferError(
                                        f"download throughput too low: {window_bps:.1f} B/s over {window_elapsed:.0f}s (minimum {min_throughput_bps:.1f} B/s)",
                                        bytes_downloaded=bytes_downloaded,
                                        bytes_total=bytes_total,
                                        retryable=True,
                                    )

        if bytes_downloaded <= 0:
            raise DownloadTransferError(
                "download completed without receiving any bytes",
                bytes_downloaded=0,
                bytes_total=bytes_total,
                retryable=False,
            )

        tmp_path.replace(target)
        metadata = parse_gguf_metadata(target).to_dict()
        return {
            "command_result_path": str(target),
            "command_result_metadata": metadata,
            "command_result_bytes_downloaded": bytes_downloaded,
            "command_result_bytes_total": bytes_total,
        }
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundled_binary_dirs() -> list[Path]:
    dirs = [app_dir()]
    if hasattr(sys, "_MEIPASS"):
        dirs.insert(0, Path(sys._MEIPASS))
    return dirs


def default_config_path() -> Path:
    if is_windows_frozen():
        return windows_config_path()
    return app_dir() / "ramdeck_config.json"


def has_interactive_stdin() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def is_windows_frozen() -> bool:
    return os.name == "nt" and getattr(sys, "frozen", False)


def windows_install_dir() -> Path:
    program_data = os.environ.get("PROGRAMDATA") or r"C:\ProgramData"
    return Path(program_data) / WINDOWS_APP_NAME


def windows_installed_exe_path() -> Path:
    return windows_install_dir() / WINDOWS_INSTALLED_EXE


def windows_bin_dir() -> Path:
    return windows_install_dir() / WINDOWS_BIN_DIR_NAME


def windows_log_dir() -> Path:
    return windows_install_dir() / WINDOWS_LOG_DIR_NAME


def windows_config_path() -> Path:
    return windows_install_dir() / WINDOWS_CONFIG_NAME


def windows_models_dir() -> Path:
    return windows_install_dir() / "models"


def windows_startup_shortcut_path() -> Path:
    startup = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    return startup / WINDOWS_STARTUP_SHORTCUT


def powershell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def windows_is_admin() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def relaunch_windows_as_admin(argv: list[str] | None = None) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        executable = str(Path(sys.executable).resolve())
        parameters = subprocess.list2cmdline(list(argv if argv is not None else sys.argv[1:]))
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            executable,
            parameters,
            None,
            1,
        )
        if int(result) <= 32:
            raise RuntimeError(f"ShellExecuteW returned {result}")
        logger.info("Relaunched RAMDeck installer with administrator rights")
        return True
    except Exception as exc:
        raise RuntimeError(f"Could not relaunch installer with administrator rights: {exc}") from exc


def run_powershell(script: str, elevated: bool = False, timeout: int = 30) -> None:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    if not elevated:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=creationflags,
        )
        return

    import base64
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand','{encoded}'",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=creationflags,
    )


def create_windows_startup_shortcut(installed_exe: Path) -> None:
    shortcut = windows_startup_shortcut_path()
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    script = f"""
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut({powershell_quote(shortcut)})
$shortcut.TargetPath = {powershell_quote(installed_exe)}
$shortcut.Arguments = '--start-minimized'
$shortcut.WorkingDirectory = {powershell_quote(installed_exe.parent)}
$shortcut.Description = 'Start RAMDeck node at Windows sign-in'
$shortcut.Save()
"""
    run_powershell(script, timeout=20)


def remove_windows_startup_shortcut() -> None:
    shortcut = windows_startup_shortcut_path()
    for _ in range(10):
        try:
            shortcut.unlink(missing_ok=True)
            return
        except OSError:
            time.sleep(0.2)
    logger.warning("Could not remove legacy Startup shortcut at %s", shortcut)


def register_windows_logon_task(installed_exe: Path) -> None:
    shortcut = windows_startup_shortcut_path()
    script = f"""
$ErrorActionPreference = 'Stop'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute {powershell_quote(installed_exe)} -Argument '--start-minimized' -WorkingDirectory {powershell_quote(installed_exe.parent)}
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskPath {powershell_quote(WINDOWS_SCHEDULED_TASK_PATH)} -TaskName {powershell_quote(WINDOWS_SCHEDULED_TASK_NAME)} -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Remove-Item {powershell_quote(shortcut)} -Force -ErrorAction SilentlyContinue
"""
    run_powershell(script, timeout=30)


def windows_logon_task_exists() -> bool:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", f"{WINDOWS_SCHEDULED_TASK_PATH}{WINDOWS_SCHEDULED_TASK_NAME}"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=creationflags,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.warning("Could not query Windows logon task: %s", exc)
        return False


def bundled_asset(name: str) -> Path:
    for base in bundled_binary_dirs():
        candidate = base / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing bundled Windows payload: {name}")


def bundled_asset_optional(name: str) -> Path | None:
    for base in bundled_binary_dirs():
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def run_hidden(command: list[str], check: bool = True, timeout: int = 30) -> subprocess.CompletedProcess:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=creationflags,
    )


def windows_service_exists() -> bool:
    result = run_hidden(["sc.exe", "query", WINDOWS_SERVICE_NAME], check=False)
    return result.returncode == 0


def windows_service_state(service_name: str) -> str:
    query = run_hidden(["sc.exe", "query", service_name], check=False, timeout=15)
    text = f"{query.stdout}\n{query.stderr}"
    for line in text.splitlines():
        if "STATE" in line and ":" in line:
            state_part = line.split(":", 1)[1]
            pieces = state_part.strip().split()
            if len(pieces) >= 2:
                return pieces[1].upper()
    return "UNKNOWN"


def wait_for_windows_service_running(service_name: str, timeout_sec: float = 30.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        state = windows_service_state(service_name)
        if state == "RUNNING":
            return True
        if state in {"STOPPED", "PAUSED", "UNKNOWN"}:
            time.sleep(0.5)
            continue
        time.sleep(0.5)
    return windows_service_state(service_name) == "RUNNING"


def remove_legacy_windows_startup() -> None:
    run_hidden(
        ["schtasks", "/Delete", "/TN", f"{WINDOWS_SCHEDULED_TASK_PATH}{WINDOWS_SCHEDULED_TASK_NAME}", "/F"],
        check=False,
    )
    remove_windows_startup_shortcut()


def configure_windows_tray_task(installed_exe: Path) -> None:
    script = f"""
$ErrorActionPreference = 'Stop'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute {powershell_quote(installed_exe)} -Argument '--service-tray --no-ui --start-minimized' -WorkingDirectory {powershell_quote(installed_exe.parent)}
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskPath {powershell_quote(WINDOWS_SCHEDULED_TASK_PATH)} -TaskName {powershell_quote(WINDOWS_TRAY_TASK_NAME)} -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
"""
    run_powershell(script, timeout=30)


def load_windows_config_safely(path: Path | None = None) -> dict:
    config_path = path or windows_config_path()
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except Exception as exc:
        logger.warning("Could not read Windows config %s: %s", config_path, exc)
        return {}


def windows_dashboard_url_from_config(path: Path | None = None) -> str | None:
    config = load_windows_config_safely(path)
    coordinator_ip = str(config.get("coordinator_ip") or "").strip()
    if not coordinator_ip:
        return None
    coordinator_port = int(config.get("coordinator_port") or DEFAULT_COORDINATOR_PORT)
    return dashboard_url(f"http://{coordinator_ip}:{coordinator_port}")


def windows_service_control(action: str) -> str:
    action_key = action.lower().strip()
    if action_key not in {"start", "stop", "restart"}:
        raise ValueError(f"unsupported service action: {action}")

    nssm = windows_bin_dir() / "nssm.exe"
    if action_key == "restart":
        windows_service_control("stop")
        time.sleep(1.0)
        return windows_service_control("start")

    if nssm.exists():
        cmd = [str(nssm), action_key, WINDOWS_SERVICE_NAME]
        if action_key == "stop":
            cmd.append("confirm")
    else:
        cmd = ["sc.exe", action_key, WINDOWS_SERVICE_NAME]

    result = run_hidden(cmd, check=False, timeout=45)
    state = windows_service_state(WINDOWS_SERVICE_NAME)
    detail = " ".join(filter(None, [result.stdout.strip(), result.stderr.strip()])).strip()
    if detail:
        return f"{action_key} requested; service state is {state}. {detail}"
    return f"{action_key} requested; service state is {state}."


def start_windows_service() -> None:
    nssm = windows_bin_dir() / "nssm.exe"
    start_result = run_hidden([str(nssm), "start", WINDOWS_SERVICE_NAME], check=False, timeout=45)
    if wait_for_windows_service_running(WINDOWS_SERVICE_NAME, timeout_sec=35.0):
        return

    query = run_hidden(["sc.exe", "query", WINDOWS_SERVICE_NAME], check=False, timeout=15)
    detail = "\n".join(filter(None, [start_result.stdout.strip(), start_result.stderr.strip(), query.stdout.strip(), query.stderr.strip()]))
    raise RuntimeError(f"Failed to start {WINDOWS_SERVICE_NAME} service. {detail}")


def configure_windows_service(installed_exe: Path, config: dict, start_service: bool = True) -> None:
    nssm = windows_bin_dir() / "nssm.exe"
    rpc_binary = windows_bin_dir() / "ggml-rpc-server.exe"
    rpc_log = windows_log_dir() / "ggml-rpc-server.log"
    service_stdout = windows_log_dir() / "node-service.out.log"
    service_stderr = windows_log_dir() / "node-service.err.log"
    service_args = [
        "--config", str(windows_config_path()),
        "--coordinator-ip", str(config["coordinator_ip"]),
        "--coordinator-port", str(config["coordinator_port"]),
        "--rpc-port", str(config["rpc_port"]),
        "--rpc-binary", str(rpc_binary),
        "--rpc-log-file", str(rpc_log),
        "--no-discovery",
        "--no-ui",
        "--skip-firewall-rule",
    ]
    
    cpu_rpc_binary = windows_bin_dir() / "cpu" / "cpu-ggml-rpc-server.exe"
    if cpu_rpc_binary.exists():
        cpu_rpc_port = int(config.get("cpu_rpc_port", DEFAULT_CPU_RPC_PORT))
        cpu_rpc_log = windows_log_dir() / "cpu-ggml-rpc-server.log"
        service_args.extend([
            "--cpu-rpc-binary", str(cpu_rpc_binary),
            "--cpu-rpc-port", str(cpu_rpc_port),
            "--cpu-rpc-log-file", str(cpu_rpc_log),
        ])

    if windows_service_exists():
        run_hidden([str(nssm), "stop", WINDOWS_SERVICE_NAME, "confirm"], check=False, timeout=45)
    else:
        run_hidden([str(nssm), "install", WINDOWS_SERVICE_NAME, str(installed_exe)])

    settings = {
        "Application": str(installed_exe),
        "AppDirectory": str(windows_install_dir()),
        "AppParameters": subprocess.list2cmdline(service_args),
        "DisplayName": WINDOWS_SERVICE_DISPLAY_NAME,
        "Description": "RAMDeck worker and primary-node runtime",
        "Start": "SERVICE_AUTO_START",
        "AppRestartDelay": "5000",
        "AppStdout": str(service_stdout),
        "AppStderr": str(service_stderr),
        "AppRotateFiles": "1",
    }
    for key, value in settings.items():
        run_hidden([str(nssm), "set", WINDOWS_SERVICE_NAME, key, value])
    run_hidden([str(nssm), "set", WINDOWS_SERVICE_NAME, "AppExit", "Default", "Restart"])

    run_hidden(
        ["sc.exe", "failure", WINDOWS_SERVICE_NAME, "reset=", "86400", "actions=", "restart/5000/restart/5000/restart/10000"],
    )
    if not start_service:
        return
    start_windows_service()


def install_windows_app(config: dict, status_callback=None) -> Path | None:
    if not is_windows_frozen():
        return None

    install_dir = windows_install_dir()
    installed_exe = windows_installed_exe_path()
    source_exe = Path(sys.executable).resolve()
    bin_dir = windows_bin_dir()
    log_dir = windows_log_dir()
    for path in (install_dir, bin_dir, log_dir, windows_models_dir()):
        path.mkdir(parents=True, exist_ok=True)

    notify(status_callback, "Stopping any previous RAMDeck service...")
    if windows_service_exists():
        installed_nssm = bin_dir / "nssm.exe"
        if installed_nssm.exists():
            run_hidden([str(installed_nssm), "stop", WINDOWS_SERVICE_NAME, "confirm"], check=False, timeout=45)
        else:
            run_hidden(["sc.exe", "stop", WINDOWS_SERVICE_NAME], check=False, timeout=45)
        time.sleep(1)

    if source_exe != installed_exe.resolve():
        shutil.copy2(source_exe, installed_exe)
        notify(status_callback, f"Installed RAMDeck node to {installed_exe}")

    payload_names = [
        "nssm.exe",
        "ggml-rpc-server.exe",
        "llama-server.exe",
        "ggml-base.dll",
        "ggml-rpc.dll",
        "ggml.dll",
        "llama-common.dll",
        "llama-server-impl.dll",
        "llama.dll",
        "mtmd.dll",
        "cpu-ggml-rpc-server.exe",
    ]
    for name in payload_names:
        asset_path = bundled_asset_optional(name) if name == "cpu-ggml-rpc-server.exe" else bundled_asset(name)
        if asset_path:
            # cpu-ggml-rpc-server.exe must live in a subdirectory that has NO
            # ggml-cuda.dll alongside it — ggml-rpc-server.exe discovers the GPU
            # backend at runtime via DLL probe of its working directory (cwd).
            # Putting it in bin\cpu\ isolates it from ggml-cuda.dll in bin\.
            if name == "cpu-ggml-rpc-server.exe":
                cpu_bin_subdir = bin_dir / "cpu"
                cpu_bin_subdir.mkdir(exist_ok=True)
                shutil.copy2(asset_path, cpu_bin_subdir / name)
            else:
                shutil.copy2(asset_path, bin_dir / name)

    # Support both payload layouts:
    # 1) Legacy CPU lane with multiple architecture-specific ggml-cpu-*.dll files.
    # 2) Newer CPU/CUDA lanes with a single ggml-cpu.dll (and optional ggml-cuda.dll).
    cpu_payload_names = [
        "ggml-cpu.dll",
        "ggml-cpu-alderlake.dll",
        "ggml-cpu-cannonlake.dll",
        "ggml-cpu-cascadelake.dll",
        "ggml-cpu-cooperlake.dll",
        "ggml-cpu-haswell.dll",
        "ggml-cpu-icelake.dll",
        "ggml-cpu-ivybridge.dll",
        "ggml-cpu-piledriver.dll",
        "ggml-cpu-sandybridge.dll",
        "ggml-cpu-sapphirerapids.dll",
        "ggml-cpu-skylakex.dll",
        "ggml-cpu-sse42.dll",
        "ggml-cpu-x64.dll",
        "ggml-cpu-zen4.dll",
    ]
    copied_cpu_backend = False
    for name in cpu_payload_names:
        candidate = bundled_asset_optional(name)
        if candidate is None:
            continue
        copied_cpu_backend = True
        shutil.copy2(candidate, bin_dir / name)

    if not copied_cpu_backend:
        raise FileNotFoundError(
            "missing bundled Windows payload: ggml-cpu.dll or ggml-cpu-*.dll"
        )

    optional_payload_names = [
        "ggml-cuda.dll",
        "libomp140.x86_64.dll",
    ]
    for name in optional_payload_names:
        candidate = bundled_asset_optional(name)
        if candidate is None:
            continue
        shutil.copy2(candidate, bin_dir / name)

    # Mirror the full CPU runtime into bin\cpu\ so the secondary CPU RPC
    # worker (cwd = bin\cpu\) is self-contained and has no ggml-cuda.dll.
    # We copy every .dll already installed into bin\ EXCEPT ggml-cuda.dll,
    # rather than maintaining a manual allowlist that risks a missing-DLL crash.
    cpu_bin_subdir = bin_dir / "cpu"
    if cpu_bin_subdir.exists():
        for src in bin_dir.iterdir():
            if src.suffix.lower() != ".dll":
                continue
            if src.name.lower() == "ggml-cuda.dll":
                # Never mirror the CUDA backend — its absence is what keeps
                # the secondary worker CPU-only at runtime.
                continue
            shutil.copy2(src, cpu_bin_subdir / src.name)
        # Sanity: assert ggml-cuda.dll was NOT copied in
        if (cpu_bin_subdir / "ggml-cuda.dll").exists():
            raise RuntimeError(
                "ggml-cuda.dll was unexpectedly copied into bin\\cpu\\. "
                "Secondary CPU worker would acquire GPU backend. Aborting install."
            )
        logger.info(
            "Mirrored %d DLLs into %s (ggml-cuda.dll excluded)",
            sum(1 for f in cpu_bin_subdir.iterdir() if f.suffix.lower() == ".dll"),
            cpu_bin_subdir,
        )

    config["models_dir"] = str(windows_models_dir())
    config["llama_server_path"] = str(bin_dir / "llama-server.exe")
    save_config(windows_config_path(), config)
    remove_legacy_windows_startup()
    notify(status_callback, "Installing RAMDeck auto-restart service...")
    configure_windows_service(installed_exe, config, start_service=False)
    notify(status_callback, "Configuring Windows firewall rules...")
    ensure_windows_firewall_rule(FIREWALL_RULE_RPC_DISPLAY_NAME, int(config["rpc_port"]))
    cpu_rpc_binary = windows_bin_dir() / "cpu" / "cpu-ggml-rpc-server.exe"
    if cpu_rpc_binary.exists():
        cpu_rpc_port = int(config.get("cpu_rpc_port", DEFAULT_CPU_RPC_PORT))
        ensure_windows_firewall_rule(FIREWALL_RULE_CPU_RPC_DISPLAY_NAME, cpu_rpc_port)
    ensure_windows_firewall_rule(FIREWALL_RULE_LLAMA_SERVER_DISPLAY_NAME, DEFAULT_LLAMA_SERVER_PORT)
    notify(status_callback, "Starting RAMDeck service...")
    start_windows_service()
    notify(status_callback, "Installing RAMDeck tray companion...")
    configure_windows_tray_task(installed_exe)
    notify(status_callback, f"RAMDeck service installed. Models folder: {windows_models_dir()}")
    return installed_exe


def default_rpc_log_path() -> Path:
    return app_dir() / "ggml-rpc-server.log"


def load_or_create_config(path: Path, coordinator_ip: str | None, coordinator_port: int) -> dict:
    if path.exists():
        data = json.loads(path.read_text())
    else:
        if coordinator_ip is None:
            if not has_interactive_stdin():
                template = {
                    "coordinator_ip": DEFAULT_COORDINATOR_IP,
                    "coordinator_port": coordinator_port,
                    "node_id": str(uuid.uuid4())[:8],
                    "rpc_port": DEFAULT_RPC_PORT,
                }
                path.write_text(json.dumps(template, indent=2) + "\n")
                raise ConfigRequiredError(f"missing coordinator config; wrote template to {path}")
            coordinator_ip = input("RAMDeck coordinator IP: ").strip()
        data = {
            "coordinator_ip": coordinator_ip,
            "coordinator_port": coordinator_port,
            "node_id": str(uuid.uuid4())[:8],
            "rpc_port": DEFAULT_RPC_PORT,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    data.setdefault("coordinator_port", coordinator_port)
    data.setdefault("node_id", str(uuid.uuid4())[:8])
    data.setdefault("rpc_port", DEFAULT_RPC_PORT)
    return data


def save_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def notify(status_callback, message: str, dashboard_url: str | None = None) -> None:
    logger.info(message)
    if status_callback:
        status_callback(message, dashboard_url)


def config_from_coordinator(coordinator: DiscoveredCoordinator, existing: dict | None = None) -> dict:
    config = dict(existing or {})
    config["coordinator_ip"] = coordinator.ip
    config["coordinator_port"] = coordinator.port
    config.setdefault("node_id", str(uuid.uuid4())[:8])
    config.setdefault("rpc_port", DEFAULT_RPC_PORT)
    config["coordinator_name"] = coordinator.display_name
    return config


def choose_coordinator_with_tk(coordinators: list[DiscoveredCoordinator]) -> DiscoveredCoordinator | None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return None

    selected: dict[str, DiscoveredCoordinator | None] = {"value": None}
    root = tk.Tk()
    root.title("Choose RAMDeck Coordinator")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    ttk.Label(root, text="Choose a RAMDeck coordinator:").pack(padx=16, pady=(16, 8), anchor="w")
    options = [f"{item.display_name} ({item.ip}:{item.port})" for item in coordinators]
    choice = tk.StringVar(value=options[0])
    combo = ttk.Combobox(root, values=options, textvariable=choice, state="readonly", width=44)
    combo.pack(padx=16, pady=8)

    def accept():
        selected["value"] = coordinators[options.index(choice.get())]
        root.destroy()

    ttk.Button(root, text="Connect", command=accept).pack(padx=16, pady=(8, 16), anchor="e")
    root.mainloop()
    return selected["value"]


def choose_coordinator(coordinators: list[DiscoveredCoordinator]) -> DiscoveredCoordinator:
    if len(coordinators) == 1:
        return coordinators[0]

    if not has_interactive_stdin():
        selected = choose_coordinator_with_tk(coordinators)
        if selected:
            return selected
        raise RuntimeError("multiple RAMDeck coordinators found; no UI available to choose one")

    print("Multiple RAMDeck coordinators found:")
    for idx, coordinator in enumerate(coordinators, start=1):
        print(f"{idx}. {coordinator.display_name} ({coordinator.ip}:{coordinator.port})")
    while True:
        raw = input("Choose coordinator number: ").strip()
        try:
            index = int(raw) - 1
            if 0 <= index < len(coordinators):
                return coordinators[index]
        except ValueError:
            pass
        print("Invalid selection")


def discover_config(path: Path, timeout_sec: float, existing: dict | None = None) -> dict | None:
    coordinators = discover_coordinators(timeout_sec=timeout_sec)
    if not coordinators:
        return None
    coordinator = choose_coordinator(coordinators)
    config = config_from_coordinator(coordinator, existing=existing)
    save_config(path, config)
    logger.info("Discovered RAMDeck coordinator %s at %s:%s", coordinator.display_name, coordinator.ip, coordinator.port)
    return config


def resolve_config(args: argparse.Namespace) -> dict:
    if args.config.exists():
        return load_or_create_config(args.config, args.coordinator_ip, args.coordinator_port)
    if args.coordinator_ip:
        return load_or_create_config(args.config, args.coordinator_ip, args.coordinator_port)
    if not args.no_discovery:
        discovered = discover_config(args.config, args.discovery_timeout)
        if discovered:
            return discovered
    return load_or_create_config(args.config, args.coordinator_ip, args.coordinator_port)


def resolve_models_dir(args: argparse.Namespace, config: dict) -> Path | None:
    configured = str(config.get("models_dir") or "").strip()
    if configured:
        return Path(configured)
    return args.models_dir


def local_ip_for_coordinator(coordinator_ip: str, coordinator_port: int) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((coordinator_ip, coordinator_port))
        return sock.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        sock.close()


def locate_rpc_binary(explicit_path: str | None = None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    names = ["ggml-rpc-server.exe", "rpc-server.exe", "ggml-rpc-server", "rpc-server"]
    for base in bundled_binary_dirs():
        candidates.extend(base / name for name in names)
        candidates.extend(base / "bin" / name for name in names)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find bundled llama.cpp RPC server binary")


def locate_llama_server_binary(explicit_path: str | None = None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    names = ["llama-server.exe", "llama-server"]
    for base in bundled_binary_dirs():
        candidates.extend(base / name for name in names)
        candidates.extend(base / "bin" / name for name in names)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else app_dir() / names[0 if os.name == "nt" else 1]


def detect_gpu() -> dict:
    """Best-effort GPU detection. Extend per-platform as compatibility
    matrix grows (see docs/HARDWARE_COMPATIBILITY.md)."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) >= 3:
                name = parts[0]
                mem_total = int(float(parts[1].split()[0]))
                mem_free = int(float(parts[2].split()[0]))
                return {
                    "gpu_vendor": "nvidia",
                    "gpu_model": name,
                    "gpu_vram_mb": mem_free,
                    "gpu_total_vram_mb": mem_total,
                    "gpu_supported": True,  # NOTE: cross-check against tested matrix before trusting
                }
    except Exception:
        pass

    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import psutil
        mem = psutil.virtual_memory()
        mem_mb = mem.total // (1024 * 1024)
        mem_avail_mb = mem.available // (1024 * 1024)
        # Assuming Apple Silicon can use up to ~70% of unified memory for Metal
        max_vram_mb = int(mem_mb * 0.70)
        vram_free_mb = min(max_vram_mb, mem_avail_mb)
        return {
            "gpu_vendor": "apple",
            "gpu_model": "Apple Silicon",
            "gpu_vram_mb": vram_free_mb,
            "gpu_total_vram_mb": mem_mb,
            "gpu_supported": True,
        }

    return {"gpu_vendor": "none", "gpu_model": "", "gpu_vram_mb": 0, "gpu_total_vram_mb": 0, "gpu_supported": False}


def _platform_compute_label() -> str:
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or platform.platform()


def _is_android_runtime() -> bool:
    if platform.system() != "Linux":
        return False
    return bool(
        os.environ.get("ANDROID_ROOT")
        or os.environ.get("ANDROID_DATA")
        or os.environ.get("TERMUX_VERSION")
    )


def _score_compute_label(label: str, cores: int) -> int:
    score = max(1, cores) * 10
    lowered = label.lower()
    apple_match = re.search(r"apple\s+m(\d+)", lowered)
    if apple_match:
        score += 1000 + int(apple_match.group(1)) * 100
    if "rtx" in lowered or "geforce" in lowered or "nvidia" in lowered:
        score += 900
    if "ryzen" in lowered or "intel" in lowered or "amd" in lowered:
        score += 100
    return score


def detect_device_capability(gpu_info: dict | None = None) -> dict:
    label = _platform_compute_label()
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    if _is_android_runtime():
        return {
            "device_class": "android_mobile",
            "compute_label": f"Android ({label})",
            "compute_score": min(250, _score_compute_label(label, cores)),
            "training_capable": False,
            "training_backend": "none",
        }
    try:
        import mlx.core as mx
        if bool(mx.metal.is_available()):
            return {
                "device_class": "apple_silicon_mlx",
                "compute_label": label,
                "compute_score": _score_compute_label(label, cores),
                "training_capable": True,
                "training_backend": "mlx",
            }
    except Exception:
        pass
    
    if gpu_info and gpu_info.get("gpu_supported"):
        cuda_label = gpu_info.get("gpu_model") or "CUDA GPU"
        return {
            "device_class": "cuda",
            "compute_label": cuda_label,
            "compute_score": _score_compute_label(cuda_label, cores),
            "training_capable": True,
            "training_backend": "cuda",
        }
        
    for base in bundled_binary_dirs():
        if (base / "ggml-cuda.dll").exists() or (base / "bin" / "ggml-cuda.dll").exists():
            return {
                "device_class": "cuda",
                "compute_label": "CUDA GPU (Fallback)",
                "compute_score": _score_compute_label("CUDA", cores),
                "training_capable": True,
                "training_backend": "cuda",
            }

    return {
        "device_class": "cpu_only",
        "compute_label": label,
        "compute_score": _score_compute_label(label, cores),
        "training_capable": False,
        "training_backend": "none",
    }


def _memory_pressure_state(device_class: str, available_mb: int) -> str:
    if device_class != "android_mobile":
        return "unknown"
    if available_mb <= ANDROID_CRITICAL_AVAILABLE_MB:
        return "critical"
    if available_mb <= ANDROID_HIGH_AVAILABLE_MB:
        return "high"
    return "normal"


def discover_available_models(models_dir: Path) -> list[str]:
    if not models_dir.is_dir():
        return []
    return sorted(str(path.resolve()) for path in models_dir.rglob("*.gguf") if path.is_file())


def discover_available_model_metadata(models_dir: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    if not models_dir.is_dir():
        return metadata
    for path in sorted(models_dir.rglob("*.gguf")):
        if not path.is_file():
            continue
        try:
            parsed = parse_gguf_metadata(path)
        except Exception as exc:
            logger.warning("Could not parse GGUF metadata for %s: %s", path, exc)
            continue
        metadata[str(path.resolve())] = parsed.to_dict()
    return metadata


def compute_live_rpc_endpoints(rpc_port: int, cpu_rpc_port: int, has_cpu_rpc_process: bool) -> list[dict]:
    mem = psutil.virtual_memory()
    gpu_info = detect_gpu()
    available_ram_mb = mem.available // (1024 * 1024)
    endpoints = []
    if gpu_info.get("gpu_supported"):
        vendor = gpu_info.get("gpu_vendor")
        backend_name = "cuda" if vendor == "nvidia" else "metal" if vendor == "apple" else "gpu"
        endpoints.append({
            "port": rpc_port,
            "backend": backend_name,
            "capacity_mb": int(gpu_info.get("gpu_vram_mb", 0) * 0.85)
        })
        if has_cpu_rpc_process:
            endpoints.append({
                "port": cpu_rpc_port,
                "backend": "cpu",
                "capacity_mb": int(available_ram_mb * 0.80)
            })
    else:
        endpoints.append({
            "port": rpc_port,
            "backend": "cpu",
            "capacity_mb": int(available_ram_mb * 0.80)
        })
    return endpoints


def poll_gpu_telemetry() -> dict:
    return {}


def build_registration_payload(
    coordinator_ip: str,
    coordinator_port: int,
    node_id: str,
    rpc_endpoints: list[dict],
    models_dir: Path | None = None,
) -> dict:
    mem = psutil.virtual_memory()
    gpu_info = detect_gpu()
    gpu_telemetry = poll_gpu_telemetry()
    capability = detect_device_capability(gpu_info)
    payload = {
        "node_id": node_id,
        "hostname": socket.gethostname(),
        "ip": local_ip_for_coordinator(coordinator_ip, coordinator_port),
        "kind": "phone" if capability["device_class"] == "android_mobile" else ("laptop" if "laptop" in platform.node().lower() else "desktop"),
        "total_ram_mb": mem.total // (1024 * 1024),
        "available_ram_mb": mem.available // (1024 * 1024),
        "os": platform.system(),
        "ramdeck_version": "0.1.0",
        "memory_pressure_state": _memory_pressure_state(
            capability["device_class"],
            mem.available // (1024 * 1024),
        ),
        "gpu_temp_c": gpu_telemetry.get("gpu_temp_c"),
        "gpu_util_pct": gpu_telemetry.get("gpu_util_pct"),
        "gpu_power_w": gpu_telemetry.get("gpu_power_w"),
        "engine_state": "idle_worker",
        "loaded_model": None,
        "available_models": discover_available_models(models_dir) if models_dir else [],
        "available_model_metadata": discover_available_model_metadata(models_dir) if models_dir else {},
        "rpc_endpoints": rpc_endpoints,
    }
    payload.update(gpu_info)
    payload.update(capability)
    return payload


def start_rpc_process(binary: Path, port: int, log_handle: TextIO | None = None) -> subprocess.Popen:
    cmd = [str(binary), "-H", "0.0.0.0", "-p", str(port)]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    logger.info("Starting RPC worker: %s", " ".join(cmd))
    popen_kwargs = {
        "cwd": str(binary.parent),
        "creationflags": creationflags,
    }
    if log_handle is not None:
        popen_kwargs["stdout"] = log_handle
        popen_kwargs["stderr"] = subprocess.STDOUT
    return subprocess.Popen(cmd, **popen_kwargs)


def probe_local_rpc(port: int, timeout: float = 0.5) -> tuple[bool, str | None]:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True, None
    except OSError as exc:
        return False, f"{type(exc).__name__}: errno={exc.errno} message={exc}"


def can_connect_local_rpc(port: int, timeout: float = 0.5) -> bool:
    connected, _ = probe_local_rpc(port, timeout)
    return connected


class RpcWorkerHealthMonitor:
    def __init__(
        self,
        failure_threshold: int = DEFAULT_RPC_HEALTH_FAILURE_THRESHOLD,
        probe: Callable[[int], tuple[bool, str | None]] = probe_local_rpc,
        timeout_failure_threshold: int | None = None,
    ):
        self.failure_threshold = max(1, int(failure_threshold))
        if timeout_failure_threshold is None:
            multiplier = float(
                os.environ.get(
                    "RAMDECK_RPC_HEALTH_TIMEOUT_FAILURE_MULTIPLIER",
                    str(DEFAULT_RPC_HEALTH_TIMEOUT_FAILURE_MULTIPLIER),
                )
            )
            if multiplier > 0:
                timeout_failure_threshold = max(
                    self.failure_threshold,
                    int(self.failure_threshold * multiplier),
                )
            else:
                timeout_failure_threshold = None
        elif int(timeout_failure_threshold) <= 0:
            timeout_failure_threshold = None
        else:
            timeout_failure_threshold = max(self.failure_threshold, int(timeout_failure_threshold))

        self.timeout_failure_threshold = timeout_failure_threshold
        self.probe = probe
        self.consecutive_failures = 0
        self.consecutive_timeouts = 0

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.consecutive_timeouts = 0

    @staticmethod
    def _is_timeout_probe_error(tcp_error: str | None) -> bool:
        text = str(tcp_error or "").lower()
        return "timed out" in text or "timeouterror" in text

    def check(self, rpc_process: subprocess.Popen, port: int) -> dict[str, Any]:
        worker_pid = getattr(rpc_process, "pid", None)
        exit_code = rpc_process.poll()
        if exit_code is not None:
            self.consecutive_failures = 0
            result = {
                "healthy": False,
                "restart_required": True,
                "reason": "process_exited",
                "worker_pid": worker_pid,
                "port": int(port),
                "poll": exit_code,
                "tcp_error": "not_checked",
                "consecutive_failures": 0,
                "failure_threshold": self.failure_threshold,
            }
            self._log_failure(result)
            return result

        # Skip TCP probe for ggml-rpc-server because rapid TCP connects/disconnects 
        # cause the acceptor thread in some llama.cpp versions to deadlock.
        # Since the process poll() is successful, we assume it's healthy.
        self.consecutive_failures = 0
        self.consecutive_timeouts = 0
        return {
            "healthy": True,
                "restart_required": False,
                "reason": "healthy",
                "worker_pid": worker_pid,
                "port": int(port),
                "poll": None,
                "tcp_error": None,
                "consecutive_failures": 0,
                "consecutive_timeouts": 0,
                "failure_threshold": self.failure_threshold,
                "timeout_failure_threshold": self.timeout_failure_threshold,
            }

        if self._is_timeout_probe_error(tcp_error):
            self.consecutive_failures = 0
            self.consecutive_timeouts += 1
            restart_required = (
                self.timeout_failure_threshold is not None
                and self.consecutive_timeouts >= self.timeout_failure_threshold
            )
            result = {
                "healthy": False,
                "restart_required": restart_required,
                "reason": "tcp_connect_timeout",
                "worker_pid": worker_pid,
                "port": int(port),
                "poll": None,
                "tcp_error": tcp_error or "unknown",
                "consecutive_failures": 0,
                "consecutive_timeouts": self.consecutive_timeouts,
                "failure_threshold": self.failure_threshold,
                "timeout_failure_threshold": self.timeout_failure_threshold,
            }
            self._log_failure(result)
            return result

        self.consecutive_timeouts = 0
        self.consecutive_failures += 1
        result = {
            "healthy": False,
            "restart_required": self.consecutive_failures >= self.failure_threshold,
            "reason": "tcp_connect_failed",
            "worker_pid": worker_pid,
            "port": int(port),
            "poll": None,
            "tcp_error": tcp_error or "unknown",
            "consecutive_failures": self.consecutive_failures,
            "consecutive_timeouts": 0,
            "failure_threshold": self.failure_threshold,
            "timeout_failure_threshold": self.timeout_failure_threshold,
        }
        self._log_failure(result)
        return result

    @staticmethod
    def _log_failure(result: dict[str, Any]) -> None:
        logger.warning(
            "RPC worker health failure: reason=%s worker_pid=%s port=%s poll=%s "
            "tcp_error=%s consecutive_failures=%s timeout_failures=%s threshold=%s timeout_threshold=%s restart_required=%s",
            result["reason"],
            result["worker_pid"],
            result["port"],
            result["poll"],
            result["tcp_error"],
            result["consecutive_failures"],
            result.get("consecutive_timeouts", 0),
            result["failure_threshold"],
            result.get("timeout_failure_threshold", result["failure_threshold"]),
            result["restart_required"],
        )


def wait_for_rpc_shutdown(
    rpc_process: subprocess.Popen,
    port: int,
    timeout_sec: float = DEFAULT_RPC_SHUTDOWN_TIMEOUT_SEC,
) -> None:
    worker_pid = getattr(rpc_process, "pid", None)
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    exit_code = rpc_process.poll()
    if exit_code is None:
        rpc_process.terminate()
        try:
            exit_code = rpc_process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            message = (
                f"RPC worker shutdown failed: worker_pid={worker_pid} port={int(port)} "
                f"process did not exit within {float(timeout_sec):g}s"
            )
            logger.error(message)
            raise RuntimeError(message) from exc

    last_tcp_error = "unknown"
    while time.monotonic() < deadline:
        connected, tcp_error = probe_local_rpc(int(port))
        if not connected:
            logger.info(
                "RPC worker shutdown confirmed: worker_pid=%s port=%s exit_code=%s tcp_error=%s",
                worker_pid,
                int(port),
                exit_code,
                tcp_error,
            )
            return
        last_tcp_error = tcp_error or "port still accepts connections"
        time.sleep(0.1)

    message = (
        f"RPC worker shutdown failed: worker_pid={worker_pid} port={int(port)} "
        f"exit_code={exit_code} port_not_released={last_tcp_error}"
    )
    logger.error(message)
    raise RuntimeError(message)


def restart_rpc_process_safely(
    rpc_process: subprocess.Popen,
    port: int,
    shutdown_timeout_sec: float,
    restart_backoff_sec: float,
    start_callback: Callable[[], subprocess.Popen],
    wait_callback: Callable[[float], Any] = time.sleep,
) -> subprocess.Popen:
    wait_for_rpc_shutdown(rpc_process, port, shutdown_timeout_sec)
    wait_callback(max(0.0, float(restart_backoff_sec)))
    return start_callback()


def agent_instance_lock_path(config_path: Path, node_id: str, rpc_port: int) -> Path:
    identity = f"{config_path.resolve()}\0{node_id}\0{int(rpc_port)}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return Path(tempfile.gettempdir()) / f"ramdeck-node-{digest}.lock"


def acquire_agent_instance_lock(config_path: Path, node_id: str, rpc_port: int) -> Path:
    lock_path = agent_instance_lock_path(config_path, node_id, rpc_port)
    process = psutil.Process(os.getpid())
    owner = {
        "pid": process.pid,
        "create_time": process.create_time(),
        "config_path": str(config_path.resolve()),
        "node_id": node_id,
        "rpc_port": int(rpc_port),
    }

    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                existing_process = psutil.Process(int(existing["pid"]))
                same_process = abs(existing_process.create_time() - float(existing["create_time"])) < 0.01
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, psutil.Error):
                same_process = False
            if same_process:
                raise RuntimeError(
                    "RAMDeck node is already running for "
                    f"config {config_path.resolve()} on RPC port {rpc_port} "
                    f"(pid {existing['pid']})"
                )
            lock_path.unlink(missing_ok=True)
            continue

        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(owner, handle)
        return lock_path

    raise RuntimeError(f"could not acquire RAMDeck node instance lock {lock_path}")


def release_agent_instance_lock(lock_path: Path) -> None:
    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        if int(owner.get("pid", -1)) == os.getpid():
            lock_path.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return


def wait_for_rpc_ready(rpc_process: subprocess.Popen, port: int, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if rpc_process.poll() is not None:
            return False
        if can_connect_local_rpc(port):
            return True
        time.sleep(0.2)
    return can_connect_local_rpc(port)


def ensure_windows_firewall_rule(display_name: str, port: int) -> None:
    if os.name != "nt":
        return

    script = f"""
$ErrorActionPreference = 'Stop'
$name = '{display_name}'
$existing = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
if (-not $existing) {{
    New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP -LocalPort {int(port)} -Action Allow -Profile Any | Out-Null
}} else {{
    Set-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow -Profile Any | Out-Null
}}
"""
    run_powershell(script, elevated=not windows_is_admin(), timeout=60)
    logger.info("Ensured Windows firewall rule %s for TCP %s", display_name, port)


def dashboard_url(base_url: str) -> str:
    return f"{base_url}/dashboard"


def _download_model_to_primary(url: str, filename: str, hf_token: str | None, models_dir: Path | None) -> dict[str, Any]:
    if models_dir is None:
        raise RuntimeError("primary node has no models_dir configured")
    attempts = _download_retry_budget()
    last_error: DownloadTransferError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _download_model_once(url, filename, hf_token, models_dir)
        except DownloadTransferError as exc:
            last_error = exc
            if not exc.retryable or attempt >= attempts:
                raise
            backoff = _download_backoff_seconds(attempt)
            logger.warning(
                "download attempt %s/%s failed after %s; retrying in %.1fs: %s",
                attempt,
                attempts,
                _format_download_progress(exc.bytes_downloaded, exc.bytes_total),
                backoff,
                exc,
            )
            time.sleep(backoff)
        except httpx.TimeoutException as exc:
            last_error = DownloadTransferError(
                f"download timed out: {exc}",
                bytes_downloaded=0,
                bytes_total=None,
                retryable=True,
            )
            if attempt >= attempts:
                raise last_error from exc
            backoff = _download_backoff_seconds(attempt)
            logger.warning("download attempt %s/%s timed out; retrying in %.1fs", attempt, attempts, backoff)
            time.sleep(backoff)
        except httpx.TransportError as exc:
            last_error = DownloadTransferError(
                f"download transport error: {exc}",
                bytes_downloaded=0,
                bytes_total=None,
                retryable=True,
            )
            if attempt >= attempts:
                raise last_error from exc
            backoff = _download_backoff_seconds(attempt)
            logger.warning("download attempt %s/%s hit transport error; retrying in %.1fs", attempt, attempts, backoff)
            time.sleep(backoff)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            retryable = _download_is_retryable_http_status(status_code)
            last_error = DownloadTransferError(
                f"download failed with HTTP {status_code}",
                bytes_downloaded=0,
                bytes_total=None,
                retryable=retryable,
            )
            if not retryable or attempt >= attempts:
                raise last_error from exc
            backoff = _download_backoff_seconds(attempt)
            logger.warning("download attempt %s/%s got HTTP %s; retrying in %.1fs", attempt, attempts, status_code, backoff)
            time.sleep(backoff)
    if last_error is not None:
        raise last_error
    raise RuntimeError("download failed without a captured error")


def register_with_coordinator(client: httpx.Client, base_url: str, payload: dict) -> None:
    response = client.post(f"{base_url}/api/v1/nodes/register", json=payload)
    response.raise_for_status()
    logger.info("Registered with coordinator: %s", response.json())


def send_heartbeat(
    client: httpx.Client,
    base_url: str,
    node_id: str,
    last_latency_ms: float,
    rpc_endpoints: list[dict],
    models_dir: Path | None = None,
    engine_report: dict | None = None,
    command_result: dict | None = None,
) -> tuple[httpx.Response, float]:
    mem = psutil.virtual_memory()
    gpu_info = detect_gpu()
    capability = detect_device_capability(gpu_info)
    available_ram_mb = mem.available // (1024 * 1024)
    heartbeat_payload = {
        "node_id": node_id,
        "latency_ms": last_latency_ms,
        "available_ram_mb": available_ram_mb,
        "memory_pressure_state": _memory_pressure_state(capability["device_class"], available_ram_mb),
        "engine_state": "idle_worker",
        "loaded_model": None,
        "available_models": discover_available_models(models_dir) if models_dir else [],
        "available_model_metadata": discover_available_model_metadata(models_dir) if models_dir else {},
        "rpc_endpoints": rpc_endpoints,
        **capability,
    }
    heartbeat_payload.update(gpu_info)
    heartbeat_payload.update(engine_report or {})
    heartbeat_payload.update(command_result or {})
    
    t0 = time.time()
    response = client.post(f"{base_url}/api/v1/nodes/heartbeat", json=heartbeat_payload)
    round_trip_ms = (time.time() - t0) * 1000
    response.raise_for_status()
    
    return response, round_trip_ms


def heartbeat_or_reregister(
    client: httpx.Client,
    base_url: str,
    node_id: str,
    payload: dict,
    last_latency_ms: float,
    rpc_endpoints: list[dict],
    models_dir: Path | None = None,
    engine_report: dict | None = None,
    command_result: dict | None = None,
) -> tuple[dict | None, float]:
    try:
        response, round_trip_ms = send_heartbeat(
            client,
            base_url,
            node_id,
            last_latency_ms,
            rpc_endpoints,
            models_dir,
            engine_report,
            command_result,
        )
        body = response.json()
        command = body.get("command") if isinstance(body, dict) else None
        return command, round_trip_ms
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != UNKNOWN_NODE_STATUS:
            raise
        logger.warning("Coordinator no longer recognizes node %s; re-registering", node_id)
        register_with_coordinator(client, base_url, payload)
        return None, last_latency_ms


def execute_node_command(
    command: dict,
    engine_supervisor: EngineSupervisor,
    base_url: str | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any | None]:
    action = command.get("action")
    

    if command.get("action") == "download_model":
        command_id = str(command.get("command_id") or "")
        try:
            download_result = _download_model_to_primary(
                str(command.get("url") or ""),
                str(command.get("filename") or "model.gguf"),
                str(command.get("hf_token") or "").strip() or None,
                models_dir,
            )
            return {"command_id": command_id, "command_status": "ok", "command_error": None, **download_result}
        except DownloadTransferError as exc:
            progress = _format_download_progress(exc.bytes_downloaded, exc.bytes_total)
            return {
                "command_id": command_id,
                "command_status": "error",
                "command_error": f"{exc} ({progress})",
                "command_result_bytes_downloaded": exc.bytes_downloaded,
                "command_result_bytes_total": exc.bytes_total,
            }
        except Exception as exc:
            return {"command_id": command_id, "command_status": "error", "command_error": str(exc)}
    return engine_supervisor.execute(command)


def run_agent(args: argparse.Namespace, status_callback=None, stop_event: threading.Event | None = None) -> int:
    notify(status_callback, f"RAMDeck node build: {AGENT_BUILD_LABEL}")
    notify(status_callback, "Finding RAMDeck coordinator...")
    config = resolve_config(args)
    models_dir = resolve_models_dir(args, config)
    coordinator_ip = config["coordinator_ip"]
    coordinator_port = int(config["coordinator_port"])
    node_id = config["node_id"]
    rpc_port = int(config.get("rpc_port", args.rpc_port))
    
    # We will build payload later once we know if we can start both processes
    base_url = f"http://{coordinator_ip}:{coordinator_port}"
    coordinator_name = config.get("coordinator_name") or f"{coordinator_ip}:{coordinator_port}"
    notify(status_callback, f"Coordinator found: {coordinator_name}", dashboard_url(base_url))

    if args.dry_run:
        payload = build_registration_payload(
            coordinator_ip, coordinator_port, node_id,
            compute_live_rpc_endpoints(rpc_port, args.cpu_rpc_port, bool(args.cpu_rpc_binary)),
            models_dir
        )
        payload["rpc_port"] = rpc_port
        print(json.dumps({"config": config, "registration": payload}, indent=2))
        return 0
    if not args.skip_firewall_rule:
        notify(status_callback, f"Configuring Windows firewall for TCP {rpc_port}...")
        ensure_windows_firewall_rule(FIREWALL_RULE_RPC_DISPLAY_NAME, rpc_port)
        notify(status_callback, f"Firewall ready on Private/Domain networks for TCP {rpc_port}")

    rpc_binary = locate_rpc_binary(args.rpc_binary)
    notify(status_callback, f"Starting RAMDeck RPC worker on port {rpc_port}...")
    rpc_log_handle: TextIO | None = None
    rpc_log_path = Path(args.rpc_log_file)
    rpc_log_path.parent.mkdir(parents=True, exist_ok=True)
    instance_lock = acquire_agent_instance_lock(Path(args.config), node_id, rpc_port)
    if can_connect_local_rpc(rpc_port):
        release_agent_instance_lock(instance_lock)
        raise RuntimeError(
            f"RPC port {rpc_port} is already owned; refusing to start a duplicate RAMDeck node"
        )
    rpc_log_handle = rpc_log_path.open("a", buffering=1, encoding="utf-8", errors="replace")
    logger.info("RPC worker log file: %s", rpc_log_path)

    def start_and_verify_rpc(start_reason: str, bin_path: Path, port: int, log: TextIO) -> subprocess.Popen:
        notify(status_callback, start_reason)
        rpc_proc = start_rpc_process(bin_path, port, log)
        if not wait_for_rpc_ready(rpc_proc, port, args.rpc_startup_timeout):
            try:
                if rpc_proc.poll() is None:
                    rpc_proc.terminate()
            except Exception:
                pass
            raise RuntimeError(
                f"RPC worker failed preflight on 127.0.0.1:{port} within {args.rpc_startup_timeout:g}s"
            )
        notify(status_callback, f"RPC preflight passed on 127.0.0.1:{port}")
        return rpc_proc

    try:
        rpc_process = start_and_verify_rpc(f"Starting RAMDeck RPC worker on port {rpc_port}...", rpc_binary, rpc_port, rpc_log_handle)
    except BaseException:
        release_agent_instance_lock(instance_lock)
        rpc_log_handle.close()
        raise
        
    cpu_rpc_process = None
    cpu_rpc_log_handle = None
    cpu_rpc_binary = None
    
    cpu_rpc_binary_str = args.cpu_rpc_binary
    if not cpu_rpc_binary_str and is_windows_frozen():
        for base in bundled_binary_dirs():
            candidate = base / "cpu-ggml-rpc-server.exe"
            if candidate.exists():
                cpu_rpc_binary_str = str(candidate)
                break
            candidate = base / "bin" / "cpu-ggml-rpc-server.exe"
            if candidate.exists():
                cpu_rpc_binary_str = str(candidate)
                break
                
    if cpu_rpc_binary_str:
        try:
            cpu_rpc_binary = locate_rpc_binary(cpu_rpc_binary_str)
        except FileNotFoundError:
            logger.warning("Secondary CPU RPC binary not found at %s. Running single-endpoint.", cpu_rpc_binary_str)
    
    init_endpoints = compute_live_rpc_endpoints(rpc_port, args.cpu_rpc_port, True)
    if len(init_endpoints) > 1 and cpu_rpc_binary:
        cpu_rpc_log_path = Path(args.cpu_rpc_log_file)
        cpu_rpc_log_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_rpc_log_handle = cpu_rpc_log_path.open("a", buffering=1, encoding="utf-8", errors="replace")
        try:
            cpu_rpc_process = start_and_verify_rpc(f"Starting secondary CPU RPC worker on port {args.cpu_rpc_port}...", cpu_rpc_binary, args.cpu_rpc_port, cpu_rpc_log_handle)
        except BaseException as exc:
            logger.warning("Failed to start secondary CPU RPC process, will continue with single backend: %s", exc)
            cpu_rpc_process = None
            cpu_rpc_log_handle.close()
            
    payload = build_registration_payload(
        coordinator_ip, coordinator_port, node_id,
        compute_live_rpc_endpoints(rpc_port, args.cpu_rpc_port, cpu_rpc_process is not None),
        models_dir
    )
    payload["rpc_port"] = rpc_port
    restart_count = 0
    cpu_restart_count = 0
    rpc_health_monitor = RpcWorkerHealthMonitor(args.rpc_health_failure_threshold)
    cpu_rpc_health_monitor = RpcWorkerHealthMonitor(args.rpc_health_failure_threshold)
    engine_log_handle: TextIO | None = None
    try:
        engine_log_path = Path(args.llama_server_log_file)
        engine_log_path.parent.mkdir(parents=True, exist_ok=True)
        engine_log_handle = engine_log_path.open("a", buffering=1, encoding="utf-8", errors="replace")
        engine_supervisor = EngineSupervisor(
            locate_llama_server_binary(args.llama_server_binary),
            log_handle=engine_log_handle,
            health_timeout_sec=args.engine_start_timeout,
        )

        payload.update(engine_supervisor.report())
        client = httpx.Client(timeout=5.0)
    except BaseException:
        if rpc_process.poll() is None:
            rpc_process.terminate()
        if cpu_rpc_process is not None and cpu_rpc_process.poll() is None:
            cpu_rpc_process.terminate()
        if engine_log_handle is not None:
            engine_log_handle.close()
        rpc_log_handle.close()
        if cpu_rpc_log_handle is not None:
            cpu_rpc_log_handle.close()
        release_agent_instance_lock(instance_lock)
        raise
    opened_dashboard = False
    command_result: dict | None = None
    last_latency_ms = 0.0
    current_backoff = args.retry_backoff

    try:
        registered = False
        while not registered and (stop_event is None or not stop_event.is_set()):
            try:
                register_with_coordinator(client, base_url, payload)
                notify(status_callback, f"Node registered as {payload.get('hostname', node_id)} ({node_id})", dashboard_url(base_url))
                registered = True
                break
            except Exception as exc:
                if not args.no_discovery:
                    logger.warning("Cached/manual coordinator failed (%s); retrying LAN discovery", exc)
                    notify(status_callback, "Cached coordinator failed; retrying LAN discovery...")
                    discovered = discover_config(args.config, args.discovery_timeout, existing=config)
                    if discovered:
                        config = discovered
                        models_dir = resolve_models_dir(args, config)
                        coordinator_ip = config["coordinator_ip"]
                        coordinator_port = int(config["coordinator_port"])
                        node_id = config["node_id"]
                        base_url = f"http://{coordinator_ip}:{coordinator_port}"
                        payload = build_registration_payload(
                            coordinator_ip, coordinator_port, node_id,
                            compute_live_rpc_endpoints(rpc_port, args.cpu_rpc_port, cpu_rpc_process is not None),
                            models_dir
                        )
                        payload["rpc_port"] = rpc_port
                        payload.update(engine_supervisor.report())
                        continue
                    raise ConfigRequiredError("could not reach the configured coordinator and LAN discovery found no RAMDeck hub") from exc

                logger.warning("Coordinator registration failed (%s); will retry", exc)
                notify(status_callback, f"Coordinator unreachable; retrying in {args.retry_backoff:g}s")
                if args.once:
                    return 1
                if stop_event is None:
                    time.sleep(args.retry_backoff)
                else:
                    stop_event.wait(args.retry_backoff)

        if not registered:
            return 1

        if args.open_dashboard:
            webbrowser.open(dashboard_url(base_url))
            opened_dashboard = True
            notify(status_callback, "Dashboard opened in your browser", dashboard_url(base_url))

        while stop_event is None or not stop_event.is_set():
            rpc_health = rpc_health_monitor.check(rpc_process, rpc_port)
            if not rpc_health["healthy"] and rpc_health["restart_required"]:
                if restart_count >= args.rpc_max_restarts:
                    logger.error("RPC worker unhealthy and restart budget exhausted (%s)", args.rpc_max_restarts)
                    notify(status_callback, f"RPC worker unhealthy after {args.rpc_max_restarts} restarts; stopping RAMDeck node")
                    return 1
                restart_count += 1
                logger.warning(
                    "RPC worker restart attempt %s/%s: reason=%s worker_pid=%s port=%s poll=%s "
                    "tcp_error=%s consecutive_failures=%s",
                    restart_count,
                    args.rpc_max_restarts,
                    rpc_health["reason"],
                    rpc_health["worker_pid"],
                    rpc_health["port"],
                    rpc_health["poll"],
                    rpc_health["tcp_error"],
                    rpc_health["consecutive_failures"],
                )
                notify(status_callback, f"RPC worker unhealthy; restart {restart_count}/{args.rpc_max_restarts}...")
                wait_callback = time.sleep if stop_event is None else stop_event.wait
                rpc_process = restart_rpc_process_safely(
                    rpc_process,
                    rpc_port,
                    args.rpc_shutdown_timeout,
                    args.rpc_restart_backoff,
                    lambda: start_and_verify_rpc(f"Restarting RAMDeck RPC worker on port {rpc_port}...", rpc_binary, rpc_port, rpc_log_handle),
                    wait_callback,
                )
                rpc_health_monitor.reset()
            elif rpc_health["healthy"]:
                restart_count = 0

            if cpu_rpc_process is not None:
                cpu_rpc_health = cpu_rpc_health_monitor.check(cpu_rpc_process, args.cpu_rpc_port)
                if not cpu_rpc_health["healthy"] and cpu_rpc_health["restart_required"]:
                    if cpu_restart_count >= args.rpc_max_restarts:
                        logger.error("CPU RPC worker unhealthy and restart budget exhausted; dropping secondary backend")
                        cpu_rpc_process.terminate()
                        cpu_rpc_process = None
                        cpu_rpc_log_handle.close()
                    else:
                        cpu_restart_count += 1
                        logger.warning("CPU RPC worker restart attempt %s", cpu_restart_count)
                        wait_callback = time.sleep if stop_event is None else stop_event.wait
                        cpu_rpc_process = restart_rpc_process_safely(
                            cpu_rpc_process,
                            args.cpu_rpc_port,
                            args.rpc_shutdown_timeout,
                            args.rpc_restart_backoff,
                            lambda: start_and_verify_rpc(f"Restarting CPU RPC worker...", cpu_rpc_binary, args.cpu_rpc_port, cpu_rpc_log_handle),
                            wait_callback,
                        )
                        cpu_rpc_health_monitor.reset()
                elif cpu_rpc_health["healthy"]:
                    cpu_restart_count = 0

            try:
                command, next_latency_ms = heartbeat_or_reregister(
                    client,
                    base_url,
                    node_id,
                    payload,
                    last_latency_ms,
                    compute_live_rpc_endpoints(rpc_port, args.cpu_rpc_port, cpu_rpc_process is not None),
                    models_dir,
                    engine_supervisor.report(),
                    command_result,
                )
                last_latency_ms = next_latency_ms
                command_result = None
                if command:
                    action = str(command.get("action") or "")
                    # Long-running commands (engine transitions, downloads) can
                    # block the main loop; keep heartbeats flowing so the
                    # coordinator does not mark this node unreachable.
                    if action in {"start", "stop", "download_model"}:
                        command_result_holder: dict[str, dict] = {}
                        command_done = threading.Event()

                        def _run_command() -> None:
                            command_result_holder["value"] = execute_node_command(
                                command,
                                engine_supervisor,
                                base_url=base_url,
                                models_dir=models_dir,
                            )
                            command_done.set()

                        threading.Thread(target=_run_command, daemon=True).start()
                        if action in {"start", "stop"}:
                            pulse_every_sec = 1.0
                        else:
                            pulse_every_sec = max(1.0, min(float(args.heartbeat_interval), 4.0))
                        next_pulse = time.monotonic() + pulse_every_sec
                        while not command_done.wait(timeout=0.5):
                            now_monotonic = time.monotonic()
                            if now_monotonic < next_pulse:
                                continue
                            next_pulse = now_monotonic + pulse_every_sec
                            try:
                                # Send a pulse heartbeat only; do not fetch/act
                                # on new commands while one is in-flight.
                                send_heartbeat(
                                    client,
                                    base_url,
                                    node_id,
                                    time.time(),
                                    compute_live_rpc_endpoints(rpc_port, args.cpu_rpc_port, cpu_rpc_process is not None),
                                    models_dir,
                                    engine_supervisor.report(),
                                    None,
                                )
                            except Exception as pulse_exc:
                                logger.warning("Progress heartbeat during %s failed: %s", action, pulse_exc)

                        command_result = command_result_holder.get(
                            "value",
                            {
                                "command_id": str(command.get("command_id") or ""),
                                "command_status": "error",
                                "command_error": f"{action} command thread exited unexpectedly",
                            },
                        )
                    else:
                        command_result = execute_node_command(
                            command,
                            engine_supervisor,
                            base_url=base_url,
                            models_dir=models_dir,
                        )

                notify(status_callback, f"Online: contributing {payload['available_ram_mb']} MB RAM", dashboard_url(base_url))
                if args.open_dashboard and not opened_dashboard:
                    webbrowser.open(dashboard_url(base_url))
                    opened_dashboard = True
                
                # Reset exponential backoff on success
                current_backoff = args.retry_backoff
            except Exception as exc:
                if "Connection" in str(type(exc)) or "Timeout" in str(type(exc)):
                    logger.warning("Heartbeat failed (network issue): %s", exc)
                else:
                    logger.warning("Heartbeat failed: %s", exc)
                
                notify(status_callback, f"Heartbeat failed; retrying in {current_backoff:g}s")
                if not args.once:
                    if stop_event is None:
                        time.sleep(current_backoff)
                    else:
                        stop_event.wait(current_backoff)
                    # Exponential backoff up to 60 seconds
                    current_backoff = min(60.0, current_backoff * 1.5)
                    continue
            if args.once:
                return 0
            if stop_event is None:
                time.sleep(args.heartbeat_interval)
            else:
                stop_event.wait(args.heartbeat_interval)
        notify(status_callback, "Stopping RAMDeck node...")
        return 0
    finally:
        try:
            for cleanup in (

                engine_supervisor.stop,
                engine_log_handle.close,
                rpc_log_handle.close,
                client.close,
            ):
                try:
                    cleanup()
                except Exception as exc:
                    logger.warning("Node shutdown cleanup failed: %s", exc)
            try:
                if rpc_process.poll() is None:
                    rpc_process.terminate()
            except Exception as exc:
                logger.warning("RPC worker shutdown failed: %s", exc)
        finally:
            release_agent_instance_lock(instance_lock)


def should_use_windows_ui(args: argparse.Namespace) -> bool:
    return is_windows_frozen() and not args.no_ui


def create_tray_icon(open_status, open_dashboard, exit_app, extra_items: list[tuple[str, Callable[[], None]]] | None = None):
    try:
        import pystray
        from PIL import Image, ImageDraw
    except Exception as exc:
        logger.warning("Tray icon unavailable: %s", exc)
        return None

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 56, 56), radius=12, fill=(32, 84, 130, 255))
    draw.rectangle((18, 20, 46, 44), fill=(255, 255, 255, 255))
    draw.rectangle((23, 25, 41, 39), fill=(46, 125, 50, 255))
    draw.line((18, 48, 46, 48), fill=(255, 255, 255, 255), width=4)
    menu_items = [
        pystray.MenuItem("Show Status", lambda: open_status()),
        pystray.MenuItem("Open Dashboard", lambda: open_dashboard()),
    ]
    for label, callback in (extra_items or []):
        menu_items.append(pystray.MenuItem(label, lambda cb=callback: cb()))
    menu_items.append(pystray.MenuItem("Exit RAMDeck Node", lambda: exit_app()))
    menu = pystray.Menu(*menu_items)
    return pystray.Icon("RAMDeck Node", image, "RAMDeck Node", menu)


def run_windows_service_tray(args: argparse.Namespace) -> int:
    if os.name != "nt":
        return 1

    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:
        logger.error("Windows tray mode requires tkinter: %s", exc)
        return 1

    root = tk.Tk()
    root.withdraw()
    stop_event = threading.Event()

    def open_dashboard() -> None:
        url = windows_dashboard_url_from_config(args.config)
        if url:
            webbrowser.open(url)
            return
        messagebox.showwarning("RAMDeck Node", "Coordinator is not configured yet.")

    def show_status() -> None:
        state = windows_service_state(WINDOWS_SERVICE_NAME)
        messagebox.showinfo("RAMDeck Node", f"{WINDOWS_SERVICE_DISPLAY_NAME}: {state}")

    def start_node() -> None:
        try:
            detail = windows_service_control("start")
            logger.info(detail)
            messagebox.showinfo("RAMDeck Node", detail)
        except Exception as exc:
            messagebox.showerror("RAMDeck Node", f"Could not start service: {exc}")

    def stop_node() -> None:
        try:
            detail = windows_service_control("stop")
            logger.info(detail)
            messagebox.showinfo("RAMDeck Node", detail)
        except Exception as exc:
            messagebox.showerror("RAMDeck Node", f"Could not stop service: {exc}")

    tray_icon_holder: dict[str, Any] = {}

    def exit_app() -> None:
        stop_event.set()
        icon = tray_icon_holder.get("icon")
        if icon is not None:
            icon.stop()

    tray_icon = create_tray_icon(
        show_status,
        open_dashboard,
        exit_app,
        extra_items=[
            ("Start Node Service", start_node),
            ("Stop Node Service", stop_node),
        ],
    )
    if tray_icon is None:
        return 1

    tray_icon_holder["icon"] = tray_icon
    tray_icon.run()
    stop_event.set()
    try:
        root.destroy()
    except Exception:
        pass
    return 0


def run_windows_ui(args: argparse.Namespace) -> int:
    if is_windows_frozen() and not windows_is_admin():
        relaunch_windows_as_admin()
        return 0

    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog

    args.open_dashboard = True
    events: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stop_event = threading.Event()
    result = {"code": None, "error": None}

    root = tk.Tk()
    root.title(f"RAMDeck Node Setup ({AGENT_BUILD_LABEL})")
    root.geometry("520x300")
    root.resizable(False, False)

    title = ttk.Label(root, text="RAMDeck Node", font=("Segoe UI", 16, "bold"))
    title.pack(anchor="w", padx=18, pady=(18, 4))

    subtitle = ttk.Label(root, text=f"Preparing this Windows PC to contribute RAM on your LAN. Build {AGENT_BUILD_LABEL}.")
    subtitle.pack(anchor="w", padx=18, pady=(0, 16))

    status_var = tk.StringVar(value="Starting...")
    dashboard_var = tk.StringVar(value="")
    status_label = ttk.Label(root, textvariable=status_var, wraplength=470, justify="left")
    status_label.pack(anchor="w", padx=18, pady=(0, 10))

    progress = ttk.Progressbar(root, mode="indeterminate", length=470)
    progress.pack(anchor="w", padx=18, pady=(0, 14))
    progress.start(12)

    details = ttk.Label(root, textvariable=dashboard_var, foreground="#345995", wraplength=470, justify="left")
    details.pack(anchor="w", padx=18, pady=(0, 14))

    buttons = ttk.Frame(root)
    buttons.pack(fill="x", padx=18, pady=(8, 18), side="bottom")

    dashboard_button = ttk.Button(buttons, text="Open Dashboard", state="disabled", command=lambda: webbrowser.open(dashboard_var.get()))
    dashboard_button.pack(side="left")
    stop_button = ttk.Button(buttons, text="Exit Node", command=lambda: exit_app())
    stop_button.pack(side="right")

    def open_dashboard() -> None:
        url = dashboard_var.get()
        if url:
            webbrowser.open(url)

    def show_status() -> None:
        root.after(0, lambda: (root.deiconify(), root.lift(), root.focus_force()))

    def exit_app() -> None:
        stop_event.set()
        root.after(800, root.destroy)

    tray_icon = create_tray_icon(show_status, open_dashboard, exit_app)
    if tray_icon:
        threading.Thread(target=tray_icon.run, daemon=True).start()

    def status_callback(message: str, url: str | None = None) -> None:
        events.put((message, url))

    def ask_coordinator_ip() -> str | None:
        answer = {"value": None}
        done = threading.Event()

        def ask() -> None:
            root.deiconify()
            root.lift()
            root.focus_force()
            answer["value"] = simpledialog.askstring(
                "RAMDeck Coordinator",
                "RAMDeck could not auto-discover the coordinator.\n\nEnter the Mac coordinator IP address:",
                parent=root,
                initialvalue=args.coordinator_ip or "",
            )
            done.set()

        root.after(0, ask)
        done.wait()
        value = (answer["value"] or "").strip()
        return value or None

    def save_manual_coordinator_config(coordinator_ip: str) -> None:
        existing = {}
        if args.config.exists():
            try:
                existing = json.loads(args.config.read_text())
            except Exception:
                existing = {}
        existing.update({
            "coordinator_ip": coordinator_ip,
            "coordinator_port": args.coordinator_port,
            "node_id": existing.get("node_id") or str(uuid.uuid4())[:8],
            "rpc_port": existing.get("rpc_port") or args.rpc_port,
        })
        save_config(args.config, existing)

    def worker() -> None:
        try:
            args.no_discovery = True
            args.config = windows_config_path()
            if not args.config.exists():
                status_callback("Enter the Orange Pi coordinator IP to install RAMDeck.")
                coordinator_ip = ask_coordinator_ip()
                if not coordinator_ip:
                    raise RuntimeError("Coordinator IP is required to install RAMDeck")
                save_manual_coordinator_config(coordinator_ip)
                args.coordinator_ip = coordinator_ip
            config = load_or_create_config(args.config, args.coordinator_ip, args.coordinator_port)
            installed_exe = install_windows_app(config, status_callback=status_callback)
            if installed_exe:
                base_url = f"http://{config['coordinator_ip']}:{config['coordinator_port']}"
                status_callback("RAMDeck installation complete. The node service is running in the background. You can safely close this window.", dashboard_url(base_url))
                result["code"] = 0
                return
            attempts = 0
            while True:
                try:
                    result["code"] = run_agent(args, status_callback=status_callback, stop_event=stop_event)
                    break
                except ConfigRequiredError as exc:
                    attempts += 1
                    if attempts > 3:
                        raise
                    status_callback(f"{exc}. Enter the coordinator IP to continue.")
                    coordinator_ip = ask_coordinator_ip()
                    if not coordinator_ip:
                        raise RuntimeError("Coordinator IP is required to finish RAMDeck setup")
                    save_manual_coordinator_config(coordinator_ip)
                    args.coordinator_ip = coordinator_ip
                    status_callback(f"Saved coordinator {coordinator_ip}; continuing setup...")
        except Exception as exc:
            result["error"] = exc
            events.put((f"RAMDeck setup failed: {exc}", None))

    def pump() -> None:
        while True:
            try:
                message, url = events.get_nowait()
            except queue.Empty:
                break
            status_var.set(message)
            if url:
                dashboard_var.set(url)
                dashboard_button.configure(state="normal")
        if result["error"] is not None:
            progress.stop()
            messagebox.showerror("RAMDeck Node", str(result["error"]))
        elif result["code"] is not None:
            progress.stop()
            if not status_var.get().startswith("RAMDeck installation complete"):
                status_var.set("RAMDeck node stopped.")
        if result["code"] is None and result["error"] is None:
            root.after(250, pump)

    def on_close() -> None:
        root.withdraw()

    root.protocol("WM_DELETE_WINDOW", on_close)
    threading.Thread(target=worker, daemon=True).start()
    root.after(100, pump)
    if args.start_minimized:
        root.withdraw()
    root.mainloop()
    stop_event.set()
    if tray_icon:
        tray_icon.stop()
    return int(result["code"] or 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAMDeck contributor agent")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--coordinator-ip", default=DEFAULT_COORDINATOR_IP if is_windows_frozen() else None)
    parser.add_argument("--coordinator-port", type=int, default=DEFAULT_COORDINATOR_PORT)
    parser.add_argument("--rpc-binary")
    parser.add_argument("--rpc-log-file", default=str(default_rpc_log_path()))
    parser.add_argument("--rpc-port", type=int, default=DEFAULT_RPC_PORT)
    parser.add_argument("--cpu-rpc-binary")
    parser.add_argument("--cpu-rpc-log-file", default=str(app_dir() / "cpu-ggml-rpc-server.log"))
    parser.add_argument("--cpu-rpc-port", type=int, default=DEFAULT_CPU_RPC_PORT)
    parser.add_argument("--models-dir", type=Path, default=app_dir() / "models")
    parser.add_argument("--llama-server-binary")
    parser.add_argument("--llama-server-log-file", default=str(app_dir() / "llama-server.log"))
    parser.add_argument("--engine-start-timeout", type=float, default=180.0)
    parser.add_argument("--heartbeat-interval", type=float, default=DEFAULT_HEARTBEAT_INTERVAL_SEC)
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF_SEC)
    parser.add_argument("--rpc-startup-timeout", type=float, default=DEFAULT_RPC_STARTUP_TIMEOUT_SEC)
    parser.add_argument("--rpc-restart-backoff", type=float, default=DEFAULT_RPC_RESTART_BACKOFF_SEC)
    parser.add_argument("--rpc-max-restarts", type=int, default=DEFAULT_RPC_MAX_RESTARTS)
    parser.add_argument("--rpc-health-failure-threshold", type=int, default=DEFAULT_RPC_HEALTH_FAILURE_THRESHOLD)
    parser.add_argument("--rpc-shutdown-timeout", type=float, default=DEFAULT_RPC_SHUTDOWN_TIMEOUT_SEC)
    parser.add_argument("--discovery-timeout", type=float, default=5.0)
    parser.add_argument("--no-discovery", action="store_true", help="skip LAN discovery and use config/manual coordinator only")
    parser.add_argument("--skip-firewall-rule", action="store_true", help="do not create/update the Windows RPC firewall rule")
    parser.add_argument("--open-dashboard", action="store_true", help="open the coordinator dashboard after registration")
    parser.add_argument("--no-ui", action="store_true", help="run without the Windows status window")
    parser.add_argument("--start-minimized", action="store_true", help="start with only the Windows tray icon visible")
    parser.add_argument("--service-tray", action="store_true", help="run only the Windows tray controller for service start/stop")
    parser.add_argument("--dry-run", action="store_true", help="print config/registration payload and exit")
    parser.add_argument("--once", action="store_true", help="register, send one heartbeat, then exit")
    return parser.parse_args(argv)


def run():
    args = parse_args()
    try:
        if args.service_tray and is_windows_frozen():
            raise SystemExit(run_windows_service_tray(args))
        if should_use_windows_ui(args):
            raise SystemExit(run_windows_ui(args))
        raise SystemExit(run_agent(args))
    except ConfigRequiredError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    run()

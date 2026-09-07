"""Cross-host process ownership for the split diagnostic controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from typing import Any, Callable, IO, Sequence

from .split_diagnostic import SplitDiagnosticError


@dataclass(frozen=True)
class ProcessIdentity:
    host: str
    pid: int
    started_ns: int
    executable: str
    process_group_id: int


@dataclass(frozen=True)
class CleanupReceipt:
    identity: ProcessIdentity
    return_code: int | None
    graceful: bool
    forced: bool
    identity_absent: bool

    def validate(self) -> None:
        if self.return_code is None:
            raise SplitDiagnosticError(f"cleanup did not collect exit status for PID {self.identity.pid}")
        if not self.identity_absent:
            raise SplitDiagnosticError(f"cleanup could not prove PID identity {self.identity.pid} absent")


def _sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalOwnedProcess:
    def __init__(
        self,
        arguments: Sequence[str],
        *,
        stdout_path: str | Path,
        stderr_path: str | Path,
        expected_sha256: str,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        kill_group: Callable[[int, int], None] = os.killpg,
        group_id: Callable[[int], int] = os.getpgid,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.arguments = list(arguments)
        self.stdout_path = Path(stdout_path)
        self.stderr_path = Path(stderr_path)
        self.expected_sha256 = expected_sha256.lower()
        self.popen_factory = popen_factory
        self.kill_group = kill_group
        self.group_id = group_id
        self.clock_ns = clock_ns
        self.process: subprocess.Popen | None = None
        self.identity: ProcessIdentity | None = None
        self._stdout: IO[bytes] | None = None
        self._stderr: IO[bytes] | None = None

    def start(self) -> ProcessIdentity:
        if self.process is not None:
            raise SplitDiagnosticError("local process owner cannot be started twice")
        executable = str(Path(self.arguments[0]).expanduser().resolve())
        actual_hash = _sha256_file(executable)
        if actual_hash.lower() != self.expected_sha256:
            raise SplitDiagnosticError(
                f"local executable hash mismatch: expected {self.expected_sha256}, got {actual_hash}"
            )
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout = self.stdout_path.open("ab")
        self._stderr = self.stderr_path.open("ab")
        try:
            self.process = self.popen_factory(
                self.arguments,
                stdout=self._stdout,
                stderr=self._stderr,
                start_new_session=True,
            )
            self.identity = ProcessIdentity(
                host="mac",
                pid=int(self.process.pid),
                started_ns=self.clock_ns(),
                executable=executable,
                process_group_id=int(self.group_id(self.process.pid)),
            )
            return self.identity
        except Exception:
            self._close_logs()
            raise

    def stop(self, graceful_timeout: float = 5.0, kill_timeout: float = 5.0) -> CleanupReceipt:
        if self.process is None or self.identity is None:
            raise SplitDiagnosticError("local process owner was not started")
        graceful = False
        forced = False
        try:
            if self.process.poll() is None:
                self.kill_group(self.identity.process_group_id, signal.SIGTERM)
                try:
                    self.process.wait(timeout=graceful_timeout)
                    graceful = True
                except subprocess.TimeoutExpired:
                    forced = True
                    self.kill_group(self.identity.process_group_id, signal.SIGKILL)
                    try:
                        self.process.wait(timeout=kill_timeout)
                    except subprocess.TimeoutExpired as exc:
                        raise SplitDiagnosticError(
                            f"local PID {self.identity.pid} did not exit after SIGKILL"
                        ) from exc
            receipt = CleanupReceipt(
                identity=self.identity,
                return_code=self.process.poll(),
                graceful=graceful,
                forced=forced,
                identity_absent=self.process.poll() is not None,
            )
            receipt.validate()
            return receipt
        finally:
            self._close_logs()

    def _close_logs(self) -> None:
        for handle in (self._stdout, self._stderr):
            if handle is not None:
                handle.close()
        self._stdout = None
        self._stderr = None


_REMOTE_HELPER = r'''
import base64, json, os, signal, subprocess, sys, threading, time, urllib.request

cfg = json.loads(sys.argv[1])
creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
stdout_handle = open(cfg["stdout_path"], "ab", buffering=0)
stderr_handle = open(cfg["stderr_path"], "ab", buffering=0)
proc = subprocess.Popen(cfg["arguments"], stdout=stdout_handle, stderr=subprocess.PIPE, creationflags=creationflags)
identity = {
    "host": "tiny",
    "pid": proc.pid,
    "started_ns": time.time_ns(),
    "executable": os.path.abspath(cfg["arguments"][0]),
    "process_group_id": proc.pid,
}
print("RAMDECK_IDENTITY " + json.dumps(identity, separators=(",", ":")), flush=True)

stop_requested = threading.Event()
request_count = 0
def stream_stderr():
    if proc.stderr is None:
        return
    for raw in iter(proc.stderr.readline, b""):
        stderr_handle.write(raw)
        encoded = base64.b64encode(raw.rstrip(b"\r\n")).decode("ascii")
        print("RAMDECK_LOG " + encoded, flush=True)

def read_commands():
    global request_count
    try:
        for line in sys.stdin:
            command = line.strip()
            if command == "STOP":
                stop_requested.set()
                return
            if command.startswith("REQUEST "):
                request_count += 1
                if request_count != 1:
                    print("RAMDECK_RESPONSE " + json.dumps({"error":"more than one request attempted"}), flush=True)
                    continue
                request_cfg = json.loads(base64.b64decode(command[8:]).decode("utf-8"))
                try:
                    body = json.dumps(request_cfg["body"]).encode("utf-8")
                    request = urllib.request.Request(
                        request_cfg["url"], data=body, headers={"Content-Type":"application/json"}, method="POST"
                    )
                    with urllib.request.urlopen(request, timeout=float(request_cfg["timeout"])) as response:
                        response_body = response.read().decode("utf-8", errors="replace")
                        result = {"status": response.status, "body": response_body, "request_count": request_count}
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}", "request_count": request_count}
                print("RAMDECK_RESPONSE " + json.dumps(result, separators=(",", ":")), flush=True)
    finally:
        stop_requested.set()

stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
stderr_thread.start()
threading.Thread(target=read_commands, daemon=True).start()
graceful = False
forced = False
try:
    while proc.poll() is None and not stop_requested.wait(0.1):
        pass
finally:
    if proc.poll() is None:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=float(cfg["graceful_timeout"]))
            graceful = True
        except (subprocess.TimeoutExpired, OSError):
            forced = True
            proc.kill()
            try:
                proc.wait(timeout=float(cfg["kill_timeout"]))
            except subprocess.TimeoutExpired:
                pass
    stdout_handle.close()
    stderr_handle.close()
    receipt = {
        "identity": identity,
        "return_code": proc.poll(),
        "graceful": graceful,
        "forced": forced,
        "identity_absent": proc.poll() is not None,
    }
    print("RAMDECK_CLEANUP " + json.dumps(receipt, separators=(",", ":")), flush=True)
    if proc.poll() is None:
        sys.exit(70)
'''


class RemoteOwnedProcess:
    def __init__(
        self,
        *,
        ssh_host: str,
        python_executable: str,
        arguments: Sequence[str],
        stdout_path: str,
        stderr_path: str,
        expected_sha256: str,
        ssh_options: Sequence[str] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8"),
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.ssh_host = ssh_host
        self.python_executable = python_executable
        self.arguments = list(arguments)
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.expected_sha256 = expected_sha256.lower()
        self.ssh_options = list(ssh_options)
        self.popen_factory = popen_factory
        self.transport: subprocess.Popen | None = None
        self.identity: ProcessIdentity | None = None

    @property
    def helper_sha256(self) -> str:
        import hashlib

        return hashlib.sha256(_REMOTE_HELPER.encode("utf-8")).hexdigest()

    def build_ssh_arguments(self, graceful_timeout: float = 5.0, kill_timeout: float = 5.0) -> list[str]:
        encoded_helper = base64.b64encode(_REMOTE_HELPER.encode("utf-8")).decode("ascii")
        expression = f"exec(__import__('base64').b64decode('{encoded_helper}'))"
        config = json.dumps({
            "arguments": self.arguments,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "graceful_timeout": graceful_timeout,
            "kill_timeout": kill_timeout,
        }, separators=(",", ":"))
        return [
            "ssh", *self.ssh_options, self.ssh_host,
            self.python_executable, "-u", "-c", expression, config,
        ]

    def start(self, timeout: float = 10.0) -> ProcessIdentity:
        if self.transport is not None:
            raise SplitDiagnosticError("remote process owner cannot be started twice")
        self.transport = self.popen_factory(
            self.build_ssh_arguments(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        line = self._read_line(timeout)
        if not line.startswith("RAMDECK_IDENTITY "):
            raise SplitDiagnosticError(f"remote owner did not return process identity: {line!r}")
        payload = json.loads(line.removeprefix("RAMDECK_IDENTITY "))
        self.identity = ProcessIdentity(**payload)
        return self.identity

    def stop(self, timeout: float = 15.0) -> CleanupReceipt:
        if self.transport is None or self.identity is None or self.transport.stdin is None:
            raise SplitDiagnosticError("remote process owner was not started")
        self.transport.stdin.write("STOP\n")
        self.transport.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SplitDiagnosticError(f"remote cleanup receipt timed out for PID {self.identity.pid}")
            line = self._read_line(remaining)
            if not line.startswith("RAMDECK_CLEANUP "):
                continue
            payload = json.loads(line.removeprefix("RAMDECK_CLEANUP "))
            receipt_identity = ProcessIdentity(**payload.pop("identity"))
            if receipt_identity != self.identity:
                raise SplitDiagnosticError("remote cleanup receipt identity mismatch")
            receipt = CleanupReceipt(identity=receipt_identity, **payload)
            receipt.validate()
            self.transport.wait(timeout=max(0.1, deadline - time.monotonic()))
            return receipt

    def read_event(self, timeout: float) -> tuple[str, Any]:
        line = self._read_line(timeout)
        if line.startswith("RAMDECK_LOG "):
            raw = base64.b64decode(line.removeprefix("RAMDECK_LOG "))
            return "log", raw.decode("utf-8", errors="replace")
        if line.startswith("RAMDECK_RESPONSE "):
            return "response", json.loads(line.removeprefix("RAMDECK_RESPONSE "))
        if line.startswith("RAMDECK_CLEANUP "):
            return "cleanup", json.loads(line.removeprefix("RAMDECK_CLEANUP "))
        return "unknown", line

    def request_once(self, *, url: str, body: dict[str, Any], timeout: float) -> None:
        if self.transport is None or self.identity is None or self.transport.stdin is None:
            raise SplitDiagnosticError("remote process owner was not started")
        payload = base64.b64encode(json.dumps({
            "url": url,
            "body": body,
            "timeout": timeout,
        }, separators=(",", ":")).encode("utf-8")).decode("ascii")
        self.transport.stdin.write(f"REQUEST {payload}\n")
        self.transport.stdin.flush()

    def _read_line(self, timeout: float) -> str:
        if self.transport is None or self.transport.stdout is None:
            raise SplitDiagnosticError("remote transport stdout is unavailable")
        selector = selectors.DefaultSelector()
        try:
            selector.register(self.transport.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise SplitDiagnosticError("remote owner response timed out")
            line = self.transport.stdout.readline()
        finally:
            selector.close()
        if not line:
            stderr = ""
            if self.transport.stderr is not None:
                stderr = self.transport.stderr.read().strip()
            raise SplitDiagnosticError(f"remote owner exited without response: {stderr}")
        return line.strip()


def identity_manifest(identity: ProcessIdentity, *, executable_sha256: str, helper_sha256: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = asdict(identity)
    payload["executable_sha256"] = executable_sha256.lower()
    if helper_sha256 is not None:
        payload["helper_sha256"] = helper_sha256.lower()
    return payload
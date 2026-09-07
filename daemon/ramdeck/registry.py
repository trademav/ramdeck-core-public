"""
Node Registry - the live source of truth for cluster topology.

This is the heart of Layer 2 (Orchestration). It tracks every known device,
their heartbeat freshness, and computes the aggregate pooled capacity.

Design notes:
- Heartbeat timeout and degraded/unreachable thresholds are intentionally
  conservative given documented real-world instability in existing
  distributed-inference tooling (nodes flapping in/out of clusters).
- All state mutations go through this class so we have one place to add
  persistence, metrics, or alerting later without touching call sites.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import time
import threading
from typing import Any, Dict, List, Optional

from .models import Node, DeviceKind, ConnectionState, DeviceClass, EngineState, MemoryPressureState, TrainingBackend

from collections import deque
from dataclasses import dataclass, field

MAX_EVENTS_PER_NODE = 2000
HISTORY_RETENTION_SECONDS = 7 * 24 * 3600

@dataclass
class NodeHistoryEvent:
    ts: float
    state: str
    allocated_mb: float | None = None
    memory_pressure: str | None = None
    rpc_healthy: bool | None = None
    latency_ms: float | None = None

class NodeHistoryStore:
    def __init__(self):
        self._events: dict[str, deque[NodeHistoryEvent]] = {}

    def record(self, node_id: str, state: str, allocated_mb=None,
               memory_pressure=None, rpc_healthy=None, latency_ms=None):
        buf = self._events.setdefault(node_id, deque(maxlen=MAX_EVENTS_PER_NODE))
        buf.append(NodeHistoryEvent(time.time(), state, allocated_mb,
                                     memory_pressure, rpc_healthy, latency_ms))

    def get(self, node_id: str, since: float) -> list[NodeHistoryEvent]:
        buf = self._events.get(node_id, deque())
        return [e for e in buf if e.ts >= since]

node_history_store = NodeHistoryStore()

HEARTBEAT_DEGRADED_SEC = 20     # miss several beats -> degraded
HEARTBEAT_UNREACHABLE_SEC = 60  # miss many beats -> drop from pool
DEFAULT_STATE_DIR = Path(__file__).resolve().parents[2] / ".ramdeck"
REGISTRY_STATE_FILENAME = "registry_state.json"

logger = logging.getLogger("ramdeck.registry")


def default_state_path() -> Path:
    state_dir = Path(os.environ.get("RAMDECK_STATE_DIR", str(DEFAULT_STATE_DIR)))
    return state_dir / REGISTRY_STATE_FILENAME


class NodeRegistry:
    def __init__(self, state_path: Path | str | None = None):
        self._nodes: Dict[str, Node] = {}
        self._pending_reconfirmation: set[str] = set()
        self._lock = threading.RLock()
        self.state_path = Path(state_path) if state_path is not None else None
        self._load_state()

    def upsert(self, node: Node) -> None:
        with self._lock:
            existing = self._nodes.get(node.node_id)
            if existing is not None:
                # Preserve explicit operator policy across re-register events.
                node.inference_enabled = existing.inference_enabled
                node.cpu_offload_enabled = existing.cpu_offload_enabled
            node.last_heartbeat = time.time()
            node.state = ConnectionState.ONLINE
            self._nodes[node.node_id] = node
            self._pending_reconfirmation.discard(node.node_id)
            self._persist_state()

    def get(self, node_id: str) -> Optional[Node]:
        with self._lock:
            return self._nodes.get(node_id)

    def heartbeat(
        self,
        node_id: str,
        latency_ms: Optional[float] = None,
        available_ram_mb: Optional[int] = None,
        rpc_endpoints: list[dict[str, Any]] | None = None,
        memory_pressure_state: MemoryPressureState | None = None,
        engine_state: EngineState | None = None,
        loaded_model: str | None = None,
        available_models: list[str] | None = None,
        available_model_metadata: dict[str, dict[str, Any]] | None = None,
        training_capable: bool | None = None,
        training_backend: str | None = None,
        client_ip: str | None = None,
    ) -> bool:
        with self._lock:
            n = self._nodes.get(node_id)
            if not n:
                return False
            n.last_heartbeat = time.time()
            n.state = ConnectionState.ONLINE
            self._pending_reconfirmation.discard(node_id)
            if client_ip is not None and n.ip != client_ip:
                n.ip = client_ip
            if latency_ms is not None:
                n.latency_ms = latency_ms
            if available_ram_mb is not None:
                n.available_ram_mb = max(0, min(int(available_ram_mb), n.total_ram_mb))
            if rpc_endpoints is not None:
                n.rpc_endpoints = [dict(endpoint) for endpoint in rpc_endpoints]
            if engine_state is not None:
                n.engine_state = engine_state
            n.loaded_model = loaded_model
            if available_models is not None:
                n.available_models = list(dict.fromkeys(available_models))
            if available_model_metadata is not None:
                n.available_model_metadata = {str(path): dict(metadata) for path, metadata in available_model_metadata.items()}
            if training_capable is not None:
                n.training_capable_reported = bool(training_capable)
            if training_backend is not None:
                backend = str(training_backend).strip().lower()
                n.training_backend = backend if backend in {
                    TrainingBackend.MLX.value,
                    TrainingBackend.CUDA.value,
                    TrainingBackend.NONE.value,
                } else TrainingBackend.NONE.value
            if memory_pressure_state is not None:
                n.memory_pressure_state = memory_pressure_state
                if (
                    n.device_class == DeviceClass.ANDROID_MOBILE
                    and memory_pressure_state == MemoryPressureState.CRITICAL
                ):
                    n.state = ConnectionState.DEGRADED
                    n.last_heartbeat = min(
                        n.last_heartbeat,
                        time.time() - (HEARTBEAT_DEGRADED_SEC + 0.001),
                    )
            self._persist_state()
            
            node_history_store.record(
                node_id,
                n.state.value,
                allocated_mb=n.total_ram_mb - n.available_ram_mb,
                memory_pressure=n.memory_pressure_state.value if n.memory_pressure_state else None,
                rpc_healthy=None,
                latency_ms=n.latency_ms
            )
            return True

    def update_capability(
        self,
        node_id: str,
        device_class: DeviceClass | None = None,
        compute_label: str | None = None,
        compute_score: int | None = None,
        training_capable: bool | None = None,
        training_backend: str | None = None,
    ) -> bool:
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            if device_class is not None:
                node.device_class = device_class
            if compute_label:
                node.compute_label = compute_label
            if compute_score is not None:
                node.compute_score = compute_score
            if training_capable is not None:
                node.training_capable_reported = bool(training_capable)
            if training_backend is not None:
                backend = str(training_backend).strip().lower()
                node.training_backend = backend if backend in {
                    TrainingBackend.MLX.value,
                    TrainingBackend.CUDA.value,
                    TrainingBackend.NONE.value,
                } else TrainingBackend.NONE.value
            self._persist_state()
            return True

    def remove(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)
            self._pending_reconfirmation.discard(node_id)
            self._persist_state()

    def sweep_stale(self) -> List[str]:
        """Call periodically. Downgrades/drops nodes that stopped heartbeating.
        Returns list of node_ids whose state changed (for event emission)."""
        changed = []
        now = time.time()
        with self._lock:
            for node_id, n in list(self._nodes.items()):
                age = now - n.last_heartbeat
                if node_id in self._pending_reconfirmation:
                    if age > HEARTBEAT_UNREACHABLE_SEC:
                        self._pending_reconfirmation.discard(node_id)
                        n.state = ConnectionState.UNREACHABLE
                    continue
                prev_state = n.state
                if age > HEARTBEAT_UNREACHABLE_SEC:
                    n.state = ConnectionState.UNREACHABLE
                elif age > HEARTBEAT_DEGRADED_SEC:
                    n.state = ConnectionState.DEGRADED
                else:
                    n.state = ConnectionState.ONLINE
                if n.state != prev_state:
                    changed.append(node_id)
                    node_history_store.record(
                        node_id,
                        n.state.value,
                        allocated_mb=n.total_ram_mb - n.available_ram_mb,
                        memory_pressure=n.memory_pressure_state.value if n.memory_pressure_state else None,
                        latency_ms=n.latency_ms
                    )
            if changed:
                self._persist_state()
        return changed

    def mark_degraded(self, node_id: str) -> bool:
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            node.state = ConnectionState.DEGRADED
            node.last_heartbeat = min(node.last_heartbeat, time.time() - (HEARTBEAT_DEGRADED_SEC + 0.001))
            self._persist_state()
            return True

    def inference_ready_nodes(self, now: float | None = None) -> List[Node]:
        now = time.time() if now is None else now
        with self._lock:
            return [
                node for node in self._nodes.values()
                if node.inference_enabled
                if node.state == ConnectionState.ONLINE
                and now - node.last_heartbeat <= HEARTBEAT_DEGRADED_SEC
            ]

    def nodes_not_ready_for_inference(self, now: float | None = None) -> List[Node]:
        now = time.time() if now is None else now
        with self._lock:
            return [
                node for node in self._nodes.values()
                if node.inference_enabled
                if node.state != ConnectionState.UNREACHABLE
                and (
                    node.state != ConnectionState.ONLINE
                    or now - node.last_heartbeat > HEARTBEAT_DEGRADED_SEC
                )
            ]

    def set_inference_enabled(self, node_id: str, enabled: bool) -> bool:
        with self._lock:
            n = self._nodes.get(node_id)
            if not n:
                return False
            n.inference_enabled = enabled
            self._persist_state()
            return True

    def set_cpu_offload_enabled(self, node_id: str, enabled: bool) -> bool:
        with self._lock:
            n = self._nodes.get(node_id)
            if not n:
                return False
            n.cpu_offload_enabled = enabled
            self._persist_state()
            return True

    def capacity_breakdown_for(self, nodes: List[Node]) -> dict:
        vram_total = 0
        ram_total = 0
        for n in nodes:
            vram_total += n.gpu_capacity_mb()
            ram_total += n.cpu_capacity_mb()
        return {
            "vram_mb": vram_total,
            "system_ram_mb": ram_total,
            "total_mb": vram_total + ram_total,
        }

    def active_nodes(self) -> List[Node]:
        with self._lock:
            return [n for n in self._nodes.values()
                    if n.state != ConnectionState.UNREACHABLE]

    def training_ready_nodes(self, now: float | None = None) -> List[Node]:
        now = time.time() if now is None else now
        with self._lock:
            return [
                node for node in self._nodes.values()
                if node.training_capable
                and node.state == ConnectionState.ONLINE
                and now - node.last_heartbeat <= HEARTBEAT_DEGRADED_SEC
            ]

    def all_nodes(self) -> List[Node]:
        with self._lock:
            return list(self._nodes.values())

    def total_pooled_capacity_mb(self) -> int:
        return sum(n.contributed_capacity_mb() for n in self.active_nodes())

    def capacity_breakdown(self) -> dict:
        """Split pooled capacity into fast (VRAM) vs standard (system RAM) tiers.
        This distinction matters: bundled model-fit calculations should not
        treat all pooled memory as equally fast."""
        return self.capacity_breakdown_for(self.active_nodes())

    def llama_rpc_peer_list(self, port: int = 50052, exclude_node_ids: Optional[List[str]] = None) -> str:
        """Build the comma-separated host:port list passed to llama-server --rpc."""
        excluded = set(exclude_node_ids or [])
        peers = []
        for node in self.active_nodes():
            if node.node_id in excluded or node.kind == DeviceKind.RAMDECK_HUB:
                continue
            if not node.inference_enabled:
                continue
            if node.ip:
                peers.append(f"{node.ip}:{port}")
        return ",".join(peers)

    def _persist_state(self) -> None:
        if self.state_path is None:
            return
        payload = {
            "schema_version": 1,
            "nodes": [self._node_to_record(node) for node in self._nodes.values()],
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
            tmp_path.replace(self.state_path)
        except Exception as exc:
            logger.warning("failed to persist registry state to %s: %s", self.state_path, exc)

    def _load_state(self) -> None:
        if self.state_path is None:
            return
        if not self.state_path.exists():
            return
        try:
            raw = self.state_path.read_text().strip()
            if not raw:
                logger.warning("registry state file %s is empty; starting with empty registry", self.state_path)
                return
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
                logger.warning("registry state file %s is malformed; starting with empty registry", self.state_path)
                return
        except Exception as exc:
            logger.warning("failed to load registry state from %s: %s", self.state_path, exc)
            return

        now = time.time()
        for record in payload["nodes"]:
            try:
                node = self._node_from_record(record)
            except Exception as exc:
                logger.warning("skipping malformed registry node record: %s", exc)
                continue
            if now - node.last_heartbeat > HEARTBEAT_UNREACHABLE_SEC:
                continue
            node.state = ConnectionState.UNREACHABLE
            self._nodes[node.node_id] = node
            self._pending_reconfirmation.add(node.node_id)

    def _node_to_record(self, node: Node) -> dict:
        return {
            "node_id": node.node_id,
            "hostname": node.hostname,
            "ip": node.ip,
            "kind": node.kind.value,
            "total_ram_mb": node.total_ram_mb,
            "available_ram_mb": node.available_ram_mb,
            "gpu": {
                "vendor": node.gpu.vendor,
                "model": node.gpu.model,
                "vram_mb": node.gpu.vram_mb,
                "total_vram_mb": node.gpu.total_vram_mb,
                "driver_version": node.gpu.driver_version,
                "supported": node.gpu.supported,
            } if node.gpu else None,
            "os": node.os,
            "ramdeck_version": node.ramdeck_version,
            "device_class": node.device_class.value,
            "compute_label": node.compute_label,
            "compute_score": node.compute_score,
            "memory_pressure_state": node.memory_pressure_state.value,
            "inference_enabled": node.inference_enabled,
            "cpu_offload_enabled": getattr(node, "cpu_offload_enabled", True),
            "rpc_port": node.rpc_port,
            "engine_state": node.engine_state.value,
            "loaded_model": node.loaded_model,
            "available_models": list(node.available_models),
            "available_model_metadata": {path: dict(metadata) for path, metadata in node.available_model_metadata.items()},
            "training_capable_reported": node.training_capable_reported,
            "training_backend": node.effective_training_backend,
            "last_heartbeat": node.last_heartbeat,
            "state": node.state.value,
            "latency_ms": node.latency_ms,
        }

    def _node_from_record(self, record: dict) -> Node:
        from .models import GPUInfo

        gpu_record = record.get("gpu")
        gpu = None
        if gpu_record:
            gpu = GPUInfo(
                vendor=gpu_record["vendor"],
                model=gpu_record["model"],
                vram_mb=int(gpu_record["vram_mb"]),
                total_vram_mb=int(gpu_record.get("total_vram_mb", 0)),
                driver_version=gpu_record.get("driver_version"),
                supported=bool(gpu_record.get("supported", False)),
            )
        return Node(
            node_id=record["node_id"],
            hostname=record["hostname"],
            ip=record["ip"],
            kind=DeviceKind(record["kind"]),
            total_ram_mb=int(record["total_ram_mb"]),
            available_ram_mb=int(record["available_ram_mb"]),
            gpu=gpu,
            os=record.get("os", "unknown"),
            ramdeck_version=record.get("ramdeck_version", "unknown"),
            device_class=DeviceClass(record.get("device_class", DeviceClass.CPU_ONLY.value)),
            compute_label=record.get("compute_label", "unknown"),
            compute_score=record.get("compute_score", 0),
            memory_pressure_state=MemoryPressureState(record.get("memory_pressure_state", "unknown")),
            inference_enabled=record.get("inference_enabled", True),
            cpu_offload_enabled=record.get("cpu_offload_enabled", True),
            rpc_port=record.get("rpc_port", 50052),
            engine_state=EngineState(record.get("engine_state", EngineState.IDLE_WORKER.value)),
            loaded_model=record.get("loaded_model"),
            available_models=list(record.get("available_models") or []),
            available_model_metadata={
                str(path): dict(metadata)
                for path, metadata in dict(record.get("available_model_metadata") or {}).items()
                if isinstance(metadata, dict)
            },
            training_capable_reported=record.get("training_capable_reported"),
            training_backend=str(record.get("training_backend", TrainingBackend.NONE.value)),
            last_heartbeat=float(record.get("last_heartbeat", 0)),
            state=ConnectionState(record.get("state", ConnectionState.UNREACHABLE.value)),
            latency_ms=record.get("latency_ms"),
        )

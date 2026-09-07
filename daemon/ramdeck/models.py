"""
RAMDeck data models.
Defines the shape of a Node (a device on the network) and the Cluster
(the live, aggregated state of all connected nodes).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Optional


class DeviceKind(str, Enum):
    RAMDECK_HUB = "ramdeck_hub"      # the connector itself (coordinator)
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    PHONE = "phone"
    GPU_HOST = "gpu_host"


class ConnectionState(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"           # missed 1-2 heartbeats
    UNREACHABLE = "unreachable"     # missed threshold, dropped from active pool


class DeviceClass(str, Enum):
    APPLE_SILICON_MLX = "apple_silicon_mlx"
    CUDA = "cuda"
    CPU_ONLY = "cpu_only"
    ANDROID_MOBILE = "android_mobile"


class MemoryPressureState(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EngineState(str, Enum):
    PRIMARY = "primary"
    REBALANCING = "rebalancing"
    IDLE_WORKER = "idle_worker"
    OFFLINE = "offline"


class ModelType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


class TrainingBackend(str, Enum):
    MLX = "mlx"
    CUDA = "cuda"
    NONE = "none"


# Android nodes are intentionally hard-capped below the LMKD risk zone.
# Empirical reports place LMKD kills around 2-3 GB active process memory.
# We cap model-shard contribution at 1.5 GB to leave roughly >= 1 GB headroom
# for runtime overhead and KV/cache growth during generation.
ANDROID_SAFE_CAPACITY_MB = 1536


@dataclass
class GPUInfo:
    vendor: str            # "nvidia" | "amd" | "apple" | "none"
    model: str
    vram_mb: int
    total_vram_mb: int = 0
    driver_version: Optional[str] = None
    supported: bool = False   # whether it's on RAMDeck's tested compatibility matrix


@dataclass
class Node:
    node_id: str
    hostname: str
    ip: str
    kind: DeviceKind
    total_ram_mb: int
    available_ram_mb: int
    gpu: Optional[GPUInfo] = None
    os: str = "unknown"
    ramdeck_version: str = "unknown"
    device_class: DeviceClass = DeviceClass.CPU_ONLY
    compute_label: str = "unknown"
    compute_score: int = 0
    memory_pressure_state: MemoryPressureState = MemoryPressureState.UNKNOWN
    inference_enabled: bool = field(default=True)
    cpu_offload_enabled: bool = field(default=True)
    rpc_port: int = field(default=50052)
    rpc_endpoints: list[dict[str, Any]] = field(default_factory=list)
    engine_state: EngineState = EngineState.IDLE_WORKER
    loaded_model: Optional[str] = None
    available_models: list[str] = field(default_factory=list)
    available_model_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    training_capable_reported: Optional[bool] = None
    training_backend: str = TrainingBackend.NONE.value
    last_heartbeat: float = field(default_factory=time)
    state: ConnectionState = ConnectionState.ONLINE
    latency_ms: Optional[float] = None

    @property
    def training_capable(self) -> bool:
        if self.training_capable_reported is not None:
            return bool(self.training_capable_reported)
        return self.device_class in {DeviceClass.APPLE_SILICON_MLX, DeviceClass.CUDA}

    @property
    def effective_training_backend(self) -> str:
        backend = str(self.training_backend or "").strip().lower()
        if backend in {TrainingBackend.MLX.value, TrainingBackend.CUDA.value, TrainingBackend.NONE.value}:
            return backend
        if self.device_class == DeviceClass.APPLE_SILICON_MLX:
            return TrainingBackend.MLX.value
        if self.device_class == DeviceClass.CUDA:
            return TrainingBackend.CUDA.value
        return TrainingBackend.NONE.value

    def effective_available_ram_mb(self) -> int:
        if self.device_class == DeviceClass.ANDROID_MOBILE:
            return min(self.available_ram_mb, ANDROID_SAFE_CAPACITY_MB)
        return self.available_ram_mb

    def cpu_capacity_mb(self) -> int:
        if self.state == ConnectionState.UNREACHABLE:
            return 0
        if self.device_class == DeviceClass.APPLE_SILICON_MLX:
            return 0
        if self.device_class == DeviceClass.CUDA and not self.cpu_offload_enabled:
            return 0
        return self.effective_available_ram_mb()

    def gpu_capacity_mb(self) -> int:
        if self.state == ConnectionState.UNREACHABLE:
            return 0
        return self.gpu.vram_mb if (self.gpu and self.gpu.supported) else 0

    def contributed_capacity_mb(self) -> int:
        """What this node meaningfully contributes to the pool."""
        return self.cpu_capacity_mb() + self.gpu_capacity_mb()

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ip": self.ip,
            "kind": self.kind.value,
            "total_ram_mb": self.total_ram_mb,
            "available_ram_mb": self.available_ram_mb,
            "gpu": {
                "vendor": self.gpu.vendor,
                "model": self.gpu.model,
                "vram_mb": self.gpu.vram_mb,
                "total_vram_mb": self.gpu.total_vram_mb,
                "supported": self.gpu.supported,
            } if self.gpu else None,
            "os": self.os,
            "ramdeck_version": self.ramdeck_version,
            "device_class": self.device_class.value,
            "compute_label": self.compute_label,
            "compute_score": self.compute_score,
            "memory_pressure_state": self.memory_pressure_state.value,
            "inference_enabled": self.inference_enabled,
            "cpu_offload_enabled": self.cpu_offload_enabled,
            "rpc_port": self.rpc_port,
            "rpc_endpoints": list(self.rpc_endpoints),
            "engine_state": self.engine_state.value,
            "loaded_model": self.loaded_model,
            "available_models": list(self.available_models),
            "available_model_metadata": {path: dict(metadata) for path, metadata in self.available_model_metadata.items()},
            "training_backend": self.effective_training_backend,
            "training_capable": self.training_capable,
            "state": self.state.value,
            "latency_ms": self.latency_ms,
            "effective_available_ram_mb": self.effective_available_ram_mb(),
            "cpu_capacity_mb": self.cpu_capacity_mb(),
            "gpu_capacity_mb": self.gpu_capacity_mb(),
            "contributed_capacity_mb": self.contributed_capacity_mb(),
        }

"""
Layer 1: Transport & Discovery

Uses mDNS/Zeroconf to advertise this RAMDeck hub on the local network and
to discover companion-app clients / paired devices announcing themselves.

Service type: _ramdeck._tcp.local.
TXT record carries: version, role (hub|node), node_id.

Rationale: mDNS is the same zero-configuration mechanism used by Bonjour/
AirDrop/Chromecast-style discovery -- no manual IP entry, works out of the
box on macOS, Windows 10+, iOS, Android (with a small library), and Linux
(via Avahi, already installed in setup/01_provision_os.sh).
"""
from __future__ import annotations
from dataclasses import dataclass
import socket
import logging
import threading
import time
from typing import Callable, Optional

from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser, ServiceStateChange

logger = logging.getLogger("ramdeck.discovery")

SERVICE_TYPE = "_ramdeck._tcp.local."
PROTOCOL_VERSION = "ramdeck-0.1"


@dataclass(frozen=True)
class DiscoveredCoordinator:
    name: str
    ip: str
    port: int
    properties: dict

    @property
    def display_name(self) -> str:
        return self.properties.get("cluster_name") or self.properties.get("hostname") or self.name


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class RamdeckAdvertiser:
    """Advertises this hub's presence so companion apps / nodes can find it."""

    def __init__(self, node_id: str, port: int = 8420, version: str = "0.1.0", cluster_name: str | None = None):
        self.zeroconf = Zeroconf()
        self.node_id = node_id
        self.port = port
        self.version = version
        self.cluster_name = cluster_name or socket.gethostname()
        self._info: Optional[ServiceInfo] = None

    def build_service_info(self, ip: str) -> ServiceInfo:
        # Audit note: this advertiser is engine-agnostic. It publishes the
        # RAMDeck coordinator API endpoint, not inference engine internals.
        name = f"ramdeck-hub-{self.node_id}.{SERVICE_TYPE}"
        return ServiceInfo(
            SERVICE_TYPE,
            name,
            addresses=[socket.inet_aton(ip)],
            port=self.port,
            properties={
                "role": "hub",
                "node_id": self.node_id,
                "version": self.version,
                "protocol": PROTOCOL_VERSION,
                "api_port": str(self.port),
                "cluster_name": self.cluster_name,
                "hostname": socket.gethostname(),
            },
            server=f"ramdeck-hub-{self.node_id}.local.",
        )

    def start(self):
        ip = get_local_ip()
        self._info = self.build_service_info(ip)
        self.zeroconf.register_service(self._info)
        logger.info(f"Advertising RAMDeck hub {self.cluster_name} on {ip}:{self.port}")

    def stop(self):
        if self._info:
            self.zeroconf.unregister_service(self._info)
        self.zeroconf.close()


class NodeDiscoverer:
    """Browses for other RAMDeck-capable devices (companion app instances
    that are also advertising _ramdeck._tcp.local, e.g. a laptop running
    the RAMDeck contributor agent)."""

    def __init__(self, on_found: Callable[[str, str, int, dict], None],
                 on_lost: Callable[[str], None]):
        self.zeroconf = Zeroconf()
        self.on_found = on_found
        self.on_lost = on_lost
        self.browser: Optional[ServiceBrowser] = None

    def _handler(self, zeroconf, service_type, name, state_change):
        if state_change is ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                ip = socket.inet_ntoa(info.addresses[0]) if info.addresses else ""
                props = {k.decode(): v.decode() for k, v in info.properties.items()}
                self.on_found(name, ip, info.port, props)
        elif state_change is ServiceStateChange.Removed:
            self.on_lost(name)

    def start(self):
        self.browser = ServiceBrowser(self.zeroconf, SERVICE_TYPE, handlers=[self._handler])
        logger.info("Browsing for RAMDeck-capable devices on LAN...")

    def stop(self):
        self.zeroconf.close()


def decode_properties(properties: dict) -> dict:
    decoded = {}
    for key, value in properties.items():
        decoded_key = key.decode() if isinstance(key, bytes) else str(key)
        decoded_value = value.decode() if isinstance(value, bytes) else str(value)
        decoded[decoded_key] = decoded_value
    return decoded


def coordinator_from_service_info(name: str, info: ServiceInfo) -> DiscoveredCoordinator | None:
    properties = decode_properties(info.properties)
    if properties.get("role") != "hub":
        return None
    if not info.addresses:
        return None
    ip = socket.inet_ntoa(info.addresses[0])
    port = int(properties.get("api_port") or info.port)
    return DiscoveredCoordinator(name=name, ip=ip, port=port, properties=properties)


def discover_coordinators(timeout_sec: float = 5.0) -> list[DiscoveredCoordinator]:
    """Bounded mDNS browse for RAMDeck coordinators on the local LAN."""
    found: dict[str, DiscoveredCoordinator] = {}
    lock = threading.Lock()
    zeroconf = Zeroconf()

    def handler(zeroconf, service_type, name, state_change):
        if state_change is not ServiceStateChange.Added:
            return
        info = zeroconf.get_service_info(service_type, name)
        if not info:
            return
        coordinator = coordinator_from_service_info(name, info)
        if coordinator:
            with lock:
                found[name] = coordinator

    browser = ServiceBrowser(zeroconf, SERVICE_TYPE, handlers=[handler])
    try:
        time.sleep(timeout_sec)
        return list(found.values())
    finally:
        browser.cancel()
        zeroconf.close()

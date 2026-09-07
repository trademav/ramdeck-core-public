"""
Layer 4 bridge: talks to the local inference engine process.

The product path now targets llama.cpp's server and RPC backend. RAMDeck still
keeps inference out-of-process: this module only probes llama-server state and
issues OpenAI-compatible chat requests to the already-running server.
"""

from __future__ import annotations

from enum import Enum
import json
import ipaddress
import logging
import os
import socket
import subprocess
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from .reasoning_policy import apply_reasoning_request_policy

logger = logging.getLogger("ramdeck.inference_bridge")

DEFAULT_LLAMA_SERVER_BASE_URL = "http://127.0.0.1:8080"


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


class InferenceHealthState(str, Enum):
    HEALTHY = "healthy"
    UNREACHABLE = "unreachable"
    UNKNOWN_ERROR = "unknown_error"


class InferenceBridgeError(Exception):
    pass


class InferenceTimeoutError(InferenceBridgeError):
    pass


class InferenceConnectionError(InferenceBridgeError):
    pass


class InferenceServerError(InferenceBridgeError):
    pass


def _coerce_message_content(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(value)


def _sanitize_tool_call_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    function = entry.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False)
        except Exception:
            arguments = "{}"
    return {
        "id": str(entry.get("id") or "call"),
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _sanitize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue

        raw_role = str(message.get("role") or "user").strip().lower()
        # Copilot can emit developer role; llama-server accepts system semantics.
        role = "system" if raw_role == "developer" else raw_role
        if role == "function":
            role = "tool"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"

        normalized_message: dict[str, Any] = {"role": role}
        content = _coerce_message_content(message.get("content"))

        if role == "assistant":
            tool_calls_raw = message.get("tool_calls")
            if isinstance(tool_calls_raw, list):
                tool_calls = [
                    item for item in (_sanitize_tool_call_entry(entry) for entry in tool_calls_raw)
                    if item is not None
                ]
                if tool_calls:
                    normalized_message["tool_calls"] = tool_calls
                    if content is None:
                        content = ""

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                normalized_message["tool_call_id"] = tool_call_id

        if content is not None:
            normalized_message["content"] = content
        elif role == "assistant":
            normalized_message["content"] = ""
        elif role in {"user", "system", "tool"}:
            normalized_message["content"] = ""

        name = message.get("name")
        if isinstance(name, str) and name:
            normalized_message["name"] = name

        normalized.append(normalized_message)

    return normalized


def _sanitize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue

        normalized_function: dict[str, Any] = {"name": name}
        description = function.get("description")
        if isinstance(description, str) and description:
            normalized_function["description"] = description

        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            # Drop metadata keys that commonly cause strict schema parsers to reject.
            cleaned = {k: v for k, v in parameters.items() if k not in {"$schema", "strict"}}
            if "type" not in cleaned:
                cleaned["type"] = "object"
            normalized_function["parameters"] = cleaned

        normalized.append({
            "type": "function",
            "function": normalized_function,
        })

    return normalized or None


def _sanitize_tool_choice(tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        lowered = tool_choice.strip().lower()
        if lowered in {"auto", "none"}:
            return lowered
        # Some llama-server builds reject "required"; fall back to best-effort auto.
        return "auto"
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                return {
                    "type": "function",
                    "function": {"name": name},
                }
    return None


def _compat_fallback_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best-effort conversion to plain chat history for strict llama-server builds."""
    fallback: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "tool":
            # Tool result records are invalid when tools are removed.
            continue

        stripped = dict(message)
        stripped.pop("tool_calls", None)
        stripped.pop("tool_call_id", None)
        if role == "assistant" and "content" not in stripped:
            stripped["content"] = ""
        fallback.append(stripped)
    return fallback


def _compact_context_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shrink chat history aggressively to avoid n_ctx overflow.

    Keeps only the last system/developer instruction and the most recent user turn.
    If no user turn exists, keeps the tail message as a best-effort prompt.
    """
    if not messages:
        return []

    last_system: dict[str, Any] | None = None
    last_user: dict[str, Any] | None = None
    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            last_system = message
        elif role == "user":
            last_user = message

    compacted: list[dict[str, Any]] = []
    if last_system is not None:
        compacted.append(last_system)
    if last_user is not None:
        compacted.append(last_user)

    if compacted:
        return compacted

    tail = messages[-1]
    if tail.get("role") != "user":
        tail = {"role": "user", "content": str(tail.get("content") or "")}
    return [tail]

def _truncate_text_for_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    return (
        text[:keep_head]
        + "\n\n[... context truncated for model context limit ...]\n\n"
        + text[-keep_tail:]
    )


def _truncate_messages_for_context(messages: list[dict[str, Any]], max_chars: int = 7000) -> list[dict[str, Any]]:
    truncated: list[dict[str, Any]] = []
    for message in messages:
        cloned = dict(message)
        content = cloned.get("content")
        if isinstance(content, str):
            cloned["content"] = _truncate_text_for_context(content, max_chars)
        elif isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part_clone = dict(part)
                    part_clone["text"] = _truncate_text_for_context(part["text"], max_chars)
                    parts.append(part_clone)
                else:
                    parts.append(part)
            cloned["content"] = parts
        truncated.append(cloned)
    return truncated


class InferenceBridge:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 5.0,
        offline_mode: bool | None = None,
    ):
        self.base_url = self._resolve_base_url(base_url)
        self.client = httpx.Client(timeout=timeout)
        if offline_mode is None:
            offline_mode = os.environ.get("RAMDECK_OFFLINE_MODE", "1").lower() not in {"0", "false", "no"}
        self.offline_mode = offline_mode
        allowed_hosts_raw = os.environ.get("RAMDECK_OFFLINE_ALLOWED_HOSTS", "")
        self._offline_allowed_hosts = {
            host.strip().lower()
            for host in allowed_hosts_raw.split(",")
            if host.strip()
        }
        self._autostart_cmd = os.environ.get("RAMDECK_LLAMA_SERVER_AUTOSTART_CMD", "").strip()
        self._autostart_enabled = bool(self._autostart_cmd) and _bool_env("RAMDECK_LLAMA_SERVER_AUTOSTART", True)
        self._autostart_cooldown_sec = float(os.environ.get("RAMDECK_LLAMA_SERVER_AUTOSTART_COOLDOWN_SEC", "15"))
        self._autostart_wait_sec = float(os.environ.get("RAMDECK_LLAMA_SERVER_AUTOSTART_WAIT_SEC", "1.0"))
        self._last_autostart_attempt = 0.0
        self._autostart_lock = threading.Lock()
        self._max_completion_tokens_cap = max(1, int(os.environ.get("RAMDECK_MAX_COMPLETION_TOKENS", "1200")))
        self._router_mode_enabled = _bool_env("RAMDECK_LLAMA_ROUTER_MODE", False)

    def endpoint_summary(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "offline_mode": self.offline_mode,
            "autostart_enabled": self._autostart_enabled,
            "router_mode_enabled": self._router_mode_enabled,
        }

    def _ensure_router_mode(self, operation: str) -> None:
        if self._router_mode_enabled:
            return
        raise InferenceServerError(
            f"{operation} is disabled for classic llama-server instances launched with -m. "
            "Use process restart model switching or enable RAMDECK_LLAMA_ROUTER_MODE for --model-dir deployments."
        )

    def _resolve_base_url(self, explicit_base_url: str | None) -> str:
        if explicit_base_url and explicit_base_url.strip():
            return self._normalize_base_url(explicit_base_url)

        env_url = (
            os.environ.get("RAMDECK_LLAMA_SERVER_BASE_URL")
            or os.environ.get("RAMDECK_LLAMA_BASE_URL")
        )
        if env_url and env_url.strip():
            return self._normalize_base_url(env_url)

        host = os.environ.get("RAMDECK_LLAMA_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        port = (
            os.environ.get("RAMDECK_LLAMA_SERVER_PORT")
            or os.environ.get("LLAMA_SERVER_PORT")
            or "8080"
        )
        return f"http://{host}:{port}".rstrip("/")

    def _normalize_base_url(self, value: str) -> str:
        raw = value.strip()
        if not raw:
            return DEFAULT_LLAMA_SERVER_BASE_URL
        if "://" not in raw:
            raw = f"http://{raw}"
        return raw.rstrip("/")

    def _attempt_autostart(self) -> bool:
        if not self._autostart_enabled:
            return False

        with self._autostart_lock:
            now = time.time()
            if now - self._last_autostart_attempt < self._autostart_cooldown_sec:
                return False
            self._last_autostart_attempt = now

        try:
            subprocess.Popen(  # noqa: S602
                self._autostart_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.warning("llama-server autostart invoked via RAMDECK_LLAMA_SERVER_AUTOSTART_CMD")
            if self._autostart_wait_sec > 0:
                time.sleep(self._autostart_wait_sec)
            return True
        except Exception as exc:
            logger.warning("llama-server autostart failed: %s", exc)
            return False

    def _is_private_destination(self, host: str) -> bool:
        if not host:
            return False

        normalized = host.strip().lower()
        if normalized in {"localhost", "127.0.0.1", "::1"}:
            return True
        if normalized.endswith(".local") or normalized.endswith(".lan"):
            return True
        if normalized in self._offline_allowed_hosts:
            return True

        try:
            ip = ipaddress.ip_address(normalized)
            return bool(ip.is_private or ip.is_loopback or ip.is_link_local)
        except ValueError:
            pass

        # DNS lookups are best-effort; unknown hosts are denied in offline mode.
        try:
            infos = socket.getaddrinfo(normalized, None)
        except OSError:
            return False

        if not infos:
            return False

        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                return False
            if not (ip.is_private or ip.is_loopback or ip.is_link_local):
                return False
        return True

    def _enforce_offline_destination_policy(self) -> None:
        if not self.offline_mode:
            return
        parsed = urlparse(self.base_url)
        host = parsed.hostname or ""
        if self._is_private_destination(host):
            return
        raise InferenceServerError(
            "offline mode blocks non-private inference endpoint "
            f"'{host}'. Set RAMDECK_OFFLINE_ALLOWED_HOSTS or disable RAMDECK_OFFLINE_MODE if intentional."
        )

    def is_healthy(self) -> bool:
        return self.detailed_health() == InferenceHealthState.HEALTHY

    def detailed_health(self) -> InferenceHealthState:
        return self.get_engine_health()

    def get_engine_health(self) -> InferenceHealthState:
        try:
            self._enforce_offline_destination_policy()
            response = self.client.get(f"{self.base_url}/health")
            if response.status_code != 200:
                if self._attempt_autostart():
                    retry = self.client.get(f"{self.base_url}/health")
                    if retry.status_code == 200:
                        payload = retry.json()
                        if payload.get("status") == "ok":
                            return InferenceHealthState.HEALTHY
                return InferenceHealthState.UNREACHABLE
            payload = response.json()
            if payload.get("status") == "ok":
                return InferenceHealthState.HEALTHY
            return InferenceHealthState.UNKNOWN_ERROR
        except Exception as exc:
            logger.warning(f"llama-server health check failed: {exc}")
            if self._attempt_autostart():
                try:
                    retry = self.client.get(f"{self.base_url}/health")
                    if retry.status_code == 200 and retry.json().get("status") == "ok":
                        return InferenceHealthState.HEALTHY
                except Exception:
                    pass
            return InferenceHealthState.UNREACHABLE

    def get_topology(self) -> Optional[dict]:
        return self.get_engine_topology()

    def get_engine_topology(self) -> Optional[dict]:
        try:
            self._enforce_offline_destination_policy()
            props_response = self.client.get(f"{self.base_url}/props")
            props_response.raise_for_status()
            props = props_response.json()

            slots_response = self.client.get(f"{self.base_url}/slots")
            slots_response.raise_for_status()
            slots = slots_response.json()

            rpc_servers = [
                server.strip()
                for server in os.environ.get("RAMDECK_LLAMA_RPC_SERVERS", "").split(",")
                if server.strip()
            ]
            return {
                "engine": "llama.cpp",
                "model_alias": props.get("model_alias"),
                "model_path": props.get("model_path"),
                "model_ftype": props.get("model_ftype"),
                "build_info": props.get("build_info"),
                "total_slots": props.get("total_slots", len(slots) if isinstance(slots, list) else None),
                "slots": slots,
                "rpc_servers": rpc_servers,
            }
        except Exception as exc:
            logger.warning(f"failed to fetch llama-server topology: {exc}")
            return None

    def load_model(self, model_id: str) -> bool:
        """llama-server loads its model at process startup in this phase."""
        topology = self.get_engine_topology()
        if topology is None:
            return False
        loaded_model = topology.get("model_alias") or topology.get("model_path") or ""
        return model_id in loaded_model or self.is_healthy()

    def list_router_models(self) -> list[dict]:
        try:
            self._enforce_offline_destination_policy()
            response = self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                return payload["data"]
            if isinstance(payload, dict) and isinstance(payload.get("models"), list):
                return payload["models"]
            return []
        except Exception as exc:
            logger.warning("failed to list llama-server router models: %s", exc)
            return []

    def load_router_model(self, model: str) -> dict:
        self._ensure_router_mode("router model load")
        try:
            self._enforce_offline_destination_policy()
            response = self.client.post(f"{self.base_url}/models/load", json={"model": model})
            response.raise_for_status()
            return response.json() if response.content else {"status": "ok"}
        except httpx.TimeoutException as exc:
            raise InferenceTimeoutError("llama-server timed out while loading model") from exc
        except httpx.TransportError as exc:
            raise InferenceConnectionError("llama-server router model load connection failed") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise InferenceServerError(
                    "llama-server runtime model switching is unavailable on this build (/models/load returned 404). "
                    "Restart the coordinator with the target model or use a router-capable llama-server build."
                ) from exc
            raise InferenceServerError(f"llama-server model load returned HTTP {exc.response.status_code}") from exc

    def unload_router_model(self, model: str) -> dict:
        self._ensure_router_mode("router model unload")
        try:
            self._enforce_offline_destination_policy()
            response = self.client.post(f"{self.base_url}/models/unload", json={"model": model})
            response.raise_for_status()
            return response.json() if response.content else {"status": "ok"}
        except httpx.TimeoutException as exc:
            raise InferenceTimeoutError("llama-server timed out while unloading model") from exc
        except httpx.TransportError as exc:
            raise InferenceConnectionError("llama-server router model unload connection failed") from exc
        except httpx.HTTPStatusError as exc:
            raise InferenceServerError(f"llama-server model unload returned HTTP {exc.response.status_code}") from exc

    def chat_completion(
        self,
        model_id: str,
        messages: list,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        reasoning_mode: bool = False,
    ) -> Optional[dict]:
        self._enforce_offline_destination_policy()
        payload = {"model": model_id, "messages": _sanitize_messages(messages)}
        payload = apply_reasoning_request_policy(
            model_id=model_id,
            payload=payload,
            reasoning_mode=reasoning_mode,
        )
        if max_tokens is None:
            payload["max_tokens"] = self._max_completion_tokens_cap
        else:
            payload["max_tokens"] = min(max_tokens, self._max_completion_tokens_cap)
        sanitized_tools = _sanitize_tools(tools)
        if sanitized_tools is not None:
            payload["tools"] = sanitized_tools
        sanitized_tool_choice = _sanitize_tool_choice(tool_choice)
        if sanitized_tool_choice is not None:
            payload["tool_choice"] = sanitized_tool_choice
        request_url = f"{self.base_url}/v1/chat/completions"
        timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)

        def _post_or_raise(request_payload: dict[str, Any]) -> dict[str, Any]:
            response = self.client.post(
                request_url,
                json=request_payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()

        try:
            return _post_or_raise(payload)
        except httpx.TimeoutException as exc:
            logger.error("chat completion timed out after waiting for llama-server generation: %s", exc)
            raise InferenceTimeoutError("llama-server timed out during generation") from exc
        except httpx.ConnectError as exc:
            logger.error("chat completion failed because llama-server connection failed: %s", exc)
            raise InferenceConnectionError("llama-server connection failed during inference") from exc
        except httpx.TransportError as exc:
            logger.error("chat completion failed because llama-server transport failed: %s", exc)
            raise InferenceConnectionError("llama-server transport failed during inference") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body_preview = (exc.response.text or "").strip()
            if len(body_preview) > 1500:
                body_preview = f"{body_preview[:1500]}..."

            # Large agent prompts can overflow n_ctx even without tool envelopes.
            # Retry once with aggressively compacted history when overflow is explicit.
            overflow_hint = "exceeds the available context size"
            if status == 400 and overflow_hint in body_preview:
                compact_payload = dict(payload)
                compact_payload["messages"] = _compact_context_messages(payload.get("messages", []))
                try:
                    logger.warning(
                        "llama-server context overflow detected on initial request; retrying with compacted context"
                    )
                    return _post_or_raise(compact_payload)
                except httpx.HTTPStatusError as compact_exc:
                    compact_preview = (compact_exc.response.text or "").strip()
                    if len(compact_preview) > 1500:
                        compact_preview = f"{compact_preview[:1500]}..."
                    if compact_exc.response.status_code == 400 and overflow_hint in compact_preview:
                        truncated_payload = dict(compact_payload)
                        truncated_payload["messages"] = _truncate_messages_for_context(
                            compact_payload.get("messages", [])
                        )
                        try:
                            logger.warning(
                                "llama-server context overflow persisted after compaction; retrying with truncated message content"
                            )
                            return _post_or_raise(truncated_payload)
                        except httpx.HTTPStatusError as truncated_exc:
                            truncated_preview = (truncated_exc.response.text or "").strip()
                            if len(truncated_preview) > 1500:
                                truncated_preview = f"{truncated_preview[:1500]}..."
                            logger.error(
                                "chat completion truncated-context retry failed with llama-server HTTP status %s; initial_400_body=%s; compact_body=%s; truncated_body=%s",
                                truncated_exc.response.status_code,
                                body_preview,
                                compact_preview,
                                truncated_preview,
                            )
                            raise InferenceServerError(
                                f"llama-server returned HTTP {truncated_exc.response.status_code}"
                            ) from truncated_exc

                    compact_preview = (compact_exc.response.text or "").strip()
                    if len(compact_preview) > 1500:
                        compact_preview = f"{compact_preview[:1500]}..."
                    logger.error(
                        "chat completion compacted-context retry failed with llama-server HTTP status %s; initial_400_body=%s; compact_body=%s",
                        compact_exc.response.status_code,
                        body_preview,
                        compact_preview,
                    )
                    raise InferenceServerError(
                        f"llama-server returned HTTP {compact_exc.response.status_code}"
                    ) from compact_exc

            # Some llama-server builds reject tool metadata despite valid chat payloads.
            # Retry once as plain chat by removing tool envelopes/history.
            if status == 400 and (
                "tools" in payload
                or "tool_choice" in payload
                or any("tool_calls" in msg or msg.get("role") == "tool" for msg in payload.get("messages", []))
            ):
                fallback_payload = dict(payload)
                fallback_payload.pop("tools", None)
                fallback_payload.pop("tool_choice", None)
                fallback_payload["messages"] = _compat_fallback_messages(payload.get("messages", []))
                try:
                    logger.warning(
                        "llama-server returned HTTP 400; retrying chat completion without tools/tool history"
                    )
                    return _post_or_raise(fallback_payload)
                except httpx.HTTPStatusError as fallback_exc:
                    fallback_preview = (fallback_exc.response.text or "").strip()
                    if len(fallback_preview) > 1500:
                        fallback_preview = f"{fallback_preview[:1500]}..."

                    # Context overflows are common with long Copilot threads on small n_ctx.
                    # Make one final attempt with aggressively compacted history.
                    if fallback_exc.response.status_code == 400 and (
                        overflow_hint in body_preview or overflow_hint in fallback_preview
                    ):
                        compact_payload = dict(fallback_payload)
                        compact_payload["messages"] = _compact_context_messages(fallback_payload.get("messages", []))
                        try:
                            logger.warning(
                                "llama-server context overflow detected; retrying with compacted context"
                            )
                            return _post_or_raise(compact_payload)
                        except httpx.HTTPStatusError as compact_exc:
                            compact_preview = (compact_exc.response.text or "").strip()
                            if len(compact_preview) > 1500:
                                compact_preview = f"{compact_preview[:1500]}..."
                            logger.error(
                                "chat completion compacted-context fallback failed with llama-server HTTP status %s; initial_400_body=%s; fallback_body=%s; compact_body=%s",
                                compact_exc.response.status_code,
                                body_preview,
                                fallback_preview,
                                compact_preview,
                            )
                            raise InferenceServerError(
                                f"llama-server returned HTTP {compact_exc.response.status_code}"
                            ) from compact_exc

                    logger.error(
                        "chat completion fallback failed with llama-server HTTP status %s; initial_400_body=%s; fallback_body=%s",
                        fallback_exc.response.status_code,
                        body_preview,
                        fallback_preview,
                    )
                    raise InferenceServerError(
                        f"llama-server returned HTTP {fallback_exc.response.status_code}"
                    ) from fallback_exc
                except Exception as fallback_exc:
                    logger.error(
                        "chat completion fallback failed after initial HTTP 400; initial_400_body=%s; error=%s",
                        body_preview,
                        fallback_exc,
                    )
                    raise InferenceBridgeError("llama-server inference request failed") from fallback_exc

            logger.error(
                "chat completion failed with llama-server HTTP status %s: %s; response_body=%s",
                status,
                exc,
                body_preview,
            )
            raise InferenceServerError(f"llama-server returned HTTP {exc.response.status_code}") from exc
        except Exception as exc:
            logger.error(f"chat completion failed: {exc}")
            raise InferenceBridgeError("llama-server inference request failed") from exc
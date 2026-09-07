"""GGUF compatibility checks for RAMDeck model loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
import math
import struct
from typing import Any, BinaryIO

MIB = 1024 * 1024
DEFAULT_MIN_CONTEXT = 512
DEFAULT_SAFETY_MARGIN = 1.2
WEIGHTS_OVERHEAD = 1.1


class CompatibilityState(str, Enum):
    FITS = "fits"
    FITS_WITH_REDUCED_CONTEXT = "fits_with_reduced_context"
    DOES_NOT_FIT = "does_not_fit"


GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}

FILE_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    33: "Q4_0_4_4",
    34: "Q4_0_4_8",
    35: "Q4_0_8_8",
}


@dataclass(frozen=True)
class GGUFMetadata:
    path: str
    file_size_bytes: int
    architecture: str | None = None
    parameter_count: int | None = None
    quantization: str | None = None
    native_context_length: int | None = None
    block_count: int | None = None
    embedding_length: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CompatibilityReport:
    model_path: str
    state: CompatibilityState
    requested_context: int
    safe_context_size: int | None
    required_mb: int
    required_with_margin_mb: int
    effective_required_mb: int
    effective_required_with_margin_mb: int
    pooled_ram_mb: int
    metadata: GGUFMetadata
    suggestions: list[str]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["metadata"] = self.metadata.to_dict()
        return payload


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of GGUF header")
    return data


def _read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _read_scalar(handle: BinaryIO, value_type: int) -> Any:
    if value_type == 8:
        return _read_string(handle)
    fmt = SCALAR_FORMATS.get(value_type)
    if not fmt:
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))[0]


def _read_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type == 9:
        item_type = _read_u32(handle)
        count = _read_u64(handle)
        return [_read_scalar(handle, item_type) for _ in range(count)]
    return _read_scalar(handle, value_type)


def parse_gguf_metadata(model_path: str | Path) -> GGUFMetadata:
    path = Path(model_path).expanduser()
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = _read_u32(handle)
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version {version}")
        _read_u64(handle)  # tensor_count
        metadata_count = _read_u64(handle)
        raw_metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _read_string(handle)
            value_type = _read_u32(handle)
            if value_type not in GGUF_VALUE_TYPES:
                raise ValueError(f"unsupported GGUF metadata value type {value_type}")
            raw_metadata[key] = _read_value(handle, value_type)

    architecture = raw_metadata.get("general.architecture")
    file_type = raw_metadata.get("general.file_type")
    quantization = FILE_TYPE_NAMES.get(file_type, str(file_type) if file_type is not None else None)
    arch_prefix = f"{architecture}." if architecture else ""
    return GGUFMetadata(
        path=str(path),
        file_size_bytes=file_size,
        architecture=architecture,
        parameter_count=_maybe_int(raw_metadata.get("general.parameter_count")),
        quantization=quantization,
        native_context_length=_maybe_int(raw_metadata.get(f"{arch_prefix}context_length")),
        block_count=_maybe_int(raw_metadata.get(f"{arch_prefix}block_count")),
        embedding_length=_maybe_int(raw_metadata.get(f"{arch_prefix}embedding_length")),
    )


def check_model_compatibility_from_metadata(
    metadata: GGUFMetadata,
    pooled_ram_mb: int,
    requested_context: int | None = None,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    min_context: int = DEFAULT_MIN_CONTEXT,
) -> CompatibilityReport:
    requested = requested_context or metadata.native_context_length or 2048
    required_mb = estimate_required_mb(metadata, requested)
    required_with_margin_mb = math.ceil(required_mb * safety_margin)
    safe_ceiling = safe_context_ceiling(metadata, pooled_ram_mb, safety_margin)

    suggestions: list[str] = []
    if required_with_margin_mb <= pooled_ram_mb:
        state = CompatibilityState.FITS
        safe_context_size = requested
        effective_required_mb = required_mb
        effective_required_with_margin_mb = required_with_margin_mb
    elif safe_ceiling is not None and safe_ceiling >= min_context:
        safe_context_size = min(safe_ceiling, requested)
        state = CompatibilityState.FITS_WITH_REDUCED_CONTEXT
        effective_required_mb = estimate_required_mb(metadata, safe_context_size)
        effective_required_with_margin_mb = math.ceil(effective_required_mb * safety_margin)
        while safe_context_size > 0 and effective_required_with_margin_mb > pooled_ram_mb:
            safe_context_size -= 1
            effective_required_mb = estimate_required_mb(metadata, safe_context_size)
            effective_required_with_margin_mb = math.ceil(effective_required_mb * safety_margin)
        suggestions.append(f"Load at ctx-size {safe_context_size} instead of {requested}.")
    else:
        state = CompatibilityState.DOES_NOT_FIT
        safe_context_size = safe_ceiling
        effective_required_mb = required_mb
        effective_required_with_margin_mb = required_with_margin_mb
        missing_mb = max(0, required_with_margin_mb - pooled_ram_mb)
        suggestions.append(f"Add at least {missing_mb} MB of fresh node capacity.")
        if metadata.quantization and not metadata.quantization.startswith("Q4"):
            suggestions.append("Try a smaller Q4 quantization of this model.")
        if safe_ceiling and safe_ceiling > 0:
            suggestions.append(f"Advanced override only: context may need to be below {safe_ceiling}.")

    return CompatibilityReport(
        model_path=str(Path(metadata.path).expanduser()),
        state=state,
        requested_context=requested,
        safe_context_size=safe_context_size,
        required_mb=required_mb,
        required_with_margin_mb=required_with_margin_mb,
        effective_required_mb=effective_required_mb,
        effective_required_with_margin_mb=effective_required_with_margin_mb,
        pooled_ram_mb=pooled_ram_mb,
        metadata=metadata,
        suggestions=suggestions,
    )


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def estimate_required_mb(metadata: GGUFMetadata, ctx_size: int) -> int:
    weights_bytes = metadata.file_size_bytes * WEIGHTS_OVERHEAD
    kv_cache_bytes = estimate_kv_cache_bytes(metadata, ctx_size)
    return math.ceil((weights_bytes + kv_cache_bytes) / MIB)


def estimate_kv_cache_bytes(metadata: GGUFMetadata, ctx_size: int) -> int:
    if not metadata.block_count or not metadata.embedding_length:
        return 0
    return int(ctx_size * metadata.block_count * metadata.embedding_length * 2 * 2)


def safe_context_ceiling(metadata: GGUFMetadata, pooled_ram_mb: int, safety_margin: float = DEFAULT_SAFETY_MARGIN) -> int | None:
    kv_per_token = estimate_kv_cache_bytes(metadata, 1)
    if kv_per_token <= 0:
        return metadata.native_context_length
    usable_bytes = (pooled_ram_mb * MIB) / safety_margin
    weights_bytes = metadata.file_size_bytes * WEIGHTS_OVERHEAD
    remaining_bytes = usable_bytes - weights_bytes
    if remaining_bytes <= 0:
        return None
    return max(0, int(remaining_bytes // kv_per_token))


def check_model_compatibility(
    model_path: str | Path,
    pooled_ram_mb: int,
    requested_context: int | None = None,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    min_context: int = DEFAULT_MIN_CONTEXT,
) -> CompatibilityReport:
    return check_model_compatibility_from_metadata(
        parse_gguf_metadata(model_path),
        pooled_ram_mb=pooled_ram_mb,
        requested_context=requested_context,
        safety_margin=safety_margin,
        min_context=min_context,
    )

"""Deterministic, strict JSON conversion and SHA-256 helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import UUID

from .errors import JsonEncodingError


def jsonable(value: Any) -> Any:
    """Convert supported Python values into a strict JSON-compatible value.

    Mappings must use string keys. Dataclass instances, enums, paths, UUIDs,
    dates, and timezone-aware datetimes are supported in addition to ordinary
    JSON values. Sets, arbitrary objects, non-finite floats, naive datetimes,
    and dataclass *types* are rejected so that serialization cannot silently
    become non-deterministic.
    """

    if isinstance(value, Enum):
        return jsonable(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JsonEncodingError("Non-finite floats are not valid canonical JSON")
        return value
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise JsonEncodingError("Naive datetimes are not valid canonical JSON")
        normalized = value.astimezone(timezone.utc)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise JsonEncodingError(
                    f"JSON mapping keys must be strings, got {type(key).__name__}"
                )
            converted[key] = jsonable(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [jsonable(item) for item in value]
    raise JsonEncodingError(
        f"Unsupported JSON value of type {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Return the unique compact UTF-8 JSON representation of ``value``."""

    try:
        return json.dumps(
            jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, JsonEncodingError):
            raise
        raise JsonEncodingError(str(error)) from error


def canonical_json_bytes(value: Any) -> bytes:
    """Return :func:`canonical_json` encoded as UTF-8."""

    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return the lowercase SHA-256 digest of raw bytes."""

    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_digest(value: Any) -> str:
    """Hash raw bytes, or the canonical JSON representation of any other value.

    The dispatch is deliberately type-based: bytes-like inputs are hashed
    verbatim, while strings are JSON strings (including their JSON quotes).
    Use :func:`sha256_bytes` when hashing encoded text as raw bytes.
    """

    if isinstance(value, (bytes, bytearray, memoryview)):
        return sha256_bytes(value)
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | PurePath, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file and return its lowercase SHA-256 digest."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


# A concise alias for callers that describe a file digest by algorithm.
file_sha256 = sha256_file

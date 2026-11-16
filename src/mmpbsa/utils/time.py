"""UTC timestamp and collision-resistant identifier helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import uuid4

_ID_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def utc_now_datetime() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)


def utc_now() -> str:
    """Return the current time as an RFC 3339 UTC timestamp."""

    return utc_now_datetime().isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_id(prefix: str | None = None) -> str:
    """Return a sortable UTC-and-random identifier safe for path components."""

    if prefix is not None and not _ID_PREFIX.fullmatch(prefix):
        raise ValueError(
            "ID prefix must begin with an ASCII letter or digit and contain "
            "only letters, digits, '.', '_', or '-'"
        )
    timestamp = utc_now_datetime().strftime("%Y%m%dT%H%M%S%fZ")
    token = f"{timestamp}-{uuid4().hex}"
    return f"{prefix}-{token}" if prefix else token

"""Small numerical helpers for trajectory v3 geometry and audit digests."""

from __future__ import annotations

import hashlib
from typing import Any


def box_tuple(box: Any) -> tuple[float, ...]:
    return tuple(float(value) for row in box for value in row)


def box_volume(box: tuple[float, ...]) -> float:
    a, b, c, d, e, f, g, h, i = box
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def box_metric(box: tuple[float, ...]) -> tuple[float, ...]:
    rows = (box[0:3], box[3:6], box[6:9])
    return tuple(
        sum(left * right for left, right in zip(first, second, strict=True))
        for first in rows
        for second in rows
    )


def float_digest(values: tuple[float, ...]) -> str:
    payload = "\n".join(f"{value:.6f}" for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

"""Read-only GPU identity resolution and non-expanding CUDA visibility guards."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence

FULL_GPU_UUID_PATTERN = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


class GpuVisibilityError(ValueError):
    """A GPU identity or CUDA-visible authority cannot be proven safely."""


def canonical_gpu_uuid(value: str) -> str:
    if not isinstance(value, str) or not FULL_GPU_UUID_PATTERN.fullmatch(value):
        raise GpuVisibilityError(
            "GPU identity must be a full GPU UUID; MIG/prefix IDs are unsupported"
        )
    return "GPU-" + value[4:].lower()


def gpu_index_uuid_inventory() -> dict[str, str]:
    """Resolve physical nvidia-smi indices, not CUDA's remapped ordinals."""
    try:
        result = subprocess.run(
            ("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"),
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuVisibilityError("cannot verify GPU index/UUID inventory with nvidia-smi") from exc
    inventory = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isascii() or not fields[0].isdecimal():
            raise GpuVisibilityError("GPU index/UUID inventory is malformed")
        index, gpu_uuid = str(int(fields[0])), canonical_gpu_uuid(fields[1])
        if index in inventory or gpu_uuid in inventory.values():
            raise GpuVisibilityError("GPU index/UUID inventory is ambiguous")
        inventory[index] = gpu_uuid
    if not inventory:
        raise GpuVisibilityError("GPU index/UUID inventory is empty")
    return inventory


def normalize_gpu_ids(
    gpu_ids: Sequence[str],
    *,
    inventory: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    resolved = []
    for gpu_id in gpu_ids:
        if not isinstance(gpu_id, str) or gpu_id != gpu_id.strip() or not gpu_id:
            raise GpuVisibilityError("GPU IDs must be non-empty device strings")
        if gpu_id.isascii() and gpu_id.isdecimal():
            if inventory is None:
                inventory = gpu_index_uuid_inventory()
            gpu_id = inventory.get(str(int(gpu_id)), "")
            if not gpu_id:
                raise GpuVisibilityError("GPU index is absent from the verified physical inventory")
        resolved.append(canonical_gpu_uuid(gpu_id))
    if len(set(resolved)) != len(resolved):
        raise GpuVisibilityError("GPU device IDs resolve to duplicate physical UUIDs")
    return tuple(resolved)


def visible_gpu_ids(value: str | None) -> tuple[str, ...] | None:
    """None means unset; empty/-1 means no authorized GPU; invalid fails closed."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise GpuVisibilityError("CUDA_VISIBLE_DEVICES must be a string")
    if value.strip() in {"", "-1"}:
        return ()
    return normalize_gpu_ids(tuple(item.strip() for item in value.split(",")))


def require_gpu_subset(
    allocated: Sequence[str],
    visible: Sequence[str] | None,
    *,
    label: str,
) -> None:
    if visible is not None and not set(allocated) <= set(visible):
        raise GpuVisibilityError(
            f"GPU lease would expand the {label} CUDA_VISIBLE_DEVICES restriction"
        )

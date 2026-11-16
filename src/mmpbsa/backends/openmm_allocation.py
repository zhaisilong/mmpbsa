"""Adapter-owned resource evidence checks for native OpenMM checkpoint resume."""

from __future__ import annotations

from collections.abc import Mapping

from mmpbsa.core.models import StageRequest

from .errors import BackendConfigurationError


def validate_resume_allocation(
    request: StageRequest, previous_allocation: Mapping[str, object] | None
) -> None:
    current = getattr(request.allocation, "physical_gpu_ids", ())
    if not current:
        return  # Frozen local/CPU resume semantics are unchanged.
    previous = (previous_allocation or {}).get("physical_gpu_ids")
    if not isinstance(previous, (list, tuple)) or tuple(previous) != tuple(current):
        raise BackendConfigurationError(
            "OpenMM host checkpoint resume requires the original physical GPU UUID; "
            "a legacy ordinal or a different leased GPU is not sufficient"
        )

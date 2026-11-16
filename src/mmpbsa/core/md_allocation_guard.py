"""Adapter-owned resume allocation guard with read-only invocation evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .errors import ContractError
from .models import StageRequest


def previous_allocation(request: StageRequest) -> Mapping[str, object] | None:
    """Read the most recent recorded allocation in the same attempt, never infer IDs."""
    invocation_root = request.attempt_dir / "invocations"
    current_id = request.invocation_dir.name
    candidates = (
        sorted(
            (
                path
                for path in invocation_root.iterdir()
                if path.is_dir()
                and path.name.isdigit()
                and len(path.name) == 4
                and path.name < current_id
            ),
            reverse=True,
        )
        if invocation_root.is_dir()
        else []
    )
    for candidate in candidates:
        path = candidate / "invocation.json"
        if candidate.is_symlink() or path.is_symlink():
            raise ContractError(f"resume allocation evidence must not be symlinked: {path}")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read previous resume allocation: {path}") from exc
        if not isinstance(record, Mapping):
            raise ContractError(f"previous invocation must be a JSON object: {path}")
        runtime = record.get("runtime")
        if runtime is None:
            # A guard-rejected invocation never ran and has no runtime receipt.
            continue
        if not isinstance(runtime, Mapping):
            raise ContractError(f"previous invocation runtime is invalid: {path}")
        allocation = runtime.get("allocation")
        if allocation is None:
            # A runtime receipt without device evidence is authoritative
            # unknown evidence; do not borrow an older device assignment.
            return None
        if not isinstance(allocation, Mapping):
            raise ContractError(f"previous invocation allocation is invalid: {path}")
        return dict(allocation)
    return None


def validate_resume_allocation(service: object | None, request: StageRequest) -> None:
    """Core transports evidence; each adapter owns its checkpoint portability policy."""
    validate = getattr(service, "validate_resume_allocation", None)
    if request.resume and callable(validate):
        validate(request, previous_allocation(request))

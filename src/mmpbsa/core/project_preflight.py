"""Read-only dependency probes and disk facts for explicit project preflight."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from .errors import ContractError
from .runner import effective_stage_runtime
from .state import StageId


def inspect_execution_environment(
    workflow, runtime, providers, output_paths: Sequence[Path]
) -> dict[str, object]:
    probes = []
    errors = []
    stages = {"md": StageId.MD, "trajectory": StageId.TRAJECTORY}
    for capability, provider_id in sorted(runtime.providers.items()):
        try:
            service = providers.service(provider_id, capability)
            science = workflow.stage_science(stages[capability]) if capability in stages else {}
            compatibility = service.compatibility(
                effective_stage_runtime(runtime, capability, science)
            )
            probes.append(
                {
                    "capability": capability,
                    "provider_id": provider_id,
                    "compatibility": compatibility.as_dict(),
                }
            )
        except Exception as exc:
            errors.append(f"{capability}: {type(exc).__name__}: {exc}")
    if errors:
        raise ContractError("project dependency preflight failed: " + "; ".join(errors))
    disks = []
    for path in output_paths:
        existing = Path(path).resolve()
        while not existing.exists() and existing.parent != existing:
            existing = existing.parent
        usage = shutil.disk_usage(existing)
        if not usage.free:
            raise ContractError(f"no free disk space for project output: {path}")
        disks.append(
            {
                "output": str(path),
                "inspected_parent": str(existing),
                "free_bytes": usage.free,
                "total_bytes": usage.total,
            }
        )
    return {
        "backend_probed": True,
        "dependency_probes": probes,
        "disk": disks,
        "disk_note": "free-space facts only; final storage demand depends on system/frames",
    }

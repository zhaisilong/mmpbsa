"""Create a new workflow-only project from explicitly prepared systems/runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from mmpbsa.utils import atomic_write_text

from .campaign import CAMPAIGN_SCHEMA
from .config import load_runtime
from .errors import ContractError
from .models import require_identifier
from .prepared import load_prepared_case
from .protocols import available_protocol_presets

_CONTROLLER = '''#!/usr/bin/env python3
"""Explicit new-project launcher; never starts a simulation merely on import."""
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"  # before importing scientific libraries

from mmpbsa.core.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["project", *sys.argv[1:], "--campaign",
                          str(Path(__file__).resolve().parent / "campaign.yaml")]))
'''


def create_project(
    output_dir: Path,
    *,
    project_id: str,
    preset: str,
    cases: Mapping[str, Path],
    runtime_template: Path,
    registry_path: Path,
    seeds: tuple[int, int, int] = (41001, 41002, 41003),
) -> Mapping[str, Path]:
    """No chemistry guessing, hidden host budget or replacement of existing files."""
    output = Path(output_dir).resolve()
    require_identifier(project_id, "project_id")
    if preset not in available_protocol_presets():
        raise ContractError("project init requires an explicit supported production preset")
    if not cases or len(seeds) != 3 or len(set(seeds)) != 3 or any(seed < 1 for seed in seeds):
        raise ContractError("project requires cases and three distinct positive replica seeds")
    if not registry_path.is_absolute():
        raise ContractError("project registry path must be explicit and absolute")
    if output.exists():
        raise ContractError("project init refuses any existing destination")
    rows = []
    for case_id, path in cases.items():
        require_identifier(case_id, "case_id")
        prepared = load_prepared_case(path, expected_case_id=case_id)
        rows.append({"case_id": case_id, "prepared_case": str(prepared.manifest_path)})
    native_runtime = load_runtime(runtime_template)
    if native_runtime.providers.get("md") != "backends":
        raise ContractError(
            "new-project default is GROMACS; choose experimental OpenMM explicitly afterwards"
        )
    raw_runtime = yaml.safe_load(Path(runtime_template).read_text(encoding="utf-8"))
    scheduler = raw_runtime.setdefault("scheduler", {})
    scheduler.update(
        {
            "execution_mode": "pipeline",
            "cpu_affinity": "physical",
            "resource_coordination": {"mode": "host", "registry_path": str(registry_path)},
        }
    )
    workflow = {
        "schema_version": "mmpbsa.workflow.v3",
        "project_id": project_id,
        "protocol": {"preset": preset, "models": ["gb", "pb"], "completion_policy": "all"},
        "cases": rows,
        "replicas": [
            {"replica_id": f"rep{i:02d}", "replica_index": i, "seed": seed}
            for i, seed in enumerate(seeds, 1)
        ],
    }
    smoke = {
        **workflow,
        "project_id": project_id + "_smoke",
        "protocol": {**workflow["protocol"], "preset": "smoke_20ps@" + preset.rsplit("@", 1)[1]},
        "replicas": workflow["replicas"],
    }
    campaign = {
        "schema_version": CAMPAIGN_SCHEMA,
        "workflow": "workflow.yaml",
        "runtime": "runtime.yaml",
        "workspace": "_production",
        "smoke_workflow": "smoke.workflow.yaml",
        "smoke_workspace": "_smoke",
        "gate": "smoke_gate.json",
        "exports": "exports",
    }
    output.mkdir(parents=True, exist_ok=False)
    contents = {
        "workflow.yaml": yaml.safe_dump(workflow, sort_keys=False),
        "smoke.workflow.yaml": yaml.safe_dump(smoke, sort_keys=False),
        "runtime.yaml": yaml.safe_dump(raw_runtime, sort_keys=False),
        "campaign.yaml": yaml.safe_dump(campaign, sort_keys=False),
        "run_project.py": _CONTROLLER,
    }
    return {name: atomic_write_text(output / name, text) for name, text in contents.items()}

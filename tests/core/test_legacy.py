from __future__ import annotations

import json
from pathlib import Path

from mmpbsa.core.legacy import (
    inspect_legacy_workflow,
    inspect_legacy_workspace,
)


def test_legacy_inspection_is_summary_only_and_read_only(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        """
schema_version: mmpbsa.workflow.v2
project_id: old_project
protocol_id: old_protocol
profile: {factory: missing.py:factory}
cases: [{case_id: a}]
replicas: [{replica_id: rep01}]
""".lstrip(),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.json").write_text(
        json.dumps(
            {
                "schema_version": "mmpbsa.workspace.v1",
                "project_id": "old_project",
                "protocol_id": "old_protocol",
            }
        ),
        encoding="utf-8",
    )
    attempt = workspace / "cases/a/replicas/rep01/md/attempts/0001"
    attempt.mkdir(parents=True)
    (attempt / "state.json").write_text(
        json.dumps(
            {
                "schema_version": "mmpbsa.attempt.v1",
                "state": "completed",
                "task": {"stage": "md"},
            }
        ),
        encoding="utf-8",
    )
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    workflow_result = inspect_legacy_workflow(workflow)
    workspace_result = inspect_legacy_workspace(workspace)

    assert workflow_result["schema_version"] == "mmpbsa.workflow.v2"
    assert workflow_result["executable"] is False
    assert workspace_result["workspace_schema"] == "mmpbsa.workspace.v1"
    assert workspace_result["state_documents"] == 1
    assert workspace_result["stages"] == {"md": 1}
    assert workspace_result["writes"] is False
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before

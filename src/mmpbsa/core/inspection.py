"""Public read-only workspace inspection: no runner, locks, probes or pools."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from .config import WorkflowDefinition
from .errors import MmpbsaError
from .models import TaskKey
from .state import REPLICA_STAGES, AttemptState, QcVerdict, StageId
from .workspace import WORKSPACE_SCHEMA, Workspace


class WorkspaceReader:
    """Only expose workspace reads; constructing this view never initializes it."""

    task_dir = Workspace.task_dir
    _attempt_dirs = Workspace._attempt_dirs
    _read_json = staticmethod(Workspace._read_json)
    _task_from_mapping = staticmethod(Workspace._task_from_mapping)
    _read_summary = Workspace._read_summary
    latest = Workspace.latest
    verify_artifacts = Workspace.verify_artifacts

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()


def workflow_tasks(workflow: WorkflowDefinition) -> tuple[TaskKey, ...]:
    tasks: list[TaskKey] = []
    for case in workflow.cases:
        tasks.append(TaskKey(StageId.ADMIT, case_id=case.case_id))
        for replica in workflow.replicas:
            for stage in REPLICA_STAGES:
                models = workflow.scoring_model_ids if stage is StageId.SCORING else (None,)
                tasks.extend(
                    TaskKey(stage, case.case_id, replica.replica_id, model) for model in models
                )
        tasks.append(TaskKey(StageId.FINALIZE_CASE, case_id=case.case_id))
    tasks.extend(TaskKey(stage) for stage in (StageId.ANALYSIS, StageId.ASSETS, StageId.REPORT))
    return tuple(tasks)


def _record(reader: WorkspaceReader, task: TaskKey) -> dict[str, object]:
    attempts = reader.task_dir(task) / "attempts"
    # A half-written/corrupt latest attempt must not silently reveal an older one.
    observed = (
        sorted(
            p for p in attempts.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 4
        )
        if attempts.is_dir()
        else []
    )
    if observed and not (observed[-1] / "state.json").is_file():
        raise ValueError(f"latest attempt has no state.json: {observed[-1]}")
    latest = reader.latest(task)
    if latest is None:
        return {
            "task": task.label,
            "latest": None,
            "state": "not_started",
            "result_status": "not_started",
        }
    if (
        task.stage is StageId.TRAJECTORY_QC
        and latest.state is AttemptState.COMPLETED
        and latest.qc_verdict is None
    ):
        raise ValueError(f"completed trajectory_qc has no verdict: {task.label}")
    if latest.state is AttemptState.COMPLETED:
        status = "completed"
    elif latest.state is AttemptState.FAILED:
        status = "execution_failed"
    elif latest.state is AttemptState.INTERRUPTED:
        status = "cancelled"
    else:
        status = "running"
    attempt = {
        "task": task.label,
        "attempt_id": latest.attempt_id,
        "state": latest.state.value,
        "artifact_state": latest.artifact_state.value,
        "science_hash": latest.science_hash,
        "compatibility_key": latest.compatibility_key,
        "qc_verdict": latest.qc_verdict.value if latest.qc_verdict is not None else None,
        "artifacts": {key: value.as_dict() for key, value in latest.artifacts.items()},
    }
    return {**attempt, "latest": dict(attempt), "result_status": status}


def read_workspace_status(
    workflow: WorkflowDefinition, root: Path, *, summary: bool = False
) -> dict[str, object]:
    """Return readable missing/corrupt states, including per-model scientific skips."""
    reader = WorkspaceReader(root)
    errors: list[dict[str, str]] = []
    state = "ready"
    manifest: Mapping[str, object] = {}
    if not reader.root.exists():
        state = "missing"
    else:
        try:
            manifest = reader._read_json(reader.root / "workspace.json", "workspace manifest")
            if manifest.get("schema_version") != WORKSPACE_SCHEMA:
                raise ValueError("unsupported workspace schema")
            if (manifest.get("project_id"), manifest.get("protocol_id")) != (
                workflow.project_id,
                workflow.protocol_id,
            ):
                raise ValueError("workspace identity differs from requested workflow")
        except (MmpbsaError, OSError, ValueError) as exc:
            state = "corrupt"
            errors.append({"path": str(reader.root / "workspace.json"), "message": str(exc)})
    records: dict[TaskKey, dict[str, object]] = {}
    for task in workflow_tasks(workflow):
        try:
            records[task] = _record(reader, task)
        except (MmpbsaError, OSError, ValueError) as exc:
            state = "corrupt"
            records[task] = {
                "task": task.label,
                "latest": None,
                "state": "corrupt",
                "result_status": "unknown",
            }
            errors.append({"path": str(reader.task_dir(task)), "message": str(exc)})
    for task, record in records.items():
        if task.stage is not StageId.SCORING:
            continue
        qc = records[TaskKey(StageId.TRAJECTORY_QC, task.case_id, task.replica_id)]
        if qc.get("qc_verdict") == QcVerdict.BLOCK.value and record["state"] == "not_started":
            record["result_status"] = "skipped_qc_block"
    counts: dict[str, dict[str, int]] = {}
    for stage in (StageId.MD, StageId.TRAJECTORY_QC, StageId.SCORING):
        models = workflow.scoring_model_ids if stage is StageId.SCORING else (None,)
        for model in models:
            selected = [r for t, r in records.items() if t.stage is stage and t.model_id == model]
            counter = Counter(str(r["result_status"]) for r in selected)
            if stage is StageId.TRAJECTORY_QC:
                counter.update(str(r["qc_verdict"]) for r in selected if r.get("qc_verdict"))
            counts[model or stage.value] = {"total": len(selected), **dict(counter)}
    telemetry: object = None
    status_path = reader.root / "scheduler_status.json"
    if status_path.is_file():
        try:
            telemetry = reader._read_json(status_path, "scheduler telemetry")
        except (MmpbsaError, OSError, json.JSONDecodeError) as exc:
            errors.append({"path": str(status_path), "message": str(exc)})
    result = {
        "schema_version": "mmpbsa.workspace-status.v1",
        "workspace": str(reader.root),
        "workspace_state": state,
        "project_id": workflow.project_id,
        "protocol_id": workflow.protocol_id,
        "summary": counts,
        "errors": errors,
        "scheduler": telemetry,
        "inspection_semantics": "recorded_state_only_no_backend_or_artifact_freshness_probe",
    }
    if not summary:
        result["tasks"] = list(records.values())
    return result

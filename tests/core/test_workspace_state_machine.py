from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import mmpbsa.core.workspace as workspace_module
from mmpbsa.core.errors import (
    ActiveAttemptError,
    ArtifactIntegrityError,
    ContractError,
    ResumeError,
    RetryError,
    WorkspaceError,
)
from mmpbsa.core.models import BackendCompatibility, StageResult, TaskKey
from mmpbsa.core.state import AttemptState, StageId
from mmpbsa.core.workspace import Workspace


def _compat(
    *,
    major: str = "2024",
    version: str = "2024.1",
    contract: str = "md.v1",
) -> BackendCompatibility:
    return BackendCompatibility(
        provider_id="fake_science",
        adapter_contract=contract,
        engine="fake-md",
        engine_major=major,
        engine_version=version,
    )


def _workspace(tmp_path: Path) -> Workspace:
    return Workspace(
        tmp_path / "workspace",
        project_id="contract_tests",
        protocol_id="state_machine_v1",
    )


def test_task_keys_require_model_scope_only_for_scoring() -> None:
    scoring = TaskKey(
        StageId.SCORING,
        case_id="case01",
        replica_id="rep01",
        model_id="gb",
    )

    assert scoring.label == "scoring/case01/rep01/gb"
    with pytest.raises(
        ContractError,
        match="scoring is replica-and-model-scoped",
    ):
        TaskKey(
            StageId.SCORING,
            case_id="case01",
            replica_id="rep01",
        )
    with pytest.raises(ContractError, match="md is replica-scoped"):
        TaskKey(
            StageId.MD,
            case_id="case01",
            replica_id="rep01",
            model_id="gb",
        )
    with pytest.raises(ContractError, match="model_id"):
        TaskKey(
            StageId.SCORING,
            case_id="case01",
            replica_id="rep01",
            model_id="not/a/model",
        )


def test_scoring_models_have_independent_v2_attempt_paths(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    gb = TaskKey(
        StageId.SCORING,
        case_id="case01",
        replica_id="rep01",
        model_id="gb",
    )
    pb = TaskKey(
        StageId.SCORING,
        case_id="case01",
        replica_id="rep01",
        model_id="pb",
    )

    gb_reservation = workspace.reserve(
        gb,
        action="run",
        science_hash="1" * 64,
        compatibility=_compat(),
    )
    pb_reservation = workspace.reserve(
        pb,
        action="run",
        science_hash="2" * 64,
        compatibility=_compat(),
    )

    expected_root = (
        workspace.root / "cases/case01/replicas/rep01/scoring/models"
    )
    assert workspace.task_dir(gb) == expected_root / "gb"
    assert workspace.task_dir(pb) == expected_root / "pb"
    assert gb_reservation.summary.attempt_id == "0001"
    assert pb_reservation.summary.attempt_id == "0001"
    manifest = json.loads(
        (workspace.root / "workspace.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "mmpbsa.workspace.v2"
    state_path = gb_reservation.attempt_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["schema_version"] == "mmpbsa.attempt.v2"
    assert state["task"] == {
        "stage": "scoring",
        "case_id": "case01",
        "replica_id": "rep01",
        "model_id": "gb",
    }

    del state["task"]["model_id"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(
        WorkspaceError,
        match="invalid task key: missing model_id",
    ):
        workspace.latest(gb)


def test_workspace_rejects_task_identity_moved_between_model_paths(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    gb = TaskKey(
        StageId.SCORING,
        case_id="case01",
        replica_id="rep01",
        model_id="gb",
    )
    reservation = workspace.reserve(
        gb,
        action="run",
        science_hash="1" * 64,
        compatibility=_compat(),
    )
    state_path = reservation.attempt_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["task"]["model_id"] = "pb"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(
        WorkspaceError,
        match="attempt task differs from its workspace path",
    ):
        workspace.latest(gb)


def test_workspace_rejects_attempt_id_moved_between_attempt_paths(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(StageId.MD, case_id="case01", replica_id="rep01")
    reservation = workspace.reserve(
        task,
        action="run",
        science_hash="1" * 64,
        compatibility=_compat(),
    )
    state_path = reservation.attempt_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["attempt_id"] = "0002"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(
        WorkspaceError,
        match="attempt ID differs from its workspace path",
    ):
        workspace.latest(task)


def test_workspace_v2_rejects_a_v1_manifest(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "workspace.json").write_text(
        json.dumps(
            {
                "schema_version": "mmpbsa.workspace.v1",
                "project_id": "contract_tests",
                "protocol_id": "state_machine_v1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceError, match="unsupported workspace schema"):
        Workspace(
            root,
            project_id="contract_tests",
            protocol_id="state_machine_v1",
        )


def test_run_resume_and_reuse_preserve_one_active_attempt(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(StageId.MD, case_id="case01", replica_id="rep01")
    science_hash = "1" * 64
    original = workspace.reserve(
        task,
        action="run",
        science_hash=science_hash,
        compatibility=_compat(),
    )
    assert original.summary.state is AttemptState.RUNNING
    assert original.summary.attempt_id == "0001"

    with pytest.raises(ActiveAttemptError, match="explicit resume"):
        workspace.reserve(
            task,
            action="run",
            science_hash=science_hash,
            compatibility=_compat(),
        )
    with pytest.raises(ActiveAttemptError, match="still running"):
        workspace.reserve(
            task,
            action="resume",
            science_hash=science_hash,
            compatibility=_compat(),
        )

    checkpoint = original.attempt_dir / "output" / "restart.cpt"
    checkpoint.write_bytes(b"checkpoint-v1")
    interrupted = workspace.fail(
        original,
        RuntimeError("controlled interruption"),
        interrupted=True,
        checkpoint=Path("output/restart.cpt"),
    )
    assert interrupted.state is AttemptState.INTERRUPTED

    with pytest.raises(ResumeError, match="science inputs changed"):
        workspace.reserve(
            task,
            action="resume",
            science_hash="2" * 64,
            compatibility=_compat(),
        )
    with pytest.raises(ResumeError, match="major compatibility changed"):
        workspace.reserve(
            task,
            action="resume",
            science_hash=science_hash,
            compatibility=_compat(major="2025", version="2025.0"),
        )

    resumed = workspace.reserve(
        task,
        action="resume",
        science_hash=science_hash,
        compatibility=_compat(version="2024.9"),
    )
    assert resumed.resume is True
    assert resumed.summary.attempt_id == "0001"
    assert resumed.checkpoint == checkpoint.resolve()
    assert resumed.invocation_dir is not None
    assert resumed.invocation_dir.name == "0002"

    output = resumed.attempt_dir / "output" / "trajectory.xtc"
    payload = b"verified trajectory"
    output.write_bytes(payload)
    completed = workspace.complete(
        resumed,
        StageResult(
            artifacts={"trajectory": Path("output/trajectory.xtc")},
            payload={"frames": 51},
            invocation_provenance={"engine_version": "2024.9"},
        ),
    )
    assert completed.state is AttemptState.COMPLETED
    assert completed.artifacts["trajectory"].sha256 == hashlib.sha256(
        payload
    ).hexdigest()

    reused = workspace.reserve(
        task,
        action="run",
        science_hash=science_hash,
        compatibility=_compat(version="2024.10"),
    )
    assert reused.reused is True
    assert reused.invocation_dir is None
    assert reused.summary.attempt_id == "0001"
    with pytest.raises(ResumeError, match="no active attempt"):
        workspace.reserve(
            task,
            action="resume",
            science_hash=science_hash,
            compatibility=_compat(),
        )


def test_adapter_contract_bump_prevents_completed_attempt_reuse(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(StageId.MD, case_id="case01", replica_id="rep01")
    science_hash = "1" * 64
    original = workspace.reserve(
        task,
        action="run",
        science_hash=science_hash,
        compatibility=_compat(contract="md.v1"),
    )
    artifact = original.attempt_dir / "output" / "result.bin"
    artifact.write_bytes(b"old wire result")
    workspace.complete(
        original,
        StageResult(artifacts={"result": Path("output/result.bin")}),
    )

    with pytest.raises(
        RetryError,
        match="different science/backend compatibility",
    ):
        workspace.reserve(
            task,
            action="run",
            science_hash=science_hash,
            compatibility=_compat(contract="md.v2"),
        )

    replacement = workspace.reserve(
        task,
        action="new",
        science_hash=science_hash,
        compatibility=_compat(contract="md.v2"),
    )
    assert replacement.reused is False
    assert replacement.summary.attempt_id == "0002"


def test_retry_creates_attempt_and_new_rejects_live_owner(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(
        StageId.TRAJECTORY,
        case_id="case01",
        replica_id="rep01",
    )
    science_hash = "3" * 64
    first = workspace.reserve(
        task,
        action="run",
        science_hash=science_hash,
        compatibility=_compat(),
    )
    failed = workspace.fail(first, RuntimeError("backend failed"))
    assert failed.state is AttemptState.FAILED

    with pytest.raises(RetryError, match="explicit retry"):
        workspace.reserve(
            task,
            action="run",
            science_hash=science_hash,
            compatibility=_compat(),
        )
    with pytest.raises(RetryError, match="science inputs changed"):
        workspace.reserve(
            task,
            action="retry",
            science_hash="4" * 64,
            compatibility=_compat(),
        )
    with pytest.raises(RetryError, match="major compatibility changed"):
        workspace.reserve(
            task,
            action="retry",
            science_hash=science_hash,
            compatibility=_compat(major="2025", version="2025.0"),
        )

    retried = workspace.reserve(
        task,
        action="retry",
        science_hash=science_hash,
        compatibility=_compat(version="2024.8"),
    )
    assert retried.summary.attempt_id == "0002"
    assert retried.resume is False
    assert retried.summary.state is AttemptState.RUNNING

    with pytest.raises(ActiveAttemptError, match="owned by a live process"):
        workspace.reserve(
            task,
            action="new",
            science_hash="5" * 64,
            compatibility=_compat(major="2025", version="2025.0"),
        )
    assert workspace.latest(task).state is AttemptState.RUNNING

    workspace.fail(retried, RuntimeError("retry failed"))
    replacement = workspace.reserve(
        task,
        action="new",
        science_hash="5" * 64,
        compatibility=_compat(major="2025", version="2025.0"),
    )
    assert replacement.summary.attempt_id == "0003"
    assert [item.state for item in workspace.attempts(task)] == [
        AttemptState.FAILED,
        AttemptState.FAILED,
        AttemptState.RUNNING,
    ]


def test_completed_artifact_is_rehashed_before_reuse_and_new_is_explicit(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(
        StageId.TRAJECTORY_QC,
        case_id="case01",
        replica_id="rep02",
    )
    reservation = workspace.reserve(
        task,
        action="run",
        science_hash="6" * 64,
        compatibility=_compat(),
    )
    output = reservation.attempt_dir / "output" / "qc.json"
    output.write_bytes(b'{"verdict":"PASS"}')
    completed = workspace.complete(
        reservation,
        StageResult(artifacts={"qc": Path("output/qc.json")}),
    )
    assert workspace.verify_artifacts(completed)["qc"] == output.resolve()

    output.write_bytes(b'{"verdict":"FAIL"}')
    with pytest.raises(ArtifactIntegrityError, match="artifact changed"):
        workspace.verify_artifacts(completed)
    with pytest.raises(ArtifactIntegrityError, match="artifact changed"):
        workspace.reserve(
            task,
            action="run",
            science_hash="6" * 64,
            compatibility=_compat(version="2024.2"),
        )

    with pytest.raises(RetryError, match="use --new"):
        workspace.reserve(
            task,
            action="run",
            science_hash="7" * 64,
            compatibility=_compat(),
        )
    fresh = workspace.reserve(
        task,
        action="new",
        science_hash="7" * 64,
        compatibility=_compat(major="2025", version="2025.0"),
    )
    assert fresh.summary.attempt_id == "0002"
    assert [item.state for item in workspace.attempts(task)] == [
        AttemptState.COMPLETED,
        AttemptState.RUNNING,
    ]


def test_resume_recovers_only_an_abandoned_running_invocation(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(StageId.MD, case_id="case01", replica_id="rep01")
    reservation = workspace.reserve(
        task,
        action="run",
        science_hash="8" * 64,
        compatibility=_compat(),
    )
    checkpoint = reservation.attempt_dir / "output" / "restart.cpt"
    checkpoint.write_bytes(b"restart")
    state_path = reservation.attempt_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["checkpoint"] = {
        "path": "output/restart.cpt",
        "sha256": hashlib.sha256(b"restart").hexdigest(),
        "size": len(b"restart"),
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    invocation_path = reservation.invocation_dir / "invocation.json"
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    invocation["owner"] = {
        "hostname": invocation["owner"]["hostname"],
        "pid": 999_999_999,
        "process_start_ticks": "not-alive",
    }
    invocation_path.write_text(json.dumps(invocation), encoding="utf-8")

    resumed = workspace.reserve(
        task,
        action="resume",
        science_hash="8" * 64,
        compatibility=_compat(version="2024.9"),
    )

    assert resumed.invocation_dir.name == "0002"
    assert resumed.checkpoint == checkpoint.resolve()
    abandoned = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert abandoned["outcome"] == "abandoned"


def test_orphan_numeric_attempt_does_not_hide_future_committed_attempt(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(StageId.MD, case_id="case01", replica_id="rep01")
    orphan = workspace.task_dir(task) / "attempts" / "0001"
    (orphan / "output").mkdir(parents=True)

    reservation = workspace.reserve(
        task,
        action="run",
        science_hash="9" * 64,
        compatibility=_compat(),
    )

    assert reservation.summary.attempt_id == "0002"
    assert [item.attempt_id for item in workspace.attempts(task)] == ["0002"]
    assert orphan.is_dir()


def test_terminal_state_repairs_invocation_ledger_after_sync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(
        StageId.TRAJECTORY_QC,
        case_id="case01",
        replica_id="rep01",
    )
    reservation = workspace.reserve(
        task,
        action="run",
        science_hash="a" * 64,
        compatibility=_compat(),
    )
    output = reservation.attempt_dir / "output" / "qc.json"
    output.write_text('{"verdict":"PASS"}', encoding="utf-8")
    invocation_path = reservation.invocation_dir / "invocation.json"
    real_atomic_write = workspace_module.atomic_write_json
    failed_once = False

    def fail_terminal_ledger_sync(
        path: object, payload: object, *args: object, **kwargs: object
    ) -> object:
        nonlocal failed_once
        if (
            not failed_once
            and Path(path) == invocation_path
            and isinstance(payload, dict)
            and payload.get("outcome") == "completed"
        ):
            failed_once = True
            raise OSError("simulated crash after state commit")
        return real_atomic_write(path, payload, *args, **kwargs)

    monkeypatch.setattr(
        workspace_module,
        "atomic_write_json",
        fail_terminal_ledger_sync,
    )
    completed = workspace.complete(
        reservation,
        StageResult(
            artifacts={"qc": Path("output/qc.json")},
            invocation_provenance={"command": ["qc", "--strict"]},
        ),
    )
    assert completed.state is AttemptState.COMPLETED
    stale = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert stale["outcome"] == "running"
    assert stale["pending_terminal"]["outcome"] == "completed"

    reused = workspace.reserve(
        task,
        action="run",
        science_hash="a" * 64,
        compatibility=_compat(version="2024.2"),
    )

    assert reused.reused is True
    repaired = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert repaired["outcome"] == "completed"
    assert repaired["provenance"] == {"command": ["qc", "--strict"]}
    assert repaired["error"] is None
    assert repaired["pending_terminal"] is None


def test_resume_seals_invocation_published_before_state_commit(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(StageId.MD, case_id="case01", replica_id="rep01")
    original = workspace.reserve(
        task,
        action="run",
        science_hash="b" * 64,
        compatibility=_compat(),
    )
    workspace.fail(
        original,
        RuntimeError("controlled interruption"),
        interrupted=True,
    )
    orphan_dir = original.attempt_dir / "invocations" / "0002"
    orphan_dir.mkdir()
    orphan = json.loads(
        (original.invocation_dir / "invocation.json").read_text(
            encoding="utf-8"
        )
    )
    orphan.update(
        {
            "invocation_id": "0002",
            "outcome": "running",
            "finished_at": None,
            "pending_terminal": None,
            "owner": {
                "hostname": orphan["owner"]["hostname"],
                "pid": 999_999_999,
                "process_start_ticks": "not-alive",
            },
        }
    )
    (orphan_dir / "invocation.json").write_text(
        json.dumps(orphan), encoding="utf-8"
    )

    resumed = workspace.reserve(
        task,
        action="resume",
        science_hash="b" * 64,
        compatibility=_compat(version="2024.7"),
    )

    assert resumed.invocation_dir.name == "0003"
    recovered = json.loads(
        (orphan_dir / "invocation.json").read_text(encoding="utf-8")
    )
    assert recovered["outcome"] == "abandoned_before_state_commit"


def test_workspace_initialization_serializes_competing_identities(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared-workspace"
    barrier = threading.Barrier(2)

    def open_workspace(project_id: str) -> tuple[str, bool]:
        barrier.wait()
        try:
            Workspace(
                root,
                project_id=project_id,
                protocol_id="shared_protocol",
            )
        except WorkspaceError:
            return project_id, False
        return project_id, True

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(open_workspace, ("project_a", "project_b"))
        )

    winners = [project_id for project_id, succeeded in results if succeeded]
    assert len(winners) == 1
    manifest = json.loads(
        (root / "workspace.json").read_text(encoding="utf-8")
    )
    assert manifest["project_id"] == winners[0]


def test_prune_is_dry_run_by_default_and_limited_to_report_outputs(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    task = TaskKey(StageId.REPORT)
    for index in range(3):
        reservation = workspace.reserve(
            task,
            action="new",
            science_hash=f"{index + 1:x}" * 64,
            compatibility=_compat(),
        )
        artifact = reservation.attempt_dir / "output" / "index.html"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"report {index}\n", encoding="utf-8")
        workspace.complete(
            reservation,
            StageResult(artifacts={"report_index": artifact}),
        )

    planned = workspace.prune_attempts(task, keep=1)
    assert [record.attempt_id for record in planned] == ["0001", "0002"]
    assert all(not record.deleted and record.path.is_dir() for record in planned)

    deleted = workspace.prune_attempts(task, keep=1, confirm=True)
    assert [record.attempt_id for record in deleted] == ["0001", "0002"]
    assert all(record.deleted and not record.path.exists() for record in deleted)
    assert workspace.latest(task).attempt_id == "0003"

    with pytest.raises(WorkspaceError, match="limited"):
        workspace.prune_attempts(
            TaskKey(StageId.MD, case_id="case01", replica_id="rep01")
        )

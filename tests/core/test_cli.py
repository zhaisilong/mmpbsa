from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from mmpbsa.core import cli
from mmpbsa.core.models import BackendCompatibility
from mmpbsa.core.state import StageId


class _Workflow:
    schema_version = "mmpbsa.workflow.v3"
    project_id = "project"
    scoring_model_ids = ("gb", "pb")
    cases = (SimpleNamespace(case_id="case_a"),)
    replicas = (SimpleNamespace(replica_id="rep01"),)

    @staticmethod
    def case(case_id: str) -> None:
        if case_id != "case_a":
            raise ValueError(case_id)

    @staticmethod
    def replica(replica_id: str) -> None:
        if replica_id != "rep01":
            raise ValueError(replica_id)


def test_job_plan_emits_exact_four_field_v2_task_keys(
    monkeypatch, capsys
) -> None:
    runtime = SimpleNamespace(
        scheduler=SimpleNamespace(
            max_workers=1,
            cpu_slots=8,
            gpu_ids=("2",),
        )
    )
    monkeypatch.setattr(cli, "_load", lambda _arguments: (_Workflow(), runtime))
    arguments = Namespace(
        case_id="case_a",
        replica_id="rep01",
        model_ids=["pb"],
        through="scoring",
    )

    assert cli._plan_jobs(arguments, batch=False) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["jobs"][0]["tasks"] == [
        {
            "stage": "admit",
            "case_id": "case_a",
            "replica_id": None,
            "model_id": None,
        },
        {
            "stage": "md",
            "case_id": "case_a",
            "replica_id": "rep01",
            "model_id": None,
        },
        {
            "stage": "trajectory",
            "case_id": "case_a",
            "replica_id": "rep01",
            "model_id": None,
        },
        {
            "stage": "trajectory_qc",
            "case_id": "case_a",
            "replica_id": "rep01",
            "model_id": None,
        },
        {
            "stage": "scoring",
            "case_id": "case_a",
            "replica_id": "rep01",
            "model_id": "pb",
        },
    ]


def test_workspace_inspect_accepts_read_only_target_paths() -> None:
    arguments = cli._parser().parse_args(
        [
            "workspace",
            "inspect",
            "--legacy-workspace",
            "/tmp/workspace",
            "--legacy-workflow",
            "/tmp/workflow.yaml",
        ]
    )

    assert arguments.legacy_workspace == Path("/tmp/workspace")
    assert arguments.legacy_workflow == Path("/tmp/workflow.yaml")


def test_workspace_prune_requires_an_explicit_safe_mode() -> None:
    parser = cli._parser()
    arguments = parser.parse_args(
        [
            "workspace",
            "prune",
            "--stage",
            "assets",
            "--stage",
            "report",
            "--keep",
            "2",
            "--dry-run",
        ]
    )

    assert arguments.stage == ["assets", "report"]
    assert arguments.keep == 2
    assert arguments.dry_run is True
    assert arguments.confirm is False


def test_report_can_explicitly_reuse_verified_assets() -> None:
    arguments = cli._parser().parse_args(["report", "--reuse-assets"])

    assert arguments.reuse_assets is True


def test_doctor_probes_with_resolved_stage_contracts(
    monkeypatch, capsys
) -> None:
    observed: dict[str, dict[str, object]] = {}

    class Workflow:
        @staticmethod
        def stage_science(stage: StageId) -> dict[str, object]:
            if stage is StageId.MD:
                return {
                    "schema_version": "mmpbsa.md-phase-plan.v2",
                    "md_engine_profile": "openmm_8_5@2",
                }
            assert stage is StageId.TRAJECTORY
            return {"adapter_contract": "mmpbsa.trajectory.v3"}

    class Runtime:
        providers = {"md": "openmm", "trajectory": "backends"}

        @staticmethod
        def stage_runtime(
            capability: str, *, gpu_id: str | None = None
        ) -> dict[str, object]:
            assert gpu_id is None
            return {"capability": capability}

    class Service:
        def __init__(self, capability: str) -> None:
            self.capability = capability

        def compatibility(
            self, runtime: dict[str, object]
        ) -> BackendCompatibility:
            observed[self.capability] = runtime
            return BackendCompatibility(
                provider_id="test",
                adapter_contract="test.v1",
                engine="test",
                engine_major="1",
                engine_version="1.0",
            )

    class Registry:
        @staticmethod
        def service(_provider_id: str, capability: str) -> Service:
            return Service(capability)

        @staticmethod
        def describe() -> tuple[object, ...]:
            return ()

    monkeypatch.setattr(cli, "_load", lambda _arguments: (Workflow(), Runtime()))
    monkeypatch.setattr(cli, "discover_providers", Registry)

    assert cli._doctor(Namespace()) == 0
    assert observed["md"]["_stage_contract"] == "mmpbsa.md-phase-plan.v2"
    assert observed["md"]["_md_engine_profile"] == "openmm_8_5@2"
    assert observed["trajectory"]["_stage_contract"] == "mmpbsa.trajectory.v3"
    assert json.loads(capsys.readouterr().out)["ok"] is True

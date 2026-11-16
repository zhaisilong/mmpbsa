from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
import yaml

from mmpbsa.core import cli
from mmpbsa.core.config import (
    WORKFLOW_SCHEMA,
    CaseConfig,
    RuntimeDefinition,
    WorkflowDefinition,
    load_runtime,
)
from mmpbsa.core.errors import (
    ArtifactIntegrityError,
    ConfigurationError,
    ContractError,
    RetryError,
    TaskExecutionError,
)
from mmpbsa.core.models import (
    BackendCompatibility,
    JobKey,
    ReplicaSpec,
    ResourceRequest,
    StageInput,
    StageRequest,
    StageResult,
    TaskKey,
    WorkflowProfile,
)
from mmpbsa.core.prepared import (
    PreparedBindingView,
    PreparedCase,
    PreparedEntity,
    PreparedMdView,
    PreparedTrajectoryView,
)
from mmpbsa.core.protocols import resolve_protocol
from mmpbsa.core.providers import ProviderRegistry
from mmpbsa.core.runner import WorkflowRunner
from mmpbsa.core.state import AttemptState, QcVerdict, StageId, WorkflowOutcome
from mmpbsa.utils import sha256_digest

_Coordinate = tuple[str, str, str, str | None]


@dataclass(frozen=True)
class FakeEvent:
    capability: str
    method: str
    case_id: str | None
    replica_id: str | None
    model_id: str | None
    gpu_id: str | None
    prepared_view: str | None
    prepared_roles: tuple[str, ...]
    md_system_roles: tuple[str, ...]
    has_binding: bool
    has_entities: bool


@dataclass
class FakeControls:
    events: list[FakeEvent] = field(default_factory=list)
    fail: set[_Coordinate] = field(default_factory=set)
    interrupt_once: set[_Coordinate] = field(default_factory=set)
    blocked_qc: set[tuple[str, str]] = field(default_factory=set)
    warned_qc: set[tuple[str, str]] = field(default_factory=set)
    interrupted: set[_Coordinate] = field(default_factory=set)
    generations: dict[tuple[str, str | None], int] = field(default_factory=dict)
    scoring_frame_ids: dict[tuple[str, str, str], tuple[int, ...]] = field(
        default_factory=dict
    )
    scoring_barrier: threading.Barrier | None = None
    resources: Mapping[str, ResourceRequest] = field(
        default_factory=lambda: {
            "md": ResourceRequest(cpu_slots=2, gpu_slots=1),
            "trajectory": ResourceRequest(cpu_slots=1),
            "trajectory_qc": ResourceRequest(cpu_slots=1),
            "scoring": ResourceRequest(cpu_slots=2),
            "report.assets": ResourceRequest(cpu_slots=1),
            "report": ResourceRequest(cpu_slots=1),
        }
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin(
        self, capability: str, method: str, request: StageRequest
    ) -> tuple[_Coordinate, bool, bool, int]:
        coordinate = (
            capability,
            request.task.case_id or "",
            request.task.replica_id or "",
            request.task.model_id,
        )
        generation_key = (capability, request.task.model_id)
        prepared = request.prepared_case
        with self.lock:
            self.events.append(
                FakeEvent(
                    capability=capability,
                    method=method,
                    case_id=request.task.case_id,
                    replica_id=request.task.replica_id,
                    model_id=request.task.model_id,
                    gpu_id=(
                        None
                        if request.runtime.get("gpu_id") is None
                        else str(request.runtime["gpu_id"])
                    ),
                    prepared_view=(
                        None
                        if prepared is None
                        else type(prepared).__name__
                    ),
                    prepared_roles=tuple(
                        sorted(
                            getattr(prepared, "resolved_artifacts", {})
                        )
                    ),
                    md_system_roles=tuple(
                        sorted(
                            item.role
                            for item in getattr(
                                getattr(prepared, "system", None),
                                "artifacts",
                                (),
                            )
                        )
                    ),
                    has_binding=hasattr(prepared, "binding"),
                    has_entities=hasattr(prepared, "entities"),
                )
            )
            generation = self.generations.get(generation_key, 0) + 1
            self.generations[generation_key] = generation
            should_interrupt = (
                coordinate in self.interrupt_once
                and coordinate not in self.interrupted
            )
            if should_interrupt:
                self.interrupted.add(coordinate)
            should_fail = coordinate in self.fail
        return coordinate, should_interrupt, should_fail, generation


class ControlledInterrupt(KeyboardInterrupt):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__("controlled checkpoint interruption")
        self.checkpoint = checkpoint


class FakeStageService:
    def __init__(self, capability: str, controls: FakeControls) -> None:
        self.capability = capability
        self.controls = controls

    def resource_request(self, request: StageRequest) -> ResourceRequest:
        if self.capability == "scoring":
            assert request.task.model_id in {"gb", "pb"}
        else:
            assert request.task.model_id is None
        return self.controls.resources.get(self.capability, ResourceRequest())

    def compatibility(
        self, runtime: Mapping[str, object]
    ) -> BackendCompatibility:
        return BackendCompatibility(
            provider_id="fake",
            adapter_contract=f"{self.capability}.v2",
            engine=f"fake-{self.capability}",
            engine_major="2",
            engine_version="2.0.4",
        )

    def execute(self, request: StageRequest) -> StageResult:
        return self._invoke(request, "execute")

    def resume(self, request: StageRequest) -> StageResult:
        assert request.resume is True
        assert request.checkpoint is not None
        assert request.checkpoint.read_bytes() == b"checkpoint-v2"
        return self._invoke(request, "resume")

    def _invoke(self, request: StageRequest, method: str) -> StageResult:
        coordinate, interrupt, fail, generation = self.controls.begin(
            self.capability, method, request
        )
        if (
            self.capability == "scoring"
            and self.controls.scoring_barrier is not None
        ):
            self.controls.scoring_barrier.wait(timeout=2.0)
        if interrupt:
            checkpoint = request.attempt_dir / "output" / "restart.chk"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint-v2")
            raise ControlledInterrupt(Path("output/restart.chk"))
        if fail:
            target = "/".join(
                item
                for item in (
                    coordinate[1],
                    coordinate[2],
                    coordinate[3],
                )
                if item
            )
            raise RuntimeError(
                f"deliberate {self.capability} failure for {target}"
            )

        output = request.attempt_dir / "output"
        output.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "stage": self.capability,
            "case_id": request.task.case_id,
            "replica_id": request.task.replica_id,
            "model_id": request.task.model_id,
        }
        verdict = None
        role = self.capability
        filename = f"{self.capability.replace('.', '_')}.json"

        if self.capability == "md":
            payload["steps"] = 10
        elif self.capability == "trajectory":
            payload["frames"] = 3
        elif self.capability == "trajectory_qc":
            qc_key = (coordinate[1], coordinate[2])
            if qc_key in self.controls.blocked_qc:
                verdict = QcVerdict.BLOCK
            elif qc_key in self.controls.warned_qc:
                verdict = QcVerdict.WARN
            else:
                verdict = QcVerdict.PASS
            payload["verdict"] = verdict.value
        elif self.capability == "scoring":
            model_id = request.task.model_id
            assert model_id is not None
            expected_native = (
                {"main": "gb.mdin"}
                if model_id == "gb"
                else {
                    "complex_receptor": "pb.mdin",
                    "ligand": "pb_ligand.mdin",
                }
            )
            assert request.science["native_inputs"] == expected_native
            score = -7.5 if model_id == "gb" else -17.5
            frame_ids = self.controls.scoring_frame_ids.get(
                (coordinate[1], coordinate[2], model_id),
                (1, 2, 3),
            )
            payload.update(
                {
                    "score": score,
                    "frame_count": len(frame_ids),
                    "frame_ids": {
                        "count": len(frame_ids),
                        "first": frame_ids[0],
                        "last": frame_ids[-1],
                        "sha256": sha256_digest(frame_ids),
                    },
                    "frame_manifest": {
                        "digest": request.science["frame_manifest_digest"],
                    },
                    "tables": [
                        {
                            "model_id": model_id,
                            "component": "delta",
                            "label": "DELTA_TOTAL",
                            "average": score,
                        }
                    ],
                }
            )
            role = "scoring_summary"
            filename = f"{model_id}.json"
        elif self.capability == "report.assets":
            role = "asset_manifest"
            payload = {
                "generation": generation,
                "analysis_attempt": request.upstream["analysis"].attempt_id,
            }
        elif self.capability == "report":
            role = "report_index"
            filename = "index.html"
            cases = request.upstream["analysis"].payload.get("cases", ())
            payload = {
                "generation": generation,
                "outcomes": [
                    str(item.get("outcome"))
                    for item in cases
                    if isinstance(item, Mapping)
                ],
            }

        artifact = output / filename
        if self.capability == "report":
            artifact.write_text(
                "<!doctype html>"
                f"<title>fake report {generation}</title>"
                f"<p>{','.join(payload['outcomes'])}</p>",
                encoding="utf-8",
            )
        else:
            artifact.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
        artifacts = {role: artifact}
        if self.capability == "report" and "assets" in request.upstream:
            published_asset = output / "assets" / "display.json"
            published_asset.parent.mkdir(parents=True, exist_ok=True)
            published_asset.write_text(
                json.dumps(
                    {
                        "generation": generation,
                        "source_attempt": request.upstream["assets"].attempt_id,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            artifacts["report_asset_0001"] = published_asset
        if self.capability == "scoring":
            delta = output / f"{request.task.model_id}_delta.csv"
            delta.write_text(
                "Frame,DELTA_TOTAL\n"
                + "".join(
                    f"{frame_id},{float(payload['score']) - index}\n"
                    for index, frame_id in enumerate(frame_ids)
                ),
                encoding="utf-8",
            )
            artifacts["delta_output"] = delta
        return StageResult(
            artifacts=artifacts,
            payload=payload,
            qc_verdict=verdict,
            invocation_provenance={
                "fake_capability": self.capability,
                "method": method,
                "model_id": request.task.model_id,
            },
        )


@dataclass(frozen=True)
class FakeProvider:
    services: Mapping[str, FakeStageService]
    provider_id: str = "fake"
    api_version: str = "2"
    package_version: str = "0.7.0"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(self.services)

    def get_service(self, capability: str) -> object | None:
        return self.services.get(capability)


@dataclass(frozen=True)
class ProjectFixture:
    runtime_path: Path
    workflow: WorkflowDefinition
    runtime: RuntimeDefinition
    registry: ProviderRegistry
    controls: FakeControls


def _artifact(root: Path, relative: str, role: str, data: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "role": role,
        "media_type": "application/octet-stream",
    }


def _write_prepared_case(root: Path, case_id: str) -> Path:
    case_root = root / case_id
    signature = hashlib.sha256(f"{case_id}:atoms".encode()).hexdigest()
    manifest = {
        "schema_version": "mmpbsa.prepared_case.v2",
        "case_id": case_id,
        "md": {
            "backend": "fake_md",
            "atom_count": 4,
            "system_signature": hashlib.sha256(
                f"{case_id}:md".encode()
            ).hexdigest(),
            "artifacts": [
                _artifact(
                    case_root,
                    "md/system.bin",
                    "md_topology",
                    f"{case_id}:md-system".encode(),
                ),
                _artifact(
                    case_root,
                    "md/system.gro",
                    "md_coordinates",
                    f"{case_id}:md-coordinates".encode(),
                ),
                _artifact(
                    case_root,
                    "md/index.ndx",
                    "index",
                    f"{case_id}:index".encode(),
                ),
            ],
        },
        "scoring": {
            "backend": "fake_scoring",
            "atom_count": 4,
            "atom_order_signature": signature,
            "source_md_atom_indices": [1, 2, 3, 4],
            "radii_model": "test-radii",
            "artifacts": [
                _artifact(
                    case_root,
                    "scoring/complex.top",
                    "complex_topology",
                    f"{case_id}:complex".encode(),
                ),
                _artifact(
                    case_root,
                    "scoring/receptor.top",
                    "receptor_topology",
                    f"{case_id}:receptor".encode(),
                ),
                _artifact(
                    case_root,
                    "scoring/binder.top",
                    "binder_topology",
                    f"{case_id}:binder".encode(),
                ),
                _artifact(
                    case_root,
                    "scoring/coordinates.crd",
                    "coordinates",
                    f"{case_id}:coordinates".encode(),
                ),
            ],
        },
        "entities": [
            {
                "entity_id": "receptor",
                "kind": "protein",
                "md_atom_indices": [1, 2],
                "scoring_atom_indices": [1, 2],
                "residues": [
                    {
                        "residue_id": "receptor_1",
                        "residue_name": "ALA",
                        "md_residue_index": 1,
                        "scoring_residue_index": 1,
                    }
                ],
                "named_selections": {
                    "all": {
                        "md_atom_indices": [1, 2],
                        "scoring_atom_indices": [1, 2],
                    },
                    "backbone": {
                        "md_atom_indices": [1],
                        "scoring_atom_indices": [1],
                    },
                    "heavy_atoms": {
                        "md_atom_indices": [1, 2],
                        "scoring_atom_indices": [1, 2],
                    },
                    "interface_backbone": {
                        "md_atom_indices": [1],
                        "scoring_atom_indices": [1],
                    },
                },
                "selection_signature": "",
            },
            {
                "entity_id": "ligand",
                "kind": "peptide",
                "md_atom_indices": [3, 4],
                "scoring_atom_indices": [3, 4],
                "residues": [
                    {
                        "residue_id": "ligand_1",
                        "residue_name": "LIG",
                        "md_residue_index": 2,
                        "scoring_residue_index": 2,
                    }
                ],
                "named_selections": {
                    "all": {
                        "md_atom_indices": [3, 4],
                        "scoring_atom_indices": [3, 4],
                    },
                    "backbone": {
                        "md_atom_indices": [3],
                        "scoring_atom_indices": [3],
                    },
                    "heavy_atoms": {
                        "md_atom_indices": [3, 4],
                        "scoring_atom_indices": [3, 4],
                    },
                },
                "selection_signature": "",
            },
        ],
        "primary_binding": {
            "binding_id": "primary",
            "receptor_entity_ids": ["receptor"],
            "ligand_entity_ids": ["ligand"],
            "excluded_entity_ids": [],
            "topology_artifact_roles": {
                "complex": "complex_topology",
                "receptor": "receptor_topology",
                "ligand": "binder_topology",
            },
        },
        "provenance": {"producer": "core-runner-v2-test"},
    }
    path = case_root / "prepared_case.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _analysis_model() -> dict[str, object]:
    return {
        "score_selector": {
            "component": "delta",
            "label": "DELTA_TOTAL",
            "value_key": "average",
        },
        "score_transform": {
            "scale": 1.0,
            "offset": 0.0,
            "input_unit": "kcal/mol",
            "output_unit": "kcal/mol",
        },
    }


def _project(
    root: Path,
    *,
    case_ids: Sequence[str] = ("case_a",),
    replica_ids: Sequence[str] = ("rep01",),
    controls: FakeControls | None = None,
    completion_policy: str = "any",
    with_assets: bool = False,
) -> ProjectFixture:
    root.mkdir(parents=True, exist_ok=True)
    prepared_root = root / "prepared"
    prepared = {
        case_id: _write_prepared_case(prepared_root, case_id)
        for case_id in case_ids
    }
    resolved = resolve_protocol(
        "production_3x5ns@1",
        models=("gb", "pb"),
        completion_policy=completion_policy,
    )
    frame_manifest_digest = resolved.preset.frame_plan.manifest_digest
    stages = {
        "md": {"steps": 10},
        "trajectory": {"stride": 2},
        "trajectory_qc": {"block_threshold": 1.0},
        "scoring": {
            "completion_policy": completion_policy,
            "models": {
                "gb": {
                    "expected_frames": 3,
                    "frame_manifest_digest": frame_manifest_digest,
                    "control_text": "&gb igb=5 /",
                    "native_inputs": {"main": "gb.mdin"},
                },
                "pb": {
                    "expected_frames": 3,
                    "frame_manifest_digest": frame_manifest_digest,
                    "control_text": "&pb istrng=0.15 /",
                    "native_inputs": {
                        "complex_receptor": "pb.mdin",
                        "ligand": "pb_ligand.mdin",
                    },
                },
            },
        },
    }
    analysis = {
        "reference_key": "value",
        "models": {"gb": _analysis_model(), "pb": _analysis_model()},
    }
    report = {
        "title": "Runner execution contract",
        "model_order": ["gb", "pb"],
    }
    workflow = WorkflowDefinition(
        source_path=root / "workflow.yaml",
        project_id="runner_contract",
        protocol_id="production_3x5ns_v1",
        profile_factory="mmpbsa.core.protocols:resolve_protocol",
        profile_configuration={
            "preset": "production_3x5ns@1",
            "models": ["gb", "pb"],
            "completion_policy": completion_policy,
        },
        profile=WorkflowProfile(
            profile_id="production_3x5ns",
            profile_version="1",
            normalized=resolved.as_dict(),
            extensions={},
        ),
        cases=tuple(
            CaseConfig(
                case_id=case_id,
                prepared_case=prepared[case_id],
                label=case_id.replace("_", " ").title(),
                reference={"value": -8.0},
            )
            for case_id in case_ids
        ),
        replicas=tuple(
            ReplicaSpec(
                replica_id=replica_id,
                replica_index=index,
                seed=41000 + index,
            )
            for index, replica_id in enumerate(replica_ids, start=1)
        ),
        stages=stages,
        analysis=analysis,
        report=report,
        resolved_protocol=resolved,
        schema_version=WORKFLOW_SCHEMA,
    )
    capabilities = [
        "md",
        "trajectory",
        "trajectory_qc",
        "scoring",
        "report",
    ]
    if with_assets:
        capabilities.append("report.assets")
    runtime_data = {
        "schema_version": "mmpbsa.runtime.v1",
        "providers": {capability: "fake" for capability in capabilities},
        "scheduler": {
            "gpu_ids": ["0", "1"],
            "max_workers": 2,
            "threads_per_job": 1,
            "cpu_slots": 4,
        },
        "backends": {"fake": {"common": {"executable": "fake-engine"}}},
    }
    runtime_path = root / "runtime.yaml"
    runtime_path.write_text(
        yaml.safe_dump(runtime_data, sort_keys=False), encoding="utf-8"
    )
    effective_controls = controls or FakeControls()
    services = {
        capability: FakeStageService(capability, effective_controls)
        for capability in capabilities
    }
    return ProjectFixture(
        runtime_path=runtime_path,
        # Compact stage science keeps these tests focused on execution-graph
        # behavior; the resolved package protocol makes the fixture a genuine
        # v3 execution contract rather than a relabeled legacy workflow.
        workflow=workflow,
        runtime=load_runtime(runtime_path),
        registry=ProviderRegistry([FakeProvider(services)]),
        controls=effective_controls,
    )


def _runner(project: ProjectFixture, workspace: Path) -> WorkflowRunner:
    return WorkflowRunner(
        project.workflow,
        project.runtime,
        workspace,
        providers=project.registry,
    )


def _scoring_task(model_id: str) -> TaskKey:
    return TaskKey(
        StageId.SCORING,
        case_id="case_a",
        replica_id="rep01",
        model_id=model_id,
    )


def _pipeline_project(project):
    return replace(
        project,
        runtime=replace(
            project.runtime,
            scheduler=replace(
                project.runtime.scheduler,
                execution_mode="pipeline",
                cpu_slots=8,
                max_workers=5,
                stage_limits={"md": 1, "postprocess": 1, "scoring": 2},
                cpu_reservations={"md": 2, "postprocess": 1},
            ),
        ),
    )


def test_pipeline_starts_next_md_while_previous_scores_wait(tmp_path):
    project = _pipeline_project(_project(tmp_path / "project", replica_ids=("rep01", "rep02")))
    runner = _runner(project, tmp_path / "workspace")
    scoring_started = threading.Event()
    second_md_started = threading.Event()
    md = project.registry.service("fake", "md")
    scoring = project.registry.service("fake", "scoring")
    execute_md, execute_scoring = md.execute, scoring.execute

    def md_execute(request):
        if request.task.replica_id == "rep02":
            assert scoring_started.wait(10), "previous scoring never started"
            second_md_started.set()
        return execute_md(request)

    def scoring_execute(request):
        if request.task.replica_id == "rep01":
            scoring_started.set()
            assert second_md_started.wait(10), "scoring prevented next MD from starting"
        return execute_scoring(request)

    md.execute, scoring.execute = md_execute, scoring_execute
    result = runner.run_batch()
    assert not result.failures
    assert len(result.results) == 2
    status = json.loads((runner.workspace.root / "scheduler_status.json").read_text())
    assert status["active"] == []
    assert status["terminal_stage_counts"]["md"] == 2
    before = len(project.controls.events)
    assert not runner.run_batch().failures
    assert len(project.controls.events) == before


@pytest.mark.parametrize("policy,failed", [("all", True), ("any", False)])
def test_pipeline_model_failure_isolated(tmp_path, policy, failed):
    controls = FakeControls(fail={("scoring", "case_a", "rep01", "pb")})
    project = _pipeline_project(
        _project(tmp_path / "project", controls=controls, completion_policy=policy)
    )
    result = _runner(project, tmp_path / "workspace").run_batch()
    assert bool(result.failures) is failed
    assert any(event.model_id == "gb" for event in controls.events)


def test_pipeline_qc_block_and_other_replica_continue(tmp_path):
    controls = FakeControls(blocked_qc={("case_a", "rep01")})
    project = _pipeline_project(
        _project(tmp_path / "project", controls=controls, replica_ids=("rep01", "rep02"))
    )
    result = _runner(project, tmp_path / "workspace").run_batch()
    assert not result.failures
    assert result.outcome is WorkflowOutcome.SCIENTIFICALLY_BLOCKED
    assert not any(
        event.capability == "scoring" and event.replica_id == "rep01" for event in controls.events
    )
    assert any(
        event.capability == "scoring" and event.replica_id == "rep02" for event in controls.events
    )


def test_pipeline_physical_provenance_does_not_change_identity(tmp_path):
    project = _pipeline_project(_project(tmp_path / "project"))
    runner = _runner(project, tmp_path / "workspace")
    original = runner.run_batch()
    assert not original.failures
    original_identity = {
        attempt.task: attempt.science_hash
        for result in original.results.values()
        for attempt in result.attempts
    }
    runtime = replace(
        project.runtime, scheduler=replace(project.runtime.scheduler, cpu_affinity="physical")
    )
    physical = WorkflowRunner(
        project.workflow, runtime, runner.workspace.root, providers=project.registry
    )
    reused = physical.run_batch()
    assert not reused.failures
    assert {
        attempt.task: attempt.science_hash
        for result in reused.results.values()
        for attempt in result.attempts
    } == original_identity
    fresh = WorkflowRunner(
        project.workflow, runtime, tmp_path / "physical", providers=project.registry
    )
    result = fresh.run_batch()
    assert not result.failures
    attempt = result.results[JobKey("case_a", "rep01")].attempts[-1]
    invocation = (
        fresh.workspace.task_dir(attempt.task)
        / "attempts"
        / attempt.attempt_id
        / "invocations/0001/invocation.json"
    )
    provenance = json.loads(invocation.read_text())["runtime"]
    assert len(provenance["allocation"]["cpu_ids"]) == 2
    assert provenance["timing"]["execution_seconds"] >= 0


def _stage_task(stage: StageId) -> TaskKey:
    if stage in {StageId.ADMIT, StageId.FINALIZE_CASE}:
        return TaskKey(stage, case_id="case_a")
    if stage in {
        StageId.MD,
        StageId.TRAJECTORY,
        StageId.TRAJECTORY_QC,
    }:
        return TaskKey(
            stage,
            case_id="case_a",
            replica_id="rep01",
        )
    if stage is StageId.SCORING:
        return _scoring_task("gb")
    return TaskKey(stage)


def _prepared_science_identity(
    runner: WorkflowRunner,
    prepared: PreparedCase,
    stage: StageId,
) -> Mapping[str, object]:
    task = _stage_task(stage)
    science = runner._science(task, prepared)
    return runner._science_identity(
        task,
        science,
        prepared,
        {},
    )


def _with_prepared_entity(
    prepared: PreparedCase,
    changed: PreparedEntity,
) -> PreparedCase:
    manifest = prepared.manifest
    entities = tuple(
        changed if item.entity_id == changed.entity_id else item
        for item in manifest.entities
    )
    return replace(
        prepared,
        manifest=replace(manifest, entities=entities),
    )


def test_runner_rejects_read_only_legacy_workflow(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    legacy = replace(
        project.workflow,
        schema_version="mmpbsa.workflow.v2",
        resolved_protocol=None,
    )

    with pytest.raises(ConfigurationError, match="read-only legacy"):
        WorkflowRunner(
            legacy,
            project.runtime,
            tmp_path / "workspace",
            providers=project.registry,
        )


def test_stage_requests_receive_only_their_prepared_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    admission_views: list[PreparedBindingView] = []
    original_execute_core = runner._execute_core

    def capture_core_request(request: StageRequest) -> StageResult:
        if request.task.stage is StageId.ADMIT:
            assert isinstance(request.prepared_case, PreparedBindingView)
            admission_views.append(request.prepared_case)
        return original_execute_core(request)

    monkeypatch.setattr(runner, "_execute_core", capture_core_request)

    runner.run_job(
        JobKey("case_a", "rep01"),
        model_ids=("gb",),
    )

    assert len(admission_views) == 1
    assert not isinstance(admission_views[0], PreparedCase)
    expected_binding_roles = {
        "binder_topology",
        "complex_topology",
        "coordinates",
        "index",
        "receptor_topology",
    }
    assert set(admission_views[0].resolved_artifacts) == (
        expected_binding_roles
    )

    events = {
        event.capability: event
        for event in project.controls.events
        if event.method == "execute"
    }
    md = events["md"]
    assert md.prepared_view == PreparedMdView.__name__
    assert set(md.prepared_roles) == {
        "md_coordinates",
        "md_topology",
    }
    assert set(md.md_system_roles) == {
        "md_coordinates",
        "md_topology",
    }
    assert md.has_binding is False
    assert md.has_entities is False

    trajectory = events["trajectory"]
    assert trajectory.prepared_view == PreparedTrajectoryView.__name__
    assert set(trajectory.prepared_roles) == {
        "complex_topology",
        "index",
        "md_topology",
        "md_coordinates",
    }
    assert trajectory.has_binding is False
    assert trajectory.has_entities is False

    for capability in ("trajectory_qc", "scoring"):
        event = events[capability]
        assert event.prepared_view == PreparedBindingView.__name__
        assert set(event.prepared_roles) == expected_binding_roles
        assert event.has_binding is True
        assert event.has_entities is True


def test_prepared_case_science_identity_has_stage_boundaries(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    prepared = runner._prepared("case_a")
    manifest = prepared.manifest
    ligand = manifest.entity("ligand")

    baseline = {
        stage: _prepared_science_identity(runner, prepared, stage)
        for stage in (
            StageId.ADMIT,
            StageId.MD,
            StageId.TRAJECTORY,
            StageId.TRAJECTORY_QC,
            StageId.SCORING,
        )
    }
    admission_projection = baseline[StageId.ADMIT]["prepared_case"]
    assert isinstance(admission_projection, Mapping)
    assert set(admission_projection) == {
        "artifacts",
        "binding_selections",
        "case_id",
        "primary_binding",
        "schema_version",
        "scoring_projection",
    }
    admission_artifacts = admission_projection["artifacts"]
    assert isinstance(admission_artifacts, Mapping)
    assert {
        item["role"]
        for artifacts in admission_artifacts.values()
        for item in artifacts
    } == {
        "binder_topology",
        "complex_topology",
        "coordinates",
        "index",
        "receptor_topology",
    }

    with pytest.raises(
        ContractError,
        match="full_system_topology_role must be 'md_topology'",
    ):
        runner._science_identity(
            _stage_task(StageId.TRAJECTORY),
            {
                "adapter_contract": "mmpbsa.trajectory.v3",
                "full_system_topology_role": "solvated_topology",
            },
            prepared,
            {},
        )

    trajectory_task = _stage_task(StageId.TRAJECTORY)
    conflicting_science = runner._science(trajectory_task, prepared)
    conflicting_science["complex_topology_role"] = "binder_topology"
    conflicting_identity = runner._science_identity(
        trajectory_task,
        conflicting_science,
        prepared,
        {},
    )
    trajectory_projection = conflicting_identity["prepared_case"]
    assert isinstance(trajectory_projection, Mapping)
    projected_artifacts = trajectory_projection["artifacts"]
    assert isinstance(projected_artifacts, Mapping)
    scoring_artifacts = projected_artifacts["scoring"]
    assert isinstance(scoring_artifacts, list)
    assert [item["role"] for item in scoring_artifacts] == [
        "complex_topology"
    ]

    changed_kind = _with_prepared_entity(
        prepared,
        replace(ligand, kind="small_molecule"),
    )
    assert (
        _prepared_science_identity(runner, changed_kind, StageId.MD)
        == baseline[StageId.MD]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_kind,
            StageId.TRAJECTORY,
        )
        != baseline[StageId.TRAJECTORY]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_kind,
            StageId.TRAJECTORY_QC,
        )
        != baseline[StageId.TRAJECTORY_QC]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_kind,
            StageId.SCORING,
        )
        != baseline[StageId.SCORING]
    )

    changed_crosswalk = _with_prepared_entity(
        prepared,
        replace(
            ligand,
            residues=(
                replace(ligand.residues[0], residue_name="LI2"),
            ),
        ),
    )
    trajectory_task = _stage_task(StageId.TRAJECTORY)
    v3_science = {
        "adapter_contract": "mmpbsa.trajectory.v3",
        "full_system_topology_role": "md_topology",
        "full_system_coordinates_role": "md_coordinates",
    }
    baseline_v3 = runner._science_identity(
        trajectory_task,
        v3_science,
        prepared,
        {},
    )
    changed_v3 = runner._science_identity(
        trajectory_task,
        v3_science,
        changed_crosswalk,
        {},
    )
    assert (
        _prepared_science_identity(runner, changed_crosswalk, StageId.MD)
        == baseline[StageId.MD]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_crosswalk,
            StageId.TRAJECTORY,
        )
        == baseline[StageId.TRAJECTORY]
    )
    assert changed_v3 != baseline_v3
    for downstream in (StageId.TRAJECTORY_QC, StageId.SCORING):
        assert (
            _prepared_science_identity(runner, changed_crosswalk, downstream)
            != baseline[downstream]
        )

    changed_metadata = _with_prepared_entity(
        prepared,
        replace(ligand, metadata={"formal_charge": 1}),
    )
    assert (
        _prepared_science_identity(runner, changed_metadata, StageId.MD)
        == baseline[StageId.MD]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_metadata,
            StageId.TRAJECTORY_QC,
        )
        == baseline[StageId.TRAJECTORY_QC]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_metadata,
            StageId.SCORING,
        )
        != baseline[StageId.SCORING]
    )

    topology_roles = manifest.primary_binding.topology_artifact_roles
    changed_binding = replace(
        prepared,
        manifest=replace(
            manifest,
            primary_binding=replace(
                manifest.primary_binding,
                topology_artifact_roles=replace(
                    topology_roles,
                    receptor=topology_roles.ligand,
                    ligand=topology_roles.receptor,
                ),
            ),
        ),
    )
    assert (
        _prepared_science_identity(runner, changed_binding, StageId.MD)
        == baseline[StageId.MD]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_binding,
            StageId.ADMIT,
        )
        != baseline[StageId.ADMIT]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_binding,
            StageId.TRAJECTORY,
        )
        == baseline[StageId.TRAJECTORY]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_binding,
            StageId.TRAJECTORY_QC,
        )
        != baseline[StageId.TRAJECTORY_QC]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_binding,
            StageId.SCORING,
        )
        != baseline[StageId.SCORING]
    )

    changed_complex_binding = replace(
        prepared,
        manifest=replace(
            manifest,
            primary_binding=replace(
                manifest.primary_binding,
                topology_artifact_roles=replace(
                    topology_roles,
                    complex=topology_roles.ligand,
                    receptor=topology_roles.complex,
                    ligand=topology_roles.receptor,
                ),
            ),
        ),
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_complex_binding,
            StageId.TRAJECTORY,
        )
        != baseline[StageId.TRAJECTORY]
    )

    changed_radii = replace(
        prepared,
        manifest=replace(
            manifest,
            scoring=replace(
                manifest.scoring,
                radii_model="different-radii",
            ),
        ),
    )
    for stage in (
        StageId.MD,
        StageId.TRAJECTORY,
        StageId.TRAJECTORY_QC,
    ):
        assert (
            _prepared_science_identity(runner, changed_radii, stage)
            == baseline[stage]
        )
    assert (
        _prepared_science_identity(
            runner,
            changed_radii,
            StageId.SCORING,
        )
        != baseline[StageId.SCORING]
    )

    changed_md = replace(
        prepared,
        manifest=replace(
            manifest,
            md=replace(manifest.md, system_signature="f" * 64),
        ),
    )
    assert (
        _prepared_science_identity(runner, changed_md, StageId.MD)
        != baseline[StageId.MD]
    )

    heavy = ligand.named_selections["heavy_atoms"]
    changed_md_selection = _with_prepared_entity(
        prepared,
        replace(
            ligand,
            named_selections={
                **ligand.named_selections,
                "heavy_atoms": replace(
                    heavy,
                    md_atom_indices=(3,),
                    scoring_atom_indices=(3,),
                ),
            },
            selection_signature="",
        ),
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_md_selection,
            StageId.MD,
        )
        == baseline[StageId.MD]
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_md_selection,
            StageId.TRAJECTORY_QC,
        )
        != baseline[StageId.TRAJECTORY_QC]
    )

    changed_provenance = replace(
        prepared,
        manifest=replace(
            manifest,
            provenance={"producer": "different"},
        ),
    )
    assert (
        _prepared_science_identity(
            runner,
            changed_provenance,
            StageId.ADMIT,
        )
        == baseline[StageId.ADMIT]
    )
    for stage in (
        StageId.MD,
        StageId.TRAJECTORY,
        StageId.TRAJECTORY_QC,
        StageId.SCORING,
    ):
        assert (
            _prepared_science_identity(
                runner,
                changed_provenance,
                stage,
            )
            == baseline[stage]
        )


@pytest.mark.parametrize(
    "stage",
    (
        StageId.FINALIZE_CASE,
        StageId.ANALYSIS,
        StageId.ASSETS,
        StageId.REPORT,
    ),
)
def test_summary_stages_do_not_directly_hash_prepared_case(
    tmp_path: Path,
    stage: StageId,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    prepared = runner._prepared("case_a")
    task = _stage_task(stage)
    science = runner._science(task, prepared)

    identity = runner._science_identity(task, science, prepared, {})

    assert "_prepared_case_schema" not in science
    assert identity["prepared_case"] is None


def test_upstream_identity_hashes_payload_and_attempt_id(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    task = TaskKey(StageId.ANALYSIS)
    artifact = tmp_path / "case.json"
    artifact.write_text("{}\n", encoding="utf-8")
    baseline_input = StageInput(
        task=TaskKey(StageId.FINALIZE_CASE, case_id="case_a"),
        attempt_id="0001",
        root=tmp_path,
        artifacts={"case_record": artifact},
        science_hash="a" * 64,
        payload={"outcome": "pass"},
    )

    def identity(item: StageInput) -> Mapping[str, object]:
        return runner._science_identity(
            task,
            runner._science(task, None),
            None,
            {"case:case_a": item},
        )

    baseline = identity(baseline_input)
    changed_payload = identity(
        replace(baseline_input, payload={"outcome": "partial"})
    )
    changed_attempt = identity(replace(baseline_input, attempt_id="0002"))

    assert baseline != changed_payload
    assert baseline != changed_attempt
    upstream = baseline["upstream"]["case:case_a"]
    assert upstream["attempt_id"] == "0001"
    assert upstream["payload_sha256"] == sha256_digest(
        baseline_input.payload
    )


def test_cli_refuses_to_execute_read_only_v2_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    legacy_workflow = tmp_path / "legacy-workflow.yaml"
    legacy_workflow.write_text(
        "schema_version: mmpbsa.workflow.v2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "discover_providers", lambda: project.registry)
    arguments = [
        "--workflow",
        str(legacy_workflow),
        "--runtime",
        str(project.runtime_path),
        "--workspace",
        str(workspace),
        "job",
        "run",
        "--case",
        "case_a",
        "--replica",
        "rep01",
    ]

    assert cli.main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "read-only legacy configuration" in captured.err


def test_scoring_models_have_independent_attempts_and_retry(
    tmp_path: Path,
) -> None:
    controls = FakeControls(
        fail={("scoring", "case_a", "rep01", "pb")}
    )
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")
    job = JobKey("case_a", "rep01")

    first = runner.run_job(job)

    assert first.completed is True
    assert set(first.model_failures) == {"pb"}
    assert runner.workspace.latest(_scoring_task("gb")).state is AttemptState.COMPLETED
    assert runner.workspace.latest(_scoring_task("pb")).state is AttemptState.FAILED
    assert [item.attempt_id for item in runner.workspace.attempts(_scoring_task("gb"))] == [
        "0001"
    ]
    assert [item.attempt_id for item in runner.workspace.attempts(_scoring_task("pb"))] == [
        "0001"
    ]

    controls.fail.clear()
    retried = runner.run_job(job, action="retry", model_ids=("pb",))

    assert retried.completed is True
    assert retried.expected_model_ids == ("pb",)
    assert not retried.model_failures
    assert [item.attempt_id for item in runner.workspace.attempts(_scoring_task("gb"))] == [
        "0001"
    ]
    assert [
        (item.attempt_id, item.state)
        for item in runner.workspace.attempts(_scoring_task("pb"))
    ] == [
        ("0001", AttemptState.FAILED),
        ("0002", AttemptState.COMPLETED),
    ]


def test_gb_and_pb_execute_concurrently_when_scheduler_allows(
    tmp_path: Path,
) -> None:
    controls = FakeControls(scoring_barrier=threading.Barrier(2))
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")

    result = runner.run_job(JobKey("case_a", "rep01"))

    assert result.completed is True
    assert not result.model_failures
    assert [
        item.task.model_id
        for item in result.attempts
        if item.task.stage is StageId.SCORING
    ] == ["gb", "pb"]


def test_multi_model_retry_requires_explicit_model(tmp_path: Path) -> None:
    controls = FakeControls(
        fail={("scoring", "case_a", "rep01", "pb")}
    )
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")
    runner.run_job(JobKey("case_a", "rep01"))

    with pytest.raises(
        ContractError, match="requires exactly one explicit model"
    ):
        runner.run_job(JobKey("case_a", "rep01"), action="retry")


def test_trajectory_qc_block_skips_all_scoring(tmp_path: Path) -> None:
    controls = FakeControls(blocked_qc={("case_a", "rep01")})
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")

    result = runner.run_job(JobKey("case_a", "rep01"))

    assert [item.task.stage for item in result.attempts] == [
        StageId.ADMIT,
        StageId.MD,
        StageId.TRAJECTORY,
        StageId.TRAJECTORY_QC,
    ]
    assert result.model_failures == {
        "gb": "skipped: trajectory_qc verdict is BLOCK",
        "pb": "skipped: trajectory_qc verdict is BLOCK",
    }
    assert not [
        event for event in controls.events if event.capability == "scoring"
    ]
    assert runner.workspace.latest(_scoring_task("gb")) is None
    assert runner.workspace.latest(_scoring_task("pb")) is None

    finalized = runner.finalize_case("case_a")
    assert finalized.qc_verdict is QcVerdict.BLOCK
    assert finalized.payload["partial"] is False
    assert finalized.payload["outcome"] == (
        WorkflowOutcome.SCIENTIFICALLY_BLOCKED.value
    )


def test_blocked_qc_never_exposes_even_current_scoring_results(
    tmp_path: Path,
) -> None:
    controls = FakeControls(blocked_qc={("case_a", "rep01")})
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")
    partial = runner.run_job(
        JobKey("case_a", "rep01"),
        through=StageId.TRAJECTORY_QC,
    )
    by_stage = {item.task.stage: item for item in partial.attempts}
    trajectory = runner.workspace.as_stage_input(
        by_stage[StageId.TRAJECTORY]
    )
    qc = runner.workspace.as_stage_input(
        by_stage[StageId.TRAJECTORY_QC]
    )
    prepared = runner._prepared("case_a")
    for model_id in project.workflow.scoring_model_ids:
        runner._execute_task(
            _scoring_task(model_id),
            action="run",
            prepared_case=prepared,
            upstream={
                "trajectory": trajectory,
                "trajectory_qc": qc,
            },
        )

    finalized = runner.finalize_case("case_a")

    replica = finalized.payload["replicas"][0]
    assert replica["qc_verdict"] == "BLOCK"
    assert replica["scoring_models"] == {}
    assert replica["completed_models"] == []
    assert {
        model_id: (
            status["usable"],
            status["identity"],
        )
        for model_id, status in replica["model_statuses"].items()
    } == {
        "gb": (True, "native"),
        "pb": (True, "native"),
    }


def test_finalize_case_rejects_same_count_different_gb_pb_frame_ids(
    tmp_path: Path,
) -> None:
    controls = FakeControls(
        scoring_frame_ids={
            ("case_a", "rep01", "pb"): (1, 99, 3),
        }
    )
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")
    result = runner.run_job(JobKey("case_a", "rep01"))
    assert result.completed is True

    with pytest.raises(
        ContractError,
        match="observed frame-ID digests differ",
    ):
        runner.finalize_case("case_a")


def test_finalize_case_rejects_different_gb_pb_planned_frame_manifests(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    scoring = dict(project.workflow.stages["scoring"])
    models = {
        model_id: dict(model)
        for model_id, model in scoring["models"].items()
    }
    models["pb"]["frame_manifest_digest"] = "a" * 64
    scoring["models"] = models
    workflow = replace(
        project.workflow,
        stages={**project.workflow.stages, "scoring": scoring},
    )
    runner = WorkflowRunner(
        workflow,
        project.runtime,
        tmp_path / "workspace",
        providers=project.registry,
    )
    runner.run_job(JobKey("case_a", "rep01"))

    with pytest.raises(
        ContractError,
        match="planned frame-manifest digests differ",
    ):
        runner.finalize_case("case_a")


def test_any_policy_one_failure_warns_and_analysis_never_mixes_models(
    tmp_path: Path,
) -> None:
    controls = FakeControls(
        fail={("scoring", "case_a", "rep01", "pb")}
    )
    project = _project(
        tmp_path / "project",
        controls=controls,
        completion_policy="any",
    )
    runner = _runner(project, tmp_path / "workspace")

    result = runner.run_job(JobKey("case_a", "rep01"))
    assert result.completed is True
    assert set(result.model_failures) == {"pb"}

    finalized = runner.finalize_case("case_a")
    assert finalized.payload["partial"] is False
    assert finalized.payload["outcome"] == (
        WorkflowOutcome.COMPLETE_WITH_WARNINGS.value
    )
    replica = finalized.payload["replicas"][0]
    assert replica["completed_models"] == ["gb"]
    assert set(replica["scoring_models"]) == {"gb"}
    assert replica["model_statuses"]["pb"]["state"] == "failed"
    scoring_artifact = replica["scoring_models"]["gb"]["artifacts"][
        "scoring_summary"
    ]
    assert set(scoring_artifact) == {"path", "sha256", "size"}
    assert len(scoring_artifact["sha256"]) == 64
    assert scoring_artifact["size"] > 0
    assert replica["scoring_models"]["gb"]["payload"]["per_frame"] == {
        "columns": ["Frame", "DELTA_TOTAL"],
        "rows": [
            {"Frame": 1.0, "DELTA_TOTAL": -7.5},
            {"Frame": 2.0, "DELTA_TOTAL": -8.5},
            {"Frame": 3.0, "DELTA_TOTAL": -9.5},
        ],
    }

    project_result = runner.finalize_project()
    case_models = project_result.analysis.payload["cases"][0]["models"]
    assert case_models["gb"]["replica_scores"] == [-7.5]
    assert case_models["gb"]["replica_count"] == 1
    assert case_models["pb"]["replica_scores"] == []
    assert case_models["pb"]["replica_count"] == 0


def test_all_models_failing_rejects_job_even_under_any_policy(
    tmp_path: Path,
) -> None:
    controls = FakeControls(
        fail={
            ("scoring", "case_a", "rep01", "gb"),
            ("scoring", "case_a", "rep01", "pb"),
        }
    )
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")

    with pytest.raises(TaskExecutionError, match="no acceptable scoring result"):
        runner.run_job(JobKey("case_a", "rep01"))

    assert runner.workspace.latest(_scoring_task("gb")).state is AttemptState.FAILED
    assert runner.workspace.latest(_scoring_task("pb")).state is AttemptState.FAILED


def test_all_policy_rejects_one_failed_model(tmp_path: Path) -> None:
    controls = FakeControls(
        fail={("scoring", "case_a", "rep01", "pb")}
    )
    project = _project(
        tmp_path / "project",
        controls=controls,
        completion_policy="all",
    )
    runner = _runner(project, tmp_path / "workspace")

    with pytest.raises(TaskExecutionError, match="no acceptable scoring result"):
        runner.run_job(JobKey("case_a", "rep01"))

    assert runner.workspace.latest(_scoring_task("gb")).state is AttemptState.COMPLETED
    assert runner.workspace.latest(_scoring_task("pb")).state is AttemptState.FAILED


def test_stage_scoped_resource_requests_release_gpu_after_md(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")

    runner.run_job(JobKey("case_a", "rep01"))

    executed = [
        event for event in project.controls.events if event.method == "execute"
    ]
    observed = [
        (event.capability, event.model_id, event.gpu_id)
        for event in executed
    ]
    assert observed[:3] == [
        ("md", None, "0"),
        ("trajectory", None, None),
        ("trajectory_qc", None, None),
    ]
    assert set(observed[3:]) == {
        ("scoring", "gb", None),
        ("scoring", "pb", None),
    }
    tasks_and_resources = [
        (
            TaskKey(
                StageId.MD,
                case_id="case_a",
                replica_id="rep01",
            ),
            {"cpu_slots": 2, "gpu_slots": 1},
        ),
        (
            TaskKey(
                StageId.TRAJECTORY,
                case_id="case_a",
                replica_id="rep01",
            ),
            {"cpu_slots": 1, "gpu_slots": 0},
        ),
        (
            TaskKey(
                StageId.TRAJECTORY_QC,
                case_id="case_a",
                replica_id="rep01",
            ),
            {"cpu_slots": 1, "gpu_slots": 0},
        ),
        (_scoring_task("gb"), {"cpu_slots": 2, "gpu_slots": 0}),
        (_scoring_task("pb"), {"cpu_slots": 2, "gpu_slots": 0}),
    ]
    for task, expected in tasks_and_resources:
        invocation = json.loads(
            (
                runner.workspace.task_dir(task)
                / "attempts/0001/invocations/0001/invocation.json"
            ).read_text(encoding="utf-8")
        )
        assert invocation["runtime"]["resource_request"] == expected


def test_model_selection_is_rejected_before_non_scoring_execution(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")

    with pytest.raises(
        ContractError, match="only valid when through is scoring"
    ):
        runner.run_job(
            JobKey("case_a", "rep01"),
            through=StageId.TRAJECTORY_QC,
            model_ids=("gb",),
        )
    with pytest.raises(
        ContractError, match="only valid when through is scoring"
    ):
        runner.run_batch(
            through=StageId.MD,
            model_ids=("pb",),
        )
    assert project.controls.events == []


@pytest.mark.parametrize(
    ("case_ids", "replica_ids", "message"),
    [
        ((), None, "at least one case"),
        (None, (), "at least one replica"),
    ],
)
def test_batch_rejects_empty_case_or_replica_selection(
    tmp_path: Path,
    case_ids: Sequence[str] | None,
    replica_ids: Sequence[str] | None,
    message: str,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")

    with pytest.raises(ContractError, match=message):
        runner.run_batch(case_ids=case_ids, replica_ids=replica_ids)
    assert project.controls.events == []


def test_stage_contract_context_survives_scheduler_gpu_lease(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    science = resolve_protocol(
        "smoke_20ps@2",
        md_engine_profile="openmm_8_5@2",
    ).stages["md"]

    before_lease = runner._effective_stage_runtime("md", science)
    after_lease = runner._effective_stage_runtime(
        "md",
        science,
        gpu_id="1",
    )

    assert before_lease["_stage_contract"] == "mmpbsa.md-phase-plan.v2"
    assert before_lease["_md_engine_profile"] == "openmm_8_5@2"
    assert "gpu_id" not in before_lease
    assert after_lease["_stage_contract"] == before_lease["_stage_contract"]
    assert after_lease["_md_engine_profile"] == before_lease["_md_engine_profile"]
    assert after_lease["gpu_id"] == "1"


def test_scoring_models_have_isolated_attempt_hashes(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    initial_workflow = project.workflow
    first_runner = WorkflowRunner(
        initial_workflow,
        project.runtime,
        workspace,
        providers=project.registry,
    )
    first_runner.run_job(JobKey("case_a", "rep01"))
    event_count = len(project.controls.events)
    pb_before = first_runner.workspace.latest(_scoring_task("pb"))
    assert pb_before is not None

    scoring = dict(initial_workflow.stages["scoring"])
    models = {
        model_id: dict(model_science)
        for model_id, model_science in scoring["models"].items()
    }
    models["gb"]["control_text"] = "&gb igb=8 /"
    scoring["models"] = models
    changed_workflow = replace(
        initial_workflow,
        stages={**initial_workflow.stages, "scoring": scoring},
    )
    second_runner = WorkflowRunner(
        changed_workflow,
        project.runtime,
        workspace,
        providers=project.registry,
    )

    reused_pb = second_runner.run_job(
        JobKey("case_a", "rep01"),
        model_ids=("pb",),
    )

    assert reused_pb.completed is True
    assert second_runner.workspace.latest(_scoring_task("pb")).science_hash == (
        pb_before.science_hash
    )
    assert len(project.controls.events) == event_count

    rebuilt_gb = second_runner.run_job(
        JobKey("case_a", "rep01"),
        action="new",
        model_ids=("gb",),
    )

    assert rebuilt_gb.completed is True
    assert [
        item.attempt_id
        for item in second_runner.workspace.attempts(_scoring_task("gb"))
    ] == ["0001", "0002"]
    assert [
        (event.capability, event.model_id)
        for event in project.controls.events[event_count:]
    ] == [("scoring", "gb")]


def test_finalize_case_marks_stale_dependency_chain_unusable(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    initial_runner = _runner(project, workspace)
    initial_runner.run_job(JobKey("case_a", "rep01"))
    changed_workflow = replace(
        project.workflow,
        stages={
            **project.workflow.stages,
            "trajectory_qc": {"block_threshold": 0.5},
        },
    )
    changed_runner = WorkflowRunner(
        changed_workflow,
        project.runtime,
        workspace,
        providers=project.registry,
    )

    finalized = changed_runner.finalize_case("case_a", action="new")

    assert finalized.payload["outcome"] == WorkflowOutcome.PARTIAL.value
    assert finalized.payload["complete_replicas"] == []
    missing = finalized.payload["missing_replicas"][0]
    assert missing["reason"] == "no current completed trajectory_qc"
    assert missing["stages"]["md"]["usable"] is True
    assert missing["stages"]["trajectory"]["usable"] is True
    qc_status = missing["stages"]["trajectory_qc"]
    assert qc_status["state"] == "completed"
    assert qc_status["usable"] is False
    assert qc_status["reason"] == "science_hash_mismatch"
    assert qc_status["observed_science_hash"] != (
        qc_status["expected_science_hash"]
    )
    assert {
        model_id: (
            status["usable"],
            status["reason"],
        )
        for model_id, status in missing["stages"][
            "scoring_models"
        ].items()
    } == {
        "gb": (False, "upstream_unavailable"),
        "pb": (False, "upstream_unavailable"),
    }


def test_finalize_case_reports_md_backend_as_first_stale_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    _runner(project, workspace).run_job(JobKey("case_a", "rep01"))
    md_service = project.registry.service("fake", "md")
    monkeypatch.setattr(
        md_service,
        "compatibility",
        lambda runtime: BackendCompatibility(
            provider_id="fake",
            adapter_contract="md.v3",
            engine="fake-md",
            engine_major="3",
            engine_version="3.0",
        ),
    )

    finalized = _runner(project, workspace).finalize_case("case_a")

    assert finalized.payload["outcome"] == WorkflowOutcome.PARTIAL.value
    stages = finalized.payload["missing_replicas"][0]["stages"]
    assert stages["md"]["reason"] == "backend_compatibility_mismatch"
    assert stages["trajectory"]["reason"] == "upstream_unavailable"
    assert stages["trajectory_qc"]["reason"] == "upstream_unavailable"
    assert {
        model_id: status["reason"]
        for model_id, status in stages["scoring_models"].items()
    } == {"gb": "upstream_unavailable", "pb": "upstream_unavailable"}


def test_finalize_case_reports_trajectory_science_as_first_stale_stage(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    _runner(project, workspace).run_job(JobKey("case_a", "rep01"))
    changed_workflow = replace(
        project.workflow,
        stages={**project.workflow.stages, "trajectory": {"stride": 3}},
    )
    changed_runner = WorkflowRunner(
        changed_workflow,
        project.runtime,
        workspace,
        providers=project.registry,
    )

    finalized = changed_runner.finalize_case("case_a")

    assert finalized.payload["outcome"] == WorkflowOutcome.PARTIAL.value
    stages = finalized.payload["missing_replicas"][0]["stages"]
    assert stages["md"]["usable"] is True
    assert stages["trajectory"]["reason"] == "science_hash_mismatch"
    assert stages["trajectory"]["observed_science_hash"] != (
        stages["trajectory"]["expected_science_hash"]
    )
    assert stages["trajectory_qc"]["reason"] == "upstream_unavailable"


def test_finalize_case_rejects_completed_scoring_from_old_backend_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    runner = _runner(project, workspace)
    runner.run_job(JobKey("case_a", "rep01"))

    scoring = project.registry.service("fake", "scoring")
    monkeypatch.setattr(
        scoring,
        "compatibility",
        lambda runtime: BackendCompatibility(
            provider_id="fake",
            adapter_contract="scoring.v3",
            engine="fake-scoring",
            engine_major="3",
            engine_version="3.0",
        ),
    )
    finalized = _runner(project, workspace).finalize_case("case_a")

    assert finalized.payload["outcome"] == WorkflowOutcome.PARTIAL.value
    assert finalized.payload["complete_replicas"] == []
    statuses = finalized.payload["missing_replicas"][0]["stages"][
        "scoring_models"
    ]
    for status in statuses.values():
        assert status["usable"] is False
        assert status["reason"] == "backend_compatibility_mismatch"
        assert status["observed_science_hash"] == status["expected_science_hash"]
        assert status["observed_compatibility_key"] != (
            status["expected_compatibility_key"]
        )


def test_finalize_case_reports_unavailable_scoring_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    _runner(project, workspace).run_job(JobKey("case_a", "rep01"))
    scoring = project.registry.service("fake", "scoring")

    def unavailable(runtime: Mapping[str, object]) -> BackendCompatibility:
        raise RuntimeError("scoring engine probe failed")

    monkeypatch.setattr(scoring, "compatibility", unavailable)
    finalized = _runner(project, workspace).finalize_case("case_a")

    assert finalized.payload["outcome"] == WorkflowOutcome.PARTIAL.value
    assert finalized.payload["complete_replicas"] == []
    statuses = finalized.payload["missing_replicas"][0]["stages"][
        "scoring_models"
    ]
    for status in statuses.values():
        assert status["usable"] is False
        assert status["reason"] == "backend_unavailable"
        assert status["backend_error"] == {
            "type": "RuntimeError",
            "message": "scoring engine probe failed",
        }
        assert status["expected_compatibility_key"] is None
        assert status["observed_science_hash"] == status["expected_science_hash"]


def test_finalize_case_does_not_hide_artifact_integrity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    _runner(project, workspace).run_job(JobKey("case_a", "rep01"))
    scoring = project.registry.service("fake", "scoring")

    def corrupt(runtime: Mapping[str, object]) -> BackendCompatibility:
        raise ArtifactIntegrityError("scoring artifact changed")

    monkeypatch.setattr(scoring, "compatibility", corrupt)
    with pytest.raises(ArtifactIntegrityError, match="scoring artifact changed"):
        _runner(project, workspace).finalize_case("case_a")


def test_finalize_project_excludes_stale_finalized_case(
    tmp_path: Path,
) -> None:
    project = _project(
        tmp_path / "project",
        case_ids=("case_a", "case_b"),
    )
    workspace = tmp_path / "workspace"
    initial_runner = _runner(project, workspace)
    initial_runner.run_job(JobKey("case_a", "rep01"))
    initial_runner.run_job(JobKey("case_b", "rep01"))
    initial_runner.finalize_case("case_a")
    initial_runner.finalize_case("case_b")

    changed_workflow = replace(
        project.workflow,
        cases=(
            replace(project.workflow.case("case_a"), label="Changed case A"),
            project.workflow.case("case_b"),
        ),
    )
    changed_runner = WorkflowRunner(
        changed_workflow,
        project.runtime,
        workspace,
        providers=project.registry,
    )

    project_result = changed_runner.finalize_project(action="new")

    assert project_result.analysis.payload["outcome"] == (
        WorkflowOutcome.PARTIAL.value
    )
    cases = {
        case["case_id"]: case
        for case in project_result.analysis.payload["cases"]
    }
    assert cases["case_a"]["finalized"] is False
    assert cases["case_a"]["outcome"] == "missing"
    assert cases["case_a"]["record"] is None
    assert cases["case_a"]["models"]["gb"]["replica_count"] == 0
    assert cases["case_a"]["models"]["pb"]["replica_count"] == 0
    assert cases["case_b"]["finalized"] is True
    assert cases["case_b"]["outcome"] == WorkflowOutcome.COMPLETE.value


def test_finalize_project_requires_one_current_finalized_case(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")

    with pytest.raises(
        ContractError,
        match="requires at least one finalized case",
    ):
        runner.finalize_project()


def test_report_assets_exclude_stale_trajectory_chain(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project", with_assets=True)
    workspace = tmp_path / "workspace"
    initial_runner = _runner(project, workspace)
    initial_runner.run_job(JobKey("case_a", "rep01"))
    initial_runner.finalize_case("case_a")
    project_result = initial_runner.finalize_project()
    analysis_input = initial_runner.workspace.as_stage_input(
        project_result.analysis
    )
    changed_workflow = replace(
        project.workflow,
        stages={
            **project.workflow.stages,
            "trajectory": {"stride": 3},
        },
    )
    changed_runner = WorkflowRunner(
        changed_workflow,
        project.runtime,
        workspace,
        providers=project.registry,
    )

    assets = changed_runner._asset_inputs(analysis_input)

    assert "prepared:case_a" in assets
    assert "trajectory:case_a:rep01" not in assets


def test_explicit_resume_appends_invocation_to_same_attempt(
    tmp_path: Path,
) -> None:
    controls = FakeControls(
        interrupt_once={("md", "case_a", "rep01", None)}
    )
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")
    job = JobKey("case_a", "rep01")

    with pytest.raises(ControlledInterrupt):
        runner.run_job(job)
    task = TaskKey(StageId.MD, case_id="case_a", replica_id="rep01")
    interrupted = runner.workspace.latest(task)
    assert interrupted is not None
    assert interrupted.state is AttemptState.INTERRUPTED
    assert interrupted.attempt_id == "0001"

    result = runner.run_job(job, action="resume", model_ids=("gb",))
    resumed = next(
        item for item in result.attempts if item.task.stage is StageId.MD
    )
    invocation_root = (
        runner.workspace.task_dir(task) / "attempts/0001/invocations"
    )
    assert resumed.state is AttemptState.COMPLETED
    assert resumed.attempt_id == "0001"
    assert sorted(path.name for path in invocation_root.iterdir()) == [
        "0001",
        "0002",
    ]
    assert [
        event.method
        for event in controls.events
        if event.capability == "md"
    ] == ["execute", "resume"]


def test_batch_continues_other_replicas_after_non_scoring_failure(
    tmp_path: Path,
) -> None:
    controls = FakeControls(
        fail={("md", "case_a", "rep02", None)}
    )
    project = _project(
        tmp_path / "project",
        replica_ids=("rep01", "rep02", "rep03"),
        controls=controls,
    )
    runner = _runner(project, tmp_path / "workspace")

    result = runner.run_batch()

    assert set(result.results) == {
        JobKey("case_a", "rep01"),
        JobKey("case_a", "rep03"),
    }
    assert set(result.failures) == {JobKey("case_a", "rep02")}
    assert result.outcome is WorkflowOutcome.PARTIAL
    assert all(item.completed for item in result.results.values())
    assert {
        event.replica_id
        for event in controls.events
        if event.capability == "trajectory_qc"
    } == {"rep01", "rep03"}


def test_batch_qc_warn_is_complete_with_warnings(tmp_path: Path) -> None:
    controls = FakeControls(warned_qc={("case_a", "rep01")})
    project = _project(
        tmp_path / "project",
        controls=controls,
    )
    runner = _runner(project, tmp_path / "workspace")

    result = runner.run_batch()

    assert not result.failures
    assert result.completed is True
    assert result.outcome is WorkflowOutcome.COMPLETE_WITH_WARNINGS


def test_partial_finalize_case_records_model_specific_inventory(
    tmp_path: Path,
) -> None:
    project = _project(
        tmp_path / "project", replica_ids=("rep01", "rep02")
    )
    runner = _runner(project, tmp_path / "workspace")
    runner.run_job(JobKey("case_a", "rep01"))

    finalized = runner.finalize_case("case_a")

    assert finalized.payload["partial"] is True
    assert finalized.payload["expected_replicas"] == ["rep01", "rep02"]
    assert finalized.payload["expected_models"] == ["gb", "pb"]
    assert finalized.payload["complete_replicas"] == ["rep01"]
    missing = finalized.payload["missing_replicas"]
    assert missing[0]["replica_id"] == "rep02"
    assert missing[0]["reason"] == "missing completed trajectory_qc"
    assert missing[0]["stages"]["md"] == {
        "state": "absent",
        "attempt_id": None,
    }
    assert missing[0]["stages"]["trajectory"] == {
        "state": "absent",
        "attempt_id": None,
    }
    assert missing[0]["stages"]["trajectory_qc"] == {
        "state": "absent",
        "attempt_id": None,
    }
    assert missing[0]["stages"]["scoring_models"] == {
        "gb": {"state": "absent", "attempt_id": None},
        "pb": {"state": "absent", "attempt_id": None},
    }
    assert finalized.payload["outcome"] == WorkflowOutcome.PARTIAL.value


def test_finalize_case_identity_includes_expected_replica_inventory(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    first_runner = _runner(project, workspace)
    first_runner.run_job(JobKey("case_a", "rep01"))
    first = first_runner.finalize_case("case_a")
    assert first.payload["partial"] is False

    expanded_workflow = replace(
        project.workflow,
        replicas=(
            *project.workflow.replicas,
            ReplicaSpec(replica_id="rep02", replica_index=2, seed=41002),
        ),
    )
    second_runner = WorkflowRunner(
        expanded_workflow,
        project.runtime,
        workspace,
        providers=project.registry,
    )
    with pytest.raises(RetryError, match="different science"):
        second_runner.finalize_case("case_a")
    second = second_runner.finalize_case("case_a", action="new")

    assert second.attempt_id == "0002"
    assert second.science_hash != first.science_hash
    assert second.payload["expected_replicas"] == ["rep01", "rep02"]
    assert second.payload["missing_replicas"][0]["replica_id"] == "rep02"


def test_prepared_case_is_revalidated_between_long_running_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    md_service = project.registry.service("fake", "md")
    original_execute = md_service.execute

    def execute_then_mutate(request: StageRequest) -> StageResult:
        result = original_execute(request)
        source = project.workflow.case("case_a").prepared_case.parent / "md/system.bin"
        assert request.prepared_case.artifact("md_topology") != source
        source.write_bytes(b"changed")
        return result

    monkeypatch.setattr(md_service, "execute", execute_then_mutate)

    with pytest.raises(ArtifactIntegrityError, match="size changed|digest changed"):
        runner.run_job(JobKey("case_a", "rep01"))


def test_md_science_hash_includes_declared_gromacs_include_bytes(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    manifest_path = project.workflow.case("case_a").prepared_case
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    topology = manifest_path.parent / "md/system.bin"
    topology_data = b'#include "forcefield/ligand.itp"\n[ system ]\nTest\n'
    topology.write_bytes(topology_data)
    topology_ref = next(
        item for item in raw["md"]["artifacts"] if item["role"] == "md_topology"
    )
    topology_ref["sha256"] = hashlib.sha256(topology_data).hexdigest()
    topology_ref["size"] = len(topology_data)
    include = _artifact(
        manifest_path.parent,
        "md/forcefield/ligand.itp",
        "ligand_parameters",
        b"[ moleculetype ]\nLIG 3\n",
    )
    raw["md"]["artifacts"].append(include)
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    runner = _runner(project, tmp_path / "workspace")
    first_case = runner._prepared("case_a")
    first = _prepared_science_identity(runner, first_case, StageId.MD)
    assert {item["role"] for item in first["prepared_case"]["md"]["artifacts"]} == {
        "md_topology",
        "md_coordinates",
        "ligand_parameters",
    }

    new_data = b"[ moleculetype ]\nLIG 4\n"
    (manifest_path.parent / include["path"]).write_bytes(new_data)
    include["sha256"] = hashlib.sha256(new_data).hexdigest()
    include["size"] = len(new_data)
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    second = _prepared_science_identity(
        runner, runner._prepared("case_a"), StageId.MD
    )
    assert first != second


def test_attempt_md_input_snapshot_is_independent_and_compatibility_is_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    md_service = project.registry.service("fake", "md")
    original_execute = md_service.execute
    observed: list[Path] = []

    def check_input(request: StageRequest) -> StageResult:
        assert request.compatibility == md_service.compatibility(request.runtime)
        snapshot = request.prepared_case.artifact("md_topology")
        assert snapshot.is_relative_to(request.attempt_dir / "prepared_inputs")
        observed.append(snapshot)
        return original_execute(request)

    monkeypatch.setattr(md_service, "execute", check_input)
    runner.run_job(JobKey("case_a", "rep01"), through=StageId.MD)
    assert len(observed) == 1
    expected = observed[0].read_bytes()
    source = project.workflow.case("case_a").prepared_case.parent / "md/system.bin"
    source.write_bytes(b"changed after attempt")
    assert observed[0].read_bytes() == expected


def test_attempt_snapshot_rejects_source_change_after_prepared_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    md_service = project.registry.service("fake", "md")
    original_compatibility = md_service.compatibility

    def mutate_during_compatibility(runtime: Mapping[str, object]) -> BackendCompatibility:
        source = project.workflow.case("case_a").prepared_case.parent / "md/system.bin"
        source.write_bytes(b"changed before snapshot")
        return original_compatibility(runtime)

    monkeypatch.setattr(md_service, "compatibility", mutate_during_compatibility)
    with pytest.raises(ArtifactIntegrityError, match="source changed"):
        runner.run_job(JobKey("case_a", "rep01"), through=StageId.MD)


def test_resume_rejects_tampered_prepared_snapshot(tmp_path: Path) -> None:
    controls = FakeControls(interrupt_once={("md", "case_a", "rep01", None)})
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")
    job = JobKey("case_a", "rep01")
    with pytest.raises(ControlledInterrupt):
        runner.run_job(job, through=StageId.MD)
    task = TaskKey(StageId.MD, case_id="case_a", replica_id="rep01")
    snapshot = runner.workspace.task_dir(task) / "attempts/0001/prepared_inputs/md/system.bin"
    snapshot.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="snapshot differs"):
        runner.run_job(job, through=StageId.MD, action="resume")


def test_resume_rejects_symlinked_prepared_snapshot_parent(tmp_path: Path) -> None:
    controls = FakeControls(interrupt_once={("md", "case_a", "rep01", None)})
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")
    job = JobKey("case_a", "rep01")
    with pytest.raises(ControlledInterrupt):
        runner.run_job(job, through=StageId.MD)
    task = TaskKey(StageId.MD, case_id="case_a", replica_id="rep01")
    snapshot_root = runner.workspace.task_dir(task) / "attempts/0001/prepared_inputs"
    (snapshot_root / "md").rename(snapshot_root / "md_original")
    (snapshot_root / "md").symlink_to(snapshot_root / "md_original", target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="symlinked parent"):
        runner.run_job(job, through=StageId.MD, action="resume")


@pytest.mark.parametrize("damage", ("tamper", "symlink"))
def test_completed_attempt_reuse_rechecks_prepared_snapshot(
    tmp_path: Path,
    damage: str,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    job = JobKey("case_a", "rep01")
    runner.run_job(job, through=StageId.MD)
    task = TaskKey(StageId.MD, case_id="case_a", replica_id="rep01")
    snapshot_root = runner.workspace.task_dir(task) / "attempts/0001/prepared_inputs"
    if damage == "tamper":
        (snapshot_root / "md/system.bin").write_bytes(b"tampered")
        expected = "snapshot differs"
    else:
        (snapshot_root / "md").rename(snapshot_root / "md_original")
        (snapshot_root / "md").symlink_to(
            snapshot_root / "md_original", target_is_directory=True
        )
        expected = "symlinked parent"
    with pytest.raises(ArtifactIntegrityError, match=expected):
        runner.run_job(job, through=StageId.MD)


def test_explicit_new_authorizes_changed_upstream_attempt_chain(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    first_runner = _runner(project, workspace)
    first_runner.run_job(JobKey("case_a", "rep01"))

    changed_workflow = replace(
        project.workflow,
        stages={**project.workflow.stages, "md": {"steps": 20}},
    )
    second_runner = WorkflowRunner(
        changed_workflow,
        project.runtime,
        workspace,
        providers=project.registry,
    )

    result = second_runner.run_job(
        JobKey("case_a", "rep01"), action="new"
    )

    by_task = {item.task: item for item in result.attempts}
    assert by_task[
        TaskKey(StageId.ADMIT, case_id="case_a")
    ].attempt_id == "0001"
    for stage in (StageId.MD, StageId.TRAJECTORY, StageId.TRAJECTORY_QC):
        assert by_task[
            TaskKey(stage, case_id="case_a", replica_id="rep01")
        ].attempt_id == "0002"
    assert by_task[_scoring_task("gb")].attempt_id == "0002"
    assert by_task[_scoring_task("pb")].attempt_id == "0002"


def test_blocked_qc_reaches_analysis_and_stable_report_publication(
    tmp_path: Path,
) -> None:
    controls = FakeControls(blocked_qc={("case_a", "rep01")})
    project = _project(
        tmp_path / "project",
        controls=controls,
        with_assets=True,
    )
    runner = _runner(project, tmp_path / "workspace")
    runner.run_job(JobKey("case_a", "rep01"))
    runner.finalize_case("case_a")

    first = runner.finalize_project()
    stable_path = (tmp_path / "workspace" / "report" / "index.html").resolve()
    report_attempt_index = (
        runner.workspace.task_dir(first.report.task)
        / "attempts"
        / first.report.attempt_id
        / "output"
        / "index.html"
    )
    assert first.publication.resolve() == stable_path
    assert not stable_path.samefile(report_attempt_index)
    assert stable_path.stat().st_nlink == 1
    assert "scientifically_blocked" in stable_path.read_text(encoding="utf-8")
    assert first.analysis.payload["cases"][0]["outcome"] == (
        "scientifically_blocked"
    )
    assert first.assets is not None
    assert (stable_path.parent / "assets" / "display.json").is_file()
    inventory = json.loads(
        (stable_path.parent / "publication_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert {item["path"] for item in inventory["artifacts"]} == {
        "index.html",
        "assets/display.json",
    }

    stale = stable_path.parent / "stale-from-old-publication.txt"
    stale.write_text("must be replaced", encoding="utf-8")
    rebuilt = runner.rebuild_report()

    assert rebuilt.publication.resolve() == stable_path
    assert rebuilt.analysis.attempt_id == first.analysis.attempt_id
    assert rebuilt.assets is not None
    assert rebuilt.assets.attempt_id == "0002"
    assert rebuilt.report.attempt_id == "0002"
    assert not stale.exists()
    assert "fake report 2" in stable_path.read_text(encoding="utf-8")
    assert json.loads(
        (stable_path.parent / "assets" / "display.json").read_text(
            encoding="utf-8"
        )
    )["generation"] == 2


def test_stable_report_publication_reaudits_final_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    report_service = project.registry.service("fake", "report")
    original_execute = report_service.execute

    def execute_with_broken_link(request: StageRequest) -> StageResult:
        result = original_execute(request)
        result.artifacts["report_index"].write_text(
            '<!doctype html><a href="missing.csv">missing</a>',
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(report_service, "execute", execute_with_broken_link)
    runner.run_job(JobKey("case_a", "rep01"))
    runner.finalize_case("case_a")

    with pytest.raises(ValueError, match="Report link audit failed"):
        runner.finalize_project()
    assert not (tmp_path / "workspace" / "report").exists()


def test_stable_report_publication_reaudits_portable_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    report_service = project.registry.service("fake", "report")
    original_execute = report_service.execute

    def execute_with_broken_manifest(request: StageRequest) -> StageResult:
        result = original_execute(request)
        manifest = request.attempt_dir / "output" / "assets" / "display_assets.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "mmpbsa.display_assets.v4",
                    "topologies": {"rep01": "structures/missing.pdb"},
                    "trajectories": {},
                    "pymol_scripts": {},
                }
            ),
            encoding="utf-8",
        )
        return replace(
            result,
            artifacts={**result.artifacts, "report_asset_manifest": manifest},
        )

    monkeypatch.setattr(report_service, "execute", execute_with_broken_manifest)
    runner.run_job(JobKey("case_a", "rep01"))
    runner.finalize_case("case_a")

    with pytest.raises(ValueError, match="Portable manifest audit failed"):
        runner.finalize_project()
    assert not (tmp_path / "workspace" / "report").exists()


def test_stable_report_publication_rejects_undeclared_staging_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    runner.run_job(JobKey("case_a", "rep01"))
    runner.finalize_case("case_a")

    from mmpbsa.core import runner as runner_module

    real_materialize = runner_module.materialize_immutable_file

    def materialize_with_leak(source: Path, destination: Path) -> Path:
        result = real_materialize(source, destination)
        if destination.name == "index.html":
            (destination.parent / "undeclared.txt").write_text(
                "not in the sealed report inventory",
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(
        runner_module,
        "materialize_immutable_file",
        materialize_with_leak,
    )
    with pytest.raises(
        ArtifactIntegrityError,
        match="tree differs from its inventory",
    ):
        runner.finalize_project()
    assert not (tmp_path / "workspace" / "report").exists()


def test_stable_report_verifies_published_hashes_under_publication_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    workspace = tmp_path / "workspace"
    runner = _runner(project, workspace)
    runner.run_job(JobKey("case_a", "rep01"))
    runner.finalize_case("case_a")

    from mmpbsa.core import runner as runner_module

    real_sha256_file = runner_module.sha256_file
    published_checks: list[bool] = []

    def sha256_while_locked(path: Path) -> str:
        if path.is_relative_to(workspace / "report"):
            published_checks.append(
                (workspace / ".report-publication.lock").is_file()
            )
        return real_sha256_file(path)

    monkeypatch.setattr(runner_module, "sha256_file", sha256_while_locked)
    runner.finalize_project()

    assert published_checks
    assert all(published_checks)


def test_core_compatibility_preserves_analysis_v2_and_bumps_finalizer() -> None:
    compatibility = WorkflowRunner._core_compatibility(StageId.ANALYSIS)
    finalizer = WorkflowRunner._core_compatibility(StageId.FINALIZE_CASE)
    legacy_finalizer = replace(
        finalizer,
        adapter_contract="stage.finalize_case.v2",
    )

    assert compatibility.adapter_contract == "stage.analysis.v2"
    assert compatibility.engine_major == "0.5"
    assert compatibility.engine_version == "0.7.0"
    assert finalizer.adapter_contract == "stage.finalize_case.v3"
    assert finalizer.key != legacy_finalizer.key

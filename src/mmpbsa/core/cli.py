"""Unified command line for the 0.7 workflow framework."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .config import WORKFLOW_SCHEMA, load_runtime, load_workflow
from .errors import MmpbsaError
from .models import AttemptSummary, JobKey, TaskKey
from .prepared import load_prepared_case
from .prepared_draft import (
    inspect_prepared_case_draft,
    seal_prepared_case_draft,
    validate_prepared_case_draft,
)
from .providers import discover_providers
from .runner import (
    BatchResult,
    ProjectResult,
    WorkflowRunner,
    effective_stage_runtime,
)
from .state import REPLICA_STAGES, StageId


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmpbsa",
        description="Replica-addressable MMPBSA workflow framework",
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=Path("workflow.yaml"),
        help="versioned scientific workflow YAML",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("runtime.yaml"),
        help="host/runtime YAML",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("_workspace"),
        help="canonical framework-owned workspace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser(
        "config",
        help="check, resolve, or explain the typed scientific workflow",
    )
    config_sub = config.add_subparsers(dest="config_action", required=True)
    config_sub.add_parser("check", help="validate workflow and PreparedCase inputs")
    config_sub.add_parser("resolve", help="print canonical resolved science")
    config_sub.add_parser("explain", help="summarize preset decisions and frame counts")
    commands.add_parser("validate", help="validate configs and PreparedCase inputs")
    commands.add_parser("doctor", help="discover and probe configured providers")
    status = commands.add_parser("status", help="read-only current workspace status")
    status.add_argument("--summary", action="store_true", help="stage/model counts and telemetry")

    prepared_case = commands.add_parser(
        "prepared-case",
        help="validate, inspect, or seal existing prepared artifacts",
    )
    prepared_case_sub = prepared_case.add_subparsers(dest="prepared_case_action", required=True)
    prepared_validate = prepared_case_sub.add_parser(
        "validate",
        help="validate either a strict draft or a sealed PreparedCase v2",
    )
    prepared_validate_input = prepared_validate.add_mutually_exclusive_group(required=True)
    prepared_validate_input.add_argument("--draft", type=Path)
    prepared_validate_input.add_argument("--manifest", type=Path)
    prepared_inspect = prepared_case_sub.add_parser(
        "inspect",
        help="inspect a strict PreparedCase draft without publishing it",
    )
    prepared_inspect.add_argument("--draft", type=Path, required=True)
    seal = prepared_case_sub.add_parser(
        "seal",
        help="atomically publish a validated draft as PreparedCase v2",
    )
    seal.add_argument("--draft", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)

    workspace = commands.add_parser(
        "workspace", help="inspect legacy workflow/workspace schemas read-only"
    )
    workspace_sub = workspace.add_subparsers(dest="workspace_action", required=True)
    inspect = workspace_sub.add_parser(
        "inspect", help="read-only inventory of legacy workflow/workspace state"
    )
    inspect.add_argument("--legacy-workspace", type=Path)
    inspect.add_argument("--legacy-workflow", type=Path)
    prune = workspace_sub.add_parser(
        "prune", help="plan or explicitly delete old assets/report attempts"
    )
    prune.add_argument(
        "--stage",
        action="append",
        choices=(StageId.ASSETS.value, StageId.REPORT.value),
        required=True,
    )
    prune.add_argument("--keep", type=int, default=1)
    prune_mode = prune.add_mutually_exclusive_group(required=True)
    prune_mode.add_argument("--dry-run", action="store_true")
    prune_mode.add_argument("--confirm", action="store_true")

    job = commands.add_parser("job", help="operate on one case/replica")
    job_sub = job.add_subparsers(dest="job_action", required=True)
    for action in ("plan", "run", "resume", "retry"):
        item = job_sub.add_parser(action)
        _job_arguments(item)
        if action == "run":
            item.add_argument(
                "--new",
                action="store_true",
                help="force a new attempt for the requested through-stage",
            )

    batch = commands.add_parser("batch", help="operate on a set of atomic jobs")
    batch_sub = batch.add_subparsers(dest="batch_action", required=True)
    for action in ("plan", "run", "resume", "retry"):
        item = batch_sub.add_parser(action)
        item.add_argument("--case", action="append", dest="case_ids")
        item.add_argument("--replica", action="append", dest="replica_ids")
        item.add_argument("--model", action="append", dest="model_ids")
        item.add_argument(
            "--through",
            choices=[stage.value for stage in REPLICA_STAGES],
            default=StageId.SCORING.value,
        )
        if action == "run":
            item.add_argument(
                "--new",
                action="store_true",
                help="force new through-stage attempts for every selected job",
            )
            item.add_argument(
                "--resume-md",
                action="store_true",
                help=(
                    "with --new in pipeline mode, preserve matching MD "
                    "and resume abandoned active MD"
                ),
            )

    finalize = commands.add_parser("finalize", help="explicitly finalize a case or project")
    finalize_sub = finalize.add_subparsers(dest="finalize_scope", required=True)
    case = finalize_sub.add_parser("case")
    case.add_argument("--case", required=True, dest="case_id")
    _action_flags(case)
    project = finalize_sub.add_parser("project")
    _action_flags(project)

    report = commands.add_parser(
        "report",
        help="transactionally rebuild assets and HTML from current analysis",
    )
    report.add_argument(
        "--reuse-assets",
        action="store_true",
        help="verify and reuse compatible current assets; rebuild only HTML",
    )
    from .project_cli import add_project_commands

    add_project_commands(commands)
    return parser


def _job_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case", required=True, dest="case_id")
    parser.add_argument("--replica", required=True, dest="replica_id")
    parser.add_argument("--model", action="append", dest="model_ids")
    parser.add_argument(
        "--through",
        choices=[StageId.ADMIT.value, *[stage.value for stage in REPLICA_STAGES]],
        default=StageId.SCORING.value,
    )


def _action_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--retry", action="store_true")
    group.add_argument("--new", action="store_true")


def _selected_action(arguments: argparse.Namespace) -> str:
    if getattr(arguments, "resume", False):
        return "resume"
    if getattr(arguments, "retry", False):
        return "retry"
    if getattr(arguments, "new", False):
        return "new"
    return "run"


def _attempt_json(value: AttemptSummary | None, *, include_payload: bool = False) -> object:
    if value is None:
        return None
    result = {
        "task": value.task.label,
        "attempt_id": value.attempt_id,
        "state": value.state.value,
        "artifact_state": value.artifact_state.value,
        "science_hash": value.science_hash,
        "compatibility_key": value.compatibility_key,
        "qc_verdict": (None if value.qc_verdict is None else value.qc_verdict.value),
        "artifacts": {role: artifact.as_dict() for role, artifact in value.artifacts.items()},
    }
    if include_payload:
        result["payload"] = dict(value.payload)
    return result


def _task_json(task: TaskKey) -> dict[str, object]:
    """Emit the same exact four-field task identity used by workspace v2."""

    return {
        "stage": task.stage.value,
        "case_id": task.case_id,
        "replica_id": task.replica_id,
        "model_id": task.model_id,
    }


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _load(arguments: argparse.Namespace):
    workflow = load_workflow(arguments.workflow)
    runtime = load_runtime(arguments.runtime)
    return workflow, runtime


def _runner(arguments: argparse.Namespace) -> WorkflowRunner:
    workflow, runtime = _load(arguments)
    _require_executable_workflow(workflow)
    if arguments.command in {"report", "finalize"}:
        from .campaign import postprocessing_runtime

        runtime = postprocessing_runtime(runtime)
    return WorkflowRunner(
        workflow,
        runtime,
        arguments.workspace,
        providers=discover_providers(),
    )


def _require_executable_workflow(workflow) -> None:
    if workflow.schema_version != WORKFLOW_SCHEMA:
        raise ValueError(
            f"{workflow.schema_version} is read-only legacy configuration; "
            f"execution requires {WORKFLOW_SCHEMA}"
        )


def _all_tasks(runner: WorkflowRunner) -> Sequence[TaskKey]:
    tasks: list[TaskKey] = []
    for case in runner.workflow.cases:
        tasks.append(TaskKey(StageId.ADMIT, case_id=case.case_id))
        for replica in runner.workflow.replicas:
            for stage in REPLICA_STAGES:
                if stage is StageId.SCORING:
                    tasks.extend(
                        TaskKey(
                            stage,
                            case_id=case.case_id,
                            replica_id=replica.replica_id,
                            model_id=model_id,
                        )
                        for model_id in runner.workflow.scoring_model_ids
                    )
                else:
                    tasks.append(
                        TaskKey(
                            stage,
                            case_id=case.case_id,
                            replica_id=replica.replica_id,
                        )
                    )
        tasks.append(TaskKey(StageId.FINALIZE_CASE, case_id=case.case_id))
    tasks.extend(TaskKey(stage) for stage in (StageId.ANALYSIS, StageId.ASSETS, StageId.REPORT))
    return tasks


def _validate(arguments: argparse.Namespace) -> int:
    workflow, runtime = _load(arguments)
    cases = []
    for case in workflow.cases:
        prepared = load_prepared_case(case.prepared_case, expected_case_id=case.case_id)
        cases.append(
            {
                "case_id": case.case_id,
                "manifest": str(prepared.manifest_path),
                "artifact_count": len(prepared.resolved_artifacts),
            }
        )
    _json(
        {
            "valid": True,
            "workflow_schema": workflow.schema_version,
            "runtime_schema": runtime.schema_version,
            "project_id": workflow.project_id,
            "protocol_id": workflow.protocol_id,
            "profile": {
                "profile_id": workflow.profile.profile_id,
                "profile_version": workflow.profile.profile_version,
            },
            "cases": cases,
            "replicas": [replica.replica_id for replica in workflow.replicas],
            "scoring": {
                "completion_policy": workflow.completion_policy,
                "models": list(workflow.scoring_model_ids),
            },
            "scheduler": {
                "cpu_slots": runtime.scheduler.cpu_slots,
                "gpu_ids": list(runtime.scheduler.gpu_ids),
            },
        }
    )
    return 0


def _config_command(arguments: argparse.Namespace) -> int:
    workflow = load_workflow(arguments.workflow)
    if arguments.config_action == "resolve":
        _json(workflow.normalized_science())
        return 0
    if arguments.config_action == "explain":
        resolved = workflow.resolved_protocol
        assert resolved is not None
        frame_plan = resolved.preset.frame_plan
        _json(
            {
                "workflow_schema": workflow.schema_version,
                "preset": resolved.preset.preset_id,
                "models": list(resolved.models),
                "completion_policy": resolved.completion_policy,
                "replicas": resolved.preset.replicas,
                "production_ns_per_replica": frame_plan.production_ps / 1000,
                "analysis_window_ns": [
                    frame_plan.analysis_start_ps / 1000,
                    frame_plan.analysis_end_ps / 1000,
                ],
                "stored_trajectory_interval_ps": frame_plan.trajectory_interval_ps,
                "scoring_interval_ps": frame_plan.scoring_interval_ps,
                "frames_per_model_per_replica": frame_plan.expected_frames,
                "frames_per_model_total": (frame_plan.expected_frames * resolved.preset.replicas),
                "settings": dict(resolved.settings),
                "engine_overrides": {
                    engine: {section: dict(values) for section, values in sections.items()}
                    for engine, sections in resolved.engine_overrides.items()
                },
            }
        )
        return 0

    _require_executable_workflow(workflow)
    cases = []
    for case in workflow.cases:
        prepared = load_prepared_case(
            case.prepared_case,
            expected_case_id=case.case_id,
        )
        cases.append(
            {
                "case_id": case.case_id,
                "manifest": str(prepared.manifest_path),
                "entities": [
                    {
                        "entity_id": entity.entity_id,
                        "kind": entity.kind,
                    }
                    for entity in prepared.manifest.entities
                ],
                "primary_binding": prepared.manifest.primary_binding.as_dict(),
            }
        )
    _json(
        {
            "valid": True,
            "executable": True,
            "workflow_schema": workflow.schema_version,
            "project_id": workflow.project_id,
            "preset": workflow.resolved_protocol.preset.preset_id,
            "cases": cases,
        }
    )
    return 0


def _prepared_case_command(arguments: argparse.Namespace) -> int:
    if arguments.prepared_case_action == "validate":
        if arguments.draft is not None:
            inspection = validate_prepared_case_draft(arguments.draft)
            _json(inspection.validation_summary())
        else:
            prepared = load_prepared_case(arguments.manifest)
            _json(
                {
                    "valid": True,
                    "schema_version": prepared.manifest.schema_version,
                    "case_id": prepared.case_id,
                    "manifest": str(prepared.manifest_path),
                    "artifact_count": len(prepared.resolved_artifacts),
                }
            )
        return 0
    if arguments.prepared_case_action == "inspect":
        _json(inspect_prepared_case_draft(arguments.draft).as_dict())
        return 0
    if arguments.prepared_case_action == "seal":
        prepared = seal_prepared_case_draft(arguments.draft, arguments.output)
        _json(
            {
                "sealed": True,
                "schema_version": prepared.manifest.schema_version,
                "case_id": prepared.case_id,
                "root": str(prepared.root),
                "manifest": str(prepared.manifest_path),
                "artifact_count": len(prepared.resolved_artifacts),
            }
        )
        return 0
    raise AssertionError(f"unhandled prepared-case action: {arguments.prepared_case_action}")


def _doctor(arguments: argparse.Namespace) -> int:
    workflow, runtime = _load(arguments)
    registry = discover_providers()
    probes: list[dict[str, object]] = []
    ok = True
    for capability, provider_id in sorted(runtime.providers.items()):
        try:
            service = registry.service(provider_id, capability)
            compatibility_method = getattr(service, "compatibility", None)
            if not callable(compatibility_method):
                raise TypeError("service has no compatibility()")
            stage = {
                "md": StageId.MD,
                "trajectory": StageId.TRAJECTORY,
            }.get(capability)
            science = {} if stage is None else workflow.stage_science(stage)
            compatibility = compatibility_method(
                effective_stage_runtime(runtime, capability, science)
            )
            probes.append(
                {
                    "capability": capability,
                    "provider_id": provider_id,
                    "ok": True,
                    "compatibility": compatibility.as_dict(),
                }
            )
        except Exception as exc:
            ok = False
            probes.append(
                {
                    "capability": capability,
                    "provider_id": provider_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    _json(
        {
            "ok": ok,
            "providers": [
                {
                    "provider_id": item.provider_id,
                    "api_version": item.api_version,
                    "package_version": item.package_version,
                    "capabilities": item.capabilities,
                }
                for item in registry.describe()
            ],
            "probes": probes,
        }
    )
    return 0 if ok else 1


def _plan_jobs(arguments: argparse.Namespace, *, batch: bool) -> int:
    workflow, runtime = _load(arguments)
    _require_executable_workflow(workflow)
    case_ids = (
        list(dict.fromkeys(arguments.case_ids))
        if batch and arguments.case_ids
        else ([item.case_id for item in workflow.cases] if batch else [arguments.case_id])
    )
    replica_ids = (
        list(dict.fromkeys(arguments.replica_ids))
        if batch and arguments.replica_ids
        else ([item.replica_id for item in workflow.replicas] if batch else [arguments.replica_id])
    )
    through = StageId(arguments.through)
    selected_models = (
        tuple(dict.fromkeys(arguments.model_ids))
        if arguments.model_ids
        else workflow.scoring_model_ids
    )
    unknown_models = set(selected_models).difference(workflow.scoring_model_ids)
    if unknown_models:
        raise ValueError("unknown scoring models: " + ", ".join(sorted(unknown_models)))
    if through is not StageId.SCORING and arguments.model_ids:
        raise ValueError("--model is only valid when --through=scoring")
    stages = [StageId.ADMIT]
    if through is not StageId.ADMIT:
        for stage in REPLICA_STAGES:
            stages.append(stage)
            if stage is through:
                break
    jobs = []
    for case_id, replica_id in (
        (case_id, replica_id) for case_id in case_ids for replica_id in replica_ids
    ):
        workflow.case(case_id)
        workflow.replica(replica_id)
        tasks = [
            TaskKey(
                stage,
                case_id=case_id,
                replica_id=(None if stage is StageId.ADMIT else replica_id),
            )
            for stage in stages
            if stage is not StageId.SCORING
        ]
        if StageId.SCORING in stages:
            tasks.extend(
                TaskKey(
                    StageId.SCORING,
                    case_id=case_id,
                    replica_id=replica_id,
                    model_id=model_id,
                )
                for model_id in selected_models
            )
        jobs.append(
            {
                "case_id": case_id,
                "replica_id": replica_id,
                "tasks": [_task_json(task) for task in tasks],
            }
        )
    _json(
        {
            "project_id": workflow.project_id,
            "through": through.value,
            "models": (list(selected_models) if through is StageId.SCORING else []),
            "jobs": jobs,
            "max_workers": runtime.scheduler.max_workers,
            "cpu_slots": runtime.scheduler.cpu_slots,
            "gpu_ids": list(runtime.scheduler.gpu_ids),
            "writes": False,
        }
    )
    return 0


def _batch_json(value: BatchResult) -> Mapping[str, object]:
    return {
        "outcome": value.outcome.value,
        "results": {
            job.label: {
                "completed": result.completed,
                "through": result.through.value,
                "completion_policy": result.completion_policy,
                "expected_models": list(result.expected_model_ids),
                "model_failures": dict(result.model_failures),
                "attempts": [_attempt_json(item) for item in result.attempts],
            }
            for job, result in sorted(value.results.items())
        },
        "failures": {
            job.label: {
                "error_type": failure.error_type,
                "message": failure.message,
            }
            for job, failure in sorted(value.failures.items())
        },
    }


def _project_json(value: ProjectResult) -> Mapping[str, object]:
    return {
        "outcome": value.analysis.payload.get("outcome"),
        "case_outcomes": {
            str(case.get("case_id")): case.get("outcome")
            for case in value.analysis.payload.get("cases", ())
            if isinstance(case, Mapping)
        },
        "analysis": _attempt_json(value.analysis),
        "assets": _attempt_json(value.assets),
        "report": _attempt_json(value.report),
        "publication": str(value.publication),
    }


def dispatch(arguments: argparse.Namespace) -> int:
    if arguments.command in {"project", "resources", "export"}:
        from .project_cli import dispatch_project

        return dispatch_project(arguments, _json, _batch_json, _project_json)
    if arguments.command == "status":
        from .inspection import read_workspace_status

        workflow = load_workflow(arguments.workflow)
        _require_executable_workflow(workflow)
        result = read_workspace_status(
            workflow, arguments.workspace, summary=getattr(arguments, "summary", False)
        )
        _json(result)
        return 1 if result["workspace_state"] == "corrupt" else 0
    if arguments.command == "config":
        return _config_command(arguments)
    if arguments.command == "prepared-case":
        return _prepared_case_command(arguments)
    if arguments.command == "workspace":
        if arguments.workspace_action == "inspect":
            from .legacy import inspect_legacy_workflow, inspect_legacy_workspace

            workspace_path = arguments.legacy_workspace or arguments.workspace
            workflow_path = arguments.legacy_workflow or arguments.workflow
            result: dict[str, object] = {
                "workspace": inspect_legacy_workspace(workspace_path),
            }
            if workflow_path.is_file():
                result["workflow"] = inspect_legacy_workflow(workflow_path)
            _json(result)
            return 0
        runner = _runner(arguments)
        records = tuple(
            record
            for stage in arguments.stage
            for record in runner.workspace.prune_attempts(
                TaskKey(StageId(stage)),
                keep=arguments.keep,
                confirm=arguments.confirm,
            )
        )
        _json(
            {
                "mode": "confirm" if arguments.confirm else "dry-run",
                "keep": arguments.keep,
                "attempts": [
                    {
                        "task": record.task.label,
                        "attempt_id": record.attempt_id,
                        "path": str(record.path),
                        "size_bytes": record.size_bytes,
                        "deleted": record.deleted,
                    }
                    for record in records
                ],
                "total_bytes": sum(record.size_bytes for record in records),
            }
        )
        return 0
    if arguments.command == "validate":
        return _validate(arguments)
    if arguments.command == "doctor":
        return _doctor(arguments)
    if arguments.command == "job" and arguments.job_action == "plan":
        return _plan_jobs(arguments, batch=False)
    if arguments.command == "batch" and arguments.batch_action == "plan":
        return _plan_jobs(arguments, batch=True)

    runner = _runner(arguments)
    if arguments.command == "job":
        action = "new" if arguments.job_action == "run" and arguments.new else arguments.job_action
        result = runner.run_job(
            JobKey(arguments.case_id, arguments.replica_id),
            through=StageId(arguments.through),
            action=action,
            model_ids=arguments.model_ids,
        )
        _json(
            {
                "job": result.job.label,
                "through": result.through.value,
                "completed": result.completed,
                "completion_policy": result.completion_policy,
                "expected_models": list(result.expected_model_ids),
                "model_failures": dict(result.model_failures),
                "attempts": [_attempt_json(item) for item in result.attempts],
            }
        )
        return 0
    if arguments.command == "batch":
        action = (
            "new" if arguments.batch_action == "run" and arguments.new else arguments.batch_action
        )
        result = runner.run_batch(
            case_ids=arguments.case_ids,
            replica_ids=arguments.replica_ids,
            model_ids=arguments.model_ids,
            through=StageId(arguments.through),
            action=action,
            **({"resume_md": True} if getattr(arguments, "resume_md", False) else {}),
        )
        _json(_batch_json(result))
        return 0 if not result.failures else 1
    if arguments.command == "finalize":
        action = _selected_action(arguments)
        if arguments.finalize_scope == "case":
            result = runner.finalize_case(arguments.case_id, action=action)
            _json(_attempt_json(result, include_payload=True))
        else:
            _json(_project_json(runner.finalize_project(action=action)))
        return 0
    if arguments.command == "report":
        _json(_project_json(runner.rebuild_report(rebuild_assets=not arguments.reuse_assets)))
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        return dispatch(arguments)
    except (MmpbsaError, OSError, ValueError) as exc:
        print(f"mmpbsa: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

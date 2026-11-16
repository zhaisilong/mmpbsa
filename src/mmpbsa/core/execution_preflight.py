"""Read-only task resource planning before the first admission/attempt write."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .config import RuntimeDefinition, WorkflowDefinition
from .errors import ContractError
from .models import JobKey, ResourceRequest, StageRequest, TaskKey
from .prepared import PreparedCase, load_prepared_case
from .prepared_projection import PREPARED_CASE_CONSUMERS, prepared_request_view
from .providers import ProviderRegistry
from .state import REPLICA_STAGES, StageId

_CAPABILITIES = {
    StageId.MD: "md",
    StageId.TRAJECTORY: "trajectory",
    StageId.TRAJECTORY_QC: "trajectory_qc",
    StageId.SCORING: "scoring",
}


def require_host_cooperation(service: object, runtime: RuntimeDefinition, capability: str) -> None:
    """Host leases require explicit cooperation with child lifecycle observation."""
    if (
        runtime.scheduler.resource_coordination.mode == "host"
        and getattr(service, "supports_host_resources", False) is not True
    ):
        raise ContractError(
            f"{capability!r} service has not declared supports_host_resources=True; "
            "host resource coordination cannot safely observe its child processes"
        )


def task_science(
    workflow: WorkflowDefinition, task: TaskKey, prepared_case: PreparedCase | None
) -> dict[str, object]:
    science = dict(
        workflow.scoring_model(str(task.model_id))
        if task.stage is StageId.SCORING
        else workflow.stage_science(task.stage)
    )
    if task.replica_id is not None:
        replica = workflow.replica(task.replica_id)
        science["_replica"] = {
            "replica_id": replica.replica_id,
            "replica_index": replica.replica_index,
            "seed": replica.seed,
        }
    if prepared_case is not None and task.stage in PREPARED_CASE_CONSUMERS:
        science["_prepared_case_schema"] = prepared_case.manifest.schema_version
    return science


def _tasks(
    workflow: WorkflowDefinition, job: JobKey, through: StageId, selected: Sequence[str]
) -> tuple[TaskKey, ...]:
    if through is StageId.ADMIT:
        return ()
    if through not in REPLICA_STAGES:
        raise ContractError("preflight through-stage must be admit or a replica stage")
    result: list[TaskKey] = []
    for stage in REPLICA_STAGES:
        models = selected if stage is StageId.SCORING else (None,)
        result.extend(TaskKey(stage, job.case_id, job.replica_id, model) for model in models)
        if stage is through:
            break
    return tuple(result)


def validate_dispatchability(
    requests: Sequence[tuple[TaskKey, ResourceRequest]], runtime: RuntimeDefinition
) -> None:
    """Simulate a feasible DAG completion; reject stalled roots before admission."""
    scheduler = runtime.scheduler

    def group(task: TaskKey) -> str:
        return (
            "md"
            if task.stage is StageId.MD
            else "scoring"
            if task.stage is StageId.SCORING
            else "postprocess"
        )

    for task, request in requests:
        if request.cpu_slots > scheduler.cpu_slots:
            raise ContractError(
                f"{task.label} requests {request.cpu_slots} CPU slots, "
                f"scheduler provides {scheduler.cpu_slots}"
            )
        if request.gpu_slots > len(scheduler.gpu_ids):
            raise ContractError(
                f"{task.label} requests {request.gpu_slots} GPUs, "
                f"scheduler provides {len(scheduler.gpu_ids)}"
            )
    if scheduler.execution_mode != "pipeline":
        return
    pending = dict(requests)
    completed: set[TaskKey] = set()
    predecessor = {
        StageId.TRAJECTORY: StageId.MD,
        StageId.TRAJECTORY_QC: StageId.TRAJECTORY,
        StageId.SCORING: StageId.TRAJECTORY_QC,
    }
    while pending:
        needed = {group(task) for task in pending}
        blocked: list[str] = []
        selected = None
        for task, request in pending.items():
            parent = predecessor.get(task.stage)
            if (
                parent is not None
                and TaskKey(parent, task.case_id, task.replica_id) not in completed
            ):
                continue
            reserved = sum(
                value
                for other, value in scheduler.cpu_reservations.items()
                if other != group(task) and other in needed
            )
            if request.cpu_slots + reserved <= scheduler.cpu_slots:
                selected = task
                break
            blocked.append(f"{task.label}: {request.cpu_slots} CPU slots plus {reserved} reserved")
        if selected is None:
            raise ContractError(
                "DAG is not dispatchable: "
                + "; ".join(blocked)
                + f"; scheduler provides {scheduler.cpu_slots}"
            )
        completed.add(selected)
        del pending[selected]


def preflight_execution(
    workflow: WorkflowDefinition,
    runtime: RuntimeDefinition,
    providers: ProviderRegistry,
    jobs: Sequence[JobKey],
    *,
    through: StageId = StageId.SCORING,
    model_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Verify sealed inputs and request contracts without opening a workspace/probing."""
    from .runner import effective_stage_runtime

    if not jobs:
        raise ContractError("preflight requires at least one job")
    if model_ids is not None and through is not StageId.SCORING:
        raise ContractError("model_ids is only valid when through is scoring")
    selected = (
        tuple(dict.fromkeys(model_ids)) if model_ids is not None else workflow.scoring_model_ids
    )
    for model in selected:
        workflow.scoring_model(model)
    if through is StageId.SCORING and not selected:
        raise ContractError("scoring preflight requires at least one model")
    admission_provider = runtime.provider_for("admission")
    if admission_provider is not None:
        require_host_cooperation(
            providers.service(admission_provider, "admission"), runtime, "admission"
        )
    prepared_cases: dict[str, PreparedCase] = {}
    requests: list[tuple[TaskKey, ResourceRequest]] = []
    rows: list[dict[str, object]] = []
    for job in jobs:
        case = workflow.case(job.case_id)
        workflow.replica(job.replica_id)
        if job.case_id not in prepared_cases:
            prepared_cases[job.case_id] = load_prepared_case(
                case.prepared_case, expected_case_id=case.case_id
            )
        prepared = prepared_cases[job.case_id]
        for task in _tasks(workflow, job, through, selected):
            science = task_science(workflow, task, prepared)
            capability = _CAPABILITIES[task.stage]
            provider_id = runtime.provider_for(capability)
            if provider_id is None:
                raise ContractError(f"runtime does not select a provider for {capability!r}")
            service = providers.service(provider_id, capability)
            require_host_cooperation(service, runtime, capability)
            for method in ("resource_request", "compatibility", "execute", "resume"):
                if not callable(getattr(service, method, None)):
                    raise ContractError(f"{capability!r} service is missing {method}()")
            prototype = StageRequest(
                task,
                science,
                effective_stage_runtime(runtime, capability, science),
                prepared_request_view(task, prepared, science),
                {},
                Path("."),
                Path("."),
            )
            request = service.resource_request(prototype)
            if not isinstance(request, ResourceRequest):
                raise ContractError(f"{task.label} returned an invalid resource request")
            requests.append((task, request))
            rows.append(
                {"task": task.label, "cpu_slots": request.cpu_slots, "gpu_slots": request.gpu_slots}
            )
    validate_dispatchability(requests, runtime)
    scheduler = runtime.scheduler
    coordination = scheduler.resource_coordination
    if coordination.mode == "host":
        from .host_resources import HostResourcePool

        validation_pool = HostResourcePool(
            registry_path=coordination.registry_path,
            project_id=workflow.project_id,
            cpu_slots=scheduler.cpu_slots,
            gpu_ids=scheduler.gpu_ids,
        )
    else:
        from .resources import ResourcePool

        validation_pool = ResourcePool(
            cpu_slots=scheduler.cpu_slots,
            gpu_ids=scheduler.gpu_ids,
            cpu_affinity=scheduler.cpu_affinity,
        )
    for _, request in requests:
        validation_pool.validate(request)
    return {
        "schema_version": "mmpbsa.execution-preflight.v1",
        "valid": True,
        "project_id": workflow.project_id,
        "job_count": len(jobs),
        "prepared_case_count": len(prepared_cases),
        "execution_mode": runtime.scheduler.execution_mode,
        "tasks": rows,
        "backend_probed": False,
        "workspace_modified": False,
        "resource_inventory_checked": True,
        "resource_coordination": coordination.mode,
        "physical_cpu_candidates": list(validation_pool.cpu_ids),
    }

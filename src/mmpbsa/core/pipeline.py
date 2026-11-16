"""Stage-ready batch dispatcher. Waiting tasks never occupy worker threads."""

from __future__ import annotations

import json
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING

from mmpbsa.utils import atomic_write_json, sha256_digest, utc_now
from mmpbsa.utils.cpu import thread_cpu_affinity

from .errors import ContractError, TaskExecutionError
from .execution_telemetry import TimingSamples, waiting_record, wall_clock_bounds
from .models import (
    AttemptSummary,
    JobKey,
    JobResult,
    ResourceAllocation,
    ResourceRequest,
    StageInput,
    StageRequest,
    TaskKey,
)
from .prepared_projection import prepared_request_view
from .state import AttemptState, QcVerdict, StageId
from .workspace import ExecutionAction

if TYPE_CHECKING:
    from .runner import BatchResult, WorkflowRunner

_GROUPS = ("md", "postprocess", "scoring")


def wait_for_resources() -> None:
    """Bounded, interruptible idle wait; resource scarcity has no timeout."""
    Event().wait(5)


def resource_group(task: TaskKey) -> str:
    if task.stage is StageId.MD:
        return "md"
    if task.stage is StageId.SCORING:
        return "scoring"
    return "postprocess"


@dataclass
class _Job:
    key: JobKey
    tasks: tuple[TaskKey, ...]
    actions: dict[TaskKey, ExecutionAction]
    inputs: dict[StageId, StageInput] = field(default_factory=dict)
    attempts: dict[TaskKey, AttemptSummary] = field(default_factory=dict)
    model_failures: dict[str, str] = field(default_factory=dict)
    model_statuses: dict[str, str] = field(default_factory=dict)
    failed: bool = False


@dataclass(frozen=True)
class _Ready:
    job: _Job
    task: TaskKey
    request: ResourceRequest
    queued_at: str
    queued_monotonic: float
    comparison_key: str


def validate_recovery_mode(through: StageId, action: ExecutionAction, mode: str) -> None:
    if (
        action != "new"
        or mode != "pipeline"
        or through not in {StageId.TRAJECTORY, StageId.TRAJECTORY_QC, StageId.SCORING}
    ):
        raise ContractError(
            "--resume-md requires batch run --new, pipeline mode, and a downstream through-stage"
        )


def recovery_md_action(runner: WorkflowRunner, task: TaskKey) -> ExecutionAction:
    """Explicit recovery preserves MD science and uses native workspace resume."""
    latest = runner.workspace.latest(task)
    if latest is None:
        return "run"
    prepared = runner._prepared(task.case_id)
    science = runner._science(task, prepared)
    current_hash = runner._science_hash(task, science, prepared, {})
    service, _, _, compatibility = runner._stage_execution_contract(task.stage, science)
    if latest.science_hash != current_hash or latest.compatibility_key != compatibility.key:
        raise ContractError("--resume-md cannot preserve changed MD science/backend compatibility")
    if latest.state.active:
        return "resume"  # reserve() still rejects a live owner or subprocess.
    if latest.state is AttemptState.COMPLETED:
        return "run"
    if latest.state is AttemptState.FAILED and callable(
        getattr(service, "recover_failed_md", None)
    ):
        return "retry"  # Fresh attempt; backend requires a valid native checkpoint.
    raise ContractError(
        "--resume-md requires completed or resumable MD; repair failed MD explicitly"
    )


def run_pipeline(
    runner: WorkflowRunner,
    jobs: Sequence[JobKey],
    *,
    through: StageId,
    action: ExecutionAction,
    model_ids: Sequence[str] | None,
    resume_md: bool = False,
) -> BatchResult:
    from .runner import BatchFailure, BatchResult

    selected = runner._selected_models(model_ids) if through is StageId.SCORING else ()
    if through is StageId.SCORING and action in {"resume", "retry"} and len(selected) != 1:
        raise ContractError(f"multi-model {action} requires exactly one explicit model")
    scheduler = runner.runtime.scheduler
    queues: dict[str, deque[_Ready]] = {group: deque() for group in _GROUPS}
    states: list[_Job] = []
    failures: dict[JobKey, BatchFailure] = {}
    active: dict[Future, tuple[_Ready, ResourceAllocation]] = {}
    terminal_counts: Counter[str] = Counter()
    completed_frames: Counter[str] = Counter()
    execution_seconds: Counter[str] = Counter()
    timed_frames: Counter[str] = Counter()
    timing_samples = TimingSamples()
    active_started: dict[Future, float] = {}
    last_snapshot = 0.0
    wait_reasons: dict[TaskKey, str] = {}

    def fail_job(state: _Job, exc: Exception) -> None:
        state.failed = True
        failures[state.key] = BatchFailure(state.key, type(exc).__name__, str(exc))

    def enqueue(state: _Job, task: TaskKey) -> None:
        prepared = runner._prepared(state.key.case_id)
        science = runner._science(task, prepared)
        service, capability = runner._provider_service(task.stage)
        request = ResourceRequest()
        if service is not None:
            assert capability is not None
            prototype = StageRequest(
                task=task,
                science=science,
                runtime=runner._effective_stage_runtime(capability, science),
                prepared_case=prepared_request_view(task, prepared, science),
                upstream=runner._job_upstream(task.stage, state.inputs),
                attempt_dir=runner.workspace.task_dir(task),
                invocation_dir=runner.workspace.task_dir(task),
            )
            request = runner._require_service(service, capability).resource_request(prototype)
        if not isinstance(request, ResourceRequest):
            raise ContractError(f"{task.label} returned an invalid resource request")
        runner._resources.validate(request)
        comparison_science = {key: value for key, value in science.items() if key != "_replica"}
        comparison_key = sha256_digest(
            {
                "stage": task.stage.value,
                "model": task.model_id,
                "case": task.case_id,
                "science": comparison_science,
                "cpu_slots": request.cpu_slots,
                "gpu_slots": request.gpu_slots,
                "runtime": runner._effective_stage_runtime(capability, science)
                if capability
                else {},
            }
        )
        queues[resource_group(task)].append(
            _Ready(state, task, request, utc_now(), monotonic(), comparison_key)
        )

    def advance(state: _Job, task: TaskKey) -> None:
        if task.stage is StageId.SCORING:
            return
        prerequisites = [t for t in state.tasks if t.stage is not StageId.SCORING]
        index = prerequisites.index(task) + 1
        if index < len(prerequisites):
            enqueue(state, prerequisites[index])
            return
        scoring = [t for t in state.tasks if t.stage is StageId.SCORING]
        if scoring and state.inputs[StageId.TRAJECTORY_QC].qc_verdict is QcVerdict.BLOCK:
            for model_task in scoring:
                state.model_failures[model_task.model_id] = (
                    "skipped: trajectory_qc verdict is BLOCK"
                )
                state.model_statuses[model_task.model_id] = "skipped_qc_block"
            return
        for model_task in scoring:
            try:
                enqueue(state, model_task)
            except Exception as exc:
                state.model_failures[model_task.model_id] = f"{type(exc).__name__}: {exc}"
                state.model_statuses[model_task.model_id] = "execution_failed"

    def needed(group: str) -> bool:
        return any(
            not state.failed
            and any(
                resource_group(task) == group
                and task not in state.attempts
                and (task.model_id is None or task.model_id not in state.model_failures)
                for task in state.tasks
            )
            for state in states
        )

    def snapshot() -> None:
        nonlocal last_snapshot
        last_snapshot = monotonic()
        heartbeat = getattr(runner._resources, "heartbeat", None)
        if heartbeat is not None:
            heartbeat()
        external_wait = bool(
            getattr(runner._resources, "is_host_pool", False)
            and getattr(runner._resources, "last_wait", None)
            and getattr(runner._resources.last_wait, "reason", None) != "project_capacity"
        )
        estimates = {}
        for name in ("md", "gb", "pb"):
            selected_pending = []
            for group_queue in queues.values():
                selected_pending.extend(
                    (item.comparison_key, 0.0)
                    for item in group_queue
                    if not item.job.failed
                    and (item.task.stage.value == name or item.task.model_id == name)
                )
            selected_pending.extend(
                (item.comparison_key, max(0.0, monotonic() - active_started[future]))
                for future, (item, _) in active.items()
                if item.task.stage.value == name or item.task.model_id == name
            )
            future_unqueued = any(
                not state.failed
                and any(
                    (task.stage.value == name or task.model_id == name)
                    and task not in state.attempts
                    and (task.model_id is None or task.model_id not in state.model_failures)
                    and not any(item.task == task for queue in queues.values() for item in queue)
                    and not any(item.task == task for item, _ in active.values())
                    for task in state.tasks
                )
                for state in states
            )
            group = "md" if name == "md" else "scoring"
            estimates[name] = (
                {"available": False, "reason": "upstream_tasks_not_yet_ready"}
                if future_unqueued
                else wall_clock_bounds(
                    timing_samples.estimate(selected_pending, external_wait=external_wait),
                    max_parallel=min(
                        scheduler.max_workers,
                        scheduler.stage_limits.get(group, scheduler.max_workers),
                    ),
                )
            )
        unready = any(
            not state.failed
            and any(
                task.stage is not StageId.ADMIT
                and task not in state.attempts
                and (task.model_id is None or task.model_id not in state.model_failures)
                and not any(item.task == task for queue in queues.values() for item in queue)
                and not any(item.task == task for item, _ in active.values())
                for task in state.tasks
            )
            for state in states
        )
        all_pending = [
            (item.comparison_key, 0.0)
            for queue in queues.values()
            for item in queue
            if not item.job.failed
        ]
        all_pending.extend(
            (item.comparison_key, max(0.0, monotonic() - active_started[future]))
            for future, (item, _) in active.items()
        )
        overall_eta = (
            {"available": False, "reason": "upstream_tasks_not_yet_ready"}
            if unready
            else wall_clock_bounds(
                timing_samples.estimate(all_pending, external_wait=external_wait),
                max_parallel=scheduler.max_workers,
            )
        )
        atomic_write_json(
            runner.workspace.root / "scheduler_status.json",
            {
                "schema_version": "mmpbsa.scheduler-status.v1",
                "updated_at": utc_now(),
                "execution_mode": "pipeline",
                "cpu_slots": scheduler.cpu_slots,
                "cpu_ids": list(runner._resources.cpu_ids),
                "queued": {
                    group: [item.task.label for item in queue if not item.job.failed]
                    for group, queue in queues.items()
                },
                "queued_details": [
                    {
                        "task": item.task.label,
                        "group": resource_group(item.task),
                        "queued_at": item.queued_at,
                        "wait_seconds": max(0.0, monotonic() - item.queued_monotonic),
                        "wait_reason": wait_reasons.get(item.task, "project_dispatch_queue"),
                        "long_wait_warning": monotonic() - item.queued_monotonic >= 15 * 60,
                    }
                    for queue in queues.values()
                    for item in queue
                    if not item.job.failed
                ],
                "active": [
                    {
                        "task": item.task.label,
                        "group": resource_group(item.task),
                        "cpu_ids": list(allocation.cpu_ids),
                        "cpu_slots": allocation.cpu_slots,
                        "gpu_ids": list(allocation.gpu_ids),
                        "physical_gpu_ids": list(getattr(allocation, "physical_gpu_ids", ())),
                        "execution_elapsed_seconds": max(0.0, monotonic() - active_started[future]),
                        "queue_wait_seconds": max(
                            0.0, active_started[future] - item.queued_monotonic
                        ),
                    }
                    for future, (item, allocation) in active.items()
                ],
                "resource_wait": waiting_record(getattr(runner._resources, "last_wait", None)),
                "eta": estimates,
                "overall_eta": overall_eta,
                "terminal_stage_counts": dict(terminal_counts),
                "completed_frames_by_model": dict(completed_frames),
                "observed_seconds_per_frame": {
                    model: execution_seconds[model] / frames
                    for model, frames in timed_frames.items()
                    if frames
                },
                "failed_jobs": [job.label for job in failures],
                "model_failures": {
                    state.key.label: dict(state.model_failures)
                    for state in states
                    if state.model_failures
                },
                "model_statuses": {state.key.label: dict(state.model_statuses) for state in states},
                "qc_blocked_jobs": [
                    state.key.label
                    for state in states
                    if StageId.TRAJECTORY_QC in state.inputs
                    and state.inputs[StageId.TRAJECTORY_QC].qc_verdict is QcVerdict.BLOCK
                ],
            },
        )

    # A single coordinator owns this workspace. It never nests a job executor
    # or starts a second CLI process with an independent resource pool.
    with thread_cpu_affinity(runner._resources.cpu_ids):
        for key in jobs:
            try:
                tasks = runner._job_tasks(key, through, model_ids=selected or None)
                actions = dict(
                    zip(tasks, runner._actions_for_tasks(tasks, action, strict=False), strict=True)
                )
                if resume_md:
                    md_task = next(task for task in tasks if task.stage is StageId.MD)
                    actions[md_task] = recovery_md_action(runner, md_task)
                state = _Job(key, tasks, actions)
                state.model_statuses = {model: "not_started" for model in selected}
                states.append(state)
                admitted = runner.workspace.latest(tasks[0])
                assert admitted is not None
                state.attempts[tasks[0]] = admitted
                state.inputs[StageId.ADMIT] = runner.workspace.as_stage_input(admitted)
                advance(state, tasks[0])
            except Exception as exc:
                if states and states[-1].key == key:
                    fail_job(states[-1], exc)
                else:
                    failures[key] = BatchFailure(key, type(exc).__name__, str(exc))

        with ThreadPoolExecutor(
            max_workers=scheduler.max_workers, thread_name_prefix="mmpbsa-stage"
        ) as executor:
            try:
                while any(queues.values()) or active:
                    active_counts = Counter(
                        resource_group(item.task) for item, _ in active.values()
                    )
                    active_cpu = Counter()
                    for item, allocation in active.values():
                        active_cpu[resource_group(item.task)] += allocation.cpu_slots
                    for group in _GROUPS:
                        queue = queues[group]
                        remaining = len(queue)
                        for _ in range(remaining):
                            item = queue.popleft()
                            if item.job.failed:
                                cancel_request = getattr(runner._resources, "cancel_request", None)
                                if cancel_request is not None:
                                    cancel_request(item.task.label)
                                continue
                            if len(active) >= scheduler.max_workers or active_counts[
                                group
                            ] >= scheduler.stage_limits.get(group, scheduler.max_workers):
                                queue.appendleft(item)
                                wait_reasons[item.task] = "project_concurrency_limit"
                                cancel_request = getattr(runner._resources, "cancel_request", None)
                                if cancel_request is not None:
                                    cancel_request(item.task.label)
                                break
                            reserved = sum(
                                max(0, slots - active_cpu[other])
                                for other, slots in scheduler.cpu_reservations.items()
                                if other != group and needed(other)
                            )
                            arguments = (
                                {"request_key": item.task.label, "priority": _GROUPS.index(group)}
                                if getattr(runner._resources, "is_host_pool", False)
                                else {}
                            )
                            allocation = runner._resources.try_acquire(
                                item.request, reserved=reserved, **arguments
                            )
                            if allocation is None:
                                queue.appendleft(item)
                                wait_record = getattr(runner._resources, "last_wait", None)
                                wait_reasons[item.task] = getattr(
                                    wait_record, "reason", "project_cpu_gpu_capacity"
                                )
                                break
                            wait_reasons.pop(item.task, None)
                            try:
                                future = executor.submit(
                                    runner._execute_task,
                                    item.task,
                                    action=item.job.actions[item.task],
                                    prepared_case=runner._prepared(item.job.key.case_id),
                                    upstream=runner._job_upstream(item.task.stage, item.job.inputs),
                                    new_on_conflict=action == "new"
                                    and item.job.actions[item.task] == "run",
                                    allocation=allocation,
                                    queued_at=item.queued_at,
                                    queued_monotonic=item.queued_monotonic,
                                    resume_failed_md=resume_md
                                    and item.task.stage is StageId.MD
                                    and item.job.actions[item.task] == "retry",
                                )
                            except BaseException:
                                runner._resources.release(allocation)
                                raise
                            active[future] = (item, allocation)
                            active_started[future] = monotonic()
                            active_counts[group] += 1
                            active_cpu[group] += allocation.cpu_slots
                    if monotonic() - last_snapshot >= 30:
                        snapshot()
                    if not active:
                        if any(queues.values()):
                            if getattr(runner._resources, "is_host_pool", False):
                                # External leases are normal queueing, not a stalled local DAG.
                                wait_for_resources()
                                continue
                            raise ContractError(
                                "pipeline cannot dispatch ready tasks; "
                                "check CPU reservations and requests"
                            )
                        break
                    done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                    for future in done:
                        item, allocation = active.pop(future)
                        active_started.pop(future)
                        runner._resources.release(allocation)
                        state, task = item.job, item.task
                        try:
                            summary = future.result()
                            state.attempts[task] = summary
                            terminal_counts[task.stage.value] += 1
                            if task.stage is StageId.SCORING:
                                state.model_statuses[task.model_id] = "completed"
                                frames = int(summary.payload.get("frame_count", 0))
                                completed_frames[task.model_id] += frames
                                invocations = (
                                    runner.workspace.task_dir(task)
                                    / "attempts"
                                    / summary.attempt_id
                                    / "invocations"
                                )
                                paths = sorted(invocations.glob("*/invocation.json"))
                                if paths:
                                    record = json.loads(paths[-1].read_text())
                                    elapsed = (
                                        (record.get("runtime") or {})
                                        .get("timing", {})
                                        .get("execution_seconds")
                                    )
                                    if isinstance(elapsed, (int, float)) and elapsed > 0 and frames:
                                        execution_seconds[task.model_id] += elapsed
                                        timed_frames[task.model_id] += frames
                            invocations = (
                                runner.workspace.task_dir(task)
                                / "attempts"
                                / summary.attempt_id
                                / "invocations"
                            )
                            paths = sorted(invocations.glob("*/invocation.json"))
                            if paths:
                                invocation = json.loads(paths[-1].read_text())
                                elapsed = (
                                    (invocation.get("runtime") or {})
                                    .get("timing", {})
                                    .get("execution_seconds")
                                )
                                if isinstance(elapsed, (int, float)):
                                    timing_samples.observe(item.comparison_key, elapsed)
                            if task.stage is not StageId.SCORING:
                                state.inputs[task.stage] = runner.workspace.as_stage_input(summary)
                            advance(state, task)
                        except Exception as exc:
                            terminal_counts[task.stage.value + "_failed"] += 1
                            if task.stage is StageId.SCORING:
                                state.model_failures[task.model_id] = f"{type(exc).__name__}: {exc}"
                                state.model_statuses[task.model_id] = "execution_failed"
                                summary = runner.workspace.latest(task)
                                if summary is not None:
                                    state.attempts[task] = summary
                            else:
                                fail_job(state, exc)
            finally:
                # Drain already-running stages; never free a lease while its
                # child is still calculating, including on coordinator errors.
                for future, (_, allocation) in list(active.items()):
                    try:
                        future.result()
                    except BaseException:
                        pass
                    finally:
                        runner._resources.release(allocation)
                        del active[future]
                snapshot()

        results = {}
        for state in states:
            if state.failed:
                continue
            completed_models = {
                task.model_id
                for task, summary in state.attempts.items()
                if task.stage is StageId.SCORING and summary.state is AttemptState.COMPLETED
            }
            acceptable = (
                bool(completed_models)
                if runner.workflow.completion_policy == "any"
                else set(selected).issubset(completed_models)
            )
            if (
                through is StageId.SCORING
                and not acceptable
                and state.inputs[StageId.TRAJECTORY_QC].qc_verdict is not QcVerdict.BLOCK
            ):
                fail_job(
                    state,
                    TaskExecutionError(
                        "no acceptable scoring result: " + str(state.model_failures)
                    ),
                )
                continue
            results[state.key] = JobResult(
                job=state.key,
                through=through,
                attempts=tuple(state.attempts[t] for t in state.tasks if t in state.attempts),
                completion_policy=runner.workflow.completion_policy
                if through is StageId.SCORING
                else "all",
                expected_model_ids=selected,
                model_failures=state.model_failures,
                model_statuses=state.model_statuses,
            )
        snapshot()
        return BatchResult(results, failures)

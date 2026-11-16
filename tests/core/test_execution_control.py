"""Regression coverage for non-mutating monitoring and early execution gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_runner import ControlledInterrupt, FakeControls, _pipeline_project, _project, _runner

from mmpbsa.core.config import ResourceCoordinationConfig
from mmpbsa.core.errors import ConfigurationError, ContractError
from mmpbsa.core.execution_preflight import preflight_execution
from mmpbsa.core.execution_telemetry import TimingSamples
from mmpbsa.core.inspection import WorkspaceReader, read_workspace_status
from mmpbsa.core.md_allocation_guard import previous_allocation
from mmpbsa.core.models import JobKey, ResourceAllocation, ResourceRequest, StageRequest, TaskKey
from mmpbsa.core.state import StageId
from mmpbsa.utils import FileLock


def _inventory(root):
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in root.rglob("*")
        if p.is_file()
    }


def test_missing_workspace_status_never_initializes_any_files(tmp_path):
    project = _project(tmp_path / "project")
    root = tmp_path / "missing"
    status = read_workspace_status(project.workflow, root, summary=True)
    assert status["workspace_state"] == "missing"
    assert status["summary"]["md"] == {"total": 1, "not_started": 1}
    assert not root.exists()
    assert "tasks" not in status


def test_complete_and_blocked_status_are_read_only_and_typed(tmp_path):
    controls = FakeControls(blocked_qc={("case_a", "rep01")})
    project = _pipeline_project(_project(tmp_path / "project", controls=controls))
    runner = _runner(project, tmp_path / "workspace")
    result = runner.run_batch()
    assert result.results[JobKey("case_a", "rep01")].model_statuses == {
        "gb": "skipped_qc_block",
        "pb": "skipped_qc_block",
    }
    before = _inventory(runner.workspace.root)
    status = read_workspace_status(project.workflow, runner.workspace.root)
    assert status["summary"]["trajectory_qc"]["BLOCK"] == 1
    assert status["summary"]["gb"]["skipped_qc_block"] == 1
    assert status["summary"]["pb"].get("execution_failed", 0) == 0
    assert _inventory(runner.workspace.root) == before


def test_corrupt_latest_attempt_is_not_hidden_by_prior_success(tmp_path):
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    runner.run_job(JobKey("case_a", "rep01"), through=StageId.MD)
    task = TaskKey(StageId.MD, "case_a", "rep01")
    incomplete = runner.workspace.task_dir(task) / "attempts" / "0002"
    incomplete.mkdir()
    before = _inventory(runner.workspace.root)
    status = read_workspace_status(project.workflow, runner.workspace.root)
    assert status["workspace_state"] == "corrupt"
    md = next(r for r in status["tasks"] if r["task"] == task.label)
    assert md["state"] == "corrupt"
    assert status["errors"]
    assert _inventory(runner.workspace.root) == before


def test_preflight_blocks_reserved_dag_before_any_admission(tmp_path):
    controls = FakeControls()
    controls.resources = {**controls.resources, "md": ResourceRequest(cpu_slots=4, gpu_slots=1)}
    project = _pipeline_project(_project(tmp_path / "project", controls=controls))
    project = replace(
        project,
        runtime=replace(
            project.runtime,
            scheduler=replace(
                project.runtime.scheduler, cpu_slots=4, cpu_reservations={"postprocess": 1}
            ),
        ),
    )
    runner = _runner(project, tmp_path / "workspace")
    before = _inventory(runner.workspace.root)
    with pytest.raises(ContractError, match="not dispatchable"):
        runner.run_batch()
    assert not controls.events
    assert not (runner.workspace.root / "cases").exists()
    assert _inventory(runner.workspace.root) == before


def test_preflight_does_not_open_workspace_or_probe_backend(tmp_path):
    project = _pipeline_project(_project(tmp_path / "project"))
    before = _inventory(tmp_path)
    report = preflight_execution(
        project.workflow, project.runtime, project.registry, [JobKey("case_a", "rep01")]
    )
    assert report["valid"] and not report["backend_probed"]
    assert report["tasks"][0]["cpu_slots"] == 2
    assert _inventory(tmp_path) == before
    assert not project.controls.events


def test_preflight_allows_large_postprocess_after_remaining_md_completes(tmp_path):
    controls = FakeControls()
    controls.resources = {**controls.resources, "trajectory": ResourceRequest(cpu_slots=7)}
    project = _pipeline_project(
        _project(tmp_path / "project", controls=controls, replica_ids=("rep01", "rep02"))
    )
    report = preflight_execution(
        project.workflow,
        project.runtime,
        project.registry,
        [JobKey("case_a", "rep01"), JobKey("case_a", "rep02")],
    )
    assert report["valid"]


def test_host_preflight_requires_initialized_registry_without_creating_it(tmp_path):
    project = _pipeline_project(_project(tmp_path / "project"))
    registry_path = tmp_path / "missing_registry.json"
    runtime = replace(
        project.runtime,
        scheduler=replace(
            project.runtime.scheduler,
            cpu_affinity="physical",
            gpu_ids=("GPU-00000000-0000-0000-0000-000000000001",),
            resource_coordination=ResourceCoordinationConfig(
                mode="host", registry_path=registry_path
            ),
        ),
    )
    for capability in project.registry.describe()[0].capabilities:
        project.registry.service("fake", capability).supports_host_resources = True
    before = _inventory(tmp_path)
    with pytest.raises(ConfigurationError):
        preflight_execution(
            project.workflow, runtime, project.registry, [JobKey("case_a", "rep01")]
        )
    assert _inventory(tmp_path) == before
    assert not registry_path.exists()


def test_host_preflight_rejects_unknown_child_lifecycle_contract_but_local_still_works(tmp_path):
    project = _pipeline_project(_project(tmp_path / "project"))
    runtime = replace(
        project.runtime,
        scheduler=replace(
            project.runtime.scheduler,
            cpu_affinity="physical",
            gpu_ids=("GPU-00000000-0000-0000-0000-000000000001",),
            resource_coordination=ResourceCoordinationConfig(
                mode="host", registry_path=tmp_path / "never_created.json"
            ),
        ),
    )
    before = _inventory(tmp_path)
    with pytest.raises(ContractError, match="supports_host_resources=True"):
        preflight_execution(
            project.workflow, runtime, project.registry, [JobKey("case_a", "rep01")]
        )
    assert preflight_execution(
        project.workflow, project.runtime, project.registry, [JobKey("case_a", "rep01")]
    )["valid"]
    assert _inventory(tmp_path) == before


def test_read_only_status_retains_legacy_latest_shape_and_artifact_verification(tmp_path):
    from mmpbsa.core.cli import _attempt_json

    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    runner.run_job(JobKey("case_a", "rep01"), through=StageId.MD)
    task = TaskKey(StageId.MD, "case_a", "rep01")
    expected = _attempt_json(runner.workspace.latest(task))
    before = _inventory(runner.workspace.root)
    reader = WorkspaceReader(runner.workspace.root)
    summary = reader.latest(task)
    artifacts = reader.verify_artifacts(summary)
    assert artifacts and all(path.is_file() for path in artifacts.values())
    status = read_workspace_status(project.workflow, runner.workspace.root)
    record = next(value for value in status["tasks"] if value["task"] == task.label)
    assert record["latest"] == expected
    assert record["result_status"] == "completed"
    missing = next(value for value in status["tasks"] if value["state"] == "not_started")
    assert missing["latest"] is None
    assert _inventory(runner.workspace.root) == before


def test_physical_preflight_checks_actual_allowed_core_inventory(tmp_path, monkeypatch):
    from mmpbsa.core import resources
    from mmpbsa.utils.cpu import PhysicalCpu

    project = _pipeline_project(_project(tmp_path / "project"))
    runtime = replace(
        project.runtime, scheduler=replace(project.runtime.scheduler, cpu_affinity="physical")
    )
    monkeypatch.setattr(resources, "physical_cpu_inventory", lambda: (PhysicalCpu(0, 0, 0, 0),))
    with pytest.raises(ConfigurationError, match="only 1 allowed"):
        preflight_execution(
            project.workflow, runtime, project.registry, [JobKey("case_a", "rep01")]
        )


def test_batch_ownership_precedes_admission_write(tmp_path):
    project = _pipeline_project(_project(tmp_path / "project"))
    runner = _runner(project, tmp_path / "workspace")
    with FileLock(runner.workspace.root / ".batch.lock", timeout_seconds=0):
        with pytest.raises(Exception, match="lock"):
            runner.run_batch(action="new")
        assert not (runner.workspace.root / "cases").exists()
        assert not project.controls.events


def test_runner_does_not_discover_resources_until_execution(tmp_path):
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    assert runner._resource_pool is None
    read_workspace_status(project.workflow, runner.workspace.root)
    assert runner._resource_pool is None


def test_host_gpu_visibility_stays_child_only_not_service_runtime(tmp_path):
    project = _project(tmp_path / "project")
    runner = _runner(project, tmp_path / "workspace")
    md = project.registry.service("fake", "md")
    execute = md.execute
    physical_gpu = "GPU-00000000-0000-0000-0000-000000000001"

    def check_runtime(request):
        assert "CUDA_VISIBLE_DEVICES" not in request.runtime.get("environment", {})
        assert request.runtime["gpu_id"] == "0"
        assert request.allocation.physical_gpu_ids == (physical_gpu,)
        return execute(request)

    md.execute = check_runtime
    summary = runner._execute_task(
        TaskKey(StageId.MD, "case_a", "rep01"),
        action="run",
        prepared_case=runner._prepared("case_a"),
        upstream={},
        allocation=ResourceAllocation(
            cpu_slots=2, gpu_ids=("0",), physical_gpu_ids=(physical_gpu,), lease_token="test-lease"
        ),
    )
    assert summary.state.value == "completed"


def test_eta_never_uses_incomparable_or_single_samples():
    samples = TimingSamples()
    samples.observe("gb_case_a_cpu4", 10)
    assert not samples.estimate([("gb_case_a_cpu4", 0)])["available"]
    samples.observe("gb_case_a_cpu4", 14)
    estimate = samples.estimate([("gb_case_a_cpu4", 3)])
    assert estimate["seconds_interval"] == [7, 11]
    assert not samples.estimate([("pb_case_a_cpu4", 0)])["available"]
    assert not samples.estimate([("gb_case_a_cpu4", 0)], external_wait=True)["available"]


def test_adapter_resume_guard_receives_previous_allocation_without_mutating_receipt(tmp_path):
    controls = FakeControls(interrupt_once={("md", "case_a", "rep01", None)})
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")
    task = TaskKey(StageId.MD, "case_a", "rep01")
    physical_gpu = "GPU-00000000-0000-0000-0000-000000000001"
    allocation = ResourceAllocation(
        cpu_slots=2, gpu_ids=("0",), physical_gpu_ids=(physical_gpu,), lease_token="test-lease"
    )
    with pytest.raises(ControlledInterrupt):
        runner._execute_task(
            task,
            action="run",
            prepared_case=runner._prepared("case_a"),
            upstream={},
            allocation=allocation,
        )
    first_receipt = (
        runner.workspace.task_dir(task) / "attempts/0001/invocations/0001/invocation.json"
    )
    original = first_receipt.read_bytes()
    calls = []
    md = project.registry.service("fake", "md")

    def guard(request, previous):
        assert request.resume
        assert request.invocation_dir.name == "0002"
        assert request.allocation.physical_gpu_ids == (physical_gpu,)
        assert previous["physical_gpu_ids"] == [physical_gpu]
        calls.append(previous)

    md.validate_resume_allocation = guard
    resume = md.resume

    def checked_resume(request):
        assert len(calls) == 1
        return resume(request)

    md.resume = checked_resume
    summary = runner._execute_task(
        task,
        action="resume",
        prepared_case=runner._prepared("case_a"),
        upstream={},
        allocation=allocation,
    )
    assert summary.state.value == "completed"
    assert first_receipt.read_bytes() == original


@pytest.mark.parametrize("latest_allocation", ({"gpu_ids": ["0"]}, None))
def test_resume_evidence_does_not_guess_uuid_from_current_or_older_allocations(
    tmp_path, latest_allocation
):
    attempt = tmp_path / "attempt"
    invocations = attempt / "invocations"
    for number, allocation in (
        ("0001", {"physical_gpu_ids": ["GPU-old"]}),
        ("0002", latest_allocation),
        ("0003", {"physical_gpu_ids": ["GPU-current"]}),
    ):
        directory = invocations / number
        directory.mkdir(parents=True)
        (directory / "invocation.json").write_text(
            json.dumps({"runtime": {"allocation": allocation}})
        )
    request = StageRequest(
        TaskKey(StageId.MD, "case_a", "rep01"),
        {},
        {},
        None,
        {},
        attempt,
        invocations / "0003",
        resume=True,
    )
    before = _inventory(tmp_path)
    assert previous_allocation(request) == latest_allocation
    assert _inventory(tmp_path) == before


def test_resume_allocation_guard_refusal_never_calls_backend_resume(tmp_path):
    controls = FakeControls(interrupt_once={("md", "case_a", "rep01", None)})
    project = _project(tmp_path / "project", controls=controls)
    runner = _runner(project, tmp_path / "workspace")
    task = TaskKey(StageId.MD, "case_a", "rep01")
    with pytest.raises(ControlledInterrupt):
        runner.run_job(JobKey("case_a", "rep01"), through=StageId.MD)
    md = project.registry.service("fake", "md")

    def guard(request, previous):
        raise ContractError("adapter refuses changed checkpoint allocation")

    md.validate_resume_allocation = guard
    with pytest.raises(ContractError, match="adapter refuses"):
        runner._execute_task(
            task, action="resume", prepared_case=runner._prepared("case_a"), upstream={}
        )
    assert not any(event.method == "resume" for event in controls.events)


def test_host_contention_wait_does_not_become_deadlock(tmp_path, monkeypatch):
    from mmpbsa.core import pipeline

    project = _pipeline_project(_project(tmp_path / "project"))
    runner = _runner(project, tmp_path / "workspace")
    local = runner._resources
    counter = {"tries": 0, "waits": 0}

    def acquire(request, *, reserved=0, request_key=None, priority=0):
        counter["tries"] += 1
        if counter["tries"] == 1:
            return None
        return local.try_acquire(request, reserved=reserved)

    runner._resource_pool = SimpleNamespace(
        is_host_pool=True,
        last_wait=SimpleNamespace(reason="host_capacity", since="now", detail="busy"),
        cpu_ids=local.cpu_ids,
        validate=local.validate,
        try_acquire=acquire,
        release=local.release,
        lease=local.lease,
        observe=local.observe,
        heartbeat=local.heartbeat,
        cancel_pending=local.cancel_pending,
    )
    monkeypatch.setattr(
        pipeline, "wait_for_resources", lambda: counter.__setitem__("waits", counter["waits"] + 1)
    )
    result = runner.run_batch()
    assert not result.failures
    assert counter["waits"] == 1

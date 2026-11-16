"""Mock-only host device allocation and OpenMM resume portability regressions."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import pytest
from test_openmm_backend import _FakeBackend, _prepared_view

from mmpbsa.backends.errors import BackendConfigurationError
from mmpbsa.backends.openmm import (
    OPENMM_PROBE_SCHEMA,
    OpenMMMdBackend,
    OpenMMWorkerConfig,
)
from mmpbsa.backends.openmm_provider import (
    MD_ENGINE_PROFILE,
    MD_PHASE_PLAN_SCHEMA,
    OpenMMStageService,
)
from mmpbsa.core.models import ResourceAllocation, StageRequest, TaskKey
from mmpbsa.core.protocols import resolve_protocol
from mmpbsa.core.state import StageId
from mmpbsa.utils.process_identity import observe_processes

GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _stage_request(tmp_path, *, allocation=None, platform="CUDA"):
    science = dict(resolve_protocol(
        "smoke_20ps@2", md_engine_profile=MD_ENGINE_PROFILE,
    ).stages["md"])
    science["_replica"] = {"replica_id": "rep01", "replica_index": 1, "seed": 4101}
    runtime = {
        "platform": {"name": "CUDA", "precision": "mixed"} if platform == "CUDA" else platform,
        "threads_per_job": 1,
        "probe_cwd": str(tmp_path),
        "_stage_contract": MD_PHASE_PLAN_SCHEMA,
        "_md_engine_profile": MD_ENGINE_PROFILE,
        "environment": {"PYTHONNOUSERSITE": "1"},
    }
    if platform == "CUDA":
        runtime["gpu_id"] = "0"
    attempt = tmp_path / "attempt"
    return StageRequest(
        task=TaskKey(StageId.MD, case_id="case-a", replica_id="rep01"),
        science=science, runtime=runtime, prepared_case=_prepared_view(tmp_path), upstream={},
        attempt_dir=attempt, invocation_dir=attempt / "invocations" / "0002",
        resume=True, allocation=allocation,
    )


def _host_allocation(gpu=GPU_A):
    return ResourceAllocation(1, (0,), ("0",), (gpu,), "mock-lease")


@pytest.mark.parametrize("previous", [
    {"physical_gpu_ids": [GPU_A]}, {"physical_gpu_ids": (GPU_A,)},
])
def test_same_physical_host_uuid_allows_checkpoint_resume(tmp_path, previous):
    request = _stage_request(tmp_path, allocation=_host_allocation())
    assert OpenMMStageService().validate_resume_allocation(request, previous) is None


@pytest.mark.parametrize("previous", [
    None, {}, {"gpu_ids": ["0"]}, {"physical_gpu_ids": []},
    {"physical_gpu_ids": [GPU_B]}, {"physical_gpu_ids": GPU_A},
])
def test_other_host_uuid_or_legacy_ordinal_cannot_authorize_checkpoint_resume(tmp_path, previous):
    request = _stage_request(tmp_path, allocation=_host_allocation())
    with pytest.raises(BackendConfigurationError, match="original physical GPU UUID"):
        OpenMMStageService().validate_resume_allocation(request, previous)


@pytest.mark.parametrize("allocation,platform", [
    (None, "CPU"), (ResourceAllocation(1), "CPU"),
    (ResourceAllocation(1, (0,), ("0",)), "CUDA"),
])
def test_frozen_local_and_cpu_resume_semantics_are_unchanged(tmp_path, allocation, platform):
    request = _stage_request(tmp_path, allocation=allocation, platform=platform)
    service = OpenMMStageService()
    assert service.validate_resume_allocation(request, None) is None
    assert service.validate_resume_allocation(request, {"physical_gpu_ids": [GPU_B]}) is None


class _Observer:
    """No registry/native processes: record the command's execution-only overlay."""

    def __init__(self, gpu):
        self.environment = {"CUDA_VISIBLE_DEVICES": gpu}
        self.events = []

    def launching(self, invocation_id):
        self.events.append(("launching", invocation_id))

    def started(self, invocation_id, identity):
        self.events.append(("started", invocation_id))

    def launch_failed(self, invocation_id):
        self.events.append(("failed", invocation_id))

    def finished(self, invocation_id):
        self.events.append(("finished", invocation_id))


class _CudaFakeBackend(_FakeBackend):
    """Run only fake phase receipts; exercise the real worker probe command path."""

    command_runner = None

    def probe(self, *, cwd):
        return OpenMMMdBackend(worker=self.worker, runner=self.command_runner).probe(cwd=cwd)


@pytest.mark.parametrize("gpu", [GPU_A, GPU_B])
def test_host_uuid_is_command_overlay_not_reserved_openmm_environment_identity(
    tmp_path, monkeypatch, gpu,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    configurations = []
    command_environments = []
    fake = _CudaFakeBackend()
    raw_probe = {
        "schema_version": OPENMM_PROBE_SCHEMA, "openmm_version": "8.5.2",
        "platforms": [{"name": "CUDA", "properties": ["Precision", "DeviceIndex"]}],
    }

    class FakePopen:
        pid = 99999999
        returncode = 0

        def __init__(self, argv, **kwargs):
            command_environments.append(dict(kwargs["env"]))
            assert kwargs["start_new_session"] is True

        def communicate(self, stdin, timeout=None):
            return json.dumps(raw_probe), ""

    def factory(config, runner=None):
        configurations.append(dict(config))
        fake.worker = OpenMMWorkerConfig(
            python_executable=sys.executable, environment=dict(config.get("environment", {})),
        )
        fake.command_runner = runner
        return fake

    monkeypatch.setattr("mmpbsa.backends.command.subprocess.Popen", FakePopen)
    monkeypatch.setattr("mmpbsa.backends.command.process_identity", lambda pid: {
        "pid": pid, "start_ticks": 1, "pgid": pid, "session_id": pid, "boot_id": "mock-boot",
    })
    monkeypatch.setattr("mmpbsa.backends.command.which", lambda command: "/mock/taskset")
    monkeypatch.setattr("mmpbsa.backends.openmm_provider.create_openmm_md_backend", factory)
    request = _stage_request(tmp_path, allocation=_host_allocation(gpu))
    service = OpenMMStageService()
    before_lease = {key: value for key, value in request.runtime.items() if key != "gpu_id"}
    reserved = service.compatibility(before_lease)
    assert service.compatibility(request.runtime) == reserved
    assert all("CUDA_VISIBLE_DEVICES" not in config.get("environment", {})
               for config in configurations)
    request = replace(request, resume=False, compatibility=reserved)
    observer = _Observer(gpu)
    with observe_processes(observer):
        result = service.execute(request)

    # Successful execute includes the adapter's actual reserved/observed
    # compatibility comparison, not merely an assertion about env text.
    assert [item.phase.kind for item in fake.requests] == ["em", "nvt", "npt", "production"]
    assert result.artifacts["checkpoint"].is_file()
    assert all(item.platform.device_index == "0" for item in fake.requests)
    assert "CUDA_VISIBLE_DEVICES" not in command_environments[0]
    assert command_environments[-1]["CUDA_VISIBLE_DEVICES"] == gpu
    assert all("CUDA_VISIBLE_DEVICES" not in config.get("environment", {})
               for config in configurations)
    assert "CUDA_VISIBLE_DEVICES" not in request.runtime.get("environment", {})
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is None
    assert [event[0] for event in observer.events] == ["launching", "started", "finished"]

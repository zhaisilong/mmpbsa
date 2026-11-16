from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmpbsa.backends.api import BackendReceipt, CommandRecord, ReceiptState
from mmpbsa.backends.errors import BackendConfigurationError, BackendOutputError
from mmpbsa.backends.openmm import (
    OPENMM_AMBER_INPUT_FORMAT,
    OPENMM_PROBE_SCHEMA,
    OpenMMIdentity,
    OpenMMMdBackend,
    OpenMMMdRequest,
    OpenMMMergeRequest,
    OpenMMPhaseSpec,
    OpenMMPlatformSpec,
    OpenMMWorkerConfig,
    openmm_artifact_paths,
    openmm_merge_artifact_paths,
)
from mmpbsa.backends.openmm_provider import (
    ADAPTER_CONTRACT,
    MD_ENGINE_PROFILE,
    MD_PHASE_PLAN_SCHEMA,
    OpenMMStageService,
    _phase_plans,
    _reset_output,
    _sealed_md_receipt,
    _sealed_merge_receipt,
    _validate_runtime_contract,
    create_provider,
)
from mmpbsa.backends.openmm_worker import (
    OpenMMWorkerError,
    _actual_platform,
    _constraint_policy,
    _platform,
)
from mmpbsa.core import (
    ArtifactRef,
    PreparedMdSystem,
    PreparedMdView,
    StageRequest,
    TaskKey,
    resolve_protocol,
)
from mmpbsa.core.state import StageId


def _input(path: Path, text: str = "input\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_openmm_constraint_policy_matches_unconstrained_em_contract() -> None:
    h_bonds = object()
    app = SimpleNamespace(HBonds=h_bonds)

    assert _constraint_policy(app, "em") == {
        "constraints": None,
        "rigidWater": False,
    }
    for phase_kind in ("nvt", "npt", "production"):
        assert _constraint_policy(app, phase_kind) == {
            "constraints": h_bonds,
            "rigidWater": True,
        }


def _phase(kind: str = "nvt", *, steps: int = 10) -> OpenMMPhaseSpec:
    return OpenMMPhaseSpec(
        phase_id=kind,
        kind=kind,
        steps=steps,
        timestep_fs=2.0,
        temperature_kelvin=300.0,
        pressure_bar=(None if kind == "nvt" else 1.0),
        initialize_velocities=(kind == "nvt"),
        report_interval_steps=5,
        trajectory_interval_steps=5,
        checkpoint_interval_steps=5,
    )


def _request(tmp_path: Path, *, platform: OpenMMPlatformSpec | None = None):
    return OpenMMMdRequest(
        topology=_input(tmp_path / "input" / "topol.top"),
        coordinates=_input(tmp_path / "input" / "system.gro"),
        output_dir=tmp_path / "output",
        phase=_phase(),
        platform=platform
        or OpenMMPlatformSpec(
            "CUDA",
            device_index="0",
            precision="mixed",
            deterministic_forces=True,
        ),
        seed=4101,
    )


def _em_request(tmp_path: Path) -> OpenMMMdRequest:
    return OpenMMMdRequest(
        topology=_input(tmp_path / "input" / "topol.top"),
        coordinates=_input(tmp_path / "input" / "system.gro"),
        output_dir=tmp_path / "em-output",
        phase=OpenMMPhaseSpec(
            phase_id="em",
            kind="em",
            minimization_tolerance_kj_mol_nm=10.0,
            maximum_force_threshold_kj_mol_nm=1_000.0,
            max_iterations=200,
        ),
        platform=OpenMMPlatformSpec("CPU"),
        seed=4101,
    )


def _amber_em_request(tmp_path: Path) -> OpenMMMdRequest:
    topology = _input(tmp_path / "input" / "solvated.prmtop")
    coordinates = _input(tmp_path / "input" / "solvated.inpcrd")
    return replace(
        _em_request(tmp_path),
        topology=topology,
        coordinates=coordinates,
        input_format=OPENMM_AMBER_INPUT_FORMAT,
        topology_sha256=hashlib.sha256(topology.read_bytes()).hexdigest(),
        coordinates_sha256=hashlib.sha256(coordinates.read_bytes()).hexdigest(),
        expected_openmm_version="8.5.2",
        expected_probe_sha256="a" * 64,
    )


def test_amber_request_rejects_changed_snapshot(tmp_path: Path) -> None:
    request = _amber_em_request(tmp_path)
    request.validate()
    request.topology.write_text("changed\n", encoding="utf-8")
    with pytest.raises(BackendConfigurationError, match="sealed digest"):
        request.validate()


def test_amber_seal_rejects_probe_and_input_tampering(tmp_path: Path) -> None:
    request = _amber_em_request(tmp_path)
    fake = _FakeBackend()
    fake.run(request)
    receipt_path = openmm_artifact_paths(request)["worker_receipt"]
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert _sealed_md_receipt(request) is not None
    for field, value in (
        ("probe_sha256", "b" * 64),
        (
            "input",
            {
                **original["input"],
                "topology_sha256": "b" * 64,
            },
        ),
    ):
        receipt = dict(original)
        receipt[field] = value
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        assert _sealed_md_receipt(request) is None


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _derived_seed(master_seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


def _platform_receipt(platform: OpenMMPlatformSpec) -> dict[str, object]:
    if platform.name == "CUDA":
        properties = {
            "DeviceIndex": str(platform.device_index),
            "Precision": str(platform.precision),
            "DeterministicForces": "true",
        }
    elif platform.name == "CPU":
        properties = {
            "Threads": str(platform.threads),
            "DeterministicForces": ("true" if platform.deterministic_forces else "false"),
        }
    else:
        properties = {}
    return {
        "name": platform.name,
        "properties": properties,
        "diagnostics": {},
    }


def _md_receipt_provenance(request: OpenMMMdRequest) -> dict[str, object]:
    platform = _platform_receipt(request.platform)
    system_sha256 = "a" * 64
    result = {
        "openmm": {"version": "8.5.2"},
        "platform": platform,
        "system_sha256": system_sha256,
        "checkpoint_provenance": {
            "openmm_version": "8.5.2",
            "system_sha256": system_sha256,
            "platform": {
                "name": platform["name"],
                "properties": dict(platform["properties"]),
            },
        },
        "system": {
            "particles": 1,
            "constraints": 0,
            "forces": ["NonbondedForce"],
        },
        "seeds": {
            "master": request.seed,
            "integrator": _derived_seed(request.seed, "integrator"),
            "barostat": _derived_seed(request.seed, "barostat"),
            "velocities": _derived_seed(request.seed, "velocities"),
        },
    }
    if request.input_format == OPENMM_AMBER_INPUT_FORMAT:
        result["probe_sha256"] = request.expected_probe_sha256
        result["input"] = {
            "format": request.input_format,
            "topology_sha256": request.topology_sha256,
            "coordinates_sha256": request.coordinates_sha256,
        }
    return result


class _RecordingRunner:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.commands = []

    def run(self, command):
        self.commands.append(command)
        return CommandRecord(
            argv=command.argv,
            cwd=command.cwd,
            returncode=0,
            stdout=json.dumps(self.payload),
            stderr="",
        )


class _WrongRequestDigestRunner:
    def __init__(
        self,
        *,
        correct_digest: bool = False,
        maximum_force: float | None = None,
    ) -> None:
        self.correct_digest = correct_digest
        self.maximum_force = maximum_force

    def run(self, command):
        request_path = Path(command.argv[command.argv.index("--request") + 1])
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        if command.argv[3] == "run":
            request = OpenMMMdRequest.from_mapping(payload)
            artifacts = openmm_artifact_paths(request)
            for role, path in artifacts.items():
                if role != "worker_receipt":
                    path.write_text(f"{role}\n", encoding="utf-8")
            receipt = {
                "schema_version": "mmpbsa.openmm.worker-receipt.v1",
                "status": "completed",
                "request_sha256": (_digest(request.as_dict()) if self.correct_digest else "f" * 64),
                **_md_receipt_provenance(request),
                "phase": {
                    "phase_id": request.phase.phase_id,
                    "kind": request.phase.kind,
                    "chunk_id": request.chunk_id,
                    "start_step": 0,
                    "end_step": request.phase.steps,
                    "executed_steps": request.phase.steps,
                    "emit_initial_frame": request.emit_initial_frame,
                    "frame_count": (
                        1
                        if request.phase.kind == "em"
                        else request.phase.steps // request.phase.trajectory_interval_steps
                        + int(request.emit_initial_frame)
                    ),
                    "maximum_force_kj_mol_nm": (
                        self.maximum_force if request.phase.kind == "em" else None
                    ),
                },
                "artifacts": {
                    role: _artifact_record(path)
                    for role, path in artifacts.items()
                    if role != "worker_receipt"
                },
            }
        else:
            request = OpenMMMergeRequest.from_mapping(payload)
            artifacts = openmm_merge_artifact_paths(request)
            for role, path in artifacts.items():
                if role != "worker_receipt":
                    path.write_text(f"{role}\n", encoding="utf-8")
            receipt = {
                "schema_version": "mmpbsa.openmm.merge-receipt.v1",
                "status": "completed",
                "request_sha256": (_digest(request.as_dict()) if self.correct_digest else "f" * 64),
                "frame_count": request.expected_frames,
                "energy_rows": 1,
                "artifacts": {
                    role: _artifact_record(path)
                    for role, path in artifacts.items()
                    if role != "worker_receipt"
                },
            }
        artifacts["worker_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
        return CommandRecord(
            argv=command.argv,
            cwd=command.cwd,
            returncode=0,
            stdout=json.dumps(receipt),
            stderr="",
        )


def test_provider_discovery_is_lazy_and_declares_only_md() -> None:
    source_root = Path(__file__).parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mmpbsa.backends.openmm_provider as p; "
            "print('openmm' in sys.modules, p.get_provider().capabilities)",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stdout.strip() == "False ('md',)"
    provider = create_provider()
    assert provider.provider_id == "openmm"
    assert provider.api_version == "2"
    assert provider.package_version == "0.7.0"
    assert provider.get_service("md") is not None
    assert provider.get_service("scoring") is None


def test_probe_uses_worker_subprocess_and_returns_platform_inventory(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner(
        {
            "schema_version": OPENMM_PROBE_SCHEMA,
            "openmm_version": "8.5.2",
            "platforms": [
                {"name": "Reference", "properties": []},
                {"name": "CUDA", "properties": ["DeviceIndex", "Precision"]},
            ],
        }
    )
    backend = OpenMMMdBackend(
        OpenMMWorkerConfig(python_executable="openmm-python"),
        runner=runner,
    )

    identity = backend.probe(cwd=tmp_path)

    assert identity.version == "8.5.2"
    assert identity.platforms == ("Reference", "CUDA")
    assert runner.commands[0].argv == (
        "openmm-python",
        "-m",
        "mmpbsa.backends.openmm_worker",
        "probe",
    )


def test_request_validation_requires_explicit_fail_closed_cuda(
    tmp_path: Path,
) -> None:
    for device_index in ("", " 0", "0 ", "0,1", "0\t1"):
        with pytest.raises(BackendConfigurationError, match="device_index"):
            _request(
                tmp_path,
                platform=OpenMMPlatformSpec(
                    "CUDA",
                    device_index=device_index,
                    precision="mixed",
                ),
            ).validate()
    with pytest.raises(BackendConfigurationError, match="device_index"):
        _request(
            tmp_path,
            platform=OpenMMPlatformSpec("CUDA", precision="mixed"),
        ).validate()
    with pytest.raises(BackendConfigurationError, match="precision"):
        _request(
            tmp_path,
            platform=OpenMMPlatformSpec("CUDA", device_index="0"),
        ).validate()
    with pytest.raises(BackendConfigurationError, match="deterministic_forces"):
        _request(
            tmp_path,
            platform=OpenMMPlatformSpec(
                "CUDA",
                device_index="0",
                precision="mixed",
                deterministic_forces=False,
            ),
        ).validate()

    checkpoint = _input(tmp_path / "input" / "restart.chk")
    with pytest.raises(BackendConfigurationError, match="checkpoint_provenance"):
        replace(
            _request(tmp_path),
            checkpoint_input=checkpoint,
            emit_initial_frame=False,
        ).validate()


def test_worker_command_is_json_only_and_never_appends_trajectory(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    backend = OpenMMMdBackend(
        OpenMMWorkerConfig(
            python_executable="openmm-python",
            environment={"OPENMM_PLUGIN_DIR": "/plugins"},
        )
    )

    receipt = backend.run(request, dry_run=True)

    assert receipt.state is ReceiptState.PLANNED
    assert len(receipt.commands) == 1
    command = receipt.commands[0]
    assert command.argv[:4] == (
        "openmm-python",
        "-m",
        "mmpbsa.backends.openmm_worker",
        "run",
    )
    assert "--request" in command.argv
    assert all("append" not in item.lower() for item in command.argv)
    payload = json.loads(receipt.generated_inputs[0].content)
    assert payload["schema_version"] == "mmpbsa.openmm.md-request.v1"
    assert payload["platform"]["device_index"] == "0"
    assert receipt.artifacts["trajectory"].name == "nvt.xtc"


def test_run_rejects_receipt_for_a_different_request(tmp_path: Path) -> None:
    backend = OpenMMMdBackend(
        OpenMMWorkerConfig(python_executable="openmm-python"),
        runner=_WrongRequestDigestRunner(),
    )

    with pytest.raises(BackendOutputError, match="different MD request"):
        backend.run(_request(tmp_path))


def test_merge_rejects_receipt_for_a_different_request(tmp_path: Path) -> None:
    request = OpenMMMergeRequest(
        trajectories=(_input(tmp_path / "chunks" / "chunk.xtc"),),
        energy_series=(_input(tmp_path / "chunks" / "chunk.csv"),),
        topology=_input(tmp_path / "input" / "topol.top"),
        coordinates=_input(tmp_path / "input" / "system.gro"),
        output_dir=tmp_path / "merged",
        expected_frames=2,
        trajectory_interval_steps=10,
        timestep_fs=2.0,
    )
    backend = OpenMMMdBackend(
        OpenMMWorkerConfig(python_executable="openmm-python"),
        runner=_WrongRequestDigestRunner(),
    )

    with pytest.raises(BackendOutputError, match="different merge request"):
        backend.merge(request)


def test_run_rejects_em_receipt_above_force_threshold(tmp_path: Path) -> None:
    backend = OpenMMMdBackend(
        OpenMMWorkerConfig(python_executable="openmm-python"),
        runner=_WrongRequestDigestRunner(
            correct_digest=True,
            maximum_force=1_000.1,
        ),
    )

    with pytest.raises(BackendOutputError, match="exceeds the request threshold"):
        backend.run(_em_request(tmp_path))


def test_worker_platform_selection_has_no_fallback(tmp_path: Path) -> None:
    request = _request(tmp_path)

    class MissingPlatformApi:
        class Platform:
            @staticmethod
            def getPlatformByName(name):
                raise RuntimeError(name)

    with pytest.raises(OpenMMWorkerError, match="unavailable"):
        _platform(MissingPlatformApi, request)

    class SelectedPlatform:
        def getName(self):
            return "CPU"

    simulation = SimpleNamespace(context=SimpleNamespace(getPlatform=lambda: SelectedPlatform()))
    with pytest.raises(OpenMMWorkerError, match="fail-closed"):
        _actual_platform(simulation, "CUDA", {"DeviceIndex": "0"})


def test_v2_phase_contract_maps_to_exact_openmm_chunks() -> None:
    science = resolve_protocol(
        "smoke_20ps@2",
        md_engine_profile=MD_ENGINE_PROFILE,
    ).stages["md"]

    plans = _phase_plans(science)

    assert [plan.phase_id for plan in plans] == ["em", "nvt", "npt", "production"]
    production = plans[-1]
    assert production.transaction_chunk_steps == 10_000
    assert production.phase.trajectory_interval_steps == 1_000
    assert production.expected_trajectory_frames == 11
    assert production.emit_initial_frame is True
    assert ADAPTER_CONTRACT == "mmpbsa.backends.openmm.md.v2"
    assert science["schema_version"] == MD_PHASE_PLAN_SCHEMA


def test_runtime_and_phase_contracts_fail_closed() -> None:
    with pytest.raises(BackendConfigurationError, match="stage contract"):
        _validate_runtime_contract({"_md_engine_profile": MD_ENGINE_PROFILE})
    with pytest.raises(BackendConfigurationError, match="md_engine_profile"):
        _validate_runtime_contract({"_stage_contract": MD_PHASE_PLAN_SCHEMA})

    science = dict(
        resolve_protocol(
            "smoke_20ps@2",
            md_engine_profile=MD_ENGINE_PROFILE,
        ).stages["md"]
    )
    science["unexpected_public_field"] = True
    with pytest.raises(BackendConfigurationError, match="unknown"):
        _phase_plans(science)
    science.pop("unexpected_public_field")
    phases = [dict(item) for item in science["phases"]]
    phases[1]["kind"] = "production"
    science["phases"] = phases
    with pytest.raises(BackendConfigurationError, match="kind='nvt'"):
        _phase_plans(science)


def test_compatibility_key_includes_exact_version_and_platform_controls(
    tmp_path: Path,
) -> None:
    def service(version: str) -> OpenMMStageService:
        return OpenMMStageService(
            runner=_RecordingRunner(
                {
                    "schema_version": OPENMM_PROBE_SCHEMA,
                    "openmm_version": version,
                    "platforms": [
                        {
                            "name": "CPU",
                            "properties": ["Threads", "DeterministicForces"],
                        }
                    ],
                }
            )
        )

    runtime = {
        "platform": "CPU",
        "threads_per_job": 2,
        "deterministic_forces": True,
        "probe_cwd": str(tmp_path),
        "_stage_contract": MD_PHASE_PLAN_SCHEMA,
        "_md_engine_profile": MD_ENGINE_PROFILE,
    }
    baseline = service("8.5.1").compatibility(runtime)
    patch_release = service("8.5.2").compatibility(runtime)
    nondeterministic = service("8.5.1").compatibility({**runtime, "deterministic_forces": False})
    different_threads = service("8.5.1").compatibility({**runtime, "threads_per_job": 3})

    assert (
        len(
            {
                baseline.key,
                patch_release.key,
                nondeterministic.key,
                different_threads.key,
            }
        )
        == 4
    )
    assert baseline.engine_major == "8.5.1"

    with pytest.raises(BackendConfigurationError, match=">=8.5.1,<8.6"):
        service("8.5.0").compatibility(runtime)


def test_compatibility_ignores_lazy_rdkit_environment_but_tracks_worker_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _RecordingRunner(
        {
            "schema_version": OPENMM_PROBE_SCHEMA,
            "openmm_version": "8.5.1",
            "platforms": [{"name": "CPU", "properties": ["Threads"]}],
        }
    )
    service = OpenMMStageService(runner=runner)
    runtime = {
        "platform": "CPU",
        "threads_per_job": 1,
        "probe_cwd": str(tmp_path),
        "_stage_contract": MD_PHASE_PLAN_SCHEMA,
        "_md_engine_profile": MD_ENGINE_PROFILE,
    }
    monkeypatch.delenv("RDBASE", raising=False)
    baseline = service.compatibility(runtime)
    monkeypatch.setenv("RDBASE", "/late/rdkit/import")
    assert service.compatibility(runtime) == baseline
    with monkeypatch.context() as changed:
        changed.setenv("OPENMM_PLUGIN_DIR", "/different/openmm/plugins")
        assert service.compatibility(runtime).key != baseline.key
    with monkeypatch.context() as changed:
        changed.setenv("PYTHONPATH", "/different/python/modules")
        assert service.compatibility(runtime).key != baseline.key
    for name in (
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "GOTO_NUM_THREADS",
        "KMP_BLOCKTIME",
        "GOMP_CPU_AFFINITY",
    ):
        with monkeypatch.context() as changed:
            changed.setenv(name, "999" if os.environ.get(name) != "999" else "998")
            assert service.compatibility(runtime).key != baseline.key, name
    configured = {**runtime, "environment": {"RDBASE": "/explicit/worker/value"}}
    configured_key = service.compatibility(configured).key
    assert configured_key != baseline.key
    monkeypatch.setenv("RDBASE", "/another/late/value")
    assert service.compatibility(configured).key == configured_key


class _FakeBackend:
    def __init__(self, *, interrupt_kind: str | None = None) -> None:
        self.requests: list[OpenMMMdRequest] = []
        self.merges: list[OpenMMMergeRequest] = []
        self.interrupt_kind = interrupt_kind
        self.did_interrupt = False
        self.worker = OpenMMWorkerConfig(python_executable=sys.executable)

    def probe(self, *, cwd: Path) -> OpenMMIdentity:
        del cwd
        raw = {
            "schema_version": OPENMM_PROBE_SCHEMA,
            "openmm_version": "8.5.2",
            "platforms": [{"name": "CPU", "properties": ["Threads"]}],
        }
        return OpenMMIdentity("8.5.2", ("CPU",), raw)

    def preflight(self, request: OpenMMMdRequest, *, invocation_dir: Path):
        request.validate()
        invocation_dir.mkdir(parents=True, exist_ok=True)
        return {
            "schema_version": "mmpbsa.openmm.amber-preflight.v1",
            "status": "passed",
            "request_sha256": _digest(request.as_dict()),
            "openmm_version": request.expected_openmm_version,
            "probe_sha256": request.expected_probe_sha256,
            "input_format": request.input_format,
            "topology_sha256": request.topology_sha256,
            "coordinates_sha256": request.coordinates_sha256,
            "atom_count": 1,
            "bond_count": 0,
            "particles": 1,
            "has_periodic_box": True,
            "dispersion_correction": True,
        }

    def run(self, request: OpenMMMdRequest) -> BackendReceipt:
        request.validate()
        if request.phase.kind == self.interrupt_kind and not self.did_interrupt:
            self.did_interrupt = True
            raise KeyboardInterrupt(f"interrupted {request.phase.kind}")
        request.output_dir.mkdir(parents=True)
        (request.output_dir / "openmm_request.json").write_text(
            json.dumps(request.as_dict()), encoding="utf-8"
        )
        artifacts = openmm_artifact_paths(request)
        for role, path in artifacts.items():
            if role != "worker_receipt":
                path.write_text(f"{role}\n", encoding="utf-8")
        start_step = (
            0
            if request.checkpoint_input is None
            else sum(
                item.phase.steps
                for item in self.requests
                if item.phase.phase_id == request.phase.phase_id
            )
        )
        receipt = {
            "schema_version": "mmpbsa.openmm.worker-receipt.v1",
            "status": "completed",
            "request_sha256": _digest(request.as_dict()),
            **_md_receipt_provenance(request),
            "phase": {
                "phase_id": request.phase.phase_id,
                "kind": request.phase.kind,
                "chunk_id": request.chunk_id,
                "start_step": start_step,
                "end_step": start_step + request.phase.steps,
                "executed_steps": request.phase.steps,
                "emit_initial_frame": request.emit_initial_frame,
                "frame_count": (
                    1
                    if request.phase.kind == "em"
                    else request.phase.steps // request.phase.trajectory_interval_steps
                    + int(request.emit_initial_frame)
                ),
                "maximum_force_kj_mol_nm": (0.0 if request.phase.kind == "em" else None),
            },
            "artifacts": {
                role: _artifact_record(path)
                for role, path in artifacts.items()
                if role != "worker_receipt"
            },
        }
        if request.input_format == OPENMM_AMBER_INPUT_FORMAT:
            receipt.update(
                {
                    "openmm": {"version": request.expected_openmm_version},
                    "probe_sha256": request.expected_probe_sha256,
                    "input": {
                        "format": request.input_format,
                        "topology_sha256": request.topology_sha256,
                        "coordinates_sha256": request.coordinates_sha256,
                    },
                }
            )
        artifacts["worker_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
        self.requests.append(request)
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=(),
            artifacts=artifacts,
            parsed=receipt,
        )

    def merge(self, request: OpenMMMergeRequest) -> BackendReceipt:
        request.validate()
        request.output_dir.mkdir(parents=True)
        (request.output_dir / "openmm_merge_request.json").write_text(
            json.dumps(request.as_dict()), encoding="utf-8"
        )
        artifacts = openmm_merge_artifact_paths(request)
        for role, path in artifacts.items():
            if role != "worker_receipt":
                path.write_text(f"{role}\n", encoding="utf-8")
        receipt = {
            "schema_version": "mmpbsa.openmm.merge-receipt.v1",
            "status": "completed",
            "request_sha256": _digest(request.as_dict()),
            "frame_count": request.expected_frames,
            "energy_rows": 1,
            "artifacts": {
                role: _artifact_record(path)
                for role, path in artifacts.items()
                if role != "worker_receipt"
            },
        }
        if request.input_format == OPENMM_AMBER_INPUT_FORMAT:
            receipt.update(
                {
                    "openmm": {"version": request.expected_openmm_version},
                    "probe_sha256": request.expected_probe_sha256,
                    "input": {
                        "format": request.input_format,
                        "topology_sha256": request.topology_sha256,
                        "coordinates_sha256": request.coordinates_sha256,
                    },
                }
            )
        artifacts["worker_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
        self.merges.append(request)
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=(),
            artifacts=artifacts,
            parsed=receipt,
        )


def _prepared_view(tmp_path: Path) -> PreparedMdView:
    coordinates = _input(tmp_path / "prepared" / "solvated.inpcrd")
    topology = _input(tmp_path / "prepared" / "solvated.prmtop")

    def artifact(path: Path, role: str) -> ArtifactRef:
        return ArtifactRef(
            path=Path(path.name),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size=path.stat().st_size,
            role=role,
        )

    system = PreparedMdSystem(
        backend="gromacs",
        atom_count=1,
        system_signature="b" * 64,
        artifacts=(
            artifact(coordinates, "solvated_coordinates"),
            artifact(topology, "solvated_topology"),
        ),
    )
    return PreparedMdView(
        case_id="case-a",
        system=system,
        resolved_artifacts={
            "solvated_coordinates": coordinates,
            "solvated_topology": topology,
        },
        input_format="amber",
    )


def test_stage_service_orchestrates_four_phases_and_public_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeBackend()
    monkeypatch.setattr(
        "mmpbsa.backends.openmm_provider.create_openmm_md_backend",
        lambda config, runner=None: fake,
    )
    science = dict(
        resolve_protocol(
            "smoke_20ps@2",
            md_engine_profile=MD_ENGINE_PROFILE,
        ).stages["md"]
    )
    science["_replica"] = {
        "replica_id": "rep01",
        "replica_index": 1,
        "seed": 4101,
    }
    attempt = tmp_path / "attempt"
    request = StageRequest(
        task=TaskKey(StageId.MD, case_id="case-a", replica_id="rep01"),
        science=science,
        runtime={
            "platform": "CPU",
            "threads_per_job": 2,
            "_stage_contract": MD_PHASE_PLAN_SCHEMA,
            "_md_engine_profile": MD_ENGINE_PROFILE,
        },
        prepared_case=_prepared_view(tmp_path),
        upstream={},
        attempt_dir=attempt,
        invocation_dir=attempt / "invocations" / "0001",
    )

    service = OpenMMStageService()
    request = replace(request, compatibility=service.compatibility(request.runtime))
    with monkeypatch.context() as changed:
        changed.setenv("OPENMM_PLUGIN_DIR", "/different/openmm/plugins")
        with pytest.raises(BackendConfigurationError, match="reserved attempt"):
            service.execute(request)
    with monkeypatch.context() as changed:
        changed.setenv(
            "OPENBLAS_NUM_THREADS",
            "3" if os.environ.get("OPENBLAS_NUM_THREADS") != "3" else "4",
        )
        with pytest.raises(BackendConfigurationError, match="reserved attempt"):
            service.execute(request)
    monkeypatch.setenv("RDBASE", "/late/rdkit/import")
    result = service.execute(request)

    assert [item.phase.kind for item in fake.requests] == [
        "em",
        "nvt",
        "npt",
        "production",
    ]
    assert fake.requests[-1].emit_initial_frame is True
    assert len(fake.merges) == 1
    assert fake.merges[0].expected_frames == 11
    assert set(result.artifacts) == {
        "full_trajectory",
        "coordinates",
        "state",
        "checkpoint",
        "energy_log",
    }
    assert result.payload["state_format"] == "openmm_xml_state"
    assert result.payload["checkpoint_format"] == "openmm_binary_checkpoint"
    assert result.checkpoint == result.artifacts["checkpoint"]


def test_merge_seal_rejects_tampered_public_trajectory(tmp_path: Path) -> None:
    source_trajectory = _input(tmp_path / "chunks" / "chunk.xtc")
    source_energy = _input(tmp_path / "chunks" / "chunk.csv")
    request = OpenMMMergeRequest(
        trajectories=(source_trajectory,),
        energy_series=(source_energy,),
        topology=_input(tmp_path / "input" / "topol.top"),
        coordinates=_input(tmp_path / "input" / "system.gro"),
        output_dir=tmp_path / "final",
        expected_frames=2,
        trajectory_interval_steps=10,
        timestep_fs=2.0,
    )
    fake = _FakeBackend()
    fake.merge(request)

    assert _sealed_merge_receipt(request) is not None
    openmm_merge_artifact_paths(request)["full_trajectory"].write_text(
        "tampered\n", encoding="utf-8"
    )
    assert _sealed_merge_receipt(request) is None


def test_phase_seal_rejects_tampered_step_fields(tmp_path: Path) -> None:
    request = replace(
        _request(tmp_path),
        platform=OpenMMPlatformSpec("CPU"),
    )
    fake = _FakeBackend()
    fake.run(request)
    receipt_path = openmm_artifact_paths(request)["worker_receipt"]

    assert _sealed_md_receipt(request) is not None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["phase"]["end_step"] += 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert _sealed_md_receipt(request) is None


def test_phase_seal_rejects_em_force_above_threshold(tmp_path: Path) -> None:
    request = _em_request(tmp_path)
    fake = _FakeBackend()
    fake.run(request)
    receipt_path = openmm_artifact_paths(request)["worker_receipt"]

    assert _sealed_md_receipt(request) is not None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["phase"]["maximum_force_kj_mol_nm"] = 1_000.1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert _sealed_md_receipt(request) is None


def test_phase_seal_rejects_tampered_provenance_and_em_system(
    tmp_path: Path,
) -> None:
    request = _em_request(tmp_path)
    fake = _FakeBackend()
    fake.run(request)
    receipt_path = openmm_artifact_paths(request)["worker_receipt"]
    original = json.loads(receipt_path.read_text(encoding="utf-8"))

    tampered: list[tuple[str, dict[str, object]]] = []

    value = json.loads(json.dumps(original))
    value["unexpected"] = True
    tampered.append(("top-level fields", value))

    value = json.loads(json.dumps(original))
    value["openmm"]["version"] = "8.5.99"
    tampered.append(("OpenMM identity", value))

    value = json.loads(json.dumps(original))
    value["platform"]["properties"]["Threads"] = "99"
    tampered.append(("platform properties", value))

    value = json.loads(json.dumps(original))
    value["checkpoint_provenance"]["system_sha256"] = "b" * 64
    tampered.append(("checkpoint provenance", value))

    value = json.loads(json.dumps(original))
    value["seeds"]["integrator"] += 1
    tampered.append(("seed provenance", value))

    value = json.loads(json.dumps(original))
    value["system"]["constraints"] = 1
    tampered.append(("EM constraints", value))

    for label, payload in tampered:
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        assert _sealed_md_receipt(request) is None, label


def test_merge_seal_rejects_tampered_energy_rows(tmp_path: Path) -> None:
    request = OpenMMMergeRequest(
        trajectories=(_input(tmp_path / "chunks" / "chunk.xtc"),),
        energy_series=(_input(tmp_path / "chunks" / "chunk.csv"),),
        topology=_input(tmp_path / "input" / "topol.top"),
        coordinates=_input(tmp_path / "input" / "system.gro"),
        output_dir=tmp_path / "final",
        expected_frames=2,
        trajectory_interval_steps=10,
        timestep_fs=2.0,
    )
    fake = _FakeBackend()
    fake.merge(request)
    receipt_path = openmm_merge_artifact_paths(request)["worker_receipt"]

    assert _sealed_merge_receipt(request) is not None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["energy_rows"] = 0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert _sealed_merge_receipt(request) is None


def test_merge_seal_rejects_unknown_receipt_fields(tmp_path: Path) -> None:
    request = OpenMMMergeRequest(
        trajectories=(_input(tmp_path / "chunks" / "chunk.xtc"),),
        energy_series=(_input(tmp_path / "chunks" / "chunk.csv"),),
        topology=_input(tmp_path / "input" / "topol.top"),
        coordinates=_input(tmp_path / "input" / "system.gro"),
        output_dir=tmp_path / "final",
        expected_frames=2,
        trajectory_interval_steps=10,
        timestep_fs=2.0,
    )
    fake = _FakeBackend()
    fake.merge(request)
    receipt_path = openmm_merge_artifact_paths(request)["worker_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert _sealed_merge_receipt(request) is not None
    receipt["unexpected"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert _sealed_merge_receipt(request) is None


def test_resume_reset_rejects_symlink_phase_directory(tmp_path: Path) -> None:
    root = tmp_path / "attempt" / "output"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = _input(outside / "must-survive.txt")
    (root / "em").symlink_to(outside, target_is_directory=True)

    with pytest.raises(BackendConfigurationError, match="symlink"):
        _reset_output(root / "em", root)

    assert marker.read_text(encoding="utf-8") == "input\n"


def test_interruption_exposes_last_sealed_checkpoint_and_resume_skips_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeBackend(interrupt_kind="nvt")
    monkeypatch.setattr(
        "mmpbsa.backends.openmm_provider.create_openmm_md_backend",
        lambda config, runner=None: fake,
    )
    science = dict(
        resolve_protocol(
            "smoke_20ps@2",
            md_engine_profile=MD_ENGINE_PROFILE,
        ).stages["md"]
    )
    science["_replica"] = {
        "replica_id": "rep01",
        "replica_index": 1,
        "seed": 4101,
    }
    attempt = tmp_path / "attempt"
    initial = StageRequest(
        task=TaskKey(StageId.MD, case_id="case-a", replica_id="rep01"),
        science=science,
        runtime={
            "platform": "CPU",
            "threads_per_job": 1,
            "_stage_contract": MD_PHASE_PLAN_SCHEMA,
            "_md_engine_profile": MD_ENGINE_PROFILE,
        },
        prepared_case=_prepared_view(tmp_path),
        upstream={},
        attempt_dir=attempt,
        invocation_dir=attempt / "invocations" / "0001",
    )
    service = OpenMMStageService()
    initial = replace(initial, compatibility=service.compatibility(initial.runtime))

    with pytest.raises(KeyboardInterrupt) as interrupted:
        service.execute(initial)

    checkpoint = Path(interrupted.value.checkpoint)
    assert checkpoint == (attempt / "output" / "em" / "em.checkpoint.chk").resolve()
    assert checkpoint.is_file()

    fake.interrupt_kind = None
    resumed = service.resume(
        replace(
            initial,
            invocation_dir=attempt / "invocations" / "0002",
            resume=True,
            checkpoint=checkpoint,
        )
    )

    assert sum(item.phase.kind == "em" for item in fake.requests) == 1
    assert resumed.payload["recovery"]["skipped"] == ["em"]


def _write_water_system(root: Path, *, grid: int = 1) -> tuple[Path, Path]:
    molecule_count = grid**3
    topology = _input(
        root / "water.top",
        f"""[ defaults ]
1 2 yes 0.5 0.83333333

[ atomtypes ]
OW 8 15.9994 0.0 A 0.31507524 0.635968
HW 1 1.0080 0.0 A 0.0 0.0

[ moleculetype ]
SOL 2

[ atoms ]
1 OW 1 SOL OW 1 -0.834 15.9994
2 HW 1 SOL HW1 1 0.417 1.008
3 HW 1 SOL HW2 1 0.417 1.008

[ settles ]
1 1 0.09572 0.15139

[ exclusions ]
1 2 3
2 1 3
3 1 2

[ system ]
water

[ molecules ]
SOL {molecule_count}
""",
    )
    if grid == 1:
        coordinate_lines = [
            "    1SOL     OW    1   1.000   1.000   1.000",
            "    1SOL    HW1    2   1.096   1.000   1.000",
            "    1SOL    HW2    3   0.976   1.093   1.000",
        ]
        box_length = 2.0
    else:
        spacing = 0.31
        coordinate_lines = []
        atom_index = 1
        residue_index = 1
        for x_index in range(grid):
            for y_index in range(grid):
                for z_index in range(grid):
                    x = 0.08 + x_index * spacing
                    y = 0.08 + y_index * spacing
                    z = 0.08 + z_index * spacing
                    for atom_name, dx, dy in (
                        ("OW", 0.0, 0.0),
                        ("HW1", 0.09572, 0.0),
                        ("HW2", -0.02399, 0.09266),
                    ):
                        coordinate_lines.append(
                            f"{residue_index:5d}{'SOL':<5}{atom_name:>5}"
                            f"{atom_index:5d}{x + dx:8.3f}{y + dy:8.3f}{z:8.3f}"
                        )
                        atom_index += 1
                    residue_index += 1
        box_length = grid * spacing
    coordinates = _input(
        root / "water.gro",
        "water\n"
        f"{molecule_count * 3}\n"
        + "\n".join(coordinate_lines)
        + "\n"
        + f"{box_length:10.5f}{box_length:10.5f}{box_length:10.5f}\n",
    )
    return topology, coordinates


def _write_argon_system(root: Path, *, grid: int = 6) -> tuple[Path, Path]:
    atom_count = grid**3
    topology = _input(
        root / "argon.top",
        f"""[ defaults ]
1 2 yes 0.5 0.83333333

[ atomtypes ]
AR 18 39.948 0.0 A 0.340 0.997

[ moleculetype ]
ARG 1

[ atoms ]
1 AR 1 ARG AR 1 0.0 39.948

[ system ]
argon

[ molecules ]
ARG {atom_count}
""",
    )
    spacing = 0.38
    lines = []
    atom_index = 1
    for x_index in range(grid):
        for y_index in range(grid):
            for z_index in range(grid):
                lines.append(
                    f"{atom_index:5d}{'ARG':<5}{'AR':>5}{atom_index:5d}"
                    f"{(x_index + 0.5) * spacing:8.3f}"
                    f"{(y_index + 0.5) * spacing:8.3f}"
                    f"{(z_index + 0.5) * spacing:8.3f}"
                )
                atom_index += 1
    box_length = grid * spacing
    coordinates = _input(
        root / "argon.gro",
        "argon\n"
        f"{atom_count}\n"
        + "\n".join(lines)
        + "\n"
        + f"{box_length:10.5f}{box_length:10.5f}{box_length:10.5f}\n",
    )
    return topology, coordinates


def _write_amber_argon_system(root: Path) -> tuple[Path, Path]:
    import parmed as pmd

    gromacs_topology, gromacs_coordinates = _write_argon_system(root / "gromacs")
    structure = pmd.load_file(str(gromacs_topology), xyz=str(gromacs_coordinates))
    root.mkdir(parents=True, exist_ok=True)
    topology = root / "solvated.prmtop"
    coordinates = root / "solvated.inpcrd"
    structure.save(str(topology), overwrite=True)
    structure.save(str(coordinates), overwrite=True)
    return topology, coordinates


@pytest.mark.skipif(
    os.environ.get("MMPBSA_OPENMM_SMOKE") != "1",
    reason="set MMPBSA_OPENMM_SMOKE=1 for a real OpenMM CPU subprocess",
)
def test_real_openmm_cpu_worker_smoke(tmp_path: Path) -> None:
    topology, coordinates = _write_water_system(tmp_path / "input")
    request = OpenMMMdRequest(
        topology=topology,
        coordinates=coordinates,
        output_dir=tmp_path / "reference",
        phase=_phase(steps=10),
        platform=OpenMMPlatformSpec("CPU", deterministic_forces=True, threads=1),
        seed=4101,
    )
    source_root = Path(__file__).parents[2] / "src"
    backend = OpenMMMdBackend(
        OpenMMWorkerConfig(
            python_executable=sys.executable,
            environment={"PYTHONPATH": str(source_root)},
        )
    )

    receipt = backend.run(request)

    assert receipt.parsed["phase"]["frame_count"] == 3
    assert receipt.parsed["platform"]["name"] == "CPU"
    assert all(path.is_file() for path in receipt.artifacts.values())


@pytest.mark.skipif(
    os.environ.get("MMPBSA_OPENMM_SMOKE") != "1",
    reason="set MMPBSA_OPENMM_SMOKE=1 for the real OpenMM CPU phase pipeline",
)
def test_real_openmm_cpu_stage_pipeline_smoke(tmp_path: Path) -> None:
    topology, coordinates = _write_amber_argon_system(tmp_path / "prepared")

    def artifact(path: Path, role: str) -> ArtifactRef:
        return ArtifactRef(
            path=Path(path.name),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size=path.stat().st_size,
            role=role,
        )

    prepared = PreparedMdView(
        case_id="water",
        system=PreparedMdSystem(
            backend="gromacs",
            atom_count=216,
            system_signature="b" * 64,
            artifacts=(
                artifact(coordinates, "solvated_coordinates"),
                artifact(topology, "solvated_topology"),
            ),
        ),
        resolved_artifacts={
            "solvated_coordinates": coordinates,
            "solvated_topology": topology,
        },
        input_format="amber",
    )
    science = dict(
        resolve_protocol(
            "smoke_20ps@2",
            md_engine_profile=MD_ENGINE_PROFILE,
        ).stages["md"]
    )
    science["_replica"] = {
        "replica_id": "rep01",
        "replica_index": 1,
        "seed": 4101,
    }
    source_root = Path(__file__).parents[2] / "src"
    attempt = tmp_path / "attempt"
    request = StageRequest(
        task=TaskKey(StageId.MD, case_id="water", replica_id="rep01"),
        science=science,
        runtime={
            "python_executable": sys.executable,
            "environment": {"PYTHONPATH": str(source_root)},
            "platform": "CPU",
            "threads_per_job": 2,
            "deterministic_forces": True,
            "_stage_contract": MD_PHASE_PLAN_SCHEMA,
            "_md_engine_profile": MD_ENGINE_PROFILE,
        },
        prepared_case=prepared,
        upstream={},
        attempt_dir=attempt,
        invocation_dir=attempt / "invocations" / "0001",
    )

    service = OpenMMStageService()
    request = replace(request, compatibility=service.compatibility(request.runtime))
    result = service.execute(request)

    assert result.payload["expected_trajectory_frames"] == 11
    assert set(result.artifacts) == {
        "full_trajectory",
        "coordinates",
        "state",
        "checkpoint",
        "energy_log",
    }
    assert all(path.is_file() for path in result.artifacts.values())
    private_artifacts = result.invocation_provenance["private_artifacts"]
    manifest = json.loads(
        Path(private_artifacts["trajectory_manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["frame_count"] == 11
    assert manifest["first_step"] == 0
    assert manifest["last_step"] == 10_000


@pytest.mark.skipif(
    os.environ.get("MMPBSA_OPENMM_CUDA_SMOKE") != "1",
    reason="set MMPBSA_OPENMM_CUDA_SMOKE=1 for a real CUDA worker smoke",
)
def test_real_openmm_cuda_worker_has_no_platform_or_device_fallback(
    tmp_path: Path,
) -> None:
    topology, coordinates = _write_water_system(tmp_path / "input")
    device_index = os.environ.get("MMPBSA_OPENMM_CUDA_DEVICE", "0")
    request = OpenMMMdRequest(
        topology=topology,
        coordinates=coordinates,
        output_dir=tmp_path / "cuda",
        phase=_phase(steps=10),
        platform=OpenMMPlatformSpec(
            "CUDA",
            device_index=device_index,
            precision="mixed",
            deterministic_forces=True,
        ),
        seed=4101,
    )
    source_root = Path(__file__).parents[2] / "src"
    receipt = OpenMMMdBackend(
        OpenMMWorkerConfig(
            python_executable=sys.executable,
            environment={"PYTHONPATH": str(source_root)},
        )
    ).run(request)

    assert receipt.parsed["phase"]["frame_count"] == 3
    assert receipt.parsed["platform"]["name"] == "CUDA"
    assert receipt.parsed["platform"]["properties"] == {
        "DeviceIndex": device_index,
        "Precision": "mixed",
        "DeterministicForces": "true",
    }


@pytest.mark.skipif(
    os.environ.get("MMPBSA_OPENMM_REFERENCE_SMOKE") != "1",
    reason="set MMPBSA_OPENMM_REFERENCE_SMOKE=1 for Reference diagnostics",
)
def test_real_openmm_reference_worker_diagnostic(tmp_path: Path) -> None:
    topology, coordinates = _write_water_system(tmp_path / "input")
    request = OpenMMMdRequest(
        topology=topology,
        coordinates=coordinates,
        output_dir=tmp_path / "reference",
        phase=_phase(steps=10),
        platform=OpenMMPlatformSpec("Reference"),
        seed=4101,
    )
    source_root = Path(__file__).parents[2] / "src"
    receipt = OpenMMMdBackend(
        OpenMMWorkerConfig(
            python_executable=sys.executable,
            environment={"PYTHONPATH": str(source_root)},
        )
    ).run(request)

    assert receipt.parsed["phase"]["frame_count"] == 3
    assert receipt.parsed["platform"]["name"] == "Reference"

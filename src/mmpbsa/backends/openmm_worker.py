"""Subprocess worker for OpenMM MD execution.

All imports of the optional OpenMM package are localized below the command
boundary.  Importing this module for provider discovery is therefore safe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .openmm import (
    OPENMM_AMBER_INPUT_FORMAT,
    OPENMM_MERGE_RECEIPT_SCHEMA,
    OPENMM_PROBE_SCHEMA,
    OPENMM_RECEIPT_SCHEMA,
    OpenMMMdRequest,
    OpenMMMergeRequest,
    openmm_artifact_paths,
    openmm_merge_artifact_paths,
)


class OpenMMWorkerError(RuntimeError):
    """A deterministic worker contract or runtime failure."""


def _load_openmm() -> tuple[Any, Any, Any]:
    try:
        openmm = importlib.import_module("openmm")
        app = importlib.import_module("openmm.app")
        unit = importlib.import_module("openmm.unit")
    except ImportError as exc:
        raise OpenMMWorkerError(
            "OpenMM is not installed in the configured worker environment"
        ) from exc
    return openmm, app, unit


def probe_openmm() -> dict[str, object]:
    openmm, _, _ = _load_openmm()
    platforms = []
    for index in range(openmm.Platform.getNumPlatforms()):
        platform = openmm.Platform.getPlatform(index)
        platforms.append(
            {
                "name": platform.getName(),
                "properties": list(platform.getPropertyNames()),
            }
        )
    return {
        "schema_version": OPENMM_PROBE_SCHEMA,
        "openmm_version": str(openmm.__version__),
        "platforms": platforms,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_reserved_probe(request: Any, openmm: Any) -> None:
    if request.expected_openmm_version is None:
        return
    if (
        str(openmm.__version__) != request.expected_openmm_version
        or _sha256_json(probe_openmm()) != request.expected_probe_sha256
    ):
        raise OpenMMWorkerError("OpenMM worker probe differs from reserved attempt")


def _artifact_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise OpenMMWorkerError(f"worker artifact is missing or empty: {resolved}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
    )


def _derive_seed(master_seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


def _coordinate_data(app: Any, path: Path) -> tuple[Any, Any, Any | None]:
    suffix = path.suffix.lower()
    if suffix == ".gro":
        source = app.GromacsGroFile(str(path))
        return source.positions, source.getPeriodicBoxVectors(), None
    if suffix == ".pdb":
        source = app.PDBFile(str(path))
        return (
            source.positions,
            source.topology.getPeriodicBoxVectors(),
            source.topology,
        )
    if suffix in {".inpcrd", ".rst7", ".crd"}:
        source = app.AmberInpcrdFile(str(path))
        return source.positions, source.getBoxVectors(), None
    raise OpenMMWorkerError(f"unsupported OpenMM coordinate format: {path}")


def _build_system(openmm: Any, app: Any, unit: Any, request: OpenMMMdRequest):
    positions, box_vectors, _ = _coordinate_data(app, request.coordinates)
    amber = request.input_format == OPENMM_AMBER_INPUT_FORMAT
    topology_source = (
        app.AmberPrmtopFile(str(request.topology))
        if amber
        else app.GromacsTopFile(str(request.topology), periodicBoxVectors=box_vectors)
    )
    if topology_source.topology.getNumAtoms() != len(positions):
        raise OpenMMWorkerError("OpenMM topology and coordinates contain different atom counts")
    if box_vectors is None:
        raise OpenMMWorkerError("OpenMM periodic MD requires coordinate box vectors")
    if amber:
        topology_source.topology.setPeriodicBoxVectors(box_vectors)
    constraint_policy = _constraint_policy(app, request.phase.kind)
    options = dict(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        **constraint_policy,
        ewaldErrorTolerance=1.0e-5,
        removeCMMotion=False,
    )
    if not amber:
        options["useDispersionCorrection"] = True
    system = topology_source.createSystem(**options)
    if amber:
        nonbonded = [
            system.getForce(index)
            for index in range(system.getNumForces())
            if isinstance(system.getForce(index), openmm.NonbondedForce)
        ]
        if len(nonbonded) != 1:
            raise OpenMMWorkerError("Amber System must contain exactly one NonbondedForce")
        nonbonded[0].setUseDispersionCorrection(True)
    system.addForce(openmm.CMMotionRemover(100))
    if request.phase.kind in {"npt", "production"}:
        barostat = openmm.MonteCarloBarostat(
            float(request.phase.pressure_bar) * unit.bar,
            float(request.phase.temperature_kelvin) * unit.kelvin,
            25,
        )
        barostat.setRandomNumberSeed(_derive_seed(request.seed, "barostat"))
        system.addForce(barostat)
    return topology_source.topology, positions, box_vectors, system


def run_preflight_worker(request: OpenMMMdRequest) -> dict[str, object]:
    """Parse the sealed Amber system before any phase output is created."""
    request.validate()
    if request.input_format != OPENMM_AMBER_INPUT_FORMAT or request.phase.kind != "em":
        raise OpenMMWorkerError("OpenMM Amber preflight requires the EM input request")
    openmm, app, unit = _load_openmm()
    _verify_reserved_probe(request, openmm)
    topology, positions, box_vectors, system = _build_system(openmm, app, unit, request)
    nonbonded = [
        system.getForce(index)
        for index in range(system.getNumForces())
        if isinstance(system.getForce(index), openmm.NonbondedForce)
    ]
    if len(nonbonded) != 1 or not nonbonded[0].getUseDispersionCorrection():
        raise OpenMMWorkerError("OpenMM Amber preflight dispersion policy differs")
    if (
        system.getNumParticles() != topology.getNumAtoms()
        or len(positions) != topology.getNumAtoms()
        or box_vectors is None
    ):
        raise OpenMMWorkerError("OpenMM Amber preflight particle/coordinate/box mismatch")
    return {
        "schema_version": "mmpbsa.openmm.amber-preflight.v1",
        "status": "passed",
        "request_sha256": _sha256_json(request.as_dict()),
        "openmm_version": str(openmm.__version__),
        "probe_sha256": request.expected_probe_sha256,
        "input_format": request.input_format,
        "topology_sha256": _sha256_file(request.topology),
        "coordinates_sha256": _sha256_file(request.coordinates),
        "atom_count": topology.getNumAtoms(),
        "bond_count": sum(1 for _ in topology.bonds()),
        "particles": system.getNumParticles(),
        "has_periodic_box": True,
        "dispersion_correction": True,
    }


def _constraint_policy(app: Any, phase_kind: str) -> dict[str, object]:
    """Lower the common protocol to an unconstrained EM and constrained MD."""

    if phase_kind == "em":
        return {"constraints": None, "rigidWater": False}
    return {"constraints": app.HBonds, "rigidWater": True}


def _integrator(openmm: Any, unit: Any, request: OpenMMMdRequest):
    if request.phase.kind == "em":
        return openmm.VerletIntegrator(1.0 * unit.femtosecond)
    integrator = openmm.LangevinMiddleIntegrator(
        float(request.phase.temperature_kelvin) * unit.kelvin,
        1.0 / unit.picosecond,
        float(request.phase.timestep_fs) * unit.femtosecond,
    )
    integrator.setConstraintTolerance(1.0e-5)
    integrator.setRandomNumberSeed(_derive_seed(request.seed, "integrator"))
    return integrator


def _platform(openmm: Any, request: OpenMMMdRequest) -> tuple[Any, dict[str, str]]:
    try:
        platform = openmm.Platform.getPlatformByName(request.platform.name)
    except Exception as exc:
        raise OpenMMWorkerError(
            f"requested OpenMM platform is unavailable: {request.platform.name}"
        ) from exc
    if request.platform.name == "CUDA":
        properties = {
            "DeviceIndex": str(request.platform.device_index),
            "Precision": str(request.platform.precision),
            "DeterministicForces": "true",
        }
    elif request.platform.name == "CPU":
        properties = {
            "Threads": str(request.platform.threads),
            "DeterministicForces": ("true" if request.platform.deterministic_forces else "false"),
        }
    else:
        properties = {}
    return platform, properties


def _actual_platform(
    simulation: Any,
    requested_name: str,
    requested_properties: Mapping[str, str],
) -> dict[str, object]:
    selected = simulation.context.getPlatform()
    name = selected.getName()
    if name != requested_name:
        raise OpenMMWorkerError(
            f"OpenMM selected {name!r}, expected fail-closed {requested_name!r}"
        )
    actual: dict[str, str] = {}
    for key, expected in requested_properties.items():
        try:
            value = selected.getPropertyValue(simulation.context, key)
        except Exception as exc:
            raise OpenMMWorkerError(
                f"OpenMM platform did not expose requested property {key!r}"
            ) from exc
        actual[key] = value
        if key != "DeviceIndex" and value.lower() != expected.lower():
            raise OpenMMWorkerError(
                f"OpenMM platform property {key}={value!r}, expected {expected!r}"
            )
        if key == "DeviceIndex" and value != expected:
            raise OpenMMWorkerError(
                f"OpenMM CUDA DeviceIndex={value!r}, expected leased {expected!r}"
            )
    diagnostics = {}
    for key in ("DeviceName",):
        if key in selected.getPropertyNames():
            try:
                diagnostics[key] = selected.getPropertyValue(simulation.context, key)
            except Exception:
                pass
    return {
        "name": name,
        "properties": actual,
        "diagnostics": diagnostics,
    }


def _checkpoint_provenance(
    *,
    openmm_version: str,
    system_sha256: str,
    platform: Mapping[str, object],
) -> dict[str, object]:
    return {
        "openmm_version": openmm_version,
        "system_sha256": system_sha256,
        "platform": {
            "name": platform["name"],
            "properties": dict(platform["properties"]),  # type: ignore[arg-type]
        },
    }


def _validate_checkpoint_provenance(
    expected: Mapping[str, object] | None,
    current: Mapping[str, object],
) -> None:
    if expected is None:
        raise OpenMMWorkerError("checkpoint provenance is absent")
    if dict(expected) != dict(current):
        raise OpenMMWorkerError("checkpoint provenance differs from OpenMM version/System/platform")


class _XtcReporter:
    def __init__(
        self,
        app: Any,
        file_name: str,
        topology: Any,
        timestep: Any,
        interval: int,
        first_step: int,
    ) -> None:
        self._interval = interval
        self._file = app.XTCFile(
            file_name,
            topology,
            timestep,
            firstStep=first_step,
            interval=interval,
            append=False,
        )

    def describeNextReport(self, simulation: Any) -> dict[str, object]:
        return {
            "steps": self._interval - simulation.currentStep % self._interval,
            "periodic": True,
            "include": ["positions"],
        }

    def report(self, simulation: Any, state: Any) -> None:
        self._file.writeModel(
            state.getPositions(asNumpy=True),
            periodicBoxVectors=state.getPeriodicBoxVectors(),
        )


class _AtomicCheckpointReporter:
    def __init__(self, path: Path, interval: int) -> None:
        self._path = path
        self._interval = interval

    def describeNextReport(self, simulation: Any) -> dict[str, object]:
        return {
            "steps": self._interval - simulation.currentStep % self._interval,
            "periodic": None,
            "include": [],
        }

    def report(self, simulation: Any, state: Any) -> None:
        del state
        _atomic_bytes(self._path, simulation.context.createCheckpoint())


def _max_force(state: Any, unit: Any) -> float:
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    return max(
        math.sqrt(float(vector[0]) ** 2 + float(vector[1]) ** 2 + float(vector[2]) ** 2)
        for vector in forces
    )


def _save_final_artifacts(
    openmm: Any,
    app: Any,
    simulation: Any,
    topology: Any,
    artifacts: Mapping[str, Path],
) -> None:
    _atomic_bytes(artifacts["checkpoint"], simulation.context.createCheckpoint())
    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getParameters=True,
        enforcePeriodicBox=True,
    )
    _atomic_text(artifacts["portable_state"], openmm.XmlSerializer.serialize(state))
    temporary = artifacts["final_coordinates"].with_name(
        artifacts["final_coordinates"].name + ".tmp"
    )
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with temporary.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(
            topology,
            state.getPositions(),
            handle,
            keepIds=False,
        )
    os.replace(temporary, artifacts["final_coordinates"])


def _frame_count(openmm: Any, path: Path) -> int:
    xtc_utils = importlib.import_module("openmm.app.internal.xtc_utils")
    return int(xtc_utils.get_xtc_nframes(str(path).encode()))


def _ensure_outputs_absent(artifacts: Mapping[str, Path]) -> None:
    existing = [str(path) for path in artifacts.values() if path.exists()]
    if existing:
        raise OpenMMWorkerError(
            "OpenMM worker refuses to append or overwrite output artifacts: " + ", ".join(existing)
        )


def run_md_worker(request: OpenMMMdRequest) -> dict[str, object]:
    request.validate()
    root = request.output_dir.resolve()
    if not root.is_dir():
        raise OpenMMWorkerError(f"OpenMM output directory does not exist: {root}")
    artifacts = openmm_artifact_paths(request)
    _ensure_outputs_absent(artifacts)

    openmm, app, unit = _load_openmm()
    _verify_reserved_probe(request, openmm)
    topology, positions, box_vectors, system = _build_system(openmm, app, unit, request)
    integrator = _integrator(openmm, unit, request)
    requested_platform, requested_properties = _platform(openmm, request)
    try:
        simulation = app.Simulation(
            topology,
            system,
            integrator,
            requested_platform,
            requested_properties,
        )
    except Exception as exc:
        raise OpenMMWorkerError(
            f"cannot create requested OpenMM {request.platform.name} Context"
        ) from exc
    platform = _actual_platform(
        simulation,
        request.platform.name,
        requested_properties,
    )
    system_xml = openmm.XmlSerializer.serialize(system)
    system_sha256 = hashlib.sha256(system_xml.encode()).hexdigest()
    provenance = _checkpoint_provenance(
        openmm_version=str(openmm.__version__),
        system_sha256=system_sha256,
        platform=platform,
    )

    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    if request.checkpoint_input is not None:
        _validate_checkpoint_provenance(request.checkpoint_provenance, provenance)
        try:
            simulation.loadCheckpoint(str(request.checkpoint_input))
        except Exception as exc:
            raise OpenMMWorkerError("OpenMM binary checkpoint cannot be loaded") from exc
    elif request.state_input is not None:
        try:
            simulation.loadState(str(request.state_input))
        except Exception as exc:
            raise OpenMMWorkerError("OpenMM portable State cannot be loaded") from exc
        simulation.currentStep = 0
        simulation.context.setTime(0.0 * unit.picosecond)
    if request.phase.initialize_velocities:
        simulation.context.setVelocitiesToTemperature(
            float(request.phase.temperature_kelvin) * unit.kelvin,
            _derive_seed(request.seed, "velocities"),
        )

    start_step = int(simulation.currentStep)
    max_force = None
    if request.phase.kind == "em":
        simulation.minimizeEnergy(
            tolerance=(
                float(request.phase.minimization_tolerance_kj_mol_nm)
                * unit.kilojoule_per_mole
                / unit.nanometer
            ),
            maxIterations=int(request.phase.max_iterations),
        )
        force_state = simulation.context.getState(getForces=True)
        max_force = _max_force(force_state, unit)
        if max_force > float(request.phase.maximum_force_threshold_kj_mol_nm):
            raise OpenMMWorkerError(
                f"OpenMM minimization exceeded maximum force threshold: {max_force:g} kJ mol-1 nm-1"
            )

    interval = request.phase.trajectory_interval_steps
    if start_step % interval:
        raise OpenMMWorkerError("OpenMM chunk start step must align to trajectory interval")
    first_step = start_step if request.emit_initial_frame else start_step + interval
    xtc_reporter = _XtcReporter(
        app,
        str(artifacts["trajectory"]),
        topology,
        integrator.getStepSize(),
        interval,
        first_step,
    )
    with artifacts["energy_series"].open("w", encoding="utf-8", newline="") as stream:
        energy_reporter = app.StateDataReporter(
            stream,
            request.phase.report_interval_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            separator=",",
        )
        if request.emit_initial_frame or request.phase.kind == "em":
            initial_positions = simulation.context.getState(
                getPositions=True,
                enforcePeriodicBox=True,
            )
            xtc_reporter.report(simulation, initial_positions)
            initial_energy = simulation.context.getState(getEnergy=True)
            energy_reporter.report(simulation, initial_energy)
        if request.phase.kind != "em":
            simulation.reporters.extend(
                (
                    xtc_reporter,
                    energy_reporter,
                    _AtomicCheckpointReporter(
                        artifacts["checkpoint"],
                        request.phase.checkpoint_interval_steps,
                    ),
                )
            )
            simulation.step(request.phase.steps)
            simulation.reporters.clear()

    _save_final_artifacts(openmm, app, simulation, topology, artifacts)
    end_step = int(simulation.currentStep)
    expected_frames = (
        1
        if request.phase.kind == "em"
        else request.phase.steps // interval + int(request.emit_initial_frame)
    )
    frame_count = _frame_count(openmm, artifacts["trajectory"])
    if frame_count != expected_frames:
        raise OpenMMWorkerError(f"OpenMM wrote {frame_count} frames; expected {expected_frames}")
    seeds = {
        "master": request.seed,
        "integrator": _derive_seed(request.seed, "integrator"),
        "barostat": _derive_seed(request.seed, "barostat"),
        "velocities": _derive_seed(request.seed, "velocities"),
    }
    artifact_records = {
        role: _artifact_record(path) for role, path in artifacts.items() if role != "worker_receipt"
    }
    receipt = {
        "schema_version": OPENMM_RECEIPT_SCHEMA,
        "status": "completed",
        "request_sha256": _sha256_json(request.as_dict()),
        "openmm": {
            "version": str(openmm.__version__),
        },
        "platform": platform,
        "checkpoint_provenance": provenance,
        "system_sha256": system_sha256,
        "system": {
            "particles": system.getNumParticles(),
            "constraints": system.getNumConstraints(),
            "forces": [
                system.getForce(index).__class__.__name__ for index in range(system.getNumForces())
            ],
        },
        "phase": {
            "phase_id": request.phase.phase_id,
            "kind": request.phase.kind,
            "chunk_id": request.chunk_id,
            "start_step": start_step,
            "end_step": end_step,
            "executed_steps": request.phase.steps,
            "emit_initial_frame": request.emit_initial_frame,
            "frame_count": frame_count,
            "maximum_force_kj_mol_nm": max_force,
        },
        "seeds": seeds,
        "artifacts": artifact_records,
    }
    if request.input_format == OPENMM_AMBER_INPUT_FORMAT:
        receipt["probe_sha256"] = request.expected_probe_sha256
        receipt["input"] = {
            "format": request.input_format,
            "topology_sha256": _sha256_file(request.topology),
            "coordinates_sha256": _sha256_file(request.coordinates),
        }
    _atomic_json(artifacts["worker_receipt"], receipt)
    return receipt


def _load_merge_topology(app: Any, request: OpenMMMergeRequest):
    if request.input_format == OPENMM_AMBER_INPUT_FORMAT:
        topology = app.AmberPrmtopFile(str(request.topology)).topology
        positions, box_vectors, _ = _coordinate_data(app, request.coordinates)
        if topology.getNumAtoms() != len(positions):
            raise OpenMMWorkerError(
                "Amber merge topology and coordinates have different atom counts"
            )
        if box_vectors is None:
            raise OpenMMWorkerError("Amber merge requires periodic box vectors")
        topology.setPeriodicBoxVectors(box_vectors)
        return topology
    _, box_vectors, _ = _coordinate_data(app, request.coordinates)
    return app.GromacsTopFile(str(request.topology), periodicBoxVectors=box_vectors).topology


def _read_xtc(openmm: Any, path: Path):
    del openmm
    xtc_utils = importlib.import_module("openmm.app.internal.xtc_utils")
    return xtc_utils.read_xtc(str(path).encode())


def _merge_energy_series(sources: Sequence[Path], destination: Path) -> int:
    header: list[str] | None = None
    rows: list[list[str]] = []
    last_step: int | None = None
    for source in sources:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                current_header = next(reader)
            except StopIteration as exc:
                raise OpenMMWorkerError(f"empty OpenMM energy chunk: {source}") from exc
            if header is None:
                header = current_header
            elif current_header != header:
                raise OpenMMWorkerError("OpenMM energy chunk headers differ")
            for row in reader:
                if not row:
                    continue
                try:
                    step = int(row[0])
                except (ValueError, IndexError) as exc:
                    raise OpenMMWorkerError(f"invalid OpenMM energy step in {source}") from exc
                if last_step is not None and step <= last_step:
                    raise OpenMMWorkerError(
                        "OpenMM energy chunks contain duplicate/non-monotonic steps"
                    )
                rows.append(row)
                last_step = step
    if header is None or not rows:
        raise OpenMMWorkerError("OpenMM merged energy series is empty")
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, destination)
    return len(rows)


def run_merge_worker(request: OpenMMMergeRequest) -> dict[str, object]:
    request.validate()
    root = request.output_dir.resolve()
    if not root.is_dir():
        raise OpenMMWorkerError(f"OpenMM merge output directory does not exist: {root}")
    artifacts = openmm_merge_artifact_paths(request)
    _ensure_outputs_absent(artifacts)
    openmm, app, unit = _load_openmm()
    _verify_reserved_probe(request, openmm)
    topology = _load_merge_topology(app, request)
    expected_step = 0
    frame_count = 0
    source_records = []
    temporary = artifacts["full_trajectory"].with_name(artifacts["full_trajectory"].name + ".tmp")
    writer = app.XTCFile(
        str(temporary),
        topology,
        request.timestep_fs * unit.femtosecond,
        firstStep=0,
        interval=request.trajectory_interval_steps,
        append=False,
    )
    for source in request.trajectories:
        positions, boxes, times, steps = _read_xtc(openmm, source)
        chunk_frames = int(len(steps))
        if positions.shape[0] != topology.getNumAtoms():
            raise OpenMMWorkerError(f"OpenMM XTC atom count differs from topology: {source}")
        for index in range(chunk_frames):
            step = int(steps[index])
            if step != expected_step:
                raise OpenMMWorkerError(
                    f"OpenMM XTC step {step} differs from expected {expected_step}"
                )
            expected_time = step * request.timestep_fs / 1000.0
            if not math.isclose(float(times[index]), expected_time, rel_tol=1.0e-6, abs_tol=1.0e-4):
                raise OpenMMWorkerError(f"OpenMM XTC time differs at step {step}: {times[index]}")
            writer.writeModel(
                positions[:, :, index],
                periodicBoxVectors=tuple(
                    openmm.Vec3(*(float(value) for value in row)) for row in boxes[:, :, index]
                ),
            )
            expected_step += request.trajectory_interval_steps
            frame_count += 1
        source_records.append(
            {
                "path": str(source.resolve()),
                "sha256": _sha256_file(source),
                "frames": chunk_frames,
            }
        )
    del writer
    os.replace(temporary, artifacts["full_trajectory"])
    if frame_count != request.expected_frames:
        raise OpenMMWorkerError(
            f"OpenMM merged {frame_count} frames; expected {request.expected_frames}"
        )
    if _frame_count(openmm, artifacts["full_trajectory"]) != frame_count:
        raise OpenMMWorkerError("OpenMM merged trajectory failed frame-count verification")
    energy_rows = _merge_energy_series(
        request.energy_series,
        artifacts["energy_log"],
    )
    manifest = {
        "schema_version": "mmpbsa.openmm.trajectory-manifest.v1",
        "frame_count": frame_count,
        "first_step": 0,
        "last_step": (frame_count - 1) * request.trajectory_interval_steps,
        "trajectory_interval_steps": request.trajectory_interval_steps,
        "timestep_fs": request.timestep_fs,
        "chunks": source_records,
    }
    _atomic_json(artifacts["trajectory_manifest"], manifest)
    artifact_records = {
        role: _artifact_record(path) for role, path in artifacts.items() if role != "worker_receipt"
    }
    receipt = {
        "schema_version": OPENMM_MERGE_RECEIPT_SCHEMA,
        "status": "completed",
        "request_sha256": _sha256_json(request.as_dict()),
        "frame_count": frame_count,
        "energy_rows": energy_rows,
        "artifacts": artifact_records,
    }
    if request.input_format == OPENMM_AMBER_INPUT_FORMAT:
        receipt["openmm"] = {"version": str(openmm.__version__)}
        receipt["probe_sha256"] = request.expected_probe_sha256
        receipt["input"] = {
            "format": request.input_format,
            "topology_sha256": _sha256_file(request.topology),
            "coordinates_sha256": _sha256_file(request.coordinates),
        }
    _atomic_json(artifacts["worker_receipt"], receipt)
    return receipt


def _load_request(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenMMWorkerError(f"cannot read worker request: {path}") from exc
    if not isinstance(value, Mapping):
        raise OpenMMWorkerError("worker request root must be a mapping")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmpbsa-openmm-worker")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("probe")
    for operation in ("run", "merge", "preflight"):
        child = subparsers.add_parser(operation)
        child.add_argument("--request", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "probe":
            payload = probe_openmm()
        elif args.operation == "run":
            payload = run_md_worker(OpenMMMdRequest.from_mapping(_load_request(args.request)))
        elif args.operation == "preflight":
            payload = run_preflight_worker(
                OpenMMMdRequest.from_mapping(_load_request(args.request))
            )
        else:
            payload = run_merge_worker(OpenMMMergeRequest.from_mapping(_load_request(args.request)))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": "mmpbsa.openmm.worker-error.v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())

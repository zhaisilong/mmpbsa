"""GROMACS rendering for the engine-neutral MD phase-plan contract."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ._support import clean_output_dir
from .errors import BackendConfigurationError, BackendOutputError

MD_PHASE_PLAN_V2: Final = "mmpbsa.md-phase-plan.v2"
GROMACS_CANONICAL_PROFILE: Final = "gromacs_canonical@1"

_PHASE_IDS = ("em", "nvt", "npt", "production")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "md_engine_profile",
    "topology_role",
    "primary_phase",
    "replica_seed",
    "seed_derivation",
    "system",
    "phases",
}
_SYSTEM_KEYS = {
    "nonbonded_method",
    "nonbonded_cutoff_nm",
    "ewald_error_tolerance",
    "dispersion_correction",
    "constraints",
    "rigid_water",
    "constraint_tolerance",
    "center_of_mass_removal_interval_steps",
}


@dataclass(frozen=True, slots=True)
class RenderedGromacsPhasePlan:
    """A validated v2 plan lowered to the established GROMACS pipeline."""

    pipeline_science: Mapping[str, object]
    audit: Mapping[str, object]


def materialize_gromacs_public_artifacts(
    output_root: Path,
    native: Mapping[str, object],
) -> Mapping[str, Path]:
    """Create a role-only public boundary without exposing native paths."""

    public_root = clean_output_dir(Path(output_root) / "public")
    roles = {
        "full_trajectory": ("xtc", "full_trajectory.xtc"),
        "coordinates": ("gro", "coordinates.gro"),
        "state": ("cpt", "state.cpt"),
        "checkpoint": ("cpt", "checkpoint.cpt"),
        "energy_log": ("log", "energy.log"),
    }
    result: dict[str, Path] = {}
    for role, (native_role, filename) in roles.items():
        source = _artifact_path(native, native_role, "production")
        destination = public_root / filename
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        result[role] = destination
    return result


def validate_gromacs_phase_output(
    science: Mapping[str, object],
    phase_id: str,
    artifacts: Mapping[str, object],
) -> Mapping[str, object]:
    """Fail closed on v2 scientific gates not guaranteed by process exit."""

    phases = _mapping_by_phase(science)
    try:
        phase = phases[phase_id]
    except KeyError as exc:
        raise BackendConfigurationError(
            f"MD phase output names undeclared phase {phase_id!r}"
        ) from exc
    if phase_id == "em":
        minimization = _mapping(phase["minimization"], "MD phase em.minimization")
        threshold = _positive_float(
            minimization["maximum_force_threshold_kj_mol_nm"],
            "em.minimization.maximum_force_threshold_kj_mol_nm",
        )
        log = _artifact_path(artifacts, "log", phase_id)
        matches = re.findall(
            r"(?im)^\s*Maximum\s+force\s*=\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
            log.read_text(encoding="utf-8", errors="replace"),
        )
        if not matches:
            raise BackendOutputError(
                "GROMACS EM log does not report the final maximum force"
            )
        maximum_force = float(matches[-1])
        if not math.isfinite(maximum_force) or maximum_force > threshold:
            raise BackendOutputError(
                "GROMACS EM final maximum force "
                f"{maximum_force:g} exceeds acceptance threshold {threshold:g} kJ/mol/nm"
            )
        return {
            "maximum_force_kj_mol_nm": maximum_force,
            "acceptance_threshold_kj_mol_nm": threshold,
        }
    if phase_id != "production":
        return {"phase_transition": "validated"}

    output = _mapping(phase["output"], "MD phase production.output")
    timestep_fs = _positive_int(phase["timestep_fs"], "production.timestep_fs")
    interval_steps = _positive_int(
        output["trajectory_interval_steps"],
        "production.output.trajectory_interval_steps",
    )
    expected_frames = _positive_int(
        output["expected_trajectory_frames"],
        "production.output.expected_trajectory_frames",
    )
    interval_ps = interval_steps * timestep_fs / 1_000
    return _validate_xtc_timeline(
        _artifact_path(artifacts, "xtc", phase_id),
        expected_frames=expected_frames,
        interval_steps=interval_steps,
        interval_ps=interval_ps,
    )


def gromacs_operational_mapping(
    science: Mapping[str, object],
    production_mdrun_arguments: Sequence[str],
) -> Mapping[str, object]:
    """Audit resilience hints that do not map one-to-one across engines."""

    phases = _mapping_by_phase(science)
    production = phases.get("production")
    if production is None:
        raise BackendConfigurationError("MD phase plan has no production phase")
    output = _mapping(production["output"], "MD phase production.output")
    checkpoint_steps_by_phase = {
        phase_id: _positive_int(
            _mapping(phase["output"], f"MD phase {phase_id}.output")[
                "checkpoint_interval_steps"
            ],
            f"{phase_id}.output.checkpoint_interval_steps",
        )
        for phase_id, phase in phases.items()
        if "output" in phase
    }
    transaction_steps = _positive_int(
        output["transaction_chunk_steps"],
        "production.output.transaction_chunk_steps",
    )
    interval_minutes = 15.0
    source = "gromacs_default"
    arguments = tuple(production_mdrun_arguments)
    for index, argument in enumerate(arguments):
        raw: str | None = None
        if argument == "-cpt":
            if index + 1 >= len(arguments):
                raise BackendConfigurationError(
                    "production mdrun_arguments -cpt requires a value"
                )
            raw = arguments[index + 1]
        elif argument.startswith("-cpt="):
            raw = argument.split("=", 1)[1]
        if raw is not None:
            try:
                interval_minutes = float(raw)
            except ValueError as exc:
                raise BackendConfigurationError(
                    "production mdrun_arguments -cpt must be numeric minutes"
                ) from exc
            if not math.isfinite(interval_minutes) or interval_minutes < -1:
                raise BackendConfigurationError(
                    "production mdrun_arguments -cpt must be >= -1 minutes"
                )
            source = "runtime.mdrun_arguments"
    return {
        "classification": "operational_resilience_not_ensemble_science",
        "checkpoint": {
            "requested_interval_steps_by_phase": checkpoint_steps_by_phase,
            "mapping_status": "profile_approximation",
            "exact_step_schedule": False,
            "phase_end_checkpoint_guaranteed": True,
            "native_control": "gmx mdrun -cpt",
            "native_control_scope": "production",
            "native_interval_minutes": interval_minutes,
            "native_control_source": source,
            "reason": (
                "modern GROMACS has no nstcheckpoint; -cpt is a wall-clock "
                "interval and cannot be derived exactly from simulation steps"
            ),
        },
        "transaction_chunk": {
            "requested_steps": transaction_steps,
            "mapping_status": "openmm_journal_hint",
            "applied_by_gromacs": False,
            "gromacs_strategy": "single mdrun with native checkpoint append",
        },
    }


def is_md_phase_plan_v2(science: Mapping[str, object]) -> bool:
    return science.get("schema_version") == MD_PHASE_PLAN_V2


def render_gromacs_phase_plan(
    science: Mapping[str, object],
) -> RenderedGromacsPhasePlan:
    """Validate and lower one canonical phase plan without executing GROMACS."""

    public = {key for key in science if not key.startswith("_")}
    _require_exact_keys(public, _TOP_LEVEL_KEYS, "MD phase plan")
    _require_equal(science["schema_version"], MD_PHASE_PLAN_V2, "schema_version")
    _require_equal(
        science["md_engine_profile"],
        GROMACS_CANONICAL_PROFILE,
        "md_engine_profile",
    )
    _require_equal(science["primary_phase"], "production", "primary_phase")
    _require_equal(
        science["seed_derivation"],
        "sha256-phase-stream-v1",
        "seed_derivation",
    )
    topology_role = _text(science["topology_role"], "topology_role")
    _require_equal(
        science["replica_seed"], "{{replica.seed}}", "replica_seed"
    )
    replica = _mapping(science.get("_replica"), "MD phase plan _replica")
    replica_seed = replica.get("seed")
    if (
        isinstance(replica_seed, bool)
        or not isinstance(replica_seed, int)
        or not 1 <= replica_seed < 2**31
    ):
        raise BackendConfigurationError(
            "MD phase plan _replica.seed must be a non-zero 31-bit integer"
        )

    system = _system(science["system"])
    raw_phases = science["phases"]
    if not _sequence(raw_phases) or len(raw_phases) != len(_PHASE_IDS):
        raise BackendConfigurationError(
            "MD phase plan phases must contain em, nvt, npt, production"
        )
    phase_seeds = {
        phase_id: _phase_seed(replica_seed, phase_id) for phase_id in _PHASE_IDS
    }
    phases = tuple(
        _phase(
            raw,
            expected_id=phase_id,
            system=system,
            phase_seed=phase_seeds[phase_id],
        )
        for phase_id, raw in zip(_PHASE_IDS, raw_phases, strict=True)
    )

    production = _mapping(raw_phases[-1], "MD phase production")
    production_output = _mapping(
        production["output"], "MD phase production.output"
    )
    pipeline = {
        "topology_role": topology_role,
        "primary_stage": "production",
        "stages": phases,
    }
    audit = {
        "source_contract": MD_PHASE_PLAN_V2,
        "md_engine_profile": GROMACS_CANONICAL_PROFILE,
        "lowering": "gromacs-mdp.v1",
        "coordinate_format": "gromacs_gro",
        "state_format": "gromacs_checkpoint",
        "checkpoint_format": "gromacs_checkpoint",
        "energy_log_format": "gromacs_log",
        "trajectory_format": "xtc",
        "emit_initial_frame": True,
        "trajectory_interval_steps": production_output[
            "trajectory_interval_steps"
        ],
        "expected_trajectory_frames": production_output[
            "expected_trajectory_frames"
        ],
        "random_seeds": {
            phase_id: {
                "phase": seed,
                **(
                    {"velocities": _stream_seed(seed, "velocities")}
                    if phase_id == "nvt"
                    else {}
                ),
                **(
                    {"integrator": _stream_seed(seed, "integrator")}
                    if phase_id != "em"
                    else {}
                ),
            }
            for phase_id, seed in phase_seeds.items()
        },
        "constraint_mapping": {
            "status": "profile_approximation",
            "declared_tolerance": system["constraint_tolerance"],
            "native_controls": "LINCS order 4, one correction iteration",
            "direct_numeric_mapping": False,
        },
        "minimization_mapping": {
            "native_algorithm": "steepest_descent",
            "native_force_threshold_kj_mol_nm": _mapping(
                _mapping(raw_phases[0], "MD phase em")["minimization"],
                "MD phase em.minimization",
            )["maximum_force_threshold_kj_mol_nm"],
            "constraints": "none",
            "rigid_water": False,
            "topology_define": "-DFLEXIBLE",
        },
    }
    return RenderedGromacsPhasePlan(pipeline_science=pipeline, audit=audit)


def _system(value: object) -> Mapping[str, object]:
    result = _mapping(value, "MD phase plan system")
    _require_exact_keys(set(result), _SYSTEM_KEYS, "MD phase plan system")
    _require_equal(
        result["nonbonded_method"],
        "particle_mesh_ewald",
        "system.nonbonded_method",
    )
    _require_equal(result["constraints"], "hydrogen_bonds", "system.constraints")
    if not _boolean(result["rigid_water"], "system.rigid_water"):
        raise BackendConfigurationError(
            "gromacs_canonical@1 requires system.rigid_water=true"
        )
    _positive_float(result["nonbonded_cutoff_nm"], "system.nonbonded_cutoff_nm")
    _positive_float(result["ewald_error_tolerance"], "system.ewald_error_tolerance")
    _positive_float(result["constraint_tolerance"], "system.constraint_tolerance")
    _positive_int(
        result["center_of_mass_removal_interval_steps"],
        "system.center_of_mass_removal_interval_steps",
    )
    _boolean(result["dispersion_correction"], "system.dispersion_correction")
    return result


def _phase(
    value: object,
    *,
    expected_id: str,
    system: Mapping[str, object],
    phase_seed: int,
) -> Mapping[str, object]:
    phase = _mapping(value, f"MD phase {expected_id}")
    _require_equal(phase.get("phase_id"), expected_id, f"{expected_id}.phase_id")
    _require_equal(phase.get("kind"), expected_id, f"{expected_id}.kind")
    if expected_id == "em":
        _require_exact_keys(
            set(phase),
            {"phase_id", "kind", "input", "minimization"},
            "MD phase em",
        )
        _require_input(phase["input"], coordinates=("prepared", None, "md_coordinates"))
        minimization = _mapping(phase["minimization"], "MD phase em.minimization")
        _require_exact_keys(
            set(minimization),
            {
                "maximum_force_threshold_kj_mol_nm",
                "max_iterations",
            },
            "MD phase em.minimization",
        )
        threshold = _positive_float(
            minimization["maximum_force_threshold_kj_mol_nm"],
            "em.minimization.maximum_force_threshold_kj_mol_nm",
        )
        iterations = _positive_int(
            minimization["max_iterations"], "em.minimization.max_iterations"
        )
        values = {
            "title": "MMPBSA energy minimization",
            "integrator": "steep",
            "emtol": threshold,
            "emstep": 0.001,
            "nsteps": iterations,
            "define": "-DFLEXIBLE",
            **_nonbonded(system),
            "constraints": "none",
            "pbc": "xyz",
        }
        return _legacy_stage(
            expected_id,
            input_value={"kind": "prepared", "role": "md_coordinates"},
            mdp=_render_mdp(values),
            suffixes=("tpr", "log", "edr", "gro"),
        )

    required = {
        "phase_id",
        "kind",
        "input",
        "duration_ps",
        "steps",
        "timestep_fs",
        "temperature_kelvin",
        "initialize_velocities",
        "output",
    }
    if expected_id in {"npt", "production"}:
        required.add("pressure_bar")
    _require_exact_keys(set(phase), required, f"MD phase {expected_id}")
    previous = {"nvt": "em", "npt": "nvt", "production": "npt"}[expected_id]
    _require_input(
        phase["input"],
        coordinates=("phase", previous, "final_coordinates"),
        state=(None if expected_id == "nvt" else ("phase", previous, "portable_state")),
    )
    steps = _positive_int(phase["steps"], f"{expected_id}.steps")
    timestep_fs = _positive_int(
        phase["timestep_fs"], f"{expected_id}.timestep_fs"
    )
    duration_ps = _positive_float(
        phase["duration_ps"], f"{expected_id}.duration_ps"
    )
    if not math.isclose(duration_ps, steps * timestep_fs / 1_000, abs_tol=1.0e-12):
        raise BackendConfigurationError(
            f"{expected_id}.duration_ps is inconsistent with steps/timestep_fs"
        )
    temperature = _positive_float(
        phase["temperature_kelvin"], f"{expected_id}.temperature_kelvin"
    )
    initialize = _boolean(
        phase["initialize_velocities"], f"{expected_id}.initialize_velocities"
    )
    if initialize != (expected_id == "nvt"):
        raise BackendConfigurationError(
            f"{expected_id}.initialize_velocities violates the canonical phase transition"
        )
    output = _phase_output(phase["output"], phase_id=expected_id, steps=steps)

    values: dict[str, object] = {
        "title": f"MMPBSA {expected_id.upper()} {duration_ps:g} ps",
        "integrator": "md",
        "dt": timestep_fs / 1_000,
        "nsteps": steps,
        "tinit": 0,
        "init-step": 0,
        "continuation": "no" if expected_id == "nvt" else "yes",
        **_nonbonded(system),
        "constraints": "h-bonds",
        "constraint-algorithm": "lincs",
        "lincs-iter": 1,
        "lincs-order": 4,
        "tcoupl": "V-rescale",
        "tc-grps": "System",
        "tau-t": 1.0,
        "ref-t": temperature,
        "comm-mode": "Linear",
        "nstcomm": system["center_of_mass_removal_interval_steps"],
        "pbc": "xyz",
        "nstxout": 0,
        "nstvout": 0,
        "nstfout": 0,
        "nstcalcenergy": output["report_interval_steps"],
        "nstenergy": output["report_interval_steps"],
        "nstlog": output["report_interval_steps"],
        "nstxout-compressed": output.get("trajectory_interval_steps", 0),
    }
    if expected_id == "nvt":
        values.update(
            {
                "pcoupl": "no",
                "gen-vel": "yes",
                "gen-temp": temperature,
                "gen-seed": _stream_seed(phase_seed, "velocities"),
            }
        )
    else:
        values.update(
            {
                "pcoupl": "C-rescale",
                "pcoupltype": "isotropic",
                "tau-p": 5.0,
                "ref-p": _positive_float(
                    phase["pressure_bar"], f"{expected_id}.pressure_bar"
                ),
                "compressibility": 4.5e-5,
                "gen-vel": "no",
            }
        )
    values["ld-seed"] = _stream_seed(phase_seed, "integrator")
    if expected_id == "production":
        values["compressed-x-precision"] = 1_000

    stage_input = {
        "kind": "stage",
        "stage_id": previous,
        "artifact": "gro",
    }
    state_input = (
        None
        if expected_id == "nvt"
        else {"kind": "stage", "stage_id": previous, "artifact": "cpt"}
    )
    suffixes = ["tpr", "log", "edr", "gro", "cpt"]
    if expected_id == "production":
        suffixes.append("xtc")
    return _legacy_stage(
        expected_id,
        input_value=stage_input,
        state_input=state_input,
        mdp=_render_mdp(values),
        suffixes=tuple(suffixes),
        primary=expected_id == "production",
    )


def _phase_output(value: object, *, phase_id: str, steps: int) -> Mapping[str, int]:
    output = _mapping(value, f"MD phase {phase_id}.output")
    required = {"report_interval_steps", "checkpoint_interval_steps"}
    if phase_id == "production":
        required |= {
            "trajectory_interval_steps",
            "transaction_chunk_steps",
            "emit_initial_frame",
            "expected_trajectory_frames",
        }
    _require_exact_keys(set(output), required, f"MD phase {phase_id}.output")
    parsed = {
        key: _positive_int(raw, f"{phase_id}.output.{key}")
        for key, raw in output.items()
        if key != "emit_initial_frame"
    }
    for key in ("report_interval_steps", "checkpoint_interval_steps"):
        if parsed[key] > steps:
            raise BackendConfigurationError(f"{phase_id}.output.{key} exceeds phase steps")
    if phase_id == "production":
        if not _boolean(output["emit_initial_frame"], "production.output.emit_initial_frame"):
            raise BackendConfigurationError(
                "gromacs_canonical@1 requires production.output.emit_initial_frame=true"
            )
        interval = parsed["trajectory_interval_steps"]
        chunk = parsed["transaction_chunk_steps"]
        if steps % interval or steps % chunk:
            raise BackendConfigurationError(
                "production output intervals must divide production steps exactly"
            )
        expected = steps // interval + 1
        if parsed["expected_trajectory_frames"] != expected:
            raise BackendConfigurationError(
                "production expected_trajectory_frames does not include the initial frame"
            )
    return parsed


def _phase_seed(master: int, phase_id: str) -> int:
    digest = hashlib.sha256(f"{master}:{phase_id}:phase".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


def _stream_seed(phase_seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{phase_seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


def _mapping_by_phase(
    science: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    raw = science.get("phases")
    if not _sequence(raw):
        raise BackendConfigurationError("MD phase plan phases must be a sequence")
    result: dict[str, Mapping[str, object]] = {}
    for item in raw:
        phase = _mapping(item, "MD phase")
        phase_id = _text(phase.get("phase_id"), "MD phase phase_id")
        if phase_id in result:
            raise BackendConfigurationError(f"duplicate MD phase_id: {phase_id}")
        result[phase_id] = phase
    return result


def _artifact_path(
    artifacts: Mapping[str, object], role: str, phase_id: str
) -> Path:
    try:
        path = Path(artifacts[role]).resolve()  # type: ignore[arg-type]
    except (KeyError, TypeError) as exc:
        raise BackendOutputError(
            f"GROMACS {phase_id} output is missing artifact {role!r}"
        ) from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise BackendOutputError(
            f"GROMACS {phase_id} artifact is missing or empty: {path}"
        )
    return path


def _validate_xtc_timeline(
    trajectory: Path,
    *,
    expected_frames: int,
    interval_steps: int,
    interval_ps: float,
) -> Mapping[str, object]:
    try:
        from mdtraj.formats import XTCTrajectoryFile
    except ModuleNotFoundError as exc:
        raise BackendConfigurationError(
            "GROMACS phase-plan v2 timeline validation requires MDTraj"
        ) from exc
    frame_count = 0
    first_time: float | None = None
    last_time: float | None = None
    try:
        with XTCTrajectoryFile(str(trajectory), "r") as stream:
            while True:
                xyz, times, steps, _boxes = stream.read(n_frames=100)
                count = int(xyz.shape[0])
                if count == 0:
                    break
                for time, step in zip(times, steps, strict=True):
                    expected_time = frame_count * interval_ps
                    expected_step = frame_count * interval_steps
                    observed_time = float(time)
                    if not math.isclose(
                        observed_time,
                        expected_time,
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-3,
                    ) or int(step) != expected_step:
                        raise BackendOutputError(
                            "GROMACS production XTC does not use the declared t=0 timeline "
                            f"at frame {frame_count + 1}: time={observed_time:g} ps, "
                            f"step={int(step)}"
                        )
                    first_time = observed_time if first_time is None else first_time
                    last_time = observed_time
                    frame_count += 1
    except (BackendConfigurationError, BackendOutputError):
        raise
    except Exception as exc:
        raise BackendOutputError(
            f"cannot inspect GROMACS production XTC timeline: {trajectory}"
        ) from exc
    if frame_count != expected_frames:
        raise BackendOutputError(
            f"GROMACS production XTC has {frame_count} frames; expected {expected_frames}"
        )
    return {
        "frame_count": frame_count,
        "first_time_ps": first_time,
        "last_time_ps": last_time,
        "interval_ps": interval_ps,
        "initial_frame_included": True,
    }


def _require_input(
    value: object,
    *,
    coordinates: tuple[str, str | None, str],
    state: tuple[str, str | None, str] | None = None,
) -> None:
    inputs = _mapping(value, "MD phase input")
    expected_keys = {"coordinates"} | ({"state"} if state else set())
    _require_exact_keys(set(inputs), expected_keys, "MD phase input")
    _require_reference(inputs["coordinates"], coordinates, "coordinates")
    if state is not None:
        _require_reference(inputs["state"], state, "state")


def _require_reference(
    value: object,
    expected: tuple[str, str | None, str],
    label: str,
) -> None:
    reference = _mapping(value, f"MD phase input {label}")
    kind, phase_id, artifact = expected
    expected_keys = {"kind", "role"} if kind == "prepared" else {
        "kind",
        "phase_id",
        "artifact",
    }
    _require_exact_keys(set(reference), expected_keys, f"MD phase input {label}")
    observed = (
        reference.get("kind"),
        reference.get("phase_id"),
        reference.get("role") if kind == "prepared" else reference.get("artifact"),
    )
    if observed != (kind, phase_id, artifact):
        raise BackendConfigurationError(
            f"MD phase input {label} does not match the canonical phase transition"
        )


def _nonbonded(system: Mapping[str, object]) -> Mapping[str, object]:
    cutoff = system["nonbonded_cutoff_nm"]
    return {
        "cutoff-scheme": "Verlet",
        "nstlist": 20,
        "rlist": cutoff,
        "coulombtype": "PME",
        "rcoulomb": cutoff,
        "ewald-rtol": system["ewald_error_tolerance"],
        "vdwtype": "Cut-off",
        "rvdw": cutoff,
        "DispCorr": (
            "EnerPres" if system["dispersion_correction"] else "no"
        ),
    }


def _legacy_stage(
    stage_id: str,
    *,
    input_value: Mapping[str, object],
    mdp: str,
    suffixes: tuple[str, ...],
    state_input: Mapping[str, object] | None = None,
    primary: bool = False,
) -> Mapping[str, object]:
    return {
        "stage_id": stage_id,
        "mdp_text": mdp,
        "coordinates": input_value,
        **({"state_input": state_input} if state_input is not None else {}),
        "expected_suffixes": list(suffixes),
        "checkpoint": "cpt" in suffixes,
        **({"primary": True} if primary else {}),
    }


def _render_mdp(values: Mapping[str, object]) -> str:
    return "".join(f"{key:<25} = {_scalar(value)}\n" for key, value in values.items())


def _scalar(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise BackendConfigurationError(f"{label} must be a mapping with string keys")
    return value


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise BackendConfigurationError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BackendConfigurationError(f"{label} must be a positive integer")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendConfigurationError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise BackendConfigurationError(f"{label} must be finite and positive")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise BackendConfigurationError(f"{label} must be boolean")
    return value


def _require_equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise BackendConfigurationError(f"{label} must be {expected!r}")


def _require_exact_keys(observed: set[str], expected: set[str], label: str) -> None:
    missing = expected - observed
    unknown = observed - expected
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise BackendConfigurationError(f"{label} is invalid: " + "; ".join(details))

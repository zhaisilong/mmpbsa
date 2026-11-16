"""Process-isolated OpenMM MD adapter and strict worker wire contracts.

This module deliberately does not import :mod:`openmm`.  Provider discovery,
configuration validation, and dry runs therefore remain usable in an
environment where the optional OpenMM runtime is not installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ._support import (
    clean_output_dir,
    execute_commands,
    materialize_inputs,
    require_artifacts,
    require_file,
)
from .api import (
    BackendReceipt,
    CommandRunner,
    CommandSpec,
    GeneratedInput,
    ReceiptState,
)
from .errors import BackendConfigurationError, BackendOutputError
from .openmm_receipt import validate_md_receipt, validate_merge_receipt

OPENMM_REQUEST_SCHEMA = "mmpbsa.openmm.md-request.v1"
OPENMM_MERGE_SCHEMA = "mmpbsa.openmm.trajectory-merge-request.v1"
OPENMM_RECEIPT_SCHEMA = "mmpbsa.openmm.worker-receipt.v1"
OPENMM_MERGE_RECEIPT_SCHEMA = "mmpbsa.openmm.merge-receipt.v1"
OPENMM_PROBE_SCHEMA = "mmpbsa.openmm.probe.v1"
OPENMM_AMBER_INPUT_FORMAT = "mmpbsa.openmm.input.amber.v1"
OPENMM_GROMACS_INPUT_FORMAT = "mmpbsa.openmm.input.gromacs.v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PLATFORMS = frozenset({"CUDA", "CPU", "Reference"})
_PRECISIONS = frozenset({"single", "mixed", "double"})
_PHASE_KINDS = frozenset({"em", "nvt", "npt", "production"})
_MAX_SEED = 2**31 - 1


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendConfigurationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise BackendConfigurationError(f"{label} must be finite and positive")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BackendConfigurationError(f"{label} must be a positive integer")
    return value


def _optional_path(value: object, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or "\x00" in value:
        raise BackendConfigurationError(f"{label} must be a non-empty path string")
    return Path(value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BackendConfigurationError(f"{label} must be a string-keyed mapping")
    return value


def _strict_fields(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    label: str,
) -> None:
    missing = required.difference(value)
    unknown = set(value).difference(required | optional)
    details = []
    if missing:
        details.append("missing " + ", ".join(sorted(missing)))
    if unknown:
        details.append("unknown " + ", ".join(sorted(unknown)))
    if details:
        raise BackendConfigurationError(f"invalid {label}: " + "; ".join(details))


def _validate_input_identity(path: Path, expected: str | None, label: str) -> None:
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise BackendConfigurationError(f"OpenMM Amber {label} requires a SHA-256 digest")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise BackendConfigurationError(f"OpenMM Amber {label} differs from its sealed digest")


@dataclass(frozen=True, slots=True)
class OpenMMPlatformSpec:
    """One explicitly selected OpenMM platform and its allowed properties."""

    name: str
    device_index: str | None = None
    precision: str | None = None
    deterministic_forces: bool = True
    threads: int = 1

    def validate(self) -> None:
        if self.name not in _PLATFORMS:
            raise BackendConfigurationError(
                "OpenMM platform must be one of CUDA, CPU, or Reference"
            )
        _positive_integer(self.threads, "OpenMM platform threads")
        if not isinstance(self.deterministic_forces, bool):
            raise BackendConfigurationError("OpenMM deterministic_forces must be boolean")
        if self.name == "CUDA":
            if (
                not isinstance(self.device_index, str)
                or not self.device_index.strip()
                or self.device_index != self.device_index.strip()
                or "," in self.device_index
                or "\x00" in self.device_index
                or any(character.isspace() for character in self.device_index)
            ):
                raise BackendConfigurationError(
                    "OpenMM CUDA requires exactly one explicit leased device_index"
                )
            if self.precision not in _PRECISIONS:
                raise BackendConfigurationError(
                    "OpenMM CUDA precision must be single, mixed, or double"
                )
            if not self.deterministic_forces:
                raise BackendConfigurationError("OpenMM CUDA requires deterministic_forces=true")
        elif self.device_index is not None or self.precision is not None:
            raise BackendConfigurationError(
                f"OpenMM {self.name} does not accept CUDA device/precision fields"
            )
        if self.name == "Reference" and self.threads != 1:
            raise BackendConfigurationError("OpenMM Reference requires threads=1")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "device_index": self.device_index,
            "precision": self.precision,
            "deterministic_forces": self.deterministic_forces,
            "threads": self.threads,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> OpenMMPlatformSpec:
        _strict_fields(
            value,
            required=frozenset({"name"}),
            optional=frozenset({"device_index", "precision", "deterministic_forces", "threads"}),
            label="OpenMM platform",
        )
        result = cls(
            name=str(value["name"]),
            device_index=(
                None if value.get("device_index") is None else str(value["device_index"])
            ),
            precision=(None if value.get("precision") is None else str(value["precision"])),
            deterministic_forces=value.get("deterministic_forces", True),  # type: ignore[arg-type]
            threads=value.get("threads", 1),  # type: ignore[arg-type]
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class OpenMMPhaseSpec:
    """Flattened phase contract consumed by one worker invocation."""

    phase_id: str
    kind: str
    steps: int = 0
    timestep_fs: float = 2.0
    temperature_kelvin: float | None = None
    pressure_bar: float | None = None
    initialize_velocities: bool = False
    report_interval_steps: int = 1
    trajectory_interval_steps: int = 1
    checkpoint_interval_steps: int = 1
    minimization_tolerance_kj_mol_nm: float | None = None
    maximum_force_threshold_kj_mol_nm: float | None = None
    max_iterations: int | None = None

    def validate(self) -> None:
        if not _IDENTIFIER.fullmatch(self.phase_id):
            raise BackendConfigurationError(f"invalid OpenMM phase_id: {self.phase_id!r}")
        if self.kind not in _PHASE_KINDS:
            raise BackendConfigurationError(f"invalid OpenMM phase kind: {self.kind!r}")
        if not isinstance(self.initialize_velocities, bool):
            raise BackendConfigurationError("initialize_velocities must be boolean")
        if self.kind == "em":
            if self.steps != 0:
                raise BackendConfigurationError("OpenMM EM steps must be zero")
            if self.temperature_kelvin is not None or self.pressure_bar is not None:
                raise BackendConfigurationError("OpenMM EM cannot declare temperature/pressure")
            if self.initialize_velocities:
                raise BackendConfigurationError("OpenMM EM cannot initialize velocities")
            _finite_positive(
                self.minimization_tolerance_kj_mol_nm,
                "OpenMM minimization tolerance",
            )
            _finite_positive(
                self.maximum_force_threshold_kj_mol_nm,
                "OpenMM maximum force threshold",
            )
            _positive_integer(self.max_iterations, "OpenMM max_iterations")
            return

        _positive_integer(self.steps, "OpenMM phase steps")
        timestep = _finite_positive(self.timestep_fs, "OpenMM timestep_fs")
        if not math.isclose(timestep, 2.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise BackendConfigurationError("openmm_8_5@1 requires a 2 fs timestep")
        _finite_positive(self.temperature_kelvin, "OpenMM temperature_kelvin")
        if self.kind == "nvt":
            if self.pressure_bar is not None:
                raise BackendConfigurationError("OpenMM NVT cannot declare pressure")
        else:
            _finite_positive(self.pressure_bar, "OpenMM pressure_bar")
        for label, interval in (
            ("report_interval_steps", self.report_interval_steps),
            ("trajectory_interval_steps", self.trajectory_interval_steps),
            ("checkpoint_interval_steps", self.checkpoint_interval_steps),
        ):
            _positive_integer(interval, f"OpenMM {label}")
        if self.steps % self.report_interval_steps:
            raise BackendConfigurationError(
                "OpenMM phase steps must align to report_interval_steps"
            )
        if self.steps % self.trajectory_interval_steps:
            raise BackendConfigurationError(
                "OpenMM phase steps must align to trajectory_interval_steps"
            )
        if any(
            value is not None
            for value in (
                self.minimization_tolerance_kj_mol_nm,
                self.maximum_force_threshold_kj_mol_nm,
                self.max_iterations,
            )
        ):
            raise BackendConfigurationError(
                "dynamic OpenMM phases cannot declare minimization controls"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "kind": self.kind,
            "steps": self.steps,
            "timestep_fs": self.timestep_fs,
            "temperature_kelvin": self.temperature_kelvin,
            "pressure_bar": self.pressure_bar,
            "initialize_velocities": self.initialize_velocities,
            "report_interval_steps": self.report_interval_steps,
            "trajectory_interval_steps": self.trajectory_interval_steps,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "minimization_tolerance_kj_mol_nm": (self.minimization_tolerance_kj_mol_nm),
            "maximum_force_threshold_kj_mol_nm": (self.maximum_force_threshold_kj_mol_nm),
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> OpenMMPhaseSpec:
        required = frozenset({"phase_id", "kind"})
        optional = frozenset(
            {
                "steps",
                "timestep_fs",
                "temperature_kelvin",
                "pressure_bar",
                "initialize_velocities",
                "report_interval_steps",
                "trajectory_interval_steps",
                "checkpoint_interval_steps",
                "minimization_tolerance_kj_mol_nm",
                "maximum_force_threshold_kj_mol_nm",
                "max_iterations",
            }
        )
        _strict_fields(value, required=required, optional=optional, label="OpenMM phase")
        result = cls(
            phase_id=str(value["phase_id"]),
            kind=str(value["kind"]),
            steps=value.get("steps", 0),  # type: ignore[arg-type]
            timestep_fs=value.get("timestep_fs", 2.0),  # type: ignore[arg-type]
            temperature_kelvin=value.get("temperature_kelvin"),  # type: ignore[arg-type]
            pressure_bar=value.get("pressure_bar"),  # type: ignore[arg-type]
            initialize_velocities=value.get("initialize_velocities", False),  # type: ignore[arg-type]
            report_interval_steps=value.get("report_interval_steps", 1),  # type: ignore[arg-type]
            trajectory_interval_steps=value.get("trajectory_interval_steps", 1),  # type: ignore[arg-type]
            checkpoint_interval_steps=value.get("checkpoint_interval_steps", 1),  # type: ignore[arg-type]
            minimization_tolerance_kj_mol_nm=value.get(  # type: ignore[arg-type]
                "minimization_tolerance_kj_mol_nm"
            ),
            maximum_force_threshold_kj_mol_nm=value.get(  # type: ignore[arg-type]
                "maximum_force_threshold_kj_mol_nm"
            ),
            max_iterations=value.get("max_iterations"),  # type: ignore[arg-type]
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class OpenMMMdRequest:
    topology: Path
    coordinates: Path
    output_dir: Path
    phase: OpenMMPhaseSpec
    platform: OpenMMPlatformSpec
    seed: int
    state_input: Path | None = None
    checkpoint_input: Path | None = None
    checkpoint_provenance: Mapping[str, object] | None = None
    chunk_id: str | None = None
    emit_initial_frame: bool = True
    input_format: str = OPENMM_GROMACS_INPUT_FORMAT
    topology_sha256: str | None = None
    coordinates_sha256: str | None = None
    expected_openmm_version: str | None = None
    expected_probe_sha256: str | None = None

    def validate(self) -> None:
        require_file(self.topology, "OpenMM topology")
        require_file(self.coordinates, "OpenMM coordinates")
        if self.input_format == OPENMM_AMBER_INPUT_FORMAT:
            if self.topology.suffix.lower() not in {".prmtop", ".parm7"}:
                raise BackendConfigurationError("OpenMM Amber topology must be prmtop/parm7")
            suffixes = {".inpcrd", ".rst7", ".crd"} if self.phase.kind == "em" else {".pdb"}
            if self.coordinates.suffix.lower() not in suffixes:
                raise BackendConfigurationError(
                    "OpenMM Amber phase has an invalid coordinate format"
                )
            _validate_input_identity(self.topology, self.topology_sha256, "topology")
            _validate_input_identity(self.coordinates, self.coordinates_sha256, "coordinates")
            if (
                not isinstance(self.expected_openmm_version, str)
                or not self.expected_openmm_version
            ):
                raise BackendConfigurationError(
                    "OpenMM Amber request requires exact probed version"
                )
            if (
                not isinstance(self.expected_probe_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.expected_probe_sha256) is None
            ):
                raise BackendConfigurationError("OpenMM Amber request requires exact probe digest")
        elif self.input_format == OPENMM_GROMACS_INPUT_FORMAT:
            if self.coordinates.suffix.lower() not in {".gro", ".pdb"}:
                raise BackendConfigurationError("OpenMM coordinates must be GRO or PDB")
            if any(
                value is not None
                for value in (
                    self.topology_sha256,
                    self.coordinates_sha256,
                    self.expected_openmm_version,
                    self.expected_probe_sha256,
                )
            ):
                raise BackendConfigurationError(
                    "legacy OpenMM request cannot declare Amber input identity"
                )
        else:
            raise BackendConfigurationError("unsupported OpenMM input format")
        self.phase.validate()
        self.platform.validate()
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise BackendConfigurationError("OpenMM seed must be an integer")
        if not 1 <= self.seed <= _MAX_SEED:
            raise BackendConfigurationError(f"OpenMM seed must be between 1 and {_MAX_SEED}")
        if self.state_input is not None:
            require_file(self.state_input, "OpenMM portable state input")
        if self.checkpoint_input is not None:
            require_file(self.checkpoint_input, "OpenMM checkpoint input")
        if self.state_input is not None and self.checkpoint_input is not None:
            raise BackendConfigurationError("OpenMM request cannot load both state and checkpoint")
        if self.checkpoint_input is not None:
            if self.checkpoint_provenance is None:
                raise BackendConfigurationError(
                    "OpenMM checkpoint resume requires checkpoint_provenance"
                )
            if self.emit_initial_frame:
                raise BackendConfigurationError(
                    "OpenMM checkpoint continuation cannot re-emit its initial frame"
                )
        elif self.checkpoint_provenance is not None:
            raise BackendConfigurationError("checkpoint_provenance requires checkpoint_input")
        if not isinstance(self.emit_initial_frame, bool):
            raise BackendConfigurationError("emit_initial_frame must be boolean")
        if self.chunk_id is not None and not _IDENTIFIER.fullmatch(self.chunk_id):
            raise BackendConfigurationError(f"invalid OpenMM chunk_id: {self.chunk_id!r}")
        if self.phase.kind != "production" and self.chunk_id is not None:
            raise BackendConfigurationError("OpenMM chunk_id is only valid for production")

    @property
    def prefix(self) -> str:
        return (
            self.phase.phase_id
            if self.chunk_id is None
            else f"{self.phase.phase_id}.{self.chunk_id}"
        )

    def as_dict(self) -> dict[str, object]:
        result = {
            "schema_version": OPENMM_REQUEST_SCHEMA,
            "topology": str(Path(self.topology).resolve()),
            "coordinates": str(Path(self.coordinates).resolve()),
            "output_dir": str(Path(self.output_dir).resolve()),
            "phase": self.phase.as_dict(),
            "platform": self.platform.as_dict(),
            "seed": self.seed,
            "state_input": (
                None if self.state_input is None else str(Path(self.state_input).resolve())
            ),
            "checkpoint_input": (
                None
                if self.checkpoint_input is None
                else str(Path(self.checkpoint_input).resolve())
            ),
            "checkpoint_provenance": (
                None if self.checkpoint_provenance is None else dict(self.checkpoint_provenance)
            ),
            "chunk_id": self.chunk_id,
            "emit_initial_frame": self.emit_initial_frame,
        }
        if self.input_format == OPENMM_AMBER_INPUT_FORMAT:
            result.update(
                {
                    "input_format": self.input_format,
                    "topology_sha256": self.topology_sha256,
                    "coordinates_sha256": self.coordinates_sha256,
                    "expected_openmm_version": self.expected_openmm_version,
                    "expected_probe_sha256": self.expected_probe_sha256,
                }
            )
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> OpenMMMdRequest:
        required = frozenset(
            {"schema_version", "topology", "coordinates", "output_dir", "phase", "platform", "seed"}
        )
        optional = frozenset(
            {
                "state_input",
                "checkpoint_input",
                "checkpoint_provenance",
                "chunk_id",
                "emit_initial_frame",
                "input_format",
                "topology_sha256",
                "coordinates_sha256",
                "expected_openmm_version",
                "expected_probe_sha256",
            }
        )
        _strict_fields(value, required=required, optional=optional, label="OpenMM request")
        if value["schema_version"] != OPENMM_REQUEST_SCHEMA:
            raise BackendConfigurationError("unsupported OpenMM request schema")
        checkpoint_provenance = value.get("checkpoint_provenance")
        result = cls(
            topology=Path(str(value["topology"])),
            coordinates=Path(str(value["coordinates"])),
            output_dir=Path(str(value["output_dir"])),
            phase=OpenMMPhaseSpec.from_mapping(_mapping(value["phase"], "phase")),
            platform=OpenMMPlatformSpec.from_mapping(_mapping(value["platform"], "platform")),
            seed=value["seed"],  # type: ignore[arg-type]
            state_input=_optional_path(value.get("state_input"), "state_input"),
            checkpoint_input=_optional_path(value.get("checkpoint_input"), "checkpoint_input"),
            checkpoint_provenance=(
                None
                if checkpoint_provenance is None
                else _mapping(checkpoint_provenance, "checkpoint_provenance")
            ),
            chunk_id=(None if value.get("chunk_id") is None else str(value["chunk_id"])),
            emit_initial_frame=value.get("emit_initial_frame", True),  # type: ignore[arg-type]
            input_format=str(value.get("input_format", OPENMM_GROMACS_INPUT_FORMAT)),
            topology_sha256=value.get("topology_sha256"),  # type: ignore[arg-type]
            coordinates_sha256=value.get("coordinates_sha256"),  # type: ignore[arg-type]
            expected_openmm_version=value.get("expected_openmm_version"),  # type: ignore[arg-type]
            expected_probe_sha256=value.get("expected_probe_sha256"),  # type: ignore[arg-type]
        )
        result.validate()
        return result


def openmm_artifact_paths(request: OpenMMMdRequest) -> dict[str, Path]:
    root = Path(request.output_dir).resolve()
    prefix = request.prefix
    return {
        "trajectory": root / f"{prefix}.xtc",
        "energy_series": root / f"{prefix}.csv",
        "portable_state": root / f"{prefix}.state.xml",
        "checkpoint": root / f"{prefix}.checkpoint.chk",
        "final_coordinates": root / f"{prefix}.final.pdb",
        "worker_receipt": root / f"{prefix}.receipt.json",
    }


@dataclass(frozen=True, slots=True)
class OpenMMMergeRequest:
    trajectories: tuple[Path, ...]
    energy_series: tuple[Path, ...]
    topology: Path
    coordinates: Path
    output_dir: Path
    expected_frames: int
    trajectory_interval_steps: int
    timestep_fs: float
    input_format: str = OPENMM_GROMACS_INPUT_FORMAT
    topology_sha256: str | None = None
    coordinates_sha256: str | None = None
    expected_openmm_version: str | None = None
    expected_probe_sha256: str | None = None

    def validate(self) -> None:
        if not self.trajectories:
            raise BackendConfigurationError("OpenMM merge trajectories cannot be empty")
        if len(self.trajectories) != len(self.energy_series):
            raise BackendConfigurationError("OpenMM merge trajectory/energy chunk counts differ")
        for path in self.trajectories:
            require_file(path, "OpenMM trajectory chunk")
        for path in self.energy_series:
            require_file(path, "OpenMM energy chunk")
        require_file(self.topology, "OpenMM merge topology")
        require_file(self.coordinates, "OpenMM merge coordinates")
        if self.input_format == OPENMM_AMBER_INPUT_FORMAT:
            if self.topology.suffix.lower() not in {
                ".prmtop",
                ".parm7",
            } or self.coordinates.suffix.lower() not in {".inpcrd", ".rst7", ".crd"}:
                raise BackendConfigurationError(
                    "OpenMM Amber merge requires prmtop/parm7 and inpcrd/rst7/crd"
                )
            _validate_input_identity(self.topology, self.topology_sha256, "merge topology")
            _validate_input_identity(self.coordinates, self.coordinates_sha256, "merge coordinates")
            if (
                not isinstance(self.expected_openmm_version, str)
                or not self.expected_openmm_version
            ):
                raise BackendConfigurationError("OpenMM Amber merge requires exact probed version")
            if (
                not isinstance(self.expected_probe_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.expected_probe_sha256) is None
            ):
                raise BackendConfigurationError("OpenMM Amber merge requires exact probe digest")
        elif self.input_format != OPENMM_GROMACS_INPUT_FORMAT:
            raise BackendConfigurationError("unsupported OpenMM merge input format")
        _positive_integer(self.expected_frames, "OpenMM expected_frames")
        _positive_integer(
            self.trajectory_interval_steps,
            "OpenMM trajectory_interval_steps",
        )
        timestep = _finite_positive(self.timestep_fs, "OpenMM merge timestep_fs")
        if not math.isclose(timestep, 2.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise BackendConfigurationError("OpenMM merge requires a 2 fs timestep")

    def as_dict(self) -> dict[str, object]:
        result = {
            "schema_version": OPENMM_MERGE_SCHEMA,
            "trajectories": [str(path.resolve()) for path in self.trajectories],
            "energy_series": [str(path.resolve()) for path in self.energy_series],
            "topology": str(self.topology.resolve()),
            "coordinates": str(self.coordinates.resolve()),
            "output_dir": str(self.output_dir.resolve()),
            "expected_frames": self.expected_frames,
            "trajectory_interval_steps": self.trajectory_interval_steps,
            "timestep_fs": self.timestep_fs,
        }
        if self.input_format == OPENMM_AMBER_INPUT_FORMAT:
            result.update(
                {
                    "input_format": self.input_format,
                    "topology_sha256": self.topology_sha256,
                    "coordinates_sha256": self.coordinates_sha256,
                    "expected_openmm_version": self.expected_openmm_version,
                    "expected_probe_sha256": self.expected_probe_sha256,
                }
            )
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> OpenMMMergeRequest:
        required = frozenset(
            {
                "schema_version",
                "trajectories",
                "energy_series",
                "topology",
                "coordinates",
                "output_dir",
                "expected_frames",
                "trajectory_interval_steps",
                "timestep_fs",
            }
        )
        _strict_fields(
            value,
            required=required,
            optional=frozenset(
                {
                    "input_format",
                    "topology_sha256",
                    "coordinates_sha256",
                    "expected_openmm_version",
                    "expected_probe_sha256",
                }
            ),
            label="merge request",
        )
        if value["schema_version"] != OPENMM_MERGE_SCHEMA:
            raise BackendConfigurationError("unsupported OpenMM merge schema")
        trajectories = value["trajectories"]
        energies = value["energy_series"]
        if (
            isinstance(trajectories, (str, bytes))
            or not isinstance(trajectories, Sequence)
            or isinstance(energies, (str, bytes))
            or not isinstance(energies, Sequence)
        ):
            raise BackendConfigurationError("OpenMM merge chunks must be sequences")
        result = cls(
            trajectories=tuple(Path(str(path)) for path in trajectories),
            energy_series=tuple(Path(str(path)) for path in energies),
            topology=Path(str(value["topology"])),
            coordinates=Path(str(value["coordinates"])),
            output_dir=Path(str(value["output_dir"])),
            expected_frames=value["expected_frames"],  # type: ignore[arg-type]
            trajectory_interval_steps=value["trajectory_interval_steps"],  # type: ignore[arg-type]
            timestep_fs=value["timestep_fs"],  # type: ignore[arg-type]
            input_format=str(value.get("input_format", OPENMM_GROMACS_INPUT_FORMAT)),
            topology_sha256=value.get("topology_sha256"),  # type: ignore[arg-type]
            coordinates_sha256=value.get("coordinates_sha256"),  # type: ignore[arg-type]
            expected_openmm_version=value.get("expected_openmm_version"),  # type: ignore[arg-type]
            expected_probe_sha256=value.get("expected_probe_sha256"),  # type: ignore[arg-type]
        )
        result.validate()
        return result


def openmm_merge_artifact_paths(request: OpenMMMergeRequest) -> dict[str, Path]:
    root = request.output_dir.resolve()
    return {
        "full_trajectory": root / "production.xtc",
        "energy_log": root / "production.csv",
        "trajectory_manifest": root / "production.manifest.json",
        "worker_receipt": root / "production.merge.receipt.json",
    }


@dataclass(frozen=True, slots=True)
class OpenMMWorkerConfig:
    python_executable: str = sys.executable
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None

    def validate(self) -> None:
        if (
            not isinstance(self.python_executable, str)
            or not self.python_executable.strip()
            or "\x00" in self.python_executable
        ):
            raise BackendConfigurationError("OpenMM Python executable cannot be empty")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "=" in key
            or "\x00" in key
            or "\x00" in value
            for key, value in self.environment.items()
        ):
            raise BackendConfigurationError("OpenMM worker environment is invalid")
        if self.timeout_seconds is not None:
            _finite_positive(self.timeout_seconds, "OpenMM worker timeout_seconds")


@dataclass(frozen=True, slots=True)
class OpenMMIdentity:
    version: str
    platforms: tuple[str, ...]
    raw: Mapping[str, object]

    @property
    def major(self) -> int:
        try:
            return int(self.version.split(".", 1)[0])
        except ValueError as exc:
            raise BackendOutputError(
                f"OpenMM worker returned an invalid version: {self.version!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class OpenMMMdBackend:
    worker: OpenMMWorkerConfig = field(default_factory=OpenMMWorkerConfig)
    runner: CommandRunner | None = None

    def probe(self, *, cwd: Path) -> OpenMMIdentity:
        self.worker.validate()
        record = execute_commands(
            (
                CommandSpec(
                    (
                        self.worker.python_executable,
                        "-m",
                        "mmpbsa.backends.openmm_worker",
                        "probe",
                    ),
                    Path(cwd).resolve(),
                    self.worker.environment,
                    timeout_seconds=self.worker.timeout_seconds,
                ),
            ),
            runner=self.runner,
        )[0]
        payload = _json_stdout(record.stdout, "OpenMM probe")
        if payload.get("schema_version") != OPENMM_PROBE_SCHEMA:
            raise BackendOutputError("OpenMM probe returned an unsupported schema")
        version = payload.get("openmm_version")
        platforms = payload.get("platforms")
        if (
            not isinstance(version, str)
            or not version.strip()
            or isinstance(platforms, (str, bytes))
            or not isinstance(platforms, Sequence)
            or any(not isinstance(item, Mapping) or not item.get("name") for item in platforms)
        ):
            raise BackendOutputError("OpenMM probe returned malformed identity data")
        return OpenMMIdentity(
            version=version,
            platforms=tuple(str(item["name"]) for item in platforms),
            raw=payload,
        )

    def preflight(self, request: OpenMMMdRequest, *, invocation_dir: Path) -> Mapping[str, object]:
        request.validate()
        if request.input_format != OPENMM_AMBER_INPUT_FORMAT or request.phase.kind != "em":
            raise BackendConfigurationError("Amber preflight requires an EM request")
        self.worker.validate()
        root = Path(invocation_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        generated = GeneratedInput(
            Path("openmm_preflight_request.json"),
            _canonical_json(request.as_dict()),
        )
        materialize_inputs(root, (generated,))
        command = CommandSpec(
            (
                self.worker.python_executable,
                "-m",
                "mmpbsa.backends.openmm_worker",
                "preflight",
                "--request",
                str(root / generated.relative_path),
            ),
            root,
            self.worker.environment,
            timeout_seconds=self.worker.timeout_seconds,
        )
        record = execute_commands((command,), runner=self.runner)[0]
        payload = _json_stdout(record.stdout, "OpenMM Amber preflight")
        expected = {
            "schema_version": "mmpbsa.openmm.amber-preflight.v1",
            "status": "passed",
            "request_sha256": hashlib.sha256(
                json.dumps(request.as_dict(), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "openmm_version": request.expected_openmm_version,
            "probe_sha256": request.expected_probe_sha256,
            "input_format": request.input_format,
            "topology_sha256": request.topology_sha256,
            "coordinates_sha256": request.coordinates_sha256,
            "has_periodic_box": True,
            "dispersion_correction": True,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise BackendOutputError("OpenMM Amber preflight differs from sealed request")
        atoms = payload.get("atom_count")
        particles = payload.get("particles")
        bonds = payload.get("bond_count")
        if (
            isinstance(atoms, bool)
            or not isinstance(atoms, int)
            or atoms < 1
            or particles != atoms
            or isinstance(bonds, bool)
            or not isinstance(bonds, int)
            or bonds < 0
        ):
            raise BackendOutputError("OpenMM Amber preflight has invalid system counts")
        return payload

    def build_command(
        self, request: OpenMMMdRequest
    ) -> tuple[GeneratedInput, CommandSpec, Mapping[str, Path]]:
        request.validate()
        self.worker.validate()
        root = request.output_dir.resolve()
        request_path = root / "openmm_request.json"
        generated = GeneratedInput(Path(request_path.name), _canonical_json(request.as_dict()))
        command = CommandSpec(
            (
                self.worker.python_executable,
                "-m",
                "mmpbsa.backends.openmm_worker",
                "run",
                "--request",
                str(request_path),
            ),
            root,
            self.worker.environment,
            timeout_seconds=self.worker.timeout_seconds,
        )
        return (
            generated,
            command,
            {
                **openmm_artifact_paths(request),
                "request_input": request_path,
            },
        )

    def run(
        self,
        request: OpenMMMdRequest,
        *,
        dry_run: bool = False,
    ) -> BackendReceipt:
        generated, command, artifacts = self.build_command(request)
        if dry_run:
            return BackendReceipt(
                state=ReceiptState.PLANNED,
                commands=(command,),
                artifacts=artifacts,
                generated_inputs=(generated,),
            )
        root = clean_output_dir(request.output_dir)
        materialize_inputs(root, (generated,))
        record = execute_commands((command,), runner=self.runner)[0]
        require_artifacts(artifacts)
        payload = _read_worker_receipt(artifacts["worker_receipt"], OPENMM_RECEIPT_SCHEMA)
        validate_openmm_md_worker_receipt(payload, request)
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=(record,),
            artifacts=artifacts,
            metadata={
                "openmm": payload["openmm"],
                "platform": payload["platform"],
                "system_sha256": payload["system_sha256"],
            },
            parsed=payload,
            generated_inputs=(generated,),
        )

    def build_merge_command(
        self, request: OpenMMMergeRequest
    ) -> tuple[GeneratedInput, CommandSpec, Mapping[str, Path]]:
        request.validate()
        self.worker.validate()
        root = request.output_dir.resolve()
        request_path = root / "openmm_merge_request.json"
        generated = GeneratedInput(Path(request_path.name), _canonical_json(request.as_dict()))
        command = CommandSpec(
            (
                self.worker.python_executable,
                "-m",
                "mmpbsa.backends.openmm_worker",
                "merge",
                "--request",
                str(request_path),
            ),
            root,
            self.worker.environment,
            timeout_seconds=self.worker.timeout_seconds,
        )
        return (
            generated,
            command,
            {
                **openmm_merge_artifact_paths(request),
                "request_input": request_path,
            },
        )

    def merge(
        self,
        request: OpenMMMergeRequest,
        *,
        dry_run: bool = False,
    ) -> BackendReceipt:
        generated, command, artifacts = self.build_merge_command(request)
        if dry_run:
            return BackendReceipt(
                state=ReceiptState.PLANNED,
                commands=(command,),
                artifacts=artifacts,
                generated_inputs=(generated,),
            )
        root = clean_output_dir(request.output_dir)
        materialize_inputs(root, (generated,))
        record = execute_commands((command,), runner=self.runner)[0]
        require_artifacts(artifacts)
        payload = _read_worker_receipt(artifacts["worker_receipt"], OPENMM_MERGE_RECEIPT_SCHEMA)
        validate_openmm_merge_worker_receipt(payload, request)
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=(record,),
            artifacts=artifacts,
            parsed=payload,
            generated_inputs=(generated,),
        )


def create_openmm_md_backend(
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None = None,
) -> OpenMMMdBackend:
    nested_raw = config.get("tool", {})
    nested = _mapping(nested_raw, "OpenMM tool") if nested_raw else {}
    executable = nested.get(
        "executable",
        config.get("python_executable", config.get("executable", sys.executable)),
    )
    environment_raw = config.get("environment", {})
    tool_environment_raw = nested.get("environment", {})
    environment = _mapping(environment_raw, "OpenMM environment")
    tool_environment = _mapping(tool_environment_raw, "OpenMM tool environment")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in {**environment, **tool_environment}.items()
    ):
        raise BackendConfigurationError("OpenMM environment keys and values must be strings")
    timeout = nested.get("timeout_seconds", config.get("timeout_seconds"))
    result = OpenMMMdBackend(
        worker=OpenMMWorkerConfig(
            python_executable=str(executable),
            environment={**dict(environment), **dict(tool_environment)},
            timeout_seconds=(None if timeout is None else float(timeout)),
        ),
        runner=runner,
    )
    result.worker.validate()
    return result


def _json_stdout(stdout: str, label: str) -> Mapping[str, object]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise BackendOutputError(f"{label} returned no JSON")
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise BackendOutputError(f"{label} returned invalid JSON") from exc
    return _mapping(value, f"{label} payload")


def _read_worker_receipt(path: Path, schema: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendOutputError(f"invalid OpenMM worker receipt: {path}") from exc
    result = _mapping(payload, "OpenMM worker receipt")
    if result.get("schema_version") != schema or result.get("status") != "completed":
        raise BackendOutputError("OpenMM worker did not report a completed operation")
    return result


def validate_openmm_md_worker_receipt(
    receipt: Mapping[str, object],
    request: OpenMMMdRequest,
) -> None:
    validate_md_receipt(
        receipt,
        request,
        schema=OPENMM_RECEIPT_SCHEMA,
        artifacts=openmm_artifact_paths(request),
    )


def validate_openmm_merge_worker_receipt(
    receipt: Mapping[str, object],
    request: OpenMMMergeRequest,
) -> None:
    validate_merge_receipt(
        receipt,
        request,
        schema=OPENMM_MERGE_RECEIPT_SCHEMA,
        artifacts=openmm_merge_artifact_paths(request),
    )

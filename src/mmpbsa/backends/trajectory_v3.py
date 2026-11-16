"""Engine-neutral, streaming trajectory preparation contract.

The v3 adapter deliberately consumes only a full-system XTC trajectory, the
    matching full-system topology and coordinates, and an explicit scoring-to-MD
    atom projection.
It does not depend on a GROMACS TPR/index or infer the producing MD engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Protocol

from ._support import (
    clean_output_dir,
    execute_commands,
    materialize_inputs,
    require_artifacts,
    require_file,
    tool_from_config,
)
from .api import (
    BackendReceipt,
    CommandRunner,
    CommandSpec,
    GeneratedInput,
    ReceiptState,
)
from .errors import (
    BackendConfigurationError,
    BackendOutputError,
    IncompatibleToolVersion,
    ToolProbeError,
)
from .trajectory_geometry import PREPROCESSING_REVISION
from .trajectory_v3_cpptraj import (
    _reference_preprocessing,
    _render_cpptraj_input,
    _render_visualization_cpptraj_input,
)
from .trajectory_v3_numeric import box_metric, box_tuple, box_volume, float_digest
from .versioning import ToolConfig, ToolIdentity, probe_tool

_DEFAULT_CHUNK_FRAMES = 100
_MAX_CHUNK_FRAMES = 10_000
_TIME_TOLERANCE_PS = 1.0e-3
_BOX_METRIC_TOLERANCE_NM2 = 1.0e-4
_MDTRAJ_MINIMUM = (1, 11)
_PARMED_MAJOR = 4


@dataclass(frozen=True, slots=True)
class TrajectoryPreparationV3Request:
    """Inputs required to project an engine-neutral MD trajectory for scoring.

    ``source_md_atom_indices`` is one-based and ordered by the dry scoring
    topology.  It may be non-monotonic; the declared order is authoritative.
    Frame numbers are also one-based and ``end_frame`` is inclusive.
    """

    full_trajectory: Path
    full_system_topology: Path
    full_system_coordinates: Path
    dry_complex_topology: Path
    source_md_atom_indices: tuple[int, ...]
    output_dir: Path
    start_frame: int
    end_frame: int | None
    stride: int
    expected_atom_order_signature: str
    residue_crosswalk: tuple[tuple[int, int, str], ...]
    source_first_time_ps: float
    source_interval_ps: float
    source_interval_steps: int
    expected_source_frames: int
    autoimage_anchor_mask: str | None = None
    autoimage_fixed_mask: str | None = None
    fit_mask: str | None = None
    expected_frames: int | None = None

    def validate(self) -> None:
        trajectory = require_file(self.full_trajectory, "full-system trajectory")
        require_file(self.full_system_topology, "full-system topology")
        coordinates = require_file(self.full_system_coordinates, "full-system coordinates")
        require_file(self.dry_complex_topology, "dry complex topology")
        if trajectory.suffix.lower() != ".xtc":
            raise BackendConfigurationError("trajectory v3 full_trajectory must be an XTC file")
        if coordinates.suffix.lower() not in {".gro", ".pdb"}:
            raise BackendConfigurationError(
                "trajectory v3 full_system_coordinates must be GRO or PDB"
            )
        if not self.source_md_atom_indices:
            raise BackendConfigurationError("trajectory v3 source_md_atom_indices cannot be empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 1
            for index in self.source_md_atom_indices
        ) or len(set(self.source_md_atom_indices)) != len(self.source_md_atom_indices):
            raise BackendConfigurationError(
                "trajectory v3 source_md_atom_indices must be positive and unique"
            )
        if re.fullmatch(r"[0-9a-f]{64}", self.expected_atom_order_signature) is None:
            raise BackendConfigurationError(
                "trajectory v3 expected_atom_order_signature must be a SHA-256 digest"
            )
        if not self.residue_crosswalk:
            raise BackendConfigurationError("trajectory v3 residue_crosswalk cannot be empty")
        if any(not isinstance(item, tuple) or len(item) != 3 for item in self.residue_crosswalk):
            raise BackendConfigurationError(
                "trajectory v3 residue_crosswalk entries must be triples"
            )
        if any(
            isinstance(md_index, bool)
            or not isinstance(md_index, int)
            or md_index < 1
            or isinstance(scoring_index, bool)
            or not isinstance(scoring_index, int)
            or scoring_index < 1
            or not isinstance(name, str)
            or not name.strip()
            for md_index, scoring_index, name in self.residue_crosswalk
        ):
            raise BackendConfigurationError("trajectory v3 residue_crosswalk entries are invalid")
        md_residues = tuple(item[0] for item in self.residue_crosswalk)
        scoring_residues = tuple(item[1] for item in self.residue_crosswalk)
        if (
            len(set(md_residues)) != len(md_residues)
            or len(set(scoring_residues)) != len(scoring_residues)
            or scoring_residues != tuple(sorted(scoring_residues))
        ):
            raise BackendConfigurationError(
                "trajectory v3 residue_crosswalk must be unique and scoring-ordered"
            )
        if (
            isinstance(self.source_first_time_ps, bool)
            or not isinstance(self.source_first_time_ps, (int, float))
            or not math.isfinite(float(self.source_first_time_ps))
            or float(self.source_first_time_ps) != 0.0
        ):
            raise BackendConfigurationError(
                "trajectory v3 source_first_time_ps must be exactly zero"
            )
        if (
            isinstance(self.source_interval_ps, bool)
            or not isinstance(self.source_interval_ps, (int, float))
            or not math.isfinite(float(self.source_interval_ps))
            or float(self.source_interval_ps) <= 0.0
        ):
            raise BackendConfigurationError(
                "trajectory v3 source_interval_ps must be finite and positive"
            )
        for label, value in (
            ("source_interval_steps", self.source_interval_steps),
            ("expected_source_frames", self.expected_source_frames),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise BackendConfigurationError(f"trajectory v3 {label} must be a positive integer")
        for label, value in (("start_frame", self.start_frame), ("stride", self.stride)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise BackendConfigurationError(f"trajectory v3 {label} must be a positive integer")
        if self.end_frame is not None and (
            isinstance(self.end_frame, bool)
            or not isinstance(self.end_frame, int)
            or self.end_frame < self.start_frame
        ):
            raise BackendConfigurationError("trajectory v3 end_frame must not precede start_frame")
        if self.end_frame is not None and self.end_frame > self.expected_source_frames:
            raise BackendConfigurationError(
                "trajectory v3 end_frame exceeds expected_source_frames"
            )
        if self.expected_frames is not None:
            if (
                isinstance(self.expected_frames, bool)
                or not isinstance(self.expected_frames, int)
                or self.expected_frames < 1
            ):
                raise BackendConfigurationError(
                    "trajectory v3 expected_frames must be a positive integer"
                )
            if self.end_frame is not None:
                selected = 1 + (self.end_frame - self.start_frame) // self.stride
                if selected != self.expected_frames:
                    raise BackendConfigurationError(
                        "trajectory v3 frame window does not produce expected_frames"
                    )
        masks = (self.autoimage_anchor_mask, self.autoimage_fixed_mask)
        if (masks[0] is None) != (masks[1] is None):
            raise BackendConfigurationError(
                "trajectory v3 autoimage anchor and fixed masks must be supplied together"
            )
        for mask in (*masks, self.fit_mask):
            if mask is not None and (
                not mask.strip() or any(token in mask for token in ("\n", "\r", "\x00", ";"))
            ):
                raise BackendConfigurationError("trajectory v3 mask is unsafe")


@dataclass(frozen=True, slots=True)
class TrajectoryV3Audit:
    """Small in-memory identity for a trajectory inspected chunk by chunk."""

    atom_count: int
    frame_count: int
    times_ps: tuple[float, ...]
    boxes_nm: tuple[tuple[float, ...], ...]
    steps: tuple[int, ...] | None = None

    def metadata(self) -> Mapping[str, object]:
        volumes = tuple(box_volume(box) for box in self.boxes_nm)
        return {
            "atom_count": self.atom_count,
            "frame_count": self.frame_count,
            "first_time_ps": self.times_ps[0],
            "last_time_ps": self.times_ps[-1],
            "time_sha256": float_digest(self.times_ps),
            "box_sha256": float_digest(tuple(value for box in self.boxes_nm for value in box)),
            "minimum_box_volume_nm3": min(volumes),
            "maximum_box_volume_nm3": max(volumes),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryV3Projection:
    """Audit returned by the streaming full-system-to-scoring projection."""

    source_atom_count: int
    source_frames_scanned: int
    atom_order_signature: str
    projection_sha256: str
    selected: TrajectoryV3Audit

    def metadata(self) -> Mapping[str, object]:
        return {
            "source_atom_count": self.source_atom_count,
            "source_frames_scanned": self.source_frames_scanned,
            "projected_atom_count": self.selected.atom_count,
            "atom_order_signature": self.atom_order_signature,
            "projection_sha256": self.projection_sha256,
            "selected": dict(self.selected.metadata()),
        }


def probe_mdtraj_identity() -> ToolIdentity:
    """Probe the optional distribution without importing compiled modules."""

    try:
        raw_version = metadata.version("mdtraj")
    except metadata.PackageNotFoundError as exc:
        raise ToolProbeError(
            "trajectory v3 requires the optional mdtraj>=1.11,<2 dependency"
        ) from exc
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", raw_version)
    if match is None:
        raise ToolProbeError(f"mdtraj distribution has an unparseable version: {raw_version!r}")
    version = tuple(int(item or 0) for item in match.groups())
    if version[0] != 1 or version[:2] < _MDTRAJ_MINIMUM:
        raise IncompatibleToolVersion(
            f"mdtraj {raw_version} is incompatible; trajectory v3 requires >=1.11,<2"
        )
    return ToolIdentity(
        executable="python-distribution:mdtraj",
        raw_version=f"mdtraj {raw_version}",
        version=version,
    )


def probe_parmed_identity() -> ToolIdentity:
    """Probe the canonical topology parser used for manifest identity checks."""

    try:
        raw_version = metadata.version("ParmEd")
    except metadata.PackageNotFoundError as exc:
        raise ToolProbeError(
            "trajectory v3 requires the optional ParmEd>=4.3,<5 dependency"
        ) from exc
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", raw_version)
    if match is None:
        raise ToolProbeError(f"ParmEd distribution has an unparseable version: {raw_version!r}")
    version = tuple(int(item or 0) for item in match.groups())
    if version[0] != _PARMED_MAJOR or version < (4, 3, 0):
        raise IncompatibleToolVersion(
            f"ParmEd {raw_version} is incompatible; trajectory v3 requires >=4.3,<5"
        )
    return ToolIdentity(
        executable="python-distribution:ParmEd",
        raw_version=f"ParmEd {raw_version}",
        version=version,
    )


class TrajectoryV3Processor(Protocol):
    """Injectable boundary for MDTraj projection and output inspection."""

    def project(
        self,
        request: TrajectoryPreparationV3Request,
        destination: Path,
        *,
        chunk_frames: int,
    ) -> TrajectoryV3Projection: ...

    def inspect(
        self,
        trajectory: Path,
        topology: Path,
        *,
        chunk_frames: int,
    ) -> TrajectoryV3Audit: ...

    def reconstruct(
        self, source: Path, destination: Path, topology: Path, *, chunk_frames: int
    ) -> Mapping[str, object]: ...

    def geometry(
        self, trajectory: Path, topology: Path, *, chunk_frames: int, require_whole: bool = True
    ) -> Mapping[str, object]: ...

    def aligned_geometry(
        self, physical: Path, fitted: Path, topology: Path, *, chunk_frames: int
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class MdtrajTrajectoryV3Processor:
    """MDTraj implementation; importing this module does not import MDTraj."""

    def reconstruct(
        self, source: Path, destination: Path, topology: Path, *, chunk_frames: int
    ) -> Mapping[str, object]:
        from .trajectory_geometry import reconstruct

        return reconstruct(source, destination, topology, chunk_frames=chunk_frames)

    def geometry(
        self, trajectory: Path, topology: Path, *, chunk_frames: int, require_whole: bool = True
    ) -> Mapping[str, object]:
        from .trajectory_geometry import validate

        return validate(
            trajectory, topology, chunk_frames=chunk_frames, require_whole=require_whole
        )

    def aligned_geometry(
        self, physical: Path, fitted: Path, topology: Path, *, chunk_frames: int
    ) -> Mapping[str, object]:
        from .trajectory_geometry import validate_rigid_geometry

        return validate_rigid_geometry(physical, fitted, topology, chunk_frames=chunk_frames)

    def project(
        self,
        request: TrajectoryPreparationV3Request,
        destination: Path,
        *,
        chunk_frames: int,
    ) -> TrajectoryV3Projection:
        md, np, xtc_file = _load_mdtraj()
        try:
            full_topology = md.load_topology(str(Path(request.full_system_coordinates).resolve()))
            dry_topology = md.load_topology(str(Path(request.dry_complex_topology).resolve()))
        except Exception as exc:
            raise BackendConfigurationError(
                "trajectory v3 could not parse the declared topologies with MDTraj"
            ) from exc

        full_contract_topology, dry_contract_topology = _load_parmed_topologies(request)
        if int(full_topology.n_atoms) != len(full_contract_topology.atoms) or int(
            dry_topology.n_atoms
        ) != len(dry_contract_topology.atoms):
            raise BackendConfigurationError("trajectory v3 topology parsers disagree on atom count")
        _validate_coordinate_topology_order(
            full_topology,
            full_contract_topology,
            check_elements=Path(request.full_system_coordinates).suffix.lower() != ".gro",
        )

        source_indices = tuple(index - 1 for index in request.source_md_atom_indices)
        atom_order_signature, projection_sha256 = _validate_topology_projection(
            full_contract_topology,
            dry_contract_topology,
            source_indices,
            expected_atom_order_signature=request.expected_atom_order_signature,
            residue_crosswalk=request.residue_crosswalk,
        )
        destination = Path(destination).resolve()
        times: list[float] = []
        steps: list[int] = []
        boxes: list[tuple[float, ...]] = []
        source_frames = 0
        previous_source_time: float | None = None
        previous_source_step: int | None = None

        try:
            with (
                xtc_file(str(Path(request.full_trajectory).resolve()), "r") as source,
                xtc_file(str(destination), "w") as target,
            ):
                while True:
                    xyz, chunk_times, chunk_steps, chunk_boxes = source.read(n_frames=chunk_frames)
                    frame_count = int(xyz.shape[0])
                    if frame_count == 0:
                        break
                    if int(xyz.shape[1]) != int(full_topology.n_atoms):
                        raise BackendOutputError(
                            "trajectory v3 XTC atom count differs from full-system topology"
                        )
                    _validate_numeric_chunk(
                        np,
                        xyz,
                        chunk_times,
                        chunk_boxes,
                        label="source XTC",
                    )
                    _validate_increasing(
                        tuple(float(value) for value in chunk_times),
                        previous=previous_source_time,
                        label="source XTC time",
                    )
                    _validate_increasing(
                        tuple(int(value) for value in chunk_steps),
                        previous=previous_source_step,
                        label="source XTC step",
                    )
                    previous_source_time = float(chunk_times[-1])
                    previous_source_step = int(chunk_steps[-1])

                    source_frames += frame_count
                    projected_xyz = xyz[:, source_indices, :]
                    target.write(
                        projected_xyz,
                        time=chunk_times,
                        step=chunk_steps,
                        box=chunk_boxes,
                    )
                    times.extend(float(value) for value in chunk_times)
                    steps.extend(int(value) for value in chunk_steps)
                    boxes.extend(box_tuple(value) for value in chunk_boxes)
        except BackendOutputError:
            raise
        except Exception as exc:
            raise BackendOutputError(
                "trajectory v3 failed while streaming the full-system XTC"
            ) from exc

        if request.end_frame is not None and source_frames < request.end_frame:
            raise BackendOutputError("trajectory v3 source XTC ends before the declared end_frame")
        selected = TrajectoryV3Audit(
            atom_count=len(source_indices),
            frame_count=len(times),
            times_ps=tuple(times),
            boxes_nm=tuple(boxes),
            steps=tuple(steps),
        )
        _validate_audit(selected, expected_frames=None, label="projected XTC")
        _validate_source_trajectory(selected, request)
        return TrajectoryV3Projection(
            source_atom_count=int(full_topology.n_atoms),
            source_frames_scanned=source_frames,
            atom_order_signature=atom_order_signature,
            projection_sha256=projection_sha256,
            selected=selected,
        )

    def inspect(
        self,
        trajectory: Path,
        topology: Path,
        *,
        chunk_frames: int,
    ) -> TrajectoryV3Audit:
        md, np, xtc_file = _load_mdtraj()
        try:
            loaded_topology = md.load_topology(str(Path(topology).resolve()))
        except Exception as exc:
            raise BackendConfigurationError(
                "trajectory v3 could not parse the dry complex topology with MDTraj"
            ) from exc
        resolved = Path(trajectory).resolve()
        if resolved.suffix.lower() == ".xtc":
            return _inspect_xtc(
                np,
                xtc_file,
                resolved,
                atom_count=int(loaded_topology.n_atoms),
                chunk_frames=chunk_frames,
            )
        return _inspect_mdtraj(
            md,
            np,
            resolved,
            loaded_topology,
            chunk_frames=chunk_frames,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryPreparationV3Backend:
    cpptraj: ToolConfig
    chunk_frames: int = _DEFAULT_CHUNK_FRAMES
    runner: CommandRunner | None = None
    processor: TrajectoryV3Processor | None = None

    def __post_init__(self) -> None:
        _validate_chunk_frames(self.chunk_frames)

    def build_commands(
        self,
        request: TrajectoryPreparationV3Request,
    ) -> tuple[tuple[GeneratedInput, ...], tuple[CommandSpec, ...], Mapping[str, Path]]:
        request.validate()
        root = Path(request.output_dir).resolve()
        selected = root / "selected.xtc"
        whole = root / "whole.xtc"
        prepared_reference = root / "prepared_reference.nc"
        prepared = root / "prepared.nc"
        centered = root / "centered.xtc"
        script = root / "trajectory.in"
        generated = GeneratedInput(
            Path(script.name),
            _render_cpptraj_input(
                request,
                selected=whole,
                prepared_reference=prepared_reference,
                prepared=prepared,
                centered=centered,
            ),
        )
        command = CommandSpec(
            argv=(self.cpptraj.executable, "-i", str(script)),
            cwd=root,
            environment=self.cpptraj.environment,
        )
        display_script = root / "trajectory_display.in"
        display_generated = GeneratedInput(
            Path(display_script.name),
            _render_visualization_cpptraj_input(request, whole=whole, centered=centered),
        )
        display_command = CommandSpec(
            argv=(self.cpptraj.executable, "-i", str(display_script)),
            cwd=root,
            environment=self.cpptraj.environment,
        )
        return (
            (generated, display_generated),
            (command, display_command),
            {
                "selected_trajectory": selected,
                "whole_trajectory": whole,
                "prepared_reference_trajectory": prepared_reference,
                "prepared_trajectory": prepared,
                "centered_trajectory": centered,
                "cpptraj_input": script,
                "cpptraj_display_input": display_script,
            },
        )

    def run(
        self,
        request: TrajectoryPreparationV3Request,
        *,
        dry_run: bool = False,
    ) -> BackendReceipt:
        generated, commands, artifacts = self.build_commands(request)
        if dry_run:
            return BackendReceipt(
                state=ReceiptState.PLANNED,
                commands=commands,
                artifacts=artifacts,
                generated_inputs=generated,
            )

        root = clean_output_dir(request.output_dir)
        identity = probe_tool(self.cpptraj, cwd=root, runner=self.runner)
        materialize_inputs(root, generated)
        processor = self.processor or MdtrajTrajectoryV3Processor()
        projection = processor.project(
            request,
            artifacts["selected_trajectory"],
            chunk_frames=self.chunk_frames,
        )
        _validate_source_trajectory(projection.selected, request)
        if (
            projection.source_frames_scanned != request.expected_source_frames
            or projection.source_frames_scanned != projection.selected.frame_count
        ):
            raise BackendOutputError("trajectory v3 projection source-frame audit is inconsistent")
        if projection.atom_order_signature != request.expected_atom_order_signature:
            raise BackendOutputError(
                "trajectory v3 projection did not preserve the expected atom order"
            )
        if re.fullmatch(r"[0-9a-f]{64}", projection.projection_sha256) is None:
            raise BackendOutputError(
                "trajectory v3 projection did not return a valid projection digest"
            )
        analysis_reference = _analysis_window(projection.selected, request)
        reconstruction = processor.reconstruct(
            artifacts["selected_trajectory"],
            artifacts["whole_trajectory"],
            request.dry_complex_topology,
            chunk_frames=self.chunk_frames,
        )
        records = execute_commands(commands, runner=self.runner)
        if any(
            "Box cannot be properly stored" in record.stdout + record.stderr for record in records
        ):
            raise BackendOutputError(
                "trajectory unit cell cannot be represented by the output format"
            )
        require_artifacts(artifacts)

        whole_audit = processor.inspect(
            artifacts["whole_trajectory"],
            request.dry_complex_topology,
            chunk_frames=self.chunk_frames,
        )
        _validate_derived_trajectory(projection.selected, whole_audit, label="whole XTC")

        prepared = processor.inspect(
            artifacts["prepared_trajectory"],
            request.dry_complex_topology,
            chunk_frames=self.chunk_frames,
        )
        prepared_reference = processor.inspect(
            artifacts["prepared_reference_trajectory"],
            request.dry_complex_topology,
            chunk_frames=self.chunk_frames,
        )
        centered = processor.inspect(
            artifacts["centered_trajectory"],
            request.dry_complex_topology,
            chunk_frames=self.chunk_frames,
        )
        _validate_derived_trajectory(analysis_reference, prepared, label="prepared NetCDF")
        source_t0 = projection.selected
        _validate_derived_trajectory(
            TrajectoryV3Audit(
                atom_count=source_t0.atom_count,
                frame_count=1,
                times_ps=source_t0.times_ps[:1],
                boxes_nm=source_t0.boxes_nm[:1],
                steps=source_t0.steps[:1] if source_t0.steps is not None else None,
            ),
            prepared_reference,
            label="prepared t0 reference NetCDF",
        )
        _validate_derived_trajectory(analysis_reference, centered, label="centered XTC")

        geometry = {
            role: dict(
                processor.geometry(
                    artifacts[role],
                    request.dry_complex_topology,
                    chunk_frames=self.chunk_frames,
                    require_whole=True,
                )
            )
            for role in ("whole_trajectory", "prepared_reference_trajectory", "prepared_trajectory")
        }
        # Fitted XTC retains rotated box vectors; compare rigid invariants rather
        # than apply minimum-image kernels requiring a reduced box orientation.
        geometry["centered_trajectory"] = dict(
            processor.geometry(
                artifacts["centered_trajectory"],
                request.dry_complex_topology,
                chunk_frames=self.chunk_frames,
                require_whole=False,
            )
        )
        alignment_geometry = processor.aligned_geometry(
            artifacts["prepared_trajectory"],
            artifacts["centered_trajectory"],
            request.dry_complex_topology,
            chunk_frames=self.chunk_frames,
        )
        if any(
            item.get("status") != "passed"
            for item in (reconstruction, alignment_geometry, *geometry.values())
        ):
            raise BackendOutputError("trajectory geometry validation did not pass")

        validation = {
            "selected_trajectory": dict(projection.selected.metadata()),
            "whole_trajectory": dict(whole_audit.metadata()),
            "prepared_reference_trajectory": dict(prepared_reference.metadata()),
            "prepared_trajectory": dict(prepared.metadata()),
            "centered_trajectory": dict(centered.metadata()),
        }
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=records,
            artifacts=artifacts,
            metadata={
                "contract": "mmpbsa.trajectory.v3",
                "preprocessing_revision": PREPROCESSING_REVISION,
                "geometry": {
                    "revision": PREPROCESSING_REVISION,
                    "status": "passed",
                    "reconstruction": dict(reconstruction),
                    "artifacts": geometry,
                    "alignment": dict(alignment_geometry),
                },
                "tools": {"cpptraj": identity},
                "selection": {
                    "start_frame": request.start_frame,
                    "end_frame": request.end_frame,
                    "stride": request.stride,
                    "expected_frames": request.expected_frames,
                },
                "source_trajectory_contract": {
                    "first_time_ps": request.source_first_time_ps,
                    "interval_ps": request.source_interval_ps,
                    "interval_steps": request.source_interval_steps,
                    "expected_frames": request.expected_source_frames,
                },
                "reference": {
                    "artifact": "prepared_reference_trajectory",
                    "frame": 1,
                    "semantics": "post_npt_production_start",
                    "source_artifact": "selected_trajectory",
                    "preprocessing": _reference_preprocessing(request),
                },
                "projection": dict(projection.metadata()),
                "validation": validation,
            },
            parsed={
                "frame_count": prepared.frame_count,
                "source_frame_count": projection.selected.frame_count,
                "atom_count": prepared.atom_count,
                "atom_order_signature": projection.atom_order_signature,
                "projection_sha256": projection.projection_sha256,
            },
            generated_inputs=generated,
        )


def create_trajectory_v3_backend(
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None = None,
    processor: TrajectoryV3Processor | None = None,
) -> TrajectoryPreparationV3Backend:
    """Create the v3 adapter without importing MDTraj until execution."""

    chunk_frames = _validate_chunk_frames(
        config.get("projection_chunk_frames", _DEFAULT_CHUNK_FRAMES)
    )
    return TrajectoryPreparationV3Backend(
        cpptraj=tool_from_config(
            config,
            key="cpptraj",
            default_executable="cpptraj",
            default_version_arguments=("-V",),
        ),
        chunk_frames=chunk_frames,
        runner=runner,
        processor=processor,
    )


def _selected_frame(frame: int, request: TrajectoryPreparationV3Request) -> bool:
    if frame < request.start_frame:
        return False
    if request.end_frame is not None and frame > request.end_frame:
        return False
    return (frame - request.start_frame) % request.stride == 0


def _analysis_window(
    projected: TrajectoryV3Audit,
    request: TrajectoryPreparationV3Request,
) -> TrajectoryV3Audit:
    selected = tuple(
        index for index in range(projected.frame_count) if _selected_frame(index + 1, request)
    )
    audit = TrajectoryV3Audit(
        atom_count=projected.atom_count,
        frame_count=len(selected),
        times_ps=tuple(projected.times_ps[index] for index in selected),
        boxes_nm=tuple(projected.boxes_nm[index] for index in selected),
        steps=(
            None if projected.steps is None else tuple(projected.steps[index] for index in selected)
        ),
    )
    _validate_audit(
        audit,
        expected_frames=request.expected_frames,
        label="analysis-window projection",
    )
    if request.end_frame is not None and projected.frame_count < request.end_frame:
        raise BackendOutputError("trajectory v3 source XTC ends before the declared end_frame")
    return audit


def _validate_coordinate_topology_order(
    coordinate_topology: Any,
    chemical_topology: Any,
    *,
    check_elements: bool = True,
) -> None:
    """Check the coordinate order without treating format-specific names as chemistry.

    GRO/PDB readers may serialize HIE as HIS and WAT as HOH.  The ordered
    residue index, atom name, and known element still identify each position.
    The single H/H1 alias is emitted by the current GRO/PDB conversion for an
    N-terminal hydrogen; all other atom-name changes fail closed.

    GRO has no element field: readers guess FE as fluorine or CL1 as carbon.
    Those guesses are not independent chemical evidence. For GRO, validate
    exact names and residue positions, and use the separately validated
    chemical topology for elements. Explicit coordinate elements remain
    checked for formats that provide them.
    """

    coordinate_atoms = tuple(coordinate_topology.atoms)
    chemical_atoms = tuple(chemical_topology.atoms)
    if len(coordinate_atoms) != len(chemical_atoms) or int(coordinate_topology.n_residues) != len(
        chemical_topology.residues
    ):
        raise BackendConfigurationError(
            "trajectory v3 MD coordinates and topology differ in atom/residue count"
        )
    for ordinal, (coordinate, chemical) in enumerate(
        zip(coordinate_atoms, chemical_atoms, strict=True), start=1
    ):
        coordinate_element = int(getattr(getattr(coordinate, "element", None), "number", 0) or 0)
        chemical_element = int(getattr(chemical, "atomic_number", 0) or 0)
        same_name = str(coordinate.name) == str(chemical.name)
        terminal_h_alias = (
            str(coordinate.name) == "H"
            and str(chemical.name) == "H1"
            and coordinate_element == chemical_element == 1
        )
        if (
            int(coordinate.residue.index) != int(chemical.residue.idx)
            or not (same_name or terminal_h_alias)
            or (
                check_elements and coordinate_element > 0 and coordinate_element != chemical_element
            )
        ):
            raise BackendConfigurationError(
                "trajectory v3 MD coordinate atom order differs from the "
                f"chemical topology at atom {ordinal}"
            )


def _validate_topology_projection(
    full_topology: Any,
    dry_topology: Any,
    source_indices: tuple[int, ...],
    *,
    expected_atom_order_signature: str,
    residue_crosswalk: tuple[tuple[int, int, str], ...],
) -> tuple[str, str]:
    if len(dry_topology.atoms) != len(source_indices):
        raise BackendConfigurationError(
            "trajectory v3 atom mapping length differs from dry complex topology"
        )
    if max(source_indices) >= len(full_topology.atoms):
        raise BackendConfigurationError("trajectory v3 atom mapping exceeds full-system topology")
    atom_order_signature = _parmed_atom_order_signature(dry_topology)
    if atom_order_signature != expected_atom_order_signature:
        raise BackendConfigurationError(
            "trajectory v3 dry topology atom order differs from PreparedCase signature"
        )

    crosswalk = {
        scoring_index: (md_index, residue_name)
        for md_index, scoring_index, residue_name in residue_crosswalk
    }
    dry_residue_indices = {int(atom.residue.idx) + 1 for atom in dry_topology.atoms}
    if set(crosswalk) != dry_residue_indices:
        raise BackendConfigurationError(
            "trajectory v3 residue_crosswalk does not cover the dry complex topology"
        )

    projected_identities: list[tuple[int, int, str, str, int]] = []
    for scoring_atom, (source_index, dry_atom) in enumerate(
        zip(source_indices, dry_topology.atoms, strict=True),
        start=1,
    ):
        source_atom = full_topology.atoms[source_index]
        scoring_residue_index = int(dry_atom.residue.idx) + 1
        expected_md_residue, expected_name = crosswalk[scoring_residue_index]
        source_identity = _parmed_atom_identity(source_atom)
        dry_identity = _parmed_atom_identity(dry_atom)
        if (
            int(source_atom.residue.idx) + 1 != expected_md_residue
            or source_identity != dry_identity
            or source_identity[0] != expected_name
        ):
            raise BackendConfigurationError(
                "trajectory v3 mapped atom/residue identity differs from the "
                f"PreparedCase crosswalk at scoring atom {scoring_atom}"
            )
        projected_identities.append(
            (
                expected_md_residue,
                scoring_residue_index,
                *source_identity,
            )
        )

    source_to_scoring = {source: scoring for scoring, source in enumerate(source_indices)}
    projected_bonds = {
        tuple(sorted((source_to_scoring[first.idx], source_to_scoring[second.idx])))
        for bond in full_topology.bonds
        for first, second in ((bond.atom1, bond.atom2),)
        if first.idx in source_to_scoring and second.idx in source_to_scoring
    }
    dry_bonds = {tuple(sorted((bond.atom1.idx, bond.atom2.idx))) for bond in dry_topology.bonds}
    if projected_bonds != dry_bonds:
        raise BackendConfigurationError(
            "trajectory v3 mapped bond graph differs from dry complex topology"
        )
    encoded = json.dumps(
        {
            "atom_order_signature": atom_order_signature,
            "source_md_atom_indices": [index + 1 for index in source_indices],
            "residue_crosswalk": residue_crosswalk,
            "atom_identities": projected_identities,
            "bonds": sorted(projected_bonds),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return atom_order_signature, hashlib.sha256(encoded).hexdigest()


def _parmed_atom_identity(atom: Any) -> tuple[str, str, int]:
    return (
        str(atom.residue.name),
        str(atom.name),
        int(getattr(atom, "atomic_number", 0) or 0),
    )


def _parmed_atom_order_signature(topology: Any) -> str:
    encoded = json.dumps(
        {
            "schema": "mmpbsa.atom_order.v1",
            "atoms": [
                {
                    "residue_index": int(atom.residue.idx) + 1,
                    "residue_name": str(atom.residue.name),
                    "atom_name": str(atom.name),
                    "atomic_number": int(getattr(atom, "atomic_number", 0) or 0),
                }
                for atom in topology.atoms
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inspect_xtc(
    np: Any,
    xtc_file: Any,
    trajectory: Path,
    *,
    atom_count: int,
    chunk_frames: int,
) -> TrajectoryV3Audit:
    times: list[float] = []
    steps: list[int] = []
    boxes: list[tuple[float, ...]] = []
    previous_time: float | None = None
    previous_step: int | None = None
    try:
        with xtc_file(str(trajectory), "r") as source:
            while True:
                xyz, chunk_times, chunk_steps, chunk_boxes = source.read(n_frames=chunk_frames)
                if int(xyz.shape[0]) == 0:
                    break
                if int(xyz.shape[1]) != atom_count:
                    raise BackendOutputError(
                        f"trajectory atom count differs from dry topology: {trajectory}"
                    )
                _validate_numeric_chunk(
                    np,
                    xyz,
                    chunk_times,
                    chunk_boxes,
                    label=str(trajectory),
                )
                values = tuple(float(value) for value in chunk_times)
                step_values = tuple(int(value) for value in chunk_steps)
                _validate_increasing(values, previous=previous_time, label=f"{trajectory} time")
                _validate_increasing(
                    step_values,
                    previous=previous_step,
                    label=f"{trajectory} step",
                )
                previous_time = values[-1]
                previous_step = step_values[-1]
                times.extend(values)
                steps.extend(step_values)
                boxes.extend(box_tuple(value) for value in chunk_boxes)
    except BackendOutputError:
        raise
    except Exception as exc:
        raise BackendOutputError(f"trajectory v3 could not inspect XTC: {trajectory}") from exc
    audit = TrajectoryV3Audit(
        atom_count=atom_count,
        frame_count=len(times),
        times_ps=tuple(times),
        boxes_nm=tuple(boxes),
        steps=tuple(steps),
    )
    _validate_audit(audit, expected_frames=None, label=str(trajectory))
    return audit


def _inspect_mdtraj(
    md: Any,
    np: Any,
    trajectory: Path,
    topology: Any,
    *,
    chunk_frames: int,
) -> TrajectoryV3Audit:
    times: list[float] = []
    boxes: list[tuple[float, ...]] = []
    previous_time: float | None = None
    try:
        for chunk in md.iterload(
            str(trajectory),
            chunk=chunk_frames,
            top=topology,
        ):
            if int(chunk.n_atoms) != int(topology.n_atoms):
                raise BackendOutputError(
                    f"trajectory atom count differs from dry topology: {trajectory}"
                )
            _validate_numeric_chunk(
                np,
                chunk.xyz,
                chunk.time,
                chunk.unitcell_vectors,
                label=str(trajectory),
            )
            values = tuple(float(value) for value in chunk.time)
            _validate_increasing(values, previous=previous_time, label=f"{trajectory} time")
            previous_time = values[-1]
            times.extend(values)
            boxes.extend(box_tuple(value) for value in chunk.unitcell_vectors)
    except BackendOutputError:
        raise
    except Exception as exc:
        raise BackendOutputError(
            f"trajectory v3 could not inspect trajectory: {trajectory}"
        ) from exc
    audit = TrajectoryV3Audit(
        atom_count=int(topology.n_atoms),
        frame_count=len(times),
        times_ps=tuple(times),
        boxes_nm=tuple(boxes),
    )
    _validate_audit(audit, expected_frames=None, label=str(trajectory))
    return audit


def _validate_numeric_chunk(
    np: Any,
    xyz: Any,
    times: Any,
    boxes: Any,
    *,
    label: str,
) -> None:
    frame_count = int(xyz.shape[0])
    if times is None or len(times) != frame_count:
        raise BackendOutputError(f"trajectory v3 {label} has invalid frame times")
    if boxes is None or len(boxes) != frame_count:
        raise BackendOutputError(f"trajectory v3 {label} has missing periodic boxes")
    if not bool(np.isfinite(xyz).all()) or not bool(np.isfinite(times).all()):
        raise BackendOutputError(f"trajectory v3 {label} contains non-finite coordinates/time")
    if not bool(np.isfinite(boxes).all()):
        raise BackendOutputError(f"trajectory v3 {label} contains non-finite box vectors")
    determinants = np.linalg.det(boxes)
    if bool((determinants <= 1.0e-9).any()):
        raise BackendOutputError(f"trajectory v3 {label} contains a non-positive periodic box")


def _validate_increasing(
    values: tuple[float | int, ...],
    *,
    previous: float | int | None,
    label: str,
) -> None:
    candidate = (() if previous is None else (previous,)) + values
    if any(right <= left for left, right in zip(candidate, candidate[1:], strict=False)):
        raise BackendOutputError(f"trajectory v3 {label} is not strictly increasing")


def _validate_audit(
    audit: TrajectoryV3Audit,
    *,
    expected_frames: int | None,
    label: str,
) -> None:
    if audit.atom_count < 1 or audit.frame_count < 1:
        raise BackendOutputError(f"trajectory v3 {label} is empty")
    if len(audit.times_ps) != audit.frame_count or len(audit.boxes_nm) != audit.frame_count:
        raise BackendOutputError(f"trajectory v3 {label} audit is internally inconsistent")
    if audit.steps is not None and len(audit.steps) != audit.frame_count:
        raise BackendOutputError(f"trajectory v3 {label} step audit is inconsistent")
    if expected_frames is not None and audit.frame_count != expected_frames:
        raise BackendOutputError(
            f"trajectory v3 {label} frame count {audit.frame_count} differs from "
            f"expected {expected_frames}"
        )
    if any(not math.isfinite(value) for value in audit.times_ps):
        raise BackendOutputError(f"trajectory v3 {label} contains non-finite time")
    _validate_increasing(audit.times_ps, previous=None, label=f"{label} time")
    if any(
        len(box) != 9 or any(not math.isfinite(value) for value in box) or box_volume(box) <= 1.0e-9
        for box in audit.boxes_nm
    ):
        raise BackendOutputError(f"trajectory v3 {label} contains an invalid periodic box")


def _validate_source_trajectory(
    audit: TrajectoryV3Audit,
    request: TrajectoryPreparationV3Request,
) -> None:
    """Enforce the engine-neutral public production trajectory timeline."""

    _validate_audit(
        audit,
        expected_frames=request.expected_source_frames,
        label="source production XTC",
    )
    if audit.steps is None:
        raise BackendOutputError(
            "trajectory v3 source production XTC does not expose simulation steps"
        )
    for index, (time_ps, step) in enumerate(zip(audit.times_ps, audit.steps, strict=True)):
        expected_time = float(request.source_first_time_ps) + index * float(
            request.source_interval_ps
        )
        expected_step = index * request.source_interval_steps
        if not math.isclose(
            time_ps,
            expected_time,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_PS,
        ):
            raise BackendOutputError(
                "trajectory v3 source production XTC violates the declared "
                f"time grid at frame {index + 1}"
            )
        if step != expected_step:
            raise BackendOutputError(
                "trajectory v3 source production XTC violates the declared "
                f"step grid at frame {index + 1}"
            )


def _validate_derived_trajectory(
    reference: TrajectoryV3Audit,
    observed: TrajectoryV3Audit,
    *,
    label: str,
) -> None:
    _validate_audit(observed, expected_frames=reference.frame_count, label=label)
    if observed.atom_count != reference.atom_count:
        raise BackendOutputError(
            f"trajectory v3 {label} atom count differs from the scoring projection"
        )
    if any(
        not math.isclose(left, right, rel_tol=1.0e-7, abs_tol=_TIME_TOLERANCE_PS)
        for left, right in zip(reference.times_ps, observed.times_ps, strict=True)
    ):
        raise BackendOutputError(f"trajectory v3 {label} times differ from the scoring projection")
    if any(
        not math.isclose(
            left,
            right,
            rel_tol=1.0e-7,
            abs_tol=_BOX_METRIC_TOLERANCE_NM2,
        )
        for reference_box, observed_box in zip(reference.boxes_nm, observed.boxes_nm, strict=True)
        for left, right in zip(
            box_metric(reference_box),
            box_metric(observed_box),
            strict=True,
        )
    ):
        raise BackendOutputError(
            f"trajectory v3 {label} box geometry differs from the scoring projection"
        )


def _load_mdtraj() -> tuple[Any, Any, Any]:
    try:
        md = import_module("mdtraj")
        np = import_module("numpy")
        xtc_file = import_module("mdtraj.formats").XTCTrajectoryFile
    except (ImportError, AttributeError) as exc:
        raise BackendConfigurationError(
            "trajectory v3 requires the optional MDTraj dependency"
        ) from exc
    return md, np, xtc_file


def _load_parmed_topologies(
    request: TrajectoryPreparationV3Request,
) -> tuple[Any, Any]:
    probe_parmed_identity()
    try:
        pmd = import_module("parmed")
        from parmed.gromacs.gromacstop import GromacsWarning

        with warnings.catch_warnings():
            # filterwarnings is case-insensitive by default; keep this exact.
            warnings.filterwarnings(
                "ignore",
                message=r"^(?-i:pairs funct != 1; unknown functional)$",
                category=GromacsWarning,
            )
            full = pmd.load_file(str(Path(request.full_system_topology).resolve()))
            dry = pmd.load_file(str(Path(request.dry_complex_topology).resolve()))
    except Exception as exc:
        raise BackendConfigurationError(
            "trajectory v3 could not parse the declared topologies with ParmEd"
        ) from exc
    return full, dry


def _validate_chunk_frames(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_CHUNK_FRAMES:
        raise BackendConfigurationError(
            f"projection_chunk_frames must be an integer in [1, {_MAX_CHUNK_FRAMES}]"
        )
    return value

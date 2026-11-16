"""cpptraj/PyMOL display-asset generation without report coupling."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
from .cpptraj import parse_cpptraj_table
from .errors import BackendConfigurationError
from .versioning import ToolConfig, probe_tool

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class DisplayAssetRequest:
    """Explicit inputs for creating portable structures and a PyMOL loader."""

    complex_topology: Path
    trajectories: Mapping[str, Path]
    component_atom_indices: Mapping[str, tuple[int, ...]]
    output_dir: Path
    alignment_mask: str | None = None
    start_frame: int = 1
    end_frame: int | None = None
    stride: int = 1
    default_replica: str | None = None
    default_state: int = 1
    reference_trajectories: Mapping[str, Path] | None = None
    reference_frame: int = 1
    reference_autoimage_anchor_atom_indices: tuple[int, ...] | None = None
    reference_autoimage_fixed_atom_indices: tuple[int, ...] | None = None
    reference_alignment_atom_indices: tuple[int, ...] | None = None
    ligand_rmsd_atom_indices: tuple[int, ...] | None = None
    reference_artifact: str = "selected_trajectory"

    def validate(self) -> None:
        if self.reference_artifact not in {"selected_trajectory", "prepared_reference_trajectory"}:
            raise BackendConfigurationError("unsupported display reference_artifact")
        require_file(self.complex_topology, "display complex topology")
        if not self.trajectories:
            raise BackendConfigurationError("display trajectories cannot be empty")
        for replica, path in self.trajectories.items():
            if not _IDENTIFIER.fullmatch(str(replica)):
                raise BackendConfigurationError(f"invalid display replica identifier: {replica!r}")
            require_file(path, f"display trajectory {replica}")
        if not self.component_atom_indices:
            raise BackendConfigurationError("display component_atom_indices cannot be empty")
        selected: list[int] = []
        for component, indices in self.component_atom_indices.items():
            if not _IDENTIFIER.fullmatch(str(component)):
                raise BackendConfigurationError(
                    f"invalid display component identifier: {component!r}"
                )
            if (
                not indices
                or any(isinstance(index, bool) or index < 1 for index in indices)
                or tuple(sorted(set(indices))) != indices
            ):
                raise BackendConfigurationError(
                    f"display component {component!r} indices must be sorted, unique, and positive"
                )
            selected.extend(indices)
        if len(selected) != len(set(selected)):
            raise BackendConfigurationError("display component selections overlap")
        for label, value in (
            ("start_frame", self.start_frame),
            ("stride", self.stride),
            ("default_state", self.default_state),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise BackendConfigurationError(f"display {label} must be positive")
        if self.end_frame is not None and (
            isinstance(self.end_frame, bool)
            or not isinstance(self.end_frame, int)
            or self.end_frame < self.start_frame
        ):
            raise BackendConfigurationError("display end_frame precedes start_frame")
        if self.alignment_mask is not None and (
            not self.alignment_mask.strip()
            or any(token in self.alignment_mask for token in ("\n", "\r", "\x00", ";"))
        ):
            raise BackendConfigurationError("display alignment_mask is unsafe")
        if self.default_replica is not None and self.default_replica not in self.trajectories:
            raise BackendConfigurationError("display default_replica is not in trajectories")
        diagnostics = (
            self.reference_trajectories,
            self.reference_autoimage_anchor_atom_indices,
            self.reference_autoimage_fixed_atom_indices,
            self.reference_alignment_atom_indices,
            self.ligand_rmsd_atom_indices,
        )
        if any(value is not None for value in diagnostics) and not all(
            value is not None for value in diagnostics
        ):
            raise BackendConfigurationError(
                "display equilibrated-pose RMSD requires reference trajectories, "
                "reference autoimage atoms, reference alignment atoms, and ligand "
                "RMSD atoms together"
            )
        if self.reference_trajectories is not None:
            if set(self.reference_trajectories) != set(self.trajectories):
                raise BackendConfigurationError(
                    "display reference trajectories must exactly match replicas"
                )
            for replica, path in self.reference_trajectories.items():
                require_file(
                    path,
                    f"display equilibrated-pose reference trajectory {replica}",
                )
            if (
                isinstance(self.reference_frame, bool)
                or not isinstance(self.reference_frame, int)
                or self.reference_frame < 1
            ):
                raise BackendConfigurationError("display reference_frame must be positive")
            for label, indices in (
                (
                    "reference_autoimage_anchor_atom_indices",
                    self.reference_autoimage_anchor_atom_indices,
                ),
                (
                    "reference_autoimage_fixed_atom_indices",
                    self.reference_autoimage_fixed_atom_indices,
                ),
                (
                    "reference_alignment_atom_indices",
                    self.reference_alignment_atom_indices,
                ),
                ("ligand_rmsd_atom_indices", self.ligand_rmsd_atom_indices),
            ):
                if (
                    indices is None
                    or not indices
                    or any(isinstance(index, bool) or index < 1 for index in indices)
                    or tuple(sorted(set(indices))) != indices
                ):
                    raise BackendConfigurationError(
                        f"display {label} must be sorted, unique, and positive"
                    )


@dataclass(frozen=True, slots=True)
class DisplayAssetBackend:
    cpptraj: ToolConfig
    runner: CommandRunner | None = None

    def build_commands(
        self,
        request: DisplayAssetRequest,
    ) -> tuple[
        tuple[GeneratedInput, ...],
        tuple[CommandSpec, ...],
        Mapping[str, Path],
    ]:
        request.validate()
        root = Path(request.output_dir).resolve()
        structures = root / "structures"
        trajectories = root / "trajectories"
        diagnostics = root / "diagnostics"
        scripts = root / "inputs"
        generated: list[GeneratedInput] = []
        commands: list[CommandSpec] = []
        artifacts: dict[str, Path] = {}

        for replica, trajectory in sorted(request.trajectories.items()):
            script = scripts / f"{replica}.in"
            structure = structures / f"{replica}.pdb"
            display_trajectory = trajectories / f"{replica}.xtc"
            pymol = root / f"load_{replica}.pml"
            diagnostic = diagnostics / f"{replica}_ligand_equilibrated_rmsd.dat"
            reference_structure = diagnostics / f"{replica}_equilibrated_reference.pdb"
            if request.reference_trajectories is not None:
                reference_script = scripts / f"{replica}_reference.in"
                generated.append(
                    GeneratedInput(
                        reference_script.relative_to(root),
                        _render_reference_cpptraj(
                            request,
                            replica,
                            reference_structure,
                        ),
                    )
                )
                commands.append(
                    CommandSpec(
                        argv=(
                            self.cpptraj.executable,
                            "-i",
                            str(reference_script),
                        ),
                        cwd=root,
                        environment=self.cpptraj.environment,
                    )
                )
                artifacts[f"reference_input.{replica}"] = reference_script
                artifacts[f"equilibrated_reference.{replica}"] = reference_structure
            content = _render_cpptraj(
                request,
                replica,
                trajectory,
                structure,
                display_trajectory,
                diagnostic,
                reference_structure,
            )
            generated.extend(
                (
                    GeneratedInput(script.relative_to(root), content),
                    GeneratedInput(
                        pymol.relative_to(root),
                        _render_pymol(request, replica),
                    ),
                )
            )
            commands.append(
                CommandSpec(
                    argv=(self.cpptraj.executable, "-i", str(script)),
                    cwd=root,
                    environment=self.cpptraj.environment,
                )
            )
            artifacts[f"structure.{replica}"] = structure
            artifacts[f"trajectory.{replica}"] = display_trajectory
            artifacts[f"cpptraj_input.{replica}"] = script
            artifacts[f"pymol_script.{replica}"] = pymol
            if request.reference_trajectories is not None:
                artifacts[f"ligand_equilibrated_rmsd.{replica}"] = diagnostic

        manifest = root / "display_assets.json"
        generated.append(
            GeneratedInput(
                manifest.relative_to(root),
                _render_manifest(request),
            )
        )
        artifacts["asset_manifest"] = manifest
        return tuple(generated), tuple(commands), artifacts

    def run(
        self,
        request: DisplayAssetRequest,
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
        (root / "structures").mkdir(parents=True, exist_ok=True)
        (root / "trajectories").mkdir(parents=True, exist_ok=True)
        if request.reference_trajectories is not None:
            (root / "diagnostics").mkdir(parents=True, exist_ok=True)
        records = execute_commands(commands, runner=self.runner)
        require_artifacts(artifacts)
        parsed = {
            label.removeprefix("ligand_equilibrated_rmsd."): (parse_cpptraj_table(path))
            for label, path in artifacts.items()
            if label.startswith("ligand_equilibrated_rmsd.")
        }
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=records,
            artifacts=artifacts,
            metadata={
                "tool": identity,
                "replicas": tuple(sorted(request.trajectories)),
                "default_replica": (request.default_replica or sorted(request.trajectories)[0]),
                "default_state": request.default_state,
            },
            parsed={"ligand_equilibrated_rmsd": parsed} if parsed else {},
            generated_inputs=generated,
        )


def create_display_asset_backend(
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None = None,
) -> DisplayAssetBackend:
    return DisplayAssetBackend(
        cpptraj=tool_from_config(
            config,
            key="cpptraj",
            default_executable="cpptraj",
            default_version_arguments=("-V",),
        ),
        runner=runner,
    )


def _render_cpptraj(
    request: DisplayAssetRequest,
    replica: str,
    trajectory: Path,
    structure: Path,
    display_trajectory: Path,
    diagnostic: Path,
    reference_structure: Path,
) -> str:
    end = "last" if request.end_frame is None else str(request.end_frame)
    lines = [
        f'parm "{Path(request.complex_topology).resolve()}"',
    ]
    if request.reference_trajectories is not None:
        lines.extend(
            (
                f'reference "{reference_structure}" 1 name equilibrated_pose',
                (
                    f'trajin "{Path(trajectory).resolve()}" '
                    f"{request.start_frame} {end} {request.stride}"
                ),
                (
                    "rms equilibrated_pose_fit "
                    f"{_atom_mask(request.reference_alignment_atom_indices)} "
                    "ref equilibrated_pose mass"
                ),
                (
                    "rms ligand_equilibrated "
                    f"{_atom_mask(request.ligand_rmsd_atom_indices)} "
                    f'ref equilibrated_pose nofit mass out "{diagnostic}"'
                ),
            )
        )
    else:
        lines.append(
            f'trajin "{Path(trajectory).resolve()}" {request.start_frame} {end} {request.stride}'
        )
        if request.alignment_mask is not None:
            lines.extend(
                (
                    f'reference "{Path(trajectory).resolve()}" 1',
                    f"rms display_fit {request.alignment_mask} reference",
                )
            )
    lines.extend(
        (
            f'trajout "{structure}" pdb onlyframes 1',
            f'trajout "{display_trajectory}" xtc',
            "run",
            "",
        )
    )
    return "\n".join(lines)


def _render_reference_cpptraj(
    request: DisplayAssetRequest,
    replica: str,
    reference_structure: Path,
) -> str:
    if request.reference_trajectories is None:
        raise BackendConfigurationError("display reference trajectory is unavailable")
    reference = request.reference_trajectories[replica]
    lines = [
        f'parm "{Path(request.complex_topology).resolve()}"',
        f'trajin "{Path(reference).resolve()}" {request.reference_frame} {request.reference_frame}',
    ]
    if request.reference_artifact == "selected_trajectory":
        lines.append(
            "autoimage anchor "
            f"{_atom_mask(request.reference_autoimage_anchor_atom_indices)} "
            "fixed "
            f"{_atom_mask(request.reference_autoimage_fixed_atom_indices)}"
        )
    lines.extend((f'trajout "{reference_structure}" pdb', "run", ""))
    return "\n".join(lines)


def _render_pymol(request: DisplayAssetRequest, replica: str) -> str:
    if replica not in request.trajectories:
        raise BackendConfigurationError(f"unknown display replica: {replica}")
    object_name = f"replica_{replica.replace('.', '_').replace('-', '_')}"
    lines = [
        "# Generated by mmpbsa.backends 0.7; edit presentation here.",
        "reinitialize",
        "set defer_builds_mode, 3",
        f"load {json.dumps(str(Path('structures') / f'{replica}.pdb'))}, {object_name}",
        (
            f"load_traj {json.dumps(str(Path('trajectories') / f'{replica}.xtc'))}, "
            f"{object_name}, state=1"
        ),
    ]
    for component, indices in sorted(request.component_atom_indices.items()):
        selection = "+".join(str(index) for index in indices)
        selection_name = f"{object_name}_{component.replace('.', '_').replace('-', '_')}"
        lines.append(f"select {selection_name}, {object_name} and index {selection}")
    lines.extend(
        (
            f"frame {request.default_state}",
            "orient",
            "",
        )
    )
    return "\n".join(lines)


def _render_manifest(request: DisplayAssetRequest) -> str:
    payload = {
        "schema_version": "mmpbsa.display_assets.v4",
        "topologies": {
            replica: f"structures/{replica}.pdb" for replica in sorted(request.trajectories)
        },
        "trajectories": {
            replica: f"trajectories/{replica}.xtc" for replica in sorted(request.trajectories)
        },
        "components": {
            component: list(indices)
            for component, indices in sorted(request.component_atom_indices.items())
        },
        "pymol_scripts": {
            replica: f"load_{replica}.pml" for replica in sorted(request.trajectories)
        },
        "default_replica": (request.default_replica or sorted(request.trajectories)[0]),
        "default_state": request.default_state,
    }
    if request.reference_trajectories is not None:
        payload["equilibrated_pose_rmsd"] = {
            "reference_artifact": request.reference_artifact,
            "reference_frame": request.reference_frame,
            "reference_time_ps": 0.0,
            "reference_semantics": "post_npt_production_start",
            "reference_preprocessing": (
                "whole_molecules_then_autoimage_ligand_anchor_receptor_fixed"
                if request.reference_artifact == "prepared_reference_trajectory"
                else "autoimage_ligand_anchor_receptor_fixed"
            ),
            "reference_source": {
                "artifact_role": request.reference_artifact,
                "scope": "same_case_and_replica",
            },
            "alignment_atom_indices": list(request.reference_alignment_atom_indices or ()),
            "autoimage_anchor_atom_indices": list(
                request.reference_autoimage_anchor_atom_indices or ()
            ),
            "autoimage_fixed_atom_indices": list(
                request.reference_autoimage_fixed_atom_indices or ()
            ),
            "ligand_atom_indices": list(request.ligand_rmsd_atom_indices or ()),
            "replicas": {
                replica: (f"diagnostics/{replica}_ligand_equilibrated_rmsd.dat")
                for replica in sorted(request.trajectories)
            },
        }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _atom_mask(indices: tuple[int, ...] | None) -> str:
    if not indices:
        raise BackendConfigurationError("display atom mask cannot be empty")
    return "@" + ",".join(str(index) for index in indices)

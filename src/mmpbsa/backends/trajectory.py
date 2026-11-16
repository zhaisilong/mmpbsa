"""GROMACS-to-cpptraj trajectory preparation adapter."""

from __future__ import annotations

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
from .errors import BackendConfigurationError
from .versioning import ToolConfig, probe_tool


@dataclass(frozen=True, slots=True)
class TrajectoryPreparationRequest:
    source_tpr: Path
    source_trajectory: Path
    index: Path
    complex_topology: Path
    output_dir: Path
    group_name: str
    start_frame: int
    end_frame: int | None
    stride: int
    autoimage_anchor_mask: str | None = None
    autoimage_fixed_mask: str | None = None
    fit_mask: str | None = None
    expected_frames: int | None = None

    def validate(self) -> None:
        require_file(self.source_tpr, "trajectory source TPR")
        require_file(self.source_trajectory, "trajectory source trajectory")
        require_file(self.index, "trajectory GROMACS index")
        require_file(self.complex_topology, "trajectory complex topology")
        if not self.group_name.strip() or any(
            token in self.group_name for token in ("\n", "\r", "\x00")
        ):
            raise BackendConfigurationError("trajectory group_name is invalid")
        for label, value in (("start_frame", self.start_frame), ("stride", self.stride)):
            if isinstance(value, bool) or value < 1:
                raise BackendConfigurationError(f"trajectory {label} must be positive")
        if self.end_frame is not None and (
            isinstance(self.end_frame, bool)
            or self.end_frame < self.start_frame
        ):
            raise BackendConfigurationError(
                "trajectory end_frame must not precede start_frame"
            )
        if self.expected_frames is not None:
            if isinstance(self.expected_frames, bool) or self.expected_frames < 1:
                raise BackendConfigurationError("expected_frames must be positive")
            if self.end_frame is not None:
                observed = 1 + (self.end_frame - self.start_frame) // self.stride
                if observed != self.expected_frames:
                    raise BackendConfigurationError(
                        "trajectory selection does not produce expected_frames"
                    )
        masks = (self.autoimage_anchor_mask, self.autoimage_fixed_mask)
        if (masks[0] is None) != (masks[1] is None):
            raise BackendConfigurationError(
                "autoimage anchor and fixed masks must be supplied together"
            )
        for mask in (*masks, self.fit_mask):
            if mask is not None and (
                not mask.strip()
                or any(
                    token in mask
                    for token in ("\n", "\r", "\x00", ";")
                )
            ):
                raise BackendConfigurationError("trajectory mask is unsafe")


@dataclass(frozen=True, slots=True)
class TrajectoryPreparationBackend:
    gromacs: ToolConfig
    cpptraj: ToolConfig
    runner: CommandRunner | None = None

    def build_commands(
        self,
        request: TrajectoryPreparationRequest,
    ) -> tuple[tuple[GeneratedInput, ...], tuple[CommandSpec, ...], Mapping[str, Path]]:
        request.validate()
        root = Path(request.output_dir).resolve()
        selected = root / "selected.xtc"
        prepared = root / "prepared.nc"
        centered = root / "centered.xtc"
        script = root / "trajectory.in"
        gmx = CommandSpec(
            argv=(
                self.gromacs.executable,
                "trjconv",
                "-s",
                str(Path(request.source_tpr).resolve()),
                "-f",
                str(Path(request.source_trajectory).resolve()),
                "-n",
                str(Path(request.index).resolve()),
                "-o",
                str(selected),
                "-pbc",
                "mol",
                "-ur",
                "compact",
            ),
            cwd=root,
            environment=self.gromacs.environment,
            stdin=request.group_name + "\n",
        )
        cpptraj_text = _render_cpptraj_input(
            request,
            selected=selected,
            prepared=prepared,
            centered=centered,
        )
        cpptraj = CommandSpec(
            argv=(self.cpptraj.executable, "-i", str(script)),
            cwd=root,
            environment=self.cpptraj.environment,
        )
        return (
            (GeneratedInput(Path(script.name), cpptraj_text),),
            (gmx, cpptraj),
            {
                "selected_trajectory": selected,
                "prepared_trajectory": prepared,
                "centered_trajectory": centered,
                "cpptraj_input": script,
            },
        )

    def run(
        self,
        request: TrajectoryPreparationRequest,
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
        identities = {
            "gromacs": probe_tool(self.gromacs, cwd=root, runner=self.runner),
            "cpptraj": probe_tool(self.cpptraj, cwd=root, runner=self.runner),
        }
        materialize_inputs(root, generated)
        records = execute_commands(commands, runner=self.runner)
        require_artifacts(artifacts)
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=records,
            artifacts=artifacts,
            metadata={
                "tools": identities,
                "selection": {
                    "start_frame": request.start_frame,
                    "end_frame": request.end_frame,
                    "stride": request.stride,
                    "expected_frames": request.expected_frames,
                },
            },
            generated_inputs=generated,
        )


def create_trajectory_backend(
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None = None,
) -> TrajectoryPreparationBackend:
    return TrajectoryPreparationBackend(
        gromacs=tool_from_config(
            config,
            key="gromacs",
            default_executable="gmx",
        ),
        cpptraj=tool_from_config(
            config,
            key="cpptraj",
            default_executable="cpptraj",
            default_version_arguments=("-V",),
        ),
        runner=runner,
    )


def _render_cpptraj_input(
    request: TrajectoryPreparationRequest,
    *,
    selected: Path,
    prepared: Path,
    centered: Path,
) -> str:
    end = "last" if request.end_frame is None else str(request.end_frame)
    lines = [
        f'parm "{Path(request.complex_topology).resolve()}"',
        f'trajin "{selected}" {request.start_frame} {end} {request.stride}',
        f'reference "{selected}" 1',
    ]
    if request.autoimage_anchor_mask is not None:
        lines.append(
            f"autoimage anchor {request.autoimage_anchor_mask} "
            f"fixed {request.autoimage_fixed_mask}"
        )
    if request.fit_mask is not None:
        lines.append(f"rms fit {request.fit_mask} reference")
    lines.extend(
        (
            f'trajout "{prepared}" netcdf',
            f'trajout "{centered}" xtc',
            "run",
            "",
        )
    )
    return "\n".join(lines)

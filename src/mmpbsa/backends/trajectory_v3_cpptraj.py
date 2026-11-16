"""Separate physical NetCDF and fitted visualization CPPTRAJ streams."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trajectory_v3 import TrajectoryPreparationV3Request


def _window(request: TrajectoryPreparationV3Request) -> str:
    result = f"start {request.start_frame}"
    if request.end_frame is not None:
        result += f" stop {request.end_frame}"
    return result + f" offset {request.stride}"


def _imaging(request: TrajectoryPreparationV3Request, source: Path) -> list[str]:
    lines = [
        f'parm "{Path(request.dry_complex_topology).resolve()}"',
        f'trajin "{source}" 1 last 1',
    ]
    if request.autoimage_anchor_mask is not None:
        lines.append(
            f"autoimage anchor {request.autoimage_anchor_mask} fixed {request.autoimage_fixed_mask}"
        )
    return lines


def _render_cpptraj_input(
    request: TrajectoryPreparationV3Request,
    *,
    selected: Path,
    prepared_reference: Path,
    prepared: Path,
    centered: Path,
) -> str:
    # Never rotate physical coordinates: NetCDF cannot store a rotated cell.
    lines = _imaging(request, selected)
    lines.extend(
        (
            f'trajout "{prepared_reference}" netcdf onlyframes 1',
            f'trajout "{prepared}" netcdf {_window(request)}',
            "run",
            "",
        )
    )
    return "\n".join(lines)


def _render_visualization_cpptraj_input(
    request: TrajectoryPreparationV3Request, *, whole: Path, centered: Path
) -> str:
    lines = _imaging(request, whole)
    if request.fit_mask is not None:
        lines.append(f"rms fit first {request.fit_mask}")
    lines.extend((f'trajout "{centered}" xtc {_window(request)}', "run", ""))
    return "\n".join(lines)


def _reference_preprocessing(request: TrajectoryPreparationV3Request) -> str:
    return (
        "whole_molecules_then_autoimage_ligand_anchor_receptor_fixed"
        if request.autoimage_anchor_mask is not None
        else "whole_molecules"
    )

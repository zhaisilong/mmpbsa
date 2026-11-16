"""Closed, sealed GROMACS topology dependency inventories.

GROMACS resolves quoted includes relative to the containing topology.  The
prepared package must declare every file in that recursive closure; otherwise
the input identity cannot be content addressed or copied into an attempt.
"""

from __future__ import annotations

import posixpath
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

from mmpbsa.utils import sha256_file

from .errors import ArtifactIntegrityError, ContractError

_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"\n]+)"\s*$')
_INCLUDE_DIRECTIVE = re.compile(r"^\s*#\s*include\b")


@dataclass(frozen=True, slots=True)
class AmberPairValidation:
    atom_count: int
    bond_count: int
    maximum_coordinate_delta_nm: float
    maximum_box_vector_delta_nm: float
    suppressed_pair_function_warnings: int


def validate_amber_md_pair(
    md_topology: tuple[Path, str],
    md_coordinates: tuple[Path, str],
    amber_topology: tuple[Path, str],
    amber_coordinates: tuple[Path, str],
) -> AmberPairValidation:
    """Check that sealed AMBER files describe the same atoms and initial pose.

    This checks structural identity, not force-field parameter equivalence.
    Coordinates and box vectors use nanometers at the public boundary.
    """

    files = (md_topology, md_coordinates, amber_topology, amber_coordinates)
    for path, expected_sha256 in files:
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ArtifactIntegrityError(f"PreparedCase MD input changed: {path}")
    return _validate_amber_md_pair_cached(
        tuple((str(path), digest) for path, digest in files)
    )


@lru_cache(maxsize=16)
def _validate_amber_md_pair_cached(
    files: tuple[tuple[str, str], ...],
) -> AmberPairValidation:
    try:
        import numpy as np
        import parmed as pmd
        from parmed.gromacs.gromacstop import GromacsWarning
    except ImportError as exc:
        raise ContractError(
            "OpenMM Amber input preflight requires ParmEd and NumPy"
        ) from exc

    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter("always")
        try:
            gromacs = pmd.load_file(files[0][0], xyz=files[1][0])
            amber = pmd.load_file(files[2][0], xyz=files[3][0])
        except Exception as exc:
            raise ContractError("cannot parse paired GROMACS/AMBER MD inputs") from exc
    suppressed = 0
    for warning in observed_warnings:
        if (
            warning.category is GromacsWarning
            and str(warning.message) == "pairs funct != 1; unknown functional"
        ):
            suppressed += 1
        else:
            warnings.warn_explicit(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
            )

    if len(gromacs.atoms) != len(amber.atoms):
        raise ContractError("GROMACS/AMBER MD atom counts differ")
    for index, (left, right) in enumerate(
        zip(gromacs.atoms, amber.atoms, strict=True), start=1
    ):
        left_id = (
            left.residue.idx,
            left.residue.name,
            left.name,
            int(left.atomic_number or 0),
        )
        right_id = (
            right.residue.idx,
            right.residue.name,
            right.name,
            int(right.atomic_number or 0),
        )
        if left_id != right_id:
            raise ContractError(
                f"GROMACS/AMBER MD atom order differs at atom {index}: "
                f"{left_id!r} != {right_id!r}"
            )

    def bond_pairs(structure: object) -> set[tuple[int, int]]:
        return {
            tuple(sorted((bond.atom1.idx, bond.atom2.idx)))
            for bond in structure.bonds
        }

    gromacs_bonds = bond_pairs(gromacs)
    if gromacs_bonds != bond_pairs(amber):
        raise ContractError("GROMACS/AMBER MD bond graphs differ")
    if gromacs.coordinates is None or amber.coordinates is None:
        raise ContractError("GROMACS/AMBER MD initial coordinates are missing")
    left_xyz = np.asarray(gromacs.coordinates, dtype=float)
    right_xyz = np.asarray(amber.coordinates, dtype=float)
    if (
        left_xyz.shape != right_xyz.shape
        or not np.isfinite(left_xyz).all()
        or not np.isfinite(right_xyz).all()
    ):
        raise ContractError("GROMACS/AMBER MD initial coordinates are invalid")
    coordinate_delta_nm = float(np.max(np.abs(left_xyz - right_xyz))) / 10.0
    if coordinate_delta_nm > 0.00051:
        raise ContractError(
            "GROMACS/AMBER MD initial coordinates differ by more than 0.00051 nm"
        )

    def box_vectors(structure: object) -> object:
        box = structure.box
        if box is None or len(box) != 6:
            raise ContractError("GROMACS/AMBER MD periodic box is missing")
        a, b, c = (float(value) for value in box[:3])
        alpha, beta, gamma = (np.deg2rad(float(value)) for value in box[3:])
        sin_gamma = float(np.sin(gamma))
        if min(a, b, c) <= 0 or abs(sin_gamma) < 1.0e-8:
            raise ContractError("GROMACS/AMBER MD periodic box is invalid")
        cx = c * float(np.cos(beta))
        cy = c * (float(np.cos(alpha)) - float(np.cos(beta) * np.cos(gamma))) / sin_gamma
        cz2 = c * c - cx * cx - cy * cy
        if cz2 < -1.0e-6:
            raise ContractError("GROMACS/AMBER MD periodic box is invalid")
        return np.asarray(
            [
                [a, 0.0, 0.0],
                [b * float(np.cos(gamma)), b * sin_gamma, 0.0],
                [cx, cy, float(np.sqrt(max(0.0, cz2)))],
            ],
            dtype=float,
        )

    box_delta_nm = float(np.max(np.abs(box_vectors(gromacs) - box_vectors(amber)))) / 10.0
    if box_delta_nm > 1.0e-4:
        raise ContractError(
            "GROMACS/AMBER MD box vectors differ by more than 0.0001 nm"
        )
    return AmberPairValidation(
        atom_count=len(gromacs.atoms),
        bond_count=len(gromacs_bonds),
        maximum_coordinate_delta_nm=coordinate_delta_nm,
        maximum_box_vector_delta_nm=box_delta_nm,
        suppressed_pair_function_warnings=suppressed,
    )


def topology_include_closure(
    topology_path: Path,
    sealed_files: Mapping[Path, Path],
) -> tuple[Path, ...]:
    """Return transitive include paths relative to the sealed package root.

    ``sealed_files`` maps manifest relative paths to their verified source
    files.  Angle bracket, macro, absolute, escaping, and undeclared includes
    are intentionally rejected because their actual bytes are not known.
    """

    normalized = {Path(key): Path(value) for key, value in sealed_files.items()}
    if topology_path not in normalized:
        raise ContractError("GROMACS topology is absent from sealed artifacts")
    visited: set[Path] = set()
    active: set[Path] = set()
    closure: set[Path] = set()

    def visit(relative: Path) -> None:
        if relative in active:
            raise ContractError(f"cyclic GROMACS topology include: {relative}")
        if relative in visited:
            return
        active.add(relative)
        source = normalized[relative]
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ContractError(f"cannot read GROMACS topology include: {source}") from exc
        for lineno, line in enumerate(lines, start=1):
            directive = line.split(";", 1)[0]
            if not _INCLUDE_DIRECTIVE.match(directive):
                continue
            match = _INCLUDE.fullmatch(directive)
            if match is None:
                raise ContractError(
                    f"GROMACS include must be a quoted literal at {relative}:{lineno}"
                )
            name = match.group(1)
            if "\\" in name or PurePosixPath(name).is_absolute():
                raise ContractError(
                    f"unsafe GROMACS topology include at {relative}:{lineno}"
                )
            target_name = posixpath.normpath(
                posixpath.join(relative.parent.as_posix(), name)
            )
            target = Path(target_name)
            if target_name in {"", ".", ".."} or target_name.startswith("../"):
                raise ContractError(
                    f"GROMACS topology include escapes package at {relative}:{lineno}"
                )
            if target not in normalized:
                raise ContractError(
                    f"undeclared GROMACS topology include {target} at "
                    f"{relative}:{lineno}"
                )
            closure.add(target)
            visit(target)
        active.remove(relative)
        visited.add(relative)

    visit(topology_path)
    return tuple(sorted(closure, key=lambda path: path.as_posix()))

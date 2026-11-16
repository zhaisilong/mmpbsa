"""Engine-independent, topology-authoritative periodic geometry checks.

Coordinates are never optimized here: only lattice translations are applied.
MDTraj is imported lazily, and the bond graph comes from the chemical topology,
not connectivity inferred from coordinate names or residue templates.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from .errors import BackendOutputError

PREPROCESSING_REVISION = "whole_molecules_v1"
BOND_VECTOR_TOLERANCE_NM = 0.005  # 0.05 Angstrom; includes XTC quantization.


def chemical_topology(path: Path) -> Any:
    import mdtraj as md
    import parmed as pmd

    structure = pmd.load_file(str(path))
    topology = md.Topology()
    chain = topology.add_chain()
    atoms = []
    for residue in structure.residues:
        target = topology.add_residue(residue.name, chain, resSeq=residue.idx + 1)
        for atom in residue.atoms:
            number = int(atom.atomic_number or 0)
            element = md.element.Element.getByAtomicNumber(number) if number else None
            atoms.append(topology.add_atom(atom.name, element, target))
    for bond in structure.bonds:
        topology.add_bond(atoms[bond.atom1.idx], atoms[bond.atom2.idx])
    return topology


def _bond_vectors(chunk: Any, bonds: Any) -> Any:
    return chunk.xyz[:, bonds[:, 1], :] - chunk.xyz[:, bonds[:, 0], :]


def _spanning_bonds(bonds: Any, atom_count: int) -> Any:
    """Deterministic graph traversal; do not depend on MDTraj private methods."""
    import numpy as np

    adjacent: list[list[int]] = [[] for _ in range(atom_count)]
    for a, b in bonds:
        adjacent[a].append(int(b))
        adjacent[b].append(int(a))
    visited: set[int] = set()
    result = []
    for root in range(atom_count):
        if root in visited:
            continue
        visited.add(root)
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacent[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    result.append((current, neighbor))
    return np.asarray(result, dtype=np.int32).reshape(-1, 2)


def _check_vectors(
    actual: Any,
    expected: Any,
    bonds: Any,
    *,
    label: str,
    frame_offset: int,
    tolerance_nm: float = BOND_VECTOR_TOLERANCE_NM,
) -> float:
    import numpy as np

    errors = np.linalg.norm(actual - expected, axis=2)
    maximum = float(errors.max(initial=0.0))
    if maximum > tolerance_nm:
        frame, bond = np.unravel_index(int(errors.argmax()), errors.shape)
        first, second = bonds[bond]
        raise BackendOutputError(
            f"{label}: periodic bond integrity failure at frame {frame_offset + frame + 1}, "
            f"atoms {first + 1}-{second + 1}; bond-vector error {maximum * 10:.6f} Angstrom"
        )
    return maximum


def reconstruct(
    source: Path, destination: Path, topology_path: Path, *, chunk_frames: int
) -> dict[str, object]:
    """Make each connected molecule whole, preserving raw projection separately."""
    import mdtraj as md
    import numpy as np
    from mdtraj.formats import XTCTrajectoryFile

    topology = chemical_topology(topology_path)
    bonds = np.asarray([(a.index, b.index) for a, b in topology.bonds], dtype=np.int32).reshape(
        -1, 2
    )
    sorted_bonds = _spanning_bonds(bonds, topology.n_atoms)
    frames = 0
    maximum = 0.0
    raw_maximum = 0.0
    with (
        XTCTrajectoryFile(str(source), "r") as reader,
        XTCTrajectoryFile(str(destination), "w") as writer,
    ):
        while True:
            xyz, times, steps, boxes = reader.read(n_frames=chunk_frames)
            if not len(xyz):
                break
            if not np.isfinite(xyz).all() or not np.isfinite(boxes).all():
                raise BackendOutputError("whole-molecule source has non-finite coordinates or box")
            chunk = md.Trajectory(xyz, topology, time=times)
            chunk.unitcell_vectors = boxes
            if len(bonds):
                minimum_image = md.compute_displacements(chunk, bonds, periodic=True)
                raw_maximum = max(
                    raw_maximum, float(np.linalg.norm(_bond_vectors(chunk, bonds), axis=2).max())
                )
                chunk.make_molecules_whole(inplace=True, sorted_bonds=sorted_bonds)
                maximum = max(
                    maximum,
                    _check_vectors(
                        _bond_vectors(chunk, bonds),
                        minimum_image,
                        bonds,
                        label=str(source),
                        frame_offset=frames,
                    ),
                )
            writer.write(chunk.xyz, time=times, step=steps, box=boxes)
            frames += len(xyz)
    if not frames:
        raise BackendOutputError("whole-molecule source has no frames")
    return {
        "revision": PREPROCESSING_REVISION,
        "status": "passed",
        "frame_count": frames,
        "bond_count": len(bonds),
        "raw_maximum_bond_angstrom": raw_maximum * 10,
        "maximum_bond_vector_error_angstrom": maximum * 10,
        "bond_vector_tolerance_angstrom": BOND_VECTOR_TOLERANCE_NM * 10,
    }


def validate(
    trajectory: Path, topology_path: Path, *, chunk_frames: int, require_whole: bool = True
) -> dict[str, object]:
    """Inspect all bonds, including those outside the RMSD/contact selection."""
    import mdtraj as md
    import numpy as np

    topology = chemical_topology(topology_path)
    bonds = np.asarray([(a.index, b.index) for a, b in topology.bonds], dtype=np.int32).reshape(
        -1, 2
    )
    frames = 0
    maximum = 0.0
    maximum_bond = 0.0
    for chunk in md.iterload(str(trajectory), top=topology, chunk=chunk_frames):
        if not np.isfinite(chunk.xyz).all():
            raise BackendOutputError(f"{trajectory}: non-finite coordinates")
        if len(bonds):
            vectors = _bond_vectors(chunk, bonds)
            maximum_bond = max(maximum_bond, float(np.linalg.norm(vectors, axis=2).max()))
            if require_whole:
                if chunk.unitcell_vectors is None:
                    raise BackendOutputError(f"{trajectory}: periodic box is missing")
                maximum = max(
                    maximum,
                    _check_vectors(
                        vectors,
                        md.compute_displacements(chunk, bonds, periodic=True),
                        bonds,
                        label=str(trajectory),
                        frame_offset=frames,
                    ),
                )
        frames += len(chunk)
    if not frames:
        raise BackendOutputError(f"{trajectory}: no frames")
    return {
        "revision": PREPROCESSING_REVISION,
        "status": "passed",
        "frame_count": frames,
        "bond_count": len(bonds),
        "maximum_bond_angstrom": maximum_bond * 10,
        "maximum_bond_vector_error_angstrom": maximum * 10,
    }


def validate_rigid_geometry(
    physical: Path, fitted: Path, topology_path: Path, *, chunk_frames: int
) -> dict[str, object]:
    """Check all per-bond lengths before/after fitting, including rotated boxes."""
    import mdtraj as md
    import numpy as np

    topology = chemical_topology(topology_path)
    bonds = np.asarray([(a.index, b.index) for a, b in topology.bonds], dtype=np.int32).reshape(
        -1, 2
    )
    frames = 0
    maximum = 0.0
    for first, second in zip(
        md.iterload(str(physical), top=topology, chunk=chunk_frames),
        md.iterload(str(fitted), top=topology, chunk=chunk_frames),
        strict=True,
    ):
        if len(first) != len(second):
            raise BackendOutputError(
                "fitted trajectory frame count differs from physical trajectory"
            )
        if len(bonds):
            errors = np.abs(
                np.linalg.norm(_bond_vectors(first, bonds), axis=2)
                - np.linalg.norm(_bond_vectors(second, bonds), axis=2)
            )
            maximum = max(maximum, float(errors.max(initial=0.0)))
            if maximum > BOND_VECTOR_TOLERANCE_NM:
                frame, bond = np.unravel_index(int(errors.argmax()), errors.shape)
                a, b = bonds[bond]
                raise BackendOutputError(
                    f"fitted geometry changed at frame {frames + frame + 1}, atoms {a + 1}-{b + 1}"
                )
        frames += len(first)
    return {
        "revision": PREPROCESSING_REVISION,
        "status": "passed",
        "frame_count": frames,
        "maximum_bond_length_error_angstrom": maximum * 10,
    }

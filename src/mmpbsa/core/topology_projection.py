"""Backend-neutral topology projection comparison.

The functions operate on ParmEd-like objects by protocol/duck typing but do
not import ParmEd or a concrete mmpbsa backend.  Their semantics intentionally
match ``mmpbsa.backends.parmed`` admission: atom identities and non-bonded
parameters must project exactly within tolerance, as must every internal
bonded term.  Keep the two entry points contract-equivalent until the backend
can migrate to this core-owned helper without changing its public API.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence


def _atom_record(atom: object) -> tuple[object, ...]:
    atom_type = getattr(atom, "atom_type", None)
    return (
        str(atom.residue.name),
        str(atom.name),
        int(getattr(atom, "atomic_number", 0) or 0),
        str(getattr(atom, "type", "")),
        float(getattr(atom, "charge", 0.0)),
        float(getattr(atom, "mass", 0.0)),
        float(getattr(atom_type, "epsilon", getattr(atom, "epsilon", 0.0)) or 0.0),
        float(getattr(atom_type, "rmin", getattr(atom, "rmin", 0.0)) or 0.0),
    )


def _term_parameters(value: object) -> tuple[str | int | float, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[str | int | float] = ["terms", len(value)]
        for index, item in enumerate(value):
            result.extend(("term", index, *_term_parameters(item)))
        return tuple(result)
    result = []
    for attribute in (
        "k",
        "req",
        "theteq",
        "phi_k",
        "per",
        "phase",
        "scee",
        "scnb",
    ):
        if hasattr(value, attribute):
            item = getattr(value, attribute)
            result.extend(
                (
                    attribute,
                    float(item) if isinstance(item, (int, float)) else str(item),
                )
            )
    return tuple(result)


def _canonical_term_atoms(kind: str, atoms: tuple[int, ...]) -> tuple[int, ...]:
    reverse = tuple(reversed(atoms))
    if kind in {"bond", "angle", "dihedral"} and reverse < atoms:
        return reverse
    return atoms


def _canonical_projection_atoms(kind: str, atoms: tuple[int, ...]) -> tuple[int, ...]:
    # A complete four-atom reversal leaves the signed torsion unchanged.
    # Amber serialization may reverse an improper when a local atom index is
    # zero (the sign encodes improper/exclusion flags). Do not accept other
    # improper permutations, which can change the torsion. Keep historical
    # topology signatures unchanged; normalize only physical comparisons.
    if kind == "improper":
        return min(atoms, tuple(reversed(atoms)))
    return _canonical_term_atoms(kind, atoms)


def _term_record(
    kind: str, term: object
) -> tuple[str, tuple[int, ...], tuple[str | int | float, ...]]:
    atoms = tuple(
        int(getattr(term, name).idx) + 1
        for name in ("atom1", "atom2", "atom3", "atom4")
        if hasattr(term, name)
    )
    atoms = _canonical_term_atoms(kind, atoms)
    return kind, atoms, _term_parameters(getattr(term, "type", None))


def _topology_terms(
    structure: object,
) -> tuple[tuple[str, tuple[int, ...], tuple[str | int | float, ...]], ...]:
    terms = [
        *(_term_record("bond", item) for item in getattr(structure, "bonds", ())),
        *(_term_record("angle", item) for item in getattr(structure, "angles", ())),
        *(
            _term_record(
                "improper" if bool(getattr(item, "improper", False)) else "dihedral",
                item,
            )
            for item in getattr(structure, "dihedrals", ())
        ),
        *(_term_record("improper", item) for item in getattr(structure, "impropers", ())),
    ]
    return tuple(sorted(terms, key=lambda item: (item[0], item[1], item[2])))


def topology_signature(structure: object) -> str:
    payload = {
        "schema": "mmpbsa.parmed_topology.v1",
        "atoms": [_atom_record(atom) for atom in structure.atoms],
        "terms": _topology_terms(structure),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parameters_close(
    left: Sequence[str | int | float],
    right: Sequence[str | int | float],
    tolerance: float,
) -> bool:
    if len(left) != len(right):
        return False
    for first, second in zip(left, right, strict=True):
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            first_value = float(first)
            second_value = float(second)
            if (
                not math.isfinite(first_value)
                or not math.isfinite(second_value)
                or abs(first_value - second_value) > tolerance
            ):
                return False
        elif first != second:
            return False
    return True


def _physical_term_parameters(
    kind: str, parameters: Sequence[str | int | float]
) -> tuple[str | int | float, ...]:
    """Remove serialization-only fields from a bonded-term record.

    Amber does not apply 1-4 electrostatic/LJ scaling to an improper torsion,
    but ParmEd's GROMACS writer attaches the topology defaults to the emitted
    improper type.  Comparing those two unused fields would reject an
    otherwise identical Hamiltonian after a GROMACS round trip.
    """

    if kind != "improper":
        return tuple(parameters)
    result: list[str | int | float] = []
    skip_value = False
    for item in parameters:
        if skip_value:
            skip_value = False
            continue
        if item in {"scee", "scnb"}:
            skip_value = True
            continue
        result.append(item)
    return tuple(result)


def mapped_system_projection_issues(
    source_structure: object,
    target_structure: object,
    source_indices: tuple[int, ...],
    label: str,
    *,
    tolerance: float = 1.0e-6,
) -> tuple[str, ...]:
    """Compare a target topology with an explicitly mapped source subsystem.

    The comparison is mechanical: atom identity/type/LJ parameters and every
    bonded term wholly internal to the mapped subsystem must survive.  The
    looser default than the Amber-to-Amber component check accommodates the
    decimal precision of GROMACS topology serialization, not chemistry.
    """

    if len(source_indices) != len(target_structure.atoms):
        return (f"{label}: atom count differs",)
    if any(index < 1 or index > len(source_structure.atoms) for index in source_indices) or len(
        set(source_indices)
    ) != len(source_indices):
        return (f"{label}: source atom mapping is invalid",)

    issues: list[str] = []
    projected_atoms = tuple(source_structure.atoms[index - 1] for index in source_indices)
    for ordinal, (source, target) in enumerate(
        zip(projected_atoms, target_structure.atoms, strict=True), start=1
    ):
        left = _atom_record(source)
        right = _atom_record(target)
        if left[:4] != right[:4]:
            issues.append(f"{label}: atom identity/type differs at target atom {ordinal}")
            continue
        for parameter, first, second in zip(
            ("charge", "mass", "epsilon", "rmin"),
            left[4:],
            right[4:],
            strict=True,
        ):
            if (
                not math.isfinite(float(first))
                or not math.isfinite(float(second))
                or abs(float(first) - float(second)) > tolerance
            ):
                issues.append(f"{label}: {parameter} differs at target atom {ordinal}")

    local = {source: ordinal for ordinal, source in enumerate(source_indices, start=1)}
    projected: dict[tuple[str, tuple[int, ...]], list[tuple[str | int | float, ...]]] = {}
    for kind, atoms, parameters in _topology_terms(source_structure):
        if not all(index in local for index in atoms):
            continue
        identity = (
            kind,
            _canonical_projection_atoms(kind, tuple(local[index] for index in atoms)),
        )
        projected.setdefault(identity, []).append(_physical_term_parameters(kind, parameters))
    target_terms: dict[tuple[str, tuple[int, ...]], list[tuple[str | int | float, ...]]] = {}
    for kind, atoms, parameters in _topology_terms(target_structure):
        target_terms.setdefault((kind, _canonical_projection_atoms(kind, atoms)), []).append(
            _physical_term_parameters(kind, parameters)
        )

    if projected.keys() != target_terms.keys():
        issues.append(f"{label}: internal term identities differ")
    else:
        for identity in sorted(projected):
            source_parameters = projected[identity]
            unmatched = list(target_terms[identity])
            if len(source_parameters) != len(unmatched):
                issues.append(
                    f"{label}: internal term multiplicity differs for {identity[0]} {identity[1]}"
                )
                continue
            for parameters in source_parameters:
                match = next(
                    (
                        index
                        for index, candidate in enumerate(unmatched)
                        if _parameters_close(parameters, candidate, tolerance)
                    ),
                    None,
                )
                if match is None:
                    issues.append(
                        f"{label}: internal term parameters differ for {identity[0]} {identity[1]}"
                    )
                    break
                unmatched.pop(match)
    return tuple(issues)


def component_projection_issues(
    complex_structure: object,
    component_structure: object,
    indices: tuple[int, ...],
    component: str,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[str, ...]:
    """Compare one serialized component with its complex-topology projection."""

    issues: list[str] = []
    projected_atoms = tuple(complex_structure.atoms[index - 1] for index in indices)
    if len(projected_atoms) != len(component_structure.atoms):
        return (f"{component}: atom count differs",)
    for ordinal, (source, target) in enumerate(
        zip(projected_atoms, component_structure.atoms, strict=True), start=1
    ):
        left = _atom_record(source)
        right = _atom_record(target)
        if left[:4] != right[:4]:
            issues.append(f"{component}: atom identity differs at component atom {ordinal}")
            continue
        for label, first, second in zip(
            ("charge", "mass", "epsilon", "rmin"), left[4:], right[4:], strict=True
        ):
            if (
                not math.isfinite(float(first))
                or not math.isfinite(float(second))
                or abs(float(first) - float(second)) > tolerance
            ):
                issues.append(f"{component}: {label} differs at component atom {ordinal}")

    local = {source: ordinal for ordinal, source in enumerate(indices, start=1)}
    projected_terms = tuple(
        sorted(
            (
                kind,
                _canonical_projection_atoms(kind, tuple(local[index] for index in atoms)),
                parameters,
            )
            for kind, atoms, parameters in _topology_terms(complex_structure)
            if all(index in local for index in atoms)
        )
    )
    component_terms = tuple(
        sorted(
            (kind, _canonical_projection_atoms(kind, atoms), parameters)
            for kind, atoms, parameters in _topology_terms(component_structure)
        )
    )
    if len(projected_terms) != len(component_terms):
        issues.append(f"{component}: internal term count differs")
    else:
        for ordinal, (source, target) in enumerate(
            zip(projected_terms, component_terms, strict=True), start=1
        ):
            if source[:2] != target[:2]:
                issues.append(f"{component}: internal term identity differs at term {ordinal}")
            elif not _parameters_close(source[2], target[2], tolerance):
                issues.append(f"{component}: internal term parameters differ at term {ordinal}")
    return tuple(issues)


def binding_boundary_bond_issues(
    complex_structure: object,
    receptor_indices: tuple[int, ...],
    ligand_indices: tuple[int, ...],
) -> tuple[str, ...]:
    """Report covalent bonds that make endpoint MM/PB(GB)SA undefined.

    This check is derived from the serialized complex topology itself.  A
    draft cannot evade it by omitting a bond from its semantic sidecar.
    """

    receptor = set(receptor_indices)
    ligand = set(ligand_indices)
    issues: list[str] = []
    for bond in getattr(complex_structure, "bonds", ()):
        first = int(bond.atom1.idx) + 1
        second = int(bond.atom2.idx) + 1
        crosses = (first in receptor and second in ligand) or (
            first in ligand and second in receptor
        )
        if crosses:
            left, right = sorted((first, second))
            issues.append(
                f"complex topology contains a receptor-ligand covalent bond: {left}--{right}"
            )
    return tuple(issues)


__all__ = [
    "binding_boundary_bond_issues",
    "component_projection_issues",
    "mapped_system_projection_issues",
    "topology_signature",
]

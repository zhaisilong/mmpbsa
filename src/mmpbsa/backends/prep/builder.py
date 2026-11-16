"""Transactional Amber/ParmEd materialization for typed preparation specs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from importlib import metadata
from pathlib import Path

from mmpbsa.backends.parmed import (
    ParmEdAdmissionRequest,
    ProjectionTolerances,
    create_parmed_admission_backend,
)
from mmpbsa.core import (
    ArtifactRef,
    BindingTopologyRoles,
    EntitySelection,
    PreparedCaseManifest,
    PreparedEntity,
    PreparedMdSystem,
    PreparedScoringSystem,
    PrimaryBinding,
    ResidueIdentity,
    load_prepared_case,
    write_prepared_case_manifest,
)
from mmpbsa.utils import sha256_file

from .amber import AmberAssemblyRequest, create_amber_preparation_tools
from .errors import PreparationError
from .models import PreparationIntent, PreparationSpec
from .planner import (
    BACKBONE_NAMES,
    WATER_NAMES,
    initial_tleap_text,
    inspect_sources,
    parameterization_request,
    residue_chemistry_class,
    salted_tleap_text,
    write_receptor_views,
)

INTEGRAL_CHARGE_TOLERANCE_E = 1.0e-5
PREPARED_CHARGE_TOLERANCE_E = 1.0e-6
MOL2_CHARGE_QUANTUM = Decimal("0.00000001")
MAX_MOL2_CHARGE_RESIDUAL_E = Decimal("0.005")


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_mol2_formal_charge(
    path: Path, *, formal_charge: int
) -> Mapping[str, object]:
    """Remove only a small serialized AM1-BCC residual, preserving symmetry."""

    source = Path(path)
    raw_sha256 = sha256_file(source)
    lines = source.read_text(encoding="utf-8").splitlines()
    atom_rows: list[tuple[int, list[str], Decimal]] = []
    in_atoms = False
    for line_number, line in enumerate(lines):
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atoms = False
            continue
        if not in_atoms or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            raise PreparationError("MOL2 atom row has fewer than nine fields")
        try:
            charge = Decimal(fields[8])
        except InvalidOperation as exc:
            raise PreparationError("MOL2 contains an invalid atom charge") from exc
        atom_rows.append((line_number, fields, charge))
    if not atom_rows:
        raise PreparationError("MOL2 contains no atom charge records")
    target = Decimal(formal_charge)
    raw_charge = sum((item[2] for item in atom_rows), Decimal(0))
    residual = target - raw_charge
    if abs(residual) > MAX_MOL2_CHARGE_RESIDUAL_E:
        raise PreparationError(
            "MOL2 charge residual is too large to normalize safely: "
            f"{abs(residual)} e"
        )
    correction = (residual / Decimal(len(atom_rows))).quantize(
        MOL2_CHARGE_QUANTUM
    )
    corrected = [item[2] + correction for item in atom_rows]
    corrected[-1] += target - sum(corrected, Decimal(0))
    for (line_number, fields, _), charge in zip(atom_rows, corrected, strict=True):
        fields[8] = f"{charge:.8f}"
        lines[line_number] = " ".join(fields)
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    normalized = sum(corrected, Decimal(0))
    if normalized != target:
        raise PreparationError("MOL2 charge normalization did not reach target")
    return {
        "method": "uniform minimum-L2 residual correction",
        "atom_count": len(atom_rows),
        "formal_charge_e": formal_charge,
        "raw_serialized_charge_e": float(raw_charge),
        "absolute_residual_e": float(abs(residual)),
        "correction_per_atom_e": float(correction),
        "normalized_charge_e": float(normalized),
        "raw_mol2_sha256": raw_sha256,
        "normalized_mol2_sha256": sha256_file(source),
    }


def _water_count(structure: object) -> int:
    return sum(
        str(residue.name).upper() in WATER_NAMES for residue in structure.residues
    )


def _salt_inventory(
    structure: object, *, target_molar: float
) -> Mapping[str, object]:
    water_count = _water_count(structure)
    if water_count < 1:
        raise PreparationError("TIP3P solvation produced no water molecules")
    net_charge = float(sum(float(atom.charge) for atom in structure.atoms))
    integral_charge = int(round(net_charge))
    if abs(net_charge - integral_charge) > INTEGRAL_CHARGE_TOLERANCE_E:
        raise PreparationError(
            f"unsolvated ionic charge is not integral: {net_charge:.8f}"
        )
    salt_pairs = int(round(water_count * target_molar / 55.5))
    if salt_pairs < 1:
        raise PreparationError("solvent box is too small to represent requested salt")
    sodium = salt_pairs + max(0, -integral_charge)
    chloride = salt_pairs + max(0, integral_charge)
    return {
        "water_molecules_before_ions": water_count,
        "solute_net_charge_e": net_charge,
        "integral_solute_charge_e": integral_charge,
        "salt_pairs": salt_pairs,
        "sodium_ions": sodium,
        "chloride_ions": chloride,
        "target_molar": target_molar,
        "water_ratio_estimated_molar": salt_pairs * 55.5 / water_count,
        "method": "round(N_water * target / 55.5) plus neutralizing counterions",
    }


def _atom_identity(atom: object) -> tuple[str, str, int]:
    return (
        str(atom.residue.name).upper(),
        str(atom.name),
        int(getattr(atom, "atomic_number", 0) or 0),
    )


def _atom_order_signature(structure: object) -> str:
    return _sha256_json(
        {
            "schema": "mmpbsa.atom_order.v1",
            "atoms": [
                {
                    "residue_index": int(atom.residue.idx) + 1,
                    "residue_name": str(atom.residue.name),
                    "atom_name": str(atom.name),
                    "atomic_number": int(
                        getattr(atom, "atomic_number", 0) or 0
                    ),
                }
                for atom in structure.atoms
            ],
        }
    )


def _md_system_signature(structure: object) -> str:
    bonds = sorted(
        (
            min(int(bond.atom1.idx), int(bond.atom2.idx)) + 1,
            max(int(bond.atom1.idx), int(bond.atom2.idx)) + 1,
        )
        for bond in structure.bonds
    )
    return _sha256_json(
        {
            "schema": "mmpbsa.md_system.v1",
            "atom_order": _atom_order_signature(structure),
            "atom_count": len(structure.atoms),
            "residue_count": len(structure.residues),
            "bonds": bonds,
            "box": (
                None
                if getattr(structure, "box", None) is None
                else [float(value) for value in structure.box]
            ),
        }
    )


def _partition(
    scoring: object,
    *,
    receptor_atom_count: int,
    ligand_atom_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Project the explicit ``combine { receptor ligand }`` assembly order."""

    expected = receptor_atom_count + ligand_atom_count
    if len(scoring.atoms) != expected:
        raise PreparationError(
            "dry topology does not match the authoritative receptor/ligand "
            f"inventories: expected {expected}, observed {len(scoring.atoms)}"
        )
    receptor = tuple(range(1, receptor_atom_count + 1))
    ligand = tuple(range(receptor_atom_count + 1, expected + 1))
    if not receptor or not ligand:
        raise PreparationError("dry complex must contain receptor and ligand atoms")
    return receptor, ligand


def _map_scoring_atoms(
    scoring: object,
    md: object,
) -> tuple[int, ...]:
    """Find the unique strict ordered dry-system projection in the MD order."""

    wanted = tuple(_atom_identity(atom) for atom in scoring.atoms)
    observed = tuple(_atom_identity(atom) for atom in md.atoms)

    forward: list[int] = []
    cursor = 0
    for identity in wanted:
        while cursor < len(observed) and observed[cursor] != identity:
            cursor += 1
        if cursor == len(observed):
            raise PreparationError(
                "dry scoring atom sequence is not an ordered projection of MD"
            )
        forward.append(cursor + 1)
        cursor += 1

    backward: list[int] = []
    cursor = len(observed) - 1
    for identity in reversed(wanted):
        while cursor >= 0 and observed[cursor] != identity:
            cursor -= 1
        if cursor < 0:
            raise PreparationError(
                "dry scoring atom sequence is not an ordered projection of MD"
            )
        backward.append(cursor + 1)
        cursor -= 1
    unique_from_end = tuple(reversed(backward))
    result = tuple(forward)
    if result != unique_from_end:
        raise PreparationError(
            "dry scoring atom sequence has an ambiguous ordered projection in MD"
        )
    return result


def _improper_terms(structure: object) -> tuple[object, ...]:
    dihedrals = tuple(
        term
        for term in getattr(structure, "dihedrals", ())
        if bool(getattr(term, "improper", False))
    )
    return (*dihedrals, *tuple(getattr(structure, "impropers", ())))


def _term_atom_ordinals(term: object) -> tuple[int, int, int, int]:
    return tuple(
        int(getattr(term, name).idx) + 1
        for name in ("atom1", "atom2", "atom3", "atom4")
    )  # type: ignore[return-value]


def _align_serialized_component_impropers(
    scoring: object,
    components: Mapping[str, tuple[object, Sequence[int]]],
) -> Mapping[str, int]:
    complex_terms = _improper_terms(scoring)
    reversals: dict[str, int] = {}
    for component_name, (component, source_indices) in components.items():
        global_to_local = {
            int(source): local
            for local, source in enumerate(source_indices, start=1)
        }
        source_by_atoms: dict[frozenset[int], object] = {}
        for term in complex_terms:
            global_ordinals = _term_atom_ordinals(term)
            if all(index in global_to_local for index in global_ordinals):
                key = frozenset(global_ordinals)
                if key in source_by_atoms:
                    raise PreparationError(
                        f"{component_name} repeats a complex improper atom set"
                    )
                source_by_atoms[key] = term
        target_by_atoms: dict[frozenset[int], tuple[int, ...]] = {}
        for term in _improper_terms(component):
            local_ordinals = _term_atom_ordinals(term)
            try:
                global_ordinals = tuple(
                    int(source_indices[index - 1]) for index in local_ordinals
                )
            except IndexError as exc:
                raise PreparationError("component improper escapes component") from exc
            key = frozenset(global_ordinals)
            if key in target_by_atoms:
                raise PreparationError(
                    f"{component_name} repeats a serialized improper atom set"
                )
            target_by_atoms[key] = local_ordinals
        if set(source_by_atoms) != set(target_by_atoms):
            raise PreparationError(
                f"{component_name} improper inventory changed on serialization"
            )
        count = 0
        for key, source_term in source_by_atoms.items():
            source_global = _term_atom_ordinals(source_term)
            source_local = tuple(global_to_local[index] for index in source_global)
            target_local = target_by_atoms[key]
            if source_local == target_local:
                continue
            if tuple(reversed(source_local)) != target_local:
                raise PreparationError(
                    f"{component_name} improper changed beyond exact reversal"
                )
            (
                source_term.atom1,
                source_term.atom2,
                source_term.atom3,
                source_term.atom4,
            ) = (
                source_term.atom4,
                source_term.atom3,
                source_term.atom2,
                source_term.atom1,
            )
            count += 1
        reversals[component_name] = count
    return reversals


def _write_index(
    path: Path,
    *,
    system_indices: Sequence[int],
    complex_indices: Sequence[int],
    receptor_indices: Sequence[int],
    ligand_indices: Sequence[int],
) -> None:
    def group(name: str, values: Sequence[int]) -> list[str]:
        result = [f"[ {name} ]"]
        for start in range(0, len(values), 15):
            result.append(
                " ".join(str(value) for value in values[start : start + 15])
            )
        return result

    content = [
        *group("System", system_indices),
        "",
        *group("Complex", complex_indices),
        "",
        *group("Receptor", receptor_indices),
        "",
        *group("Ligand", ligand_indices),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def _artifact(
    root: Path, path: Path, role: str, media_type: str
) -> ArtifactRef:
    target = Path(path).resolve()
    try:
        relative = target.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise PreparationError(f"artifact escapes PreparedCase root: {target}") from exc
    if not target.is_file() or target.stat().st_size == 0:
        raise PreparationError(f"artifact is missing or empty: {target}")
    return ArtifactRef(
        path=relative,
        sha256=sha256_file(target),
        size=target.stat().st_size,
        role=role,
        media_type=media_type,
    )


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_jsonable(item) for item in value]
    return str(value)


def _receipt_summary(receipt: object, root: Path) -> Mapping[str, object]:
    prefix = str(Path(root).resolve())

    def portable(value: object) -> object:
        converted = _jsonable(value)
        if isinstance(converted, str):
            return converted.replace(prefix, "${PREPARED_CASE_ROOT}")
        if isinstance(converted, list):
            return [portable(item) for item in converted]
        if isinstance(converted, Mapping):
            return {str(key): portable(item) for key, item in converted.items()}
        return converted

    commands = []
    for command in receipt.commands:
        stdout = str(getattr(command, "stdout", ""))
        stderr = str(getattr(command, "stderr", ""))
        commands.append(
            {
                "argv": list(command.argv),
                "cwd": str(command.cwd),
                "returncode": getattr(command, "returncode", None),
                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            }
        )
    return {
        "state": receipt.state.value,
        "commands": portable(commands),
        "metadata": portable(receipt.metadata),
        "artifacts": portable(receipt.artifacts),
    }


def _entity_selection(
    md_indices: Sequence[int], scoring_indices: Sequence[int]
) -> EntitySelection:
    return EntitySelection(
        md_atom_indices=tuple(sorted(int(item) for item in md_indices)),
        scoring_atom_indices=tuple(sorted(int(item) for item in scoring_indices)),
    )


def _stable_residue_id(chain: str, number: int, insertion: str) -> str:
    chain_text = chain or "chain"
    insertion_text = f"_{insertion}" if insertion else ""
    return f"{chain_text}_{number}{insertion_text}"


def _coordinates(atom: object) -> tuple[float, float, float]:
    return (float(atom.xx), float(atom.xy), float(atom.xz))


def _residues_for_atom_indices(
    structure: object, indices: Sequence[int]
) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[int] = set()
    for index in indices:
        residue = structure.atoms[index - 1].residue
        residue_index = int(residue.idx)
        if residue_index not in seen:
            seen.add(residue_index)
            result.append(residue)
    return tuple(result)


def _receptor_heavy_coordinate_pairs(
    inspection: object,
    scoring: object,
    receptor_indices: Sequence[int],
) -> tuple[tuple[tuple[float, float, float], int], ...]:
    scoring_residues = _residues_for_atom_indices(scoring, receptor_indices)
    if len(scoring_residues) != len(inspection.protein_residue_keys):
        raise PreparationError("cannot map source receptor residues to scoring")
    source_by_residue: dict[
        tuple[str, int, str, str], dict[str, object]
    ] = {}
    for atom in inspection.protein_atoms:
        names = source_by_residue.setdefault(atom.residue_key, {})
        if atom.name in names:
            raise PreparationError(
                f"source receptor repeats atom name {atom.name!r} in one residue"
            )
        names[atom.name] = atom
    pairs: list[tuple[tuple[float, float, float], int]] = []
    for source_key, residue in zip(
        inspection.protein_residue_keys, scoring_residues, strict=True
    ):
        source_atoms = source_by_residue[source_key]
        for atom in residue.atoms:
            if int(getattr(atom, "atomic_number", 0) or 0) == 1:
                continue
            try:
                source_atom = source_atoms[str(atom.name)]
            except KeyError as exc:
                raise PreparationError(
                    "prepared receptor atom cannot be mapped to the source: "
                    f"residue={source_key!r}, atom={atom.name!r}"
                ) from exc
            if not source_atom.heavy:
                raise PreparationError("source/prepared receptor element mismatch")
            pairs.append((source_atom.coordinates, int(atom.idx) + 1))
    if len(pairs) < 4:
        raise PreparationError("receptor has too few heavy atoms for water matching")
    return tuple(pairs)


def _squared_distance(
    left: Sequence[float], right: Sequence[float]
) -> float:
    return sum(
        (first - second) ** 2
        for first, second in zip(left, right, strict=True)
    )


def _spatially_distributed_anchors(
    pairs: Sequence[tuple[tuple[float, float, float], int]],
    *,
    count: int = 32,
) -> tuple[tuple[tuple[float, float, float], int], ...]:
    """Deterministic farthest-point anchors, distributed across the receptor."""

    target = min(count, len(pairs))
    centroid = tuple(
        sum(pair[0][axis] for pair in pairs) / len(pairs)
        for axis in range(3)
    )
    first = max(
        range(len(pairs)),
        key=lambda index: (_squared_distance(pairs[index][0], centroid), -index),
    )
    selected = [first]
    selected_set = {first}
    while len(selected) < target:
        candidate = max(
            (index for index in range(len(pairs)) if index not in selected_set),
            key=lambda index: (
                min(
                    _squared_distance(pairs[index][0], pairs[item][0])
                    for item in selected
                ),
                -index,
            ),
        )
        selected.append(candidate)
        selected_set.add(candidate)
    anchors = tuple(pairs[index] for index in selected)
    first_vector = tuple(
        anchors[1][0][axis] - anchors[0][0][axis] for axis in range(3)
    )
    second_vector = tuple(
        anchors[2][0][axis] - anchors[0][0][axis] for axis in range(3)
    )
    cross = (
        first_vector[1] * second_vector[2]
        - first_vector[2] * second_vector[1],
        first_vector[2] * second_vector[0]
        - first_vector[0] * second_vector[2],
        first_vector[0] * second_vector[1]
        - first_vector[1] * second_vector[0],
    )
    if sum(value * value for value in cross) < 1.0:
        raise PreparationError("receptor water-matching anchors are nearly collinear")
    return anchors


def _distance_fingerprint(
    point: Sequence[float], anchors: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    return tuple(_squared_distance(point, anchor) ** 0.5 for anchor in anchors)


def _match_crystal_waters(
    spec: PreparationSpec,
    *,
    inspection: object,
    scoring: object,
    md: object,
    receptor_indices: Sequence[int],
    scoring_to_md: Mapping[int, int],
) -> tuple[tuple[object, ...], tuple[Mapping[str, object], ...]]:
    coordinate_pairs = _receptor_heavy_coordinate_pairs(
        inspection, scoring, receptor_indices
    )
    anchors = _spatially_distributed_anchors(coordinate_pairs)
    source_anchors = tuple(pair[0] for pair in anchors)
    md_anchors = tuple(
        _coordinates(md.atoms[scoring_to_md[pair[1]] - 1]) for pair in anchors
    )
    water_residues = tuple(
        residue
        for residue in md.residues
        if str(residue.name).upper() in WATER_NAMES
    )
    matched_residues: list[object] = []
    audits: list[Mapping[str, object]] = []
    used_water_indices: set[int] = set()
    for declared in spec.crystal_waters:
        source_heavy = tuple(
            atom
            for atom in inspection.crystal_water_atoms
            if atom.residue_key == declared.source_key and atom.heavy
        )
        if len(source_heavy) != 1:
            raise PreparationError(
                f"crystal water {declared.entity_id!r} must have one source oxygen"
            )
        source_fingerprint = _distance_fingerprint(
            source_heavy[0].coordinates, source_anchors
        )
        ranked: list[tuple[float, object]] = []
        for residue in water_residues:
            if int(residue.idx) in used_water_indices:
                continue
            heavy = tuple(
                atom
                for atom in residue.atoms
                if int(getattr(atom, "atomic_number", 0) or 0) != 1
            )
            if len(heavy) != 1:
                continue
            candidate = _distance_fingerprint(_coordinates(heavy[0]), md_anchors)
            error = (
                sum(
                    (left - right) ** 2
                    for left, right in zip(
                        source_fingerprint, candidate, strict=True
                    )
                )
                / len(source_fingerprint)
            ) ** 0.5
            ranked.append((error, residue))
        if len(ranked) < 2:
            raise PreparationError("too few MD water candidates for unique matching")
        ranked.sort(key=lambda item: (item[0], int(item[1].idx)))
        best_error, matched = ranked[0]
        second_error = ranked[1][0]
        margin = second_error - best_error
        if best_error > 0.02:
            raise PreparationError(
                f"crystal water {declared.entity_id!r} is not preserved: "
                f"anchor fingerprint RMS error={best_error:.6f} A"
            )
        if margin < 0.25:
            raise PreparationError(
                f"crystal water {declared.entity_id!r} mapping is ambiguous: "
                f"best/second margin={margin:.6f} A"
            )
        used_water_indices.add(int(matched.idx))
        matched_residues.append(matched)
        audits.append(
            {
                "entity_id": declared.entity_id,
                "method": "32 spatially distributed receptor-heavy anchor distances",
                "anchor_count": len(anchors),
                "md_residue_index": int(matched.idx) + 1,
                "fingerprint_rms_error_angstrom": best_error,
                "second_best_error_angstrom": second_error,
                "candidate_margin_angstrom": margin,
                "maximum_error_angstrom": 0.02,
                "minimum_margin_angstrom": 0.25,
            }
        )
    return tuple(matched_residues), tuple(audits)


def _build_entities(
    spec: PreparationSpec,
    *,
    scoring: object,
    md: object,
    receptor_indices: tuple[int, ...],
    ligand_indices: tuple[int, ...],
    source_indices: tuple[int, ...],
) -> tuple[tuple[PreparedEntity, ...], tuple[Mapping[str, object], ...]]:
    inspection = inspect_sources(spec)
    scoring_to_md = {
        scoring_index: source_indices[scoring_index - 1]
        for scoring_index in range(1, len(source_indices) + 1)
    }
    receptor_md = tuple(scoring_to_md[index] for index in receptor_indices)
    ligand_md = tuple(scoring_to_md[index] for index in ligand_indices)

    ligand_atoms = tuple(scoring.atoms[index - 1] for index in ligand_indices)
    ligand_heavy = tuple(
        index
        for index, atom in zip(ligand_indices, ligand_atoms, strict=True)
        if int(getattr(atom, "atomic_number", 0) or 0) != 1
    )
    interface_residue_indices: set[int] = set()
    for index in receptor_indices:
        atom = scoring.atoms[index - 1]
        if int(getattr(atom, "atomic_number", 0) or 0) == 1:
            continue
        if any(
            int(getattr(ligand, "atomic_number", 0) or 0) != 1
            and sum(
                (left - right) ** 2
                for left, right in zip(
                    _coordinates(atom), _coordinates(ligand), strict=True
                )
            )
            <= spec.interface_cutoff_angstrom**2
            for ligand in ligand_atoms
        ):
            interface_residue_indices.add(int(atom.residue.idx) + 1)
    if not interface_residue_indices:
        raise PreparationError("prepared receptor interface is empty")

    receptor_heavy = tuple(
        index
        for index in receptor_indices
        if int(getattr(scoring.atoms[index - 1], "atomic_number", 0) or 0) != 1
    )
    receptor_backbone = tuple(
        index
        for index in receptor_indices
        if str(scoring.atoms[index - 1].name) in BACKBONE_NAMES
    )
    interface_backbone = tuple(
        index
        for index in receptor_backbone
        if int(scoring.atoms[index - 1].residue.idx) + 1
        in interface_residue_indices
    )
    if not interface_backbone:
        raise PreparationError("prepared receptor interface backbone is empty")

    receptor_residues: list[ResidueIdentity] = []
    scoring_receptor_residues = _residues_for_atom_indices(
        scoring, receptor_indices
    )
    if len(scoring_receptor_residues) != len(inspection.protein_residue_keys):
        raise PreparationError("prepared receptor residue inventory changed")
    interface_residue_ids: list[str] = []
    interface_classes: set[str] = set()
    for source_key, residue in zip(
        inspection.protein_residue_keys, scoring_receptor_residues, strict=True
    ):
        scoring_residue_index = int(residue.idx) + 1
        first_scoring_atom = int(residue.atoms[0].idx) + 1
        first_md_atom = scoring_to_md[first_scoring_atom]
        md_residue_index = int(md.atoms[first_md_atom - 1].residue.idx) + 1
        residue_id = _stable_residue_id(source_key[0], source_key[1], source_key[2])
        receptor_residues.append(
            ResidueIdentity(
                residue_id=residue_id,
                residue_name=str(residue.name),
                md_residue_index=md_residue_index,
                scoring_residue_index=scoring_residue_index,
            )
        )
        if scoring_residue_index in interface_residue_indices:
            interface_residue_ids.append(residue_id)
            interface_classes.add(residue_chemistry_class(str(residue.name)))

    ligand_residues = _residues_for_atom_indices(scoring, ligand_indices)
    if len(ligand_residues) != 1:
        raise PreparationError(
            "the small-molecule preparation adapter requires one ligand residue"
        )
    ligand_residue = ligand_residues[0]
    ligand_md_residue = md.atoms[ligand_md[0] - 1].residue
    ligand = PreparedEntity(
        entity_id=spec.ligand.entity_id,
        kind="small_molecule",
        md_atom_indices=ligand_md,
        scoring_atom_indices=ligand_indices,
        residues=(
            ResidueIdentity(
                residue_id=f"{spec.ligand.residue_name}_1",
                residue_name=str(ligand_residue.name),
                md_residue_index=int(ligand_md_residue.idx) + 1,
                scoring_residue_index=int(ligand_residue.idx) + 1,
            ),
        ),
        named_selections={
            "all": _entity_selection(ligand_md, ligand_indices),
            "heavy_atoms": _entity_selection(
                tuple(scoring_to_md[index] for index in ligand_heavy),
                ligand_heavy,
            ),
        },
        metadata={
            "formal_charge": spec.ligand.formal_charge,
            "force_field": spec.ligand.force_field,
            "charge_method": spec.ligand.charge_method,
            "qc_selection": "heavy_atoms",
        },
    )
    receptor = PreparedEntity(
        entity_id=spec.receptor.entity_id,
        kind="protein",
        md_atom_indices=receptor_md,
        scoring_atom_indices=receptor_indices,
        residues=tuple(receptor_residues),
        named_selections={
            "all": _entity_selection(receptor_md, receptor_indices),
            "heavy_atoms": _entity_selection(
                tuple(scoring_to_md[index] for index in receptor_heavy),
                receptor_heavy,
            ),
            "backbone": _entity_selection(
                tuple(scoring_to_md[index] for index in receptor_backbone),
                receptor_backbone,
            ),
            "interface_backbone": _entity_selection(
                tuple(scoring_to_md[index] for index in interface_backbone),
                interface_backbone,
            ),
        },
        metadata={
            "interface_cutoff_angstrom": spec.interface_cutoff_angstrom,
            "interface_residue_ids": interface_residue_ids,
            "interface_chemistry_classes": sorted(interface_classes),
        },
    )

    structural_water_residues, water_mapping_audit = _match_crystal_waters(
        spec,
        inspection=inspection,
        scoring=scoring,
        md=md,
        receptor_indices=receptor_indices,
        scoring_to_md=scoring_to_md,
    )
    waters: list[PreparedEntity] = []
    for declared, residue in zip(
        spec.crystal_waters, structural_water_residues, strict=True
    ):
        atom_indices = tuple(int(atom.idx) + 1 for atom in residue.atoms)
        heavy_indices = tuple(
            int(atom.idx) + 1
            for atom in residue.atoms
            if int(getattr(atom, "atomic_number", 0) or 0) != 1
        )
        waters.append(
            PreparedEntity(
                entity_id=declared.entity_id,
                kind="water",
                md_atom_indices=atom_indices,
                scoring_atom_indices=(),
                residues=(
                    ResidueIdentity(
                        residue_id=_stable_residue_id(
                            declared.chain_id,
                            declared.residue_number,
                            declared.insertion_code,
                        ),
                        residue_name=str(residue.name),
                        md_residue_index=int(residue.idx) + 1,
                        scoring_residue_index=None,
                    ),
                ),
                named_selections={
                    "all": _entity_selection(atom_indices, ()),
                    "heavy_atoms": _entity_selection(heavy_indices, ()),
                },
                metadata={
                    "crystallographic": True,
                    "source_chain_id": declared.chain_id,
                    "source_residue_number": declared.residue_number,
                    "scoring_policy": "excluded_dry_baseline",
                },
            )
        )
    return (receptor, ligand, *waters), water_mapping_audit


def _materialize(
    staging: Path,
    *,
    spec: PreparationSpec,
    tool_config: Mapping[str, object],
) -> Path:
    try:
        import parmed as pmd
    except ImportError as exc:
        raise PreparationError(
            "actual preparation requires ParmEd (install mmpbsa[parmed])"
        ) from exc

    inspection = inspect_sources(spec)
    work = staging / "_build"
    work.mkdir()
    md_receptor = work / "receptor_md_with_crystal_waters.pdb"
    dry_receptor = work / "receptor_dry.pdb"
    receptor_views = write_receptor_views(
        spec, md_path=md_receptor, dry_path=dry_receptor
    )
    tools = create_amber_preparation_tools(tool_config)
    parameterization = tools.parameterize(
        parameterization_request(spec, work / "parameterization")
    )
    ligand_mol2 = parameterization.artifacts["molecule"]
    ligand_frcmod = parameterization.artifacts["parameters"]
    charge_normalization = _normalize_mol2_formal_charge(
        ligand_mol2, formal_charge=spec.ligand.formal_charge
    )
    if charge_normalization["atom_count"] != spec.expectations.ligand_atoms:
        raise PreparationError("parameterized ligand atom inventory changed")

    initial = tools.assemble(
        AmberAssemblyRequest(
            input_text=initial_tleap_text(
                spec,
                md_receptor=md_receptor,
                dry_receptor=dry_receptor,
                ligand_mol2=ligand_mol2,
                ligand_frcmod=ligand_frcmod,
            ),
            declared_inputs=(
                md_receptor,
                dry_receptor,
                ligand_mol2,
                ligand_frcmod,
            ),
            output_artifacts={
                "dry_topology": Path("dry.prmtop"),
                "dry_coordinates": Path("dry.inpcrd"),
                "noions_topology": Path("noions.prmtop"),
                "noions_coordinates": Path("noions.inpcrd"),
            },
            output_dir=work / "initial",
        )
    )
    noions = pmd.load_file(
        str(initial.artifacts["noions_topology"]),
        str(initial.artifacts["noions_coordinates"]),
    )
    salt = _salt_inventory(
        noions, target_molar=spec.solvation.salt_concentration_molar
    )
    salted = tools.assemble(
        AmberAssemblyRequest(
            input_text=salted_tleap_text(
                spec,
                md_receptor=md_receptor,
                ligand_mol2=ligand_mol2,
                ligand_frcmod=ligand_frcmod,
                sodium_count=int(salt["sodium_ions"]),
                chloride_count=int(salt["chloride_ions"]),
            ),
            declared_inputs=(md_receptor, ligand_mol2, ligand_frcmod),
            output_artifacts={
                "solvated_topology": Path("solvated.prmtop"),
                "solvated_coordinates": Path("solvated.inpcrd"),
            },
            output_dir=work / "salted",
        )
    )

    md_dir = staging / "md"
    scoring_dir = staging / "scoring"
    provenance_dir = staging / "provenance"
    md_dir.mkdir()
    scoring_dir.mkdir()
    provenance_dir.mkdir()
    amber_topology = md_dir / "solvated.prmtop"
    amber_coordinates = md_dir / "solvated.inpcrd"
    shutil.copy2(salted.artifacts["solvated_topology"], amber_topology)
    shutil.copy2(salted.artifacts["solvated_coordinates"], amber_coordinates)
    md = pmd.load_file(str(amber_topology), str(amber_coordinates))
    md_net_charge = float(sum(float(atom.charge) for atom in md.atoms))
    if abs(md_net_charge) > PREPARED_CHARGE_TOLERANCE_E:
        raise PreparationError(
            f"neutralized MD system retains {md_net_charge:.12f} e net charge"
        )
    md_topology = md_dir / "topol.top"
    md_coordinates = md_dir / "system.gro"
    md.save(str(md_topology), format="gromacs", overwrite=False)
    md.save(str(md_coordinates), format="gro", overwrite=False)

    scoring = pmd.load_file(
        str(initial.artifacts["dry_topology"]),
        str(initial.artifacts["dry_coordinates"]),
    )
    pmd.tools.changeRadii(scoring, spec.force_fields.pb_radii).execute()
    receptor_indices, ligand_indices = _partition(
        scoring,
        receptor_atom_count=spec.expectations.receptor_protein_atoms,
        ligand_atom_count=spec.expectations.ligand_atoms,
    )
    expected_scoring_atoms = (
        spec.expectations.receptor_protein_atoms + spec.expectations.ligand_atoms
    )
    if len(scoring.atoms) != expected_scoring_atoms:
        raise PreparationError(
            "dry scoring atom inventory differs from source expectations"
        )
    receptor_index_set = set(receptor_indices)
    ligand_index_set = set(ligand_indices)
    receptor = deepcopy(scoring)
    receptor.strip(
        tuple(
            index in ligand_index_set
            for index in range(1, len(scoring.atoms) + 1)
        )
    )
    ligand = deepcopy(scoring)
    ligand.strip(
        tuple(
            index in receptor_index_set
            for index in range(1, len(scoring.atoms) + 1)
        )
    )
    complex_topology = scoring_dir / "complex.prmtop"
    receptor_topology = scoring_dir / "receptor.prmtop"
    ligand_topology = scoring_dir / "ligand.prmtop"
    scoring_coordinates = scoring_dir / "coordinates.inpcrd"
    scoring.save(str(scoring_coordinates), format="rst7", overwrite=False)
    receptor.save(str(receptor_topology), overwrite=False)
    ligand.save(str(ligand_topology), overwrite=False)
    improper_reversals = _align_serialized_component_impropers(
        scoring,
        {
            "receptor": (
                pmd.load_file(str(receptor_topology)),
                receptor_indices,
            ),
            "ligand": (
                pmd.load_file(str(ligand_topology)),
                ligand_indices,
            ),
        },
    )
    scoring.remake_parm()
    scoring.save(str(complex_topology), overwrite=False)

    source_indices = _map_scoring_atoms(scoring, md)
    mapped_receptor = tuple(source_indices[index - 1] for index in receptor_indices)
    mapped_ligand = tuple(source_indices[index - 1] for index in ligand_indices)
    index_path = md_dir / "Complex.ndx"
    _write_index(
        index_path,
        system_indices=tuple(range(1, len(md.atoms) + 1)),
        complex_indices=source_indices,
        receptor_indices=mapped_receptor,
        ligand_indices=mapped_ligand,
    )
    entities, crystal_water_mapping = _build_entities(
        spec,
        scoring=scoring,
        md=md,
        receptor_indices=receptor_indices,
        ligand_indices=ligand_indices,
        source_indices=source_indices,
    )
    admission = create_parmed_admission_backend(
        {"major_policy": {"accepted": [4]}}
    ).admit(
        ParmEdAdmissionRequest(
            complex_topology=complex_topology,
            component_topologies={
                "receptor": receptor_topology,
                "ligand": ligand_topology,
            },
            component_atom_indices={
                "receptor": receptor_indices,
                "ligand": ligand_indices,
            },
            tolerances=ProjectionTolerances(
                charge=1.0e-8,
                mass=1.0e-8,
                parameter=1.0e-8,
            ),
            require_complete_partition=True,
        )
    )
    admission.require_accepted()
    receipt_path = provenance_dir / "preparation.json"
    receipt = {
        "schema_version": "mmpbsa.preparation.receipt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_id": spec.case_id,
        "config_sha256": sha256_file(spec.config_path),
        "source_inputs": {
            "receptor": spec.receptor.source.sha256,
            "ligand": spec.ligand.source.sha256,
        },
        "scientific_protocol": {
            "protein_force_field": spec.force_fields.protein,
            "ligand_force_field": spec.ligand.force_field,
            "ligand_charge_method": spec.ligand.charge_method,
            "ligand_formal_charge": spec.ligand.formal_charge,
            "ligand_charge_normalization": charge_normalization,
            "water_model": spec.force_fields.water,
            "crystal_waters_retained_in_md": [
                item.entity_id for item in spec.crystal_waters
            ],
            "crystal_water_mapping": list(crystal_water_mapping),
            "all_water_excluded_from_dry_scoring": True,
            "receptor_views": dict(receptor_views),
            "solvent_shape": spec.solvation.shape,
            "solvent_padding_angstrom": spec.solvation.padding_angstrom,
            "salt": salt,
            "pb_radii": spec.force_fields.pb_radii,
            "interface_cutoff_angstrom": spec.interface_cutoff_angstrom,
        },
        "tools": {
            "ParmEd": metadata.version("ParmEd"),
            "parameterization": _receipt_summary(parameterization, staging),
            "initial_assembly": _receipt_summary(initial, staging),
            "salted_assembly": _receipt_summary(salted, staging),
        },
        "projection_admission": {
            "accepted": admission.accepted,
            "issues": list(admission.issues),
            "complex_signature": admission.complex_signature,
            "component_signatures": dict(admission.component_signatures),
            "serialized_improper_reversals": dict(improper_reversals),
            "tool": _jsonable(admission.tool),
        },
        "inventories": {
            "md_atoms": len(md.atoms),
            "scoring_atoms": len(scoring.atoms),
            "receptor_atoms": len(receptor_indices),
            "ligand_atoms": len(ligand_indices),
            "crystal_water_atoms": len(inspection.crystal_water_atoms),
            "scoring_to_md_atom_indices": list(source_indices),
            "charges": {
                "receptor_e": float(sum(float(atom.charge) for atom in receptor.atoms)),
                "ligand_e": float(sum(float(atom.charge) for atom in ligand.atoms)),
                "dry_complex_e": float(
                    sum(float(atom.charge) for atom in scoring.atoms)
                ),
                "solvated_system_e": md_net_charge,
            },
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    atom_signature = _atom_order_signature(scoring)
    manifest = PreparedCaseManifest(
        case_id=spec.case_id,
        md=PreparedMdSystem(
            backend="gromacs",
            atom_count=len(md.atoms),
            system_signature=_md_system_signature(md),
            artifacts=(
                _artifact(staging, md_coordinates, "md_coordinates", "chemical/x-gro"),
                _artifact(staging, md_topology, "md_topology", "text/plain"),
                _artifact(staging, index_path, "index", "text/plain"),
                _artifact(
                    staging,
                    amber_topology,
                    "solvated_topology",
                    "chemical/x-amber-topology",
                ),
                _artifact(
                    staging,
                    amber_coordinates,
                    "solvated_coordinates",
                    "chemical/x-amber-coordinate",
                ),
                _artifact(
                    staging,
                    receipt_path,
                    "preparation_receipt",
                    "application/json",
                ),
            ),
            metadata={
                "force_fields": [spec.force_fields.protein, spec.ligand.force_field],
                "water_model": spec.force_fields.water,
                "crystal_water_entity_ids": [
                    item.entity_id for item in spec.crystal_waters
                ],
                "padding_angstrom": spec.solvation.padding_angstrom,
                "salt_target_molar": spec.solvation.salt_concentration_molar,
                "salt_inventory": salt,
            },
        ),
        scoring=PreparedScoringSystem(
            backend="ambertools",
            atom_count=len(scoring.atoms),
            atom_order_signature=atom_signature,
            source_md_atom_indices=source_indices,
            radii_model=spec.force_fields.pb_radii,
            artifacts=(
                _artifact(
                    staging,
                    complex_topology,
                    "complex_topology",
                    "chemical/x-amber-topology",
                ),
                _artifact(
                    staging,
                    receptor_topology,
                    "receptor_topology",
                    "chemical/x-amber-topology",
                ),
                _artifact(
                    staging,
                    ligand_topology,
                    "ligand_topology",
                    "chemical/x-amber-topology",
                ),
                _artifact(
                    staging,
                    scoring_coordinates,
                    "coordinates",
                    "chemical/x-amber-coordinate",
                ),
            ),
            metadata={
                "dry_complex": True,
                "water_policy": "exclude_all",
            },
        ),
        entities=entities,
        primary_binding=PrimaryBinding(
            binding_id="primary",
            receptor_entity_ids=(spec.receptor.entity_id,),
            ligand_entity_ids=(spec.ligand.entity_id,),
            excluded_entity_ids=tuple(
                sorted(item.entity_id for item in spec.crystal_waters)
            ),
            topology_artifact_roles=BindingTopologyRoles(
                complex="complex_topology",
                receptor="receptor_topology",
                ligand="ligand_topology",
            ),
        ),
        provenance={
            "preparer": "mmpbsa.backends.prep.prepare_case",
            "preparation_schema": spec.schema_version,
            "receipt_role": "preparation_receipt",
            "source_data_policy": "source hashes only; no prior prepared/run import",
        },
    )
    manifest_path = staging / "prepared_case.json"
    write_prepared_case_manifest(manifest_path, manifest)
    shutil.rmtree(work)
    load_prepared_case(manifest_path, expected_case_id=spec.case_id)
    return manifest_path


@contextmanager
def _locked_parent(parent: Path):
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / ".prepare.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_case(
    spec: PreparationIntent,
    destination: Path,
    *,
    tool_config: Mapping[str, object] | None = None,
) -> Path:
    """Create one immutable PreparedCase and atomically publish it."""

    from .models_v2 import PreparationSpecV2

    if isinstance(spec, PreparationSpecV2):
        from .builder_v2 import prepare_case_v2

        return prepare_case_v2(spec, destination, tool_config=tool_config)
    if not isinstance(spec, PreparationSpec):
        raise PreparationError("prepare_case requires a supported preparation spec")

    target = Path(destination).expanduser().resolve()
    if target == Path(target.anchor) or target.name in {"", ".", ".."}:
        raise PreparationError(f"unsafe PreparedCase destination: {target}")
    with _locked_parent(target.parent):
        if target.exists():
            raise PreparationError(f"PreparedCase destination already exists: {target}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
        )
        try:
            manifest = _materialize(
                staging, spec=spec, tool_config=tool_config or {}
            )
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    published = target / manifest.relative_to(staging)
    load_prepared_case(published, expected_case_id=spec.case_id)
    return published


__all__ = ["prepare_case"]

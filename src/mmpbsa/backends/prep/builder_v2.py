"""Transactional mmCIF preparation for modified peptide binding partners."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from mmpbsa.backends.parmed import (
    ParmEdAdmissionRequest,
    ProjectionTolerances,
    create_parmed_admission_backend,
)
from mmpbsa.core import (
    BindingTopologyRoles,
    EntityFeature,
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

from .._support import tool_from_config
from .amber import AmberAssemblyRequest, create_amber_preparation_tools
from .builder import (
    PREPARED_CHARGE_TOLERANCE_E,
    _align_serialized_component_impropers,
    _artifact,
    _atom_order_signature,
    _coordinates,
    _entity_selection,
    _jsonable,
    _locked_parent,
    _map_scoring_atoms,
    _md_system_signature,
    _partition,
    _receipt_summary,
    _salt_inventory,
    _write_index,
)
from .errors import PreparationError
from .models_v2 import PreparationSpecV2, ResidueRange, ResidueRef
from .peptide_backbone import _peptide_backbone_indices
from .planner import BACKBONE_NAMES, residue_chemistry_class


def _selected_residues(
    structure: object,
    ranges: Sequence[ResidueRange],
) -> tuple[tuple[ResidueRef, object], ...]:
    by_chain: dict[str, list[ResidueRange]] = {}
    for selected in ranges:
        by_chain.setdefault(selected.chain_id, []).append(selected)
    result: list[tuple[ResidueRef, object]] = []
    for chain in structure[0]:
        if chain.name not in by_chain:
            continue
        for residue in chain:
            number = int(residue.seqid.num)
            if not any(item.start <= number <= item.end for item in by_chain[chain.name]):
                continue
            atoms = tuple(atom for atom in residue if atom.altloc in ("\x00", " ", "A"))
            if atoms:
                result.append((ResidueRef(chain.name, number), residue))
    expected = {
        ResidueRef(item.chain_id, number)
        for item in ranges
        for number in range(item.start, item.end + 1)
    }
    observed = {item[0] for item in result}
    if observed != expected:
        raise PreparationError(
            "mmCIF selection does not exactly cover declared residues: "
            f"missing={sorted(expected - observed)!r}, extra={sorted(observed - expected)!r}"
        )
    return tuple(result)


def _pdb_atom_line(
    *,
    serial: int,
    atom: object,
    residue_name: str,
    chain_id: str,
    residue_number: int,
) -> str:
    name = str(atom.name)
    element = str(atom.element.name)
    atom_field = f" {name:<3}" if len(name) < 4 and not name[:1].isdigit() else f"{name:>4}"
    return (
        f"ATOM  {serial:5d} {atom_field} {residue_name:>3} {chain_id[:1]:1}"
        f"{residue_number:4d}    {float(atom.pos.x):8.3f}{float(atom.pos.y):8.3f}"
        f"{float(atom.pos.z):8.3f}{1.0:6.2f}{float(atom.b_iso):6.2f}          "
        f"{element:>2}"
    )


def _write_selected_pdb(
    path: Path,
    residues: Sequence[tuple[ResidueRef, object]],
    *,
    transforms: Mapping[ResidueRef, tuple[str, Mapping[str, str]]] | None = None,
) -> tuple[ResidueRef, ...]:
    transform_map = transforms or {}
    lines: list[str] = []
    serial = 1
    previous_chain: str | None = None
    for reference, residue in residues:
        if previous_chain is not None and reference.chain_id != previous_chain:
            lines.append("TER")
        expected_name = str(residue.name).upper()
        prepared_name, atom_renames = transform_map.get(reference, (expected_name, {}))
        for atom in residue:
            if atom.altloc not in ("\x00", " ", "A"):
                continue
            original = str(atom.name)
            renamed = atom_renames.get(original, original)
            original_name = atom.name
            atom.name = renamed
            try:
                lines.append(
                    _pdb_atom_line(
                        serial=serial,
                        atom=atom,
                        residue_name=prepared_name,
                        chain_id=reference.chain_id,
                        residue_number=reference.residue_number,
                    )
                )
            finally:
                atom.name = original_name
            serial += 1
        previous_chain = reference.chain_id
    lines.extend(("TER", "END"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tuple(item[0] for item in residues)


def _run_reduce(
    source: Path,
    target: Path,
    log: Path,
    tool_config: Mapping[str, object],
) -> Mapping[str, object]:
    tool = tool_from_config(
        tool_config,
        key="reduce",
        default_executable="reduce",
        default_version_arguments=("-version",),
    )
    environment = os.environ.copy()
    environment.update(tool.environment)
    command = (tool.executable, "-BUILD", "-NOFLIP", str(source.resolve()))
    completed = subprocess.run(
        command,
        cwd=target.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = completed.stderr + "\n" + completed.stdout
    if completed.returncode not in {0, 1} or (
        completed.returncode == 1 and "BADBUMP" not in combined
    ):
        raise PreparationError(
            f"Reduce failed with exit code {completed.returncode}: " + completed.stderr[-2000:]
        )
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in completed.stdout.splitlines()):
        raise PreparationError("Reduce produced no PDB atom records")
    target.write_text(completed.stdout, encoding="utf-8")
    log.write_text(combined, encoding="utf-8")
    return {
        "argv": list(command),
        "returncode": completed.returncode,
        "status": "warn" if completed.returncode else "completed",
        "bad_bump": "BADBUMP" in combined,
    }


def _histidine_names(lines: Sequence[str]) -> Mapping[tuple[str, int], str]:
    atoms: dict[tuple[str, int], set[str]] = {}
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and line[17:20].strip() == "HIS":
            atoms.setdefault((line[21:22], int(line[22:26])), set()).add(line[12:16].strip())
    result: dict[tuple[str, int], str] = {}
    for key, names in atoms.items():
        if {"HD1", "HE2"}.issubset(names):
            result[key] = "HIP"
        elif "HD1" in names:
            result[key] = "HID"
        elif "HE2" in names:
            result[key] = "HIE"
        else:
            raise PreparationError(f"Reduce did not resolve histidine {key!r}")
    return result


def _prepare_receptor(
    source_structure: object,
    spec: PreparationSpecV2,
    root: Path,
    tool_config: Mapping[str, object],
) -> tuple[Path, tuple[ResidueRef, ...], Mapping[str, object]]:
    residues = _selected_residues(source_structure, spec.receptor.ranges)
    context = root / "receptor_context.pdb"
    keys = _write_selected_pdb(context, residues)
    reduced = root / "receptor_reduced.pdb"
    reduce_receipt = _run_reduce(context, reduced, root / "reduce.log", tool_config)
    lines = reduced.read_text(encoding="utf-8", errors="strict").splitlines()
    histidines = _histidine_names(lines)
    cap = spec.receptor.cap
    output: list[str] = []
    retained_cap_atoms: set[str] = set()
    for raw in lines:
        if not raw.startswith(("ATOM  ", "HETATM")) or len(raw) < 54:
            continue
        chain = raw[21:22].strip()
        number = int(raw[22:26])
        reference = ResidueRef(chain, number)
        if reference not in set(keys):
            continue
        residue_name = raw[17:20].strip()
        atom_name = raw[12:16].strip()
        if cap is not None and reference == cap.source:
            if residue_name != cap.expected_name or atom_name not in cap.retain_atoms:
                continue
            retained_cap_atoms.add(atom_name)
            renamed = cap.atom_renames.get(atom_name, atom_name)
            raw = raw[:12] + f"{renamed:>4}" + raw[16:17] + f"{cap.prepared_name:>3}" + raw[20:]
        elif residue_name == "HIS":
            raw = raw[:17] + f"{histidines[(raw[21:22], number)]:>3}" + raw[20:]
        output.append(raw)
    if cap is not None and retained_cap_atoms != set(cap.retain_atoms):
        raise PreparationError(
            f"cap retained atoms differ: expected {sorted(cap.retain_atoms)!r}, "
            f"observed {sorted(retained_cap_atoms)!r}"
        )
    output.extend(("TER", "END"))
    prepared = root / "receptor.pdb"
    prepared.write_text("\n".join(output) + "\n", encoding="utf-8")
    return prepared, keys, reduce_receipt


def _prepare_ligand(
    source_structure: object,
    spec: PreparationSpecV2,
    root: Path,
) -> tuple[Path, tuple[ResidueRef, ...]]:
    residues = _selected_residues(source_structure, spec.ligand.ranges)
    transform_by_ref = {item.source: item for item in spec.ligand.transforms}
    for reference, residue in residues:
        transform = transform_by_ref.get(reference)
        if transform is not None and str(residue.name).upper() != transform.expected_name:
            raise PreparationError(
                f"residue transform expected {reference!r}={transform.expected_name}, "
                f"observed {residue.name}"
            )
    transforms = {
        reference: (item.prepared_name, item.atom_renames)
        for reference, item in transform_by_ref.items()
    }
    path = root / "ligand.pdb"
    keys = _write_selected_pdb(path, residues, transforms=transforms)
    return path, keys


def _quoted(path: Path) -> str:
    text = str(path.resolve())
    if any(item in text for item in ('"', "\n", "\r", "\x00")):
        raise PreparationError(f"path cannot be represented in tleap input: {path}")
    return f'"{text}"'


def _tleap_prelude(spec: PreparationSpecV2) -> list[str]:
    return [
        *(f"source {item}" for item in spec.force_fields.leaprcs),
        f"set default PBRadii {spec.force_fields.pb_radii}",
    ]


def _bond_commands(spec: PreparationSpecV2, ligand_keys: Sequence[ResidueRef]) -> list[str]:
    index = {key: ordinal for ordinal, key in enumerate(ligand_keys, start=1)}
    return [
        f"bond LIG.{index[bond.first.residue]}.{bond.first.atom_name} "
        f"LIG.{index[bond.second.residue]}.{bond.second.atom_name}"
        for bond in spec.ligand.bonds
    ]


def _initial_tleap(
    spec: PreparationSpecV2,
    receptor: Path,
    ligand: Path,
    ligand_keys: Sequence[ResidueRef],
) -> str:
    return "\n".join(
        (
            *_tleap_prelude(spec),
            f"REC = loadpdb {_quoted(receptor)}",
            f"LIG = loadpdb {_quoted(ligand)}",
            *_bond_commands(spec, ligand_keys),
            "DRY = combine { REC LIG }",
            "check DRY",
            "saveamberparm DRY dry.prmtop dry.inpcrd",
            "NOIONS = combine { REC LIG }",
            (
                f"solvateOct NOIONS {spec.force_fields.water_box} "
                f"{spec.solvation.padding_angstrom:g} iso"
            ),
            "saveamberparm NOIONS noions.prmtop noions.inpcrd",
            "quit",
        )
    )


def _salted_tleap(
    spec: PreparationSpecV2,
    receptor: Path,
    ligand: Path,
    ligand_keys: Sequence[ResidueRef],
    sodium: int,
    chloride: int,
) -> str:
    return "\n".join(
        (
            *_tleap_prelude(spec),
            f"REC = loadpdb {_quoted(receptor)}",
            f"LIG = loadpdb {_quoted(ligand)}",
            *_bond_commands(spec, ligand_keys),
            "SYSTEM = combine { REC LIG }",
            (
                f"solvateOct SYSTEM {spec.force_fields.water_box} "
                f"{spec.solvation.padding_angstrom:g} iso"
            ),
            f"addIonsRand SYSTEM Na+ {sodium} Cl- {chloride}",
            "check SYSTEM",
            "saveamberparm SYSTEM solvated.prmtop solvated.inpcrd",
            "quit",
        )
    )


def _residues_for_indices(structure: object, indices: Sequence[int]) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[int] = set()
    for index in indices:
        residue = structure.atoms[index - 1].residue
        if int(residue.idx) not in seen:
            seen.add(int(residue.idx))
            result.append(residue)
    return tuple(result)


def _save_gromacs_topology(
    structure: object,
    path: Path,
    *,
    receptor_atom_count: int,
    ligand_atom_count: int,
) -> Mapping[str, object]:
    """Write mixed Amber scaling as explicit GROMACS pair function type 2."""

    pair_scales: dict[tuple[int, int], tuple[float, float]] = {}
    unique_types: dict[int, object] = {}
    for dihedral in structure.dihedrals:
        raw_type = dihedral.type
        try:
            types = tuple(raw_type)
        except TypeError:
            types = (raw_type,)
        for item in types:
            unique_types[id(item)] = item
        if bool(dihedral.improper) or bool(dihedral.ignore_end):
            continue
        scales = {(float(item.scee), float(item.scnb)) for item in types}
        if len(scales) != 1:
            raise PreparationError("one 1-4 pair has conflicting Amber scaling")
        scee, scnb = next(iter(scales))
        if scee <= 0.0 or scnb <= 0.0:
            raise PreparationError("proper 1-4 scaling must be positive")
        key = tuple(sorted((int(dihedral.atom1.idx) + 1, int(dihedral.atom4.idx) + 1)))
        observed = pair_scales.get(key)
        if observed is not None and observed != (scee, scnb):
            raise PreparationError("duplicate 1-4 pair has conflicting scaling")
        pair_scales[key] = (scee, scnb)
    if not pair_scales:
        raise PreparationError("Amber system contains no proper 1-4 pairs")
    charge_scales = sorted({round(1.0 / item[0], 10) for item in pair_scales.values()})
    lj_scales = sorted({round(1.0 / item[1], 10) for item in pair_scales.values()})
    if len(charge_scales) == 1 and len(lj_scales) == 1:
        structure.save(str(path), format="gromacs", overwrite=False)
        return {
            "mode": "global_fudge",
            "charge_scales": charge_scales,
            "lj_scales": lj_scales,
            "pair_count": len(pair_scales),
        }

    for item in unique_types.values():
        item.scee = 1.0
        item.scnb = 1.0
    structure.save(str(path), format="gromacs", overwrite=False)
    lines = path.read_text(encoding="utf-8").splitlines()
    offsets = {"system1": 0, "system2": receptor_atom_count}
    expected_sizes = {"system1": receptor_atom_count, "system2": ligand_atom_count}
    current_molecule: str | None = None
    pending_molecule_name = False
    section = ""
    observed_pairs: set[tuple[int, int]] = set()
    rewritten: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[] ").casefold()
            pending_molecule_name = section == "moleculetype"
            rewritten.append(line)
            continue
        if pending_molecule_name and stripped and not stripped.startswith(";"):
            current_molecule = stripped.split()[0]
            pending_molecule_name = False
            rewritten.append(line)
            continue
        if section != "pairs" or not stripped or stripped.startswith(";"):
            rewritten.append(line)
            continue
        fields = stripped.split()
        if len(fields) < 3 or current_molecule not in offsets:
            rewritten.append(line)
            continue
        try:
            local_a, local_b = int(fields[0]), int(fields[1])
        except ValueError:
            rewritten.append(line)
            continue
        if not (
            1 <= local_a <= expected_sizes[current_molecule]
            and 1 <= local_b <= expected_sizes[current_molecule]
        ):
            raise PreparationError("GROMACS pair escapes its solute molecule")
        offset = offsets[current_molecule]
        key = tuple(sorted((local_a + offset, local_b + offset)))
        try:
            scee, scnb = pair_scales[key]
        except KeyError as exc:
            raise PreparationError(f"GROMACS emitted an undeclared 1-4 pair: {key!r}") from exc
        atom_a = structure.atoms[key[0] - 1]
        atom_b = structure.atoms[key[1] - 1]
        sigma_nm = (float(atom_a.sigma) + float(atom_b.sigma)) / 20.0
        epsilon_kj = math.sqrt(float(atom_a.epsilon) * float(atom_b.epsilon)) * 4.184 / scnb
        rewritten.append(
            f"{local_a:7d}{local_b:7d}     2 "
            f"{1.0 / scee:.10f} {float(atom_a.charge):.10f} "
            f"{float(atom_b.charge):.10f} {sigma_nm:.9f} {epsilon_kj:.9f}"
        )
        observed_pairs.add(key)
    if observed_pairs != set(pair_scales):
        missing = len(set(pair_scales).difference(observed_pairs))
        extra = len(observed_pairs.difference(pair_scales))
        raise PreparationError(
            f"explicit GROMACS 1-4 inventory differs: missing={missing}, extra={extra}"
        )
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return {
        "mode": "explicit_pair_function_2",
        "charge_scales": charge_scales,
        "lj_scales": lj_scales,
        "pair_count": len(pair_scales),
        "solute_molecule_types": ["system1", "system2"],
        "global_fudge_after_expansion": [1.0, 1.0],
    }


def _build_entities(
    spec: PreparationSpecV2,
    *,
    scoring: object,
    md: object,
    receptor_indices: tuple[int, ...],
    ligand_indices: tuple[int, ...],
    source_indices: tuple[int, ...],
    receptor_keys: Sequence[ResidueRef],
    ligand_keys: Sequence[ResidueRef],
) -> tuple[PreparedEntity, PreparedEntity]:
    scoring_to_md = {
        scoring_index: source_indices[scoring_index - 1]
        for scoring_index in range(1, len(source_indices) + 1)
    }
    receptor_md = tuple(scoring_to_md[item] for item in receptor_indices)
    ligand_md = tuple(scoring_to_md[item] for item in ligand_indices)
    receptor_residues = _residues_for_indices(scoring, receptor_indices)
    ligand_residues = _residues_for_indices(scoring, ligand_indices)
    if len(receptor_residues) != len(receptor_keys) or len(ligand_residues) != len(ligand_keys):
        raise PreparationError("prepared residue inventory differs from explicit source selection")

    def residue_id(key: ResidueRef) -> str:
        return f"{key.chain_id}_{key.residue_number}"

    def identities(
        keys: Sequence[ResidueRef], residues: Sequence[object]
    ) -> tuple[ResidueIdentity, ...]:
        result = []
        for key, residue in zip(keys, residues, strict=True):
            scoring_index = int(residue.idx) + 1
            md_atom = scoring_to_md[int(residue.atoms[0].idx) + 1]
            result.append(
                ResidueIdentity(
                    residue_id=residue_id(key),
                    residue_name=str(residue.name),
                    md_residue_index=int(md.atoms[md_atom - 1].residue.idx) + 1,
                    scoring_residue_index=scoring_index,
                )
            )
        return tuple(result)

    receptor_identity = identities(receptor_keys, receptor_residues)
    ligand_identity = identities(ligand_keys, ligand_residues)
    ligand_atoms = tuple(scoring.atoms[index - 1] for index in ligand_indices)
    interface_receptor_residues: set[int] = set()
    interface_ligand_residues: set[int] = set()
    cutoff_squared = spec.interface_cutoff_angstrom**2
    for receptor_index in receptor_indices:
        receptor_atom = scoring.atoms[receptor_index - 1]
        if int(getattr(receptor_atom, "atomic_number", 0) or 0) == 1:
            continue
        for ligand_atom in ligand_atoms:
            if int(getattr(ligand_atom, "atomic_number", 0) or 0) == 1:
                continue
            if (
                sum(
                    (left - right) ** 2
                    for left, right in zip(
                        _coordinates(receptor_atom), _coordinates(ligand_atom), strict=True
                    )
                )
                <= cutoff_squared
            ):
                interface_receptor_residues.add(int(receptor_atom.residue.idx) + 1)
                interface_ligand_residues.add(int(ligand_atom.residue.idx) + 1)
    if not interface_receptor_residues or not interface_ligand_residues:
        raise PreparationError("prepared receptor-ligand interface is empty")

    def selected(indices: Sequence[int], predicate: object) -> tuple[int, ...]:
        return tuple(index for index in indices if predicate(scoring.atoms[index - 1]))

    def heavy(atom: object) -> bool:
        return int(getattr(atom, "atomic_number", 0) or 0) != 1

    def backbone(atom: object) -> bool:
        return str(atom.name) in BACKBONE_NAMES

    receptor_heavy = selected(receptor_indices, heavy)
    receptor_backbone = selected(receptor_indices, backbone)
    receptor_interface_backbone = tuple(
        index
        for index in receptor_backbone
        if int(scoring.atoms[index - 1].residue.idx) + 1 in interface_receptor_residues
    )
    ligand_heavy = selected(ligand_indices, heavy)
    ligand_backbone = _peptide_backbone_indices(ligand_keys, ligand_residues, ligand_indices)
    if not receptor_interface_backbone or not ligand_backbone:
        raise PreparationError("protein interface or peptide QC backbone is empty")

    ligand_residue_by_key = dict(zip(ligand_keys, ligand_residues, strict=True))
    features: list[EntityFeature] = []
    for feature in spec.ligand.features:
        feature_residue_indices = {
            int(ligand_residue_by_key[key].idx) + 1 for key in feature.residues
        }
        scoring_atoms = tuple(
            index
            for index in ligand_indices
            if int(scoring.atoms[index - 1].residue.idx) + 1 in feature_residue_indices
        )
        features.append(
            EntityFeature(
                feature_id=feature.feature_id,
                kind=feature.kind,
                selection=_entity_selection(
                    tuple(scoring_to_md[index] for index in scoring_atoms),
                    scoring_atoms,
                ),
                residue_ids=tuple(sorted(residue_id(key) for key in feature.residues)),
                metadata=dict(feature.metadata),
            )
        )

    receptor_classes = sorted(
        {
            residue_chemistry_class(str(residue.name))
            for residue in receptor_residues
            if int(residue.idx) + 1 in interface_receptor_residues
        }
    )
    ligand_classes = sorted(
        {
            residue_chemistry_class(str(residue.name))
            for residue in ligand_residues
            if int(residue.idx) + 1 in interface_ligand_residues
        }
    )
    receptor = PreparedEntity(
        entity_id=spec.receptor.entity_id,
        kind="protein",
        md_atom_indices=receptor_md,
        scoring_atom_indices=receptor_indices,
        residues=receptor_identity,
        named_selections={
            "all": _entity_selection(receptor_md, receptor_indices),
            "heavy_atoms": _entity_selection(
                tuple(scoring_to_md[index] for index in receptor_heavy), receptor_heavy
            ),
            "backbone": _entity_selection(
                tuple(scoring_to_md[index] for index in receptor_backbone), receptor_backbone
            ),
            "interface_backbone": _entity_selection(
                tuple(scoring_to_md[index] for index in receptor_interface_backbone),
                receptor_interface_backbone,
            ),
        },
        metadata={
            "interface_cutoff_angstrom": spec.interface_cutoff_angstrom,
            "interface_chemistry_classes": receptor_classes,
        },
    )
    ligand = PreparedEntity(
        entity_id=spec.ligand.entity_id,
        kind="peptide",
        md_atom_indices=ligand_md,
        scoring_atom_indices=ligand_indices,
        residues=ligand_identity,
        named_selections={
            "all": _entity_selection(ligand_md, ligand_indices),
            "heavy_atoms": _entity_selection(
                tuple(scoring_to_md[index] for index in ligand_heavy), ligand_heavy
            ),
            "backbone": _entity_selection(
                tuple(scoring_to_md[index] for index in ligand_backbone), ligand_backbone
            ),
        },
        features=tuple(features),
        metadata={
            "formal_charge": spec.ligand.formal_charge,
            "qc_selection": "backbone",
            "interface_chemistry_classes": ligand_classes,
            "heavy_atoms_include_all_features": True,
        },
    )
    return receptor, ligand


def _materialize_v2(
    staging: Path,
    *,
    spec: PreparationSpecV2,
    tool_config: Mapping[str, object],
) -> Path:
    try:
        import gemmi
        import parmed as pmd
    except ImportError as exc:
        raise PreparationError("preparation v2 requires Gemmi and ParmEd") from exc

    spec.verify_sources()
    structure = gemmi.read_structure(str(spec.source.path))
    if len(structure) != 1:
        raise PreparationError("mmCIF preparation source must contain exactly one model")
    work = staging / "_build"
    work.mkdir()
    receptor_pdb, receptor_keys, reduce_receipt = _prepare_receptor(
        structure, spec, work, tool_config
    )
    ligand_pdb, ligand_keys = _prepare_ligand(structure, spec, work)
    tools = create_amber_preparation_tools(tool_config)
    initial = tools.assemble(
        AmberAssemblyRequest(
            input_text=_initial_tleap(spec, receptor_pdb, ligand_pdb, ligand_keys),
            declared_inputs=(receptor_pdb, ligand_pdb),
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
        str(initial.artifacts["noions_topology"]), str(initial.artifacts["noions_coordinates"])
    )
    salt = _salt_inventory(noions, target_molar=spec.solvation.salt_concentration_molar)
    salted = tools.assemble(
        AmberAssemblyRequest(
            input_text=_salted_tleap(
                spec,
                receptor_pdb,
                ligand_pdb,
                ligand_keys,
                int(salt["sodium_ions"]),
                int(salt["chloride_ions"]),
            ),
            declared_inputs=(receptor_pdb, ligand_pdb),
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
        raise PreparationError(f"neutralized MD system retains {md_net_charge:.12f} e")
    md_topology = md_dir / "topol.top"
    md_coordinates = md_dir / "system.gro"
    gromacs_scaling = _save_gromacs_topology(
        md,
        md_topology,
        receptor_atom_count=spec.expectations.receptor_atoms,
        ligand_atom_count=spec.expectations.ligand_atoms,
    )
    md.save(str(md_coordinates), format="gro", overwrite=False)

    scoring = pmd.load_file(
        str(initial.artifacts["dry_topology"]), str(initial.artifacts["dry_coordinates"])
    )
    pmd.tools.changeRadii(scoring, spec.force_fields.pb_radii).execute()
    receptor_indices, ligand_indices = _partition(
        scoring,
        receptor_atom_count=spec.expectations.receptor_atoms,
        ligand_atom_count=spec.expectations.ligand_atoms,
    )
    if len(_residues_for_indices(scoring, receptor_indices)) != spec.expectations.receptor_residues:
        raise PreparationError("prepared receptor residue count differs from expectation")
    if len(_residues_for_indices(scoring, ligand_indices)) != spec.expectations.ligand_residues:
        raise PreparationError("prepared ligand residue count differs from expectation")
    receptor_index_set = set(receptor_indices)
    ligand_index_set = set(ligand_indices)
    receptor = deepcopy(scoring)
    receptor.strip(tuple(index in ligand_index_set for index in range(1, len(scoring.atoms) + 1)))
    ligand = deepcopy(scoring)
    ligand.strip(tuple(index in receptor_index_set for index in range(1, len(scoring.atoms) + 1)))
    ligand_charge = float(sum(float(atom.charge) for atom in ligand.atoms))
    if abs(ligand_charge - spec.expectations.ligand_charge_e) > 1.0e-6:
        raise PreparationError(
            "prepared ligand charge differs: expected "
            f"{spec.expectations.ligand_charge_e}, observed {ligand_charge:.8f}"
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
            "receptor": (pmd.load_file(str(receptor_topology)), receptor_indices),
            "ligand": (pmd.load_file(str(ligand_topology)), ligand_indices),
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
    entities = _build_entities(
        spec,
        scoring=scoring,
        md=md,
        receptor_indices=receptor_indices,
        ligand_indices=ligand_indices,
        source_indices=source_indices,
        receptor_keys=receptor_keys,
        ligand_keys=ligand_keys,
    )
    admission = create_parmed_admission_backend({"major_policy": {"accepted": [4]}}).admit(
        ParmEdAdmissionRequest(
            complex_topology=complex_topology,
            component_topologies={"receptor": receptor_topology, "ligand": ligand_topology},
            component_atom_indices={"receptor": receptor_indices, "ligand": ligand_indices},
            tolerances=ProjectionTolerances(charge=1.0e-8, mass=1.0e-8, parameter=1.0e-8),
            require_complete_partition=True,
        )
    )
    admission.require_accepted()
    receipt_path = provenance_dir / "preparation.json"
    receipt = {
        "schema_version": "mmpbsa.preparation.receipt.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_id": spec.case_id,
        "config_sha256": sha256_file(spec.config_path),
        "source_sha256": spec.source.sha256,
        "scientific_protocol": {
            "force_fields": list(spec.force_fields.leaprcs),
            "water_model": spec.force_fields.water,
            "pb_radii": spec.force_fields.pb_radii,
            "solvent_shape": spec.solvation.shape,
            "solvent_padding_angstrom": spec.solvation.padding_angstrom,
            "salt": salt,
            "receptor_ranges": _jsonable(spec.receptor.ranges),
            "ligand_ranges": _jsonable(spec.ligand.ranges),
            "residue_transforms": _jsonable(spec.ligand.transforms),
            "intra_entity_bonds": _jsonable(spec.ligand.bonds),
            "features": _jsonable(spec.ligand.features),
            "gromacs_one_four_scaling": dict(gromacs_scaling),
        },
        "tools": {
            "ParmEd": metadata.version("ParmEd"),
            "Reduce": reduce_receipt,
            "initial_assembly": _receipt_summary(initial, staging),
            "salted_assembly": _receipt_summary(salted, staging),
        },
        "projection_admission": {
            "accepted": admission.accepted,
            "issues": list(admission.issues),
            "serialized_improper_reversals": dict(improper_reversals),
        },
        "inventories": {
            "md_atoms": len(md.atoms),
            "scoring_atoms": len(scoring.atoms),
            "receptor_atoms": len(receptor_indices),
            "receptor_residues": len(receptor_keys),
            "ligand_atoms": len(ligand_indices),
            "ligand_residues": len(ligand_keys),
            "charges": {
                "receptor_e": float(sum(float(atom.charge) for atom in receptor.atoms)),
                "ligand_e": ligand_charge,
                "dry_complex_e": float(sum(float(atom.charge) for atom in scoring.atoms)),
                "solvated_system_e": md_net_charge,
            },
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
                    staging, amber_topology, "solvated_topology", "chemical/x-amber-topology"
                ),
                _artifact(
                    staging,
                    amber_coordinates,
                    "solvated_coordinates",
                    "chemical/x-amber-coordinate",
                ),
                _artifact(staging, receipt_path, "preparation_receipt", "application/json"),
            ),
            metadata={
                "force_fields": list(spec.force_fields.leaprcs),
                "water_model": spec.force_fields.water,
                "padding_angstrom": spec.solvation.padding_angstrom,
                "salt_target_molar": spec.solvation.salt_concentration_molar,
                "salt_inventory": salt,
            },
        ),
        scoring=PreparedScoringSystem(
            backend="ambertools",
            atom_count=len(scoring.atoms),
            atom_order_signature=_atom_order_signature(scoring),
            source_md_atom_indices=source_indices,
            radii_model=spec.force_fields.pb_radii,
            artifacts=(
                _artifact(
                    staging, complex_topology, "complex_topology", "chemical/x-amber-topology"
                ),
                _artifact(
                    staging, receptor_topology, "receptor_topology", "chemical/x-amber-topology"
                ),
                _artifact(staging, ligand_topology, "ligand_topology", "chemical/x-amber-topology"),
                _artifact(
                    staging, scoring_coordinates, "coordinates", "chemical/x-amber-coordinate"
                ),
            ),
            metadata={"dry_complex": True, "water_policy": "exclude_all"},
        ),
        entities=entities,
        primary_binding=PrimaryBinding(
            binding_id="primary",
            receptor_entity_ids=(spec.receptor.entity_id,),
            ligand_entity_ids=(spec.ligand.entity_id,),
            excluded_entity_ids=(),
            topology_artifact_roles=BindingTopologyRoles(
                complex="complex_topology",
                receptor="receptor_topology",
                ligand="ligand_topology",
            ),
            cross_entity_bonds=(),
        ),
        provenance={
            "preparer": "mmpbsa.backends.prep.prepare_case",
            "preparation_schema": spec.schema_version,
            "receipt_role": "preparation_receipt",
            "source_data_policy": "SHA-bound source mmCIF; no prior prepared/run import",
            "ligand_entity_policy": "peptide and attached modifications remain one explicit entity",
        },
    )
    manifest_path = staging / "prepared_case.json"
    write_prepared_case_manifest(manifest_path, manifest)
    shutil.rmtree(work)
    load_prepared_case(manifest_path, expected_case_id=spec.case_id)
    return manifest_path


def prepare_case_v2(
    spec: PreparationSpecV2,
    destination: Path,
    *,
    tool_config: Mapping[str, object] | None = None,
) -> Path:
    target = Path(destination).expanduser().resolve()
    if target == Path(target.anchor) or target.name in {"", ".", ".."}:
        raise PreparationError(f"unsafe PreparedCase destination: {target}")
    with _locked_parent(target.parent):
        if target.exists():
            raise PreparationError(f"PreparedCase destination already exists: {target}")
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
        try:
            manifest = _materialize_v2(staging, spec=spec, tool_config=tool_config or {})
            relative = manifest.relative_to(staging)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    published = target / relative
    load_prepared_case(published, expected_case_id=spec.case_id)
    return published


__all__ = ["prepare_case_v2"]

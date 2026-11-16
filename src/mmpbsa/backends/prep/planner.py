"""Source inspection and deterministic AmberTools preparation plans."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from math import dist
from pathlib import Path

from .amber import (
    AmberAssemblyRequest,
    SmallMoleculeParameterizationRequest,
    create_amber_preparation_tools,
)
from .errors import PreparationError
from .models import PreparationSpec

WATER_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP3P"})
BACKBONE_NAMES = frozenset({"N", "CA", "C", "O"})
CHARGED_RESIDUES = frozenset({"ARG", "LYS", "ASP", "GLU", "HIP"})
POLAR_RESIDUES = frozenset(
    {"ASN", "GLN", "HIS", "HID", "HIE", "SER", "THR", "TYR", "CYS"}
)
NONPOLAR_RESIDUES = frozenset(
    {
        "ALA",
        "ACE",
        "GLY",
        "ILE",
        "LEU",
        "MET",
        "NME",
        "PHE",
        "PRO",
        "TRP",
        "VAL",
    }
)


@dataclass(frozen=True, slots=True)
class PdbAtom:
    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    element: str
    coordinates: tuple[float, float, float]
    line: str

    @property
    def residue_key(self) -> tuple[str, int, str, str]:
        return (
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.residue_name,
        )

    @property
    def heavy(self) -> bool:
        return self.element.upper() != "H"


@dataclass(frozen=True, slots=True)
class SdfAtom:
    ordinal: int
    element: str
    coordinates: tuple[float, float, float]

    @property
    def heavy(self) -> bool:
        return self.element.upper() != "H"


@dataclass(frozen=True, slots=True)
class SourceInspection:
    receptor_atoms: tuple[PdbAtom, ...]
    ligand_atoms: tuple[SdfAtom, ...]
    protein_residue_keys: tuple[tuple[str, int, str, str], ...]
    water_residue_keys: tuple[tuple[str, int, str, str], ...]
    interface_residue_keys: tuple[tuple[str, int, str, str], ...]
    interface_chemistry_classes: tuple[str, ...]
    histidine_states: Mapping[str, int]

    @property
    def protein_atoms(self) -> tuple[PdbAtom, ...]:
        return tuple(
            atom for atom in self.receptor_atoms if atom.residue_name not in WATER_NAMES
        )

    @property
    def crystal_water_atoms(self) -> tuple[PdbAtom, ...]:
        return tuple(
            atom for atom in self.receptor_atoms if atom.residue_name in WATER_NAMES
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "receptor_protein_atoms": len(self.protein_atoms),
            "receptor_protein_residues": len(self.protein_residue_keys),
            "ligand_atoms": len(self.ligand_atoms),
            "ligand_heavy_atoms": sum(atom.heavy for atom in self.ligand_atoms),
            "crystal_water_atoms": len(self.crystal_water_atoms),
            "crystal_water_residues": [
                _residue_key_dict(key) for key in self.water_residue_keys
            ],
            "interface_cutoff_residues": [
                _residue_key_dict(key) for key in self.interface_residue_keys
            ],
            "interface_chemistry_classes": list(
                self.interface_chemistry_classes
            ),
            "histidine_states": dict(self.histidine_states),
        }


def _residue_key_dict(
    key: tuple[str, int, str, str],
) -> dict[str, object]:
    return {
        "chain_id": key[0],
        "residue_number": key[1],
        "insertion_code": key[2],
        "residue_name": key[3],
    }


def _pdb_element(line: str, atom_name: str) -> str:
    declared = line[76:78].strip() if len(line) >= 78 else ""
    if declared:
        letters = "".join(character for character in declared if character.isalpha())
        if letters:
            return letters.capitalize()
    letters = "".join(character for character in atom_name if character.isalpha())
    return (letters[:1] or "?").capitalize()


def read_pdb_atoms(path: Path) -> tuple[PdbAtom, ...]:
    result: list[PdbAtom] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            atom_name = line[12:16].strip()
            result.append(
                PdbAtom(
                    serial=int(line[6:11]),
                    name=atom_name,
                    residue_name=line[17:20].strip().upper(),
                    chain_id=line[21:22].strip(),
                    residue_number=int(line[22:26]),
                    insertion_code=line[26:27].strip(),
                    element=_pdb_element(line, atom_name),
                    coordinates=(
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ),
                    line=line,
                )
            )
        except (ValueError, IndexError) as exc:
            raise PreparationError(
                f"invalid PDB atom record at {path}:{line_number}"
            ) from exc
    if not result:
        raise PreparationError(f"PDB contains no atoms: {path}")
    serials = tuple(atom.serial for atom in result)
    if len(serials) != len(set(serials)):
        raise PreparationError("PDB atom serials must be unique")
    return tuple(result)


def read_sdf_atoms(path: Path) -> tuple[SdfAtom, ...]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise PreparationError(f"SDF is truncated: {path}")
    try:
        atom_count = int(lines[3][0:3])
    except ValueError as exc:
        raise PreparationError(f"SDF atom count is invalid: {path}") from exc
    if atom_count < 1 or len(lines) < 4 + atom_count:
        raise PreparationError(f"SDF atom block is incomplete: {path}")
    result: list[SdfAtom] = []
    for ordinal, line in enumerate(lines[4 : 4 + atom_count], start=1):
        try:
            result.append(
                SdfAtom(
                    ordinal=ordinal,
                    element=line[31:34].strip().capitalize(),
                    coordinates=(
                        float(line[0:10]),
                        float(line[10:20]),
                        float(line[20:30]),
                    ),
                )
            )
        except ValueError as exc:
            raise PreparationError(
                f"invalid SDF atom record at ordinal {ordinal}: {path}"
            ) from exc
    if any(not atom.element for atom in result):
        raise PreparationError(f"SDF atom element is empty: {path}")
    return tuple(result)


def _ordered_residues(
    atoms: tuple[PdbAtom, ...], *, water: bool
) -> tuple[tuple[str, int, str, str], ...]:
    result: list[tuple[str, int, str, str]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for atom in atoms:
        is_water = atom.residue_name in WATER_NAMES
        if is_water != water or atom.residue_key in seen:
            continue
        seen.add(atom.residue_key)
        result.append(atom.residue_key)
    return tuple(result)


def _histidine_states(atoms: tuple[PdbAtom, ...]) -> Mapping[str, int]:
    by_residue: dict[tuple[str, int, str, str], set[str]] = {}
    for atom in atoms:
        if atom.residue_name == "HIS":
            by_residue.setdefault(atom.residue_key, set()).add(atom.name)
    result: dict[str, int] = {}
    for key, names in by_residue.items():
        has_hd1 = "HD1" in names
        has_he2 = "HE2" in names
        if has_hd1 and has_he2:
            state = "HIP"
        elif has_hd1:
            state = "HID"
        elif has_he2:
            state = "HIE"
        else:
            raise PreparationError(
                f"histidine {key!r} has neither explicit HD1 nor HE2"
            )
        result[state] = result.get(state, 0) + 1
    return dict(sorted(result.items()))


def residue_chemistry_class(residue_name: str) -> str:
    normalized = residue_name.upper()
    if normalized in CHARGED_RESIDUES:
        return "charged"
    if normalized in POLAR_RESIDUES:
        return "polar"
    if normalized in NONPOLAR_RESIDUES:
        return "nonpolar"
    return "unknown"


def inspect_sources(spec: PreparationSpec) -> SourceInspection:
    spec.verify_sources()
    receptor_atoms = read_pdb_atoms(spec.receptor.source.path)
    ligand_atoms = read_sdf_atoms(spec.ligand.source.path)
    protein_residues = _ordered_residues(receptor_atoms, water=False)
    water_residues = _ordered_residues(receptor_atoms, water=True)
    declared_waters = tuple(item.source_key for item in spec.crystal_waters)
    if set(water_residues) != set(declared_waters):
        raise PreparationError(
            "source waters and explicit crystal_waters differ: "
            f"source={water_residues!r}, declared={declared_waters!r}"
        )

    ligand_heavy = tuple(atom for atom in ligand_atoms if atom.heavy)
    if not ligand_heavy:
        raise PreparationError("ligand has no heavy atoms")
    interface: set[tuple[str, int, str, str]] = set()
    for atom in receptor_atoms:
        if atom.residue_name in WATER_NAMES or not atom.heavy:
            continue
        if any(
            dist(atom.coordinates, ligand.coordinates)
            <= spec.interface_cutoff_angstrom
            for ligand in ligand_heavy
        ):
            interface.add(atom.residue_key)
    ordered_interface = tuple(key for key in protein_residues if key in interface)
    if not ordered_interface:
        raise PreparationError("interface cutoff selected no receptor residues")
    classes = tuple(
        sorted({residue_chemistry_class(key[3]) for key in ordered_interface})
    )
    inspection = SourceInspection(
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        protein_residue_keys=protein_residues,
        water_residue_keys=water_residues,
        interface_residue_keys=ordered_interface,
        interface_chemistry_classes=classes,
        histidine_states=_histidine_states(receptor_atoms),
    )
    _validate_expectations(spec, inspection)
    return inspection


def _validate_expectations(
    spec: PreparationSpec, inspection: SourceInspection
) -> None:
    observed: Mapping[str, object] = {
        "receptor_protein_atoms": len(inspection.protein_atoms),
        "receptor_protein_residues": len(inspection.protein_residue_keys),
        "ligand_atoms": len(inspection.ligand_atoms),
        "crystal_water_atoms": len(inspection.crystal_water_atoms),
        "histidine_states": dict(inspection.histidine_states),
    }
    expected = {
        "receptor_protein_atoms": spec.expectations.receptor_protein_atoms,
        "receptor_protein_residues": spec.expectations.receptor_protein_residues,
        "ligand_atoms": spec.expectations.ligand_atoms,
        "crystal_water_atoms": spec.expectations.crystal_water_atoms,
        "histidine_states": dict(spec.expectations.histidine_states),
    }
    if observed != expected:
        raise PreparationError(
            "source inventory differs from preparation expectations: "
            f"expected={expected!r}, observed={observed!r}"
        )


def _histidine_names(atoms: tuple[PdbAtom, ...]) -> Mapping[tuple[str, int, str], str]:
    by_residue: dict[tuple[str, int, str], set[str]] = {}
    for atom in atoms:
        if atom.residue_name == "HIS":
            key = (atom.chain_id, atom.residue_number, atom.insertion_code)
            by_residue.setdefault(key, set()).add(atom.name)
    result: dict[tuple[str, int, str], str] = {}
    for key, names in by_residue.items():
        if {"HD1", "HE2"}.issubset(names):
            result[key] = "HIP"
        elif "HD1" in names:
            result[key] = "HID"
        elif "HE2" in names:
            result[key] = "HIE"
        else:
            raise PreparationError(f"cannot assign histidine protonation for {key!r}")
    return result


def write_receptor_views(
    spec: PreparationSpec,
    *,
    md_path: Path,
    dry_path: Path,
) -> Mapping[str, object]:
    """Write normalized receptor inputs: wet for MD, water-free for scoring."""

    inspection = inspect_sources(spec)
    source_lines = spec.receptor.source.path.read_text(encoding="utf-8").splitlines()
    histidines = _histidine_names(inspection.receptor_atoms)
    water_serials = {
        atom.serial for atom in inspection.receptor_atoms if atom.residue_name in WATER_NAMES
    }
    md_lines: list[str] = []
    dry_lines: list[str] = []
    for line in source_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            residue_name = line[17:20].strip().upper()
            if residue_name == "HIS":
                key = (
                    line[21:22].strip(),
                    int(line[22:26]),
                    line[26:27].strip(),
                )
                line = line[:17] + f"{histidines[key]:>3}" + line[20:]
            md_lines.append(line)
            if residue_name not in WATER_NAMES:
                dry_lines.append(line)
            continue
        if line.startswith("CONECT"):
            try:
                referenced = {int(item) for item in line.split()[1:]}
            except ValueError as exc:
                raise PreparationError("invalid PDB CONECT record") from exc
            md_lines.append(line)
            if referenced.isdisjoint(water_serials):
                dry_lines.append(line)
            continue
        if line.startswith("END"):
            continue
        if line.startswith("TER"):
            md_lines.append(line)
            dry_lines.append(line)
    md_lines.append("END")
    dry_lines.append("END")
    for destination, lines in ((Path(md_path), md_lines), (Path(dry_path), dry_lines)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "md_receptor_preserves_crystal_waters": True,
        "md_crystal_water_count": len(inspection.water_residue_keys),
        "dry_receptor_water_count": 0,
        "histidine_states": dict(inspection.histidine_states),
    }


def parameterization_request(
    spec: PreparationSpec, output_dir: Path
) -> SmallMoleculeParameterizationRequest:
    ligand = spec.ligand
    return SmallMoleculeParameterizationRequest(
        source=ligand.source.path,
        output_dir=output_dir,
        input_format="sdf",
        residue_name=ligand.residue_name,
        formal_charge=ligand.formal_charge,
        charge_method=ligand.charge_method,
        atom_type=ligand.atom_type,
        force_field=ligand.force_field,
        molecule_filename=f"{spec.case_id}.mol2",
        parameter_filename=f"{spec.case_id}.frcmod",
    )


def _quoted(path: Path) -> str:
    value = str(Path(path).resolve())
    if any(item in value for item in ('"', "\n", "\r", "\x00")):
        raise PreparationError(f"path cannot be represented in tleap input: {path}")
    return f'"{value}"'


def initial_tleap_text(
    spec: PreparationSpec,
    *,
    md_receptor: Path,
    dry_receptor: Path,
    ligand_mol2: Path,
    ligand_frcmod: Path,
) -> str:
    return "\n".join(
        (
            f"source leaprc.protein.{spec.force_fields.protein}",
            f"source leaprc.{spec.ligand.force_field}",
            f"source leaprc.water.{spec.force_fields.water}",
            f"set default PBRadii {spec.force_fields.pb_radii}",
            f"loadamberparams {_quoted(ligand_frcmod)}",
            f"DRY_REC = loadpdb {_quoted(dry_receptor)}",
            f"MD_REC = loadpdb {_quoted(md_receptor)}",
            f"LIG = loadmol2 {_quoted(ligand_mol2)}",
            "DRY = combine { DRY_REC LIG }",
            "check DRY",
            "saveamberparm DRY dry.prmtop dry.inpcrd",
            "NOIONS = combine { MD_REC LIG }",
            f"solvateOct NOIONS TIP3PBOX {spec.solvation.padding_angstrom:g} iso",
            "saveamberparm NOIONS noions.prmtop noions.inpcrd",
            "quit",
        )
    )


def salted_tleap_text(
    spec: PreparationSpec,
    *,
    md_receptor: Path,
    ligand_mol2: Path,
    ligand_frcmod: Path,
    sodium_count: int,
    chloride_count: int,
) -> str:
    return "\n".join(
        (
            f"source leaprc.protein.{spec.force_fields.protein}",
            f"source leaprc.{spec.ligand.force_field}",
            f"source leaprc.water.{spec.force_fields.water}",
            f"set default PBRadii {spec.force_fields.pb_radii}",
            f"loadamberparams {_quoted(ligand_frcmod)}",
            f"MD_REC = loadpdb {_quoted(md_receptor)}",
            f"LIG = loadmol2 {_quoted(ligand_mol2)}",
            "SYSTEM = combine { MD_REC LIG }",
            f"solvateOct SYSTEM TIP3PBOX {spec.solvation.padding_angstrom:g} iso",
            f"addIonsRand SYSTEM Na+ {sodium_count} Cl- {chloride_count}",
            "check SYSTEM",
            "saveamberparm SYSTEM solvated.prmtop solvated.inpcrd",
            "quit",
        )
    )


def _receipt_commands(receipt: object, step: str) -> list[dict[str, object]]:
    return [
        {"step": step, "argv": list(command.argv)}
        for command in receipt.commands
    ]


def build_preparation_plan(
    spec: PreparationSpec,
    *,
    tool_config: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Resolve and validate a preparation without creating its destination."""

    inspection = inspect_sources(spec)
    tools = create_amber_preparation_tools(tool_config or {})
    with tempfile.TemporaryDirectory(prefix="mmpbsa-preparation-plan-") as raw:
        root = Path(raw)
        md_receptor = root / "receptor_md.pdb"
        dry_receptor = root / "receptor_dry.pdb"
        receptor_views = write_receptor_views(
            spec, md_path=md_receptor, dry_path=dry_receptor
        )
        parameterization = tools.parameterize(
            parameterization_request(spec, root / "parameterization"),
            dry_run=True,
        )
        ligand_mol2 = root / "future" / f"{spec.case_id}.mol2"
        ligand_frcmod = root / "future" / f"{spec.case_id}.frcmod"
        ligand_mol2.parent.mkdir(parents=True)
        ligand_mol2.write_text("dry-run placeholder\n", encoding="utf-8")
        ligand_frcmod.write_text("dry-run placeholder\n", encoding="utf-8")
        initial_text = initial_tleap_text(
            spec,
            md_receptor=md_receptor,
            dry_receptor=dry_receptor,
            ligand_mol2=ligand_mol2,
            ligand_frcmod=ligand_frcmod,
        )
        initial = tools.assemble(
            AmberAssemblyRequest(
                input_text=initial_text,
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
                output_dir=root / "initial",
            ),
            dry_run=True,
        )
        commands = [
            *_receipt_commands(parameterization, "parameterize_ligand"),
            *_receipt_commands(initial, "assemble_dry_and_unsalted_md"),
        ]
        root_text = str(root)
        case_root_text = str(spec.config_path.parent)
        commands = [
            {
                **command,
                "argv": [
                    str(item)
                    .replace(root_text, "${PLAN_ROOT}")
                    .replace(case_root_text, "${CASE_ROOT}")
                    for item in command["argv"]
                ],
            }
            for command in commands
        ]
        initial_text = initial_text.replace(root_text, "${PLAN_ROOT}")
    canonical = {
        "case_id": spec.case_id,
        "entities": {
            "receptor": {
                "entity_id": spec.receptor.entity_id,
                "kind": "protein",
                "interface_chemistry_classes": list(
                    inspection.interface_chemistry_classes
                ),
            },
            "ligand": {
                "entity_id": spec.ligand.entity_id,
                "kind": "small_molecule",
                "formal_charge": spec.ligand.formal_charge,
                "qc_selection": "heavy_atoms",
            },
            "crystal_waters": [
                {
                    "entity_id": item.entity_id,
                    "kind": "water",
                    "source_identity": _residue_key_dict(item.source_key),
                    "scoring_policy": "excluded",
                }
                for item in spec.crystal_waters
            ],
        },
        "primary_binding": {
            "binding_id": "primary",
            "receptor_entity_ids": [spec.receptor.entity_id],
            "ligand_entity_ids": [spec.ligand.entity_id],
            "excluded_entity_ids": [item.entity_id for item in spec.crystal_waters],
            "cross_entity_bonds": [],
        },
        "systems": {
            "md": {
                "role_neutral": True,
                "explicit_solvent": True,
                "crystal_waters_retained": True,
            },
            "scoring": {
                "dry": True,
                "all_water_excluded": True,
                "topology_roles": [
                    "complex_topology",
                    "receptor_topology",
                    "ligand_topology",
                ],
            },
        },
        "source_inspection": inspection.as_dict(),
        "receptor_views": dict(receptor_views),
        "commands": commands,
        "deferred_two_phase": {
            "step": "assemble_salted_md",
            "state": "deferred_until_noions_inventory",
            "reason": (
                "Na+/Cl- counts require the actual noions water count and "
                "integral solute charge"
            ),
            "target_salt_concentration_molar": (
                spec.solvation.salt_concentration_molar
            ),
            "salt_pair_formula": "round(N_water * target_molar / 55.5)",
            "neutralization_policy": (
                "add counterions for integral solute charge, then add equal "
                "Na+/Cl- salt pairs"
            ),
            "resolved_counts_recorded_by": "materialized preparation receipt",
        },
        "generated_tleap_inputs": {
            "initial": initial_text,
        },
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "mmpbsa.preparation.plan.v1",
        "dry_run": True,
        "writes_destination": False,
        "plan_digest": digest,
        **canonical,
    }


__all__ = [
    "BACKBONE_NAMES",
    "SourceInspection",
    "WATER_NAMES",
    "build_preparation_plan",
    "initial_tleap_text",
    "inspect_sources",
    "parameterization_request",
    "read_pdb_atoms",
    "read_sdf_atoms",
    "residue_chemistry_class",
    "salted_tleap_text",
    "write_receptor_views",
]

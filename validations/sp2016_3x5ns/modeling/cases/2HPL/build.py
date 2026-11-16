#!/usr/bin/env python3
"""Build and seal the validation-local 2HPL PreparedCase proof.

This is deliberately a single-case implementation.  Every modeling decision
belongs to ``build_plan.json`` and no SP2016-specific chemistry enters
``src/mmpbsa``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VALIDATION_ROOT = HERE.parents[2]
REPOSITORY_ROOT = HERE.parents[4]
PLAN_PATH = HERE / "build_plan.json"
DEFAULT_OUTPUT = HERE / "_integration_build"
BACKBONE_NAMES = frozenset({"N", "CA", "C", "O"})
WATER_NAMES = frozenset({"WAT", "HOH", "SOL"})
ION_NAMES = frozenset({"NA", "NA+", "CL", "CL-"})
TERMINAL_TEMPLATES = {
    "receptor_n": {
        "amber_template": "NSER",
        "auth_residue_id": "A:12",
        "tleap_mapping": ("Mapped residue SER, term: Terminal/beginning, seq. number: 0 to: NSER."),
    },
    "receptor_c": {
        "amber_template": "CSER",
        "auth_residue_id": "A:111",
        "tleap_mapping": ("Mapped residue SER, term: Terminal/last, seq. number: 99 to: CSER."),
    },
    "ligand_n": {
        "amber_template": "NASP",
        "auth_residue_id": "B:802",
        "tleap_mapping": (
            "Mapped residue ASP, term: Terminal/beginning, seq. number: 100 to: NASP."
        ),
    },
    "ligand_c": {
        "amber_template": "CGLY",
        "auth_residue_id": "B:806",
        "tleap_mapping": ("Mapped residue GLY, term: Terminal/last, seq. number: 104 to: CGLY."),
    },
}


class BuildError(RuntimeError):
    """A fail-closed validation build error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BuildError(f"{path} must contain one mapping")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def load_build_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    plan = _load_json(path)
    validate_build_plan(plan)
    return plan


def validate_build_plan(plan: Mapping[str, object]) -> None:
    """Reject drift from the audited single-case scientific plan."""

    required = {
        "schema_version",
        "case_id",
        "pdb_id",
        "source",
        "entities",
        "source_filter",
        "protonation",
        "terminal_policy",
        "force_field",
        "solvation",
        "interface",
        "outputs",
    }
    if set(plan) != required:
        raise BuildError(f"build plan fields differ: {sorted(set(plan) ^ required)}")
    expected_scalars = {
        "schema_version": "sp2016.2hpl_build_plan.v1",
        "case_id": "sp2016_16",
        "pdb_id": "2HPL",
    }
    for key, expected in expected_scalars.items():
        if plan[key] != expected:
            raise BuildError(f"build plan {key} must be {expected!r}")
    source = _mapping(plan["source"], "source")
    if source.get("sha256") != "771a6653e8dc513cf76a6a7e897f0f23a1ca85a95afc67b74aee52a4725e7e6b":
        raise BuildError("2HPL source SHA-256 drifted")
    assembly = _mapping(source.get("biological_assembly"), "source biological assembly")
    if (
        assembly.get("assembly_id"),
        assembly.get("author_defined"),
        assembly.get("operator_id"),
        assembly.get("operator"),
        assembly.get("label_asym_ids"),
    ) != ("1", True, "1", "identity", ["A", "B", "C", "D"]):
        raise BuildError("2HPL biological assembly 1 identity contract drifted")
    entities = _mapping(plan["entities"], "entities")
    receptor = _mapping(entities.get("receptor"), "receptor")
    ligand = _mapping(entities.get("ligand"), "ligand")
    if (receptor.get("auth_chain"), receptor.get("residues")) != ("A", "12-111"):
        raise BuildError("2HPL receptor selector drifted")
    if (
        ligand.get("auth_chain"),
        ligand.get("residues"),
        ligand.get("sequence"),
    ) != ("B", "802-806", "DDLYG"):
        raise BuildError("2HPL peptide selector or identity drifted")
    for entity, expected_templates in (
        (receptor, {"n_terminal": "NSER", "c_terminal": "CSER"}),
        (ligand, {"n_terminal": "NASP", "c_terminal": "CGLY"}),
    ):
        termini = _mapping(entity.get("termini"), "entity termini")
        if (
            termini.get("n_terminal"),
            termini.get("c_terminal"),
        ) != ("standard_charged", "standard_charged"):
            raise BuildError("2HPL must use deposited entities with charged termini")
        templates = _mapping(
            termini.get("expected_amber_templates"), "expected Amber terminal templates"
        )
        if dict(templates) != expected_templates:
            raise BuildError("2HPL Amber terminal-template contract drifted")
    terminal_policy = _mapping(plan["terminal_policy"], "terminal_policy")
    if "validation default" not in str(terminal_policy.get("ownership", "")):
        raise BuildError("2HPL terminal chemistry must be a validation-owned decision")
    force_field = _mapping(plan["force_field"], "force_field")
    if (
        force_field.get("protein_and_peptide"),
        force_field.get("water"),
        force_field.get("pb_radii"),
    ) != ("ff14SB", "TIP3P", "mbondi2"):
        raise BuildError("2HPL force-field contract drifted")
    solvation = _mapping(plan["solvation"], "solvation")
    if (
        solvation.get("shape"),
        float(solvation.get("buffer_angstrom", 0)),
        float(solvation.get("salt_target_molar", 0)),
    ) != ("truncated_octahedron", 10.0, 0.15):
        raise BuildError("2HPL solvation contract drifted")
    outputs = _mapping(plan["outputs"], "outputs")
    if outputs.get("starts_md") is not False:
        raise BuildError("2HPL build proof must not start MD")
    if outputs.get("target_must_not_exist") is not True:
        raise BuildError("2HPL output must be create-only")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise BuildError(f"{label} must be a string-keyed mapping")
    return value


def _require_tool(name: str) -> Path:
    resolved = shutil.which(name)
    if resolved is None:
        raise BuildError(f"required executable is not on PATH: {name}")
    return Path(resolved).resolve()


def _conda_package_version(executable: Path, package: str) -> str | None:
    metadata_root = executable.parent.parent / "conda-meta"
    candidates = sorted(metadata_root.glob(f"{package}-*.json"))
    if not candidates:
        return None
    try:
        return str(_load_json(candidates[-1]).get("version"))
    except (OSError, json.JSONDecodeError, BuildError):
        return None


def _run_command(
    *,
    root: Path,
    argv: Sequence[str],
    cwd: Path,
    log: Path,
    records: list[dict[str, object]],
) -> None:
    if Path(argv[0]).name in {"tleap", "teLeap"} and (cwd / "leap.log").exists():
        raise BuildError(f"refusing to append to a pre-existing tleap cwd log: {cwd / 'leap.log'}")
    started = _utc_now()
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    record = {
        "argv": list(argv),
        "cwd": cwd.relative_to(root).as_posix(),
        "started_at": started,
        "completed_at": _utc_now(),
        "return_code": completed.returncode,
        "log": _file_record(root, log),
    }
    records.append(record)
    if completed.returncode != 0:
        raise BuildError(f"command failed ({completed.returncode}); see {log}")
    if Path(argv[0]).name in {"tleap", "teLeap"} and "Errors = 0" not in completed.stdout:
        raise BuildError(f"tleap did not report a clean exit; see {log}")


def _capture_tleap_engine_log(
    *, root: Path, destination: Path, command_record: dict[str, object]
) -> dict[str, dict[str, str]]:
    """Validate exact terminal mappings and move LEaP's cwd log under logs/."""

    generated = root / "leap.log"
    if not generated.is_file():
        raise BuildError("tleap did not create its engine log in the staging directory")
    content = generated.read_text(encoding="utf-8", errors="replace")
    evidence: dict[str, dict[str, str]] = {}
    for label, expected in TERMINAL_TEMPLATES.items():
        mapping = expected["tleap_mapping"]
        if content.count(mapping) != 1:
            raise BuildError(
                f"tleap engine log does not contain exactly one {label} template mapping"
            )
        evidence[label] = {
            "auth_residue_id": expected["auth_residue_id"],
            "amber_template": expected["amber_template"],
            "log_evidence": mapping,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(generated, destination)
    command_record["engine_log"] = _file_record(root, destination)
    command_record["terminal_template_mappings"] = evidence
    return evidence


def _extract_selected_pdb(
    source: Path, target: Path, plan: Mapping[str, object]
) -> dict[str, object]:
    import gemmi

    structure = gemmi.read_structure(str(source))
    if len(structure) != 1:
        raise BuildError(f"expected one coordinate model, observed {len(structure)}")
    if len(structure.assemblies) != 1:
        raise BuildError(f"expected one biological assembly, observed {len(structure.assemblies)}")
    assembly = structure.assemblies[0]
    if (
        assembly.name != "1"
        or not assembly.author_determined
        or assembly.oligomeric_details != "dimeric"
        or len(assembly.generators) != 1
    ):
        raise BuildError("official mmCIF biological assembly 1 metadata drifted")
    generator = assembly.generators[0]
    if set(generator.subchains) != {"A", "B", "C", "D"} or len(generator.operators) != 1:
        raise BuildError("official mmCIF biological assembly 1 membership drifted")
    operator = generator.operators[0]
    if operator.name != "1" or not operator.transform.is_identity():
        raise BuildError("official mmCIF biological assembly 1 is no longer identity-only")
    output = gemmi.Structure()
    output.name = "2HPL_A_B_deposited_entities"
    model = gemmi.Model("1")
    audit: dict[str, object] = {
        "biological_assembly": {
            "assembly_id": assembly.name,
            "author_defined": assembly.author_determined,
            "oligomeric_details": assembly.oligomeric_details,
            "operator_id": operator.name,
            "operator": "identity",
            "label_asym_ids": sorted(generator.subchains),
            "selected_polymer_label_asym_ids": ["A", "B"],
            "excluded_water_label_asym_ids": ["C", "D"],
        },
        "chains": {},
        "renamed_residues": [],
    }
    expected = {"A": (12, 111, 100), "B": (802, 806, 5)}
    for chain_name in ("A", "B"):
        source_chain = structure[0].find_chain(chain_name)
        if source_chain is None:
            raise BuildError(f"source is missing auth chain {chain_name}")
        residues = [item for item in source_chain if item.entity_type == gemmi.EntityType.Polymer]
        first, last, count = expected[chain_name]
        if (
            len(residues) != count
            or residues[0].seqid.num != first
            or residues[-1].seqid.num != last
            or [item.seqid.num for item in residues] != list(range(first, last + 1))
        ):
            raise BuildError(f"source polymer range drifted for auth chain {chain_name}")
        selected_chain = gemmi.Chain(chain_name)
        for residue in residues:
            if residue.seqid.icode.strip():
                raise BuildError("2HPL proof does not admit insertion codes")
            selected = residue.clone()
            for atom in selected:
                if atom.altloc not in ("\x00", " ") or abs(float(atom.occ) - 1.0) > 1.0e-6:
                    raise BuildError(
                        "selected atom has altloc/partial occupancy: "
                        f"{chain_name}:{residue.seqid.num}:{atom.name}"
                    )
            if chain_name == "A" and residue.seqid.num == 86:
                if residue.name != "HIS":
                    raise BuildError("expected deposited HIS A86")
                selected.name = "HIE"
                audit["renamed_residues"].append(
                    {"auth_chain": "A", "auth_seq_id": 86, "from": "HIS", "to": "HIE"}
                )
            selected_chain.add_residue(selected)
        model.add_chain(selected_chain)
        audit["chains"][chain_name] = {
            "first_auth_seq_id": first,
            "last_auth_seq_id": last,
            "residue_count": count,
        }
    output.add_model(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.with_suffix(".raw.pdb")
    output.write_pdb(str(raw))
    lines = [
        line
        for line in raw.read_text(encoding="utf-8").splitlines()
        if not line.startswith("CRYST1")
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw.unlink()
    audit["selected_pdb"] = _file_record(target.parents[1], target)
    audit["source_policy"] = (
        "selected deposited polymer entities from biological assembly 1 under its identity "
        "operator; no atom or residue rebuilding; water-only label asym C/D excluded"
    )
    return audit


def _tleap_header() -> list[str]:
    return [
        "source leaprc.protein.ff14SB",
        "source leaprc.water.tip3p",
        "set default PBRadii mbondi2",
    ]


def _write_tleap_probe(root: Path, selected: Path) -> Path:
    script = root / "inputs" / "tleap_probe.in"
    complex_top = root / "artifacts" / "scoring" / "complex.prmtop"
    complex_crd = root / "artifacts" / "scoring" / "complex.inpcrd"
    noion_top = root / "intermediate" / "solvated_noions.prmtop"
    noion_crd = root / "intermediate" / "solvated_noions.rst7"
    lines = [
        *_tleap_header(),
        f"complex = loadPdb {selected}",
        "check complex",
        f"saveAmberParm complex {complex_top} {complex_crd}",
        "solvateOct complex TIP3PBOX 10.0",
        f"saveAmberParm complex {noion_top} {noion_crd}",
        "quit",
    ]
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script


def _write_tleap_final(
    root: Path,
    selected: Path,
    *,
    net_charge: int,
    salt_pairs: int,
) -> Path:
    script = root / "inputs" / "tleap_final.in"
    solvated_top = root / "intermediate" / "solvated.prmtop"
    solvated_crd = root / "intermediate" / "solvated.rst7"
    if net_charge < 0:
        neutralize = "addIonsRand complex Na+ 0"
    elif net_charge > 0:
        neutralize = "addIonsRand complex Cl- 0"
    else:
        neutralize = "# complex is already neutral"
    lines = [
        *_tleap_header(),
        f"complex = loadPdb {selected}",
        "solvateOct complex TIP3PBOX 10.0",
        neutralize,
        f"addIonsRand complex Na+ {salt_pairs} Cl- {salt_pairs}",
        f"saveAmberParm complex {solvated_top} {solvated_crd}",
        "quit",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script


def _validate_dry_structure(
    dry: object, tleap_terminal_mappings: Mapping[str, Mapping[str, str]]
) -> dict[str, object]:
    if len(dry.residues) != 105:
        raise BuildError(f"dry complex residue count differs: {len(dry.residues)}")
    sequence = "".join(
        {"ASP": "D", "LEU": "L", "TYR": "Y", "GLY": "G"}.get(residue.name, "X")
        for residue in dry.residues[100:105]
    )
    if sequence != "DDLYG":
        raise BuildError(f"dry peptide identity differs: {sequence}")
    expected_terminal_atoms = {
        "receptor_n": (0, {"H1", "H2", "H3"}),
        "receptor_c": (99, {"OXT"}),
        "ligand_n": (100, {"H1", "H2", "H3"}),
        "ligand_c": (104, {"OXT"}),
    }
    terminal_receipt = {}
    for label, (index, required) in expected_terminal_atoms.items():
        residue = dry.residues[index]
        atom_names = {atom.name for atom in residue.atoms}
        if not required <= atom_names:
            raise BuildError(f"{label} is not a standard charged terminus")
        terminal_receipt[label] = {
            "residue_index": index + 1,
            "residue_name": residue.name,
            "auth_residue_id": TERMINAL_TEMPLATES[label]["auth_residue_id"],
            "amber_template": TERMINAL_TEMPLATES[label]["amber_template"],
            "required_atoms": sorted(required),
            "residue_charge": sum(float(atom.charge) for atom in residue.atoms),
            "tleap_log_evidence": tleap_terminal_mappings[label]["log_evidence"],
        }
    charge = sum(float(atom.charge) for atom in dry.atoms)
    rounded = round(charge)
    if abs(charge - rounded) > 1.0e-5:
        raise BuildError(f"dry complex charge is not integral: {charge}")
    radius_set = [str(item) for item in dry.parm_data.get("RADIUS_SET", [])]
    if not any("mbondi2" in item.lower() for item in radius_set):
        raise BuildError(f"dry topology does not report mbondi2 radii: {radius_set}")
    return {
        "atom_count": len(dry.atoms),
        "residue_count": len(dry.residues),
        "net_charge": charge,
        "integral_net_charge": rounded,
        "radius_set": radius_set,
        "terminal_checks": terminal_receipt,
    }


def _component_coordinates(root: Path, dry: object, receptor_atoms: int) -> None:
    import parmed as pmd

    scoring = root / "artifacts" / "scoring"
    receptor = pmd.load_file(str(scoring / "receptor.prmtop"))
    ligand = pmd.load_file(str(scoring / "ligand.prmtop"))
    if len(receptor.atoms) != receptor_atoms or len(receptor.atoms) + len(ligand.atoms) != len(
        dry.atoms
    ):
        raise BuildError("ante-MMPBSA component atom counts do not partition dry complex")
    receptor.coordinates = dry.coordinates[:receptor_atoms]
    ligand.coordinates = dry.coordinates[receptor_atoms:]
    receptor.save(str(scoring / "receptor.inpcrd"), overwrite=True)
    ligand.save(str(scoring / "ligand.inpcrd"), overwrite=True)


def _same_atom(left: object, right: object) -> bool:
    return (
        (
            str(left.residue.name).upper(),
            str(left.name),
            int(left.atomic_number or 0),
        )
        == (
            str(right.residue.name).upper(),
            str(right.name),
            int(right.atomic_number or 0),
        )
        and abs(float(left.charge) - float(right.charge)) <= 1.0e-8
        and abs(float(left.mass) - float(right.mass)) <= 1.0e-8
    )


def _convert_gromacs(root: Path, dry: object, solvated: object) -> tuple[object, dict[str, object]]:
    import parmed as pmd

    md = root / "artifacts" / "md"
    md.mkdir(parents=True, exist_ok=True)
    top = md / "topol.top"
    gro = md / "system.gro"
    solvated.save(str(top), format="gromacs", overwrite=True)
    solvated.save(str(gro), format="gro", overwrite=True)
    projected = pmd.load_file(str(top), xyz=str(gro))
    if len(projected.atoms) != len(solvated.atoms):
        raise BuildError("GROMACS conversion changed MD atom count")
    for index, (dry_atom, md_atom) in enumerate(
        zip(dry.atoms, projected.atoms, strict=False), start=1
    ):
        if not _same_atom(dry_atom, md_atom):
            raise BuildError(f"dry-to-GROMACS mapping differs at atom {index}")
    return projected, {
        "backend": "gromacs",
        "conversion": "ParmEd AmberParm -> standalone GROMACS topology/GRO",
        "atom_count": len(projected.atoms),
        "residue_count": len(projected.residues),
        "dry_prefix_atom_count": len(dry.atoms),
        "dry_prefix_identity_charge_mass_exact": True,
        "box": [float(item) for item in projected.box],
    }


def _indices(atoms: Sequence[object]) -> list[int]:
    return sorted(int(atom.idx) + 1 for atom in atoms)


def _selection(md: Sequence[int], scoring: Sequence[int]) -> dict[str, object]:
    return {"md_atom_indices": list(md), "scoring_atom_indices": list(scoring)}


def _dry_entity(
    *,
    entity_id: str,
    kind: str,
    dry: object,
    residue_start: int,
    residue_stop: int,
    auth_chain: str,
    auth_first: int,
    interface_residue_indices: set[int] | None = None,
) -> dict[str, object]:
    residues = list(dry.residues[residue_start:residue_stop])
    atoms = [atom for residue in residues for atom in residue.atoms]
    all_indices = _indices(atoms)
    heavy = _indices([atom for atom in atoms if int(atom.atomic_number or 0) > 1])
    backbone = _indices([atom for atom in atoms if atom.name in BACKBONE_NAMES])
    named = {
        "all": _selection(all_indices, all_indices),
        "heavy_atoms": _selection(heavy, heavy),
        "backbone": _selection(backbone, backbone),
    }
    if interface_residue_indices is not None:
        interface = _indices(
            [
                atom
                for residue in residues
                if int(residue.idx) + 1 in interface_residue_indices
                for atom in residue.atoms
                if atom.name in BACKBONE_NAMES
            ]
        )
        if not interface:
            raise BuildError("2HPL interface-backbone selection is empty")
        named["interface_backbone"] = _selection(interface, interface)
    identities = [
        {
            "residue_id": f"{auth_chain}_{auth_first + offset}",
            "residue_name": residue.name,
            "md_residue_index": int(residue.idx) + 1,
            "scoring_residue_index": int(residue.idx) + 1,
        }
        for offset, residue in enumerate(residues)
    ]
    return {
        "entity_id": entity_id,
        "kind": kind,
        "md_atom_indices": all_indices,
        "scoring_atom_indices": all_indices,
        "residues": identities,
        "named_selections": named,
        "features": [],
        "metadata": {
            "auth_chain": auth_chain,
            "auth_first_residue": auth_first,
            "auth_last_residue": auth_first + len(residues) - 1,
            "terminal_policy": "deposited entity; standard charged N and C termini",
        },
    }


def _md_only_entity(entity_id: str, kind: str, residues: Sequence[object]) -> dict[str, object]:
    atoms = [atom for residue in residues for atom in residue.atoms]
    all_indices = _indices(atoms)
    heavy = _indices([atom for atom in atoms if int(atom.atomic_number or 0) > 1])
    return {
        "entity_id": entity_id,
        "kind": kind,
        "md_atom_indices": all_indices,
        "scoring_atom_indices": [],
        "residues": [
            {
                "residue_id": f"{entity_id}_{ordinal:05d}",
                "residue_name": residue.name,
                "md_residue_index": int(residue.idx) + 1,
                "scoring_residue_index": None,
            }
            for ordinal, residue in enumerate(residues, start=1)
        ],
        "named_selections": {
            "all": _selection(all_indices, []),
            "heavy_atoms": _selection(heavy, []),
        },
        "features": [],
        "metadata": {"scoring_policy": "excluded from dry endpoint scoring"},
    }


def _coordinate_vector(atom: object) -> tuple[float, float, float]:
    return float(atom.xx), float(atom.xy), float(atom.xz)


def build_inventory(dry: object, md: object, cutoff: float) -> dict[str, object]:
    # ParmEd atoms are not iterable coordinate vectors; use explicit xyz values.
    ligand_heavy = [
        atom
        for residue in dry.residues[100:105]
        for atom in residue.atoms
        if int(atom.atomic_number or 0) > 1
    ]
    cutoff_squared = cutoff * cutoff
    interface: set[int] = set()
    for residue in dry.residues[:100]:
        for receptor_atom in (atom for atom in residue.atoms if int(atom.atomic_number or 0) > 1):
            rx, ry, rz = _coordinate_vector(receptor_atom)
            if any(
                (rx - lx) ** 2 + (ry - ly) ** 2 + (rz - lz) ** 2 <= cutoff_squared
                for lx, ly, lz in (_coordinate_vector(atom) for atom in ligand_heavy)
            ):
                interface.add(int(residue.idx) + 1)
                break
    receptor = _dry_entity(
        entity_id="receptor",
        kind="protein",
        dry=dry,
        residue_start=0,
        residue_stop=100,
        auth_chain="A",
        auth_first=12,
        interface_residue_indices=interface,
    )
    ligand = _dry_entity(
        entity_id="ligand_peptide",
        kind="peptide",
        dry=dry,
        residue_start=100,
        residue_stop=105,
        auth_chain="B",
        auth_first=802,
    )
    water_residues = [
        residue for residue in md.residues[105:] if residue.name.upper() in WATER_NAMES
    ]
    ion_residues = [residue for residue in md.residues[105:] if residue.name.upper() in ION_NAMES]
    unknown = [
        residue.name
        for residue in md.residues[105:]
        if residue not in water_residues and residue not in ion_residues
    ]
    if unknown:
        raise BuildError(f"unexpected solvated residue types: {sorted(set(unknown))}")
    water = _md_only_entity("bulk_water", "water", water_residues)
    ions = _md_only_entity("bulk_ions", "ion", ion_residues)
    covered = sorted(
        atom for entity in (receptor, ligand, water, ions) for atom in entity["md_atom_indices"]
    )
    if covered != list(range(1, len(md.atoms) + 1)):
        raise BuildError("PreparedCase entities do not cover every solvated MD atom")
    return {
        "md_atom_count": len(md.atoms),
        "scoring_atom_count": len(dry.atoms),
        "source_md_atom_indices": list(range(1, len(dry.atoms) + 1)),
        "entities": [receptor, ligand, ions, water],
        "interface_receptor_md_residue_indices": sorted(interface),
        "water_residue_count": len(water_residues),
        "ion_residue_count": len(ion_residues),
    }


def make_draft_manifest(
    plan: Mapping[str, object],
    inventory: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Pure draft renderer used by ordinary contract tests and the real build."""

    return {
        "schema_version": "mmpbsa.prepared_case_draft.v1",
        "case_id": str(plan["case_id"]),
        "md": {
            "backend": "gromacs",
            "artifacts": [
                {
                    "role": "md_topology",
                    "source": "artifacts/md/topol.top",
                    "path": "md/topol.top",
                    "media_type": "text/plain",
                },
                {
                    "role": "md_coordinates",
                    "source": "artifacts/md/system.gro",
                    "path": "md/system.gro",
                    "media_type": "chemical/x-gro",
                },
                {
                    "role": "solvated_topology",
                    "source": "intermediate/solvated.prmtop",
                    "path": "md/solvated.prmtop",
                    "media_type": "chemical/x-amber-topology",
                },
                {
                    "role": "index",
                    "source": "artifacts/md/index.ndx",
                    "path": "md/index.ndx",
                    "media_type": "text/plain",
                },
                {
                    "role": "preparation_receipt",
                    "source": "provenance/preparation_receipt.json",
                    "path": "provenance/preparation_receipt.json",
                    "media_type": "application/json",
                },
            ],
            "metadata": {
                "force_field": "ff14SB",
                "water_model": "TIP3P",
                "box_shape": "truncated_octahedron",
                "buffer_angstrom": 10.0,
                "salt_target_molar": 0.15,
                "md_started": False,
            },
        },
        "scoring": {
            "backend": "ambertools",
            "radii_model": "mbondi2",
            "source_md_atom_indices": list(inventory["source_md_atom_indices"]),
            "artifacts": [
                {
                    "role": "complex_topology",
                    "source": "artifacts/scoring/complex.prmtop",
                    "path": "scoring/complex.prmtop",
                    "media_type": "application/octet-stream",
                },
                {
                    "role": "receptor_topology",
                    "source": "artifacts/scoring/receptor.prmtop",
                    "path": "scoring/receptor.prmtop",
                    "media_type": "application/octet-stream",
                },
                {
                    "role": "ligand_topology",
                    "source": "artifacts/scoring/ligand.prmtop",
                    "path": "scoring/ligand.prmtop",
                    "media_type": "application/octet-stream",
                },
                {
                    "role": "coordinates",
                    "source": "artifacts/scoring/complex.inpcrd",
                    "path": "scoring/complex.inpcrd",
                    "media_type": "application/octet-stream",
                },
            ],
            "metadata": {
                "dry_complex": True,
                "bulk_water_excluded": True,
                "ion_entities_excluded": True,
            },
        },
        "entities": list(inventory["entities"]),
        "primary_binding": {
            "binding_id": "primary",
            "receptor_entity_ids": ["receptor"],
            "ligand_entity_ids": ["ligand_peptide"],
            "excluded_entity_ids": ["bulk_ions", "bulk_water"],
            "topology_artifact_roles": {
                "complex": "complex_topology",
                "receptor": "receptor_topology",
                "ligand": "ligand_topology",
            },
            "cross_entity_bonds": [],
        },
        "provenance": dict(provenance),
    }


def _write_index(path: Path, *, md_atoms: int, dry_atoms: int, receptor_atoms: int) -> None:
    groups = {
        "System": list(range(1, md_atoms + 1)),
        "Complex": list(range(1, dry_atoms + 1)),
        "Receptor": list(range(1, receptor_atoms + 1)),
        "Ligand": list(range(receptor_atoms + 1, dry_atoms + 1)),
    }
    lines: list[str] = []
    for name, values in groups.items():
        lines.append(f"[ {name} ]")
        lines.extend(
            " ".join(str(item) for item in values[index : index + 15])
            for index in range(0, len(values), 15)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@contextmanager
def _create_only_transaction(output: Path) -> Iterator[Path]:
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise BuildError(f"output must not already exist: {output}")
    lock = output.parent / f".{output.name}.build.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise BuildError(f"build output is locked: {output}") from exc
    os.write(descriptor, f"pid={os.getpid()} started={_utc_now()}\n".encode())
    os.close(descriptor)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        yield staging
        if output.exists() or output.is_symlink():
            raise BuildError(f"output appeared during build: {output}")
        os.replace(staging, output)
    except BaseException:
        failure = (
            output.parent
            / f".{output.name}.failed.{datetime.now().strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
        )
        if staging.exists():
            try:
                os.replace(staging, failure)
                print(f"failed staging preserved at {failure}", file=sys.stderr)
            except OSError:
                pass
        raise
    finally:
        lock.unlink(missing_ok=True)


def _build_into(root: Path, plan: Mapping[str, object]) -> None:
    import gemmi
    import parmed as pmd

    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from mmpbsa.core import inspect_prepared_case_draft, seal_prepared_case_draft

    if int(str(pmd.__version__).partition(".")[0]) != 4:
        raise BuildError(f"ParmEd major 4 is required, observed {pmd.__version__}")
    source_config = _mapping(plan["source"], "source")
    source = (VALIDATION_ROOT / str(source_config["path"])).resolve()
    if not source.is_file() or _sha256(source) != source_config["sha256"]:
        raise BuildError("official 2HPL source is missing or differs from pinned SHA-256")

    tleap = _require_tool("tleap")
    ante = _require_tool("ante-MMPBSA.py")
    for directory in (
        "artifacts/md",
        "artifacts/scoring",
        "inputs",
        "intermediate",
        "logs",
        "provenance",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, object]] = []
    operations: list[dict[str, object]] = []
    selected = root / "inputs" / "selected_A_B.pdb"
    extraction = _extract_selected_pdb(source, selected, plan)
    operations.append(
        {
            "operation": "mmcif_polymer_selection",
            "tool": "gemmi",
            "version": gemmi.__version__,
            "result": extraction,
        }
    )

    probe = _write_tleap_probe(root, selected)
    _run_command(
        root=root,
        argv=[str(tleap), "-f", str(probe)],
        cwd=root,
        log=root / "logs" / "tleap_probe.log",
        records=commands,
    )
    probe_terminal_mappings = _capture_tleap_engine_log(
        root=root,
        destination=root / "logs" / "tleap_probe.engine.log",
        command_record=commands[-1],
    )
    scoring = root / "artifacts" / "scoring"
    dry = pmd.load_file(str(scoring / "complex.prmtop"), xyz=str(scoring / "complex.inpcrd"))
    dry_receipt = _validate_dry_structure(dry, probe_terminal_mappings)
    receptor_atoms = sum(len(residue.atoms) for residue in dry.residues[:100])
    ligand_mask = ":101-105"
    _run_command(
        root=root,
        argv=[
            str(ante),
            "-p",
            str(scoring / "complex.prmtop"),
            "-r",
            str(scoring / "receptor.prmtop"),
            "-l",
            str(scoring / "ligand.prmtop"),
            "-n",
            ligand_mask,
            "--radii=mbondi2",
        ],
        cwd=root,
        log=root / "logs" / "ante_mmpbsa.log",
        records=commands,
    )
    _component_coordinates(root, dry, receptor_atoms)
    operations.append(
        {
            "operation": "component_coordinate_projection",
            "tool": "ParmEd",
            "version": pmd.__version__,
            "receptor_atom_count": receptor_atoms,
            "ligand_atom_count": len(dry.atoms) - receptor_atoms,
        }
    )

    noions = pmd.load_file(
        str(root / "intermediate" / "solvated_noions.prmtop"),
        xyz=str(root / "intermediate" / "solvated_noions.rst7"),
    )
    preion_waters = sum(residue.name.upper() in WATER_NAMES for residue in noions.residues)
    target_molar = float(_mapping(plan["solvation"], "solvation")["salt_target_molar"])
    salt_pairs = round(preion_waters * target_molar / 55.5)
    final_script = _write_tleap_final(
        root, selected, net_charge=int(dry_receipt["integral_net_charge"]), salt_pairs=salt_pairs
    )
    _run_command(
        root=root,
        argv=[str(tleap), "-f", str(final_script)],
        cwd=root,
        log=root / "logs" / "tleap_final.log",
        records=commands,
    )
    final_terminal_mappings = _capture_tleap_engine_log(
        root=root,
        destination=root / "logs" / "tleap_final.engine.log",
        command_record=commands[-1],
    )
    solvated = pmd.load_file(
        str(root / "intermediate" / "solvated.prmtop"),
        xyz=str(root / "intermediate" / "solvated.rst7"),
    )
    if abs(sum(float(atom.charge) for atom in solvated.atoms)) > 1.0e-5:
        raise BuildError("final solvated system is not neutral")
    water_count = sum(residue.name.upper() in WATER_NAMES for residue in solvated.residues)
    ion_counts = {
        name: sum(residue.name.upper() == name for residue in solvated.residues)
        for name in ("NA+", "CL-")
    }
    md, gromacs_receipt = _convert_gromacs(root, dry, solvated)
    operations.append(
        {
            "operation": "amber_to_gromacs_projection",
            "tool": "ParmEd",
            "version": pmd.__version__,
            "result": gromacs_receipt,
        }
    )

    inventory = build_inventory(
        dry,
        md,
        float(
            _mapping(plan["interface"], "interface")["receptor_residue_heavy_atom_cutoff_angstrom"]
        ),
    )
    _write_index(
        root / "artifacts" / "md" / "index.ndx",
        md_atoms=len(md.atoms),
        dry_atoms=len(dry.atoms),
        receptor_atoms=receptor_atoms,
    )

    artifact_paths = [
        scoring / "complex.prmtop",
        scoring / "complex.inpcrd",
        scoring / "receptor.prmtop",
        scoring / "receptor.inpcrd",
        scoring / "ligand.prmtop",
        scoring / "ligand.inpcrd",
        root / "artifacts" / "md" / "topol.top",
        root / "artifacts" / "md" / "system.gro",
        root / "artifacts" / "md" / "index.ndx",
        root / "intermediate" / "solvated.prmtop",
        root / "intermediate" / "solvated.rst7",
        probe,
        final_script,
        selected,
    ]
    tools = {
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "gemmi": {"version": gemmi.__version__},
        "ParmEd": {"version": pmd.__version__},
        "AmberTools": {"version": _conda_package_version(tleap, "ambertools")},
        "tleap": {"path": str(tleap), "sha256": _sha256(tleap)},
        "ante-MMPBSA.py": {"path": str(ante), "sha256": _sha256(ante)},
    }
    receipt = {
        "schema_version": "sp2016.2hpl_preparation_receipt.v1",
        "case_id": plan["case_id"],
        "pdb_id": plan["pdb_id"],
        "created_at": _utc_now(),
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "bytes": source.stat().st_size,
            "revision_date": "2024-02-14",
            "biological_assembly": extraction["biological_assembly"],
        },
        "plan": {"path": str(PLAN_PATH), "sha256": _sha256(PLAN_PATH)},
        "tools": tools,
        "commands": commands,
        "python_operations": operations,
        "dry": dry_receipt,
        "terminal_chemistry": {
            "decision": (
                "Both complete deposited entities use free charged N and C termini as "
                "an explicit validation-owned scientific default."
            ),
            "not_implicit_tool_behavior": True,
            "probe_tleap_template_mappings": probe_terminal_mappings,
            "final_tleap_template_mappings": final_terminal_mappings,
            "topology_atom_and_charge_evidence": dry_receipt["terminal_checks"],
        },
        "solvation": {
            "shape": "truncated_octahedron",
            "buffer_angstrom": 10.0,
            "preion_water_count": preion_waters,
            "final_water_count": water_count,
            "dry_integral_net_charge": dry_receipt["integral_net_charge"],
            "neutralizing_counterion_count": abs(int(dry_receipt["integral_net_charge"])),
            "salt_pair_count": salt_pairs,
            "salt_target_molar": target_molar,
            "nominal_salt_molar": salt_pairs * 55.5 / preion_waters,
            "ion_residue_counts": ion_counts,
            "final_net_charge": sum(float(atom.charge) for atom in solvated.atoms),
        },
        "inventory": {key: value for key, value in inventory.items() if key != "entities"},
        "artifacts": [_file_record(root, path) for path in artifact_paths],
        "md_started": False,
        "build_boundary": (
            "Topology/coordinate construction and PreparedCase sealing only; "
            "no minimization or MD command was run."
        ),
    }
    receipt_path = root / "provenance" / "preparation_receipt.json"
    _write_json(receipt_path, receipt)
    draft = make_draft_manifest(
        plan,
        inventory,
        {
            "owner": "validations/sp2016_3x5ns/modeling/cases/2HPL",
            "source_sha256": source_config["sha256"],
            "build_plan_sha256": _sha256(PLAN_PATH),
            "preparation_receipt_sha256": _sha256(receipt_path),
            "modeling_scope": "validation-local 2HPL end-to-end build proof",
        },
    )
    draft_path = root / "prepared_case_draft.json"
    _write_json(draft_path, draft)
    inspection = inspect_prepared_case_draft(draft_path)
    _write_json(root / "provenance" / "draft_inspection.json", inspection.as_dict())
    prepared = seal_prepared_case_draft(draft_path, root / "sealed")
    sealed_manifest = root / "sealed" / "prepared_case.json"
    _write_json(
        root / "provenance" / "seal_summary.json",
        {
            "schema_version": "sp2016.2hpl_seal_summary.v1",
            "sealed_at": _utc_now(),
            "api": "mmpbsa.core.seal_prepared_case_draft",
            "case_id": prepared.case_id,
            "prepared_schema": prepared.manifest.schema_version,
            "prepared_case_manifest": _file_record(root, sealed_manifest),
            "draft": _file_record(root, draft_path),
            "projection_admission": inspection.projection_admission,
            "md_started": False,
        },
    )


def build(output: Path = DEFAULT_OUTPUT) -> Path:
    plan = load_build_plan()
    output = output.absolute()
    with _create_only_transaction(output) as staging:
        try:
            _build_into(staging, plan)
        except BaseException as exc:
            (staging / "logs").mkdir(parents=True, exist_ok=True)
            (staging / "logs" / "python_exception.log").write_text(
                "".join(traceback.format_exception(exc)), encoding="utf-8"
            )
            raise
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plan", action="store_true", help="Validate and print the immutable plan only"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.plan:
        print(json.dumps(load_build_plan(), indent=2, sort_keys=True))
        return 0
    destination = build(args.output)
    summary = _load_json(destination / "provenance" / "seal_summary.json")
    print(json.dumps({"output": str(destination), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

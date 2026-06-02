from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import (
    count_selected_residues,
    count_waters,
    mamba_command,
    run_logged,
    split_chain_spec,
    write_json_atomic,
    write_receptor_only_pdb,
    write_text_atomic,
)


STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASH",
    "ASN",
    "ASP",
    "CYM",
    "CYS",
    "CYX",
    "GLH",
    "GLN",
    "GLU",
    "GLY",
    "HID",
    "HIE",
    "HIP",
    "HIS",
    "ILE",
    "LEU",
    "LYN",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def prepare_input_structure(paths: Any, manifest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    policy = str(profile.get("amber_prep", {}).get("nonstandard_policy", "strip")).lower()
    findings = inspect_receptor_residues(paths.input / "complex.pdb", manifest["receptor_chains"])
    cofactor_count = int(manifest.get("receptor_cofactor_residue_count") or 0)
    blocking = []
    if findings["nonstandard_atom_residues"]:
        blocking.append("nonstandard ATOM receptor residues")
    if findings["hetero_residues"] and policy != "strip":
        blocking.append("receptor-chain HETATM residues")
    if blocking:
        raise SystemExit(
            "Small-molecule MMPBSA receptor preparation requires explicit handling for "
            + ", ".join(blocking)
            + ". Set amber_prep.nonstandard_policy: strip only for removable HETATM records."
        )

    dropped = write_receptor_only_pdb(paths.input / "complex.pdb", paths.input / "receptor.pdb", manifest["receptor_chains"])
    receptor_residues = count_selected_residues(paths.input / "receptor.pdb", manifest["receptor_chains"])
    if receptor_residues <= 0:
        raise SystemExit("Could not derive receptor residue count from receptor chains")
    receptor_mask_count = receptor_residues + cofactor_count
    ligand_first = receptor_mask_count + 1
    manifest.update(
        {
            "input_preparation": "small_molecule_complex_pose",
            "amber_prep_recipe": profile.get("amber_prep", {}).get("recipe", "protein_ligand_gaff2_tip3p"),
            "dropped_receptor_hetero_residues": dropped,
            "input_residue_findings": findings,
            "protein_receptor_residue_count": receptor_residues,
            "receptor_cofactor_residue_count": cofactor_count,
            "receptor_residue_count": receptor_mask_count,
            "ligand_residue_count": 1,
            "receptor_residue_mask": f":1-{receptor_mask_count}",
            "ligand_residue_mask": f":{ligand_first}",
            "note": "Residue masks assume tleap combines cleaned receptor first, receptor cofactors second, and one small-molecule ligand last.",
        }
    )
    return manifest


def inspect_receptor_residues(pdb: Path, receptor_chains: str) -> dict[str, Any]:
    selected = set(split_chain_spec(receptor_chains))
    nonstandard_atom: dict[tuple[str, str, str, str], str] = {}
    hetero: dict[tuple[str, str, str, str], str] = {}
    for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip()
        if chain not in selected:
            continue
        resname = line[17:20].strip()
        key = (chain, resname, line[22:26].strip(), line[26].strip())
        if line.startswith("HETATM"):
            hetero[key] = line[76:78].strip()
        elif resname not in STANDARD_AMINO_ACIDS:
            nonstandard_atom[key] = line[76:78].strip()
    return {
        "nonstandard_atom_residues": residue_records(nonstandard_atom),
        "hetero_residues": residue_records(hetero),
    }


def residue_records(items: dict[tuple[str, str, str, str], str]) -> list[dict[str, str]]:
    return [
        {"chain": chain, "resname": resname, "resseq": resseq, "icode": icode, "element": element}
        for (chain, resname, resseq, icode), element in sorted(items.items(), key=lambda item: (item[0][0], item[0][2], item[0][1], item[0][3]))
    ]


def run_amber_prepare(paths: Any, profile: dict[str, Any]) -> None:
    paths.amber.mkdir(parents=True, exist_ok=True)
    run_logged(
        mamba_command(
            profile,
            [
                "pdb4amber",
                "-i",
                str(paths.input / "receptor.pdb"),
                "-o",
                str(paths.amber / "receptor_pdb4amber.pdb"),
                "--dry",
                "--nohyd",
                "-l",
                str(paths.logs / "pdb4amber.detail.log"),
            ],
        ),
        paths.logs / "pdb4amber.log",
    )
    ligand_mol2 = paths.ligand / "ligand.mol2"
    ligand_frcmod = paths.ligand / "ligand.frcmod"
    ligand_libs = sorted([*paths.ligand.glob("*.lib"), *paths.ligand.glob("*.off")])
    manifest = read_manifest(paths)
    cofactor_files = [Path(item) for item in manifest.get("receptor_cofactor_files", [])]
    cofactor_frcmods = [Path(item) for item in manifest.get("receptor_cofactor_frcmods", [])]
    cofactor_libs = [Path(item) for item in manifest.get("receptor_cofactor_libs", [])]
    if not ligand_mol2.exists():
        raise SystemExit(f"Missing prepared ligand mol2: {ligand_mol2}")

    write_text_atomic(
        paths.amber / "build_presalt.leap",
        tleap_text(
            paths.amber / "receptor_pdb4amber.pdb",
            ligand_mol2,
            ligand_frcmod if ligand_frcmod.exists() else None,
            ligand_libs,
            None,
            profile,
            cofactor_files=cofactor_files,
            cofactor_frcmods=cofactor_frcmods,
            cofactor_libs=cofactor_libs,
        ),
    )
    run_logged(mamba_command(profile, ["tleap", "-f", "build_presalt.leap"]), paths.logs / "tleap_presalt.log", cwd=paths.amber)
    apply_radii(paths.amber / "complex_dry.prmtop", paths.logs / "parmed_complex_dry_radii.log", profile)

    water_count = count_waters(paths.amber / "system_solvated.pdb")
    salt_pairs = max(1, round(float(profile["system"]["salt_molar"]) * water_count / 55.5))
    write_text_atomic(
        paths.amber / "build_final.leap",
        tleap_text(
            paths.amber / "receptor_pdb4amber.pdb",
            ligand_mol2,
            ligand_frcmod if ligand_frcmod.exists() else None,
            ligand_libs,
            salt_pairs,
            profile,
            cofactor_files=cofactor_files,
            cofactor_frcmods=cofactor_frcmods,
            cofactor_libs=cofactor_libs,
        ),
    )
    run_logged(mamba_command(profile, ["tleap", "-f", "build_final.leap"]), paths.logs / "tleap_final.log", cwd=paths.amber)
    apply_radii(paths.amber / "complex_dry.prmtop", paths.logs / "parmed_complex_dry_final_radii.log", profile)
    apply_radii(paths.amber / "system_solvated.prmtop", paths.logs / "parmed_system_solvated_radii.log", profile)

    write_json_atomic(
        paths.amber / "summary.json",
        {
            "recipe": profile.get("amber_prep", {}).get("recipe", "protein_ligand_gaff2_tip3p"),
            "leaprc": [profile["amber_prep"]["protein_ff"], profile["amber_prep"]["ligand_ff"], profile["amber_prep"]["water_ff"]],
            "ligand_mol2": str(ligand_mol2),
            "ligand_frcmod": str(ligand_frcmod) if ligand_frcmod.exists() else "",
            "ligand_libs": [str(path) for path in ligand_libs],
            "receptor_cofactor_files": [str(path) for path in cofactor_files],
            "receptor_cofactor_frcmods": [str(path) for path in cofactor_frcmods],
            "receptor_cofactor_libs": [str(path) for path in cofactor_libs],
            "pb_radii": profile["amber_prep"].get("pb_radii", "mbondi2"),
            "solvent_shape": profile["system"].get("solvent_shape", "oct"),
            "water_count_presalt": water_count,
            "salt_pairs": salt_pairs,
        },
    )


def tleap_text(
    receptor_pdb: Path,
    ligand_mol2: Path,
    ligand_frcmod: Path | None,
    ligand_libs: list[Path],
    salt_pairs: int | None,
    profile: dict[str, Any],
    cofactor_files: list[Path] | None = None,
    cofactor_frcmods: list[Path] | None = None,
    cofactor_libs: list[Path] | None = None,
) -> str:
    padding = float(profile["system"]["solvent_padding_angstrom"])
    solvent_shape = str(profile["system"].get("solvent_shape", "oct")).lower()
    if solvent_shape == "oct":
        solvate_command = f"solvateOct mol TIP3PBOX {padding:.3f}"
    elif solvent_shape == "box":
        solvate_command = f"solvateBox mol TIP3PBOX {padding:.3f}"
    else:
        raise SystemExit(f"Unsupported solvent_shape: {solvent_shape}")
    lines = [
        f"source {profile['amber_prep']['protein_ff']}",
        f"source {profile['amber_prep']['ligand_ff']}",
        f"source {profile['amber_prep']['water_ff']}",
        f"set default PBRadii {profile['amber_prep'].get('pb_radii', 'mbondi2')}",
    ]
    for lib in [*(cofactor_libs or []), *ligand_libs]:
        lines.append(tleap_parameter_load_command(lib))
    for frcmod in cofactor_frcmods or []:
        lines.append(f'loadamberparams "{frcmod}"')
    if ligand_frcmod is not None:
        lines.append(f'loadamberparams "{ligand_frcmod}"')
    cofactor_units: list[str] = []
    for idx, cofactor_file in enumerate(cofactor_files or [], start=1):
        unit = f"cof{idx}"
        suffix = cofactor_file.suffix.lower()
        if suffix == ".mol2":
            lines.append(f'{unit} = loadmol2 "{cofactor_file}"')
        elif suffix == ".pdb":
            lines.append(f'{unit} = loadpdb "{cofactor_file}"')
        else:
            raise SystemExit(f"Unsupported receptor cofactor file format {suffix!r}; use mol2 or pdb: {cofactor_file}")
        cofactor_units.append(unit)
    combine_units = " ".join(["rec", *cofactor_units, "lig"])
    lines.extend(
        [
            f'rec = loadpdb "{receptor_pdb}"',
            f'lig = loadmol2 "{ligand_mol2}"',
            f"mol = combine {{ {combine_units} }}",
            "check mol",
            "savepdb mol complex_dry.pdb",
            "saveamberparm mol complex_dry.prmtop complex_dry.inpcrd",
            solvate_command,
            "addIonsRand mol Na+ 0",
            "addIonsRand mol Cl- 0",
        ]
    )
    if salt_pairs is not None and salt_pairs > 0:
        lines.extend([f"addIonsRand mol Na+ {salt_pairs}", f"addIonsRand mol Cl- {salt_pairs}"])
    lines.extend(["check mol", "savepdb mol system_solvated.pdb", "saveamberparm mol system_solvated.prmtop system_solvated.inpcrd", "quit"])
    return "\n".join(lines) + "\n"


def read_manifest(paths: Any) -> dict[str, Any]:
    manifest_path = paths.root / "manifest.json"
    if not manifest_path.exists():
        return {}
    import json

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def tleap_parameter_load_command(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".lib", ".off"}:
        return f'loadoff "{path}"'
    if suffix in {".prep", ".prepi"}:
        return f'loadamberprep "{path}"'
    raise SystemExit(f"Unsupported Amber parameter library format {suffix!r}: {path}")


def apply_radii(prmtop: Path, log: Path, profile: dict[str, Any]) -> None:
    radii = str(profile["amber_prep"].get("pb_radii", "mbondi2"))
    if radii.lower() == "none":
        return
    script = f"changeRadii {radii}\noutparm {prmtop}\nquit\n"
    run_logged(mamba_command(profile, ["parmed", "-O", "-p", str(prmtop)]), log, stdin=script)


def ligand_residue_name_from_mol2(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "@<TRIPOS>ATOM" not in text:
        return None
    in_atoms = False
    for raw in text.splitlines():
        if raw.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if raw.startswith("@<TRIPOS>") and in_atoms:
            return None
        if in_atoms and raw.strip():
            parts = re.split(r"\s+", raw.strip())
            if len(parts) >= 8:
                return parts[7]
    return None

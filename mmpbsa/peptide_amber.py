from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .common import (
    count_selected_residues,
    count_waters,
    mamba_command,
    run_logged,
    split_chain_spec,
    write_json_atomic,
    write_protein_only_pdb,
    write_text_atomic,
)


STANDARD_AMINO_ACIDS = {
    "ACE",
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
    "NHE",
    "NH2",
    "NME",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def prepare_input_structure(paths: Any, manifest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    policy = str(profile.get("amber_prep", {}).get("nonstandard_policy", "fail")).lower()
    if policy not in {"fail", "strip"}:
        raise SystemExit(f"Unsupported amber_prep.nonstandard_policy for peptide workflow: {policy!r}; use 'fail' or 'strip'.")
    source_pdb = paths.input / "selected_raw.pdb"
    if not source_pdb.exists():
        source_pdb = paths.input / "selected.pdb"
    findings = inspect_selected_residues(source_pdb, manifest["receptor_chains"], manifest["peptide_chains"])
    cofactor_count = int(manifest.get("receptor_cofactor_residue_count") or 0)
    blocking = []
    if findings["nonstandard_atom_residues"]:
        blocking.append("nonstandard ATOM residues")
    if findings["hetero_residues"] and policy != "strip":
        blocking.append("HETATM residues")
    if blocking:
        raise SystemExit(
            "Amber preparation requires explicit handling for "
            + ", ".join(blocking)
            + ". Set amber_prep.nonstandard_policy: strip only for removable HETATM records; parameterized residues need a custom recipe. "
            + f"Findings: {findings}"
        )

    dropped = write_protein_only_pdb(
        source_pdb,
        paths.input / "selected.pdb",
        manifest["receptor_chains"],
        manifest["peptide_chains"],
        accepted_hetero_resnames=STANDARD_AMINO_ACIDS,
    )
    shutil.copy2(paths.input / "selected.pdb", paths.input / "selected_protein.pdb")
    write_protein_only_pdb(
        source_pdb,
        paths.input / "selected_receptor.pdb",
        manifest["receptor_chains"],
        "",
        accepted_hetero_resnames=STANDARD_AMINO_ACIDS,
    )
    write_protein_only_pdb(
        source_pdb,
        paths.input / "selected_peptide.pdb",
        "",
        manifest["peptide_chains"],
        accepted_hetero_resnames=STANDARD_AMINO_ACIDS,
    )
    receptor_residues = count_selected_residues(paths.input / "selected_receptor.pdb", manifest["receptor_chains"])
    peptide_residues = count_selected_residues(paths.input / "selected_peptide.pdb", manifest["peptide_chains"])
    if receptor_residues <= 0 or peptide_residues <= 0:
        raise SystemExit("Could not derive receptor/peptide residue counts from selected structure")
    receptor_mask_count = receptor_residues + cofactor_count
    peptide_first = receptor_mask_count + 1
    peptide_last = receptor_mask_count + peptide_residues
    manifest.update(
        {
            "input_preparation": "standard_peptide_atom_records_with_receptor_cofactors" if cofactor_count else "standard_peptide_atom_records",
            "raw_input_pdb": str(source_pdb),
            "clean_input_pdb": str(paths.input / "selected.pdb"),
            "amber_prep_recipe": profile.get("amber_prep", {}).get("recipe", "standard_peptide_ff14sb_tip3p"),
            "dropped_nonprotein_residues": dropped,
            "dropped_nonprotein_residue_count": len(dropped),
            "input_residue_findings": findings,
            "protein_receptor_residue_count": receptor_residues,
            "receptor_cofactor_residue_count": cofactor_count,
            "receptor_residue_count": receptor_mask_count,
            "peptide_residue_count": peptide_residues,
            "receptor_residue_mask": f":1-{receptor_mask_count}",
            "peptide_residue_mask": f":{peptide_first}-{peptide_last}",
            "note": "Residue masks assume tleap combines receptor chain(s), receptor cofactors, then peptide chain(s).",
        }
    )
    return manifest


def inspect_selected_residues(pdb: Path, receptor_chains: str, ligand_chains: str) -> dict[str, Any]:
    selected = set(split_chain_spec(receptor_chains) + split_chain_spec(ligand_chains))
    nonstandard_atom: dict[tuple[str, str, str, str], str] = {}
    hetero: dict[tuple[str, str, str, str], str] = {}
    accepted_hetero: dict[tuple[str, str, str, str], str] = {}
    for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip()
        if chain not in selected:
            continue
        resname = line[17:20].strip().upper()
        key = (chain, resname, line[22:26].strip(), line[26].strip())
        if line.startswith("HETATM"):
            if resname in STANDARD_AMINO_ACIDS:
                accepted_hetero[key] = line[76:78].strip()
            else:
                hetero[key] = line[76:78].strip()
        elif resname not in STANDARD_AMINO_ACIDS:
            nonstandard_atom[key] = line[76:78].strip()
    return {
        "nonstandard_atom_residues": residue_records(nonstandard_atom),
        "hetero_residues": residue_records(hetero),
        "accepted_hetero_residues": residue_records(accepted_hetero),
    }


def residue_records(items: dict[tuple[str, str, str, str], str]) -> list[dict[str, str]]:
    return [
        {"chain": chain, "resname": resname, "resseq": resseq, "icode": icode, "element": element}
        for (chain, resname, resseq, icode), element in sorted(items.items(), key=lambda item: (item[0][0], item[0][2], item[0][1], item[0][3]))
    ]


def run_amber_prepare(paths: Any, profile: dict[str, Any]) -> None:
    paths.amber.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(paths)
    cofactor_files = [Path(item) for item in manifest.get("receptor_cofactor_files", [])]
    cofactor_frcmods = [Path(item) for item in manifest.get("receptor_cofactor_frcmods", [])]
    cofactor_libs = [Path(item) for item in manifest.get("receptor_cofactor_libs", [])]
    if cofactor_files:
        run_logged(
            mamba_command(
                profile,
                [
                    "pdb4amber",
                    "-i",
                    str(paths.input / "selected_receptor.pdb"),
                    "-o",
                    str(paths.amber / "receptor_pdb4amber.pdb"),
                    "--dry",
                    "--nohyd",
                    "-l",
                    str(paths.logs / "pdb4amber_receptor.detail.log"),
                ],
            ),
            paths.logs / "pdb4amber_receptor.log",
        )
        run_logged(
            mamba_command(
                profile,
                [
                    "pdb4amber",
                    "-i",
                    str(paths.input / "selected_peptide.pdb"),
                    "-o",
                    str(paths.amber / "peptide_pdb4amber.pdb"),
                    "--dry",
                    "--nohyd",
                    "-l",
                    str(paths.logs / "pdb4amber_peptide.detail.log"),
                ],
            ),
            paths.logs / "pdb4amber_peptide.log",
        )
        write_text_atomic(
            paths.amber / "build_presalt.leap",
            tleap_text_with_cofactors(
                paths.amber / "receptor_pdb4amber.pdb",
                paths.amber / "peptide_pdb4amber.pdb",
                None,
                profile,
                cofactor_files=cofactor_files,
                cofactor_frcmods=cofactor_frcmods,
                cofactor_libs=cofactor_libs,
            ),
        )
        run_logged(mamba_command(profile, ["tleap", "-f", "build_presalt.leap"]), paths.logs / "tleap_presalt.log", cwd=paths.amber)
        water_count = count_waters(paths.amber / "system_solvated.pdb")
        salt_pairs = max(1, round(float(profile["system"]["salt_molar"]) * water_count / 55.5))
        write_text_atomic(
            paths.amber / "build_final.leap",
            tleap_text_with_cofactors(
                paths.amber / "receptor_pdb4amber.pdb",
                paths.amber / "peptide_pdb4amber.pdb",
                salt_pairs,
                profile,
                cofactor_files=cofactor_files,
                cofactor_frcmods=cofactor_frcmods,
                cofactor_libs=cofactor_libs,
            ),
        )
        run_logged(mamba_command(profile, ["tleap", "-f", "build_final.leap"]), paths.logs / "tleap_final.log", cwd=paths.amber)
        write_json_atomic(
            paths.amber / "summary.json",
            {
                "recipe": profile.get("amber_prep", {}).get("recipe", "standard_peptide_ff14sb_tip3p"),
                "nonstandard_policy": profile.get("amber_prep", {}).get("nonstandard_policy", "fail"),
                "leaprc": ["leaprc.protein.ff14SB", "leaprc.water.tip3p"],
                "receptor_cofactor_files": [str(path) for path in cofactor_files],
                "receptor_cofactor_frcmods": [str(path) for path in cofactor_frcmods],
                "receptor_cofactor_libs": [str(path) for path in cofactor_libs],
                "pb_radii": "mbondi2",
                "solvent_shape": profile["system"].get("solvent_shape", "oct"),
                "solvent_shape_actual": profile["system"].get("solvent_shape", "oct"),
                "box_retry_used": bool(manifest.get("box_retry_used", False)),
                "box_retry_reason": manifest.get("box_retry_reason", ""),
                "water_count_presalt": water_count,
                "salt_pairs": salt_pairs,
            },
        )
        return

    run_logged(
        mamba_command(
            profile,
            [
                "pdb4amber",
                "-i",
                str(paths.input / "selected_protein.pdb"),
                "-o",
                str(paths.amber / "complex_pdb4amber.pdb"),
                "--dry",
                "--nohyd",
                "-l",
                str(paths.logs / "pdb4amber.detail.log"),
            ],
        ),
        paths.logs / "pdb4amber.log",
    )
    write_text_atomic(paths.amber / "build_presalt.leap", tleap_text(paths.amber / "complex_pdb4amber.pdb", None, profile))
    run_logged(mamba_command(profile, ["tleap", "-f", "build_presalt.leap"]), paths.logs / "tleap_presalt.log", cwd=paths.amber)
    water_count = count_waters(paths.amber / "system_solvated.pdb")
    salt_pairs = max(1, round(float(profile["system"]["salt_molar"]) * water_count / 55.5))
    write_text_atomic(paths.amber / "build_final.leap", tleap_text(paths.amber / "complex_pdb4amber.pdb", salt_pairs, profile))
    run_logged(mamba_command(profile, ["tleap", "-f", "build_final.leap"]), paths.logs / "tleap_final.log", cwd=paths.amber)
    write_json_atomic(
        paths.amber / "summary.json",
        {
            "recipe": profile.get("amber_prep", {}).get("recipe", "standard_peptide_ff14sb_tip3p"),
            "nonstandard_policy": profile.get("amber_prep", {}).get("nonstandard_policy", "fail"),
            "leaprc": ["leaprc.protein.ff14SB", "leaprc.water.tip3p"],
            "pb_radii": "mbondi2",
            "solvent_shape": profile["system"].get("solvent_shape", "oct"),
            "solvent_shape_actual": profile["system"].get("solvent_shape", "oct"),
            "box_retry_used": bool(manifest.get("box_retry_used", False)),
            "box_retry_reason": manifest.get("box_retry_reason", ""),
            "water_count_presalt": water_count,
            "salt_pairs": salt_pairs,
        },
    )


def tleap_text(clean_pdb: Path, salt_pairs: int | None, profile: dict[str, Any]) -> str:
    padding = float(profile["system"]["solvent_padding_angstrom"])
    solvent_shape = str(profile["system"].get("solvent_shape", "oct")).lower()
    if solvent_shape == "oct":
        solvate_command = f"solvateOct mol TIP3PBOX {padding:.3f}"
    elif solvent_shape == "box":
        solvate_command = f"solvateBox mol TIP3PBOX {padding:.3f}"
    else:
        raise SystemExit(f"Unsupported solvent_shape: {solvent_shape}")
    lines = [
        "source leaprc.protein.ff14SB",
        "source leaprc.water.tip3p",
        "set default PBRadii mbondi2",
        f"mol = loadpdb {clean_pdb}",
        "check mol",
        "savepdb mol complex_dry.pdb",
        "saveamberparm mol complex_dry.prmtop complex_dry.inpcrd",
        solvate_command,
        "addIonsRand mol Na+ 0",
        "addIonsRand mol Cl- 0",
    ]
    if salt_pairs is not None and salt_pairs > 0:
        lines.extend([f"addIonsRand mol Na+ {salt_pairs}", f"addIonsRand mol Cl- {salt_pairs}"])
    lines.extend(["check mol", "savepdb mol system_solvated.pdb", "saveamberparm mol system_solvated.prmtop system_solvated.inpcrd", "quit"])
    return "\n".join(lines) + "\n"


def tleap_text_with_cofactors(
    receptor_pdb: Path,
    peptide_pdb: Path,
    salt_pairs: int | None,
    profile: dict[str, Any],
    *,
    cofactor_files: list[Path],
    cofactor_frcmods: list[Path],
    cofactor_libs: list[Path],
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
        "source leaprc.protein.ff14SB",
        "source leaprc.water.tip3p",
        "set default PBRadii mbondi2",
    ]
    for lib in cofactor_libs:
        lines.append(tleap_parameter_load_command(lib))
    for frcmod in cofactor_frcmods:
        lines.append(f'loadamberparams "{frcmod}"')
    lines.append(f'rec = loadpdb "{receptor_pdb}"')
    cofactor_units: list[str] = []
    for idx, cofactor_file in enumerate(cofactor_files, start=1):
        unit = f"cof{idx}"
        suffix = cofactor_file.suffix.lower()
        if suffix == ".mol2":
            lines.append(f'{unit} = loadmol2 "{cofactor_file}"')
        elif suffix == ".pdb":
            lines.append(f'{unit} = loadpdb "{cofactor_file}"')
        else:
            raise SystemExit(f"Unsupported receptor cofactor file format {suffix!r}; use mol2 or pdb: {cofactor_file}")
        cofactor_units.append(unit)
    lines.extend(
        [
            f'pep = loadpdb "{peptide_pdb}"',
            f"mol = combine {{ {' '.join(['rec', *cofactor_units, 'pep'])} }}",
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

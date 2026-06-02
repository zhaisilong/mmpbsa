from __future__ import annotations

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
    policy = str(profile.get("amber_prep", {}).get("nonstandard_policy", "fail")).lower()
    findings = inspect_selected_residues(paths.input / "selected.pdb", manifest["receptor_chains"], manifest["peptide_chains"])
    blocking = []
    if findings["nonstandard_atom_residues"]:
        blocking.append("nonstandard ATOM residues")
    if findings["hetero_residues"] and policy != "strip":
        blocking.append("HETATM residues")
    if blocking:
        raise SystemExit(
            "Amber preparation requires explicit handling for "
            + ", ".join(blocking)
            + ". Set amber_prep.nonstandard_policy: strip only for removable HETATM records; parameterized residues need a custom recipe."
        )

    dropped = write_protein_only_pdb(paths.input / "selected.pdb", paths.input / "selected_protein.pdb", manifest["receptor_chains"], manifest["peptide_chains"])
    receptor_residues = count_selected_residues(paths.input / "selected_protein.pdb", manifest["receptor_chains"])
    peptide_residues = count_selected_residues(paths.input / "selected_protein.pdb", manifest["peptide_chains"])
    if receptor_residues <= 0 or peptide_residues <= 0:
        raise SystemExit("Could not derive receptor/peptide residue counts from selected structure")
    peptide_first = receptor_residues + 1
    peptide_last = receptor_residues + peptide_residues
    manifest.update(
        {
            "input_preparation": "standard_peptide_atom_records",
            "amber_prep_recipe": profile.get("amber_prep", {}).get("recipe", "standard_peptide_ff14sb_tip3p"),
            "dropped_nonprotein_residues": dropped,
            "input_residue_findings": findings,
            "receptor_residue_count": receptor_residues,
            "peptide_residue_count": peptide_residues,
            "receptor_residue_mask": f":1-{receptor_residues}",
            "peptide_residue_mask": f":{peptide_first}-{peptide_last}",
            "note": "Residue masks assume selected PDB order is receptor chain(s) followed by peptide chain(s).",
        }
    )
    return manifest


def inspect_selected_residues(pdb: Path, receptor_chains: str, ligand_chains: str) -> dict[str, Any]:
    selected = set(split_chain_spec(receptor_chains) + split_chain_spec(ligand_chains))
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

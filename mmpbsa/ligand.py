from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .common import mamba_command, resolve_project_path, run_logged, split_path_list, write_json_atomic


FORMAT_BY_SUFFIX = {
    ".mol2": "mol2",
    ".sdf": "sdf",
    ".sd": "sdf",
    ".pdb": "pdb",
    ".mol": "mdl",
}


def ligand_input_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in FORMAT_BY_SUFFIX:
        raise SystemExit(f"Unsupported ligand input format {suffix!r}; use mol2, sdf, mol, or pdb")
    return FORMAT_BY_SUFFIX[suffix]


def mol2_charges(path: Path) -> list[float]:
    charges: list[float] = []
    in_atoms = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("@<TRIPOS>"):
            in_atoms = line.upper() == "@<TRIPOS>ATOM"
            continue
        if not in_atoms:
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        try:
            charges.append(float(fields[-1]))
        except ValueError:
            continue
    return charges


def mol2_total_charge(path: Path) -> float | None:
    charges = mol2_charges(path)
    if not charges:
        return None
    return sum(charges)


def antechamber_command(input_file: Path, output_mol2: Path, input_format: str, resname: str, charge: int, profile: dict[str, Any]) -> list[str]:
    amber_prep = profile["amber_prep"]
    return [
        "antechamber",
        "-i",
        str(input_file),
        "-fi",
        input_format,
        "-o",
        str(output_mol2),
        "-fo",
        "mol2",
        "-at",
        str(amber_prep.get("gaff_version", "gaff2")),
        "-c",
        str(amber_prep.get("charge_method", "bcc")),
        "-nc",
        str(charge),
        "-rn",
        resname,
    ]


def antechamber_read_mol2_charge_command(input_mol2: Path, charge_file: Path, output_mol2: Path, resname: str, charge: int, profile: dict[str, Any]) -> list[str]:
    amber_prep = profile["amber_prep"]
    return [
        "antechamber",
        "-i",
        str(input_mol2),
        "-fi",
        "mol2",
        "-o",
        str(output_mol2),
        "-fo",
        "mol2",
        "-at",
        str(amber_prep.get("gaff_version", "gaff2")),
        "-c",
        "rc",
        "-cf",
        str(charge_file),
        "-nc",
        str(charge),
        "-rn",
        resname,
    ]


def parmchk2_command(input_mol2: Path, output_frcmod: Path, profile: dict[str, Any]) -> list[str]:
    return [
        "parmchk2",
        "-i",
        str(input_mol2),
        "-f",
        "mol2",
        "-o",
        str(output_frcmod),
        "-s",
        str(profile["amber_prep"].get("gaff_version", "gaff2")),
    ]


def obabel_to_mol2_command(input_file: Path, output_mol2: Path, input_format: str) -> list[str]:
    return [
        "obabel",
        f"-i{input_format}",
        str(input_file),
        "-omol2",
        "-O",
        str(output_mol2),
    ]


def run_ligand_prepare(paths: Any, manifest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    paths.ligand.mkdir(parents=True, exist_ok=True)
    mode = str(manifest.get("ligand_param_mode") or profile["amber_prep"].get("default_ligand_param_mode", "auto")).lower()
    if mode not in {"auto", "mol2", "direct_mol2", "preparam"}:
        raise SystemExit(f"Unsupported ligand_param_mode: {mode}")
    charge_method = str(profile["amber_prep"].get("charge_method", "bcc")).lower()
    if mode == "auto" and charge_method == "resp" and not bool(profile["amber_prep"].get("allow_charge_fallback", False)):
        raise SystemExit(
            "ligand_param_mode=auto cannot provide validated RESP charges. "
            "Provide a RESP-charged mol2/preparam ligand, or use a test profile with amber_prep.charge_method=bcc."
        )
    if mode == "auto":
        summary = run_auto_ligand_prepare(paths, manifest, profile)
    elif mode in {"mol2", "direct_mol2"}:
        summary = run_direct_mol2_ligand_prepare(paths, manifest, profile, mode)
    else:
        summary = run_preparam_ligand_prepare(paths, manifest, profile)
    write_json_atomic(paths.ligand / "summary.json", summary)
    manifest["ligand_prepare"] = summary
    return manifest


def run_auto_ligand_prepare(paths: Any, manifest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(manifest["input_ligand_file"]))
    input_format = ligand_input_format(source)
    output_mol2 = paths.ligand / "ligand.mol2"
    output_frcmod = paths.ligand / "ligand.frcmod"
    charge = int(manifest["ligand_charge"])
    antechamber_input = source
    antechamber_format = input_format
    fallback_mol2 = paths.ligand / "ligand_obabel_input.mol2"
    fallback_reason = ""
    try:
        run_logged(
            mamba_command(profile, antechamber_command(antechamber_input, output_mol2, antechamber_format, manifest["ligand_resname"], charge, profile)),
            paths.logs / "antechamber_ligand.log",
            cwd=paths.ligand,
        )
    except RuntimeError as exc:
        log_text = (paths.logs / "antechamber_ligand.log").read_text(encoding="utf-8", errors="replace")
        can_fallback = input_format in {"sdf", "mdl"} and (
            "Invalid number of atoms" in log_text
            or "Check Format for sdf File" in log_text
            or "MDL SDF supports" in log_text
        )
        if not can_fallback:
            raise
        fallback_reason = str(exc)
        run_logged(
            mamba_command(profile, obabel_to_mol2_command(source, fallback_mol2, input_format)),
            paths.logs / "obabel_ligand_to_mol2.log",
            cwd=paths.ligand,
        )
        antechamber_input = fallback_mol2
        antechamber_format = "mol2"
        run_logged(
            mamba_command(profile, antechamber_command(antechamber_input, output_mol2, antechamber_format, manifest["ligand_resname"], charge, profile)),
            paths.logs / "antechamber_ligand_from_obabel_mol2.log",
            cwd=paths.ligand,
        )
    run_logged(
        mamba_command(profile, parmchk2_command(output_mol2, output_frcmod, profile)),
        paths.logs / "parmchk2_ligand.log",
        cwd=paths.ligand,
    )
    return {
        "mode": "auto",
        "status": "prepared",
        "input_ligand_file": str(source),
        "input_format": input_format,
        "antechamber_input_file": str(antechamber_input),
        "antechamber_input_format": antechamber_format,
        "fallback_mol2": str(fallback_mol2) if fallback_mol2.exists() else "",
        "fallback_reason": fallback_reason,
        "ligand_resname": manifest["ligand_resname"],
        "ligand_charge": charge,
        "gaff_version": profile["amber_prep"].get("gaff_version", "gaff2"),
        "charge_method": profile["amber_prep"].get("charge_method", "bcc"),
        "mol2": str(output_mol2),
        "frcmod": str(output_frcmod),
        "libs": [],
    }


def run_direct_mol2_ligand_prepare(paths: Any, manifest: dict[str, Any], profile: dict[str, Any], mode: str) -> dict[str, Any]:
    src_mol2_text = str(manifest.get("ligand_mol2") or "").strip()
    if src_mol2_text:
        src_mol2 = resolve_project_path(src_mol2_text)
    else:
        src_mol2 = Path(str(manifest["input_ligand_file"]))
    if src_mol2.suffix.lower() != ".mol2":
        raise SystemExit(f"{mode} mode requires a mol2 ligand_file or ligand_mol2, got: {src_mol2}")
    if not src_mol2.exists():
        raise SystemExit(f"Missing ligand mol2: {src_mol2}")

    source_charges = mol2_charges(src_mol2)
    if not source_charges:
        raise SystemExit(f"{mode} mode requires partial charges in the mol2 ATOM records: {src_mol2}")

    charge_file = paths.ligand / "ligand_mol2_charges.dat"
    charge_file.write_text("".join(f"{charge:.8f}\n" for charge in source_charges), encoding="utf-8")
    output_mol2 = paths.ligand / "ligand.mol2"
    declared_charge = int(manifest["ligand_charge"])
    run_logged(
        mamba_command(profile, antechamber_read_mol2_charge_command(src_mol2, charge_file, output_mol2, manifest["ligand_resname"], declared_charge, profile)),
        paths.logs / "antechamber_ligand_direct_mol2.log",
        cwd=paths.ligand,
    )

    frcmod_text = str(manifest.get("ligand_frcmod") or "").strip()
    frcmod_generated = False
    if frcmod_text:
        src_frcmod = resolve_project_path(frcmod_text)
        if not src_frcmod.exists():
            raise SystemExit(f"Missing ligand_frcmod: {src_frcmod}")
        output_frcmod = paths.ligand / "ligand.frcmod"
        shutil.copy2(src_frcmod, output_frcmod)
    else:
        output_frcmod = paths.ligand / "ligand.frcmod"
        run_logged(
            mamba_command(profile, parmchk2_command(output_mol2, output_frcmod, profile)),
            paths.logs / "parmchk2_ligand_direct_mol2.log",
            cwd=paths.ligand,
        )
        frcmod_generated = True

    copied_libs: list[str] = []
    for item in split_path_list(str(manifest.get("ligand_lib") or "")):
        src = resolve_project_path(item)
        if not src.exists():
            raise SystemExit(f"Missing ligand library/off file: {src}")
        dst = paths.ligand / src.name
        shutil.copy2(src, dst)
        copied_libs.append(str(dst))

    total_charge = mol2_total_charge(output_mol2)
    charge_delta = None if total_charge is None else total_charge - declared_charge
    charge_warning = ""
    if charge_delta is not None and abs(charge_delta) > 0.2:
        charge_warning = (
            f"mol2 partial charges sum to {total_charge:.4f}, "
            f"but ligand_charge is {declared_charge}; verify protonation and charge assignment."
        )

    return {
        "mode": mode,
        "status": "prepared",
        "input_ligand_file": str(manifest["input_ligand_file"]),
        "input_format": "mol2",
        "ligand_resname": manifest["ligand_resname"],
        "ligand_charge": declared_charge,
        "mol2_total_charge": total_charge,
        "mol2_charge_delta": charge_delta,
        "charge_warning": charge_warning,
        "gaff_version": profile["amber_prep"].get("gaff_version", "gaff2"),
        "charge_method": "read_from_mol2",
        "antechamber_charge_file": str(charge_file),
        "mol2": str(output_mol2),
        "frcmod": str(output_frcmod),
        "frcmod_generated": frcmod_generated,
        "libs": copied_libs,
    }


def run_preparam_ligand_prepare(paths: Any, manifest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    src_mol2_text = str(manifest.get("ligand_mol2") or "").strip()
    if not src_mol2_text:
        source = Path(str(manifest["input_ligand_file"]))
        if source.suffix.lower() != ".mol2":
            raise SystemExit("preparam mode requires ligand_mol2 or a mol2 ligand_file")
        src_mol2 = source
    else:
        src_mol2 = resolve_project_path(src_mol2_text)
    if not src_mol2.exists():
        raise SystemExit(f"Missing ligand_mol2: {src_mol2}")
    output_mol2 = paths.ligand / "ligand.mol2"
    shutil.copy2(src_mol2, output_mol2)

    frcmod_out = ""
    frcmod_text = str(manifest.get("ligand_frcmod") or "").strip()
    if frcmod_text:
        src_frcmod = resolve_project_path(frcmod_text)
        if not src_frcmod.exists():
            raise SystemExit(f"Missing ligand_frcmod: {src_frcmod}")
        dst_frcmod = paths.ligand / "ligand.frcmod"
        shutil.copy2(src_frcmod, dst_frcmod)
        frcmod_out = str(dst_frcmod)

    copied_libs: list[str] = []
    for item in split_path_list(str(manifest.get("ligand_lib") or "")):
        src = resolve_project_path(item)
        if not src.exists():
            raise SystemExit(f"Missing ligand library/off file: {src}")
        dst = paths.ligand / src.name
        shutil.copy2(src, dst)
        copied_libs.append(str(dst))

    return {
        "mode": "preparam",
        "status": "prepared",
        "input_ligand_file": str(manifest["input_ligand_file"]),
        "ligand_resname": manifest["ligand_resname"],
        "ligand_charge": int(manifest["ligand_charge"]),
        "mol2": str(output_mol2),
        "frcmod": frcmod_out,
        "libs": copied_libs,
    }

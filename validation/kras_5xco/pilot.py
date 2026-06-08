from __future__ import annotations

import json
import math
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmpbsa.common import write_json_atomic, write_text_atomic


@dataclass(frozen=True)
class CifAtom:
    group: str
    atom: str
    resname: str
    chain: str
    resseq: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str


@dataclass(frozen=True)
class VariantSpec:
    job_id: str
    name: str
    ligand_id: str
    keep_positions: tuple[int, ...]
    mutations: dict[int, str]


PILOT_VARIANTS: dict[str, VariantSpec] = {
    "WT_PEP_0001": VariantSpec("WT_PEP_0001", "WT", "PEP_0001", tuple(range(1, 22)), {}),
    "del4R_PEP_0002": VariantSpec(
        "del4R_PEP_0002",
        "del4R",
        "PEP_0002",
        (1, 4, 5, *range(6, 19), 21),
        {},
    ),
    "core13_PEP_0003": VariantSpec("core13_PEP_0003", "core13", "PEP_0003", (1, *range(6, 17), 21), {}),
    "L8A_PEP_0006": VariantSpec("L8A_PEP_0006", "L8A", "PEP_0006", tuple(range(1, 22)), {8: "ALA"}),
    "D13A_PEP_0011": VariantSpec("D13A_PEP_0011", "D13A", "PEP_0011", tuple(range(1, 22)), {13: "ALA"}),
    "P14A_PEP_0012": VariantSpec("P14A_PEP_0012", "P14A", "PEP_0012", tuple(range(1, 22)), {14: "ALA"}),
}

STATE_LABELS = {"gdp_only": "GDP-only", "gdp_mg": "GDP+Mg"}


def default_variant_ids() -> list[str]:
    return ["WT_PEP_0001", "D13A_PEP_0011", "L8A_PEP_0006", "P14A_PEP_0012", "del4R_PEP_0002", "core13_PEP_0003"]


def parse_variant_ids(text: str | None) -> list[str]:
    if not text:
        return default_variant_ids()
    values = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    missing = [item for item in values if item not in PILOT_VARIANTS]
    if missing:
        raise SystemExit(f"Unknown KRAS pilot variant(s): {', '.join(missing)}")
    return values


def parse_states(text: str | None) -> list[str]:
    if not text:
        return ["gdp_only", "gdp_mg"]
    values = [item.strip().replace("-", "_") for item in text.replace(";", ",").split(",") if item.strip()]
    missing = [item for item in values if item not in STATE_LABELS]
    if missing:
        raise SystemExit(f"Unknown KRAS pilot receptor state(s): {', '.join(missing)}")
    return values


def build_kras_5xco_pilot(
    output_dir: Path,
    *,
    template_cif: Path,
    gdp_lib: Path,
    gdp_frcmod: Path,
    mg_source_cif: Path | None = None,
    download_mg_source: bool = False,
    variants: list[str] | None = None,
    states: list[str] | None = None,
    force: bool = False,
    production_ns: float = 20.0,
    mmpbsa_start_ns: float = 10.0,
    seed_base: int = 2026060401,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        if not force:
            raise SystemExit(f"Output directory is not empty: {output}; use --force to rebuild it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    template_cif = template_cif.resolve()
    gdp_lib = gdp_lib.resolve()
    gdp_frcmod = gdp_frcmod.resolve()
    require_file(template_cif, "5XCO template CIF")
    require_file(gdp_lib, "GDP Amber prep")
    require_file(gdp_frcmod, "GDP frcmod")
    if mg_source_cif is None:
        mg_source_cif = output / "sources" / "5US4.cif"
    mg_source_cif = mg_source_cif.resolve()
    if download_mg_source and not mg_source_cif.exists():
        download_rcsb_cif("5US4", mg_source_cif)
    require_file(mg_source_cif, "Mg source CIF")

    variant_ids = variants or default_variant_ids()
    state_ids = states or ["gdp_only", "gdp_mg"]
    template_atoms = load_cif_atoms(template_cif)
    mg_source_atoms = load_cif_atoms(mg_source_cif)

    protein_atoms = template_protein_atoms(template_atoms)
    peptide_template = template_peptide_atoms(template_atoms)
    gdp_atoms = template_gdp_atoms(template_atoms)
    protein_residue_count = count_template_residues(protein_atoms)
    if protein_residue_count <= 0:
        raise SystemExit("5XCO template has no protein residues for chain A")
    if len(gdp_atoms) != 28:
        raise SystemExit(f"Expected 28 5XCO GDP heavy atoms, found {len(gdp_atoms)}")

    mg_report = transfer_mg_from_source(template_atoms, mg_source_atoms)
    protocol_path = output / "kras_5xco_pilot_3x20ns.yaml"
    smoke_protocol_path = output / "kras_5xco_smoke_20ps.yaml"
    write_text_atomic(protocol_path, pilot_protocol_text(production_ns, mmpbsa_start_ns, seed_base))
    write_text_atomic(smoke_protocol_path, smoke_protocol_text(seed_base))

    jobs: list[dict[str, Any]] = []
    for variant_id in variant_ids:
        spec = PILOT_VARIANTS[variant_id]
        for state in state_ids:
            cofactor_count = 2 if state == "gdp_mg" else 1
            job_id = f"{spec.job_id}_{state}"
            job_dir = output / job_id
            input_dir = job_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            selected_pdb = input_dir / "selected.pdb"
            write_selected_pdb(
                selected_pdb,
                protein_atoms=protein_atoms,
                peptide_atoms=variant_peptide_atoms(peptide_template, spec),
                protein_residue_count=protein_residue_count,
                receptor_cofactor_count=cofactor_count,
            )
            write_cofactor_pdb(input_dir / "gdp.pdb", gdp_atoms, resname="gdp", chain="G", atom_serial_start=1)
            cofactor_files = ["input/gdp.pdb"]
            if state == "gdp_mg":
                write_mg_pdb(input_dir / "mg.pdb", mg_report["mg_xyz"])
                cofactor_files.append("input/mg.pdb")

            config = {
                "job_id": job_id,
                "name": f"{spec.name}_{state}",
                "selected_pdb": "input/selected.pdb",
                "receptor_chains": "A",
                "peptide_chains": "B",
                "receptor_cofactor_files": ",".join(cofactor_files),
                "receptor_cofactor_libs": str(gdp_lib),
                "receptor_cofactor_frcmods": str(gdp_frcmod),
                "receptor_cofactor_residue_count": cofactor_count,
                "source": f"5XCO-derived KRAS peptide pilot, receptor_state={STATE_LABELS[state]}",
                "template_pdb_id": "5XCO",
                "template_source_cif": str(template_cif),
                "mg_source_pdb_id": "5US4",
                "mg_source_cif": str(mg_source_cif),
                "base_job_id": spec.job_id,
                "ligand_id": spec.ligand_id,
                "receptor_state": state,
                "variant_keep_positions": list(spec.keep_positions),
                "variant_mutations": {str(k): v for k, v in spec.mutations.items()},
            }
            write_json_atomic(job_dir / f"{job_id}.json", config)
            jobs.append(
                {
                    "job_id": job_id,
                    "base_job_id": spec.job_id,
                    "name": config["name"],
                    "receptor_state": state,
                    "selected_pdb": str(selected_pdb),
                    "receptor_cofactor_residue_count": cofactor_count,
                    "peptide_residue_count": len(spec.keep_positions),
                }
            )

    report = {
        "schema_version": "mmpbsa.kras_5xco_pilot.v1",
        "output_dir": str(output),
        "template_cif": str(template_cif),
        "mg_source_cif": str(mg_source_cif),
        "protocol": str(protocol_path),
        "smoke_protocol": str(smoke_protocol_path),
        "variant_ids": variant_ids,
        "states": state_ids,
        "job_count": len(jobs),
        "protein_residue_count": protein_residue_count,
        "template_missing_notes": ["5XCO coordinate model starts at SER auth_seq 0; no artificial N-terminal Gly was added."],
        "gdp_atom_count": len(gdp_atoms),
        "mg_transfer": {
            "source_pdb_id": "5US4",
            "source_chain": mg_report["source_chain"],
            "gdp_fit_atom_count": mg_report["fit_atom_count"],
            "gdp_fit_rmsd_angstrom": mg_report["fit_rmsd_angstrom"],
            "mg_xyz": [round(float(v), 3) for v in mg_report["mg_xyz"]],
        },
        "jobs": jobs,
        "run_hints": {
            "dry_prepare": f"mmpbsa peptide run {output} --protocol {protocol_path} --mode prepare --resume",
            "smoke_build": f"python -m validation.kras_5xco.build_pilot {output}_smoke --template-cif {template_cif} --mg-source-cif {mg_source_cif} --gdp-lib {gdp_lib} --gdp-frcmod {gdp_frcmod} --variants WT_PEP_0001 --force",
            "smoke_wt_gdp_only": f"mmpbsa peptide run {output}_smoke --job-id WT_PEP_0001_gdp_only --protocol {output}_smoke/kras_5xco_smoke_20ps.yaml --force",
            "smoke_wt_gdp_mg": f"mmpbsa peptide run {output}_smoke --job-id WT_PEP_0001_gdp_mg --protocol {output}_smoke/kras_5xco_smoke_20ps.yaml --force",
        },
    }
    write_json_atomic(output / "pilot_manifest.json", report)
    write_text_atomic(output / "run_pilot_one_gpu.sh", run_script_text(output, protocol_path, [job["job_id"] for job in jobs]))
    return report


def require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


def download_rcsb_cif(pdb_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    with urllib.request.urlopen(url, timeout=60) as response:
        output.write_bytes(response.read())


def load_cif_atoms(path: Path) -> list[CifAtom]:
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ModuleNotFoundError as exc:
        raise SystemExit("KRAS pilot building requires Biopython; run inside the md mamba environment.") from exc

    data = MMCIF2Dict(str(path))
    required = [
        "_atom_site.group_PDB",
        "_atom_site.auth_atom_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.type_symbol",
    ]
    for key in required:
        if key not in data:
            raise SystemExit(f"mmCIF file is missing {key}: {path}")
    count = len(data["_atom_site.group_PDB"])
    alt_ids = data.get("_atom_site.label_alt_id", ["."] * count)
    models = data.get("_atom_site.pdbx_PDB_model_num", ["1"] * count)
    occupancies = data.get("_atom_site.occupancy", ["1.00"] * count)
    bfactors = data.get("_atom_site.B_iso_or_equiv", ["0.00"] * count)
    atoms: list[CifAtom] = []
    for idx in range(count):
        if str(models[idx]) not in {"1", ".", "?"}:
            continue
        if str(alt_ids[idx]) not in {".", "?", "A"}:
            continue
        atoms.append(
            CifAtom(
                group=str(data["_atom_site.group_PDB"][idx]),
                atom=str(data["_atom_site.auth_atom_id"][idx]),
                resname=str(data["_atom_site.auth_comp_id"][idx]).upper(),
                chain=str(data["_atom_site.auth_asym_id"][idx]),
                resseq=str(data["_atom_site.auth_seq_id"][idx]),
                x=float(data["_atom_site.Cartn_x"][idx]),
                y=float(data["_atom_site.Cartn_y"][idx]),
                z=float(data["_atom_site.Cartn_z"][idx]),
                occupancy=safe_float(occupancies[idx], 1.0),
                bfactor=safe_float(bfactors[idx], 0.0),
                element=str(data["_atom_site.type_symbol"][idx]).strip(),
            )
        )
    return atoms


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def template_protein_atoms(atoms: list[CifAtom]) -> list[CifAtom]:
    return [atom for atom in atoms if atom.group == "ATOM" and atom.chain == "A"]


def template_peptide_atoms(atoms: list[CifAtom]) -> dict[int, list[CifAtom]]:
    residues: dict[int, list[CifAtom]] = {}
    for atom in atoms:
        if atom.chain != "B" or atom.resname == "HOH":
            continue
        if atom.group not in {"ATOM", "HETATM"}:
            continue
        try:
            logical_position = int(atom.resseq) + 1
        except ValueError:
            continue
        if 1 <= logical_position <= 21:
            residues.setdefault(logical_position, []).append(atom)
    if set(residues) != set(range(1, 22)):
        raise SystemExit(f"5XCO peptide template does not contain all logical positions 1-21: {sorted(residues)}")
    return residues


def template_gdp_atoms(atoms: list[CifAtom]) -> list[CifAtom]:
    return [atom for atom in atoms if atom.chain == "A" and atom.resname == "GDP"]


def count_template_residues(atoms: list[CifAtom]) -> int:
    residues: list[tuple[str, str, str]] = []
    for atom in atoms:
        key = (atom.chain, atom.resseq, atom.resname)
        if key not in residues:
            residues.append(key)
    return len(residues)


def variant_peptide_atoms(template: dict[int, list[CifAtom]], spec: VariantSpec) -> list[list[CifAtom]]:
    residues: list[list[CifAtom]] = []
    for position in spec.keep_positions:
        target_resname = spec.mutations.get(position)
        source_atoms = template[position]
        if target_resname == "ALA":
            kept = [atom for atom in source_atoms if normalize_atom_name(atom.atom) in {"N", "CA", "C", "O", "CB"}]
            residues.append([replace_resname(atom, "ALA") for atom in kept])
        else:
            residues.append(source_atoms)
    return residues


def replace_resname(atom: CifAtom, resname: str) -> CifAtom:
    return CifAtom(atom.group, atom.atom, resname, atom.chain, atom.resseq, atom.x, atom.y, atom.z, atom.occupancy, atom.bfactor, atom.element)


def write_selected_pdb(
    output: Path,
    *,
    protein_atoms: list[CifAtom],
    peptide_atoms: list[list[CifAtom]],
    protein_residue_count: int,
    receptor_cofactor_count: int,
) -> None:
    lines: list[str] = []
    serial = 1
    residue_map: dict[tuple[str, str, str], int] = {}
    next_residue = 1
    for atom in protein_atoms:
        key = (atom.chain, atom.resseq, atom.resname)
        if key not in residue_map:
            residue_map[key] = next_residue
            next_residue += 1
        resseq = residue_map[key]
        lines.append(format_pdb_line("ATOM", serial, normalize_atom_name(atom.atom), atom.resname, "A", resseq, atom.x, atom.y, atom.z, atom.occupancy, atom.bfactor, atom.element))
        serial += 1
    lines.append("TER\n")
    peptide_first_resseq = protein_residue_count + receptor_cofactor_count + 1
    for offset, residue_atoms in enumerate(peptide_atoms):
        resseq = peptide_first_resseq + offset
        source_resname = residue_atoms[0].resname
        resname = peptide_resname_for_amber(source_resname)
        for atom in residue_atoms:
            lines.append(
                format_pdb_line("ATOM", serial, normalize_atom_name(atom.atom), resname, "B", resseq, atom.x, atom.y, atom.z, atom.occupancy, atom.bfactor, atom.element)
            )
            serial += 1
    lines.extend(["TER\n", "END\n"])
    write_text_atomic(output, "".join(lines))


def peptide_resname_for_amber(resname: str) -> str:
    if resname == "NH2":
        return "NHE"
    if resname == "CYS":
        return "CYX"
    return resname


def normalize_atom_name(atom_name: str) -> str:
    return atom_name.strip().replace("'", "*")


def write_cofactor_pdb(output: Path, atoms: list[CifAtom], *, resname: str, chain: str, atom_serial_start: int) -> None:
    lines: list[str] = []
    serial = atom_serial_start
    for atom in atoms:
        lines.append(format_pdb_line("HETATM", serial, normalize_atom_name(atom.atom), resname, chain, 1, atom.x, atom.y, atom.z, 1.0, 0.0, atom.element))
        serial += 1
    lines.append("END\n")
    write_text_atomic(output, "".join(lines))


def write_mg_pdb(output: Path, xyz: tuple[float, float, float]) -> None:
    line = format_pdb_line("HETATM", 1, "MG", "MG", "M", 1, xyz[0], xyz[1], xyz[2], 1.0, 0.0, "Mg")
    write_text_atomic(output, line + "END\n")


def transfer_mg_from_source(template_atoms: list[CifAtom], source_atoms: list[CifAtom]) -> dict[str, Any]:
    ref_gdp = gdp_atom_map(template_atoms, chain="A")
    best: dict[str, Any] | None = None
    for chain in sorted({atom.chain for atom in source_atoms}):
        source_gdp = gdp_atom_map(source_atoms, chain=chain)
        source_mg = [atom for atom in source_atoms if atom.chain == chain and atom.resname == "MG" and normalize_atom_name(atom.atom).upper() == "MG"]
        common = sorted(set(ref_gdp) & set(source_gdp))
        if len(common) < 10 or not source_mg:
            continue
        moving = np.array([source_gdp[name] for name in common], dtype=float)
        fixed = np.array([ref_gdp[name] for name in common], dtype=float)
        rotation, translation = kabsch(moving, fixed)
        fitted = moving @ rotation + translation
        rmsd = math.sqrt(float(np.mean(np.sum((fitted - fixed) ** 2, axis=1))))
        mg_xyz_array = np.array([[source_mg[0].x, source_mg[0].y, source_mg[0].z]], dtype=float) @ rotation + translation
        candidate = {
            "source_chain": chain,
            "fit_atom_count": len(common),
            "fit_rmsd_angstrom": rmsd,
            "mg_xyz": tuple(float(v) for v in mg_xyz_array[0]),
        }
        if best is None or candidate["fit_rmsd_angstrom"] < best["fit_rmsd_angstrom"]:
            best = candidate
    if best is None:
        raise SystemExit("Could not transfer Mg: no source chain had GDP and MG")
    return best


def gdp_atom_map(atoms: list[CifAtom], *, chain: str) -> dict[str, tuple[float, float, float]]:
    result: dict[str, tuple[float, float, float]] = {}
    for atom in atoms:
        if atom.chain == chain and atom.resname == "GDP":
            name = normalize_atom_name(atom.atom)
            if not name.upper().startswith("H"):
                result[name] = (atom.x, atom.y, atom.z)
    return result


def kabsch(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    x = moving - moving_center
    y = fixed - fixed_center
    covariance = x.T @ y
    v, _s, wt = np.linalg.svd(covariance)
    determinant = np.sign(np.linalg.det(v @ wt))
    correction = np.diag([1.0, 1.0, determinant])
    rotation = v @ correction @ wt
    translation = fixed_center - moving_center @ rotation
    return rotation, translation


def format_pdb_line(
    record: str,
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    bfactor: float,
    element: str,
) -> str:
    atom_field = pdb_atom_field(atom_name, element)
    return f"{record:<6}{serial:5d} {atom_field}{resname:>4s} {chain:1s}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element.strip():>2s}\n"


def pdb_atom_field(atom_name: str, element: str) -> str:
    name = atom_name.strip()[:4]
    elem = element.strip().upper()
    if len(name) == 4:
        return name
    if len(elem) == 1 and name and not name[0].isdigit():
        return f" {name:<3s}"
    return f"{name:<4s}"


def pilot_protocol_text(production_ns: float, mmpbsa_start_ns: float, seed_base: int) -> str:
    return f"""protocol:
  replica_indices: [1, 2, 3]
  replica_count: 3
  production_ns: {production_ns:.1f}
  xtc_interval_ps: 20.0
  mmpbsa_start_ns: {mmpbsa_start_ns:.1f}
  mmpbsa_interval_ps: 20.0
  min_mmpbsa_frames: 1500
system:
  temperature_k: 300.0
  salt_molar: 0.15
  solvent_padding_angstrom: 10.0
  solvent_shape: box
  allow_box_retry: false
amber_prep:
  recipe: 5xco_kras_gdp_mg_capped_cyclic_peptide
  nonstandard_policy: fail
  default_ligand_param_mode: auto
  protein_ff: leaprc.protein.ff14SB
  ligand_ff: leaprc.gaff2
  water_ff: leaprc.water.tip3p
  gaff_version: gaff2
  charge_method: bcc
  pb_radii: mbondi2
md:
  em_steps: 50000
  emstep: 0.001
  nvt_steps: 50000
  npt_steps: 100000
  seed_base: {seed_base}
  ntomp: 8
mmpbsa:
  enabled: true
  mpi: true
  np: 16
  keep_tmp: false
  explicit_water_count: 0
  entropy: none
  gb_igb: 5
  gb_epsout: 78.5
  pb_exdi: 80.0
  pb_inp: 2
  pb_radiopt: 0
  pb_prbrad: 1.4
  pb_fillratio: 4.0
  internal_limit_kcal_mol: 20000.0
  internal_std_limit_kcal_mol: 5000.0
  epsilon: 4.0
qc:
  receptor_rmsd_fail_angstrom: 5.0
  ligand_rmsd_warn_angstrom: 10.0
  peptide_rmsd_warn_angstrom: 10.0
  native_contacts_fail_min: 0
  interface_distance_fail_angstrom: 8.0
export:
  pymol_stride: 25
runtime:
  mamba_env: md
  gmxrc: ${{GMXRC}}
  gmx_bin: gmx_mpi
  gpu_id: 0
  mpi4py_path: ''
debug:
  keep_step_tmp: false
"""


def smoke_protocol_text(seed_base: int) -> str:
    return f"""protocol:
  replica_indices: [1]
  replica_count: 1
  production_ns: 0.02
  xtc_interval_ps: 2.0
  mmpbsa_start_ns: 0.0
  mmpbsa_interval_ps: 2.0
  min_mmpbsa_frames: 5
system:
  temperature_k: 300.0
  salt_molar: 0.15
  solvent_padding_angstrom: 10.0
  solvent_shape: box
  allow_box_retry: false
amber_prep:
  recipe: 5xco_kras_gdp_mg_capped_cyclic_peptide_smoke
  nonstandard_policy: fail
  default_ligand_param_mode: auto
  protein_ff: leaprc.protein.ff14SB
  ligand_ff: leaprc.gaff2
  water_ff: leaprc.water.tip3p
  gaff_version: gaff2
  charge_method: bcc
  pb_radii: mbondi2
md:
  em_steps: 200
  emstep: 0.001
  nvt_steps: 500
  npt_steps: 500
  seed_base: {seed_base}
  ntomp: 8
mmpbsa:
  enabled: false
  mpi: false
  np: 1
  keep_tmp: false
  explicit_water_count: 0
  entropy: none
  gb_igb: 5
  gb_epsout: 78.5
  pb_exdi: 80.0
  pb_inp: 2
  pb_radiopt: 0
  pb_prbrad: 1.4
  pb_fillratio: 4.0
  internal_limit_kcal_mol: 20000.0
  internal_std_limit_kcal_mol: 5000.0
  epsilon: 4.0
qc:
  receptor_rmsd_fail_angstrom: 5.0
  ligand_rmsd_warn_angstrom: 10.0
  peptide_rmsd_warn_angstrom: 10.0
  native_contacts_fail_min: 0
  interface_distance_fail_angstrom: 8.0
export:
  pymol_stride: 1
runtime:
  mamba_env: md
  gmxrc: ${{GMXRC}}
  gmx_bin: gmx_mpi
  gpu_id: 0
  mpi4py_path: ''
debug:
  keep_step_tmp: false
"""


def run_script_text(output: Path, protocol_path: Path, job_ids: list[str]) -> str:
    jobs = " ".join(job_ids)
    return f"""#!/usr/bin/env bash
set -euo pipefail

RUN_DIR={output}
PROTOCOL={protocol_path}
PYTHON=${{PYTHON:-python}}
export MAMBA_ENV=${{MAMBA_ENV:-md}}
export GMXRC=${{GMXRC:-/home/silong/projects/gromacs/gromacs202602/bin/GMXRC}}
export GMX_BIN=${{GMX_BIN:-gmx_mpi}}
export GPU_ID=${{GPU_ID:-0}}
export NTOMP=${{NTOMP:-8}}
export MMPBSA_NP=${{MMPBSA_NP:-16}}
export PYTHONUNBUFFERED=1

for job in {jobs}; do
  date -Is
  echo "running $job on GPU_ID=$GPU_ID"
  "$PYTHON" -m mmpbsa peptide run "$RUN_DIR" --job-id "$job" --protocol "$PROTOCOL" --resume
done
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from mmpbsa.common import utc_now, write_csv_atomic, write_json_atomic, write_text_atomic
from mmpbsa.ligand import mol2_total_charge


DEFAULT_RESOURCES = Path("/data2/silong/projects/resources/boltz_kras")
DEFAULT_BOLTZ2_CIF_DIR = Path("/data2/silong/projects/tmp/20260629_kras_boltz2_top10")
DEFAULT_BOLTZ2_MANIFEST = DEFAULT_BOLTZ2_CIF_DIR / "md_selected_manifest.csv"
DEFAULT_BOLTZ2_IPTM_MANIFEST = DEFAULT_BOLTZ2_CIF_DIR / "md_selected_iptm_only_manifest.csv"
DEFAULT_BOLTZ_SMILES_MANIFEST = DEFAULT_RESOURCES / "md_selected_manifest.csv"
DEFAULT_PDB_PREP_SRC = Path("/data2/silong/projects/tools/pdb_tools/pdb_prep/src")
DEFAULT_GMXRC = Path("/data2/silong/projects/gromacs/gromacs202602/bin/GMXRC")
DEFAULT_STRICT_REPORT_DIRNAME = "final_strict_3x5ns_10prod"
DEFAULT_STRICT_3X15_REPORT_DIRNAME = "final_strict_3x15ns_10prod"
RUN_MANIFEST_CANDIDATES = (
    "boltz_6wgn_gnp_mg_manifest.json",
    "boltz2_6wgn_gnp_mg_manifest.json",
)

STRICT_REPORT_PROFILES = {
    "3x5ns": {
        "display_name": "3x5 ns",
        "output_dirname": DEFAULT_STRICT_REPORT_DIRNAME,
        "md_protocol": "3 replicas x 5 ns",
        "mmpbsa_window": "3-5 ns",
        "expected_startframe": 151,
        "expected_mmpbsa_start_ns": 3.0,
        "expected_production_ns": 5.0,
        "expected_replicas": 3,
        "expected_frames": 303,
        "expected_frames_per_replica": 101,
    },
    "3x15ns": {
        "display_name": "3x15 ns",
        "output_dirname": DEFAULT_STRICT_3X15_REPORT_DIRNAME,
        "md_protocol": "3 replicas x 15 ns",
        "mmpbsa_window": "5-15 ns",
        "expected_startframe": 251,
        "expected_mmpbsa_start_ns": 5.0,
        "expected_production_ns": 15.0,
        "expected_replicas": 3,
        "expected_frames": 1503,
        "expected_frames_per_replica": 501,
    },
}

PROTEIN_CHAIN = "A"
LIGAND_SELECTOR = "LIG1:L:1"
GNP_SELECTOR = "GNP:G:1"
MG_SELECTOR = "MG:M:1"

DATASET_FIELDS = [
    "job_id",
    "name",
    "complex_pdb",
    "receptor_chains",
    "ligand_file",
    "ligand_resname",
    "ligand_charge",
    "ligand_param_mode",
    "pdb_id",
    "model_id",
    "source",
    "ligand_chain",
    "ligand_resseq",
    "ligand_mol2",
    "ligand_frcmod",
    "ligand_lib",
    "receptor_cofactor_files",
    "receptor_cofactor_frcmods",
    "receptor_cofactor_libs",
    "receptor_cofactor_residue_count",
    "deltaG_exp_kJ_mol",
    "ic50_nM",
    "kd_nM",
]

IPTM_TO_CONFIG_FIELDS = {
    "selection_rank": "selection_rank",
    "id": "boltz2_id",
    "iptm": "boltz2_iptm",
    "kras_kd_pred": "boltz2_kras_kd_pred",
    "kd": "boltz2_kd",
    "kd_log10_nM_from_M": "boltz2_kd_log10_nM_from_M",
    "prediction_dir": "boltz2_prediction_dir",
}


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


def load_boltz_manifest(path: Path, *, selection_tier: str = "primary", limit: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if str(row.get("selection_tier") or "").strip() == selection_tier]
    if limit is not None:
        selected = selected[:limit]
    return selected


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing Boltz2 manifest with ligand SMILES: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Boltz2 manifest is empty: {path}")
    missing = [field for field in ("rank", "smiles") if field not in (rows[0] or {})]
    if missing:
        raise SystemExit(f"Boltz2 manifest {path} is missing required columns: {', '.join(missing)}")
    return rows


def cif_rank_model(path: Path) -> tuple[int, str]:
    match = re.search(r"rank_(\d+)_model_(\d+)", path.stem)
    if not match:
        raise SystemExit(f"Could not parse rank/model from CIF filename: {path.name}")
    rank_text, model_text = match.groups()
    return int(rank_text), f"rank_{int(rank_text):04d}_model_{int(model_text)}"


def manifest_model_id(row: dict[str, str]) -> str:
    for key in ("representative_model", "model_id", "model"):
        value = str(row.get(key) or "").strip()
        if value:
            return normalized_model_id(value)
    rank = str(row.get("rank") or "").strip()
    if rank:
        return f"rank_{int(float(rank)):04d}"
    return ""


def normalized_model_id(value: str) -> str:
    match = re.search(r"rank_(\d+)_model_(\d+)", value)
    if match:
        rank_text, model_text = match.groups()
        return f"rank_{int(rank_text):04d}_model_{int(model_text)}"
    return value.strip()


def find_manifest_row_for_cif(cif: Path, rows: Sequence[dict[str, str]]) -> dict[str, str]:
    rank, model_id = cif_rank_model(cif)
    exact = [row for row in rows if manifest_model_id(row) == model_id]
    if not exact:
        exact = [row for row in rows if int(float(str(row.get("rank") or "-1"))) == rank]
    if len(exact) != 1:
        raise SystemExit(f"Expected one manifest row for {cif.name} ({model_id}), found {len(exact)}")
    row = dict(exact[0])
    smiles = str(row.get("smiles") or "").strip()
    if not smiles:
        raise SystemExit(f"Manifest row for {cif.name} does not provide a ligand SMILES")
    row.setdefault("representative_model", model_id)
    row["representative_model"] = normalized_model_id(str(row.get("representative_model") or model_id))
    row["rank"] = str(rank)
    return row


def manifest_smiles(row: dict[str, str]) -> str:
    for key in ("smiles", "SMILES", "canonical_smiles", "canonical_SMILES"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def load_iptm_manifest_rows(path: Path, *, set_name: str = "primary", limit: int | None = 10) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing Boltz2 iPTM manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Boltz2 iPTM manifest is empty: {path}")
    required = ("set", "rank", "name", "model", "cif_path")
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise SystemExit(f"Boltz2 iPTM manifest {path} is missing required columns: {', '.join(missing)}")
    selected = [dict(row) for row in rows if str(row.get("set") or "").strip() == set_name]
    has_selection_rank = any(str(row.get("selection_rank") or "").strip() for row in selected)
    if has_selection_rank:
        selected.sort(key=lambda row: int(float(str(row.get("selection_rank") or "999999"))))
    for index, row in enumerate(selected, start=1):
        if not str(row.get("selection_rank") or "").strip():
            row["selection_rank"] = str(index)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise SystemExit(f"No Boltz2 iPTM rows selected from {path} with set={set_name!r}")
    return selected


def default_search_roots(manifest_path: Path) -> list[Path]:
    roots = [
        manifest_path.resolve().parent,
        manifest_path.resolve().parent.parent,
        Path("/data2/silong/projects/tmp"),
        Path("/data2/silong/projects"),
        Path("/data2/silong/projects/resources"),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def resolve_existing_path(path_text: str, *, manifest_path: Path, search_roots: Sequence[Path]) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    raw = Path(text).expanduser()
    if raw.is_absolute():
        return raw if raw.exists() else None
    candidates = [manifest_path.resolve().parent / raw, *[root.resolve() / raw for root in search_roots]]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_iptm_cif(
    row: dict[str, str],
    *,
    manifest_path: Path,
    search_roots: Sequence[Path],
    local_cif_dir: Path | None = None,
) -> Path | None:
    resolved = resolve_existing_path(str(row.get("cif_path") or ""), manifest_path=manifest_path, search_roots=search_roots)
    if resolved is not None:
        return resolved
    model = iptm_model_id(row)
    if local_cif_dir is not None:
        candidate = local_cif_dir.resolve() / f"{model}.cif"
        if candidate.exists():
            return candidate
    local_candidate = manifest_path.resolve().parent / f"{model}.cif"
    if local_candidate.exists():
        return local_candidate
    return None


def load_smiles_fallbacks(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fallbacks: dict[str, str] = {}
    for row in rows:
        smiles = manifest_smiles(row)
        if not smiles:
            continue
        for key in smiles_lookup_keys(row):
            fallbacks.setdefault(key, smiles)
    return fallbacks


def smiles_lookup_keys(row: dict[str, str]) -> list[str]:
    keys: list[str] = []
    for field in ("rank", "name", "model", "representative_model"):
        value = str(row.get(field) or "").strip()
        if not value:
            continue
        keys.append(value)
        rank_match = re.search(r"rank_(\d+)", value)
        if rank_match:
            rank_int = int(rank_match.group(1))
            keys.extend([str(rank_int), f"rank_{rank_int:04d}"])
    rank_text = str(row.get("rank") or "").strip()
    if rank_text:
        rank_int = int(float(rank_text))
        keys.extend([str(rank_int), f"rank_{rank_int:04d}"])
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def fallback_smiles_for_row(row: dict[str, str], smiles_fallbacks: dict[str, str]) -> str:
    for key in smiles_lookup_keys(row):
        smiles = smiles_fallbacks.get(key)
        if smiles:
            return smiles
    return ""


def iptm_model_id(row: dict[str, str]) -> str:
    model = str(row.get("model") or "").strip()
    if model:
        return model
    identifier = str(row.get("id") or row.get("name") or "").strip()
    if not identifier:
        raise SystemExit(f"Boltz2 iPTM row has no model/id/name: {row}")
    return identifier


def iptm_job_id(row: dict[str, str]) -> str:
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", iptm_model_id(row)).strip("_")
    return f"kras6wgnb2_{model}"


def resolve_iptm_topology(
    row: dict[str, str],
    *,
    manifest_path: Path,
    search_roots: Sequence[Path],
    smiles_fallbacks: dict[str, str] | None = None,
) -> dict[str, Any]:
    smiles = manifest_smiles(row)
    if not smiles and smiles_fallbacks:
        smiles = fallback_smiles_for_row(row, smiles_fallbacks)
    if smiles:
        return {
            "kind": "smiles",
            "smiles": smiles,
            "ligand_file": "",
            "ligand_mol2": "",
            "ligand_frcmod": "",
            "ligand_lib": "",
        }

    ligand_file = None
    for key in ("ligand_sdf", "ligand_file", "source_ligand_sdf", "ligand_path"):
        ligand_file = resolve_existing_path(str(row.get(key) or ""), manifest_path=manifest_path, search_roots=search_roots)
        if ligand_file is not None:
            break
    ligand_mol2 = None
    for key in ("ligand_mol2", "mol2"):
        ligand_mol2 = resolve_existing_path(str(row.get(key) or ""), manifest_path=manifest_path, search_roots=search_roots)
        if ligand_mol2 is not None:
            break
    ligand_frcmod = None
    for key in ("ligand_frcmod", "frcmod"):
        ligand_frcmod = resolve_existing_path(str(row.get(key) or ""), manifest_path=manifest_path, search_roots=search_roots)
        if ligand_frcmod is not None:
            break
    ligand_libs: list[str] = []
    for item in split_semicolon_paths(str(row.get("ligand_lib") or row.get("ligand_libs") or "")):
        resolved = resolve_existing_path(item, manifest_path=manifest_path, search_roots=search_roots)
        if resolved is None:
            raise SystemExit(f"Missing ligand library for {iptm_model_id(row)}: {item}")
        ligand_libs.append(str(resolved))

    if ligand_mol2 is not None and ligand_frcmod is not None:
        return {
            "kind": "preparam",
            "smiles": "",
            "ligand_file": str(ligand_file or ligand_mol2),
            "ligand_mol2": str(ligand_mol2),
            "ligand_frcmod": str(ligand_frcmod),
            "ligand_lib": ";".join(ligand_libs),
        }
    return {
        "kind": "missing",
        "smiles": "",
        "ligand_file": str(ligand_file or ""),
        "ligand_mol2": str(ligand_mol2 or ""),
        "ligand_frcmod": str(ligand_frcmod or ""),
        "ligand_lib": ";".join(ligand_libs),
    }


def split_semicolon_paths(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;,\n]+", value) if item.strip()]


def preflight_iptm_manifest(
    manifest_path: Path,
    *,
    set_name: str = "primary",
    limit: int | None = 10,
    search_roots: Sequence[Path] | None = None,
    local_cif_dir: Path | None = DEFAULT_BOLTZ2_CIF_DIR,
    smiles_manifest: Path | None = DEFAULT_BOLTZ_SMILES_MANIFEST,
) -> dict[str, Any]:
    manifest = manifest_path.resolve()
    roots = list(search_roots or default_search_roots(manifest))
    rows = load_iptm_manifest_rows(manifest, set_name=set_name, limit=limit)
    smiles_fallbacks = load_smiles_fallbacks(smiles_manifest)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in rows:
        model = iptm_model_id(row)
        cif = resolve_iptm_cif(row, manifest_path=manifest, search_roots=roots, local_cif_dir=local_cif_dir)
        topology = resolve_iptm_topology(row, manifest_path=manifest, search_roots=roots, smiles_fallbacks=smiles_fallbacks)
        issues: list[str] = []
        if cif is None:
            issues.append(f"missing CIF path: {row.get('cif_path')}")
        if topology["kind"] == "missing":
            issues.append("missing ligand topology: provide smiles/canonical_smiles or ligand_mol2+ligand_frcmod")
        if issues:
            failures.append(f"{model}: " + "; ".join(issues))
        records.append(
            {
                "selection_rank": row.get("selection_rank", ""),
                "id": row.get("id", ""),
                "name": row.get("name", ""),
                "model": model,
                "job_id": iptm_job_id(row),
                "cif": str(cif) if cif else "",
                "topology_kind": topology["kind"],
                "topology_source": "fallback_smiles" if topology["kind"] == "smiles" and not manifest_smiles(row) else topology["kind"],
                "ligand_file": topology.get("ligand_file", ""),
                "ligand_mol2": topology.get("ligand_mol2", ""),
                "ligand_frcmod": topology.get("ligand_frcmod", ""),
                "smiles": topology.get("smiles", ""),
                "iptm": row.get("iptm", ""),
                "kras_kd_pred": row.get("kras_kd_pred", ""),
                "issues": "; ".join(issues),
            }
        )
    return {
        "schema_version": "mmpbsa.validation.kras_6wgn_boltz.iptm_preflight.v1",
        "manifest": str(manifest),
        "set": set_name,
        "limit": limit,
        "local_cif_dir": str(local_cif_dir.resolve()) if local_cif_dir else "",
        "smiles_manifest": str(smiles_manifest.resolve()) if smiles_manifest else "",
        "selected": len(records),
        "passed": not failures,
        "failures": failures,
        "records": records,
    }


def find_representative_cif(row: dict[str, str], top_dir: Path) -> Path:
    model = str(row.get("representative_model") or "").strip()
    candidates: list[Path] = []
    if model:
        candidates = sorted(top_dir.glob(f"*{model}.cif"))
    if not candidates:
        rank = int(str(row["rank"]).strip())
        candidates = sorted(top_dir.glob(f"*rank_{rank:04d}*.cif"))
    if len(candidates) != 1:
        raise SystemExit(f"Expected one representative CIF for rank={row.get('rank')} model={model!r}, found {len(candidates)}")
    return candidates[0]


def parse_cif_atoms(path: Path) -> list[CifAtom]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    headers: list[str] = []
    in_atom_loop = False
    atoms: list[CifAtom] = []
    for raw in lines:
        stripped = raw.strip()
        if stripped == "loop_":
            headers = []
            in_atom_loop = False
            continue
        if stripped.startswith("_atom_site."):
            headers.append(stripped)
            in_atom_loop = True
            continue
        if not in_atom_loop:
            continue
        if not stripped or stripped.startswith("#"):
            in_atom_loop = False
            continue
        if not stripped.startswith(("ATOM ", "HETATM ")):
            continue
        parts = shlex.split(stripped)
        if len(parts) < len(headers):
            raise SystemExit(f"Could not parse atom_site row in {path}: {raw}")
        data = {header.rsplit(".", 1)[1]: parts[idx] for idx, header in enumerate(headers)}
        model = data.get("pdbx_PDB_model_num", "1")
        if model not in {"1", ".", "?"}:
            continue
        alt_id = data.get("label_alt_id", ".")
        if alt_id not in {".", "?", "A"}:
            continue
        atoms.append(
            CifAtom(
                group=data["group_PDB"],
                atom=data.get("auth_atom_id") or data["label_atom_id"],
                resname=(data.get("auth_comp_id") or data["label_comp_id"]).upper(),
                chain=data.get("auth_asym_id") or data["label_asym_id"],
                resseq=data.get("auth_seq_id") or data.get("label_seq_id") or "1",
                x=float(data["Cartn_x"]),
                y=float(data["Cartn_y"]),
                z=float(data["Cartn_z"]),
                occupancy=safe_float(data.get("occupancy"), 1.0),
                bfactor=safe_float(data.get("B_iso_or_equiv"), 0.0),
                element=str(data.get("type_symbol") or "").strip(),
            )
        )
    if not atoms:
        raise SystemExit(f"No atom_site records found in {path}")
    return atoms


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def cif_summary(path: Path) -> dict[str, Any]:
    atoms = parse_cif_atoms(path)
    protein_residues = {
        (atom.chain, atom.resseq, atom.resname)
        for atom in atoms
        if atom.group == "ATOM" and atom.chain == PROTEIN_CHAIN
    }
    return {
        "chains": sorted({atom.chain for atom in atoms}),
        "protein_residue_count": len(protein_residues),
        "ligand_atom_count": sum(1 for atom in atoms if atom.resname == "LIG1"),
        "gnp_atom_count": sum(1 for atom in atoms if atom.resname == "GNP"),
        "mg_atom_count": sum(1 for atom in atoms if atom.resname == "MG"),
        "total_atom_count": len(atoms),
    }


def job_id_for_row(row: dict[str, str]) -> str:
    model = str(row.get("representative_model") or "").strip()
    if model:
        return f"kras6wgn_{model}"
    return f"kras6wgn_rank_{int(row['rank']):04d}"


def job_config(
    *,
    job_id: str,
    row: dict[str, str],
    relative_input_dir: str = "input",
    ligand_charge: int = 0,
    gnp_charge: int = -4,
    mg_charge: int = 2,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "name": f"Boltz KRAS 6WGN GNP-Mg {row.get('representative_model') or row.get('rank')}",
        "complex_pdb": f"{relative_input_dir}/complex.pdb",
        "receptor_chains": PROTEIN_CHAIN,
        "ligand_file": f"{relative_input_dir}/ligand.sdf",
        "ligand_resname": "LIG",
        "ligand_charge": ligand_charge,
        "ligand_param_mode": "preparam",
        "pdb_id": "KRAS_6WGN_BOLTZ",
        "model_id": row.get("representative_model") or f"rank_{int(row['rank']):04d}",
        "source": "Boltz KRAS cyclic ligand pose; 6WGN-like KRAS(G12D)-GNP-Mg active receptor state",
        "ligand_chain": "L",
        "ligand_resseq": "1",
        "ligand_mol2": f"{relative_input_dir}/ligand.mol2",
        "ligand_frcmod": f"{relative_input_dir}/ligand.frcmod",
        "receptor_cofactor_files": f"{relative_input_dir}/gnp.mol2;{relative_input_dir}/mg.pdb",
        "receptor_cofactor_frcmods": f"{relative_input_dir}/gnp.frcmod",
        "receptor_cofactor_residue_count": 2,
        "nucleotide_state": "GNP",
        "reference_pdb_id": "6WGN",
        "gnp_charge": gnp_charge,
        "mg_charge": mg_charge,
        "receptor_cofactor_net_charge": gnp_charge + mg_charge,
        "boltz_rank": row.get("rank", ""),
        "boltz_smiles": row.get("smiles", ""),
        "boltz_composite_score": row.get("representative_composite_score", ""),
        "boltz_ligand_iptm": row.get("representative_ligand_iptm", ""),
        "boltz_pose_cluster_size": row.get("pose_cluster_size", ""),
        "boltz_pose_cluster_max_rmsd": row.get("pose_cluster_max_rmsd", ""),
    }


def make_boltz_jobs(
    resources_dir: Path,
    run_dir: Path,
    *,
    selection_tier: str = "primary",
    limit: int | None = 10,
    force: bool = False,
    prepare_inputs: bool = True,
    charge_method: str = "gas",
    pdb_prep_src: Path = DEFAULT_PDB_PREP_SRC,
) -> dict[str, Any]:
    resources = resources_dir.resolve()
    manifest_path = resources / "md_selected_manifest.csv"
    top_dir = resources / "top10"
    if not manifest_path.exists():
        raise SystemExit(f"Missing Boltz manifest: {manifest_path}")
    if not top_dir.exists():
        raise SystemExit(f"Missing Boltz top10 directory: {top_dir}")

    output = run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_boltz_manifest(manifest_path, selection_tier=selection_tier, limit=limit)
    if not rows:
        raise SystemExit(f"No Boltz rows selected from {manifest_path} with selection_tier={selection_tier!r}")

    jobs: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        cif = find_representative_cif(row, top_dir)
        summary = cif_summary(cif)
        validate_boltz_cif_summary(cif, summary)
        job_id = job_id_for_row(row)
        job_dir = output / job_id
        if job_dir.exists() and force:
            shutil.rmtree(job_dir)
        source_dir = job_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        config = job_config(job_id=job_id, row=row, relative_input_dir="source")
        if prepare_inputs:
            prepared = prepare_job_inputs(
                cif,
                source_dir,
                ligand_smiles=str(row.get("smiles") or ""),
                ligand_charge=int(config["ligand_charge"]),
                gnp_charge=int(config["gnp_charge"]),
                charge_method=charge_method,
                pdb_prep_src=pdb_prep_src,
                force=force,
            )
        else:
            prepared = {"status": "skipped"}
        config.update(
            {
                "source_cif": str(cif),
                "source_manifest": str(manifest_path),
                "input_summary": summary,
                "input_preparation": prepared,
            }
        )
        write_json_atomic(job_dir / f"{job_id}.json", config)
        jobs.append(
            {
                "index": index,
                "job_id": job_id,
                "rank": row.get("rank"),
                "representative_model": row.get("representative_model"),
                "cif": str(cif),
                "job_dir": str(job_dir),
                "ligand_charge": config["ligand_charge"],
                "gnp_charge": config["gnp_charge"],
                "mg_charge": config["mg_charge"],
                "cofactor_net_charge": config["receptor_cofactor_net_charge"],
                "input_summary": summary,
                "prepared": prepared,
            }
        )
        dataset_rows.append(config)

    dataset_path = output / "boltz_6wgn_gnp_mg_jobs.csv"
    write_dataset_csv(dataset_path, dataset_rows)
    protocol = "configs/ligand_crystal_3x5ns_mmpbsa_bcc.yaml"
    run_script = output / "run_top10_3x5ns.sh"
    write_text_atomic(run_script, run_script_text(output, protocol, [job["job_id"] for job in jobs]))
    run_script.chmod(0o755)
    report = {
        "schema_version": "mmpbsa.validation.kras_6wgn_boltz.v1",
        "resources_dir": str(resources),
        "run_dir": str(output),
        "selection_tier": selection_tier,
        "limit": limit,
        "job_count": len(jobs),
        "reference_state": "6WGN-like KRAS(G12D)-GNP-Mg active state",
        "charge_reference": {
            "ligand_charge": 0,
            "gnp_charge": -4,
            "mg_charge": 2,
            "receptor_cofactor_net_charge": -2,
            "charge_method": charge_method,
        },
        "dataset_csv": str(dataset_path),
        "production_protocol": protocol,
        "smoke_protocol": "configs/smoke_20ps.yaml",
        "jobs": jobs,
    }
    write_json_atomic(output / "boltz_6wgn_gnp_mg_manifest.json", report)
    return report


def make_boltz_jobs_from_cifs(
    cif_dir: Path,
    manifest_path: Path,
    run_dir: Path,
    *,
    dataset_label: str = "boltz2",
    limit: int | None = None,
    force: bool = False,
    prepare_inputs: bool = True,
    charge_method: str = "gas",
    pdb_prep_src: Path = DEFAULT_PDB_PREP_SRC,
) -> dict[str, Any]:
    source_dir = cif_dir.resolve()
    if not source_dir.exists():
        raise SystemExit(f"Missing Boltz2 CIF directory: {source_dir}")
    cifs = sorted(source_dir.glob("*.cif"), key=lambda path: cif_rank_model(path))
    if limit is not None:
        cifs = cifs[:limit]
    if not cifs:
        raise SystemExit(f"No CIF files found in {source_dir}")

    manifest_rows = load_manifest_rows(manifest_path.resolve())
    output = run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for index, cif in enumerate(cifs, start=1):
        row = find_manifest_row_for_cif(cif, manifest_rows)
        rank, model_id = cif_rank_model(cif)
        summary = cif_summary(cif)
        validate_boltz_cif_summary(cif, summary)
        job_id = f"kras6wgnb2_{model_id}"
        job_dir = output / job_id
        if job_dir.exists() and force:
            shutil.rmtree(job_dir)
        input_dir = job_dir / "source"
        input_dir.mkdir(parents=True, exist_ok=True)
        config = job_config(job_id=job_id, row=row, relative_input_dir="source")
        config.update(
            {
                "name": f"Boltz2 KRAS 6WGN GNP-Mg {model_id}",
                "source": "Boltz2 KRAS cyclic ligand pose; 6WGN-like KRAS(G12D)-GNP-Mg active receptor state",
                "dataset_label": dataset_label,
                "model_id": model_id,
                "boltz_rank": rank,
                "boltz_smiles": row.get("smiles", ""),
            }
        )
        if prepare_inputs:
            prepared = prepare_job_inputs(
                cif,
                input_dir,
                ligand_smiles=str(row.get("smiles") or ""),
                ligand_charge=int(config["ligand_charge"]),
                gnp_charge=int(config["gnp_charge"]),
                charge_method=charge_method,
                pdb_prep_src=pdb_prep_src,
                force=force,
            )
        else:
            prepared = {"status": "skipped"}
        config.update(
            {
                "source_cif": str(cif),
                "source_manifest": str(manifest_path.resolve()),
                "input_summary": summary,
                "input_preparation": prepared,
            }
        )
        write_json_atomic(job_dir / f"{job_id}.json", config)
        jobs.append(
            {
                "index": index,
                "job_id": job_id,
                "rank": rank,
                "representative_model": model_id,
                "cif": str(cif),
                "job_dir": str(job_dir),
                "ligand_charge": config["ligand_charge"],
                "gnp_charge": config["gnp_charge"],
                "mg_charge": config["mg_charge"],
                "cofactor_net_charge": config["receptor_cofactor_net_charge"],
                "input_summary": summary,
                "prepared": prepared,
            }
        )
        dataset_rows.append(config)

    dataset_path = output / f"{dataset_label}_6wgn_gnp_mg_jobs.csv"
    write_dataset_csv(dataset_path, dataset_rows)
    protocol = "configs/ligand_crystal_3x15ns_mmpbsa_bcc.yaml"
    run_script = output / "run_top10_3x15ns.sh"
    write_text_atomic(run_script, run_script_text(output, protocol, [job["job_id"] for job in jobs]))
    run_script.chmod(0o755)
    report = {
        "schema_version": "mmpbsa.validation.kras_6wgn_boltz.cif_dir.v1",
        "dataset_label": dataset_label,
        "cif_dir": str(source_dir),
        "source_manifest": str(manifest_path.resolve()),
        "run_dir": str(output),
        "limit": limit,
        "job_count": len(jobs),
        "reference_state": "6WGN-like KRAS(G12D)-GNP-Mg active state",
        "charge_reference": {
            "ligand_charge": 0,
            "gnp_charge": -4,
            "mg_charge": 2,
            "receptor_cofactor_net_charge": -2,
            "charge_method": charge_method,
        },
        "dataset_csv": str(dataset_path),
        "production_protocol": protocol,
        "smoke_protocol": "",
        "jobs": jobs,
    }
    write_json_atomic(output / f"{dataset_label}_6wgn_gnp_mg_manifest.json", report)
    return report


def make_boltz_jobs_from_iptm_manifest(
    manifest_path: Path,
    run_dir: Path,
    *,
    dataset_label: str = "boltz2",
    set_name: str = "primary",
    limit: int | None = 10,
    search_roots: Sequence[Path] | None = None,
    local_cif_dir: Path | None = DEFAULT_BOLTZ2_CIF_DIR,
    smiles_manifest: Path | None = DEFAULT_BOLTZ_SMILES_MANIFEST,
    force: bool = False,
    prepare_inputs: bool = True,
    charge_method: str = "gas",
    pdb_prep_src: Path = DEFAULT_PDB_PREP_SRC,
) -> dict[str, Any]:
    manifest = manifest_path.resolve()
    roots = list(search_roots or default_search_roots(manifest))
    smiles_fallbacks = load_smiles_fallbacks(smiles_manifest)
    preflight = preflight_iptm_manifest(
        manifest,
        set_name=set_name,
        limit=limit,
        search_roots=roots,
        local_cif_dir=local_cif_dir,
        smiles_manifest=smiles_manifest,
    )
    if not preflight["passed"]:
        raise SystemExit("Boltz2 iPTM preflight failed:\n" + "\n".join(preflight["failures"]))

    rows = load_iptm_manifest_rows(manifest, set_name=set_name, limit=limit)
    output = run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        model_id = iptm_model_id(row)
        cif = resolve_iptm_cif(row, manifest_path=manifest, search_roots=roots, local_cif_dir=local_cif_dir)
        if cif is None:
            raise SystemExit(f"Missing CIF for {model_id}: {row.get('cif_path')}")
        topology = resolve_iptm_topology(row, manifest_path=manifest, search_roots=roots, smiles_fallbacks=smiles_fallbacks)
        if topology["kind"] == "missing":
            raise SystemExit(f"Missing ligand topology for {model_id}")
        summary = cif_summary(cif)
        validate_boltz_cif_summary(cif, summary)
        job_id = iptm_job_id(row)
        job_dir = output / job_id
        if job_dir.exists() and force:
            shutil.rmtree(job_dir)
        input_dir = job_dir / "source"
        input_dir.mkdir(parents=True, exist_ok=True)

        config_row = {
            "rank": row.get("selection_rank", index),
            "representative_model": model_id,
            "smiles": topology.get("smiles", ""),
            "representative_ligand_iptm": row.get("iptm", ""),
        }
        config = job_config(job_id=job_id, row=config_row, relative_input_dir="source")
        config.update(
            {
                "name": f"Boltz2 KRAS 6WGN GNP-Mg {model_id}",
                "source": "Boltz2 iPTM-selected KRAS cyclic ligand pose; 6WGN-like KRAS(G12D)-GNP-Mg active receptor state",
                "dataset_label": dataset_label,
                "model_id": model_id,
                "selection_rank": row.get("selection_rank", index),
                "boltz2_id": row.get("id") or row.get("name", ""),
                "boltz2_name": row.get("name", ""),
                "boltz2_iptm": row.get("iptm", ""),
                "boltz2_kras_kd_pred": row.get("kras_kd_pred", ""),
                "boltz2_kd": row.get("kd", ""),
                "boltz2_kd_log10_nM_from_M": row.get("kd_log10_nM_from_M", ""),
                "boltz2_prediction_dir": row.get("prediction_dir", ""),
                "boltz_rank": row.get("selection_rank", index),
                "boltz_smiles": topology.get("smiles", ""),
            }
        )
        for source_key, config_key in (("ligand_file", "ligand_file"), ("ligand_mol2", "ligand_mol2"), ("ligand_frcmod", "ligand_frcmod"), ("ligand_lib", "ligand_lib")):
            value = str(topology.get(source_key) or "").strip()
            if value:
                config[config_key] = value
        if topology["kind"] == "preparam":
            config["ligand_param_mode"] = "preparam"

        if prepare_inputs:
            if topology["kind"] == "smiles":
                prepared = prepare_job_inputs(
                    cif,
                    input_dir,
                    ligand_smiles=str(topology.get("smiles") or ""),
                    ligand_charge=int(config["ligand_charge"]),
                    gnp_charge=int(config["gnp_charge"]),
                    charge_method=charge_method,
                    pdb_prep_src=pdb_prep_src,
                    force=force,
                )
            else:
                prepared = prepare_job_inputs_with_preparam_ligand(
                    cif,
                    input_dir,
                    topology=topology,
                    ligand_charge=int(config["ligand_charge"]),
                    gnp_charge=int(config["gnp_charge"]),
                    charge_method=charge_method,
                    pdb_prep_src=pdb_prep_src,
                    force=force,
                )
                config["ligand_file"] = "source/" + Path(str(prepared["ligand_file"])).name
                config["ligand_mol2"] = "source/ligand.mol2"
                config["ligand_frcmod"] = "source/ligand.frcmod"
        else:
            prepared = {"status": "skipped", "topology_kind": topology["kind"]}

        config.update(
            {
                "source_cif": str(cif),
                "source_manifest": str(manifest),
                "input_summary": summary,
                "input_preparation": prepared,
            }
        )
        write_json_atomic(job_dir / f"{job_id}.json", config)
        jobs.append(
            {
                "index": index,
                "job_id": job_id,
                "rank": row.get("selection_rank", index),
                "representative_model": model_id,
                "cif": str(cif),
                "job_dir": str(job_dir),
                "ligand_charge": config["ligand_charge"],
                "gnp_charge": config["gnp_charge"],
                "mg_charge": config["mg_charge"],
                "cofactor_net_charge": config["receptor_cofactor_net_charge"],
                "boltz2_id": row.get("id") or row.get("name", ""),
                "boltz2_iptm": row.get("iptm", ""),
                "input_summary": summary,
                "prepared": prepared,
            }
        )
        dataset_rows.append(config)

    dataset_path = output / f"{dataset_label}_iptm_6wgn_gnp_mg_jobs.csv"
    write_dataset_csv(dataset_path, dataset_rows)
    protocol = "configs/ligand_crystal_3x15ns_mmpbsa_bcc.yaml"
    run_script = output / "run_top10_3x15ns.sh"
    write_text_atomic(run_script, run_script_text(output, protocol, [job["job_id"] for job in jobs]))
    run_script.chmod(0o755)
    parallel_script = output / "run_top10_3x15ns_gpu_workers.sh"
    write_text_atomic(parallel_script, gpu_worker_script_text(output, protocol, [job["job_id"] for job in jobs]))
    parallel_script.chmod(0o755)
    report = {
        "schema_version": "mmpbsa.validation.kras_6wgn_boltz.iptm_manifest.v1",
        "dataset_label": dataset_label,
        "source_manifest": str(manifest),
        "local_cif_dir": str(local_cif_dir.resolve()) if local_cif_dir else "",
        "smiles_manifest": str(smiles_manifest.resolve()) if smiles_manifest else "",
        "run_dir": str(output),
        "set": set_name,
        "limit": limit,
        "job_count": len(jobs),
        "reference_state": "6WGN-like KRAS(G12D)-GNP-Mg active state",
        "charge_reference": {
            "ligand_charge": 0,
            "gnp_charge": -4,
            "mg_charge": 2,
            "receptor_cofactor_net_charge": -2,
            "charge_method": charge_method,
        },
        "dataset_csv": str(dataset_path),
        "production_protocol": protocol,
        "worker_script": str(parallel_script),
        "jobs": jobs,
    }
    write_json_atomic(output / f"{dataset_label}_6wgn_gnp_mg_manifest.json", report)
    return report


def validate_boltz_cif_summary(cif: Path, summary: dict[str, Any]) -> None:
    expected_chains = {"A", "G", "L", "M"}
    if set(summary["chains"]) != expected_chains:
        raise SystemExit(f"{cif} chains are {summary['chains']}, expected {sorted(expected_chains)}")
    if int(summary["protein_residue_count"]) <= 0:
        raise SystemExit(f"{cif} has no KRAS protein residues on chain A")
    if int(summary["ligand_atom_count"]) <= 0:
        raise SystemExit(f"{cif} has no LIG1 atoms")
    if int(summary["gnp_atom_count"]) <= 0:
        raise SystemExit(f"{cif} has no GNP atoms")
    if int(summary["mg_atom_count"]) != 1:
        raise SystemExit(f"{cif} should contain exactly one MG atom, found {summary['mg_atom_count']}")


def prepare_job_inputs(
    cif: Path,
    input_dir: Path,
    *,
    ligand_smiles: str,
    ligand_charge: int,
    gnp_charge: int,
    charge_method: str,
    pdb_prep_src: Path,
    force: bool,
) -> dict[str, Any]:
    ensure_pdb_prep_on_path(pdb_prep_src)
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
        from pdb_prep import extract_ligand
    except ModuleNotFoundError as exc:
        raise SystemExit("KRAS Boltz scaffold requires pdbfixer, openmm, and pdb_prep; run inside the md mamba environment.") from exc

    input_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = input_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    complex_pdb = input_dir / "complex.pdb"
    ligand_sdf = input_dir / "ligand.sdf"
    ligand_mol2 = input_dir / "ligand.mol2"
    ligand_frcmod = input_dir / "ligand.frcmod"
    gnp_sdf = input_dir / "gnp.sdf"
    gnp_mol2 = input_dir / "gnp.mol2"
    gnp_frcmod = input_dir / "gnp.frcmod"
    mg_pdb = input_dir / "mg.pdb"

    if force or not complex_pdb.exists():
        fixer = PDBFixer(filename=str(cif))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0, forcefield=None)
        with complex_pdb.open("w", encoding="utf-8") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)

    if force or not ligand_sdf.exists():
        if not ligand_smiles:
            raise SystemExit(f"Boltz manifest did not provide a ligand SMILES for {cif}")
        extract_ligand(
            cif,
            selectors=[LIGAND_SELECTOR],
            output_path=ligand_sdf,
            force_write=True,
            smiles=ligand_smiles,
            add_hydrogens=True,
        )
    if force or not ligand_mol2.exists() or not ligand_frcmod.exists():
        run_antechamber(ligand_sdf, ligand_mol2, ligand_charge, "LIG", charge_method, logs_dir / "antechamber_ligand.log", cwd=input_dir)
        normalize_mol2_charge(ligand_mol2, ligand_charge, preferred_atom_names=set())
        run_parmchk2(ligand_mol2, ligand_frcmod, logs_dir / "parmchk2_ligand.log", cwd=input_dir)
        cleanup_antechamber_scratch(input_dir)

    if force or not gnp_sdf.exists():
        extract_ligand(
            cif,
            selectors=[GNP_SELECTOR],
            output_path=gnp_sdf,
            force_write=True,
            ccd_code="GNP",
            add_hydrogens=True,
        )
    if force or not gnp_mol2.exists() or not gnp_frcmod.exists():
        run_antechamber(gnp_sdf, gnp_mol2, gnp_charge, "GNP", charge_method, logs_dir / "antechamber_gnp.log", cwd=input_dir)
        normalize_mol2_charge(
            gnp_mol2,
            gnp_charge,
            preferred_atom_names={"O1G", "O2G", "O3G", "O1B", "O2B", "O1A", "O2A"},
        )
        run_parmchk2(gnp_mol2, gnp_frcmod, logs_dir / "parmchk2_gnp.log", cwd=input_dir)
        cleanup_antechamber_scratch(input_dir)

    if force or not mg_pdb.exists():
        extract_pdb_residue(complex_pdb, mg_pdb, selector=MG_SELECTOR)

    ligand_charge_sum = mol2_total_charge(ligand_mol2)
    gnp_charge_sum = mol2_total_charge(gnp_mol2)
    charge_audit = {
        "ligand_charge_sum": ligand_charge_sum,
        "gnp_charge_sum": gnp_charge_sum,
        "ligand_charge_delta": None if ligand_charge_sum is None else ligand_charge_sum - ligand_charge,
        "gnp_charge_delta": None if gnp_charge_sum is None else gnp_charge_sum - gnp_charge,
    }
    if charge_audit["ligand_charge_delta"] is not None and abs(charge_audit["ligand_charge_delta"]) > 0.02:
        raise SystemExit(f"Ligand mol2 charge audit failed for {ligand_mol2}: {charge_audit}")
    if charge_audit["gnp_charge_delta"] is not None and abs(charge_audit["gnp_charge_delta"]) > 0.02:
        raise SystemExit(f"GNP mol2 charge audit failed for {gnp_mol2}: {charge_audit}")

    summary = {
        "status": "prepared",
        "source_cif": str(cif),
        "complex_pdb": str(complex_pdb),
        "ligand_sdf": str(ligand_sdf),
        "ligand_mol2": str(ligand_mol2),
        "ligand_frcmod": str(ligand_frcmod),
        "ligand_smiles": ligand_smiles,
        "ligand_charge_method": charge_method,
        "ligand_charge": ligand_charge,
        "gnp_sdf": str(gnp_sdf),
        "gnp_mol2": str(gnp_mol2),
        "gnp_frcmod": str(gnp_frcmod),
        "gnp_charge_method": charge_method,
        "gnp_charge": gnp_charge,
        "mg_pdb": str(mg_pdb),
        "mg_charge": 2,
        "cofactor_residue_count": 2,
        **charge_audit,
    }
    write_json_atomic(input_dir / "prep_summary.json", summary)
    return summary


def prepare_job_inputs_with_preparam_ligand(
    cif: Path,
    input_dir: Path,
    *,
    topology: dict[str, Any],
    ligand_charge: int,
    gnp_charge: int,
    charge_method: str,
    pdb_prep_src: Path,
    force: bool,
) -> dict[str, Any]:
    ensure_pdb_prep_on_path(pdb_prep_src)
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
        from pdb_prep import extract_ligand
    except ModuleNotFoundError as exc:
        raise SystemExit("KRAS Boltz scaffold requires pdbfixer, openmm, and pdb_prep; run inside the md mamba environment.") from exc

    input_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = input_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    complex_pdb = input_dir / "complex.pdb"
    ligand_mol2 = input_dir / "ligand.mol2"
    ligand_frcmod = input_dir / "ligand.frcmod"
    ligand_file_source = Path(str(topology["ligand_file"]))
    ligand_file = input_dir / ("ligand" + ligand_file_source.suffix.lower())
    gnp_sdf = input_dir / "gnp.sdf"
    gnp_mol2 = input_dir / "gnp.mol2"
    gnp_frcmod = input_dir / "gnp.frcmod"
    mg_pdb = input_dir / "mg.pdb"

    if force or not complex_pdb.exists():
        fixer = PDBFixer(filename=str(cif))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0, forcefield=None)
        with complex_pdb.open("w", encoding="utf-8") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)

    if force or not ligand_file.exists():
        shutil.copy2(ligand_file_source, ligand_file)
    if force or not ligand_mol2.exists():
        shutil.copy2(Path(str(topology["ligand_mol2"])), ligand_mol2)
    if force or not ligand_frcmod.exists():
        shutil.copy2(Path(str(topology["ligand_frcmod"])), ligand_frcmod)
    for lib in split_semicolon_paths(str(topology.get("ligand_lib") or "")):
        src = Path(lib)
        dst = input_dir / src.name
        if force or not dst.exists():
            shutil.copy2(src, dst)

    if force or not gnp_sdf.exists():
        extract_ligand(
            cif,
            selectors=[GNP_SELECTOR],
            output_path=gnp_sdf,
            force_write=True,
            ccd_code="GNP",
            add_hydrogens=True,
        )
    if force or not gnp_mol2.exists() or not gnp_frcmod.exists():
        run_antechamber(gnp_sdf, gnp_mol2, gnp_charge, "GNP", charge_method, logs_dir / "antechamber_gnp.log", cwd=input_dir)
        normalize_mol2_charge(
            gnp_mol2,
            gnp_charge,
            preferred_atom_names={"O1G", "O2G", "O3G", "O1B", "O2B", "O1A", "O2A"},
        )
        run_parmchk2(gnp_mol2, gnp_frcmod, logs_dir / "parmchk2_gnp.log", cwd=input_dir)
        cleanup_antechamber_scratch(input_dir)

    if force or not mg_pdb.exists():
        extract_pdb_residue(complex_pdb, mg_pdb, selector=MG_SELECTOR)

    ligand_charge_sum = mol2_total_charge(ligand_mol2)
    gnp_charge_sum = mol2_total_charge(gnp_mol2)
    charge_audit = {
        "ligand_charge_sum": ligand_charge_sum,
        "gnp_charge_sum": gnp_charge_sum,
        "ligand_charge_delta": None if ligand_charge_sum is None else ligand_charge_sum - ligand_charge,
        "gnp_charge_delta": None if gnp_charge_sum is None else gnp_charge_sum - gnp_charge,
    }
    if charge_audit["ligand_charge_delta"] is not None and abs(charge_audit["ligand_charge_delta"]) > 0.2:
        raise SystemExit(f"Ligand mol2 charge audit failed for {ligand_mol2}: {charge_audit}")
    if charge_audit["gnp_charge_delta"] is not None and abs(charge_audit["gnp_charge_delta"]) > 0.02:
        raise SystemExit(f"GNP mol2 charge audit failed for {gnp_mol2}: {charge_audit}")

    summary = {
        "status": "prepared",
        "topology_kind": "preparam",
        "source_cif": str(cif),
        "complex_pdb": str(complex_pdb),
        "ligand_file": str(ligand_file),
        "ligand_mol2": str(ligand_mol2),
        "ligand_frcmod": str(ligand_frcmod),
        "ligand_charge": ligand_charge,
        "gnp_sdf": str(gnp_sdf),
        "gnp_mol2": str(gnp_mol2),
        "gnp_frcmod": str(gnp_frcmod),
        "gnp_charge_method": charge_method,
        "gnp_charge": gnp_charge,
        "mg_pdb": str(mg_pdb),
        "mg_charge": 2,
        "cofactor_residue_count": 2,
        **charge_audit,
    }
    write_json_atomic(input_dir / "prep_summary.json", summary)
    return summary


def ensure_pdb_prep_on_path(pdb_prep_src: Path) -> None:
    if str(pdb_prep_src) not in sys.path:
        sys.path.insert(0, str(pdb_prep_src))


def run_antechamber(input_sdf: Path, output_mol2: Path, charge: int, resname: str, charge_method: str, log: Path, *, cwd: Path) -> None:
    run(
        [
            "antechamber",
            "-i",
            str(input_sdf),
            "-fi",
            "sdf",
            "-o",
            str(output_mol2),
            "-fo",
            "mol2",
            "-at",
            "gaff2",
            "-c",
            charge_method,
            "-nc",
            str(charge),
            "-rn",
            resname,
        ],
        log,
        cwd=cwd,
    )


def run_parmchk2(input_mol2: Path, output_frcmod: Path, log: Path, *, cwd: Path) -> None:
    run(["parmchk2", "-i", str(input_mol2), "-f", "mol2", "-o", str(output_frcmod), "-s", "gaff2"], log, cwd=cwd)


def extract_pdb_residue(source_pdb: Path, output_pdb: Path, *, selector: str) -> None:
    resname, chain, resseq = selector_parts(selector)
    lines: list[str] = []
    for raw in source_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.startswith(("ATOM  ", "HETATM")):
            continue
        if raw[17:20].strip() == resname and raw[21].strip() == chain and raw[22:26].strip() == resseq:
            lines.append(raw.rstrip() + "\n")
    if not lines:
        raise SystemExit(f"Selector {selector!r} did not match {source_pdb}")
    write_text_atomic(output_pdb, "".join(lines) + "TER\nEND\n")


def selector_parts(selector: str) -> tuple[str, str, str]:
    parts = selector.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected selector RESNAME:CHAIN:RESSEQ, got {selector!r}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def normalize_mol2_charge(path: Path, target_charge: int, *, preferred_atom_names: set[str]) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atom_indices: list[int] = []
    charges: dict[int, float] = {}
    in_atoms = False
    for idx, line in enumerate(lines):
        if line.upper().startswith("@<TRIPOS>"):
            in_atoms = line.upper() == "@<TRIPOS>ATOM"
            continue
        if not in_atoms or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        charges[idx] = float(fields[-1])
        if fields[1] in preferred_atom_names:
            atom_indices.append(idx)
    if not charges:
        raise SystemExit(f"No mol2 atom charges found in {path}")
    if not atom_indices:
        atom_indices = list(charges)
    current = sum(charges.values())
    delta = (float(target_charge) - current) / len(atom_indices)
    for idx in atom_indices:
        charges[idx] += delta
        raw = lines[idx]
        lines[idx] = f"{raw[:69]}{charges[idx]:10.6f}" if len(raw) >= 69 else replace_last_field(raw, f"{charges[idx]:.6f}")
    write_text_atomic(path, "\n".join(lines) + "\n")


def replace_last_field(line: str, value: str) -> str:
    fields = line.split()
    fields[-1] = value
    return " ".join(fields)


def run(command: list[str], log: Path, *, cwd: Path) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError(f"Required executable not found on PATH: {command[0]}")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    with log.open("w", encoding="utf-8") as handle:
        handle.write("# command: " + " ".join(shlex.quote(str(part)) for part in command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"\n# returncode: {result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}; see {log}")


def cleanup_antechamber_scratch(directory: Path) -> None:
    for pattern in ("ANTECHAMBER_*", "ATOMTYPE.INF", "sqm.in", "sqm.out", "sqm.pdb"):
        for path in directory.glob(pattern):
            path.unlink(missing_ok=True)


def write_dataset_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DATASET_FIELDS})


def run_script_text(run_dir: Path, protocol: str, job_ids: list[str]) -> str:
    jobs = " ".join(job_ids)
    return f"""#!/usr/bin/env bash
set -euo pipefail

RUN_DIR={run_dir}
PROTOCOL=${{PROTOCOL:-{protocol}}}
PYTHON=${{PYTHON:-python}}
export MAMBA_ENV=${{MAMBA_ENV:-md}}
export GMXRC=${{GMXRC:-{DEFAULT_GMXRC}}}
export GMX_BIN=${{GMX_BIN:-gmx_mpi}}
export GPU_ID=${{GPU_ID:-0}}
export NTOMP=${{NTOMP:-8}}
export MMPBSA_NP=${{MMPBSA_NP:-16}}
export PYTHONUNBUFFERED=1

for job in {jobs}; do
  date -Is
  echo "running $job on GPU_ID=$GPU_ID"
  "$PYTHON" -m mmpbsa ligand run "$RUN_DIR" --job-id "$job" --protocol "$PROTOCOL" --resume
done
"""


def gpu_worker_script_text(run_dir: Path, protocol: str, job_ids: list[str]) -> str:
    jobs = " ".join(shlex.quote(job) for job in job_ids)
    return f"""#!/usr/bin/env bash
set -euo pipefail

RUN_DIR={shlex.quote(str(run_dir))}
PROTOCOL="${{PROTOCOL:-{shlex.quote(protocol)}}}"
PYTHON="${{PYTHON:-python}}"
export MAMBA_ENV="${{MAMBA_ENV:-md}}"
export GMXRC="${{GMXRC:-{DEFAULT_GMXRC}}}"
export GMX_BIN="${{GMX_BIN:-gmx_mpi}}"
export NTOMP="${{NTOMP:-8}}"
export MMPBSA_NP="${{MMPBSA_NP:-16}}"
export PYTHONUNBUFFERED=1

IFS=', ' read -r -a GPU_LIST <<< "${{GPUS:-4,5,6,7}}"
JOBS=({jobs})
if [[ "${{#GPU_LIST[@]}}" -eq 0 ]]; then
  echo "No GPUs configured; set GPUS=4,5,6,7" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR"/gpu*.pid "$RUN_DIR"/gpu*.log

pids=()
for worker_index in "${{!GPU_LIST[@]}}"; do
  gpu="${{GPU_LIST[$worker_index]}}"
  (
    set -euo pipefail
    for ((job_index=worker_index; job_index<${{#JOBS[@]}}; job_index+=${{#GPU_LIST[@]}})); do
      job="${{JOBS[$job_index]}}"
      date -Is
      echo "running $job on GPU_ID=$gpu"
      GPU_ID="$gpu" "$PYTHON" -m mmpbsa ligand run "$RUN_DIR" \
        --job-id "$job" --protocol "$PROTOCOL" --resume
    done
  ) > "$RUN_DIR/gpu${{gpu}}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "$pid" > "$RUN_DIR/gpu${{gpu}}.pid"
  echo "gpu${{gpu}} pid=$pid log=$RUN_DIR/gpu${{gpu}}.log"
done

failed=0
for pid in "${{pids[@]}}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
"""


def strict_report_profile(profile_name: str) -> dict[str, Any]:
    try:
        return STRICT_REPORT_PROFILES[profile_name]
    except KeyError as exc:
        options = ", ".join(sorted(STRICT_REPORT_PROFILES))
        raise SystemExit(f"Unknown strict report profile {profile_name!r}; choose one of: {options}") from exc


def strict_report_output_dirname(profile_name: str) -> str:
    return str(strict_report_profile(profile_name)["output_dirname"])


def load_run_manifest(run_root: Path) -> tuple[Path, dict[str, Any]]:
    for name in RUN_MANIFEST_CANDIDATES:
        path = run_root / name
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8"))
    candidates = sorted(run_root.glob("*_6wgn_gnp_mg_manifest.json"))
    if len(candidates) == 1:
        path = candidates[0]
        return path, json.loads(path.read_text(encoding="utf-8"))
    if candidates:
        names = ", ".join(path.name for path in candidates)
        raise SystemExit(f"Multiple Boltz run manifests found in {run_root}: {names}")
    expected = ", ".join(RUN_MANIFEST_CANDIDATES)
    raise SystemExit(f"Missing Boltz run manifest in {run_root}; expected one of: {expected}")


def write_strict_3x5ns_report(
    run_dir: Path,
    output_dir: Path,
    *,
    profile_name: str = "3x5ns",
    expected_jobs: int = 10,
    expected_replicas: int | None = None,
    expected_frames: int | None = None,
    expected_frames_per_replica: int | None = None,
) -> dict[str, Any]:
    profile = strict_report_profile(profile_name)
    expected_replicas = int(expected_replicas if expected_replicas is not None else profile["expected_replicas"])
    expected_frames = int(expected_frames if expected_frames is not None else profile["expected_frames"])
    expected_frames_per_replica = int(
        expected_frames_per_replica if expected_frames_per_replica is not None else profile["expected_frames_per_replica"]
    )
    expected_startframe = int(profile["expected_startframe"])
    expected_mmpbsa_start_ns = float(profile["expected_mmpbsa_start_ns"])
    expected_production_ns = float(profile["expected_production_ns"])
    run_root = run_dir.resolve()
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, run_manifest = load_run_manifest(run_root)
    selected_jobs = run_manifest.get("jobs", [])
    if len(selected_jobs) != expected_jobs:
        raise SystemExit(f"Expected {expected_jobs} selected jobs in {manifest_path}, found {len(selected_jobs)}")

    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for job in selected_jobs:
        job_id = str(job["job_id"])
        job_dir = run_root / job_id
        summary_path = job_dir / "result" / "summary.json"
        job_manifest_path = job_dir / "manifest.json"
        audit_path = job_dir / "analysis" / "mmpbsa" / "audit.json"
        qc_path = job_dir / "analysis" / "qc" / "summary.json"
        summary = read_json_or_empty(summary_path)
        job_manifest = read_json_or_empty(job_manifest_path)
        audit = read_json_or_empty(audit_path)
        qc = read_json_or_empty(qc_path)

        frame_settings = job_manifest.get("frame_settings", {})
        protocol = job_manifest.get("profile", {}).get("protocol", {})
        replica_frames = [safe_number(rep.get("frames")) for rep in audit.get("replicas", [])]
        issues = list(audit.get("issues", []))
        strict_issues: list[str] = []
        if not summary_path.exists():
            strict_issues.append("missing result/summary.json")
        if not job_manifest_path.exists():
            strict_issues.append("missing manifest.json")
        if not audit_path.exists():
            strict_issues.append("missing analysis/mmpbsa/audit.json")
        if int_or_none(summary.get("replica_count")) != expected_replicas:
            strict_issues.append(f"replica_count={summary.get('replica_count')} expected {expected_replicas}")
        if int_or_none(summary.get("frames_per_replica")) != expected_frames_per_replica:
            strict_issues.append(
                f"frames_per_replica={summary.get('frames_per_replica')} expected {expected_frames_per_replica}"
            )
        if int_or_none(summary.get("mmpbsa_frames")) != expected_frames:
            strict_issues.append(f"mmpbsa_frames={summary.get('mmpbsa_frames')} expected {expected_frames}")
        if int_or_none(frame_settings.get("startframe")) != expected_startframe:
            strict_issues.append(f"startframe={frame_settings.get('startframe')} expected {expected_startframe}")
        if float_or_none(protocol.get("mmpbsa_start_ns")) != expected_mmpbsa_start_ns:
            strict_issues.append(f"mmpbsa_start_ns={protocol.get('mmpbsa_start_ns')} expected {expected_mmpbsa_start_ns}")
        if float_or_none(protocol.get("production_ns")) != expected_production_ns:
            strict_issues.append(f"production_ns={protocol.get('production_ns')} expected {expected_production_ns}")
        if int_or_none(audit.get("frames")) != expected_frames:
            strict_issues.append(f"audit frames={audit.get('frames')} expected {expected_frames}")
        if replica_frames and any(int_or_none(value) != expected_frames_per_replica for value in replica_frames):
            strict_issues.append(f"replica frames={replica_frames} expected all {expected_frames_per_replica}")
        if summary.get("status") != "valid":
            strict_issues.append(f"summary status={summary.get('status')}")
        if summary.get("trajectory_qc_status") != "valid":
            strict_issues.append(f"trajectory_qc_status={summary.get('trajectory_qc_status')}")
        if summary.get("mmpbsa_qc_status") != "valid":
            strict_issues.append(f"mmpbsa_qc_status={summary.get('mmpbsa_qc_status')}")

        strict_pass = not strict_issues
        if not strict_pass:
            failures.append(f"{job_id}: " + "; ".join(strict_issues))
        row = {
            "selection_index": job.get("index", ""),
            "job_id": job_id,
            "boltz_rank": job.get("rank", ""),
            "model_id": summary.get("model_id", ""),
            "strict_profile": profile_name,
            "strict_mmpbsa": strict_pass,
            "strict_3_5ns_mmpbsa": strict_pass,
            "status": summary.get("status", ""),
            "trajectory_qc_status": summary.get("trajectory_qc_status", qc.get("status", "")),
            "mmpbsa_qc_status": summary.get("mmpbsa_qc_status", audit.get("status", "")),
            "replica_count": summary.get("replica_count", ""),
            "frames_per_replica": summary.get("frames_per_replica", ""),
            "mmpbsa_frames": summary.get("mmpbsa_frames", audit.get("frames", "")),
            "explicit_water_count": summary.get("explicit_water_count", ""),
            "ligand_charge": summary.get("ligand_charge", ""),
            "charge_method": summary.get("charge_method", ""),
            "GB_delta_total_kJ_mol": summary.get("GB_delta_total_kJ_mol", ""),
            "GB_delta_total_kJ_mol_replica_sd": summary.get("GB_delta_total_kJ_mol_replica_sd", ""),
            "GB_delta_total_kJ_mol_replica_sem": summary.get("GB_delta_total_kJ_mol_replica_sem", ""),
            "GB_dMM_kJ_mol": summary.get("GB_dMM_kJ_mol", ""),
            "GB_dMM_kJ_mol_replica_sd": summary.get("GB_dMM_kJ_mol_replica_sd", ""),
            "GB_dMM_kJ_mol_replica_sem": summary.get("GB_dMM_kJ_mol_replica_sem", ""),
            "PB_delta_total_kJ_mol": summary.get("PB_delta_total_kJ_mol", ""),
            "PB_delta_total_kJ_mol_replica_sd": summary.get("PB_delta_total_kJ_mol_replica_sd", ""),
            "PB_delta_total_kJ_mol_replica_sem": summary.get("PB_delta_total_kJ_mol_replica_sem", ""),
            "PB_dMM_kJ_mol": summary.get("PB_dMM_kJ_mol", ""),
            "PB_dMM_kJ_mol_replica_sd": summary.get("PB_dMM_kJ_mol_replica_sd", ""),
            "PB_dMM_kJ_mol_replica_sem": summary.get("PB_dMM_kJ_mol_replica_sem", ""),
            "strict_issues": "; ".join(strict_issues),
        }
        rows.append(row)
        qc_rows.append(
            {
                "job_id": job_id,
                "strict_profile": profile_name,
                "strict_mmpbsa": strict_pass,
                "strict_3_5ns_mmpbsa": strict_pass,
                "summary_status": summary.get("status", ""),
                "trajectory_qc_status": summary.get("trajectory_qc_status", qc.get("status", "")),
                "mmpbsa_qc_status": summary.get("mmpbsa_qc_status", audit.get("status", "")),
                "replica_count": summary.get("replica_count", ""),
                "frames_per_replica": summary.get("frames_per_replica", ""),
                "mmpbsa_frames": summary.get("mmpbsa_frames", audit.get("frames", "")),
                "replica_frames": ",".join(format_number(value) for value in replica_frames),
                "audit_issue_count": len(issues),
                "strict_issues": "; ".join(strict_issues),
            }
        )

    ranking_rows = sorted(rows, key=lambda row: sort_number(row.get("GB_delta_total_kJ_mol")))
    for rank, row in enumerate(ranking_rows, start=1):
        row["GB_primary_rank"] = rank
    write_csv_atomic(output / "ranking_strict_3x5ns_10prod.csv", ranking_rows)
    write_csv_atomic(output / "qc_summary.csv", qc_rows)
    report = {
        "schema_version": f"mmpbsa.validation.kras_6wgn_boltz.strict_{profile_name}_report.v1",
        "generated_at": utc_now(),
        "run_dir": str(run_root),
        "output_dir": str(output),
        "source_manifest": str(manifest_path),
        "profile": profile_name,
        "profile_display_name": profile["display_name"],
        "expected_jobs": expected_jobs,
        "production_jobs": len(rows),
        "smoke_jobs_included": 0,
        "md_protocol": profile["md_protocol"],
        "mmpbsa_window": profile["mmpbsa_window"],
        "expected_replicas": expected_replicas,
        "expected_frames_per_job": expected_frames,
        "expected_frames_per_replica": expected_frames_per_replica,
        "strict_pass": len(failures) == 0,
        "failures": failures,
        "primary_score": "GB_delta_total_kJ_mol",
        "secondary_score": "PB_delta_total_kJ_mol",
    }
    write_json_atomic(output / "summary.json", report)
    write_text_atomic(output / "report.md", strict_report_markdown(report, ranking_rows, qc_rows))
    if failures:
        raise SystemExit("Strict report failed:\n" + "\n".join(failures))
    return report


def read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sort_number(value: Any) -> float:
    number = safe_number(value)
    return number if number is not None else float("inf")


def format_number(value: Any, digits: int = 2) -> str:
    number = safe_number(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def strict_report_markdown(report: dict[str, Any], ranking_rows: list[dict[str, Any]], qc_rows: list[dict[str, Any]]) -> str:
    profile_display = str(report.get("profile_display_name") or report.get("profile") or "").strip()
    title_suffix = f" {profile_display}" if profile_display else ""
    lines = [
        f"# KRAS 6WGN/GNP-Mg Boltz Strict{title_suffix} Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Run directory: `{report['run_dir']}`",
        f"- Production jobs: {report['production_jobs']}",
        f"- Smoke jobs included in ranking: {report['smoke_jobs_included']}",
        f"- MD protocol: {report['md_protocol']}",
        f"- MMPBSA scoring window: {report['mmpbsa_window']}",
        f"- Expected frames/job: {frame_count_label(report.get('expected_replicas'), report.get('expected_frames_per_replica'), report.get('expected_frames_per_job'))}",
        f"- Strict pass: {report['strict_pass']}",
        "",
        "## Ranking",
        "",
        "Primary score is `GB_delta_total_kJ_mol`; more negative is ranked better. PB is a secondary check.",
        "",
        "| GB rank | Boltz rank | Job | GB mean kJ/mol | GB replica SD | PB mean kJ/mol | PB replica SD | Frames | QC |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranking_rows:
        qc = "pass" if row.get("strict_mmpbsa", row.get("strict_3_5ns_mmpbsa")) else "fail"
        lines.append(
            "| {rank} | {boltz} | `{job}` | {gb} | {gb_sd} | {pb} | {pb_sd} | {frames} | {qc} |".format(
                rank=row.get("GB_primary_rank", ""),
                boltz=row.get("boltz_rank", ""),
                job=row.get("job_id", ""),
                gb=format_number(row.get("GB_delta_total_kJ_mol")),
                gb_sd=format_number(row.get("GB_delta_total_kJ_mol_replica_sd")),
                pb=format_number(row.get("PB_delta_total_kJ_mol")),
                pb_sd=format_number(row.get("PB_delta_total_kJ_mol_replica_sd")),
                frames=frame_count_label(row.get("replica_count"), row.get("frames_per_replica"), row.get("mmpbsa_frames")),
                qc=qc,
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is a Boltz pose-rescoring scaffold, not a public affinity benchmark: these cyclic ligands do not yet have matched experimental KD/IC50 values in this report.",
            "- The receptor state is KRAS(G12D)-GNP-Mg based on 6WGN, so the scores should not be mixed with GDP-only or no-Mg reports.",
            f"- The ranking excludes smoke outputs and excludes any full-window MMPBSA summaries by requiring exactly {frame_count_label(report.get('expected_replicas'), report.get('expected_frames_per_replica'), report.get('expected_frames_per_job'))} strict {report.get('mmpbsa_window')} frames.",
            "- Entropy is disabled; GB is the primary ranking score and PB is a secondary diagnostic.",
            "",
            "## QC",
            "",
            "| Job | Strict | MMPBSA frames | Replica frames | Audit issues | Strict issues |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for row in qc_rows:
        lines.append(
            "| `{job}` | {strict} | {frames} | {repframes} | {audit} | {issues} |".format(
                job=row.get("job_id", ""),
                strict=row.get("strict_mmpbsa", row.get("strict_3_5ns_mmpbsa", "")),
                frames=frame_count_label(row.get("replica_count"), row.get("frames_per_replica"), row.get("mmpbsa_frames")),
                repframes=row.get("replica_frames", ""),
                audit=row.get("audit_issue_count", ""),
                issues=row.get("strict_issues", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def frame_count_label(replicas: Any, frames_per_replica: Any, total_frames: Any) -> str:
    rep = int_or_none(replicas)
    per_rep = int_or_none(frames_per_replica)
    total = int_or_none(total_frames)
    if rep and rep > 1 and per_rep:
        expected = rep * per_rep
        if total is None:
            return f"{rep} x {per_rep}"
        if expected == total:
            return f"{rep} x {per_rep} = {total}"
        return f"{rep} x {per_rep} != {total}"
    return str(total_frames or "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build KRAS 6WGN/GNP-Mg Boltz ligand MMPBSA jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_parser = subparsers.add_parser("make-jobs", help="Create job directories from Boltz KRAS top CIF files.")
    make_parser.add_argument("run_dir", type=Path)
    make_parser.add_argument("--resources-dir", type=Path, default=DEFAULT_RESOURCES)
    make_parser.add_argument("--selection-tier", default="primary")
    make_parser.add_argument("--limit", type=int, default=10)
    make_parser.add_argument("--charge-method", default="gas", choices=["gas", "bcc"])
    make_parser.add_argument("--pdb-prep-src", type=Path, default=DEFAULT_PDB_PREP_SRC)
    make_parser.add_argument("--skip-prepare-inputs", action="store_true", help="Only write job JSON/dataset files; do not generate SDF/MOL2/FRCMOD inputs.")
    make_parser.add_argument("--force", action="store_true")

    make_cifs_parser = subparsers.add_parser("make-jobs-from-cifs", help="Create job directories from a CIF directory and SMILES manifest.")
    make_cifs_parser.add_argument("run_dir", type=Path)
    make_cifs_parser.add_argument("--cif-dir", type=Path, default=DEFAULT_BOLTZ2_CIF_DIR)
    make_cifs_parser.add_argument("--manifest", type=Path, default=DEFAULT_BOLTZ2_MANIFEST)
    make_cifs_parser.add_argument("--dataset-label", default="boltz2")
    make_cifs_parser.add_argument("--limit", type=int, default=None)
    make_cifs_parser.add_argument("--charge-method", default="gas", choices=["gas", "bcc"])
    make_cifs_parser.add_argument("--pdb-prep-src", type=Path, default=DEFAULT_PDB_PREP_SRC)
    make_cifs_parser.add_argument("--skip-prepare-inputs", action="store_true", help="Only write job JSON/dataset files; do not generate SDF/MOL2/FRCMOD inputs.")
    make_cifs_parser.add_argument("--force", action="store_true")

    preflight_iptm_parser = subparsers.add_parser("preflight-iptm-manifest", help="Validate a Boltz2 iPTM manifest before staging jobs.")
    preflight_iptm_parser.add_argument("--manifest", type=Path, default=DEFAULT_BOLTZ2_IPTM_MANIFEST)
    preflight_iptm_parser.add_argument("--set", dest="set_name", default="primary")
    preflight_iptm_parser.add_argument("--limit", type=int, default=10)
    preflight_iptm_parser.add_argument("--search-root", type=Path, action="append", default=[])
    preflight_iptm_parser.add_argument("--local-cif-dir", type=Path, default=DEFAULT_BOLTZ2_CIF_DIR)
    preflight_iptm_parser.add_argument("--smiles-manifest", type=Path, default=DEFAULT_BOLTZ_SMILES_MANIFEST)

    make_iptm_parser = subparsers.add_parser("make-jobs-from-iptm-manifest", help="Create jobs from a Boltz2 iPTM manifest with CIF and ligand topology paths.")
    make_iptm_parser.add_argument("run_dir", type=Path)
    make_iptm_parser.add_argument("--manifest", type=Path, default=DEFAULT_BOLTZ2_IPTM_MANIFEST)
    make_iptm_parser.add_argument("--dataset-label", default="boltz2")
    make_iptm_parser.add_argument("--set", dest="set_name", default="primary")
    make_iptm_parser.add_argument("--limit", type=int, default=10)
    make_iptm_parser.add_argument("--search-root", type=Path, action="append", default=[])
    make_iptm_parser.add_argument("--local-cif-dir", type=Path, default=DEFAULT_BOLTZ2_CIF_DIR)
    make_iptm_parser.add_argument("--smiles-manifest", type=Path, default=DEFAULT_BOLTZ_SMILES_MANIFEST)
    make_iptm_parser.add_argument("--charge-method", default="gas", choices=["gas", "bcc"])
    make_iptm_parser.add_argument("--pdb-prep-src", type=Path, default=DEFAULT_PDB_PREP_SRC)
    make_iptm_parser.add_argument("--skip-prepare-inputs", action="store_true", help="Only write job JSON/dataset files; do not generate PDB/MOL2/FRCMOD inputs.")
    make_iptm_parser.add_argument("--force", action="store_true")

    summary_parser = subparsers.add_parser("summarize-cifs", help="Print CIF summaries for selected Boltz rows.")
    summary_parser.add_argument("--resources-dir", type=Path, default=DEFAULT_RESOURCES)
    summary_parser.add_argument("--selection-tier", default="primary")
    summary_parser.add_argument("--limit", type=int, default=10)

    report_parser = subparsers.add_parser("report-strict", help="Generate a strict production-only KRAS Boltz report.")
    report_parser.add_argument("run_dir", type=Path)
    report_parser.add_argument("--profile", default="3x5ns", choices=sorted(STRICT_REPORT_PROFILES))
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: RUN_DIR/reports/final_strict_<profile>_10prod",
    )
    report_parser.add_argument("--expected-jobs", type=int, default=10)
    report_parser.add_argument("--expected-replicas", type=int, default=None)
    report_parser.add_argument("--expected-frames", type=int, default=None)
    report_parser.add_argument("--expected-frames-per-replica", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "make-jobs":
        report = make_boltz_jobs(
            args.resources_dir,
            args.run_dir,
            selection_tier=args.selection_tier,
            limit=args.limit,
            force=args.force,
            prepare_inputs=not args.skip_prepare_inputs,
            charge_method=args.charge_method,
            pdb_prep_src=args.pdb_prep_src,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "make-jobs-from-cifs":
        report = make_boltz_jobs_from_cifs(
            args.cif_dir,
            args.manifest,
            args.run_dir,
            dataset_label=args.dataset_label,
            limit=args.limit,
            force=args.force,
            prepare_inputs=not args.skip_prepare_inputs,
            charge_method=args.charge_method,
            pdb_prep_src=args.pdb_prep_src,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "preflight-iptm-manifest":
        roots = args.search_root or None
        report = preflight_iptm_manifest(
            args.manifest,
            set_name=args.set_name,
            limit=args.limit,
            search_roots=roots,
            local_cif_dir=args.local_cif_dir,
            smiles_manifest=args.smiles_manifest,
        )
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            return 1
    elif args.command == "make-jobs-from-iptm-manifest":
        roots = args.search_root or None
        report = make_boltz_jobs_from_iptm_manifest(
            args.manifest,
            args.run_dir,
            dataset_label=args.dataset_label,
            set_name=args.set_name,
            limit=args.limit,
            search_roots=roots,
            local_cif_dir=args.local_cif_dir,
            smiles_manifest=args.smiles_manifest,
            force=args.force,
            prepare_inputs=not args.skip_prepare_inputs,
            charge_method=args.charge_method,
            pdb_prep_src=args.pdb_prep_src,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "summarize-cifs":
        rows = load_boltz_manifest(args.resources_dir / "md_selected_manifest.csv", selection_tier=args.selection_tier, limit=args.limit)
        payload = []
        for row in rows:
            cif = find_representative_cif(row, args.resources_dir / "top10")
            payload.append({"rank": row["rank"], "representative_model": row.get("representative_model"), "cif": str(cif), **cif_summary(cif)})
        print(json.dumps(payload, indent=2))
    elif args.command == "report-strict":
        output_dir = args.output_dir or args.run_dir / "reports" / strict_report_output_dirname(args.profile)
        report = write_strict_3x5ns_report(
            args.run_dir,
            output_dir,
            profile_name=args.profile,
            expected_jobs=args.expected_jobs,
            expected_replicas=args.expected_replicas,
            expected_frames=args.expected_frames,
            expected_frames_per_replica=args.expected_frames_per_replica,
        )
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

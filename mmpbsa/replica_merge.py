from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import aggregate_replica_values, read_json, write_csv_atomic, write_json_atomic


def merge_peptide_replicas(output_job_dir: Path, source_job_dirs: list[Path], force: bool = False) -> dict[str, Any]:
    return merge_replicas(output_job_dir, source_job_dirs, force=force, label="peptide")


def merge_ligand_replicas(output_job_dir: Path, source_job_dirs: list[Path], force: bool = False) -> dict[str, Any]:
    return merge_replicas(output_job_dir, source_job_dirs, force=force, label="ligand")


def merge_replicas(output_job_dir: Path, source_job_dirs: list[Path], force: bool = False, label: str = "job") -> dict[str, Any]:
    if not source_job_dirs:
        raise SystemExit("At least one source job directory is required")
    output = output_job_dir.resolve()
    audit_path = output / "analysis" / "mmpbsa" / "audit.json"
    summary_path = output / "result" / "summary.json"
    if not force and (audit_path.exists() or summary_path.exists()):
        raise SystemExit(f"Merged outputs already exist under {output}; use --force to overwrite audit/summary files")

    source_records = [load_source_record(path.resolve()) for path in source_job_dirs]
    merged_replicas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in source_records:
        for replica in record["replicas"]:
            name = str(replica["replica"])
            if name in seen:
                raise SystemExit(f"Duplicate replica {name} while merging; use distinct replica indices")
            seen.add(name)
            merged = dict(replica)
            merged["source_job_dir"] = str(record["job_dir"])
            merged["source_job_id"] = record["job_id"]
            merged.setdefault("replica_index", replica_index(name))
            merged_replicas.append(merged)

    merged_replicas.sort(key=lambda item: int(item.get("replica_index") or replica_index(str(item["replica"]))))
    values = aggregate_replica_values([dict(item.get("values", {})) for item in merged_replicas])
    invalid = any(str(item.get("audit", {}).get("status")) != "valid" for item in merged_replicas)
    frames = sum(float(item.get("frames") or item.get("audit", {}).get("frames") or 0.0) for item in merged_replicas)
    replica_min_frames = replica_min_frame_floor(merged_replicas)
    indices = [int(item.get("replica_index") or replica_index(str(item["replica"]))) for item in merged_replicas]
    seeds = {str(item["replica"]): item.get("seed") for item in merged_replicas if item.get("seed") is not None}
    audit = {
        "status": "invalid" if invalid else "valid",
        "job_id": output.name,
        "frames": frames,
        "min_frames": replica_min_frames * len(merged_replicas),
        "replica_min_frames": replica_min_frames,
        "replica_count": len(merged_replicas),
        "replica_indices": indices,
        "replica_seeds": seeds,
        "issues": [issue | {"replica": item["replica"]} for item in merged_replicas for issue in item.get("audit", {}).get("issues", [])],
        "notes": [f"Merged from independently calculated {label} replica jobs."],
        "values": values,
        "replicas": merged_replicas,
        "merged_from": [record["job_id"] for record in source_records],
    }

    summary = merged_summary(output.name, source_records, audit, values)
    write_json_atomic(audit_path, audit)
    write_json_atomic(summary_path, summary)
    write_csv_atomic(output / "result" / "summary.csv", [summary])
    return {"output_job_dir": str(output), "replica_count": len(merged_replicas), "replica_indices": indices, "status": audit["status"]}


def load_source_record(job_dir: Path) -> dict[str, Any]:
    audit_path = job_dir / "analysis" / "mmpbsa" / "audit.json"
    if not audit_path.exists():
        raise SystemExit(f"Missing source audit: {audit_path}")
    audit = read_json(audit_path)
    replicas = audit.get("replicas")
    if not isinstance(replicas, list) or not replicas:
        raise SystemExit(f"Source audit has no per-replica records: {audit_path}")
    summary_path = job_dir / "result" / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    return {"job_dir": job_dir, "job_id": str(audit.get("job_id") or job_dir.name), "audit": audit, "summary": summary, "replicas": replicas}


def replica_index(name: str) -> int:
    match = re.fullmatch(r"rep(\d+)", name)
    if not match:
        raise SystemExit(f"Invalid replica name {name!r}; expected repNN")
    return int(match.group(1))


def replica_min_frame_floor(replicas: list[dict[str, Any]]) -> int:
    values = [
        int(replica.get("audit", {}).get("min_frames"))
        for replica in replicas
        if replica.get("audit", {}).get("min_frames") not in (None, "")
    ]
    if values:
        return min(values)
    return 0


def merged_summary(job_id: str, sources: list[dict[str, Any]], audit: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    first = sources[0]["summary"]
    source_statuses = [record["summary"].get("status") for record in sources if record["summary"]]
    status = "valid" if audit["status"] == "valid" and all(status in (None, "", "valid") for status in source_statuses) else "invalid"
    summary: dict[str, Any] = {
        "job_id": job_id,
        "name": first.get("name") or job_id,
        "pdb_id": first.get("pdb_id", ""),
        "model_id": first.get("model_id", ""),
        "source": first.get("source", ""),
        "status": status,
        "trajectory_qc_status": "merged",
        "mmpbsa_qc_status": audit["status"],
        "mmpbsa_frames": audit["frames"],
        "mmpbsa_frames_total": audit["frames"],
        "replica_count": audit["replica_count"],
        "replica_indices": audit["replica_indices"],
        "replica_seeds": audit["replica_seeds"],
        "replicas": [item["replica"] for item in audit["replicas"]],
        "merged_from": audit["merged_from"],
    }
    for key in [
        "deltaG_exp_kJ_mol",
        "paper_mm_pbsa_kJ_mol",
        "paper_dmm_pbsa_kJ_mol",
        "paper_vdw_kJ_mol",
        "ligand_resname",
        "ligand_charge",
        "ligand_param_mode",
        "charge_method",
        "ic50_nM",
        "kd_nM",
        "dielectric_source",
        "dielectric_class",
        "dielectric_epsilon",
        "explicit_water_count",
        "entropy_enabled",
        "entropy_method",
        "frames_per_replica",
        "mmpbsa_enabled",
    ]:
        if key in first:
            summary[key] = first[key]
    summary.update(values)
    return summary

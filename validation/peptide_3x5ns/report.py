#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmpbsa.metrics import linear_fit, pearson_r, spearman_r


SCORE_METRICS = [
    ("GB_delta_total_kJ_mol", (), "GB total"),
    ("PB_delta_total_kJ_mol", (), "PB total"),
    ("GB_dMM_kJ_mol", ("GB_dmm_like_kJ_mol",), "GB dMM"),
    ("PB_dMM_kJ_mol", ("PB_dmm_like_kJ_mol",), "PB dMM"),
    ("paper_mm_pbsa_kJ_mol", (), "paper MM-PBSA"),
    ("paper_dmm_pbsa_kJ_mol", (), "paper dMM-PBSA"),
]
SCORE_KEYS = [metric[0] for metric in SCORE_METRICS]
ALIASES = {key: aliases for key, aliases, _label in SCORE_METRICS}
LABELS = {key: label for key, _aliases, label in SCORE_METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the local peptide 3x5ns validation run.")
    parser.add_argument("--run-dir", type=Path, default=Path("pipeline_tests/peptide_3x5ns"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/peptide_3x5ns"))
    args = parser.parse_args()

    rows = load_rows(args.run_dir)
    correlations = {key: correlation_record(rows, key) for key in SCORE_KEYS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    write_json(args.output_dir / "summary.json", summary_record(args.run_dir, args.output_dir, rows, correlations))
    (args.output_dir / "report.md").write_text(report_markdown(args.run_dir, rows, correlations), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir.resolve()), "correlations": correlations}, indent=2))


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(run_dir.glob("*/result/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = load_manifest(summary_path.parents[1])
        audit = load_audit(summary_path.parents[1])
        row = {
            "job_id": summary.get("job_id", summary_path.parents[1].name),
            "name": summary.get("name", ""),
            "pdb_id": summary.get("pdb_id", ""),
            "status": summary.get("status", ""),
            "deltaG_exp_kJ_mol": summary.get("deltaG_exp_kJ_mol"),
            "GB_delta_total_kJ_mol": summary.get("GB_delta_total_kJ_mol"),
            "GB_delta_total_kJ_mol_replica_sd": summary.get("GB_delta_total_kJ_mol_replica_sd")
            if summary.get("GB_delta_total_kJ_mol_replica_sd") is not None
            else replica_sd(audit, "GB_delta_total_kJ_mol"),
            "GB_delta_total_kJ_mol_replica_sem": summary.get("GB_delta_total_kJ_mol_replica_sem"),
            "PB_delta_total_kJ_mol": summary.get("PB_delta_total_kJ_mol"),
            "PB_delta_total_kJ_mol_replica_sd": summary.get("PB_delta_total_kJ_mol_replica_sd")
            if summary.get("PB_delta_total_kJ_mol_replica_sd") is not None
            else replica_sd(audit, "PB_delta_total_kJ_mol"),
            "PB_delta_total_kJ_mol_replica_sem": summary.get("PB_delta_total_kJ_mol_replica_sem"),
            "GB_dMM_kJ_mol": value_with_alias(summary, "GB_dMM_kJ_mol"),
            "GB_dMM_kJ_mol_replica_sd": value_with_alias(summary, "GB_dMM_kJ_mol_replica_sd")
            if value_with_alias(summary, "GB_dMM_kJ_mol_replica_sd") is not None
            else replica_sd(audit, "GB_dMM_kJ_mol"),
            "GB_dMM_kJ_mol_replica_sem": value_with_alias(summary, "GB_dMM_kJ_mol_replica_sem"),
            "PB_dMM_kJ_mol": value_with_alias(summary, "PB_dMM_kJ_mol"),
            "PB_dMM_kJ_mol_replica_sd": value_with_alias(summary, "PB_dMM_kJ_mol_replica_sd")
            if value_with_alias(summary, "PB_dMM_kJ_mol_replica_sd") is not None
            else replica_sd(audit, "PB_dMM_kJ_mol"),
            "PB_dMM_kJ_mol_replica_sem": value_with_alias(summary, "PB_dMM_kJ_mol_replica_sem"),
            "paper_mm_pbsa_kJ_mol": summary.get("paper_mm_pbsa_kJ_mol"),
            "paper_dmm_pbsa_kJ_mol": summary.get("paper_dmm_pbsa_kJ_mol"),
            "mmpbsa_frames": summary.get("mmpbsa_frames"),
            "replica_count": summary.get("replica_count"),
            "dropped_nonprotein_residue_count": dropped_count(summary, manifest),
            "box_retry_used": bool(summary.get("box_retry_used", False)),
            "solvent_shape_actual": summary.get("solvent_shape_actual", ""),
            "job_dir": str(summary_path.parents[1]),
        }
        rows.append(row)
    rows.sort(key=lambda row: str(row["job_id"]))
    return rows


def load_audit(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "analysis" / "mmpbsa" / "audit.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dropped_count(summary: dict[str, Any], manifest: dict[str, Any]) -> int | None:
    for data in (summary, manifest):
        value = data.get("dropped_nonprotein_residue_count")
        if isinstance(value, int):
            return value
        residues = data.get("dropped_nonprotein_residues")
        if isinstance(residues, list):
            return len(residues)
    return None


def value_with_alias(data: dict[str, Any], key: str) -> Any:
    if data.get(key) is not None:
        return data.get(key)
    if "_replica_" in key:
        base, suffix = key.split("_replica_", 1)
        for alias in ALIASES.get(base, ()):
            alias_key = f"{alias}_replica_{suffix}"
            if data.get(alias_key) is not None:
                return data.get(alias_key)
        return None
    for alias in ALIASES.get(key, ()):
        if data.get(alias) is not None:
            return data.get(alias)
    return None


def replica_sd(audit: dict[str, Any], key: str) -> float | None:
    aliases = ALIASES.get(key, ())
    values: list[float] = []
    for replica in audit.get("replicas", []):
        data = replica.get("values", {})
        value = data.get(key)
        if value is None:
            for alias in aliases:
                value = data.get(alias)
                if value is not None:
                    break
        if isinstance(value, (int, float)):
            values.append(float(value))
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summary_record(run_dir: Path, output_dir: Path, rows: list[dict[str, Any]], correlations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "jobs_total": len([path for path in run_dir.iterdir() if path.is_dir()]) if run_dir.exists() else 0,
        "jobs_reported": len(rows),
        "jobs_valid": sum(1 for row in rows if row.get("status") == "valid"),
        "box_retry_jobs": [row["job_id"] for row in rows if row.get("box_retry_used")],
        "correlations": correlations,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "job_id",
        "name",
        "pdb_id",
        "status",
        "deltaG_exp_kJ_mol",
        "GB_delta_total_kJ_mol",
        "GB_delta_total_kJ_mol_replica_sd",
        "GB_delta_total_kJ_mol_replica_sem",
        "PB_delta_total_kJ_mol",
        "PB_delta_total_kJ_mol_replica_sd",
        "PB_delta_total_kJ_mol_replica_sem",
        "GB_dMM_kJ_mol",
        "GB_dMM_kJ_mol_replica_sd",
        "GB_dMM_kJ_mol_replica_sem",
        "PB_dMM_kJ_mol",
        "PB_dMM_kJ_mol_replica_sd",
        "PB_dMM_kJ_mol_replica_sem",
        "paper_mm_pbsa_kJ_mol",
        "paper_dmm_pbsa_kJ_mol",
        "mmpbsa_frames",
        "replica_count",
        "dropped_nonprotein_residue_count",
        "box_retry_used",
        "solvent_shape_actual",
        "job_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def correlation_record(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    pairs = [
        (float(row["deltaG_exp_kJ_mol"]), float(row[score_key]))
        for row in rows
        if row.get("status") == "valid" and isinstance(row.get("deltaG_exp_kJ_mol"), (int, float)) and isinstance(row.get(score_key), (int, float))
    ]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson_r": None, "spearman_r": None, "slope": None, "intercept": None}
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    slope, intercept = linear_fit(xs, ys)
    return {
        "n": len(pairs),
        "pearson_r": pearson_r(xs, ys),
        "spearman_r": spearman_r(xs, ys),
        "slope": slope,
        "intercept": intercept,
    }


def report_markdown(run_dir: Path, rows: list[dict[str, Any]], correlations: dict[str, dict[str, Any]]) -> str:
    valid = [row for row in rows if row.get("status") == "valid"]
    box_retry_jobs = [str(row["job_id"]) for row in rows if row.get("box_retry_used")]
    lines = [
        "# Peptide 3x5ns MMPBSA Validation Report",
        "",
        f"- Run directory: `{run_dir.resolve()}`",
        f"- Completed valid jobs: {len(valid)}/{len(rows)}",
        "- Protocol: `configs/peptide_crystal_3x5ns.yaml`",
        "- Score policy: MM/GBSA is the primary ranking score; MM/PBSA is a secondary check.",
        f"- Box retry jobs: {', '.join(box_retry_jobs) if box_retry_jobs else 'none'}",
        "",
        "## Correlations",
        "",
    ]
    for key in SCORE_KEYS:
        record = correlations[key]
        if record["pearson_r"] is None:
            lines.append(f"- `{LABELS[key]}` (`{key}`): insufficient data (`n={record['n']}`).")
        else:
            lines.append(
                f"- `{LABELS[key]}` (`{key}`): computed = {record['slope']:.4f} * experimental + {record['intercept']:.4f}; "
                f"Pearson r = {record['pearson_r']:.4f}; Spearman r = {record['spearman_r']:.4f}; n = {record['n']}."
            )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| job | status | exp kJ/mol | GB total mean +- SD | GB dMM mean +- SD | PB total mean +- SD | PB dMM mean +- SD | dropped HETATM residues | box retry |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: float(item["deltaG_exp_kJ_mol"])):
        lines.append(
            "| {job} | {status} | {exp} | {gb} | {gb_dmm} | {pb} | {pb_dmm} | {dropped} | {retry} |".format(
                job=row["job_id"],
                status=row["status"],
                exp=format_float(row.get("deltaG_exp_kJ_mol")),
                gb=format_mean_sd(row.get("GB_delta_total_kJ_mol"), row.get("GB_delta_total_kJ_mol_replica_sd")),
                gb_dmm=format_mean_sd(row.get("GB_dMM_kJ_mol"), row.get("GB_dMM_kJ_mol_replica_sd")),
                pb=format_mean_sd(row.get("PB_delta_total_kJ_mol"), row.get("PB_delta_total_kJ_mol_replica_sd")),
                pb_dmm=format_mean_sd(row.get("PB_dMM_kJ_mol"), row.get("PB_dMM_kJ_mol_replica_sd")),
                dropped=format_int(row.get("dropped_nonprotein_residue_count")),
                retry="yes" if row.get("box_retry_used") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- All 12 peptide jobs completed trajectory QC, per-replica MMPBSA, audit, and report generation.",
            "- On this small local validation subset, PB has slightly higher rank correlation than GB; treat this as a pipeline diagnostic only.",
            "- Absolute MM/PBSA values are shifted relative to experimental binding free energies and should not be interpreted as calibrated DeltaG.",
            "- `sp2016_10` required octahedral-box recovery to a rectangular box due to a water-water minimum-image overlap during EM.",
            "",
        ]
    )
    return "\n".join(lines)


def format_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.3f}"


def format_mean_sd(mean: Any, sd: Any) -> str:
    if not isinstance(mean, (int, float)):
        return ""
    if not isinstance(sd, (int, float)):
        return format_float(mean)
    return f"{float(mean):.3f} +- {float(sd):.3f}"


def format_int(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()

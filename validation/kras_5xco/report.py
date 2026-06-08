from __future__ import annotations

import html
import math
import re
from pathlib import Path
from typing import Any

from mmpbsa.common import read_json, utc_now, write_csv_atomic, write_json_atomic, write_text_atomic
from mmpbsa.metrics import pearson_r, spearman_r

try:
    import yaml
except ModuleNotFoundError:  # Keep --help usable in lean environments.
    yaml = None


GAS_CONSTANT_KJ_MOL_K = 8.31446261815324e-3
DEFAULT_TEMPERATURE_K = 298.15
SCORE_KEYS = [
    "GB_delta_total_kJ_mol",
    "PB_delta_total_kJ_mol",
    "GB_dMM_kJ_mol",
    "PB_dMM_kJ_mol",
    "GB_vdw_kJ_mol",
    "PB_vdw_kJ_mol",
]
PILOT_STATE_LABELS = {
    "gdp_only": "5xco_gdp_only_3x20ns",
    "gdp_mg": "5xco_gdp_mg_3x20ns",
}
BASELINE_LABEL = "baseline_noMg_AF3_3x5ns"
POLICIES = {
    "all_as_reported": lambda row: True,
    "full_length_point": lambda row: row.get("series_class") == "full_length_point",
    "truncation": lambda row: row.get("series_class") == "truncation",
    "non_censored": lambda row: not bool(row.get("is_censored")),
}


def report_kras_5xco_pilot(
    run_dir: Path,
    output_dir: Path,
    assay_dir: Path,
    baseline_run_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    assay_dir = assay_dir.resolve()
    baseline_run_dir = baseline_run_dir.resolve() if baseline_run_dir else None
    assay_records = load_assay_records(assay_dir)

    rows = load_result_rows(run_dir, assay_records, run_kind="pilot")
    if baseline_run_dir and baseline_run_dir.exists():
        rows.extend(load_result_rows(baseline_run_dir, assay_records, run_kind="baseline"))
    if not rows:
        raise SystemExit(f"No completed KRAS pilot results found in {run_dir}")

    correlations = compute_correlation_rows(rows)
    decision = build_decision(rows, correlations, run_dir, output_dir, assay_dir, baseline_run_dir)
    outputs = {
        "results": str(output_dir / "kras_5xco_mg_pilot_results.csv"),
        "correlations": str(output_dir / "kras_5xco_mg_pilot_correlations.csv"),
        "decision": str(output_dir / "kras_5xco_mg_pilot_decision.json"),
        "html": str(output_dir / "report_kras_5xco_mg_pilot.html"),
    }
    report = {
        "schema_version": "mmpbsa.kras_5xco_pilot_report.v1",
        "generated_at": utc_now(),
        "run_dir": str(run_dir),
        "baseline_run_dir": str(baseline_run_dir) if baseline_run_dir else "",
        "assay_dir": str(assay_dir),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "pilot_row_count": sum(1 for row in rows if row.get("run_kind") == "pilot"),
        "baseline_row_count": sum(1 for row in rows if row.get("run_kind") == "baseline"),
        "valid_pilot_rows": sum(1 for row in rows if row.get("run_kind") == "pilot" and row.get("status") == "valid"),
        "correlations": correlations,
        "decision": decision,
        "outputs": outputs,
    }
    write_csv_atomic(output_dir / "kras_5xco_mg_pilot_results.csv", rows)
    write_csv_atomic(output_dir / "kras_5xco_mg_pilot_correlations.csv", correlations)
    write_json_atomic(output_dir / "kras_5xco_mg_pilot_decision.json", decision)
    write_text_atomic(output_dir / "report_kras_5xco_mg_pilot.html", render_html_report(report, rows, correlations, decision))
    return report


def load_assay_records(assay_dir: Path) -> dict[str, dict[str, Any]]:
    if yaml is None:
        raise SystemExit("PyYAML is required to load KRAS assay YAML files.")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(assay_dir.glob("assay_*.yaml")):
        assay = yaml.safe_load(path.read_text(encoding="utf-8"))
        group = assay.get("assay_group", {}) or {}
        if group.get("nucleotide_state") != "GDP":
            continue
        for record in assay.get("records", []) or []:
            if record.get("protein_id") != "PRO_0001":
                continue
            value_nm = float(record["standard_value_nM"])
            relation = str(record.get("endpoint_relation") or "").strip()
            records[str(record["ligand_id"])] = {
                "ligand_id": record.get("ligand_id"),
                "mutation_label": record.get("mutation_label"),
                "endpoint_type": record.get("endpoint_type"),
                "endpoint_relation": relation,
                "standard_value_nM": value_nm,
                "experimental_deltaG_kJ_mol": delta_g_from_nm(value_nm),
                "is_censored": relation.startswith(">") or relation.startswith("<"),
                "assay_file": str(path),
            }
    return records


def load_result_rows(run_dir: Path, assay_records: dict[str, dict[str, Any]], run_kind: str) -> list[dict[str, Any]]:
    rows = []
    for job_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        summary_path = job_dir / "result" / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        if summary.get("status") != "valid":
            continue
        ligand_id = ligand_id_for_job(job_dir.name, job_dir)
        assay = assay_records.get(ligand_id)
        if not assay:
            continue
        manifest = read_optional_json(job_dir / "manifest.json")
        job_config = read_optional_json(job_dir / f"{job_dir.name}.json")
        state = receptor_state_for_job(job_dir.name, run_kind)
        row = {
            "run_kind": run_kind,
            "run_label": PILOT_STATE_LABELS.get(state, BASELINE_LABEL),
            "job_id": job_dir.name,
            "variant_id": variant_id_for_job(job_dir.name),
            "ligand_id": ligand_id,
            "receptor_state": state,
            "series_class": series_class_for_job(job_dir.name),
            "status": summary.get("status"),
            "trajectory_qc_status": summary.get("trajectory_qc_status"),
            "mmpbsa_qc_status": summary.get("mmpbsa_qc_status"),
            "mmpbsa_frames": summary.get("mmpbsa_frames"),
            "replica_count": summary.get("replica_count"),
            "peptide_residue_count": manifest.get("peptide_residue_count") or summary.get("peptide_residue_count"),
            "peptide_charge": peptide_charge(job_dir, summary, job_config, manifest),
            "peptide_atom_count": peptide_atom_count(job_dir),
            "peptide_heavy_atom_count": peptide_atom_count(job_dir, heavy_only=True),
            "peptide_rmsd_mean": average_replica_qc(summary, "peptide_bb_rmsd_after_receptor_fit_angstrom", "mean"),
            "receptor_rmsd_mean": average_replica_qc(summary, "receptor_bb_rmsd_angstrom", "mean"),
            "native_contacts_mean": average_native_contact(summary, "rec_pep[native]", "mean"),
            **assay,
        }
        for key in SCORE_KEYS:
            row[key] = summary.get(key)
            row[f"{key}_replica_sd"] = summary.get(f"{key}_replica_sd")
            row[f"{key}_replica_sem"] = summary.get(f"{key}_replica_sem")
        rows.append(row)
    return rows


def read_optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def ligand_id_for_job(job_id: str, job_dir: Path) -> str:
    config_path = job_dir / f"{job_id}.json"
    if config_path.exists():
        value = read_json(config_path).get("ligand_id")
        if value:
            return str(value)
    match = re.search(r"(PEP_\d{4})", job_id)
    if not match:
        raise SystemExit(f"Cannot infer ligand_id from job id {job_id!r}")
    return match.group(1)


def variant_id_for_job(job_id: str) -> str:
    return re.sub(r"_gdp_(?:mg|only)$", "", job_id)


def receptor_state_for_job(job_id: str, run_kind: str) -> str:
    if run_kind == "baseline":
        return "baseline_noMg_AF3"
    if job_id.endswith("_gdp_mg"):
        return "gdp_mg"
    if job_id.endswith("_gdp_only"):
        return "gdp_only"
    return "unknown"


def series_class_for_job(job_id: str) -> str:
    variant = variant_id_for_job(job_id)
    if variant.startswith(("core13_", "del4R_")):
        return "truncation"
    return "full_length_point"


def peptide_charge(job_dir: Path, summary: dict[str, Any], job_config: dict[str, Any], manifest: dict[str, Any]) -> float | None:
    for source in (summary, job_config, manifest):
        value = source.get("peptide_charge")
        if isinstance(value, (int, float)):
            return float(value)
    descriptor = peptide_prmtop_descriptor(job_dir)
    return descriptor.get("peptide_charge")


def peptide_atom_count(job_dir: Path, heavy_only: bool = False) -> int | None:
    descriptor = peptide_prmtop_descriptor(job_dir)
    key = "peptide_heavy_atom_count" if heavy_only else "peptide_atom_count"
    value = descriptor.get(key)
    return int(value) if isinstance(value, int) else None


def peptide_prmtop_descriptor(job_dir: Path) -> dict[str, Any]:
    prmtop = job_dir / "analysis" / "mmpbsa" / "rep01" / "peptide.prmtop"
    if not prmtop.exists():
        return {}
    try:
        import parmed as pmd
    except ModuleNotFoundError:
        return {}
    structure = pmd.load_file(str(prmtop))
    return {
        "peptide_charge": round(sum(atom.charge for atom in structure.atoms), 6),
        "peptide_atom_count": len(structure.atoms),
        "peptide_heavy_atom_count": sum(1 for atom in structure.atoms if getattr(atom, "atomic_number", 0) != 1),
    }


def average_replica_qc(summary: dict[str, Any], section: str, field: str) -> float | None:
    values = []
    for replica in summary.get("replica_qc", []) or []:
        value = ((replica.get(section) or {}).get(field))
        if isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def average_native_contact(summary: dict[str, Any], contact: str, field: str) -> float | None:
    values = []
    for replica in summary.get("replica_qc", []) or []:
        value = (((replica.get("native_contacts") or {}).get(contact) or {}).get(field))
        if isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def compute_correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for run_label in sorted({str(row["run_label"]) for row in rows}):
        run_rows = [row for row in rows if row.get("run_label") == run_label and row.get("status") == "valid"]
        for policy, predicate in POLICIES.items():
            subset = [row for row in run_rows if predicate(row)]
            for metric in SCORE_KEYS:
                score_pairs = [(float(row["experimental_deltaG_kJ_mol"]), float(row[metric])) for row in subset if is_number(row.get("experimental_deltaG_kJ_mol")) and is_number(row.get(metric))]
                charge_pairs = [(float(row["peptide_charge"]), float(row[metric])) for row in subset if is_number(row.get("peptide_charge")) and is_number(row.get(metric))]
                sem_key = f"{metric}_replica_sem"
                sem_values = [float(row[sem_key]) for row in subset if is_number(row.get(sem_key))]
                out.append(
                    {
                        "run_label": run_label,
                        "policy": policy,
                        "metric": metric,
                        "n_experiment": len(score_pairs),
                        "experiment_pearson_r": safe_pearson([x for x, _ in score_pairs], [y for _, y in score_pairs]),
                        "experiment_spearman_r": safe_spearman([x for x, _ in score_pairs], [y for _, y in score_pairs]),
                        "n_charge": len(charge_pairs),
                        "charge_pearson_r": safe_pearson([x for x, _ in charge_pairs], [y for _, y in charge_pairs]),
                        "charge_spearman_r": safe_spearman([x for x, _ in charge_pairs], [y for _, y in charge_pairs]),
                        "replica_sem_mean": sum(sem_values) / len(sem_values) if sem_values else None,
                    }
                )
    return out


def build_decision(
    rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    run_dir: Path,
    output_dir: Path,
    assay_dir: Path,
    baseline_run_dir: Path | None,
) -> dict[str, Any]:
    primary = find_correlation(correlations, "5xco_gdp_mg_3x20ns", "all_as_reported", "GB_delta_total_kJ_mol")
    pb_primary = find_correlation(correlations, "5xco_gdp_mg_3x20ns", "all_as_reported", "PB_delta_total_kJ_mol")
    primary_pass = passes_stop_rule(primary)
    pb_pass = passes_stop_rule(pb_primary)
    return {
        "generated_at": utc_now(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "assay_dir": str(assay_dir),
        "baseline_run_dir": str(baseline_run_dir) if baseline_run_dir else "",
        "pilot_status": {
            "expected_pilot_rows": 12,
            "valid_pilot_rows": sum(1 for row in rows if row.get("run_kind") == "pilot" and row.get("status") == "valid"),
            "valid_gdp_only_rows": sum(1 for row in rows if row.get("run_label") == "5xco_gdp_only_3x20ns" and row.get("status") == "valid"),
            "valid_gdp_mg_rows": sum(1 for row in rows if row.get("run_label") == "5xco_gdp_mg_3x20ns" and row.get("status") == "valid"),
            "frames_per_job": sorted({row.get("mmpbsa_frames") for row in rows if row.get("run_kind") == "pilot"}),
        },
        "primary_score": {
            "run_label": "5xco_gdp_mg_3x20ns",
            "metric": "GB_delta_total_kJ_mol",
            "passed_stop_rule": primary_pass,
            "experiment_pearson_r": primary.get("experiment_pearson_r"),
            "charge_pearson_r": primary.get("charge_pearson_r"),
            "interpretation": "candidate_primary_ranking_score" if primary_pass else "diagnostic_only",
        },
        "pb_total_check": {
            "run_label": "5xco_gdp_mg_3x20ns",
            "metric": "PB_delta_total_kJ_mol",
            "passed_stop_rule": pb_pass,
            "experiment_pearson_r": pb_primary.get("experiment_pearson_r"),
            "charge_pearson_r": pb_primary.get("charge_pearson_r"),
            "interpretation": "reference_only_when_charge_bias_is_high" if not pb_pass else "candidate_secondary_score",
        },
        "next_compute_recommendation": next_compute_recommendation(primary_pass),
    }


def next_compute_recommendation(primary_pass: bool) -> dict[str, Any]:
    if primary_pass:
        return {
            "action": "expand_gdp_mg_full_length_point_variants",
            "receptor_state": "gdp_mg",
            "protocol": "3 replicas x 20 ns",
            "variants": ["P7A_PEP_0005", "Y9A_PEP_0007", "I10A_PEP_0008", "S11A_PEP_0009", "Y12A_PEP_0010", "V15A_PEP_0013"],
            "do_not_expand_yet": ["core13_PEP_0003", "del4R_PEP_0002"],
        }
    return {
        "action": "do_not_expand_raw_mmpbsa_ranking",
        "reason": "Primary GDP+Mg GB total did not pass the charge/experiment stop rule.",
        "fallback": "Use MM/PBSA as pose/QC diagnostic and consider RBFE/TI for same-scaffold point mutations.",
    }


def find_correlation(correlations: list[dict[str, Any]], run_label: str, policy: str, metric: str) -> dict[str, Any]:
    for row in correlations:
        if row.get("run_label") == run_label and row.get("policy") == policy and row.get("metric") == metric:
            return row
    return {}


def passes_stop_rule(row: dict[str, Any]) -> bool:
    exp_r = abs(float(row["experiment_pearson_r"])) if is_number(row.get("experiment_pearson_r")) else None
    charge_r = abs(float(row["charge_pearson_r"])) if is_number(row.get("charge_pearson_r")) else None
    return bool(exp_r is not None and charge_r is not None and exp_r >= 0.3 and charge_r <= 0.7)


def delta_g_from_nm(value_nm: float) -> float:
    return GAS_CONSTANT_KJ_MOL_K * DEFAULT_TEMPERATURE_K * math.log(value_nm * 1e-9)


def safe_pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    value = pearson_r(xs, ys)
    return None if math.isnan(value) else value


def safe_spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    value = spearman_r(xs, ys)
    return None if math.isnan(value) else value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def render_html_report(report: dict[str, Any], rows: list[dict[str, Any]], correlations: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    primary = decision["primary_score"]
    pb_check = decision["pb_total_check"]
    summary_rows = [
        {"field": "Pilot rows", "value": decision["pilot_status"]["valid_pilot_rows"]},
        {"field": "GDP-only rows", "value": decision["pilot_status"]["valid_gdp_only_rows"]},
        {"field": "GDP+Mg rows", "value": decision["pilot_status"]["valid_gdp_mg_rows"]},
        {"field": "Primary score", "value": f"{primary['run_label']} / {primary['metric']}"},
        {"field": "Primary experiment r", "value": primary.get("experiment_pearson_r")},
        {"field": "Primary charge r", "value": primary.get("charge_pearson_r")},
        {"field": "Primary stop rule", "value": primary.get("passed_stop_rule")},
        {"field": "PB total stop rule", "value": pb_check.get("passed_stop_rule")},
        {"field": "Next action", "value": decision["next_compute_recommendation"]["action"]},
    ]
    total_corr = [
        row
        for row in correlations
        if row.get("policy") == "all_as_reported" and row.get("metric") in {"GB_delta_total_kJ_mol", "PB_delta_total_kJ_mol"}
    ]
    pilot_rows = [row for row in rows if row.get("run_kind") == "pilot"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>KRAS 5XCO GDP/Mg pilot report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #20242a; }}
    h1, h2 {{ margin-bottom: 0.35rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; font-size: 13px; }}
    th, td {{ border: 1px solid #d3d7dc; padding: 6px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f5f7; }}
    .note {{ color: #555f6d; max-width: 980px; }}
    .ok {{ color: #116329; font-weight: 600; }}
    .bad {{ color: #9b1c1c; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>KRAS 5XCO GDP/Mg pilot report</h1>
  <p class="note">Generated {escape(report.get("generated_at"))}. This report compares completed 5XCO-derived GDP-only/GDP+Mg pilot jobs against KRAS GDP assay records and the optional AF3/no-Mg baseline.</p>
  <h2>Decision Summary</h2>
  {render_table(summary_rows, ["field", "value"], class_field="value")}
  <h2>Total Score Correlations</h2>
  {render_table(total_corr, ["run_label", "policy", "metric", "n_experiment", "experiment_pearson_r", "experiment_spearman_r", "n_charge", "charge_pearson_r", "charge_spearman_r", "replica_sem_mean"])}
  <h2>Pilot Result Rows</h2>
  {render_table(pilot_rows, ["run_label", "job_id", "ligand_id", "series_class", "standard_value_nM", "experimental_deltaG_kJ_mol", "peptide_charge", "peptide_residue_count", "GB_delta_total_kJ_mol", "GB_delta_total_kJ_mol_replica_sem", "PB_delta_total_kJ_mol", "PB_delta_total_kJ_mol_replica_sem", "peptide_rmsd_mean", "native_contacts_mean"])}
</body>
</html>
"""


def render_table(rows: list[dict[str, Any]], fields: list[str], class_field: str | None = None) -> str:
    if not rows:
        return '<p class="note">No rows available.</p>'
    lines = ["<table>", "<thead><tr>" + "".join(f"<th>{escape(field)}</th>" for field in fields) + "</tr></thead>", "<tbody>"]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            klass = ""
            if class_field == field and isinstance(row.get("value"), bool):
                klass = ' class="ok"' if row.get("value") else ' class="bad"'
            cells.append(f"<td{klass}>{escape(format_value(value))}</td>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(format_value(item) for item in value)
    return str(value)


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))

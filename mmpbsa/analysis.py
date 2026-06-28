from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from .common import write_csv_atomic, write_json_atomic


NUM_RE = re.compile(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?")
SECTION_MARKERS = {
    "GENERALIZED BORN": "GB",
    "POISSON BOLTZMANN": "PB",
}
COMPONENTS = {"Complex:", "Receptor:", "Ligand:", "Differences (Complex - Receptor - Ligand):"}
INTERNAL_TERMS = ("BOND", "ANGLE", "DIHED")
TERMS = {
    "VDWAALS": "vdw",
    "EEL": "electrostatic",
    "EGB": "polar_solvation",
    "EPB": "polar_solvation",
    "ESURF": "nonpolar_solvation",
    "ENPOLAR": "nonpolar_solvation",
    "EDISPER": "dispersion",
    "DELTA TOTAL": "delta_total",
}


def parse_mmpbsa_full(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    frame_match = re.search(r"Calculations performed using\s+([0-9.]+)\s+complex frames", text)
    frames = float(frame_match.group(1)) if frame_match else None
    section: str | None = None
    component: str | None = None
    rows: dict[str, dict[str, float]] = {}
    values: dict[str, float] = {}

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        for marker, name in SECTION_MARKERS.items():
            if marker in stripped:
                section = name
                component = None
                break
        if stripped in COMPONENTS:
            component = "Differences" if stripped.startswith("Differences") else stripped.rstrip(":")
            continue
        if section is None or component is None:
            continue

        nums = [float(x) for x in NUM_RE.findall(line)]
        if len(nums) < 3:
            continue
        first = NUM_RE.search(line)
        if first is None:
            continue
        term = line[: first.start()].strip()
        if not term:
            continue
        rows[f"{section}.{component}.{term}"] = {"average": nums[0], "std_dev": nums[1], "sem": nums[2]}
        if component == "Differences" and term in TERMS:
            key = f"{section}_{TERMS[term]}_kcal_mol"
            values[key] = nums[0]
            values[key.replace("_kcal_mol", "_sem_kcal_mol")] = nums[2]

    add_kj(values)
    add_dmm(values)
    return {"frames": frames, "rows": rows, "values": values}


def add_kj(values: dict[str, float]) -> None:
    for key, value in list(values.items()):
        if key.endswith("_kcal_mol"):
            values[key.replace("_kcal_mol", "_kJ_mol")] = value * 4.184


def add_dmm(values: dict[str, float]) -> None:
    gb_terms = {"GB_vdw_kcal_mol", "GB_electrostatic_kcal_mol", "GB_polar_solvation_kcal_mol", "GB_nonpolar_solvation_kcal_mol"}
    if gb_terms <= values.keys():
        gb_dmm = (
            values["GB_vdw_kcal_mol"]
            + 0.2 * values["GB_electrostatic_kcal_mol"]
            + 0.2 * values["GB_polar_solvation_kcal_mol"]
            + values["GB_nonpolar_solvation_kcal_mol"]
        )
        values["GB_dMM_kcal_mol"] = gb_dmm
        values["GB_dMM_kJ_mol"] = gb_dmm * 4.184
        # Compatibility alias for reports generated before v0.1.3.
        values["GB_dmm_like_kcal_mol"] = gb_dmm
        values["GB_dmm_like_kJ_mol"] = gb_dmm * 4.184
    pb_terms = {"PB_vdw_kcal_mol", "PB_electrostatic_kcal_mol", "PB_polar_solvation_kcal_mol", "PB_nonpolar_solvation_kcal_mol"}
    if pb_terms <= values.keys():
        pb_nonpolar = values["PB_nonpolar_solvation_kcal_mol"] + values.get("PB_dispersion_kcal_mol", 0.0)
        pb_dmm = (
            values["PB_vdw_kcal_mol"]
            + 0.2 * values["PB_electrostatic_kcal_mol"]
            + 0.2 * values["PB_polar_solvation_kcal_mol"]
            + pb_nonpolar
        )
        values["PB_nonpolar_total_kcal_mol"] = pb_nonpolar
        values["PB_nonpolar_total_kJ_mol"] = pb_nonpolar * 4.184
        values["PB_dMM_kcal_mol"] = pb_dmm
        values["PB_dMM_kJ_mol"] = pb_dmm * 4.184
        # Compatibility alias for reports generated before v0.1.3.
        values["PB_dmm_like_kcal_mol"] = pb_dmm
        values["PB_dmm_like_kJ_mol"] = pb_dmm * 4.184


def add_dmm_like(values: dict[str, float]) -> None:
    add_dmm(values)


def audit_mmpbsa(parsed: dict[str, Any], min_frames: int, internal_limit: float, internal_std_limit: float) -> dict[str, Any]:
    rows = parsed["rows"]
    frames = parsed["frames"]
    issues: list[dict[str, Any]] = []
    notes: list[str] = []

    if frames is None:
        issues.append({"severity": "fail", "code": "missing_frame_count", "message": "Could not parse MMPBSA frame count."})
    elif frames < min_frames:
        issues.append({"severity": "fail", "code": "too_few_frames", "message": f"MMPBSA used {frames:g} frames; required at least {min_frames}."})

    max_internal_avg = 0.0
    max_internal_std = 0.0
    worst_avg_key = ""
    worst_std_key = ""
    for key, values in rows.items():
        parts = key.split(".")
        if len(parts) != 3:
            continue
        _, component, term = parts
        if component == "Differences" or term not in INTERNAL_TERMS:
            continue
        average = abs(values["average"])
        std_dev = abs(values["std_dev"])
        if average > max_internal_avg:
            max_internal_avg = average
            worst_avg_key = key
        if std_dev > max_internal_std:
            max_internal_std = std_dev
            worst_std_key = key

    if max_internal_avg > internal_limit:
        issues.append({"severity": "fail", "code": "internal_energy_too_large", "message": f"{worst_avg_key} average is {max_internal_avg:.3f} kcal/mol."})
    if max_internal_std > internal_std_limit:
        issues.append({"severity": "fail", "code": "internal_energy_std_too_large", "message": f"{worst_std_key} std dev is {max_internal_std:.3f} kcal/mol."})

    max_delta_internal = 0.0
    for section in ("GB", "PB"):
        for term in INTERNAL_TERMS:
            values = rows.get(f"{section}.Differences.{term}")
            if values is not None:
                max_delta_internal = max(max_delta_internal, abs(values["average"]))
    if max_delta_internal > 0.1:
        issues.append({"severity": "fail", "code": "internal_terms_do_not_cancel", "message": f"Difference internal terms deviate by {max_delta_internal:.3f} kcal/mol."})
    else:
        notes.append("Difference BOND/ANGLE/DIHED terms cancel within 0.1 kcal/mol.")

    required_values = ["GB_delta_total_kJ_mol", "PB_delta_total_kJ_mol", "GB_dMM_kJ_mol", "PB_dMM_kJ_mol"]
    missing = [key for key in required_values if key not in parsed["values"]]
    if missing:
        issues.append({"severity": "fail", "code": "missing_energy_terms", "message": "Missing energy terms: " + ", ".join(missing)})

    return {
        "status": "invalid" if issues else "valid",
        "frames": frames,
        "min_frames": min_frames,
        "max_internal_average_kcal_mol": max_internal_avg,
        "max_internal_average_key": worst_avg_key,
        "max_internal_std_kcal_mol": max_internal_std,
        "max_internal_std_key": worst_std_key,
        "max_delta_internal_average_kcal_mol": max_delta_internal,
        "issues": issues,
        "notes": notes,
    }


def load_cpptraj_table(path: Path) -> tuple[list[str], list[list[float]]]:
    header: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            header = line.split()
            continue
        rows.append([float(value) for value in line.split()])
    if not header or not rows:
        raise SystemExit(f"Could not parse cpptraj table: {path}")
    return header, rows


def column_stats(header: list[str], rows: list[list[float]]) -> dict[str, dict[str, float]]:
    columns = list(zip(*rows))
    stats: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(header[1:], start=1):
        values = [float(value) for value in columns[idx]]
        stats[name] = {"min": min(values), "mean": sum(values) / len(values), "max": max(values), "last": values[-1]}
    return stats


def write_trajectory_qc_csv(
    output: Path,
    receptor_rows: list[list[float]],
    partner_rows: list[list[float]],
    contact_header: list[str],
    contact_rows: list[list[float]],
    partner_field: str = "ligand_heavy_rmsd_after_receptor_fit_angstrom",
    replica_names: list[str] | None = None,
    frames_per_replica: int | None = None,
) -> None:
    contact_names = contact_header[1:]
    rows: list[dict[str, Any]] = []
    names = replica_names or []
    per_replica = int(frames_per_replica or 0)
    for idx, receptor in enumerate(receptor_rows):
        row: dict[str, Any] = {
            "frame": int(receptor[0]),
            "receptor_bb_rmsd_angstrom": receptor[1],
            partner_field: partner_rows[idx][1],
        }
        if names and per_replica > 0:
            replica_idx = min(idx // per_replica, len(names) - 1)
            row["replica"] = names[replica_idx]
            row["replica_frame"] = idx - replica_idx * per_replica + 1
            row["global_frame"] = idx + 1
        for name_idx, name in enumerate(contact_names, start=1):
            row[name] = contact_rows[idx][name_idx]
        rows.append(row)
    write_csv_atomic(output, rows)


def evaluate_trajectory_qc(
    summary: dict[str, Any],
    thresholds: dict[str, Any],
    partner_field: str = "ligand_heavy_rmsd_after_receptor_fit_angstrom",
    partner_threshold_key: str = "ligand_rmsd_warn_angstrom",
    partner_label: str = "ligand",
    contact_prefix: str = "rec_lig",
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    receptor = summary["receptor_bb_rmsd_angstrom"]
    partner = summary[partner_field]
    contacts = summary["native_contacts"]
    if receptor["max"] > float(thresholds["receptor_rmsd_fail_angstrom"]):
        issues.append({"severity": "fail", "code": "receptor_rmsd_high", "message": f"Receptor backbone RMSD max is {receptor['max']:.3f}."})
    if partner["max"] > float(thresholds[partner_threshold_key]):
        issues.append({"severity": "warn", "code": f"{partner_label}_rmsd_high", "message": f"{partner_label.title()} RMSD max is {partner['max']:.3f}."})
    native = contacts.get(f"{contact_prefix}[native]")
    mindist = contacts.get(f"{contact_prefix}[mindist]")
    if native and native["min"] < float(thresholds["native_contacts_fail_min"]):
        issues.append({"severity": "fail", "code": "native_contacts_low", "message": f"Minimum native contacts is {native['min']:.0f}."})
    if mindist and mindist["max"] > float(thresholds["interface_distance_fail_angstrom"]):
        issues.append({"severity": "fail", "code": "interface_distance_high", "message": f"Interface minimum distance max is {mindist['max']:.3f}."})
    return issues


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank(xs), rank(ys))


def write_debug_json(path: Path, data: Any) -> None:
    write_json_atomic(path, json.loads(json.dumps(data, default=str)))

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from .common import label_chains, mamba_command, parse_simple_mask, read_json, run_logged, utc_now, write_csv_atomic, write_json_atomic, write_text_atomic


PARTNER_COLUMNS = [
    ("ligand_heavy_rmsd_after_receptor_fit_angstrom", "Ligand heavy RMSD"),
    ("peptide_bb_rmsd_after_receptor_fit_angstrom", "Peptide backbone RMSD"),
]
SCORE_FIELDS = [
    ("GB_delta_total_kJ_mol", "GB total", "GB_delta_total_kJ_mol_replica_sd"),
    ("PB_delta_total_kJ_mol", "PB total", "PB_delta_total_kJ_mol_replica_sd"),
    ("GB_dMM_kJ_mol", "GB dMM", "GB_dMM_kJ_mol_replica_sd"),
    ("PB_dMM_kJ_mol", "PB dMM", "PB_dMM_kJ_mol_replica_sd"),
]
COFACTOR_RESN = "GNP+GDP+GTP+ATP+ADP+MG+MN+ZN+CA"
BINDER_CARBON_COLORS = ["cyan", "yelloworange", "tv_green", "magenta", "salmon"]
RECEPTOR_COLORS = ["gray70", "gray60", "slate"]
ELEMENT_COLORS = {"N": "blue", "O": "red", "S": "yellow", "P": "orange"}
ROW_METADATA_COLUMNS = {"frame", "global_frame", "replica", "replica_frame", "state"}
REPLICA_TRACE_COLORS = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#0891b2", "#dc2626", "#475569"]
CHARGED_POSITIVE_RESIDUES = {"ARG", "LYS", "HIS", "HIP", "HID", "HIE"}
CHARGED_NEGATIVE_RESIDUES = {"ASP", "GLU"}
COMPOSITE_SORT = "composite"
COMPOSITE_SORT_FIELDS = [
    "PB_delta_total_kJ_mol",
    "PB_dMM_kJ_mol",
    "GB_delta_total_kJ_mol",
    "GB_dMM_kJ_mol",
]
FIELD_LABELS = {
    COMPOSITE_SORT: "PB > PB dMM > GB > GB dMM",
    "GB_delta_total_kJ_mol": "GB total",
    "PB_delta_total_kJ_mol": "PB total",
    "GB_dMM_kJ_mol": "GB dMM",
    "PB_dMM_kJ_mol": "PB dMM",
    "GB_delta_total_kJ_mol_replica_sd": "GB SD",
    "PB_delta_total_kJ_mol_replica_sd": "PB SD",
    "GB_dMM_kJ_mol_replica_sd": "GB dMM SD",
    "PB_dMM_kJ_mol_replica_sd": "PB dMM SD",
    "GB_delta_total_kcal_mol": "GB total",
    "PB_delta_total_kcal_mol": "PB total",
}
RUN_TABLE_FIELDS = [
    "GB_delta_total_kJ_mol",
    "GB_delta_total_kJ_mol_replica_sd",
    "PB_delta_total_kJ_mol",
    "PB_delta_total_kJ_mol_replica_sd",
    "GB_dMM_kJ_mol",
    "GB_dMM_kJ_mol_replica_sd",
    "PB_dMM_kJ_mol",
    "PB_dMM_kJ_mol_replica_sd",
]
RUN_TABLE_GROUPS = [
    (
        "PB score",
        [
            ("PB_delta_total_kJ_mol", "PB total", "kJ/mol", ""),
            ("PB_delta_total_kJ_mol_replica_sd", "PB SD", "kJ/mol", "sd"),
            ("PB_dMM_kJ_mol", "PB dMM", "kJ/mol", ""),
            ("PB_dMM_kJ_mol_replica_sd", "PB dMM SD", "kJ/mol", "sd"),
        ],
    ),
    (
        "GB score",
        [
            ("GB_delta_total_kJ_mol", "GB total", "kJ/mol", ""),
            ("GB_delta_total_kJ_mol_replica_sd", "GB SD", "kJ/mol", "sd"),
            ("GB_dMM_kJ_mol", "GB dMM", "kJ/mol", ""),
            ("GB_dMM_kJ_mol_replica_sd", "GB dMM SD", "kJ/mol", "sd"),
        ],
    ),
    (
        "Trajectory QC",
        [
            ("receptor_rmsd_mean", "Receptor RMSD", "Angstrom", ""),
            ("partner_rmsd_mean", "Partner RMSD", "Angstrom", ""),
            ("native_contacts_mean", "Native contacts mean", "contacts", ""),
            ("native_contacts_min", "Native contacts min", "contacts", ""),
            ("interface_distance_min", "Interface distance min", "Angstrom", ""),
        ],
    ),
]


def visualize_job(
    job_dir: Path,
    output_dir: Path,
    *,
    export_visual: bool = False,
    align: bool = True,
    movie_stride: int = 5,
    render_video: bool = False,
    zip_archive: bool = False,
    archive_name: str | None = None,
) -> dict[str, Any]:
    job = job_dir.resolve()
    qc_csv = job / "analysis" / "qc" / "trajectory_qc.csv"
    if not qc_csv.exists():
        raise SystemExit(f"Missing trajectory QC CSV: {qc_csv}")
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary = read_optional_json(job / "result" / "summary.json")
    qc_summary = read_optional_json(job / "analysis" / "qc" / "summary.json")
    manifest = read_optional_json(job / "manifest.json")
    qc_rows = normalize_replica_rows(read_csv_rows(qc_csv), manifest)
    thresholds = infer_qc_thresholds(job)
    title = str(summary.get("name") or qc_summary.get("name") or job.name)

    metric_rows = qc_metric_rows(qc_rows, thresholds=thresholds)
    write_csv_atomic(out / "qc_metrics.csv", metric_rows)
    copy_if_exists(qc_csv, out / "trajectory_qc.csv")
    write_csv_atomic(out / "trajectory_qc_by_replica.csv", qc_rows)

    qc_svg = trajectory_qc_svg(qc_rows, title, thresholds=thresholds)
    write_text_atomic(out / "trajectory_qc.svg", qc_svg)

    score_paths: dict[str, str] = {}
    score_svg = score_bar_svg(summary, title)
    if score_svg:
        write_text_atomic(out / "mmpbsa_scores.svg", score_svg)
        score_paths = {
            "mmpbsa_scores_svg": str(out / "mmpbsa_scores.svg"),
        }
    visual_paths: dict[str, Any] = {}
    if export_visual:
        visual_record = copy_job_bundle_files(job, out / "pymol", snapshots_only=False, align=align, movie_stride=movie_stride, render_video=render_video)
        visual_paths = {"pymol_dir": str(out / "pymol"), "pymol_visual": visual_record}
    analysis_paths = write_sample_analysis_outputs(job, out, qc_rows, visual_paths)
    write_text_atomic(out / "index.html", sample_index_html(job, title, summary, qc_summary, metric_rows, qc_svg, score_svg, visual_paths, analysis_paths))

    report = {
        "schema_version": "mmpbsa.visualize.job.v1",
        "created_at": utc_now(),
        "job_dir": str(job),
        "output_dir": str(out),
        "job_id": job.name,
        "index_html": str(out / "index.html"),
        "qc_metrics_csv": str(out / "qc_metrics.csv"),
        "trajectory_qc_by_replica_csv": str(out / "trajectory_qc_by_replica.csv"),
        "trajectory_qc_svg": str(out / "trajectory_qc.svg"),
        **score_paths,
        **visual_paths,
        **analysis_paths,
    }
    if zip_archive or archive_name:
        archive_path = report_archive_path(out, archive_name)
        report["zip_archive"] = True
        report["archive"] = str(archive_path)
        write_json_atomic(out / "manifest.json", report)
        write_zip(out, archive_path)
    else:
        report["zip_archive"] = False
    write_json_atomic(out / "manifest.json", report)
    return report


def visualize_run(
    run_dir: Path,
    output_dir: Path,
    *,
    job_ids: list[str] | None = None,
    sort_by: str = COMPOSITE_SORT,
    limit: int | None = None,
    include_jobs: bool = False,
    include_samples: bool | None = None,
    export_pymol: bool = False,
    align: bool = True,
    movie_stride: int = 5,
    render_video: bool = False,
    zip_archive: bool = False,
    archive_name: str | None = None,
) -> dict[str, Any]:
    root = run_dir.resolve()
    rows = completed_summary_rows(root)
    if job_ids:
        selected = set(job_ids)
        rows = [row for row in rows if str(row.get("job_id") or row.get("job_dir") or "") in selected or Path(str(row.get("job_dir") or "")).name in selected]
        missing = sorted(selected - {str(row.get("job_id") or Path(str(row.get("job_dir") or "")).name) for row in rows})
        if missing:
            raise SystemExit("Missing completed jobs: " + ", ".join(missing))
    rows = sort_summary_rows(rows, sort_by)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"No completed jobs found under {root}")

    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_sample_reports = include_jobs if include_samples is None else include_samples
    job_visuals: dict[str, dict[str, Any]] = {}
    if write_sample_reports:
        for row in rows:
            job_id = str(row.get("job_id") or Path(str(row["job_dir"])).name)
            job_visuals[job_id] = visualize_job(
                Path(str(row["job_dir"])),
                out / "samples" / job_id,
                export_visual=export_pymol,
                align=align,
                movie_stride=movie_stride,
                render_video=render_video,
            )
    ranking_rows = [ranking_row(row, sort_by) for row in rows]
    qc_rows = [run_qc_row(Path(str(row["job_dir"])), row) for row in rows]
    write_csv_atomic(out / "ranking.csv", ranking_rows)
    write_csv_atomic(out / "qc_summary.csv", qc_rows)
    write_csv_atomic(out / "qc_overview.csv", qc_rows)

    ranking_svg = ranking_svg_plot(ranking_rows, sort_by)
    qc_svg = qc_overview_svg(qc_rows)
    write_text_atomic(out / "ranking.svg", ranking_svg)
    write_text_atomic(out / "qc_summary.svg", qc_svg)
    write_text_atomic(out / "qc_overview.svg", qc_svg)
    write_text_atomic(out / "index.html", run_index_html(root, rows, ranking_rows, qc_rows, ranking_svg, qc_svg, sort_by, include_samples=write_sample_reports))

    report = {
        "schema_version": "mmpbsa.visualize.run.v1",
        "created_at": utc_now(),
        "run_dir": str(root),
        "output_dir": str(out),
        "jobs": len(rows),
        "sort_by": sort_by,
        "include_samples": write_sample_reports,
        "include_jobs": write_sample_reports,
        "export_pymol": export_pymol,
        "index_html": str(out / "index.html"),
        "ranking_csv": str(out / "ranking.csv"),
        "qc_summary_csv": str(out / "qc_summary.csv"),
    }
    if job_visuals:
        report["sample_reports"] = job_visuals
    if zip_archive or archive_name:
        archive_path = report_archive_path(out, archive_name)
        report["zip_archive"] = True
        report["archive"] = str(archive_path)
        write_json_atomic(out / "manifest.json", report)
        write_zip(out, archive_path)
    else:
        report["zip_archive"] = False
    write_json_atomic(out / "manifest.json", report)
    return report


def bundle_pymol(
    run_dir: Path,
    output_dir: Path,
    *,
    job_ids: list[str],
    archive_name: str | None = None,
    zip_archive: bool = False,
    snapshots_only: bool = False,
    align: bool = True,
    movie_stride: int = 5,
    keep_plots: bool = False,
    render_video: bool = False,
) -> dict[str, Any]:
    if not job_ids:
        raise SystemExit("At least one --job-id is required for a PyMOL bundle.")
    root = run_dir.resolve()
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_archive = zip_archive or bool(archive_name)
    bundle_name = archive_stem(archive_name or "pymol_bundle")
    bundle_root = out / bundle_name
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)

    jobs: list[dict[str, Any]] = []
    for job_id in job_ids:
        source_job = root / job_id
        if not source_job.exists():
            raise SystemExit(f"Job directory does not exist: {source_job}")
        target_job = bundle_root / "jobs" / job_id
        target_job.mkdir(parents=True)
        bundle_record = copy_job_bundle_files(
            source_job,
            target_job,
            snapshots_only=snapshots_only,
            align=align,
            movie_stride=movie_stride,
            render_video=render_video,
        )
        if keep_plots:
            bundle_record["visual"] = visualize_job(source_job, target_job / "visual")
        jobs.append({"job_id": job_id, "source_job_dir": str(source_job), "bundle_job_dir": str(target_job.relative_to(bundle_root)), **bundle_record})

    manifest = {
        "schema_version": "mmpbsa.visualize.pymol_bundle.v1",
        "created_at": utc_now(),
        "source_run_dir": str(root),
        "snapshots_only": snapshots_only,
        "align": align,
        "movie_stride": movie_stride,
        "keep_plots": keep_plots,
        "render_video": render_video,
        "zip_archive": write_archive,
        "jobs": jobs,
    }
    write_json_atomic(bundle_root / "manifest.json", manifest)
    write_text_atomic(bundle_root / "README.md", bundle_readme(jobs, snapshots_only))
    write_text_atomic(bundle_root / "index.html", bundle_index_html(jobs, snapshots_only=snapshots_only, keep_plots=keep_plots))
    report = {
        "schema_version": "mmpbsa.visualize.bundle_report.v1",
        "created_at": utc_now(),
        "bundle_dir": str(bundle_root),
        "index_html": str(bundle_root / "index.html"),
        "align": align,
        "movie_stride": movie_stride,
        "keep_plots": keep_plots,
        "zip_archive": write_archive,
        "jobs": len(jobs),
    }
    if write_archive:
        archive_path = out / f"{bundle_name}.zip"
        write_zip(bundle_root, archive_path)
        report["archive"] = str(archive_path)
    write_json_atomic(bundle_root / "bundle_report.json", report)
    return report


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def read_numeric_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float]] = []
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                parsed = numeric(value)
                if parsed is not None:
                    row[key] = parsed
            rows.append(row)
    if not rows:
        raise SystemExit(f"No numeric rows found in {path}")
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                parsed = numeric(value)
                row[key] = parsed if parsed is not None else (value or "")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def normalize_replica_rows(rows: list[dict[str, Any]], manifest: dict[str, Any], *, stride: int = 1) -> list[dict[str, Any]]:
    if not rows:
        return []
    names, per_replica = replica_layout_from_manifest(manifest, len(rows), stride=stride)
    frame_settings = manifest.get("frame_settings") or {}
    full_frames_per_replica = int(numeric(frame_settings.get("frames_per_replica")) or per_replica or 0)
    counters: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        normalized = dict(row)
        raw_frame = numeric(normalized.get("frame"))
        raw_global_frame = numeric(normalized.get("global_frame"))
        if raw_global_frame is not None:
            global_frame = int(raw_global_frame)
        elif stride > 1:
            normalized.setdefault("state", int(raw_frame or idx + 1))
            global_frame = idx * stride + 1
            normalized["frame"] = global_frame
        else:
            global_frame = int(raw_frame or idx + 1)
        normalized["global_frame"] = global_frame
        replica = str(normalized.get("replica") or "")
        replica_frame = numeric(normalized.get("replica_frame"))
        if not replica and names and per_replica > 0:
            if full_frames_per_replica:
                replica_idx = min(max((global_frame - 1) // full_frames_per_replica, 0), len(names) - 1)
            else:
                replica_idx = min(idx // per_replica, len(names) - 1)
            replica = names[replica_idx]
            if full_frames_per_replica:
                replica_frame = global_frame - replica_idx * full_frames_per_replica
            else:
                replica_frame = idx - replica_idx * per_replica + 1
        elif replica and replica_frame is None:
            counters[replica] = counters.get(replica, 0) + 1
            replica_frame = (counters[replica] - 1) * stride + 1 if stride > 1 else counters[replica]
        if replica:
            normalized["replica"] = replica
        if replica_frame is not None:
            normalized["replica_frame"] = int(replica_frame)
        out.append(normalized)
    return out


def replica_layout_from_manifest(manifest: dict[str, Any], row_count: int, *, stride: int = 1) -> tuple[list[str], int]:
    frame_settings = manifest.get("frame_settings") or {}
    names = list(frame_settings.get("replica_names") or manifest.get("replicas") or [])
    replica_count = int(numeric(frame_settings.get("replica_count")) or numeric(manifest.get("replica_count")) or len(names) or 0)
    if not names and replica_count:
        names = [f"rep{idx:02d}" for idx in range(1, replica_count + 1)]
    frames_per_replica = int(numeric(frame_settings.get("frames_per_replica")) or 0)
    if stride > 1 and frames_per_replica:
        frames_per_replica = ((frames_per_replica - 1) // stride) + 1
    if not frames_per_replica and names:
        frames_per_replica = max(1, math.ceil(row_count / len(names)))
    return names, frames_per_replica


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def trajectory_qc_svg(rows: list[dict[str, Any]], title: str, thresholds: dict[str, float] | None = None) -> str:
    panels: list[tuple[str, list[tuple[str, str]]]] = []
    if not rows:
        return replica_line_panels_svg([], [], f"{title} trajectory QC", thresholds=thresholds or {})
    panels.append(("Receptor backbone RMSD (angstrom)", [("receptor_bb_rmsd_angstrom", "Receptor")]))
    for column, label in PARTNER_COLUMNS:
        if column in rows[0]:
            panels.append((f"{label} (angstrom)", [(column, label)]))
            break
    native = first_column(rows, "[native]")
    nonnative = first_column(rows, "[nonnative]")
    mindist = first_column(rows, "[mindist]")
    maxdist = first_column(rows, "[maxdist]")
    if native:
        metrics = [(native, "Native")]
        if nonnative:
            metrics.append((nonnative, "Nonnative"))
        panels.append(("Interface contacts", metrics))
    if mindist:
        metrics = [(mindist, "Minimum distance")]
        if maxdist:
            metrics.append((maxdist, "Maximum distance"))
        panels.append(("Interface distance (angstrom)", metrics))
    return replica_line_panels_svg(rows, panels, f"{title} trajectory QC", thresholds=thresholds or {})


def score_bar_svg(summary: dict[str, Any], title: str) -> str:
    bars = []
    for key, label, sd_key in SCORE_FIELDS:
        value = numeric(summary.get(key))
        if value is not None:
            bars.append({"label": label, "value": value, "sd": numeric(summary.get(sd_key))})
    if not bars:
        return ""
    return bar_svg(bars, f"{title} MMPBSA scores", "kJ/mol")


def write_sample_analysis_outputs(job: Path, out: Path, qc_rows: list[dict[str, Any]], visual_paths: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    manifest = read_optional_json(job / "manifest.json")

    landscape_rows = md_energy_landscape_rows(job, out)
    if landscape_rows:
        write_csv_atomic(out / "md_energy_landscape.csv", landscape_rows)
        landscape_svg = md_energy_landscape_svg(landscape_rows)
        write_text_atomic(out / "md_energy_landscape.svg", landscape_svg)
        basin_rows = md_energy_basin_rows(landscape_rows)
        if basin_rows:
            write_csv_atomic(out / "md_energy_basin_by_replica.csv", basin_rows)
            paths["md_energy_basin_by_replica_csv"] = str(out / "md_energy_basin_by_replica.csv")
        basin_svg = md_energy_basin_svg(landscape_rows)
        write_text_atomic(out / "md_energy_basin.svg", basin_svg)
        paths["md_energy_landscape_csv"] = str(out / "md_energy_landscape.csv")
        paths["md_energy_trace_svg"] = str(out / "md_energy_landscape.svg")
        paths["md_energy_landscape_svg"] = str(out / "md_energy_landscape.svg")
        paths["md_energy_basin_svg"] = str(out / "md_energy_basin.svg")

    pdb_path = exported_trajectory_path(job, out, visual_paths)
    if pdb_path:
        visual_record = visual_paths.get("pymol_visual") or {}
        profile = manifest.get("profile") or {}
        export_config = profile.get("export") or {}
        stride = int(numeric(visual_record.get("movie_stride")) or numeric(export_config.get("pymol_stride")) or 1)
        interaction_rows = interaction_contact_rows(pdb_path, manifest, stride=stride)
        if interaction_rows:
            write_csv_atomic(out / "interaction_contacts.csv", interaction_rows)
            write_csv_atomic(out / "interaction_contacts_by_replica.csv", interaction_rows)
            interaction_svg = interaction_contacts_svg(interaction_rows)
            write_text_atomic(out / "interaction_contacts.svg", interaction_svg)
            paths["interaction_contacts_csv"] = str(out / "interaction_contacts.csv")
            paths["interaction_contacts_by_replica_csv"] = str(out / "interaction_contacts_by_replica.csv")
            paths["interaction_contacts_svg"] = str(out / "interaction_contacts.svg")
            paths["interaction_source_pdb"] = str(pdb_path)
    return paths


def exported_trajectory_path(job: Path, out: Path, visual_paths: dict[str, Any]) -> Path | None:
    pymol_dir = visual_paths.get("pymol_dir")
    candidates: list[Path] = []
    if pymol_dir:
        base = Path(str(pymol_dir))
        candidates.extend([base / "structures" / "aligned_trajectory.pdb", base / "structures" / "pymol_trajectory.pdb"])
    candidates.extend([out / "pymol" / "structures" / "aligned_trajectory.pdb", job / "analysis" / "structures" / "pymol_trajectory.pdb"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def md_energy_landscape_rows(job: Path, out: Path) -> list[dict[str, Any]]:
    manifest = read_optional_json(job / "manifest.json")
    md_root = job / "md"
    if not md_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    analysis_root = out / "md_analysis"
    for rep_dir in sorted(md_root.glob("rep*")):
        if not rep_dir.is_dir():
            continue
        rep_out = analysis_root / rep_dir.name
        rep_out.mkdir(parents=True, exist_ok=True)
        potential_rows = extract_potential_rows(rep_dir, rep_out, manifest)
        rmsd_rows = mdtraj_binder_rmsd_rows(rep_dir, manifest)
        if not potential_rows or not rmsd_rows:
            continue
        reference = potential_rows[0]["potential_kJ_mol"]
        for idx, rmsd in enumerate(rmsd_rows, start=1):
            potential = nearest_potential_row(potential_rows, float(rmsd["time_ps"]), fallback_index=idx - 1)
            if potential is None:
                continue
            rows.append(
                {
                    "replica": rep_dir.name,
                    "frame": idx,
                    "time_ps": rmsd["time_ps"],
                    "binder_rmsd_angstrom": rmsd["binder_rmsd_angstrom"],
                    "potential_kJ_mol": potential["potential_kJ_mol"],
                    "delta_potential_kJ_mol": potential["potential_kJ_mol"] - reference,
                }
            )
    return rows


def extract_potential_rows(rep_dir: Path, out_dir: Path, manifest: dict[str, Any]) -> list[dict[str, float]]:
    edr = rep_dir / "md_prod.edr"
    if not edr.exists():
        return []
    xvg = out_dir / "potential.xvg"
    log = out_dir / "gmx_energy_potential.log"
    if not xvg.exists():
        runtime = ((manifest.get("profile") or {}).get("runtime") or {}) if manifest else {}
        gmx_bin = str(runtime.get("gmx_bin") or "gmx")
        gmxrc = str(runtime.get("gmxrc") or "")
        source = f"source {shlex.quote(gmxrc)} && " if gmxrc else ""
        command = f"{source}printf 'Potential\\n0\\n' | {shlex.quote(gmx_bin)} energy -f {shlex.quote(str(edr))} -o {shlex.quote(str(xvg))}"
        process = subprocess.run(["bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        write_text_atomic(log, process.stdout)
        if process.returncode != 0 or not xvg.exists():
            return []
    return parse_potential_xvg(xvg)


def parse_potential_xvg(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append({"time_ps": float(parts[0]), "potential_kJ_mol": float(parts[1])})
        except ValueError:
            continue
    return rows


def nearest_potential_row(rows: list[dict[str, float]], time_ps: float, *, fallback_index: int) -> dict[str, float] | None:
    if not rows:
        return None
    if fallback_index < len(rows):
        candidate = rows[fallback_index]
        if abs(candidate["time_ps"] - time_ps) <= 1e-6:
            return candidate
    return min(rows, key=lambda row: abs(row["time_ps"] - time_ps))


def mdtraj_binder_rmsd_rows(rep_dir: Path, manifest: dict[str, Any]) -> list[dict[str, float]]:
    gro = rep_dir / "md_prod.gro"
    xtc = rep_dir / "md_prod.xtc"
    if not gro.exists() or not xtc.exists():
        return []
    try:
        import mdtraj as md  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return []
    try:
        traj = md.load(str(xtc), top=str(gro))
        traj.image_molecules(inplace=True)
        align_indices = receptor_alignment_indices(traj.topology, manifest)
        if len(align_indices) >= 3:
            traj.superpose(traj, 0, atom_indices=align_indices)
        binder_indices = binder_rmsd_indices(traj.topology, manifest)
        if not binder_indices:
            return []
        rmsd_nm = md.rmsd(traj, traj, 0, atom_indices=binder_indices)
    except Exception:
        return []
    return [{"time_ps": float(traj.time[idx]), "binder_rmsd_angstrom": float(value) * 10.0} for idx, value in enumerate(rmsd_nm)]


def md_energy_landscape_svg(rows: list[dict[str, Any]]) -> str:
    width = 900
    first_plot_top = 78
    panel_height = 205
    plot_h = 122
    height = first_plot_top + panel_height * 2
    left, right = 82, 34
    points = md_energy_points(rows)
    if not points:
        return "\n".join(
            [
                svg_header(width, height),
                '<text x="24" y="30" class="title">MD RMSD / potential trace</text>',
                '<text x="24" y="64" class="tick">No paired RMSD/energy rows were found.</text>',
                "</svg>",
            ]
        )

    by_replica = md_energy_points_by_replica(points)
    legend_rows = max(1, math.ceil(len(by_replica) / 4))
    legend_height = 34 + legend_rows * 18
    height = first_plot_top + panel_height * 2 + legend_height
    basin = min(points, key=lambda point: float(point["y"]))
    time_values = [float(point["time_ns"]) for point in points]
    time_data_min, time_data_max = min(time_values), max(time_values)
    x_min, x_max = padded_range(time_values, include_zero=True)
    plot_w = width - left - right
    colors = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#dc2626", "#0891b2"]

    def map_x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    parts = [
        svg_header(width, height),
        '<text x="24" y="30" class="title">MD RMSD / potential trace</text>',
    ]
    panels = [
        ("x", "Binder RMSD to production start (angstrom)", "rmsd"),
        ("y", "Delta potential (kJ/mol)", "energy"),
    ]
    for panel_idx, (key, label, kind) in enumerate(panels):
        top = first_plot_top + panel_idx * panel_height
        base_y = top + plot_h
        values = [float(point[key]) for point in points]
        y_min, y_max = padded_range(values, include_zero=(kind == "energy"))
        y_data_min, y_data_max = min(values), max(values)

        def map_y(value: float) -> float:
            return base_y - (value - y_min) / (y_max - y_min) * plot_h

        parts.extend(
            [
                f'<text x="{left}" y="{top - 18}" class="subtitle">{escape(label)}</text>',
                f'<line x1="{left}" y1="{base_y}" x2="{left + plot_w}" y2="{base_y}" class="axis"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{base_y}" class="axis"/>',
            ]
        )
        for tick in axis_ticks(time_data_min, time_data_max, max_ticks=7, map_value=map_x, min_px=48, include_low=True, include_high=True):
            x_tick = map_x(float(tick))
            parts.extend(
                [
                    f'<line x1="{x_tick:.2f}" y1="{base_y}" x2="{x_tick:.2f}" y2="{base_y + 4}" class="axis"/>',
                    f'<text x="{x_tick:.2f}" y="{base_y + 20}" class="tick" text-anchor="middle">{tick_label(tick)}</text>',
                ]
            )
        for tick in axis_ticks(y_data_min, y_data_max, max_ticks=5, map_value=map_y, min_px=18):
            y_tick = map_y(float(tick))
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y_tick:.2f}" x2="{left + plot_w}" y2="{y_tick:.2f}" class="grid"/>',
                    f'<text x="{left - 8}" y="{y_tick + 4:.2f}" class="tick" text-anchor="end">{tick_label(tick)}</text>',
                ]
            )
        basin_x = map_x(float(basin["time_ns"]))
        basin_y = map_y(float(basin[key]))
        parts.append(f'<line x1="{basin_x:.2f}" y1="{top}" x2="{basin_x:.2f}" y2="{base_y}" class="basin-line"/>')
        for idx, (replica, rep_points) in enumerate(sorted(by_replica.items())):
            mapped = [f'{map_x(float(point["time_ns"])):.2f},{map_y(float(point[key])):.2f}' for point in rep_points]
            if not mapped:
                continue
            color = colors[idx % len(colors)]
            parts.append(f'<polyline class="time-trace" points="{" ".join(mapped)}" fill="none" stroke="{color}" stroke-width="2.2" opacity="0.82"><title>{escape(replica)} {escape(label)}</title></polyline>')
        parts.append(
            f'<circle class="basin-point" cx="{basin_x:.2f}" cy="{basin_y:.2f}" r="4.5">'
            f'<title>Basin: {escape(str(basin["replica"]))} frame {float(basin["frame"]):.0f}, {float(basin["time_ns"]):.2f} ns, {float(basin["x"]):.2f} angstrom, {float(basin["y"]):.2f} kJ/mol</title>'
            "</circle>"
        )
        if panel_idx == 1:
            parts.append(f'<text x="{left + plot_w / 2:.2f}" y="{base_y + 40}" class="tick" text-anchor="middle">Time (ns)</text>')
    legend_y = height - legend_height + 32
    legend_x = left
    for idx, replica in enumerate(sorted(by_replica)):
        if idx and idx % 4 == 0:
            legend_y += 18
            legend_x = left
        color = colors[idx % len(colors)]
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2.2"/>',
                f'<text x="{legend_x + 31}" y="{legend_y}" class="legend">{escape(replica)}</text>',
            ]
        )
        legend_x += 150
    parts.append(
        f'<text x="{left}" y="{height - legend_height + 10}" class="basin-label">'
        f'Basin {escape(str(basin["replica"]))} frame {float(basin["frame"]):.0f}: {float(basin["time_ns"]):.2f} ns, {float(basin["x"]):.2f} A, {float(basin["y"]):.2f} kJ/mol'
        "</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def md_energy_basin_svg(rows: list[dict[str, Any]]) -> str:
    width = 900
    left, right, top = 82, 32, 92
    plot_h = 235
    x_label_y = top + plot_h + 42
    points = md_energy_points(rows)
    height = x_label_y + 58
    if not points:
        return "\n".join(
            [
                svg_header(width, height),
                '<text x="24" y="30" class="title">1D RMSD basin</text>',
                '<text x="24" y="64" class="tick">No paired RMSD/energy rows were found.</text>',
                "</svg>",
            ]
        )
    basin_rows = md_energy_basin_rows(rows)
    if not basin_rows:
        return "\n".join(
            [
                svg_header(width, height),
                '<text x="24" y="30" class="title">1D RMSD basin</text>',
                '<text x="24" y="64" class="tick">Not enough RMSD samples for a basin curve.</text>',
                "</svg>",
            ]
        )
    by_replica: dict[str, list[dict[str, Any]]] = {}
    for row in basin_rows:
        by_replica.setdefault(str(row["replica"]), []).append(row)
    for rep_rows in by_replica.values():
        rep_rows.sort(key=lambda row: float(row["rmsd_bin_center_angstrom"]))
    legend_rows = max(1, math.ceil(len(by_replica) / 4))
    legend_top = x_label_y + 28
    height = legend_top + legend_rows * 18 + 12
    xs = [float(point["rmsd_bin_center_angstrom"]) for point in basin_rows]
    ys = [float(point["free_energy_kJ_mol"]) for point in basin_rows]
    x_data_min, x_data_max = min(xs), max(xs)
    y_data_min, y_data_max = min(ys), max(ys)
    x_min, x_max = padded_range(xs, include_zero=False)
    y_min, y_max = padded_range(ys, include_zero=True)
    plot_w = width - left - right
    base_y = top + plot_h

    def map_x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def map_y(value: float) -> float:
        return base_y - (value - y_min) / (y_max - y_min) * plot_h

    parts = [
        svg_header(width, height),
        '<text x="24" y="30" class="title">1D RMSD basin</text>',
        '<text x="24" y="48" class="tick">Per-replica occupancy-derived free-energy profiles from MD RMSD; QC view only.</text>',
        f'<text x="{left}" y="{top - 18}" class="subtitle">Free energy (kJ/mol)</text>',
        f'<line x1="{left}" y1="{base_y}" x2="{left + plot_w}" y2="{base_y}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{base_y}" class="axis"/>',
        f'<text x="{left + plot_w / 2:.2f}" y="{x_label_y:.2f}" class="tick" text-anchor="middle">Binder RMSD to production start (angstrom)</text>',
    ]
    for tick in axis_ticks(x_data_min, x_data_max, max_ticks=7, map_value=map_x, min_px=48, include_low=True, include_high=True):
        x_tick = map_x(float(tick))
        parts.extend(
            [
                f'<line x1="{x_tick:.2f}" y1="{base_y}" x2="{x_tick:.2f}" y2="{base_y + 4}" class="axis"/>',
                f'<text x="{x_tick:.2f}" y="{base_y + 20}" class="tick" text-anchor="middle">{tick_label(tick)}</text>',
            ]
        )
    for tick in axis_ticks(y_data_min, y_data_max, max_ticks=6, map_value=map_y, min_px=18):
        y_tick = map_y(float(tick))
        parts.extend(
            [
                f'<line x1="{left}" y1="{y_tick:.2f}" x2="{left + plot_w}" y2="{y_tick:.2f}" class="grid"/>',
                f'<text x="{left - 8}" y="{y_tick + 4:.2f}" class="tick" text-anchor="end">{tick_label(tick)}</text>',
            ]
        )
    for idx, (replica, rep_rows) in enumerate(sorted(by_replica.items())):
        color = REPLICA_TRACE_COLORS[idx % len(REPLICA_TRACE_COLORS)]
        curve_points = [f'{map_x(float(point["rmsd_bin_center_angstrom"])):.2f},{map_y(float(point["free_energy_kJ_mol"])):.2f}' for point in rep_rows]
        if not curve_points:
            continue
        basin = next((point for point in rep_rows if int(numeric(point.get("is_basin")) or 0) == 1), min(rep_rows, key=lambda point: float(point["free_energy_kJ_mol"])))
        basin_x = map_x(float(basin["rmsd_bin_center_angstrom"]))
        basin_y = map_y(float(basin["free_energy_kJ_mol"]))
        parts.append(f'<line x1="{basin_x:.2f}" y1="{top}" x2="{basin_x:.2f}" y2="{base_y}" class="basin-line" style="stroke:{color};opacity:.58"/>')
        parts.append(
            f'<polyline class="basin-curve" points="{" ".join(curve_points)}" fill="none" stroke="{color}" stroke-width="2.4" opacity="0.86">'
            f'<title>{escape(replica)} 1D RMSD basin</title></polyline>'
        )
        parts.append(
            f'<circle class="basin-point" cx="{basin_x:.2f}" cy="{basin_y:.2f}" r="4.7" style="fill:{color}">'
            f'<title>{escape(replica)} basin center: {float(basin["rmsd_bin_center_angstrom"]):.2f} angstrom, {float(basin["free_energy_kJ_mol"]):.2f} kJ/mol, n={int(numeric(basin.get("count")) or 0)}</title>'
            "</circle>"
        )
    legend_y = legend_top
    legend_x = left
    for idx, replica in enumerate(sorted(by_replica)):
        if idx and idx % 4 == 0:
            legend_y += 18
            legend_x = left
        color = REPLICA_TRACE_COLORS[idx % len(REPLICA_TRACE_COLORS)]
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2.4"/>',
                f'<text x="{legend_x + 31}" y="{legend_y}" class="legend">{escape(replica)}</text>',
            ]
        )
        legend_x += 150
    parts.append("</svg>")
    return "\n".join(parts)


def md_energy_basin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = md_energy_points(rows)
    by_replica = md_energy_points_by_replica(points)
    out: list[dict[str, Any]] = []
    for replica, rep_points in sorted(by_replica.items()):
        curve = rmsd_free_energy_curve(rep_points)
        if not curve:
            continue
        basin_index, _ = min(enumerate(curve), key=lambda item: float(item[1]["free_energy_kJ_mol"]))
        for idx, point in enumerate(curve):
            out.append(
                {
                    "replica": replica,
                    "rmsd_bin_center_angstrom": float(point["x"]),
                    "free_energy_kJ_mol": float(point["free_energy_kJ_mol"]),
                    "count": int(point["count"]),
                    "is_basin": 1 if idx == basin_index else 0,
                }
            )
    return out


def rmsd_free_energy_curve(points: list[dict[str, Any]], *, bins: int = 40, temperature_k: float = 298.15) -> list[dict[str, float]]:
    rmsd_values = [float(point["x"]) for point in points if numeric(point.get("x")) is not None]
    if len(rmsd_values) < 2:
        return []
    low, high = min(rmsd_values), max(rmsd_values)
    if low == high:
        return [{"x": low, "free_energy_kJ_mol": 0.0, "count": float(len(rmsd_values))}]
    bin_count = max(6, min(bins, int(math.sqrt(len(rmsd_values)) * 4)))
    width = (high - low) / bin_count
    counts = [0] * bin_count
    for value in rmsd_values:
        index = min(bin_count - 1, max(0, int((value - low) / width)))
        counts[index] += 1
    total = sum(counts)
    if total == 0:
        return []
    kb_kj_mol_k = 0.00831446261815324
    raw: list[dict[str, float]] = []
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        probability = count / total
        free_energy = -kb_kj_mol_k * temperature_k * math.log(probability)
        raw.append({"x": low + (index + 0.5) * width, "free_energy_kJ_mol": free_energy, "count": float(count)})
    if not raw:
        return []
    minimum = min(point["free_energy_kJ_mol"] for point in raw)
    for point in raw:
        point["free_energy_kJ_mol"] -= minimum
    return raw


def md_energy_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        rmsd = numeric(row.get("binder_rmsd_angstrom"))
        energy = numeric(row.get("delta_potential_kJ_mol"))
        if rmsd is None or energy is None:
            continue
        frame = numeric(row.get("frame")) or 0.0
        time_ps = numeric(row.get("time_ps"))
        points.append(
            {
                "replica": str(row.get("replica") or "replica"),
                "frame": frame,
                "time_ps": time_ps if time_ps is not None else frame,
                "time_ns": (time_ps / 1000.0) if time_ps is not None else frame,
                "x": rmsd,
                "y": energy,
            }
        )
    return points


def md_energy_points_by_replica(points: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_replica: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_replica.setdefault(str(point["replica"]), []).append(point)
    for rep_points in by_replica.values():
        rep_points.sort(key=lambda point: (float(point["time_ps"]), float(point["frame"])))
    return by_replica


def receptor_alignment_indices(topology: Any, manifest: dict[str, Any]) -> list[int]:
    binder = set(binder_atom_indices(topology, manifest))
    indices = []
    try:
        selected = list(topology.select("protein and (name N or name CA or name C)"))
    except Exception:
        selected = []
    for idx in selected:
        if int(idx) not in binder:
            indices.append(int(idx))
    if indices:
        return indices
    try:
        return [int(idx) for idx in topology.select("protein and name CA")]
    except Exception:
        return []


def binder_rmsd_indices(topology: Any, manifest: dict[str, Any]) -> list[int]:
    binder = binder_atom_indices(topology, manifest)
    if not binder:
        return []
    ca_indices = [idx for idx in binder if atom_name(topology, idx) == "CA"]
    if ca_indices and binder_is_peptide_like(topology, binder, manifest):
        return ca_indices
    return [idx for idx in binder if atom_element(topology, idx) != "H"]


def binder_atom_indices(topology: Any, manifest: dict[str, Any]) -> list[int]:
    ranged = manifest_atom_range_indices(topology, manifest, "binder_atom_range", "peptide_atom_range", "ligand_atom_range")
    if ranged:
        return ranged
    resname = str(manifest.get("ligand_resname") or "").strip()
    if resname:
        indices = [atom.index for atom in topology.atoms if atom.residue.name.upper() == resname.upper()]
        if indices:
            return sorted(indices)
    non_protein = [atom.index for atom in topology.atoms if not atom.residue.is_water and not atom.residue.is_protein]
    return sorted(non_protein)


def manifest_atom_range_indices(topology: Any, manifest: dict[str, Any], *keys: str) -> list[int]:
    atom_count = topology.n_atoms
    for key in keys:
        value = manifest.get(key)
        if not isinstance(value, list | tuple) or len(value) != 2:
            continue
        try:
            start = max(0, int(value[0]) - 1)
            stop = min(atom_count, int(value[1]))
        except (TypeError, ValueError):
            continue
        if start < stop:
            return list(range(start, stop))
    return []


def binder_is_peptide_like(topology: Any, indices: list[int], manifest: dict[str, Any]) -> bool:
    if manifest.get("peptide_atom_range") or manifest.get("peptide_residue_mask"):
        return True
    residues = {topology.atom(idx).residue.index for idx in indices}
    ca_count = sum(1 for idx in indices if atom_name(topology, idx) == "CA")
    return len(residues) > 1 and ca_count >= 2


def atom_name(topology: Any, index: int) -> str:
    return str(topology.atom(index).name).upper()


def atom_element(topology: Any, index: int) -> str:
    element = topology.atom(index).element
    return str(getattr(element, "symbol", "") or "").upper()


def energy_basin_rows(job: Path, qc_rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    partner_column = next((column for column, _ in PARTNER_COLUMNS if any(column in row for row in qc_rows)), "")
    if not partner_column:
        return []
    rep_files = sorted((job / "analysis" / "mmpbsa").glob("rep*/per_frame_energy.csv"))
    rows: list[dict[str, Any]] = []
    qc_offset = 0
    for rep_path in rep_files:
        energies = parse_per_frame_energy(rep_path)
        frames = sorted({frame for model in energies.values() for frame in model})
        if not frames:
            continue
        replica = rep_path.parent.name
        for idx, frame in enumerate(frames):
            qc = qc_rows[qc_offset + idx] if qc_offset + idx < len(qc_rows) else {}
            row: dict[str, Any] = {
                "replica": replica,
                "frame": frame,
                "sample_index": qc_offset + idx + 1,
                "partner_rmsd_angstrom": qc.get(partner_column, ""),
                "receptor_rmsd_angstrom": qc.get("receptor_bb_rmsd_angstrom", ""),
            }
            gb = energies.get("GB", {}).get(frame)
            pb = energies.get("PB", {}).get(frame)
            if gb is not None:
                row["GB_delta_total_kcal_mol"] = gb
            if pb is not None:
                row["PB_delta_total_kcal_mol"] = pb
            if "GB_delta_total_kcal_mol" in row or "PB_delta_total_kcal_mol" in row:
                rows.append(row)
        qc_offset += len(frames)
    return rows


def parse_per_frame_energy(path: Path) -> dict[str, dict[int, float]]:
    energies: dict[str, dict[int, float]] = {"GB": {}, "PB": {}}
    model: str | None = None
    in_delta = False
    header: list[str] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("GENERALIZED BORN"):
            model = "GB"
            in_delta = False
            header = None
            continue
        if line.startswith("POISSON BOLTZMANN"):
            model = "PB"
            in_delta = False
            header = None
            continue
        if line.startswith("DELTA Energy Terms"):
            in_delta = True
            header = None
            continue
        if not in_delta or model is None:
            continue
        values = [item.strip() for item in line.split(",")]
        if values and values[0] == "Frame #":
            header = values
            continue
        if header is None or not values or not values[0].lstrip("-").isdigit():
            continue
        try:
            frame = int(values[0])
            idx = header.index("DELTA TOTAL")
            value = float(values[idx])
        except (ValueError, IndexError):
            continue
        energies.setdefault(model, {})[frame] = value
    return energies


def energy_basin_svg(rows: list[dict[str, Any]]) -> str:
    scatter = []
    for row in rows:
        rmsd = numeric(row.get("partner_rmsd_angstrom"))
        energy = numeric(row.get("GB_delta_total_kcal_mol"))
        if rmsd is not None and energy is not None:
            scatter.append({"x": rmsd, "y": energy, "label": f"{row.get('replica')} frame {row.get('frame')}"})
    return scatter_svg(scatter, "Energy basin: GB delta total vs partner RMSD", "Partner RMSD (angstrom)", "GB delta total (kcal/mol)")


def interaction_contact_rows(pdb_path: Path, manifest: dict[str, Any] | None = None, *, stride: int = 1) -> list[dict[str, Any]]:
    models = parse_pdb_models(pdb_path)
    if not models:
        return []
    rows: list[dict[str, Any]] = []
    for frame, atoms in enumerate(models, start=1):
        receptor = [atom for atom in atoms if atom["chain"] == "A" and atom["element"] != "H"]
        binder = [atom for atom in atoms if atom["chain"] == "B" and atom["element"] != "H"]
        if not receptor or not binder:
            continue
        hbond_count = count_hbond_like_contacts(receptor, binder)
        salt_count = count_salt_bridge_like_contacts(receptor, binder)
        rows.append(
            {
                "frame": frame,
                "hbond_like_contacts": hbond_count,
                "salt_bridge_like_contacts": salt_count,
            }
        )
    if manifest:
        return normalize_replica_rows(rows, manifest, stride=max(1, int(stride)))
    return rows


def interaction_contacts_svg(rows: list[dict[str, Any]]) -> str:
    return replica_line_panels_svg(
        rows,
        [
            (
                "Interaction contacts",
                [
                    ("hbond_like_contacts", "H-bond-like"),
                    ("salt_bridge_like_contacts", "Salt-bridge-like"),
                ],
            )
        ],
        "Interaction contacts",
    )


def parse_pdb_models(path: Path) -> list[list[dict[str, Any]]]:
    models: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    saw_model = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            saw_model = True
            if current:
                models.append(current)
                current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                models.append(current)
                current = []
            continue
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        element = (line[76:78].strip() if len(line) >= 78 else "") or re.sub(r"[^A-Za-z]", "", line[12:16]).strip()[:1]
        try:
            atom = {
                "atom": line[12:16].strip(),
                "resn": line[17:20].strip().upper(),
                "chain": (line[21].strip() or ""),
                "resi": line[22:26].strip(),
                "element": element.upper(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
            }
        except ValueError:
            continue
        current.append(atom)
    if current:
        models.append(current)
    if not saw_model and len(models) > 1:
        return [models[0]]
    return models


def count_hbond_like_contacts(receptor: list[dict[str, Any]], binder: list[dict[str, Any]]) -> int:
    rec_polar = [atom for atom in receptor if atom["element"] in {"N", "O", "S"}]
    binder_polar = [atom for atom in binder if atom["element"] in {"N", "O", "S"}]
    return count_pairs_within(rec_polar, binder_polar, 3.5)


def count_salt_bridge_like_contacts(receptor: list[dict[str, Any]], binder: list[dict[str, Any]]) -> int:
    rec_pos, rec_neg = charged_atom_sets(receptor)
    bind_pos, bind_neg = charged_atom_sets(binder)
    return count_pairs_within(rec_pos, bind_neg, 4.0) + count_pairs_within(rec_neg, bind_pos, 4.0)


def charged_atom_sets(atoms: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for atom in atoms:
        resn = str(atom["resn"])
        element = str(atom["element"])
        atom_name = str(atom["atom"]).upper()
        if resn in CHARGED_POSITIVE_RESIDUES and element == "N":
            positive.append(atom)
        elif resn in CHARGED_NEGATIVE_RESIDUES and element == "O":
            negative.append(atom)
        elif resn not in CHARGED_POSITIVE_RESIDUES | CHARGED_NEGATIVE_RESIDUES:
            if element == "N" and atom_name.startswith("N"):
                positive.append(atom)
            elif element == "O" and atom_name.startswith("O"):
                negative.append(atom)
    return positive, negative


def count_pairs_within(group_a: list[dict[str, Any]], group_b: list[dict[str, Any]], cutoff: float) -> int:
    cutoff2 = cutoff * cutoff
    count = 0
    for atom_a in group_a:
        ax, ay, az = float(atom_a["x"]), float(atom_a["y"]), float(atom_a["z"])
        for atom_b in group_b:
            dx = ax - float(atom_b["x"])
            dy = ay - float(atom_b["y"])
            dz = az - float(atom_b["z"])
            if dx * dx + dy * dy + dz * dz <= cutoff2:
                count += 1
    return count


def qc_metric_rows(rows: list[dict[str, Any]], *, thresholds: dict[str, float] | None = None) -> list[dict[str, Any]]:
    threshold_values = thresholds or {}
    metrics: list[dict[str, Any]] = []
    columns = [column for column in rows[0] if column not in ROW_METADATA_COLUMNS]
    for column in columns:
        values = [float(value) for value in (numeric(row.get(column)) for row in rows) if value is not None]
        if not values:
            continue
        metrics.append(
            {
                "metric": column,
                "label": qc_metric_label(column),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "first": values[0],
                "last": values[-1],
                "threshold": threshold_values.get(column, ""),
            }
        )
    return metrics


def qc_metric_label(column: str) -> str:
    labels = {
        "receptor_bb_rmsd_angstrom": "Receptor backbone RMSD (angstrom)",
        "ligand_heavy_rmsd_after_receptor_fit_angstrom": "Ligand heavy RMSD after receptor fit (angstrom)",
        "peptide_bb_rmsd_after_receptor_fit_angstrom": "Peptide backbone RMSD after receptor fit (angstrom)",
    }
    if column in labels:
        return labels[column]
    if column.endswith("[native]"):
        return "Native contacts"
    if column.endswith("[mindist]"):
        return "Interface minimum distance (angstrom)"
    return pretty_field_label(column)


def infer_qc_thresholds(job_dir: Path) -> dict[str, float]:
    manifest = read_optional_json(job_dir / "manifest.json")
    qc = ((manifest.get("profile") or {}).get("qc") or {}) if manifest else {}
    thresholds: dict[str, float] = {}
    mappings = {
        "receptor_bb_rmsd_angstrom": "receptor_rmsd_fail_angstrom",
        "ligand_heavy_rmsd_after_receptor_fit_angstrom": "ligand_rmsd_warn_angstrom",
        "peptide_bb_rmsd_after_receptor_fit_angstrom": "peptide_rmsd_warn_angstrom",
        "rec_lig[native]": "native_contacts_fail_min",
        "rec_pep[native]": "native_contacts_fail_min",
        "rec_lig[mindist]": "interface_distance_fail_angstrom",
        "rec_pep[mindist]": "interface_distance_fail_angstrom",
    }
    for column, key in mappings.items():
        value = numeric(qc.get(key))
        if value is not None:
            thresholds[column] = value
    return thresholds


def completed_summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(run_dir.glob("*/result/summary.json")):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        row["job_dir"] = str(summary_path.parents[1])
        row.setdefault("job_id", summary_path.parents[1].name)
        rows.append(row)
    return rows


def sort_summary_rows(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == COMPOSITE_SORT:
        return sorted(rows, key=lambda row: (*composite_sort_key(row), str(row.get("job_id") or "")))
    return sorted(rows, key=lambda row: (numeric(row.get(sort_by)) is None, numeric(row.get(sort_by)) or 0.0, str(row.get("job_id") or "")))


def composite_sort_key(row: dict[str, Any]) -> tuple[tuple[bool, float], ...]:
    key: list[tuple[bool, float]] = []
    for field in COMPOSITE_SORT_FIELDS:
        value = numeric(row.get(field))
        key.append((value is None, value if value is not None else 0.0))
    return tuple(key)


def ranking_row(row: dict[str, Any], sort_by: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "job_id": row.get("job_id", ""),
        "status": row.get("status", ""),
        "mmpbsa_qc_status": row.get("mmpbsa_qc_status", ""),
        "replica_count": row.get("replica_count", ""),
        "mmpbsa_frames": row.get("mmpbsa_frames", ""),
    }
    if sort_by != COMPOSITE_SORT:
        out[sort_by] = row.get(sort_by, "")
    sd_key = f"{sort_by}_replica_sd"
    if sd_key in row:
        out[sd_key] = row.get(sd_key, "")
    for key, _, replica_sd in SCORE_FIELDS:
        if key not in out and key in row:
            out[key] = row.get(key, "")
        if replica_sd not in out and replica_sd in row:
            out[replica_sd] = row.get(replica_sd, "")
    return out


def run_qc_row(job_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    qc = read_optional_json(job_dir / "analysis" / "qc" / "summary.json")
    qc_csv = job_dir / "analysis" / "qc" / "trajectory_qc.csv"
    metric_summary: dict[str, dict[str, float]] = {}
    if qc_csv.exists():
        try:
            metric_summary = {row["metric"]: row for row in qc_metric_rows(read_numeric_csv(qc_csv))}
        except SystemExit:
            metric_summary = {}
    partner_key = "ligand_heavy_rmsd_after_receptor_fit_angstrom"
    partner_label = "ligand_rmsd"
    if "peptide_bb_rmsd_after_receptor_fit_angstrom" in qc:
        partner_key = "peptide_bb_rmsd_after_receptor_fit_angstrom"
        partner_label = "peptide_rmsd"
    receptor = qc.get("receptor_bb_rmsd_angstrom", {})
    partner = qc.get(partner_key, {})
    native = first_metric_with_suffix(metric_summary, "[native]")
    mindist = first_metric_with_suffix(metric_summary, "[mindist]")
    return {
        "job_id": summary.get("job_id", job_dir.name),
        "status": summary.get("status", ""),
        "trajectory_qc_status": summary.get("trajectory_qc_status", ""),
        "receptor_rmsd_mean": receptor.get("mean", ""),
        "receptor_rmsd_max": receptor.get("max", ""),
        f"{partner_label}_mean": partner.get("mean", ""),
        f"{partner_label}_max": partner.get("max", ""),
        "native_contacts_mean": native.get("mean", "") if native else "",
        "native_contacts_min": native.get("min", "") if native else "",
        "native_contacts_last": native.get("last", "") if native else "",
        "interface_distance_min": mindist.get("min", "") if mindist else "",
        "interface_distance_last": mindist.get("last", "") if mindist else "",
        "qc_issue_count": len(qc.get("issues", [])) if qc else "",
    }


def first_metric_with_suffix(metrics: dict[str, dict[str, float]], suffix: str) -> dict[str, float] | None:
    for key, row in metrics.items():
        if key.endswith(suffix):
            return row
    return None


def ranking_svg_plot(rows: list[dict[str, Any]], sort_by: str) -> str:
    bars = []
    if sort_by == COMPOSITE_SORT:
        for idx, row in enumerate(rows, start=1):
            job_id = str(row.get("job_id") or "")
            bars.append({"label": compact_job_id(job_id), "full_label": job_id, "value": float(idx), "sd": None})
        return bar_svg(bars, "Composite rank", "rank", palette="qc")
    sd_key = f"{sort_by}_replica_sd"
    for row in rows:
        value = numeric(row.get(sort_by))
        if value is not None:
            job_id = str(row.get("job_id") or "")
            bars.append({"label": compact_job_id(job_id), "full_label": job_id, "value": value, "sd": numeric(row.get(sd_key))})
    if not bars:
        bars = [{"label": compact_job_id(str(row.get("job_id") or "")), "full_label": str(row.get("job_id") or ""), "value": float(idx + 1), "sd": None} for idx, row in enumerate(rows)]
    return bar_svg(bars, f"Run ranking by {pretty_field_label(sort_by)}", pretty_field_label(sort_by))


def qc_overview_svg(rows: list[dict[str, Any]]) -> str:
    bars = []
    for row in rows:
        receptor = numeric(row.get("receptor_rmsd_mean"))
        ligand = numeric(row.get("ligand_rmsd_mean"))
        peptide = numeric(row.get("peptide_rmsd_mean"))
        value = ligand if ligand is not None else peptide
        if value is None:
            value = receptor
        if value is not None:
            job_id = str(row.get("job_id") or "")
            bars.append({"label": compact_job_id(job_id), "full_label": job_id, "value": value, "sd": None})
    if not bars:
        bars = [{"label": compact_job_id(str(row.get("job_id") or "")), "full_label": str(row.get("job_id") or ""), "value": 0.0, "sd": None} for row in rows]
    return bar_svg(bars, "Run partner RMSD mean", "angstrom", palette="qc")


def first_column(rows: list[dict[str, float]], suffix: str) -> str | None:
    for key in rows[0]:
        if key.endswith(suffix):
            return key
    return None


def replica_line_panels_svg(rows: list[dict[str, Any]], panels: list[tuple[str, list[tuple[str, str]]]], title: str, thresholds: dict[str, float] | None = None) -> str:
    width = 900
    left, right = 82, 26
    first_plot_top = 78
    panel_height = 190
    plot_h = 112
    active_panels = [(label, metrics) for label, metrics in panels if any(column_has_numeric(rows, column) for column, _ in metrics)]
    if not rows or not active_panels:
        return "\n".join(
            [
                svg_header(width, 120),
                f'<text x="24" y="30" class="title">{escape(title)}</text>',
                '<text x="24" y="64" class="tick">No trajectory rows were found.</text>',
                "</svg>",
            ]
        )

    groups = rows_by_replica(rows)
    replicas = sorted(groups)
    legend_rows = max(1, math.ceil(len(replicas) / 4))
    style_metric_labels = next(([label for _, label in metrics[:2]] for _, metrics in active_panels if len(metrics) > 1), [])
    style_legend_height = 24 if style_metric_labels else 0
    legend_height = 42 + legend_rows * 18 + style_legend_height
    height = first_plot_top + panel_height * len(active_panels) + legend_height
    plot_w = width - left - right
    threshold_values = thresholds or {}
    x_key = "replica_frame" if any(numeric(row.get("replica_frame")) is not None for row in rows) else "frame"
    x_values = [numeric(row.get(x_key)) for row in rows]
    x_numbers = [float(value) for value in x_values if value is not None]
    x_data_min, x_data_max = min(x_numbers or [1.0]), max(x_numbers or [1.0])
    x_min, x_max = padded_range(x_numbers or [1.0], include_zero=False)

    def map_x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    parts = [
        svg_header(width, height),
        f'<text x="24" y="30" class="title">{escape(title)}</text>',
    ]
    for panel_idx, (panel_label, metrics) in enumerate(active_panels):
        top = first_plot_top + panel_idx * panel_height
        base_y = top + plot_h
        metric_values: list[float] = []
        for column, _ in metrics:
            metric_values.extend(column_numeric_values(rows, column))
            threshold = numeric(threshold_values.get(column))
            if threshold is not None:
                metric_values.append(threshold)
        y_min, y_max = padded_range(metric_values or [0.0], include_zero=False)
        y_data_min = min(metric_values or [0.0])
        y_data_max = max(metric_values or [0.0])

        def map_y(value: float) -> float:
            return base_y - (value - y_min) / (y_max - y_min) * plot_h

        parts.extend(
            [
                f'<text x="{left}" y="{top - 18}" class="subtitle">{escape(panel_label)}</text>',
                f'<line x1="{left}" y1="{base_y}" x2="{left + plot_w}" y2="{base_y}" class="axis"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{base_y}" class="axis"/>',
            ]
        )
        x_ticks = axis_ticks(x_data_min, x_data_max, max_ticks=8, map_value=map_x, min_px=48, integer=True, include_low=True, include_high=True)
        for tick in x_ticks:
            x_tick = map_x(float(tick))
            parts.extend(
                [
                    f'<line x1="{x_tick:.2f}" y1="{base_y}" x2="{x_tick:.2f}" y2="{base_y + 4}" class="axis"/>',
                    f'<text x="{x_tick:.2f}" y="{base_y + 20}" class="tick" text-anchor="middle">{tick_label(tick)}</text>',
                ]
            )
        for tick in axis_ticks(y_data_min, y_data_max, max_ticks=5, map_value=map_y, min_px=18):
            y_tick = map_y(float(tick))
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y_tick:.2f}" x2="{left + plot_w}" y2="{y_tick:.2f}" class="grid"/>',
                    f'<text x="{left - 8}" y="{y_tick + 4:.2f}" class="tick" text-anchor="end">{tick_label(tick)}</text>',
                ]
            )
        for metric_idx, (column, metric_label) in enumerate(metrics):
            threshold = numeric(threshold_values.get(column))
            if threshold is not None:
                y_threshold = map_y(threshold)
                label = f"threshold {tick_label(threshold)}"
                label_y = min(max(y_threshold - 5, top + 13), base_y - 7)
                parts.extend(
                    [
                        f'<line x1="{left}" y1="{y_threshold:.2f}" x2="{left + plot_w}" y2="{y_threshold:.2f}" class="threshold"/>',
                        *svg_label_with_background(label, left + plot_w - 8, label_y, "threshold-label", anchor="end"),
                    ]
                )
            dash = "" if metric_idx == 0 else ' stroke-dasharray="6 4"'
            opacity = 0.90 if metric_idx == 0 else 0.68
            for replica_idx, replica in enumerate(replicas):
                points = []
                for row in groups[replica]:
                    x_value = numeric(row.get(x_key))
                    y_value = numeric(row.get(column))
                    if x_value is None or y_value is None:
                        continue
                    points.append(f"{map_x(x_value):.2f},{map_y(y_value):.2f}")
                if not points:
                    continue
                color = REPLICA_TRACE_COLORS[replica_idx % len(REPLICA_TRACE_COLORS)]
                parts.append(
                    f'<polyline class="replica-trace" points="{" ".join(points)}" fill="none" stroke="{color}" '
                    f'stroke-width="2.1" opacity="{opacity:.2f}"{dash}><title>{escape(replica)} {escape(metric_label)}</title></polyline>'
                )
    parts.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - legend_height + 10}" class="tick" text-anchor="middle">Frame within replica</text>')
    legend_y = height - legend_height + 32
    legend_x = left
    for idx, replica in enumerate(replicas):
        if idx and idx % 4 == 0:
            legend_y += 18
            legend_x = left
        color = REPLICA_TRACE_COLORS[idx % len(REPLICA_TRACE_COLORS)]
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2.2"/>',
                f'<text x="{legend_x + 31}" y="{legend_y}" class="legend">{escape(replica)}</text>',
            ]
        )
        legend_x += 150
    if style_legend_height:
        style_y = legend_y + 18
        solid_label = style_metric_labels[0] if style_metric_labels else "primary metric"
        dashed_label = style_metric_labels[1] if len(style_metric_labels) > 1 else "secondary metric"
        parts.extend(
            [
                f'<line x1="{left}" y1="{style_y - 4}" x2="{left + 24}" y2="{style_y - 4}" stroke="#111827" stroke-width="2.0"/>',
                f'<text x="{left + 31}" y="{style_y}" class="legend">solid: {escape(solid_label)}</text>',
                f'<line x1="{left + 230}" y1="{style_y - 4}" x2="{left + 254}" y2="{style_y - 4}" stroke="#111827" stroke-width="2.0" stroke-dasharray="6 4"/>',
                f'<text x="{left + 261}" y="{style_y}" class="legend">dashed: {escape(dashed_label)}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def rows_by_replica(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        replica = str(row.get("replica") or "trajectory")
        groups.setdefault(replica, []).append(row)
    for values in groups.values():
        values.sort(key=lambda row: (numeric(row.get("replica_frame")) or numeric(row.get("frame")) or 0.0, numeric(row.get("global_frame")) or 0.0))
    return groups


def column_has_numeric(rows: list[dict[str, Any]], column: str) -> bool:
    return any(numeric(row.get(column)) is not None for row in rows)


def column_numeric_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    return [float(value) for value in (numeric(row.get(column)) for row in rows) if value is not None]


def tick_label(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def stacked_line_svg(rows: list[dict[str, float]], series: list[tuple[str, str]], title: str, thresholds: dict[str, float] | None = None) -> str:
    width = 900
    panel_height = 165
    margin = {"left": 72, "right": 24, "top": 34, "bottom": 34}
    height = 54 + panel_height * len(series)
    frame_values = [row.get("frame", idx + 1.0) for idx, row in enumerate(rows)]
    x_min, x_max = min(frame_values), max(frame_values)
    if x_min == x_max:
        x_max = x_min + 1.0
    parts = [svg_header(width, height), f'<text x="24" y="28" class="title">{escape(title)}</text>']
    colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c"]
    threshold_values = thresholds or {}
    for idx, (column, label) in enumerate(series):
        top = 46 + idx * panel_height
        values = [row[column] for row in rows if column in row]
        if not values:
            continue
        threshold = threshold_values.get(column)
        if threshold is not None:
            values.append(float(threshold))
        y_min, y_max = padded_range(values, include_zero=False)
        plot_w = width - margin["left"] - margin["right"]
        plot_h = panel_height - margin["top"] - margin["bottom"]
        left = margin["left"]
        base_y = top + margin["top"] + plot_h
        points = []
        for row_idx, row in enumerate(rows):
            if column not in row:
                continue
            x = left + (row.get("frame", row_idx + 1.0) - x_min) / (x_max - x_min) * plot_w
            y = base_y - (row[column] - y_min) / (y_max - y_min) * plot_h
            points.append(f"{x:.2f},{y:.2f}")
        parts.extend(
            [
                f'<text x="{left}" y="{top + 18}" class="subtitle">{escape(label)}</text>',
                f'<line x1="{left}" y1="{base_y}" x2="{left + plot_w}" y2="{base_y}" class="axis"/>',
                f'<line x1="{left}" y1="{top + margin["top"]}" x2="{left}" y2="{base_y}" class="axis"/>',
                f'<text x="8" y="{top + margin["top"] + 6}" class="tick">{y_max:.2f}</text>',
                f'<text x="8" y="{base_y}" class="tick">{y_min:.2f}</text>',
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="2.4"/>',
            ]
        )
        for tick in nice_ticks(x_min, x_max, max_ticks=6):
            x_tick = left + (tick - x_min) / (x_max - x_min) * plot_w
            parts.extend(
                [
                    f'<line x1="{x_tick:.2f}" y1="{base_y}" x2="{x_tick:.2f}" y2="{base_y + 4}" class="axis"/>',
                    f'<text x="{x_tick - 18:.2f}" y="{base_y + 22}" class="tick">{tick:g}</text>',
                ]
            )
        if threshold is not None:
            y_threshold = base_y - (float(threshold) - y_min) / (y_max - y_min) * plot_h
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y_threshold:.2f}" x2="{left + plot_w}" y2="{y_threshold:.2f}" class="threshold"/>',
                    f'<text x="{left + plot_w - 112}" y="{y_threshold - 4:.2f}" class="threshold-label">threshold {threshold:g}</text>',
                ]
            )
        if points:
            x, y = points[-1].split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{colors[idx % len(colors)]}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def bar_svg(bars: list[dict[str, Any]], title: str, ylabel: str, *, palette: str = "score") -> str:
    width = max(900, 78 * len(bars) + 120)
    height = 390
    left, right, top, bottom = 82, 24, 74, 104
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(bar["value"]) for bar in bars]
    for bar in bars:
        sd = numeric(bar.get("sd"))
        if sd is not None:
            values.extend([float(bar["value"]) - sd, float(bar["value"]) + sd])
    y_min, y_max = padded_range(values, include_zero=True)
    zero_y = top + plot_h - (0.0 - y_min) / (y_max - y_min) * plot_h
    gap = 8
    slot = plot_w / max(1, len(bars))
    bar_w = max(12, min(48, slot - gap))
    parts = [
        svg_header(width, height),
        f'<text x="24" y="30" class="title">{escape(title)}</text>',
        f'<text x="24" y="50" class="tick">{escape(ylabel)}</text>',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'<text x="8" y="{top + 6}" class="tick">{y_max:.2f}</text>',
        f'<text x="8" y="{top + plot_h}" class="tick">{y_min:.2f}</text>',
    ]
    for idx, bar in enumerate(bars):
        value = float(bar["value"])
        x = left + idx * slot + (slot - bar_w) / 2.0
        y = top + plot_h - (value - y_min) / (y_max - y_min) * plot_h
        rect_y = min(y, zero_y)
        rect_h = max(1.0, abs(zero_y - y))
        color = "#0f766e" if palette == "qc" else ("#2563eb" if value <= 0 else "#dc2626")
        parts.append(f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_w:.2f}" height="{rect_h:.2f}" fill="{color}" opacity="0.82"/>')
        sd = numeric(bar.get("sd"))
        if sd is not None:
            y_hi = top + plot_h - (value + sd - y_min) / (y_max - y_min) * plot_h
            y_lo = top + plot_h - (value - sd - y_min) / (y_max - y_min) * plot_h
            cx = x + bar_w / 2.0
            parts.extend(
                [
                    f'<line x1="{cx:.2f}" y1="{y_hi:.2f}" x2="{cx:.2f}" y2="{y_lo:.2f}" class="error"/>',
                    f'<line x1="{cx - 5:.2f}" y1="{y_hi:.2f}" x2="{cx + 5:.2f}" y2="{y_hi:.2f}" class="error"/>',
                    f'<line x1="{cx - 5:.2f}" y1="{y_lo:.2f}" x2="{cx + 5:.2f}" y2="{y_lo:.2f}" class="error"/>',
                ]
            )
        full_label = str(bar.get("full_label") or bar["label"])
        label = shorten(str(bar["label"]), 20)
        parts.append(f'<title>{escape(full_label)}: {value:.3f}</title>')
        parts.append(f'<text x="{x + bar_w / 2.0:.2f}" y="{top + plot_h + 20}" class="label" transform="rotate(45 {x + bar_w / 2.0:.2f},{top + plot_h + 20})">{escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def scatter_svg(points: list[dict[str, Any]], title: str, xlabel: str, ylabel: str) -> str:
    width = 900
    height = 420
    left, right, top, bottom = 82, 28, 56, 70
    if not points:
        return "\n".join(
            [
                svg_header(width, height),
                f'<text x="24" y="30" class="title">{escape(title)}</text>',
                '<text x="24" y="64" class="tick">No paired RMSD/energy rows were found.</text>',
                "</svg>",
            ]
        )
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    x_min, x_max = padded_range(xs, include_zero=False)
    y_min, y_max = padded_range(ys, include_zero=False)
    plot_w = width - left - right
    plot_h = height - top - bottom
    base_y = top + plot_h
    parts = [
        svg_header(width, height),
        f'<text x="24" y="30" class="title">{escape(title)}</text>',
        f'<line x1="{left}" y1="{base_y}" x2="{left + plot_w}" y2="{base_y}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{base_y}" class="axis"/>',
        f'<text x="{left + plot_w / 2 - 70:.2f}" y="{height - 22}" class="tick">{escape(xlabel)}</text>',
        f'<text x="10" y="{top - 12}" class="tick">{escape(ylabel)}</text>',
    ]
    for idx in range(6):
        frac = idx / 5.0
        x = left + frac * plot_w
        value = x_min + frac * (x_max - x_min)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{base_y}" x2="{x:.2f}" y2="{base_y + 4}" class="axis"/>',
                f'<text x="{x - 14:.2f}" y="{base_y + 20}" class="tick">{value:.2f}</text>',
            ]
        )
    for idx in range(5):
        frac = idx / 4.0
        y = base_y - frac * plot_h
        value = y_min + frac * (y_max - y_min)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>',
                f'<text x="10" y="{y + 4:.2f}" class="tick">{value:.2f}</text>',
            ]
        )
    for point in points:
        x = left + (float(point["x"]) - x_min) / (x_max - x_min) * plot_w
        y = base_y - (float(point["y"]) - y_min) / (y_max - y_min) * plot_h
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="#2563eb" opacity="0.42"><title>{escape(str(point.get("label") or ""))}: {float(point["x"]):.3f}, {float(point["y"]):.3f}</title></circle>')
    parts.append("</svg>")
    return "\n".join(parts)


def padded_range(values: list[float], *, include_zero: bool) -> tuple[float, float]:
    vals = list(values)
    if include_zero:
        vals.append(0.0)
    low, high = min(vals), max(vals)
    if low == high:
        pad = max(1.0, abs(low) * 0.1)
    else:
        pad = (high - low) * 0.08
    return low - pad, high + pad


def nice_tick_values(low: float, high: float, *, max_ticks: int) -> list[float]:
    if max_ticks <= 1 or high <= low:
        return [low]
    span = high - low
    raw_step = span / (max_ticks - 1)
    power = 10 ** math.floor(math.log10(raw_step))
    candidates = [1.0, 2.0, 2.5, 5.0, 10.0]
    step = min(candidates, key=lambda value: abs(value * power - raw_step)) * power
    start = math.ceil(low / step) * step
    ticks: list[float] = []
    current = start
    while current <= high + step * 0.5:
        ticks.append(round(current, 10))
        current += step
    return [tick for tick in sorted(dict.fromkeys(ticks)) if low <= tick <= high]


def nice_ticks(low: float, high: float, *, max_ticks: int) -> list[float]:
    if max_ticks <= 1 or high <= low:
        return [low]
    ticks = nice_tick_values(low, high, max_ticks=max_ticks)
    if not ticks or ticks[0] > low:
        ticks.insert(0, low)
    if ticks[-1] < high:
        ticks.append(high)
    if len(ticks) > max_ticks + 2:
        stride = math.ceil(len(ticks) / (max_ticks + 2))
        ticks = ticks[::stride]
        if ticks[-1] != high:
            ticks.append(high)
    ticks = sorted(dict.fromkeys(round(tick, 10) for tick in ticks))
    if len(ticks) >= 3:
        min_gap = (high - low) * 0.055
        if ticks[1] - ticks[0] < min_gap:
            del ticks[1]
        if len(ticks) >= 3 and ticks[-1] - ticks[-2] < min_gap:
            del ticks[-2]
    return ticks


def axis_ticks(
    low: float,
    high: float,
    *,
    max_ticks: int,
    map_value: Any | None = None,
    min_px: float = 40.0,
    integer: bool = False,
    include_low: bool = False,
    include_high: bool = False,
) -> list[float]:
    if high < low:
        low, high = high, low
    if high <= low:
        return [round(low) if integer else low]
    ticks = nice_tick_values(low, high, max_ticks=max_ticks)
    if include_low:
        ticks.append(low)
    if include_high:
        ticks.append(high)
    if integer:
        ticks = [float(round(tick)) for tick in ticks if low - 1e-8 <= round(tick) <= high + 1e-8]
    ticks = [round(tick, 10) for tick in ticks if low - 1e-8 <= tick <= high + 1e-8]
    ticks = sorted(dict.fromkeys(ticks))
    if map_value is not None:
        ticks = filter_ticks_by_pixel(ticks, map_value, min_px=min_px, low=low, high=high)
    if len(ticks) > max_ticks + 2:
        removable = [tick for tick in ticks if not nearly_equal(tick, low) and not nearly_equal(tick, high)]
        while len(ticks) > max_ticks + 2 and removable:
            victim = min(removable, key=lambda tick: tick_preference(tick, low=low, high=high))
            ticks.remove(victim)
            removable.remove(victim)
    return ticks or [round(low) if integer else low]


def filter_ticks_by_pixel(ticks: list[float], map_value: Any, *, min_px: float, low: float, high: float) -> list[float]:
    selected: list[float] = []
    for tick in ticks:
        if not selected:
            selected.append(tick)
            continue
        distance = abs(float(map_value(tick)) - float(map_value(selected[-1])))
        if distance >= min_px:
            selected.append(tick)
            continue
        previous = selected[-1]
        if tick_preference(tick, low=low, high=high) > tick_preference(previous, low=low, high=high):
            if len(selected) < 2 or abs(float(map_value(tick)) - float(map_value(selected[-2]))) >= min_px:
                selected[-1] = tick
    return selected


def tick_preference(value: float, *, low: float, high: float) -> int:
    score = 0
    if nearly_equal(value, low) or nearly_equal(value, high):
        score += 8
    if nearly_equal(value, round(value)):
        score += 2
    abs_value = abs(value)
    for base, weight in ((100.0, 3), (50.0, 2), (10.0, 1)):
        if abs_value >= base and nearly_equal(value / base, round(value / base)):
            score += weight
    if nearly_equal(value, 0.0):
        score += 3
    return score


def nearly_equal(left: float, right: float, *, tol: float = 1e-8) -> bool:
    return abs(left - right) <= tol


def svg_label_with_background(text: str, x: float, y: float, klass: str, *, anchor: str = "start") -> list[str]:
    width = max(22.0, len(text) * 6.2 + 8.0)
    height = 14.0
    if anchor == "end":
        rect_x = x - width - 1.0
    elif anchor == "middle":
        rect_x = x - width / 2.0
    else:
        rect_x = x - 4.0
    rect_y = y - height + 3.0
    return [
        f'<rect x="{rect_x:.2f}" y="{rect_y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="2" class="label-bg"/>',
        f'<text x="{x:.2f}" y="{y:.2f}" class="{klass}" text-anchor="{anchor}">{escape(text)}</text>',
    ]


def svg_header(width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
            "<style>",
            ".title{font:700 18px sans-serif;fill:#111827}.subtitle{font:700 13px sans-serif;fill:#111827}",
            ".axis{stroke:#6b7280;stroke-width:1}.tick{font:11px sans-serif;fill:#4b5563}.label{font:10px sans-serif;fill:#374151}",
            ".error{stroke:#111827;stroke-width:1.4}.grid{stroke:#e5e7eb;stroke-width:1}",
            ".threshold{stroke:#f59e0b;stroke-width:1.6;stroke-dasharray:6 4}.threshold-label{font:10px sans-serif;fill:#b45309}",
            ".basin-line{stroke:#111827;stroke-width:1.8;stroke-dasharray:7 4}.basin-label{font:11px sans-serif;fill:#111827;font-weight:700}",
            ".basin-point{fill:#111827;stroke:#ffffff;stroke-width:1.5}.basin-sample{stroke:none}.legend{font:11px sans-serif;fill:#374151}",
            ".label-bg{fill:#ffffff;opacity:.86;stroke:none}",
            ".replica-path,.time-trace,.replica-trace,.basin-curve{stroke-linejoin:round;stroke-linecap:round}",
            "</style>",
        ]
    )


def run_index_html(
    run_dir: Path,
    rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    ranking_svg: str,
    qc_svg: str,
    sort_by: str,
    *,
    include_samples: bool,
) -> str:
    row_by_job = {str(row.get("job_id") or ""): row for row in rows}
    qc_by_job = {str(row.get("job_id") or ""): row for row in qc_rows}
    valid_count = sum(1 for row in rows if str(row.get("status") or "") == "valid")
    invalid_count = len(rows) - valid_count
    generated_at = utc_now()
    summary_items = {
        "Run": run_dir.name,
        "Generated": generated_at,
        "Jobs": f"{len(rows)} total / {valid_count} valid / {invalid_count} invalid",
        "Sort": pretty_field_label(sort_by),
        "MMPBSA window": run_frame_window_label(run_dir, rows),
        "MD settings": run_protocol_label(run_dir, rows),
    }
    table_rows = []
    for rank, ranking in enumerate(ranking_rows, start=1):
        job_id = str(ranking.get("job_id") or "")
        summary = row_by_job.get(job_id, {})
        qc = qc_by_job.get(job_id, {})
        status = str(summary.get("status") or ranking.get("status") or "")
        traj_status = str(summary.get("trajectory_qc_status") or "")
        mmpbsa_status = str(summary.get("mmpbsa_qc_status") or "")
        job_link = f"samples/{url_component(job_id)}/index.html" if include_samples else ""
        job_cell = f'<a href="{escape(job_link)}">{escape(job_id)}</a>' if job_link else f"<code>{escape(job_id)}</code>"
        score_cells = []
        for _, fields in RUN_TABLE_GROUPS[:2]:
            for field, _, _, klass in fields:
                score_cells.append(sortable_cell(summary.get(field), classes=klass))
        partner_rmsd = qc.get("ligand_rmsd_mean") or qc.get("peptide_rmsd_mean")
        qc_cells = [
            sortable_cell(qc.get("receptor_rmsd_mean")),
            sortable_cell(partner_rmsd),
            sortable_cell(qc.get("native_contacts_mean")),
            sortable_cell(qc.get("native_contacts_min")),
            sortable_cell(qc.get("interface_distance_min")),
        ]
        table_rows.append(
            "<tr>"
            f"{sortable_cell(rank, classes='sticky-rank')}"
            f"{plain_cell(job_cell, classes='sticky-job job-cell')}"
            f"{qc_status_cell(status, traj_status, mmpbsa_status)}"
            f"{sortable_cell(summary.get('mmpbsa_frames'))}"
            f"{''.join(score_cells)}"
            f"{''.join(qc_cells)}"
            "</tr>"
        )
    group_header = (
        '<tr class="group-row">'
        '<th class="sticky-rank" colspan="2">Identity</th>'
        '<th colspan="2">Run state</th>'
        '<th colspan="4">PB score</th>'
        '<th colspan="4">GB score</th>'
        '<th colspan="5">Trajectory QC</th>'
        "</tr>"
    )
    identity_headers = (
        f"{sortable_header('Rank', classes='sticky-rank')}"
        '<th class="sticky-job">Job</th>'
        '<th>QC</th>'
        f"{sortable_header('Frames')}"
    )
    score_headers = "".join(sortable_header(label, unit=unit, classes=klass) for _, fields in RUN_TABLE_GROUPS[:2] for _, label, unit, klass in fields)
    qc_headers = "".join(sortable_header(label, unit=unit, classes=klass) for _, label, unit, klass in RUN_TABLE_GROUPS[2][1])
    ranking_table = (
        '<table class="wide sortable report-table"><thead>'
        f"{group_header}<tr>{identity_headers}{score_headers}{qc_headers}"
        "</tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>"
    )
    return full_html(
        "MMPBSA Group Report",
        [
            f'<section class="summary-section"><h2>Summary</h2>{metadata_strip(summary_items, title=str(run_dir))}</section>',
            f'<section class="ranking-section"><h2>Ranking</h2>{ranking_table}</section>',
        ],
    )


def sample_index_html(
    job_dir: Path,
    title: str,
    summary: dict[str, Any],
    qc_summary: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    qc_svg: str,
    score_svg: str,
    visual_paths: dict[str, Any],
    analysis_paths: dict[str, Any],
) -> str:
    metric_table = metric_rows_table(metric_rows)
    visual_record = visual_paths.get("pymol_visual")
    visual_actions = ""
    if isinstance(visual_record, dict):
        visual_actions = pymol_action_row(visual_record)
    sections = [
        f'<section class="summary-section"><h2>Summary</h2>{metadata_strip(job_metadata(job_dir, summary, qc_summary), title=str(job_dir))}</section>',
    ]
    if visual_actions:
        sections.append(f"<section><h2>PyMOL</h2>{visual_actions}</section>")
    sections.extend(
        [
            f"<section><h2>QC Audit</h2>{metric_table}</section>",
            f'<section class="plot-section"><h2>Trajectory QC</h2>{qc_svg}</section>',
        ]
    )
    if analysis_paths.get("interaction_contacts_svg"):
        sections.append(
            '<section class="plot-section"><h2>Interaction Contacts</h2>'
            '<p class="muted">Replica-aware geometric estimate from exported PDB states: color encodes replica, solid lines are H-bond-like contacts, and dashed lines are salt-bridge-like contacts.</p>'
            '<img class="svg-img" src="interaction_contacts.svg" alt="Interaction contacts plot">'
            "</section>"
        )
    if analysis_paths.get("md_energy_trace_svg") or analysis_paths.get("md_energy_landscape_svg"):
        sections.append(
            '<section class="plot-section"><h2>MD RMSD / Potential Trace</h2>'
            '<p class="muted">Replica traces are plotted against production time; the vertical dashed line marks the frame with the lowest sampled potential.</p>'
            '<img class="svg-img" src="md_energy_landscape.svg" alt="MD energy landscape plot">'
            "</section>"
        )
    if analysis_paths.get("md_energy_basin_svg"):
        sections.append(
            '<section class="plot-section"><h2>Energy Basin</h2>'
            '<p class="muted">Per-replica occupancy-derived 1D RMSD basins from MD frames for QC only; these are not rigorous free-energy landscapes.</p>'
            '<img class="svg-img" src="md_energy_basin.svg" alt="Sampled energy basin plot">'
            "</section>"
        )
    return full_html(f"{title} Sample Report", sections)


def bundle_index_html(jobs: list[dict[str, Any]], *, snapshots_only: bool, keep_plots: bool) -> str:
    rows = []
    for job in jobs:
        job_dir = str(job["bundle_job_dir"])
        job_id = str(job["job_id"])
        trajectory_link = ""
        if "aligned_trajectory.pdb" in set(job.get("files", [])):
            trajectory_link = f'<a href="{escape(job_dir)}/structures/aligned_trajectory.pdb">aligned trajectory</a>'
        elif "pymol_trajectory.pdb" in set(job.get("files", [])):
            trajectory_link = f'<a href="{escape(job_dir)}/structures/pymol_trajectory.pdb">trajectory</a>'
        else:
            trajectory_link = '<span class="muted">snapshots only</span>'
        plot_links = ""
        if keep_plots:
            plot_links = f'<a href="{escape(job_dir)}/visual/index.html">sample report</a>'
        else:
            plot_links = '<span class="muted">compact</span>'
        rows.append(
            "<tr>"
            f"<td><code>{escape(job_id)}</code></td>"
            f"<td>{status_badge(str(job.get('align_status') or 'skipped'))}</td>"
            f"<td>{status_badge(str(job.get('video_status') or 'not_requested'))}</td>"
            f'<td><a href="{escape(job_dir)}/movie.pml">movie.pml</a> <a href="{escape(job_dir)}/load_pymol.pml">load</a></td>'
            f"<td>{trajectory_link}</td>"
            f'<td><a href="{escape(job_dir)}/trajectory_qc.csv">QC CSV</a></td>'
            f"<td>{plot_links}</td>"
            f'<td><a href="{escape(job_dir)}/summary.json">summary.json</a></td>'
            "</tr>"
        )
    table = (
        '<table class="wide"><thead><tr><th>Job</th><th>Align</th><th>Video</th><th>PyMOL</th><th>Trajectory</th><th>QC</th><th>Plots</th><th>Summary</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )
    return full_html(
        "PyMOL Bundle Index",
        [
            f"<section><h2>Summary</h2>{card_grid({'Jobs': len(jobs), 'Snapshots only': str(snapshots_only).lower(), 'Keep plots': str(keep_plots).lower()})}</section>",
            '<section><h2>Usage</h2><p>Open any selected job directory and run <code>pymol movie.pml</code> for playback, or <code>pymol load_pymol.pml</code> for static inspection. Run <code>./render_video.sh</code> locally when PyMOL and ffmpeg are installed.</p></section>',
            f"<section><h2>Jobs</h2>{table}</section>",
        ],
    )


def full_html(title: str, sections: list[str]) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{escape(title)}</title>",
            f"<style>{base_css()}</style>",
            "</head>",
            "<body>",
            f"<h1>{escape(title)}</h1>",
            *sections,
            f"<script>{sortable_table_script()}</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


def base_css() -> str:
    return (
        ":root{color-scheme:light;--ink:#172033;--muted:#667085;--line:#d9dee8;--panel:#f7f8fb;--panel2:#ffffff;--blue:#2457c5;--sticky:#fff}"
        "*{box-sizing:border-box}body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;margin:0;color:var(--ink);background:#eef1f6}"
        "body:before{content:'';display:block;height:6px;background:#2457c5}.page,body>h1,body>section{max-width:1280px;margin-left:auto;margin-right:auto}body>section.ranking-section{max-width:min(1840px,calc(100vw - 32px))}"
        "h1{font-size:26px;margin:28px auto 18px;padding:0 24px;letter-spacing:0}h2{font-size:17px;margin:0 0 14px;padding-left:10px;border-left:3px solid var(--blue);line-height:1.2;letter-spacing:0}h3{font-size:14px;margin:0 0 8px;color:#344054}"
        "section{margin:16px auto;padding:20px 24px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;box-shadow:0 1px 2px rgba(16,24,40,.04)}"
        "table{border-collapse:separate;border-spacing:0;margin:10px 0;width:100%;max-width:100%;background:#fff}"
        "td,th{border-bottom:1px solid #e5e8ef;padding:8px 10px;text-align:left;font-size:13px;vertical-align:top}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.sd{color:var(--muted)}"
        "th{position:sticky;top:0;background:#f3f5f9;font-weight:700;color:#344054;z-index:3}.group-row th{top:0;background:#e9edf5;text-align:center;border-bottom:1px solid #cfd6e4}.report-table thead tr:last-child th{top:35px}.unit{display:block;margin-top:2px;font-size:10px;font-weight:600;color:var(--muted)}"
        "th.sortable-col{cursor:pointer}th.sortable-col:after{content:'';display:inline-block;min-width:12px;margin-left:5px;color:#667085}th.sortable-col[data-dir='asc']:after{content:'^'}th.sortable-col[data-dir='desc']:after{content:'v'}"
        "tbody tr:nth-child(even) td{background:#fbfcfe}tr:hover td{background:#f4f7ff}.wide{display:block;overflow-x:auto;white-space:nowrap;border:1px solid var(--line);border-radius:8px}.full-table{display:table;width:100%;border:1px solid var(--line);border-radius:8px;overflow:hidden}"
        ".sticky-rank{position:sticky;left:0;z-index:4;min-width:64px;background:var(--sticky);box-shadow:1px 0 0 var(--line)}th.sticky-rank{background:#f3f5f9}.group-row .sticky-rank{background:#e9edf5}.sticky-job{position:sticky;left:64px;z-index:4;min-width:248px;background:var(--sticky);box-shadow:1px 0 0 var(--line)}th.sticky-job{background:#f3f5f9}.job-cell code,.job-cell a{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}"
        "code{background:#eef2f7;padding:2px 5px;border-radius:4px;color:#101828}a{color:var(--blue);text-decoration:none;font-weight:600}a:hover{text-decoration:underline}"
        "ul{margin:8px 0 0;padding-left:20px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}"
        ".card{border:1px solid var(--line);border-radius:8px;padding:12px;background:linear-gradient(180deg,#fff,#f8fafc)}.card b{display:block;font-size:11px;text-transform:uppercase;color:var(--muted)}.card span{display:block;margin-top:5px;font-size:14px;overflow-wrap:anywhere}.card.path span{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".meta-strip{display:flex;flex-wrap:wrap;gap:8px 10px;align-items:stretch}.meta-item{display:flex;gap:7px;align-items:baseline;border:1px solid var(--line);border-radius:6px;padding:7px 9px;background:#fbfcfe;min-height:34px}.meta-item b{font-size:10px;text-transform:uppercase;color:var(--muted);letter-spacing:.02em}.meta-item span{font-size:13px;font-weight:650;color:#1f2937}"
        ".action-row{display:flex;gap:12px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap}.actions{display:flex;flex-wrap:wrap;gap:8px}.action-link{display:inline-flex;align-items:center;height:32px;padding:0 10px;border:1px solid #c9d6f5;border-radius:6px;background:#eff4ff}.compact-note{flex-basis:100%;margin:0}.movie-player{flex-basis:100%;width:100%;max-width:920px;margin:4px auto 0;border:1px solid var(--line);border-radius:8px;background:#000}.svg-img{display:block;width:100%;max-width:920px;height:auto}"
        ".charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.chart-panel{min-width:0;border:1px solid var(--line);border-radius:8px;padding:12px;background:#fff}section svg{width:100%;height:auto;max-width:900px;display:block}.plot-section svg,.plot-section .svg-img{margin-left:auto;margin-right:auto}"
        ".badge{display:inline-block;border-radius:999px;padding:3px 9px;font-size:12px;font-weight:700;background:#e5e7eb;color:#374151}.muted{color:var(--muted)}"
        ".valid,.aligned,.rendered{background:#dcfce7;color:#166534}.invalid,.fail,.failed{background:#fee2e2;color:#991b1b}.warn,.warning{background:#fef3c7;color:#92400e}.skipped,.not_requested{background:#e0f2fe;color:#075985}"
        "iframe{width:100%;height:440px;border:1px solid var(--line);border-radius:8px;margin:10px 0;background:#fff}"
        "@media print{body{background:#fff}body:before{display:none}body>h1,body>section{max-width:none;margin:0 0 12px;padding-left:0;padding-right:0}section{box-shadow:none;border-color:#cfd6e4;break-inside:avoid}.wide{overflow:visible;white-space:normal}.sticky-rank,.sticky-job{position:static;box-shadow:none}th{position:static}.chart-panel{break-inside:avoid}}"
    )


def card_grid(data: dict[str, Any]) -> str:
    cards = []
    for key, value in data.items():
        if value in (None, "", [], {}):
            continue
        key_text = str(key)
        value_text = str(value)
        path_class = " path" if key_text.lower() in {"run", "job_dir", "output_dir"} else ""
        title = f' title="{escape(value_text)}"' if path_class else ""
        cards.append(f'<div class="card{path_class}"{title}><b>{escape(key_text)}</b><span>{escape(value_text)}</span></div>')
    return '<div class="cards">' + "".join(cards) + "</div>"


def metadata_strip(data: dict[str, Any], *, title: str = "") -> str:
    items = []
    title_attr = f' title="{escape(title)}"' if title else ""
    for key, value in data.items():
        if value in (None, "", [], {}):
            continue
        items.append(f'<span class="meta-item"{title_attr if key in {"Run", "job_id"} else ""}><b>{escape(str(key))}</b><span>{escape(str(value))}</span></span>')
    return '<div class="meta-strip">' + "".join(items) + "</div>"


def run_frame_window_label(run_dir: Path, rows: list[dict[str, Any]]) -> str:
    manifest = first_manifest(rows)
    profile = manifest.get("profile") or {}
    protocol = profile.get("protocol") or {}
    frame_settings = manifest.get("frame_settings") or {}
    replicas = numeric(frame_settings.get("replica_count") or protocol.get("replica_count"))
    frames_per_replica = numeric(frame_settings.get("frames_per_replica"))
    total_frames = numeric(frame_settings.get("expected_mmpbsa_frames"))
    start_ns = numeric(protocol.get("mmpbsa_start_ns"))
    end_ns = numeric(protocol.get("production_ns"))
    interval_ps = numeric(protocol.get("mmpbsa_interval_ps") or protocol.get("xtc_interval_ps"))
    if total_frames is None:
        frame_values = [value for value in (numeric(row.get("mmpbsa_frames")) for row in rows) if value is not None]
        total_frames = frame_values[0] if frame_values and min(frame_values) == max(frame_values) else None
    parts = []
    if replicas is not None and frames_per_replica is not None:
        parts.append(f"{int(replicas)} replicas x {int(frames_per_replica)} frames")
    elif total_frames is not None:
        parts.append(f"{int(total_frames)} frames")
    if start_ns is not None and end_ns is not None:
        parts.append(f"{start_ns:g}-{end_ns:g} ns")
    if interval_ps is not None:
        parts.append(f"every {interval_ps:g} ps")
    return ", ".join(parts) or "see manifest"


def run_protocol_label(run_dir: Path, rows: list[dict[str, Any]]) -> str:
    manifest = first_manifest(rows)
    profile = manifest.get("profile") or {}
    protocol = profile.get("protocol") or {}
    mmpbsa = profile.get("mmpbsa") or {}
    amber = profile.get("amber_prep") or {}
    production_ns = numeric(protocol.get("production_ns"))
    explicit_waters = numeric(mmpbsa.get("explicit_water_count"))
    gb = mmpbsa.get("gb_igb")
    pb = mmpbsa.get("pb_inp")
    protein_ff = str(amber.get("protein_ff") or "").replace("leaprc.protein.", "")
    parts = []
    if production_ns is not None:
        parts.append(f"{production_ns:g} ns production")
    if protein_ff:
        parts.append(protein_ff)
    if gb:
        parts.append(f"GB igb={gb}")
    if pb:
        parts.append(f"PB inp={pb}")
    if explicit_waters is not None:
        parts.append(f"{int(explicit_waters)} explicit waters")
    return ", ".join(parts) or "see manifest"


def first_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        job_dir = row.get("job_dir")
        if not job_dir:
            continue
        path = Path(str(job_dir)) / "manifest.json"
        if path.exists():
            return read_optional_json(path)
    return {}


def pymol_action_row(visual_record: dict[str, Any]) -> str:
    items = [
        ("load_pymol.pml", "Load"),
        ("movie.pml", "Movie script"),
        ("load_static.pml", "Static snapshots"),
    ]
    for script in visual_record.get("full_md_scripts", [])[:3]:
        items.append((str(script), f"Full MD {Path(str(script)).parent.name}"))
    links = "".join(f'<a class="action-link" href="pymol/{escape(href)}">{escape(label)}</a>' for href, label in items)
    meta = {
        "Align": visual_record.get("align_status", "skipped"),
        "States": visual_record.get("trajectory_states", ""),
        "Stride": visual_record.get("movie_stride", ""),
        "Video": visual_record.get("video_status", "not_requested"),
    }
    video = ""
    if visual_record.get("video_status") == "rendered":
        video = '<video class="movie-player" controls src="pymol/movie.mp4"></video>'
    elif visual_record.get("movie_requested"):
        video = f'<p class="muted compact-note">movie.mp4 was requested but not rendered: {escape(str(visual_record.get("video_reason") or visual_record.get("video_status") or "unknown"))}.</p>'
    caveat = ""
    if visual_record.get("full_md_scripts"):
        caveat = '<p class="muted compact-note">Movie/Load use aligned MMPBSA-window PDB states. Full MD links use mdtraj-processed production trajectories with PBC imaging and receptor fitting.</p>'
    return '<div class="action-row"><div class="actions">' + links + "</div>" + metadata_strip(meta) + video + caveat + "</div>"


def link_list(links: list[tuple[str, str]]) -> str:
    items = [f'<li><a href="{escape(href)}">{escape(label)}</a></li>' for href, label in links]
    return "<ul>" + "".join(items) + "</ul>"


def metric_rows_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">No QC metrics available.</p>'
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('label') or row.get('metric') or ''))}</td>"
            f"{sortable_cell(row.get('mean'))}"
            f"{sortable_cell(row.get('min'))}"
            f"{sortable_cell(row.get('max'))}"
            f"{sortable_cell(row.get('first'))}"
            f"{sortable_cell(row.get('last'))}"
            f"{sortable_cell(row.get('threshold'))}"
            "</tr>"
        )
    return (
        '<table class="sortable full-table"><thead><tr><th>Metric</th>'
        f"{sortable_header('Mean')}{sortable_header('Min')}{sortable_header('Max')}{sortable_header('First')}{sortable_header('Last')}{sortable_header('Threshold')}"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def sortable_header(label: str, *, unit: str = "", classes: str = "") -> str:
    klass = class_attr("sortable-col", "num", classes)
    unit_text = f'<span class="unit">{escape(unit)}</span>' if unit else ""
    return f'<th{klass}>{escape(label)}{unit_text}</th>'


def sortable_cell(value: Any, *, classes: str = "") -> str:
    sort_value = cell_sort_value(value)
    klass = class_attr("num" if numeric(value) is not None else "", classes)
    return f'<td{klass} data-sort="{escape(sort_value)}">{format_cell(value)}</td>'


def plain_cell(html_text: str, *, classes: str = "", title: str = "") -> str:
    klass = class_attr(classes)
    title_attr = f' title="{escape(title)}"' if title else ""
    return f"<td{klass}{title_attr}>{html_text}</td>"


def class_attr(*classes: str) -> str:
    names: list[str] = []
    for group in classes:
        for name in str(group or "").split():
            if name and name not in names:
                names.append(name)
    return f' class="{" ".join(escape(name) for name in names)}"' if names else ""


def qc_status_cell(status: str, trajectory_status: str, mmpbsa_status: str) -> str:
    values = [value for value in (status, trajectory_status, mmpbsa_status) if value]
    normalized = [value.lower() for value in values]
    if values and all(value in {"valid", "merged"} for value in normalized):
        display = "valid"
    elif any(value in {"warn", "warning"} for value in normalized):
        display = "warn"
    elif any(value in {"fail", "failed", "invalid"} for value in normalized):
        display = "fail"
    else:
        display = values[0] if values else "unknown"
    title = f"Status: {status or 'unknown'}; Trajectory QC: {trajectory_status or 'unknown'}; MMPBSA QC: {mmpbsa_status or 'unknown'}"
    return plain_cell(status_badge(display), title=title)


def cell_sort_value(value: Any) -> str:
    parsed = numeric(value)
    if parsed is not None:
        return f"{parsed:.12g}"
    return str(value or "").lower()


def format_cell(value: Any) -> str:
    parsed = numeric(value)
    if parsed is not None:
        if abs(parsed - round(parsed)) < 1e-9:
            return str(int(round(parsed)))
        return f"{parsed:.2f}"
    return escape(str(value or ""))


def sortable_table_script() -> str:
    return (
        "document.querySelectorAll('table.sortable').forEach(table=>{const heads=[...table.querySelectorAll('thead tr:last-child th')];"
        "heads.forEach((th,i)=>{if(!th.classList.contains('sortable-col'))return;th.addEventListener('click',()=>{"
        "const tbody=table.querySelector('tbody');const rows=[...tbody.querySelectorAll('tr')];"
        "const dir=th.dataset.dir==='asc'?'desc':'asc';heads.forEach(h=>{delete h.dataset.dir;h.removeAttribute('aria-sort');});th.dataset.dir=dir;th.setAttribute('aria-sort',dir==='asc'?'ascending':'descending');"
        "rows.sort((a,b)=>{const av=(a.children[i]?.dataset.sort||a.children[i]?.textContent||'').trim();const bv=(b.children[i]?.dataset.sort||b.children[i]?.textContent||'').trim();"
        "const an=Number(av),bn=Number(bv);const numeric=av!==''&&bv!==''&&Number.isFinite(an)&&Number.isFinite(bn);const cmp=numeric?an-bn:av.localeCompare(bv);return dir==='asc'?cmp:-cmp;});"
        "rows.forEach(r=>tbody.appendChild(r));});});});"
    )


def status_badge(status: str) -> str:
    text = status or "unknown"
    klass = re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-") or "unknown"
    return f'<span class="badge {escape(klass)}">{escape(text)}</span>'


def format_number(value: Any) -> str:
    parsed = numeric(value)
    if parsed is None:
        return ""
    return f"{parsed:.2f}"


def pretty_field_label(field: str) -> str:
    if field in FIELD_LABELS:
        return FIELD_LABELS[field]
    label = field
    label = label.replace("_kJ_mol", " (kJ/mol)").replace("_kcal_mol", " (kcal/mol)")
    label = label.replace("_replica_sd", " SD").replace("_replica_sem", " SEM")
    label = label.replace("_", " ")
    return label


def compact_job_id(job_id: str) -> str:
    match = re.search(r"rank_(\d+)_model_(\d+)", job_id)
    if match:
        return f"r{match.group(1)}_m{match.group(2)}"
    text = re.sub(r"^(kras6wgn_|kras_|job_)", "", job_id)
    return shorten(text, 20)


def url_component(text: str) -> str:
    return text.replace("/", "_")


def job_metadata(job_dir: Path, summary: dict[str, Any], qc_summary: dict[str, Any]) -> dict[str, Any]:
    manifest = read_optional_json(job_dir / "manifest.json")
    profile = manifest.get("profile") or {}
    protocol = profile.get("protocol") or {}
    mmpbsa = profile.get("mmpbsa") or {}
    production_ns = numeric(protocol.get("production_ns"))
    start_ns = numeric(protocol.get("mmpbsa_start_ns"))
    interval_ps = numeric(protocol.get("mmpbsa_interval_ps") or protocol.get("xtc_interval_ps"))
    window = ""
    if start_ns is not None and production_ns is not None:
        window = f"{start_ns:g}-{production_ns:g} ns"
        if interval_ps is not None:
            window += f", every {interval_ps:g} ps"
    replica_count = summary.get("replica_count") or ((manifest.get("frame_settings") or {}).get("replica_count"))
    frames = summary.get("mmpbsa_frames") or ((manifest.get("frame_settings") or {}).get("expected_mmpbsa_frames"))
    md = f"{production_ns:g} ns" if production_ns is not None else ""
    return {
        "job_id": summary.get("job_id") or job_dir.name,
        "status": summary.get("status", ""),
        "QC": f"{summary.get('trajectory_qc_status') or qc_summary.get('status', 'unknown')} / {summary.get('mmpbsa_qc_status', 'unknown')}",
        "production MD": md,
        "MMPBSA window": window,
        "frames": frames,
        "replicas": replica_count,
        "GB/PB": f"igb={mmpbsa.get('gb_igb', '')}, inp={mmpbsa.get('pb_inp', '')}",
    }


def copy_job_bundle_files(
    source_job: Path,
    target_job: Path,
    *,
    snapshots_only: bool,
    align: bool,
    movie_stride: int,
    render_video: bool,
) -> dict[str, Any]:
    structures = source_job / "analysis" / "structures"
    if not structures.exists():
        raise SystemExit(f"Missing structures directory: {structures}")
    target_structures = target_job / "structures"
    target_structures.mkdir(parents=True, exist_ok=True)
    movie_stride = max(1, int(movie_stride))
    copied: set[str] = set()
    visual_record: dict[str, Any] = {"align_requested": align, "snapshots_only": snapshots_only, "movie_stride": movie_stride, "movie_requested": render_video}
    if align:
        visual_record.update(export_aligned_visual_assets(source_job, target_job, snapshots_only=snapshots_only, movie_stride=movie_stride))
        copied.update(visual_record.get("files", []))
    if visual_record.get("align_status") != "aligned":
        copied.update(copy_existing_structures(structures, target_structures, snapshots_only=snapshots_only))
    if not copied:
        raise SystemExit(f"No PyMOL structures found in {structures}")
    if "aligned_trajectory.pdb" in copied:
        state_count = int(visual_record.get("trajectory_states") or 1)
    elif "pymol_trajectory.pdb" in copied:
        state_count = None
    else:
        state_count = 1
    write_text_atomic(target_job / "load_pymol.pml", portable_pymol_script(snapshots_only=snapshots_only, copied=copied, state_count=state_count, autoplay=False))
    write_text_atomic(target_job / "movie.pml", portable_pymol_script(snapshots_only=snapshots_only, copied=copied, state_count=state_count, autoplay=True))
    write_text_atomic(target_job / "load_static.pml", static_snapshots_pml(copied))
    write_text_atomic(target_job / "render_video.sh", render_video_shell_script())
    write_text_atomic(target_job / "render_frames.pml", render_frames_pml(copied, state_count))
    full_md_scripts = write_full_md_references(source_job, target_job, movie_stride=movie_stride)
    if full_md_scripts:
        visual_record["full_md_scripts"] = full_md_scripts
    os.chmod(target_job / "render_video.sh", 0o755)
    if render_video:
        visual_record.update(try_render_video(target_job))
    else:
        visual_record["video_status"] = "not_requested"
    copy_if_exists(source_job / "result" / "summary.json", target_job / "summary.json")
    copy_if_exists(source_job / "result" / "summary.csv", target_job / "summary.csv")
    copy_if_exists(source_job / "analysis" / "qc" / "trajectory_qc.csv", target_job / "trajectory_qc.csv")
    copy_if_exists(source_job / "manifest.json", target_job / "job_manifest.json")
    visual_record["files"] = sorted(copied)
    write_json_atomic(target_job / "bundle_manifest.json", visual_record)
    return visual_record


def copy_existing_structures(source_structures: Path, target_structures: Path, *, snapshots_only: bool) -> set[str]:
    copied: set[str] = set()
    for name in ("first.pdb", "mid.pdb", "last.pdb"):
        source = source_structures / name
        if source.exists():
            shutil.copy2(source, target_structures / name)
            copied.add(name)
    if not snapshots_only and (source_structures / "pymol_trajectory.pdb").exists():
        shutil.copy2(source_structures / "pymol_trajectory.pdb", target_structures / "pymol_trajectory.pdb")
        copied.add("pymol_trajectory.pdb")
    return copied


def write_full_md_references(source_job: Path, target_job: Path, *, movie_stride: int) -> list[str]:
    scripts: list[str] = []
    md_root = source_job / "md"
    if not md_root.exists():
        return scripts
    manifest = read_optional_json(source_job / "manifest.json")
    full_root = target_job / "full_md"
    full_root.mkdir(parents=True, exist_ok=True)
    for rep_dir in sorted(md_root.glob("rep*")):
        if not rep_dir.is_dir():
            continue
        rep_out = full_root / rep_dir.name
        rep_out.mkdir(parents=True, exist_ok=True)
        if write_full_md_assets(rep_dir, rep_out, manifest, stride=movie_stride):
            scripts.append(str((rep_out / "load_full_md.pml").relative_to(target_job)))
    return scripts


def write_full_md_assets(rep_dir: Path, rep_out: Path, manifest: dict[str, Any], *, stride: int) -> bool:
    gro = rep_dir / "md_prod.gro"
    xtc = rep_dir / "md_prod.xtc"
    if not gro.exists() or not xtc.exists():
        return False
    try:
        import mdtraj as md  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        write_text_atomic(rep_out / "full_md_status.txt", "mdtraj is not installed; full-MD PyMOL assets were skipped.\n")
        return False
    try:
        traj = md.load(str(xtc), top=str(gro), stride=max(1, int(stride)))
        traj.image_molecules(inplace=True)
        align_indices = receptor_alignment_indices(traj.topology, manifest)
        if len(align_indices) >= 3:
            traj.superpose(traj, 0, atom_indices=align_indices)
        visual_indices = full_md_visual_indices(traj.topology, manifest)
        if not visual_indices:
            return False
        visual = traj.atom_slice(visual_indices)
        visual[0].save_pdb(str(rep_out / "full_md_aligned.pdb"))
        visual.save_xtc(str(rep_out / "full_md_aligned.xtc"))
        visual[0].save_pdb(str(rep_out / "first.pdb"))
        visual[len(visual) // 2].save_pdb(str(rep_out / "mid.pdb"))
        visual[-1].save_pdb(str(rep_out / "last.pdb"))
    except Exception as exc:
        write_text_atomic(rep_out / "full_md_status.txt", f"full-MD mdtraj export failed: {exc}\n")
        return False
    write_text_atomic(rep_out / "load_full_md.pml", full_md_pymol_script(rep_dir.name))
    return True


def full_md_visual_indices(topology: Any, manifest: dict[str, Any]) -> list[int]:
    indices: set[int] = set()
    binder = set(binder_atom_indices(topology, manifest))
    for atom in topology.atoms:
        resname = atom.residue.name.upper()
        if atom.residue.is_water:
            continue
        if atom.residue.is_protein or atom.index in binder or resname in set(COFACTOR_RESN.split("+")):
            indices.add(atom.index)
    return sorted(indices)


def full_md_pymol_script(replica: str) -> str:
    lines = [
        "reinitialize",
        "load full_md_aligned.pdb, full_md",
        "load_traj full_md_aligned.xtc, full_md",
        "hide everything",
        "show cartoon, polymer.protein",
        "show sticks, organic",
        f"show sticks, resn {COFACTOR_RESN}",
        "show spheres, resn MG",
        "color gray70, polymer.protein",
        "color cyan, organic and elem C",
    ]
    for element, color in ELEMENT_COLORS.items():
        lines.append(f"color {color}, organic and elem {element}")
    lines.extend(
        [
            f"color orange, resn {COFACTOR_RESN}",
            "color lime, resn MG",
            "set cartoon_transparency, 0.12",
            "set stick_radius, 0.14",
            "set sphere_scale, 0.35",
            "set all_states, off",
            "bg_color white",
            "orient",
            "python",
            "from pymol import cmd",
            "n = max(1, cmd.count_states('full_md'))",
            "cmd.mset(f'1 -{n}')",
            "python end",
            "set movie_loop, on",
            f"# {replica}: mdtraj-processed production trajectory with PBC imaging and receptor fitting.",
            "mplay",
            "",
        ]
    )
    return "\n".join(lines)


def export_aligned_visual_assets(source_job: Path, target_job: Path, *, snapshots_only: bool, movie_stride: int) -> dict[str, Any]:
    manifest = read_optional_json(source_job / "manifest.json")
    mmpbsa_dir = source_job / "analysis" / "mmpbsa"
    complex_prmtop = mmpbsa_dir / "complex.prmtop"
    trajectory = mmpbsa_dir / "md_prod_dry_center.nc"
    if not complex_prmtop.exists() or not trajectory.exists():
        return {
            "align_status": "skipped",
            "align_reason": f"missing {complex_prmtop.name if not complex_prmtop.exists() else trajectory.name}",
            "files": [],
        }
    receptor_mask = str(manifest.get("receptor_residue_mask") or "").strip()
    if not receptor_mask:
        return {"align_status": "skipped", "align_reason": "missing receptor_residue_mask in job manifest", "files": []}
    try:
        receptor_last = parse_simple_mask(receptor_mask)[1]
    except SystemExit as exc:
        return {"align_status": "skipped", "align_reason": str(exc), "files": []}
    frames = trajectory_frame_count(source_job, manifest)
    if frames <= 0:
        return {"align_status": "skipped", "align_reason": "could not determine trajectory frame count", "files": []}

    structures = target_job / "structures"
    logs = target_job / "logs"
    structures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for label, mask in alignment_masks(receptor_mask):
        script = logs / f"cpptraj_align_{label}.in"
        log = logs / f"cpptraj_align_{label}.log"
        write_text_atomic(script, cpptraj_alignment_text(complex_prmtop, trajectory, structures, mask, frames=frames, stride=movie_stride, snapshots_only=snapshots_only))
        try:
            run_logged(cpptraj_command(manifest, script), log)
        except RuntimeError as exc:
            errors.append(f"{label}: {exc}")
            cleanup_raw_alignment_outputs(structures)
            continue
        files = finalize_aligned_pdbs(structures, receptor_last, snapshots_only=snapshots_only)
        if files:
            state_count = 0 if snapshots_only else ((frames - 1) // movie_stride) + 1
            return {
                "align_status": "aligned",
                "alignment_mask": mask,
                "alignment_mask_kind": label,
                "trajectory_frames": frames,
                "trajectory_states": state_count,
                "files": sorted(files),
            }
    return {"align_status": "skipped", "align_reason": "; ".join(errors) if errors else "alignment produced no files", "files": []}


def trajectory_frame_count(source_job: Path, manifest: dict[str, Any]) -> int:
    summary = read_optional_json(source_job / "result" / "summary.json")
    qc_summary = read_optional_json(source_job / "analysis" / "qc" / "summary.json")
    for value in (
        summary.get("trajectory_frames"),
        summary.get("mmpbsa_frames"),
        qc_summary.get("frames"),
        (manifest.get("frame_settings") or {}).get("expected_mmpbsa_frames"),
    ):
        parsed = numeric(value)
        if parsed is not None and parsed > 0:
            return int(round(parsed))
    return 0


def alignment_masks(receptor_mask: str) -> list[tuple[str, str]]:
    return [("backbone", f"{receptor_mask}@N,CA,C"), ("heavy", f"{receptor_mask}&!@H=")]


def cpptraj_command(manifest: dict[str, Any], script: Path) -> list[str]:
    profile = manifest.get("profile") or {}
    if isinstance(profile, dict) and isinstance(profile.get("runtime"), dict) and profile["runtime"].get("mamba_env"):
        return mamba_command(profile, ["cpptraj", "-i", str(script)])
    return ["cpptraj", "-i", str(script)]


def cpptraj_alignment_text(
    complex_prmtop: Path,
    trajectory: Path,
    output_dir: Path,
    alignment_mask: str,
    *,
    frames: int,
    stride: int,
    snapshots_only: bool,
) -> str:
    stride = max(1, int(stride))
    frames = max(1, int(frames))
    mid = max(1, frames // 2)
    lines = [f"parm {complex_prmtop}"]
    if snapshots_only:
        lines.extend(
            [
                f"trajin {trajectory} 1 1 1",
                f"trajin {trajectory} {mid} {mid} 1",
                f"trajin {trajectory} {frames} {frames} 1",
            ]
        )
    else:
        lines.extend(
            [
                f"trajin {trajectory} 1 {frames} {stride}",
            ]
        )
    lines.extend(
        [
            f"reference {trajectory} 1",
            f"rms visual_fit {alignment_mask} reference",
            f"trajout {output_dir / 'aligned_trajectory.raw.pdb'} pdb multi",
            "run",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize_aligned_pdbs(structures: Path, receptor_last: int, *, snapshots_only: bool) -> set[str]:
    files: set[str] = set()
    numbered = sorted(structures.glob("aligned_trajectory.raw.pdb.*"), key=numbered_pdb_suffix)
    raw_traj = structures / "aligned_trajectory.raw.pdb"
    if raw_traj.exists() and not numbered:
        numbered = [raw_traj]
    if numbered:
        snapshot_sources = [numbered[0], numbered[len(numbered) // 2], numbered[-1]]
        for stem, source in zip(("first", "mid", "last"), snapshot_sources):
            label_chains(source, structures / f"{stem}.pdb", receptor_last)
            files.add(f"{stem}.pdb")
    if numbered and not snapshots_only:
        merge_numbered_pdbs(numbered, structures / "aligned_trajectory.pdb", receptor_last)
        files.add("aligned_trajectory.pdb")
    cleanup_raw_alignment_outputs(structures)
    return files


def cleanup_raw_alignment_outputs(structures: Path) -> None:
    for pattern in ("*.raw.pdb", "*.raw.pdb.*"):
        for path in structures.glob(pattern):
            path.unlink(missing_ok=True)


def numbered_pdb_suffix(path: Path) -> int:
    try:
        return int(path.suffix.lstrip("."))
    except ValueError:
        return 0


def merge_numbered_pdbs(frame_paths: list[Path], output: Path, receptor_last: int) -> None:
    lines: list[str] = ["REMARK aligned trajectory generated by mmpbsa visualize bundle"]
    for idx, frame_path in enumerate(frame_paths, start=1):
        lines.append(f"MODEL     {idx:4d}")
        for raw in frame_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.startswith(("END", "MODEL", "ENDMDL")):
                continue
            lines.append(label_pdb_line(raw, receptor_last))
        lines.append("ENDMDL")
    lines.append("END")
    write_text_atomic(output, "\n".join(lines) + "\n")


def label_pdb_line(line: str, receptor_last: int) -> str:
    # Keep the public helper path-based; reproduce the same residue-chain rule for in-memory trajectory merging.
    if not line.startswith(("ATOM  ", "HETATM", "TER")) or len(line) < 26:
        return line
    try:
        residue_index = int(line[22:26])
    except ValueError:
        return line
    chain = "A" if residue_index <= receptor_last else "B"
    return f"{line[:21]}{chain}{line[22:]}"


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def portable_pymol_script(*, snapshots_only: bool, copied: set[str], state_count: int | None, autoplay: bool) -> str:
    lines = [
        "reinitialize",
    ]
    if not snapshots_only and "aligned_trajectory.pdb" in copied:
        lines.append("load structures/aligned_trajectory.pdb, complex")
    elif not snapshots_only and "pymol_trajectory.pdb" in copied:
        lines.append("load structures/pymol_trajectory.pdb, complex")
    else:
        for stem in ("first", "mid", "last"):
            if f"{stem}.pdb" in copied:
                lines.append(f"load structures/{stem}.pdb, {stem}")
    lines.extend(pymol_style_lines())
    if state_count and state_count > 1:
        lines.extend(
            [
                f"mset 1 -{state_count}",
                "set movie_loop, on",
            ]
        )
        if autoplay:
            lines.append("mplay")
    lines.append("")
    return "\n".join(lines)


def static_snapshots_pml(copied: set[str]) -> str:
    lines = ["reinitialize"]
    loaded = []
    for stem in ("first", "mid", "last"):
        if f"{stem}.pdb" in copied:
            lines.append(f"load structures/{stem}.pdb, {stem}")
            loaded.append(stem)
    if not loaded and "aligned_trajectory.pdb" in copied:
        lines.extend(
            [
                "load structures/aligned_trajectory.pdb, complex",
                "python",
                "from pymol import cmd",
                "n = max(1, cmd.count_states('complex'))",
                "cmd.create('first', 'complex', 1, 1)",
                "cmd.create('mid', 'complex', 1, max(1, n // 2))",
                "cmd.create('last', 'complex', 1, n)",
                "python end",
                "delete complex",
            ]
        )
        loaded = ["first", "mid", "last"]
    lines.extend(pymol_style_lines())
    for idx, stem in enumerate(loaded):
        receptor = RECEPTOR_COLORS[idx % len(RECEPTOR_COLORS)]
        binder_carbon = BINDER_CARBON_COLORS[idx % len(BINDER_CARBON_COLORS)]
        lines.extend(
            [
                f"color {receptor}, {stem} and chain A",
                f"color {binder_carbon}, {stem} and chain B and elem C",
            ]
        )
    if loaded:
        lines.extend(["set grid_mode, 1", "zoom all, 8"])
    lines.append("")
    return "\n".join(lines)


def pymol_style_lines() -> list[str]:
    lines = [
        "hide everything",
        "show cartoon, chain A",
        "show sticks, chain B",
        f"show sticks, resn {COFACTOR_RESN}",
        "show spheres, resn MG",
        "color gray70, chain A",
        "color cyan, chain B and elem C",
    ]
    for element, color in ELEMENT_COLORS.items():
        lines.append(f"color {color}, chain B and elem {element}")
    lines.extend(
        [
            f"color orange, resn {COFACTOR_RESN}",
            "color lime, resn MG",
            "set cartoon_transparency, 0.08",
            "set stick_radius, 0.16",
            "set sphere_scale, 0.35",
            "set cartoon_fancy_helices, on",
            "set all_states, off",
            "bg_color white",
            "orient",
            "zoom chain B, 8",
        ]
    )
    return lines


def render_frames_pml(copied: set[str], state_count: int | None) -> str:
    return "\n".join(
        [
            portable_pymol_script(snapshots_only=False, copied=copied, state_count=state_count, autoplay=False),
            "set ray_opaque_background, off",
            "set antialias, 2",
            "viewport 1280, 900",
            "bg_color white",
            "python",
            "import os",
            "os.makedirs('frames', exist_ok=True)",
            "python end",
            f"mpng frames/frame, first=1, last={max(1, int(state_count or 1))}",
            "quit",
            "",
        ]
    )


def render_video_shell_script() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")"',
            'if ! command -v pymol >/dev/null 2>&1; then echo "missing pymol"; exit 2; fi',
            'if ! command -v ffmpeg >/dev/null 2>&1; then echo "missing ffmpeg"; exit 2; fi',
            "rm -rf frames movie.mp4",
            "mkdir -p frames",
            "pymol -cq render_frames.pml",
            "ffmpeg -y -framerate 12 -pattern_type glob -i 'frames/frame*.png' -pix_fmt yuv420p movie.mp4",
            "",
        ]
    )


def try_render_video(target_job: Path) -> dict[str, Any]:
    missing = [name for name in ("pymol", "ffmpeg") if shutil.which(name) is None]
    if missing:
        return {"video_status": "skipped", "video_reason": "missing tools: " + ", ".join(missing)}
    log = target_job / "logs" / "render_video.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(["bash", "render_video.sh"], cwd=target_job, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    write_text_atomic(log, process.stdout)
    if process.returncode != 0:
        return {"video_status": "failed", "video_reason": f"render_video.sh exited {process.returncode}", "video_log": str(log)}
    return {"video_status": "rendered", "video_path": str(target_job / "movie.mp4"), "video_log": str(log)}


def bundle_readme(jobs: list[dict[str, Any]], snapshots_only: bool) -> str:
    lines = [
        "# MMPBSA PyMOL Bundle",
        "",
        "Open a selected job directory and run:",
        "",
        "```bash",
        "pymol load_pymol.pml",
        "```",
        "",
        f"- Snapshots only: {str(snapshots_only).lower()}",
        "- Relative paths are used, so the bundle can be moved to another machine.",
        "",
        "## Jobs",
        "",
    ]
    for job in jobs:
        lines.append(f"- `{job['job_id']}`: `{job['bundle_job_dir']}`")
    lines.append("")
    return "\n".join(lines)


def archive_stem(name: str) -> str:
    path = Path(name)
    return path.stem if path.suffix == ".zip" else path.name


def report_archive_path(output_dir: Path, archive_name: str | None) -> Path:
    if archive_name:
        path = Path(archive_name)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        if not path.is_absolute():
            path = output_dir.parent / path
        return path.resolve()
    return (output_dir.parent / f"{output_dir.name}.zip").resolve()


def write_zip(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if archive_path.resolve().is_relative_to(source_dir.resolve()):
            raise SystemExit(f"Archive path must be outside the source directory: {archive_path}")
    except AttributeError:
        pass
    temp = archive_path.with_name(f".{archive_path.name}.tmp")
    temp.unlink(missing_ok=True)
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir.parent))
    temp.replace(archive_path)


def escape(text: str) -> str:
    return html.escape(text, quote=True)


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "."

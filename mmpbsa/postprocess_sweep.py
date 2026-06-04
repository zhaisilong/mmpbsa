from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import parse_mmpbsa_full
from .common import aggregate_replica_values, mamba_command, mpi_pythonpath, read_json, run_logged, utc_now, write_csv_atomic, write_json_atomic, write_text_atomic
from .metrics import pearson_r, spearman_r

try:
    import yaml
except ModuleNotFoundError:  # Keep import cheap for --help in lean environments.
    yaml = None


GAS_CONSTANT_KJ_MOL_K = 8.31446261815324e-3
DEFAULT_TEMPERATURE_K = 298.15
DEFAULT_EPSILONS = [4.0, 8.0, 12.0, 20.0]
SCORE_KEYS = [
    "GB_delta_total_kJ_mol",
    "PB_delta_total_kJ_mol",
    "GB_dMM_kJ_mol",
    "PB_dMM_kJ_mol",
    "GB_vdw_kJ_mol",
    "PB_vdw_kJ_mol",
]
POLICIES = {
    "all_as_reported": lambda row: True,
    "non_censored": lambda row: not bool(row.get("is_censored")),
    "exact_only": lambda row: str(row.get("endpoint_relation") or "").strip() == "=",
}
E_FLOAT_RE = re.compile(r"(?P<key>\b(?:epsin|indi|dielc|saltcon|istrng)\s*=\s*)(?P<value>[-+]?\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class ReplicaInput:
    name: str
    source_dir: Path
    complex_prmtop: Path
    receptor_prmtop: Path
    peptide_prmtop: Path
    trajectory: Path
    mmpbsa_input: Path
    baseline_output: Path


@dataclass(frozen=True)
class SweepJob:
    job_id: str
    job_dir: Path
    job_config: dict[str, Any]
    manifest: dict[str, Any]
    summary: dict[str, Any]
    replicas: list[ReplicaInput]


def parse_epsilons(text: str | None) -> list[float]:
    if not text:
        return list(DEFAULT_EPSILONS)
    values = []
    for chunk in text.replace(";", ",").split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise SystemExit("At least one epsilon value is required.")
    return sorted(dict.fromkeys(values))


def format_epsilon(value: float) -> str:
    text = f"{value:g}".replace(".", "p")
    return f"eps{text}"


def mmpbsa_input_with_epsilon(template: str, epsilon: float, salt_molar: float) -> str:
    replacements = {
        "epsin": epsilon,
        "indi": epsilon,
        "dielc": epsilon,
        "saltcon": salt_molar,
        "istrng": salt_molar,
    }

    def replace(match: re.Match[str]) -> str:
        key = match.group("key")
        name = key.split("=", 1)[0].strip().lower()
        return f"{key}{replacements[name]:.3f}"

    return E_FLOAT_RE.sub(replace, template)


def peptide_postprocess_sweep(
    run_dir: Path,
    output_dir: Path,
    assay_dir: Path | None,
    epsilons: list[float] | None = None,
    salt_molar: float = 0.150,
    np_ranks: int = 16,
    max_workers: int = 1,
    job_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    epsilons = epsilons or list(DEFAULT_EPSILONS)
    output_dir = output_dir.resolve()
    run_dir = run_dir.resolve()
    jobs = discover_sweep_jobs(run_dir, job_id=job_id)
    if not jobs:
        raise SystemExit(f"No completed peptide jobs found in {run_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "mmpbsa.peptide.postprocess_sweep.v1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "job_count": len(jobs),
        "jobs": [job.job_id for job in jobs],
        "epsilons": epsilons,
        "salt_molar": salt_molar,
        "np": np_ranks,
        "max_workers": max_workers,
        "dry_run": dry_run,
        "baseline_note": "Existing epsilon=4 results are imported when present; non-baseline epsilons are recomputed.",
    }
    write_json_atomic(output_dir / "sweep_manifest.json", manifest)
    if dry_run:
        return write_dry_run_report(output_dir, manifest, jobs)

    assay_records = load_assay_records(assay_dir) if assay_dir else {}
    tasks = [(job, epsilon, replica) for job in jobs for epsilon in epsilons for replica in job.replicas]
    if max_workers <= 1:
        replica_rows = [run_or_import_replica(job, replica, epsilon, salt_molar, np_ranks, output_dir, force=force) for job, epsilon, replica in tasks]
    else:
        replica_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_or_import_replica, job, replica, epsilon, salt_molar, np_ranks, output_dir, force)
                for job, epsilon, replica in tasks
            ]
            for future in as_completed(futures):
                replica_rows.append(future.result())
        replica_rows.sort(key=lambda row: (str(row.get("job_id")), float(row.get("epsilon") or 0.0), str(row.get("replica"))))

    job_rows = aggregate_job_rows(jobs, replica_rows, assay_records)
    descriptor_rows = build_descriptor_rows(jobs, output_dir)
    attach_descriptors(job_rows, descriptor_rows)
    correlation_rows = compute_correlation_rows(job_rows)
    residual_rows, residual_correlation_rows = compute_residual_rows(job_rows)
    correlation_rows.extend(residual_correlation_rows)
    stop_rule_rows = compute_stop_rule_rows(correlation_rows)
    effect_rows = compute_sweep_effect_rows(job_rows)

    write_csv_atomic(output_dir / "sweep_replica_results.csv", replica_rows)
    write_csv_atomic(output_dir / "sweep_results.csv", job_rows)
    write_csv_atomic(output_dir / "sweep_descriptors.csv", list(descriptor_rows.values()))
    write_csv_atomic(output_dir / "sweep_correlations.csv", correlation_rows)
    write_csv_atomic(output_dir / "sweep_residuals.csv", residual_rows)
    write_csv_atomic(output_dir / "sweep_stop_rules.csv", stop_rule_rows)
    write_csv_atomic(output_dir / "sweep_effects.csv", effect_rows)
    report = {
        **manifest,
        "generated_at": utc_now(),
        "replica_result_count": len(replica_rows),
        "job_result_count": len(job_rows),
        "failed_replica_count": sum(1 for row in replica_rows if row.get("status") != "valid"),
        "correlations": correlation_rows,
        "stop_rules": stop_rule_rows,
        "sweep_effects": effect_rows,
        "outputs": {
            "replica_results": str(output_dir / "sweep_replica_results.csv"),
            "results": str(output_dir / "sweep_results.csv"),
            "descriptors": str(output_dir / "sweep_descriptors.csv"),
            "correlations": str(output_dir / "sweep_correlations.csv"),
            "residuals": str(output_dir / "sweep_residuals.csv"),
            "stop_rules": str(output_dir / "sweep_stop_rules.csv"),
            "effects": str(output_dir / "sweep_effects.csv"),
            "html": str(output_dir / "report_mmpbsa_sweep.html"),
        },
    }
    write_json_atomic(output_dir / "sweep_report.json", report)
    write_text_atomic(output_dir / "report_mmpbsa_sweep.html", render_html_report(report, job_rows, correlation_rows, stop_rule_rows))
    copy_report_to_correlation_dir(output_dir / "report_mmpbsa_sweep.html", run_dir)
    return report


def discover_sweep_jobs(run_dir: Path, job_id: str | None = None) -> list[SweepJob]:
    directories = [run_dir / job_id] if job_id else sorted(path for path in run_dir.iterdir() if path.is_dir())
    jobs: list[SweepJob] = []
    for job_dir in directories:
        config_path = job_dir / f"{job_dir.name}.json"
        manifest_path = job_dir / "manifest.json"
        summary_path = job_dir / "result" / "summary.json"
        mmpbsa_dir = job_dir / "analysis" / "mmpbsa"
        if not (config_path.exists() and manifest_path.exists() and summary_path.exists() and mmpbsa_dir.exists()):
            continue
        manifest = read_json(manifest_path)
        summary = read_json(summary_path)
        replicas = discover_replicas(mmpbsa_dir, manifest)
        if not replicas:
            continue
        jobs.append(SweepJob(job_dir.name, job_dir, read_json(config_path), manifest, summary, replicas))
    return jobs


def discover_replicas(mmpbsa_dir: Path, manifest: dict[str, Any]) -> list[ReplicaInput]:
    names = list(manifest.get("replicas") or [])
    if not names:
        names = sorted(path.name for path in mmpbsa_dir.glob("rep*") if path.is_dir())
    replicas = []
    for name in names:
        rep_dir = mmpbsa_dir / name
        replica = ReplicaInput(
            name=name,
            source_dir=rep_dir,
            complex_prmtop=rep_dir / "complex.prmtop",
            receptor_prmtop=rep_dir / "receptor.prmtop",
            peptide_prmtop=rep_dir / "peptide.prmtop",
            trajectory=rep_dir / "md_prod_dry_center.nc",
            mmpbsa_input=rep_dir / "mmpbsa.in",
            baseline_output=rep_dir / "FINAL_RESULTS_MMPBSA.dat",
        )
        if all(path.exists() for path in (replica.complex_prmtop, replica.receptor_prmtop, replica.peptide_prmtop, replica.trajectory, replica.mmpbsa_input)):
            replicas.append(replica)
    return replicas


def write_dry_run_report(output_dir: Path, manifest: dict[str, Any], jobs: list[SweepJob]) -> dict[str, Any]:
    rows = [
        {
            "job_id": job.job_id,
            "replica_count": len(job.replicas),
            "replicas": ",".join(replica.name for replica in job.replicas),
            "status": job.summary.get("status"),
        }
        for job in jobs
    ]
    write_csv_atomic(output_dir / "sweep_dry_run_jobs.csv", rows)
    report = {**manifest, "jobs_discovered": rows, "outputs": {"jobs": str(output_dir / "sweep_dry_run_jobs.csv")}}
    write_json_atomic(output_dir / "sweep_dry_run.json", report)
    return report


def run_or_import_replica(
    job: SweepJob,
    replica: ReplicaInput,
    epsilon: float,
    salt_molar: float,
    np_ranks: int,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    is_baseline = math.isclose(epsilon, 4.0, rel_tol=0.0, abs_tol=1e-9) and replica.baseline_output.exists()
    if is_baseline:
        output = replica.baseline_output
        source = "baseline_import"
        work_dir = replica.source_dir
        log = ""
    else:
        work_dir = output_dir / "work" / format_epsilon(epsilon) / job.job_id / replica.name
        output = work_dir / "FINAL_RESULTS_MMPBSA.dat"
        log_path = output_dir / "logs" / f"{job.job_id}_{replica.name}_{format_epsilon(epsilon)}.log"
        log = str(log_path)
        if force and work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        stage_replica_inputs(replica, work_dir, epsilon, salt_molar)
        if not output.exists():
            profile = profile_for_job(job, np_ranks)
            command = mmpbsa_command(profile, np_ranks)
            try:
                run_logged(mamba_command(profile, command), log_path, cwd=work_dir, env={"PYTHONPATH": mpi_pythonpath(profile)})
            except Exception as exc:
                return {
                    "job_id": job.job_id,
                    "replica": replica.name,
                    "epsilon": epsilon,
                    "salt_molar": salt_molar,
                    "status": "failed",
                    "source": "computed",
                    "work_dir": str(work_dir),
                    "output": str(output),
                    "log": log,
                    "error": str(exc),
                }
        source = "computed"

    try:
        parsed = parse_mmpbsa_full(output)
    except Exception as exc:
        return {
            "job_id": job.job_id,
            "replica": replica.name,
            "epsilon": epsilon,
            "salt_molar": salt_molar,
            "status": "failed",
            "source": source,
            "work_dir": str(work_dir),
            "output": str(output),
            "log": log,
            "error": f"parse failed: {exc}",
        }

    row: dict[str, Any] = {
        "job_id": job.job_id,
        "replica": replica.name,
        "epsilon": epsilon,
        "salt_molar": salt_molar,
        "status": "valid",
        "source": source,
        "frames": parsed.get("frames"),
        "work_dir": str(work_dir),
        "output": str(output),
        "log": log,
    }
    row.update(parsed.get("values", {}))
    return row


def profile_for_job(job: SweepJob, np_ranks: int) -> dict[str, Any]:
    profile = json.loads(json.dumps(job.manifest.get("profile") or {}))
    if not profile:
        raise RuntimeError(f"{job.job_id} manifest does not contain profile settings")
    env = os.environ.get("MAMBA_ENV")
    if env:
        profile.setdefault("runtime", {})["mamba_env"] = env
    profile.setdefault("mmpbsa", {})["np"] = np_ranks
    return profile


def mmpbsa_command(profile: dict[str, Any], np_ranks: int) -> list[str]:
    if np_ranks > 1:
        return [
            "mpirun",
            "-np",
            str(np_ranks),
            "MMPBSA.py.MPI",
            "-O",
            "-i",
            "mmpbsa.in",
            "-o",
            "FINAL_RESULTS_MMPBSA.dat",
            "-eo",
            "per_frame_energy.csv",
            "-cp",
            "complex.prmtop",
            "-rp",
            "receptor.prmtop",
            "-lp",
            "peptide.prmtop",
            "-y",
            "md_prod_dry_center.nc",
        ]
    return [
        "MMPBSA.py",
        "-O",
        "-i",
        "mmpbsa.in",
        "-o",
        "FINAL_RESULTS_MMPBSA.dat",
        "-eo",
        "per_frame_energy.csv",
        "-cp",
        "complex.prmtop",
        "-rp",
        "receptor.prmtop",
        "-lp",
        "peptide.prmtop",
        "-y",
        "md_prod_dry_center.nc",
    ]


def stage_replica_inputs(replica: ReplicaInput, work_dir: Path, epsilon: float, salt_molar: float) -> None:
    for source, name in (
        (replica.complex_prmtop, "complex.prmtop"),
        (replica.receptor_prmtop, "receptor.prmtop"),
        (replica.peptide_prmtop, "peptide.prmtop"),
        (replica.trajectory, "md_prod_dry_center.nc"),
    ):
        target = work_dir / name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source.resolve())
    template = replica.mmpbsa_input.read_text(encoding="utf-8")
    write_text_atomic(work_dir / "mmpbsa.in", mmpbsa_input_with_epsilon(template, epsilon, salt_molar))


def aggregate_job_rows(jobs: list[SweepJob], replica_rows: list[dict[str, Any]], assay_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows_by_job_eps: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in replica_rows:
        if row.get("status") == "valid":
            rows_by_job_eps.setdefault((str(row["job_id"]), float(row["epsilon"])), []).append(row)

    for job in jobs:
        ligand_id = ligand_id_for_job(job)
        assay = assay_records.get(ligand_id, {})
        for epsilon in sorted({float(row["epsilon"]) for row in replica_rows if row.get("job_id") == job.job_id}):
            reps = rows_by_job_eps.get((job.job_id, epsilon), [])
            values = aggregate_replica_values([{key: float(rep[key]) for key in SCORE_KEYS if isinstance(rep.get(key), (int, float))} for rep in reps])
            row: dict[str, Any] = {
                "job_id": job.job_id,
                "ligand_id": ligand_id,
                "name": job.job_config.get("name") or job.summary.get("name") or job.job_id,
                "mutation_label": job.job_config.get("mutation_label") or assay.get("mutation_label") or job.job_config.get("name") or job.job_id,
                "epsilon": epsilon,
                "salt_molar": reps[0].get("salt_molar") if reps else "",
                "status": "valid" if reps else "missing",
                "replica_count": len(reps),
                "mmpbsa_frames": sum(float(rep.get("frames") or 0.0) for rep in reps),
                "endpoint_type": assay.get("endpoint_type"),
                "endpoint_relation": assay.get("endpoint_relation"),
                "standard_value_nM": assay.get("standard_value_nM"),
                "experimental_deltaG_kJ_mol": assay.get("experimental_deltaG_kJ_mol"),
                "is_censored": assay.get("is_censored"),
                "peptide_rmsd_mean": average_replica_qc(job.summary, "peptide_bb_rmsd_after_receptor_fit_angstrom", "mean"),
                "native_contacts_mean": average_native_contact(job.summary, "rec_pep[native]", "mean"),
            }
            row.update(values)
            rows.append(row)
    return rows


def ligand_id_for_job(job: SweepJob) -> str:
    value = job.job_config.get("ligand_id")
    if value:
        return str(value)
    match = re.search(r"(PEP_\d{4})", job.job_id)
    return match.group(1) if match else job.job_id


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


def load_assay_records(assay_dir: Path) -> dict[str, dict[str, Any]]:
    if yaml is None:
        raise SystemExit("PyYAML is required to load assay files.")
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
            endpoint_relation = str(record.get("endpoint_relation") or "").strip()
            records[str(record["ligand_id"])] = {
                "ligand_id": record.get("ligand_id"),
                "mutation_label": record.get("mutation_label"),
                "endpoint_type": record.get("endpoint_type"),
                "endpoint_relation": endpoint_relation,
                "standard_value_nM": value_nm,
                "experimental_deltaG_kJ_mol": delta_g_from_nm(value_nm),
                "is_censored": endpoint_relation.startswith(">") or endpoint_relation.startswith("<"),
                "assay_file": str(path),
            }
    return records


def delta_g_from_nm(value_nm: float) -> float:
    return GAS_CONSTANT_KJ_MOL_K * DEFAULT_TEMPERATURE_K * math.log(value_nm * 1e-9)


def build_descriptor_rows(jobs: list[SweepJob], output_dir: Path) -> dict[str, dict[str, Any]]:
    descriptors = {}
    for job in jobs:
        first_replica = job.replicas[0]
        row: dict[str, Any] = {
            "job_id": job.job_id,
            "ligand_id": ligand_id_for_job(job),
            "peptide_residue_count": job.manifest.get("peptide_residue_count"),
            "receptor_residue_mask": job.manifest.get("receptor_residue_mask"),
            "peptide_residue_mask": job.manifest.get("peptide_residue_mask"),
            "native_contacts_mean": average_native_contact(job.summary, "rec_pep[native]", "mean"),
        }
        row.update(peptide_prmtop_descriptors(first_replica.peptide_prmtop))
        row["peptide_sasa_lcpo_angstrom2"] = peptide_sasa(job, output_dir)
        descriptors[job.job_id] = row
    return descriptors


def peptide_prmtop_descriptors(prmtop: Path) -> dict[str, Any]:
    try:
        import parmed as pmd
    except ModuleNotFoundError:
        return {"peptide_charge": None, "peptide_atom_count": None, "peptide_heavy_atom_count": None}
    structure = pmd.load_file(str(prmtop))
    return {
        "peptide_charge": round(sum(atom.charge for atom in structure.atoms), 6),
        "peptide_atom_count": len(structure.atoms),
        "peptide_heavy_atom_count": sum(1 for atom in structure.atoms if getattr(atom, "atomic_number", 0) != 1),
    }


def peptide_sasa(job: SweepJob, output_dir: Path) -> float | None:
    profile = profile_for_job(job, int((job.manifest.get("profile", {}).get("mmpbsa", {}) or {}).get("np", 1) or 1))
    work = output_dir / "descriptors" / job.job_id
    work.mkdir(parents=True, exist_ok=True)
    out = work / "peptide_sasa_lcpo.dat"
    if not out.exists():
        script = work / "peptide_sasa.in"
        write_text_atomic(
            script,
            f"""parm {job.replicas[0].complex_prmtop}
trajin {job.replicas[0].trajectory}
surf peptide_sasa {job.manifest["peptide_residue_mask"]}&!@H= out {out}
run
""",
        )
        try:
            run_logged(mamba_command(profile, ["cpptraj", "-i", str(script)]), output_dir / "logs" / f"{job.job_id}_peptide_sasa.log")
        except Exception:
            return None
    values = read_cpptraj_single_series(out)
    return sum(values) / len(values) if values else None


def read_cpptraj_single_series(path: Path) -> list[float]:
    values = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("@"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            values.append(float(parts[1]))
        except ValueError:
            continue
    return values


def attach_descriptors(job_rows: list[dict[str, Any]], descriptor_rows: dict[str, dict[str, Any]]) -> None:
    for row in job_rows:
        descriptor = descriptor_rows.get(str(row["job_id"]), {})
        for key, value in descriptor.items():
            if key not in {"job_id", "ligand_id"}:
                row[key] = value


def compute_correlation_rows(job_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    epsilons = sorted({float(row["epsilon"]) for row in job_rows})
    for epsilon in epsilons:
        by_eps = [row for row in job_rows if float(row["epsilon"]) == epsilon and row.get("status") == "valid"]
        for policy, predicate in POLICIES.items():
            subset = [row for row in by_eps if predicate(row)]
            for metric in SCORE_KEYS:
                score_pairs = [(float(row["experimental_deltaG_kJ_mol"]), float(row[metric])) for row in subset if is_number(row.get("experimental_deltaG_kJ_mol")) and is_number(row.get(metric))]
                charge_pairs = [(float(row["peptide_charge"]), float(row[metric])) for row in subset if is_number(row.get("peptide_charge")) and is_number(row.get(metric))]
                sd_key = f"{metric}_replica_sd"
                sd_values = [float(row[sd_key]) for row in subset if is_number(row.get(sd_key))]
                rows.append(
                    {
                        "score_type": "raw",
                        "epsilon": epsilon,
                        "policy": policy,
                        "metric": metric,
                        "n_experiment": len(score_pairs),
                        "experiment_pearson_r": pearson_r([x for x, _ in score_pairs], [y for _, y in score_pairs]),
                        "experiment_spearman_r": spearman_r([x for x, _ in score_pairs], [y for _, y in score_pairs]),
                        "n_charge": len(charge_pairs),
                        "charge_pearson_r": pearson_r([x for x, _ in charge_pairs], [y for _, y in charge_pairs]),
                        "charge_spearman_r": spearman_r([x for x, _ in charge_pairs], [y for _, y in charge_pairs]),
                        "replica_sd_mean": sum(sd_values) / len(sd_values) if sd_values else None,
                    }
                )
    return rows


def compute_residual_rows(job_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    residual_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    predictors = ["peptide_charge", "peptide_residue_count", "peptide_sasa_lcpo_angstrom2", "native_contacts_mean"]
    for epsilon in sorted({float(row["epsilon"]) for row in job_rows}):
        by_eps = [row for row in job_rows if float(row["epsilon"]) == epsilon and row.get("status") == "valid"]
        for metric in SCORE_KEYS:
            usable = [row for row in by_eps if is_number(row.get(metric)) and all(is_number(row.get(key)) for key in predictors)]
            if len(usable) < 3:
                continue
            residuals = ridge_loocv_residuals(usable, metric, predictors)
            for row, residual in zip(usable, residuals):
                residual_rows.append(
                    {
                        "epsilon": epsilon,
                        "metric": metric,
                        "job_id": row["job_id"],
                        "ligand_id": row.get("ligand_id"),
                        "raw_score": row.get(metric),
                        "residual_score": residual,
                        "experimental_deltaG_kJ_mol": row.get("experimental_deltaG_kJ_mol"),
                        "endpoint_relation": row.get("endpoint_relation"),
                        "is_censored": row.get("is_censored"),
                    }
                )
            for policy, predicate in POLICIES.items():
                subset = [
                    (row, residual)
                    for row, residual in zip(usable, residuals)
                    if predicate(row) and is_number(row.get("experimental_deltaG_kJ_mol"))
                ]
                xs = [float(row["experimental_deltaG_kJ_mol"]) for row, _ in subset]
                ys = [float(residual) for _, residual in subset]
                correlation_rows.append(
                    {
                        "score_type": "ridge_residual",
                        "epsilon": epsilon,
                        "policy": policy,
                        "metric": metric,
                        "n_experiment": len(xs),
                        "experiment_pearson_r": pearson_r(xs, ys),
                        "experiment_spearman_r": spearman_r(xs, ys),
                        "n_charge": len(subset),
                        "charge_pearson_r": pearson_r(
                            [float(row["peptide_charge"]) for row, _ in subset if is_number(row.get("peptide_charge"))],
                            [float(residual) for row, residual in subset if is_number(row.get("peptide_charge"))],
                        ),
                        "charge_spearman_r": spearman_r(
                            [float(row["peptide_charge"]) for row, _ in subset if is_number(row.get("peptide_charge"))],
                            [float(residual) for row, residual in subset if is_number(row.get("peptide_charge"))],
                        ),
                        "replica_sd_mean": None,
                    }
                )
    return residual_rows, correlation_rows


def ridge_loocv_residuals(rows: list[dict[str, Any]], metric: str, predictors: list[str], alpha: float = 1.0) -> list[float]:
    x_all = np.array([[float(row[key]) for key in predictors] for row in rows], dtype=float)
    y_all = np.array([float(row[metric]) for row in rows], dtype=float)
    residuals = []
    for index in range(len(rows)):
        train_mask = np.ones(len(rows), dtype=bool)
        train_mask[index] = False
        x_train = x_all[train_mask]
        y_train = y_all[train_mask]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0.0] = 1.0
        x_train_s = (x_train - mean) / std
        x_hold_s = (x_all[index : index + 1] - mean) / std
        x_aug = np.column_stack([np.ones(len(x_train_s)), x_train_s])
        penalty = np.eye(x_aug.shape[1]) * alpha
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(x_aug.T @ x_aug + penalty) @ x_aug.T @ y_train
        y_hat = float((np.column_stack([np.ones(1), x_hold_s]) @ beta)[0])
        residuals.append(float(y_all[index] - y_hat))
    return residuals


def compute_stop_rule_rows(correlation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    raw = [row for row in correlation_rows if row.get("score_type") == "raw" and row.get("policy") == "all_as_reported"]
    for row in raw:
        charge_r = abs(float(row["charge_pearson_r"])) if is_number(row.get("charge_pearson_r")) else None
        exp_r = abs(float(row["experiment_pearson_r"])) if is_number(row.get("experiment_pearson_r")) else None
        passed = charge_r is not None and exp_r is not None and charge_r <= 0.7 and exp_r >= 0.3
        rows.append(
            {
                "epsilon": row["epsilon"],
                "metric": row["metric"],
                "score_type": "raw",
                "abs_charge_pearson_r": charge_r,
                "abs_experiment_pearson_r": exp_r,
                "passed_basic_stop_rule": passed,
                "interpretation": "candidate_raw_ranking" if passed else "raw_total_not_reliable_for_mixed_charge_length_ranking",
            }
        )
    return rows


def compute_sweep_effect_rows(job_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    job_ids = sorted({str(row["job_id"]) for row in job_rows})
    for metric in SCORE_KEYS:
        max_deltas = []
        changed_jobs = 0
        for job_id in job_ids:
            values = [float(row[metric]) for row in job_rows if str(row["job_id"]) == job_id and is_number(row.get(metric))]
            if len(values) < 2:
                continue
            delta = max(values) - min(values)
            max_deltas.append(delta)
            if abs(delta) > 1e-6:
                changed_jobs += 1
        max_abs_delta = max((abs(value) for value in max_deltas), default=None)
        mean_abs_delta = sum(abs(value) for value in max_deltas) / len(max_deltas) if max_deltas else None
        rows.append(
            {
                "metric": metric,
                "jobs_checked": len(max_deltas),
                "jobs_changed": changed_jobs,
                "max_abs_delta_across_eps": max_abs_delta,
                "mean_abs_delta_across_eps": mean_abs_delta,
                "epsilon_effective": bool(max_abs_delta is not None and max_abs_delta > 1e-6),
                "note": effect_note(metric, bool(max_abs_delta is not None and max_abs_delta > 1e-6)),
            }
        )
    return rows


def effect_note(metric: str, effective: bool) -> str:
    if effective:
        return "Score changed across epsilon values."
    if metric.startswith("GB_"):
        return "Score did not change across epsilon values; current Amber MMPBSA mmpbsa_py_energy GB backend does not expose an effective solute dielectric sweep for this igb=5 run."
    return "Score did not change across epsilon values."


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def copy_report_to_correlation_dir(report_path: Path, run_dir: Path) -> None:
    project_root = run_dir.resolve().parent
    correlation_dir = project_root / "correlation"
    if not correlation_dir.exists():
        return
    shutil.copyfile(report_path, correlation_dir / report_path.name)


def render_html_report(report: dict[str, Any], job_rows: list[dict[str, Any]], correlation_rows: list[dict[str, Any]], stop_rule_rows: list[dict[str, Any]]) -> str:
    raw_all = [
        row
        for row in correlation_rows
        if row.get("score_type") == "raw" and row.get("policy") == "all_as_reported" and row.get("metric") in {"PB_delta_total_kJ_mol", "GB_delta_total_kJ_mol"}
    ]
    residual_all = [
        row
        for row in correlation_rows
        if row.get("score_type") == "ridge_residual" and row.get("policy") == "all_as_reported" and row.get("metric") in {"PB_delta_total_kJ_mol", "GB_delta_total_kJ_mol"}
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>KRAS MMPBSA post-processing sweep</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #20242a; }}
    h1, h2 {{ margin-bottom: 0.35rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; font-size: 13px; }}
    th, td {{ border: 1px solid #d3d7dc; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f5f7; }}
    .note {{ color: #555f6d; max-width: 980px; }}
    .bad {{ color: #9b1c1c; font-weight: 600; }}
    .ok {{ color: #116329; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>KRAS MMPBSA post-processing sweep</h1>
  <p class="note">Generated {escape(report.get("generated_at"))}. Existing epsilon=4 results are imported as the AF3/GDP/no-Mg baseline; other epsilon values are recomputed from the same dry trajectories and topology files.</p>
  <h2>Run Summary</h2>
  <table>
    <tr><th>Run dir</th><td>{escape(report.get("run_dir"))}</td></tr>
    <tr><th>Output dir</th><td>{escape(report.get("output_dir"))}</td></tr>
    <tr><th>Jobs</th><td>{report.get("job_count")}</td></tr>
    <tr><th>Replica rows</th><td>{report.get("replica_result_count")}</td></tr>
    <tr><th>Failed replica rows</th><td>{report.get("failed_replica_count")}</td></tr>
    <tr><th>Epsilons</th><td>{escape(", ".join(str(x) for x in report.get("epsilons", [])))}</td></tr>
    <tr><th>Salt molar</th><td>{report.get("salt_molar")}</td></tr>
  </table>
  <h2>Sweep Effect Check</h2>
  <p class="note">This table verifies whether each score actually changed when epsilon changed. In this AmberTools build, PB responds to <code>indi</code>; GB rows are unchanged for the current <code>igb=5</code>/<code>mmpbsa_py_energy</code> backend, so GB epsilon rows should be interpreted as baseline controls rather than a true GB dielectric sweep.</p>
  {render_table(report.get("sweep_effects", []), ["metric", "jobs_checked", "jobs_changed", "max_abs_delta_across_eps", "mean_abs_delta_across_eps", "epsilon_effective", "note"], class_field="epsilon_effective")}
  <h2>Raw Total Correlations</h2>
  {render_table(raw_all, ["epsilon", "metric", "n_experiment", "experiment_pearson_r", "experiment_spearman_r", "n_charge", "charge_pearson_r", "charge_spearman_r", "replica_sd_mean"])}
  <h2>Residual Total Correlations</h2>
  {render_table(residual_all, ["epsilon", "metric", "n_experiment", "experiment_pearson_r", "experiment_spearman_r", "n_charge", "charge_pearson_r", "charge_spearman_r"])}
  <h2>Stop Rule</h2>
  {render_stop_table(stop_rule_rows)}
  <h2>Per-Variant PB Total</h2>
  {render_table([row for row in job_rows if is_number(row.get("PB_delta_total_kJ_mol"))], ["epsilon", "job_id", "mutation_label", "standard_value_nM", "peptide_charge", "peptide_residue_count", "PB_delta_total_kJ_mol", "PB_delta_total_kJ_mol_replica_sd", "native_contacts_mean"])}
</body>
</html>
"""


def render_stop_table(rows: list[dict[str, Any]]) -> str:
    fields = ["epsilon", "metric", "abs_charge_pearson_r", "abs_experiment_pearson_r", "passed_basic_stop_rule", "interpretation"]
    return render_table(rows, fields, class_field="passed_basic_stop_rule")


def render_table(rows: list[dict[str, Any]], fields: list[str], class_field: str | None = None) -> str:
    if not rows:
        return "<p class=\"note\">No rows available.</p>"
    lines = ["<table>", "<thead><tr>" + "".join(f"<th>{escape(field)}</th>" for field in fields) + "</tr></thead>", "<tbody>"]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            klass = ""
            if class_field == field:
                klass = " class=\"ok\"" if bool(value) else " class=\"bad\""
            cells.append(f"<td{klass}>{escape(format_value(value))}</td>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))

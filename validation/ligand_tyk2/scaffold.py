from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmpbsa.metrics import linear_fit, pearson_r, spearman_r


DEFAULT_TYK2_LIGANDS = ["lig_ejm_46", "lig_ejm_54", "lig_ejm_31", "lig_ejm_50", "lig_ejm_43"]
GAS_CONSTANT_KJ_MOL_K = 8.31446261815324e-3
DEFAULT_EXPERIMENT_TEMPERATURE_K = 298.15
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "ligand_crystal_3x15ns_mmpbsa_bcc.yaml"


@dataclass(frozen=True)
class LigandMeasurement:
    name: str
    measurement_type: str
    value: float
    unit: str
    delta_g_kj_mol: float


@dataclass(frozen=True)
class RunResult:
    job_id: str
    gpu_id: str
    returncode: int
    log: Path


def parse_csv_option(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_gpu_list(value: str) -> list[str]:
    gpus = parse_csv_option(value, [])
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU id")
    return gpus


def target_root(resources_dir: Path, target: str) -> Path:
    root = resources_dir.resolve() / "data" / target
    if not root.exists():
        raise SystemExit(f"Missing validation target directory: {root}")
    return root


def load_ligand_measurements(resources_dir: Path, target: str) -> dict[str, LigandMeasurement]:
    path = target_root(resources_dir, target) / "00_data" / "ligands.yml"
    if not path.exists():
        raise SystemExit(f"Missing ligand metadata: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    measurements: dict[str, LigandMeasurement] = {}
    for name, entry in raw.items():
        measurement = entry.get("measurement", {}) or {}
        unit = str(measurement.get("unit") or "").strip()
        value = float(measurement["value"])
        measurements[name] = LigandMeasurement(
            name=name,
            measurement_type=str(measurement.get("type") or "").strip(),
            value=value,
            unit=unit,
            delta_g_kj_mol=experimental_delta_g_kj_mol(value, unit),
        )
    return measurements


def experimental_delta_g_kj_mol(value: float, unit: str, temperature_k: float = DEFAULT_EXPERIMENT_TEMPERATURE_K) -> float:
    unit_key = unit.strip().lower()
    scale_by_unit = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "nm": 1e-9,
    }
    if unit_key not in scale_by_unit:
        raise SystemExit(f"Unsupported affinity unit {unit!r}; supported units: M, mM, uM, nM")
    concentration_molar = value * scale_by_unit[unit_key]
    if concentration_molar <= 0:
        raise SystemExit(f"Affinity value must be positive, got {value!r} {unit}")
    return GAS_CONSTANT_KJ_MOL_K * temperature_k * math.log(concentration_molar)


def load_sdf_records(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Missing ligand SDF: {path}")
    records: dict[str, str] = {}
    for block in path.read_text(encoding="utf-8", errors="replace").split("$$$$"):
        lines = block.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue
        name = lines[0].strip()
        records[name] = "\n".join(lines).rstrip() + "\n$$$$\n"
    return records


def make_ligand_jobs(resources_dir: Path, run_dir: Path, target: str, ligand_names: list[str]) -> list[Path]:
    root = target_root(resources_dir, target)
    protein = root / "01_protein" / "crd" / "protein.pdb"
    sdf = root / "02_ligands" / "ligands.sdf"
    if not protein.exists():
        raise SystemExit(f"Missing validation protein PDB: {protein}")
    measurements = load_ligand_measurements(resources_dir, target)
    records = load_sdf_records(sdf)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ligand_name in ligand_names:
        if ligand_name not in records:
            raise SystemExit(f"Ligand {ligand_name!r} is not present in {sdf}")
        if ligand_name not in measurements:
            raise SystemExit(f"Ligand {ligand_name!r} is not present in ligand metadata")
        job_id = f"{target}_{ligand_name}"
        job_dir = run_dir / job_id
        source_dir = job_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        ligand_sdf = source_dir / f"{ligand_name}.sdf"
        ligand_sdf.write_text(records[ligand_name], encoding="utf-8")
        measurement = measurements[ligand_name]
        config = {
            "job_id": job_id,
            "name": ligand_name,
            "pdb_id": target.upper(),
            "source": "openforcefield/protein-ligand-benchmark",
            "validation_target": target,
            "validation_ligand_name": ligand_name,
            "benchmark_target": target,
            "benchmark_ligand_name": ligand_name,
            "complex_pdb": str(protein.resolve()),
            "receptor_chains": "A",
            "ligand_file": f"source/{ligand_name}.sdf",
            "ligand_resname": "LIG",
            "ligand_charge": 0,
            "ligand_param_mode": "auto",
            "measurement_type": measurement.measurement_type,
            "measurement_value": measurement.value,
            "measurement_unit": measurement.unit,
            "deltaG_exp_kJ_mol": measurement.delta_g_kj_mol,
        }
        config_path = job_dir / f"{job_id}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        written.append(config_path)
    return written


def discover_job_ids(run_dir: Path) -> list[str]:
    root = run_dir.resolve()
    if not root.exists():
        raise SystemExit(f"RUN_DIR does not exist: {root}")
    return [path.name for path in sorted(root.iterdir()) if path.is_dir() and (path / f"{path.name}.json").exists()]


def assign_jobs_to_gpus(job_ids: list[str], gpu_ids: list[str], max_workers: int) -> dict[str, list[str]]:
    worker_count = min(max_workers, len(gpu_ids), len(job_ids))
    if worker_count <= 0:
        return {}
    assignments = {gpu_ids[idx]: [] for idx in range(worker_count)}
    worker_gpus = list(assignments)
    for idx, job_id in enumerate(job_ids):
        assignments[worker_gpus[idx % worker_count]].append(job_id)
    return assignments


def run_ligand_jobs(
    run_dir: Path,
    protocol_path: Path,
    gpu_ids: list[str],
    max_workers: int,
    ntomp: int,
    mmpbsa_np: int,
    mode: str = "full",
    force: bool = False,
) -> list[RunResult]:
    job_ids = discover_job_ids(run_dir)
    assignments = assign_jobs_to_gpus(job_ids, gpu_ids, max_workers)
    if not assignments:
        raise SystemExit(f"No ligand job configs found under {run_dir.resolve()}")
    results: list[RunResult] = []
    print("GPU assignments:")
    for gpu_id, assigned in assignments.items():
        print(f"  GPU {gpu_id}: {', '.join(assigned)}")
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [
            executor.submit(run_assigned_jobs, run_dir.resolve(), protocol_path.resolve(), gpu_id, assigned, ntomp, mmpbsa_np, mode, force)
            for gpu_id, assigned in assignments.items()
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    failures = [result for result in results if result.returncode != 0]
    if failures:
        details = ", ".join(f"{item.job_id} on GPU {item.gpu_id} (log: {item.log})" for item in failures)
        raise SystemExit(f"One or more ligand jobs failed: {details}")
    return sorted(results, key=lambda item: item.job_id)


def run_assigned_jobs(
    run_dir: Path,
    protocol_path: Path,
    gpu_id: str,
    job_ids: list[str],
    ntomp: int,
    mmpbsa_np: int,
    mode: str,
    force: bool = False,
) -> list[RunResult]:
    results: list[RunResult] = []
    for job_id in job_ids:
        job_dir = run_dir / job_id
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        driver_log = logs_dir / f"validation_runner_gpu{gpu_id}.log"
        env = os.environ.copy()
        env.update({"GPU_ID": gpu_id, "NTOMP": str(ntomp), "MMPBSA_NP": str(mmpbsa_np)})
        command = [
            sys.executable,
            "-m",
            "mmpbsa",
            "ligand",
            "run",
            str(run_dir),
            "--job-id",
            job_id,
            "--protocol",
            str(protocol_path),
            "--mode",
            mode,
        ]
        command.append("--force" if force else "--resume")
        print(f"{job_id}: start on GPU {gpu_id}")
        with driver_log.open("w", encoding="utf-8") as handle:
            handle.write("# command: " + " ".join(command) + "\n")
            handle.write(f"# GPU_ID={gpu_id} NTOMP={ntomp} MMPBSA_NP={mmpbsa_np}\n\n")
            handle.flush()
            process = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        print(f"{job_id}: finished on GPU {gpu_id} with code {process.returncode}")
        results.append(RunResult(job_id=job_id, gpu_id=gpu_id, returncode=process.returncode, log=driver_log))
        if process.returncode != 0:
            break
    return results


def write_validation_report(run_dir: Path, resources_dir: Path, target: str, output: Path) -> dict[str, Any]:
    measurements = load_ligand_measurements(resources_dir, target)
    rows: list[dict[str, Any]] = []
    for job_id in discover_job_ids(run_dir):
        config = json.loads((run_dir / job_id / f"{job_id}.json").read_text(encoding="utf-8"))
        ligand_name = config.get("validation_ligand_name") or config.get("benchmark_ligand_name") or config.get("name") or job_id
        measurement = measurements[str(ligand_name)]
        job_dir = run_dir / job_id
        summary = load_json_if_exists(job_dir / "result" / "summary.json")
        audit = load_json_if_exists(job_dir / "analysis" / "mmpbsa" / "audit.json")
        row = {
            "job_id": job_id,
            "ligand": ligand_name,
            "measurement_type": measurement.measurement_type,
            "measurement_value": measurement.value,
            "measurement_unit": measurement.unit,
            "experimental_deltaG_kJ_mol": measurement.delta_g_kj_mol,
            "status": summary.get("status", "incomplete"),
            "replica_count": summary.get("replica_count", audit.get("replica_count")),
            "mmpbsa_frames": summary.get("mmpbsa_frames", audit.get("frames")),
            "GB_delta_total_kJ_mol": metric_value(summary, audit, "GB_delta_total_kJ_mol"),
            "PB_delta_total_kJ_mol": metric_value(summary, audit, "PB_delta_total_kJ_mol"),
            "GB_dMM_kJ_mol": metric_value(summary, audit, "GB_dMM_kJ_mol"),
            "PB_dMM_kJ_mol": metric_value(summary, audit, "PB_dMM_kJ_mol"),
            "GB_delta_total_kJ_mol_replica_sd": metric_sd(summary, audit, "GB_delta_total_kJ_mol"),
            "PB_delta_total_kJ_mol_replica_sd": metric_sd(summary, audit, "PB_delta_total_kJ_mol"),
            "GB_dMM_kJ_mol_replica_sd": metric_sd(summary, audit, "GB_dMM_kJ_mol"),
            "PB_dMM_kJ_mol_replica_sd": metric_sd(summary, audit, "PB_dMM_kJ_mol"),
        }
        rows.append(row)
    rows.sort(key=lambda row: float(row["experimental_deltaG_kJ_mol"]))
    correlations = {
        key: correlation_record(rows, key)
        for key in [
            "GB_delta_total_kJ_mol",
            "PB_delta_total_kJ_mol",
            "GB_dMM_kJ_mol",
            "PB_dMM_kJ_mol",
        ]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_markdown(target, run_dir, rows, correlations), encoding="utf-8")
    return {"output": str(output.resolve()), "rows": len(rows), "correlations": correlations}


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {}


def metric_value(summary: dict[str, Any], audit: dict[str, Any], key: str) -> float | None:
    for alias in metric_aliases(key):
        value = summary.get(alias)
        if isinstance(value, (int, float)):
            return float(value)
    audit_values = audit.get("values", {})
    if isinstance(audit_values, dict):
        for alias in metric_aliases(key):
            value = audit_values.get(alias)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def metric_sd(summary: dict[str, Any], audit: dict[str, Any], key: str) -> float | None:
    for alias in metric_aliases(key):
        for suffix in ("_replica_sd", "_sd"):
            value = summary.get(f"{alias}{suffix}")
            if isinstance(value, (int, float)):
                return float(value)
    audit_values = audit.get("values", {})
    if isinstance(audit_values, dict):
        for alias in metric_aliases(key):
            for suffix in ("_replica_sd", "_sd"):
                value = audit_values.get(f"{alias}{suffix}")
                if isinstance(value, (int, float)):
                    return float(value)
    replicas = audit.get("replicas", [])
    if not isinstance(replicas, list):
        return None
    values: list[float] = []
    for replica in replicas:
        if not isinstance(replica, dict) or not isinstance(replica.get("values"), dict):
            continue
        for alias in metric_aliases(key):
            value = replica["values"].get(alias)
            if isinstance(value, (int, float)):
                values.append(float(value))
                break
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def metric_aliases(key: str) -> list[str]:
    if "_dMM_" in key:
        return [key, key.replace("_dMM_", "_dmm_like_")]
    return [key]


def correlation_record(rows: list[dict[str, Any]], computed_key: str) -> dict[str, Any]:
    pairs = [
        (float(row["experimental_deltaG_kJ_mol"]), float(row[computed_key]))
        for row in rows
        if isinstance(row.get(computed_key), (int, float))
    ]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson_r": None, "spearman_r": None, "slope": None, "intercept": None}
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    slope, intercept = linear_fit(xs, ys)
    return {"n": len(pairs), "pearson_r": pearson_r(xs, ys), "spearman_r": spearman_r(xs, ys), "slope": slope, "intercept": intercept}


def report_markdown(target: str, run_dir: Path, rows: list[dict[str, Any]], correlations: dict[str, dict[str, Any]]) -> str:
    completed = [row for row in rows if row["status"] == "valid"]
    best_key = best_correlation_key(correlations)
    lines = [
        f"# {target.upper()} Ligand MMPBSA Validation",
        "",
        f"- Run directory: `{project_relative(run_dir)}`",
        "- Source: `openforcefield/protein-ligand-benchmark`",
        f"- Experimental conversion: `DeltaG = RT ln(K)`, `T = {DEFAULT_EXPERIMENT_TEMPERATURE_K:.2f} K`",
        "",
        "## Results",
        "",
        "| ligand | status | replicas | frames | exp DeltaG kJ/mol | GB total mean +- SD | PB total mean +- SD | GB dMM mean +- SD | PB dMM mean +- SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {ligand} | {status} | {replicas} | {frames} | {exp} | {gb} | {pb} | {gbd} | {pbd} |".format(
                ligand=row["ligand"],
                status=row["status"],
                replicas=format_int(row.get("replica_count")),
                frames=format_int(row.get("mmpbsa_frames")),
                exp=format_float(row["experimental_deltaG_kJ_mol"]),
                gb=format_mean_sd(row.get("GB_delta_total_kJ_mol"), row.get("GB_delta_total_kJ_mol_replica_sd")),
                pb=format_mean_sd(row.get("PB_delta_total_kJ_mol"), row.get("PB_delta_total_kJ_mol_replica_sd")),
                gbd=format_mean_sd(row.get("GB_dMM_kJ_mol"), row.get("GB_dMM_kJ_mol_replica_sd")),
                pbd=format_mean_sd(row.get("PB_dMM_kJ_mol"), row.get("PB_dMM_kJ_mol_replica_sd")),
            )
        )
    lines.extend(["", "## Correlation Lines", ""])
    for key, record in correlations.items():
        if record["pearson_r"] is None:
            lines.append(f"- `{key}`: insufficient completed jobs (`n={record['n']}`).")
        else:
            lines.append(
                f"- `{key}`: computed = {record['slope']:.4f} * experimental + {record['intercept']:.4f}; "
                f"Pearson r = {record['pearson_r']:.4f}; Spearman r = {record['spearman_r']:.4f}; n = {record['n']}."
            )
    lines.extend(["", "## Interpretation", ""])
    if best_key is None:
        lines.append("- No completed correlation line is available yet.")
    else:
        best = correlations[best_key]
        lines.append(
            f"- The strongest line in this subset is `{best_key}` with Pearson r = {best['pearson_r']:.4f} "
            f"over {best['n']} completed ligands."
        )
    if completed:
        lines.append(
            f"- All {len(completed)} completed ligand jobs passed trajectory QC and MMPBSA audit."
        )
    lines.append(
        "- Computed MM/PBSA values are substantially shifted relative to experimental DeltaG; "
        "interpret this validation primarily through relative ordering and correlation, not absolute agreement."
    )
    lines.append(
        "- The five-ligand subset is a pipeline validation run. A production validation should expand the ligand set "
        "before drawing method-level conclusions."
    )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report is generated from a small five-ligand TYK2 subset and is versioned as project validation evidence.",
            f"- Reported frame counts are read from existing MMPBSA audit files; completed rows here use {frame_count_note(rows)}.",
            "- MM/PBSA absolute values are not expected to match experimental affinities directly; use the correlation lines as a pipeline diagnostic.",
            "- Entropy is disabled in the default ligand validation profiles; the report therefore focuses on GB, PB, and dMM scores.",
        ]
    )
    return "\n".join(lines) + "\n"


def best_correlation_key(correlations: dict[str, dict[str, Any]]) -> str | None:
    candidates = [
        (abs(float(record["pearson_r"])), key)
        for key, record in correlations.items()
        if record.get("pearson_r") is not None and not math.isnan(float(record["pearson_r"]))
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def frame_count_note(rows: list[dict[str, Any]]) -> str:
    counts = sorted(
        {
            int(float(row["mmpbsa_frames"]))
            for row in rows
            if isinstance(row.get("mmpbsa_frames"), (int, float))
        }
    )
    if not counts:
        return "no completed MMPBSA frames"
    if len(counts) == 1:
        return f"{counts[0]} total frames per job"
    return ", ".join(str(count) for count in counts) + " total frames per job"


def format_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.3f}"


def format_int(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return str(int(float(value)))


def format_mean_sd(value: Any, sd: Any) -> str:
    mean = format_float(value)
    if not mean:
        return ""
    if not isinstance(sd, (int, float)):
        return mean
    return f"{mean} +- {float(sd):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local TYK2 ligand validation helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_parser = subparsers.add_parser("make-ligand-jobs", help="Write TYK2 ligand job directories from validation resources.")
    make_parser.add_argument("resources_dir", type=Path)
    make_parser.add_argument("run_dir", type=Path)
    make_parser.add_argument("--target", default="tyk2")
    make_parser.add_argument("--ligands", help="Comma-separated ligand names. Defaults to the selected TYK2 subset.")

    run_parser = subparsers.add_parser("run-ligand-jobs", help="Run prepared ligand jobs across local GPUs.")
    run_parser.add_argument("run_dir", type=Path)
    run_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    run_parser.add_argument("--gpus", default="2,3")
    run_parser.add_argument("--jobs", dest="max_workers", type=int, default=2)
    run_parser.add_argument("--ntomp", type=int, default=4)
    run_parser.add_argument("--mmpbsa-np", type=int, default=16)
    run_parser.add_argument("--mode", choices=["full", "prepare", "md", "analysis", "report"], default="full")
    run_parser.add_argument("--force", action="store_true")

    report_parser = subparsers.add_parser("report", help="Generate the local TYK2 validation report.")
    report_parser.add_argument("run_dir", type=Path)
    report_parser.add_argument("resources_dir", type=Path)
    report_parser.add_argument("--target", default="tyk2")
    report_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "make-ligand-jobs":
        selected = parse_csv_option(args.ligands, DEFAULT_TYK2_LIGANDS)
        paths = make_ligand_jobs(args.resources_dir, args.run_dir, args.target, selected)
        print(json.dumps({"jobs_written": len(paths), "configs": [str(path) for path in paths]}, indent=2))
    elif args.command == "run-ligand-jobs":
        results = run_ligand_jobs(
            args.run_dir,
            args.protocol,
            parse_gpu_list(args.gpus),
            args.max_workers,
            args.ntomp,
            args.mmpbsa_np,
            mode=args.mode,
            force=args.force,
        )
        print(json.dumps([result.__dict__ | {"log": str(result.log)} for result in results], indent=2))
    elif args.command == "report":
        report = write_validation_report(args.run_dir, args.resources_dir, args.target, args.output)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

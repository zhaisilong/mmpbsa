from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import write_csv_atomic, write_json_atomic, write_text_atomic


def completed_summaries(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(run_dir.resolve().glob("*/result/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data.setdefault("job_dir", str(summary_path.parents[1]))
        rows.append(data)
    return rows


def aggregate_run_dir(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows = completed_summaries(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(output_dir / "summary.csv", rows)
    qc_rows = [
        {
            "job_id": row.get("job_id", ""),
            "status": row.get("status", ""),
            "trajectory_qc_status": row.get("trajectory_qc_status", ""),
            "mmpbsa_qc_status": row.get("mmpbsa_qc_status", ""),
            "mmpbsa_frames": row.get("mmpbsa_frames", ""),
            "trajectory_frames": row.get("trajectory_frames", ""),
        }
        for row in rows
    ]
    write_csv_atomic(output_dir / "qc_summary.csv", qc_rows)
    report = {
        "run_dir": str(run_dir.resolve()),
        "jobs_total": len(list(path for path in run_dir.resolve().iterdir() if path.is_dir())),
        "jobs_completed": len(rows),
        "output_dir": str(output_dir.resolve()),
    }
    write_json_atomic(output_dir / "summary.json", report)
    write_text_atomic(
        output_dir / "report.md",
        "\n".join(
            [
                "# MMPBSA Result Report",
                "",
                f"- Run directory: `{report['run_dir']}`",
                f"- Completed jobs: {report['jobs_completed']}",
                f"- Total job directories: {report['jobs_total']}",
                "",
                "See `summary.csv` and `qc_summary.csv` for tabular results.",
                "",
            ]
        ),
    )
    return report

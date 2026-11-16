#!/usr/bin/env python3
"""Run all SP2016 modeling preflights and emit a concise or JSON summary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from modeling.common.preflight import preflight_case

ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, help="Write the complete JSON result to this path")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete JSON result instead of a one-line summary",
    )
    parser.add_argument("--eligible-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog = json.loads((ROOT / "catalog" / "cases.json").read_text(encoding="utf-8"))
    selected = [
        case for case in catalog["cases"] if not args.eligible_only or case["status"] == "eligible"
    ]
    results = [preflight_case(ROOT / case["model_file"], args.source_root) for case in selected]
    status_counts: dict[str, int] = {}
    for result in results:
        key = result["preflight_status"]
        status_counts[key] = status_counts.get(key, 0) + 1
    failed = status_counts.get("failed", 0)
    payload = {
        "schema_version": 1,
        "validation_id": catalog["validation_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": "eligible_only" if args.eligible_only else "all",
        "summary": {
            "selected": len(selected),
            "eligible_catalog": sum(case["status"] == "eligible" for case in selected),
            "rejected_catalog": sum(case["status"] == "rejected" for case in selected),
            "preflight_failed": failed,
            "status_counts": status_counts,
            "production_topologies_present": sum(
                result["modeling"]["production_topology_present"] is True
                for result in results
                if "modeling" in result
            ),
        },
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.json:
        sys.stdout.write(rendered)
    else:
        summary = payload["summary"]
        destination = str(args.output) if args.output else "-"
        print(
            f"{catalog['validation_id']}: selected={summary['selected']} "
            f"eligible={summary['eligible_catalog']} "
            f"rejected={summary['rejected_catalog']} "
            f"failed={summary['preflight_failed']} output={destination}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

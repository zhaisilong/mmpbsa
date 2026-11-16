#!/usr/bin/env python3
"""Thin CLI for the package-level preparation contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mmpbsa.backends.prep import (
    build_preparation_plan,
    load_preparation_runtime,
    load_preparation_spec,
    prepare_case,
)

BASE_DIR = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=BASE_DIR / "preparation.yaml",
        help="typed preparation document",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=BASE_DIR / "runtime.local.yaml",
        help="shared mmpbsa.runtime.v1 tool configuration",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate sources and print the resolved, non-writing plan",
    )
    mode.add_argument(
        "--output",
        type=Path,
        help="new immutable PreparedCase directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    spec = load_preparation_spec(arguments.config)
    runtime = load_preparation_runtime(arguments.runtime)
    if arguments.dry_run:
        result: object = build_preparation_plan(spec, tool_config=runtime)
    else:
        result = {
            "manifest": str(
                prepare_case(spec, arguments.output, tool_config=runtime)
            ),
            "case_id": spec.case_id,
        }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

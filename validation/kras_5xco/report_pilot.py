from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import report_kras_5xco_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the KRAS 5XCO GDP-only/GDP+Mg validation pilot.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assay-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path)
    args = parser.parse_args()

    report = report_kras_5xco_pilot(args.run_dir, args.output_dir, args.assay_dir, baseline_run_dir=args.baseline_run_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

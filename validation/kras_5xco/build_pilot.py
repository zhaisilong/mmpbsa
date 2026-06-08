from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the KRAS 5XCO GDP-only/GDP+Mg validation pilot.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--template-cif", type=Path, required=True)
    parser.add_argument("--mg-source-cif", type=Path)
    parser.add_argument("--download-mg-source", action="store_true")
    parser.add_argument("--gdp-lib", type=Path, required=True)
    parser.add_argument("--gdp-frcmod", type=Path, required=True)
    parser.add_argument("--variants", help="Comma-separated variant IDs. Defaults to WT,D13A,L8A,P14A,del4R,core13.")
    parser.add_argument("--states", help="Comma-separated receptor states: gdp_only,gdp_mg.")
    parser.add_argument("--production-ns", type=float, default=20.0)
    parser.add_argument("--mmpbsa-start-ns", type=float, default=10.0)
    parser.add_argument("--seed-base", type=int, default=2026060401)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from .pilot import build_kras_5xco_pilot, parse_states, parse_variant_ids

    report = build_kras_5xco_pilot(
        args.output_dir,
        template_cif=args.template_cif,
        mg_source_cif=args.mg_source_cif,
        download_mg_source=args.download_mg_source,
        gdp_lib=args.gdp_lib,
        gdp_frcmod=args.gdp_frcmod,
        variants=parse_variant_ids(args.variants),
        states=parse_states(args.states),
        production_ns=args.production_ns,
        mmpbsa_start_ns=args.mmpbsa_start_ns,
        seed_base=args.seed_base,
        force=args.force,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

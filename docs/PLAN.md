# Unified MMPBSA Pipeline Plan

## Summary

- v0.1.1 is prepared for GitHub publication at `git@github.com:zhaisilong/mmpbsa.git`.
- Initialize the project as a git repository with `README.md`, `CHANGELOG.md`, `docs/PLAN.md`, `docs/setups.md`, `scripts/`, `configs/`, `tests/`, and a Python package.
- Refactor the peptide workflow from `mmpbsa_v3` and the small-molecule workflow from `mmpbsa_sm` into one package named `mmpbsa`.
- Use explicit CLI flows for peptide and small-molecule jobs instead of inferring type from fields.
- Use `RUN_DIR/<job_id>/<job_id>.json` as the run-independent job configuration.
- Use only `.xxx_done` files for step state control.

## Interfaces

- `mmpbsa peptide run RUN_DIR [--job-id JOB_ID] [--protocol PATH] [--mode full] [--resume|--force]`
- `mmpbsa ligand run RUN_DIR [--job-id JOB_ID] [--protocol PATH] [--mode full] [--resume|--force]`
- `mmpbsa status RUN_DIR [--job-id JOB_ID]`
- `mmpbsa aggregate RUN_DIR --output-dir PATH`
- `mmpbsa doctor [--protocol PATH]`
- `mmpbsa frame-settings [--protocol PATH]`
- `mmpbsa benchmark make-ligand-jobs BENCHMARK_DATA_DIR RUN_DIR [--target tyk2]`
- `mmpbsa benchmark run-ligand-jobs RUN_DIR [--gpus 2,3] [--jobs 2]`
- `mmpbsa benchmark report RUN_DIR BENCHMARK_DATA_DIR --output docs/tyk2_ligand_benchmark.md`

`--protocol` points to YAML files such as `configs/default_15ns.yaml`,
`configs/ligand_crystal_3x5ns.yaml`, `configs/ligand_crystal_5x5ns.yaml`,
`configs/ligand_crystal_3x5ns_mmpbsa_bcc.yaml`,
`configs/peptide_crystal_3x5ns.yaml`, `configs/peptide_crystal_5x5ns.yaml`,
and `configs/smoke_20ps.yaml`.

## Runtime Model

- `full`: run all stages.
- `prepare`: input preparation, ligand preparation when needed, Amber preparation.
- `md`: GROMACS conversion, EM, NVT, NPT, production MD.
- `analysis`: trajectory processing, QC, MMPBSA, audit.
- `report`: summarize existing QC/MMPBSA outputs into `result/summary.json` and `result/summary.csv`.

Default behavior fails when selected steps already have done files. `--resume` skips them. `--force` clears the selected mode and downstream done files/outputs.

## Implementation Notes

- Shared `JobContext` handles job discovery, protocol loading, path resolution, and environment overrides.
- Shared `DoneFileRunner` handles mode expansion and `.xxx_done/.xxx_failed` policy.
- `PeptidePipeline` and `LigandPipeline` own only domain-specific calculation steps.
- The package keeps generated `manifest.json` as derived run metadata, but it is not used for state control.
- TYK2 ligand benchmark helpers generate one `RUN_DIR/<job_id>/<job_id>.json`
  per ligand and schedule jobs across the requested GPU IDs.
- Ligand crystal defaults use independent replica directories under
  `md/repXX`; MMPBSA can be skipped for production MD-only runs or enabled for
  benchmark profiles.
- Ligand MMPBSA uses MM/GBSA as the primary ranking score and MM/PBSA as a
  secondary score. Entropy is off by default and reserved for explicit
  sensitivity profiles. Replica results are
  averaged in `analysis/mmpbsa/audit.json`.
- Peptide crystal profiles use independent `md/repXX` directories and aggregate
  dry frames for MMPBSA. Stability QC is reported per replica.
- Peptide dielectric uses explicit profile config when present; otherwise it
  uses charged/polar/nonpolar interface classification. PB entropy-corrected
  values are diagnostic outputs only when entropy is explicitly enabled; peptide
  explicit-water MMPBSA is intentionally fail-fast until a dedicated sensitivity
  branch is implemented.

## Test Plan

- Unit tests cover job discovery, job ID validation, done-file behavior, mode dependency checks, frame settings, ligand helpers, and tleap text generation.
- CLI help test is present and runs when `click` is installed.
- Real MD/MMPBSA execution should be validated through `configs/smoke_20ps.yaml` after the `md` mamba environment and GROMACS are available.
- Peptide migration was validated against v3 outputs for `sp2016_11`.
- Ligand benchmark validation records experimental DeltaG conversion and MMPBSA
  correlation lines when an external benchmark dataset is provided.
- Ligand unit tests cover 3x5ns frame accounting, RESP fail-fast behavior,
  automatic dielectric classification, fixed explicit-water selection, and
  MMPBSA input generation.
- Peptide unit tests cover 3x5ns frame accounting, dielectric config priority,
  automatic peptide interface dielectric classification, and MMPBSA input
  generation.

## Defaults

- CLI uses `--protocol`, not `--profile`.
- Ligand CLI default is `configs/ligand_crystal_3x5ns.yaml`; peptide CLI
  default remains `configs/default_15ns.yaml`.
- Production ligand defaults expect RESP-charged input. The TYK2 benchmark
  helper defaults to `configs/ligand_crystal_3x5ns_mmpbsa_bcc.yaml` because the
  benchmark source does not provide RESP charges.
- `report` is the user-facing replacement for the old `finalize` step name.
- Relative paths in job JSON resolve against the job directory.
- Multi-job execution is directory traversal over immediate child directories.

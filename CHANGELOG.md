# Changelog

## v0.1.0 - 2026-06-02

- Added the local Spiliotopoulos 2016 `core8 + second4` peptide 3x5ns MMPBSA
  test pipeline documentation under `pipeline_tests/peptide_3x5ns`.
- Documented the GPU 4-7 four-worker run layout for peptide test jobs.
- Initialized unified `mmpbsa` package with Click-based peptide and ligand command groups.
- Added directory-based job discovery using `RUN_DIR/<job_id>/<job_id>.json`.
- Replaced runtime state JSON control with `.xxx_done` and `.xxx_failed` sentinel files.
- Migrated peptide and small-molecule Amber/GROMACS/MMPBSA logic from the reference workflows.
- Added shared `--protocol` YAML handling, status, aggregate, frame-settings, and doctor commands.
- Added setup documentation and unittest coverage for discovery, mode/done policy, protocol parsing, and ligand helpers.
- Added MPI readiness checks to `doctor` and cleanup of stale `_MMPBSA_*` files on forced analysis reruns.
- Validated peptide migration from v3 MD outputs using `pipeline_tests/peptide/sp2016_11`.
- Added TYK2 ligand benchmark helpers for `openforcefield/protein-ligand-benchmark`, including job generation, two-GPU scheduling, and correlation report generation.
- Added `configs/ligand_default_15ns.yaml` and EM-log validation for protein-ligand MD stability checks.
- Added ligand crystal defaults with 3x5ns and optional 5x5ns independent replicas.
- Switched protein defaults to Amber ff14SB and ligand PB/GB radii to `mbondi2`.
- Added RESP fail-fast policy for production ligand defaults and an explicit AM1-BCC benchmark profile.
- Added ligand MMPBSA gating, automatic job-level dielectric selection, fixed interface-water retention, and replica-level MMPBSA aggregation.
- Added peptide crystal 3x5ns/5x5ns profiles, peptide MMPBSA dielectric policy, per-replica stability QC, and PB entropy-corrected diagnostic output.

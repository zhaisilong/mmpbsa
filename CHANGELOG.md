# Changelog

## v0.1.5 - 2026-06-03

- Added ligand `3x15ns`, `3x15ns_mmpbsa_bcc`, and `1x15ns` crystal-start
  profiles matching the peptide long-replica workflow.
- Added `mmpbsa ligand run --replica-index N` and
  `mmpbsa ligand merge-replicas` for independently calculated ligand replica
  extensions.
- Extended ligand manifest, MMPBSA manifest, audit, and summary outputs with
  stable replica indices and seed metadata.
- Added a versioned TYK2 five-ligand 3x5ns validation report with replica
  `mean +- SD` tables and correlation lines.
- Updated the TYK2 validation scaffold default to the longer 3x15ns MMPBSA-BCC
  profile while keeping it under `validation/` as project validation tooling.

## v0.1.4 - 2026-06-03

- Added peptide `3x15ns` and `1x15ns` crystal-start profiles for longer AF3,
  cofold, and docked-structure relaxation workflows.
- Added explicit replica index handling: replica names, seeds, manifest fields,
  audit records, and summaries now track stable `repNN` indices.
- Added `mmpbsa peptide run --replica-index N` for single-replica reruns and
  `mmpbsa peptide merge-replicas` for combining independently calculated
  peptide replica summaries.

## v0.1.3 - 2026-06-03

- Made peptide HETATM handling strict by default: new jobs preserve
  `input/selected_raw.pdb`, write ATOM-only cleaned inputs for Amber, and require
  explicit `amber_prep.nonstandard_policy: strip` before dropping HETATM records.
- Added replica sample SD fields alongside existing replica SEM fields in
  aggregated MMPBSA summaries.
- Renamed computed dMM outputs to `GB_dMM_*` and `PB_dMM_*`, while retaining
  `*_dmm_like_*` compatibility aliases.
- Updated peptide and ligand validation reports to display dMM and peptide
  replica `mean +- SD`.
- Moved local validation scaffolding out of the public CLI/package surface:
  fixed peptide and TYK2 validation helpers now live under `validation/`, and
  `mmpbsa benchmark` is no longer a public command.

## v0.1.2 - 2026-06-03

- Changed peptide MMPBSA analysis to calculate each replica independently and
  aggregate replica means with SEM, matching the ligand workflow.
- Added peptide EM recovery for octahedral solvent-box water overlaps: affected
  jobs can retry once with a rectangular box and record the fallback in manifest
  and summary outputs.
- Enabled `allow_box_retry` for peptide crystal 3x5ns and 5x5ns profiles.
- Added local peptide 3x5ns validation reporting with 12/12 completed jobs and
  correlation diagnostics under `validation/peptide_3x5ns/report.py`.

## v0.1.1 - 2026-06-02

- Replaced machine-specific GROMACS `GMXRC` paths in checked-in protocols with
  the portable `${GMXRC}` environment placeholder.
- Updated runtime and doctor handling so `GMXRC`, `GMX_BIN`, and `MAMBA_ENV`
  overrides are resolved consistently.
- Added tests for portable GROMACS runtime resolution and doctor overrides.

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
- Added TYK2 ligand validation helpers for `openforcefield/protein-ligand-benchmark`, including job generation, two-GPU scheduling, and correlation report generation.
- Added `configs/ligand_default_15ns.yaml` and EM-log validation for protein-ligand MD stability checks.
- Added ligand crystal defaults with 3x5ns and optional 5x5ns independent replicas.
- Switched protein defaults to Amber ff14SB and ligand PB/GB radii to `mbondi2`.
- Added RESP fail-fast policy for production ligand defaults and an explicit AM1-BCC validation profile.
- Added ligand MMPBSA gating, automatic job-level dielectric selection, fixed interface-water retention, and replica-level MMPBSA aggregation.
- Added peptide crystal 3x5ns/5x5ns profiles, peptide MMPBSA dielectric policy, per-replica stability QC, and opt-in PB entropy-corrected diagnostic output.

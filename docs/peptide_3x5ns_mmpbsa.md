# Peptide 3x5ns MMPBSA Test Pipeline

## Current Result

- Status: 12/12 jobs complete.
- Failed markers: none.
- MMPBSA mode: each replica is calculated separately and then averaged across
  replicas with sample SD and SEM.
- Recovery note: `sp2016_10` used automatic `oct -> box` solvent fallback after
  a water-water minimum-image overlap caused EM infinite force in the original
  octahedral box.
- Input note: the historical `sp2016_10` run started from a raw selected PDB
  containing 9 sulfate HETATM residues; they were stripped before Amber
  preparation. Current peptide defaults fail fast on HETATM unless `strip` is
  configured explicitly.

Correlation diagnostics against experimental DeltaG:

| score | Pearson r | Spearman r | n |
| --- | ---: | ---: | ---: |
| `GB_delta_total_kJ_mol` | 0.4604 | 0.4545 | 12 |
| `PB_delta_total_kJ_mol` | 0.5339 | 0.5455 | 12 |
| `GB_dMM_kJ_mol` | 0.4032 | 0.4336 | 12 |
| `PB_dMM_kJ_mol` | 0.3321 | 0.4056 | 12 |
| `paper_mm_pbsa_kJ_mol` | 0.4497 | 0.3497 | 12 |
| `paper_dmm_pbsa_kJ_mol` | 0.6564 | 0.6154 | 12 |

Interpretation: this is a local pipeline validation subset. PB ranks slightly
better than GB here, but the values are not calibrated absolute binding free
energies. Use the lines as diagnostics for the run and implementation.
The `*_dMM_kJ_mol` fields are the pipeline-computed Spiliotopoulos-style dMM
scores; `paper_dmm_pbsa_kJ_mol` is the literature reference value from the
dataset.

## Dataset

- Source: Spiliotopoulos 2016 protein-peptide set.
- Cases: `core8 + second4`, 12 jobs total.
- Run directory: `pipeline_tests/peptide_3x5ns/`.
- Protocol: `configs/peptide_crystal_3x5ns.yaml`.
- MMPBSA score policy: MM/GBSA is the primary ranking score; MM/PBSA is a
  secondary check.
- Explicit interface water is not forced in this profile.
- HETATM handling is strict in the current code: new peptide jobs preserve raw
  input as `input/selected_raw.pdb`, write ATOM-only cleaned structures for
  Amber, and fail unless `amber_prep.nonstandard_policy: strip` is set
  explicitly for removable HETATM records.
- Amber-supported peptide caps and residue variants such as `ACE`, `NME`,
  `NHE`, `NH2`, `CYX`, `CYM`, `ASH`, `GLH`, `HID`, `HIE`, `HIP`, and `LYN` are
  treated as peptide residues and retained even if the input PDB uses `HETATM`.

## Job List

```text
GPU 4: sp2016_09, sp2016_01, sp2016_14
GPU 5: sp2016_17, sp2016_02, sp2016_16
GPU 6: sp2016_15, sp2016_05, sp2016_10
GPU 7: sp2016_03, sp2016_08, sp2016_06
```

## Run Commands

```bash
RUN_DIR=pipeline_tests/peptide_3x5ns
PROTOCOL=configs/peptide_crystal_3x5ns.yaml
```

Launch the GPU 4-7 workers:

```bash
setsid bash -c 'echo $$ > pipeline_tests/peptide_3x5ns/run_gpu4567.pid; \
  exec bash validation/peptide_3x5ns/run_gpu4567.sh' \
  > pipeline_tests/peptide_3x5ns/run_gpu4567.log 2>&1 < /dev/null &
```

Check frame accounting:

```bash
mamba run -n md python -m mmpbsa frame-settings --protocol "$PROTOCOL"
```

Check status:

```bash
mamba run -n md python -m mmpbsa status "$RUN_DIR"
```

Aggregate completed outputs:

```bash
mamba run -n md python -m mmpbsa aggregate "$RUN_DIR" \
  --output-dir results/peptide_3x5ns
```

Generate the local validation report:

```bash
python validation/peptide_3x5ns/report.py \
  --run-dir pipeline_tests/peptide_3x5ns \
  --output-dir results/peptide_3x5ns
```

## Acceptance Criteria

- Each completed job writes `result/summary.json`.
- Each completed job reports at least 300 MMPBSA frames.
- Each completed job includes per-replica trajectory QC.
- `GB_delta_total_kJ_mol`, `PB_delta_total_kJ_mol`, `GB_dMM_kJ_mol`, and
  `PB_dMM_kJ_mol` are present. New summaries also include
  `_replica_sd` and `_replica_sem` fields.
- Correlations are interpreted as pipeline diagnostics, not absolute binding
  free-energy validation.

All criteria are met for the current 12-job validation run.

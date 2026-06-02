# Peptide 3x5ns MMPBSA Test Pipeline

## Dataset

- Source: Spiliotopoulos 2016 protein-peptide set.
- Cases: `core8 + second4`, 12 jobs total.
- Run directory: `pipeline_tests/peptide_3x5ns/`.
- Protocol: `configs/peptide_crystal_3x5ns.yaml`.
- MMPBSA score policy: MM/GBSA is the primary ranking score; MM/PBSA is a
  secondary check.
- Explicit interface water is not forced in this profile.

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
  exec bash scripts/run_peptide_3x5ns_gpu4567.sh' \
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

## Acceptance Criteria

- Each completed job writes `result/summary.json`.
- Each completed job reports at least 300 MMPBSA frames.
- Each completed job includes per-replica trajectory QC.
- `GB_delta_total_kJ_mol` and `PB_delta_total_kJ_mol` are present.
- Correlations are interpreted as pipeline diagnostics, not absolute binding
  free-energy validation.

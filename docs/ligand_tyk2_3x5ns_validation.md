# TYK2 Ligand MMPBSA Validation

- Run directory: `pipeline_tests/ligand/tyk2_3x5ns`
- Source: `openforcefield/protein-ligand-benchmark`
- Experimental conversion: `DeltaG = RT ln(K)`, `T = 298.15 K`

## Results

| ligand | status | replicas | frames | exp DeltaG kJ/mol | GB total mean +- SD | PB total mean +- SD | GB dMM mean +- SD | PB dMM mean +- SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lig_ejm_46 | valid | 3 | 753 | -47.484 | -160.441 +- 3.133 | -89.491 +- 2.309 | -202.597 +- 3.205 | -103.234 +- 1.774 |
| lig_ejm_54 | valid | 3 | 753 | -44.207 | -162.292 +- 0.738 | -89.655 +- 0.845 | -200.513 +- 2.214 | -101.702 +- 1.217 |
| lig_ejm_31 | valid | 3 | 753 | -40.057 | -148.540 +- 3.323 | -84.610 +- 2.923 | -188.323 +- 3.872 | -97.103 +- 3.142 |
| lig_ejm_50 | valid | 3 | 753 | -37.685 | -141.285 +- 3.591 | -78.794 +- 6.719 | -184.411 +- 4.311 | -91.895 +- 5.813 |
| lig_ejm_43 | valid | 3 | 753 | -34.680 | -148.598 +- 6.021 | -81.146 +- 8.434 | -195.749 +- 7.074 | -96.239 +- 8.083 |

## Correlation Lines

- `GB_delta_total_kJ_mol`: computed = 1.4255 * experimental + -94.0385; Pearson r = 0.8187; Spearman r = 0.6000; n = 5.
- `PB_delta_total_kJ_mol`: computed = 0.8570 * experimental + -49.7550; Pearson r = 0.8965; Spearman r = 0.8000; n = 5.
- `GB_dMM_kJ_mol`: computed = 0.9905 * experimental + -153.8828; Pearson r = 0.6479; Spearman r = 0.7000; n = 5.
- `PB_dMM_kJ_mol`: computed = 0.7506 * experimental + -67.3931; Pearson r = 0.8439; Spearman r = 0.9000; n = 5.

## Interpretation

- The strongest line in this subset is `PB_delta_total_kJ_mol` with Pearson r = 0.8965 over 5 completed ligands.
- All 5 completed ligand jobs passed trajectory QC and MMPBSA audit.
- Computed MM/PBSA values are substantially shifted relative to experimental DeltaG; interpret this validation primarily through relative ordering and correlation, not absolute agreement.
- The five-ligand subset is a pipeline validation run. A production validation should expand the ligand set before drawing method-level conclusions.

## Notes

- This report is generated from a small five-ligand TYK2 subset and is versioned as project validation evidence.
- Reported frame counts are read from existing MMPBSA audit files; completed rows here use 753 total frames per job.
- MM/PBSA absolute values are not expected to match experimental affinities directly; use the correlation lines as a pipeline diagnostic.
- Entropy is disabled in the default ligand validation profiles; the report therefore focuses on GB, PB, and dMM scores.

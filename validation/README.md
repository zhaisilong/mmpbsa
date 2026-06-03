# Validation Scaffolding

This directory contains local validation helpers for this repository. These
scripts are intentionally kept outside the public `mmpbsa` package and CLI.

- `peptide_3x5ns/`: fixed Spiliotopoulos peptide 3x5ns launcher and report.
- `ligand_tyk2/`: fixed TYK2 ligand validation scaffold for the
  `openforcefield/protein-ligand-benchmark` resource layout.

Use the public pipeline through `mmpbsa peptide ...` and `mmpbsa ligand ...`.

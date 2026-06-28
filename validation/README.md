# Validation Scaffolding

This directory contains local validation helpers for this repository. These
scripts are intentionally kept outside the public `mmpbsa` package and CLI.

- `peptide_3x5ns/`: fixed Spiliotopoulos peptide 3x5ns launcher and report.
- `ligand_tyk2/`: fixed TYK2 ligand validation scaffold for the
  `openforcefield/protein-ligand-benchmark` resource layout.
- `kras_6wgn_boltz/`: local KRAS(G12D)-GNP-Mg Boltz pose scaffold. It prepares
  GNP/Mg as receptor cofactors and treats the Boltz `LIG1` cyclic ligand as the
  scored ligand. Use its strict report command for production summaries so
  smoke runs and full-window `0-5 ns` MMPBSA outputs are excluded.

Use the public pipeline through `mmpbsa peptide ...` and `mmpbsa ligand ...`.

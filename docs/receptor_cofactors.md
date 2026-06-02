# Receptor Cofactors In MMPBSA

Some targets need a bound cofactor to define the receptor state. KRAS-GDP is the
canonical case: GDP is not the ligand being scored, but part of the background
receptor state.

Use this partition when scoring a peptide or small molecule bound to a
GDP-loaded target:

```text
Complex  = Protein + GDP + ligand
Receptor = Protein + GDP
Ligand   = ligand
```

If a metal ion stabilizes the cofactor, include it with the receptor too:

```text
Receptor = Protein + GDP + Mg2+
```

This is equivalent to the multicomponent receptor pattern in gmx_MMPBSA, where
the receptor group can include more than protein. In this package, Amber
`MMPBSA.py` gets the same partition through `ante-MMPBSA.py`: the ligand mask
selects only the peptide or small molecule being scored, and the receptor
topology is the rest of the dry complex.

## Job Fields

For peptide jobs, receptor cofactors can be provided with:

```json
{
  "job_id": "kras_gdp_peptide",
  "selected_pdb": "input/complex.pdb",
  "receptor_chains": "A",
  "peptide_chains": "B",
  "receptor_cofactor_files": "input/gdp.pdb",
  "receptor_cofactor_libs": "input/GDP.prep",
  "receptor_cofactor_frcmods": "input/frcmod.phos",
  "receptor_cofactor_residue_count": 1
}
```

For small-molecule jobs, use the same `receptor_cofactor_*` fields with the
ligand workflow. The pipeline combines receptor units in this order:

```text
protein receptor, receptor cofactors, ligand
```

That order keeps the receptor residue mask contiguous and leaves the ligand mask
as the final residue range.

## GDP Parameters

Do not rely on GDP being a standard Amber protein residue. Use validated cofactor
parameters whenever possible. A practical public source is the Manchester/Bryce
AMBER parameter database entry for GDP revised phosphate parameters:

- `GDP.prep`: http://amber.manchester.ac.uk/cof/GDP.prep
- `frcmod.phos`: http://amber.manchester.ac.uk/cof/frcmod.phos
- contributor notes: http://amber.manchester.ac.uk/cof/phos_inf.html

Load those files with `loadamberprep` and `loadamberparams` in tleap. The
pipeline accepts `.prep` and `.prepi` files in `receptor_cofactor_libs`, and
`.frcmod` files in `receptor_cofactor_frcmods`.

Antechamber/GAFF parameterization is acceptable only as an exploratory fallback
for GDP-like highly charged cofactors. Production calculations should record the
parameter source, total charge, and any missing-parameter warnings.

## References

- KRAS-GDP can be treated as a receptor background state when scoring an
  additional ligand, as in KRASG12D-GDP-MRTX1133 MM/PBSA studies:
  https://www.nature.com/articles/s41598-022-22668-1
- gmx_MMPBSA documents multicomponent receptor groups:
  https://valdes-tresanco-ms.github.io/gmx_MMPBSA/v1.5.7/examples/Comp_receptor/
- Amber LEaP uses `.pdb`, `.mol2`, `.lib`, `.prepi`, and `.frcmod` inputs to
  build topology files:
  https://ambermd.org/tutorials/pengfei/index.php

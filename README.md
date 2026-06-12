# MMPBSA

Unified protein-peptide and protein-small-molecule MD + MM/PBSA pipeline.

The package provides two explicit workflows:

- `mmpbsa peptide ...`: protein-peptide MD and MMPBSA analysis.
- `mmpbsa ligand ...`: protein-small-molecule explicit-water MD, with optional MMPBSA.

## Install

```bash
git clone git@github.com:zhaisilong/mmpbsa.git
cd mmpbsa
python -m pip install -e .
```

The command-line entrypoint is `mmpbsa`.

## Environment

The pipeline expects an MD environment with AmberTools/MMPBSA.py, GROMACS,
ACPYPE, and common Amber ligand tools available. The checked-in protocols assume
a mamba environment named `md` and use `${GMXRC}` as a portable GROMACS setup
placeholder. Export runtime settings for the target machine before running:

```bash
MAMBA_ENV=md \
GMXRC=/path/to/gromacs/bin/GMXRC \
GMX_BIN=gmx_mpi \
GPU_ID=0 \
NTOMP=8 \
MMPBSA_NP=16 \
mmpbsa doctor --protocol configs/default_15ns.yaml
```

See [docs/setups.md](docs/setups.md) for dependency checks and machine-specific
configuration notes.

## Job Layout

Jobs are directory based. Given a `RUN_DIR`, each job lives in a subdirectory:

```text
RUN_DIR/
  demo_peptide/
    demo_peptide.json
  demo_ligand/
    demo_ligand.json
```

The `<job_id>.json` file is the run-independent system configuration. Runtime
outputs, logs, temporary files, and `.xxx_done` sentinel files are written in the
same job directory. Relative paths in job JSON files resolve against the job
directory.

Protein-peptide job:

```json
{
  "job_id": "demo_peptide",
  "name": "Demo peptide",
  "selected_pdb": "input/complex.pdb",
  "receptor_chains": "A",
  "peptide_chains": "B"
}
```

Peptide jobs can keep receptor cofactors such as GDP in the receptor state:

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

Protein-small-molecule job:

```json
{
  "job_id": "demo_ligand",
  "name": "Demo ligand",
  "complex_pdb": "input/complex.pdb",
  "receptor_chains": "A",
  "ligand_file": "input/ligand.sdf",
  "ligand_resname": "LIG",
  "ligand_charge": 0,
  "ligand_param_mode": "auto",
  "receptor_cofactor_files": "input/gdp.pdb",
  "receptor_cofactor_libs": "input/GDP.prep",
  "receptor_cofactor_frcmods": "input/frcmod.phos",
  "receptor_cofactor_residue_count": 1
}
```

## Run

Run one job:

```bash
mmpbsa peptide run RUN_DIR --job-id demo_peptide --protocol configs/default_15ns.yaml
mmpbsa ligand run RUN_DIR --job-id demo_ligand
```

Run every job under a run directory:

```bash
mmpbsa peptide run RUN_DIR --protocol configs/default_15ns.yaml --resume
mmpbsa ligand run RUN_DIR --resume
```

Stage modes:

```bash
--mode full      # default: prepare + md + analysis + report
--mode prepare   # input cleanup, ligand prep when needed, Amber prep
--mode md        # Amber to GROMACS, EM, NVT, NPT, production
--mode analysis  # trajectory processing, QC, MMPBSA, audit
--mode report    # summarize existing outputs only
```

Done-file policy:

- Default: fail if a selected step already has `.<step>_done`.
- `--resume`: skip selected steps with existing done files.
- `--force`: delete selected mode and downstream done files/outputs, then rerun.

Shared utilities:

```bash
mmpbsa status RUN_DIR
mmpbsa aggregate RUN_DIR --output-dir OUTPUT_DIR
mmpbsa frame-settings --protocol configs/default_15ns.yaml
mmpbsa doctor --protocol configs/default_15ns.yaml
```

## Protocol Defaults

Peptide profiles:

- `configs/default_15ns.yaml`: compatibility default, 15 ns production.
- `configs/peptide_crystal_3x5ns.yaml`: 3 independent 5 ns replicas.
- `configs/peptide_crystal_3x15ns.yaml`: 3 independent 15 ns replicas for AF3/cofold/docked starts when GPU time is available.
- `configs/peptide_crystal_1x15ns.yaml`: single-replica 15 ns template for targeted `repNN` reruns.
- `configs/peptide_crystal_5x5ns.yaml`: optional 5-replica version.

Ligand profiles:

- `configs/ligand_crystal_3x5ns.yaml`: ligand crystal-start default, MMPBSA disabled.
- `configs/ligand_crystal_3x15ns.yaml`: 3 independent 15 ns ligand replicas for GPU-rich crystal/docked starts.
- `configs/ligand_crystal_1x15ns.yaml`: single-replica 15 ns template for targeted `repNN` ligand reruns.
- `configs/ligand_crystal_5x5ns.yaml`: optional 5-replica version.
- `configs/ligand_crystal_3x5ns_mmpbsa_bcc.yaml`: local validation profile with MMPBSA enabled and AM1-BCC fallback charges.
- `configs/ligand_crystal_3x15ns_mmpbsa_bcc.yaml`: longer local validation profile with MMPBSA enabled and AM1-BCC fallback charges.

Peptide MMPBSA uses Amber ff14SB and `mbondi2`. Profiles may explicitly set
`mmpbsa.epsilon` or `mmpbsa.dielectric`; otherwise the pipeline applies the
charged/polar/nonpolar interface rule and records the selected value in the
summary. Crystal-start peptide profiles run MMPBSA separately for each
independent replica and report the replica mean plus sample SD and SEM. Explicit
interface water defaults to 0. Entropy is disabled by default; PB
entropy-corrected output is available only for explicit sensitivity profiles and
is not treated as a validated default score.

Replica indices are stable. `rep01`, `rep02`, and `rep03` use seeds
`seed_base+1`, `seed_base+2`, and `seed_base+3`. A single extra replica can be
run and later merged:

```bash
mmpbsa peptide run RUN_DIR --job-id demo_peptide \
  --protocol configs/peptide_crystal_1x15ns.yaml \
  --replica-index 4 --resume

mmpbsa peptide merge-replicas RUN_DIR/demo_peptide_merged \
  RUN_DIR/demo_peptide_rep01 RUN_DIR/demo_peptide_rep04
```

The ligand workflow supports the same explicit replica controls:

```bash
mmpbsa ligand run RUN_DIR --job-id demo_ligand \
  --protocol configs/ligand_crystal_1x15ns.yaml \
  --replica-index 4 --resume

mmpbsa ligand merge-replicas RUN_DIR/demo_ligand_merged \
  RUN_DIR/demo_ligand_rep01 RUN_DIR/demo_ligand_rep04
```

Peptide input preparation is strict about HETATM records. New jobs keep the
original selected structure as `input/selected_raw.pdb`; the Amber-facing
`input/selected.pdb` and `input/selected_protein.pdb` are ATOM-only cleaned
structures. The default peptide policy is `amber_prep.nonstandard_policy: fail`;
set it to `strip` only when the HETATM records are known removable crystallization
or buffer species. Dropped residues are recorded in the job manifest and
`result/summary.json`.

Common Amber peptide residues and caps such as `ACE`, `NME`, `NHE`, `NH2`,
`CYX`, `CYM`, `ASH`, `GLH`, `HID`, `HIE`, `HIP`, and `LYN` are retained even if
the input PDB marks them as `HETATM`; the cleaned Amber-facing files normalize
those records to `ATOM`.

Peptide crystal profiles keep `solvent_shape: oct` as the first choice and can
automatically retry with `solvent_shape: box` if EM detects a water-box overlap
or non-finite force. The retry decision and actual solvent shape are recorded in
the job manifest and summary.

Ligand production defaults expect RESP-charged ligand input. Use AM1-BCC
fallback profiles only when that approximation is intentional.

## Outputs

Typical outputs are written under each job directory:

```text
input/
ligand/              # small-molecule jobs only
amber/
md/
analysis/
logs/
result/summary.json
result/summary.csv
.<step>_done
```

Local validation scaffolding is kept outside the core package under
`validation/`. It is for checking this repository's peptide/ligand behavior, not
for the public pipeline API. Peptide validation summaries can be regenerated
from existing outputs with:

```bash
python validation/peptide_3x5ns/report.py \
  --run-dir pipeline_tests/peptide_3x5ns \
  --output-dir results/peptide_3x5ns
```

## Documentation

- [Setup guide](docs/setups.md)
- [Receptor cofactor guide](docs/receptor_cofactors.md)
- [Peptide local validation notes](docs/peptide_3x5ns_mmpbsa.md)
- [TYK2 ligand 3x5ns validation report](docs/ligand_tyk2_3x5ns_validation.md)

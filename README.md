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
- `configs/peptide_crystal_5x5ns.yaml`: optional 5-replica version.

Ligand profiles:

- `configs/ligand_crystal_3x5ns.yaml`: ligand crystal-start default, MMPBSA disabled.
- `configs/ligand_crystal_5x5ns.yaml`: optional 5-replica version.
- `configs/ligand_crystal_3x5ns_mmpbsa_bcc.yaml`: benchmark/test profile with MMPBSA enabled and AM1-BCC fallback charges.

Peptide MMPBSA uses Amber ff14SB and `mbondi2`. Profiles may explicitly set
`mmpbsa.epsilon` or `mmpbsa.dielectric`; otherwise the pipeline applies the
charged/polar/nonpolar interface rule and records the selected value in the
summary. Explicit interface water defaults to 0. Entropy is disabled by default;
PB entropy-corrected output is available only for explicit sensitivity profiles
and is not treated as a validated default score.

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

## Documentation

- [Setup guide](docs/setups.md)
- [Receptor cofactor guide](docs/receptor_cofactors.md)
- [Implementation plan](docs/PLAN.md)
- [Peptide local validation notes](docs/peptide_3x5ns_mmpbsa.md)

# Setup Guide

This guide describes how to run the `mmpbsa` package on a new machine after
cloning the GitHub repository.

## Install The Package

```bash
git clone git@github.com:zhaisilong/mmpbsa.git
cd mmpbsa
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
```

## External MD Tools

The Python package is only the pipeline driver. Production runs need an MD
software environment that provides:

- AmberTools commands: `MMPBSA.py`, `MMPBSA.py.MPI`, `cpptraj`, `antechamber`,
  `parmchk2`, `parmed`.
- MPI support for MMPBSA when `mmpbsa.mpi: true`: `mpirun` and `mpi4py`.
- GROMACS with GPU support, usually exposed through `gmx_mpi`.
- ACPYPE for Amber-to-GROMACS conversion.

The default YAML protocols refer to a local mamba environment named `md`.
Create or adapt an environment that provides the tools above, then verify it:

```bash
mamba run -n md which MMPBSA.py
mamba run -n md which MMPBSA.py.MPI
mamba run -n md which mpirun
mamba run -n md python -c "import mpi4py; print(mpi4py.__version__)"
mamba run -n md which cpptraj
mamba run -n md which acpype
mamba run -n md which antechamber
mamba run -n md which parmchk2
mamba run -n md which parmed
```

Verify GROMACS separately:

```bash
source /path/to/gromacs/bin/GMXRC
which gmx_mpi
gmx_mpi --version
```

## Machine-Specific Runtime Settings

The protocol YAML contains runtime defaults. Checked-in configs use `${GMXRC}`
for portability, so export `GMXRC` on each machine or replace it in a local copy:

```yaml
runtime:
  mamba_env: md
  gmxrc: ${GMXRC}
  gmx_bin: gmx_mpi
  gpu_id: 0
```

You can either edit a local copy of the YAML or override the runtime from the
shell:

```bash
MAMBA_ENV=md \
GMXRC=/path/to/gromacs/bin/GMXRC \
GMX_BIN=gmx_mpi \
GPU_ID=0 \
NTOMP=8 \
MMPBSA_NP=16 \
mmpbsa doctor --protocol configs/default_15ns.yaml
```

`GPU_ID`, `NTOMP`, and `MMPBSA_NP` are read at runtime, so the same job config
can be reused across machines.

## Protocols

Core protocols:

- `configs/default_15ns.yaml`: peptide compatibility default, 15 ns production
  with MMPBSA over the final 10 ns.
- `configs/peptide_crystal_3x5ns.yaml`: peptide crystal-start protocol with
  3 independent 5 ns replicas and per-replica stability QC.
- `configs/peptide_crystal_5x5ns.yaml`: optional 5-replica peptide protocol.
- `configs/ligand_crystal_3x5ns.yaml`: ligand crystal-start default, 3
  independent 5 ns replicas, explicit water MD, RESP input expected, MMPBSA
  disabled by default.
- `configs/ligand_crystal_5x5ns.yaml`: optional 5-replica ligand protocol.
- `configs/ligand_crystal_3x5ns_mmpbsa_bcc.yaml`: test profile with MMPBSA
  enabled and AM1-BCC ligand charges.
- `configs/smoke_20ps.yaml`: short smoke settings for environment validation.

Inspect frame selection:

```bash
mmpbsa frame-settings --protocol configs/default_15ns.yaml
mmpbsa frame-settings --protocol configs/peptide_crystal_3x5ns.yaml
```

## Job Directory Template

Protein-peptide:

```text
RUN_DIR/
  demo_peptide/
    demo_peptide.json
    input/
      complex.pdb
```

`demo_peptide.json`:

```json
{
  "job_id": "demo_peptide",
  "name": "Demo peptide",
  "selected_pdb": "input/complex.pdb",
  "receptor_chains": "A",
  "peptide_chains": "B"
}
```

Protein-small-molecule:

```text
RUN_DIR/
  demo_ligand/
    demo_ligand.json
    input/
      complex.pdb
      ligand.sdf
```

`demo_ligand.json`:

```json
{
  "job_id": "demo_ligand",
  "name": "Demo ligand",
  "complex_pdb": "input/complex.pdb",
  "receptor_chains": "A",
  "ligand_file": "input/ligand.sdf",
  "ligand_resname": "LIG",
  "ligand_charge": 0,
  "ligand_param_mode": "auto"
}
```

## Running

Run one smoke job:

```bash
mmpbsa peptide run RUN_DIR --job-id demo_peptide --protocol configs/smoke_20ps.yaml
mmpbsa ligand run RUN_DIR --job-id demo_ligand --protocol configs/smoke_20ps.yaml
```

Resume a job:

```bash
mmpbsa peptide run RUN_DIR --job-id demo_peptide --resume
```

Force rerun MD and downstream state:

```bash
mmpbsa peptide run RUN_DIR --job-id demo_peptide --mode md --force
```

Summarize:

```bash
mmpbsa status RUN_DIR
mmpbsa aggregate RUN_DIR --output-dir OUTPUT_DIR
```

## Validation

Run the unit tests after installation:

```bash
python -m unittest discover -s tests -v
```

For real MD validation, create a small peptide or ligand job and use
`configs/smoke_20ps.yaml` before launching production protocols.

# Reproducible environment

Use a dedicated environment for this repository. Do not install the retired
`mmpbsa-core`, `mmpbsa-backends`, `mmpbsa-report`, or `mmpbsa-utils`
distributions alongside the unified `mmpbsa` package.

The audited reference environment is Python 3.12 with AmberTools 24.8,
ParmEd 4.3.1, gemmi 0.7.5, PyYAML 6.0, Open Babel 3.1.1, MDTraj 1.11,
and OpenMM 8.5.1. GROMACS and CUDA remain host/runtime dependencies and must be declared in `runtime.yaml`;
`mmpbsa doctor` records their observed versions before execution.

```bash
conda env create -f environment.example.yml
conda activate mmpbsa-v2
mmpbsa --workflow workflow.yaml --runtime runtime.yaml doctor
python scripts/release_check.py
```

OpenMM is optional. A GROMACS-only installation may omit it; engine-neutral
trajectory v3 needs `.[trajectory]` (MDTraj plus ParmEd), while OpenMM MD and
trajectory v3 can be installed together with `.[openmm]`.

`scripts/release_check.py` rejects a dirty tree for releases, runs Ruff and
pytest, builds from a temporary copy of Git-tracked files, and compares the
wheel's `mmpbsa/` inventory with `src/mmpbsa/`. This prevents stale
`build/lib` modules from leaking into a release wheel. During development,
`--allow-dirty` also includes non-ignored untracked files, so newly added
modules cannot be silently omitted from the pre-commit wheel audit.

`--allow-dirty` is a development audit, not release approval. Before publishing,
audit a clean source snapshot, install its wheel in an isolated environment and
check provider discovery. Optional OpenMM acceptance gates are described in
[openmm.md](openmm.md); a past workspace does not validate a changed snapshot.

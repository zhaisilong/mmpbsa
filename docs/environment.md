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

## Git publication and local results

Develop and publish from the same `main` repository. Public validation source
structures and literature measurements remain tracked with their provenance and
licences. Local workspaces, prepared structures, trajectories, logs, reports,
scores and host-specific runtime files are not publication inputs; they may
remain in ignored directories of the local checkout. Never use cleanup commands
that remove ignored files as a release preparation step.

Install the local pre-push guard once and inspect outgoing commits before pushing:

```bash
python scripts/publication_audit.py install-hook
python scripts/publication_audit.py check
```

The guard permits only fast-forward pushes to `origin/main`, and installation
refuses to overwrite a different existing hook or custom hooks path. Before
committing and pushing, review the staged paths, contents and outgoing history by
hand. Confirm that no local project data, private paths or credentials are
present, including in an earlier commit whose files were later deleted.
Automated publication checks and Git hooks only screen known patterns; an
unrecognized secret or result may pass, and hooks can be bypassed. They do not
replace manual review or protect files that have already been published.

Preserve existing scientific receipts, file hashes and source fingerprints
unchanged. A historical receipt may refer to a commit that is no longer present
locally; the current source and its tests are not evidence that the old run was
performed with the current version.

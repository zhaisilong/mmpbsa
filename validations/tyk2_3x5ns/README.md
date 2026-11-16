# TYK2 4GIH / lig_ejm_31 validation

This directory is a clean 0.6 validation input. The `data/` tree is the only
content copied from the archived validation; no prepared topology, trajectory,
workspace, result, or report is reused.

`preparation.yaml` declares chemical entities before any MD is run. The MD
system itself remains role-neutral, but the resulting PreparedCase has one
explicit primary binding: `receptor_protein` versus the small-molecule
`ligand`. Its QC selection is ligand heavy atoms. The two source waters,
`HOH A:1318` and `HOH A:1325`, are named MD-only entities and remain in the
explicit-solvent simulation. Every water is excluded from the dry GB/PB
scoring topology.

Inspect the complete non-writing plan from the repository root:

```bash
PYTHONPATH=src python validations/tyk2_3x5ns/prepare.py --dry-run
```

The plan verifies source hashes and inventories, assigns explicit histidine
states, derives the 5 Å receptor interface, records its chemistry classes for
automatic dielectric selection, and emits AmberTools commands from typed
settings. Preparation and workflow execution share the same complete runtime
document; `prepare.py` defaults to `runtime.local.yaml`. To materialize a new
case (the destination must not exist):

```bash
PYTHONPATH=src python validations/tyk2_3x5ns/prepare.py \
  --output validations/tyk2_3x5ns/prepared/lig_ejm_31
```

The validation has two deliberately separate workflows and workspaces:

- `workflow.smoke.yaml` uses `smoke_20ps@1` only as an integration check and
  writes `_runs/smoke/`.
- `workflow.yaml` uses the scientific `production_3x5ns@1` preset and writes
  `_runs/production/`.

Validate and run the smoke workflow first:

```bash
PYTHONPATH=src python -m mmpbsa.core.cli \
  --workflow validations/tyk2_3x5ns/workflow.smoke.yaml \
  --runtime validations/tyk2_3x5ns/runtime.local.yaml \
  --workspace validations/tyk2_3x5ns/_runs/smoke \
  config check

PYTHONPATH=src python -m mmpbsa.core.cli \
  --workflow validations/tyk2_3x5ns/workflow.smoke.yaml \
  --runtime validations/tyk2_3x5ns/runtime.local.yaml \
  --workspace validations/tyk2_3x5ns/_runs/smoke \
  job run --case lig_ejm_31 --replica rep01 --through scoring
```

Only after the smoke passes should the full batch be run with `workflow.yaml`
and `_runs/production`. The production and smoke state must never share a
workspace. The checked-in runner performs batch scoring, case finalization and
project/report finalization in that order:

```bash
MMPBSA_CLI=/path/to/mmpbsa \
  bash validations/tyk2_3x5ns/run_production.sh
```

`runtime.example.yaml` is portable documentation; `runtime.local.yaml` records
this host's executable paths and is not scientific configuration.

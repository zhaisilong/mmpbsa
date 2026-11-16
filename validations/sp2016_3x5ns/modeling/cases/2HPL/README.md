# sp2016_16 / 2HPL

Status: **eligible; validation-local builder and sealing workflow implemented**.

Receptor: auth A / label A (12-111 observed; 12-111 declared).

Peptide: auth B / label B, `DDLYG` observed versus `DDLYG` declared.

## Decision

- The peptide is fully observed and this source may enter explicit case-specific modeling.
- Official biological assembly 1 is author-defined and uses identity operator 1.
  Polymer label asym A/B are retained; water-only label asym C/D are excluded.
- Both complete deposited entities are modeled with free charged termini.  This
  is a validation-owned scientific default, not an implicit LEaP decision:
  receptor A:12/A:111 must map to `NSER`/`CSER`, and peptide
  B:802/B:806 to `NASP`/`CGLY`.  The build also requires terminal
  `H1/H2/H3` and `OXT` topology atoms before admission.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`; the immutable executable decisions
are in `build_plan.json`.  No production topology or MD result is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/2HPL/prepare.py --json
```

The real build requires AmberTools 24 and the project preparation extra in the
active environment. Install the extra with `pip install -e '.[preparation]'`
from the repository root; it supplies ParmEd 4 and gemmi. Return to the
validation directory, inspect the immutable plan, and choose a new create-only
output path:

```bash
python modeling/cases/2HPL/build.py --plan
python modeling/cases/2HPL/build.py \
  --output modeling/cases/2HPL/_integration_build
```

The default target, `_integration_build/`, must not already exist.  The build
uses a create-only lock, stages beside the target, atomically publishes only
after `mmpbsa.core.seal_prepared_case_draft` succeeds, and never starts MD.
It writes dry Amber complex/receptor/ligand artifacts, solvated GROMACS
topology/coordinates, the full-system Amber topology retained canonically as
`solvated_topology` for engine-neutral trajectory projection, standard index
groups, full command/tool/source-hash receipts, the explicit draft, and the
sealed PreparedCase.  The full-system Amber coordinates remain preparation
intermediate data because the production trajectory supplies coordinates.
These generated
integration artifacts (and preserved failed staging directories) are ignored;
only the plan, builder, documentation, and tests belong in version control.
The builder validates the draft and seals it itself.  The published manifest
can be checked independently without rerunning or starting MD:

```bash
mmpbsa prepared-case validate \
  --manifest modeling/cases/2HPL/_integration_build/sealed/prepared_case.json
```

Run the external integration test explicitly:

```bash
RUN_SP2016_2HPL_INTEGRATION=1 \
  python -m pytest \
  tests/test_2hpl_build_contract.py -k integration
```

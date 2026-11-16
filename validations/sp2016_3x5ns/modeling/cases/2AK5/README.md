# sp2016_11 / 2AK5

Status: **eligible for case-specific modeling**.

Receptor: auth A / label A (6-65 observed; 3-66 declared); auth B / label B (3-66 observed; 3-66 declared).

Peptide: auth D / label C, `RPPKPRPR` observed versus `RPPKPRPR` declared.

## Decision

- The peptide is fully observed and this source may enter explicit case-specific modeling.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/2AK5/prepare.py --json
```

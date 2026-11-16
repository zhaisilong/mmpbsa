# sp2016_10 / 1X2R

Status: **eligible for case-specific modeling**.

Receptor: auth A / label A (324-613 observed; 309-624 declared).

Peptide: auth B / label B, `LDEETGEFL` observed versus `LDEETGEFL` declared.

## Decision

- The peptide is fully observed and this source may enter explicit case-specific modeling.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/1X2R/prepare.py --json
```

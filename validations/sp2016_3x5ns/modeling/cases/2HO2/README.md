# sp2016_15 / 2HO2

Status: **eligible for case-specific modeling**.

Receptor: auth A / label A (253-285 observed; 252-289 declared).

Peptide: auth B / label B, `PPPPPPPPPL` observed versus `PPPPPPPPPL` declared.

## Decision

- The peptide is fully observed and this source may enter explicit case-specific modeling.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/2HO2/prepare.py --json
```

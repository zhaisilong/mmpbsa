# sp2016_06 / 1T4F

Status: **eligible for case-specific modeling**.

Receptor: auth M / label A (23-110 observed; 16-125 declared).

Peptide: auth P / label B, `RFMDYWEGL` observed versus `RFMDYWEGL` declared.

## Decision

- The peptide is fully observed and this source may enter explicit case-specific modeling.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/1T4F/prepare.py --json
```

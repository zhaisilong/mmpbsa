# sp2016_17 / 2O9V

Status: **eligible for case-specific modeling**.

Receptor: auth A / label A (818-884 observed; 818-884 declared).

Peptide: auth B / label B, `VPPPVPPPPS` observed versus `VPPPVPPPPS` declared.

## Decision

- The peptide is fully observed and this source may enter explicit case-specific modeling.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/2O9V/prepare.py --json
```

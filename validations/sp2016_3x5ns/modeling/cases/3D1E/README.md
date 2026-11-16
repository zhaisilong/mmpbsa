# sp2016_19 / 3D1E

Status: **scientifically rejected**.

Receptor: auth A / label A (1-366 observed; 1-366 declared).

Peptide: auth P / label C, `GQLGLF` observed versus `TLMTGQLGLF` declared.

## Decision

- `PEPTIDE_RESIDUES_MISSING`: The first four residues THR501-THR504 of the declared decamer are absent.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/3D1E/prepare.py --json
```

# sp2016_04 / 1PZ5

Status: **eligible for case-specific modeling**.

Receptor: auth A / label A (1-27,27A,27B,27C,27D,27E,28-95,97-211 observed; 1-27,27A,27B,27C,27D,27E,28-95,97-211 declared); auth B / label B (1-52,52A,52B,52C,53-82,82A,82B,82C,83-100,100A,101-213 observed; 1-52,52A,52B,52C,53-82,82A,82B,82C,83-100,100A,101-213 declared).

Peptide: auth C / label C, `MDWNMHAA` observed versus `MDWNMHAA` declared.

## Decision

- The peptide is fully observed and this source may enter explicit case-specific modeling.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/1PZ5/prepare.py --json
```

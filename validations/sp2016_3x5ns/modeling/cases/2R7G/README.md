# sp2016_18 / 2R7G

Status: **scientifically rejected**.

Receptor: auth A / label A (380-502,509-578,643-785 observed; 380-581,643-787 declared).

Peptide: auth B / label B, `PPTLHELYDL` observed versus `PPTLHELYDL` declared.

## Decision

- `RECEPTOR_INTERNAL_GAPS`: Receptor A lacks internal residues 503-508 and 579-581; terminal residues 786-787 are also absent.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/2R7G/prepare.py --json
```

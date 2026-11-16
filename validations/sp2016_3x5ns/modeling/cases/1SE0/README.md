# sp2016_05 / 1SE0

Status: **scientifically rejected**.

Receptor: auth A / label A (39-135 observed; 30-145 declared).

Peptide: auth B / label B, `AIAYFIP` observed versus `AIAYFIPDQA` declared.

## Decision

- `PEPTIDE_RESIDUES_MISSING`: Declared ASP8, GLN9, and ALA10 are absent.
- `RECEPTOR_TERMINI_MISSING`: Receptor residues 30-38 and 136-145 are absent.
- `REQUIRED_METAL_MODEL_UNAVAILABLE`: Structural Zn A201 must not be stripped, but no validated case-local metal model exists.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/1SE0/prepare.py --json
```

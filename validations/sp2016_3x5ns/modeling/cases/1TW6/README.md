# sp2016_08 / 1TW6

Status: **scientifically rejected**.

Receptor: auth B / label B (78-172 observed; 40-172 declared).

Peptide: auth D / label D, `AVPIAQ` observed versus `AVPIAQKSE` declared.

## Decision

- `PEPTIDE_RESIDUES_MISSING`: Peptide D lacks declared LYS7-SER8-GLU9.
- `REQUIRED_METAL_MODEL_UNAVAILABLE`: Receptor B contains structural Zn B1001 without a validated case-local model.
- `RECEPTOR_TERMINUS_MISSING`: Receptor B residues 40-77 are absent.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/1TW6/prepare.py --json
```

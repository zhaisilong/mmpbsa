# sp2016_14 / 2FOJ

Status: **scientifically rejected**.

Receptor: auth A / label A (63-105,112-205 observed; 51-205 declared).

Peptide: auth B / label B, `GARAHSS` observed versus `GARAHSS` declared.

## Decision

- `RECEPTOR_INTERNAL_GAP`: Receptor residues 106-111 are absent inside the selected domain; residues 51-62 are also absent at the N terminus.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/2FOJ/prepare.py --json
```

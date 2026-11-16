# sp2016_09 / 1W9E

Status: **scientifically rejected**.

Receptor: auth B / label B (110-273 observed; 108-273 declared).

Peptide: auth S / label D, `NEFYF` observed versus `TNEFYF` declared.

## Decision

- `PEPTIDE_RESIDUES_MISSING`: N-terminal THR1 of declared TNEFYF is absent.
- `RECEPTOR_TERMINUS_MISSING`: Receptor B residues 108-109 are absent.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/1W9E/prepare.py --json
```

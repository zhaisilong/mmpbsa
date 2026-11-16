# sp2016_01 / 1CKA

Status: **scientifically rejected**.

Receptor: auth A / label A (134-190 observed; 134-190 declared).

Peptide: auth B / label B, `PPPALPPKK` observed versus `PPPALPPKKR` declared.

## Decision

- `PEPTIDE_RESIDUES_MISSING`: Declared ARG B10 is absent; the observed chain is a 9-mer, not the benchmark 10-mer.
- `PEPTIDE_TERMINAL_ATOM_MISSING`: Backbone O is absent at LYS B9; TYR A190 O is also absent.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/1CKA/prepare.py --json
```

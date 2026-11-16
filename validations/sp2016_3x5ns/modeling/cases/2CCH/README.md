# sp2016_13 / 2CCH

Status: **scientifically rejected**.

Receptor: auth A / label A (0-296 observed; 0-298 declared); auth B / label B (173-432 observed; 173-432 declared).

Peptide: auth E / label E, `HTLKGRRLVFDN` observed versus `HTLKGRRLVFDN` declared.

## Decision

- `REQUIRED_NUCLEOTIDE_COFACTOR_MODEL_UNAVAILABLE`: ATP A1297 occupies the kinase site and requires validated parameters and altloc treatment.
- `MODIFIED_SUBSTRATE_STATE_UNRESOLVED`: The entry describes a gamma-linked ATP substrate analogue, while current struct_conn has no peptide-ATP covalent link.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/2CCH/prepare.py --json
```

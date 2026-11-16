# sp2016_12 / 2B9H

Status: **scientifically rejected**.

Receptor: auth A / label A (1-163,180-353 observed; 1-353 declared).

Peptide: auth C / label B, `RRNLKGLNLNLH` observed versus `RRNLKGLNLNLHPD` declared.

## Decision

- `PEPTIDE_RESIDUES_MISSING`: Peptide C lacks declared PRO21 and ASP22.
- `RECEPTOR_INTERNAL_GAP`: Receptor A has an unresolved internal segment 164-179.
- `REQUIRED_NUCLEOTIDE_METAL_MODEL_UNAVAILABLE`: ADP A500 and coordinated Mg A700 lack a validated case-local model.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/2B9H/prepare.py --json
```

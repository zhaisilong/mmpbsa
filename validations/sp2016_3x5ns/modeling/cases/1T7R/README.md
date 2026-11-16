# sp2016_07 / 1T7R

Status: **scientifically rejected**.

Receptor: auth A / label A (669-918 observed; 651-919 declared).

Peptide: auth B / label B, `SSRFESLFAG` observed versus `SSRFESLFAGEKESR` declared.

## Decision

- `PEPTIDE_RESIDUES_MISSING`: Declared peptide residues GLU108-ARG112 are absent.
- `REQUIRED_ORGANIC_COFACTOR_MODEL_UNAVAILABLE`: DHT A300 is a functional receptor co-ligand and lacks validated case-local parameters.

Exact source identity, residue ranges, heterogen decisions, altloc audit, and
force-field boundary are in `model.yaml`.  No production topology is claimed.

## Run

Run the following commands from the validation directory
`validations/sp2016_3x5ns` (relative to the repository root).

```bash
python modeling/cases/1T7R/prepare.py --json
```

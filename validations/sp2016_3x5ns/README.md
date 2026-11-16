# SP2016 protein-peptide 3x5 ns validation modeling

This validation-owned layer covers all 19 Spiliotopoulos 2016 complexes.
Molecular-dynamics production is **not started here**: official RCSB sources are
hash-pinned, receptor/peptide/cofactor decisions are explicit, and every case
has an executable fail-closed preflight.

## Scientific admission contract

Exactly nine cases are eligible for case-specific modeling: `1D4T, 1MFG, 1PZ5,
1T4F, 1X2R, 2AK5, 2HO2, 2HPL, 2O9V`.  The other ten are intentionally rejected
for the concrete reasons stored in their model files.  A scientific rejection
is a successful `rejected_expected` preflight outcome, not a software crash.

`eligible` is weaker than `buildable`: eight eligible entries still say
`build.status = not_implemented`. Only `2HPL` has an implemented
validation-local builder and sealing workflow. No generated topology, build
artifact, trajectory or simulation result is included. For example, 1PZ5 still
requires insertion-code-aware selection and four explicit receptor disulfides.
The catalog describes expected validation outcomes, not a record of prior runs.

## Layout and commands

- `catalog/`: machine-readable inventory, outcomes, provenance, and licenses.
- `sources/mmcif/`: byte-identical official coordinate downloads.
- `modeling/common/`: validation-local, gemmi-backed source-evidence preflight.
- `modeling/cases/<PDB>/`: explicit model data, thin executable, and README.
- `tests/`: local tests; global pytest configuration is untouched.

```bash
python -m pip install -e '.[preparation]'
python validations/sp2016_3x5ns/prepare_all.py --output /tmp/sp2016_preflight.json
python validations/sp2016_3x5ns/prepare_all.py --json
python validations/sp2016_3x5ns/modeling/cases/1D4T/prepare.py --json
python -m pytest validations/sp2016_3x5ns/tests
```

`--output` writes the full machine-readable result while stdout remains one
summary line.  Add `--json` only when full JSON on stdout is wanted.

## Source-derived evidence boundary

After byte count and SHA-256 admission, preflight parses the pinned official
mmCIF with gemmi and independently derives every selected receptor/ligand
observed residue selector, including insertion codes.  It also derives the
peptide's deposited entity sequence and atom-observed sequence.  A mismatch
between those facts and `model.yaml` fails closed.

Each result embeds `source_evidence` with biological assembly generators and
operators; selected polymer `_entity.id`, description, and details; non-water
non-polymer residues on selected auth chains; and selected-chain nonstandard
polymer residues as classified by gemmi (so PTMs such as TPO remain visible
independently).  It also records selected-chain altloc/occupancy facts and all
`_struct_conn` rows with the selected subset identified.  Consequently, the
3D1E evidence retains the deposited Rhodamine 6G/TLMT statement even though
the dye is not observed.  This is observation only: it never converts source contents into protonation,
parameterization, inclusion/exclusion, force-field, or eligibility decisions.
Those remain explicit, reviewable case-local choices.

gemmi is therefore a required preflight dependency, supplied by the project's
`.[preparation]` extra.  If it is unavailable, preflight reports an actionable
`source_evidence_derivation` failure instead of silently trusting duplicated
model declarations.

Experimental delta-G and paper MM-PBSA/dMM-PBSA/vdW values are historical
labels, not acceptance values for a new 3-replica x 5 ns ff14SB workflow.

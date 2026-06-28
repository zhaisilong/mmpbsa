# KRAS 5XCO GDP/Mg Pilot

Historical note: this page documents the 5XCO-derived GDP/GDP+Mg peptide pilot.
It is not the charge or receptor-state reference for Boltz poses modeled against
the 6WGN-like KRAS(G12D)-GNP-Mg active state. For those cases, use
`docs/kras_6wgn_boltz.md`.

This validation case study is for KRAS-GDP cyclic peptide cases where raw
AF3/no-Mg MM/PBSA ranking remains dominated by peptide charge.

## Build Inputs

Use 5XCO as the common KRAS-peptide pose and 5US4 as the Mg geometry source:

```bash
python -m validation.kras_5xco.build_pilot \
  /home/silong/projects/homework/kras/mmpbsa_5xco_pilot \
  --template-cif /home/silong/projects/tmp/KRAS_G12D_cyclic_peptide_interface/structures/5xco/5xco.cif \
  --mg-source-cif /home/silong/projects/homework/kras/mmpbsa_5xco_pilot_sources/5US4.cif \
  --gdp-lib /home/silong/projects/homework/kras/mmpbsa/gdp_params/GDP.prep \
  --gdp-frcmod /home/silong/projects/homework/kras/mmpbsa/gdp_params/frcmod.phos \
  --force
```

The default pilot builds six variants: `WT`, `D13A`, `L8A`, `P14A`, `del4R`,
and `core13`. Each variant gets two receptor states:

- `gdp_only`: receptor = KRAS + GDP
- `gdp_mg`: receptor = KRAS + GDP + Mg

The builder writes `pilot_manifest.json`, `kras_5xco_pilot_3x20ns.yaml`, and a
one-GPU run script in the output directory.

## Validation

Before long MD, run prepare and topology partition checks:

```bash
mmpbsa peptide run /home/silong/projects/homework/kras/mmpbsa_5xco_pilot \
  --protocol /home/silong/projects/homework/kras/mmpbsa_5xco_pilot/kras_5xco_pilot_3x20ns.yaml \
  --mode prepare --resume
```

Expected masks are generated from the manifest, not hard-coded:

- GDP-only WT: receptor `:1-171`, peptide `:172-192`
- GDP+Mg WT: receptor `:1-172`, peptide `:173-193`

For Mg branches, receptor topology must contain both `gdp` and `MG`, and peptide
topology must contain neither.

## Run Policy

Run a short WT smoke test first. If EM/NVT/NPT/short production is valid for
both receptor states, run the 20 ns pilot. On a single RTX 3090, use two jobs in
parallel to improve GPU occupancy without oversubscribing the 24 CPU threads too
heavily.

Use raw PB/GB total only as a diagnostic unless the pilot reduces charge
domination. If `|corr(raw_score, peptide_charge)| > 0.7` and experimental
correlation remains poor after 5XCO + GDP+Mg + 20 ns, stop using raw MM/PBSA
total as the KRAS peptide KD ranking score.

## Completed Pilot

The first 5XCO pilot completed on 2026-06-06 03:37 CST:

- 12 jobs completed: six variants times `gdp_only`/`gdp_mg`
- all trajectory QC and MM/PBSA audits were `valid`
- each job used 3 replicas and 1503 MM/PBSA frames
- no failed markers were found

Generate the GDP-only/GDP+Mg comparison report with:

```bash
python -m validation.kras_5xco.report_pilot \
  /home/silong/projects/homework/kras/mmpbsa_5xco_pilot \
  --output-dir /home/silong/projects/homework/kras/correlation \
  --assay-dir /home/silong/projects/tmp/KRAS_G12D_cyclic_peptide_interface \
  --baseline-run-dir /home/silong/projects/homework/kras/mmpbsa
```

The report writes:

- `kras_5xco_mg_pilot_results.csv`
- `kras_5xco_mg_pilot_correlations.csv`
- `kras_5xco_mg_pilot_decision.json`
- `report_kras_5xco_mg_pilot.html`

Initial read-only checks indicate that `GDP+Mg + GB_delta_total_kJ_mol` is the
best current primary score candidate for the six-variant pilot. PB total remains
more charge-sensitive and should be treated as a reference diagnostic unless the
report shows it passes the charge/experiment stop rule.

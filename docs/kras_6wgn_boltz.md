# KRAS 6WGN/GNP-Mg Boltz Scaffold

This validation scaffold is for Boltz-predicted KRAS cyclic ligand poses under
the 6WGN-like KRAS(G12D)-GNP-Mg active receptor state. It must not reuse the
5XCO/GDP charge reference.

## State And Charge Reference

Use a single receptor state:

```text
Complex  = KRAS + GNP + Mg2+ + LIG
Receptor = KRAS + GNP + Mg2+
Ligand   = Boltz LIG1 cyclic ligand
```

The local charge reference is:

```text
LIG charge = 0
GNP charge = -4
Mg charge  = +2
receptor cofactor net charge = -2
```

This matches the existing KRAS AF3/GNP preparation records under
`/data2/silong/projects/homework/kras_cyc_mmpbsa`, where 42 prepared cases have
`GNP` mol2 charge sums near `-4.000018` and neutral cyclic ligand mol2 charge
sums.

## Build Jobs

Create the local job scaffold from the Boltz resources:

```bash
mamba run -n md python -m validation.kras_6wgn_boltz.scaffold make-jobs \
  /data2/silong/projects/homework/kras_cyc_mmpbsa/boltz_6wgn_gnp_mg \
  --resources-dir /data2/silong/projects/resources/boltz_kras \
  --limit 10 \
  --force
```

The builder writes:

- `boltz_6wgn_gnp_mg_manifest.json`
- `boltz_6wgn_gnp_mg_jobs.csv`
- one `RUN_DIR/<job_id>/<job_id>.json` per selected Boltz pose
- prepared `source/ligand.mol2`, `source/ligand.frcmod`, `source/gnp.mol2`,
  `source/gnp.frcmod`, and `source/mg.pdb`

The job config uses the ligand workflow with:

```json
{
  "receptor_chains": "A",
  "ligand_chain": "L",
  "ligand_resname": "LIG",
  "ligand_charge": 0,
  "ligand_param_mode": "preparam",
  "receptor_cofactor_files": "source/gnp.mol2;source/mg.pdb",
  "receptor_cofactor_frcmods": "source/gnp.frcmod",
  "receptor_cofactor_residue_count": 2,
  "nucleotide_state": "GNP",
  "reference_pdb_id": "6WGN"
}
```

## Smoke And Production

Run one smoke job before launching the top-10 set:

```bash
GMXRC=/data2/silong/projects/gromacs/gromacs202602/bin/GMXRC \
GPU_ID=0 \
mmpbsa ligand run /data2/silong/projects/homework/kras_cyc_mmpbsa/boltz_6wgn_gnp_mg \
  --job-id kras6wgn_rank_0987_model_1 \
  --protocol configs/smoke_20ps.yaml \
  --resume
```

The smoke run is only a readiness check. It must not be included in production
ranking. If the smoke check used a top-10 job id, rerun that same job with the
production protocol and `--mode full --force` before generating the final
report:

```bash
GMXRC=/data2/silong/projects/gromacs/gromacs202602/bin/GMXRC \
GPU_ID=0 \
mmpbsa ligand run /data2/silong/projects/homework/kras_cyc_mmpbsa/boltz_6wgn_gnp_mg \
  --job-id kras6wgn_rank_0987_model_1 \
  --protocol configs/ligand_crystal_3x5ns_mmpbsa_bcc.yaml \
  --mode full \
  --force
```

If prepare and smoke pass, run the top-10 3x5 ns validation using the generated
`run_top10_3x5ns.sh` or explicit `mmpbsa ligand run` commands with
`configs/ligand_crystal_3x5ns_mmpbsa_bcc.yaml`. Treat GB as the primary ranking
diagnostic and PB as a secondary check.

## Strict Report

Do not use generic aggregate output as the final KRAS Boltz report, because it
cannot distinguish smoke summaries from production summaries. Generate the
production-only report with the strict reporter:

```bash
mamba run -n md python -m validation.kras_6wgn_boltz.scaffold report-strict \
  /data2/silong/projects/homework/kras_cyc_mmpbsa/boltz_6wgn_gnp_mg
```

The strict reporter fails if any selected job has the wrong protocol, fewer than
three replicas, or anything other than 303 total MMPBSA frames from the 3-5 ns
window. It writes:

- `reports/final_strict_3x5ns_10prod/report.md`
- `reports/final_strict_3x5ns_10prod/ranking_strict_3x5ns_10prod.csv`
- `reports/final_strict_3x5ns_10prod/qc_summary.csv`
- `reports/final_strict_3x5ns_10prod/summary.json`

## Interpretation

This is a pose rescoring validation scaffold, not a new public benchmark. The
reported MM/GBSA and MM/PBSA values should be compared against Boltz confidence,
pose stability, and any downstream assay data when available. Do not interpret
the old 5XCO/GDP pilot correlations as evidence for this active-state GNP/Mg
Boltz set.

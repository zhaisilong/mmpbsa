from __future__ import annotations

import os
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .peptide_amber import prepare_input_structure, run_amber_prepare
from .analysis import (
    audit_mmpbsa,
    column_stats,
    evaluate_trajectory_qc,
    load_cpptraj_table,
    parse_mmpbsa_full,
    write_trajectory_qc_csv,
)
from .common import (
    aggregate_replica_values,
    explicit_water_count,
    flatten_atom_range,
    frame_settings,
    job_id,
    job_name,
    label_chains,
    mamba_command,
    mmpbsa_enabled,
    model_id,
    mpi_pythonpath,
    parse_simple_mask,
    pdb_id,
    peptide_chains,
    read_json,
    remove_paths,
    replica_count,
    replica_indices,
    replica_names,
    replica_seed_map,
    residue_atoms,
    run_logged,
    shlex_quote,
    split_chain_spec,
    split_path_list,
    utc_now,
    write_csv_atomic,
    write_index,
    write_json_atomic,
    write_text_atomic,
)
from .md import EmUnstableError, convert_to_gromacs, em_failure_atom_index, em_log_has_unstable_structure, find_gro_atom_overlaps, run_em, run_npt, run_nvt, run_production
from .runner import DoneFileRunner, JobContext


STEPS = [
    "init",
    "prepare_input",
    "amber_prepare",
    "md_convert",
    "md_em",
    "md_nvt",
    "md_npt",
    "md_production",
    "analysis_prepare",
    "analysis_qc",
    "analysis_mmpbsa",
    "analysis_audit",
    "report",
]

MODE_STEPS = {
    "full": STEPS,
    "prepare": ["init", "prepare_input", "amber_prepare"],
    "md": ["md_convert", "md_em", "md_nvt", "md_npt", "md_production"],
    "analysis": ["analysis_prepare", "analysis_qc", "analysis_mmpbsa", "analysis_audit"],
    "report": ["report"],
}


@dataclass(frozen=True)
class JobPaths:
    root: Path
    input: Path
    amber: Path
    md: Path
    gromacs: Path
    rep: Path
    analysis: Path
    mmpbsa: Path
    qc: Path
    structures: Path
    logs: Path
    result: Path
    manifest: Path

    @classmethod
    def from_root(cls, root: Path) -> "JobPaths":
        return cls(
            root=root,
            input=root / "input",
            amber=root / "amber",
            md=root / "md",
            gromacs=root / "md" / "gromacs",
            rep=root / "md" / "rep01",
            analysis=root / "analysis",
            mmpbsa=root / "analysis" / "mmpbsa",
            qc=root / "analysis" / "qc",
            structures=root / "analysis" / "structures",
            logs=root / "logs",
            result=root / "result",
            manifest=root / "manifest.json",
        )

    def with_rep(self, rep: Path) -> "JobPaths":
        return replace(self, rep=rep)


class PeptidePipeline(DoneFileRunner):
    STEPS = STEPS
    MODE_STEPS = MODE_STEPS

    def __init__(self, context: JobContext) -> None:
        super().__init__(context)
        self.paths = JobPaths.from_root(context.job_dir)

    def ensure_dirs(self) -> None:
        for directory in [
            self.paths.root,
            self.paths.input,
            self.paths.amber,
            self.paths.md,
            self.paths.gromacs,
            self.paths.rep,
            self.paths.analysis,
            self.paths.mmpbsa,
            self.paths.qc,
            self.paths.structures,
            self.paths.logs,
            self.paths.result,
            *self.replica_dirs(),
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def replica_dirs(self) -> list[Path]:
        return [self.paths.md / name for name in replica_names(self.profile)]

    def replica_paths(self) -> list[JobPaths]:
        return [self.paths.with_rep(rep) for rep in self.replica_dirs()]

    def replica_outputs(self, *names: str) -> list[Path]:
        return [rep / name for rep in self.replica_dirs() for name in names]

    def required_outputs(self, step: str) -> list[Path]:
        p = self.paths
        analysis_prepare = [
            p.mmpbsa / "complex.prmtop",
            p.mmpbsa / "receptor.prmtop",
            p.mmpbsa / "peptide.prmtop",
            p.mmpbsa / "md_prod_dry_center.nc",
            p.mmpbsa / "mmpbsa_manifest.json",
        ]
        if mmpbsa_enabled(self.profile):
            for name in replica_names(self.profile):
                rep = p.mmpbsa / name
                analysis_prepare.extend(
                    [
                        rep / "complex.prmtop",
                        rep / "receptor.prmtop",
                        rep / "peptide.prmtop",
                        rep / "md_prod_dry_center.nc",
                        rep / "mmpbsa.in",
                    ]
                )
        mmpbsa_outputs = [p.mmpbsa / ".mmpbsa_skipped"] if not mmpbsa_enabled(self.profile) else [p.mmpbsa / "mmpbsa_replicas.json"]
        outputs: dict[str, list[Path]] = {
            "init": [p.manifest, p.input / "selected_raw.pdb"],
            "prepare_input": [
                p.input / "selected.pdb",
                p.input / "selected_protein.pdb",
                p.input / "selected_receptor.pdb",
                p.input / "selected_peptide.pdb",
                p.manifest,
            ],
            "amber_prepare": [
                p.amber / "complex_dry.pdb",
                p.amber / "complex_dry.prmtop",
                p.amber / "system_solvated.prmtop",
                p.amber / "system_solvated.inpcrd",
                p.amber / "summary.json",
            ],
            "md_convert": self.replica_outputs("system_GMX.gro", "system_GMX.top", "em.mdp"),
            "md_em": self.replica_outputs("em.gro", "em.tpr"),
            "md_nvt": self.replica_outputs("nvt.gro", "nvt.tpr", "nvt.cpt"),
            "md_npt": self.replica_outputs("npt.gro", "npt.tpr", "npt.cpt"),
            "md_production": self.replica_outputs("md_prod.gro", "md_prod.tpr", "md_prod.xtc"),
            "analysis_prepare": analysis_prepare,
            "analysis_qc": [
                p.qc / "trajectory_qc.csv",
                p.qc / "summary.json",
                p.structures / "first.pdb",
                p.structures / "mid.pdb",
                p.structures / "last.pdb",
                p.structures / "pymol_trajectory.pdb",
            ],
            "analysis_mmpbsa": mmpbsa_outputs,
            "analysis_audit": [p.mmpbsa / "audit.json"],
            "report": [p.result / "summary.json", p.result / "summary.csv"],
        }
        return outputs[step]

    def cleanup_for_step(self, step: str) -> None:
        mapping = {
            "init": [self.paths.input, self.paths.manifest],
            "prepare_input": [
                self.paths.input / "selected.pdb",
                self.paths.input / "selected_protein.pdb",
                self.paths.input / "selected_receptor.pdb",
                self.paths.input / "selected_peptide.pdb",
            ],
            "amber_prepare": [self.paths.amber],
            "md_convert": [self.paths.gromacs, *self.replica_dirs()],
            "analysis_prepare": [self.paths.mmpbsa],
            "analysis_qc": [self.paths.qc, self.paths.structures],
            "analysis_mmpbsa": [
                self.paths.mmpbsa / ".mmpbsa_skipped",
                self.paths.mmpbsa / "mmpbsa_replicas.json",
                self.paths.mmpbsa / "FINAL_RESULTS_MMPBSA.dat",
                self.paths.mmpbsa / "FINAL_RESULTS_MMPBSA_SANITY.dat",
                self.paths.mmpbsa / "per_frame_energy.csv",
                *self.paths.mmpbsa.glob("_MMPBSA_*"),
                *self.paths.mmpbsa.glob("rep*/_MMPBSA_*"),
            ],
            "report": [self.paths.result],
        }
        remove_paths(mapping.get(step, []))

    def manifest(self) -> dict[str, Any]:
        return read_json(self.paths.manifest)

    def write_manifest(self, data: dict[str, Any]) -> None:
        write_json_atomic(self.paths.manifest, data)

    def step_init(self) -> None:
        row = dict(self.config)
        for key in ("selected_pdb", "receptor_chains", "peptide_chains"):
            if not str(row.get(key) or "").strip():
                raise SystemExit(f"{self.context.config_path} is missing required field {key}")
        source_pdb = self.context.resolve_path(row["selected_pdb"])
        if not source_pdb.exists():
            raise SystemExit(f"Missing selected PDB: {source_pdb}")
        shutil.copy2(source_pdb, self.paths.input / "selected_raw.pdb")
        cofactor_files = self.copy_optional_inputs(row.get("receptor_cofactor_files"), "cofactors")
        cofactor_frcmods = self.copy_optional_inputs(row.get("receptor_cofactor_frcmods"), "cofactor_params")
        cofactor_libs = self.copy_optional_inputs(row.get("receptor_cofactor_libs"), "cofactor_params")
        cofactor_count_text = str(row.get("receptor_cofactor_count") or row.get("receptor_cofactor_residue_count") or "").strip()
        receptor_cofactor_count = len(cofactor_files) if cofactor_count_text == "" else int(cofactor_count_text)
        settings = frame_settings(self.profile)
        receptor_chains = row["receptor_chains"]
        ligand_chains = peptide_chains(row)
        manifest = {
            "schema_version": "mmpbsa.peptide.job.v1",
            "job_id": job_id(row),
            "name": job_name(row),
            "pdb_id": pdb_id(row),
            "model_id": model_id(row),
            "source": row.get("source", ""),
            "job_config": str(self.context.config_path),
            "protocol_path": str(self.context.protocol_path),
            "profile": self.profile,
            "source_pdb": str(source_pdb),
            "raw_input_pdb": str(self.paths.input / "selected_raw.pdb"),
            "input_pdb": str(self.paths.input / "selected.pdb"),
            "clean_input_pdb": str(self.paths.input / "selected.pdb"),
            "receptor_chains": receptor_chains,
            "peptide_chains": ligand_chains,
            "receptor_cofactor_files": cofactor_files,
            "receptor_cofactor_frcmods": cofactor_frcmods,
            "receptor_cofactor_libs": cofactor_libs,
            "receptor_cofactor_residue_count": receptor_cofactor_count,
            "experimental_deltaG_kJ_mol": optional_float(row.get("deltaG_exp_kJ_mol")),
            "paper_mm_pbsa_kJ_mol": optional_float(row.get("paper_mm_pbsa_kJ_mol")),
            "paper_dmm_pbsa_kJ_mol": optional_float(row.get("paper_dmm_pbsa_kJ_mol")),
            "paper_vdw_kJ_mol": optional_float(row.get("paper_vdw_kJ_mol")),
            "frame_settings": settings,
            "replica_indices": replica_indices(self.profile),
            "replicas": replica_names(self.profile),
            "replica_seeds": replica_seed_map(self.profile),
            "solvent_shape_initial": self.profile["system"].get("solvent_shape", "oct"),
            "solvent_shape_actual": self.profile["system"].get("solvent_shape", "oct"),
            "box_retry_used": False,
            "created_at": utc_now(),
            "runtime_overrides": {
                "GPU_ID": os.environ.get("GPU_ID"),
                "NTOMP": os.environ.get("NTOMP"),
                "MMPBSA_NP": os.environ.get("MMPBSA_NP"),
            },
        }
        self.write_manifest(manifest)

    def copy_optional_inputs(self, value: str | None, dirname: str) -> list[str]:
        copied: list[str] = []
        items = split_path_list(value)
        if not items:
            return copied
        target_dir = self.paths.input / dirname
        target_dir.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(items, start=1):
            source = self.context.resolve_path(item)
            if not source.exists():
                raise SystemExit(f"Missing optional input {item!r}: {source}")
            target = target_dir / f"{idx:02d}_{source.name}"
            shutil.copy2(source, target)
            copied.append(str(target))
        return copied

    def step_prepare_input(self) -> None:
        manifest = prepare_input_structure(self.paths, self.manifest(), self.profile)
        manifest["dielectric_policy"] = peptide_dielectric_policy(self.paths.input / "selected_protein.pdb", manifest, self.profile)
        self.write_manifest(manifest)

    def step_amber_prepare(self) -> None:
        remove_paths([self.paths.amber])
        self.paths.amber.mkdir(parents=True, exist_ok=True)
        run_amber_prepare(self.paths, self.profile)

    def step_md_convert(self) -> None:
        remove_paths([self.paths.gromacs, *self.replica_dirs()])
        self.paths.gromacs.mkdir(parents=True, exist_ok=True)
        convert_to_gromacs(self.paths, self.profile)

    def step_md_em(self) -> None:
        try:
            self.run_all_em()
        except EmUnstableError as exc:
            if not self.box_retry_allowed():
                raise
            self.retry_md_em_with_box(self.box_retry_details(exc))

    def run_all_em(self) -> None:
        for paths in self.replica_paths():
            run_em(paths, self.profile)

    def box_retry_allowed(self) -> bool:
        system = self.profile.get("system", {})
        if not bool(system.get("allow_box_retry", False)):
            return False
        if str(system.get("solvent_shape", "oct")).lower() == "box":
            return False
        return not bool(self.manifest().get("box_retry_used", False))

    def box_retry_details(self, exc: Exception) -> dict[str, Any]:
        details: dict[str, Any] = {
            "reason": str(exc),
            "detected_at": utc_now(),
            "failed_logs": [],
        }
        for log in sorted(self.paths.logs.glob("*mdrun_em.log")):
            if not em_log_has_unstable_structure(log):
                continue
            rep_name = log.name.split("_", 1)[0] if "_" in log.name else self.paths.rep.name
            atom_index = em_failure_atom_index(log)
            overlaps: list[dict[str, Any]] = []
            gro = self.paths.md / rep_name / "system_GMX.gro"
            if atom_index is not None and gro.exists():
                overlaps = find_gro_atom_overlaps(gro, atom_index)
            details["failed_logs"].append(
                {
                    "replica": rep_name,
                    "log": str(log),
                    "max_force_atom_index": atom_index,
                    "nearby_overlaps": overlaps,
                }
            )
        return details

    def retry_md_em_with_box(self, details: dict[str, Any]) -> None:
        initial_shape = str(self.profile.get("system", {}).get("solvent_shape", "oct")).lower()
        retry_profile = deepcopy(self.profile)
        retry_profile.setdefault("system", {})
        retry_profile["system"]["solvent_shape"] = "box"
        retry_profile["system"]["allow_box_retry"] = False

        manifest = self.manifest()
        manifest.setdefault("solvent_shape_initial", initial_shape)
        manifest["solvent_shape_actual"] = "box"
        manifest["box_retry_used"] = True
        manifest["box_retry_reason"] = details["reason"]
        manifest["box_retry_details"] = details
        manifest["profile"] = retry_profile
        self.profile = retry_profile
        self.write_manifest(manifest)
        write_json_atomic(self.paths.logs / "box_retry.json", details)
        write_text_atomic(self.paths.logs / "box_retry.log", f"{utc_now()}\nretry solvent_shape: {initial_shape} -> box\nreason: {details['reason']}\n")

        remove_paths([self.paths.amber, self.paths.gromacs, *self.replica_dirs()])
        self.ensure_dirs()
        run_amber_prepare(self.paths, self.profile)
        write_text_atomic(self.done_file("amber_prepare"), f"{utc_now()}\nbox_retry_rebuild=true\n")
        convert_to_gromacs(self.paths, self.profile)
        write_text_atomic(self.done_file("md_convert"), f"{utc_now()}\nbox_retry_rebuild=true\n")
        self.run_all_em()

    def step_md_nvt(self) -> None:
        for paths in self.replica_paths():
            run_nvt(paths, self.profile)

    def step_md_npt(self) -> None:
        for paths in self.replica_paths():
            run_npt(paths, self.profile)

    def step_md_production(self) -> None:
        for paths in self.replica_paths():
            run_production(paths, self.profile)

    def step_analysis_prepare(self) -> None:
        remove_paths(
            [
                self.paths.mmpbsa / "md_prod_dry_center.xtc",
                self.paths.mmpbsa / "md_prod_dry_center.nc",
                self.paths.mmpbsa / "mmpbsa_replicas.json",
                self.paths.mmpbsa / ".mmpbsa_skipped",
                self.paths.mmpbsa / "FINAL_RESULTS_MMPBSA.dat",
                self.paths.mmpbsa / "FINAL_RESULTS_MMPBSA_SANITY.dat",
                self.paths.mmpbsa / "per_frame_energy.csv",
                *[self.paths.mmpbsa / name for name in replica_names(self.profile)],
            ]
        )
        manifest = self.manifest()
        self.write_layout_and_mmpbsa_inputs(manifest)
        ligand_mask = manifest["peptide_residue_mask"]
        remove_paths([self.paths.mmpbsa / "complex.prmtop", self.paths.mmpbsa / "receptor.prmtop", self.paths.mmpbsa / "peptide.prmtop"])
        run_logged(
            mamba_command(
                self.profile,
                [
                    "ante-MMPBSA.py",
                    "-p",
                    str(self.paths.amber / "complex_dry.prmtop"),
                    "-c",
                    "complex.prmtop",
                    "-r",
                    "receptor.prmtop",
                    "-l",
                    "peptide.prmtop",
                    "-n",
                    str(ligand_mask),
                    "--radii=mbondi2",
                ],
            ),
            self.paths.logs / "ante_mmpbsa.log",
            cwd=self.paths.mmpbsa,
        )
        shutil.copy2(self.paths.amber / "complex_dry.prmtop", self.paths.mmpbsa / "complex.prmtop")

        if explicit_water_count(self.profile) > 0:
            raise SystemExit("peptide explicit-water MMPBSA sensitivity is not implemented yet; keep mmpbsa.explicit_water_count=0 for this profile.")
        for rep_dir in self.replica_dirs():
            self.run_trjconv(
                rep_dir,
                self.paths.qc / "protein_peptide.ndx",
                "Complex",
                self.paths.mmpbsa / f"{rep_dir.name}_dry_center.xtc",
                self.paths.logs / f"trjconv_{rep_dir.name}_dry_center.log",
            )
        run_logged(mamba_command(self.profile, ["cpptraj", "-i", str(self.paths.mmpbsa / "convert_dry_xtc_to_nc.in")]), self.paths.logs / "cpptraj_dry_convert.log")
        mmpbsa_manifest = {
            "enabled": mmpbsa_enabled(self.profile),
            "replica_count": replica_count(self.profile),
            "replica_indices": replica_indices(self.profile),
            "replicas": replica_names(self.profile),
            "replica_seeds": replica_seed_map(self.profile),
            "explicit_water_count": explicit_water_count(self.profile),
        }
        if mmpbsa_enabled(self.profile):
            self.prepare_replica_mmpbsa_inputs(manifest)
        write_json_atomic(self.paths.mmpbsa / "mmpbsa_manifest.json", mmpbsa_manifest)

    def write_layout_and_mmpbsa_inputs(self, manifest: dict[str, Any]) -> None:
        atoms_by_residue = residue_atoms(self.paths.amber / "complex_dry.pdb")
        receptor_first, receptor_last = parse_simple_mask(str(manifest["receptor_residue_mask"]))
        peptide_first, peptide_last = parse_simple_mask(str(manifest["peptide_residue_mask"]))
        receptor_atoms = flatten_atom_range(atoms_by_residue, receptor_first, receptor_last)
        peptide_atoms = flatten_atom_range(atoms_by_residue, peptide_first, peptide_last)
        complex_atoms = receptor_atoms + peptide_atoms
        layout = {
            "receptor_atom_range": [min(receptor_atoms), max(receptor_atoms)],
            "peptide_atom_range": [min(peptide_atoms), max(peptide_atoms)],
            "complex_atom_count": len(complex_atoms),
            "receptor_atom_count": len(receptor_atoms),
            "peptide_atom_count": len(peptide_atoms),
            "mmpbsa_trajectory_preselected": replica_count(self.profile) > 1,
        }
        manifest.update(layout)
        self.write_manifest(manifest)
        write_index(self.paths.qc / "protein_peptide.ndx", {"Complex": complex_atoms, "Receptor": receptor_atoms, "Peptide": peptide_atoms})
        fixed_mask = f":{receptor_first}-{peptide_last}"
        if bool(manifest["mmpbsa_trajectory_preselected"]):
            settings = manifest["frame_settings"]
            trajin_lines = "\n".join(
                f"trajin {self.paths.mmpbsa / f'{name}_dry_center.xtc'} {settings['startframe']} {settings['total_frames']} {settings['interval']}"
                for name in replica_names(self.profile)
            )
        else:
            trajin_lines = f"trajin {self.paths.mmpbsa / f'{replica_names(self.profile)[0]}_dry_center.xtc'}"
        write_text_atomic(
            self.paths.mmpbsa / "convert_dry_xtc_to_nc.in",
            f"""parm {self.paths.mmpbsa / "complex.prmtop"}
{trajin_lines}
autoimage anchor {manifest["receptor_residue_mask"]} fixed {fixed_mask}
trajout {self.paths.mmpbsa / "md_prod_dry_center.nc"} netcdf
run
""",
        )
    def prepare_replica_mmpbsa_inputs(self, manifest: dict[str, Any]) -> None:
        receptor_first, _receptor_last = parse_simple_mask(str(manifest["receptor_residue_mask"]))
        _peptide_first, peptide_last = parse_simple_mask(str(manifest["peptide_residue_mask"]))
        settings = manifest["frame_settings"]
        for rep_dir in self.replica_dirs():
            out = self.paths.mmpbsa / rep_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.paths.mmpbsa / "complex.prmtop", out / "complex.prmtop")
            shutil.copy2(self.paths.mmpbsa / "receptor.prmtop", out / "receptor.prmtop")
            shutil.copy2(self.paths.mmpbsa / "peptide.prmtop", out / "peptide.prmtop")
            input_xtc = self.paths.mmpbsa / f"{rep_dir.name}_dry_center.xtc"
            write_text_atomic(
                out / "convert_dry_xtc_to_nc.in",
                f"""parm {out / "complex.prmtop"}
trajin {input_xtc} {settings["startframe"]} {settings["total_frames"]} {settings["interval"]}
autoimage anchor {manifest["receptor_residue_mask"]} fixed :{receptor_first}-{peptide_last}
trajout {out / "md_prod_dry_center.nc"} netcdf
run
""",
            )
            run_logged(mamba_command(self.profile, ["cpptraj", "-i", str(out / "convert_dry_xtc_to_nc.in")]), self.paths.logs / f"cpptraj_dry_convert_{rep_dir.name}.log")
            write_text_atomic(out / "mmpbsa_sanity.in", mmpbsa_input_text(manifest, self.profile, sanity=True))
            write_text_atomic(out / "mmpbsa.in", mmpbsa_input_text(manifest, self.profile, sanity=False))

    def run_trjconv(self, rep_dir: Path, index: Path, group: str, output: Path, log: Path) -> None:
        trjconv_script = (
            "set -eo pipefail; "
            "set +u; "
            f"source {shlex_quote(str(self.profile['runtime']['gmxrc']))}; "
            "set -u; "
            f"printf '{group}\\n{group}\\n' | {shlex_quote(str(self.profile['runtime']['gmx_bin']))} trjconv "
            f"-s {shlex_quote(str(rep_dir / 'md_prod.tpr'))} "
            f"-f {shlex_quote(str(rep_dir / 'md_prod.xtc'))} "
            f"-n {shlex_quote(str(index))} "
            f"-o {shlex_quote(str(output))} "
            "-pbc mol -center -ur compact"
        )
        run_logged(f"bash -lc {shlex_quote(trjconv_script)}", log)

    def step_analysis_qc(self) -> None:
        manifest = self.manifest()
        receptor_mask = manifest["receptor_residue_mask"]
        peptide_mask = manifest["peptide_residue_mask"]
        write_text_atomic(
            self.paths.qc / "trajectory_qc.in",
            f"""parm {self.paths.mmpbsa / "complex.prmtop"}
trajin {self.paths.mmpbsa / "md_prod_dry_center.nc"}
reference {self.paths.mmpbsa / "md_prod_dry_center.nc"} 1
rms receptor_bb {receptor_mask}@N,CA,C reference out {self.paths.qc / "receptor_bb_rmsd.dat"}
rms peptide_bb {peptide_mask}@N,CA,C reference nofit out {self.paths.qc / "peptide_bb_rmsd_after_receptor_fit.dat"}
nativecontacts name rec_pep {receptor_mask} {peptide_mask} distance 4.5 reference out {self.paths.qc / "native_contacts.dat"} mindist maxdist
run
""",
        )
        run_logged(mamba_command(self.profile, ["cpptraj", "-i", str(self.paths.qc / "trajectory_qc.in")]), self.paths.logs / "trajectory_qc.log")
        receptor_header, receptor_rows = load_cpptraj_table(self.paths.qc / "receptor_bb_rmsd.dat")
        peptide_header, peptide_rows = load_cpptraj_table(self.paths.qc / "peptide_bb_rmsd_after_receptor_fit.dat")
        contact_header, contact_rows = load_cpptraj_table(self.paths.qc / "native_contacts.dat")
        write_trajectory_qc_csv(
            self.paths.qc / "trajectory_qc.csv",
            receptor_rows,
            peptide_rows,
            contact_header,
            contact_rows,
            partner_field="peptide_bb_rmsd_after_receptor_fit_angstrom",
        )
        receptor_stats = column_stats(receptor_header, receptor_rows)["receptor_bb"]
        peptide_stats = column_stats(peptide_header, peptide_rows)["peptide_bb"]
        contact_stats = column_stats(contact_header, contact_rows)
        frames = len(receptor_rows)
        summary: dict[str, Any] = {
            "job_id": manifest["job_id"],
            "name": manifest["name"],
            "pdb_id": manifest["pdb_id"],
            "model_id": manifest.get("model_id", ""),
            "frames": frames,
            "receptor_bb_rmsd_angstrom": receptor_stats,
            "peptide_bb_rmsd_after_receptor_fit_angstrom": peptide_stats,
            "native_contacts": contact_stats,
        }
        issues = evaluate_trajectory_qc(
            summary,
            self.profile["qc"],
            partner_field="peptide_bb_rmsd_after_receptor_fit_angstrom",
            partner_threshold_key="peptide_rmsd_warn_angstrom",
            partner_label="peptide",
            contact_prefix="rec_pep",
        )
        summary["replica_qc"] = peptide_replica_qc_summaries(
            manifest,
            receptor_header,
            receptor_rows,
            peptide_header,
            peptide_rows,
            contact_header,
            contact_rows,
            self.profile,
        )
        for replica in summary["replica_qc"]:
            for issue in replica["issues"]:
                issues.append({**issue, "replica": replica["replica"]})
        summary["issues"] = issues
        summary["status"] = "invalid" if any(issue["severity"] == "fail" for issue in issues) else "valid"
        write_json_atomic(self.paths.qc / "summary.json", summary)
        self.export_structures(frames, int(parse_simple_mask(str(manifest["receptor_residue_mask"]))[1]))

    def export_structures(self, frames: int, receptor_last: int) -> None:
        mid = max(1, frames // 2)
        write_text_atomic(
            self.paths.qc / "extract_structures.in",
            f"""parm {self.paths.mmpbsa / "complex.prmtop"}
trajin {self.paths.mmpbsa / "md_prod_dry_center.nc"} 1 1
trajout {self.paths.structures / "first.raw.pdb"} pdb
run
clear trajin
trajin {self.paths.mmpbsa / "md_prod_dry_center.nc"} {mid} {mid}
trajout {self.paths.structures / "mid.raw.pdb"} pdb
run
clear trajin
trajin {self.paths.mmpbsa / "md_prod_dry_center.nc"} {frames} {frames}
trajout {self.paths.structures / "last.raw.pdb"} pdb
run
""",
        )
        run_logged(mamba_command(self.profile, ["cpptraj", "-i", str(self.paths.qc / "extract_structures.in")]), self.paths.logs / "extract_structures.log")
        for stem in ("first", "mid", "last"):
            label_chains(self.paths.structures / f"{stem}.raw.pdb", self.paths.structures / f"{stem}.pdb", receptor_last)
            (self.paths.structures / f"{stem}.raw.pdb").unlink(missing_ok=True)
        self.export_pymol_trajectory(frames, receptor_last)

    def export_pymol_trajectory(self, frames: int, receptor_last: int) -> None:
        stride = max(1, int(self.profile.get("export", {}).get("pymol_stride", 25)))
        temp_dir = self.paths.analysis / "tmp" / "pymol_export"
        remove_paths([temp_dir])
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_text_atomic(
                temp_dir / "export_pymol_pdb.in",
                f"""parm {self.paths.mmpbsa / "complex.prmtop"}
trajin {self.paths.mmpbsa / "md_prod_dry_center.nc"} 1 {frames} {stride}
trajout frame pdb multi
run
""",
            )
            run_logged(mamba_command(self.profile, ["cpptraj", "-i", str(temp_dir / "export_pymol_pdb.in")]), self.paths.logs / "export_pymol_pdb.log", cwd=temp_dir)
            frame_paths = sorted(temp_dir.glob("frame.*"), key=lambda path: int(path.suffix.lstrip(".")))
            if not frame_paths:
                raise RuntimeError("cpptraj did not write frame.* files for PyMOL export")
            with (self.paths.structures / ".pymol_trajectory.pdb.tmp").open("w", encoding="utf-8") as handle:
                handle.write("REMARK chain A = receptor\n")
                handle.write("REMARK chain B = peptide\n")
                for idx, frame_path in enumerate(frame_paths, start=1):
                    handle.write(f"MODEL     {idx:4d}\n")
                    for raw in frame_path.read_text(encoding="utf-8", errors="replace").splitlines():
                        if raw.startswith("END"):
                            continue
                        handle.write(label_line(raw, receptor_last) + "\n")
                    handle.write("ENDMDL\n")
                handle.write("END\n")
            (self.paths.structures / ".pymol_trajectory.pdb.tmp").replace(self.paths.structures / "pymol_trajectory.pdb")
            write_text_atomic(
                self.paths.structures / "load_pymol.pml",
                f"""load {self.paths.structures / "pymol_trajectory.pdb"}, complex
hide everything
show cartoon, chain A
show sticks, chain B
color gray70, chain A
color tv_red, chain B
""",
            )
        finally:
            if not bool(self.profile.get("debug", {}).get("keep_step_tmp", False)):
                remove_paths([temp_dir])

    def step_analysis_mmpbsa(self) -> None:
        if not mmpbsa_enabled(self.profile):
            write_text_atomic(self.paths.mmpbsa / ".mmpbsa_skipped", "mmpbsa.enabled=false\n")
            return

        replicas: list[dict[str, Any]] = []
        seeds = replica_seed_map(self.profile)
        for name in replica_names(self.profile):
            rep_dir = self.paths.mmpbsa / name
            sanity_cmd = [
                "MMPBSA.py",
                "-O",
                "-i",
                "mmpbsa_sanity.in",
                "-o",
                "FINAL_RESULTS_MMPBSA_SANITY.dat",
                "-cp",
                "complex.prmtop",
                "-rp",
                "receptor.prmtop",
                "-lp",
                "peptide.prmtop",
                "-y",
                "md_prod_dry_center.nc",
            ]
            run_logged(mamba_command(self.profile, sanity_cmd), self.paths.logs / f"mmpbsa_sanity_{name}.log", cwd=rep_dir)

            if bool(self.profile["mmpbsa"]["mpi"]):
                cmd = [
                    "mpirun",
                    "-np",
                    str(int(self.profile["mmpbsa"]["np"])),
                    "MMPBSA.py.MPI",
                    "-O",
                    "-i",
                    "mmpbsa.in",
                    "-o",
                    "FINAL_RESULTS_MMPBSA.dat",
                    "-eo",
                    "per_frame_energy.csv",
                    "-cp",
                    "complex.prmtop",
                    "-rp",
                    "receptor.prmtop",
                    "-lp",
                    "peptide.prmtop",
                    "-y",
                    "md_prod_dry_center.nc",
                ]
            else:
                cmd = [
                    "MMPBSA.py",
                    "-O",
                    "-i",
                    "mmpbsa.in",
                    "-o",
                    "FINAL_RESULTS_MMPBSA.dat",
                    "-eo",
                    "per_frame_energy.csv",
                    "-cp",
                    "complex.prmtop",
                    "-rp",
                    "receptor.prmtop",
                    "-lp",
                    "peptide.prmtop",
                    "-y",
                    "md_prod_dry_center.nc",
                ]
            env = {"PYTHONPATH": mpi_pythonpath(self.profile)}
            run_logged(mamba_command(self.profile, cmd), self.paths.logs / f"mmpbsa_{name}.log", cwd=rep_dir, env=env)
            replicas.append(
                {
                    "replica": name,
                    "replica_index": int(name.replace("rep", "")),
                    "seed": seeds[name],
                    "output": str(rep_dir / "FINAL_RESULTS_MMPBSA.dat"),
                    "per_frame_energy": str(rep_dir / "per_frame_energy.csv"),
                }
            )
        write_json_atomic(self.paths.mmpbsa / "mmpbsa_replicas.json", {"replicas": replicas})

    def step_analysis_audit(self) -> None:
        manifest = self.manifest()
        if not mmpbsa_enabled(self.profile):
            audit = {
                "status": "skipped",
                "job_id": manifest["job_id"],
                "frames": 0,
                "min_frames": int(self.profile["protocol"]["min_mmpbsa_frames"]),
                "replica_count": replica_count(self.profile),
                "replica_indices": replica_indices(self.profile),
                "replica_seeds": replica_seed_map(self.profile),
                "issues": [],
                "notes": ["MMPBSA skipped because mmpbsa.enabled=false."],
                "values": {},
                "replicas": [],
            }
            write_json_atomic(self.paths.mmpbsa / "audit.json", audit)
            return

        per_replica: list[dict[str, Any]] = []
        min_frames = max(1, int(round(int(self.profile["protocol"]["min_mmpbsa_frames"]) / replica_count(self.profile))))
        seeds = replica_seed_map(self.profile)
        for name in replica_names(self.profile):
            output = self.paths.mmpbsa / name / "FINAL_RESULTS_MMPBSA.dat"
            parsed = parse_mmpbsa_full(output)
            audit = audit_mmpbsa(
                parsed,
                min_frames,
                float(self.profile["mmpbsa"]["internal_limit_kcal_mol"]),
                float(self.profile["mmpbsa"]["internal_std_limit_kcal_mol"]),
            )
            entropy = parse_entropy_terms(output)
            values = dict(parsed["values"])
            values.update(entropy)
            add_pb_entropy_corrected(values)
            per_replica.append(
                {
                    "replica": name,
                    "replica_index": int(name.replace("rep", "")),
                    "seed": seeds[name],
                    "audit": audit,
                    "values": values,
                    "frames": parsed["frames"],
                    "output": str(output),
                }
            )
        values = aggregate_replica_values([item["values"] for item in per_replica])
        audit = {
            "status": "invalid" if any(item["audit"]["status"] != "valid" for item in per_replica) else "valid",
            "job_id": manifest["job_id"],
            "frames": sum(float(item["frames"] or 0.0) for item in per_replica),
            "min_frames": int(self.profile["protocol"]["min_mmpbsa_frames"]),
            "replica_min_frames": min_frames,
            "replica_count": replica_count(self.profile),
            "replica_indices": replica_indices(self.profile),
            "replica_seeds": seeds,
            "issues": [issue | {"replica": item["replica"]} for item in per_replica for issue in item["audit"]["issues"]],
            "notes": ["Replica MMPBSA values are averaged across independent runs."],
            "values": values,
            "replicas": per_replica,
        }
        write_json_atomic(self.paths.mmpbsa / "audit.json", audit)
        if not bool(self.profile["mmpbsa"]["keep_tmp"]):
            remove_paths(list(self.paths.mmpbsa.glob("rep*/_MMPBSA_*")))
        if audit["status"] != "valid":
            raise RuntimeError("MMPBSA audit failed; see mmpbsa/audit.json")

    def step_report(self) -> None:
        manifest = self.manifest()
        traj_qc = read_json(self.paths.qc / "summary.json")
        mmpbsa_audit = read_json(self.paths.mmpbsa / "audit.json")
        mmpbsa_status = mmpbsa_audit["status"]
        status = "valid" if traj_qc["status"] == "valid" and mmpbsa_status in {"valid", "skipped"} else "invalid"
        summary: dict[str, Any] = {
            "job_id": manifest["job_id"],
            "name": manifest["name"],
            "pdb_id": manifest["pdb_id"],
            "model_id": manifest.get("model_id", ""),
            "source": manifest.get("source", ""),
            "status": status,
            "trajectory_qc_status": traj_qc["status"],
            "mmpbsa_qc_status": mmpbsa_status,
            "mmpbsa_frames": mmpbsa_audit["frames"],
            "trajectory_frames": traj_qc["frames"],
            "replica_count": replica_count(self.profile),
            "replica_indices": replica_indices(self.profile),
            "replicas": replica_names(self.profile),
            "replica_seeds": replica_seed_map(self.profile),
            "frames_per_replica": manifest.get("frame_settings", {}).get("frames_per_replica"),
            "mmpbsa_frames_total": mmpbsa_audit["frames"],
            "solvent_shape_initial": manifest.get("solvent_shape_initial"),
            "solvent_shape_actual": manifest.get("solvent_shape_actual"),
            "box_retry_used": bool(manifest.get("box_retry_used", False)),
            "box_retry_reason": manifest.get("box_retry_reason", ""),
            "replica_qc": traj_qc.get("replica_qc", []),
            "dielectric_source": manifest.get("dielectric_policy", {}).get("source"),
            "dielectric_class": manifest.get("dielectric_policy", {}).get("classification"),
            "dielectric_epsilon": manifest.get("dielectric_policy", {}).get("epsilon"),
            "explicit_water_count": explicit_water_count(self.profile),
            "entropy_enabled": entropy_enabled(self.profile),
            "entropy_method": str(self.profile.get("mmpbsa", {}).get("entropy", "none")),
            "input_preparation": manifest.get("input_preparation", ""),
            "dropped_nonprotein_residues": manifest.get("dropped_nonprotein_residues", []),
            "dropped_nonprotein_residue_count": len(manifest.get("dropped_nonprotein_residues", [])),
            "input_residue_findings": manifest.get("input_residue_findings", {}),
            "deltaG_exp_kJ_mol": manifest.get("experimental_deltaG_kJ_mol"),
            "paper_mm_pbsa_kJ_mol": manifest.get("paper_mm_pbsa_kJ_mol"),
            "paper_dmm_pbsa_kJ_mol": manifest.get("paper_dmm_pbsa_kJ_mol"),
            "paper_vdw_kJ_mol": manifest.get("paper_vdw_kJ_mol"),
        }
        summary.update(mmpbsa_audit.get("values", {}))
        write_json_atomic(self.paths.result / "summary.json", summary)
        write_csv_atomic(self.paths.result / "summary.csv", [summary])


def optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG", "HIP"}
POLAR_RESIDUES = {"ASN", "GLN", "HIS", "HID", "HIE", "SER", "THR", "TYR", "CYS", "CYX", "TRP"}


def peptide_dielectric_policy(pdb: Path, manifest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    configured = profile.get("mmpbsa", {}).get("dielectric", profile.get("mmpbsa", {}).get("epsilon"))
    if configured not in (None, ""):
        epsilon = float(configured)
        return {"source": "config", "classification": "configured", "epsilon": epsilon, "reason": "mmpbsa dielectric configured in profile"}

    receptor_chains = set(split_chain_spec(str(manifest.get("receptor_chains") or "")))
    peptide_chain_set = set(split_chain_spec(str(manifest.get("peptide_chains") or "")))
    receptor_atoms: list[tuple[str, tuple[float, float, float]]] = []
    peptide_atoms: list[tuple[str, tuple[float, float, float]]] = []
    for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM  "):
            continue
        coords = pdb_coords(line)
        if coords is None:
            continue
        chain = line[21].strip()
        resname = line[17:20].strip().upper()
        if chain in receptor_chains:
            receptor_atoms.append((resname, coords))
        elif chain in peptide_chain_set:
            peptide_atoms.append((resname, coords))

    if not receptor_atoms or not peptide_atoms:
        return {"source": "auto", "classification": "unknown", "epsilon": 4.0, "reason": "could not identify receptor or peptide atoms"}

    interface_resnames: set[str] = set()
    cutoff2 = 25.0
    for rec_resname, rec_xyz in receptor_atoms:
        for pep_resname, pep_xyz in peptide_atoms:
            if distance2(rec_xyz, pep_xyz) <= cutoff2:
                interface_resnames.add(rec_resname)
                interface_resnames.add(pep_resname)
    if not interface_resnames:
        return {"source": "auto", "classification": "unknown", "epsilon": 4.0, "reason": "no receptor-peptide contacts within 5 A"}
    if interface_resnames & CHARGED_RESIDUES:
        return {"source": "auto", "classification": "charged", "epsilon": 4.0, "reason": "charged residue at interface", "interface_resnames": sorted(interface_resnames)}
    if interface_resnames & POLAR_RESIDUES:
        return {"source": "auto", "classification": "polar", "epsilon": 3.0, "reason": "polar residue at interface", "interface_resnames": sorted(interface_resnames)}
    return {"source": "auto", "classification": "nonpolar", "epsilon": 2.0, "reason": "only nonpolar residues identified at interface", "interface_resnames": sorted(interface_resnames)}


def pdb_coords(line: str) -> tuple[float, float, float] | None:
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        return None


def distance2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def entropy_enabled(profile: dict[str, Any]) -> bool:
    return str(profile.get("mmpbsa", {}).get("entropy", "none")).lower() in {"pb", "pb_only", "qh", "true"}


def parse_entropy_terms(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r"DELTA\s+S\s+TOTAL.*?([-+]?\d+(?:\.\d+)?)",
        r"Total\s+entropy.*?([-+]?\d+(?:\.\d+)?)",
        r"-T\*?DeltaS.*?([-+]?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return {"entropy_correction_kcal_mol": value, "entropy_correction_kJ_mol": value * 4.184}
    return {}


def add_pb_entropy_corrected(values: dict[str, float]) -> None:
    entropy = values.get("entropy_correction_kcal_mol")
    pb = values.get("PB_delta_total_kcal_mol")
    if entropy is None or pb is None:
        return
    corrected = pb + entropy
    values["PB_delta_total_entropy_corrected_kcal_mol"] = corrected
    values["PB_delta_total_entropy_corrected_kJ_mol"] = corrected * 4.184


def peptide_replica_qc_summaries(
    manifest: dict[str, Any],
    receptor_header: list[str],
    receptor_rows: list[list[float]],
    peptide_header: list[str],
    peptide_rows: list[list[float]],
    contact_header: list[str],
    contact_rows: list[list[float]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    per_replica = int(manifest.get("frame_settings", {}).get("frames_per_replica") or len(receptor_rows))
    summaries: list[dict[str, Any]] = []
    names = replica_names(profile)
    seeds = replica_seed_map(profile)
    if not names:
        names = ["rep01"]
    for idx, name in enumerate(names):
        start = idx * per_replica
        end = min(start + per_replica, len(receptor_rows))
        if start >= len(receptor_rows):
            continue
        rec_slice = receptor_rows[start:end]
        pep_slice = peptide_rows[start:end]
        contact_slice = contact_rows[start:end]
        receptor_stats = column_stats(receptor_header, rec_slice)["receptor_bb"]
        peptide_stats = column_stats(peptide_header, pep_slice)["peptide_bb"]
        contact_stats = column_stats(contact_header, contact_slice)
        summary: dict[str, Any] = {
            "replica": name,
            "replica_index": int(name.replace("rep", "")),
            "seed": seeds.get(name),
            "frames": len(rec_slice),
            "receptor_bb_rmsd_angstrom": receptor_stats,
            "peptide_bb_rmsd_after_receptor_fit_angstrom": peptide_stats,
            "native_contacts": contact_stats,
        }
        issues = evaluate_trajectory_qc(
            summary,
            profile["qc"],
            partner_field="peptide_bb_rmsd_after_receptor_fit_angstrom",
            partner_threshold_key="peptide_rmsd_warn_angstrom",
            partner_label="peptide",
            contact_prefix="rec_pep",
        )
        summary["issues"] = issues
        summary["status"] = "invalid" if any(issue["severity"] == "fail" for issue in issues) else "valid"
        summaries.append(summary)
    return summaries


def mmpbsa_input_text(manifest: dict[str, Any], profile: dict[str, Any], sanity: bool) -> str:
    entropy = 0 if sanity else 1 if entropy_enabled(profile) else 0
    if sanity:
        general = """&general
  startframe=1,
  endframe=1,
  interval=1,
  verbose=2,
  keep_files=1,
  netcdf=1,
/
"""
    else:
        settings = manifest["frame_settings"]
        startframe = 1 if bool(manifest.get("mmpbsa_trajectory_preselected", False)) else int(settings["startframe"])
        interval = 1 if bool(manifest.get("mmpbsa_trajectory_preselected", False)) else int(settings["interval"])
        general = f"""&general
  startframe={startframe},
  interval={interval},
  entropy={entropy},
  verbose=2,
  keep_files=1,
  netcdf=1,
/
"""
    salt = float(profile["system"]["salt_molar"])
    mmpbsa = profile.get("mmpbsa", {})
    epsilon = float(manifest.get("dielectric_policy", {}).get("epsilon") or mmpbsa.get("epsilon") or mmpbsa.get("dielectric") or 4.0)
    igb = int(mmpbsa.get("gb_igb", 5))
    epsout = float(mmpbsa.get("gb_epsout", 78.5))
    pb_exdi = float(mmpbsa.get("pb_exdi", 80.0))
    pb_inp = int(mmpbsa.get("pb_inp", 2))
    pb_radiopt = int(mmpbsa.get("pb_radiopt", 0))
    pb_prbrad = float(mmpbsa.get("pb_prbrad", 1.4))
    pb_fillratio = float(mmpbsa.get("pb_fillratio", 4.0))
    nmode = f"""&nmode
  dielc={epsilon:.3f},
/
""" if entropy else ""
    return (
        general
        + f"""&gb
  igb={igb},
  epsin={epsilon:.3f},
  epsout={epsout:.3f},
  saltcon={salt:.3f},
/
&pb
  istrng={salt:.3f},
  indi={epsilon:.3f},
  exdi={pb_exdi:.3f},
  inp={pb_inp},
  radiopt={pb_radiopt},
  prbrad={pb_prbrad:.3f},
  fillratio={pb_fillratio:.3f},
/
"""
        + nmode
    )


def label_line(raw: str, receptor_last: int) -> str:
    if not raw.startswith(("ATOM  ", "HETATM", "TER")) or len(raw) < 26:
        return raw
    try:
        residue_index = int(raw[22:26])
    except ValueError:
        return raw
    chain = "A" if residue_index <= receptor_last else "B"
    return f"{raw[:21]}{chain}{raw[22:]}"


def summarize_job(job: Path) -> dict[str, Any]:
    p = JobPaths.from_root(job.resolve())
    if (p.result / "summary.json").exists():
        return read_json(p.result / "summary.json")
    done = sorted(path.name for path in job.glob(".*_done"))
    failed = sorted(path.name for path in job.glob(".*_failed"))
    return {"job": str(job), "done": done, "failed": failed, "status": "incomplete"}

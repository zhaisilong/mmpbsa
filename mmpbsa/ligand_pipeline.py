from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .ligand_amber import prepare_input_structure, run_amber_prepare
from .ligand import run_ligand_prepare
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
    flatten_atom_range,
    explicit_water_count,
    frame_settings,
    job_id,
    job_name,
    label_chains,
    ligand_chain,
    mmpbsa_enabled,
    ligand_resname,
    mamba_command,
    model_id,
    mpi_pythonpath,
    parse_simple_mask,
    pdb_id,
    read_json,
    remove_paths,
    replica_count,
    replica_names,
    residue_atoms,
    run_logged,
    shlex_quote,
    split_path_list,
    WATER_NAMES,
    utc_now,
    write_csv_atomic,
    write_index,
    write_json_atomic,
    write_text_atomic,
)
from .md import convert_to_gromacs, run_em, run_npt, run_nvt, run_production
from .runner import DoneFileRunner, JobContext


STEPS = [
    "init",
    "prepare_input",
    "ligand_prepare",
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
    "prepare": ["init", "prepare_input", "ligand_prepare", "amber_prepare"],
    "md": ["md_convert", "md_em", "md_nvt", "md_npt", "md_production"],
    "analysis": ["analysis_prepare", "analysis_qc", "analysis_mmpbsa", "analysis_audit"],
    "report": ["report"],
}


@dataclass(frozen=True)
class JobPaths:
    root: Path
    input: Path
    ligand: Path
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
            ligand=root / "ligand",
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


class LigandPipeline(DoneFileRunner):
    STEPS = STEPS
    MODE_STEPS = MODE_STEPS

    def __init__(self, context: JobContext) -> None:
        super().__init__(context)
        self.paths = JobPaths.from_root(context.job_dir)

    def ensure_dirs(self) -> None:
        for directory in [
            self.paths.root,
            self.paths.input,
            self.paths.ligand,
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
                        rep / "ligand.prmtop",
                        rep / "md_prod_dry_center.nc",
                        rep / "mmpbsa.in",
                    ]
                )
        mmpbsa_outputs = [p.mmpbsa / ".mmpbsa_skipped"] if not mmpbsa_enabled(self.profile) else [p.mmpbsa / "mmpbsa_replicas.json"]
        outputs: dict[str, list[Path]] = {
            "init": [p.manifest, p.input / "complex.pdb"],
            "prepare_input": [p.input / "receptor.pdb", p.manifest],
            "ligand_prepare": [p.ligand / "ligand.mol2", p.ligand / "summary.json"],
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
            "prepare_input": [self.paths.input / "receptor.pdb"],
            "ligand_prepare": [self.paths.ligand],
            "amber_prepare": [self.paths.amber],
            "md_convert": [self.paths.md],
            "analysis_prepare": [self.paths.mmpbsa],
            "analysis_qc": [self.paths.qc, self.paths.structures],
            "analysis_mmpbsa": [self.paths.mmpbsa / ".mmpbsa_skipped", self.paths.mmpbsa / "mmpbsa_replicas.json", *self.paths.mmpbsa.glob("rep*/_MMPBSA_*")],
            "report": [self.paths.result],
        }
        remove_paths(mapping.get(step, []))

    def manifest(self) -> dict[str, Any]:
        return read_json(self.paths.manifest)

    def write_manifest(self, data: dict[str, Any]) -> None:
        write_json_atomic(self.paths.manifest, data)

    def step_init(self) -> None:
        row = dict(self.config)
        for key in ("complex_pdb", "receptor_chains", "ligand_file", "ligand_resname"):
            if not str(row.get(key) or "").strip():
                raise SystemExit(f"{self.context.config_path} is missing required field {key}")
        if row.get("ligand_charge") is None or str(row.get("ligand_charge")).strip() == "":
            raise SystemExit(f"{self.context.config_path} is missing required field ligand_charge")
        source_complex = self.context.resolve_path(row["complex_pdb"])
        if not source_complex.exists():
            raise SystemExit(f"Missing complex_pdb: {source_complex}")
        source_ligand = self.context.resolve_path(row["ligand_file"])
        if not source_ligand.exists():
            raise SystemExit(f"Missing ligand_file: {source_ligand}")

        shutil.copy2(source_complex, self.paths.input / "complex.pdb")
        ligand_copy = self.paths.input / f"ligand_source{source_ligand.suffix.lower()}"
        shutil.copy2(source_ligand, ligand_copy)
        cofactor_files = self.copy_optional_inputs(row.get("receptor_cofactor_files"), "cofactors")
        cofactor_frcmods = self.copy_optional_inputs(row.get("receptor_cofactor_frcmods"), "cofactor_params")
        cofactor_libs = self.copy_optional_inputs(row.get("receptor_cofactor_libs"), "cofactor_params")
        cofactor_count_text = str(row.get("receptor_cofactor_count") or row.get("receptor_cofactor_residue_count") or "").strip()
        receptor_cofactor_count = len(cofactor_files) if cofactor_count_text == "" else int(cofactor_count_text)
        try:
            ligand_charge = int(float(str(row["ligand_charge"]).strip()))
        except ValueError as exc:
            raise SystemExit(f"ligand_charge must be an integer: {row['ligand_charge']!r}") from exc

        settings = frame_settings(self.profile)
        manifest = {
            "schema_version": "mmpbsa.ligand.job.v1",
            "job_id": job_id(row),
            "name": job_name(row),
            "pdb_id": pdb_id(row),
            "model_id": model_id(row),
            "source": row.get("source", ""),
            "job_config": str(self.context.config_path),
            "protocol_path": str(self.context.protocol_path),
            "profile": self.profile,
            "source_complex_pdb": str(source_complex),
            "source_ligand_file": str(source_ligand),
            "input_complex_pdb": str(self.paths.input / "complex.pdb"),
            "input_ligand_file": str(ligand_copy),
            "receptor_chains": row["receptor_chains"].strip(),
            "ligand_chain": ligand_chain(row),
            "ligand_resseq": (row.get("ligand_resseq") or "").strip(),
            "ligand_resname": ligand_resname(row),
            "ligand_charge": ligand_charge,
            "ligand_param_mode": (row.get("ligand_param_mode") or self.profile["amber_prep"].get("default_ligand_param_mode", "auto")).strip().lower(),
            "ligand_mol2": self.resolve_optional_path_text(row.get("ligand_mol2")),
            "ligand_frcmod": self.resolve_optional_path_text(row.get("ligand_frcmod")),
            "ligand_lib": self.resolve_optional_path_list_text(row.get("ligand_lib")),
            "receptor_cofactor_files": cofactor_files,
            "receptor_cofactor_frcmods": cofactor_frcmods,
            "receptor_cofactor_libs": cofactor_libs,
            "receptor_cofactor_residue_count": receptor_cofactor_count,
            "experimental_deltaG_kJ_mol": optional_float(row.get("deltaG_exp_kJ_mol")),
            "ic50_nM": optional_float(row.get("ic50_nM")),
            "kd_nM": optional_float(row.get("kd_nM")),
            "frame_settings": settings,
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

    def resolve_optional_path_text(self, value: str | None) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return str(self.context.resolve_path(text))

    def resolve_optional_path_list_text(self, value: str | None) -> str:
        paths = [str(self.context.resolve_path(item)) for item in split_path_list(value)]
        return ",".join(paths)

    def step_prepare_input(self) -> None:
        manifest = prepare_input_structure(self.paths, self.manifest(), self.profile)
        manifest["dielectric_policy"] = infer_dielectric_policy(self.paths.input / "complex.pdb", manifest)
        self.write_manifest(manifest)

    def step_ligand_prepare(self) -> None:
        remove_paths([self.paths.ligand])
        self.paths.ligand.mkdir(parents=True, exist_ok=True)
        self.write_manifest(run_ligand_prepare(self.paths, self.manifest(), self.profile))

    def step_amber_prepare(self) -> None:
        remove_paths([self.paths.amber])
        self.paths.amber.mkdir(parents=True, exist_ok=True)
        run_amber_prepare(self.paths, self.profile)

    def step_md_convert(self) -> None:
        remove_paths([self.paths.gromacs, *self.replica_dirs()])
        self.paths.gromacs.mkdir(parents=True, exist_ok=True)
        convert_to_gromacs(self.paths, self.profile)

    def step_md_em(self) -> None:
        for paths in self.replica_paths():
            run_em(paths, self.profile)

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
        remove_paths([self.paths.mmpbsa / "md_prod_dry_center.xtc", self.paths.mmpbsa / "md_prod_dry_center.nc", *self.paths.mmpbsa.glob("rep*")])
        manifest = self.manifest()
        self.write_layout_and_mmpbsa_inputs(manifest)
        shutil.copy2(self.paths.amber / "complex_dry.prmtop", self.paths.mmpbsa / "complex.prmtop")
        for rep_dir in self.replica_dirs():
            self.run_trjconv(
                rep_dir,
                self.paths.qc / "protein_ligand.ndx",
                "Complex",
                self.paths.mmpbsa / f"{rep_dir.name}_dry_center.xtc",
                self.paths.logs / f"trjconv_{rep_dir.name}_dry_center.log",
            )
        run_logged(mamba_command(self.profile, ["cpptraj", "-i", str(self.paths.mmpbsa / "convert_dry_xtc_to_nc.in")]), self.paths.logs / "cpptraj_dry_convert.log")
        mmpbsa_manifest = {
            "enabled": mmpbsa_enabled(self.profile),
            "replica_count": replica_count(self.profile),
            "replicas": replica_names(self.profile),
            "explicit_water_count": explicit_water_count(self.profile),
        }
        if mmpbsa_enabled(self.profile):
            mmpbsa_manifest["selected_waters"] = self.prepare_replica_mmpbsa_inputs(manifest)
        write_json_atomic(self.paths.mmpbsa / "mmpbsa_manifest.json", mmpbsa_manifest)

    def write_layout_and_mmpbsa_inputs(self, manifest: dict[str, Any]) -> None:
        atoms_by_residue = residue_atoms(self.paths.amber / "complex_dry.pdb")
        receptor_first, receptor_last = parse_simple_mask(str(manifest["receptor_residue_mask"]))
        ligand_first, ligand_last = parse_simple_mask(str(manifest["ligand_residue_mask"]))
        receptor_atoms = flatten_atom_range(atoms_by_residue, receptor_first, receptor_last)
        ligand_atoms = flatten_atom_range(atoms_by_residue, ligand_first, ligand_last)
        complex_atoms = receptor_atoms + ligand_atoms
        layout = {
            "receptor_atom_range": [min(receptor_atoms), max(receptor_atoms)],
            "ligand_atom_range": [min(ligand_atoms), max(ligand_atoms)],
            "complex_atom_count": len(complex_atoms),
            "receptor_atom_count": len(receptor_atoms),
            "ligand_atom_count": len(ligand_atoms),
            "mmpbsa_trajectory_preselected": replica_count(self.profile) > 1,
        }
        manifest.update(layout)
        self.write_manifest(manifest)
        write_index(self.paths.qc / "protein_ligand.ndx", {"Complex": complex_atoms, "Receptor": receptor_atoms, "Ligand": ligand_atoms})
        fixed_mask = f":{receptor_first}-{ligand_last}"
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

    def prepare_replica_mmpbsa_inputs(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        selected_waters = select_interface_waters(
            self.paths.amber / "system_solvated.pdb",
            str(manifest["ligand_residue_mask"]),
            explicit_water_count(self.profile),
        )
        receptor_first, receptor_last = parse_simple_mask(str(manifest["receptor_residue_mask"]))
        ligand_first, ligand_last = parse_simple_mask(str(manifest["ligand_residue_mask"]))
        keep_residues = list(range(receptor_first, ligand_last + 1)) + [int(water["resnum"]) for water in selected_waters]
        system_atoms = residue_atoms(self.paths.amber / "system_solvated.pdb")
        complex_water_atoms: list[int] = []
        for resnum in keep_residues:
            complex_water_atoms.extend(system_atoms.get(resnum, []))
        if not complex_water_atoms:
            raise RuntimeError("Could not build explicit-water MMPBSA atom index from system_solvated.pdb")

        for rep_dir in self.replica_dirs():
            out = self.paths.mmpbsa / rep_dir.name
            out.mkdir(parents=True, exist_ok=True)
            write_index(out / "protein_ligand_water.ndx", {"ComplexWater": complex_water_atoms})
            self.write_selected_topology(out / "complex.prmtop", keep_residues, rep_dir.name)
            run_logged(
                mamba_command(self.profile, ligand_replica_ante_mmpbsa_command(manifest, self.profile)),
                self.paths.logs / f"ante_mmpbsa_{rep_dir.name}.log",
                cwd=out,
            )
            self.run_trjconv(
                rep_dir,
                out / "protein_ligand_water.ndx",
                "ComplexWater",
                out / "md_prod_dry_center.xtc",
                self.paths.logs / f"trjconv_{rep_dir.name}_mmpbsa_water.log",
            )
            write_text_atomic(
                out / "convert_dry_xtc_to_nc.in",
                f"""parm {out / "complex.prmtop"}
trajin {out / "md_prod_dry_center.xtc"}
autoimage anchor {manifest["receptor_residue_mask"]} fixed :{receptor_first}-{ligand_last}
trajout {out / "md_prod_dry_center.nc"} netcdf
run
""",
            )
            run_logged(mamba_command(self.profile, ["cpptraj", "-i", str(out / "convert_dry_xtc_to_nc.in")]), self.paths.logs / f"cpptraj_dry_convert_{rep_dir.name}.log")
            write_text_atomic(out / "mmpbsa_sanity.in", mmpbsa_input_text(manifest, self.profile, sanity=True))
            write_text_atomic(out / "mmpbsa.in", mmpbsa_input_text(manifest, self.profile, sanity=False))
        return selected_waters

    def write_selected_topology(self, output: Path, keep_residues: list[int], replica_name: str) -> None:
        import parmed as pmd
        from parmed.amber._amberparm import PrmtopPointers

        keep_mask = ",".join(str(resnum) for resnum in keep_residues)
        source = self.paths.amber / "system_solvated.prmtop"
        parm = pmd.load_file(str(source))
        parm.strip(f"!(:{keep_mask})")
        parm.box = None
        parm.hasbox = False
        if "POINTERS" in parm.parm_data:
            parm.parm_data["POINTERS"][PrmtopPointers.IFBOX] = 0
        for flag in ["BOX_DIMENSIONS", "SOLVENT_POINTERS", "ATOMS_PER_MOLECULE"]:
            if flag in parm.flag_list:
                parm.delete_flag(flag)
        parm.save(str(output), overwrite=True)
        write_text_atomic(
            self.paths.logs / f"parmed_selected_topology_{replica_name}.log",
            f"source: {source}\noutput: {output}\nstrip_mask: !(:{keep_mask})\nifbox: 0\natom_count: {len(parm.atoms)}\n",
        )

    def step_analysis_qc(self) -> None:
        manifest = self.manifest()
        receptor_mask = manifest["receptor_residue_mask"]
        ligand_mask = manifest["ligand_residue_mask"]
        write_text_atomic(
            self.paths.qc / "trajectory_qc.in",
            f"""parm {self.paths.mmpbsa / "complex.prmtop"}
trajin {self.paths.mmpbsa / "md_prod_dry_center.nc"}
reference {self.paths.mmpbsa / "md_prod_dry_center.nc"} 1
rms receptor_bb {receptor_mask}@N,CA,C reference out {self.paths.qc / "receptor_bb_rmsd.dat"}
rms ligand_heavy {ligand_mask}&!@H= reference nofit out {self.paths.qc / "ligand_heavy_rmsd_after_receptor_fit.dat"}
nativecontacts name rec_lig {receptor_mask} {ligand_mask}&!@H= distance 4.5 reference out {self.paths.qc / "native_contacts.dat"} mindist maxdist
run
""",
        )
        run_logged(mamba_command(self.profile, ["cpptraj", "-i", str(self.paths.qc / "trajectory_qc.in")]), self.paths.logs / "trajectory_qc.log")
        receptor_header, receptor_rows = load_cpptraj_table(self.paths.qc / "receptor_bb_rmsd.dat")
        ligand_header, ligand_rows = load_cpptraj_table(self.paths.qc / "ligand_heavy_rmsd_after_receptor_fit.dat")
        contact_header, contact_rows = load_cpptraj_table(self.paths.qc / "native_contacts.dat")
        write_trajectory_qc_csv(self.paths.qc / "trajectory_qc.csv", receptor_rows, ligand_rows, contact_header, contact_rows)
        receptor_stats = column_stats(receptor_header, receptor_rows)["receptor_bb"]
        ligand_stats = column_stats(ligand_header, ligand_rows)["ligand_heavy"]
        contact_stats = column_stats(contact_header, contact_rows)
        frames = len(receptor_rows)
        summary: dict[str, Any] = {
            "job_id": manifest["job_id"],
            "name": manifest["name"],
            "pdb_id": manifest["pdb_id"],
            "model_id": manifest.get("model_id", ""),
            "frames": frames,
            "receptor_bb_rmsd_angstrom": receptor_stats,
            "ligand_heavy_rmsd_after_receptor_fit_angstrom": ligand_stats,
            "native_contacts": contact_stats,
        }
        issues = evaluate_trajectory_qc(summary, self.profile["qc"])
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
                handle.write("REMARK chain B = ligand\n")
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
                "ligand.prmtop",
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
                    "ligand.prmtop",
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
                    "ligand.prmtop",
                    "-y",
                    "md_prod_dry_center.nc",
                ]
            env = {"PYTHONPATH": mpi_pythonpath(self.profile)}
            run_logged(mamba_command(self.profile, cmd), self.paths.logs / f"mmpbsa_{name}.log", cwd=rep_dir, env=env)
            replicas.append({"replica": name, "output": str(rep_dir / "FINAL_RESULTS_MMPBSA.dat"), "per_frame_energy": str(rep_dir / "per_frame_energy.csv")})
        write_json_atomic(self.paths.mmpbsa / "mmpbsa_replicas.json", {"replicas": replicas})

    def step_analysis_audit(self) -> None:
        manifest = self.manifest()
        if not mmpbsa_enabled(self.profile):
            audit = {
                "status": "skipped",
                "job_id": manifest["job_id"],
                "frames": 0,
                "min_frames": int(self.profile["protocol"]["min_mmpbsa_frames"]),
                "issues": [],
                "notes": ["MMPBSA skipped because mmpbsa.enabled=false."],
                "values": {},
                "replicas": [],
            }
            write_json_atomic(self.paths.mmpbsa / "audit.json", audit)
            return

        per_replica: list[dict[str, Any]] = []
        min_frames = max(1, int(round(int(self.profile["protocol"]["min_mmpbsa_frames"]) / replica_count(self.profile))))
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
            per_replica.append({"replica": name, "audit": audit, "values": values, "frames": parsed["frames"], "output": str(output)})
        values = aggregate_replica_values([item["values"] for item in per_replica])
        audit = {
            "status": "invalid" if any(item["audit"]["status"] != "valid" for item in per_replica) else "valid",
            "job_id": manifest["job_id"],
            "frames": sum(float(item["frames"] or 0.0) for item in per_replica),
            "min_frames": int(self.profile["protocol"]["min_mmpbsa_frames"]),
            "replica_min_frames": min_frames,
            "replica_count": replica_count(self.profile),
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
            "mmpbsa_enabled": mmpbsa_enabled(self.profile),
            "explicit_water_count": explicit_water_count(self.profile),
            "dielectric_epsilon": manifest.get("dielectric_policy", {}).get("epsilon"),
            "dielectric_class": manifest.get("dielectric_policy", {}).get("classification"),
            "deltaG_exp_kJ_mol": manifest.get("experimental_deltaG_kJ_mol"),
            "ligand_resname": manifest.get("ligand_resname"),
            "ligand_charge": manifest.get("ligand_charge"),
            "ligand_param_mode": manifest.get("ligand_param_mode"),
            "charge_method": manifest.get("ligand_prepare", {}).get("charge_method"),
            "ic50_nM": manifest.get("ic50_nM"),
            "kd_nM": manifest.get("kd_nM"),
        }
        summary.update(mmpbsa_audit.get("values", {}))
        write_json_atomic(self.paths.result / "summary.json", summary)
        write_csv_atomic(self.paths.result / "summary.csv", [summary])


def optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG", "HIP"}
POLAR_RESIDUES = {"ASN", "GLN", "HIS", "HID", "HIE", "SER", "THR", "TYR", "CYS", "CYX", "TRP"}


def infer_dielectric_policy(complex_pdb: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    ligand_charge = int(manifest.get("ligand_charge") or 0)
    if ligand_charge != 0:
        return {"classification": "charged", "epsilon": 4.0, "reason": "ligand formal charge is non-zero"}

    receptor_chains = set(str(manifest.get("receptor_chains") or "").replace(",", ""))
    ligand_atoms: list[tuple[float, float, float]] = []
    receptor_atoms: list[tuple[str, tuple[float, float, float]]] = []
    ligand_chain_value = str(manifest.get("ligand_chain") or "").strip()
    ligand_resname_value = str(manifest.get("ligand_resname") or "").strip().upper()
    ligand_resseq_value = str(manifest.get("ligand_resseq") or "").strip()
    for line in complex_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        coords = pdb_coords(line)
        if coords is None:
            continue
        chain = line[21].strip()
        resname = line[17:20].strip().upper()
        resseq = line[22:26].strip()
        if chain in receptor_chains and line.startswith("ATOM  "):
            receptor_atoms.append((resname, coords))
            continue
        chain_ok = not ligand_chain_value or chain == ligand_chain_value
        resseq_ok = not ligand_resseq_value or resseq == ligand_resseq_value
        resname_ok = not ligand_resname_value or resname == ligand_resname_value
        if chain_ok and resseq_ok and resname_ok:
            ligand_atoms.append(coords)

    if not ligand_atoms or not receptor_atoms:
        return {"classification": "unknown", "epsilon": 4.0, "reason": "could not identify ligand/receptor interface atoms"}

    interface_resnames: set[str] = set()
    cutoff2 = 25.0
    for resname, rec_xyz in receptor_atoms:
        if any(distance2(rec_xyz, lig_xyz) <= cutoff2 for lig_xyz in ligand_atoms):
            interface_resnames.add(resname)
    if not interface_resnames:
        return {"classification": "unknown", "epsilon": 4.0, "reason": "no receptor atoms within 5 A of ligand"}
    if interface_resnames & CHARGED_RESIDUES:
        return {"classification": "charged", "epsilon": 4.0, "reason": "charged receptor residue at interface", "interface_resnames": sorted(interface_resnames)}
    if interface_resnames & POLAR_RESIDUES:
        return {"classification": "polar", "epsilon": 3.0, "reason": "polar receptor residue at interface", "interface_resnames": sorted(interface_resnames)}
    return {"classification": "nonpolar", "epsilon": 2.0, "reason": "only nonpolar receptor residues identified at interface", "interface_resnames": sorted(interface_resnames)}


def pdb_coords(line: str) -> tuple[float, float, float] | None:
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        return None


def distance2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def select_interface_waters(system_pdb: Path, ligand_mask: str, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    ligand_first, ligand_last = parse_simple_mask(ligand_mask)
    ligand_atoms: list[tuple[float, float, float]] = []
    water_atoms: dict[int, dict[str, Any]] = {}
    for line in system_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        coords = pdb_coords(line)
        if coords is None:
            continue
        try:
            atom_id = int(line[6:11])
            resnum = int(line[22:26])
        except ValueError:
            continue
        resname = line[17:20].strip().upper()
        atom_name = line[12:16].strip().upper()
        if ligand_first <= resnum <= ligand_last:
            ligand_atoms.append(coords)
        if resname in WATER_NAMES:
            record = water_atoms.setdefault(resnum, {"resnum": resnum, "resname": resname, "atom_ids": [], "oxygen_coords": []})
            record["atom_ids"].append(atom_id)
            if atom_name.startswith("O"):
                record["oxygen_coords"].append(coords)
    if not ligand_atoms:
        raise RuntimeError(f"Could not locate ligand atoms {ligand_mask} in {system_pdb}")
    candidates: list[dict[str, Any]] = []
    for record in water_atoms.values():
        coords_list = record["oxygen_coords"] or []
        if not coords_list:
            continue
        min_dist2 = min(distance2(wat, lig) for wat in coords_list for lig in ligand_atoms)
        candidates.append({**record, "min_distance_angstrom": math.sqrt(min_dist2)})
    selected = sorted(candidates, key=lambda item: item["min_distance_angstrom"])[:count]
    if len(selected) < count:
        raise RuntimeError(f"Requested {count} explicit waters, but only found {len(selected)} candidates in {system_pdb}")
    return selected


def ligand_replica_ante_mmpbsa_command(manifest: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    return [
        "ante-MMPBSA.py",
        "-p",
        "complex.prmtop",
        "-r",
        "receptor.prmtop",
        "-l",
        "ligand.prmtop",
        "-n",
        str(manifest["ligand_residue_mask"]),
        f"--radii={profile['amber_prep'].get('pb_radii', 'mbondi2')}",
    ]


def entropy_enabled(profile: dict[str, Any]) -> bool:
    return str(profile.get("mmpbsa", {}).get("entropy", "none")).lower() in {"pb", "pb_only", "qh", "true"}


def parse_entropy_terms(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "entropy_correction_kcal_mol": [
            r"DELTA\s+S\s+TOTAL.*?([-+]?\d+(?:\.\d+)?)",
            r"Total\s+entropy.*?([-+]?\d+(?:\.\d+)?)",
            r"-T\*?DeltaS.*?([-+]?\d+(?:\.\d+)?)",
        ]
    }
    values: dict[str, float] = {}
    for key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, text, re.IGNORECASE)
            if match:
                values[key] = float(match.group(1))
                values[key.replace("_kcal_mol", "_kJ_mol")] = values[key] * 4.184
                break
    return values


def add_pb_entropy_corrected(values: dict[str, float]) -> None:
    entropy = values.get("entropy_correction_kcal_mol")
    pb = values.get("PB_delta_total_kcal_mol")
    if entropy is None or pb is None:
        return
    corrected = pb + entropy
    values["PB_delta_total_entropy_corrected_kcal_mol"] = corrected
    values["PB_delta_total_entropy_corrected_kJ_mol"] = corrected * 4.184


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
    epsilon = float(manifest.get("dielectric_policy", {}).get("epsilon") or mmpbsa.get("default_dielectric", 4.0))
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

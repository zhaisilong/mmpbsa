from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mmpbsa.aggregate import aggregate_run_dir
from mmpbsa.analysis import add_dmm, write_trajectory_qc_csv
from mmpbsa.common import aggregate_replica_values, frame_settings, gmx_runtime, load_profile, profile_with_replica_indices, replica_indices, replica_names, replica_seed_map, residue_atoms
from mmpbsa.ligand import ligand_input_format, mol2_total_charge, run_ligand_prepare
from mmpbsa.ligand_amber import tleap_text
from mmpbsa.ligand_pipeline import infer_dielectric_policy, ligand_replica_ante_mmpbsa_command, mmpbsa_input_text, replica_mmpbsa_convert_text, select_interface_waters
from mmpbsa.md import EmUnstableError, align_gro_to_top_molecule_order, find_gro_atom_overlaps, mdp_texts
from mmpbsa.peptide_amber import prepare_input_structure as peptide_prepare_input_structure
from mmpbsa.peptide_amber import tleap_text_with_cofactors as peptide_tleap_text_with_cofactors
from mmpbsa.peptide_pipeline import PeptidePipeline
from mmpbsa.peptide_pipeline import mmpbsa_input_text as peptide_mmpbsa_input_text
from mmpbsa.peptide_pipeline import peptide_dielectric_policy
from mmpbsa.postprocess_sweep import mmpbsa_input_with_epsilon, parse_epsilons, ridge_loocv_residuals
from mmpbsa.metrics import linear_fit, pearson_r, spearman_r
from mmpbsa.replica_merge import merge_ligand_replicas, merge_peptide_replicas
from mmpbsa.runner import DoneFileRunner, JobContext, discover_job_contexts
from mmpbsa.visualize import bundle_pymol, cpptraj_alignment_text, interaction_contact_rows, interaction_contacts_svg, md_energy_basin_rows, md_energy_basin_svg, md_energy_landscape_svg, nice_ticks, trajectory_qc_svg, visualize_job, visualize_run
from validation.kras_5xco.pilot import CifAtom, PILOT_VARIANTS, peptide_resname_for_amber, variant_peptide_atoms
from validation.kras_5xco.report import report_kras_5xco_pilot
from validation.kras_5xco import report as kras_pilot_report_module
from validation.kras_6wgn_boltz.scaffold import cif_summary as kras_boltz_cif_summary
from validation.kras_6wgn_boltz.scaffold import job_config as kras_boltz_job_config
from validation.kras_6wgn_boltz.scaffold import job_id_for_row as kras_boltz_job_id_for_row
from validation.kras_6wgn_boltz.scaffold import make_boltz_jobs_from_cifs as kras_boltz_make_jobs_from_cifs
from validation.kras_6wgn_boltz.scaffold import make_boltz_jobs_from_iptm_manifest as kras_boltz_make_jobs_from_iptm_manifest
from validation.kras_6wgn_boltz.scaffold import preflight_iptm_manifest as kras_boltz_preflight_iptm_manifest
from validation.kras_6wgn_boltz.scaffold import strict_report_markdown as kras_boltz_strict_report_markdown
from validation.kras_6wgn_boltz.scaffold import write_strict_3x5ns_report as kras_boltz_write_strict_report
from validation.ligand_tyk2.scaffold import assign_jobs_to_gpus, experimental_delta_g_kj_mol, load_sdf_records

try:
    from click.testing import CliRunner

    from mmpbsa.cli import cli

    CLICK_AVAILABLE = True
except ModuleNotFoundError:
    CLICK_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]


def make_job(tmp_path: Path, job_id: str = "job1") -> Path:
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / f"{job_id}.json").write_text(json.dumps({"job_id": job_id}), encoding="utf-8")
    return job_dir


def make_context(job_dir: Path) -> JobContext:
    protocol_path = ROOT / "configs" / "smoke_20ps.yaml"
    return JobContext(
        job_id=job_dir.name,
        job_dir=job_dir,
        config_path=job_dir / f"{job_dir.name}.json",
        config={"job_id": job_dir.name},
        protocol_path=protocol_path,
        protocol=load_profile(protocol_path),
    )


def svg_text_values(svg: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", value) for value in re.findall(r"<text\b[^>]*>(.*?)</text>", svg, flags=re.S)]


def make_visual_job(root: Path, job_id: str, score: float = -10.0, partner_column: str = "ligand_heavy_rmsd_after_receptor_fit_angstrom") -> Path:
    job_dir = root / job_id
    (job_dir / "analysis" / "qc").mkdir(parents=True)
    (job_dir / "analysis" / "mmpbsa").mkdir(parents=True)
    (job_dir / "analysis" / "structures").mkdir(parents=True)
    (job_dir / "result").mkdir(parents=True)
    (job_dir / "analysis" / "qc" / "trajectory_qc.csv").write_text(
        "\n".join(
            [
                f"frame,receptor_bb_rmsd_angstrom,{partner_column},rec_lig[native],rec_lig[mindist]",
                "1,0.0,0.0,25,2.0",
                "2,0.5,1.1,20,2.2",
                "3,0.8,1.4,18,2.4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    partner_summary_key = partner_column
    (job_dir / "analysis" / "qc" / "summary.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "frames": 3,
                "status": "valid",
                "receptor_bb_rmsd_angstrom": {"mean": 0.43, "max": 0.8},
                partner_summary_key: {"mean": 0.83, "max": 1.4},
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "result" / "summary.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "name": f"Visual {job_id}",
                "status": "valid",
                "trajectory_qc_status": "valid",
                "mmpbsa_qc_status": "valid",
                "trajectory_frames": 3,
                "mmpbsa_frames": 303,
                "replica_count": 3,
                "frames_per_replica": 101,
                "GB_delta_total_kJ_mol": score,
                "GB_delta_total_kJ_mol_replica_sd": 1.5,
                "PB_delta_total_kJ_mol": score / 2.0,
                "PB_delta_total_kJ_mol_replica_sd": 2.5,
                "GB_dMM_kJ_mol": score - 5.0,
                "GB_dMM_kJ_mol_replica_sd": 3.5,
                "PB_dMM_kJ_mol": score / 2.0 - 3.0,
                "PB_dMM_kJ_mol_replica_sd": 4.5,
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "result" / "summary.csv").write_text("job_id,GB_delta_total_kJ_mol\n" f"{job_id},{score}\n", encoding="utf-8")
    (job_dir / "analysis" / "mmpbsa" / "audit.json").write_text(
        json.dumps(
            {
                "status": "valid",
                "frames": 303,
                "replicas": [
                    {
                        "replica": "rep01",
                        "values": {
                            "GB_delta_total_kJ_mol": score + 5.0,
                            "PB_delta_total_kJ_mol": score / 2.0 + 2.0,
                            "GB_dMM_kJ_mol": score - 2.0,
                            "PB_dMM_kJ_mol": score / 2.0 - 1.0,
                        },
                    },
                    {
                        "replica": "rep02",
                        "values": {
                            "GB_delta_total_kJ_mol": score - 7.0,
                            "PB_delta_total_kJ_mol": score / 2.0 - 4.0,
                            "GB_dMM_kJ_mol": score - 9.0,
                            "PB_dMM_kJ_mol": score / 2.0 - 6.0,
                        },
                    },
                    {
                        "replica": "rep03",
                        "values": {
                            "GB_delta_total_kJ_mol": score + 1.0,
                            "PB_delta_total_kJ_mol": score / 2.0 + 1.0,
                            "GB_dMM_kJ_mol": score - 4.0,
                            "PB_dMM_kJ_mol": score / 2.0 - 2.0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "manifest.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "profile": {
                    "qc": {
                        "receptor_rmsd_fail_angstrom": 5.0,
                        "ligand_rmsd_warn_angstrom": 10.0,
                        "peptide_rmsd_warn_angstrom": 10.0,
                        "native_contacts_fail_min": 1,
                        "interface_distance_fail_angstrom": 8.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    for name in ("first.pdb", "mid.pdb", "last.pdb", "pymol_trajectory.pdb"):
        (job_dir / "analysis" / "structures" / name).write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n", encoding="utf-8")
    return job_dir


def peptide_pdb_with_hetatm() -> str:
    return "\n".join(
        [
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N",
            "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C",
            "HETATM    3  S   SO4 A 101       2.000   0.000   0.000  1.00  0.00           S",
            "HETATM    4  O1  SO4 A 101       2.500   0.000   0.000  1.00  0.00           O",
            "ATOM      5  N   ALA B   2       3.000   0.000   0.000  1.00  0.00           N",
            "ATOM      6  CA  ALA B   2       4.000   0.000   0.000  1.00  0.00           C",
            "END",
        ]
    ) + "\n"


def peptide_pdb_with_amber_hetatm_caps() -> str:
    return "\n".join(
        [
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N",
            "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C",
            "HETATM    3  C   ACE B   2       2.000   0.000   0.000  1.00  0.00           C",
            "HETATM    4  SG  CYX B   3       3.000   0.000   0.000  1.00  0.00           S",
            "HETATM    5  N   NH2 B   4       4.000   0.000   0.000  1.00  0.00           N",
            "END",
        ]
    ) + "\n"


def minimal_boltz_cif() -> str:
    headers = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.label_asym_id",
        "_atom_site.auth_asym_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    rows = [
        "ATOM 1 C CA . MET 1 1 ? A A 0.0 0.0 0.0 1.0 10.0 MET CA 1",
        "HETATM 2 C C1 . LIG1 1 1 ? L L 1.0 0.0 0.0 1.0 10.0 LIG1 C1 1",
        "HETATM 3 C C1 . GNP 1 1 ? G G 2.0 0.0 0.0 1.0 10.0 GNP C1 1",
        "HETATM 4 MG MG . MG 1 1 ? M M 3.0 0.0 0.0 1.0 10.0 MG MG 1",
    ]
    return "data_model\n#\nloop_\n" + "\n".join(headers + rows) + "\n#\n"


class FakeRunner(DoneFileRunner):
    STEPS = ["init", "prepare", "md", "report"]
    MODE_STEPS = {
        "full": STEPS,
        "prepare": ["init", "prepare"],
        "md": ["md"],
        "analysis": [],
        "report": ["report"],
    }

    def ensure_dirs(self) -> None:
        self.context.job_dir.mkdir(parents=True, exist_ok=True)

    def required_outputs(self, step: str) -> list[Path]:
        return [self.context.job_dir / f"{step}.out"]

    def cleanup_for_step(self, step: str) -> None:
        for output in self.required_outputs(step):
            output.unlink(missing_ok=True)

    def _write(self, step: str) -> None:
        self.required_outputs(step)[0].write_text(step, encoding="utf-8")

    def step_init(self) -> None:
        self._write("init")

    def step_prepare(self) -> None:
        self._write("prepare")

    def step_md(self) -> None:
        self._write("md")

    def step_report(self) -> None:
        self._write("report")


class CoreTests(unittest.TestCase):
    def test_discover_job_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_job(root, "alpha")
            make_job(root, "beta")
            contexts = discover_job_contexts(root, ROOT / "configs" / "smoke_20ps.yaml")
            self.assertEqual([context.job_id for context in contexts], ["alpha", "beta"])
            one = discover_job_contexts(root, ROOT / "configs" / "smoke_20ps.yaml", job_id="beta")
            self.assertEqual([context.job_id for context in one], ["beta"])

    def test_job_id_must_match_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = root / "alpha"
            job_dir.mkdir()
            (job_dir / "alpha.json").write_text(json.dumps({"job_id": "wrong"}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                discover_job_contexts(root, ROOT / "configs" / "smoke_20ps.yaml")

    @unittest.skipUnless(CLICK_AVAILABLE, "click is not installed")
    def test_status_skips_non_job_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_job(root, "alpha")
            (root / "gdp_params").mkdir()

            result = CliRunner().invoke(cli, ["status", str(root)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("alpha: incomplete", result.output)
            self.assertNotIn("gdp_params", result.output)

            result = CliRunner().invoke(cli, ["status", str(root), "--job-id", "gdp_params"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Missing job config", result.output)

    def test_visualize_job_writes_qc_and_score_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = make_visual_job(root, "alpha", score=-42.0)
            report = visualize_job(job_dir, root / "visual_alpha")
            self.assertEqual(report["job_id"], "alpha")
            self.assertTrue((root / "visual_alpha" / "index.html").exists())
            self.assertTrue((root / "visual_alpha" / "qc_metrics.csv").exists())
            self.assertTrue((root / "visual_alpha" / "trajectory_qc.svg").exists())
            self.assertFalse((root / "visual_alpha" / "mmpbsa_scores.svg").exists())
            self.assertFalse((root / "visual_alpha" / "trajectory_qc.html").exists())
            self.assertFalse((root / "visual_alpha" / "mmpbsa_scores.html").exists())
            html = (root / "visual_alpha" / "index.html").read_text(encoding="utf-8")
            metrics = (root / "visual_alpha" / "qc_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("Receptor backbone RMSD", html)
            self.assertIn("Ligand heavy RMSD", html)
            self.assertIn("threshold 5", html)
            self.assertIn("Native contacts", metrics)
            qc_svg = (root / "visual_alpha" / "trajectory_qc.svg").read_text(encoding="utf-8")
            self.assertIn('viewBox="0 0 900', qc_svg)
            self.assertNotIn('width="900"', qc_svg)

    def test_write_trajectory_qc_csv_adds_replica_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "trajectory_qc.csv"

            write_trajectory_qc_csv(
                output,
                [[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]],
                [[1, 1.1], [2, 1.2], [3, 1.3], [4, 1.4]],
                ["#Frame", "rec_lig[native]"],
                [[1, 10], [2, 9], [3, 8], [4, 7]],
                replica_names=["rep01", "rep02"],
                frames_per_replica=2,
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["replica"], "rep01")
            self.assertEqual(rows[1]["replica_frame"], "2")
            self.assertEqual(rows[2]["replica"], "rep02")
            self.assertEqual(rows[2]["global_frame"], "3")

    def test_visualize_job_writes_replica_aware_qc_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = make_visual_job(root, "replica_qc", score=-12.0)
            (job_dir / "analysis" / "qc" / "trajectory_qc.csv").write_text(
                "\n".join(
                    [
                        "frame,receptor_bb_rmsd_angstrom,ligand_heavy_rmsd_after_receptor_fit_angstrom,rec_lig[native],rec_lig[mindist]",
                        "1,0.1,1.1,20,2.0",
                        "2,0.2,1.2,18,2.1",
                        "3,0.3,2.1,17,2.2",
                        "4,0.4,2.2,16,2.3",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["frame_settings"] = {"replica_names": ["rep01", "rep02"], "replica_count": 2, "frames_per_replica": 2}
            (job_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            visualize_job(job_dir, root / "visual_replica_qc")

            by_replica_path = root / "visual_replica_qc" / "trajectory_qc_by_replica.csv"
            by_replica = by_replica_path.read_text(encoding="utf-8")
            with by_replica_path.open(newline="", encoding="utf-8") as handle:
                by_replica_rows = list(csv.DictReader(handle))
            qc_metrics = (root / "visual_replica_qc" / "qc_metrics.csv").read_text(encoding="utf-8")
            qc_svg = (root / "visual_replica_qc" / "trajectory_qc.svg").read_text(encoding="utf-8")
            self.assertEqual(by_replica_rows[0]["replica"], "rep01")
            self.assertEqual(by_replica_rows[2]["replica"], "rep02")
            self.assertEqual(by_replica_rows[2]["replica_frame"], "1")
            self.assertEqual(by_replica_rows[2]["global_frame"], "3")
            self.assertIn("rep01", by_replica)
            self.assertIn("rep02", by_replica)
            self.assertNotIn("replica,", qc_metrics)
            self.assertIn('class="replica-trace"', qc_svg)
            self.assertIn("rep01", qc_svg)
            self.assertIn("rep02", qc_svg)

    def test_interaction_contacts_keep_replica_identity_for_downsampled_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "trajectory.pdb"
            model = "\n".join(
                [
                    "MODEL        {state}",
                    "ATOM      1  NZ  LYS A   1       0.000   0.000   0.000  1.00  0.00           N",
                    "ATOM      2  OD1 ASP B   2       0.000   0.000   3.000  1.00  0.00           O",
                    "ENDMDL",
                ]
            )
            pdb.write_text("\n".join(model.format(state=idx) for idx in range(1, 5)) + "\nEND\n", encoding="utf-8")
            manifest = {"frame_settings": {"replica_names": ["rep01", "rep02"], "frames_per_replica": 5}}

            rows = interaction_contact_rows(pdb, manifest, stride=2)

            self.assertEqual(rows[0]["state"], 1)
            self.assertEqual(rows[0]["global_frame"], 1)
            self.assertEqual(rows[0]["replica"], "rep01")
            self.assertEqual(rows[2]["replica_frame"], 5)
            self.assertEqual(rows[3]["global_frame"], 7)
            self.assertEqual(rows[3]["replica"], "rep02")
            self.assertEqual(rows[3]["replica_frame"], 2)
            self.assertGreater(rows[0]["hbond_like_contacts"], 0)
            self.assertGreater(rows[0]["salt_bridge_like_contacts"], 0)

    def test_nice_ticks_drops_near_duplicate_end_labels(self) -> None:
        ticks = nice_ticks(0.0, 301.0, max_ticks=8)

        self.assertIn(301.0, ticks)
        self.assertNotIn(300.0, ticks)

    def test_md_energy_trace_svg_uses_time_axis_and_basin_marker(self) -> None:
        rows = [
            {"replica": "rep01", "frame": 3, "time_ps": 2000.0, "binder_rmsd_angstrom": 3.0, "delta_potential_kJ_mol": -10.0},
            {"replica": "rep01", "frame": 1, "time_ps": 0.0, "binder_rmsd_angstrom": 1.0, "delta_potential_kJ_mol": 0.0},
            {"replica": "rep01", "frame": 2, "time_ps": 1000.0, "binder_rmsd_angstrom": 2.0, "delta_potential_kJ_mol": -5.0},
            {"replica": "rep02", "frame": 1, "time_ps": 0.0, "binder_rmsd_angstrom": 1.5, "delta_potential_kJ_mol": -2.0},
            {"replica": "rep02", "frame": 2, "time_ps": 1000.0, "binder_rmsd_angstrom": 4.0, "delta_potential_kJ_mol": -30.0},
        ]
        svg = md_energy_landscape_svg(rows)

        self.assertEqual(svg.count('class="time-trace"'), 4)
        self.assertIn("Time (ns)", svg)
        self.assertIn("MD RMSD / potential trace", svg)
        self.assertIn('class="basin-line"', svg)
        self.assertIn('class="basin-point"', svg)
        self.assertIn("Basin rep02 frame 2: 1.00 ns, 4.00 A, -30.00 kJ/mol", svg)
        self.assertIn("rep01", svg)
        self.assertIn("rep02", svg)

        basin_svg = md_energy_basin_svg(rows)
        self.assertEqual(basin_svg.count('class="basin-curve"'), 2)
        self.assertEqual(basin_svg.count('class="basin-line"'), 2)
        self.assertEqual(basin_svg.count('class="basin-point"'), 2)
        self.assertNotIn('class="time-trace"', basin_svg)
        self.assertNotIn('class="replica-path"', basin_svg)
        self.assertIn("rep01 1D RMSD basin", basin_svg)
        self.assertIn("rep02 1D RMSD basin", basin_svg)
        self.assertIn("Binder RMSD to production start", basin_svg)
        self.assertIn("Per-replica occupancy-derived free-energy profiles", basin_svg)

        basin_rows = md_energy_basin_rows(rows)
        self.assertEqual({row["replica"] for row in basin_rows}, {"rep01", "rep02"})
        self.assertEqual(sum(int(row["is_basin"]) for row in basin_rows if row["replica"] == "rep01"), 1)
        self.assertEqual(sum(int(row["is_basin"]) for row in basin_rows if row["replica"] == "rep02"), 1)

    def test_visualize_job_writes_replica_basin_csv(self) -> None:
        rows = [
            {"replica": "rep01", "frame": 1, "time_ps": 0.0, "binder_rmsd_angstrom": 1.0, "delta_potential_kJ_mol": 0.0},
            {"replica": "rep01", "frame": 2, "time_ps": 1000.0, "binder_rmsd_angstrom": 1.5, "delta_potential_kJ_mol": -5.0},
            {"replica": "rep01", "frame": 3, "time_ps": 2000.0, "binder_rmsd_angstrom": 2.0, "delta_potential_kJ_mol": -10.0},
            {"replica": "rep02", "frame": 1, "time_ps": 0.0, "binder_rmsd_angstrom": 0.8, "delta_potential_kJ_mol": 0.0},
            {"replica": "rep02", "frame": 2, "time_ps": 1000.0, "binder_rmsd_angstrom": 1.2, "delta_potential_kJ_mol": -6.0},
            {"replica": "rep02", "frame": 3, "time_ps": 2000.0, "binder_rmsd_angstrom": 1.6, "delta_potential_kJ_mol": -9.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = make_visual_job(root, "basin_job", score=-12.0)
            with patch("mmpbsa.visualize.md_energy_landscape_rows", return_value=rows):
                report = visualize_job(job_dir, root / "visual_basin")

            basin_csv = root / "visual_basin" / "md_energy_basin_by_replica.csv"
            self.assertTrue(basin_csv.exists())
            self.assertEqual(report["md_energy_basin_by_replica_csv"], str(basin_csv))
            with basin_csv.open(newline="", encoding="utf-8") as handle:
                basin_rows = list(csv.DictReader(handle))
            self.assertEqual({row["replica"] for row in basin_rows}, {"rep01", "rep02"})
            self.assertEqual(sum(int(row["is_basin"]) for row in basin_rows if row["replica"] == "rep01"), 1)
            self.assertEqual(sum(int(row["is_basin"]) for row in basin_rows if row["replica"] == "rep02"), 1)
            html = (root / "visual_basin" / "index.html").read_text(encoding="utf-8")
            self.assertIn("Per-replica occupancy-derived 1D RMSD basins", html)

    def test_svg_axis_ticks_use_data_bounds_not_padded_bounds(self) -> None:
        qc_rows: list[dict[str, Any]] = []
        for replica_idx, replica in enumerate(["rep01", "rep02", "rep03"], start=1):
            for frame in range(1, 102):
                qc_rows.append(
                    {
                        "frame": (replica_idx - 1) * 101 + frame,
                        "global_frame": (replica_idx - 1) * 101 + frame,
                        "replica": replica,
                        "replica_frame": frame,
                        "receptor_bb_rmsd_angstrom": frame * 0.015,
                        "ligand_heavy_rmsd_after_receptor_fit_angstrom": 1.0 + replica_idx * 0.2 + frame * 0.01,
                        "rec_lig[native]": 25 - frame * 0.02,
                        "rec_lig[mindist]": 2.0 + frame * 0.005,
                    }
                )
        qc_svg = trajectory_qc_svg(qc_rows, "Axis test", thresholds={"ligand_heavy_rmsd_after_receptor_fit_angstrom": 10.0})
        qc_text = svg_text_values(qc_svg)
        self.assertIn("101", qc_text)
        self.assertNotIn("-7", qc_text)
        self.assertNotIn("109", qc_text)
        self.assertGreaterEqual(qc_svg.count('class="replica-trace"'), 12)

        contacts_svg = interaction_contacts_svg(
            [
                {"frame": 1, "global_frame": 1, "replica": "rep01", "replica_frame": 1, "hbond_like_contacts": 2, "salt_bridge_like_contacts": 1},
                {"frame": 2, "global_frame": 2, "replica": "rep01", "replica_frame": 2, "hbond_like_contacts": 3, "salt_bridge_like_contacts": 1},
                {"frame": 3, "global_frame": 3, "replica": "rep02", "replica_frame": 1, "hbond_like_contacts": 1, "salt_bridge_like_contacts": 0},
                {"frame": 4, "global_frame": 4, "replica": "rep02", "replica_frame": 2, "hbond_like_contacts": 2, "salt_bridge_like_contacts": 1},
            ]
        )
        self.assertIn("rep01", contacts_svg)
        self.assertIn("rep02", contacts_svg)
        self.assertGreaterEqual(contacts_svg.count('class="replica-trace"'), 4)
        self.assertIn("solid: H-bond-like", contacts_svg)
        self.assertIn("dashed: Salt-bridge-like", contacts_svg)

        energy_rows = [
            {
                "replica": "rep01",
                "frame": idx + 1,
                "time_ps": float(idx * 1000),
                "binder_rmsd_angstrom": 1.0 + idx * 0.4,
                "delta_potential_kJ_mol": -200.0 * idx,
            }
            for idx in range(6)
        ]
        landscape_text = svg_text_values(md_energy_landscape_svg(energy_rows))
        self.assertNotIn("-0.40", landscape_text)
        self.assertNotIn("5.40", landscape_text)

        basin_text = svg_text_values(md_energy_basin_svg(energy_rows))
        self.assertNotIn("-0.18", basin_text)

    def test_visualize_run_writes_sorted_ranking_and_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_visual_job(root, "weak", score=-5.0)
            make_visual_job(root, "kras6wgn_rank_0001_model_0", score=-50.0)
            report = visualize_run(root, root / "visual_run", include_samples=True)
            self.assertEqual(report["jobs"], 2)
            self.assertTrue(report["include_samples"])
            ranking = (root / "visual_run" / "ranking.csv").read_text(encoding="utf-8")
            qc_summary = (root / "visual_run" / "qc_summary.csv").read_text(encoding="utf-8")
            self.assertLess(ranking.find("kras6wgn_rank_0001_model_0"), ranking.find("weak"))
            index = (root / "visual_run" / "index.html").read_text(encoding="utf-8")
            self.assertIn("MMPBSA Group Report", index)
            self.assertIn("table.sortable", index)
            self.assertIn("GB score", index)
            self.assertIn("PB score", index)
            self.assertIn("MMPBSA Results", index)
            self.assertIn("Trajectory QC", index)
            self.assertIn("Correlation", index)
            self.assertIn("kJ/mol", index)
            self.assertIn("Å", index)
            self.assertNotIn("Angstrom", index)
            self.assertIn('class="num sticky-rank"', index)
            self.assertIn('class="sticky-job job-cell"', index)
            self.assertIn("<th>QC</th>", index)
            self.assertNotIn("<th>Status</th><th>Trajectory QC</th><th>MMPBSA QC</th>", index)
            self.assertIn("samples/kras6wgn_rank_0001_model_0/index.html", index)
            self.assertIn("Best", index)
            self.assertIn('class="badge best"', index)
            self.assertIn("3 x 101 = 303", index)
            self.assertIn('data-sort="303"', index)
            self.assertIn("GB mean", index)
            self.assertIn("GB best", index)
            self.assertIn("GB replica SD", index)
            self.assertIn("PB mean", index)
            self.assertIn("PB best", index)
            self.assertIn("PB replica SD", index)
            self.assertIn("Native contacts mean", index)
            self.assertIn("centered-chart", index)
            self.assertIn("No correlation manifest", index)
            self.assertIn("PB_delta_total_kJ_mol_replica_best", ranking)
            self.assertIn("PB_delta_total_kJ_mol_replica_best_replica", ranking)
            self.assertIn("-29", ranking)
            self.assertIn("rep02", ranking)
            self.assertIn("native_contacts_mean", qc_summary)
            self.assertIn("chart-panel", index)
            self.assertIn("aria-sort", index)
            self.assertNotIn('width="900" height="390"', index)
            self.assertNotIn("Composite rank", index)
            self.assertFalse((root / "visual_run" / "ranking.html").exists())
            self.assertFalse((root / "visual_run" / "ranking.svg").exists())
            self.assertFalse((root / "visual_run" / "qc_overview.html").exists())
            self.assertTrue((root / "visual_run" / "samples" / "kras6wgn_rank_0001_model_0" / "index.html").exists())

    def test_visualize_run_default_composite_ranking_prefers_pb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gb_good = make_visual_job(root, "gb_good_pb_bad", score=-100.0)
            pb_good = make_visual_job(root, "pb_good", score=-1.0)
            gb_summary = json.loads((gb_good / "result" / "summary.json").read_text(encoding="utf-8"))
            gb_summary["PB_delta_total_kJ_mol"] = 50.0
            (gb_good / "result" / "summary.json").write_text(json.dumps(gb_summary), encoding="utf-8")
            pb_summary = json.loads((pb_good / "result" / "summary.json").read_text(encoding="utf-8"))
            pb_summary["PB_delta_total_kJ_mol"] = -100.0
            (pb_good / "result" / "summary.json").write_text(json.dumps(pb_summary), encoding="utf-8")

            report = visualize_run(root, root / "visual_run")

            self.assertEqual(report["sort_by"], "composite")
            ranking = (root / "visual_run" / "ranking.csv").read_text(encoding="utf-8")
            self.assertLess(ranking.find("pb_good"), ranking.find("gb_good_pb_bad"))

    def test_visualize_run_writes_dynamic_manifest_correlations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_visual_job(root, "kras6wgnb2_rank_0001_model_0", score=-10.0)
            make_visual_job(root, "kras6wgnb2_rank_0002_model_0", score=-20.0)
            invalid = make_visual_job(root, "kras6wgnb2_rank_0003_model_0", score=-30.0)
            summary_path = invalid / "result" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["status"] = "invalid"
            summary["trajectory_qc_status"] = "invalid"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            source_manifest = root / "source_manifest.csv"
            source_manifest.write_text(
                "\n".join(
                    [
                        "model,iptm,kras_kd_pred,SMILES,cif_path",
                        "rank_0001_model_0,0.1,5.0,CC,a.cif",
                        "rank_0002_model_0,0.2,4.0,CC,b.cif",
                        "rank_0003_model_0,0.3,3.0,CC,c.cif",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "boltz2_6wgn_gnp_mg_manifest.json").write_text(
                json.dumps(
                    {
                        "source_manifest": str(source_manifest),
                        "jobs": [
                            {"job_id": "kras6wgnb2_rank_0001_model_0", "representative_model": "rank_0001_model_0", "boltz2_id": "rank_0001"},
                            {"job_id": "kras6wgnb2_rank_0002_model_0", "representative_model": "rank_0002_model_0", "boltz2_id": "rank_0002"},
                            {"job_id": "kras6wgnb2_rank_0003_model_0", "representative_model": "rank_0003_model_0", "boltz2_id": "rank_0003"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = visualize_run(root, root / "visual_run")

            self.assertEqual(report["jobs"], 3)
            self.assertTrue((root / "visual_run" / "correlations.csv").exists())
            self.assertTrue((root / "visual_run" / "correlation_iptm.svg").exists())
            self.assertTrue((root / "visual_run" / "correlation_kras_kd_pred.svg").exists())
            self.assertIn("correlation_scatter_svgs", report)
            with (root / "visual_run" / "correlations.csv").open(newline="") as handle:
                correlations = list(csv.DictReader(handle))
            columns = {row["manifest_column"] for row in correlations}
            self.assertIn("iptm", columns)
            self.assertIn("kras_kd_pred", columns)
            self.assertNotIn("SMILES", columns)
            self.assertTrue(all(row["n"] == "2" for row in correlations if row["status"] == "ok"))
            index = (root / "visual_run" / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("correlation_scatter.svg", index)
            self.assertIn("correlation_iptm.svg", index)
            self.assertIn("correlation_kras_kd_pred.svg", index)
            self.assertIn("correlation-grid", index)
            self.assertIn("kras_kd_pred", index)
            self.assertIn("iptm", index)

    @unittest.skipUnless(CLICK_AVAILABLE, "click is not installed")
    def test_visualize_run_cli_zip_and_pymol_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_visual_job(root, "alpha", score=-20.0)
            output_dir = root / "report"
            result = CliRunner().invoke(cli, ["visualize", "run", str(root), "--output-dir", str(output_dir), "--pymol", "--zip"])

            self.assertEqual(result.exit_code, 0, result.output)
            report = json.loads(result.output)
            self.assertTrue(report["zip_archive"])
            self.assertTrue(Path(report["archive"]).exists())
            self.assertTrue((output_dir / "samples" / "alpha" / "pymol" / "load_pymol.pml").exists())
            index = (output_dir / "index.html").read_text(encoding="utf-8")
            sample_index = (output_dir / "samples" / "alpha" / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("<h2>Files</h2>", index)
            self.assertNotIn("<h2>Files</h2>", sample_index)
            self.assertIn("<h2>PyMOL</h2>", sample_index)
            with zipfile.ZipFile(report["archive"]) as zf:
                names = set(zf.namelist())
            self.assertIn("report/index.html", names)
            self.assertIn("report/samples/alpha/index.html", names)
            self.assertIn("report/samples/alpha/pymol/load_pymol.pml", names)

    def test_bundle_pymol_uses_selected_jobs_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_visual_job(root, "alpha", score=-20.0)
            make_visual_job(root, "beta", score=-10.0)
            report = bundle_pymol(root, root / "bundles", job_ids=["alpha"])
            bundle_dir = Path(report["bundle_dir"])
            self.assertTrue((bundle_dir / "jobs" / "alpha" / "load_pymol.pml").exists())
            self.assertFalse((bundle_dir / "jobs" / "beta").exists())
            self.assertFalse(report["zip_archive"])
            self.assertNotIn("archive", report)
            self.assertFalse((root / "bundles" / "pymol_bundle.zip").exists())
            pml = (bundle_dir / "jobs" / "alpha" / "load_pymol.pml").read_text(encoding="utf-8")
            self.assertIn("load structures/pymol_trajectory.pdb", pml)
            self.assertNotIn(str(root), pml)
            self.assertTrue((bundle_dir / "jobs" / "alpha" / "movie.pml").exists())
            self.assertTrue((bundle_dir / "jobs" / "alpha" / "render_video.sh").exists())
            self.assertTrue((bundle_dir / "jobs" / "alpha" / "bundle_manifest.json").exists())
            self.assertFalse((bundle_dir / "jobs" / "alpha" / "visual").exists())
            bundle_index = (bundle_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("PyMOL Bundle Index", bundle_index)
            self.assertIn("jobs/alpha/load_pymol.pml", bundle_index)
            self.assertIn("movie.pml", bundle_index)
            zipped = bundle_pymol(root, root / "archives", job_ids=["alpha"], archive_name="alpha_bundle.zip")
            self.assertTrue(zipped["zip_archive"])
            self.assertIn("archive", zipped)
            with zipfile.ZipFile(zipped["archive"]) as zf:
                names = set(zf.namelist())
            self.assertIn("alpha_bundle/jobs/alpha/load_pymol.pml", names)
            self.assertIn("alpha_bundle/jobs/alpha/movie.pml", names)
            self.assertIn("alpha_bundle/index.html", names)
            self.assertNotIn("alpha_bundle/jobs/beta/load_pymol.pml", names)
            zipped_default = bundle_pymol(root, root / "archives_default", job_ids=["alpha"], zip_archive=True)
            self.assertTrue(zipped_default["zip_archive"])
            self.assertTrue(Path(zipped_default["archive"]).name == "pymol_bundle.zip")

    def test_cpptraj_alignment_text_fits_receptor_before_output(self) -> None:
        text = cpptraj_alignment_text(
            Path("/tmp/complex.prmtop"),
            Path("/tmp/md.nc"),
            Path("/tmp/out"),
            ":1-171@N,CA,C",
            frames=303,
            stride=5,
            snapshots_only=False,
        )
        self.assertIn("trajin /tmp/md.nc 1 303 5", text)
        self.assertIn("reference /tmp/md.nc 1", text)
        self.assertIn("rms visual_fit :1-171@N,CA,C reference", text)
        self.assertLess(text.find("rms visual_fit :1-171@N,CA,C reference"), text.find("trajout /tmp/out/aligned_trajectory.raw.pdb pdb multi"))
        snapshots = cpptraj_alignment_text(
            Path("/tmp/complex.prmtop"),
            Path("/tmp/md.nc"),
            Path("/tmp/out"),
            ":1-171@N,CA,C",
            frames=303,
            stride=5,
            snapshots_only=True,
        )
        self.assertIn("trajin /tmp/md.nc 151 151 1", snapshots)

    def test_done_policy_default_resume_and_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = make_job(root, "alpha")
            context = make_context(job_dir)
            runner = FakeRunner(context)
            runner.run(mode="prepare")
            self.assertTrue((job_dir / ".init_done").exists())
            self.assertTrue((job_dir / ".prepare_done").exists())
            with self.assertRaises(SystemExit):
                FakeRunner(context).run(mode="prepare")
            FakeRunner(context).run(mode="prepare", resume=True)
            FakeRunner(context).run(mode="prepare", force=True)
            self.assertTrue((job_dir / ".init_done").exists())
            self.assertTrue((job_dir / ".prepare_done").exists())

    def test_mode_requires_previous_done_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = make_job(root, "alpha")
            with self.assertRaises(SystemExit):
                FakeRunner(make_context(job_dir)).run(mode="md")

    def test_empty_mode_fails_with_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = make_job(root, "alpha")
            with self.assertRaisesRegex(SystemExit, "mode 'analysis' has no steps"):
                FakeRunner(make_context(job_dir)).run(mode="analysis")

    def test_aggregate_empty_run_writes_stable_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "unfinished").mkdir()

            report = aggregate_run_dir(run_dir, root / "aggregate")

            self.assertEqual(report["jobs_total"], 1)
            self.assertEqual(report["jobs_completed"], 0)
            summary_header = (
                (root / "aggregate" / "summary.csv")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            qc_header = (
                (root / "aggregate" / "qc_summary.csv")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                summary_header,
                "job_id,name,status,GB_delta_total_kJ_mol,PB_delta_total_kJ_mol,"
                "trajectory_qc_status,mmpbsa_qc_status,mmpbsa_frames,trajectory_frames,job_dir",
            )
            self.assertEqual(
                qc_header,
                "job_id,status,trajectory_qc_status,mmpbsa_qc_status,mmpbsa_frames,trajectory_frames",
            )

    def test_aggregate_completed_run_keeps_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            job_dir = make_job(run_dir, "alpha")
            (job_dir / "result").mkdir()
            (job_dir / "result" / "summary.json").write_text(
                json.dumps(
                    {
                        "job_id": "alpha",
                        "status": "valid",
                        "trajectory_qc_status": "valid",
                        "mmpbsa_qc_status": "valid",
                        "mmpbsa_frames": 303,
                        "trajectory_frames": 1501,
                        "custom_metric": 7,
                    }
                ),
                encoding="utf-8",
            )

            report = aggregate_run_dir(run_dir, root / "aggregate")

            self.assertEqual(report["jobs_total"], 1)
            self.assertEqual(report["jobs_completed"], 1)
            with (root / "aggregate" / "summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with (root / "aggregate" / "qc_summary.csv").open(newline="", encoding="utf-8") as handle:
                qc_rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["custom_metric"], "7")
            self.assertEqual(rows[0]["job_dir"], str(job_dir))
            self.assertEqual(qc_rows[0]["job_id"], "alpha")
            self.assertEqual(qc_rows[0]["mmpbsa_frames"], "303")

    def test_align_gro_to_top_molecule_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            top = root / "system_GMX.top"
            gro = root / "system_GMX.gro"
            top.write_text(
                """
[ moleculetype ]
system 3
[ atoms ]
1 C 1 SYS C1 1 0 12.0

[ moleculetype ]
WAT 3
[ atoms ]
1 OW 1 WAT OW 1 0 16.0
2 HW 1 WAT HW1 1 0 1.0
3 HW 1 WAT HW2 1 0 1.0

[ moleculetype ]
NA+ 1
[ atoms ]
1 Na 1 NA+ NA 1 1 23.0

[ moleculetype ]
CL- 1
[ atoms ]
1 Cl 1 CL- CL 1 -1 35.0

[ molecules ]
system 1
WAT 1
NA+ 1
CL- 1
""",
                encoding="utf-8",
            )

            def gro_atom(resid: int, resname: str, atom: str, serial: int) -> str:
                return f"{resid:5d}{resname:<5s}{atom:>5s}{serial:5d}{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}"

            gro.write_text(
                "\n".join(
                    [
                        "test",
                        "6",
                        gro_atom(1, "SYS", "C1", 1),
                        gro_atom(4, "CL-", "CL", 2),
                        gro_atom(2, "WAT", "OW", 3),
                        gro_atom(2, "WAT", "HW1", 4),
                        gro_atom(2, "WAT", "HW2", 5),
                        gro_atom(3, "NA+", "NA", 6),
                        "1.0 1.0 1.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            align_gro_to_top_molecule_order(gro, top)
            atom_lines = gro.read_text(encoding="utf-8").splitlines()[2:8]
            self.assertEqual([line[5:10].strip() for line in atom_lines], ["SYS", "WAT", "WAT", "WAT", "NA+", "CL-"])
            self.assertEqual([int(line[15:20]) for line in atom_lines], [1, 2, 3, 4, 5, 6])

    def test_find_gro_atom_overlaps_detects_triclinic_image(self) -> None:
        def gro_atom(resid: int, resname: str, atom: str, serial: int, x: float, y: float, z: float) -> str:
            return f"{resid:5d}{resname:<5s}{atom:>5s}{serial:5d}{x:8.3f}{y:8.3f}{z:8.3f}"

        with tempfile.TemporaryDirectory() as tmp:
            gro = Path(tmp) / "overlap.gro"
            gro.write_text(
                "\n".join(
                    [
                        "triclinic overlap",
                        "2",
                        gro_atom(1, "WAT", "O", 1, 2.734, 4.364, 6.392),
                        gro_atom(2, "WAT", "O", 2, 5.012, 1.127, 0.782),
                        "    6.85862     6.46638     5.60004     0.00000     0.00000     2.28620     0.00000    -2.28620     3.23319",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            overlaps = find_gro_atom_overlaps(gro, 1, threshold_nm=0.08)
            self.assertEqual(len(overlaps), 1)
            self.assertAlmostEqual(overlaps[0]["distance_nm"], 0.01345, places=4)
            self.assertEqual(overlaps[0]["target"]["atom"], "O")
            self.assertEqual(overlaps[0]["neighbor"]["serial"], 2)

    def test_frame_settings_default_protocol(self) -> None:
        settings = frame_settings(load_profile(ROOT / "configs" / "default_15ns.yaml"))
        self.assertEqual(settings["startframe"], 251)
        self.assertEqual(settings["interval"], 1)
        self.assertEqual(settings["total_frames"], 751)
        self.assertEqual(settings["expected_mmpbsa_frames"], 501)
        self.assertEqual(settings["replica_count"], 1)

    def test_frame_settings_ligand_crystal_replicas(self) -> None:
        settings = frame_settings(load_profile(ROOT / "configs" / "ligand_crystal_3x5ns.yaml"))
        self.assertEqual(settings["startframe"], 151)
        self.assertEqual(settings["interval"], 1)
        self.assertEqual(settings["total_frames"], 251)
        self.assertEqual(settings["frames_per_replica"], 101)
        self.assertEqual(settings["replica_count"], 3)
        self.assertEqual(settings["expected_mmpbsa_frames"], 303)

    def test_frame_settings_ligand_crystal_3x15ns_replicas(self) -> None:
        profile = load_profile(ROOT / "configs" / "ligand_crystal_3x15ns.yaml")
        settings = frame_settings(profile)
        self.assertEqual(settings["startframe"], 251)
        self.assertEqual(settings["interval"], 1)
        self.assertEqual(settings["total_frames"], 751)
        self.assertEqual(settings["frames_per_replica"], 501)
        self.assertEqual(settings["replica_count"], 3)
        self.assertEqual(settings["replica_indices"], [1, 2, 3])
        self.assertEqual(settings["replica_names"], ["rep01", "rep02", "rep03"])
        self.assertEqual(settings["expected_mmpbsa_frames"], 1503)
        self.assertEqual(profile["protocol"]["min_mmpbsa_frames"], 1500)

    def test_frame_settings_peptide_crystal_replicas(self) -> None:
        settings = frame_settings(load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml"))
        self.assertEqual(settings["startframe"], 151)
        self.assertEqual(settings["frames_per_replica"], 101)
        self.assertEqual(settings["replica_count"], 3)
        self.assertEqual(settings["expected_mmpbsa_frames"], 303)

    def test_frame_settings_peptide_crystal_3x15ns_replicas(self) -> None:
        profile = load_profile(ROOT / "configs" / "peptide_crystal_3x15ns.yaml")
        settings = frame_settings(profile)
        self.assertEqual(settings["startframe"], 251)
        self.assertEqual(settings["interval"], 1)
        self.assertEqual(settings["total_frames"], 751)
        self.assertEqual(settings["frames_per_replica"], 501)
        self.assertEqual(settings["replica_count"], 3)
        self.assertEqual(settings["replica_indices"], [1, 2, 3])
        self.assertEqual(settings["replica_names"], ["rep01", "rep02", "rep03"])
        self.assertEqual(settings["expected_mmpbsa_frames"], 1503)
        self.assertEqual(profile["protocol"]["min_mmpbsa_frames"], 1500)

    def test_replica_index_override_keeps_global_seed(self) -> None:
        profile = load_profile(ROOT / "configs" / "peptide_crystal_3x15ns.yaml")
        single = profile_with_replica_indices(profile, [4], scale_min_frames=True)
        self.assertEqual(replica_indices(single), [4])
        self.assertEqual(replica_names(single), ["rep04"])
        self.assertEqual(single["protocol"]["min_mmpbsa_frames"], 500)
        self.assertEqual(replica_seed_map(single), {"rep04": 2026052405})
        nvt = mdp_texts(single, replica_index=replica_indices(single)[0])["nvt.mdp"]
        self.assertIn("gen-seed                = 2026052405", nvt)

    def test_ligand_replica_index_override_keeps_global_seed(self) -> None:
        profile = load_profile(ROOT / "configs" / "ligand_crystal_3x15ns.yaml")
        single = profile_with_replica_indices(profile, [4], scale_min_frames=True)
        self.assertEqual(replica_indices(single), [4])
        self.assertEqual(replica_names(single), ["rep04"])
        self.assertEqual(single["protocol"]["min_mmpbsa_frames"], 500)
        self.assertEqual(replica_seed_map(single), {"rep04": 2026052405})
        nvt = mdp_texts(single, replica_index=replica_indices(single)[0])["nvt.mdp"]
        self.assertIn("gen-seed                = 2026052405", nvt)

    def test_aggregate_replica_values(self) -> None:
        values = aggregate_replica_values(
            [
                {"GB_delta_total_kcal_mol": -10.0, "only_first": 1.0},
                {"GB_delta_total_kcal_mol": -13.0, "only_second": 2.0},
                {"GB_delta_total_kcal_mol": -16.0, "only_third": 3.0},
            ]
        )
        self.assertEqual(
            set(values),
            {"GB_delta_total_kcal_mol", "GB_delta_total_kcal_mol_replica_sd", "GB_delta_total_kcal_mol_replica_sem"},
        )
        self.assertAlmostEqual(values["GB_delta_total_kcal_mol"], -13.0)
        self.assertAlmostEqual(values["GB_delta_total_kcal_mol_replica_sd"], 3.0)
        self.assertAlmostEqual(values["GB_delta_total_kcal_mol_replica_sem"], 3.0**0.5)

    def test_merge_peptide_replicas_aggregates_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = self.write_replica_source(root, "job_rep01", "rep01", 2026052402, -10.0)
            source_b = self.write_replica_source(root, "job_rep04", "rep04", 2026052405, -16.0)
            output = root / "merged"

            report = merge_peptide_replicas(output, [source_a, source_b])

            self.assertEqual(report["replica_indices"], [1, 4])
            summary = json.loads((output / "result" / "summary.json").read_text(encoding="utf-8"))
            audit = json.loads((output / "analysis" / "mmpbsa" / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["replica_count"], 2)
            self.assertEqual(summary["replica_seeds"], {"rep01": 2026052402, "rep04": 2026052405})
            self.assertAlmostEqual(summary["GB_delta_total_kJ_mol"], -13.0)
            self.assertAlmostEqual(summary["GB_delta_total_kJ_mol_replica_sd"], 3.0 * 2**0.5)
            self.assertAlmostEqual(summary["GB_delta_total_kJ_mol_replica_sem"], 3.0)
            self.assertEqual(audit["replica_min_frames"], 500)
            self.assertEqual(audit["min_frames"], 1000)

    def test_merge_peptide_replicas_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = self.write_replica_source(root, "job_a", "rep02", 2026052403, -10.0)
            source_b = self.write_replica_source(root, "job_b", "rep02", 2026052403, -16.0)
            with self.assertRaises(SystemExit):
                merge_peptide_replicas(root / "merged", [source_a, source_b])

    def test_merge_ligand_replicas_aggregates_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra = {"ligand_resname": "LIG", "ligand_charge": 0, "charge_method": "bcc", "ic50_nM": 42.0}
            source_a = self.write_replica_source(root, "lig_rep02", "rep02", 2026052403, -20.0, extra)
            source_b = self.write_replica_source(root, "lig_rep05", "rep05", 2026052406, -26.0, extra)
            output = root / "merged_ligand"

            report = merge_ligand_replicas(output, [source_a, source_b])

            self.assertEqual(report["replica_indices"], [2, 5])
            summary = json.loads((output / "result" / "summary.json").read_text(encoding="utf-8"))
            audit = json.loads((output / "analysis" / "mmpbsa" / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["replica_count"], 2)
            self.assertEqual(summary["ligand_resname"], "LIG")
            self.assertEqual(summary["charge_method"], "bcc")
            self.assertEqual(summary["ic50_nM"], 42.0)
            self.assertAlmostEqual(summary["GB_delta_total_kJ_mol"], -23.0)
            self.assertAlmostEqual(summary["PB_delta_total_kJ_mol"], -46.0)
            self.assertIn("ligand replica jobs", audit["notes"][0])

    def write_replica_source(self, root: Path, job_id: str, replica: str, seed: int, value: float, extra_summary: dict[str, Any] | None = None) -> Path:
        job_dir = root / job_id
        (job_dir / "analysis" / "mmpbsa").mkdir(parents=True)
        (job_dir / "result").mkdir()
        index = int(replica.replace("rep", ""))
        audit = {
            "status": "valid",
            "job_id": job_id,
            "frames": 501,
            "replicas": [
                {
                    "replica": replica,
                    "replica_index": index,
                    "seed": seed,
                    "audit": {"status": "valid", "min_frames": 500, "issues": []},
                    "values": {"GB_delta_total_kJ_mol": value, "PB_delta_total_kJ_mol": value * 2.0},
                    "frames": 501,
                }
            ],
        }
        summary = {"job_id": job_id, "name": job_id, "status": "valid", "frames_per_replica": 501}
        if extra_summary:
            summary.update(extra_summary)
        (job_dir / "analysis" / "mmpbsa" / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
        (job_dir / "result" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return job_dir

    def test_add_dmm_writes_new_names_and_compat_aliases(self) -> None:
        values = {
            "GB_vdw_kcal_mol": -10.0,
            "GB_electrostatic_kcal_mol": -20.0,
            "GB_polar_solvation_kcal_mol": 15.0,
            "GB_nonpolar_solvation_kcal_mol": -1.0,
            "PB_vdw_kcal_mol": -11.0,
            "PB_electrostatic_kcal_mol": -12.0,
            "PB_polar_solvation_kcal_mol": 10.0,
            "PB_nonpolar_solvation_kcal_mol": -2.0,
            "PB_dispersion_kcal_mol": 3.0,
        }
        add_dmm(values)
        self.assertAlmostEqual(values["GB_dMM_kcal_mol"], -12.0)
        self.assertAlmostEqual(values["GB_dmm_like_kcal_mol"], values["GB_dMM_kcal_mol"])
        self.assertAlmostEqual(values["PB_dMM_kcal_mol"], -10.4)
        self.assertAlmostEqual(values["PB_dmm_like_kcal_mol"], values["PB_dMM_kcal_mol"])

    def test_protocol_mmpbsa_rank_defaults(self) -> None:
        normal_protocols = [
            "default_15ns.yaml",
            "ligand_default_15ns.yaml",
            "ligand_crystal_1x15ns.yaml",
            "ligand_crystal_3x5ns.yaml",
            "ligand_crystal_3x5ns_mmpbsa_bcc.yaml",
            "ligand_crystal_3x15ns.yaml",
            "ligand_crystal_3x15ns_mmpbsa_bcc.yaml",
            "ligand_crystal_5x5ns.yaml",
            "peptide_crystal_1x15ns.yaml",
            "peptide_crystal_3x5ns.yaml",
            "peptide_crystal_3x15ns.yaml",
            "peptide_crystal_5x5ns.yaml",
        ]
        for name in normal_protocols:
            with self.subTest(name=name):
                profile = load_profile(ROOT / "configs" / name)
                self.assertEqual(profile["mmpbsa"]["np"], 16)
        smoke = load_profile(ROOT / "configs" / "smoke_20ps.yaml")
        self.assertEqual(smoke["mmpbsa"]["np"], 1)

    def test_gmx_runtime_expands_environment(self) -> None:
        profile = {"runtime": {"gmxrc": "${GMXRC}", "gmx_bin": "${GMX_BIN}"}}
        with patch.dict("os.environ", {"GMXRC": "/opt/gromacs/bin/GMXRC", "GMX_BIN": "gmx_mpi"}, clear=True):
            self.assertEqual(gmx_runtime(profile), ("/opt/gromacs/bin/GMXRC", "gmx_mpi"))

    def test_gmx_runtime_requires_set_environment(self) -> None:
        profile = {"runtime": {"gmxrc": "${GMXRC}", "gmx_bin": "gmx_mpi"}}
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit) as error:
                gmx_runtime(profile)
        self.assertIn("GMXRC", str(error.exception))

    def test_ligand_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(ligand_input_format(Path("lig.sdf")), "sdf")
            self.assertEqual(ligand_input_format(Path("lig.mol2")), "mol2")
            mol2 = root / "ligand.mol2"
            mol2.write_text(
                "\n".join(
                    [
                        "@<TRIPOS>MOLECULE",
                        "LIG",
                        "2 1 0 0 0",
                        "SMALL",
                        "USER_CHARGES",
                        "@<TRIPOS>ATOM",
                        "1 C1 0.0 0.0 0.0 c3 1 LIG -0.25",
                        "2 H1 0.0 0.0 1.0 h1 1 LIG 0.25",
                        "@<TRIPOS>BOND",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertAlmostEqual(mol2_total_charge(mol2) or 0.0, 0.0)

    def test_tleap_text_for_ligand(self) -> None:
        profile = load_profile(ROOT / "configs" / "default_15ns.yaml")
        text = tleap_text(Path("rec.pdb"), Path("ligand.mol2"), Path("ligand.frcmod"), [Path("extra.off")], 12, profile)
        self.assertIn("source leaprc.protein.ff14SB", text)
        self.assertIn("source leaprc.gaff2", text)
        self.assertIn("set default PBRadii mbondi2", text)
        self.assertIn('loadamberparams "ligand.frcmod"', text)
        self.assertIn("mol = combine { rec lig }", text)
        self.assertIn("addIonsRand mol Na+ 12", text)
        text = tleap_text(
            Path("rec.pdb"),
            Path("ligand.mol2"),
            Path("ligand.frcmod"),
            [],
            12,
            profile,
            cofactor_files=[Path("GDP.pdb")],
            cofactor_frcmods=[Path("frcmod.phos")],
            cofactor_libs=[Path("GDP.prep")],
        )
        self.assertIn('loadamberprep "GDP.prep"', text)
        self.assertIn('loadamberparams "frcmod.phos"', text)
        self.assertIn('cof1 = loadpdb "GDP.pdb"', text)
        self.assertIn("mol = combine { rec cof1 lig }", text)

    def test_resp_auto_ligand_prepare_fails_fast(self) -> None:
        profile = load_profile(ROOT / "configs" / "ligand_crystal_3x5ns.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = type("Paths", (), {"ligand": root / "ligand"})()
            manifest = {
                "ligand_param_mode": "auto",
                "input_ligand_file": str(root / "ligand.sdf"),
                "ligand_charge": 0,
                "ligand_resname": "LIG",
            }
            with self.assertRaises(SystemExit):
                run_ligand_prepare(paths, manifest, profile)

    def test_ligand_dielectric_and_water_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "complex.pdb"
            pdb.write_text(
                "\n".join(
                    [
                        "ATOM      1  CA  LEU A   1       0.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      2  CA  ASP A   2       2.000   0.000   0.000  1.00  0.00           C",
                        "HETATM    3  C1  LIG B   3       3.000   0.000   0.000  1.00  0.00           C",
                        "HETATM    4  O   WAT W   4       3.400   0.000   0.000  1.00  0.00           O",
                        "HETATM    5  H1  WAT W   4       3.500   0.100   0.000  1.00  0.00           H",
                        "HETATM    6  H2  WAT W   4       3.500  -0.100   0.000  1.00  0.00           H",
                        "HETATM    7  O   WAT W   5      30.000   0.000   0.000  1.00  0.00           O",
                        "HETATM    8  H1  WAT W   5      30.100   0.100   0.000  1.00  0.00           H",
                        "HETATM    9  H2  WAT W   5      30.100  -0.100   0.000  1.00  0.00           H",
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = {
                "ligand_charge": 0,
                "receptor_chains": "A",
                "ligand_chain": "B",
                "ligand_resname": "LIG",
                "ligand_resseq": "3",
            }
            policy = infer_dielectric_policy(pdb, manifest)
            self.assertEqual(policy["classification"], "charged")
            self.assertEqual(policy["epsilon"], 4.0)
            waters = select_interface_waters(pdb, ":3", 1)
            self.assertEqual(waters[0]["resnum"], 4)

    def test_ligand_explicit_water_selection_handles_extended_pdb_residue_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "system_solvated.pdb"
            pdb.write_text(
                "\n".join(
                    [
                        "ATOM      1  CA  LEU A   1       0.000   0.000   0.000  1.00  0.00           C",
                        "HETATM    2  C1  LIG B   3       3.000   0.000   0.000  1.00  0.00           C",
                        "HETATM    3  O   WAT  1076      30.000   0.000   0.000  1.00  0.00           O",
                        "HETATM    4  H1  WAT  1076      30.100   0.100   0.000  1.00  0.00           H",
                        "HETATM    5  H2  WAT  1076      30.100  -0.100   0.000  1.00  0.00           H",
                        "HETATM    6  O   WAT  10769      3.400   0.000   0.000  1.00  0.00           O",
                        "HETATM    7  H1  WAT  10769      3.500   0.100   0.000  1.00  0.00           H",
                        "HETATM    8  H2  WAT  10769      3.500  -0.100   0.000  1.00  0.00           H",
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            waters = select_interface_waters(pdb, ":3", 1)
            self.assertEqual(waters[0]["resnum"], 10769)
            self.assertEqual(waters[0]["atom_ids"], [6, 7, 8])
            atoms = residue_atoms(pdb)
            self.assertEqual(atoms[1076], [3, 4, 5])
            self.assertEqual(atoms[10769], [6, 7, 8])

    def test_ligand_mmpbsa_input_uses_policy_params(self) -> None:
        profile = load_profile(ROOT / "configs" / "ligand_crystal_3x5ns_mmpbsa_bcc.yaml")
        manifest = {"frame_settings": frame_settings(profile), "dielectric_policy": {"epsilon": 3.0}}
        text = mmpbsa_input_text(manifest, profile, sanity=False)
        self.assertIn("igb=5", text)
        self.assertIn("epsin=3.000", text)
        self.assertIn("indi=3.000", text)
        self.assertIn("radiopt=0", text)
        self.assertIn("inp=2", text)
        self.assertIn("entropy=0", text)
        self.assertNotIn("&nmode", text)
        configured = json.loads(json.dumps(profile))
        configured["mmpbsa"]["entropy"] = "pb"
        text = mmpbsa_input_text(manifest, configured, sanity=False)
        self.assertIn("entropy=1", text)
        self.assertIn("&nmode", text)
        sanity = mmpbsa_input_text(manifest, profile, sanity=True)
        self.assertNotIn("entropy=1", sanity)
        self.assertNotIn("&nmode", sanity)

    def test_ligand_replica_mmpbsa_convert_applies_frame_window(self) -> None:
        profile = load_profile(ROOT / "configs" / "ligand_crystal_3x5ns_mmpbsa_bcc.yaml")
        text = replica_mmpbsa_convert_text(
            Path("complex.prmtop"),
            Path("rep01.xtc"),
            Path("rep01.nc"),
            ":1-171",
            ":1-172",
            frame_settings(profile),
        )
        self.assertIn("trajin rep01.xtc 151 251 1", text)
        self.assertIn("trajout rep01.nc netcdf", text)

    def test_kras_boltz_make_jobs_from_cifs_requires_manifest_smiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cif_dir = root / "cifs"
            cif_dir.mkdir()
            (cif_dir / "rank_0122_model_1.cif").write_text(minimal_boltz_cif(), encoding="utf-8")
            manifest = root / "md_selected_manifest.csv"
            manifest.write_text("rank,representative_model,smiles\n122,rank_0122_model_1,\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                kras_boltz_make_jobs_from_cifs(cif_dir, manifest, root / "run", prepare_inputs=False)

    def test_kras_boltz_make_jobs_from_cifs_writes_3x15_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cif_dir = root / "cifs"
            cif_dir.mkdir()
            (cif_dir / "rank_0122_model_1.cif").write_text(minimal_boltz_cif(), encoding="utf-8")
            manifest = root / "md_selected_manifest.csv"
            manifest.write_text(
                "rank,representative_model,smiles,representative_composite_score\n"
                "122,rank_0122_model_1,CCO,0.9\n",
                encoding="utf-8",
            )

            report = kras_boltz_make_jobs_from_cifs(cif_dir, manifest, root / "run", prepare_inputs=False)

            self.assertEqual(report["production_protocol"], "configs/ligand_crystal_3x15ns_mmpbsa_bcc.yaml")
            self.assertEqual(report["job_count"], 1)
            job_id = "kras6wgnb2_rank_0122_model_1"
            config = json.loads((root / "run" / job_id / f"{job_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(config["job_id"], job_id)
            self.assertEqual(config["boltz_smiles"], "CCO")
            self.assertEqual(config["input_preparation"]["status"], "skipped")
            run_script = (root / "run" / "run_top10_3x15ns.sh").read_text(encoding="utf-8")
            self.assertIn("configs/ligand_crystal_3x15ns_mmpbsa_bcc.yaml", run_script)
            self.assertTrue((root / "run" / "boltz2_6wgn_gnp_mg_manifest.json").exists())

    def test_kras_boltz_iptm_preflight_reports_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "md_selected_iptm_only_manifest.csv"
            manifest.write_text(
                "set,selection_rank,rank,name,model,iptm,kras_kd_pred,kd,kd_log10_nM_from_M,cif_path,prediction_dir\n"
                "primary,1,316,PP0316,PP0316_model_0,0.93,-1.2,5e-11,-1.3,missing/PP0316_model_0.cif,missing\n",
                encoding="utf-8",
            )

            report = kras_boltz_preflight_iptm_manifest(manifest)

            self.assertFalse(report["passed"])
            self.assertIn("PP0316_model_0", report["failures"][0])
            self.assertIn("missing CIF path", report["failures"][0])
            self.assertIn("missing ligand topology", report["failures"][0])

    def test_kras_boltz_iptm_manifest_accepts_ordered_uppercase_smiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cif_dir = root / "cifs"
            cif_dir.mkdir()
            for model in ("rank_0892_model_6", "rank_0992_model_4"):
                (cif_dir / f"{model}.cif").write_text(minimal_boltz_cif(), encoding="utf-8")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "set,rank,name,model,SMILES,iptm,cif_path,prediction_dir\n"
                "primary,892,rank_0892,rank_0892_model_6,CCO,0.95,missing/rank_0892_model_6.cif,predictions/rank_0892\n"
                "primary,992,rank_0992,rank_0992_model_4,CCN,0.94,missing/rank_0992_model_4.cif,predictions/rank_0992\n",
                encoding="utf-8",
            )

            report = kras_boltz_make_jobs_from_iptm_manifest(
                manifest,
                root / "run",
                local_cif_dir=cif_dir,
                smiles_manifest=None,
                prepare_inputs=False,
            )

            self.assertEqual(report["job_count"], 2)
            first = json.loads((root / "run" / "kras6wgnb2_rank_0892_model_6" / "kras6wgnb2_rank_0892_model_6.json").read_text(encoding="utf-8"))
            second = json.loads((root / "run" / "kras6wgnb2_rank_0992_model_4" / "kras6wgnb2_rank_0992_model_4.json").read_text(encoding="utf-8"))
            self.assertEqual(first["selection_rank"], "1")
            self.assertEqual(first["boltz_smiles"], "CCO")
            self.assertEqual(second["selection_rank"], "2")
            self.assertEqual(second["boltz_smiles"], "CCN")

    def test_kras_boltz_make_jobs_from_iptm_manifest_writes_worker_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cif_dir = root / "cifs"
            cif_dir.mkdir()
            cif = cif_dir / "rank_0892_model_6.cif"
            cif.write_text(minimal_boltz_cif(), encoding="utf-8")
            manifest = root / "md_selected_iptm_only_manifest.csv"
            manifest.write_text(
                "set,selection_rank,rank,name,model,iptm,kras_kd_pred,kd,kd_log10_nM_from_M,cif_path,prediction_dir\n"
                "primary,1,892,rank_0892,rank_0892_model_6,0.93,-1.2,5e-11,-1.3,missing/rank_0892_model_6.cif,predictions/rank_0892\n",
                encoding="utf-8",
            )
            smiles_manifest = root / "old_smiles.csv"
            smiles_manifest.write_text("rank,name,representative_model,smiles\n892,rank_0892,rank_0892_model_0,CCO\n", encoding="utf-8")

            report = kras_boltz_make_jobs_from_iptm_manifest(
                manifest,
                root / "run",
                local_cif_dir=cif_dir,
                smiles_manifest=smiles_manifest,
                prepare_inputs=False,
            )

            self.assertEqual(report["job_count"], 1)
            self.assertEqual(report["production_protocol"], "configs/ligand_crystal_3x15ns_mmpbsa_bcc.yaml")
            job_id = "kras6wgnb2_rank_0892_model_6"
            config = json.loads((root / "run" / job_id / f"{job_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(config["job_id"], job_id)
            self.assertEqual(config["selection_rank"], "1")
            self.assertEqual(config["boltz2_id"], "rank_0892")
            self.assertEqual(config["boltz_smiles"], "CCO")
            worker_script = (root / "run" / "run_top10_3x15ns_gpu_workers.sh").read_text(encoding="utf-8")
            self.assertIn("GPUS:-4,5,6,7", worker_script)
            self.assertIn("--protocol \"$PROTOCOL\" --resume", worker_script)

    def test_kras_boltz_strict_report_rejects_smoke_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_id = "kras6wgn_rank_0987_model_1"
            (root / "boltz_6wgn_gnp_mg_manifest.json").write_text(
                json.dumps({"jobs": [{"index": 1, "job_id": job_id, "rank": "987"}]}),
                encoding="utf-8",
            )
            job_dir = root / job_id
            (job_dir / "result").mkdir(parents=True)
            (job_dir / "analysis" / "mmpbsa").mkdir(parents=True)
            (job_dir / "analysis" / "qc").mkdir(parents=True)
            (job_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "profile": {"protocol": {"production_ns": 0.02, "mmpbsa_start_ns": 0.0}},
                        "frame_settings": {"startframe": 1, "frames_per_replica": 11},
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "result" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "valid",
                        "trajectory_qc_status": "valid",
                        "mmpbsa_qc_status": "valid",
                        "replica_count": 1,
                        "frames_per_replica": 11,
                        "mmpbsa_frames": 11,
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "analysis" / "mmpbsa" / "audit.json").write_text(
                json.dumps({"status": "valid", "frames": 11, "replicas": [{"frames": 11}], "issues": []}),
                encoding="utf-8",
            )
            (job_dir / "analysis" / "qc" / "summary.json").write_text(json.dumps({"status": "valid"}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                kras_boltz_write_strict_report(root, root / "reports", expected_jobs=1)

    def test_kras_boltz_strict_report_accepts_3x15_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_id = "kras6wgnb2_rank_0122_model_1"
            (root / "boltz2_6wgn_gnp_mg_manifest.json").write_text(
                json.dumps({"jobs": [{"index": 1, "job_id": job_id, "rank": "122"}]}),
                encoding="utf-8",
            )
            job_dir = root / job_id
            (job_dir / "result").mkdir(parents=True)
            (job_dir / "analysis" / "mmpbsa").mkdir(parents=True)
            (job_dir / "analysis" / "qc").mkdir(parents=True)
            (job_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "profile": {"protocol": {"production_ns": 15.0, "mmpbsa_start_ns": 5.0}},
                        "frame_settings": {"startframe": 251, "frames_per_replica": 501},
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "result" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "valid",
                        "trajectory_qc_status": "valid",
                        "mmpbsa_qc_status": "valid",
                        "replica_count": 3,
                        "frames_per_replica": 501,
                        "mmpbsa_frames": 1503,
                        "GB_delta_total_kJ_mol": -100.0,
                        "PB_delta_total_kJ_mol": -50.0,
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "analysis" / "mmpbsa" / "audit.json").write_text(
                json.dumps({"status": "valid", "frames": 1503, "replicas": [{"frames": 501}, {"frames": 501}, {"frames": 501}], "issues": []}),
                encoding="utf-8",
            )
            (job_dir / "analysis" / "qc" / "summary.json").write_text(json.dumps({"status": "valid"}), encoding="utf-8")

            report = kras_boltz_write_strict_report(root, root / "reports", profile_name="3x15ns", expected_jobs=1)

            self.assertTrue(report["strict_pass"])
            self.assertEqual(report["mmpbsa_window"], "5-15 ns")
            markdown = (root / "reports" / "report.md").read_text(encoding="utf-8")
            self.assertIn("Strict 3x15 ns Report", markdown)
            self.assertIn("3 x 501 = 1503", markdown)

    def test_kras_boltz_strict_report_uses_replica_frame_labels(self) -> None:
        report = {
            "generated_at": "2026-01-01T00:00:00Z",
            "run_dir": "/tmp/run",
            "production_jobs": 1,
            "smoke_jobs_included": 0,
            "md_protocol": "3x5ns",
            "mmpbsa_window": "3-5 ns",
            "expected_replicas": 3,
            "expected_frames_per_replica": 101,
            "expected_frames_per_job": 303,
            "strict_pass": True,
        }
        rows = [
            {
                "GB_primary_rank": 1,
                "boltz_rank": 1,
                "job_id": "kras6wgn_rank_0001_model_0",
                "GB_delta_total_kJ_mol": -100.0,
                "GB_delta_total_kJ_mol_replica_sd": 5.0,
                "PB_delta_total_kJ_mol": -50.0,
                "PB_delta_total_kJ_mol_replica_sd": 3.0,
                "replica_count": 3,
                "frames_per_replica": 101,
                "mmpbsa_frames": 303,
                "strict_3_5ns_mmpbsa": True,
                "replica_frames": "101;101;101",
                "audit_issue_count": 0,
                "strict_issues": "",
            }
        ]

        markdown = kras_boltz_strict_report_markdown(report, rows, rows)

        self.assertIn("Expected frames/job: 3 x 101 = 303", markdown)
        self.assertIn("GB mean kJ/mol", markdown)
        self.assertIn("PB replica SD", markdown)
        self.assertIn("| 1 | 1 | `kras6wgn_rank_0001_model_0` | -100.00 | 5.00 | -50.00 | 3.00 | 3 x 101 = 303 | pass |", markdown)

    def test_ligand_replica_ante_mmpbsa_uses_selected_complex_input(self) -> None:
        profile = load_profile(ROOT / "configs" / "ligand_crystal_3x5ns_mmpbsa_bcc.yaml")
        command = ligand_replica_ante_mmpbsa_command({"ligand_residue_mask": ":289"}, profile)
        self.assertEqual(command[command.index("-p") + 1], "complex.prmtop")
        self.assertEqual(command[command.index("-r") + 1], "receptor.prmtop")
        self.assertEqual(command[command.index("-l") + 1], "ligand.prmtop")
        self.assertEqual(command[command.index("-n") + 1], ":289")
        self.assertIn("--radii=mbondi2", command)
        self.assertNotIn("-c", command)
        self.assertNotIn("complex_selected.prmtop", command)

    def test_peptide_dielectric_policy_config_and_auto(self) -> None:
        profile = load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml")
        configured = json.loads(json.dumps(profile))
        configured["mmpbsa"]["epsilon"] = 2.5
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "selected_protein.pdb"
            pdb.write_text(
                "\n".join(
                    [
                        "ATOM      1  CA  LEU A   1       0.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      2  CA  ASP A   2       2.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      3  CA  LYS B   3       3.000   0.000   0.000  1.00  0.00           C",
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = {"receptor_chains": "A", "peptide_chains": "B"}
            policy = peptide_dielectric_policy(pdb, manifest, profile)
            self.assertEqual(policy["source"], "auto")
            self.assertEqual(policy["classification"], "charged")
            self.assertEqual(policy["epsilon"], 4.0)
            policy = peptide_dielectric_policy(pdb, manifest, configured)
            self.assertEqual(policy["source"], "config")
            self.assertEqual(policy["epsilon"], 2.5)

    def test_peptide_mmpbsa_input_uses_policy_params(self) -> None:
        profile = load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml")
        manifest = {"frame_settings": frame_settings(profile), "dielectric_policy": {"epsilon": 3.0}}
        text = peptide_mmpbsa_input_text(manifest, profile, sanity=False)
        self.assertIn("igb=5", text)
        self.assertIn("epsin=3.000", text)
        self.assertIn("indi=3.000", text)
        self.assertIn("radiopt=0", text)
        self.assertIn("inp=2", text)
        self.assertIn("entropy=0", text)
        self.assertNotIn("&nmode", text)
        manifest["mmpbsa_trajectory_preselected"] = True
        text = peptide_mmpbsa_input_text(manifest, profile, sanity=False)
        self.assertIn("startframe=1", text)
        self.assertIn("interval=1", text)
        configured = json.loads(json.dumps(profile))
        configured["mmpbsa"]["entropy"] = "pb"
        text = peptide_mmpbsa_input_text(manifest, configured, sanity=False)
        self.assertIn("entropy=1", text)
        self.assertIn("&nmode", text)

    def test_peptide_pipeline_requires_replica_mmpbsa_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = make_job(Path(tmp), "pep")
            context = JobContext(
                job_id="pep",
                job_dir=job_dir,
                config_path=job_dir / "pep.json",
                config={"job_id": "pep"},
                protocol_path=ROOT / "configs" / "peptide_crystal_3x5ns.yaml",
                protocol=load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml"),
            )
            pipeline = PeptidePipeline(context)
            analysis_prepare = {path.relative_to(job_dir).as_posix() for path in pipeline.required_outputs("analysis_prepare")}
            self.assertIn("analysis/mmpbsa/rep01/mmpbsa.in", analysis_prepare)
            self.assertIn("analysis/mmpbsa/rep02/md_prod_dry_center.nc", analysis_prepare)
            self.assertIn("analysis/mmpbsa/rep03/peptide.prmtop", analysis_prepare)
            self.assertNotIn("analysis/mmpbsa/mmpbsa.in", analysis_prepare)
            analysis_mmpbsa = {path.relative_to(job_dir).as_posix() for path in pipeline.required_outputs("analysis_mmpbsa")}
            self.assertEqual(analysis_mmpbsa, {"analysis/mmpbsa/mmpbsa_replicas.json"})

    def test_peptide_md_em_retries_with_box_on_unstable_em(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = make_job(Path(tmp), "pep")
            profile = json.loads(json.dumps(load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml")))
            profile["system"]["allow_box_retry"] = True
            context = JobContext(
                job_id="pep",
                job_dir=job_dir,
                config_path=job_dir / "pep.json",
                config={"job_id": "pep"},
                protocol_path=ROOT / "configs" / "peptide_crystal_3x5ns.yaml",
                protocol=profile,
            )
            pipeline = PeptidePipeline(context)
            pipeline.ensure_dirs()
            pipeline.write_manifest(
                {
                    "job_id": "pep",
                    "profile": profile,
                    "solvent_shape_initial": "oct",
                    "solvent_shape_actual": "oct",
                    "box_retry_used": False,
                }
            )
            with (
                patch("mmpbsa.peptide_pipeline.run_em", side_effect=[EmUnstableError("unstable EM"), None, None, None]) as run_em_mock,
                patch("mmpbsa.peptide_pipeline.run_amber_prepare") as amber_mock,
                patch("mmpbsa.peptide_pipeline.convert_to_gromacs") as convert_mock,
            ):
                pipeline.step_md_em()
            manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(run_em_mock.call_count, 4)
            amber_mock.assert_called_once()
            convert_mock.assert_called_once()
            self.assertTrue(manifest["box_retry_used"])
            self.assertEqual(manifest["solvent_shape_actual"], "box")
            self.assertEqual(pipeline.profile["system"]["solvent_shape"], "box")

    def test_peptide_receptor_cofactor_masks_and_tleap(self) -> None:
        profile = load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            selected = input_dir / "selected.pdb"
            selected.write_text(
                "\n".join(
                    [
                        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N",
                        "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      3  N   ALA B   2       3.000   0.000   0.000  1.00  0.00           N",
                        "ATOM      4  CA  ALA B   2       4.000   0.000   0.000  1.00  0.00           C",
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            paths = type("Paths", (), {"input": input_dir})()
            manifest = {
                "receptor_chains": "A",
                "peptide_chains": "B",
                "receptor_cofactor_residue_count": 1,
            }
            prepared = peptide_prepare_input_structure(paths, manifest, profile)
            self.assertEqual(prepared["protein_receptor_residue_count"], 1)
            self.assertEqual(prepared["receptor_residue_count"], 2)
            self.assertEqual(prepared["peptide_residue_mask"], ":3-3")
            self.assertTrue((input_dir / "selected_receptor.pdb").exists())
            self.assertTrue((input_dir / "selected_peptide.pdb").exists())

        text = peptide_tleap_text_with_cofactors(
            Path("rec.pdb"),
            Path("pep.pdb"),
            2,
            profile,
            cofactor_files=[Path("GDP.pdb")],
            cofactor_frcmods=[Path("frcmod.phos")],
            cofactor_libs=[Path("GDP.prep")],
        )
        self.assertIn('loadamberprep "GDP.prep"', text)
        self.assertIn('loadamberparams "frcmod.phos"', text)
        self.assertIn('cof1 = loadpdb "GDP.pdb"', text)
        self.assertIn("mol = combine { rec cof1 pep }", text)
        self.assertIn("addIonsRand mol Na+ 2", text)

    def test_peptide_caps_are_allowed_atom_residues(self) -> None:
        profile = load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            (input_dir / "selected_raw.pdb").write_text(
                "\n".join(
                    [
                        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N",
                        "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      3  CH3 ACE B   2       2.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      4  C   ACE B   2       3.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      5  N   ARG B   3       4.000   0.000   0.000  1.00  0.00           N",
                        "ATOM      6  CA  ARG B   3       5.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      7  N   NHE B   4       6.000   0.000   0.000  1.00  0.00           N",
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            paths = type("Paths", (), {"input": input_dir})()
            manifest = {"receptor_chains": "A", "peptide_chains": "B"}

            prepared = peptide_prepare_input_structure(paths, manifest, profile)

            self.assertEqual(prepared["peptide_residue_count"], 3)
            clean = (input_dir / "selected_peptide.pdb").read_text(encoding="utf-8")
            self.assertIn(" ACE B", clean)
            self.assertIn(" NHE B", clean)

    def test_peptide_hetatm_fails_by_default(self) -> None:
        profile = json.loads(json.dumps(load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml")))
        profile["amber_prep"]["nonstandard_policy"] = "fail"
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            (input_dir / "selected_raw.pdb").write_text(peptide_pdb_with_hetatm(), encoding="utf-8")
            paths = type("Paths", (), {"input": input_dir})()
            manifest = {"receptor_chains": "A", "peptide_chains": "B"}

            with self.assertRaises(SystemExit) as raised:
                peptide_prepare_input_structure(paths, manifest, profile)

            self.assertIn("HETATM residues", str(raised.exception))

    def test_peptide_hetatm_strip_writes_clean_input_and_records_drops(self) -> None:
        profile = json.loads(json.dumps(load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml")))
        profile["amber_prep"]["nonstandard_policy"] = "strip"
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            (input_dir / "selected_raw.pdb").write_text(peptide_pdb_with_hetatm(), encoding="utf-8")
            paths = type("Paths", (), {"input": input_dir})()
            manifest = {"receptor_chains": "A", "peptide_chains": "B"}

            prepared = peptide_prepare_input_structure(paths, manifest, profile)

            clean = (input_dir / "selected.pdb").read_text(encoding="utf-8")
            self.assertNotIn("HETATM", clean)
            self.assertIn("ATOM", clean)
            self.assertEqual(prepared["dropped_nonprotein_residue_count"], 1)
            self.assertEqual(prepared["dropped_nonprotein_residues"][0]["resname"], "SO4")
            self.assertTrue((input_dir / "selected_protein.pdb").exists())
            self.assertTrue((input_dir / "selected_receptor.pdb").exists())
            self.assertTrue((input_dir / "selected_peptide.pdb").exists())

    def test_postprocess_sweep_input_rewrites_dielectric_and_salt(self) -> None:
        template = """&gb
  igb=5,
  epsin=4.000,
  epsout=78.500,
  saltcon=0.000,
/
&pb
  istrng=0.000,
  indi=4.000,
  exdi=80.000,
/
"""
        text = mmpbsa_input_with_epsilon(template, 12.0, 0.150)
        self.assertIn("epsin=12.000", text)
        self.assertIn("indi=12.000", text)
        self.assertIn("saltcon=0.150", text)
        self.assertIn("istrng=0.150", text)
        self.assertIn("epsout=78.500", text)
        self.assertEqual(parse_epsilons("20,4,8,4"), [4.0, 8.0, 20.0])

    def test_kras_pilot_variant_position_maps(self) -> None:
        self.assertEqual(PILOT_VARIANTS["WT_PEP_0001"].keep_positions, tuple(range(1, 22)))
        self.assertEqual(PILOT_VARIANTS["del4R_PEP_0002"].keep_positions, (1, 4, 5, *range(6, 19), 21))
        self.assertEqual(PILOT_VARIANTS["core13_PEP_0003"].keep_positions, (1, *range(6, 17), 21))
        self.assertEqual(PILOT_VARIANTS["L8A_PEP_0006"].mutations, {8: "ALA"})
        self.assertEqual(peptide_resname_for_amber("NH2"), "NHE")
        self.assertEqual(peptide_resname_for_amber("CYS"), "CYX")

    def test_kras_pilot_alanine_mutation_keeps_ala_heavy_atoms(self) -> None:
        def atom(position: int, atom_name: str, resname: str = "LEU") -> CifAtom:
            return CifAtom("ATOM", atom_name, resname, "B", str(position - 1), float(position), 0.0, 0.0, 1.0, 0.0, atom_name[0])

        template = {
            position: [atom(position, name, "LEU") for name in ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2")]
            for position in range(1, 22)
        }
        mutated = variant_peptide_atoms(template, PILOT_VARIANTS["L8A_PEP_0006"])
        residue_8 = mutated[7]

        self.assertEqual({item.resname for item in residue_8}, {"ALA"})
        self.assertEqual([item.atom for item in residue_8], ["N", "CA", "C", "O", "CB"])

    def test_kras_6wgn_boltz_cif_summary(self) -> None:
        cif_text = """data_model
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.label_entity_id
_atom_site.auth_asym_id
_atom_site.auth_comp_id
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA 1 1 ? A 0.0 0.0 0.0 1 1 A ALA 10.0 1
ATOM 2 C CA . ALA 1 1 ? A 1.0 0.0 0.0 1 1 A ALA 10.0 1
HETATM 3 C C1 . LIG1 . 1 ? L 2.0 0.0 0.0 1 2 L LIG1 10.0 1
HETATM 4 P PG . GNP . 1 ? G 3.0 0.0 0.0 1 3 G GNP 10.0 1
HETATM 5 MG MG . MG . 1 ? M 4.0 0.0 0.0 1 4 M MG 10.0 1
#
"""
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "model.cif"
            cif.write_text(cif_text, encoding="utf-8")
            summary = kras_boltz_cif_summary(cif)

        self.assertEqual(summary["chains"], ["A", "G", "L", "M"])
        self.assertEqual(summary["protein_residue_count"], 1)
        self.assertEqual(summary["ligand_atom_count"], 1)
        self.assertEqual(summary["gnp_atom_count"], 1)
        self.assertEqual(summary["mg_atom_count"], 1)

    def test_kras_6wgn_boltz_job_config_uses_gnp_mg_reference(self) -> None:
        row = {
            "rank": "987",
            "representative_model": "rank_0987_model_1",
            "smiles": "C1CC1",
            "representative_composite_score": "0.96",
            "representative_ligand_iptm": "0.94",
            "pose_cluster_size": "3",
            "pose_cluster_max_rmsd": "0.98",
        }
        job_id = kras_boltz_job_id_for_row(row)
        config = kras_boltz_job_config(job_id=job_id, row=row)

        self.assertEqual(job_id, "kras6wgn_rank_0987_model_1")
        self.assertEqual(config["nucleotide_state"], "GNP")
        self.assertEqual(config["reference_pdb_id"], "6WGN")
        self.assertEqual(config["ligand_charge"], 0)
        self.assertEqual(config["gnp_charge"], -4)
        self.assertEqual(config["mg_charge"], 2)
        self.assertEqual(config["receptor_cofactor_net_charge"], -2)
        self.assertEqual(config["receptor_cofactor_files"], "input/gnp.mol2;input/mg.pdb")
        self.assertEqual(config["receptor_cofactor_residue_count"], 2)
        self.assertNotIn("GDP", config["source"])

        source_config = kras_boltz_job_config(job_id=job_id, row=row, relative_input_dir="source")
        self.assertEqual(source_config["complex_pdb"], "source/complex.pdb")
        self.assertEqual(source_config["receptor_cofactor_files"], "source/gnp.mol2;source/mg.pdb")

    def test_postprocess_sweep_ridge_residuals_remove_simple_charge_trend(self) -> None:
        rows = [
            {"score": -100.0, "peptide_charge": 1.0, "peptide_residue_count": 10.0, "peptide_sasa_lcpo_angstrom2": 200.0, "native_contacts_mean": 50.0},
            {"score": -200.0, "peptide_charge": 2.0, "peptide_residue_count": 10.0, "peptide_sasa_lcpo_angstrom2": 200.0, "native_contacts_mean": 50.0},
            {"score": -300.0, "peptide_charge": 3.0, "peptide_residue_count": 10.0, "peptide_sasa_lcpo_angstrom2": 200.0, "native_contacts_mean": 50.0},
            {"score": -400.0, "peptide_charge": 4.0, "peptide_residue_count": 10.0, "peptide_sasa_lcpo_angstrom2": 200.0, "native_contacts_mean": 50.0},
        ]
        residuals = ridge_loocv_residuals(rows, "score", ["peptide_charge", "peptide_residue_count", "peptide_sasa_lcpo_angstrom2", "native_contacts_mean"], alpha=0.0)
        self.assertLess(max(abs(value) for value in residuals), 1e-9)

    def test_kras_5xco_pilot_report_writes_outputs(self) -> None:
        if kras_pilot_report_module.yaml is None:
            self.skipTest("PyYAML is not installed in this Python environment")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "mmpbsa_5xco_pilot"
            assay_dir = root / "assay"
            output_dir = root / "correlation"
            assay_dir.mkdir()
            assay_dir.joinpath("assay_0001.yaml").write_text(
                """
assay_group:
  nucleotide_state: GDP
records:
  - ligand_id: PEP_0001
    protein_id: PRO_0001
    mutation_label: WT
    endpoint_relation: "="
    endpoint_type: KD
    standard_value_nM: 10.0
  - ligand_id: PEP_0002
    protein_id: PRO_0001
    mutation_label: del4R
    endpoint_relation: "="
    endpoint_type: KD
    standard_value_nM: 100.0
""",
                encoding="utf-8",
            )

            def write_job(job_id: str, charge: float, gb: float, pb: float) -> None:
                job = run_dir / job_id
                (job / "result").mkdir(parents=True)
                (job / "analysis" / "qc").mkdir(parents=True)
                (job / "analysis" / "mmpbsa").mkdir(parents=True)
                ligand_id = "_".join(job_id.split("_gdp_")[0].split("_")[-2:])
                (job / f"{job_id}.json").write_text(json.dumps({"ligand_id": ligand_id, "peptide_charge": charge}), encoding="utf-8")
                (job / "manifest.json").write_text(json.dumps({"peptide_residue_count": 21, "replicas": ["rep01", "rep02", "rep03"]}), encoding="utf-8")
                summary = {
                    "status": "valid",
                    "trajectory_qc_status": "valid",
                    "mmpbsa_qc_status": "valid",
                    "mmpbsa_frames": 1503.0,
                    "replica_count": 3,
                    "GB_delta_total_kJ_mol": gb,
                    "GB_delta_total_kJ_mol_replica_sem": 1.0,
                    "PB_delta_total_kJ_mol": pb,
                    "PB_delta_total_kJ_mol_replica_sem": 1.5,
                    "GB_dMM_kJ_mol": gb - 10.0,
                    "PB_dMM_kJ_mol": pb - 10.0,
                    "GB_vdw_kJ_mol": -50.0,
                    "PB_vdw_kJ_mol": -50.0,
                    "replica_qc": [
                        {
                            "peptide_bb_rmsd_after_receptor_fit_angstrom": {"mean": 2.0},
                            "receptor_bb_rmsd_angstrom": {"mean": 1.0},
                            "native_contacts": {"rec_pep[native]": {"mean": 100.0}},
                        }
                    ],
                }
                (job / "result" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            write_job("WT_PEP_0001_gdp_mg", 7.0, -450.0, -210.0)
            write_job("WT_PEP_0001_gdp_only", 7.0, -400.0, -180.0)
            write_job("del4R_PEP_0002_gdp_mg", 2.0, -350.0, -120.0)
            write_job("del4R_PEP_0002_gdp_only", 2.0, -330.0, -110.0)

            report = report_kras_5xco_pilot(run_dir, output_dir, assay_dir)

            self.assertEqual(report["pilot_row_count"], 4)
            self.assertTrue((output_dir / "kras_5xco_mg_pilot_results.csv").exists())
            self.assertTrue((output_dir / "kras_5xco_mg_pilot_correlations.csv").exists())
            self.assertTrue((output_dir / "kras_5xco_mg_pilot_decision.json").exists())
            self.assertTrue((output_dir / "report_kras_5xco_mg_pilot.html").exists())

    def test_peptide_amber_hetatm_caps_are_retained_as_atom_records(self) -> None:
        profile = json.loads(json.dumps(load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml")))
        profile["amber_prep"]["nonstandard_policy"] = "fail"
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            (input_dir / "selected_raw.pdb").write_text(peptide_pdb_with_amber_hetatm_caps(), encoding="utf-8")
            paths = type("Paths", (), {"input": input_dir})()
            manifest = {"receptor_chains": "A", "peptide_chains": "B"}

            prepared = peptide_prepare_input_structure(paths, manifest, profile)

            clean = (input_dir / "selected.pdb").read_text(encoding="utf-8")
            peptide = (input_dir / "selected_peptide.pdb").read_text(encoding="utf-8")
            self.assertNotIn("HETATM", clean)
            self.assertIn("ATOM      3  C   ACE B   2", clean)
            self.assertIn("ATOM      4  SG  CYX B   3", clean)
            self.assertIn("ATOM      5  N   NH2 B   4", clean)
            self.assertIn("ACE", peptide)
            self.assertIn("CYX", peptide)
            self.assertIn("NH2", peptide)
            self.assertEqual(prepared["dropped_nonprotein_residue_count"], 0)
            self.assertEqual(prepared["peptide_residue_count"], 3)
            self.assertEqual(prepared["peptide_residue_mask"], ":2-4")
            accepted = prepared["input_residue_findings"]["accepted_hetero_residues"]
            self.assertEqual([item["resname"] for item in accepted], ["ACE", "CYX", "NH2"])

    def test_cli_help(self) -> None:
        if not CLICK_AVAILABLE:
            self.skipTest("click is not installed in this Python environment")
        result = CliRunner().invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("peptide", result.output)
        self.assertIn("ligand", result.output)
        self.assertNotIn("benchmark", result.output)
        peptide_help = CliRunner().invoke(cli, ["peptide", "--help"])
        self.assertEqual(peptide_help.exit_code, 0)
        self.assertIn("merge-replicas", peptide_help.output)
        self.assertIn("sweep-postprocess", peptide_help.output)
        self.assertNotIn("build-kras-5xco-pilot", peptide_help.output)
        self.assertNotIn("report-kras-5xco-pilot", peptide_help.output)
        peptide_run_help = CliRunner().invoke(cli, ["peptide", "run", "--help"])
        self.assertEqual(peptide_run_help.exit_code, 0)
        self.assertIn("--replica-index", peptide_run_help.output)
        peptide_sweep_help = CliRunner().invoke(cli, ["peptide", "sweep-postprocess", "--help"])
        self.assertEqual(peptide_sweep_help.exit_code, 0)
        self.assertIn("--max-workers", peptide_sweep_help.output)
        ligand_help = CliRunner().invoke(cli, ["ligand", "--help"])
        self.assertEqual(ligand_help.exit_code, 0)
        self.assertIn("merge-replicas", ligand_help.output)
        ligand_run_help = CliRunner().invoke(cli, ["ligand", "run", "--help"])
        self.assertEqual(ligand_run_help.exit_code, 0)
        self.assertIn("--replica-index", ligand_run_help.output)

    def test_doctor_applies_runtime_environment_overrides(self) -> None:
        if not CLICK_AVAILABLE:
            self.skipTest("click is not installed in this Python environment")

        completed = type("Completed", (), {"returncode": 0, "stdout": ""})()
        with patch("mmpbsa.cli.subprocess.run", return_value=completed) as run:
            with patch.dict(
                "os.environ",
                {"MAMBA_ENV": "custom_md", "GMXRC": "/opt/gromacs/bin/GMXRC", "GMX_BIN": "gmx_custom"},
                clear=True,
            ):
                result = CliRunner().invoke(cli, ["doctor", "--protocol", str(ROOT / "configs" / "default_15ns.yaml")])

        self.assertEqual(result.exit_code, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(["mamba", "run", "-n", "custom_md", "which", "MMPBSA.py"], commands)
        self.assertIn(["bash", "-lc", "source '/opt/gromacs/bin/GMXRC' && which 'gmx_custom'"], commands)

    def test_validation_sdf_split_and_delta_g(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sdf = Path(tmp) / "ligands.sdf"
            sdf.write_text(
                """lig_a
  test

  0  0  0  0  0  0            999 V2000
M  END
$$$$

lig_b
  test

  0  0  0  0  0  0            999 V2000
M  END
$$$$
""",
                encoding="utf-8",
            )
            records = load_sdf_records(sdf)
            self.assertEqual(sorted(records), ["lig_a", "lig_b"])
            self.assertTrue(records["lig_a"].endswith("$$$$\n"))
        self.assertAlmostEqual(experimental_delta_g_kj_mol(1.0, "uM"), -34.248, places=3)
        self.assertAlmostEqual(experimental_delta_g_kj_mol(1000.0, "nM"), -34.248, places=3)

    def test_validation_correlation_helpers_and_gpu_assignment(self) -> None:
        slope, intercept = linear_fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 0.0)
        self.assertAlmostEqual(pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
        self.assertAlmostEqual(spearman_r([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]), -1.0)
        assignments = assign_jobs_to_gpus(["a", "b", "c", "d", "e"], ["2", "3"], 2)
        self.assertEqual(assignments, {"2": ["a", "c", "e"], "3": ["b", "d"]})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mmpbsa.analysis import add_dmm
from mmpbsa.common import aggregate_replica_values, frame_settings, gmx_runtime, load_profile, profile_with_replica_indices, replica_indices, replica_names, replica_seed_map
from mmpbsa.ligand import ligand_input_format, mol2_total_charge, run_ligand_prepare
from mmpbsa.ligand_amber import tleap_text
from mmpbsa.ligand_pipeline import infer_dielectric_policy, ligand_replica_ante_mmpbsa_command, mmpbsa_input_text, select_interface_waters
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

    def test_postprocess_sweep_ridge_residuals_remove_simple_charge_trend(self) -> None:
        rows = [
            {"score": -100.0, "peptide_charge": 1.0, "peptide_residue_count": 10.0, "peptide_sasa_lcpo_angstrom2": 200.0, "native_contacts_mean": 50.0},
            {"score": -200.0, "peptide_charge": 2.0, "peptide_residue_count": 10.0, "peptide_sasa_lcpo_angstrom2": 200.0, "native_contacts_mean": 50.0},
            {"score": -300.0, "peptide_charge": 3.0, "peptide_residue_count": 10.0, "peptide_sasa_lcpo_angstrom2": 200.0, "native_contacts_mean": 50.0},
            {"score": -400.0, "peptide_charge": 4.0, "peptide_residue_count": 10.0, "peptide_sasa_lcpo_angstrom2": 200.0, "native_contacts_mean": 50.0},
        ]
        residuals = ridge_loocv_residuals(rows, "score", ["peptide_charge", "peptide_residue_count", "peptide_sasa_lcpo_angstrom2", "native_contacts_mean"], alpha=0.0)
        self.assertLess(max(abs(value) for value in residuals), 1e-9)

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

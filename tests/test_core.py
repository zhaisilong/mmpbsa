from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmpbsa.benchmark import assign_jobs_to_gpus, experimental_delta_g_kj_mol, linear_fit, load_sdf_records, pearson_r, spearman_r
from mmpbsa.common import frame_settings, gmx_runtime, load_profile
from mmpbsa.ligand import ligand_input_format, mol2_total_charge, run_ligand_prepare
from mmpbsa.ligand_amber import tleap_text
from mmpbsa.ligand_pipeline import infer_dielectric_policy, ligand_replica_ante_mmpbsa_command, mmpbsa_input_text, select_interface_waters
from mmpbsa.peptide_pipeline import mmpbsa_input_text as peptide_mmpbsa_input_text
from mmpbsa.peptide_pipeline import peptide_dielectric_policy
from mmpbsa.runner import DoneFileRunner, JobContext, discover_job_contexts

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

    def test_frame_settings_peptide_crystal_replicas(self) -> None:
        settings = frame_settings(load_profile(ROOT / "configs" / "peptide_crystal_3x5ns.yaml"))
        self.assertEqual(settings["startframe"], 151)
        self.assertEqual(settings["frames_per_replica"], 101)
        self.assertEqual(settings["replica_count"], 3)
        self.assertEqual(settings["expected_mmpbsa_frames"], 303)

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

    def test_cli_help(self) -> None:
        if not CLICK_AVAILABLE:
            self.skipTest("click is not installed in this Python environment")
        result = CliRunner().invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("peptide", result.output)
        self.assertIn("ligand", result.output)
        self.assertIn("benchmark", result.output)

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

    def test_benchmark_sdf_split_and_delta_g(self) -> None:
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

    def test_benchmark_correlation_helpers_and_gpu_assignment(self) -> None:
        slope, intercept = linear_fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 0.0)
        self.assertAlmostEqual(pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
        self.assertAlmostEqual(spearman_r([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]), -1.0)
        assignments = assign_jobs_to_gpus(["a", "b", "c", "d", "e"], ["2", "3"], 2)
        self.assertEqual(assignments, {"2": ["a", "c", "e"], "3": ["b", "d"]})


if __name__ == "__main__":
    unittest.main()

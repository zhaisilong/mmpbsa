from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mmpbsa.backends.ambertools import (
    AmberToolsScoringBackend,
    AmberToolsScoringRequest,
    create_ambertools_scoring_backend,
    parse_mmpbsa_models_csv,
    parse_numeric_csv,
)
from mmpbsa.backends.cpptraj import CpptrajQcBackend, CpptrajQcRequest
from mmpbsa.backends.display import DisplayAssetBackend, DisplayAssetRequest
from mmpbsa.backends.errors import BackendConfigurationError, BackendOutputError
from mmpbsa.backends.gromacs import GromacsMdBackend, GromacsMdRequest
from mmpbsa.backends.trajectory import (
    TrajectoryPreparationBackend,
    TrajectoryPreparationRequest,
)
from mmpbsa.backends.versioning import (
    MajorVersionPolicy,
    ToolConfig,
    parse_version,
)
from mmpbsa.utils import sha256_digest


class AdapterExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fixture = Path(__file__).parent / "fixtures" / "fake_tool.py"
        self.fake = self.root / "fake-tool"
        shutil.copy2(fixture, self.fake)
        self.fake.chmod(0o755)
        self.tool = ToolConfig(
            executable=str(self.fake),
            version_arguments=("--version",),
            major_policy=MajorVersionPolicy(accepted=frozenset({2024})),
        )
        amber_bin = self.root / "amber" / "bin"
        amber_bin.mkdir(parents=True)
        sander = amber_bin / "sander"
        shutil.copy2(fixture, sander)
        sander.chmod(0o755)
        amber_environment = {"AMBERHOME": str(amber_bin.parent)}
        self.scoring_tool = replace(
            self.tool,
            environment=amber_environment,
        )
        self.sander_tool = ToolConfig(
            executable=str(sander),
            version_arguments=("--version",),
            major_policy=MajorVersionPolicy(accepted=frozenset({2024})),
            environment=amber_environment,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def input(self, name: str) -> Path:
        path = self.root / "inputs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")
        return path

    def test_version_parser_and_explicit_major_policy(self) -> None:
        self.assertEqual(parse_version("GROMACS version 2024.3"), (2024, 3))
        self.assertTrue(self.tool.major_policy.accepts(2024))
        self.assertFalse(self.tool.major_policy.accepts(2023))

    def test_gromacs_commands_execute_against_fake_executable(self) -> None:
        output = self.root / "md"
        receipt = GromacsMdBackend(self.tool).run(
            GromacsMdRequest(
                stage_id="production",
                mdp_text="integrator = md\nnsteps = 100",
                coordinates=self.input("system.gro"),
                topology=self.input("topol.top"),
                output_dir=output,
                expected_suffixes=(
                    "tpr",
                    "log",
                    "edr",
                    "gro",
                    "xtc",
                    "cpt",
                ),
            )
        )

        self.assertEqual(len(receipt.commands), 2)
        self.assertEqual(receipt.commands[0].argv[1], "grompp")
        self.assertEqual(receipt.commands[1].argv[1], "mdrun")
        self.assertTrue(all(path.is_file() for path in receipt.artifacts.values()))

        resumed = GromacsMdBackend(self.tool).run(
            replace(
                GromacsMdRequest(
                    stage_id="production",
                    mdp_text="integrator = md\nnsteps = 100",
                    coordinates=self.root / "inputs" / "system.gro",
                    topology=self.root / "inputs" / "topol.top",
                    output_dir=output,
                ),
                checkpoint_input=output / "production.cpt",
                append=True,
            ),
            resume=True,
        )
        self.assertEqual(len(resumed.commands), 1)
        self.assertIn("-cpi", resumed.commands[0].argv)
        self.assertIn("-append", resumed.commands[0].argv)

        with self.assertRaises(BackendConfigurationError):
            GromacsMdBackend(self.tool).build_commands(
                GromacsMdRequest(
                    stage_id="production",
                    mdp_text="integrator = md\nnsteps = 100",
                    coordinates=self.root / "inputs" / "system.gro",
                    topology=self.root / "inputs" / "topol.top",
                    output_dir=output,
                    mdrun_arguments=("-maxh", "0.01"),
                )
            )

    def test_gromacs_unconstrained_gate_precedes_mdrun(self) -> None:
        topology = self.input("flexible.top")
        topology.write_text("[ bonds ]\n1 2 1 0.1 1000\n", encoding="utf-8")
        request = GromacsMdRequest(
            stage_id="em",
            mdp_text="integrator = steep\ndefine = -DFLEXIBLE",
            coordinates=self.input("system.gro"),
            topology=topology,
            output_dir=self.root / "em-flexible",
            require_unconstrained_topology=True,
        )

        receipt = GromacsMdBackend(self.tool).run(request)

        self.assertIn("-pp", receipt.commands[0].argv)
        self.assertEqual(receipt.commands[1].argv[1], "mdrun")

        constrained = self.input("rigid.top")
        constrained.write_text(
            "[ settles ]\n1 1 0.09572 0.15136\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(BackendOutputError, "retains.*settles"):
            GromacsMdBackend(self.tool).run(
                replace(
                    request,
                    topology=constrained,
                    output_dir=self.root / "em-rigid",
                )
            )
        self.assertFalse((self.root / "em-rigid" / "em.log").exists())

    def test_trajectory_pipeline_materializes_both_tool_outputs(self) -> None:
        output = self.root / "trajectory"
        backend = TrajectoryPreparationBackend(
            gromacs=self.tool,
            cpptraj=self.tool,
        )
        receipt = backend.run(
            TrajectoryPreparationRequest(
                source_tpr=self.input("md.tpr"),
                source_trajectory=self.input("md.xtc"),
                index=self.input("complex.ndx"),
                complex_topology=self.input("complex.prmtop"),
                output_dir=output,
                group_name="Complex",
                start_frame=1,
                end_frame=3,
                stride=1,
                expected_frames=3,
            )
        )

        self.assertEqual(len(receipt.commands), 2)
        self.assertTrue(receipt.artifacts["prepared_trajectory"].is_file())
        self.assertTrue(receipt.artifacts["centered_trajectory"].is_file())

    def test_mmpbsa_outputs_are_parsed_without_project_policy(self) -> None:
        output = self.root / "scoring"
        receipt = AmberToolsScoringBackend(
            self.scoring_tool,
            self.sander_tool,
        ).run(
            AmberToolsScoringRequest(
                model_id="gb",
                control_text="&general\n  interval=1,\n/\n&gb\n  igb=5,\n/\n",
                complex_topology=self.input("complex.prmtop"),
                receptor_topology=self.input("receptor.prmtop"),
                binder_topology=self.input("binder.prmtop"),
                trajectory=self.input("prepared.nc"),
                output_dir=output,
                expected_frames=2,
            )
        )

        summary = receipt.parsed["summary"]["rows"]
        self.assertEqual(summary[0]["label"], "DELTA TOTAL")
        self.assertEqual(
            [
                (
                    row["model_id"],
                    row["component"],
                    row["label"],
                    row["average"],
                )
                for row in summary
            ],
            [
                (
                    "gb",
                    "delta",
                    "DELTA TOTAL",
                    -10.0,
                ),
            ],
        )
        self.assertEqual(receipt.parsed["model_id"], "gb")
        self.assertEqual(receipt.parsed["frame_count"], 2)
        self.assertEqual(
            receipt.parsed["frame_ids"],
            {
                "count": 2,
                "first": 1,
                "last": 2,
                "sha256": sha256_digest((1, 2)),
            },
        )
        self.assertTrue(receipt.artifacts["delta_output"].is_file())
        self.assertNotIn("-use-mdins", receipt.commands[0].argv)

    def test_mmpbsa_factory_binds_probe_to_effective_sander(self) -> None:
        backend = create_ambertools_scoring_backend(
            {
                "environment": {
                    "AMBERHOME": str(self.root / "amber"),
                    "SHARED": "inherited",
                },
                "tool": {"executable": str(self.fake)},
                "sander": {"executable": "sander"},
            }
        )

        self.assertEqual(
            backend.sander.executable,
            str((self.root / "amber/bin/sander").resolve()),
        )
        self.assertEqual(backend.tool.environment["SHARED"], "inherited")
        self.assertEqual(backend.sander.environment["SHARED"], "inherited")
        with self.assertRaisesRegex(
            BackendConfigurationError,
            "differs from AMBERHOME/bin/sander",
        ):
            create_ambertools_scoring_backend(
                {
                    "environment": {
                        "AMBERHOME": str(self.root / "amber"),
                    },
                    "tool": {"executable": str(self.fake)},
                    "sander": {"executable": str(self.fake)},
                }
            )

    def test_numeric_csv_single_table_remains_backward_compatible(self) -> None:
        source = self.root / "single-table.csv"
        source.write_text(
            "# generated table\n"
            "Frame,DELTA TOTAL\n"
            "1,-9.0\n"
            "2,-11.0\n",
            encoding="utf-8",
        )

        parsed = parse_numeric_csv(source)

        self.assertEqual(set(parsed), {"columns", "rows"})
        self.assertEqual(parsed["columns"], ("Frame", "DELTA TOTAL"))
        self.assertEqual(parsed["rows"][-1]["DELTA TOTAL"], -11.0)

    def test_numeric_csv_selects_one_strict_mmpbsa_named_table(self) -> None:
        source = self.root / "multi-table.csv"
        source.write_text(
            "GENERALIZED BORN:\n"
            "Complex Energy Terms\n"
            "Frame #,TOTAL\n"
            "1,-100.0\n"
            "2,-101.0\n"
            "\n"
            "DELTA Energy Terms\n"
            "Frame #,DELTA TOTAL\n"
            "1,-9.0\n"
            "2,-11.0\n"
            "\n"
            "POISSON BOLTZMANN:\n"
            "DELTA Energy Terms\n"
            "Frame #,DELTA TOTAL\n"
            "1,-8.0\n",
            encoding="utf-8",
        )

        parsed = parse_numeric_csv(
            source,
            selector={
                "section": "GENERALIZED BORN",
                "subsection": "DELTA Energy Terms",
            },
        )

        self.assertEqual(parsed["columns"], ("Frame #", "DELTA TOTAL"))
        self.assertEqual(
            [row["DELTA TOTAL"] for row in parsed["rows"]],
            [-9.0, -11.0],
        )
        self.assertEqual(
            parsed["selected"],
            {
                "section": "GENERALIZED BORN",
                "subsection": "DELTA Energy Terms",
            },
        )
        self.assertEqual(
            [
                (table["section"], table["subsection"])
                for table in parsed["tables"]
            ],
            [
                ("GENERALIZED BORN", "Complex Energy Terms"),
                ("GENERALIZED BORN", "DELTA Energy Terms"),
                ("POISSON BOLTZMANN", "DELTA Energy Terms"),
            ],
        )
        with self.assertRaises(BackendOutputError):
            parse_numeric_csv(source)
        with self.assertRaises(BackendOutputError):
            parse_numeric_csv(
                source,
                selector={
                    "section": "GENERALIZED BORN",
                    "subsection": "Missing Energy Terms",
                },
            )

        malformed = self.root / "malformed-multi-table.csv"
        malformed.write_text(
            "GENERALIZED BORN:\n"
            "Frame #,DELTA TOTAL\n"
            "1,-9.0\n",
            encoding="utf-8",
        )
        with self.assertRaises(BackendOutputError):
            parse_numeric_csv(
                malformed,
                selector={
                    "section": "GENERALIZED BORN",
                    "subsection": "DELTA Energy Terms",
                },
            )

    def test_mmpbsa_native_inputs_are_model_specific_and_persisted(self) -> None:
        output = self.root / "scoring-native"
        native_inputs = {"main": "&cntrl\n  igb=5,\n/\n\n"}

        request = AmberToolsScoringRequest(
            model_id="gb",
            control_text="&general\n  use_sander=1,\n/\n&gb\n  igb=5,\n/\n",
            complex_topology=self.input("native-complex.prmtop"),
            receptor_topology=self.input("native-receptor.prmtop"),
            binder_topology=self.input("native-binder.prmtop"),
            trajectory=self.input("native-prepared.nc"),
            output_dir=output,
            native_inputs=native_inputs,
            expected_frames=2,
        )
        backend = AmberToolsScoringBackend(
            self.scoring_tool,
            self.sander_tool,
        )

        receipt = backend.run(request)

        self.assertEqual(receipt.commands[0].argv.count("-use-mdins"), 1)
        generated = {
            str(item.relative_path): item.content
            for item in receipt.generated_inputs
        }
        self.assertEqual(
            set(generated),
            {
                "mmpbsa.in",
                "_MMPBSA_gb.mdin",
                "inputs/gb.mdin",
            },
        )
        for key, working, persisted, artifact_role in (
            (
                "main",
                "_MMPBSA_gb.mdin",
                "inputs/gb.mdin",
                "native_main_input",
            ),
        ):
            expected = native_inputs[key].rstrip("\n") + "\n"
            self.assertEqual(generated[working], expected)
            self.assertEqual(generated[persisted], expected)
            self.assertFalse((output / working).exists())
            self.assertEqual(
                receipt.artifacts[artifact_role],
                output / persisted,
            )
            self.assertEqual(
                receipt.artifacts[artifact_role].read_text(encoding="utf-8"),
                expected,
            )
        self.assertEqual(receipt.parsed["model_id"], "gb")

    def test_mmpbsa_native_inputs_require_model_specific_nonempty_keys(self) -> None:
        backend = AmberToolsScoringBackend(
            self.scoring_tool,
            self.sander_tool,
        )
        request = AmberToolsScoringRequest(
            model_id="pb",
            control_text="&general\n/\n&pb\n  istrng=0.15,\n/\n",
            complex_topology=self.input("strict-complex.prmtop"),
            receptor_topology=self.input("strict-receptor.prmtop"),
            binder_topology=self.input("strict-binder.prmtop"),
            trajectory=self.input("strict-prepared.nc"),
            output_dir=self.root / "strict-scoring",
        )
        invalid = {
            "missing": {"complex_receptor": "pb"},
            "unexpected": {
                "complex_receptor": "pb",
                "ligand": "pb2",
                "other": "other",
            },
            "empty": {"complex_receptor": " ", "ligand": "pb2"},
            "nul": {"complex_receptor": "pb", "ligand": "pb2\x00"},
            "not_mapping": ("complex_receptor", "ligand"),
        }

        for label, native_inputs in invalid.items():
            with self.subTest(label=label):
                with self.assertRaises(BackendConfigurationError):
                    backend.build_commands(
                        replace(request, native_inputs=native_inputs)
                    )

    def test_mmpbsa_model_parser_requires_four_tables_and_matching_frames(self) -> None:
        source = self.root / "strict-model.csv"
        source.write_text(
            "GENERALIZED BORN:\n"
            "Complex Energy Terms\nFrame #,TOTAL\n1,-1\n2,-2\n"
            "Receptor Energy Terms\nFrame #,TOTAL\n1,-1\n2,-2\n"
            "Ligand Energy Terms\nFrame #,TOTAL\n1,-1\n2,-2\n"
            "DELTA Energy Terms\nFrame #,DELTA TOTAL\n1,-1\n2,-2\n",
            encoding="utf-8",
        )
        parsed = parse_mmpbsa_models_csv(
            source,
            model_id="gb",
            expected_frames=2,
        )
        self.assertEqual(parsed["frame_ids"], (1, 2))
        self.assertEqual(set(parsed["components"]), {"complex", "receptor", "ligand", "delta"})
        with self.assertRaises(BackendOutputError, msg="exact frame count"):
            parse_mmpbsa_models_csv(source, model_id="gb", expected_frames=3)

        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "Ligand Energy Terms\nFrame #,TOTAL\n1,-1\n2,-2",
                "Ligand Energy Terms\nFrame #,TOTAL\n1,-1\n3,-2",
            ),
            encoding="utf-8",
        )
        with self.assertRaises(BackendOutputError, msg="component frame IDs"):
            parse_mmpbsa_models_csv(source, model_id="gb", expected_frames=2)

        source.write_text(
            "GENERALIZED BORN:\n"
            "Complex Energy Terms\nFrame #,TOTAL\n0,-1\n1,-2\n"
            "Receptor Energy Terms\nFrame #,TOTAL\n0,-1\n1,-2\n"
            "Ligand Energy Terms\nFrame #,TOTAL\n0,-1\n1,-2\n"
            "DELTA Energy Terms\nFrame #,DELTA TOTAL\n0,-1\n1,-2\n",
            encoding="utf-8",
        )
        zero_based = parse_mmpbsa_models_csv(
            source,
            model_id="gb",
            expected_frames=2,
        )
        self.assertEqual(zero_based["frame_ids"], (1, 2))
        self.assertEqual(
            zero_based["frame_id_normalization"],
            {
                "source_first": 0,
                "source_last": 1,
                "offset": 1,
                "canonical_first": 1,
                "canonical_last": 2,
            },
        )
        self.assertEqual(
            tuple(
                int(row["Frame #"])
                for row in zero_based["components"]["delta"]["rows"]
            ),
            (1, 2),
        )

        source.write_text(
            "GENERALIZED BORN:\n"
            "Complex Energy Terms\nFrame #,TOTAL\n2,-1\n3,-2\n"
            "Receptor Energy Terms\nFrame #,TOTAL\n2,-1\n3,-2\n"
            "Ligand Energy Terms\nFrame #,TOTAL\n2,-1\n3,-2\n"
            "DELTA Energy Terms\nFrame #,DELTA TOTAL\n2,-1\n3,-2\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(BackendOutputError, "0..1 or 1..2"):
            parse_mmpbsa_models_csv(source, model_id="gb", expected_frames=2)

    def test_mmpbsa_rejects_backend_managed_extra_arguments(self) -> None:
        backend = AmberToolsScoringBackend(
            self.scoring_tool,
            self.sander_tool,
        )
        request = AmberToolsScoringRequest(
            model_id="gb",
            control_text="&general\n/\n&gb\n  igb=5,\n/\n",
            complex_topology=self.input("reserved-complex.prmtop"),
            receptor_topology=self.input("reserved-receptor.prmtop"),
            binder_topology=self.input("reserved-binder.prmtop"),
            trajectory=self.input("reserved-prepared.nc"),
            output_dir=self.root / "reserved-scoring",
        )

        for argument in (
            "--overwrite",
            "-prefix=other",
            "-make-mdins",
            "-use-mdins",
            "-rewrite-output",
            "--clean",
        ):
            with self.subTest(argument=argument):
                with self.assertRaises(BackendConfigurationError):
                    backend.build_commands(
                        replace(request, extra_arguments=(argument,))
                    )

    def test_cpptraj_qc_requires_and_parses_declared_tables(self) -> None:
        output = self.root / "qc"
        receipt = CpptrajQcBackend(self.tool).run(
            CpptrajQcRequest(
                input_text='writedata "rmsd.dat"\nrun\n',
                declared_inputs=(self.input("prepared.nc"),),
                output_tables={"rmsd": Path("rmsd.dat")},
                output_dir=output,
                expected_frames=2,
            )
        )

        self.assertEqual(
            receipt.parsed["rmsd"]["columns"],
            ("Frame", "RMSD"),
        )
        self.assertEqual(receipt.parsed["rmsd"]["rows"][-1][-1], 2.0)

    def test_display_assets_use_explicit_components_and_replicas(self) -> None:
        output = self.root / "display"
        receipt = DisplayAssetBackend(self.tool).run(
            DisplayAssetRequest(
                complex_topology=self.input("complex-for-display.prmtop"),
                trajectories={
                    "rep01": self.input("rep01.nc"),
                    "rep02": self.input("rep02.nc"),
                },
                component_atom_indices={
                    "receptor": (1, 2),
                    "binder": (3,),
                },
                output_dir=output,
                default_replica="rep02",
                reference_trajectories={
                    "rep01": self.input("rep01-production.xtc"),
                    "rep02": self.input("rep02-production.xtc"),
                },
                reference_autoimage_anchor_atom_indices=(3,),
                reference_autoimage_fixed_atom_indices=(1, 2),
                reference_alignment_atom_indices=(1, 2),
                ligand_rmsd_atom_indices=(3,),
            )
        )

        self.assertTrue(receipt.artifacts["structure.rep01"].is_file())
        reference_input = receipt.artifacts["reference_input.rep01"].read_text(
            encoding="utf-8"
        )
        self.assertIn('rep01-production.xtc" 1 1', reference_input)
        self.assertIn("autoimage anchor @3 fixed @1,2", reference_input)
        cpptraj_input = receipt.artifacts["cpptraj_input.rep01"].read_text(
            encoding="utf-8"
        )
        self.assertIn('trajout "', cpptraj_input)
        self.assertIn('rep01.pdb" pdb', cpptraj_input)
        self.assertNotIn('rep01.pdb" pdb multi', cpptraj_input)
        self.assertIn(
            'rep01_equilibrated_reference.pdb" 1 name equilibrated_pose',
            cpptraj_input,
        )
        self.assertIn(
            "rms equilibrated_pose_fit @1,2 ref equilibrated_pose mass",
            cpptraj_input,
        )
        self.assertIn(
            "rms ligand_equilibrated @3 ref equilibrated_pose nofit mass out",
            cpptraj_input,
        )
        rep01_pymol = receipt.artifacts["pymol_script.rep01"].read_text(
            encoding="utf-8"
        )
        rep02_pymol = receipt.artifacts["pymol_script.rep02"].read_text(
            encoding="utf-8"
        )
        self.assertIn('load "structures/rep01.pdb", replica_rep01', rep01_pymol)
        self.assertIn("index 1+2", rep01_pymol)
        self.assertNotIn("replica_rep02", rep01_pymol)
        self.assertIn('load "structures/rep02.pdb", replica_rep02', rep02_pymol)
        self.assertNotIn("replica_rep01", rep02_pymol)
        manifest = json.loads(
            receipt.artifacts["asset_manifest"].read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], "mmpbsa.display_assets.v4")
        self.assertTrue(receipt.artifacts["trajectory.rep01"].is_file())
        self.assertIn(
            'load_traj "trajectories/rep01.xtc", replica_rep01, state=1',
            rep01_pymol,
        )
        self.assertEqual(
            manifest["trajectories"],
            {"rep01": "trajectories/rep01.xtc", "rep02": "trajectories/rep02.xtc"},
        )
        self.assertEqual(
            manifest["pymol_scripts"],
            {"rep01": "load_rep01.pml", "rep02": "load_rep02.pml"},
        )
        self.assertEqual(
            manifest["equilibrated_pose_rmsd"]["ligand_atom_indices"],
            [3],
        )
        self.assertEqual(
            manifest["equilibrated_pose_rmsd"]["reference_semantics"],
            "post_npt_production_start",
        )
        self.assertEqual(
            manifest["equilibrated_pose_rmsd"]["reference_preprocessing"],
            "autoimage_ligand_anchor_receptor_fixed",
        )
        self.assertEqual(
            receipt.parsed["ligand_equilibrated_rmsd"]["rep01"]["columns"],
            ("Frame", "ligand_equilibrated"),
        )


if __name__ == "__main__":
    unittest.main()

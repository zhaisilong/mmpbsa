from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import warnings
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mmpbsa.backends.api import CommandRecord, CommandSpec
from mmpbsa.backends.errors import BackendConfigurationError, BackendOutputError
from mmpbsa.backends.trajectory_dispatch import execute_trajectory_adapter
from mmpbsa.backends.trajectory_v3 import (
    MdtrajTrajectoryV3Processor,
    TrajectoryPreparationV3Backend,
    TrajectoryPreparationV3Request,
    TrajectoryV3Audit,
    TrajectoryV3Projection,
    _load_parmed_topologies,
    _validate_coordinate_topology_order,
    _validate_topology_projection,
    create_trajectory_v3_backend,
)
from mmpbsa.backends.trajectory_v3_numeric import (
    box_metric,
    box_tuple,
    box_volume,
    float_digest,
)
from mmpbsa.backends.versioning import MajorVersionPolicy, ToolConfig
from mmpbsa.core import (
    EntitySelection,
    PreparedTrajectoryView,
    StageId,
    StageInput,
    TaskKey,
)

_HAS_MDTRAJ = importlib.util.find_spec("mdtraj") is not None
_HAS_PARMED = importlib.util.find_spec("parmed") is not None
_BOX = (3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0)


def test_trajectory_v3_numeric_helpers_preserve_audit_values() -> None:
    import hashlib

    box = box_tuple(((2, 0, 0), (1, 3, 0), (0, 0, 4)))
    assert box == (2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 4.0)
    assert box_volume(box) == 24.0
    assert box_metric(box) == (4.0, 2.0, 0.0, 2.0, 10.0, 0.0, 0.0, 0.0, 16.0)
    assert float_digest((0.0, 1.2345678)) == hashlib.sha256(b"0.000000\n1.234568").hexdigest()


@unittest.skipUnless(_HAS_MDTRAJ and _HAS_PARMED, "MDTraj/ParmEd required")
def test_gro_inferred_elements_do_not_override_chemical_topology(tmp_path: Path) -> None:
    import mdtraj as md
    import parmed as pmd

    gro = tmp_path / "metal.gro"
    gro.write_text(
        "ambiguous element names\n2\n"
        + "".join(
            f"{1:5d}{'HEM':<5}{name:>5}{serial:5d}{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
            for serial, name in enumerate(("FE", "CL1"), 1)
        )
        + "   3.00000   3.00000   3.00000\n",
        encoding="utf-8",
    )
    coordinates = md.load_topology(str(gro))
    chemistry = pmd.Structure()
    for name, element, mass in (("FE", 26, 55.0), ("CL1", 17, 35.45)):
        chemistry.add_atom(pmd.Atom(name=name, atomic_number=element, mass=mass), "HEM", 1)
    # The reader inferred F/C, but GRO itself did not encode those elements.
    assert tuple(a.element.number for a in coordinates.atoms) == (9, 6)
    _validate_coordinate_topology_order(coordinates, chemistry, check_elements=False)
    with unittest.TestCase().assertRaises(BackendConfigurationError):
        _validate_coordinate_topology_order(coordinates, chemistry)
    coordinates.atom(0).name = "CL1"
    with unittest.TestCase().assertRaises(BackendConfigurationError):
        _validate_coordinate_topology_order(coordinates, chemistry, check_elements=False)


def _audit(
    *,
    atom_count: int = 3,
    times: tuple[float, ...] = (10.0, 30.0),
    boxes: tuple[tuple[float, ...], ...] | None = None,
) -> TrajectoryV3Audit:
    selected_boxes = boxes if boxes is not None else tuple(_BOX for _ in times)
    return TrajectoryV3Audit(
        atom_count=atom_count,
        frame_count=len(times),
        times_ps=times,
        boxes_nm=selected_boxes,
        steps=tuple(index * 5_000 for index in range(1, len(times) + 1)),
    )


def _full_audit() -> TrajectoryV3Audit:
    times = (0.0, 10.0, 20.0, 30.0, 40.0)
    return TrajectoryV3Audit(
        atom_count=3,
        frame_count=len(times),
        times_ps=times,
        boxes_nm=tuple(_BOX for _ in times),
        steps=tuple(index * 5_000 for index in range(len(times))),
    )


class _FakeProcessor:
    def __init__(
        self,
        *,
        source_audit: TrajectoryV3Audit | None = None,
        centered_times: tuple[float, ...] | None = None,
        centered_boxes: tuple[tuple[float, ...], ...] | None = None,
        reference_times: tuple[float, ...] | None = None,
    ) -> None:
        self.source_audit = source_audit
        self.centered_times = centered_times
        self.centered_boxes = centered_boxes
        self.reference_times = reference_times
        self.project_calls: list[tuple[TrajectoryPreparationV3Request, Path, int]] = []
        self.inspect_calls: list[tuple[Path, Path, int]] = []

    def reconstruct(self, source, destination, topology, *, chunk_frames):
        destination.write_bytes(b"fake whole XTC\n")
        return {"status": "passed", "revision": "whole_molecules_v1"}

    def geometry(self, trajectory, topology, *, chunk_frames, require_whole=True):
        return {"status": "passed", "revision": "whole_molecules_v1"}

    def aligned_geometry(self, physical, fitted, topology, *, chunk_frames):
        return {"status": "passed", "revision": "whole_molecules_v1"}

    def project(
        self,
        request: TrajectoryPreparationV3Request,
        destination: Path,
        *,
        chunk_frames: int,
    ) -> TrajectoryV3Projection:
        self.project_calls.append((request, destination, chunk_frames))
        destination.write_bytes(b"fake projected XTC\n")
        selected = self.source_audit or _full_audit()
        return TrajectoryV3Projection(
            source_atom_count=8,
            source_frames_scanned=selected.frame_count,
            atom_order_signature="a" * 64,
            projection_sha256="b" * 64,
            selected=selected,
        )

    def inspect(
        self,
        trajectory: Path,
        topology: Path,
        *,
        chunk_frames: int,
    ) -> TrajectoryV3Audit:
        self.inspect_calls.append((trajectory, topology, chunk_frames))
        if trajectory.name == "whole.xtc":
            return self.source_audit or _full_audit()
        if trajectory.name == "prepared_reference.nc":
            return _audit(times=self.reference_times or (0.0,))
        if trajectory.name == "centered.xtc" and self.centered_times is not None:
            return _audit(times=self.centered_times)
        if trajectory.name == "centered.xtc" and self.centered_boxes is not None:
            return _audit(boxes=self.centered_boxes)
        return _audit()


class _FakeCpptrajRunner:
    def __init__(self) -> None:
        self.commands: list[CommandSpec] = []

    def run(self, command: CommandSpec) -> CommandRecord:
        self.commands.append(command)
        if "-V" in command.argv:
            return CommandRecord(
                argv=command.argv,
                cwd=command.cwd,
                returncode=0,
                stdout="cpptraj 2024.2\n",
                stderr="",
            )
        script = Path(command.argv[command.argv.index("-i") + 1])
        text = script.read_text(encoding="utf-8")
        for name in ("prepared_reference.nc", "prepared.nc", "centered.xtc"):
            path = command.cwd / name
            if str(path) not in text:
                continue
            path.write_bytes(f"fake {name}\n".encode())
        return CommandRecord(
            argv=command.argv,
            cwd=command.cwd,
            returncode=0,
            stdout="",
            stderr="",
        )


class TrajectoryV3AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tool = ToolConfig(
            executable="fake-cpptraj",
            version_arguments=("-V",),
            major_policy=MajorVersionPolicy(accepted=frozenset({2024})),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _file(self, name: str, content: bytes = b"fixture\n") -> Path:
        path = self.root / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _request(self, output_name: str = "trajectory") -> TrajectoryPreparationV3Request:
        return TrajectoryPreparationV3Request(
            full_trajectory=self._file("production.xtc"),
            full_system_topology=self._file("solvated.prmtop"),
            full_system_coordinates=self._file("solvated.gro"),
            dry_complex_topology=self._file("complex.prmtop"),
            source_md_atom_indices=(3, 1, 7),
            output_dir=self.root / output_name,
            start_frame=2,
            end_frame=4,
            stride=2,
            expected_atom_order_signature="a" * 64,
            residue_crosswalk=((1, 1, "MOL"),),
            source_first_time_ps=0.0,
            source_interval_ps=10.0,
            source_interval_steps=5_000,
            expected_source_frames=5,
            expected_frames=2,
            autoimage_anchor_mask=":3",
            autoimage_fixed_mask=":1-2",
            fit_mask="@1-2",
        )

    @unittest.skipUnless(_HAS_PARMED, "ParmEd is unavailable")
    def test_parmed_pair_warning_filter_is_exact_and_local(self) -> None:
        import parmed as pmd
        from parmed.gromacs.gromacstop import GromacsWarning

        request = self._request()

        def load_file(path: str) -> str:
            warnings.warn(
                "pairs funct != 1; unknown functional",
                GromacsWarning,
                stacklevel=2,
            )
            warnings.warn(
                "pairs funct != 1; unknown functional (different)",
                GromacsWarning,
                stacklevel=2,
            )
            warnings.warn(
                "Pairs funct != 1; unknown functional",
                GromacsWarning,
                stacklevel=2,
            )
            warnings.warn(
                "pairs funct != 1; unknown functional",
                UserWarning,
                stacklevel=2,
            )
            warnings.warn("another topology warning", UserWarning, stacklevel=2)
            return Path(path).name

        with (
            patch("mmpbsa.backends.trajectory_v3.probe_parmed_identity"),
            patch.object(pmd, "load_file", side_effect=load_file),
        ):
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                full, dry = _load_parmed_topologies(request)
                warnings.warn(
                    "pairs funct != 1; unknown functional",
                    GromacsWarning,
                    stacklevel=2,
                )

        self.assertEqual((full, dry), ("solvated.prmtop", "complex.prmtop"))
        self.assertEqual(
            [str(item.message) for item in observed],
            [
                "pairs funct != 1; unknown functional (different)",
                "Pairs funct != 1; unknown functional",
                "pairs funct != 1; unknown functional",
                "another topology warning",
                "pairs funct != 1; unknown functional (different)",
                "Pairs funct != 1; unknown functional",
                "pairs funct != 1; unknown functional",
                "another topology warning",
                "pairs funct != 1; unknown functional",
            ],
        )

    def test_contract_is_engine_neutral_and_preserves_declared_projection_order(self) -> None:
        request = self._request()
        backend = TrajectoryPreparationV3Backend(self.tool, chunk_frames=37)

        generated, commands, artifacts = backend.build_commands(request)

        self.assertEqual(commands[0].argv, ("fake-cpptraj", "-i", str(artifacts["cpptraj_input"])))
        self.assertNotIn("gmx", " ".join(commands[0].argv).lower())
        self.assertNotIn("tpr", generated[0].content.lower())
        self.assertIn('trajin "' + str(artifacts["whole_trajectory"]), generated[0].content)
        self.assertIn(
            f'trajin "{artifacts["whole_trajectory"]}" 1 last 1',
            generated[0].content,
        )
        self.assertIn("autoimage anchor :3 fixed :1-2", generated[0].content)
        self.assertNotIn("rms fit", generated[0].content)
        self.assertIn("rms fit first @1-2", generated[1].content)
        self.assertIn(
            f'trajout "{artifacts["prepared_reference_trajectory"]}" netcdf onlyframes 1',
            generated[0].content,
        )
        self.assertIn(
            f'trajout "{artifacts["prepared_trajectory"]}" netcdf start 2 stop 4 offset 2',
            generated[0].content,
        )
        self.assertIn(
            f'trajout "{artifacts["centered_trajectory"]}" xtc start 2 stop 4 offset 2',
            generated[1].content,
        )
        self.assertEqual(request.source_md_atom_indices, (3, 1, 7))

    def test_fake_runner_receipt_audits_all_outputs_without_importing_mdtraj(self) -> None:
        previously_loaded_mdtraj = sys.modules.get("mdtraj")
        runner = _FakeCpptrajRunner()
        processor = _FakeProcessor()
        request = self._request()
        backend = TrajectoryPreparationV3Backend(
            self.tool,
            chunk_frames=37,
            runner=runner,
            processor=processor,
        )

        receipt = backend.run(request)

        self.assertEqual(receipt.metadata["contract"], "mmpbsa.trajectory.v3")
        self.assertEqual(
            receipt.metadata["reference"],
            {
                "artifact": "prepared_reference_trajectory",
                "frame": 1,
                "semantics": "post_npt_production_start",
                "source_artifact": "selected_trajectory",
                "preprocessing": "whole_molecules_then_autoimage_ligand_anchor_receptor_fixed",
            },
        )
        self.assertEqual(receipt.parsed["frame_count"], 2)
        self.assertEqual(receipt.parsed["source_frame_count"], 5)
        self.assertEqual(receipt.parsed["atom_count"], 3)
        self.assertEqual(receipt.parsed["atom_order_signature"], "a" * 64)
        self.assertEqual(receipt.parsed["projection_sha256"], "b" * 64)
        self.assertEqual(
            receipt.metadata["validation"]["selected_trajectory"]["first_time_ps"],
            0.0,
        )
        self.assertEqual(
            receipt.metadata["validation"]["prepared_reference_trajectory"]["first_time_ps"],
            0.0,
        )
        self.assertEqual(processor.project_calls[0][2], 37)
        self.assertEqual(
            [item[0].name for item in processor.inspect_calls],
            ["whole.xtc", "prepared.nc", "prepared_reference.nc", "centered.xtc"],
        )
        self.assertEqual(len(receipt.commands), 2)
        self.assertEqual(len(runner.commands), 3)
        self.assertTrue(receipt.artifacts["prepared_trajectory"].is_file())
        self.assertTrue(receipt.artifacts["prepared_reference_trajectory"].is_file())
        self.assertTrue(receipt.artifacts["centered_trajectory"].is_file())
        self.assertIs(sys.modules.get("mdtraj"), previously_loaded_mdtraj)

    def test_output_time_or_box_drift_is_rejected(self) -> None:
        backend = TrajectoryPreparationV3Backend(
            self.tool,
            runner=_FakeCpptrajRunner(),
            processor=_FakeProcessor(centered_times=(10.0, 31.0)),
        )
        with self.assertRaisesRegex(BackendOutputError, "times differ"):
            backend.run(self._request("drift"))

    def test_unrepresentable_netcdf_cell_warning_is_not_success(self) -> None:
        class WarningRunner(_FakeCpptrajRunner):
            def run(self, command):
                record = super().run(command)
                if "-i" in command.argv:
                    return replace(
                        record,
                        stdout="Warning: Box cannot be properly stored as Amber NetCDF trajectory.",
                    )
                return record

        backend = TrajectoryPreparationV3Backend(
            self.tool, runner=WarningRunner(), processor=_FakeProcessor()
        )
        with self.assertRaisesRegex(BackendOutputError, "unit cell cannot be represented"):
            backend.run(self._request("bad-cell"))

    def test_prepared_t0_reference_must_be_single_frame_at_source_t0(self) -> None:
        for index, times in enumerate(((1.0,), (0.0, 10.0))):
            with self.subTest(times=times):
                backend = TrajectoryPreparationV3Backend(
                    self.tool,
                    runner=_FakeCpptrajRunner(),
                    processor=_FakeProcessor(reference_times=times),
                )
                with self.assertRaises(BackendOutputError):
                    backend.run(self._request(f"reference-drift-{index}"))

    def test_source_t0_interval_step_and_exact_frame_count_are_required(self) -> None:
        invalid_audits = (
            replace(_full_audit(), times_ps=(1.0, 11.0, 21.0, 31.0, 41.0)),
            replace(_full_audit(), times_ps=(0.0, 10.0, 21.0, 30.0, 40.0)),
            replace(_full_audit(), steps=(0, 5_000, 10_001, 15_000, 20_000)),
            TrajectoryV3Audit(
                atom_count=3,
                frame_count=4,
                times_ps=(0.0, 10.0, 20.0, 30.0),
                boxes_nm=tuple(_BOX for _ in range(4)),
                steps=(0, 5_000, 10_000, 15_000),
            ),
        )
        for index, audit in enumerate(invalid_audits):
            with self.subTest(index=index):
                backend = TrajectoryPreparationV3Backend(
                    self.tool,
                    runner=_FakeCpptrajRunner(),
                    processor=_FakeProcessor(source_audit=audit),
                )
                with self.assertRaises(BackendOutputError):
                    backend.run(self._request(f"invalid-source-{index}"))

    def test_rigid_fit_may_rotate_box_vectors_without_changing_box_geometry(self) -> None:
        rotated = (0.0, 3.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 3.0)
        backend = TrajectoryPreparationV3Backend(
            self.tool,
            runner=_FakeCpptrajRunner(),
            processor=_FakeProcessor(centered_boxes=(rotated, rotated)),
        )

        receipt = backend.run(self._request("rotated-box"))

        self.assertEqual(receipt.parsed["frame_count"], 2)

    def test_request_rejects_ambiguous_or_inconsistent_inputs(self) -> None:
        request = self._request()
        invalid = (
            replace(request, source_md_atom_indices=(3, 1, 3)),
            replace(request, expected_atom_order_signature="not-a-digest"),
            replace(request, residue_crosswalk=((1, 2, "MOL"), (2, 1, "MOL"))),
            replace(request, source_first_time_ps=1.0),
            replace(request, source_interval_ps=0.0),
            replace(request, source_interval_steps=0),
            replace(request, expected_source_frames=3),
            replace(request, expected_frames=3),
            replace(request, autoimage_fixed_mask=None),
            replace(request, fit_mask=":1; quit"),
            replace(request, full_trajectory=self._file("production.dcd")),
            replace(request, full_system_coordinates=self._file("solvated.xyz")),
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                with self.assertRaises(BackendConfigurationError):
                    candidate.validate()

    def test_factory_validates_streaming_chunk_size(self) -> None:
        processor = _FakeProcessor()
        backend = create_trajectory_v3_backend(
            {
                "projection_chunk_frames": 17,
                "cpptraj": {
                    "executable": "fake-cpptraj",
                    "version_arguments": ["-V"],
                },
            },
            processor=processor,
        )
        self.assertEqual(backend.chunk_frames, 17)
        self.assertIs(backend.processor, processor)
        for value in (True, 0, 10_001, "100"):
            with self.subTest(value=value):
                with self.assertRaises(BackendConfigurationError):
                    create_trajectory_v3_backend({"projection_chunk_frames": value})


@unittest.skipUnless(
    _HAS_MDTRAJ and _HAS_PARMED,
    "optional MDTraj/ParmEd dependencies are not installed",
)
class MdtrajStreamingProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_self_contained_gromacs_top_gro_and_amber_dry_projection(self) -> None:
        import hashlib
        import json

        import mdtraj as md
        import numpy as np
        import parmed as pmd
        from mdtraj.formats import XTCTrajectoryFile

        full_topology = self.root / "full.top"
        full_topology.write_text(
            """[ defaults ]
1 2 yes 0.5 0.83333333

[ atomtypes ]
CT 6 12.011 0.0 A 0.34 0.1
OX 8 15.999 0.0 A 0.30 0.2
HX 1 1.008 0.0 A 0.10 0.0

[ bondtypes ]
CT OX 1 0.14 100000.0
CT HX 1 0.11 100000.0

[ moleculetype ]
MOL 3

[ atoms ]
1 CT 1 MOL C1 1 0.0 12.011
2 OX 1 MOL O1 1 0.0 15.999
3 HX 1 MOL H1 1 0.0 1.008

[ bonds ]
1 2 1
1 3 1

[ system ]
minimal

[ molecules ]
MOL 2
""",
            encoding="utf-8",
        )
        full_coordinates = self.root / "full.gro"
        atoms = (
            (1, "C1", 0.10),
            (1, "O1", 0.20),
            (1, "H1", 0.30),
            (2, "C1", 0.40),
            (2, "O1", 0.50),
            (2, "H1", 0.60),
        )
        full_coordinates.write_text(
            "minimal\n6\n"
            + "".join(
                f"{residue:5d}{'MOL':<5}{name:>5}{serial:5d}{x:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
                for serial, (residue, name, x) in enumerate(atoms, start=1)
            )
            + "   3.00000   3.00000   3.00000\n",
            encoding="utf-8",
        )

        dry_topology = self.root / "complex.prmtop"
        dry = deepcopy(pmd.load_file(str(full_topology)))
        dry.strip((True, True, True, False, False, False))
        dry.atoms[:] = [dry.atoms[1], dry.atoms[0], dry.atoms[2]]
        dry.save(str(dry_topology))
        dry = pmd.load_file(str(dry_topology))
        signature = hashlib.sha256(
            json.dumps(
                {
                    "schema": "mmpbsa.atom_order.v1",
                    "atoms": [
                        {
                            "residue_index": atom.residue.idx + 1,
                            "residue_name": atom.residue.name,
                            "atom_name": atom.name,
                            "atomic_number": int(atom.atomic_number or 0),
                        }
                        for atom in dry.atoms
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        source = self.root / "production.xtc"
        first_frame = md.load(str(full_coordinates)).xyz[0]
        xyz = np.stack((first_frame, first_frame + np.float32(0.01)))
        boxes = np.tile(np.eye(3, dtype=np.float32) * 3.0, (2, 1, 1))
        with XTCTrajectoryFile(str(source), "w") as stream:
            stream.write(
                xyz,
                time=np.array((0.0, 10.0), dtype=np.float32),
                step=np.array((0, 5_000), dtype=np.int32),
                box=boxes,
            )

        index = self.root / "complex.ndx"
        index.write_text("[ Complex ]\n5 4 6\n", encoding="utf-8")
        selection = EntitySelection(md_atom_indices=(4, 5, 6), scoring_atom_indices=(1, 2, 3))
        prepared = PreparedTrajectoryView(
            case_id="synthetic",
            source_md_atom_indices=(5, 4, 6),
            atom_order_signature=signature,
            residue_crosswalk=((2, 1, "MOL"),),
            complex_topology_role="complex_topology",
            binding_selections={
                name: selection
                for name in (
                    "receptor_all",
                    "ligand_all",
                    "receptor_interface_backbone",
                    "ligand_qc",
                    "receptor_heavy",
                    "ligand_heavy",
                )
            },
            resolved_artifacts={
                "index": index,
                "md_topology": full_topology,
                "md_coordinates": full_coordinates,
                "complex_topology": dry_topology,
            },
        )
        upstream = StageInput(
            task=TaskKey(StageId.MD, case_id="synthetic", replica_id="rep01"),
            attempt_id="0001",
            root=self.root,
            artifacts={"full_trajectory": source},
        )

        class CaptureBackend:
            def run(self, request: TrajectoryPreparationV3Request):
                return request

        with patch(
            "mmpbsa.backends.trajectory_dispatch.create_trajectory_v3_backend",
            return_value=CaptureBackend(),
        ):
            request = execute_trajectory_adapter(
                science={
                    "adapter_contract": "mmpbsa.trajectory.v3",
                    "full_system_topology_role": "md_topology",
                    "full_system_coordinates_role": "md_coordinates",
                    "source_trajectory_contract": {
                        "first_time_ps": 0.0,
                        "interval_ps": 10.0,
                        "interval_steps": 5_000,
                        "expected_frames": 2,
                    },
                    "start_frame": 1,
                    "end_frame": 2,
                    "expected_frames": 2,
                },
                config={},
                prepared=prepared,
                upstream=upstream,
                output_dir=self.root / "unused",
                runner=None,
            )

        self.assertEqual(request.full_system_topology, full_topology)
        self.assertEqual(request.full_system_coordinates, full_coordinates)
        self.assertEqual(request.dry_complex_topology, dry_topology)
        self.assertEqual(request.source_md_atom_indices, (5, 4, 6))
        request.validate()
        projected_path = self.root / "selected.xtc"
        projection = MdtrajTrajectoryV3Processor().project(request, projected_path, chunk_frames=1)
        self.assertEqual(projection.source_atom_count, 6)
        self.assertEqual(projection.selected.atom_count, 3)
        self.assertEqual(projection.selected.frame_count, 2)
        self.assertEqual(projection.atom_order_signature, signature)
        with XTCTrajectoryFile(str(projected_path), "r") as stream:
            observed, _, _, _ = stream.read(n_frames=2)
        np.testing.assert_allclose(observed, xyz[:, [4, 3, 5], :], atol=1.0e-3)
        self.assertEqual(
            [atom.name for atom in md.load_topology(str(dry_topology)).atoms],
            ["O1", "C1", "H1"],
        )

    def test_residue_crosswalk_rejects_swapped_identical_residues(self) -> None:
        import parmed as pmd

        full = pmd.Structure()
        dry = pmd.Structure()
        for structure in (full, dry):
            for residue_number in (1, 2):
                carbon = pmd.Atom(name="C1", atomic_number=6, mass=12.0)
                hydrogen = pmd.Atom(name="H1", atomic_number=1, mass=1.0)
                structure.add_atom(carbon, "MOL", residue_number)
                structure.add_atom(hydrogen, "MOL", residue_number)
                structure.bonds.append(pmd.Bond(carbon, hydrogen))

        import hashlib
        import json

        payload = {
            "schema": "mmpbsa.atom_order.v1",
            "atoms": [
                {
                    "residue_index": atom.residue.idx + 1,
                    "residue_name": atom.residue.name,
                    "atom_name": atom.name,
                    "atomic_number": int(atom.atomic_number or 0),
                }
                for atom in dry.atoms
            ],
        }
        signature = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        with self.assertRaisesRegex(BackendConfigurationError, "crosswalk"):
            _validate_topology_projection(
                full,
                dry,
                (2, 3, 0, 1),
                expected_atom_order_signature=signature,
                residue_crosswalk=((1, 1, "MOL"), (2, 2, "MOL")),
            )

    def test_non_monotonic_mapping_streams_full_dry_production_for_t0_reference(
        self,
    ) -> None:
        import mdtraj as md
        import numpy as np
        from mdtraj.formats import XTCTrajectoryFile

        full_topology = self.root / "full.pdb"
        dry_topology = self.root / "dry.pdb"
        full_topology.write_text(
            _pdb(
                (("C1", "C"), ("O1", "O"), ("N1", "N"), ("H1", "H")),
                ((1, 2), (1, 3), (3, 4)),
            ),
            encoding="utf-8",
        )
        dry_topology.write_text(
            _pdb(
                (("N1", "N"), ("C1", "C"), ("H1", "H")),
                ((1, 2), (1, 3)),
            ),
            encoding="utf-8",
        )
        source = self.root / "full.xtc"
        output = self.root / "selected.xtc"
        xyz = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3) / 100.0
        times = np.arange(5, dtype=np.float32) * 10.0
        steps = np.arange(5, dtype=np.int32) * 5_000
        boxes = np.tile(np.eye(3, dtype=np.float32) * 3.0, (5, 1, 1))
        with XTCTrajectoryFile(str(source), "w") as target:
            target.write(xyz, time=times, step=steps, box=boxes)

        import parmed as pmd

        dry_structure = pmd.load_file(str(dry_topology))
        signature_payload = {
            "schema": "mmpbsa.atom_order.v1",
            "atoms": [
                {
                    "residue_index": atom.residue.idx + 1,
                    "residue_name": atom.residue.name,
                    "atom_name": atom.name,
                    "atomic_number": int(atom.atomic_number or 0),
                }
                for atom in dry_structure.atoms
            ],
        }
        import hashlib
        import json

        expected_signature = hashlib.sha256(
            json.dumps(
                signature_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        request = TrajectoryPreparationV3Request(
            full_trajectory=source,
            full_system_topology=full_topology,
            full_system_coordinates=full_topology,
            dry_complex_topology=dry_topology,
            source_md_atom_indices=(3, 1, 4),
            output_dir=self.root / "unused",
            start_frame=2,
            end_frame=5,
            stride=2,
            expected_atom_order_signature=expected_signature,
            residue_crosswalk=((1, 1, "MOL"),),
            source_first_time_ps=0.0,
            source_interval_ps=10.0,
            source_interval_steps=5_000,
            expected_source_frames=5,
            expected_frames=2,
        )
        processor = MdtrajTrajectoryV3Processor()
        projection = processor.project(request, output, chunk_frames=2)

        selected = md.load_xtc(str(output), top=str(dry_topology))
        self.assertEqual(projection.source_frames_scanned, 5)
        self.assertEqual(projection.selected.times_ps, (0.0, 10.0, 20.0, 30.0, 40.0))
        np.testing.assert_allclose(
            selected.xyz,
            xyz[:, [2, 0, 3], :],
            atol=1.0e-3,
        )
        self.assertEqual(
            [atom.name for atom in selected.topology.atoms],
            ["N1", "C1", "H1"],
        )

        wrong_coordinates = self.root / "wrong_coordinates.pdb"
        wrong_coordinates.write_text(
            _pdb(
                (("O1", "O"), ("C1", "C"), ("N1", "N"), ("H1", "H")),
                ((1, 2), (1, 3), (3, 4)),
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(BackendConfigurationError, "coordinate atom order differs"):
            processor.project(
                replace(request, full_system_coordinates=wrong_coordinates),
                self.root / "wrong_selected.xtc",
                chunk_frames=2,
            )


def _pdb(
    atoms: tuple[tuple[str, str], ...],
    bonds: tuple[tuple[int, int], ...],
) -> str:
    lines = []
    for serial, (name, element) in enumerate(atoms, start=1):
        lines.append(
            f"ATOM  {serial:5d} {name:>4s} MOL A   1    "
            f"{serial * 10.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}"
        )
    for first, second in bonds:
        lines.append(f"CONECT{first:5d}{second:5d}")
    lines.extend(("TER", "END", ""))
    return "\n".join(lines)

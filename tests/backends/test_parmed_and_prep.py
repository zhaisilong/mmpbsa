from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mmpbsa.backends.parmed import (
    AtomRecord,
    ParmEdAdmissionBackend,
    ParmEdAdmissionRequest,
    TopologyView,
)
from mmpbsa.backends.prep import (
    AmberAssemblyRequest,
    SmallMoleculeParameterizationRequest,
    create_amber_preparation_tools,
)


class FakeLoader:
    def __init__(self, views: dict[Path, TopologyView]) -> None:
        self.views = {path.resolve(): view for path, view in views.items()}

    def probe(self) -> dict[str, object]:
        return {
            "name": "FakeParmEd",
            "raw_version": "4.3.0",
            "version": (4, 3, 0),
        }

    def load(self, path: Path) -> TopologyView:
        return self.views[path.resolve()]


def atom(ordinal: int, name: str) -> AtomRecord:
    return AtomRecord(
        ordinal=ordinal,
        residue_ordinal=1,
        residue_name="MOL",
        atom_name=name,
        atomic_number=6,
        atom_type="c3",
        charge=0.0,
        mass=12.01,
        epsilon=0.1,
        rmin=1.9,
    )


class AdmissionAndPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def file(self, name: str) -> Path:
        path = self.root / name
        path.write_text(name + "\n", encoding="utf-8")
        return path

    def test_projection_admission_accepts_exact_component_views(self) -> None:
        complex_path = self.file("complex.prmtop")
        receptor_path = self.file("receptor.prmtop")
        binder_path = self.file("binder.prmtop")
        complex_view = TopologyView((atom(1, "C1"), atom(2, "C2")))
        receptor_view = TopologyView((atom(1, "C1"),))
        binder_view = TopologyView((atom(1, "C2"),))
        backend = ParmEdAdmissionBackend(
            FakeLoader(
                {
                    complex_path: complex_view,
                    receptor_path: receptor_view,
                    binder_path: binder_view,
                }
            )
        )

        result = backend.admit(
            ParmEdAdmissionRequest(
                complex_topology=complex_path,
                component_topologies={
                    "receptor": receptor_path,
                    "binder": binder_path,
                },
                component_atom_indices={
                    "receptor": (1,),
                    "binder": (2,),
                },
            )
        )

        self.assertTrue(result.accepted)
        self.assertFalse(result.issues)

    def test_preparation_tools_are_callable_but_not_stage_defaults(self) -> None:
        source = self.file("ligand.sdf")
        declared = self.file("ligand.mol2")
        tools = create_amber_preparation_tools({})
        parameterization = tools.parameterize(
            SmallMoleculeParameterizationRequest(
                source=source,
                output_dir=self.root / "parameters",
                input_format="sdf",
                residue_name="LIG",
                formal_charge=0,
                charge_method="bcc",
                atom_type="gaff2",
                force_field="gaff2",
            ),
            dry_run=True,
        )
        assembly = tools.assemble(
            AmberAssemblyRequest(
                input_text=(
                    f'lig = loadmol2 "{declared}"\n'
                    "saveamberparm lig ligand.prmtop ligand.inpcrd\n"
                    "quit\n"
                ),
                declared_inputs=(declared,),
                output_artifacts={
                    "topology": Path("ligand.prmtop"),
                    "coordinates": Path("ligand.inpcrd"),
                },
                output_dir=self.root / "assembly",
            ),
            dry_run=True,
        )

        self.assertEqual(parameterization.commands[0].argv[0], "antechamber")
        self.assertEqual(parameterization.commands[1].argv[0], "parmchk2")
        self.assertEqual(assembly.commands[0].argv[0], "tleap")
        self.assertFalse((self.root / "parameters").exists())
        self.assertFalse((self.root / "assembly").exists())


if __name__ == "__main__":
    unittest.main()

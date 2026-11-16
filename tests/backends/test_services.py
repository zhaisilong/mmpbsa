from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

try:
    import yaml  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:
    # Core declares PyYAML as an install dependency. The repository's minimal
    # test interpreter may not have optional distributions installed, and
    # these tests do not exercise YAML loading.
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.YAMLError = ValueError
    yaml_stub.safe_load = lambda stream: None
    sys.modules["yaml"] = yaml_stub

from mmpbsa.backends.api import BackendReceipt, ReceiptState  # noqa: E402
from mmpbsa.backends.errors import (  # noqa: E402
    BackendConfigurationError,
    IncompatibleToolVersion,
    ToolProbeError,
)
from mmpbsa.backends.parmed import AdmissionResult  # noqa: E402
from mmpbsa.backends.provider import create_provider  # noqa: E402
from mmpbsa.backends.services import BuiltinStageService, _prepared_components  # noqa: E402
from mmpbsa.core import (  # noqa: E402
    ArtifactRef,
    BindingTopologyRoles,
    EntitySelection,
    PreparedCase,
    PreparedCaseManifest,
    PreparedEntity,
    PreparedMdSystem,
    PreparedScoringSystem,
    PrimaryBinding,
    ProviderRegistry,
    QcVerdict,
    ResidueIdentity,
    StageInput,
    StageRequest,
    StageService,
    TaskKey,
    resolve_protocol,
)
from mmpbsa.core.state import StageId  # noqa: E402


class ProviderAndServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fixture = Path(__file__).parent / "fixtures" / "fake_tool.py"
        self.fake = self.root / "fake-tool"
        shutil.copy2(fixture, self.fake)
        self.fake.chmod(0o755)
        amber_bin = self.root / "amber" / "bin"
        amber_bin.mkdir(parents=True)
        self.sander = amber_bin / "sander"
        shutil.copy2(fixture, self.sander)
        self.sander.chmod(0o755)
        self.runtime = {
            "environment": {"AMBERHOME": str(amber_bin.parent)},
            "tool": {
                "executable": str(self.fake),
                "version_arguments": ["--version"],
                "major_policy": {"accepted": [2024]},
            },
            "sander": {
                "executable": str(self.sander),
                "version_arguments": ["--version"],
                "major_policy": {"accepted": [2024]},
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def file(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")
        return path

    def prepared_case(
        self,
        *,
        ligand_kind: str = "small_molecule",
        receptor_metadata: dict[str, object] | None = None,
        ligand_metadata: dict[str, object] | None = None,
        topology_roles: BindingTopologyRoles | None = None,
    ) -> PreparedCase:
        prepared_root = self.root / "prepared-v2"
        binding_roles = topology_roles or BindingTopologyRoles(
            complex="complex_topology",
            receptor="receptor_topology",
            ligand="ligand_topology",
        )
        paths = {
            "md_coordinates": self.file("prepared-v2/md/system.gro"),
            "md_topology": self.file("prepared-v2/md/topol.top"),
            "solvated_topology": self.file(
                "prepared-v2/md/solvated.prmtop"
            ),
            "index": self.file("prepared-v2/md/index.ndx"),
            binding_roles.complex: self.file(
                "prepared-v2/scoring/complex.prmtop"
            ),
            binding_roles.receptor: self.file(
                "prepared-v2/scoring/receptor.prmtop"
            ),
            binding_roles.ligand: self.file(
                "prepared-v2/scoring/ligand.prmtop"
            ),
            "coordinates": self.file("prepared-v2/scoring/coordinates.inpcrd"),
        }

        def artifact(role: str) -> ArtifactRef:
            path = paths[role]
            content = path.read_bytes()
            return ArtifactRef(
                path=path.relative_to(prepared_root),
                sha256=hashlib.sha256(content).hexdigest(),
                size=len(content),
                role=role,
            )

        def selection(
            md_atoms: tuple[int, ...],
            scoring_atoms: tuple[int, ...],
        ) -> EntitySelection:
            return EntitySelection(
                md_atom_indices=md_atoms,
                scoring_atom_indices=scoring_atoms,
            )

        receptor = PreparedEntity(
            entity_id="receptor",
            kind="protein",
            md_atom_indices=(1, 2),
            scoring_atom_indices=(1, 2),
            residues=(ResidueIdentity("A_1", "ALA", 1, 1),),
            named_selections={
                "all": selection((1, 2), (1, 2)),
                "heavy_atoms": selection((1, 2), (1, 2)),
                "backbone": selection((1, 2), (1, 2)),
                "interface_backbone": selection((1,), (1,)),
            },
            metadata=(
                {"interface_chemistry_classes": ("polar",)}
                if receptor_metadata is None
                else receptor_metadata
            ),
        )
        ligand_selections = {
            "all": selection((3, 4), (3, 4)),
            "heavy_atoms": selection((3,), (3,)),
        }
        if ligand_kind == "peptide":
            ligand_selections["backbone"] = selection((3,), (3,))
        ligand = PreparedEntity(
            entity_id="ligand",
            kind=ligand_kind,
            md_atom_indices=(3, 4),
            scoring_atom_indices=(3, 4),
            residues=(ResidueIdentity("L_1", "LIG", 2, 2),),
            named_selections=ligand_selections,
            metadata=({"formal_charge": 0} if ligand_metadata is None else ligand_metadata),
        )
        water = PreparedEntity(
            entity_id="structural_water",
            kind="water",
            md_atom_indices=(5,),
            scoring_atom_indices=(),
            residues=(ResidueIdentity("W_1", "HOH", 3, None),),
            named_selections={
                "all": selection((5,), ()),
                "heavy_atoms": selection((5,), ()),
            },
        )
        manifest = PreparedCaseManifest(
            case_id="case-a",
            md=PreparedMdSystem(
                backend="gromacs",
                atom_count=5,
                system_signature="b" * 64,
                artifacts=tuple(
                    artifact(role)
                    for role in (
                        "md_coordinates",
                        "md_topology",
                        "solvated_topology",
                        "index",
                    )
                ),
            ),
            scoring=PreparedScoringSystem(
                backend="ambertools",
                atom_count=4,
                atom_order_signature="a" * 64,
                source_md_atom_indices=(1, 2, 3, 4),
                radii_model="mbondi2",
                artifacts=tuple(
                    artifact(role)
                    for role in (
                        binding_roles.complex,
                        binding_roles.receptor,
                        binding_roles.ligand,
                        "coordinates",
                    )
                ),
            ),
            entities=(receptor, ligand, water),
            primary_binding=PrimaryBinding(
                binding_id="primary",
                receptor_entity_ids=("receptor",),
                ligand_entity_ids=("ligand",),
                excluded_entity_ids=("structural_water",),
                topology_artifact_roles=binding_roles,
            ),
            provenance={"fixture": "backend-service"},
        )
        return PreparedCase(
            manifest_path=self.file("prepared-v2/prepared_case.json"),
            root=prepared_root,
            manifest=manifest,
            resolved_artifacts=paths,
        )

    def test_entry_point_provider_matches_frozen_core_abi(self) -> None:
        provider = create_provider()

        self.assertEqual(provider.provider_id, "backends")
        self.assertEqual(provider.api_version, "2")
        self.assertEqual(provider.package_version, "0.7.0")
        self.assertEqual(
            set(provider.capabilities),
            {
                "admission",
                "md",
                "trajectory",
                "scoring",
                "trajectory_qc",
                "report.assets",
            },
        )
        registry = ProviderRegistry((provider,))
        for capability in provider.capabilities:
            service = registry.service("backends", capability)
            self.assertIsInstance(service, StageService)

    def test_compatibility_probes_engine_and_locks_major(self) -> None:
        service = BuiltinStageService("md")

        with patch(
            "mmpbsa.backends.trajectory_v3.metadata.version",
            side_effect=metadata.PackageNotFoundError,
        ):
            compatibility = service.compatibility(self.runtime)

        self.assertEqual(compatibility.provider_id, "backends")
        self.assertEqual(
            compatibility.adapter_contract,
            "mmpbsa.backends.md.v2",
        )
        self.assertEqual(compatibility.engine, "gromacs")
        self.assertEqual(compatibility.engine_major, "gromacs=2024")
        self.assertIn("Fake scientific tool 2024.2", compatibility.engine_version)

        with patch(
            "mmpbsa.backends.trajectory_v3.metadata.version",
            side_effect=lambda name: {
                "mdtraj": "1.11.1",
                "ParmEd": "4.3.0",
            }[name],
        ):
            phase_plan = service.compatibility(
                {
                    **self.runtime,
                    "_stage_contract": "mmpbsa.md-phase-plan.v2",
                    "_md_engine_profile": "gromacs_canonical@1",
                }
            )
        self.assertEqual(
            phase_plan.adapter_contract,
            "mmpbsa.backends.md.v3",
        )
        self.assertEqual(phase_plan.engine, "gromacs+mdtraj")
        with patch(
            "mmpbsa.backends.trajectory_v3.metadata.version",
            side_effect=metadata.PackageNotFoundError,
        ):
            with self.assertRaisesRegex(
                ToolProbeError,
                "optional mdtraj>=1.11,<2",
            ):
                service.compatibility(
                    {
                        **self.runtime,
                        "_stage_contract": "mmpbsa.md-phase-plan.v2",
                        "_md_engine_profile": "gromacs_canonical@1",
                    }
                )
        with self.assertRaisesRegex(
            BackendConfigurationError,
            "built-in GROMACS MD requires",
        ):
            service.compatibility(
                {
                    **self.runtime,
                    "_stage_contract": "mmpbsa.md-phase-plan.v2",
                    "_md_engine_profile": "openmm_8_5@1",
                }
            )

        scoring = BuiltinStageService("scoring").compatibility(self.runtime)
        self.assertEqual(
            scoring.adapter_contract,
            "mmpbsa.backends.scoring.v3",
        )

        with patch(
            "mmpbsa.backends.trajectory_v3.metadata.version",
            side_effect=lambda name: {
                "mdtraj": "1.11.1",
                "ParmEd": "4.3.0",
            }[name],
        ):
            trajectory = BuiltinStageService("trajectory").compatibility(
                {
                    "_stage_contract": "mmpbsa.trajectory.v3",
                    "cpptraj": self.runtime["tool"],
                }
            )
        self.assertEqual(
            trajectory.adapter_contract,
            "mmpbsa.backends.trajectory.v4",
        )
        self.assertEqual(trajectory.engine, "cpptraj+mdtraj+parmed")
        self.assertEqual(
            trajectory.engine_major,
            "cpptraj=2024;mdtraj=1;parmed=4",
        )
        self.assertIn("mdtraj 1.11.1", trajectory.engine_version)
        self.assertIn("ParmEd 4.3.0", trajectory.engine_version)
        with self.assertRaisesRegex(
            BackendConfigurationError,
            "unsupported trajectory stage contract",
        ):
            BuiltinStageService("trajectory").compatibility(
                {
                    "_stage_contract": "mmpbsa.trajectory-plan.v3",
                    "cpptraj": self.runtime["tool"],
                }
            )
        with patch(
            "mmpbsa.backends.trajectory_v3.metadata.version",
            side_effect=metadata.PackageNotFoundError,
        ):
            with self.assertRaisesRegex(
                ToolProbeError,
                "optional mdtraj>=1.11,<2",
            ):
                BuiltinStageService("trajectory").compatibility(
                    {
                        "_stage_contract": "mmpbsa.trajectory.v3",
                        "cpptraj": self.runtime["tool"],
                    }
                )
        with patch(
            "mmpbsa.backends.trajectory_v3.metadata.version",
            return_value="1.10.3",
        ):
            with self.assertRaisesRegex(
                IncompatibleToolVersion,
                "requires >=1.11,<2",
            ):
                BuiltinStageService("trajectory").compatibility(
                    {
                        "_stage_contract": "mmpbsa.trajectory.v3",
                        "cpptraj": self.runtime["tool"],
                    }
                )
        with patch(
            "mmpbsa.backends.trajectory_v3.metadata.version",
            side_effect=["1.11.1", metadata.PackageNotFoundError()],
        ):
            with self.assertRaisesRegex(
                ToolProbeError,
                "optional ParmEd>=4.3,<5",
            ):
                BuiltinStageService("trajectory").compatibility(
                    {
                        "_stage_contract": "mmpbsa.trajectory.v3",
                        "cpptraj": self.runtime["tool"],
                    }
                )

    def test_admission_uses_primary_binding_scoring_projection(self) -> None:
        prepared = self.prepared_case(
            topology_roles=BindingTopologyRoles(
                complex="admission_complex",
                receptor="admission_receptor",
                ligand="admission_ligand",
            )
        )
        self.assertNotIn(
            "component_topology_roles",
            resolve_protocol("production_3x5ns@1").stages["admit"],
        )
        captured = []

        class AdmissionBackend:
            def admit(self, request: object) -> AdmissionResult:
                captured.append(request)
                return AdmissionResult(
                    accepted=True,
                    issues=(),
                    complex_signature="complex",
                    component_signatures={
                        "receptor": "receptor",
                        "ligand": "ligand",
                    },
                    tool={"name": "fake-parmed"},
                )

        attempt = self.root / "admission-attempt"
        request = StageRequest(
            task=TaskKey(StageId.ADMIT, case_id="case-a"),
            science={},
            runtime={},
            prepared_case=prepared.binding_view,
            upstream={},
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        with patch(
            "mmpbsa.backends.services._create_backend",
            return_value=AdmissionBackend(),
        ):
            result = BuiltinStageService("admission").execute(request)

        admitted = captured[0]
        self.assertEqual(
            admitted.complex_topology,
            prepared.artifact("admission_complex"),
        )
        self.assertEqual(
            admitted.component_atom_indices,
            {"receptor": (1, 2), "ligand": (3, 4)},
        )
        self.assertEqual(
            admitted.component_topologies,
            {
                "receptor": prepared.artifact("admission_receptor"),
                "ligand": prepared.artifact("admission_ligand"),
            },
        )
        self.assertTrue(admitted.require_complete_partition)
        self.assertTrue(result.payload["complete_projection"])
        self.assertEqual(
            result.payload["topology_artifact_roles"],
            {
                "complex": "admission_complex",
                "receptor": "admission_receptor",
                "ligand": "admission_ligand",
            },
        )

    def test_trajectory_renders_binding_masks_in_scoring_order(self) -> None:
        prepared = self.prepared_case(
            topology_roles=BindingTopologyRoles(
                complex="primary_complex",
                receptor="primary_receptor",
                ligand="primary_ligand",
            )
        )
        source_tpr = self.file("md/production.tpr")
        source_trajectory = self.file("md/production.xtc")
        science = dict(resolve_protocol("production_3x5ns@1").stages["trajectory"])
        self.assertNotIn("complex_topology_role", science)
        science.update(
            {
                "start_frame": 1,
                "end_frame": 2,
                "stride": 1,
                "expected_frames": 2,
            }
        )
        attempt = self.root / "trajectory-attempt"
        request = StageRequest(
            task=TaskKey(
                StageId.TRAJECTORY,
                case_id="case-a",
                replica_id="rep01",
            ),
            science=science,
            runtime={
                "gromacs": self.runtime["tool"],
                "cpptraj": self.runtime["tool"],
            },
            prepared_case=prepared.binding_view,
            upstream={
                "md": StageInput(
                    task=TaskKey(
                        StageId.MD,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=source_tpr.parent,
                    artifacts={
                        "tpr": source_tpr,
                        "xtc": source_trajectory,
                    },
                )
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        result = BuiltinStageService("trajectory").execute(request)

        rendered = result.artifacts["cpptraj_input"].read_text(encoding="utf-8")
        self.assertNotIn("{{selection:", rendered)
        self.assertIn("autoimage anchor @3,4 fixed @1,2", rendered)
        self.assertIn("rms fit @1 reference", rendered)
        self.assertIn(
            str(prepared.artifact("primary_complex")),
            rendered,
        )

    def test_scoring_service_maps_one_model_and_seals_native_inputs(self) -> None:
        trajectory = self.file("trajectory/prepared.nc")
        prepared = self.prepared_case()
        attempt = self.root / "scoring-attempt"
        native_inputs = {"main": "&cntrl\n  igb=5, intdiel=4.0, extdiel=78.5,\n/\n"}
        request = StageRequest(
            task=TaskKey(
                StageId.SCORING,
                case_id="case-a",
                replica_id="rep01",
                model_id="gb",
            ),
            science={
                "control_text": ("&general\n  use_sander=1,\n/\n&gb\n  igb=5,\n/\n"),
                "native_inputs": native_inputs,
                "expected_frames": 2,
            },
            runtime=self.runtime,
            prepared_case=prepared.binding_view,
            upstream={
                "trajectory": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=trajectory.parent,
                    artifacts={"prepared_trajectory": trajectory},
                )
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        result = BuiltinStageService("scoring").execute(request)

        expected_roles = {
            "native_main_input": "gb.mdin",
        }
        for role, filename in expected_roles.items():
            self.assertEqual(
                result.artifacts[role],
                attempt / "output" / "inputs" / filename,
            )
            self.assertTrue(result.artifacts[role].is_file())
        self.assertFalse((attempt / "output" / "_MMPBSA_gb.mdin").exists())
        receipt = json.loads(
            Path(result.invocation_provenance["receipt"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(expected_roles),
            set(receipt["artifacts"]).intersection(expected_roles),
        )
        self.assertEqual(receipt["commands"][0]["argv"].count("-use-mdins"), 1)
        self.assertEqual(result.payload["model_id"], "gb")
        self.assertEqual(result.payload["frame_count"], 2)
        self.assertEqual(
            result.payload["dielectric_resolution"]["mode"],
            "fixed",
        )
        self.assertEqual(
            result.payload["dielectric_resolution"]["resolved_internal_dielectric"],
            4.0,
        )
        self.assertIn("delta_output", result.artifacts)

    def test_scoring_resolves_auto_dielectric_from_small_ligand_evidence(
        self,
    ) -> None:
        prepared = self.prepared_case(
            receptor_metadata={"interface_chemistry_classes": ("polar", "nonpolar")},
            ligand_metadata={"formal_charge": 0},
        )
        trajectory = self.file("auto-small/prepared.nc")
        science = dict(resolve_protocol("production_3x5ns@1").stages["scoring"]["models"]["gb"])
        science["expected_frames"] = 2
        attempt = self.root / "auto-small-attempt"
        request = StageRequest(
            task=TaskKey(
                StageId.SCORING,
                case_id="case-a",
                replica_id="rep01",
                model_id="gb",
            ),
            science=science,
            runtime=self.runtime,
            prepared_case=prepared.binding_view,
            upstream={
                "trajectory": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=trajectory.parent,
                    artifacts={"prepared_trajectory": trajectory},
                )
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        result = BuiltinStageService("scoring").execute(request)

        audit = result.payload["dielectric_resolution"]
        self.assertEqual(audit["mode"], "automatic")
        self.assertEqual(audit["resolved_internal_dielectric"], 3.0)
        self.assertFalse(audit["fallback"]["used"])
        self.assertEqual(
            audit["evidence"]["scope"]["interface_entity_ids"],
            ["receptor"],
        )
        native = result.artifacts["native_main_input"].read_text(encoding="utf-8")
        self.assertIn("intdiel=3", native)
        self.assertNotIn("__MMPBSA_AUTO_INTERNAL_DIELECTRIC__", native)

    def test_pb_replaces_auto_dielectric_in_control_and_both_native_inputs(
        self,
    ) -> None:
        prepared = self.prepared_case()
        trajectory = self.file("auto-pb/prepared.nc")
        science = dict(resolve_protocol("production_3x5ns@1").stages["scoring"]["models"]["pb"])
        science["expected_frames"] = 2
        attempt = self.root / "auto-pb-attempt"
        request = StageRequest(
            task=TaskKey(
                StageId.SCORING,
                case_id="case-a",
                replica_id="rep01",
                model_id="pb",
            ),
            science=science,
            runtime=self.runtime,
            prepared_case=prepared.binding_view,
            upstream={
                "trajectory": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=trajectory.parent,
                    artifacts={"prepared_trajectory": trajectory},
                )
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        result = BuiltinStageService("scoring").execute(request)

        for role in (
            "control_input",
            "native_complex_receptor_input",
            "native_ligand_input",
        ):
            content = result.artifacts[role].read_text(encoding="utf-8")
            self.assertNotIn(
                "__MMPBSA_AUTO_INTERNAL_DIELECTRIC__",
                content,
            )
        self.assertIn(
            "indi=3",
            result.artifacts["control_input"].read_text(encoding="utf-8"),
        )
        self.assertIn(
            "epsin=3",
            result.artifacts["native_ligand_input"].read_text(encoding="utf-8"),
        )

    def test_auto_dielectric_peptide_scope_and_invalid_charge_fallback(
        self,
    ) -> None:
        peptide = self.prepared_case(
            ligand_kind="peptide",
            receptor_metadata={"interface_chemistry_classes": ("nonpolar",)},
            ligand_metadata={
                "formal_charge": True,
                "interface_chemistry_classes": ("polar",),
            },
        )
        trajectory = self.file("auto-peptide/prepared.nc")
        science = dict(resolve_protocol("production_3x5ns@1").stages["scoring"]["models"]["gb"])
        science["expected_frames"] = 2
        attempt = self.root / "auto-peptide-attempt"
        request = StageRequest(
            task=TaskKey(
                StageId.SCORING,
                case_id="case-a",
                replica_id="rep01",
                model_id="gb",
            ),
            science=science,
            runtime=self.runtime,
            prepared_case=peptide.binding_view,
            upstream={
                "trajectory": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=trajectory.parent,
                    artifacts={"prepared_trajectory": trajectory},
                )
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        result = BuiltinStageService("scoring").execute(request)

        audit = result.payload["dielectric_resolution"]
        self.assertEqual(audit["resolved_internal_dielectric"], 4.0)
        self.assertTrue(audit["fallback"]["used"])
        self.assertEqual(
            audit["evidence"]["scope"]["interface_entity_ids"],
            ["receptor", "ligand"],
        )
        self.assertTrue(any("formal_charge" in warning for warning in audit["warnings"]))
        self.assertEqual(result.payload["warnings"], audit["warnings"])
        self.assertEqual(
            result.invocation_provenance["metadata"]["dielectric_resolution"]["warnings"],
            audit["warnings"],
        )

    def test_qc_service_returns_core_result_and_generic_verdict(self) -> None:
        trajectory = self.file("upstream/prepared.nc")
        attempt = self.root / "attempt"
        invocation = attempt / "invocations" / "0001"
        service = BuiltinStageService("trajectory_qc")
        request = StageRequest(
            task=TaskKey(
                StageId.TRAJECTORY_QC,
                case_id="case-a",
                replica_id="rep01",
            ),
            science={
                "input_text": 'writedata "rmsd.dat"\nrun\n',
                "declared_inputs": [
                    {
                        "stage": "trajectory",
                        "artifact": "prepared_trajectory",
                    }
                ],
                "output_tables": {"rmsd": "rmsd.dat"},
                "expected_frames": 2,
                "checks": [
                    {
                        "id": "mean_rmsd",
                        "table": "rmsd",
                        "column": "RMSD",
                        "reducer": "mean",
                        "warn_above": 1.0,
                        "block_above": 3.0,
                    }
                ],
            },
            runtime=self.runtime,
            prepared_case=None,
            upstream={
                "trajectory": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=trajectory.parent,
                    artifacts={"prepared_trajectory": trajectory},
                ),
            },
            attempt_dir=attempt,
            invocation_dir=invocation,
        )

        result = service.execute(request)

        self.assertIs(result.qc_verdict, QcVerdict.WARN)
        self.assertAlmostEqual(result.payload["metrics"]["mean_rmsd"], 1.25)
        self.assertEqual(result.payload["verdict"], "WARN")
        self.assertEqual(
            result.payload["issues"][0]["code"],
            "mean_rmsd",
        )
        self.assertEqual(
            result.payload["issues"][0]["verdict"],
            "WARN",
        )
        self.assertTrue((invocation / "backend_receipt.json").is_file())
        self.assertTrue(all(path.is_relative_to(attempt) for path in result.artifacts.values()))

    def test_protocol_qc_resolves_all_binding_selections_and_four_checks(
        self,
    ) -> None:
        prepared = self.prepared_case()
        trajectory = self.file("protocol-qc/prepared.nc")
        science = dict(resolve_protocol("production_3x5ns@1").stages["trajectory_qc"])
        science["expected_frames"] = 2
        attempt = self.root / "protocol-qc-attempt"
        request = StageRequest(
            task=TaskKey(
                StageId.TRAJECTORY_QC,
                case_id="case-a",
                replica_id="rep01",
            ),
            science=science,
            runtime=self.runtime,
            prepared_case=prepared.binding_view,
            upstream={
                "trajectory": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=trajectory.parent,
                    artifacts={"prepared_trajectory": trajectory},
                )
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        result = BuiltinStageService("trajectory_qc").execute(request)

        rendered = result.artifacts["cpptraj_input"].read_text(encoding="utf-8")
        self.assertNotIn("{{selection:", rendered)
        self.assertIn("rms interface_backbone @1", rendered)
        self.assertIn("rms ligand_primary @3", rendered)
        self.assertIn(
            "nativecontacts name primary_binding @1,2 @3",
            rendered,
        )
        self.assertEqual(len(result.payload["check_results"]), 4)
        self.assertAlmostEqual(
            result.payload["metrics"]["interface_backbone_rmsd_p95_angstrom"],
            1.925,
        )
        self.assertIs(result.qc_verdict, QcVerdict.PASS)

    def test_v2_qc_uses_post_npt_t0_reference_with_window_frames(self) -> None:
        prepared = self.prepared_case()
        t0_reference = self.file("protocol-qc-v2/prepared_reference.nc")
        window = self.file("protocol-qc-v2/prepared.nc")
        trajectory_payload = {
            "backend_metadata": {
                "reference": {
                    "artifact": "prepared_reference_trajectory",
                    "frame": 1,
                    "semantics": "post_npt_production_start",
                    "source_artifact": "selected_trajectory",
                    "preprocessing": (
                        "autoimage_ligand_anchor_receptor_fixed_then_receptor_fit"
                    ),
                },
                "validation": {
                    "prepared_reference_trajectory": {
                        "atom_count": 4,
                        "frame_count": 1,
                        "first_time_ps": 0.0,
                        "last_time_ps": 0.0,
                    },
                    "selected_trajectory": {
                        "atom_count": 4,
                        "first_time_ps": 0.0,
                    },
                },
            }
        }
        science = dict(resolve_protocol("production_3x5ns@2").stages["trajectory_qc"])
        science["expected_frames"] = 2
        attempt = self.root / "protocol-qc-v2-attempt"
        request = StageRequest(
            task=TaskKey(
                StageId.TRAJECTORY_QC,
                case_id="case-a",
                replica_id="rep01",
            ),
            science=science,
            runtime=self.runtime,
            prepared_case=prepared.binding_view,
            upstream={
                "trajectory": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=window.parent,
                    artifacts={
                        "prepared_trajectory": window,
                        "prepared_reference_trajectory": t0_reference,
                    },
                    payload=trajectory_payload,
                )
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        result = BuiltinStageService("trajectory_qc").execute(request)

        rendered = result.artifacts["cpptraj_input"].read_text(encoding="utf-8")
        self.assertIn(f'trajin "{window}"', rendered)
        self.assertIn(f'reference "{t0_reference}" 1', rendered)
        self.assertEqual(
            result.payload["reference_definition"],
            {
                "artifact": "prepared_reference_trajectory",
                "frame": 1,
                "semantics": "post_npt_production_start",
                "source_artifact": "selected_trajectory",
                "preprocessing": (
                    "autoimage_ligand_anchor_receptor_fixed_then_receptor_fit"
                ),
                "time_ps": 0.0,
            },
        )
        self.assertEqual(len(result.payload["check_results"]), 4)
        self.assertIs(result.qc_verdict, QcVerdict.PASS)
        missing_reference = StageInput(
            task=request.upstream["trajectory"].task,
            attempt_id="0001",
            root=window.parent,
            artifacts={"prepared_trajectory": window},
            payload=trajectory_payload,
        )
        missing_request = StageRequest(
            task=request.task,
            science=science,
            runtime=self.runtime,
            prepared_case=prepared.binding_view,
            upstream={"trajectory": missing_reference},
            attempt_dir=self.root / "protocol-qc-v2-missing-attempt",
            invocation_dir=self.root / "protocol-qc-v2-missing-attempt" / "invocations" / "0001",
        )
        with self.assertRaisesRegex(BackendConfigurationError, "prepared_reference_trajectory"):
            BuiltinStageService("trajectory_qc").execute(missing_request)

        bad_payload = deepcopy(trajectory_payload)
        bad_payload["backend_metadata"]["reference"]["source_artifact"] = (
            "prepared_trajectory"
        )
        bad_upstream = replace(request.upstream["trajectory"], payload=bad_payload)
        bad_request = replace(
            request,
            upstream={"trajectory": bad_upstream},
            attempt_dir=self.root / "protocol-qc-v2-bad-provenance",
            invocation_dir=(
                self.root / "protocol-qc-v2-bad-provenance" / "invocations" / "0001"
            ),
        )
        with self.assertRaisesRegex(BackendConfigurationError, "provenance source_artifact"):
            BuiltinStageService("trajectory_qc").execute(bad_request)

        bad_time_payload = deepcopy(trajectory_payload)
        bad_time_payload["backend_metadata"]["validation"][
            "prepared_reference_trajectory"
        ]["first_time_ps"] = 5.0
        bad_time_upstream = replace(
            request.upstream["trajectory"], payload=bad_time_payload
        )
        bad_time_request = replace(
            request,
            upstream={"trajectory": bad_time_upstream},
            attempt_dir=self.root / "protocol-qc-v2-bad-time",
            invocation_dir=(
                self.root / "protocol-qc-v2-bad-time" / "invocations" / "0001"
            ),
        )
        with self.assertRaisesRegex(BackendConfigurationError, "one frame at production t=0"):
            BuiltinStageService("trajectory_qc").execute(bad_time_request)

    def test_assets_use_verified_inputs_and_record_missing_replica(self) -> None:
        prepared = self.prepared_case()
        topology = prepared.artifact("complex_topology")
        coordinates = prepared.artifact("coordinates")
        trajectory = self.file("trajectory/centered.xtc")
        analysis_file = self.file("analysis/analysis.json")
        attempt = self.root / "assets-attempt"
        analysis_payload = {
            "cases": [
                {
                    "case_id": "case-a",
                    "record": {
                        "expected_replicas": ["rep01", "rep02"],
                    },
                }
            ]
        }
        request = StageRequest(
            task=TaskKey(StageId.ASSETS),
            science={"stride": 2},
            runtime={"cpptraj": self.runtime["tool"]},
            prepared_case=None,
            upstream={
                "analysis": StageInput(
                    task=TaskKey(StageId.ANALYSIS),
                    attempt_id="0001",
                    root=analysis_file.parent,
                    artifacts={"analysis": analysis_file},
                    payload=analysis_payload,
                ),
                "prepared:case-a": StageInput(
                    task=TaskKey(StageId.ADMIT, case_id="case-a"),
                    attempt_id="0001",
                    root=topology.parent,
                    artifacts={
                        "complex_topology": topology,
                        "coordinates": coordinates,
                    },
                    payload=prepared.manifest.as_dict(),
                ),
                "trajectory:case-a:rep01": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=trajectory.parent,
                    artifacts={
                        "centered_trajectory": trajectory,
                        "selected_trajectory": trajectory,
                    },
                ),
                "trajectory:case-a:rep02": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep02",
                    ),
                    attempt_id="0001",
                    root=trajectory.parent,
                    artifacts={"centered_trajectory": trajectory},
                ),
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        result = BuiltinStageService("report.assets").execute(request)

        index = json.loads(result.artifacts["asset_index"].read_text(encoding="utf-8"))
        self.assertEqual(index["availability"]["case-a"]["status"], "partial")
        self.assertEqual(
            index["availability"]["case-a"]["replicas"]["rep02"]["status"],
            "unavailable",
        )
        self.assertEqual(
            index["availability"]["case-a"]["replicas"]["rep02"]["reason"],
            "trajectory_artifact_missing:selected_trajectory",
        )
        self.assertIn("case-a.structure.rep01", result.artifacts)
        display_manifest = json.loads(
            result.artifacts["case-a.asset_manifest"].read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            display_manifest["components"],
            {"ligand": [3, 4], "receptor": [1, 2]},
        )
        self.assertEqual(
            display_manifest["pymol_scripts"],
            {"rep01": "load_rep01.pml"},
        )
        self.assertEqual(
            display_manifest["schema_version"],
            "mmpbsa.display_assets.v4",
        )
        self.assertEqual(
            display_manifest["equilibrated_pose_rmsd"]["alignment_atom_indices"],
            [1],
        )
        self.assertEqual(
            display_manifest["equilibrated_pose_rmsd"]["ligand_atom_indices"],
            [3],
        )
        self.assertEqual(
            display_manifest["equilibrated_pose_rmsd"]["reference_time_ps"],
            0.0,
        )
        self.assertNotIn("binder", display_manifest["components"])
        component_availability = index["availability"]["case-a"][
            "components"
        ]
        self.assertEqual(component_availability["projection"], "dry_scoring")
        self.assertEqual(
            component_availability["excluded"],
            {
                "status": "unavailable",
                "reason": "md_only_entities_excluded_from_dry_display",
                "entity_ids": ["structural_water"],
                "entity_kinds": ["water"],
            },
        )
        self.assertEqual(
            result.payload["schema_version"],
            "mmpbsa.report_assets.v3",
        )
        self.assertEqual(
            {item["artifact_role"] for item in result.payload["assets"]},
            set(result.artifacts).difference(
                {
                    "case-a.cpptraj_input.rep01",
                    "case-a.reference_input.rep01",
                }
            ),
        )
        hidden = {
            item["artifact_role"]
            for item in result.payload["assets"]
            if item["visibility"] == "hidden"
        }
        self.assertEqual(
            hidden,
            {
                "case-a.ligand_equilibrated_rmsd.rep01",
                "case-a.equilibrated_reference.rep01",
            },
        )
        diagnostic = result.payload["diagnostics"]["case-a"]["ligand_rmsd"]
        self.assertEqual(diagnostic["reference_artifact"], "selected_trajectory")
        self.assertEqual(diagnostic["reference_frame"], 1)
        self.assertEqual(diagnostic["reference_time_ps"], 0.0)
        self.assertEqual(
            diagnostic["reference_semantics"],
            "post_npt_production_start",
        )
        self.assertEqual(
            diagnostic["reference_preprocessing"],
            "autoimage_ligand_anchor_receptor_fixed",
        )
        self.assertEqual(
            diagnostic["alignment_selection"],
            "receptor_interface_backbone",
        )
        self.assertEqual(diagnostic["rmsd_selection"], "ligand_qc")
        self.assertEqual(
            diagnostic["replicas"]["rep01"]["columns"],
            ["Frame", "ligand_equilibrated"],
        )
        pymol_descriptors = [
            item
            for item in result.payload["assets"]
            if item["kind"] == "pymol_script"
        ]
        self.assertEqual(len(pymol_descriptors), 1)
        self.assertEqual(pymol_descriptors[0]["replica_id"], "rep01")

    def test_assets_select_explicit_holo_receptor_entities(self) -> None:
        base = self.prepared_case().manifest

        def entity(name, kind, md_atom, scoring_atom, md_residue, scoring_residue):
            atoms = EntitySelection((md_atom,), (scoring_atom,))
            selections = {"all": atoms, "heavy_atoms": atoms}
            if kind == "protein":
                selections.update({"backbone": atoms, "interface_backbone": atoms})
            return PreparedEntity(
                entity_id=name,
                kind=kind,
                md_atom_indices=(md_atom,),
                scoring_atom_indices=(scoring_atom,),
                residues=(
                    ResidueIdentity(
                        name, "ALA" if kind == "protein" else "HEM", md_residue, scoring_residue
                    ),
                ),
                named_selections=selections,
            )

        ligand = replace(base.entity("ligand"), residues=(ResidueIdentity("L_1", "LIG", 3, 3),))
        water = replace(
            base.entity("structural_water"), residues=(ResidueIdentity("W_1", "HOH", 4, None),)
        )
        manifest = replace(
            base,
            md=replace(base.md, atom_count=7),
            scoring=replace(base.scoring, atom_count=6, source_md_atom_indices=(1, 2, 3, 4, 6, 7)),
            entities=(
                entity("receptor_A", "protein", 1, 1, 1, 1),
                entity("receptor_B", "protein", 2, 2, 2, 2),
                ligand,
                water,
                entity("heme_A", "cofactor", 6, 5, 5, 4),
                entity("heme_B", "cofactor", 7, 6, 6, 5),
            ),
            primary_binding=replace(
                base.primary_binding,
                receptor_entity_ids=("heme_A", "heme_B", "receptor_A", "receptor_B"),
            ),
        )
        stage = StageInput(
            task=TaskKey(StageId.ADMIT, case_id="case-a"),
            attempt_id="0001",
            root=self.root,
            artifacts={},
            payload=manifest.as_dict(),
        )
        selected = ["receptor_A", "receptor_B", "heme_A", "heme_B", "ligand"]
        components, availability, diagnostics = _prepared_components(stage, selected, "case-a")
        self.assertEqual(
            components,
            {
                "receptor_A": (1,),
                "receptor_B": (2,),
                "heme_A": (5,),
                "heme_B": (6,),
                "ligand": (3, 4),
            },
        )
        self.assertEqual(availability["heme_A"]["entity_ids"], ["heme_A"])
        self.assertEqual(diagnostics["receptor_all"], (1, 2, 5, 6))
        with self.assertRaisesRegex(BackendConfigurationError, "components overlap"):
            _prepared_components(stage, ["receptor", "heme_A", "ligand"], "case-a")
        with self.assertRaisesRegex(BackendConfigurationError, "unsupported"):
            _prepared_components(stage, ["invented_entity"], "case-a")
        components, availability, _ = _prepared_components(
            stage, ["ligand", "structural_water"], "case-a"
        )
        self.assertNotIn("structural_water", components)
        self.assertEqual(
            availability["structural_water"]["reason"],
            "md_only_entities_excluded_from_dry_display",
        )

    def test_assets_reject_legacy_partition_payload(self) -> None:
        topology = self.file("legacy/complex.prmtop")
        trajectory = self.file("legacy/centered.xtc")
        analysis_file = self.file("legacy/analysis.json")
        attempt = self.root / "legacy-assets-attempt"
        request = StageRequest(
            task=TaskKey(StageId.ASSETS),
            science={},
            runtime={"cpptraj": self.runtime["tool"]},
            prepared_case=None,
            upstream={
                "analysis": StageInput(
                    task=TaskKey(StageId.ANALYSIS),
                    attempt_id="0001",
                    root=analysis_file.parent,
                    artifacts={"analysis": analysis_file},
                    payload={
                        "cases": [
                            {
                                "case_id": "case-a",
                                "record": {"expected_replicas": ["rep01"]},
                            }
                        ]
                    },
                ),
                "prepared:case-a": StageInput(
                    task=TaskKey(StageId.ADMIT, case_id="case-a"),
                    attempt_id="0001",
                    root=topology.parent,
                    artifacts={"complex_topology": topology},
                    payload={
                        "partition": {
                            "receptor": {"atom_indices": [1, 2]},
                            "binder": {"atom_indices": [3]},
                        }
                    },
                ),
                "trajectory:case-a:rep01": StageInput(
                    task=TaskKey(
                        StageId.TRAJECTORY,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=trajectory.parent,
                    artifacts={"centered_trajectory": trajectory},
                ),
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        with self.assertRaisesRegex(
            BackendConfigurationError,
            "mmpbsa.prepared_case.v2 payload; legacy partition payloads "
            "are not executable",
        ):
            BuiltinStageService("report.assets").execute(request)

    def test_multistage_md_aliases_primary_and_resumes_in_order(self) -> None:
        prepared = self.prepared_case().md_view
        self.assertNotIn(
            "index_role",
            resolve_protocol("production_3x5ns@1").stages["md"],
        )
        self.assertEqual(
            set(prepared.resolved_artifacts),
            {"md_coordinates", "md_topology"},
        )
        self.assertEqual(
            {artifact.role for artifact in prepared.system.artifacts},
            {"md_coordinates", "md_topology"},
        )
        self.assertFalse(hasattr(prepared, "binding"))
        self.assertFalse(hasattr(prepared, "entities"))
        science = {
            "_replica": {
                "replica_id": "rep01",
                "replica_index": 1,
                "seed": 4101,
            },
            "topology_role": "md_topology",
            "primary_stage": "production",
            "stages": [
                {
                    "stage_id": "em",
                    "mdp_text": "integrator = steep\n",
                    "coordinates": {
                        "kind": "prepared",
                        "role": "md_coordinates",
                    },
                    "expected_suffixes": ["tpr", "log", "edr", "gro"],
                },
                {
                    "stage_id": "nvt",
                    "mdp_text": "gen_seed = {{replica.seed}}\n",
                    "coordinates": {
                        "kind": "stage",
                        "stage_id": "em",
                        "artifact": "gro",
                    },
                    "reference": {
                        "kind": "prepared",
                        "role": "md_coordinates",
                    },
                    "expected_suffixes": [
                        "tpr",
                        "log",
                        "edr",
                        "gro",
                        "cpt",
                    ],
                    "checkpoint": True,
                },
                {
                    "stage_id": "npt",
                    "mdp_text": (
                        "continuation = yes\n; replica {{replica.id}}/{{replica.index}}\n"
                    ),
                    "coordinates": {
                        "kind": "stage",
                        "stage_id": "nvt",
                        "artifact": "gro",
                    },
                    "state_input": {
                        "kind": "stage",
                        "stage_id": "nvt",
                        "artifact": "cpt",
                    },
                    "expected_suffixes": [
                        "tpr",
                        "log",
                        "edr",
                        "gro",
                        "cpt",
                    ],
                    "checkpoint": True,
                },
                {
                    "stage_id": "production",
                    "mdp_text": "continuation = yes\n",
                    "coordinates": {
                        "kind": "stage",
                        "stage_id": "npt",
                        "artifact": "gro",
                    },
                    "state_input": {
                        "kind": "stage",
                        "stage_id": "npt",
                        "artifact": "cpt",
                    },
                    "expected_suffixes": [
                        "tpr",
                        "log",
                        "edr",
                        "gro",
                        "xtc",
                        "cpt",
                    ],
                    "checkpoint": True,
                },
            ],
        }
        attempt = self.root / "md-attempt"
        service = BuiltinStageService("md")
        md_runtime = {
            **self.runtime,
            "threads_per_job": 4,
            "gpu_enabled": True,
            "gpu_id": "leased-3",
            "stage_overrides": {
                "em": {
                    "threads_per_job": 2,
                    "gpu_id": "unleased-em",
                },
                "production": {
                    "threads_per_job": 8,
                    "gpu_id": "unleased-production",
                },
                "not_in_pipeline": {
                    "threads_per_job": 32,
                    "gpu_id": "unleased-unused",
                },
            },
        }

        def request(invocation: str, *, resume: bool = False) -> StageRequest:
            return StageRequest(
                task=TaskKey(
                    StageId.MD,
                    case_id="case-a",
                    replica_id="rep01",
                ),
                science=science,
                runtime=md_runtime,
                prepared_case=prepared,
                upstream={},
                attempt_dir=attempt,
                invocation_dir=attempt / "invocations" / invocation,
                resume=resume,
                checkpoint=(
                    attempt / "output" / "production" / "production.cpt" if resume else None
                ),
            )

        resources = service.resource_request(request("resource-probe"))
        self.assertEqual(resources.cpu_slots, 8)
        self.assertEqual(resources.gpu_slots, 1)

        initial = service.execute(request("0001"))
        self.assertEqual(
            initial.artifacts["xtc"],
            attempt / "output" / "production" / "production.xtc",
        )
        self.assertIn("em.gro", initial.artifacts)
        self.assertIn("nvt.cpt", initial.artifacts)
        backend_receipt = json.loads(
            Path(initial.invocation_provenance["receipt"]).read_text(encoding="utf-8")
        )
        grompp_commands = [
            command["argv"]
            for command in backend_receipt["commands"]
            if command["argv"][1] == "grompp"
        ]
        self.assertEqual(len(grompp_commands), 4)
        self.assertTrue(all("-n" not in argv for argv in grompp_commands))
        mdrun_commands = [
            command["argv"]
            for command in backend_receipt["commands"]
            if command["argv"][1] == "mdrun"
        ]
        self.assertEqual(len(mdrun_commands), 4)
        expected_threads = {
            "em": "2",
            "nvt": "4",
            "npt": "4",
            "production": "8",
        }
        for argv in mdrun_commands:
            stage_id = Path(argv[argv.index("-deffnm") + 1]).name
            self.assertEqual(
                argv[argv.index("-ntomp") + 1],
                expected_threads[stage_id],
            )
            self.assertEqual(
                argv[argv.index("-gpu_id") + 1],
                "leased-3",
            )
        rendered = initial.artifacts["nvt.mdp_input"].read_text(encoding="utf-8")
        self.assertIn("4101", rendered)
        for stage_id in ("em", "nvt", "npt", "production"):
            self.assertTrue((attempt / "output" / stage_id / "stage_complete.json").is_file())

        production_marker = attempt / "output" / "production" / "stage_complete.json"
        production_marker.unlink()
        missing_marker = service.resume(request("0002", resume=True))
        self.assertEqual(
            missing_marker.payload["recovery"]["production"],
            "checkpoint_append",
        )
        self.assertEqual(
            missing_marker.payload["skipped_complete_stages"],
            ["em", "nvt", "npt"],
        )
        stage_provenance = missing_marker.invocation_provenance["metadata"]["stage_provenance"]
        self.assertEqual(
            stage_provenance["em"]["source"],
            "completion_marker",
        )
        self.assertTrue(stage_provenance["em"]["commands"])
        self.assertIn(
            "Fake scientific tool 2024.2",
            stage_provenance["em"]["tool"]["raw_version"],
        )
        self.assertEqual(
            stage_provenance["production"]["source"],
            "current_invocation",
        )
        self.assertEqual(
            missing_marker.payload["stage_provenance"]["em"]["source"],
            "completion_marker",
        )

        production_marker.write_text("{not json\n", encoding="utf-8")
        corrupt_marker = service.resume(request("0003", resume=True))
        self.assertEqual(
            corrupt_marker.payload["recovery"]["production"],
            "checkpoint_append",
        )

        corrupt_marker.artifacts["xtc"].write_text(
            "corrupted but nonempty\n",
            encoding="utf-8",
        )
        corrupted_artifact = service.resume(request("0004", resume=True))
        self.assertEqual(
            corrupted_artifact.payload["recovery"]["production"],
            "checkpoint_append",
        )
        self.assertEqual(
            corrupted_artifact.payload["skipped_complete_stages"],
            ["em", "nvt", "npt"],
        )

        corrupted_artifact.artifacts["nvt.cpt"].unlink()
        restarted = service.resume(request("0005", resume=True))
        self.assertEqual(
            restarted.payload["recovery"]["nvt"],
            "restart_stage",
        )
        self.assertEqual(
            restarted.payload["recovery"]["production"],
            "restart_stage",
        )
        self.assertEqual(
            restarted.payload["recovery"]["npt"],
            "restart_stage",
        )

    def test_phase_plan_v2_reuses_pipeline_and_publishes_portable_aliases(
        self,
    ) -> None:
        prepared = self.prepared_case().md_view
        science = dict(resolve_protocol("smoke_20ps@2").stages["md"])
        science["_replica"] = {
            "replica_id": "rep01",
            "replica_index": 1,
            "seed": 4101,
        }
        attempt = self.root / "md-v2-attempt"
        request = StageRequest(
            task=TaskKey(StageId.MD, case_id="case-a", replica_id="rep01"),
            science=science,
            runtime={
                **self.runtime,
                "_stage_contract": "mmpbsa.md-phase-plan.v2",
                "_md_engine_profile": "gromacs_canonical@1",
                "mdrun_arguments": ["-cpt", "7.5"],
            },
            prepared_case=prepared,
            upstream={},
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        with patch(
            "mmpbsa.backends.services.validate_gromacs_phase_output",
            side_effect=lambda _science, phase_id, _artifacts: {
                "phase_id": phase_id
            },
        ) as validate:
            result = BuiltinStageService("md").execute(request)

        self.assertEqual(validate.call_count, 4)
        self.assertEqual(
            set(result.artifacts),
            {
                "full_trajectory",
                "coordinates",
                "state",
                "checkpoint",
                "energy_log",
            },
        )
        self.assertEqual(result.checkpoint, result.artifacts["checkpoint"])
        self.assertEqual(len(set(result.artifacts.values())), 5)
        self.assertTrue(
            all(path.parent.name == "public" for path in result.artifacts.values())
        )
        native = result.payload["phase_plan"]["native_artifacts"]
        self.assertEqual(
            result.artifacts["full_trajectory"].read_bytes(),
            Path(native["production.xtc"]).read_bytes(),
        )
        self.assertEqual(
            result.artifacts["coordinates"].read_bytes(),
            Path(native["production.gro"]).read_bytes(),
        )
        receipt = json.loads(
            Path(result.invocation_provenance["receipt"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("production.xtc", receipt["artifacts"])
        self.assertIn("production.mdp_input", receipt["artifacts"])
        self.assertNotIn("full_trajectory", receipt["artifacts"])
        self.assertNotIn("xtc", receipt["artifacts"])
        self.assertTrue(all("." in role for role in native))
        velocity_seed = result.payload["phase_plan"]["random_seeds"]["nvt"][
            "velocities"
        ]
        self.assertIn(
            f"gen-seed                  = {velocity_seed}",
            Path(native["nvt.mdp_input"]).read_text(encoding="utf-8"),
        )
        self.assertEqual(
            result.payload["phase_plan"]["source_contract"],
            "mmpbsa.md-phase-plan.v2",
        )
        self.assertEqual(
            result.payload["scientific_validation"]["production"],
            {"phase_id": "production"},
        )
        operational = result.payload["phase_plan"]["operational_mapping"]
        self.assertEqual(
            operational["classification"],
            "operational_resilience_not_ensemble_science",
        )
        self.assertEqual(
            operational["checkpoint"]["native_interval_minutes"],
            7.5,
        )
        self.assertFalse(
            operational["checkpoint"]["exact_step_schedule"]
        )
        self.assertEqual(
            operational["transaction_chunk"]["mapping_status"],
            "openmm_journal_hint",
        )

    def test_trajectory_v3_routes_full_system_projection_without_tpr(
        self,
    ) -> None:
        prepared = self.prepared_case()
        full_trajectory = self.file("md-v2/production.xtc")
        captured = []

        class FakeTrajectoryV3Backend:
            def run(self, request: object) -> BackendReceipt:
                captured.append(request)
                request.output_dir.mkdir(parents=True, exist_ok=True)
                artifacts = {
                    role: request.output_dir / filename
                    for role, filename in {
                        "selected_trajectory": "selected.xtc",
                        "prepared_trajectory": "prepared.nc",
                        "centered_trajectory": "centered.xtc",
                    }.items()
                }
                for path in artifacts.values():
                    path.write_text("trajectory-v3\n", encoding="utf-8")
                return BackendReceipt(
                    state=ReceiptState.COMPLETED,
                    commands=(),
                    artifacts=artifacts,
                    metadata={"contract": "mmpbsa.trajectory.v3"},
                )

        science = dict(resolve_protocol("smoke_20ps@2").stages["trajectory"])
        attempt = self.root / "trajectory-v3-attempt"
        request = StageRequest(
            task=TaskKey(
                StageId.TRAJECTORY,
                case_id="case-a",
                replica_id="rep01",
            ),
            science=science,
            runtime={"_stage_contract": "mmpbsa.trajectory.v3"},
            prepared_case=prepared.trajectory_view,
            upstream={
                "md": StageInput(
                    task=TaskKey(
                        StageId.MD,
                        case_id="case-a",
                        replica_id="rep01",
                    ),
                    attempt_id="0001",
                    root=full_trajectory.parent,
                    artifacts={"full_trajectory": full_trajectory},
                )
            },
            attempt_dir=attempt,
            invocation_dir=attempt / "invocations" / "0001",
        )

        with patch(
            "mmpbsa.backends.trajectory_dispatch.create_trajectory_v3_backend",
            return_value=FakeTrajectoryV3Backend(),
        ):
            result = BuiltinStageService("trajectory").execute(request)

        projected = captured[0]
        self.assertEqual(projected.full_trajectory, full_trajectory)
        self.assertEqual(
            projected.full_system_topology,
            prepared.artifact("md_topology"),
        )
        self.assertEqual(
            projected.full_system_coordinates,
            prepared.artifact("md_coordinates"),
        )
        self.assertEqual(
            projected.dry_complex_topology,
            prepared.artifact("complex_topology"),
        )
        self.assertEqual(projected.source_md_atom_indices, (1, 2, 3, 4))
        self.assertEqual(projected.expected_atom_order_signature, "a" * 64)
        self.assertEqual(
            projected.residue_crosswalk,
            ((1, 1, "ALA"), (2, 2, "LIG")),
        )
        self.assertEqual(projected.source_first_time_ps, 0.0)
        self.assertEqual(projected.source_interval_ps, 2.0)
        self.assertEqual(projected.source_interval_steps, 1_000)
        self.assertEqual(projected.expected_source_frames, 11)
        self.assertEqual(projected.autoimage_anchor_mask, "@3,4")
        self.assertEqual(projected.autoimage_fixed_mask, "@1,2")
        self.assertEqual(projected.fit_mask, "@1")
        self.assertEqual(
            result.payload["frame_manifest"]["selection"]["expected_frames"],
            11,
        )


if __name__ == "__main__":
    unittest.main()

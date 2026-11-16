from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
CASE_ROOT = ROOT / "modeling" / "cases" / "2HPL"
BUILD_SCRIPT = CASE_ROOT / "build.py"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mmpbsa.core import PreparedCaseDraftManifest  # noqa: E402


def _load_builder():
    spec = importlib.util.spec_from_file_location("sp2016_2hpl_build", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection(md: list[int], scoring: list[int]) -> dict[str, list[int]]:
    return {"md_atom_indices": md, "scoring_atom_indices": scoring}


def _scored_entity(
    entity_id: str,
    kind: str,
    atoms: list[int],
    residue_index: int,
    *,
    interface: bool = False,
) -> dict[str, object]:
    named = {
        "all": _selection(atoms, atoms),
        "heavy_atoms": _selection(atoms, atoms),
        "backbone": _selection(atoms, atoms),
    }
    if interface:
        named["interface_backbone"] = _selection(atoms, atoms)
    return {
        "entity_id": entity_id,
        "kind": kind,
        "md_atom_indices": atoms,
        "scoring_atom_indices": atoms,
        "residues": [
            {
                "residue_id": f"{entity_id}_1",
                "residue_name": "ALA",
                "md_residue_index": residue_index,
                "scoring_residue_index": residue_index,
            }
        ],
        "named_selections": named,
        "features": [],
        "metadata": {},
    }


def _md_only_entity(
    entity_id: str, kind: str, atoms: list[int], residue_index: int
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "kind": kind,
        "md_atom_indices": atoms,
        "scoring_atom_indices": [],
        "residues": [
            {
                "residue_id": f"{entity_id}_1",
                "residue_name": "WAT" if kind == "water" else "NA+",
                "md_residue_index": residue_index,
                "scoring_residue_index": None,
            }
        ],
        "named_selections": {
            "all": _selection(atoms, []),
            "heavy_atoms": _selection([atoms[0]], []),
        },
        "features": [],
        "metadata": {"scoring_policy": "excluded"},
    }


def _synthetic_inventory() -> dict[str, object]:
    return {
        "source_md_atom_indices": [1, 2, 3, 4],
        "entities": [
            _scored_entity("receptor", "protein", [1, 2], 1, interface=True),
            _scored_entity("ligand_peptide", "peptide", [3, 4], 2),
            _md_only_entity("bulk_ions", "ion", [5], 3),
            _md_only_entity("bulk_water", "water", [6, 7, 8], 4),
        ],
    }


def test_plan_fixes_source_assembly_termini_force_field_and_no_md() -> None:
    builder = _load_builder()
    plan = builder.load_build_plan()
    assembly = plan["source"]["biological_assembly"]
    assert assembly["assembly_id"] == "1"
    assert assembly["operator"] == "identity"
    assert assembly["label_asym_ids"] == ["A", "B", "C", "D"]
    assert plan["entities"]["receptor"]["termini"]["expected_amber_templates"] == {
        "n_terminal": "NSER",
        "c_terminal": "CSER",
    }
    assert plan["entities"]["ligand"]["termini"]["expected_amber_templates"] == {
        "n_terminal": "NASP",
        "c_terminal": "CGLY",
    }
    assert plan["force_field"] == {
        "protein_and_peptide": "ff14SB",
        "water": "TIP3P",
        "ion_parameters": ("Joung-Cheatham monovalent ions loaded by leaprc.water.tip3p"),
        "pb_radii": "mbondi2",
    }
    assert plan["outputs"]["starts_md"] is False
    assert plan["outputs"]["target_must_not_exist"] is True
    assert plan["outputs"]["transactional_publish"] is True


def test_pure_draft_renderer_satisfies_current_core_contract() -> None:
    builder = _load_builder()
    draft = builder.make_draft_manifest(
        builder.load_build_plan(),
        _synthetic_inventory(),
        {"owner": "validation-local-test"},
    )
    manifest = PreparedCaseDraftManifest.from_mapping(draft)
    assert manifest.schema_version == "mmpbsa.prepared_case_draft.v1"
    assert manifest.case_id == "sp2016_16"
    assert manifest.md.backend == "gromacs"
    assert manifest.md.metadata["md_started"] is False
    assert manifest.scoring.backend == "ambertools"
    assert manifest.scoring.radii_model == "mbondi2"
    assert {artifact.role for artifact in manifest.md.artifacts} == {
        "md_topology",
        "md_coordinates",
        "solvated_topology",
        "index",
        "preparation_receipt",
    }
    solvated_topology = next(
        artifact for artifact in manifest.md.artifacts if artifact.role == "solvated_topology"
    )
    assert solvated_topology.source == Path("intermediate/solvated.prmtop")
    assert solvated_topology.path == Path("md/solvated.prmtop")
    assert solvated_topology.media_type == "chemical/x-amber-topology"
    assert {artifact.role for artifact in manifest.scoring.artifacts} == {
        "complex_topology",
        "receptor_topology",
        "ligand_topology",
        "coordinates",
    }
    entities = {entity.entity_id: entity for entity in manifest.entities}
    assert entities["ligand_peptide"].kind == "peptide"
    assert entities["ligand_peptide"].default_qc_selection == "backbone"
    assert manifest.primary_binding.receptor_entity_ids == ("receptor",)
    assert manifest.primary_binding.ligand_entity_ids == ("ligand_peptide",)
    assert manifest.primary_binding.excluded_entity_ids == ("bulk_ions", "bulk_water")


def test_create_only_transaction_publishes_atomically_and_refuses_reuse(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    destination = tmp_path / "proof"
    with builder._create_only_transaction(destination) as staging:
        (staging / "sentinel.txt").write_text("complete\n", encoding="utf-8")
        assert not destination.exists()
    assert (destination / "sentinel.txt").read_text(encoding="utf-8") == "complete\n"
    with pytest.raises(builder.BuildError, match="must not already exist"):
        with builder._create_only_transaction(destination):
            pass


@pytest.mark.skipif(
    os.environ.get("RUN_SP2016_2HPL_INTEGRATION") != "1",
    reason="set RUN_SP2016_2HPL_INTEGRATION=1 to run AmberTools/ParmEd build+seal",
)
def test_integration_builds_and_seals_without_starting_md(tmp_path: Path) -> None:
    assert Path(sys.executable).is_file()
    output = tmp_path / "2hpl_build"
    completed = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    cli_summary = json.loads(completed.stdout)
    assert cli_summary["output"] == str(output)
    receipt = json.loads(
        (output / "provenance" / "preparation_receipt.json").read_text(encoding="utf-8")
    )
    seal = json.loads((output / "provenance" / "seal_summary.json").read_text(encoding="utf-8"))
    assert receipt["md_started"] is False
    assert receipt["source"]["biological_assembly"]["operator"] == "identity"
    assert receipt["inventory"]["scoring_atom_count"] == 1657
    assert receipt["inventory"]["md_atom_count"] > 1657
    assert {item["amber_template"] for item in receipt["dry"]["terminal_checks"].values()} == {
        "NSER",
        "CSER",
        "NASP",
        "CGLY",
    }
    assert all(command["return_code"] == 0 for command in receipt["commands"])
    assert not (output / "leap.log").exists()
    assert (output / "logs" / "tleap_probe.engine.log").is_file()
    assert (output / "logs" / "tleap_final.engine.log").is_file()
    assert seal["projection_admission"]["accepted"] is True
    assert seal["projection_admission"]["issues"] == []
    assert seal["md_started"] is False
    sealed_manifest = output / "sealed" / "prepared_case.json"
    assert sealed_manifest.is_file()
    sealed = json.loads(sealed_manifest.read_text(encoding="utf-8"))
    md_artifacts = {artifact["role"]: artifact for artifact in sealed["md"]["artifacts"]}
    assert "solvated_topology" in md_artifacts
    assert md_artifacts["solvated_topology"]["path"] == "md/solvated.prmtop"
    assert (output / "sealed" / "md" / "solvated.prmtop").is_file()

    import parmed as pmd

    from mmpbsa.backends.trajectory_v3 import _validate_topology_projection
    from mmpbsa.core import load_prepared_case

    prepared = load_prepared_case(sealed_manifest, expected_case_id="sp2016_16")
    trajectory_view = prepared.trajectory_view
    full_topology = pmd.load_file(str(prepared.artifact("solvated_topology")))
    dry_topology = pmd.load_file(str(prepared.artifact("complex_topology")))
    signature, projection_digest = _validate_topology_projection(
        full_topology,
        dry_topology,
        tuple(index - 1 for index in trajectory_view.source_md_atom_indices),
        expected_atom_order_signature=trajectory_view.atom_order_signature,
        residue_crosswalk=trajectory_view.residue_crosswalk,
    )
    assert signature == trajectory_view.atom_order_signature
    assert len(projection_digest) == 64

from __future__ import annotations

from pathlib import Path

import yaml

from mmpbsa.core import protocol_preset

ROOT = Path(__file__).resolve().parents[1]


def _yaml(name: str) -> dict[str, object]:
    value = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_production_validation_uses_the_scientific_3x5ns_contract() -> None:
    workflow = _yaml("workflow.yaml")
    protocol = workflow["protocol"]
    assert isinstance(protocol, dict)

    assert workflow["schema_version"] == "mmpbsa.workflow.v3"
    assert protocol == {
        "preset": "production_3x5ns@1",
        "models": ["gb", "pb"],
        "completion_policy": "all",
    }
    assert [replica["replica_id"] for replica in workflow["replicas"]] == [
        "rep01",
        "rep02",
        "rep03",
    ]
    preset = protocol_preset("production_3x5ns@1")
    assert preset.frame_plan.production_ps == 5_000
    assert preset.frame_plan.expected_frames == 101


def test_tyk2_ligand_and_dry_scoring_roles_are_explicit_before_md() -> None:
    preparation = _yaml("preparation.yaml")
    entities = preparation["entities"]
    assert isinstance(entities, dict)
    ligand = entities["ligand"]
    assert isinstance(ligand, dict)

    assert ligand["entity_id"] == "ligand"
    assert ligand["kind"] == "small_molecule"
    assert ligand["atom_type"] == "gaff2"
    waters = entities["crystal_waters"]
    assert isinstance(waters, list) and len(waters) == 2

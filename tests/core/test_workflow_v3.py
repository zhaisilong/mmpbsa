from __future__ import annotations

from pathlib import Path

import pytest

from mmpbsa.core.config import load_workflow
from mmpbsa.core.errors import ConfigurationError


def _workflow(
    *,
    preset: str = "production_3x50ns@1",
    protocol_extra: str = "",
    replicas: str = """
  - {replica_id: rep01, replica_index: 1, seed: 41001}
  - {replica_id: rep02, replica_index: 2, seed: 41002}
  - {replica_id: rep03, replica_index: 3, seed: 41003}
""",
) -> str:
    return f"""
schema_version: mmpbsa.workflow.v3
project_id: typed_protocol
protocol:
  preset: {preset}
{protocol_extra}cases:
  - case_id: ligand_a
    prepared_case: prepared/ligand_a/prepared_case.json
    reference:
      experimental_dg_kj_mol: -35.0
replicas:
{replicas}analysis:
  reference_key: experimental_dg_kj_mol
report:
  title: Typed protocol test
  metadata:
    Dataset: test
""".lstrip()


def test_v3_workflow_expands_package_preset_without_python_profile(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(_workflow(), encoding="utf-8")

    workflow = load_workflow(path)

    assert workflow.schema_version == "mmpbsa.workflow.v3"
    assert workflow.protocol_id == "production_3x50ns_v1"
    assert workflow.profile.profile_id == "production_3x50ns"
    assert workflow.profile.profile_version == "1"
    assert workflow.resolved_protocol is not None
    assert workflow.resolved_protocol.preset.frame_plan.expected_frames == 1001
    assert workflow.completion_policy == "all"
    assert workflow.scoring_model_ids == ("gb", "pb")
    trajectory = workflow.stages["trajectory"]
    assert trajectory["start_frame"] == 501
    assert trajectory["end_frame"] == 2501
    assert trajectory["stride"] == 2
    assert trajectory["expected_frames"] == 1001
    assert trajectory["autoimage_anchor_mask"] == "{{selection:ligand_all}}"
    assert trajectory["autoimage_fixed_mask"] == "{{selection:receptor_all}}"
    assert (
        trajectory["fit_mask"]
        == "{{selection:receptor_interface_backbone}}"
    )
    assert workflow.report["title"] == "Typed protocol test"
    assert workflow.report["metadata"] == {
        "Replicas": 3,
        "Production per replica": "50 ns",
        "Frames per model and replica": 1001,
        "Dataset": "test",
    }


def test_v3_workflow_can_select_only_pb(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        _workflow(
            preset="production_3x100ns@1",
            protocol_extra="  models: [pb]\n  completion_policy: all\n",
        ),
        encoding="utf-8",
    )

    workflow = load_workflow(path)

    assert workflow.scoring_model_ids == ("pb",)
    assert tuple(workflow.analysis["models"]) == ("pb", "pb_dmm")
    assert workflow.analysis["models"]["pb_dmm"]["source_model_id"] == "pb"
    assert workflow.report["model_order"] == ["pb", "pb_dmm"]
    assert workflow.scoring_model("pb")["expected_frames"] == 2001


@pytest.mark.parametrize(
    ("text", "message"),
    (
        (
            _workflow(replicas="""
  - {replica_id: rep01, replica_index: 1, seed: 41001}
"""),
            "requires exactly 3 replicas",
        ),
        (
            _workflow(replicas="""
  - {replica_id: rep01, replica_index: 1, seed: 41001}
  - {replica_id: rep02, replica_index: 2, seed: 41001}
  - {replica_id: rep03, replica_index: 3, seed: 41003}
"""),
            "seeds must be unique",
        ),
        (
            _workflow(protocol_extra="  control_text: raw-native-input\n"),
            "unknown fields: control_text",
        ),
        (
            _workflow(protocol_extra="  overrides: {duration_ns: 7}\n"),
            "unknown fields: duration_ns",
        ),
        (
            _workflow().replace(
                "analysis:\n  reference_key: experimental_dg_kj_mol\n",
                "analysis:\n  models: {gb: {}}\n",
            ),
            "unknown fields: models",
        ),
    ),
)
def test_v3_workflow_rejects_ambiguous_or_raw_protocol_inputs(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_workflow(path)

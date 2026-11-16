from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mmpbsa.core.config import load_runtime, load_workflow
from mmpbsa.core.errors import ConfigurationError
from mmpbsa.core.legacy import inspect_legacy_workflow
from mmpbsa.core.state import StageId
from mmpbsa.utils import sha256_digest


def _workflow_text(*, protocol_extra: str = "", extra: str = "") -> str:
    return f"""
schema_version: mmpbsa.workflow.v3
project_id: tyk2_validation
protocol:
  preset: production_3x5ns@1
  models: [gb, pb]
  completion_policy: any
{protocol_extra}
cases:
  - case_id: lig_ejm_31
    prepared_case: prepared/lig_ejm_31/prepared_case.json
    label: EJM 31
    reference:
      experimental_dg_kcal_mol: -8.4
replicas:
  - replica_id: rep01
    replica_index: 1
    seed: 31001
  - replica_id: rep02
    replica_index: 2
    seed: 31002
  - replica_id: rep03
    replica_index: 3
    seed: 31003
analysis:
  reference_key: experimental_dg_kcal_mol
  aggregation: mean
report:
  title: TYK2 acceptance
{extra}""".lstrip()


def _runtime_text(*, extra_provider: str = "") -> str:
    return f"""
schema_version: mmpbsa.runtime.v1
providers:
  admission: scientific
  md: scientific
  trajectory: scientific
  trajectory_qc: scientific
  scoring: scientific
  report: report
{extra_provider}scheduler:
  gpu_ids: ["2", "3"]
  max_workers: 2
  threads_per_job: 4
  cpu_slots: 3
backends:
  scientific:
    gmx: /opt/gromacs/bin/gmx
  report:
    theme: light
environment:
  GMX_MAXBACKUP: "-1"
  SCRATCH_TEMPLATE: "${{TMPDIR}}/mmpbsa"
""".lstrip()


def test_workflow_and_runtime_have_strict_separate_contracts(
    tmp_path: Path,
) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    runtime_path = tmp_path / "runtime.yaml"
    workflow_path.write_text(_workflow_text(), encoding="utf-8")
    runtime_path.write_text(_runtime_text(), encoding="utf-8")

    workflow = load_workflow(workflow_path)
    runtime = load_runtime(runtime_path)

    assert workflow.schema_version == "mmpbsa.workflow.v3"
    assert runtime.schema_version == "mmpbsa.runtime.v1"
    assert workflow.source_path == workflow_path.resolve()
    assert workflow.cases[0].prepared_case == (
        tmp_path / "prepared/lig_ejm_31/prepared_case.json"
    ).resolve()
    assert workflow.profile.profile_id == "production_3x5ns"
    assert workflow.profile.profile_version == "1"
    assert workflow.resolved_protocol is not None
    assert workflow.scoring_model_ids == ("gb", "pb")
    assert workflow.completion_policy == "any"
    assert workflow.resolved_protocol.md_engine_profile == "gromacs_canonical@1"
    assert workflow.scoring_model("pb")["expected_frames"] == 101
    assert runtime.scheduler.gpu_ids == ("2", "3")
    assert runtime.scheduler.cpu_slots == 3
    assert runtime.stage_runtime("md", gpu_id="3") == {
        "gmx": "/opt/gromacs/bin/gmx",
        "environment": {
            "GMX_MAXBACKUP": "-1",
            "SCRATCH_TEMPLATE": "${TMPDIR}/mmpbsa",
        },
        "threads_per_job": 4,
        "cpu_slots_total": 3,
        "gpu_enabled": True,
        "gpu_id": "3",
    }


@pytest.mark.parametrize("gpu_id", ("", " 0", "0 ", "0,1"))
def test_scheduler_rejects_ambiguous_gpu_lease_tokens(
    tmp_path: Path,
    gpu_id: str,
) -> None:
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        _runtime_text().replace('["2", "3"]', f'["{gpu_id}"]'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="each name one"):
        load_runtime(runtime_path)


def test_workflow_selects_openmm_profile_for_v2_protocol(
    tmp_path: Path,
) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        _workflow_text(
            protocol_extra="  md_engine_profile: openmm_8_5@2\n",
        ).replace("production_3x5ns@1", "production_3x5ns@2"),
        encoding="utf-8",
    )

    workflow = load_workflow(workflow_path)

    assert workflow.protocol_id == "production_3x5ns_v2"
    assert workflow.resolved_protocol is not None
    assert workflow.resolved_protocol.md_engine_profile == "openmm_8_5@2"
    assert workflow.stages["md"]["md_engine_profile"] == "openmm_8_5@2"


def test_old_openmm_profile_is_read_only_archive_not_executable(
    tmp_path: Path,
) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        _workflow_text(
            protocol_extra="  md_engine_profile: openmm_8_5@1\n",
        ).replace("production_3x5ns@1", "production_3x5ns@2"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="archived read-only"):
        load_workflow(workflow_path)
    inventory = inspect_legacy_workflow(workflow_path)
    assert inventory["schema_version"] == "mmpbsa.workflow.v3"
    assert inventory["executable"] is False
    assert inventory["writes"] is False


@pytest.mark.parametrize(
    ("workflow_text", "message"),
    [
        (
            _workflow_text(extra="scheduler: {max_workers: 99}\n"),
            "unknown fields: scheduler",
        ),
        (
            _workflow_text(extra="  metadata: {scratch: '${HOME}/science'}\n"),
            "environment interpolation",
        ),
        (
            _workflow_text(extra="stages: {md: {}}\n"),
            "unknown fields: stages",
        ),
    ],
)
def test_workflow_rejects_runtime_leakage_and_incomplete_science(
    tmp_path: Path,
    workflow_text: str,
    message: str,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_workflow(path)


def test_runtime_rejects_science_or_unknown_capability(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        _runtime_text(extra_provider="  free_energy_magic: scientific\n"),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="unknown capabilities: free_energy_magic",
    ):
        load_runtime(path)


def test_runtime_environment_layers_are_merged_without_loss(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        """
schema_version: mmpbsa.runtime.v1
providers:
  md: scientific
scheduler: {}
backends:
  scientific:
    common:
      environment:
        GLOBAL_OVERRIDE: common
        COMMON_ONLY: retained
    md:
      environment:
        GLOBAL_OVERRIDE: stage
        STAGE_ONLY: retained
environment:
  GLOBAL_OVERRIDE: global
  GLOBAL_ONLY: retained
""".lstrip(),
        encoding="utf-8",
    )

    runtime = load_runtime(path)

    assert runtime.stage_runtime("md")["environment"] == {
        "GLOBAL_OVERRIDE": "stage",
        "GLOBAL_ONLY": "retained",
        "COMMON_ONLY": "retained",
        "STAGE_ONLY": "retained",
    }
    assert runtime.stage_runtime("md")["cpu_slots_total"] == 1
    assert runtime.stage_runtime("md")["gpu_enabled"] is False


def test_runtime_gpu_lease_overrides_provider_and_md_stage_hints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        """
schema_version: mmpbsa.runtime.v1
providers:
  md: scientific
scheduler:
  gpu_ids: ["3"]
  threads_per_job: 4
  cpu_slots: 8
backends:
  scientific:
    common:
      gpu_id: common-unleased
    md:
      gpu_id: md-unleased
      stage_overrides:
        em:
          threads_per_job: 2
          gpu_id: em-unleased
        production:
          threads_per_job: 8
          gpu_id: production-unleased
environment: {}
""".lstrip(),
        encoding="utf-8",
    )

    runtime = load_runtime(path)

    before_lease = runtime.stage_runtime("md")
    assert "gpu_id" not in before_lease
    assert before_lease["stage_overrides"] == {
        "em": {"threads_per_job": 2},
        "production": {"threads_per_job": 8},
    }
    leased = runtime.stage_runtime("md", gpu_id="3")
    assert leased["gpu_id"] == "3"
    assert all(
        "gpu_id" not in override
        for override in leased["stage_overrides"].values()
    )
    # Building effective runtime views must not mutate the loaded document.
    assert (
        runtime.backends["scientific"]["md"]["stage_overrides"]["production"][
            "gpu_id"
        ]
        == "production-unleased"
    )


def test_v2_profile_factory_is_never_imported(tmp_path: Path) -> None:
    marker = tmp_path / "profile-imported"
    (tmp_path / "malicious_profile.py").write_text(
        """
from pathlib import Path
Path(__file__).with_name("profile-imported").write_text("executed", encoding="utf-8")

def make_profile(context):
    raise RuntimeError("legacy profile execution must be unreachable")
""".lstrip(),
        encoding="utf-8",
    )
    path = tmp_path / "workflow.yaml"
    path.write_text(
        """
schema_version: mmpbsa.workflow.v2
profile:
  factory: malicious_profile.py:make_profile
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="read-only legacy configuration.*workspace inspect",
    ):
        load_workflow(path)
    assert not marker.exists()


def test_nested_report_assets_settings_are_delivered_only_to_assets_stage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        _workflow_text(
            extra="""  components: [legacy_receptor, ligand]
  alignment_mask: '@CA'
  default_replica: rep01
  display: {style: legacy}
  assets:
    components: [receptor_A, receptor_B, heme_A, heme_B, ligand]
    default_replica: rep02
    display: {style: nested}
    stride: 5
""",
        ),
        encoding="utf-8",
    )
    workflow = load_workflow(path)
    original_report = copy.deepcopy(workflow.report)
    generation = original_report["assets"]
    assert isinstance(generation, dict)

    science = workflow.stage_science(StageId.ASSETS)

    assert science == {**original_report, **generation}
    assert science["components"] == [
        "receptor_A", "receptor_B", "heme_A", "heme_B", "ligand"
    ]
    assert science["default_replica"] == "rep02"
    assert science["display"] == {"style": "nested"}
    assert science["alignment_mask"] == "@CA"
    assert science["title"] == "TYK2 acceptance"
    assert science["stride"] == 5
    assert workflow.stage_science(StageId.REPORT) is workflow.report
    assert workflow.report == original_report
    assert sha256_digest(science) != sha256_digest(original_report)


def test_nested_report_assets_selection_changes_stage_science_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        _workflow_text(extra="  assets: {components: [receptor, ligand]}\n"),
        encoding="utf-8",
    )
    first = load_workflow(path)
    path.write_text(
        _workflow_text(
            extra="  assets: {components: [receptor_A, receptor_B, heme_A, heme_B, ligand]}\n"
        ),
        encoding="utf-8",
    )
    second = load_workflow(path)

    assert first.stage_science(StageId.ASSETS)["components"] == ["receptor", "ligand"]
    assert second.stage_science(StageId.ASSETS)["components"] == [
        "receptor_A", "receptor_B", "heme_A", "heme_B", "ligand"
    ]
    assert sha256_digest(first.stage_science(StageId.ASSETS)) != sha256_digest(
        second.stage_science(StageId.ASSETS)
    )


@pytest.mark.parametrize(
    "assets_configuration",
    [
        pytest.param("", id="legacy-top-level-settings"),
        pytest.param(
            "  assets:\n    - role: custom\n      label: Custom display\n",
            id="legacy-descriptor-list",
        ),
    ],
)
def test_report_asset_legacy_settings_and_descriptors_are_not_flattened(
    tmp_path: Path, assets_configuration: str
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        _workflow_text(
            extra=(
                "  components: [receptor, ligand]\n"
                "  alignment_mask: '@CA'\n"
                "  stride: 2\n"
                + assets_configuration
            ),
        ),
        encoding="utf-8",
    )
    workflow = load_workflow(path)

    assert workflow.stage_science(StageId.ASSETS) == workflow.report
    assert workflow.stage_science(StageId.REPORT) is workflow.report
    assert workflow.stage_science(StageId.ASSETS)["components"] == [
        "receptor", "ligand"
    ]
    if assets_configuration:
        assert workflow.report["assets"] == [
            {"role": "custom", "label": "Custom display"}
        ]


@pytest.mark.parametrize(
    "configuration",
    [
        pytest.param(
            "  assets: {components: [receptor_A, receptor_B, heme_A, heme_B, ligand]}\n"
            "  asset_links: [{role: custom, label: Custom display}]\n",
            id="generation-map-and-canonical-links",
        ),
        pytest.param(
            "  assets: [{role: legacy}]\n  asset_links: [{role: custom}]\n",
            id="legacy-list-and-canonical-links",
        ),
        pytest.param("  assets: {}\n  asset_links: []\n", id="empty-map"),
        pytest.param("  assets: []\n  asset_links: []\n", id="empty-list"),
        pytest.param("  assets: null\n  asset_links: null\n", id="null-as-omitted"),
    ],
)
def test_report_asset_configuration_accepts_supported_forms(
    tmp_path: Path, configuration: str
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(_workflow_text(extra=configuration), encoding="utf-8")

    workflow = load_workflow(path)

    assert "assets" in workflow.report
    assert "asset_links" in workflow.report
    assert workflow.stage_science(StageId.REPORT) is workflow.report


@pytest.mark.parametrize(
    ("configuration", "field"),
    [
        ("  assets: corrupt\n", "assets"),
        ("  assets: true\n", "assets"),
        ("  assets: 42\n", "assets"),
        ("  assets: [corrupt]\n", "assets"),
        ("  assets: [{role: custom}, false]\n", "assets"),
        ("  assets: {1: corrupt}\n", "assets"),
        ("  assets: [{1: corrupt}]\n", "assets"),
        ("  asset_links: {role: custom}\n", "asset_links"),
        ("  asset_links: corrupt\n", "asset_links"),
        ("  asset_links: 42\n", "asset_links"),
        ("  asset_links: [corrupt]\n", "asset_links"),
        ("  asset_links: [{role: custom}, false]\n", "asset_links"),
        ("  asset_links: [{1: corrupt}]\n", "asset_links"),
    ],
)
def test_report_asset_configuration_rejects_malformed_forms_at_load(
    tmp_path: Path, configuration: str, field: str
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(_workflow_text(extra=configuration), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=rf"workflow\.report\.{field}"):
        load_workflow(path)

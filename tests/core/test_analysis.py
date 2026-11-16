from __future__ import annotations

import copy
import csv
from types import SimpleNamespace

import pytest

from mmpbsa.core.analysis import (
    build_project_analysis,
    extract_score,
    write_project_analysis,
)
from mmpbsa.core.errors import ContractError


def _dual_model_payload() -> dict[str, object]:
    return {
        "summary": {
            "rows": [
                {
                    "model_id": "gb",
                    "component": "delta",
                    "label": "DELTA TOTAL",
                    "average": -10.0,
                },
                {
                    "model_id": "pb",
                    "component": "delta",
                    "label": "DELTA TOTAL",
                    "average": -20.0,
                },
            ]
        }
    }


def _model_payload(model_id: str) -> dict[str, object]:
    score = {"gb": -10.0, "pb": -20.0}[model_id]
    return {
        "model_id": model_id,
        "summary": {
            "rows": [
                {
                    "model_id": model_id,
                    "component": "delta",
                    "label": "DELTA TOTAL",
                    "average": score,
                }
            ]
        },
    }


def test_implicit_score_extraction_rejects_dual_model_ambiguity() -> None:
    with pytest.raises(ContractError, match="ambiguous"):
        extract_score(_dual_model_payload())


def _workflow() -> SimpleNamespace:
    return SimpleNamespace(
        project_id="selection_test",
        protocol_id="gb_pb_v2",
        profile=SimpleNamespace(profile_id="test", profile_version="1"),
        analysis={
            "reference_key": "experimental_dg_kj_mol",
            "models": {
                "gb": {
                    "score_selector": {
                        "component": "delta",
                        "label": "DELTA TOTAL",
                        "value_key": "average",
                    },
                    "score_transform": {
                        "scale": 4.184,
                        "offset": 0.0,
                        "input_unit": "kcal/mol",
                        "output_unit": "kJ/mol",
                    },
                },
                "pb": {
                    "score_selector": {
                        "component": "delta",
                        "label": "DELTA TOTAL",
                        "value_key": "average",
                    },
                    "score_transform": {
                        "scale": 4.184,
                        "offset": 0.0,
                        "input_unit": "kcal/mol",
                        "output_unit": "kJ/mol",
                    },
                },
            },
        },
        report={"model_order": ["gb", "pb"]},
        cases=(
            SimpleNamespace(
                case_id="case01",
                label="Case 01",
                reference={"experimental_dg_kj_mol": -42.0},
            ),
        ),
    )


def _record(*, include_pb: bool = True) -> dict[str, object]:
    scoring_models: dict[str, object] = {
        "gb": {
            "model_id": "gb",
            "task": {"model_id": "gb"},
            "payload": _model_payload("gb"),
        }
    }
    if include_pb:
        scoring_models["pb"] = {
            "model_id": "pb",
            "task": {"model_id": "pb"},
            "payload": _model_payload("pb"),
        }
    return {
        "case_id": "case01",
        "qc_verdict": "PASS",
        "outcome": "complete",
        "replicas": [
            {
                "replica_id": "rep01",
                "completed_models": list(scoring_models),
                "scoring_models": scoring_models,
                "trajectory_qc": {
                    "payload": {},
                    "qc_verdict": "PASS",
                },
            }
        ],
    }


def test_analysis_selects_models_independently_and_converts_units() -> None:
    workflow = _workflow()
    records = (
        _record(),
    )

    analysis = build_project_analysis(workflow, records)

    assert analysis["schema_version"] == "mmpbsa.analysis.v3"
    assert analysis["model_order"] == ["gb", "pb"]
    case = analysis["cases"][0]
    assert case["models"]["gb"]["replica_scores"] == [-41.84]
    assert case["models"]["gb"]["score_mean"] == pytest.approx(-41.84)
    assert case["models"]["pb"]["replica_scores"] == [-83.68]
    assert case["models"]["pb"]["score_mean"] == pytest.approx(-83.68)
    assert case["models"]["gb"]["score_unit"] == "kJ/mol"
    replica = case["record"]["replicas"][0]
    assert (
        replica["scoring_models"]["gb"]["selected_score_raw"] == -10.0
    )
    assert replica["scoring_models"]["gb"]["selected_score"] == pytest.approx(
        -41.84
    )
    assert (
        replica["scoring_models"]["pb"]["selected_score_raw"] == -20.0
    )
    assert analysis["models"]["gb"]["model_id"] == "gb"
    assert analysis["models"]["gb"]["correlation"]["model_id"] == "gb"
    assert analysis["models"]["pb"]["correlation"]["model_id"] == "pb"


def test_analysis_does_not_infer_an_occupancy_basin_from_qc_rmsd() -> None:
    record = _record()
    record["replicas"][0]["trajectory_qc"]["payload"] = {
        "ligand_rmsd": {
            "columns": ["Frame", "ligand_primary"],
            "rows": [
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 1.1],
                [4.0, 2.0],
                [5.0, 2.5],
            ],
        }
    }

    analysis = build_project_analysis(_workflow(), (record,))
    case_record = analysis["cases"][0]["record"]
    assert "basin" not in case_record
    assert "basin_note" not in case_record


def test_missing_model_remains_unavailable_without_cross_model_fallback() -> None:
    analysis = build_project_analysis(_workflow(), (_record(include_pb=False),))

    case = analysis["cases"][0]
    assert case["models"]["gb"]["replica_count"] == 1
    assert case["models"]["pb"]["replica_scores"] == []
    assert case["models"]["pb"]["score_mean"] is None
    assert case["models"]["gb"]["comparison_exclusion_reasons"] == [
        "dielectric_unresolved"
    ]
    assert case["models"]["pb"]["comparison_exclusion_reasons"] == [
        "model_incomplete",
        "model_score_missing",
    ]
    replica = case["record"]["replicas"][0]
    assert "pb" not in replica["scoring_models"]


def test_claimed_completed_model_requires_its_own_attempt_record() -> None:
    record = _record(include_pb=False)
    record["replicas"][0]["completed_models"] = ["gb", "pb"]

    with pytest.raises(ContractError, match="claims model 'pb' complete"):
        build_project_analysis(_workflow(), (record,))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", "trajectory_qc"),
        ("case_id", "another_case"),
        ("replica_id", "rep99"),
        ("model_id", "pb"),
    ],
)
def test_scoring_attempt_task_identity_must_match_finalized_record(
    field: str, value: str
) -> None:
    record = _record()
    record["replicas"][0]["scoring_models"]["gb"]["task"][field] = value
    with pytest.raises(ContractError, match=f"mismatched {field}"):
        build_project_analysis(_workflow(), (record,))


def test_scoring_attempt_model_id_cannot_impersonate_another_model() -> None:
    record = _record()
    record["replicas"][0]["scoring_models"]["gb"]["model_id"] = "pb"
    with pytest.raises(ContractError, match="mismatched model_id"):
        build_project_analysis(_workflow(), (record,))


def test_analysis_case_csv_is_one_row_per_case_and_model(tmp_path) -> None:
    analysis = build_project_analysis(_workflow(), (_record(),))

    artifacts = write_project_analysis(tmp_path, analysis)
    with artifacts["case_table"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["model_id"] for row in rows] == ["gb", "pb"]
    assert float(rows[0]["score_mean"]) == pytest.approx(-41.84)
    assert rows[0]["comparison_eligible"] == "False"
    assert rows[0]["comparison_exclusion_reasons"] == "dielectric_unresolved"
    assert rows[0]["rank"] == ""


def test_report_model_order_does_not_change_analysis_contract() -> None:
    workflow = _workflow()
    workflow.report["model_order"] = ["pb", "gb"]

    analysis = build_project_analysis(workflow, (_record(),))

    assert analysis["model_order"] == ["gb", "pb"]
    assert list(analysis["models"]) == ["gb", "pb"]


def test_one_gb_attempt_yields_direct_and_dmm_analysis_models() -> None:
    workflow = _workflow()
    workflow.analysis["models"] = {
        "gb": workflow.analysis["models"]["gb"],
        "gb_dmm": {
            "source_model_id": "gb",
            "evaluator": "v1_dmm_v1",
            "score_selector": {
                "component": "delta",
                "label": "DMM",
                "value_key": "average",
            },
            "score_transform": {
                "scale": 4.184,
                "offset": 0.0,
                "input_unit": "kcal/mol",
                "output_unit": "kJ/mol",
            },
        },
    }
    workflow.report["model_order"] = ["gb", "gb_dmm"]
    record = _record(include_pb=False)
    payload = record["replicas"][0]["scoring_models"]["gb"]["payload"]
    payload["derived_summary"] = {
        "model_id": "gb",
        "rows": [
            {
                "model_id": "gb",
                "component": "delta",
                "label": "DMM",
                "average": -7.0,
            }
        ],
    }

    analysis = build_project_analysis(workflow, (record,))

    case = analysis["cases"][0]
    assert case["models"]["gb"]["score_mean"] == pytest.approx(-41.84)
    assert case["models"]["gb_dmm"]["score_mean"] == pytest.approx(-29.288)
    selected = case["record"]["replicas"][0]["scoring_models"]["gb_dmm"]
    assert selected["source_model_id"] == "gb"


def test_explicit_selector_must_match_exactly_one_row() -> None:
    with pytest.raises(ContractError, match="exactly one"):
        extract_score(
            _dual_model_payload(),
            selector={
                "component": "not-a-component",
                "label": "DELTA TOTAL",
            },
        )


def _comparison_record(
    case_id: str,
    *,
    epsilons: tuple[float, ...] = (2.0, 2.0, 2.0),
    outcome: str = "complete",
    block_replica: int | None = None,
    missing_replica: bool = False,
    missing_pb: bool = False,
) -> dict[str, object]:
    record = copy.deepcopy(_record())
    record["case_id"] = case_id
    record["outcome"] = outcome
    record["expected_replicas"] = ["rep01", "rep02", "rep03"]
    base = record["replicas"][0]
    rows = []
    for index, epsilon in enumerate(epsilons, 1):
        if missing_replica and index == 3:
            continue
        replica = copy.deepcopy(base)
        replica["replica_id"] = f"rep{index:02d}"
        replica["completed_models"] = ["gb", "pb"]
        for model in ("gb", "pb"):
            replica["scoring_models"][model]["payload"]["dielectric_resolution"] = {
                "mode": "fixed",
                "resolved_internal_dielectric": epsilon,
            }
            replica["scoring_models"][model]["payload"]["summary"]["rows"][0][
                "average"
            ] -= float(int(case_id[-1]))
        if missing_pb and index == 2:
            del replica["scoring_models"]["pb"]
            replica["completed_models"] = ["gb"]
        if block_replica == index:
            replica["trajectory_qc"]["qc_verdict"] = "BLOCK"
        rows.append(replica)
    record["replicas"] = rows
    record["partial"] = missing_replica
    if block_replica is not None:
        record["qc_verdict"] = "BLOCK"
    return record


def test_comparison_excludes_partial_blocked_and_missing_model_per_model() -> None:
    workflow = _workflow()
    workflow.replicas = tuple(
        SimpleNamespace(replica_id=f"rep{index:02d}") for index in range(1, 4)
    )
    workflow.cases = tuple(
        SimpleNamespace(
            case_id=f"case0{index}",
            label=f"Case {index}",
            reference={"experimental_dg_kj_mol": -40.0 - index},
        )
        for index in range(1, 8)
    )
    records = (
        _comparison_record("case01"),
        _comparison_record("case02"),
        _comparison_record("case03", epsilons=(4.0, 4.0, 4.0)),
        _comparison_record("case04", outcome="partial", missing_replica=True),
        _comparison_record("case05", outcome="scientifically_blocked", block_replica=2),
        _comparison_record("case06", epsilons=(2.0, 4.0, 2.0)),
        _comparison_record("case07", missing_pb=True),
    )

    analysis = build_project_analysis(workflow, records)
    cases = {row["case_id"]: row for row in analysis["cases"]}
    assert analysis["schema_version"] == "mmpbsa.analysis.v3"
    assert [
        (group["comparison_group"], group["n"])
        for group in analysis["models"]["gb"]["correlation_groups"]
    ] == [("epsilon=2", 3), ("epsilon=4", 1)]
    assert cases["case01"]["models"]["gb"]["rank"] == 3
    assert cases["case02"]["models"]["gb"]["rank"] == 2
    assert cases["case03"]["models"]["gb"]["rank"] == 1
    assert cases["case07"]["models"]["gb"]["rank"] == 1
    assert cases["case07"]["models"]["pb"]["comparison_exclusion_reasons"] == [
        "model_incomplete",
        "model_score_missing",
    ]
    assert cases["case04"]["models"]["gb"]["rank"] is None
    assert "incomplete_replicas" in cases["case04"]["models"]["gb"][
        "comparison_exclusion_reasons"
    ]
    assert "qc_blocked" in cases["case05"]["models"]["gb"][
        "comparison_exclusion_reasons"
    ]
    assert "mixed_dielectric" in cases["case06"]["models"]["gb"][
        "comparison_exclusion_reasons"
    ]


def test_comparison_requires_resolved_epsilon_and_references_only_for_correlation() -> None:
    workflow = _workflow()
    record = _record()
    record["replicas"][0]["scoring_models"]["gb"]["payload"][
        "dielectric_resolution"
    ] = {"mode": "fixed", "resolved_internal_dielectric": 4.0}

    analysis = build_project_analysis(workflow, (record,))

    gb = analysis["cases"][0]["models"]["gb"]
    pb = analysis["cases"][0]["models"]["pb"]
    assert gb["comparison_eligible"] is True
    assert gb["rank"] == 1
    assert pb["comparison_eligible"] is False
    assert pb["comparison_exclusion_reasons"] == ["dielectric_unresolved"]
    assert analysis["models"]["gb"]["correlation_groups"][0]["n"] == 1

    workflow.cases[0].reference = {}
    analysis_without_reference = build_project_analysis(workflow, (record,))
    assert analysis_without_reference["cases"][0]["models"]["gb"]["rank"] == 1
    assert analysis_without_reference["models"]["gb"]["correlation_groups"][0][
        "n"
    ] == 0

    record["qc_verdict"] = "WARN"
    record["outcome"] = "complete_with_warnings"
    record["replicas"][0]["trajectory_qc"]["qc_verdict"] = "WARN"
    warned = build_project_analysis(workflow, (record,))
    assert warned["cases"][0]["models"]["gb"]["comparison_eligible"] is True

    del record["replicas"][0]["completed_models"]
    unverifiable = build_project_analysis(workflow, (record,))
    assert unverifiable["cases"][0]["models"]["gb"][
        "comparison_exclusion_reasons"
    ] == ["model_completion_unverified"]

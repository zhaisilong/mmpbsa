from __future__ import annotations

import copy
import csv
import gzip
import json
import statistics

import pytest

from mmpbsa.core.errors import ContractError
from mmpbsa.core.export import (
    CASE_COLUMNS,
    REPLICA_COLUMNS,
    build_export_tables,
    export_analysis,
)


def _analysis() -> dict[str, object]:
    """Invented cases and scores covering complete and partial scoring."""
    cases = []
    block_counts = [2, 1, 1]
    warning_counts = [2, 1]
    for case_index in range(7):
        blocked = block_counts[case_index - 4] if case_index >= 4 else 0
        warned = warning_counts[case_index] if case_index < 2 else 0
        verdict = "BLOCK" if blocked else "WARN" if warned else "PASS"
        replicas = []
        scores = {model: [] for model in ("gb", "pb")}
        for replica_index in range(3):
            qc = (
                "BLOCK" if replica_index < blocked
                else "WARN" if replica_index < warned else "PASS"
            )
            scoring = {}
            if qc != "BLOCK":
                for model_index, model in enumerate(("gb", "pb")):
                    raw = -(case_index + replica_index + model_index + 1.0)
                    selected = raw * 4.184
                    scores[model].append(selected)
                    scoring[model] = {
                        "model_id": model, "selected_score_raw": raw,
                        "selected_score": selected, "payload": {"frame_count": 501},
                    }
            replicas.append({
                "replica_id": f"rep{replica_index + 1:02d}", "seed": 1000 + replica_index,
                "qc_verdict": qc, "scoring_models": scoring,
                "trajectory_qc": {"payload": {
                    "metrics": {"native_contacts_min": 0 if qc == "BLOCK" else 2},
                    "check_results": [{"id": "native_contact_retention", "verdict": qc}],
                }},
            })
        models = {}
        for model, values in scores.items():
            sd = statistics.stdev(values) if len(values) > 1 else None
            models[model] = {
                "replica_count": len(values), "score_mean": statistics.mean(values),
                "score_stddev": sd, "score_sem": sd / len(values) ** 0.5 if sd else None,
                "score_unit": "kJ/mol", "comparison_eligible": not blocked,
                "comparison_exclusion_reasons": ["qc_blocked"] if blocked else [],
                "rank": None if blocked else 4 - case_index,
                "comparison_group": "epsilon=4", "resolved_internal_dielectric": 4,
            }
        cases.append({
            "case_id": f"candidate{case_index:02d}",
            "label": f"Structure {case_index}", "models": models, "qc_verdict": verdict,
            "outcome": "scientifically_blocked" if blocked else "complete",
            "record": {"replicas": replicas},
        })
    return {
        "schema_version": "mmpbsa.analysis.v3", "project_id": "synthetic_inventory",
        "model_order": ["gb", "pb"], "cases": cases,
        "models": {model: {"score_selection": {"transform": {
            "scale": 4.184, "offset": 0, "input_unit": "kcal/mol", "output_unit": "kJ/mol",
        }}} for model in ("gb", "pb")},
    }


def test_inventory_scores_and_rank_eligibility_match_synthetic_campaign() -> None:
    cases, replicas, summary = build_export_tables(_analysis())
    assert len(cases) == 14 and len(replicas) == 42
    assert summary["case_count"] == 7 and summary["replica_count"] == 21
    assert summary["scored_replicas_by_model"] == {"gb": 17, "pb": 17}
    assert summary["eligible_cases_by_model"] == {"gb": 4, "pb": 4}
    assert summary["qc_counts"] == {"PASS": 14, "WARN": 3, "BLOCK": 4}
    assert summary["case_qc_counts"] == {"PASS": 2, "WARN": 2, "BLOCK": 3}
    assert cases[0]["score_mean"] == pytest.approx(-8.368)
    assert replicas[0]["score"] == pytest.approx(-4.184)
    assert replicas[0]["score_raw"] == -1
    assert replicas[0]["score_unit"] == "kJ/mol"
    assert replicas[0]["score_raw_unit"] == "kcal/mol"
    blocked = [row for row in replicas if row["qc_verdict"] == "BLOCK"]
    assert len(blocked) == 8
    assert all(row["model_status"] == "skipped_qc_block" for row in blocked)
    assert all(row["score"] is None and row["frame_count"] is None for row in blocked)
    partial = [row for row in cases if row["qc_verdict"] == "BLOCK"]
    assert all(row["score_mean"] is not None and row["rank"] is None for row in partial)
    assert all(row["score_stddev"] is None and row["score_sem"] is None
               for row in partial if row["replica_count"] == 1)


@pytest.mark.parametrize("compressed", [False, True])
def test_offline_csv_export_is_bom_complete_and_source_is_unchanged(tmp_path, compressed) -> None:
    archive = tmp_path / ("analysis.json.gz" if compressed else "analysis.json")
    data = json.dumps(_analysis()).encode()
    archive.write_bytes(gzip.compress(data) if compressed else data)
    initial = archive.read_bytes(), archive.stat().st_mtime_ns
    artifacts = export_analysis(archive, tmp_path / "handoff")
    assert (archive.read_bytes(), archive.stat().st_mtime_ns) == initial
    for key in ("case_table", "replica_table", "field_dictionary"):
        assert artifacts[key].read_bytes().startswith(b"\xef\xbb\xbf")
    with artifacts["case_table"].open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == CASE_COLUMNS
        rows = list(reader)
    assert rows[8]["rank"] == "" and rows[8]["score_stddev"] == ""
    with artifacts["replica_table"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 42
    with artifacts["field_dictionary"].open(encoding="utf-8-sig", newline="") as handle:
        dictionary = list(csv.DictReader(handle))
    assert {row["column"] for row in dictionary} == set(CASE_COLUMNS + REPLICA_COLUMNS)
    assert all(row["description_zh"] and row["missing_value_meaning"] for row in dictionary)


@pytest.mark.parametrize("corruption,match", [
    ("double_conversion", "transform"), ("n1_zero_sd", "SD/SEM"),
    ("blocked_score", "BLOCK replica"), ("blocked_rank", "ranking"),
])
def test_export_rejects_inconsistent_scientific_records_before_writing(tmp_path, corruption, match):
    analysis = copy.deepcopy(_analysis())
    blocked_case = analysis["cases"][4]
    if corruption == "double_conversion":
        analysis["cases"][0]["record"]["replicas"][0]["scoring_models"]["gb"][
            "selected_score"
        ] *= 4.184
    elif corruption == "n1_zero_sd":
        blocked_case["models"]["gb"]["score_stddev"] = 0
    elif corruption == "blocked_score":
        blocked_case["record"]["replicas"][0]["scoring_models"]["gb"] = {
            "selected_score": -1,
        }
    else:
        blocked_case["models"]["gb"].update(comparison_eligible=True, rank=1)
    destination = tmp_path / "must_not_exist"
    with pytest.raises(ContractError, match=match):
        export_analysis(analysis, destination)
    assert not destination.exists()


def test_export_keeps_model_specific_status_and_does_not_fill_missing_scores() -> None:
    analysis = _analysis()
    replica = analysis["cases"][0]["record"]["replicas"][0]
    replica["scoring_models"].pop("pb")
    replica["model_statuses"] = {"pb": {"result_status": "execution_failed", "state": "failed"}}
    replica["model_result_statuses"] = {"pb": "cancelled"}
    remaining = [item["scoring_models"]["pb"]["selected_score"]
                 for item in analysis["cases"][0]["record"]["replicas"][1:]]
    sd = statistics.stdev(remaining)
    analysis["cases"][0]["models"]["pb"].update(
        replica_count=2, score_mean=statistics.mean(remaining),
        score_stddev=sd, score_sem=sd / 2 ** 0.5,
        comparison_eligible=False, comparison_exclusion_reasons=["model_incomplete"], rank=None,
    )
    _, rows, _ = build_export_tables(analysis)
    assert rows[1]["model_status"] == "cancelled"
    assert rows[1]["score"] is None and rows[1]["frame_count"] is None


def test_export_refuses_source_directory_as_output(tmp_path) -> None:
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(_analysis()))
    with pytest.raises(ContractError, match="separate"):
        export_analysis(source, tmp_path)
    assert not (tmp_path / "cases.csv").exists()

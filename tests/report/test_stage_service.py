from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmpbsa.report import (
    REPORT_INPUT_SCHEMA,
    ReportProvider,
    adapt_analysis_input,
    materialize_replica_inventory,
)
from mmpbsa.report.adapters import _downsample_points
from mmpbsa.report.models import PlotPoint
from mmpbsa.report.provider import _portable_analysis_value


def _analysis() -> dict[str, object]:
    correlation = {
        "n": 1,
        "pearson_r": None,
        "spearman_rho": None,
        "kendall_tau_b": None,
        "mae": 1.2,
        "rmse": 1.2,
        "mean_signed_error": -1.2,
    }
    return {
        "schema_version": "mmpbsa.analysis.v2",
        "project_id": "demo",
        "protocol_id": "p1",
        "profile": {"profile_id": "demo", "profile_version": "1"},
        "model_order": ["gb", "pb"],
        "models": {
            "gb": {
                "model_id": "gb",
                "score_selection": {
                    "model_id": "gb",
                    "selector": None,
                    "transform": {"output_unit": "kJ/mol"},
                },
                "correlation": {"model_id": "gb", **correlation},
            },
            "pb": {
                "model_id": "pb",
                "score_selection": {
                    "model_id": "pb",
                    "selector": None,
                    "transform": {"output_unit": "kJ/mol"},
                },
                "correlation": {
                    "model_id": "pb",
                    **{**correlation, "mae": 2.2, "rmse": 2.2},
                },
            },
        },
        "cases": [
            {
                "case_id": "case1",
                "label": "Case 1",
                "reference": 7.2,
                "models": {
                    "gb": {
                        "model_id": "gb",
                        "replica_scores": [-12.0, -11.0],
                        "replica_count": 2,
                        "score_mean": -11.5,
                        "score_stddev": 0.7,
                        "score_sem": 0.5,
                        "score_unit": "kJ/mol",
                    },
                    "pb": {
                        "model_id": "pb",
                        "replica_scores": [-22.0, -21.0],
                        "replica_count": 2,
                        "score_mean": -21.5,
                        "score_stddev": 0.7,
                        "score_sem": 0.5,
                        "score_unit": "kJ/mol",
                    },
                },
                "qc_verdict": "WARN",
                "outcome": "partial",
                "finalized": True,
                "record": {
                    "expected_replicas": ["rep01", "rep02", "rep03"],
                    "complete_replicas": ["rep01", "rep02"],
                    "missing_replicas": [
                        {"replica_id": "rep03", "reason": "missing completed scoring/qc"}
                    ],
                    "qc_verdict": "WARN",
                    "outcome": "partial",
                    "replicas": [
                        {
                            "replica_id": "rep01",
                            "seed": 101,
                            "scoring_models": {
                                "gb": {
                                    "model_id": "gb",
                                    "selected_score": -12.0,
                                    "payload": {"score": -12.0},
                                },
                                "pb": {
                                    "model_id": "pb",
                                    "selected_score": -22.0,
                                    "payload": {"score": -22.0},
                                },
                            },
                            "trajectory_qc": {
                                "payload": {"frame_count": 101},
                                "qc_verdict": "PASS",
                            },
                        },
                        {
                            "replica_id": "rep02",
                            "seed": 102,
                            "scoring_models": {
                                "gb": {
                                    "model_id": "gb",
                                    "selected_score": -11.0,
                                    "payload": {"score": -11.0},
                                },
                                "pb": {
                                    "model_id": "pb",
                                    "selected_score": -21.0,
                                    "payload": {"score": -21.0},
                                },
                            },
                            "trajectory_qc": {"qc_verdict": "WARN"},
                        },
                    ],
                },
            },
            {
                "case_id": "case2",
                "label": "Case 2",
                "reference": None,
                "models": {
                    "gb": {
                        "model_id": "gb",
                        "replica_scores": [],
                        "replica_count": 0,
                        "score_mean": None,
                        "score_stddev": None,
                        "score_sem": None,
                        "score_unit": "kJ/mol",
                    },
                    "pb": {
                        "model_id": "pb",
                        "replica_scores": [],
                        "replica_count": 0,
                        "score_mean": None,
                        "score_stddev": None,
                        "score_sem": None,
                        "score_unit": "kJ/mol",
                    },
                },
                "qc_verdict": None,
                "outcome": "missing",
                "finalized": False,
                "record": None,
            },
        ],
        "configuration": {},
    }


def _science() -> dict[str, object]:
    return {
        "title": "Demo report",
        "model_order": ["pb", "gb"],
        "_protocol_settings": {"temperature_k": 300.0},
        "_profile": {
            "profile_id": "demo",
            "profile_version": "1",
            "normalized": {},
            "extensions": {},
        },
        "_project": {
            "project_id": "demo",
            "protocol_id": "p1",
            "cases": [
                {"case_id": "case1", "label": "Case 1"},
                {"case_id": "case2", "label": "Case 2"},
            ],
            "replicas": [
                {"replica_id": "rep01", "replica_index": 1, "seed": 101},
                {"replica_id": "rep02", "replica_index": 2, "seed": 102},
                {"replica_id": "rep03", "replica_index": 3, "seed": 103},
            ],
        },
    }


def test_analysis_adapter_validates_schema_and_preserves_missing_inventory() -> None:
    with pytest.raises(ValueError, match="Expected analysis schema"):
        adapt_analysis_input({"schema_version": "legacy"}, report_configuration={})

    adapted = adapt_analysis_input(_analysis(), report_configuration=_science())
    assert adapted.schema_version == REPORT_INPUT_SCHEMA
    assert [column.key for column in adapted.project.score_columns] == [
        "pb_mean",
        "pb_stddev",
        "pb_sem",
        "gb_mean",
        "gb_stddev",
        "gb_sem",
    ]
    assert adapted.project.rank_by == "pb_mean"
    assert [item.score_key for item in adapted.project.correlations] == [
        "pb_mean",
        "gb_mean",
    ]
    case1 = adapted.project.cases[0]
    assert case1.scores["gb_mean"] == -11.5
    assert case1.scores["pb_mean"] == -21.5
    assert case1.replicas[0].scores["gb_mean"] == -12.0
    assert case1.replicas[0].scores["pb_mean"] == -22.0
    case2 = adapted.project.cases[1]
    assert [(item.replica_id, item.status) for item in materialize_replica_inventory(case2)] == [
        ("rep01", "missing"),
        ("rep02", "missing"),
        ("rep03", "missing"),
    ]
    assert adapted.project.correlations[0].status == "unavailable"


def test_v3_report_uses_eligibility_and_dielectric_groups_from_analysis() -> None:
    analysis = copy.deepcopy(_analysis())
    analysis["schema_version"] = "mmpbsa.analysis.v3"
    for model in analysis["models"].values():
        model["correlation_groups"] = []
    analysis["models"]["pb"]["correlation_groups"] = [
        {
            "model_id": "pb",
            "comparison_group": "epsilon=4",
            "resolved_internal_dielectric": 4.0,
            "n": 1,
            "reference_mode": "quantitative_comparison",
        }
    ]
    for model in analysis["cases"][0]["models"].values():
        model.update(
            comparison_eligible=False,
            comparison_exclusion_reasons=["case_incomplete", "incomplete_replicas"],
            comparison_group="epsilon=2",
            resolved_internal_dielectric=2.0,
            rank=None,
        )
    second = analysis["cases"][1]["models"]
    second["pb"].update(
        score_mean=-10.0,
        comparison_eligible=True,
        comparison_exclusion_reasons=[],
        comparison_group="epsilon=4",
        resolved_internal_dielectric=4.0,
        rank=1,
    )
    second["gb"].update(
        score_mean=-5.0,
        comparison_eligible=True,
        comparison_exclusion_reasons=[],
        comparison_group="epsilon=2",
        resolved_internal_dielectric=2.0,
        rank=1,
    )

    project = adapt_analysis_input(analysis, report_configuration=_science()).project
    document = ReportProvider().build_document(project)
    scores = next(
        block
        for slot in document.pages[0].slots
        for block in slot.blocks
        if block.block_id == "project-scores"
    )
    correlations = next(
        block
        for slot in document.pages[0].slots
        for block in slot.blocks
        if block.block_id == "project-correlations"
    )
    assert project.comparison_contract is True
    assert [column.key for column in scores.columns[:4]] == [
        "rank", "case", "internal_dielectric", "comparison"
    ]
    assert [column.label for column in scores.columns[2:4]] == [
        "PB rank ε", "PB rank eligibility"
    ]
    assert "apply only to PB" in scores.caption
    assert "Other model score columns are descriptive" in scores.caption
    assert project.cases[1].comparisons["gb_mean"].dielectric == 2.0
    assert [(row[0], row[1].label, row[2]) for row in scores.rows] == [
        (1, "Case 2", 4.0),
        (None, "Case 1", 2.0),
    ]
    assert "case incomplete" in scores.rows[1][3]
    assert [column.key for column in correlations.columns[:4]] == [
        "reference", "score", "dielectric", "status"
    ]
    assert correlations.rows[0][2] == 4.0

    unranked_science = _science()
    unranked_science["rank_by"] = None
    unranked = adapt_analysis_input(
        analysis, report_configuration=unranked_science
    ).project
    unranked_document = ReportProvider().build_document(unranked)
    unranked_scores = next(
        block
        for slot in unranked_document.pages[0].slots
        for block in slot.blocks
        if block.block_id == "project-scores"
    )
    assert [column.key for column in unranked_scores.columns[:2]] == [
        "case", "pb_mean"
    ]
    assert [row[0].label for row in unranked_scores.rows] == [
        "Case 1", "Case 2"
    ]


@pytest.mark.parametrize(
    ("cases", "error_type", "message"),
    [
        ("corrupt", TypeError, "analysis.cases must be a sequence"),
        ([{"case_id": "case1"}, "corrupt"], TypeError, r"analysis.cases\[1\]"),
        ([{"label": "missing ID"}], ValueError, r"analysis.cases\[0\].case_id"),
    ],
)
def test_analysis_adapter_rejects_malformed_case_inventory(
    cases: object, error_type: type[Exception], message: str
) -> None:
    analysis = _analysis()
    analysis["cases"] = cases
    with pytest.raises(error_type, match=message):
        adapt_analysis_input(analysis, report_configuration=_science())


def test_case_catalog_adds_preflight_pages_without_fake_run_inventory() -> None:
    science = _science()
    science["case_catalog"] = [
        {
            "case_id": "case1",
            "label": "Eligible case",
            "status": "eligible",
            "metadata": {"PDB": "1AAA"},
            "provenance": {"Source": "catalog"},
        },
        {
            "case_id": "case_rejected",
            "label": "Rejected case",
            "status": "preflight_rejected",
            "reason": "The deposited peptide is incomplete.",
            "metadata": {"PDB": "1BBB"},
            "provenance": {"Source": "RCSB"},
        },
    ]

    adapted = adapt_analysis_input(_analysis(), report_configuration=science)
    assert [case.case_id for case in adapted.project.cases] == [
        "case1",
        "case_rejected",
        "case2",
    ]
    rejected = adapted.project.cases[1]
    assert rejected.expected_replica_ids == ()
    assert rejected.status == "preflight_rejected"
    document = ReportProvider().build_document(adapted.project)
    project_inventory = next(
        block
        for slot in document.pages[0].slots
        for block in slot.blocks
        if block.block_id == "project-inventory"
    )
    assert project_inventory.rows[1][3:5] == (
        "not scheduled",
        "not applicable",
    )
    rejected_page = next(
        page for page in document.pages if page.target.case_id == "case_rejected"
    )
    blocks = [block for slot in rejected_page.slots for block in slot.blocks]
    assert {block.block_id for block in blocks} == {
        "case-overview",
        "case-preflight-rejected",
        "case-provenance",
    }
    notice = next(
        block for block in blocks if block.block_id == "case-preflight-rejected"
    )
    assert notice.body == "The deposited peptide is incomplete."


def test_case_catalog_rejects_conflicts_and_missing_reasons() -> None:
    science = _science()
    science["case_catalog"] = [
        {"case_id": "case1", "status": "preflight_rejected", "reason": "bad"}
    ]
    with pytest.raises(ValueError, match="marks workflow case"):
        adapt_analysis_input(_analysis(), report_configuration=science)

    science["case_catalog"] = [
        {"case_id": "case_rejected", "status": "preflight_rejected"}
    ]
    with pytest.raises(ValueError, match="reason is required"):
        adapt_analysis_input(_analysis(), report_configuration=science)


def test_profile_wire_contributions_are_typed_escaped_and_page_checked() -> None:
    science = _science()
    science["_profile"]["extensions"] = {  # type: ignore[index]
        "report": {
            "schema_version": "mmpbsa.report.contributions.v1",
            "contributions": [
                {
                    "page": "project",
                    "slot": "project.after_summary",
                    "block": {
                        "kind": "notice",
                        "title": "Project note",
                        "text": "<script>alert(1)</script>",
                        "severity": "warn",
                    },
                },
                {
                    "page": "case:*",
                    "slot": "case.before_provenance",
                    "block": {
                        "kind": "key_value",
                        "title": "Profile facts",
                        "items": {"Protocol": "locked", "Replicas": 3},
                    },
                },
                {
                    "page": "case:case1",
                    "slot": "case.after_energy",
                    "block": {
                        "kind": "table",
                        "title": "Declared values",
                        "columns": ["Name", "Value"],
                        "rows": [["alpha", 1.0]],
                    },
                },
            ],
        }
    }
    adapted = adapt_analysis_input(_analysis(), report_configuration=science)
    provider = ReportProvider()
    document = provider.build_document(
        adapted.project,
        contributors=adapted.contributors,
    )
    rendered = provider.render(document)

    assert "project.after_summary" in rendered[document.pages[0].path]
    assert "case.before_provenance" in rendered[document.pages[1].path]
    assert "case.after_energy" in rendered[document.pages[1].path]
    assert "<script>alert(1)</script>" not in rendered[document.pages[0].path]
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered[document.pages[0].path]

    science["_profile"]["extensions"]["report"]["contributions"][0]["slot"] = (  # type: ignore[index]
        "case.after_summary"
    )
    with pytest.raises(ValueError, match="incompatible"):
        adapt_analysis_input(_analysis(), report_configuration=science)


def test_profile_wire_rejects_raw_html_and_unknown_fields() -> None:
    science = _science()
    science["_profile"]["extensions"] = {  # type: ignore[index]
        "report": {
            "schema_version": "mmpbsa.report.contributions.v1",
            "contributions": [
                {
                    "page": "project",
                    "slot": "project.after_summary",
                    "block": {
                        "kind": "raw_html",
                        "title": "Unsafe",
                        "html": "<script>bad()</script>",
                    },
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="HTML/script blocks are forbidden"):
        adapt_analysis_input(_analysis(), report_configuration=science)


def test_analysis_payload_derives_energy_convergence_and_qc_series() -> None:
    analysis = copy.deepcopy(_analysis())
    replica = analysis["cases"][0]["record"]["replicas"][0]  # type: ignore[index]
    replica["scoring_models"]["gb"]["payload"]["per_frame"] = {  # type: ignore[index]
        "columns": ["Frame #", "VDWAALS", "DELTA TOTAL"],
        "rows": [
            {"Frame #": 151.0, "VDWAALS": -5.0, "DELTA TOTAL": -10.0},
            {"Frame #": 152.0, "VDWAALS": -6.0, "DELTA TOTAL": -15.0},
        ],
    }
    replica["trajectory_qc"]["payload"]["ligand_rmsd"] = {  # type: ignore[index]
        "columns": ["Frame", "RMSD"],
        "rows": [[151.0, 1.2], [152.0, 1.5]],
    }

    case = adapt_analysis_input(
        analysis,
        report_configuration=_science(),
    ).project.cases[0]
    assert [panel.title for panel in case.energy_plots] == ["GB"]
    assert [panel.title for panel in case.convergence_plots] == ["GB"]
    assert [panel.title for panel in case.trajectory_plots] == [
        "Ligand primary RMSD"
    ]
    assert [
        point.y
        for point in case.convergence_plots[0].plot.series[0].points
    ] == [
        -10.0,
        -12.5,
    ]
    assert [
        point.x
        for point in case.trajectory_plots[0].plot.series[0].points
    ] == [
        151.0,
        152.0,
    ]


def test_report_never_falls_back_to_payload_or_legacy_qc() -> None:
    analysis = copy.deepcopy(_analysis())
    replica = analysis["cases"][0]["record"]["replicas"][0]  # type: ignore[index]
    del replica["scoring_models"]["pb"]["selected_score"]  # type: ignore[index]
    replica["qc"] = {
        "qc_verdict": "BLOCK",
        "payload": {
            "ligand_rmsd": {
                "columns": ["Frame", "RMSD"],
                "rows": [[1.0, 99.0]],
            }
        },
    }
    del replica["trajectory_qc"]

    case = adapt_analysis_input(
        analysis,
        report_configuration=_science(),
    ).project.cases[0]

    assert case.replicas[0].scores["pb_mean"] is None
    assert case.replicas[0].qc_verdict is None
    assert case.trajectory is None


def test_ambiguous_energy_columns_remain_unavailable() -> None:
    analysis = copy.deepcopy(_analysis())
    replica = analysis["cases"][0]["record"]["replicas"][0]  # type: ignore[index]
    replica["scoring_models"]["gb"]["payload"]["per_frame"] = {  # type: ignore[index]
        "columns": ["Frame", "DELTA TOTAL", "DELTA_TOTAL"],
        "rows": [
            {"Frame": 1.0, "DELTA TOTAL": -1.0, "DELTA_TOTAL": -2.0},
        ],
    }
    case = adapt_analysis_input(
        analysis,
        report_configuration=_science(),
    ).project.cases[0]
    assert case.energy is None
    assert case.convergence is None


def _ligand_rmsd_payload(
    replicas: dict[str, list[list[float]]],
) -> dict[str, object]:
    return {
        "schema_version": "mmpbsa.report_assets.v3",
        "availability": {"case1": {"status": "available"}},
        "assets": [],
        "diagnostics": {
            "case1": {
                "ligand_rmsd": {
                    "reference_artifact": "selected_trajectory",
                    "reference_frame": 1,
                    "reference_time_ps": 0.0,
                    "reference_semantics": "post_npt_production_start",
                    "reference_preprocessing": (
                        "autoimage_ligand_anchor_receptor_fixed"
                    ),
                    "alignment_selection": "receptor_interface_backbone",
                    "rmsd_selection": "ligand_qc",
                    "replicas": {
                        replica_id: {
                            "columns": ["Frame", "ligand_equilibrated"],
                            "rows": rows,
                        }
                        for replica_id, rows in replicas.items()
                    },
                }
            }
        },
    }


def test_ligand_rmsd_occupancy_basin_is_one_dimensional_per_replica() -> None:
    values = [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.5]
    diagnostics = {
        replica_id: [
            [float(frame), value + offset]
            for frame, value in enumerate(values, start=1)
        ]
        for replica_id, offset in (("rep01", 0.0), ("rep02", 0.2))
    }

    case = adapt_analysis_input(
        _analysis(),
        report_configuration=_science(),
        asset_payload=_ligand_rmsd_payload(diagnostics),
    ).project.cases[0]

    basin = case.occupancy_basin
    assert basin is not None
    assert basin.kind.value == "line"
    assert basin.x_min == 0.0
    assert basin.y_min == 0.0
    assert basin.x_label == "Ligand RMSD to equilibrated starting pose (Å)"
    assert basin.y_label == "Relative occupancy free energy (kJ/mol)"
    assert [series.label for series in basin.series] == ["rep01", "rep02"]
    assert basin.series[0].points[0].y == 0.0
    assert basin.series[0].points[0].emphasized
    assert basin.series[0].points[1].y == pytest.approx(
        0.00831446261815324 * 300.0 * 0.6931471805599453
    )
    assert "300 K" in case.occupancy_basin_note
    assert "rep03" in case.occupancy_basin_note
    common_minimum = 0.5
    common_width = (1.7 - common_minimum) / 10
    expected_centers = {
        common_minimum + (index + 0.5) * common_width
        for index in range(10)
    }
    assert all(
        any(math.isclose(point.x, center) for center in expected_centers)
        for series in basin.series
        for point in series.points
    )


def test_long_plot_downsampling_preserves_endpoints_and_extrema() -> None:
    points = tuple(
        PlotPoint(float(index), 1000.0 if index == 777 else float(index % 19))
        for index in range(2001)
    )

    sampled = _downsample_points(points)

    assert len(sampled) <= 400
    assert sampled[0] == points[0]
    assert sampled[-1] == points[-1]
    assert points[777] in sampled


def test_portable_analysis_archive_redacts_only_host_path_roots() -> None:
    payload = {
        "values": [1.0, 2.0],
        "artifact": {
            "path": "/example/private/workspace/output/table.csv",
            "sha256": "a" * 64,
        },
        "windows_tool": r"C:\\AmberTools\\bin\\sander.exe",
        "relative": "assets/table.csv",
        "message": (
            "Cannot read (/example/private/workspace/output/table.csv). "
            "See https://doi.org/10.1000/example and ratio 1/2."
        ),
        "windows_message": r"Cannot open C:\AmberTools\bin\sander.exe.",
        "file_uri": "source=file:///tmp/private/input.dat",
        "quoted_space": "Cannot read '/example/private/My Project/output data.csv'.",
        "quoted_windows": r'Cannot open "C:\Program Files\Amber\sander.exe"',
        "unquoted_space": "Failed at /example/user/My Project/private/run.log",
    }

    portable = _portable_analysis_value(payload)

    assert portable == {
        "values": [1.0, 2.0],
        "artifact": {
            "path": "host-local://table.csv",
            "sha256": "a" * 64,
        },
        "windows_tool": "host-local://sander.exe",
        "relative": "assets/table.csv",
        "message": (
            "Cannot read (host-local://table.csv). "
            "See https://doi.org/10.1000/example and ratio 1/2."
        ),
        "windows_message": "Cannot open host-local://sander.exe.",
        "file_uri": "source=host-local://input.dat",
        "quoted_space": "Cannot read 'host-local://output data.csv'.",
        "quoted_windows": 'Cannot open "host-local://sander.exe"',
        "unquoted_space": "host-local://redacted",
    }


def test_portable_analysis_archive_redacts_nested_mapping_keys() -> None:
    payload = {
        "nested": {
            "/example/user/private/secret.txt": {"message": "ok"},
            "https://doi.org/10.1000/example": 1.0,
        }
    }

    assert _portable_analysis_value(payload) == {
        "nested": {
            "host-local://secret.txt": {"message": "ok"},
            "https://doi.org/10.1000/example": 1.0,
        }
    }


def test_portable_analysis_archive_rejects_redacted_key_collision() -> None:
    payload = {
        "/example/user/private/secret.txt": 1,
        "/tmp/other/secret.txt": 2,
    }

    with pytest.raises(ValueError, match="mapping keys collide after redaction"):
        _portable_analysis_value(payload)


def test_ligand_rmsd_basin_rejects_noncontiguous_frames() -> None:
    with pytest.raises(ValueError, match="contiguous from 1"):
        adapt_analysis_input(
            _analysis(),
            report_configuration=_science(),
            asset_payload=_ligand_rmsd_payload(
                {"rep01": [[1.0, 0.5], [3.0, 0.8]]}
            ),
        )


def test_ligand_rmsd_basin_requires_resolved_protocol_temperature() -> None:
    science = _science()
    del science["_protocol_settings"]
    with pytest.raises(ValueError, match="resolved protocol settings"):
        adapt_analysis_input(
            _analysis(),
            report_configuration=science,
            asset_payload=_ligand_rmsd_payload(
                {"rep01": [[1.0, 0.5], [2.0, 0.8]]}
            ),
        )


@pytest.mark.parametrize(
    "generation_configuration",
    [
        pytest.param(None, id="default"),
        pytest.param(
            {"components": ["receptor_A", "receptor_B", "heme_A", "heme_B", "ligand"]},
            id="nested-generation-map",
        ),
    ],
)
def test_stage_service_renders_and_enumerates_pages_and_assets(
    tmp_path: Path, generation_configuration: dict[str, object] | None
) -> None:
    asset_attempt = tmp_path / "asset-attempt"
    bundle = asset_attempt / "output" / "case1"
    bundle.mkdir(parents=True)
    session = bundle / "session.pse"
    script = bundle / "open.py"
    asset_index = asset_attempt / "output" / "asset_index.json"
    session.write_bytes(b"session")
    script.write_text("print('display only')\n", encoding="utf-8")
    asset_index.write_text(
        '{"schema_version":"mmpbsa.report_assets.v3","bundles":{}}\n',
        encoding="utf-8",
    )

    attempt = tmp_path / "report-attempt"
    (attempt / "output").mkdir(parents=True)
    science = _science()
    if generation_configuration is not None:
        science["assets"] = generation_configuration
    request = SimpleNamespace(
        science=science,
        upstream={
            "analysis": SimpleNamespace(payload=_analysis()),
            "assets": SimpleNamespace(
                root=asset_attempt,
                artifacts={
                    "case1.session": session,
                    "case1.script": script,
                    "asset_index": asset_index,
                },
                payload={
                    "schema_version": "mmpbsa.report_assets.v3",
                    "availability": {"case1": {"status": "available"}},
                    "diagnostics": {},
                    "assets": [
                        {
                            "artifact_role": "case1.session",
                            "case_id": "case1",
                            "replica_id": None,
                            "kind": "pymol_script",
                            "label": "Open in PyMOL",
                            "visibility": "visible",
                        },
                        {
                            "artifact_role": "case1.script",
                            "case_id": "case1",
                            "replica_id": "rep01",
                            "kind": "provenance",
                            "label": "PyMOL input",
                            "visibility": "visible",
                        },
                        {
                            "artifact_role": "asset_index",
                            "case_id": None,
                            "replica_id": None,
                            "kind": "manifest",
                            "label": "Display asset availability",
                            "visibility": "visible",
                        },
                    ],
                },
            ),
        },
        attempt_dir=attempt,
    )
    provider = ReportProvider()
    result = provider.execute(request)

    assert result.payload["schema_version"] == "mmpbsa.report.result.v1"
    assert set(result.artifacts) == {
        "report_index",
        "report_case_case1",
        "report_case_case2",
        "report_asset_0001",
        "report_asset_0002",
        "report_asset_0003",
        "report_asset_0004",
    }
    assert (attempt / "output" / "index.html").is_file()
    assert (attempt / "output" / "cases" / "case1" / "index.html").is_file()
    assert (attempt / "output" / "cases" / "case2" / "index.html").is_file()
    published_session = attempt / "output" / "assets" / "case1" / "session.pse"
    assert published_session.read_bytes() == b"session"
    assert not os.path.samefile(session, published_session)
    session.write_bytes(b"mutated upstream session")
    assert published_session.read_bytes() == b"session"
    assert (attempt / "output" / "assets" / "case1" / "open.py").is_file()
    assert (attempt / "output" / "assets" / "asset_index.json").is_file()
    assert (attempt / "output" / "data" / "analysis.json.gz").is_file()
    case_html = (attempt / "output" / "cases" / "case1" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "Open in PyMOL" in case_html
    assert "../../assets/case1/session.pse" in case_html
    assert "../../assets/case1/open.py" in case_html

    resumed = provider.resume(request)
    assert resumed.artifacts["report_index"] == attempt / "output" / "index.html"


def test_asset_descriptor_schema_kind_and_role_are_strict(tmp_path: Path) -> None:
    asset = tmp_path / "asset.txt"
    asset.write_text("asset", encoding="utf-8")
    base = {
        "schema_version": "mmpbsa.report_assets.v3",
        "availability": {},
        "diagnostics": {},
        "assets": [
            {
                "artifact_role": "missing",
                "case_id": "case1",
                "replica_id": None,
                "kind": "topology",
                "label": "Structure",
                "visibility": "visible",
            }
        ],
    }
    with pytest.raises(ValueError, match="role is unavailable"):
        adapt_analysis_input(
            _analysis(),
            report_configuration=_science(),
            asset_artifacts={"available": asset},
            asset_payload=base,
        )

    base["assets"][0]["artifact_role"] = "available"  # type: ignore[index]
    base["assets"][0]["kind"] = "raw_html"  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown asset kind"):
        adapt_analysis_input(
            _analysis(),
            report_configuration=_science(),
            asset_artifacts={"available": asset},
            asset_payload=base,
        )

    base["schema_version"] = "legacy"
    with pytest.raises(ValueError, match="Unsupported report-assets"):
        adapt_analysis_input(
            _analysis(),
            report_configuration=_science(),
            asset_artifacts={"available": asset},
            asset_payload=base,
        )


@pytest.mark.parametrize("descriptor_field", ("assets", "asset_links"))
def test_configured_asset_descriptors_are_published_with_generated_assets(
    tmp_path: Path, descriptor_field: str
) -> None:
    asset_attempt = tmp_path / "asset-attempt"
    asset_output = asset_attempt / "output"
    asset_output.mkdir(parents=True)
    generated = asset_output / "generated.json"
    custom = asset_output / "custom.txt"
    generated.write_text('{"generated": true}\n', encoding="utf-8")
    custom.write_text("custom display details\n", encoding="utf-8")
    science = _science()
    science[descriptor_field] = [
        {
            "role": "custom",
            "request_id": "custom-display",
            "case_id": "case1",
            "slot": "artifacts",
            "output": "downloads/custom.txt",
            "label": "Custom display details",
        }
    ]
    if descriptor_field == "asset_links":
        science["assets"] = {
            "components": ["receptor_A", "receptor_B", "heme_A", "heme_B", "ligand"]
        }
    request = SimpleNamespace(
        science=science,
        upstream={
            "analysis": SimpleNamespace(payload=_analysis()),
            "assets": SimpleNamespace(
                root=asset_attempt,
                artifacts={"generated": generated, "custom": custom},
                payload={
                    "schema_version": "mmpbsa.report_assets.v3",
                    "availability": {},
                    "diagnostics": {},
                    "assets": [
                        {
                            "artifact_role": "generated",
                            "case_id": None,
                            "replica_id": None,
                            "kind": "manifest",
                            "label": "Generated display index",
                            "visibility": "visible",
                        }
                    ],
                },
            ),
        },
        attempt_dir=tmp_path / "report-attempt",
    )

    result = ReportProvider().execute(request)

    output = request.attempt_dir / "output"
    assert (output / "assets" / "generated.json").read_bytes() == generated.read_bytes()
    assert (output / "downloads" / "custom.txt").read_bytes() == custom.read_bytes()
    assert result.payload["assets"] == [
        "assets/generated.json", "downloads/custom.txt", "data/analysis.json.gz"
    ]
    project_html = (output / "index.html").read_text(encoding="utf-8")
    case_html = (output / "cases" / "case1" / "index.html").read_text(encoding="utf-8")
    assert "Generated display index" in project_html
    assert 'href="assets/generated.json"' in project_html
    assert "Custom display details" in case_html
    assert 'href="../../downloads/custom.txt"' in case_html


def test_asset_links_merge_with_legacy_asset_descriptors(tmp_path: Path) -> None:
    asset = tmp_path / "asset.txt"
    asset.write_text("asset", encoding="utf-8")
    science = _science()
    science["assets"] = [
        {"role": "custom", "output": "downloads/legacy.txt", "label": "Legacy asset"}
    ]
    science["asset_links"] = [
        {"role": "custom", "output": "downloads/canonical.txt", "label": "Canonical asset"}
    ]

    adapted = adapt_analysis_input(
        _analysis(),
        report_configuration=science,
        asset_artifacts={"custom": asset},
    )

    assert [request.outputs[0].as_posix() for request in adapted.asset_requests] == [
        "downloads/legacy.txt", "downloads/canonical.txt"
    ]
    assert [(link.label, link.href) for link in adapted.project.artifacts] == [
        ("Legacy asset", "downloads/legacy.txt"),
        ("Canonical asset", "downloads/canonical.txt"),
    ]


@pytest.mark.parametrize(
    "configuration",
    [
        pytest.param(
            {
                "assets": {
                    "components": ["receptor_A", "receptor_B", "heme_A", "heme_B", "ligand"]
                }
            },
            id="generation-map",
        ),
        pytest.param({"assets": {}}, id="empty-generation-map"),
        pytest.param({"assets": [], "asset_links": []}, id="empty-descriptor-lists"),
        pytest.param({"assets": None, "asset_links": None}, id="null-as-omitted"),
    ],
)
def test_report_asset_configuration_is_valid_without_upstream_assets(
    configuration: dict[str, object],
) -> None:
    science = _science()
    science.update(configuration)

    adapted = adapt_analysis_input(_analysis(), report_configuration=science)

    assert adapted.asset_requests == ()
    assert adapted.asset_sources == {}


@pytest.mark.parametrize("has_upstream", (False, True), ids=("no-upstream", "upstream"))
@pytest.mark.parametrize(
    ("field", "configuration"),
    [
        ("assets", "corrupt"),
        ("assets", True),
        ("assets", 42),
        ("assets", ["corrupt"]),
        ("assets", [{"role": "custom"}, False]),
        ("assets", {1: "corrupt"}),
        ("asset_links", {"role": "custom"}),
        ("asset_links", "corrupt"),
        ("asset_links", 42),
        ("asset_links", ["corrupt"]),
        ("asset_links", [{"role": "custom"}, False]),
    ],
)
def test_report_asset_configuration_rejects_malformed_forms_before_merging(
    field: str, configuration: object, has_upstream: bool
) -> None:
    science = _science()
    science[field] = configuration
    payload = (
        {
            "schema_version": "mmpbsa.report_assets.v3",
            "availability": {},
            "assets": [],
            "diagnostics": {},
        }
        if has_upstream
        else None
    )

    with pytest.raises(TypeError, match=rf"report\.{field}"):
        adapt_analysis_input(
            _analysis(), report_configuration=science, asset_payload=payload
        )


@pytest.mark.parametrize("descriptor_field", ("assets", "asset_links"))
def test_configured_asset_descriptors_require_upstream_roles(
    descriptor_field: str,
) -> None:
    science = _science()
    science[descriptor_field] = [{"role": "custom"}]

    with pytest.raises(ValueError, match="role is unavailable"):
        adapt_analysis_input(_analysis(), report_configuration=science)

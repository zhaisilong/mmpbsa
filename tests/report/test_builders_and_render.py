from __future__ import annotations

from dataclasses import dataclass

import pytest

from mmpbsa.report import (
    AssetBlock,
    CaseRecord,
    Contribution,
    CorrelationRecord,
    DecompositionRecord,
    Link,
    NamedSlot,
    NoticeBlock,
    PageTarget,
    PlotKind,
    PlotPanel,
    PlotPoint,
    PlotSeries,
    PlotSpec,
    ProjectRecord,
    QcIssue,
    ReplicaRecord,
    ScoreColumn,
    TableBlock,
    build_document,
    materialize_replica_inventory,
    render_document,
    render_plot_svg,
)


def _plot(series_id: str = "rep01") -> PlotSpec:
    return PlotSpec(
        PlotKind.LINE,
        "Time (ps)",
        "Value",
        (
            PlotSeries(
                series_id,
                series_id,
                (PlotPoint(0.0, 1.0), PlotPoint(100.0, 2.0)),
            ),
        ),
    )


@pytest.mark.parametrize(
    "status,title",
    [("preflight_pending", "Preflight pending"),
     ("preflight_skipped", "Preflight skipped"),
     ("preflight_rejected", "Preflight rejected")],
)
def test_preflight_states_do_not_fabricate_science(status: str, title: str) -> None:
    case = CaseRecord(
        "pending", "Pending", (), status=status, outcome="not_run",
        metadata={"Preflight reason": "Explicit gate"}, scores={"gb_mean": -100.0},
    )
    document = build_document(ProjectRecord(
        "preflight", "Preflight", (case,), (ScoreColumn("gb_mean", "GB mean"),),
        rank_by="gb_mean",
    ))
    assert case.is_preflight
    project_blocks = [block for slot in document.pages[0].slots for block in slot.blocks]
    inventory = next(block for block in project_blocks if block.block_id == "project-inventory")
    assert len(inventory.columns) == 3
    assert inventory.rows[0][1] == status
    assert not any(block.block_id == "project-scores" for block in project_blocks)
    html = render_document(document)[document.pages[1].path]
    assert title in html and "Explicit gate" in html
    assert "Energy basin unavailable" not in html
    assert "Correlation unavailable" not in html


@pytest.mark.parametrize("status", ["preflight_pending", "preflight_skipped"])
def test_case_catalog_accepts_report_only_preflight_states(status: str) -> None:
    from mmpbsa.report.adapters import _merge_case_catalog

    entry = {"case_id": "unprepared", "status": status, "reason": "Model review"}
    case = _merge_case_catalog((), [entry])[0]
    assert case.status == status and case.expected_replica_ids == ()
    with pytest.raises(ValueError, match="reason is required"):
        _merge_case_catalog((), [{"case_id": "unprepared", "status": status}])
    with pytest.raises(ValueError, match="marks workflow case"):
        _merge_case_catalog((CaseRecord("unprepared", "Unprepared", ()),), [entry])


def _minimal_project(*, case: CaseRecord | None = None) -> ProjectRecord:
    return ProjectRecord(
        project_id="demo",
        title="Demo",
        cases=(
            case
            or CaseRecord(
                case_id="lig1",
                label="Ligand 1",
                expected_replica_ids=("rep01", "rep02", "rep03"),
            ),
        ),
    )


@dataclass
class _Contributor:
    contributor_id: str
    block_id: str
    priority: int

    def contribute(self, context):
        return (
            Contribution(
                PageTarget.project(),
                NamedSlot.OVERVIEW,
                NoticeBlock(self.block_id, self.block_id, "safe"),
                self.contributor_id,
                self.priority,
            ),
        )


def test_named_slots_and_contributors_have_deterministic_order() -> None:
    project = _minimal_project()
    contributors = (
        _Contributor("zeta", "z-block", 20),
        _Contributor("alpha", "a-block", 20),
        _Contributor("first", "first-block", 5),
    )
    first = build_document(project, contributors=contributors)
    second = build_document(project, contributors=tuple(reversed(contributors)))

    assert first == second
    overview = first.pages[0].slots[0]
    assert overview.slot is NamedSlot.OVERVIEW
    assert [block.block_id for block in overview.blocks] == [
        "first-block",
        "project-overview",
        "a-block",
        "z-block",
    ]


def test_renderer_escapes_content_and_rejects_active_link_schemes() -> None:
    contributor = _Contributor("safe", "safe-block", 50)
    project = ProjectRecord(
        project_id="demo",
        title="<script>alert(1)</script>",
        cases=(
            CaseRecord(
                case_id="lig1",
                label="<img src=x onerror=alert(1)>",
                expected_replica_ids=("rep01",),
            ),
        ),
    )
    document = build_document(project, contributors=(contributor,))
    rendered = render_document(document)

    assert "<script>alert(1)</script>" not in rendered[document.pages[0].path]
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered[document.pages[0].path]
    assert "<img src=x" not in rendered[document.pages[0].path]
    with pytest.raises(ValueError, match="Unsafe link scheme"):
        Link("bad", "javascript:alert(1)")


def test_optional_scientific_panels_are_omitted_when_data_is_absent() -> None:
    document = build_document(_minimal_project())
    project_slots = {slot.slot for slot in document.pages[0].slots}
    case_slots = {slot.slot for slot in document.pages[1].slots}

    assert project_slots == {NamedSlot.OVERVIEW, NamedSlot.INVENTORY}
    assert case_slots == {
        NamedSlot.OVERVIEW,
        NamedSlot.INVENTORY,
        NamedSlot.TRAJECTORY,
        NamedSlot.CONVERGENCE,
        NamedSlot.DECOMPOSITION,
        NamedSlot.ACTIONS,
    }


def test_full_case_builds_all_standard_mmpbsa_panels() -> None:
    case = CaseRecord(
        case_id="lig1",
        label="Ligand 1",
        expected_replica_ids=("rep01",),
        replicas=(
            ReplicaRecord(
                "rep01",
                "completed",
                frame_count=101,
                qc_verdict="PASS",
                scores={"GB": -15.2},
            ),
        ),
        status="completed",
        outcome="scientifically_valid",
        qc_verdict="PASS",
        qc_issues=(QcIssue("contact-loss", "Contact loss warning", "WARN", "rep01"),),
        protocol={"Window": "3-5 ns"},
        scores={"GB": -15.2},
        trajectory=_plot(),
        convergence=_plot(),
        occupancy_basin=_plot(),
        correlations=(CorrelationRecord("pIC50", "GB", 5, pearson_r=0.8),),
        correlation_plot=_plot("correlation"),
        decomposition=(DecompositionRecord("ASP93", {"GB": -2.1}),),
        actions=(Link("PyMOL", "assets/session.pse"),),
        artifacts=(Link("Scores", "assets/scores.csv"),),
        provenance={"Fingerprint": "abc"},
    )
    project = ProjectRecord(
        "demo",
        "Demo",
        (case,),
        score_columns=(ScoreColumn("GB", "GB"),),
        rank_by="GB",
    )
    case_page = build_document(project).pages[1]

    assert tuple(slot.slot for slot in case_page.slots) == (
        NamedSlot.OVERVIEW,
        NamedSlot.PROTOCOL,
        NamedSlot.INVENTORY,
        NamedSlot.SCORES,
        NamedSlot.QC,
        NamedSlot.TRAJECTORY,
        NamedSlot.CONVERGENCE,
        NamedSlot.DECOMPOSITION,
        NamedSlot.ACTIONS,
        NamedSlot.ARTIFACTS,
        NamedSlot.PROVENANCE,
    )
    assert [block.block_id for slot in case_page.slots for block in slot.blocks] == [
        "case-overview",
        "case-protocol",
        "case-replica-summary",
        "case-scores",
        "case-qc",
        "case-trajectory",
        "case-convergence",
        "case-basin",
        "case-decomposition",
        "case-actions",
        "case-artifacts",
        "case-provenance",
    ]
    basin = next(
        block
        for slot in case_page.slots
        for block in slot.blocks
        if block.block_id == "case-basin"
    )
    assert "receptor-interface-backbone fitted" in basin.caption
    assert "post-NPT production t=0 pose" in basin.caption
    assert (
        "mass-weighted over small-molecule heavy atoms or peptide backbone atoms"
        in basin.caption
    )


def test_project_links_are_explicit_and_unranked_scores_have_precise_title() -> None:
    case = CaseRecord(
        "lig1",
        "Ligand 1",
        ("rep01",),
        scores={"gb_mean": -10.0},
    )
    project = ProjectRecord(
        "demo",
        "Demo",
        (case,),
        score_columns=(ScoreColumn("gb_mean", "GB mean"),),
        rank_by=None,
    )

    html = render_document(build_document(project))[build_document(project).pages[0].path]

    assert html.count('href="cases/lig1/index.html"') == 2
    assert "<h2>Scores</h2>" in html
    assert "Scores and ranking" not in html


def test_project_score_table_keeps_scored_unranked_and_excludes_rejected() -> None:
    cases = (
        CaseRecord(
            "scored",
            "Scored",
            (),
            scores={"rank_score": -10.0, "other": -9.0},
        ),
        CaseRecord(
            "missing_rank",
            "Missing rank",
            (),
            scores={"other": -5.0},
        ),
        CaseRecord("unscored", "Unscored", ()),
        CaseRecord(
            "rejected",
            "Rejected",
            (),
            status="preflight_rejected",
            scores={"rank_score": -100.0, "other": -100.0},
        ),
    )
    columns = (
        ScoreColumn("rank_score", "Rank score"),
        ScoreColumn("other", "Other score"),
    )

    ranked = build_document(
        ProjectRecord("ranked", "Ranked", cases, columns, rank_by="rank_score")
    )
    ranked_table = next(
        block
        for slot in ranked.pages[0].slots
        for block in slot.blocks
        if block.block_id == "project-scores"
    )
    assert isinstance(ranked_table, TableBlock)
    assert [(row[0], row[1].label) for row in ranked_table.rows] == [
        (1, "Scored"),
        (None, "Missing rank"),
    ]

    unranked = build_document(
        ProjectRecord("unranked", "Unranked", cases, columns, rank_by=None)
    )
    unranked_table = next(
        block
        for slot in unranked.pages[0].slots
        for block in slot.blocks
        if block.block_id == "project-scores"
    )
    assert isinstance(unranked_table, TableBlock)
    assert [row[0].label for row in unranked_table.rows] == [
        "Scored",
        "Missing rank",
    ]

    rejected_only = build_document(
        ProjectRecord(
            "rejected_only",
            "Rejected only",
            (cases[-1],),
            columns,
            rank_by="rank_score",
        )
    )
    assert "project-scores" not in {
        block.block_id
        for slot in rejected_only.pages[0].slots
        for block in slot.blocks
    }


def test_correlation_columns_and_notes_are_data_driven() -> None:
    project = ProjectRecord(
        "demo",
        "Demo",
        (_minimal_project().cases[0],),
        correlations=(
            CorrelationRecord(
                "AF3 global ipTM",
                "gb_mean",
                3,
                pearson_r=0.8,
                spearman_r=0.5,
                kendall_tau_b=0.33,
                reference_mode="descriptive_association",
                notes=("Only 3 paired cases; estimates are unstable.",),
            ),
        ),
    )

    project_page = build_document(project).pages[0]
    correlation = next(
        block
        for slot in project_page.slots
        for block in slot.blocks
        if block.block_id == "project-correlations"
    )

    assert isinstance(correlation, TableBlock)
    assert tuple(column.key for column in correlation.columns) == (
        "reference",
        "score",
        "status",
        "n",
        "pearson",
        "spearman",
        "kendall",
    )
    assert "Only 3 paired cases" in correlation.caption


def test_small_multiple_groups_render_without_line_point_clouds() -> None:
    panel = PlotPanel("ligand-rmsd", "Ligand RMSD", _plot())
    case = CaseRecord(
        "lig1",
        "Ligand 1",
        ("rep01",),
        trajectory_plots=(panel,),
    )
    document = build_document(_minimal_project(case=case))
    html = render_document(document)[document.pages[1].path]

    assert 'class="plot-grid"' in html
    assert "<h3>Ligand RMSD</h3>" in html
    assert "<circle" not in render_plot_svg(_plot())
    emphasized = PlotSpec(
        PlotKind.LINE,
        "RMSD",
        "ΔG",
        (
            PlotSeries(
                "rep01",
                "rep01",
                (
                    PlotPoint(1.0, 1.0),
                    PlotPoint(2.0, 0.0, "Basin center", emphasized=True),
                ),
            ),
        ),
    )
    assert render_plot_svg(emphasized).count("<circle") == 1


def test_scatter_renderer_uses_zero_floor_and_translucent_points() -> None:
    scatter = PlotSpec(
        PlotKind.SCATTER,
        "Prediction",
        "Reference",
        (
            PlotSeries(
                "correlation",
                "Cases",
                (PlotPoint(1.2, -40.0), PlotPoint(2.1, -35.0)),
            ),
        ),
        x_min=0.0,
    )

    svg = render_plot_svg(scatter)

    assert 'fill-opacity="0.48"' in svg
    assert '>0</text>' in svg


def test_partial_replica_inventory_is_explicit_and_ordered() -> None:
    case = CaseRecord(
        case_id="lig1",
        label="Ligand 1",
        expected_replica_ids=("rep01", "rep02", "rep03"),
        replicas=(
            ReplicaRecord("rep03", "completed"),
            ReplicaRecord("extra", "completed"),
        ),
    )
    inventory = materialize_replica_inventory(case)
    assert [(record.replica_id, record.status) for record in inventory] == [
        ("rep01", "missing"),
        ("rep02", "missing"),
        ("rep03", "completed"),
        ("extra", "completed"),
    ]
    page = build_document(_minimal_project(case=case)).pages[1]
    inventory_slot = next(slot for slot in page.slots if slot.slot is NamedSlot.INVENTORY)
    assert [block.block_id for block in inventory_slot.blocks] == [
        "case-partial-inventory",
        "case-replica-summary",
    ]
    table = inventory_slot.blocks[1]
    assert isinstance(table, TableBlock)
    assert tuple(row[0] for row in table.rows) == ("rep01", "rep02", "rep03", "extra")


def test_replica_status_qc_and_model_scores_have_one_canonical_table() -> None:
    case = CaseRecord(
        case_id="lig1",
        label="Ligand 1",
        expected_replica_ids=("rep01", "rep02"),
        replicas=(
            ReplicaRecord(
                "rep01",
                "completed",
                seed=101,
                frame_count=101,
                qc_verdict="WARN",
                scores={"gb_mean": -15.2},
            ),
        ),
        status="partial",
        outcome="partial",
        qc_verdict="WARN",
        scores={
            "gb_mean": -15.2,
            "gb_stddev": 0.7,
            "gb_sem": 0.5,
        },
        qc_issues=(
            QcIssue("contact-loss", "Reference contacts were lost", "WARN", "rep01"),
        ),
    )
    project = ProjectRecord(
        "demo",
        "Demo",
        (case,),
        score_columns=(
            ScoreColumn("gb_mean", "GB mean"),
            ScoreColumn("gb_stddev", "GB sample SD"),
            ScoreColumn("gb_sem", "GB SEM"),
        ),
        rank_by="gb_mean",
    )

    document = build_document(project)
    project_page, case_page = document.pages
    project_blocks = [
        block for slot in project_page.slots for block in slot.blocks
    ]
    case_blocks = [block for slot in case_page.slots for block in slot.blocks]

    assert "project-qc" not in {block.block_id for block in project_blocks}
    project_inventory = next(
        block for block in project_blocks if block.block_id == "project-inventory"
    )
    assert isinstance(project_inventory, TableBlock)
    assert tuple(column.key for column in project_inventory.columns) == (
        "case",
        "status",
        "outcome",
        "replicas",
        "qc",
        "qc_evaluated",
        "qc_issues",
    )

    replica_summary = next(
        block for block in case_blocks if block.block_id == "case-replica-summary"
    )
    assert isinstance(replica_summary, TableBlock)
    assert tuple(column.key for column in replica_summary.columns) == (
        "replica",
        "declared",
        "status",
        "seed",
        "frames",
        "qc",
        "gb_mean",
    )
    assert replica_summary.rows == (
        ("rep01", "yes", "completed", 101, 101, "WARN", -15.2),
        ("rep02", "yes", "missing", None, None, None, None),
    )

    aggregate = next(
        block for block in case_blocks if block.block_id == "case-scores"
    )
    assert isinstance(aggregate, TableBlock)
    assert aggregate.rows == ((-15.2, 0.7, 0.5),)
    qc = next(block for block in case_blocks if block.block_id == "case-qc")
    assert isinstance(qc, TableBlock)
    assert qc.rows == (
        ("rep01", "WARN", "contact-loss", "Reference contacts were lost"),
    )

    html = render_document(document)[case_page.path]
    assert html.count("rep02</td>") == 1
    assert "No records were supplied." not in html


def test_contributor_cannot_inject_unknown_blocks_or_duplicate_ids() -> None:
    with pytest.raises(TypeError, match="typed report blocks"):
        Contribution(  # type: ignore[arg-type]
            PageTarget.project(),
            NamedSlot.OVERVIEW,
            "<script>bad()</script>",
            "unsafe",
        )

    class Duplicate:
        contributor_id = "duplicate"

        def contribute(self, context):
            return (
                Contribution(
                    PageTarget.project(),
                    NamedSlot.ARTIFACTS,
                    AssetBlock("project-overview", "Collision", (Link("x", "x.txt"),)),
                    self.contributor_id,
                ),
            )

    with pytest.raises(ValueError, match="Duplicate block ID"):
        build_document(_minimal_project(), contributors=(Duplicate(),))

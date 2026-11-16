"""Standard MMPBSA panels and deterministic named-slot composition."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from .models import (
    SLOT_ORDER,
    AssetBlock,
    CaseRecord,
    Contribution,
    CorrelationRecord,
    DocumentSpec,
    Link,
    NamedSlot,
    NoticeBlock,
    NoticeLevel,
    PageSpec,
    PageTarget,
    PlotBlock,
    PlotGroupBlock,
    ProjectRecord,
    QcIssue,
    ReplicaRecord,
    Scalar,
    ScoreColumn,
    SlotContent,
    TableBlock,
    TableColumn,
    case_page_path,
)

_STANDARD_CONTRIBUTOR = "mmpbsa.report.standard"


@dataclass(frozen=True, slots=True)
class ReportContext:
    """Read-only input exposed to structural report contributors."""

    project: ProjectRecord

    @property
    def page_targets(self) -> tuple[PageTarget, ...]:
        return (PageTarget.project(),) + tuple(
            PageTarget.case(case.case_id) for case in self.project.cases
        )

    def case(self, case_id: str) -> CaseRecord:
        return next(case for case in self.project.cases if case.case_id == case_id)


@runtime_checkable
class ReportContributor(Protocol):
    """A safe extension point; implementations return typed blocks only."""

    contributor_id: str

    def contribute(self, context: ReportContext) -> Iterable[Contribution]:
        ...


@dataclass(frozen=True, slots=True)
class StaticContributor:
    """A deterministic contributor assembled from a validated wire contract."""

    contributor_id: str
    contributions: tuple[Contribution, ...]

    def __post_init__(self) -> None:
        if not self.contributor_id:
            raise ValueError("Static contributor ID cannot be empty")
        if any(
            item.contributor_id != self.contributor_id
            for item in self.contributions
        ):
            raise ValueError("Static contribution IDs must match the contributor")

    def contribute(self, context: ReportContext) -> tuple[Contribution, ...]:
        return self.contributions


def build_document(
    project: ProjectRecord,
    *,
    contributors: Sequence[ReportContributor] = (),
) -> DocumentSpec:
    """Build the fixed project/case page tree from explicit records.

    Contributor ordering never depends on plugin discovery order. Contributions
    are ordered by canonical slot, numeric priority, contributor ID, and block
    ID. Duplicate IDs are rejected instead of silently overriding content.
    """

    context = ReportContext(project)
    contributions = list(_standard_contributions(project))
    contributor_ids: list[str] = []
    for contributor in contributors:
        contributor_id = str(getattr(contributor, "contributor_id", ""))
        if not contributor_id:
            raise TypeError("Report contributors must expose contributor_id")
        contributor_ids.append(contributor_id)
    if len(contributor_ids) != len(set(contributor_ids)):
        raise ValueError("Report contributor IDs must be unique")

    for contributor in sorted(contributors, key=lambda item: item.contributor_id):
        produced = contributor.contribute(context)
        for contribution in produced:
            if not isinstance(contribution, Contribution):
                raise TypeError("Contributors must return Contribution values")
            if contribution.contributor_id != contributor.contributor_id:
                raise ValueError("Contribution contributor_id does not match its provider")
            contributions.append(contribution)

    targets = context.page_targets
    valid_targets = set(targets)
    for contribution in contributions:
        if contribution.page not in valid_targets:
            raise ValueError(f"Contribution targets an unknown page: {contribution.page.page_id}")

    pages: list[PageSpec] = []
    for target in targets:
        if target == PageTarget.project():
            title = project.title
            path = PurePosixPath("index.html")
        else:
            case = context.case(str(target.case_id))
            title = case.label
            path = case_page_path(case.case_id)
        page_contributions = [
            contribution for contribution in contributions if contribution.page == target
        ]
        block_ids = tuple(item.block.block_id for item in page_contributions)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError(f"Duplicate block ID on {target.page_id}")
        slots: list[SlotContent] = []
        for slot in SLOT_ORDER:
            items = sorted(
                (item for item in page_contributions if item.slot is slot),
                key=lambda item: (item.priority, item.contributor_id, item.block.block_id),
            )
            if items:
                slots.append(SlotContent(slot=slot, blocks=tuple(item.block for item in items)))
        pages.append(PageSpec(target=target, path=path, title=title, slots=tuple(slots)))
    return DocumentSpec(title=project.title, pages=tuple(pages))


build_standard_document = build_document


def materialize_replica_inventory(case: CaseRecord) -> tuple[ReplicaRecord, ...]:
    """Retain all expected replicas and explicitly represent missing workers."""

    by_id = {record.replica_id: record for record in case.replicas}
    records = [
        by_id.get(
            replica_id,
            ReplicaRecord(replica_id=replica_id, status="missing"),
        )
        for replica_id in case.expected_replica_ids
    ]
    expected = set(case.expected_replica_ids)
    records.extend(
        sorted(
            (item for item in case.replicas if item.replica_id not in expected),
            key=lambda item: item.replica_id,
        )
    )
    return tuple(records)


def _standard_contributions(project: ProjectRecord) -> Iterable[Contribution]:
    yield from _project_panels(project)
    for case in project.cases:
        yield from _case_panels(project, case)


def _add(
    page: PageTarget,
    slot: NamedSlot,
    block: NoticeBlock | TableBlock | PlotBlock | PlotGroupBlock | AssetBlock,
    priority: int,
) -> Contribution:
    return Contribution(
        page=page,
        slot=slot,
        block=block,
        contributor_id=_STANDARD_CONTRIBUTOR,
        priority=priority,
    )


def _project_panels(project: ProjectRecord) -> Iterable[Contribution]:
    page = PageTarget.project()
    overview: dict[str, Scalar] = {
        "Project": project.project_id,
        "Cases": len(project.cases),
    }
    if project.generated_at:
        overview["Generated"] = project.generated_at
    overview.update(project.metadata)
    yield _add(
        page,
        NamedSlot.OVERVIEW,
        _mapping_table("project-overview", "Overview", overview),
        10,
    )

    if project.protocol:
        yield _add(
            page,
            NamedSlot.PROTOCOL,
            _mapping_table("project-protocol", "Protocol", project.protocol),
            10,
        )

    inventory_rows: list[tuple[object, ...]] = []
    preflight_only = bool(project.cases) and all(case.is_preflight for case in project.cases)
    partial_cases = 0
    for case in project.cases:
        preflight_rejected = case.is_preflight
        replicas = materialize_replica_inventory(case)
        complete = sum(replica.status.lower() in {"completed", "complete"} for replica in replicas)
        missing = sum(replica.status.lower() == "missing" for replica in replicas)
        if missing:
            partial_cases += 1
        inventory_rows.append(
            (
                Link(case.label, case_page_path(case.case_id).as_posix()),
                case.status,
                case.outcome,
                (
                    "not scheduled"
                    if preflight_rejected
                    else f"{complete}/{len(case.expected_replica_ids)}"
                    if case.expected_replica_ids
                    else "inventory unavailable"
                ),
                "not applicable" if preflight_rejected else case.qc_verdict,
                sum(replica.qc_verdict is not None for replica in replicas),
                len(case.qc_issues),
            )
        )
    if partial_cases:
        yield _add(
            page,
            NamedSlot.INVENTORY,
            NoticeBlock(
                "project-partial-inventory",
                "Partial replica inventory",
                f"{partial_cases} case(s) have one or more missing declared replicas. "
                "Available scientific results remain visible and are not imputed.",
                NoticeLevel.WARNING,
            ),
            5,
        )
    yield _add(
        page,
        NamedSlot.INVENTORY,
        TableBlock(
            "project-inventory",
            "Case inventory and status",
            (
                TableColumn("case", "Case"),
                TableColumn("status", "Execution status"),
                TableColumn("outcome", "Scientific outcome"),
            ) + (() if preflight_only else (
                TableColumn("replicas", "Completed replicas"),
                TableColumn("qc", "QC"),
                TableColumn("qc_evaluated", "QC replicas", numeric=True),
                TableColumn("qc_issues", "QC issues", numeric=True),
            )),
            tuple(row[:3] for row in inventory_rows)
            if preflight_only
            else tuple(inventory_rows),
            "Preflight admission states only; no MD replicas or scientific QC have been scheduled."
            if preflight_only else (
                "Execution completeness and scientific QC are summarized once per case. "
                "The replica denominator is protocol-declared."
            ),
        ),
        10,
    )

    score_columns = _score_columns(project)
    score_rows = (
        _ranked_case_rows(project, score_columns) if score_columns else ()
    )
    if score_rows:
        table_columns: list[TableColumn] = []
        rank_model = (
            project.rank_by.removesuffix("_mean").upper()
            if project.rank_by is not None
            else None
        )
        if project.rank_by is not None:
            table_columns.append(TableColumn("rank", "Rank", numeric=True))
        table_columns.append(TableColumn("case", "Case"))
        if project.comparison_contract and rank_model is not None:
            table_columns.extend(
                (
                    TableColumn(
                        "internal_dielectric",
                        f"{rank_model} rank ε",
                        numeric=True,
                    ),
                    TableColumn("comparison", f"{rank_model} rank eligibility"),
                )
            )
        table_columns.extend(
            TableColumn(column.key, column.label, column.unit, numeric=True)
            for column in score_columns
        )
        yield _add(
            page,
            NamedSlot.SCORES,
            TableBlock(
                "project-scores",
                "Scores and ranking" if project.rank_by else "Scores",
                tuple(table_columns),
                score_rows,
                (
                    (
                        f"Rank, rank ε, and eligibility apply only to {rank_model}; "
                        "eligible cases are ranked within that model's ε group. "
                        "Other model score columns are descriptive here; each "
                        "case page shows their own eligibility and ε."
                        if project.comparison_contract
                        else f"Finite values are ranked ascending by {project.rank_by}; "
                        "other scored cases retain rank N/A."
                    )
                    if project.rank_by
                    else "Aggregate scores are shown in declared case order."
                ),
            ),
            10,
        )

    if project.correlations:
        yield _add(
            page,
            NamedSlot.CORRELATION,
            _correlation_table("project-correlations", project.correlations),
            10,
        )
    if project.correlation_plot is not None:
        yield _add(
            page,
            NamedSlot.CORRELATION,
            PlotBlock(
                "project-correlation-plot",
                "Correlation plot",
                project.correlation_plot,
                "Descriptive analysis only; QC labels remain independent.",
            ),
            20,
        )
    if project.artifacts:
        yield _add(
            page,
            NamedSlot.ARTIFACTS,
            AssetBlock("project-artifacts", "Project artifacts", project.artifacts),
            10,
        )
    if project.provenance:
        yield _add(
            page,
            NamedSlot.PROVENANCE,
            _mapping_table("project-provenance", "Provenance", project.provenance),
            10,
        )


def _case_panels(project: ProjectRecord, case: CaseRecord) -> Iterable[Contribution]:
    page = PageTarget.case(case.case_id)
    preflight_rejected = case.is_preflight
    preflight_reason = case.metadata.get("Preflight reason")
    overview: dict[str, Scalar] = {
        "Case": case.case_id,
        "Execution status": case.status,
        "Scientific outcome": case.outcome,
        "QC": case.qc_verdict,
    }
    if case.is_preflight:
        overview.pop("QC")
    overview.update(
        {
            key: value
            for key, value in case.metadata.items()
            if key != "Preflight reason"
        }
    )
    yield _add(page, NamedSlot.OVERVIEW, _mapping_table("case-overview", "Overview", overview), 10)
    if preflight_rejected:
        title, level = {
            "preflight_pending": ("Preflight pending", NoticeLevel.WARNING),
            "preflight_skipped": ("Preflight skipped", NoticeLevel.INFO),
            "preflight_rejected": ("Preflight rejected", NoticeLevel.BLOCKED),
        }[case.status]
        yield _add(
            page,
            NamedSlot.OVERVIEW,
            NoticeBlock(
                "case-" + case.status.replace("_", "-"),
                title,
                str(preflight_reason or title + "."),
                level,
            ),
            20,
        )
        if case.provenance:
            yield _add(
                page,
                NamedSlot.PROVENANCE,
                _mapping_table(
                    "case-provenance",
                    "Provenance",
                    case.provenance,
                ),
                10,
            )
        return
    if case.protocol:
        yield _add(
            page,
            NamedSlot.PROTOCOL,
            _mapping_table("case-protocol", "Protocol", case.protocol),
            10,
        )

    inventory = materialize_replica_inventory(case)
    missing = tuple(
        replica.replica_id
        for replica in inventory
        if replica.status.lower() == "missing"
    )
    if not case.expected_replica_ids:
        yield _add(
            page,
            NamedSlot.INVENTORY,
            NoticeBlock(
                "case-inventory-unavailable",
                "Replica inventory unavailable",
                "The analysis input did not declare an expected replica inventory.",
                NoticeLevel.WARNING,
            ),
            5,
        )
    elif missing:
        yield _add(
            page,
            NamedSlot.INVENTORY,
            NoticeBlock(
                "case-partial-inventory",
                "Partial result",
                "Missing declared replicas: " + ", ".join(missing) + ".",
                NoticeLevel.WARNING,
            ),
            5,
        )
    expected = set(case.expected_replica_ids)
    score_columns = _score_columns(project)
    replica_score_columns = _replica_score_columns(project, score_columns)
    yield _add(
        page,
        NamedSlot.INVENTORY,
        TableBlock(
            "case-replica-summary",
            "Replica execution, QC, and scores",
            (
                TableColumn("replica", "Replica"),
                TableColumn("declared", "Declared"),
                TableColumn("status", "Execution status"),
                TableColumn("seed", "Seed", numeric=True),
                TableColumn("frames", "Frames", numeric=True),
                TableColumn("qc", "QC"),
            )
            + tuple(
                TableColumn(column.key, column.label, column.unit, numeric=True)
                for column in replica_score_columns
            ),
            tuple(
                (
                    replica.replica_id,
                    "yes" if replica.replica_id in expected else "unexpected",
                    replica.status,
                    replica.seed,
                    replica.frame_count,
                    replica.qc_verdict,
                )
                + tuple(
                    _format_score(replica.scores.get(column.key), column)
                    for column in replica_score_columns
                )
                for replica in inventory
            ),
            "Each replica appears once. Missing and unexpected workers are retained "
            "explicitly; absent QC and score values are not imputed.",
        ),
        10,
    )

    if score_columns and _case_has_aggregate_scores(case, score_columns):
        caption = (
            "Mean, replica sample SD, and SEM remain case-level statistics; "
            "per-replica model scores are shown only in the replica summary."
        )
        if project.comparison_contract and case.comparisons:
            statuses = []
            for score_key, comparison in case.comparisons.items():
                status = (
                    f"rank {comparison.rank} within internal ε={comparison.dielectric:g}"
                    if comparison.eligible and comparison.rank is not None
                    else "excluded: "
                    + ", ".join(reason.replace("_", " ") for reason in comparison.reasons)
                )
                statuses.append(f"{score_key}: {status}")
            caption += " Comparison eligibility: " + "; ".join(statuses) + "."
        yield _add(
            page,
            NamedSlot.SCORES,
            TableBlock(
                "case-scores",
                "Case aggregate scores",
                tuple(
                    TableColumn(column.key, column.label, column.unit, numeric=True)
                    for column in score_columns
                ),
                (
                    tuple(
                        _format_score(case.scores.get(column.key), column)
                        for column in score_columns
                    ),
                ),
                caption,
            ),
            10,
        )

    if case.qc_issues:
        yield _add(
            page,
            NamedSlot.QC,
            TableBlock(
                "case-qc",
                "QC issues",
                (
                    TableColumn("scope", "Scope"),
                    TableColumn("verdict", "Verdict"),
                    TableColumn("code", "Code"),
                    TableColumn("message", "Message"),
                ),
                tuple(_qc_issue_row(issue) for issue in case.qc_issues),
                "Only actionable issue records are listed here. Case and replica "
                "verdicts are reported in the overview and replica summary.",
            ),
            10,
        )

    if case.qc_contact_context:
        yield _add(
            page, NamedSlot.QC,
            TableBlock(
                "case-qc-contact-context", "Initial-contact loss context (BLOCK replicas)",
                (
                    TableColumn("replica", "Replica"),
                    TableColumn("samples", "Samples", numeric=True),
                    TableColumn("zero_fraction", "Initial-contact zero fraction", numeric=True),
                    TableColumn("zero_run", "Longest zero-contact run (samples)", numeric=True),
                    TableColumn(
                        "other_contacts", "Zero-initial samples with other contacts", numeric=True
                    ),
                    TableColumn("total_min", "Minimum native + nonnative contacts", numeric=True),
                ),
                tuple((
                    item.replica_id, item.sample_count, item.native_zero_fraction,
                    item.longest_native_zero_samples, item.nonnative_at_native_zero_samples,
                    item.total_contacts_min,
                ) for item in case.qc_contact_context),
                "Descriptive context, not additional QC gates. Losing all initial contacts "
                "does not mean losing all receptor contacts or prove permanent dissociation. "
                "Runs are sampled observations, not independently established durations. "
                "Other/total contacts are unavailable unless explicit nonnative counts exist.",
            ),
            20,
        )

    if case.trajectory_plots:
        yield _add(
            page,
            NamedSlot.TRAJECTORY,
            PlotGroupBlock(
                "case-trajectory",
                "Trajectory diagnostics",
                case.trajectory_plots,
                "Each panel contains one physical observable across replicas.",
            ),
            10,
        )
    elif case.trajectory is not None:
        yield _add(
            page,
            NamedSlot.TRAJECTORY,
            PlotBlock(
                "case-trajectory",
                "Trajectory diagnostics",
                case.trajectory,
                "Axes must use the absolute frame or time coordinates emitted by trajectory QC.",
            ),
            10,
        )
    else:
        yield _add(
            page,
            NamedSlot.TRAJECTORY,
            _unavailable_block(
                "case-trajectory-unavailable",
                "Trajectory diagnostics unavailable",
                "No typed trajectory diagnostic series was supplied.",
            ),
            10,
        )
    if case.energy_plots:
        yield _add(
            page,
            NamedSlot.CONVERGENCE,
            PlotGroupBlock(
                "case-energy-series",
                "Per-frame binding score",
                case.energy_plots,
                "Each panel contains one analysis model across replicas.",
            ),
            5,
        )
    elif case.energy is not None:
        yield _add(
            page,
            NamedSlot.CONVERGENCE,
            PlotBlock(
                "case-energy-series",
                "Per-frame binding score",
                case.energy,
                "Values and units are preserved from the scoring payload.",
            ),
            5,
        )
    if case.convergence_plots:
        yield _add(
            page,
            NamedSlot.CONVERGENCE,
            PlotGroupBlock(
                "case-convergence",
                "Binding-score convergence diagnostics",
                case.convergence_plots,
                "Cumulative means are separated by model and compared across replicas. "
                "A flattened cumulative mean alone does not establish equilibrium or "
                "statistical convergence; inspect replica agreement, drift and sampling.",
            ),
            10,
        )
    elif case.convergence is not None:
        yield _add(
            page,
            NamedSlot.CONVERGENCE,
            PlotBlock(
                "case-convergence", "Binding-score convergence diagnostics", case.convergence,
                "A flattened cumulative mean alone does not establish statistical convergence.",
            ),
            10,
        )
    else:
        yield _add(
            page,
            NamedSlot.CONVERGENCE,
            _unavailable_block(
                "case-convergence-unavailable",
                "Binding-score convergence diagnostics unavailable",
                "No uniquely identifiable per-frame binding-score series was supplied.",
            ),
            10,
        )
    if case.occupancy_basin is not None:
        yield _add(
            page,
            NamedSlot.CONVERGENCE,
            PlotBlock(
                "case-basin",
                "Ligand RMSD occupancy basin",
                case.occupancy_basin,
                (
                    "Each replica is PBC-normalized and receptor-interface-backbone "
                    "fitted to its own post-NPT production t=0 pose. Ligand RMSD is "
                    "mass-weighted over small-molecule heavy atoms or peptide "
                    "backbone atoms. Curves use −RT ln(P/Pmax) from RMSD occupancy; "
                    "they are conformational diagnostics, not GB/PB binding "
                    "energies or rigorous free-energy surfaces."
                    + (
                        f" {case.occupancy_basin_note}"
                        if case.occupancy_basin_note
                        else ""
                    )
                ),
            ),
            20,
        )
    else:
        yield _add(
            page,
            NamedSlot.CONVERGENCE,
            _unavailable_block(
                "case-basin-unavailable",
                "Ligand RMSD occupancy basin unavailable",
                case.occupancy_basin_note
                or "Equilibrated-pose ligand RMSD diagnostics were not supplied.",
            ),
            20,
        )
    # Correlation is a project-level comparison across cases. A single case is
    # not an unavailable correlation sample; the concept is not applicable.
    if case.decomposition:
        yield _add(
            page,
            NamedSlot.DECOMPOSITION,
            _decomposition_table(case),
            10,
        )
    else:
        yield _add(
            page,
            NamedSlot.DECOMPOSITION,
            _unavailable_block(
                "case-decomposition-unavailable",
                "Decomposition unavailable",
                "No typed per-residue decomposition result was supplied.",
            ),
            10,
        )
    if case.actions:
        yield _add(
            page,
            NamedSlot.ACTIONS,
            AssetBlock(
                "case-actions",
                "PyMOL and actions",
                case.actions,
                "Each action identifies its own replica/state and never implies a shared object.",
            ),
            10,
        )
    else:
        yield _add(
            page,
            NamedSlot.ACTIONS,
            _unavailable_block(
                "case-actions-unavailable",
                "PyMOL and actions unavailable",
                "No typed display action or PyMOL asset was supplied.",
            ),
            10,
        )
    if case.artifacts:
        yield _add(
            page,
            NamedSlot.ARTIFACTS,
            AssetBlock("case-artifacts", "Artifacts", case.artifacts),
            10,
        )
    if case.provenance:
        yield _add(
            page,
            NamedSlot.PROVENANCE,
            _mapping_table("case-provenance", "Provenance", case.provenance),
            10,
        )


def _mapping_table(block_id: str, title: str, values: Mapping[str, Scalar]) -> TableBlock:
    return TableBlock(
        block_id,
        title,
        (TableColumn("key", "Field"), TableColumn("value", "Value")),
        tuple((key, values[key]) for key in sorted(values)),
    )


def _unavailable_block(block_id: str, title: str, body: str) -> NoticeBlock:
    return NoticeBlock(block_id, title, body, NoticeLevel.INFO)


def _score_columns(project: ProjectRecord) -> tuple[ScoreColumn, ...]:
    if project.score_columns:
        return project.score_columns
    keys = sorted(
        {
            key
            for case in project.cases
            for key in (
                *case.scores.keys(),
                *(score_key for replica in case.replicas for score_key in replica.scores),
            )
        }
    )
    return tuple(ScoreColumn(key=key, label=key) for key in keys)


def _case_has_aggregate_scores(
    case: CaseRecord,
    columns: tuple[ScoreColumn, ...],
) -> bool:
    return any(case.scores.get(column.key) is not None for column in columns)


def _replica_score_columns(
    project: ProjectRecord,
    columns: tuple[ScoreColumn, ...],
) -> tuple[ScoreColumn, ...]:
    """Return columns whose keys are present in typed per-replica records.

    Aggregate-only columns such as sample SD and SEM therefore stay in the
    case table instead of producing repeated all-N/A replica cells.
    """

    replica_keys = {
        key
        for case in project.cases
        for replica in case.replicas
        for key in replica.scores
    }
    return tuple(
        column
        for column in columns
        if column.key in replica_keys
    )


def _ranked_case_rows(
    project: ProjectRecord,
    columns: tuple[ScoreColumn, ...],
) -> tuple[tuple[object, ...], ...]:
    scored_cases = tuple(
        case
        for case in project.cases
        if not case.is_preflight
        and _case_has_aggregate_scores(case, columns)
    )
    if project.comparison_contract and project.rank_by is not None:
        key = project.rank_by
        ordered = tuple(
            (
                case.comparisons[key].rank
                if key is not None and key in case.comparisons
                else None,
                case,
            )
            for case in sorted(
                scored_cases,
                key=lambda item: (
                    item.comparisons[key].rank is None
                    if key is not None and key in item.comparisons
                    else True,
                    item.comparisons[key].dielectric
                    if key is not None
                    and key in item.comparisons
                    and item.comparisons[key].dielectric is not None
                    else float("inf"),
                    item.comparisons[key].rank
                    if key is not None
                    and key in item.comparisons
                    and item.comparisons[key].rank is not None
                    else float("inf"),
                    item.case_id,
                ),
            )
        )
    elif project.rank_by is None:
        ordered: tuple[tuple[int | None, CaseRecord], ...] = tuple(
            (None, case) for case in scored_cases
        )
    else:
        finite = sorted(
            (
                (float(case.scores[project.rank_by]), case.case_id, case)
                for case in scored_cases
                if case.scores.get(project.rank_by) is not None
                and math.isfinite(float(case.scores[project.rank_by]))
            ),
            key=lambda item: (item[0], item[1]),
        )
        ordered = tuple(
            (rank, case)
            for rank, (_, _, case) in enumerate(finite, 1)
        )
        ranked_case_ids = {case.case_id for _, case in ordered}
        ordered += tuple(
            (None, case)
            for case in scored_cases
            if case.case_id not in ranked_case_ids
        )
    rows: list[tuple[object, ...]] = []
    for rank, case in ordered:
        prefix: tuple[object, ...] = ()
        if project.rank_by is not None:
            prefix += (rank,)
        prefix += (Link(case.label, case_page_path(case.case_id).as_posix()),)
        if project.comparison_contract and project.rank_by is not None:
            comparison = case.comparisons.get(project.rank_by or "")
            prefix += (
                comparison.dielectric if comparison is not None else None,
                (
                    "Eligible"
                    if comparison is not None and comparison.eligible
                    else "; ".join(reason.replace("_", " ") for reason in comparison.reasons)
                    if comparison is not None
                    else "Unavailable"
                ),
            )
        rows.append(
            prefix
            + tuple(_format_score(case.scores.get(column.key), column) for column in columns)
        )
    return tuple(rows)


def _format_score(value: float | None, column: ScoreColumn) -> float | None:
    return None if value is None else round(float(value), column.precision)


def _qc_issue_row(issue: QcIssue) -> tuple[object, ...]:
    return (issue.replica_id or "case", issue.verdict, issue.code, issue.message)


def _correlation_table(
    block_id: str,
    correlations: tuple[CorrelationRecord, ...],
) -> TableBlock:
    show_dielectric = any(
        record.resolved_internal_dielectric is not None for record in correlations
    )
    available_columns = (
        ("pearson", "Pearson r", "pearson_r"),
        ("spearman", "Spearman r", "spearman_r"),
        ("kendall", "Kendall tau-b", "kendall_tau_b"),
        ("slope", "Slope", "slope"),
        ("intercept", "Intercept", "intercept"),
        ("mae", "MAE", "mae"),
        ("rmse", "RMSE", "rmse"),
        ("mse", "Mean signed error", "mean_signed_error"),
    )
    selected = tuple(
        (key, label, attribute)
        for key, label, attribute in available_columns
        if any(getattr(record, attribute) is not None for record in correlations)
    )
    notes = tuple(
        dict.fromkeys(
            note
            for record in correlations
            for note in record.notes
        )
    )
    caption = "Correlation is descriptive and is not a baseline pass/fail gate."
    if notes:
        caption += " " + " ".join(notes)
    return TableBlock(
        block_id,
        "Correlation",
        (
            TableColumn("reference", "Reference"),
            TableColumn("score", "Score"),
            *(
                (TableColumn("dielectric", "Internal ε", numeric=True),)
                if show_dielectric
                else ()
            ),
            TableColumn("status", "Status"),
            TableColumn("n", "N", numeric=True),
            *(TableColumn(key, label, numeric=True) for key, label, _ in selected),
        ),
        tuple(
            (
                record.reference,
                record.score_key,
                *(
                    (record.resolved_internal_dielectric,)
                    if show_dielectric
                    else ()
                ),
                record.status,
                record.n,
                *(getattr(record, attribute) for _, _, attribute in selected),
            )
            for record in correlations
        ),
        caption,
    )


def _decomposition_table(case: CaseRecord) -> TableBlock:
    value_keys = sorted({key for record in case.decomposition for key in record.values})
    return TableBlock(
        "case-decomposition",
        "Per-residue decomposition",
        (TableColumn("residue", "Residue"),)
        + tuple(TableColumn(key, key, numeric=True) for key in value_keys),
        tuple(
            (record.residue,) + tuple(record.values.get(key) for key in value_keys)
            for record in case.decomposition
        ),
    )

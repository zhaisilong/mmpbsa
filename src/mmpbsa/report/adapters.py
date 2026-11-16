"""Adapters from the core-neutral analysis receipt to typed report records."""

from __future__ import annotations

import math
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from mmpbsa.core.scientific_context import verified_frame_selection

from .builders import StaticContributor
from .models import (
    PREFLIGHT_STATUSES,
    AssetRequest,
    CaseRecord,
    ComparisonRecord,
    Contribution,
    CorrelationRecord,
    DecompositionRecord,
    Link,
    NamedSlot,
    NoticeBlock,
    NoticeLevel,
    PageTarget,
    PlotKind,
    PlotPanel,
    PlotPoint,
    PlotSeries,
    PlotSpec,
    ProjectRecord,
    QcContactContext,
    QcIssue,
    ReplicaRecord,
    Scalar,
    ScoreColumn,
    TableBlock,
    TableColumn,
)

ANALYSIS_SCHEMA = "mmpbsa.analysis.v3"
LEGACY_ANALYSIS_SCHEMA = "mmpbsa.analysis.v2"
REPORT_INPUT_SCHEMA = "mmpbsa.report.input.v1"


@dataclass(frozen=True, slots=True)
class AdaptedReportInput:
    """Validated adapter result plus display-only copy requests."""

    project: ProjectRecord
    asset_requests: tuple[AssetRequest, ...] = ()
    asset_sources: Mapping[str, Path] | None = None
    contributors: tuple[StaticContributor, ...] = ()
    schema_version: str = REPORT_INPUT_SCHEMA


def adapt_analysis_input(
    analysis: Mapping[str, object],
    *,
    report_configuration: Mapping[str, object] | None = None,
    asset_artifacts: Mapping[str, Path] | None = None,
    asset_root: Path | None = None,
    asset_payload: Mapping[str, object] | None = None,
) -> AdaptedReportInput:
    """Adapt current analysis and read archived v2 analyses without file discovery."""

    if not isinstance(analysis, Mapping):
        raise TypeError("Analysis payload must be a mapping")
    if analysis.get("schema_version") not in {ANALYSIS_SCHEMA, LEGACY_ANALYSIS_SCHEMA}:
        raise ValueError(
            f"Expected analysis schema {ANALYSIS_SCHEMA!r}, "
            f"received {analysis.get('schema_version')!r}"
        )
    configuration = dict(report_configuration or {})
    comparison_contract = analysis.get("schema_version") == ANALYSIS_SCHEMA
    project_context_raw = configuration.get("_project")
    project_context = (
        project_context_raw if isinstance(project_context_raw, Mapping) else {}
    )
    project_id = _identifier_or_default(
        analysis.get("project_id")
        or project_context.get("project_id")
        or configuration.get("project_id"),
        "unavailable",
    )
    protocol_id = _text_or_none(
        analysis.get("protocol_id") or project_context.get("protocol_id")
    )
    profile = analysis.get("profile")
    profile_mapping = profile if isinstance(profile, Mapping) else {}
    injected_profile = configuration.get("_profile")
    if isinstance(injected_profile, Mapping):
        profile_mapping = {**profile_mapping, **injected_profile}

    analysis_models = _analysis_models(analysis)
    model_order = _model_order(
        analysis,
        configuration,
        tuple(analysis_models),
    )
    score_columns = _model_score_columns(
        configuration.get("score_columns"),
        model_order=model_order,
        models=analysis_models,
    )
    default_rank = (
        f"{model_order[0]}_mean"
        if f"{model_order[0]}_mean"
        in {column.key for column in score_columns}
        else score_columns[0].key
    )
    rank_by_raw = configuration.get("rank_by", default_rank)
    rank_by = str(rank_by_raw) if rank_by_raw is not None else None
    if rank_by == "score_mean":
        rank_by = default_rank
    elif rank_by in model_order:
        rank_by = f"{rank_by}_mean"
    if rank_by is not None and rank_by not in {column.key for column in score_columns}:
        raise ValueError("report.rank_by must identify a configured score column")
    if comparison_contract and rank_by is not None and rank_by not in {
        f"{model_id}_mean" for model_id in model_order
    }:
        raise ValueError("analysis v3 can rank only a model mean score")

    ligand_rmsd_diagnostics = _ligand_rmsd_diagnostics(asset_payload)
    temperature_k = _protocol_temperature_k(
        configuration,
        required=bool(ligand_rmsd_diagnostics),
    )

    case_hints, global_replicas = _case_hints(project_context)
    raw_cases = analysis.get("cases")
    if not _is_sequence(raw_cases):
        raise TypeError("analysis.cases must be a sequence")
    adapted_cases: list[CaseRecord] = []
    for index, row in enumerate(raw_cases):
        if not isinstance(row, Mapping):
            raise TypeError(f"analysis.cases[{index}] must be a mapping")
        case_id = _safe_identifier(row.get("case_id"))
        if case_id is None:
            raise ValueError(f"analysis.cases[{index}].case_id must be a safe identifier")
        adapted_cases.append(
            _adapt_case(
                row,
                model_order=model_order,
                models=analysis_models,
                hint=case_hints.get(case_id, {}),
                global_replicas=global_replicas,
                ligand_rmsd_diagnostics=ligand_rmsd_diagnostics.get(case_id),
                temperature_k=temperature_k,
                comparison_contract=comparison_contract,
            )
        )
    cases = tuple(adapted_cases)
    known_case_ids = {case.case_id for case in cases}
    unknown_diagnostic_cases = set(ligand_rmsd_diagnostics).difference(
        known_case_ids | set(case_hints)
    )
    if unknown_diagnostic_cases:
        raise ValueError(
            "Report-assets diagnostics target unknown cases: "
            + ", ".join(sorted(unknown_diagnostic_cases))
        )
    cases += tuple(
        _adapt_unavailable_case(case_id, hint, global_replicas)
        for case_id, hint in sorted(case_hints.items())
        if case_id not in known_case_ids
    )
    cases = _merge_case_catalog(cases, configuration.get("case_catalog"))

    metadata = _scalar_mapping(configuration.get("metadata"))
    metadata.setdefault("Protocol", protocol_id)
    provenance = _scalar_mapping(configuration.get("provenance"))
    for key in ("profile_id", "profile_version"):
        if (value := profile_mapping.get(key)) is not None:
            provenance.setdefault(key.replace("_", " ").title(), _scalar(value))
    correlation_inputs: list[tuple[str, object]] = []
    for model_id in model_order:
        if not comparison_contract:
            correlation_inputs.append(
                (model_id, analysis_models[model_id].get("correlation"))
            )
            continue
        groups = analysis_models[model_id].get("correlation_groups")
        if not _is_sequence(groups):
            raise TypeError(
                f"analysis.models.{model_id}.correlation_groups must be a sequence"
            )
        for index, group in enumerate(groups):
            if not isinstance(group, Mapping) or group.get("model_id") != model_id:
                raise ValueError(
                    f"analysis.models.{model_id}.correlation_groups[{index}] "
                    "must identify its model"
                )
            correlation_inputs.append((model_id, group))
    correlations = tuple(
        _adapt_project_correlation(
            value,
            reference_label=str(configuration.get("reference_label", "reference")),
            score_key=f"{model_id}_mean",
        )
        for model_id, value in correlation_inputs
    )
    project = ProjectRecord(
        project_id=project_id,
        title=str(configuration.get("title") or project_id),
        generated_at=str(configuration.get("generated_at") or ""),
        cases=cases,
        score_columns=score_columns,
        rank_by=rank_by,
        comparison_contract=comparison_contract,
        metadata=metadata,
        protocol={
            **({"Protocol ID": protocol_id} if protocol_id is not None else {}),
            **_scalar_mapping(configuration.get("protocol")),
        },
        correlations=tuple(
            correlation
            for correlation in correlations
            if correlation is not None
        ),
        artifacts=_links(configuration.get("artifacts")),
        provenance=provenance,
    )
    project, requests, sources = _adapt_assets(
        project,
        configuration=configuration,
        asset_artifacts=asset_artifacts or {},
        asset_root=asset_root,
        asset_payload=asset_payload,
    )
    profile_contributions = _profile_contributions(profile_mapping, project)
    contributors = (
        (StaticContributor("profile.report", profile_contributions),)
        if profile_contributions
        else ()
    )
    return AdaptedReportInput(
        project=project,
        asset_requests=requests,
        asset_sources=sources,
        contributors=contributors,
    )


def _adapt_case(
    row: Mapping[str, object],
    *,
    model_order: tuple[str, ...],
    models: Mapping[str, Mapping[str, object]],
    hint: Mapping[str, object],
    global_replicas: tuple[str, ...],
    ligand_rmsd_diagnostics: Mapping[str, object] | None,
    temperature_k: float | None,
    comparison_contract: bool,
) -> CaseRecord:
    case_id = _identifier_or_default(row.get("case_id"), "unavailable")
    record_raw = row.get("record")
    record = record_raw if isinstance(record_raw, Mapping) else {}
    replica_rows_raw = record.get("replicas", ())
    replica_rows = (
        tuple(item for item in replica_rows_raw if isinstance(item, Mapping))
        if _is_sequence(replica_rows_raw)
        else ()
    )
    expected = _expected_replicas(record, hint, global_replicas, replica_rows)
    replicas = tuple(
        _adapt_replica(item, model_order)
        for item in replica_rows
    )
    qc_issues = tuple(
        issue
        for replica_row in replica_rows
        for issue in _extract_qc_issues(replica_row)
    )
    energy_plots, convergence_plots = _energy_plots(
        replica_rows,
        model_order,
        models,
        scientific_context=record.get("scientific_context"),
    )
    occupancy_basin, occupancy_basin_note = _occupancy_basin_plot(
        case_id,
        expected,
        ligand_rmsd_diagnostics,
        temperature_k,
    )
    explicit_trajectory = _optional_plot(record.get("trajectory"))
    explicit_energy = _optional_plot(record.get("energy"))
    explicit_convergence = _optional_plot(record.get("convergence"))
    scores: dict[str, float | None] = {}
    comparisons: dict[str, ComparisonRecord] = {}
    case_models_raw = row.get("models")
    case_models = (
        case_models_raw if isinstance(case_models_raw, Mapping) else {}
    )
    for model_id in model_order:
        model_raw = case_models.get(model_id)
        model = model_raw if isinstance(model_raw, Mapping) else {}
        if model and str(model.get("model_id", model_id)) != model_id:
            raise ValueError(
                f"analysis case model {model_id!r} has mismatched model_id"
            )
        scores.update(
            {
                f"{model_id}_mean": _finite(model.get("score_mean")),
                f"{model_id}_stddev": _finite(
                    model.get("score_stddev")
                ),
                f"{model_id}_sem": _finite(model.get("score_sem")),
            }
        )
        if comparison_contract:
            raw_reasons = model.get("comparison_exclusion_reasons")
            if not _is_sequence(raw_reasons):
                raise ValueError(
                    f"analysis case model {model_id!r} has no comparison reasons"
                )
            comparisons[f"{model_id}_mean"] = ComparisonRecord(
                eligible=model.get("comparison_eligible") is True,
                reasons=tuple(str(reason) for reason in raw_reasons),
                group=_text_or_none(model.get("comparison_group")),
                dielectric=_finite(model.get("resolved_internal_dielectric")),
                rank=_integer_or_none(model.get("rank")),
            )
    finalized = bool(row.get("finalized"))
    qc_verdict = _text_or_none(row.get("qc_verdict") or record.get("qc_verdict"))
    return CaseRecord(
        case_id=case_id,
        label=str(row.get("label") or hint.get("label") or case_id),
        expected_replica_ids=expected,
        replicas=replicas,
        status=str(record.get("status") or ("completed" if finalized else "unavailable")),
        outcome=str(row.get("outcome") or record.get("outcome") or "unavailable"),
        qc_verdict=qc_verdict,
        metadata={
            **_scalar_mapping(hint.get("metadata")),
            **(
                {"Reference": _scalar(row.get("reference"))}
                if row.get("reference") is not None
                else {}
            ),
        },
        protocol=_scalar_mapping(hint.get("protocol")),
        scores=scores,
        comparisons=comparisons,
        qc_issues=qc_issues,
        qc_contact_context=_qc_contact_context(replica_rows),
        trajectory_plots=(
            () if explicit_trajectory is not None else _trajectory_plots(
                replica_rows, scientific_context=record.get("scientific_context")
            )
        ),
        energy_plots=() if explicit_energy is not None else energy_plots,
        convergence_plots=(
            () if explicit_convergence is not None else convergence_plots
        ),
        trajectory=explicit_trajectory,
        energy=explicit_energy,
        convergence=explicit_convergence,
        occupancy_basin=occupancy_basin,
        occupancy_basin_note=occupancy_basin_note,
        correlations=_correlations(record.get("correlations")),
        correlation_plot=_optional_plot(record.get("correlation_plot")),
        decomposition=_decomposition(record.get("decomposition")),
        actions=_links(record.get("actions")),
        artifacts=_links(record.get("report_artifacts")),
        provenance={
            **_scalar_mapping(record.get("provenance")),
            **_scientific_provenance(record.get("scientific_context")),
        },
    )


def _adapt_unavailable_case(
    case_id: str,
    hint: Mapping[str, object],
    global_replicas: tuple[str, ...],
) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        label=str(hint.get("label") or case_id),
        expected_replica_ids=_replica_ids(hint.get("replicas")) or global_replicas,
        status="unavailable",
        outcome="missing",
        metadata=_scalar_mapping(hint.get("metadata")),
        protocol=_scalar_mapping(hint.get("protocol")),
    )


def _merge_case_catalog(
    cases: tuple[CaseRecord, ...],
    value: object,
) -> tuple[CaseRecord, ...]:
    """Merge a display-only validation catalog without inventing run records.

    Catalog entries may annotate workflow cases or add preflight-rejected case
    pages.  They never participate in score aggregation or correlation because
    those values continue to come exclusively from ``mmpbsa.analysis.v2``.
    """

    if value is None:
        return cases
    if not _is_sequence(value):
        raise TypeError("report.case_catalog must be a sequence")

    allowed = {
        "case_id",
        "label",
        "status",
        "reason",
        "metadata",
        "provenance",
    }
    catalog: list[Mapping[str, object]] = []
    catalog_ids: set[str] = set()
    for index, raw in enumerate(value, 1):
        label = f"report.case_catalog[{index}]"
        if not isinstance(raw, Mapping):
            raise TypeError(f"{label} must be a mapping")
        unknown = set(raw).difference(allowed)
        missing = {"case_id", "status"}.difference(raw)
        if missing:
            raise ValueError(
                f"{label} is missing fields: " + ", ".join(sorted(missing))
            )
        if unknown:
            raise ValueError(
                f"{label} has unknown fields: " + ", ".join(sorted(unknown))
            )
        case_id = _safe_identifier(raw.get("case_id"))
        if case_id is None:
            raise ValueError(f"{label}.case_id must be a safe identifier")
        if case_id in catalog_ids:
            raise ValueError(f"report.case_catalog repeats case_id {case_id!r}")
        status = str(raw.get("status"))
        if status not in {"eligible"} | PREFLIGHT_STATUSES:
            raise ValueError(
                f"{label}.status must be eligible, preflight_pending, "
                "preflight_skipped or preflight_rejected"
            )
        reason = raw.get("reason")
        if status in PREFLIGHT_STATUSES and (
            not isinstance(reason, str) or not reason.strip()
        ):
            raise ValueError(
                f"{label}.reason is required for {status} cases"
            )
        if reason is not None and (
            not isinstance(reason, str) or not reason.strip()
        ):
            raise ValueError(f"{label}.reason must be a non-empty string")
        catalog_ids.add(case_id)
        catalog.append(raw)

    by_case = {case.case_id: case for case in cases}
    merged: list[CaseRecord] = []
    for raw in catalog:
        case_id = str(raw["case_id"])
        status = str(raw["status"])
        existing = by_case.pop(case_id, None)
        catalog_metadata = _scalar_mapping(raw.get("metadata"))
        catalog_metadata["Catalog status"] = status
        if raw.get("reason") is not None:
            catalog_metadata["Preflight reason"] = str(raw["reason"])
        catalog_provenance = _scalar_mapping(raw.get("provenance"))
        if existing is not None:
            if status in PREFLIGHT_STATUSES:
                raise ValueError(
                    f"report.case_catalog marks workflow case {case_id!r} "
                    f"as {status}"
                )
            merged.append(
                replace(
                    existing,
                    label=str(raw.get("label") or existing.label),
                    metadata={**catalog_metadata, **existing.metadata},
                    provenance={**catalog_provenance, **existing.provenance},
                )
            )
            continue
        merged.append(
            CaseRecord(
                case_id=case_id,
                label=str(raw.get("label") or case_id),
                expected_replica_ids=(),
                status=(
                    status
                    if status in PREFLIGHT_STATUSES
                    else "not_scheduled"
                ),
                outcome=("not_run" if status in PREFLIGHT_STATUSES else "eligible"),
                metadata=catalog_metadata,
                provenance=catalog_provenance,
            )
        )
    merged.extend(by_case[case.case_id] for case in cases if case.case_id in by_case)
    return tuple(merged)


def _adapt_replica(
    row: Mapping[str, object],
    model_order: tuple[str, ...],
) -> ReplicaRecord:
    replica_id = _identifier_or_default(
        row.get("replica_id") or row.get("id"),
        "unavailable",
    )
    scoring_models_raw = row.get("scoring_models")
    scoring_models = (
        scoring_models_raw if isinstance(scoring_models_raw, Mapping) else {}
    )
    scoring_payloads: list[Mapping[str, object]] = []
    scores: dict[str, float | None] = {}
    for model_id in model_order:
        scoring_raw = scoring_models.get(model_id)
        scoring = scoring_raw if isinstance(scoring_raw, Mapping) else {}
        if scoring and str(scoring.get("model_id", model_id)) != model_id:
            raise ValueError(
                f"replica scoring model {model_id!r} has mismatched model_id"
            )
        scoring_payload_raw = scoring.get("payload")
        scoring_payload = (
            scoring_payload_raw
            if isinstance(scoring_payload_raw, Mapping)
            else {}
        )
        scoring_payloads.append(scoring_payload)
        # Core analysis is the sole score-selection boundary. The report
        # adapter never searches another model's payload as a fallback.
        scores[f"{model_id}_mean"] = _finite(
            scoring.get("selected_score")
        )
    qc_raw = row.get("trajectory_qc")
    qc = qc_raw if isinstance(qc_raw, Mapping) else {}
    qc_payload_raw = qc.get("payload")
    qc_payload = qc_payload_raw if isinstance(qc_payload_raw, Mapping) else qc
    verdict = _text_or_none(
        qc.get("qc_verdict")
        or qc.get("verdict")
        or qc_payload.get("qc_verdict")
        or qc_payload.get("verdict")
    )
    status = _text_or_none(row.get("status") or row.get("execution_state"))
    if status is None:
        status = "completed" if scoring_models else "unavailable"
    return ReplicaRecord(
        replica_id=replica_id,
        status=status,
        seed=_integer_or_none(row.get("seed")),
        frame_count=_find_integer(
            (qc_payload, *scoring_payloads, row),
            ("frame_count", "n_frames", "frames"),
        ),
        qc_verdict=verdict,
        scores=scores,
    )


def _extract_qc_issues(row: Mapping[str, object]) -> tuple[QcIssue, ...]:
    qc_raw = row.get("trajectory_qc")
    qc = qc_raw if isinstance(qc_raw, Mapping) else {}
    payload_raw = qc.get("payload")
    payload = payload_raw if isinstance(payload_raw, Mapping) else qc
    candidates = payload.get("issues", ())
    if not _is_sequence(candidates):
        return ()
    replica_id = _text_or_none(row.get("replica_id") or row.get("id"))
    issues: list[QcIssue] = []
    for index, value in enumerate(candidates):
        if not isinstance(value, Mapping):
            continue
        issues.append(
            QcIssue(
                code=_identifier_or_default(value.get("code"), f"issue-{index + 1}"),
                message=str(value.get("message") or "QC issue details unavailable"),
                verdict=str(value.get("verdict") or "unavailable"),
                replica_id=replica_id,
            )
        )
    return tuple(issues)


def _qc_contact_context(
    replica_rows: tuple[Mapping[str, object], ...],
) -> tuple[QcContactContext, ...]:
    contexts = []
    for row in replica_rows:
        qc = _stage_payload(row, "trajectory_qc")
        verdict = str(row.get("qc_verdict") or qc.get("verdict") or "").upper()
        replica_id = _safe_identifier(row.get("replica_id"))
        table = qc.get("native_contacts")
        if verdict != "BLOCK" or replica_id is None or not isinstance(table, Mapping):
            continue
        columns = _table_columns(table)
        if not columns:
            continue
        native = tuple(column for column in columns[1:] if (
            _normalized_column(column).endswith("native")
            and not _normalized_column(column).endswith("nonnative")
        ))
        nonnative = tuple(column for column in columns[1:]
                          if _normalized_column(column).endswith("nonnative"))
        if len(native) != 1:
            continue
        points = _table_points(table, columns[0], native[0])
        if not points or any(point.y < 0 for point in points):
            continue
        current = longest = zero_count = 0
        for point in points:
            if point.y == 0:
                zero_count += 1
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        nonnative_points = (
            _table_points(table, columns[0], nonnative[0]) if len(nonnative) == 1 else None
        )
        paired = nonnative_points is not None and (
            tuple(point.x for point in points) == tuple(point.x for point in nonnative_points)
            and all(point.y >= 0 for point in nonnative_points)
        )
        contexts.append(QcContactContext(
            replica_id=replica_id, sample_count=len(points),
            native_zero_fraction=zero_count / len(points), longest_native_zero_samples=longest,
            nonnative_at_native_zero_samples=(
                sum(left.y == 0 and right.y > 0 for left, right in zip(
                    points, nonnative_points, strict=True
                )) if paired else None
            ),
            total_contacts_min=(
                min(left.y + right.y for left, right in zip(
                    points, nonnative_points, strict=True
                )) if paired else None
            ),
        ))
    return tuple(contexts)


def _scientific_provenance(value: object) -> dict[str, Scalar]:
    if not isinstance(value, Mapping):
        return {"Scientific context": "unknown (legacy evidence not supplied)"}
    prepared = value.get("prepared_case")
    prepared = prepared if isinstance(prepared, Mapping) else {}
    return {
        "Scientific context": _scalar(value.get("status", "unknown")),
        "Prepared manifest digest": _scalar(prepared.get("manifest_digest")),
    }


def _expected_replicas(
    record: Mapping[str, object],
    hint: Mapping[str, object],
    global_replicas: tuple[str, ...],
    rows: tuple[Mapping[str, object], ...],
) -> tuple[str, ...]:
    for candidate in (
        record.get("expected_replica_ids"),
        record.get("expected_replicas"),
        hint.get("replicas"),
    ):
        identifiers = _replica_ids(candidate)
        if identifiers:
            return identifiers
    if global_replicas:
        return global_replicas
    return tuple(
        identifier
        for row in rows
        if (identifier := _safe_identifier(row.get("replica_id") or row.get("id")))
        is not None
    )


def _case_hints(value: object) -> tuple[dict[str, Mapping[str, object]], tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}, ()
    global_replicas = _replica_ids(value.get("replicas"))
    raw_cases = value.get("cases")
    hints: dict[str, Mapping[str, object]] = {}
    if _is_sequence(raw_cases):
        for item in raw_cases:
            if not isinstance(item, Mapping):
                continue
            case_id = _safe_identifier(item.get("case_id"))
            if case_id is not None:
                hints[case_id] = item
    else:
        for key, item in value.items():
            if key == "replicas" or not isinstance(item, Mapping):
                continue
            case_id = _safe_identifier(key)
            if case_id is not None:
                hints[case_id] = item
    return hints, global_replicas


def _analysis_models(
    analysis: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw = analysis.get("models")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("analysis.models must be a non-empty mapping")
    result: dict[str, Mapping[str, object]] = {}
    for raw_model_id, value in raw.items():
        if not isinstance(raw_model_id, str) or not raw_model_id:
            raise ValueError(
                "analysis.models keys must be non-empty strings"
            )
        if not isinstance(value, Mapping):
            raise TypeError(
                f"analysis.models.{raw_model_id} must be a mapping"
            )
        if str(value.get("model_id", raw_model_id)) != raw_model_id:
            raise ValueError(
                f"analysis.models.{raw_model_id} has mismatched model_id"
            )
        for field in ("score_selection", "correlation"):
            nested = value.get(field)
            if (
                isinstance(nested, Mapping)
                and str(nested.get("model_id", raw_model_id))
                != raw_model_id
            ):
                raise ValueError(
                    f"analysis.models.{raw_model_id}.{field} has "
                    "mismatched model_id"
                )
        result[raw_model_id] = value
    return result


def _model_order(
    analysis: Mapping[str, object],
    configuration: Mapping[str, object],
    model_ids: tuple[str, ...],
) -> tuple[str, ...]:
    configured = configuration.get(
        "model_order", analysis.get("model_order", ())
    )
    if not _is_sequence(configured):
        raise TypeError("report.model_order must be a sequence")
    order: list[str] = []
    for value in configured:
        if not isinstance(value, str) or not value:
            raise ValueError(
                "report.model_order entries must be non-empty strings"
            )
        if value not in model_ids:
            raise ValueError(
                f"report.model_order names unknown analysis model {value!r}"
            )
        if value in order:
            raise ValueError("report.model_order entries must be unique")
        order.append(value)
    return tuple(order + sorted(set(model_ids).difference(order)))


def _model_unit(model: Mapping[str, object]) -> str:
    selection = model.get("score_selection")
    if not isinstance(selection, Mapping):
        return ""
    transform = selection.get("transform")
    if not isinstance(transform, Mapping):
        return ""
    unit = transform.get("output_unit")
    return "" if unit is None else str(unit)


def _score_columns(value: object) -> tuple[ScoreColumn, ...]:
    if value is None:
        return ()
    if not _is_sequence(value):
        raise TypeError("report.score_columns must be a sequence")
    columns: list[ScoreColumn] = []
    for item in value:
        if isinstance(item, str):
            columns.append(ScoreColumn(item, item))
        elif isinstance(item, Mapping):
            columns.append(
                ScoreColumn(
                    key=str(item.get("key") or ""),
                    label=str(item.get("label") or item.get("key") or ""),
                    unit=str(item.get("unit") or ""),
                    precision=int(item.get("precision", 2)),
                )
            )
        else:
            raise TypeError("report.score_columns entries must be strings or mappings")
    if not columns:
        raise ValueError("report.score_columns cannot be empty")
    return tuple(columns)


def _model_score_columns(
    value: object,
    *,
    model_order: tuple[str, ...],
    models: Mapping[str, Mapping[str, object]],
) -> tuple[ScoreColumn, ...]:
    configured = _score_columns(value)
    if not configured:
        configured = (
            ScoreColumn("score_mean", "Mean score"),
            ScoreColumn("score_stddev", "Replica SD"),
            ScoreColumn("score_sem", "Replica SEM"),
        )
    legacy_keys = {"score_mean", "score_stddev", "score_sem"}
    if all(column.key in legacy_keys for column in configured):
        suffix = {
            "score_mean": "mean",
            "score_stddev": "stddev",
            "score_sem": "sem",
        }
        expanded: list[ScoreColumn] = []
        for model_id in model_order:
            model_label = model_id.upper()
            model_unit = _model_unit(models[model_id])
            for column in configured:
                expanded.append(
                    ScoreColumn(
                        key=f"{model_id}_{suffix[column.key]}",
                        label=f"{model_label} · {column.label}",
                        unit=column.unit or model_unit,
                        precision=column.precision,
                    )
                )
        return tuple(expanded)
    declared = {column.key for column in configured}
    allowed = {
        f"{model_id}_{suffix}"
        for model_id in model_order
        for suffix in ("mean", "stddev", "sem")
    }
    unknown = declared.difference(allowed)
    if unknown:
        raise ValueError(
            "report.score_columns contains non-model score keys: "
            + ", ".join(sorted(unknown))
        )
    missing = [
        model_id
        for model_id in model_order
        if f"{model_id}_mean" not in declared
    ]
    if missing:
        raise ValueError(
            "report.score_columns must expose a distinct mean column for "
            "every model; missing " + ", ".join(missing)
        )
    return configured


def _adapt_project_correlation(
    value: object,
    *,
    reference_label: str,
    score_key: str,
) -> CorrelationRecord | None:
    if not isinstance(value, Mapping):
        return CorrelationRecord(
            reference=reference_label,
            score_key=score_key,
            n=0,
            status="unavailable",
        )
    metrics = (
        _finite(value.get("pearson_r")),
        _finite(value.get("spearman_rho")),
        _finite(value.get("kendall_tau_b")),
        _finite(value.get("mae")),
        _finite(value.get("rmse")),
        _finite(value.get("mean_signed_error")),
    )
    n = _integer_or_none(value.get("n")) or 0
    notes = tuple(
        str(note)
        for key in ("warning", "sample_size_warning")
        if (note := value.get(key)) is not None and str(note).strip()
    )
    return CorrelationRecord(
        reference=reference_label,
        score_key=score_key,
        n=n,
        pearson_r=metrics[0],
        spearman_r=metrics[1],
        kendall_tau_b=metrics[2],
        mae=metrics[3],
        rmse=metrics[4],
        mean_signed_error=metrics[5],
        status="available" if n >= 2 else "unavailable",
        reference_mode=_text_or_none(value.get("reference_mode")),
        notes=notes,
        comparison_group=_text_or_none(value.get("comparison_group")),
        resolved_internal_dielectric=_finite(
            value.get("resolved_internal_dielectric")
        ),
    )


def _correlations(value: object) -> tuple[CorrelationRecord, ...]:
    if not _is_sequence(value):
        return ()
    records: list[CorrelationRecord] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        records.append(
            CorrelationRecord(
                reference=str(item.get("reference") or "reference"),
                score_key=str(item.get("score_key") or "score_mean"),
                n=_integer_or_none(item.get("n")) or 0,
                pearson_r=_finite(item.get("pearson_r")),
                spearman_r=_finite(item.get("spearman_r") or item.get("spearman_rho")),
                kendall_tau_b=_finite(item.get("kendall_tau_b")),
                slope=_finite(item.get("slope")),
                intercept=_finite(item.get("intercept")),
                mae=_finite(item.get("mae")),
                rmse=_finite(item.get("rmse")),
                mean_signed_error=_finite(item.get("mean_signed_error")),
                status=str(item.get("status") or "available"),
                reference_mode=_text_or_none(item.get("reference_mode")),
                notes=_correlation_notes(item.get("notes")),
            )
        )
    return tuple(records)


def _correlation_notes(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not _is_sequence(value):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _decomposition(value: object) -> tuple[DecompositionRecord, ...]:
    if not _is_sequence(value):
        return ()
    records: list[DecompositionRecord] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        values = item.get("values")
        if not isinstance(values, Mapping):
            continue
        numeric = {
            str(key): number
            for key, raw in values.items()
            if (number := _finite(raw)) is not None
        }
        if numeric:
            records.append(
                DecompositionRecord(
                    residue=str(item.get("residue") or "unavailable"),
                    values=numeric,
                )
            )
    return tuple(records)


def _energy_plots(
    replica_rows: tuple[Mapping[str, object], ...],
    model_order: tuple[str, ...],
    models: Mapping[str, Mapping[str, object]],
    *,
    scientific_context: object = None,
) -> tuple[tuple[PlotPanel, ...], tuple[PlotPanel, ...]]:
    energy_panels: list[PlotPanel] = []
    convergence_panels: list[PlotPanel] = []
    for model_id in model_order:
        energy_series: list[PlotSeries] = []
        convergence_series: list[PlotSeries] = []
        axis_labels: set[str] = set()
        scale, offset, unit = _score_transform(models[model_id])
        extracted_rows = []
        for row in replica_rows:
            replica_id = _safe_identifier(row.get("replica_id") or row.get("id"))
            if replica_id is None:
                continue
            scoring = _model_stage_payload(row, model_id)
            per_frame = scoring.get("per_frame")
            extracted = _mapping_xy_series(
                per_frame, value_kind="energy"
            )
            if extracted is None:
                continue
            axis_label, _, raw_points = extracted
            selection = _replica_frame_selection(
                scientific_context, replica_id, row, model_id=model_id
            )
            extracted_rows.append((replica_id, axis_label, raw_points, selection))
        use_time = bool(extracted_rows) and all(
            _can_map_frame_axis(points, axis, selection)
            for _, axis, points, selection in extracted_rows
        )
        for replica_id, axis_label, raw_points, selection in extracted_rows:
            axis_label, raw_points = _display_axis(
                raw_points, axis_label, selection, use_time=use_time
            )
            points = tuple(
                PlotPoint(
                    point.x,
                    point.y * scale + offset,
                    point.label,
                    point.emphasized,
                )
                for point in raw_points
            )
            axis_labels.add(axis_label)
            energy_series.append(
                PlotSeries(
                    f"{replica_id}-{model_id}-energy",
                    replica_id,
                    _downsample_points(points),
                )
            )
            running = 0.0
            cumulative: list[PlotPoint] = []
            for index, point in enumerate(points, 1):
                running += point.y
                cumulative.append(
                    PlotPoint(point.x, running / index, point.label)
                )
            convergence_series.append(
                PlotSeries(
                    f"{replica_id}-{model_id}-cumulative",
                    replica_id,
                    _downsample_points(tuple(cumulative)),
                )
            )
        if not energy_series:
            continue
        x_label = next(iter(axis_labels)) if len(axis_labels) == 1 else "Frame / time"
        title = _model_label(model_id)
        energy_panels.append(
            PlotPanel(
                f"energy-{model_id}",
                title,
                PlotSpec(
                    PlotKind.LINE,
                    x_label,
                    f"Per-frame binding score ({unit})",
                    tuple(energy_series),
                ),
            )
        )
        convergence_panels.append(
            PlotPanel(
                f"convergence-{model_id}",
                title,
                PlotSpec(
                    PlotKind.LINE,
                    x_label,
                    f"Cumulative mean ({unit})",
                    tuple(convergence_series),
                ),
            )
        )
    return tuple(energy_panels), tuple(convergence_panels)


def _replica_frame_selection(
    context: object, replica_id: str, row: Mapping[str, object], *, model_id: str | None = None,
) -> Mapping[str, object]:
    if not isinstance(context, Mapping):
        return {}
    planned = context.get("planned_frame_selection")
    if not isinstance(planned, Mapping):
        return {}
    if model_id is None:
        qc = _stage_payload(row, "trajectory_qc")
        receipt = qc.get("frame_manifest") or row.get("trajectory_frame_manifest")
    else:
        receipt = _model_stage_payload(row, model_id).get("frame_manifest")
    if not isinstance(receipt, Mapping):
        return {}
    # Revalidate the actual row receipt; a summary's status alone is not proof.
    return verified_frame_selection(planned, receipt)


def _can_map_frame_axis(
    points: tuple[PlotPoint, ...], axis_label: str, selection: Mapping[str, object],
) -> bool:
    return (
        selection.get("status") == "verified"
        and selection.get("expected_frames") == len(points)
        and _normalized_column(axis_label) in {"frame", "frameindex", "framenumber"}
        and all(math.isclose(point.x, index, abs_tol=1e-9)
                for index, point in enumerate(points, 1))
    )


def _display_axis(
    points: tuple[PlotPoint, ...], axis_label: str, selection: Mapping[str, object], *,
    use_time: bool,
) -> tuple[str, tuple[PlotPoint, ...]]:
    if use_time and _can_map_frame_axis(points, axis_label, selection):
        start = float(selection["analysis_start_ps"])
        interval = float(selection["analysis_interval_ps"])
        return "Time (ns)", tuple(
            replace(point, x=(start + (point.x - 1) * interval) / 1000)
            for point in points
        )
    # Never label an ordinal scoring/QC frame as full production time.
    if _normalized_column(axis_label) in {"frame", "frameindex", "framenumber"}:
        return "Analysis frame", points
    return "Analysis frame", tuple(
        replace(point, x=float(index)) for index, point in enumerate(points, 1)
    )


def _ligand_rmsd_diagnostics(
    payload: Mapping[str, object] | None,
) -> Mapping[str, Mapping[str, object]]:
    """Validate the report-assets equilibrated-pose RMSD wire contract."""

    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("Assets StageInput.payload must be a mapping")
    schema = payload.get("schema_version")
    if schema != "mmpbsa.report_assets.v3":
        raise ValueError(
            "Unsupported report-assets payload schema; version 0.7 requires "
            "mmpbsa.report_assets.v3"
        )
    raw_diagnostics = payload.get("diagnostics")
    if not isinstance(raw_diagnostics, Mapping):
        raise TypeError("Report-assets diagnostics must be a mapping")
    result: dict[str, Mapping[str, object]] = {}
    for raw_case_id, raw_case in raw_diagnostics.items():
        case_id = _wire_text(raw_case_id, "report-assets diagnostic case ID")
        if _safe_identifier(case_id) != case_id:
            raise ValueError(
                f"Invalid report-assets diagnostic case ID: {case_id!r}"
            )
        if not isinstance(raw_case, Mapping):
            raise TypeError(
                f"Report-assets diagnostics for {case_id!r} must be a mapping"
            )
        _strict_fields(
            raw_case,
            required={"ligand_rmsd"},
            label=f"report-assets diagnostics {case_id!r}",
        )
        raw_diagnostic = raw_case["ligand_rmsd"]
        if not isinstance(raw_diagnostic, Mapping):
            raise TypeError(
                f"Report-assets ligand_rmsd for {case_id!r} must be a mapping"
            )
        _strict_fields(
            raw_diagnostic,
            required={
                "reference_artifact",
                "reference_frame",
                "reference_time_ps",
                "reference_semantics",
                "reference_preprocessing",
                "alignment_selection",
                "rmsd_selection",
                "replicas",
            },
            label=f"report-assets ligand_rmsd {case_id!r}",
        )
        reference_pair = (
            raw_diagnostic.get("reference_artifact"),
            raw_diagnostic.get("reference_preprocessing"),
        )
        if reference_pair not in {
            ("selected_trajectory", "autoimage_ligand_anchor_receptor_fixed"),
            (
                "prepared_reference_trajectory",
                "whole_molecules_then_autoimage_ligand_anchor_receptor_fixed",
            ),
        }:
            raise ValueError(
                f"Report-assets ligand_rmsd {case_id!r} has invalid reference provenance"
            )
        expected_contract = {
            "reference_semantics": "post_npt_production_start",
            "alignment_selection": "receptor_interface_backbone",
            "rmsd_selection": "ligand_qc",
        }
        for key, expected in expected_contract.items():
            observed = _wire_text(
                raw_diagnostic[key],
                f"report-assets ligand_rmsd {case_id!r} {key}",
            )
            if observed != expected:
                raise ValueError(
                    f"Report-assets ligand_rmsd {case_id!r} {key} must be "
                    f"{expected!r}"
                )
        reference_frame = raw_diagnostic["reference_frame"]
        if (
            isinstance(reference_frame, bool)
            or not isinstance(reference_frame, int)
            or reference_frame != 1
        ):
            raise ValueError(
                f"Report-assets ligand_rmsd {case_id!r} reference_frame must be 1"
            )
        reference_time_ps = _finite(raw_diagnostic["reference_time_ps"])
        if reference_time_ps is None or not math.isclose(
            reference_time_ps,
            0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                f"Report-assets ligand_rmsd {case_id!r} reference_time_ps "
                "must be 0"
            )
        raw_replicas = raw_diagnostic["replicas"]
        if not isinstance(raw_replicas, Mapping):
            raise TypeError(
                f"Report-assets ligand_rmsd {case_id!r} replicas must be a mapping"
            )
        replicas: dict[str, Mapping[str, object]] = {}
        for raw_replica_id, raw_table in raw_replicas.items():
            replica_id = _wire_text(
                raw_replica_id,
                f"report-assets ligand_rmsd {case_id!r} replica ID",
            )
            if _safe_identifier(replica_id) != replica_id:
                raise ValueError(
                    f"Invalid report-assets diagnostic replica ID: {replica_id!r}"
                )
            if not isinstance(raw_table, Mapping):
                raise TypeError(
                    f"Report-assets ligand_rmsd {case_id!r}/{replica_id!r} "
                    "table must be a mapping"
                )
            _strict_fields(
                raw_table,
                required={"columns", "rows"},
                label=(
                    f"report-assets ligand_rmsd {case_id!r}/{replica_id!r} table"
                ),
            )
            replicas[replica_id] = raw_table
        result[case_id] = {
            **expected_contract,
            "reference_frame": 1,
            "reference_time_ps": 0.0,
            "replicas": replicas,
        }
    return result


def _protocol_temperature_k(
    configuration: Mapping[str, object],
    *,
    required: bool,
) -> float | None:
    raw_settings = configuration.get("_protocol_settings")
    if raw_settings is None and not required:
        return None
    if not isinstance(raw_settings, Mapping):
        raise ValueError(
            "report._protocol_settings must provide the resolved protocol settings"
        )
    temperature = _finite(raw_settings.get("temperature_k"))
    if temperature is None or temperature <= 0.0:
        raise ValueError(
            "report._protocol_settings.temperature_k must be finite and positive"
        )
    return temperature


def _occupancy_basin_plot(
    case_id: str,
    expected_replicas: tuple[str, ...],
    diagnostics: Mapping[str, object] | None,
    temperature_k: float | None,
) -> tuple[PlotSpec | None, str]:
    if diagnostics is None:
        return None, (
            "Equilibrated-pose ligand RMSD was not computed in report.assets; "
            "MD/QC RMSD is intentionally not substituted."
        )
    if temperature_k is None:
        raise ValueError("Occupancy-basin diagnostics require protocol temperature")
    raw_tables = diagnostics.get("replicas")
    if not isinstance(raw_tables, Mapping) or not raw_tables:
        return None, "No equilibrated-pose ligand RMSD replica tables were supplied."

    table_ids = {str(replica_id) for replica_id in raw_tables}
    if expected_replicas:
        unexpected = table_ids.difference(expected_replicas)
        if unexpected:
            raise ValueError(
                f"Ligand RMSD diagnostics for {case_id!r} contain unexpected "
                "replicas: " + ", ".join(sorted(unexpected))
            )
        replica_order = expected_replicas
    else:
        replica_order = tuple(sorted(table_ids))

    values_by_replica: dict[str, tuple[float, ...]] = {}
    missing: list[str] = []
    for replica_id in replica_order:
        table = raw_tables.get(replica_id)
        if table is None:
            missing.append(replica_id)
            continue
        values = _equilibrated_rmsd_values(
            table,
            label=f"{case_id}/{replica_id}",
        )
        if len(values) < 2:
            missing.append(f"{replica_id} (fewer than two frames)")
            continue
        values_by_replica[replica_id] = values
    if not values_by_replica:
        return None, "No replica supplied enough ligand RMSD samples for a basin."
    all_values = tuple(
        value
        for values in values_by_replica.values()
        for value in values
    )
    minimum = min(all_values)
    maximum = max(all_values)
    bin_count = max(
        6,
        min(
            40,
            int(4 * math.sqrt(max(map(len, values_by_replica.values())))),
        ),
    )
    series: list[PlotSeries] = []
    for replica_id, values in values_by_replica.items():
        series.append(
            PlotSeries(
                f"{replica_id}-occupancy-basin",
                replica_id,
                _occupancy_profile(
                    values,
                    temperature_k,
                    minimum=minimum,
                    maximum=maximum,
                    bin_count=bin_count,
                ),
            )
        )
    note = f"Profiles use the resolved protocol temperature ({temperature_k:g} K)."
    if missing:
        note += " Unavailable replicas: " + ", ".join(missing) + "."
    return (
        PlotSpec(
            PlotKind.LINE,
            "Ligand RMSD to equilibrated starting pose (Å)",
            "Relative occupancy free energy (kJ/mol)",
            tuple(series),
            x_min=0.0,
            y_min=0.0,
        ),
        note,
    )


def _equilibrated_rmsd_values(
    value: object,
    *,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Equilibrated-pose RMSD {label} table must be a mapping")
    columns = _table_columns(value)
    if columns is None or len(columns) < 2:
        raise ValueError(f"Equilibrated-pose RMSD {label} has invalid columns")
    axis_matches = tuple(
        column
        for column in columns
        if _normalized_column(column) in {"frame", "frameindex", "framenumber"}
    )
    value_matches = tuple(
        column
        for column in columns
        if _normalized_column(column) in {"ligandequilibrated", "rmsd"}
    )
    if len(axis_matches) != 1 or len(value_matches) != 1:
        raise ValueError(
            f"Equilibrated-pose RMSD {label} must identify one frame and one "
            "ligand_equilibrated column"
        )
    points = _table_points(value, axis_matches[0], value_matches[0])
    if points is None:
        raise ValueError(f"Equilibrated-pose RMSD {label} has invalid rows")
    frame_map = _frame_value_map(
        points,
        offset=-1,
        label=f"equilibrated-pose RMSD {label}",
    )
    expected_frames = set(range(len(frame_map)))
    if set(frame_map) != expected_frames:
        raise ValueError(
            f"Equilibrated-pose RMSD {label} frames must be contiguous from 1"
        )
    return tuple(frame_map[frame] for frame in range(len(frame_map)))


def _occupancy_profile(
    values: tuple[float, ...],
    temperature_k: float,
    *,
    minimum: float,
    maximum: float,
    bin_count: int,
) -> tuple[PlotPoint, ...]:
    if math.isclose(minimum, maximum, abs_tol=1.0e-12):
        return (PlotPoint(minimum, 0.0, "Most occupied", True),)

    width = (maximum - minimum) / bin_count
    counts = [0] * bin_count
    for value in values:
        index = min(int((value - minimum) / width), bin_count - 1)
        counts[index] += 1
    peak = max(counts)
    gas_constant = 0.00831446261815324
    peak_marked = False
    previous_bin: int | None = None
    points: list[PlotPoint] = []
    for index, count in enumerate(counts):
        if count == 0:
            continue
        emphasized = count == peak and not peak_marked
        peak_marked = peak_marked or emphasized
        points.append(
            PlotPoint(
                minimum + (index + 0.5) * width,
                -gas_constant * temperature_k * math.log(count / peak),
                f"{count} frames",
                emphasized,
                previous_bin is None or index != previous_bin + 1,
            )
        )
        previous_bin = index
    return tuple(points)


def _frame_value_map(
    points: tuple[PlotPoint, ...],
    *,
    offset: int,
    label: str,
) -> Mapping[int, float]:
    result: dict[int, float] = {}
    for point in points:
        frame = round(point.x)
        if not math.isclose(point.x, frame, abs_tol=1e-9):
            raise ValueError(f"{label} frame axes must be integers")
        normalized = frame + offset
        if normalized < 0 or normalized in result:
            raise ValueError(
                f"{label} has an invalid or duplicate normalized frame "
                f"{normalized}"
            )
        result[normalized] = point.y
    return result


def _score_transform(
    model: Mapping[str, object],
) -> tuple[float, float, str]:
    selection = model.get("score_selection")
    transform = selection.get("transform") if isinstance(selection, Mapping) else {}
    transform = transform if isinstance(transform, Mapping) else {}
    scale = _finite(transform.get("scale"))
    offset = _finite(transform.get("offset"))
    return (
        1.0 if scale is None else scale,
        0.0 if offset is None else offset,
        str(transform.get("output_unit") or "source units"),
    )


def _trajectory_plots(
    replica_rows: tuple[Mapping[str, object], ...],
    *,
    scientific_context: object = None,
) -> tuple[PlotPanel, ...]:
    metric_specs = (
        (
            "ligand-rmsd",
            "Ligand primary RMSD",
            "ligand_rmsd",
            lambda column: _normalized_column(column) in {"ligandprimary", "rmsd"},
            "RMSD (Å)",
        ),
        (
            "interface-rmsd",
            "Interface backbone RMSD",
            "interface_rmsd",
            lambda column: _normalized_column(column)
            in {"interfacebackbone", "rmsd"},
            "RMSD (Å)",
        ),
        (
            "native-contacts",
            "Native contacts",
            "native_contacts",
            lambda column: (
                _normalized_column(column).endswith("native")
                and not _normalized_column(column).endswith("nonnative")
            ),
            "Contact count",
        ),
        (
            "minimum-distance",
            "Minimum interface distance",
            "native_contacts",
            lambda column: _normalized_column(column).endswith("mindist"),
            "Distance (Å)",
        ),
    )
    panels: list[PlotPanel] = []
    for metric_id, title, table_name, matches, y_label in metric_specs:
        series: list[PlotSeries] = []
        axis_labels: set[str] = set()
        extracted_rows = []
        for row in replica_rows:
            replica_id = _safe_identifier(row.get("replica_id") or row.get("id"))
            if replica_id is None:
                continue
            qc = _stage_payload(row, "trajectory_qc")
            table = qc.get(table_name)
            if not isinstance(table, Mapping):
                continue
            columns = _table_columns(table)
            if columns is None or len(columns) < 2:
                continue
            axis_column = columns[0]
            if _normalized_column(axis_column) not in {
                "frame",
                "frameindex",
                "framenumber",
                "time",
                "timeps",
                "timestep",
            }:
                continue
            value_columns = tuple(column for column in columns[1:] if matches(column))
            if len(value_columns) != 1:
                continue
            extracted = _table_points(table, axis_column, value_columns[0])
            if extracted is None:
                continue
            selection = _replica_frame_selection(scientific_context, replica_id, row)
            extracted_rows.append((replica_id, axis_column, extracted, selection))
        use_time = bool(extracted_rows) and all(
            _can_map_frame_axis(points, axis, selection)
            for _, axis, points, selection in extracted_rows
        )
        for replica_id, axis_column, extracted, selection in extracted_rows:
            axis_column, extracted = _display_axis(
                extracted, axis_column, selection, use_time=use_time
            )
            axis_labels.add(axis_column)
            series.append(
                PlotSeries(
                    f"{replica_id}-{metric_id}",
                    replica_id,
                    _downsample_points(extracted),
                )
            )
        if series:
            x_label = next(iter(axis_labels)) if len(axis_labels) == 1 else "Frame / time"
            panels.append(
                PlotPanel(
                    f"trajectory-{metric_id}",
                    title,
                    PlotSpec(PlotKind.LINE, x_label, y_label, tuple(series)),
                )
            )
    return tuple(panels)


def _mapping_xy_series(
    value: object,
    *,
    value_kind: str,
) -> tuple[str, str, tuple[PlotPoint, ...]] | None:
    if not isinstance(value, Mapping):
        return None
    columns = _table_columns(value)
    if columns is None:
        return None
    normalized = tuple((_normalized_column(column), column) for column in columns)
    axis_names = {
        "frame",
        "frameindex",
        "framenumber",
        "time",
        "timeps",
        "timestep",
    }
    axis_matches = [
        column for name, column in normalized if name in axis_names
    ]
    if len(axis_matches) != 1:
        return None
    if value_kind != "energy":
        return None
    priorities = {"deltatotal": 0, "deltagbind": 1, "total": 2}
    candidates = [
        (priorities[name], column)
        for name, column in normalized
        if name in priorities
    ]
    if not candidates:
        return None
    best_priority = min(priority for priority, _ in candidates)
    best = [column for priority, column in candidates if priority == best_priority]
    if len(best) != 1:
        return None
    points = _table_points(value, axis_matches[0], best[0])
    if points is None:
        return None
    return axis_matches[0], best[0], points


def _table_columns(value: Mapping[str, object]) -> tuple[str, ...] | None:
    raw = value.get("columns")
    if not _is_sequence(raw) or not all(
        isinstance(column, str) and column.strip() for column in raw
    ):
        return None
    columns = tuple(raw)
    return columns if len(columns) == len(set(columns)) else None


def _table_points(
    table: Mapping[str, object],
    axis_column: str,
    value_column: str,
) -> tuple[PlotPoint, ...] | None:
    columns = _table_columns(table)
    rows = table.get("rows")
    if columns is None or not _is_sequence(rows):
        return None
    axis_index = columns.index(axis_column)
    value_index = columns.index(value_column)
    points: list[PlotPoint] = []
    for row in rows:
        if isinstance(row, Mapping):
            x, y = _finite(row.get(axis_column)), _finite(row.get(value_column))
        elif _is_sequence(row) and len(row) == len(columns):
            x, y = _finite(row[axis_index]), _finite(row[value_index])
        else:
            return None
        if x is None or y is None:
            return None
        points.append(PlotPoint(x, y))
    return tuple(points) if points else None


def _stage_payload(row: Mapping[str, object], stage: str) -> Mapping[str, object]:
    raw = row.get(stage)
    if not isinstance(raw, Mapping):
        return {}
    payload = raw.get("payload")
    return payload if isinstance(payload, Mapping) else raw


def _model_stage_payload(
    row: Mapping[str, object],
    model_id: str,
) -> Mapping[str, object]:
    scoring_models = row.get("scoring_models")
    if not isinstance(scoring_models, Mapping):
        return {}
    raw = scoring_models.get(model_id)
    if not isinstance(raw, Mapping):
        return {}
    payload = raw.get("payload")
    return payload if isinstance(payload, Mapping) else raw


def _normalized_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _downsample_points(
    points: tuple[PlotPoint, ...],
    *,
    maximum: int = 400,
) -> tuple[PlotPoint, ...]:
    """Preserve endpoints and local extrema for compact deterministic SVGs."""

    if len(points) <= maximum:
        return points
    if maximum < 4:
        raise ValueError("Plot downsampling maximum must be at least four")
    interior = points[1:-1]
    bucket_count = max(1, (maximum - 2) // 2)
    selected: list[tuple[int, PlotPoint]] = [(0, points[0])]
    for bucket in range(bucket_count):
        start = bucket * len(interior) // bucket_count
        end = (bucket + 1) * len(interior) // bucket_count
        segment = interior[start:end]
        if not segment:
            continue
        indexed = tuple(enumerate(segment, start=start + 1))
        low = min(indexed, key=lambda item: (item[1].y, item[0]))
        high = max(indexed, key=lambda item: (item[1].y, -item[0]))
        selected.extend({low[0]: low, high[0]: high}.values())
    selected.append((len(points) - 1, points[-1]))
    return tuple(point for _, point in sorted(selected, key=lambda item: item[0]))


def _model_label(model_id: str) -> str:
    if model_id.endswith("_dmm"):
        return f"{model_id.removesuffix('_dmm').upper()}-dMM"
    return model_id.upper()


def _optional_plot(value: object) -> PlotSpec | None:
    if not isinstance(value, Mapping):
        return None
    raw_series = value.get("series")
    if not _is_sequence(raw_series):
        return None
    series: list[PlotSeries] = []
    for index, item in enumerate(raw_series):
        if not isinstance(item, Mapping) or not _is_sequence(item.get("points")):
            continue
        points: list[PlotPoint] = []
        for point in item["points"]:
            if isinstance(point, Mapping):
                x, y = _finite(point.get("x")), _finite(point.get("y"))
                label = str(point.get("label") or "")
                emphasized = bool(point.get("emphasized", False))
                break_before = bool(point.get("break_before", False))
            elif _is_sequence(point) and len(point) >= 2:
                x, y = _finite(point[0]), _finite(point[1])
                label = str(point[2]) if len(point) > 2 else ""
                emphasized = False
                break_before = False
            else:
                continue
            if x is not None and y is not None:
                points.append(PlotPoint(x, y, label, emphasized, break_before))
        if points:
            series.append(
                PlotSeries(
                    _identifier_or_default(item.get("series_id"), f"series-{index + 1}"),
                    str(item.get("label") or f"Series {index + 1}"),
                    tuple(points),
                )
            )
    if not series:
        return None
    try:
        kind = PlotKind(str(value.get("kind") or "line"))
    except ValueError as error:
        raise ValueError(f"Unsupported report plot kind: {value.get('kind')!r}") from error
    return PlotSpec(
        kind=kind,
        x_label=str(value.get("x_label") or "x"),
        y_label=str(value.get("y_label") or "y"),
        series=tuple(series),
        zero_line=bool(value.get("zero_line", False)),
        x_min=_finite(value.get("x_min")),
        y_min=_finite(value.get("y_min")),
    )


def _adapt_assets(
    project: ProjectRecord,
    *,
    configuration: Mapping[str, object],
    asset_artifacts: Mapping[str, Path],
    asset_root: Path | None,
    asset_payload: Mapping[str, object] | None,
) -> tuple[ProjectRecord, tuple[AssetRequest, ...], Mapping[str, Path]]:
    configured_specs: list[Mapping[str, object]] = []
    for key in ("assets", "asset_links"):
        configured = configuration.get(key)
        if configured is None:
            continue
        # report.assets mappings belong to the generation provider.  Their
        # output descriptors arrive via the verified upstream assets payload.
        # Keep historic descriptor sequences as an alias for asset_links.
        if key == "assets" and isinstance(configured, Mapping):
            if not all(isinstance(item, str) for item in configured):
                raise TypeError("report.assets must be a mapping with string keys")
            continue
        if not _is_sequence(configured):
            raise TypeError(f"report.{key} must be a sequence")
        if not all(isinstance(item, Mapping) for item in configured):
            raise TypeError(f"report.{key} entries must be mappings")
        configured_specs.extend(configured)
    if not asset_artifacts and asset_payload is None and not configured_specs:
        return project, (), {}
    specs = _asset_descriptor_specs(asset_payload, project)
    specs.extend(configured_specs)

    requests: list[AssetRequest] = []
    sources: dict[str, Path] = {}
    project_links = list(project.artifacts)
    case_actions: dict[str, list[Link]] = {
        case.case_id: list(case.actions) for case in project.cases
    }
    case_artifacts: dict[str, list[Link]] = {
        case.case_id: list(case.artifacts) for case in project.cases
    }
    for index, spec in enumerate(specs, 1):
        role = str(spec.get("role") or "")
        if role not in asset_artifacts:
            raise ValueError(f"Configured report asset role is unavailable: {role!r}")
        source = Path(asset_artifacts[role])
        request_id = _identifier_or_default(spec.get("request_id"), f"asset-{index:04d}")
        output_raw = spec.get("output")
        relative_source = _relative_asset_source(source, role, asset_root)
        output = (
            PurePosixPath(str(output_raw))
            if output_raw is not None
            else PurePosixPath("assets") / relative_source
        )
        request = AssetRequest(request_id, "copy", (output,), {"source_role": role})
        requests.append(request)
        sources[request_id] = source
        visibility = str(spec.get("visibility") or "visible")
        if visibility not in {"visible", "hidden"}:
            raise ValueError("Report asset visibility must be visible or hidden")
        if visibility == "hidden":
            continue
        case_id = _text_or_none(spec.get("case_id"))
        if case_id is None:
            href = output.as_posix()
        else:
            if case_id not in case_actions:
                raise ValueError(f"Report asset targets unknown case: {case_id}")
            href = posixpath.relpath(
                output.as_posix(),
                start=(PurePosixPath("cases") / case_id).as_posix(),
            )
        link = Link(str(spec.get("label") or role), href, str(spec.get("description") or ""))
        slot = str(spec.get("slot") or ("actions" if case_id else "artifacts"))
        if case_id is None:
            if slot != "artifacts":
                raise ValueError("Project assets can only target the artifacts slot")
            project_links.append(link)
        elif slot == "actions":
            case_actions[case_id].append(link)
        elif slot == "artifacts":
            case_artifacts[case_id].append(link)
        else:
            raise ValueError("Case assets can target only actions or artifacts")

    updated_cases = tuple(
        replace(
            case,
            actions=tuple(case_actions[case.case_id]),
            artifacts=tuple(case_artifacts[case.case_id]),
        )
        for case in project.cases
    )
    return (
        replace(project, cases=updated_cases, artifacts=tuple(project_links)),
        tuple(requests),
        sources,
    )


def _asset_descriptor_specs(
    payload: Mapping[str, object] | None,
    project: ProjectRecord,
) -> list[Mapping[str, object]]:
    if payload is None:
        return []
    if not isinstance(payload, Mapping):
        raise TypeError("Assets StageInput.payload must be a mapping")
    schema = payload.get("schema_version")
    if schema != "mmpbsa.report_assets.v3":
        raise ValueError(
            "Unsupported report-assets payload schema; version 0.7 requires "
            "mmpbsa.report_assets.v3"
        )
    required = {"schema_version", "availability", "assets", "diagnostics"}
    allowed = required | {"backend_metadata"}
    missing = required.difference(payload)
    unknown = set(payload).difference(allowed)
    if missing:
        raise ValueError(
            "Assets payload is missing fields: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            "Assets payload has unknown fields: " + ", ".join(sorted(unknown))
        )
    availability = payload["availability"]
    if not isinstance(availability, Mapping):
        raise TypeError("Report-assets availability must be a mapping")
    raw_assets = payload["assets"]
    if not _is_sequence(raw_assets):
        raise TypeError("Report-assets descriptors must be a sequence")
    case_ids = {case.case_id for case in project.cases}
    specs: list[Mapping[str, object]] = []
    allowed_kinds = {
        "pymol_script",
        "topology",
        "trajectory",
        "manifest",
        "diagnostic",
        "provenance",
    }
    for index, raw in enumerate(raw_assets, 1):
        label = f"report-assets descriptor {index}"
        if not isinstance(raw, Mapping):
            raise TypeError(f"{label} must be a mapping")
        _strict_fields(
            raw,
            required={
                "artifact_role",
                "case_id",
                "replica_id",
                "kind",
                "label",
                "visibility",
            },
            label=label,
        )
        role = _wire_text(raw["artifact_role"], f"{label} artifact_role")
        case_raw = raw["case_id"]
        case_id = (
            None
            if case_raw is None
            else _wire_text(case_raw, f"{label} case_id")
        )
        if case_id is not None and case_id not in case_ids:
            raise ValueError(f"{label} targets unknown case {case_id!r}")
        kind = _wire_text(raw["kind"], f"{label} kind")
        if kind not in allowed_kinds:
            raise ValueError(f"{label} has unknown asset kind {kind!r}")
        replica_raw = raw["replica_id"]
        if replica_raw is not None:
            _wire_text(replica_raw, f"{label} replica_id")
        visibility = _wire_text(raw["visibility"], f"{label} visibility")
        if visibility not in {"visible", "hidden"}:
            raise ValueError(f"{label} has unknown visibility {visibility!r}")
        description = (
            f"Replica {replica_raw}" if replica_raw is not None else ""
        )
        specs.append(
            {
                "role": role,
                "case_id": case_id,
                "slot": (
                    "actions"
                    if kind == "pymol_script" and case_id is not None
                    else "artifacts"
                ),
                "label": _wire_text(raw["label"], f"{label} label"),
                "description": description,
                "visibility": visibility,
            }
        )
    return specs


def _relative_asset_source(
    source: Path,
    role: str,
    asset_root: Path | None,
) -> PurePosixPath:
    if asset_root is not None:
        try:
            relative = source.resolve().relative_to(asset_root.resolve())
        except ValueError:
            pass
        else:
            return PurePosixPath(relative.as_posix())
    return PurePosixPath(role) / source.name


def _profile_contributions(
    profile: Mapping[str, object],
    project: ProjectRecord,
) -> tuple[Contribution, ...]:
    extensions_raw = profile.get("extensions")
    if not isinstance(extensions_raw, Mapping) or "report" not in extensions_raw:
        return ()
    report_raw = extensions_raw["report"]
    if not isinstance(report_raw, Mapping):
        raise TypeError("_profile.extensions.report must be a mapping")
    _strict_fields(
        report_raw,
        required={"schema_version", "contributions"},
        label="_profile.extensions.report",
    )
    if report_raw["schema_version"] != "mmpbsa.report.contributions.v1":
        raise ValueError("Unsupported profile report-contribution schema")
    raw_contributions = report_raw["contributions"]
    if not _is_sequence(raw_contributions):
        raise TypeError("Profile report contributions must be a sequence")
    case_ids = {case.case_id for case in project.cases}
    result: list[Contribution] = []
    for index, raw in enumerate(raw_contributions, 1):
        label = f"profile report contribution {index}"
        if not isinstance(raw, Mapping):
            raise TypeError(f"{label} must be a mapping")
        _strict_fields(raw, required={"page", "slot", "block"}, label=label)
        page_value = _wire_text(raw["page"], f"{label} page")
        slot_value = _wire_text(raw["slot"], f"{label} slot")
        targets = _wire_targets(page_value, case_ids, label)
        slot = _wire_slot(slot_value, page_value, label)
        block = _wire_block(raw["block"], block_id=f"profile-{index:04d}", label=label)
        result.extend(
            Contribution(
                page=target,
                slot=slot,
                block=block,
                contributor_id="profile.report",
                priority=100,
            )
            for target in targets
        )
    return tuple(result)


def _wire_targets(
    page: str,
    case_ids: set[str],
    label: str,
) -> tuple[PageTarget, ...]:
    if page == "project":
        return (PageTarget.project(),)
    if page == "case:*":
        return tuple(PageTarget.case(case_id) for case_id in sorted(case_ids))
    if page.startswith("case:"):
        case_id = page.removeprefix("case:")
        if case_id not in case_ids:
            raise ValueError(f"{label} targets unknown case {case_id!r}")
        return (PageTarget.case(case_id),)
    raise ValueError(f"{label} page must be 'project', 'case:*', or 'case:<id>'")


def _wire_slot(slot: str, page: str, label: str) -> NamedSlot:
    allowed_project = {
        NamedSlot.PROJECT_AFTER_SUMMARY,
        NamedSlot.PROJECT_AFTER_SCORES,
        NamedSlot.PROJECT_BEFORE_PROVENANCE,
    }
    allowed_case = {
        NamedSlot.CASE_AFTER_SUMMARY,
        NamedSlot.CASE_AFTER_ENERGY,
        NamedSlot.CASE_AFTER_QC,
        NamedSlot.CASE_BEFORE_PROVENANCE,
    }
    try:
        parsed = NamedSlot(slot)
    except ValueError as error:
        raise ValueError(f"{label} uses unknown named slot {slot!r}") from error
    allowed = allowed_project if page == "project" else allowed_case
    if parsed not in allowed:
        raise ValueError(f"{label} slot {slot!r} is incompatible with page {page!r}")
    return parsed


def _wire_block(value: object, *, block_id: str, label: str):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} block must be a mapping")
    kind = _wire_text(value.get("kind"), f"{label} block kind")
    if kind == "notice":
        _strict_fields(
            value,
            required={"kind", "title", "text", "severity"},
            label=f"{label} notice block",
        )
        severity = {
            "info": NoticeLevel.INFO,
            "warn": NoticeLevel.WARNING,
            "block": NoticeLevel.BLOCKED,
        }.get(_wire_text(value["severity"], f"{label} notice severity"))
        if severity is None:
            raise ValueError(f"{label} notice severity must be info, warn, or block")
        return NoticeBlock(
            block_id,
            _wire_text(value["title"], f"{label} notice title"),
            _wire_text(value["text"], f"{label} notice text"),
            severity,
        )
    if kind == "key_value":
        _strict_fields(
            value,
            required={"kind", "title", "items"},
            label=f"{label} key_value block",
        )
        items = value["items"]
        if not isinstance(items, Mapping):
            raise TypeError(f"{label} key_value items must be a scalar mapping")
        if not all(isinstance(key, str) and key.strip() for key in items):
            raise TypeError(f"{label} key_value keys must be non-empty strings")
        rows = tuple(
            (str(key), _wire_scalar(item, f"{label} item {key!r}"))
            for key, item in sorted(items.items(), key=lambda pair: str(pair[0]))
        )
        return TableBlock(
            block_id,
            _wire_text(value["title"], f"{label} key_value title"),
            (TableColumn("key", "Field"), TableColumn("value", "Value")),
            rows,
        )
    if kind == "table":
        _strict_fields(
            value,
            required={"kind", "title", "columns", "rows"},
            label=f"{label} table block",
        )
        columns_raw, rows_raw = value["columns"], value["rows"]
        if not _is_sequence(columns_raw) or not all(
            isinstance(column, str) and column.strip() for column in columns_raw
        ):
            raise TypeError(f"{label} table columns must be non-empty strings")
        if not _is_sequence(rows_raw):
            raise TypeError(f"{label} table rows must be a sequence")
        rows: list[tuple[Scalar, ...]] = []
        for row_index, row in enumerate(rows_raw, 1):
            if not _is_sequence(row):
                raise TypeError(f"{label} table row {row_index} must be a sequence")
            rows.append(
                tuple(
                    _wire_scalar(cell, f"{label} row {row_index}")
                    for cell in row
                )
            )
        return TableBlock(
            block_id,
            _wire_text(value["title"], f"{label} table title"),
            tuple(
                TableColumn(f"column-{index}", column)
                for index, column in enumerate(columns_raw, 1)
            ),
            tuple(rows),
        )
    raise ValueError(
        f"{label} block kind must be notice, key_value, or table; "
        "HTML/script blocks are forbidden"
    )


def _wire_scalar(value: object, label: str) -> Scalar:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"{label} must contain scalar finite values")


def _wire_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _strict_fields(
    value: Mapping[str, object],
    *,
    required: set[str],
    label: str,
) -> None:
    missing = required.difference(value)
    unknown = set(value).difference(required)
    if missing:
        raise ValueError(
            f"{label} is missing fields: "
            + ", ".join(sorted(str(item) for item in missing))
        )
    if unknown:
        raise ValueError(
            f"{label} has unknown fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )


def _links(value: object) -> tuple[Link, ...]:
    if not _is_sequence(value):
        return ()
    links: list[Link] = []
    for item in value:
        if isinstance(item, Mapping):
            links.append(
                Link(
                    str(item.get("label") or item.get("href") or "artifact"),
                    str(item.get("href") or ""),
                    str(item.get("description") or ""),
                )
            )
    return tuple(links)


def _replica_ids(value: object) -> tuple[str, ...]:
    if not _is_sequence(value):
        return ()
    result: list[str] = []
    for item in value:
        raw = item.get("replica_id") if isinstance(item, Mapping) else item
        if (identifier := _safe_identifier(raw)) is not None:
            result.append(identifier)
    return tuple(dict.fromkeys(result))


def _walk_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        records.append(value)
        for item in value.values():
            records.extend(_walk_mappings(item))
    elif _is_sequence(value):
        for item in value:
            records.extend(_walk_mappings(item))
    return tuple(records)


def _find_integer(
    values: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> int | None:
    for value in values:
        for row in _walk_mappings(value):
            for key in keys:
                if (parsed := _integer_or_none(row.get(key))) is not None:
                    return parsed
    return None


def _scalar_mapping(value: object) -> dict[str, Scalar]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Scalar] = {}
    for key, item in value.items():
        scalar = _scalar(item)
        if scalar is not None or item is None:
            result[str(key)] = scalar
    return result


def _scalar(value: object) -> Scalar:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer_or_none(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_identifier(value: object) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    try:
        CaseRecord(
            case_id=text,
            label=text,
            expected_replica_ids=(),
        )
    except ValueError:
        return None
    return text


def _identifier_or_default(value: object, default: str) -> str:
    if value is None or not str(value).strip():
        return default
    identifier = _safe_identifier(value)
    if identifier is None:
        raise ValueError(f"Unsafe identifier: {value!r}")
    return identifier


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))

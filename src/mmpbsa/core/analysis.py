"""Small, dependency-free project aggregation owned by the workflow core."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from mmpbsa.utils import atomic_write_json

from .config import WorkflowDefinition
from .errors import ContractError


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _walk(value: object) -> Sequence[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for item in value.values():
            rows.extend(_walk(item))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            rows.extend(_walk(item))
    return rows


def _normalized_label(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def extract_score(
    payload: Mapping[str, object],
    *,
    selector: Mapping[str, object] | None = None,
    model_id: str | None = None,
) -> float | None:
    """Extract one score, rejecting ambiguous model selection."""

    if selector is not None:
        allowed = {"component", "label", "value_key"}
        required = {"component", "label"}
        missing = required.difference(selector)
        unknown = set(selector).difference(allowed)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise ContractError(
                "invalid analysis.score_selector: " + "; ".join(details)
            )
        value_key = str(selector.get("value_key", "average"))
        if value_key not in {"average", "mean", "value", "score"}:
            raise ContractError(
                "analysis.score_selector.value_key is unsupported"
            )
        matches: list[float] = []
        for row in _walk(payload):
            if (
                model_id is not None
                and _normalized_label(row.get("model_id")) != model_id.casefold()
            ):
                continue
            if any(
                _normalized_label(row.get(field))
                != _normalized_label(selector[field])
                for field in ("component", "label")
            ):
                continue
            value = _finite(row.get(value_key))
            if value is not None:
                matches.append(value)
        if len(matches) != 1:
            raise ContractError(
                "analysis.score_selector must match exactly one finite score "
                f"row; observed {len(matches)}"
            )
        return matches[0]

    for key in ("score", "binding_score", "delta_total", "DELTA_TOTAL"):
        if key in payload and (value := _finite(payload[key])) is not None:
            return value
    candidates: list[tuple[int, float, str, str, str]] = []
    for row in _walk(payload):
        observed_model = str(row.get("model_id", ""))
        if (
            model_id is not None
            and observed_model.casefold() != model_id.casefold()
        ):
            continue
        label = str(row.get("label", "")).strip().upper().replace(" ", "_")
        value = None
        for key in ("average", "mean", "value", "score"):
            if key in row and (value := _finite(row[key])) is not None:
                break
        if value is None:
            continue
        priority = {
            "DELTA_TOTAL": 0,
            "DELTA_G_BIND": 1,
            "TOTAL": 2,
        }.get(label)
        if priority is not None:
            candidates.append(
                (
                    priority,
                    value,
                    observed_model,
                    str(row.get("component", "")),
                    label,
                )
            )
    if not candidates:
        return None
    best_priority = min(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_priority]
    if len(best) != 1:
        contexts = ", ".join(
            f"{observed_model}/{component}/{label}"
            for _, _, observed_model, component, label in best
        )
        raise ContractError(
            "scoring payload is ambiguous; configure analysis.score_selector "
            f"explicitly ({contexts})"
        )
    return best[0][1]


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for index in range(start, stop):
            result[ordered[index][0]] = rank
        start = stop
    return result


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    numerator = sum(
        (left - mean_x) * (right - mean_y)
        for left, right in zip(x, y, strict=True)
    )
    denom_x = sum((value - mean_x) ** 2 for value in x)
    denom_y = sum((value - mean_y) ** 2 for value in y)
    denominator = math.sqrt(denom_x * denom_y)
    return None if denominator == 0 else numerator / denominator


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(_ranks(x), _ranks(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for left in range(len(x) - 1):
        for right in range(left + 1, len(x)):
            dx = x[left] - x[right]
            dy = y[left] - y[right]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 12)


def _analysis_models(
    workflow: WorkflowDefinition,
) -> dict[str, Mapping[str, object]]:
    raw = workflow.analysis.get("models")
    if not isinstance(raw, Mapping) or not raw:
        raise ContractError("analysis.models must be a non-empty mapping")
    models: dict[str, Mapping[str, object]] = {}
    for raw_model_id, configuration in raw.items():
        model_id = str(raw_model_id)
        if not model_id or model_id != raw_model_id:
            raise ContractError("analysis.models keys must be non-empty strings")
        if not isinstance(configuration, Mapping):
            raise ContractError(
                f"analysis.models.{model_id} must be a mapping"
            )
        models[model_id] = configuration
    return models


def _score_selection(
    model_id: str,
    configuration: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, float, float, str | None]:
    selector_raw = configuration.get("score_selector")
    if selector_raw is not None and not isinstance(selector_raw, Mapping):
        raise ContractError(
            f"analysis.models.{model_id}.score_selector must be a mapping"
        )
    selector = selector_raw if isinstance(selector_raw, Mapping) else None
    transform_raw = configuration.get("score_transform", {})
    if not isinstance(transform_raw, Mapping):
        raise ContractError(
            f"analysis.models.{model_id}.score_transform must be a mapping"
        )
    unknown_transform = set(transform_raw).difference(
        {"scale", "offset", "input_unit", "output_unit"}
    )
    if unknown_transform:
        raise ContractError(
            f"analysis.models.{model_id}.score_transform contains unknown "
            "fields: "
            + ", ".join(sorted(str(item) for item in unknown_transform))
        )
    scale = _finite(transform_raw.get("scale", 1.0))
    offset = _finite(transform_raw.get("offset", 0.0))
    if scale is None or offset is None:
        raise ContractError(
            f"analysis.models.{model_id}.score_transform scale/offset "
            "must be finite"
        )
    score_unit = (
        None
        if transform_raw.get("output_unit") is None
        else str(transform_raw["output_unit"])
    )
    return selector, scale, offset, score_unit


def _claims_model_complete(
    replica: Mapping[str, object],
    model_id: str,
) -> bool:
    """Recognize explicit per-model completion declarations only.

    A replica-level ``status=completed`` cannot imply that every configured
    scoring model completed when the workflow completion policy is ``any``.
    """

    for key in ("completed_models", "complete_models", "successful_models"):
        value = replica.get(key)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if model_id in value:
                return True
    for key in (
        "model_statuses",
        "scoring_model_statuses",
        "scoring_statuses",
    ):
        value = replica.get(key)
        if not isinstance(value, Mapping) or model_id not in value:
            continue
        status = value[model_id]
        if isinstance(status, Mapping):
            status = (
                status.get("status")
                or status.get("state")
                or status.get("outcome")
            )
        return _normalized_label(status) in {
            "complete",
            "completed",
            "success",
            "succeeded",
        }
    return False


def _validate_attempt_model(
    attempt: Mapping[str, object],
    model_id: str,
    *,
    case_id: str,
    replica_id: str,
) -> None:
    declared = attempt.get("model_id")
    if declared is not None and str(declared) != model_id:
        raise ContractError(
            f"scoring_models.{model_id} declares mismatched model_id "
            f"{declared!r}"
        )
    task = attempt.get("task")
    if isinstance(task, Mapping):
        for key, expected in (
            ("stage", "scoring"),
            ("case_id", case_id),
            ("replica_id", replica_id),
        ):
            if task.get(key) is not None and str(task[key]) != expected:
                raise ContractError(
                    f"scoring_models.{model_id}.task declares mismatched {key} "
                    f"{task[key]!r}"
                )
    if (
        isinstance(task, Mapping)
        and task.get("model_id") is not None
        and str(task["model_id"]) != model_id
    ):
        raise ContractError(
            f"scoring_models.{model_id}.task declares mismatched model_id "
            f"{task['model_id']!r}"
        )


def _resolved_dielectric(attempt: Mapping[str, object]) -> float | None:
    payload = attempt.get("payload")
    audit = payload.get("dielectric_resolution") if isinstance(payload, Mapping) else None
    if not isinstance(audit, Mapping):
        provenance = attempt.get("invocation_provenance")
        metadata = provenance.get("metadata") if isinstance(provenance, Mapping) else None
        audit = metadata.get("dielectric_resolution") if isinstance(metadata, Mapping) else None
    value = (
        _finite(audit.get("resolved_internal_dielectric"))
        if isinstance(audit, Mapping)
        else None
    )
    return value if value is not None and value > 0 else None


def _qc_verdict(replica: Mapping[str, object]) -> str | None:
    qc = replica.get("trajectory_qc")
    payload = qc.get("payload") if isinstance(qc, Mapping) else None
    for candidate in (
        replica.get("qc_verdict"),
        qc.get("qc_verdict") if isinstance(qc, Mapping) else None,
        qc.get("verdict") if isinstance(qc, Mapping) else None,
        payload.get("verdict") if isinstance(payload, Mapping) else None,
    ):
        if candidate is not None:
            return str(candidate).upper()
    return None


def _comparison_status(
    record: Mapping[str, object] | None,
    replica_rows: Sequence[Mapping[str, object]],
    expected_ids: tuple[str, ...],
    model_id: str,
    source_model_id: str,
) -> tuple[bool, list[str], float | None, str | None]:
    reasons: list[str] = []
    if record is None:
        reasons.append("missing_finalized_case")
    else:
        outcome = str(record.get("outcome", "")).lower()
        if outcome not in {"complete", "complete_with_warnings"} or record.get("partial"):
            reasons.append("case_incomplete")
        if str(record.get("qc_verdict", "")).upper() == "BLOCK":
            reasons.append("qc_blocked")
        declared = record.get("expected_replicas")
        if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes, bytearray)):
            if tuple(str(item) for item in declared) != expected_ids:
                reasons.append("replica_declaration_mismatch")
    rows_by_id = {str(row.get("replica_id")): row for row in replica_rows}
    if (
        len(rows_by_id) != len(replica_rows)
        or set(rows_by_id) != set(expected_ids)
        or not expected_ids
    ):
        reasons.append("incomplete_replicas")
    dielectric_values: list[float] = []
    for replica_id in expected_ids:
        row = rows_by_id.get(replica_id)
        if row is None:
            continue
        if _qc_verdict(row) not in {"PASS", "WARN"}:
            reasons.append("replica_qc_not_accepted")
        completed = row.get("completed_models")
        if not isinstance(completed, Sequence) or isinstance(
            completed, (str, bytes, bytearray)
        ):
            reasons.append("model_completion_unverified")
        elif source_model_id not in completed:
            reasons.append("model_incomplete")
        scoring = row.get("scoring_models")
        attempt = scoring.get(model_id) if isinstance(scoring, Mapping) else None
        if not isinstance(attempt, Mapping) or _finite(attempt.get("selected_score")) is None:
            reasons.append("model_score_missing")
            continue
        dielectric = _resolved_dielectric(attempt)
        if dielectric is None:
            reasons.append("dielectric_unresolved")
        else:
            dielectric_values.append(dielectric)
    unique_dielectrics = set(dielectric_values)
    if len(unique_dielectrics) > 1:
        reasons.append("mixed_dielectric")
    dielectric = next(iter(unique_dielectrics)) if len(unique_dielectrics) == 1 else None
    group = f"epsilon={dielectric:g}" if dielectric is not None else None
    unique_reasons = list(dict.fromkeys(reasons))
    return not unique_reasons, unique_reasons, dielectric, group


def _correlation(
    scores: Sequence[float],
    references: Sequence[float],
    *,
    reference_mode: str = "quantitative_comparison",
) -> dict[str, object]:
    result: dict[str, object] = {
        "n": len(scores),
        "pearson_r": _round_or_none(pearson(scores, references)),
        "spearman_rho": _round_or_none(spearman(scores, references)),
        "kendall_tau_b": _round_or_none(kendall_tau_b(scores, references)),
    }
    if scores and reference_mode == "quantitative_comparison":
        errors = [
            score - reference
            for score, reference in zip(scores, references, strict=True)
        ]
        result.update(
            {
                "mae": statistics.fmean(abs(item) for item in errors),
                "rmse": math.sqrt(
                    statistics.fmean(item * item for item in errors)
                ),
                "mean_signed_error": statistics.fmean(errors),
            }
        )
    else:
        result.update(
            {"mae": None, "rmse": None, "mean_signed_error": None}
        )
    result["reference_mode"] = reference_mode
    if reference_mode == "descriptive_association":
        result["warning"] = (
            "Reference and score have different semantics; error metrics are "
            "not reported. Associations are descriptive only."
        )
    if len(scores) < 5:
        result["sample_size_warning"] = (
            f"Only {len(scores)} paired cases; association estimates are unstable."
        )
    return result


def build_project_analysis(
    workflow: WorkflowDefinition,
    case_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate finalized records independently for every scoring model."""

    by_case = {str(item.get("case_id")): item for item in case_records}
    models = _analysis_models(workflow)
    # Analysis ordering is part of the analysis contract and follows the
    # declared analysis model mapping. Presentation-only report.model_order is
    # applied later by the report adapter and must not change/reuse analysis.
    model_order = tuple(models)
    source_models = {
        model_id: str(models[model_id].get("source_model_id", model_id))
        for model_id in model_order
    }
    selections = {
        model_id: _score_selection(model_id, models[model_id])
        for model_id in model_order
    }
    case_rows: list[dict[str, object]] = []
    declared_replicas = tuple(
        str(item.replica_id) for item in getattr(workflow, "replicas", ())
    )
    reference_key = str(workflow.analysis.get("reference_key", "value"))
    for case in workflow.cases:
        record = by_case.get(case.case_id)
        replica_rows = []
        if isinstance(record, Mapping):
            candidate = record.get("replicas", ())
            if isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes, bytearray)
            ):
                replica_rows = [
                    row for row in candidate if isinstance(row, Mapping)
                ]
        scores: dict[str, list[float]] = {
            model_id: [] for model_id in model_order
        }
        selected_replica_rows = [dict(row) for row in replica_rows]
        for selected_row, replica_row in zip(
            selected_replica_rows, replica_rows, strict=True
        ):
            scoring_models_raw = replica_row.get("scoring_models")
            scoring_models = (
                scoring_models_raw
                if isinstance(scoring_models_raw, Mapping)
                else {}
            )
            selected_scoring_models = {
                str(key): dict(value)
                for key, value in scoring_models.items()
                if isinstance(key, str) and isinstance(value, Mapping)
            }
            for model_id in model_order:
                source_model_id = source_models[model_id]
                attempt_raw = scoring_models.get(source_model_id)
                if not isinstance(attempt_raw, Mapping):
                    if _claims_model_complete(replica_row, source_model_id):
                        replica_id = replica_row.get(
                            "replica_id", "unavailable"
                        )
                        raise ContractError(
                            f"replica {replica_id!r} claims model "
                            f"{source_model_id!r} complete but scoring_models has no "
                            "attempt record"
                        )
                    continue
                _validate_attempt_model(
                    attempt_raw,
                    source_model_id,
                    case_id=case.case_id,
                    replica_id=str(replica_row.get("replica_id")),
                )
                selector, scale, offset, _ = selections[model_id]
                payload = attempt_raw.get("payload", {})
                if (
                    isinstance(payload, Mapping)
                    and payload.get("model_id") is not None
                    and str(payload["model_id"]) != source_model_id
                ):
                    raise ContractError(
                        f"scoring_models.{model_id}.payload declares "
                        f"mismatched source model_id {payload['model_id']!r}"
                    )
                raw_score = (
                    extract_score(
                        payload,
                        selector=selector,
                        model_id=source_model_id,
                    )
                    if isinstance(payload, Mapping)
                    else None
                )
                selected_score = (
                    None if raw_score is None else raw_score * scale + offset
                )
                selected_attempt = dict(attempt_raw)
                selected_attempt["model_id"] = model_id
                selected_attempt["source_model_id"] = source_model_id
                selected_payload = dict(payload) if isinstance(payload, Mapping) else {}
                per_frame_models = selected_payload.get("per_frame_models")
                if isinstance(per_frame_models, Mapping):
                    selected_per_frame = per_frame_models.get(model_id)
                    if isinstance(selected_per_frame, Mapping):
                        selected_payload["per_frame"] = dict(selected_per_frame)
                selected_attempt["payload"] = selected_payload
                selected_attempt["selected_score_raw"] = raw_score
                selected_attempt["selected_score"] = selected_score
                transform = models[model_id].get("score_transform", {})
                if isinstance(transform, Mapping):
                    selected_attempt["selected_score_raw_unit"] = transform.get("input_unit")
                    selected_attempt["selected_score_unit"] = transform.get("output_unit")
                selected_scoring_models[model_id] = selected_attempt
                if selected_score is not None:
                    scores[model_id].append(selected_score)
            selected_row["scoring_models"] = selected_scoring_models
        selected_record = (
            None
            if record is None
            else {**dict(record), "replicas": selected_replica_rows}
        )
        reference = _finite(case.reference.get(reference_key))
        case_models: dict[str, dict[str, object]] = {}
        expected_ids = declared_replicas or tuple(
            str(item) for item in (record or {}).get("expected_replicas", ())
        ) or tuple(str(row.get("replica_id")) for row in replica_rows)
        for model_id in model_order:
            model_scores = scores[model_id]
            mean = statistics.fmean(model_scores) if model_scores else None
            stddev = (
                statistics.stdev(model_scores)
                if len(model_scores) > 1
                else None
            )
            sem = (
                stddev / math.sqrt(len(model_scores))
                if stddev is not None
                else None
            )
            eligible, exclusion_reasons, dielectric, group = _comparison_status(
                record,
                selected_replica_rows,
                expected_ids,
                model_id,
                source_models[model_id],
            )
            case_models[model_id] = {
                "model_id": model_id,
                "replica_scores": model_scores,
                "replica_count": len(model_scores),
                "score_mean": mean,
                "score_stddev": stddev,
                "score_sem": sem,
                "score_unit": selections[model_id][3],
                "comparison_eligible": eligible,
                "comparison_exclusion_reasons": exclusion_reasons,
                "resolved_internal_dielectric": dielectric,
                "comparison_group": group,
                "rank": None,
            }
        verdict = None if record is None else record.get("qc_verdict")
        outcome = "missing" if record is None else record.get("outcome")
        row = {
            "case_id": case.case_id,
            "label": case.label or case.case_id,
            "reference": reference,
            "models": case_models,
            "qc_verdict": verdict,
            "outcome": outcome,
            "finalized": record is not None,
            "record": selected_record,
        }
        case_rows.append(row)

    model_results: dict[str, dict[str, object]] = {}
    reference_mode = str(
        workflow.analysis.get("reference_mode", "quantitative_comparison")
    )
    for model_id in model_order:
        selector, scale, offset, score_unit = selections[model_id]
        transform_raw = models[model_id].get("score_transform", {})
        transform = (
            transform_raw if isinstance(transform_raw, Mapping) else {}
        )
        groups: dict[float, list[dict[str, object]]] = {}
        for case_row in case_rows:
            case_model = case_row["models"][model_id]
            if case_model["comparison_eligible"]:
                epsilon = float(case_model["resolved_internal_dielectric"])
                groups.setdefault(epsilon, []).append(case_row)
        correlation_groups: list[dict[str, object]] = []
        for epsilon, members in sorted(groups.items()):
            ordered = sorted(
                members,
                key=lambda row: (
                    float(row["models"][model_id]["score_mean"]),
                    str(row["case_id"]),
                ),
            )
            previous_score: float | None = None
            current_rank = 0
            for index, member in enumerate(ordered, 1):
                score = float(member["models"][model_id]["score_mean"])
                if score != previous_score:
                    current_rank = index
                member["models"][model_id]["rank"] = current_rank
                previous_score = score
            paired = [row for row in members if row["reference"] is not None]
            correlation_groups.append(
                {
                    "model_id": model_id,
                    "comparison_group": f"epsilon={epsilon:g}",
                    "resolved_internal_dielectric": epsilon,
                    "eligible_case_count": len(members),
                    **_correlation(
                        [float(row["models"][model_id]["score_mean"]) for row in paired],
                        [float(row["reference"]) for row in paired],
                        reference_mode=reference_mode,
                    ),
                }
            )
        legacy_correlation = (
            dict(correlation_groups[0])
            if len(correlation_groups) == 1
            else {
                "model_id": model_id,
                **_correlation([], [], reference_mode=reference_mode),
                "warning": (
                    "Comparison requires a single resolved internal-dielectric group."
                ),
            }
        )
        model_results[model_id] = {
            "model_id": model_id,
            "correlation": legacy_correlation,
            "correlation_groups": correlation_groups,
            "score_selection": {
                "model_id": model_id,
                "selector": None if selector is None else dict(selector),
                "source_model_id": source_models[model_id],
                "evaluator": models[model_id].get("evaluator", "direct"),
                "transform": {
                    "scale": scale,
                    "offset": offset,
                    "input_unit": transform.get("input_unit"),
                    "output_unit": score_unit,
                },
            },
        }
    case_outcomes = {str(row["outcome"]) for row in case_rows}
    if "failed" in case_outcomes:
        project_outcome = "failed"
    elif case_outcomes.intersection({"partial", "missing"}):
        project_outcome = "partial"
    elif "scientifically_blocked" in case_outcomes:
        project_outcome = "scientifically_blocked"
    elif "complete_with_warnings" in case_outcomes:
        project_outcome = "complete_with_warnings"
    else:
        project_outcome = "complete"
    return {
        "schema_version": "mmpbsa.analysis.v3",
        "project_id": workflow.project_id,
        "protocol_id": workflow.protocol_id,
        "outcome": project_outcome,
        "profile": {
            "profile_id": workflow.profile.profile_id,
            "profile_version": workflow.profile.profile_version,
        },
        "model_order": list(model_order),
        "models": model_results,
        "cases": case_rows,
        "configuration": dict(workflow.analysis),
    }


def write_project_analysis(
    output_dir: Path, analysis: Mapping[str, object]
) -> Mapping[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    analysis_json = atomic_write_json(output / "analysis.json", analysis)
    # The same dependency-free table contract serves in-pipeline analysis and
    # offline export. Keep historical artifact keys while adding useful tables.
    from .export import export_analysis

    return {"analysis": analysis_json, **export_analysis(analysis, output)}

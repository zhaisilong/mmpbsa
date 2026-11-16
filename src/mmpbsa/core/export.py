"""Offline, dependency-free scientific handoff from one analysis archive.

Scores and ranks are copied from authoritative analysis, never recomputed or
pooled across projects. Native and selected scores deliberately have separate
unit columns. The public exporter does not construct a runner or load providers.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from mmpbsa.utils import atomic_write_json, atomic_write_text, sha256_file

from .errors import ContractError

EXPORT_SCHEMA = "mmpbsa.export.v1"
CASE_COLUMNS = (
    "case_id", "label", "model_id", "reference", "replica_count", "score_mean",
    "score_stddev", "score_sem", "comparison_eligible", "comparison_exclusion_reasons",
    "comparison_group", "resolved_internal_dielectric", "rank", "qc_verdict", "outcome",
    "score_unit", "project_id", "source_analysis_sha256", "scientific_context_status",
)
REPLICA_COLUMNS = (
    "project_id", "case_id", "replica_id", "seed", "model_id", "source_model_id",
    "model_status", "qc_verdict", "qc_block_reasons", "qc_warn_reasons", "score",
    "score_unit", "score_raw", "score_raw_unit", "frame_count", "frame_manifest_digest",
    "frame_ids_sha256", "ligand_primary_rmsd_p95_angstrom",
    "interface_backbone_rmsd_p95_angstrom", "native_contacts_min",
    "interface_distance_max_angstrom", "source_analysis_sha256",
)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: object, label: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractError(f"{label} must be a sequence")
    if any(not isinstance(row, Mapping) for row in value):
        raise ContractError(f"{label} rows must be mappings")
    return tuple(value)


def _finite(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ContractError("export score must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("export score must be numeric") from exc
    if not math.isfinite(result):
        raise ContractError("export score must be finite")
    return result


def load_analysis(source: Mapping[str, object] | Path) -> tuple[Mapping[str, object], str]:
    """Read a JSON/gzip archive without modifying it; mappings have no file hash."""
    if isinstance(source, Mapping):
        return source, ""
    path = Path(source)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        digest = sha256_file(path)
        with opener(path, "rt", encoding="utf-8-sig") as handle:
            analysis = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read analysis archive: {path}") from exc
    if not isinstance(analysis, Mapping):
        raise ContractError("analysis archive must contain a mapping")
    if sha256_file(path) != digest:
        raise ContractError("source analysis changed while reading the archive")
    return analysis, digest


def _csv(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _reasons(qc: Mapping[str, object], verdict: str) -> str:
    checks = qc.get("check_results", qc.get("issues", ()))
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes, bytearray)):
        return ""
    return ";".join(
        str(item.get("id") or item.get("code") or "unknown")
        for item in checks
        if isinstance(item, Mapping) and str(item.get("verdict", "")).upper() == verdict
    )


def _model_status(row: Mapping[str, object], model_id: str, score: object) -> str:
    if str(row.get("qc_verdict", "")).upper() == "BLOCK":
        return "skipped_qc_block"
    if score is not None:
        return "completed"
    statuses = _mapping(row.get("model_result_statuses")) or _mapping(row.get("model_statuses"))
    status = statuses.get(model_id)
    if isinstance(status, Mapping):
        status = (
            status.get("result_status") or status.get("task_result")
            or status.get("status") or status.get("state")
        )
    if status is None:
        return "not_started"
    # Preserve archived status detail, while normalizing common execution outcomes.
    value = str(status)
    return {"failed": "execution_failed", "missing": "not_started"}.get(value, value)


def build_export_tables(
    analysis: Mapping[str, object], *, source_sha256: str = ""
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Validate and flatten a single archive; never infer missing measurements."""
    if analysis.get("schema_version") not in {"mmpbsa.analysis.v2", "mmpbsa.analysis.v3"}:
        raise ContractError("export requires an analysis v2 or v3 archive")
    cases = _rows(analysis.get("cases", ()), "analysis.cases")
    raw_order = analysis.get("model_order", ())
    if not isinstance(raw_order, Sequence) or isinstance(raw_order, (str, bytes, bytearray)):
        raise ContractError("analysis.model_order must be a sequence")
    if any(not isinstance(model, str) or not model for model in raw_order):
        raise ContractError("analysis.model_order identifiers must be nonempty strings")
    order = tuple(raw_order)
    if not order or len(order) != len(set(order)):
        raise ContractError("analysis.model_order must be nonempty and unique")
    project_id = analysis.get("project_id", "")
    models = _mapping(analysis.get("models"))
    case_rows: list[dict[str, object]] = []
    replica_rows: list[dict[str, object]] = []
    case_ids: set[str] = set()
    qc_counts: dict[str, int] = {}
    case_qc_counts: dict[str, int] = {}
    eligible_by_model = dict.fromkeys(order, 0)
    scores_by_model = dict.fromkeys(order, 0)
    blocked_replicas = 0
    total_replicas = 0
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in case_ids:
            raise ContractError("export requires unique nonempty case identifiers")
        case_ids.add(case_id)
        case_verdict = str(case.get("qc_verdict") or "unknown").upper()
        case_qc_counts[case_verdict] = case_qc_counts.get(case_verdict, 0) + 1
        record = _mapping(case.get("record"))
        context = _mapping(record.get("scientific_context"))
        case_models = _mapping(case.get("models"))
        replicas = _rows(record.get("replicas", ()), f"{case_id}.replicas")
        replica_ids: set[str] = set()
        for model_id in order:
            model = _mapping(case_models.get(model_id))
            if not model:
                raise ContractError(f"case {case_id} is missing model {model_id}")
            count = model.get("replica_count")
            sd = _finite(model.get("score_stddev"))
            sem = _finite(model.get("score_sem"))
            if isinstance(count, int) and count < 2 and (sd is not None or sem is not None):
                raise ContractError("n < 2 replica SD/SEM must be unavailable, not zero")
            eligible = model.get("comparison_eligible")
            rank = model.get("rank")
            if eligible is not True and rank is not None:
                raise ContractError("excluded cases cannot have a comparison rank")
            if eligible is True and (
                not isinstance(rank, int) or isinstance(rank, bool) or rank < 1
            ):
                raise ContractError("eligible cases require a positive original comparison rank")
            if str(case.get("qc_verdict", "")).upper() == "BLOCK" and (
                eligible is True or rank is not None
            ):
                raise ContractError("QC BLOCK case cannot enter ranking")
            eligible_by_model[model_id] += eligible is True
            case_rows.append({
                "case_id": case_id, "label": case.get("label"), "model_id": model_id,
                "reference": case.get("reference"), "replica_count": count,
                "score_mean": _finite(model.get("score_mean")), "score_stddev": sd,
                "score_sem": sem, "comparison_eligible": eligible,
                "comparison_exclusion_reasons": ";".join(
                    str(item) for item in model.get("comparison_exclusion_reasons", ())
                ),
                "comparison_group": model.get("comparison_group"),
                "resolved_internal_dielectric": model.get("resolved_internal_dielectric"),
                "rank": rank, "qc_verdict": case.get("qc_verdict"),
                "outcome": case.get("outcome"), "score_unit": model.get("score_unit"),
                "project_id": project_id, "source_analysis_sha256": source_sha256,
                "scientific_context_status": context.get("status", "unknown"),
            })
        for replica in replicas:
            replica_id = str(replica.get("replica_id", ""))
            if not replica_id or replica_id in replica_ids:
                raise ContractError(f"{case_id} requires unique nonempty replica identifiers")
            replica_ids.add(replica_id)
            total_replicas += 1
            qc = _mapping(_mapping(replica.get("trajectory_qc")).get("payload"))
            if replica.get("qc_verdict") and qc.get("verdict") and (
                str(replica["qc_verdict"]).upper() != str(qc["verdict"]).upper()
            ):
                raise ContractError("replica and QC receipt have mismatched verdicts")
            verdict = str(replica.get("qc_verdict") or qc.get("verdict") or "unknown").upper()
            qc_counts[verdict] = qc_counts.get(verdict, 0) + 1
            blocked_replicas += verdict == "BLOCK"
            metrics = _mapping(qc.get("metrics"))
            scoring = _mapping(replica.get("scoring_models"))
            for model_id in order:
                attempt = _mapping(scoring.get(model_id))
                payload = _mapping(attempt.get("payload"))
                model_config = _mapping(models.get(model_id))
                transform = _mapping(_mapping(model_config.get("score_selection")).get("transform"))
                score = _finite(attempt.get("selected_score"))
                raw_score = _finite(attempt.get("selected_score_raw"))
                if verdict == "BLOCK" and (score is not None or raw_score is not None):
                    raise ContractError("QC BLOCK replica cannot carry a selected score")
                scale, offset = _finite(transform.get("scale")), _finite(transform.get("offset"))
                if score is not None and raw_score is not None and scale is not None:
                    if not math.isclose(score, raw_score * scale + (offset or 0), rel_tol=1e-9):
                        raise ContractError("selected score disagrees with native score transform")
                if score is not None:
                    scores_by_model[model_id] += 1
                frame_manifest = _mapping(payload.get("frame_manifest"))
                identity = _mapping(payload.get("frame_ids"))
                replica_rows.append({
                    "project_id": project_id, "case_id": case_id, "replica_id": replica_id,
                    "seed": replica.get("seed"), "model_id": model_id,
                    "source_model_id": attempt.get("source_model_id", model_id),
                    "model_status": _model_status(
                        {**replica, "qc_verdict": verdict}, model_id, score
                    ),
                    "qc_verdict": verdict, "qc_block_reasons": _reasons(qc, "BLOCK"),
                    "qc_warn_reasons": _reasons(qc, "WARN"), "score": score,
                    "score_unit": _mapping(case_models.get(model_id)).get("score_unit"),
                    "score_raw": raw_score, "score_raw_unit": transform.get("input_unit"),
                    "frame_count": payload.get("frame_count") if score is not None else None,
                    "frame_manifest_digest": frame_manifest.get("digest"),
                    "frame_ids_sha256": identity.get("sha256"),
                    **{name: metrics.get(name) for name in REPLICA_COLUMNS[17:21]},
                    "source_analysis_sha256": source_sha256,
                })
        if replicas:
            for model_id in order:
                observed = [
                    _finite(_mapping(_mapping(replica.get("scoring_models")).get(model_id)).get(
                        "selected_score"
                    ))
                    for replica in replicas
                ]
                scores = [score for score in observed if score is not None]
                model = _mapping(case_models.get(model_id))
                if model.get("replica_count") != len(scores):
                    raise ContractError("case replica_count disagrees with selected replica scores")
                expected = {
                    "score_mean": statistics.fmean(scores) if scores else None,
                    "score_stddev": statistics.stdev(scores) if len(scores) > 1 else None,
                }
                expected["score_sem"] = (
                    expected["score_stddev"] / math.sqrt(len(scores))
                    if expected["score_stddev"] is not None else None
                )
                for field, value in expected.items():
                    actual = _finite(model.get(field))
                    if (actual is None) != (value is None) or (
                        actual is not None and value is not None
                        and not math.isclose(actual, value, rel_tol=1e-8, abs_tol=1e-10)
                    ):
                        raise ContractError(f"case {field} disagrees with selected replica scores")
    summary = {
        "schema_version": EXPORT_SCHEMA, "project_id": project_id, "valid": True,
        "source_analysis_sha256": source_sha256, "case_count": len(cases),
        "case_model_row_count": len(case_rows), "replica_count": total_replicas,
        "replica_model_row_count": len(replica_rows), "qc_counts": qc_counts,
        "case_qc_counts": case_qc_counts, "eligible_cases_by_model": eligible_by_model,
        "blocked_replica_count": blocked_replicas, "scored_replicas_by_model": scores_by_model,
        "ranking_policy": "copied from one source analysis within its comparison groups",
        "missing_value_policy": "blank means unavailable; never imputed as zero",
    }
    return case_rows, replica_rows, summary


def _dictionary() -> list[dict[str, object]]:
    descriptions = {
        "case_id": "体系原始编号", "label": "体系显示名", "model_id": "分析模型编号",
        "reference": "输入参考值；不是由分数推算的实验活性", "replica_count": "已计分replica数",
        "score_mean": "replica选定分数均值；含部分结果的体系仍不一定可排名",
        "score_stddev": "replica均值间的样本标准差", "score_sem": "样本标准差除以sqrt(n)",
        "comparison_eligible": "权威分析记录中的比较/排名资格",
        "comparison_exclusion_reasons": "原始排名排除原因；分号分隔",
        "comparison_group": "原项目内部可比组；不能跨项目合并排名",
        "resolved_internal_dielectric": "实际解析的内部介电常数", "rank": "原项目可比组内排名",
        "qc_verdict": "科学QC判定；BLOCK不是技术执行错误", "outcome": "原始体系完成状态",
        "score_unit": "已选定、已转换分数的单位；不得再次换算", "project_id": "原项目编号",
        "source_analysis_sha256": "输入归档字节的SHA256；内存输入时为空",
        "scientific_context_status": "科学来源摘要状态；旧来源缺失为unknown",
        "replica_id": "原始replica编号", "seed": "模拟随机种子",
        "source_model_id": "生成此分析分数的计算模型编号",
        "model_status": "completed/skipped_qc_block/execution_failed等；保留旧详细状态",
        "qc_block_reasons": "BLOCK检查编号；分号分隔", "qc_warn_reasons": "WARN检查编号；分号分隔",
        "score": "权威selected_score，直接复制，不重新选择/转换",
        "score_raw": "原生selected_score_raw；与已转换分数分列",
        "score_raw_unit": "原生分数单位；不是score_unit",
        "frame_count": "该replica该模型实际评分帧数；BLOCK未计分留空",
        "frame_manifest_digest": "实际评分selection receipt的digest",
        "frame_ids_sha256": "实际评分帧编号的SHA256",
        "ligand_primary_rmsd_p95_angstrom": "ligand primary RMSD的P95；小分子重原子/肽backbone",
        "interface_backbone_rmsd_p95_angstrom": "受体界面backbone RMSD的P95",
        "native_contacts_min": "窗口内初始原生接触最小保留数；0不等于所有受体接触为0",
        "interface_distance_max_angstrom": "窗口内受体-ligand最小重原子距离的最大值",
    }
    rows = []
    for table, columns in (("cases.csv", CASE_COLUMNS), ("replicas.csv", REPLICA_COLUMNS)):
        for name in columns:
            unit = "Å" if name.endswith("angstrom") else (
                "score_unit" if name in {"score", "score_mean", "score_stddev", "score_sem"}
                else "score_raw_unit" if name == "score_raw" else ""
            )
            rows.append({
                "table": table, "column": name, "description_zh": descriptions[name],
                "unit": unit, "source": "authoritative analysis archive",
                "missing_value_meaning": "不可用/未记录；非零值；SD/SEM在n<2时不可用",
                "caveat": "分数不是实验亲和力/IC50；累计均值走平不证明收敛；不猜采购标识",
            })
    return rows


def export_analysis(
    analysis: Mapping[str, object] | Path, output_dir: Path
) -> Mapping[str, Path]:
    """Publish UTF-8 BOM CSVs plus a validation summary from one archive.

    All validation happens before creating output. An archive cannot be its own
    output target, and its hash is rechecked after publication.
    """
    source, digest = load_analysis(analysis)
    cases, replicas, summary = build_export_tables(source, source_sha256=digest)
    output = Path(output_dir).resolve()
    if not isinstance(analysis, Mapping) and Path(analysis).resolve().parent == output:
        raise ContractError("export output must be separate from the source archive directory")
    artifacts = {}
    for key, name, columns, rows in (
        ("case_table", "cases.csv", CASE_COLUMNS, cases),
        ("replica_table", "replicas.csv", REPLICA_COLUMNS, replicas),
        ("field_dictionary", "field_dictionary.csv", tuple(_dictionary()[0]), _dictionary()),
    ):
        artifacts[key] = atomic_write_text(output / name, _csv(columns, rows), encoding="utf-8-sig")
    artifacts["validation_summary"] = atomic_write_json(output / "validation_summary.json", summary)
    if not isinstance(analysis, Mapping) and sha256_file(Path(analysis)) != digest:
        raise ContractError("source analysis changed during export")
    return artifacts

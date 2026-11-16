"""Optional evidence-backed scientific provenance, without chemistry imports.

The summary distinguishes planned frame selection from actual receipts. Missing
legacy evidence is unknown, not reconstructed from preset names or file paths.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from .prepared import PreparedCase

SCIENTIFIC_CONTEXT_SCHEMA = "mmpbsa.scientific-context.v1"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def verified_frame_selection(
    planned: Mapping[str, object] | None, receipt: Mapping[str, object] | None
) -> dict[str, object]:
    """Validate canonical selection and its *observed* receipt; never guess time.

    A digest alone is insufficient. Integer source-frame geometry and explicit
    production origin are required before analysis frame -> absolute ns mapping.
    """
    unknown: dict[str, object] = {"status": "unknown"}
    if not planned or not receipt:
        return unknown
    selection = _mapping(receipt.get("selection"))
    try:
        digest = _digest(planned)
    except (TypeError, ValueError):
        return {"status": "unknown", "reason": "invalid_canonical_selection"}
    if selection != planned or receipt.get("digest") != digest:
        return {"status": "unknown", "reason": "selection_receipt_mismatch"}
    fields = ("start_frame", "end_frame", "stride", "expected_frames")
    if any(isinstance(planned.get(key), bool) or not isinstance(planned.get(key), int)
           for key in fields):
        return {"status": "unknown", "reason": "invalid_frame_geometry"}
    start, end, stride, count = (int(planned[key]) for key in fields)
    interval = planned.get("source_trajectory_interval_ps")
    if (
        planned.get("schema_version") != "mmpbsa.frame-manifest.v1"
        or planned.get("source_stage") != "production"
        or start < 1 or end < start or stride < 1 or count < 1
        or (end - start) % stride or (end - start) // stride + 1 != count
        or isinstance(interval, bool) or not isinstance(interval, (int, float))
        or not 0 < interval < float("inf")
    ):
        return {"status": "unknown", "reason": "invalid_frame_geometry"}
    return {
        "status": "verified", "digest": receipt["digest"], "selection": dict(planned),
        "analysis_start_ps": (start - 1) * interval,
        "analysis_interval_ps": stride * interval, "expected_frames": count,
        "time_origin": "production_t0",
        "evidence": "completed_stage_selection_receipt_matches_canonical_plan",
    }


def build_scientific_context(
    prepared: PreparedCase | None, *,
    frame_selection: Mapping[str, object] | None = None,
    replicas: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Summarize a sealed PreparedCase and the supplied finalized replica evidence.

    Caller must pass the already verified PreparedCase, not an unvalidated JSON
    dictionary. No reads/writes, provider discovery, RDKit, or network requests.
    A completed trajectory receipt can accompany QC as
    ``trajectory_frame_manifest``; QC otherwise needs its own selection receipt.
    """
    result: dict[str, object] = {
        "schema_version": SCIENTIFIC_CONTEXT_SCHEMA,
        "status": "available" if prepared is not None else "unknown",
        "prepared_case": {"status": "unknown"},
        "planned_frame_selection": dict(frame_selection) if frame_selection else None,
        "replicas": {},
    }
    if prepared is not None:
        manifest = prepared.manifest
        result["prepared_case"] = {
            "status": "verified", "case_id": prepared.case_id,
            "manifest_digest": _digest(manifest.as_dict()),
            "primary_binding": manifest.primary_binding.as_dict(),
            "entities": [
                {
                    "entity_id": entity.entity_id, "kind": entity.kind,
                    "md_atom_count": len(entity.md_atom_indices),
                    "scoring_atom_count": len(entity.scoring_atom_indices),
                    "selection_signature": entity.selection_signature,
                    "named_selections": {
                        name: selection.as_dict()
                        for name, selection in entity.named_selections.items()
                    },
                    "features": [feature.as_dict() for feature in entity.features],
                    "metadata": dict(entity.metadata),
                }
                for entity in manifest.entities
            ],
            "modeling_provenance": dict(manifest.provenance),
        }
    summaries = {}
    for replica in replicas:
        replica_id = str(replica.get("replica_id", ""))
        if not replica_id:
            continue
        qc = _mapping(_mapping(replica.get("trajectory_qc")).get("payload"))
        qc_receipt = _mapping(qc.get("frame_manifest")) or _mapping(
            replica.get("trajectory_frame_manifest")
        )
        scoring = _mapping(replica.get("scoring_models"))
        summaries[replica_id] = {
            "qc_frame_selection": verified_frame_selection(frame_selection, qc_receipt),
            "qc_reference": dict(_mapping(qc.get("reference_definition"))) or {"status": "unknown"},
            "scoring": {
                str(model): {
                    "frame_selection": verified_frame_selection(
                        frame_selection, _mapping(_mapping(attempt).get("payload")).get(
                            "frame_manifest"
                        )
                    ),
                    "frame_ids": dict(_mapping(
                        _mapping(_mapping(attempt).get("payload")).get("frame_ids")
                    )) or {"status": "unknown"},
                }
                for model, attempt in scoring.items()
            },
        }
    result["replicas"] = summaries
    return result

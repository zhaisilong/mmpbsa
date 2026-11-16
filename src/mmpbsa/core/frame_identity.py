"""Cross-model frame identity checks for case finalization."""

from __future__ import annotations

from collections.abc import Mapping

from .config import WorkflowDefinition
from .errors import ContractError
from .models import StageInput


def require_gb_pb_frame_parity(
    workflow: WorkflowDefinition,
    case_id: str,
    replica_id: str,
    scoring_models: Mapping[str, StageInput],
) -> None:
    """Fail closed unless completed GB/PB used one planned and observed frame set."""

    if set(scoring_models) != {"gb", "pb"}:
        return

    planned: dict[str, str] = {}
    observed: dict[str, tuple[int, int, int, str]] = {}
    for model_id in ("gb", "pb"):
        item = scoring_models[model_id]
        model_science = workflow.scoring_model(model_id)
        raw_planned_digest = model_science.get("frame_manifest_digest")
        if not _is_sha256(raw_planned_digest):
            raise ContractError(
                "completed GB/PB scoring requires a planned "
                f"frame_manifest_digest: {case_id}/{replica_id}/{model_id}"
            )
        planned_digest = str(raw_planned_digest)
        planned[model_id] = planned_digest

        manifest = item.payload.get("frame_manifest")
        reported_digest = (
            manifest.get("digest") if isinstance(manifest, Mapping) else None
        )
        if reported_digest != planned_digest:
            raise ContractError(
                "completed scoring frame-manifest receipt differs from the "
                f"planned digest: {case_id}/{replica_id}/{model_id}"
            )

        identity = item.payload.get("frame_ids")
        if not isinstance(identity, Mapping):
            raise ContractError(
                "completed GB/PB scoring has no observed frame-ID identity: "
                f"{case_id}/{replica_id}/{model_id}"
            )
        raw_count = identity.get("count")
        raw_first = identity.get("first")
        raw_last = identity.get("last")
        raw_digest = identity.get("sha256")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (raw_count, raw_first, raw_last)
        ) or not _is_sha256(raw_digest):
            raise ContractError(
                "completed GB/PB scoring has an invalid observed frame-ID "
                f"identity: {case_id}/{replica_id}/{model_id}"
            )
        count, first, last = int(raw_count), int(raw_first), int(raw_last)
        digest = str(raw_digest)
        expected = model_science.get("expected_frames")
        frame_count = item.payload.get("frame_count")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected < 1
            or frame_count != count
            or count != expected
            or first != 1
            or last != expected
        ):
            raise ContractError(
                "completed GB/PB scoring observed-frame summary differs from "
                f"the planned 1..N range: {case_id}/{replica_id}/{model_id}"
            )
        observed[model_id] = (count, first, last, digest)

    if planned["gb"] != planned["pb"]:
        raise ContractError(
            "completed GB/PB scoring planned frame-manifest digests differ: "
            f"{case_id}/{replica_id}"
        )
    if observed["gb"] != observed["pb"]:
        raise ContractError(
            "completed GB/PB scoring observed frame-ID digests differ: "
            f"{case_id}/{replica_id}"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

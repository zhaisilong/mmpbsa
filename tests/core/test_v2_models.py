from __future__ import annotations

import pytest

from mmpbsa.core.errors import ContractError
from mmpbsa.core.models import (
    AttemptSummary,
    JobKey,
    JobResult,
    ResourceRequest,
    StageRequest,
    StageResult,
    StageService,
    TaskKey,
)
from mmpbsa.core.state import ArtifactState, AttemptState, StageId


def _attempt(
    stage: StageId,
    state: AttemptState,
    *,
    model_id: str | None = None,
) -> AttemptSummary:
    return AttemptSummary(
        task=TaskKey(
            stage,
            case_id="case01",
            replica_id="rep01",
            model_id=model_id,
        ),
        attempt_id="0001",
        state=state,
        artifact_state=ArtifactState.COMPLETE,
        science_hash="a" * 64,
        compatibility_key="fake:v2:engine:1",
        artifacts={},
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cpu_slots": 0}, "cpu_slots"),
        ({"cpu_slots": True}, "cpu_slots"),
        ({"gpu_slots": -1}, "gpu_slots"),
        ({"gpu_slots": False}, "gpu_slots"),
    ],
)
def test_resource_request_is_strict(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        ResourceRequest(**kwargs)  # type: ignore[arg-type]


def test_stage_service_v2_includes_resource_request() -> None:
    class Service:
        def resource_request(self, request: StageRequest) -> ResourceRequest:
            return ResourceRequest(cpu_slots=2, gpu_slots=1)

        def compatibility(self, runtime: object) -> object:
            return object()

        def execute(self, request: StageRequest) -> StageResult:
            return StageResult(artifacts={})

        def resume(self, request: StageRequest) -> StageResult:
            return StageResult(artifacts={})

    assert isinstance(Service(), StageService)


def test_job_result_applies_any_or_all_to_scoring_models_only() -> None:
    completed_md = _attempt(StageId.MD, AttemptState.COMPLETED)
    completed_gb = _attempt(
        StageId.SCORING,
        AttemptState.COMPLETED,
        model_id="gb",
    )
    failed_pb = _attempt(
        StageId.SCORING,
        AttemptState.FAILED,
        model_id="pb",
    )
    common = {
        "job": JobKey("case01", "rep01"),
        "through": StageId.SCORING,
        "attempts": (completed_md, completed_gb, failed_pb),
        "expected_model_ids": ("gb", "pb"),
        "model_failures": {"pb": "backend failed"},
    }

    assert JobResult(completion_policy="any", **common).completed is True
    assert JobResult(completion_policy="all", **common).completed is False

    failed_md = _attempt(StageId.MD, AttemptState.FAILED)
    assert (
        JobResult(
            completion_policy="any",
            **{**common, "attempts": (failed_md, completed_gb)},
        ).completed
        is False
    )


def test_job_result_validates_model_completion_metadata() -> None:
    job = JobKey("case01", "rep01")

    with pytest.raises(ContractError, match="completion_policy"):
        JobResult(
            job=job,
            through=StageId.SCORING,
            attempts=(),
            completion_policy="first",
        )
    with pytest.raises(ContractError, match="unique"):
        JobResult(
            job=job,
            through=StageId.SCORING,
            attempts=(),
            expected_model_ids=("gb", "gb"),
        )
    with pytest.raises(ContractError, match="unexpected model IDs"):
        JobResult(
            job=job,
            through=StageId.SCORING,
            attempts=(),
            expected_model_ids=("gb",),
            model_failures={"pb": "failed"},
        )

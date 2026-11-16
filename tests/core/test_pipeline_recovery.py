from types import SimpleNamespace

import pytest

from mmpbsa.core import ContractError, StageId, TaskKey
from mmpbsa.core.pipeline import recovery_md_action, validate_recovery_mode
from mmpbsa.core.state import AttemptState


@pytest.mark.parametrize(
    "state,expected",
    [
        (None, "run"),
        (AttemptState.COMPLETED, "run"),
        (AttemptState.RUNNING, "resume"),
        (AttemptState.INTERRUPTED, "resume"),
    ],
)
def test_recovery_uses_native_resume_without_new_md_attempt(state, expected):
    latest = (
        None
        if state is None
        else SimpleNamespace(state=state, science_hash="same", compatibility_key="same-backend")
    )
    runner = SimpleNamespace(
        workspace=SimpleNamespace(latest=lambda _: latest),
        _prepared=lambda _: object(),
        _science=lambda *_: {},
        _science_hash=lambda *_: "same",
        _stage_execution_contract=lambda *_: (
            None,
            None,
            None,
            SimpleNamespace(key="same-backend"),
        ),
    )
    assert (
        recovery_md_action(runner, TaskKey(StageId.MD, case_id="test", replica_id="rep01"))
        == expected
    )


@pytest.mark.parametrize("changed", ["science_hash", "compatibility_key", "failed"])
def test_recovery_never_silently_restarts_changed_or_failed_md(changed):
    latest = SimpleNamespace(
        state=AttemptState.COMPLETED, science_hash="same", compatibility_key="same-backend"
    )
    if changed == "failed":
        latest.state = AttemptState.FAILED
    else:
        setattr(latest, changed, "changed")
    runner = SimpleNamespace(
        workspace=SimpleNamespace(latest=lambda _: latest),
        _prepared=lambda _: object(),
        _science=lambda *_: {},
        _science_hash=lambda *_: "same",
        _stage_execution_contract=lambda *_: (
            None,
            None,
            None,
            SimpleNamespace(key="same-backend"),
        ),
    )
    with pytest.raises(ContractError):
        recovery_md_action(runner, TaskKey(StageId.MD, case_id="test", replica_id="rep01"))


@pytest.mark.parametrize(
    "through,action,mode",
    [
        (StageId.MD, "new", "pipeline"),
        (StageId.SCORING, "run", "pipeline"),
        (StageId.SCORING, "new", "jobs"),
    ],
)
def test_recovery_flag_requires_explicit_downstream_pipeline_rebuild(through, action, mode):
    with pytest.raises(ContractError, match="requires batch run --new"):
        validate_recovery_mode(through, action, mode)
    validate_recovery_mode(StageId.SCORING, "new", "pipeline")


def test_failed_md_uses_explicit_retry_only_with_native_recovery_hook():
    latest = SimpleNamespace(
        state=AttemptState.FAILED, science_hash="same", compatibility_key="same-backend"
    )
    runner = SimpleNamespace(
        workspace=SimpleNamespace(latest=lambda _: latest),
        _prepared=lambda _: object(),
        _science=lambda *_: {},
        _science_hash=lambda *_: "same",
        _stage_execution_contract=lambda *_: (
            SimpleNamespace(recover_failed_md=lambda *_: None),
            None,
            None,
            SimpleNamespace(key="same-backend"),
        ),
    )
    assert (
        recovery_md_action(runner, TaskKey(StageId.MD, case_id="test", replica_id="rep01"))
        == "retry"
    )

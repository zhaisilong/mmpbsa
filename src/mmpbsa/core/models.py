"""Stable workflow value objects and provider-facing execution requests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .errors import ContractError
from .state import ArtifactState, AttemptState, QcVerdict, StageId

if TYPE_CHECKING:
    from .prepared import PreparedBindingView, PreparedMdView, PreparedTrajectoryView

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require_identifier(value: str, label: str) -> str:
    text = str(value)
    if not _IDENTIFIER.fullmatch(text):
        raise ContractError(
            f"{label} must start with a letter or number and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return text


@dataclass(frozen=True, slots=True, order=True)
class JobKey:
    case_id: str
    replica_id: str

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        require_identifier(self.replica_id, "replica_id")

    @property
    def label(self) -> str:
        return f"{self.case_id}/{self.replica_id}"


@dataclass(frozen=True, slots=True, order=True)
class TaskKey:
    stage: StageId
    case_id: str | None = None
    replica_id: str | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, StageId):
            raise ContractError("stage must be a StageId")
        for label, value in (
            ("case_id", self.case_id),
            ("replica_id", self.replica_id),
            ("model_id", self.model_id),
        ):
            if value is not None and not isinstance(value, str):
                raise ContractError(f"{label} must be a string")
            if value is not None:
                require_identifier(value, label)
        if self.stage is StageId.ADMIT and (
            self.case_id is None
            or self.replica_id is not None
            or self.model_id is not None
        ):
            raise ContractError("admit is case-scoped")
        if self.stage in {
            StageId.MD,
            StageId.TRAJECTORY,
            StageId.TRAJECTORY_QC,
        } and (
            self.case_id is None
            or self.replica_id is None
            or self.model_id is not None
        ):
            raise ContractError(f"{self.stage.value} is replica-scoped")
        if self.stage is StageId.SCORING and (
            self.case_id is None
            or self.replica_id is None
            or self.model_id is None
        ):
            raise ContractError("scoring is replica-and-model-scoped")
        if self.stage is StageId.FINALIZE_CASE and (
            self.case_id is None
            or self.replica_id is not None
            or self.model_id is not None
        ):
            raise ContractError("finalize_case is case-scoped")
        if self.stage in {StageId.ANALYSIS, StageId.ASSETS, StageId.REPORT} and (
            self.case_id is not None
            or self.replica_id is not None
            or self.model_id is not None
        ):
            raise ContractError(f"{self.stage.value} is project-scoped")

    @property
    def label(self) -> str:
        parts = [self.stage.value]
        if self.case_id is not None:
            parts.append(self.case_id)
        if self.replica_id is not None:
            parts.append(self.replica_id)
        if self.model_id is not None:
            parts.append(self.model_id)
        return "/".join(parts)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A content-addressed file reference relative to an explicit root."""

    path: Path
    sha256: str
    size: int
    role: str
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        path = Path(self.path)
        if path.is_absolute() or path == Path(".") or ".." in path.parts:
            raise ContractError("artifact path must be a safe relative path")
        if not _SHA256.fullmatch(self.sha256):
            raise ContractError("artifact sha256 must be a lowercase SHA-256 digest")
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 1
        ):
            raise ContractError("artifact size must be a positive integer")
        if not self.role.strip() or not self.media_type.strip():
            raise ContractError("artifact role and media_type cannot be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "size": self.size,
            "role": self.role,
            "media_type": self.media_type,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ArtifactRef:
        allowed = {"path", "sha256", "size", "role", "media_type"}
        unknown = set(value).difference(allowed)
        if unknown:
            raise ContractError(
                "unknown artifact fields: " + ", ".join(sorted(unknown))
            )
        raw_size = value.get("size")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise ContractError("artifact size must be a positive integer")
        try:
            return cls(
                path=Path(str(value["path"])),
                sha256=str(value["sha256"]),
                size=raw_size,
                role=str(value["role"]),
                media_type=str(
                    value.get("media_type") or "application/octet-stream"
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("invalid artifact reference") from exc


@dataclass(frozen=True, slots=True)
class ReplicaSpec:
    replica_id: str
    replica_index: int
    seed: int

    def __post_init__(self) -> None:
        require_identifier(self.replica_id, "replica_id")
        if (
            isinstance(self.replica_index, bool)
            or not isinstance(self.replica_index, int)
            or self.replica_index < 1
        ):
            raise ContractError("replica_index must be >= 1")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ContractError("seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class BackendCompatibility:
    provider_id: str
    adapter_contract: str
    engine: str
    engine_major: str
    engine_version: str

    def __post_init__(self) -> None:
        for label, value in (
            ("provider_id", self.provider_id),
            ("adapter_contract", self.adapter_contract),
            ("engine", self.engine),
            ("engine_major", self.engine_major),
            ("engine_version", self.engine_version),
        ):
            if not str(value).strip():
                raise ContractError(f"{label} cannot be empty")

    @property
    def key(self) -> str:
        return (
            f"{self.provider_id}:{self.adapter_contract}:"
            f"{self.engine}:{self.engine_major}"
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "provider_id": self.provider_id,
            "adapter_contract": self.adapter_contract,
            "engine": self.engine,
            "engine_major": self.engine_major,
            "engine_version": self.engine_version,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> BackendCompatibility:
        allowed = {
            "provider_id",
            "adapter_contract",
            "engine",
            "engine_major",
            "engine_version",
        }
        missing = allowed.difference(value)
        unknown = set(value).difference(allowed)
        if missing or unknown:
            detail = []
            if missing:
                detail.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                detail.append("unknown " + ", ".join(sorted(unknown)))
            raise ContractError("invalid backend compatibility: " + "; ".join(detail))
        return cls(
            provider_id=str(value["provider_id"]),
            adapter_contract=str(value["adapter_contract"]),
            engine=str(value["engine"]),
            engine_major=str(value["engine_major"]),
            engine_version=str(value["engine_version"]),
        )


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    """Execution-only lease; never part of science or compatibility identity."""

    cpu_slots: int
    cpu_ids: tuple[int, ...] = ()
    gpu_ids: tuple[str, ...] = ()
    physical_gpu_ids: tuple[str, ...] = ()
    lease_token: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.cpu_slots, bool)
            or not isinstance(self.cpu_slots, int)
            or self.cpu_slots < 1
        ):
            raise ContractError("allocation cpu_slots must be positive")
        if any(
            isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in self.cpu_ids
        ):
            raise ContractError("allocation CPU IDs must be non-negative integers")
        if self.cpu_ids and len(self.cpu_ids) != self.cpu_slots:
            raise ContractError("allocation CPU IDs must match cpu_slots")
        if len(set(self.cpu_ids)) != len(self.cpu_ids) or len(set(self.gpu_ids)) != len(
            self.gpu_ids
        ):
            raise ContractError("allocation resource IDs must be unique")
        if self.physical_gpu_ids and (
            len(self.physical_gpu_ids) != len(self.gpu_ids)
            or len(set(self.physical_gpu_ids)) != len(self.physical_gpu_ids)
        ):
            raise ContractError("physical GPU IDs must uniquely match the engine GPU IDs")
        if self.lease_token is not None and (
            not isinstance(self.lease_token, str) or not self.lease_token
        ):
            raise ContractError("lease token must be a non-empty string")


@dataclass(frozen=True, slots=True)
class StageRequest:
    task: TaskKey
    science: Mapping[str, object]
    runtime: Mapping[str, object]
    prepared_case: PreparedMdView | PreparedTrajectoryView | PreparedBindingView | None
    upstream: Mapping[str, StageInput]
    attempt_dir: Path
    invocation_dir: Path
    resume: bool = False
    checkpoint: Path | None = None
    compatibility: BackendCompatibility | None = None
    allocation: ResourceAllocation | None = None


@dataclass(frozen=True, slots=True)
class StageInput:
    """Verified outputs from a stage or an admitted external artifact set."""

    task: TaskKey
    attempt_id: str
    root: Path
    artifacts: Mapping[str, Path]
    science_hash: str = ""
    payload: Mapping[str, object] = field(default_factory=dict)
    qc_verdict: QcVerdict | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    artifacts: Mapping[str, Path]
    payload: Mapping[str, object] = field(default_factory=dict)
    qc_verdict: QcVerdict | None = None
    checkpoint: Path | None = None
    invocation_provenance: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Scheduler slots requested by one stage invocation."""

    cpu_slots: int = 1
    gpu_slots: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.cpu_slots, bool)
            or not isinstance(self.cpu_slots, int)
            or self.cpu_slots < 1
        ):
            raise ContractError("cpu_slots must be an integer >= 1")
        if (
            isinstance(self.gpu_slots, bool)
            or not isinstance(self.gpu_slots, int)
            or self.gpu_slots < 0
        ):
            raise ContractError("gpu_slots must be an integer >= 0")


@runtime_checkable
class StageService(Protocol):
    """Structural port implemented by a backend capability."""

    def resource_request(self, request: StageRequest) -> ResourceRequest: ...

    def compatibility(
        self, runtime: Mapping[str, object]
    ) -> BackendCompatibility: ...

    def execute(self, request: StageRequest) -> StageResult: ...

    def resume(self, request: StageRequest) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class AttemptSummary:
    task: TaskKey
    attempt_id: str
    state: AttemptState
    artifact_state: ArtifactState
    science_hash: str
    compatibility_key: str
    artifacts: Mapping[str, ArtifactRef]
    qc_verdict: QcVerdict | None = None
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JobResult:
    job: JobKey
    through: StageId
    attempts: tuple[AttemptSummary, ...]
    completion_policy: str = "all"
    expected_model_ids: tuple[str, ...] = ()
    model_failures: Mapping[str, str] = field(default_factory=dict)
    model_statuses: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.completion_policy not in {"any", "all"}:
            raise ContractError("completion_policy must be 'any' or 'all'")
        if any(
            not isinstance(model_id, str)
            for model_id in self.expected_model_ids
        ):
            raise ContractError(
                "expected_model_ids must contain model ID strings"
            )
        for model_id in self.expected_model_ids:
            require_identifier(model_id, "expected model_id")
        if len(set(self.expected_model_ids)) != len(self.expected_model_ids):
            raise ContractError("expected_model_ids must be unique")
        if any(not isinstance(model_id, str) for model_id in self.model_failures):
            raise ContractError("model_failures keys must be model ID strings")
        for model_id in self.model_failures:
            require_identifier(model_id, "failed model_id")
        unknown_failures = set(self.model_failures).difference(
            self.expected_model_ids
        )
        if unknown_failures:
            raise ContractError(
                "model_failures contains unexpected model IDs: "
                + ", ".join(sorted(unknown_failures))
            )
        if any(
            not isinstance(message, str) or not message
            for message in self.model_failures.values()
        ):
            raise ContractError(
                "model_failures values must be non-empty strings"
            )
        allowed_statuses = {
            "completed", "skipped_qc_block", "execution_failed", "cancelled", "not_started"
        }
        if not isinstance(self.model_statuses, Mapping) or any(
            key not in self.expected_model_ids or value not in allowed_statuses
            for key, value in self.model_statuses.items()
        ):
            raise ContractError("model_statuses contains an unexpected model or invalid status")

    @property
    def completed(self) -> bool:
        if not self.attempts:
            return False
        if not all(
            item.state is AttemptState.COMPLETED
            for item in self.attempts
            if item.task.stage is not StageId.SCORING
        ):
            return False
        if not self.expected_model_ids:
            return all(
                item.state is AttemptState.COMPLETED
                for item in self.attempts
            )
        completed_models = {
            item.task.model_id
            for item in self.attempts
            if item.task.stage is StageId.SCORING
            and item.state is AttemptState.COMPLETED
        }
        expected = set(self.expected_model_ids)
        return (
            bool(expected.intersection(completed_models))
            if self.completion_policy == "any"
            else expected.issubset(completed_models)
        )


@dataclass(frozen=True, slots=True)
class WorkflowProfile:
    profile_id: str
    profile_version: str
    normalized: Mapping[str, object] = field(default_factory=dict)
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier(self.profile_id, "profile_id")
        if not self.profile_version.strip():
            raise ContractError("profile_version cannot be empty")


JsonObject = Mapping[str, Any]

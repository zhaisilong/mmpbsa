"""Orthogonal execution, artifact, and scientific state vocabularies."""

from __future__ import annotations

from enum import Enum


class AttemptState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            AttemptState.COMPLETED,
            AttemptState.FAILED,
            AttemptState.CANCELLED,
        }

    @property
    def active(self) -> bool:
        return not self.terminal


class ArtifactState(str, Enum):
    MISSING = "missing"
    PARTIAL = "partial"
    COMPLETE = "complete"
    INVALID = "invalid"


class QcVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


class WorkflowOutcome(str, Enum):
    COMPLETE = "complete"
    COMPLETE_WITH_WARNINGS = "complete_with_warnings"
    SCIENTIFICALLY_BLOCKED = "scientifically_blocked"
    PARTIAL = "partial"
    EXECUTION_FAILED = "execution_failed"


class StageId(str, Enum):
    ADMIT = "admit"
    MD = "md"
    TRAJECTORY = "trajectory"
    TRAJECTORY_QC = "trajectory_qc"
    SCORING = "scoring"
    FINALIZE_CASE = "finalize_case"
    ANALYSIS = "analysis"
    ASSETS = "assets"
    REPORT = "report"


REPLICA_STAGES = (
    StageId.MD,
    StageId.TRAJECTORY,
    StageId.TRAJECTORY_QC,
    StageId.SCORING,
)

PROJECT_STAGES = (
    StageId.ANALYSIS,
    StageId.ASSETS,
    StageId.REPORT,
)

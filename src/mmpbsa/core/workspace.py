"""Canonical workspace layout and transactional attempt state machine."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mmpbsa.utils import (
    FileLock,
    atomic_directory,
    atomic_write_json,
    resolve_within,
    sha256_file,
    utc_now,
)

from .errors import (
    ActiveAttemptError,
    ArtifactIntegrityError,
    ContractError,
    ResumeError,
    RetryError,
    WorkspaceError,
)
from .models import (
    ArtifactRef,
    AttemptSummary,
    BackendCompatibility,
    StageInput,
    StageResult,
    TaskKey,
)
from .state import ArtifactState, AttemptState, QcVerdict, StageId

WORKSPACE_SCHEMA = "mmpbsa.workspace.v2"
ATTEMPT_SCHEMA = "mmpbsa.attempt.v2"
INVOCATION_SCHEMA = "mmpbsa.invocation.v1"
ExecutionAction = Literal["run", "resume", "retry", "new"]


@dataclass(frozen=True, slots=True)
class AttemptReservation:
    summary: AttemptSummary
    attempt_dir: Path
    invocation_dir: Path | None
    reused: bool
    resume: bool
    checkpoint: Path | None = None


@dataclass(frozen=True, slots=True)
class PruneRecord:
    task: TaskKey
    attempt_id: str
    path: Path
    size_bytes: int
    deleted: bool


class Workspace:
    """Owns all mutable workflow state below one explicit root."""

    def __init__(
        self,
        root: Path,
        *,
        project_id: str,
        protocol_id: str,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "workspace.json"
        self._initialize(project_id=project_id, protocol_id=protocol_id)

    def _initialize(self, *, project_id: str, protocol_id: str) -> None:
        lock = FileLock(
            self.root / ".workspace-init.lock",
            owner=f"mmpbsa:workspace-init:{project_id}",
            timeout_seconds=30,
            stale_after_seconds=24 * 60 * 60,
        )
        with lock:
            self._initialize_locked(
                project_id=project_id,
                protocol_id=protocol_id,
            )

    def _initialize_locked(
        self, *, project_id: str, protocol_id: str
    ) -> None:
        if self._manifest_path.exists():
            raw = self._read_json(self._manifest_path, "workspace manifest")
            if raw.get("schema_version") != WORKSPACE_SCHEMA:
                raise WorkspaceError(
                    f"unsupported workspace schema in {self._manifest_path}"
                )
            observed = (raw.get("project_id"), raw.get("protocol_id"))
            expected = (project_id, protocol_id)
            if observed != expected:
                raise WorkspaceError(
                    "workspace identity differs: "
                    f"expected {expected!r}, observed {observed!r}"
                )
            return
        atomic_write_json(
            self._manifest_path,
            {
                "schema_version": WORKSPACE_SCHEMA,
                "project_id": project_id,
                "protocol_id": protocol_id,
                "created_at": utc_now(),
            },
        )
        # The initialization lock makes this a consistency assertion rather
        # than a last-writer-wins race when two identities open an empty root.
        raw = self._read_json(self._manifest_path, "workspace manifest")
        observed = (raw.get("project_id"), raw.get("protocol_id"))
        expected = (project_id, protocol_id)
        if observed != expected:
            raise WorkspaceError(
                "workspace identity differs after initialization: "
                f"expected {expected!r}, observed {observed!r}"
            )

    @staticmethod
    def _read_json(path: Path, label: str) -> Mapping[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkspaceError(f"cannot read {label}: {path}") from exc
        if not isinstance(value, Mapping):
            raise WorkspaceError(f"{label} must be a JSON object: {path}")
        return value

    def task_dir(self, task: TaskKey) -> Path:
        if task.stage is StageId.ADMIT:
            assert task.case_id is not None
            return self.root / "cases" / task.case_id / "admit"
        if task.stage in {
            StageId.MD,
            StageId.TRAJECTORY,
            StageId.TRAJECTORY_QC,
        }:
            assert task.case_id is not None and task.replica_id is not None
            return (
                self.root
                / "cases"
                / task.case_id
                / "replicas"
                / task.replica_id
                / task.stage.value
            )
        if task.stage is StageId.SCORING:
            assert (
                task.case_id is not None
                and task.replica_id is not None
                and task.model_id is not None
            )
            return (
                self.root
                / "cases"
                / task.case_id
                / "replicas"
                / task.replica_id
                / "scoring"
                / "models"
                / task.model_id
            )
        if task.stage is StageId.FINALIZE_CASE:
            assert task.case_id is not None
            return self.root / "cases" / task.case_id / "finalize_case"
        return self.root / "project" / task.stage.value

    def _attempt_dirs(self, task: TaskKey) -> tuple[Path, ...]:
        attempts = self.task_dir(task) / "attempts"
        if not attempts.is_dir():
            return ()
        result = [
            item
            for item in attempts.iterdir()
            if item.is_dir() and item.name.isdigit() and len(item.name) == 4
            and (item / "state.json").is_file()
        ]
        return tuple(sorted(result, key=lambda item: int(item.name)))

    def _observed_attempt_numbers(self, task: TaskKey) -> tuple[int, ...]:
        attempts = self.task_dir(task) / "attempts"
        if not attempts.is_dir():
            return ()
        return tuple(
            sorted(
                int(item.name)
                for item in attempts.iterdir()
                if item.is_dir()
                and item.name.isdigit()
                and len(item.name) == 4
            )
        )

    def latest(self, task: TaskKey) -> AttemptSummary | None:
        attempts = self._attempt_dirs(task)
        if not attempts:
            return None
        return self._read_summary(attempts[-1], expected_task=task)

    def attempts(self, task: TaskKey) -> tuple[AttemptSummary, ...]:
        return tuple(
            self._read_summary(path, expected_task=task)
            for path in self._attempt_dirs(task)
        )

    def prune_attempts(
        self,
        task: TaskKey,
        *,
        keep: int = 1,
        confirm: bool = False,
    ) -> tuple[PruneRecord, ...]:
        """Plan or delete old assets/report attempts under the task lock."""

        if task.stage not in {StageId.ASSETS, StageId.REPORT} or any(
            value is not None
            for value in (task.case_id, task.replica_id, task.model_id)
        ):
            raise WorkspaceError("pruning is limited to project assets/report stages")
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError("prune keep must be a positive integer")
        records: list[PruneRecord] = []
        with self._lock(task):
            attempts = self._attempt_dirs(task)
            candidates = attempts[:-keep] if len(attempts) > keep else ()
            attempts_root = (self.task_dir(task) / "attempts").resolve()
            for path in candidates:
                summary = self._read_summary(path, expected_task=task)
                if summary.state.active:
                    continue
                resolved = path.resolve()
                if resolved.parent != attempts_root or not (
                    resolved.name.isdigit() and len(resolved.name) == 4
                ):
                    raise WorkspaceError(f"refusing unsafe prune target: {resolved}")
                size = sum(
                    item.stat().st_size
                    for item in resolved.rglob("*")
                    if item.is_file() and not item.is_symlink()
                )
                if confirm:
                    shutil.rmtree(resolved)
                records.append(
                    PruneRecord(
                        task=task,
                        attempt_id=summary.attempt_id,
                        path=resolved,
                        size_bytes=size,
                        deleted=confirm,
                    )
                )
        return tuple(records)

    @staticmethod
    def _task_mapping(task: TaskKey) -> dict[str, object]:
        return {
            "stage": task.stage.value,
            "case_id": task.case_id,
            "replica_id": task.replica_id,
            "model_id": task.model_id,
        }

    @staticmethod
    def _task_from_mapping(value: Mapping[str, object]) -> TaskKey:
        expected = {"stage", "case_id", "replica_id", "model_id"}
        missing = expected.difference(value)
        unknown = set(value).difference(expected)
        if missing or unknown:
            detail: list[str] = []
            if missing:
                detail.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                detail.append("unknown " + ", ".join(sorted(unknown)))
            raise WorkspaceError(
                "attempt contains an invalid task key: " + "; ".join(detail)
            )
        try:
            return TaskKey(
                stage=StageId(str(value["stage"])),
                case_id=(
                    None
                    if value.get("case_id") is None
                    else str(value["case_id"])
                ),
                replica_id=(
                    None
                    if value.get("replica_id") is None
                    else str(value["replica_id"])
                ),
                model_id=(
                    None
                    if value.get("model_id") is None
                    else str(value["model_id"])
                ),
            )
        except (KeyError, ValueError, ContractError) as exc:
            raise WorkspaceError("attempt contains an invalid task key") from exc

    def _read_summary(
        self,
        attempt_dir: Path,
        *,
        expected_task: TaskKey | None = None,
    ) -> AttemptSummary:
        raw = self._read_json(attempt_dir / "state.json", "attempt state")
        if raw.get("schema_version") != ATTEMPT_SCHEMA:
            raise WorkspaceError(f"unsupported attempt schema: {attempt_dir}")
        task_raw = raw.get("task")
        if not isinstance(task_raw, Mapping):
            raise WorkspaceError(f"attempt task is invalid: {attempt_dir}")
        artifacts_raw = raw.get("artifacts", {})
        if not isinstance(artifacts_raw, Mapping):
            raise WorkspaceError(f"attempt artifacts are invalid: {attempt_dir}")
        if any(
            not isinstance(value, Mapping)
            for value in artifacts_raw.values()
        ):
            raise WorkspaceError(f"attempt artifacts are invalid: {attempt_dir}")
        try:
            artifacts = {
                str(role): ArtifactRef.from_mapping(value)
                for role, value in artifacts_raw.items()
            }
            if any(
                role != artifact.role
                for role, artifact in artifacts.items()
            ):
                raise WorkspaceError(
                    f"attempt artifact role keys differ: {attempt_dir}"
                )
            verdict = raw.get("qc_verdict")
            payload = raw.get("payload", {})
            if not isinstance(payload, Mapping):
                raise WorkspaceError(
                    f"attempt payload is invalid: {attempt_dir}"
                )
            observed_task = self._task_from_mapping(task_raw)
            attempt_id = str(raw["attempt_id"])
            if expected_task is not None and observed_task != expected_task:
                raise WorkspaceError(
                    "attempt task differs from its workspace path: "
                    f"expected {expected_task.label}, observed "
                    f"{observed_task.label}"
                )
            if attempt_id != attempt_dir.name:
                raise WorkspaceError(
                    "attempt ID differs from its workspace path: "
                    f"expected {attempt_dir.name}, observed {attempt_id}"
                )
            return AttemptSummary(
                task=observed_task,
                attempt_id=attempt_id,
                state=AttemptState(str(raw["state"])),
                artifact_state=ArtifactState(str(raw["artifact_state"])),
                science_hash=str(raw["science_hash"]),
                compatibility_key=str(raw["compatibility_key"]),
                artifacts=artifacts,
                qc_verdict=None if verdict is None else QcVerdict(str(verdict)),
                payload=payload,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise WorkspaceError(f"invalid attempt state: {attempt_dir}") from exc

    def _state_path(self, task: TaskKey, attempt_id: str) -> Path:
        return self.task_dir(task) / "attempts" / attempt_id / "state.json"

    def _raw_state(self, task: TaskKey, attempt_id: str) -> dict[str, object]:
        raw = self._read_json(
            self._state_path(task, attempt_id), "attempt state"
        )
        return dict(raw)

    def _write_state(
        self, task: TaskKey, attempt_id: str, value: Mapping[str, object]
    ) -> None:
        atomic_write_json(self._state_path(task, attempt_id), value)

    def _lock(self, task: TaskKey) -> FileLock:
        task_dir = self.task_dir(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        return FileLock(
            task_dir / ".lock",
            owner=f"mmpbsa:{task.label}",
            timeout_seconds=30,
            stale_after_seconds=24 * 60 * 60,
        )

    @staticmethod
    def _checkpoint_from_raw(
        attempt_dir: Path, raw: Mapping[str, object]
    ) -> Path | None:
        checkpoint = raw.get("checkpoint")
        if checkpoint is None:
            return None
        if not isinstance(checkpoint, Mapping):
            raise WorkspaceError("attempt checkpoint record is invalid")
        try:
            path = resolve_within(
                attempt_dir, Path(str(checkpoint["path"])), must_exist=True
            )
            expected_size = int(checkpoint["size"])
            expected_digest = str(checkpoint["sha256"])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ArtifactIntegrityError(
                f"attempt checkpoint is unavailable: {attempt_dir}"
            ) from exc
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_digest
        ):
            raise ArtifactIntegrityError(
                f"attempt checkpoint changed: {path}"
            )
        return path

    def reserve(
        self,
        task: TaskKey,
        *,
        action: ExecutionAction,
        science_hash: str,
        compatibility: BackendCompatibility,
    ) -> AttemptReservation:
        if action not in {"run", "resume", "retry", "new"}:
            raise WorkspaceError(f"unknown execution action: {action!r}")
        with self._lock(task):
            self._reconcile_task(task)
            latest = self.latest(task)
            if action == "run" and latest is not None:
                if latest.state.active:
                    raise ActiveAttemptError(
                        f"{task.label} has active attempt {latest.attempt_id}; "
                        "use explicit resume or --new"
                    )
                if latest.state is AttemptState.COMPLETED:
                    if (
                        latest.science_hash == science_hash
                        and latest.compatibility_key == compatibility.key
                    ):
                        self.verify_artifacts(latest)
                        return AttemptReservation(
                            summary=latest,
                            attempt_dir=(
                                self.task_dir(task)
                                / "attempts"
                                / latest.attempt_id
                            ),
                            invocation_dir=None,
                            reused=True,
                            resume=False,
                        )
                    raise RetryError(
                        f"{task.label} has a sealed completed attempt with "
                        "different science/backend compatibility; use --new"
                    )
                if latest.state is AttemptState.FAILED:
                    raise RetryError(
                        f"{task.label} last failed; use explicit retry or --new"
                    )
                raise RetryError(
                    f"{task.label} last attempt is {latest.state.value}; "
                    "use --new"
                )

            if action == "resume":
                if latest is None or not latest.state.active:
                    raise ResumeError(
                        f"{task.label} has no active attempt to resume"
                    )
                if latest.science_hash != science_hash:
                    raise ResumeError(
                        "science inputs changed; active attempt cannot be resumed"
                    )
                if latest.compatibility_key != compatibility.key:
                    raise ResumeError(
                        "backend major compatibility changed; active attempt "
                        "cannot be resumed"
                    )
                attempt_dir = (
                    self.task_dir(task) / "attempts" / latest.attempt_id
                )
                raw = self._raw_state(task, latest.attempt_id)
                if latest.state is AttemptState.RUNNING:
                    self._recover_abandoned_invocation(
                        task, raw, attempt_dir
                    )
                elif latest.state is not AttemptState.INTERRUPTED:
                    raise ResumeError(
                        f"{task.label} attempt {latest.attempt_id} is "
                        f"{latest.state.value}, not resumable"
                    )
                checkpoint = self._checkpoint_from_raw(attempt_dir, raw)
                return self._begin_invocation(
                    task,
                    raw,
                    attempt_dir,
                    resume=True,
                    checkpoint=checkpoint,
                    compatibility=compatibility,
                )

            if action == "retry":
                if latest is None or latest.state is not AttemptState.FAILED:
                    raise RetryError(
                        f"{task.label} has no failed attempt to retry"
                    )
                if latest.science_hash != science_hash:
                    raise RetryError(
                        "science inputs changed; use --new instead of retry"
                    )
                if latest.compatibility_key != compatibility.key:
                    raise RetryError(
                        "backend major compatibility changed; use --new "
                        "instead of retry"
                    )

            if action == "new" and latest is not None and latest.state.active:
                raw = self._raw_state(task, latest.attempt_id)
                attempt_dir = (
                    self.task_dir(task) / "attempts" / latest.attempt_id
                )
                if latest.state is AttemptState.RUNNING:
                    invocation = self._current_invocation(raw, attempt_dir)
                    if self._owner_is_alive(invocation.get("owner")):
                        raise ActiveAttemptError(
                            f"{task.label} is owned by a live process; "
                            "--new cannot create a concurrent attempt"
                        )
                    child_pids = self._processes_within(attempt_dir)
                    if child_pids:
                        raise ActiveAttemptError(
                            f"{task.label} still has live subprocesses in its "
                            f"attempt directory: "
                            f"{', '.join(str(pid) for pid in child_pids)}"
                        )
                raw["state"] = AttemptState.CANCELLED.value
                finished_at = utc_now()
                raw["finished_at"] = finished_at
                error = {
                    "type": "SupersededAttempt",
                    "message": "cancelled by explicit --new",
                }
                raw["error"] = error
                if latest.state is AttemptState.RUNNING:
                    raw["invocation_terminal"] = {
                        "outcome": "cancelled",
                        "finished_at": finished_at,
                        "provenance": {},
                        "error": error,
                    }
                self._write_state(task, latest.attempt_id, raw)
                self._sync_invocation_from_state(
                    task, attempt_dir, raw, tolerate_failure=True
                )

            return self._create_attempt(
                task,
                science_hash=science_hash,
                compatibility=compatibility,
            )

    def _create_attempt(
        self,
        task: TaskKey,
        *,
        science_hash: str,
        compatibility: BackendCompatibility,
    ) -> AttemptReservation:
        observed = self._observed_attempt_numbers(task)
        next_number = 1 if not observed else observed[-1] + 1
        attempt_id = f"{next_number:04d}"
        attempts_root = self.task_dir(task) / "attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        attempt_dir = attempts_root / attempt_id
        invocation_id = "0001"
        now = utc_now()
        raw: dict[str, object] = {
            "schema_version": ATTEMPT_SCHEMA,
            "attempt_id": attempt_id,
            "task": self._task_mapping(task),
            "state": AttemptState.RUNNING.value,
            "artifact_state": ArtifactState.MISSING.value,
            "science_hash": science_hash,
            "compatibility_key": compatibility.key,
            "compatibility": compatibility.as_dict(),
            "created_at": now,
            "started_at": now,
            "finished_at": None,
            "invocation_count": 1,
            "current_invocation_id": invocation_id,
            "invocation_terminal": None,
            "artifacts": {},
            "payload": {},
            "qc_verdict": None,
            "checkpoint": None,
            "error": None,
            "last_invocation_backend": compatibility.as_dict(),
        }
        with atomic_directory(attempt_dir) as staging:
            (staging / "output").mkdir()
            invocation_dir = staging / "invocations" / invocation_id
            invocation_dir.mkdir(parents=True)
            atomic_write_json(
                invocation_dir / "invocation.json",
                self._invocation_record(
                    invocation_id,
                    resume=False,
                    started_at=now,
                    compatibility=compatibility,
                ),
            )
            atomic_write_json(staging / "state.json", raw)
        published_invocation = attempt_dir / "invocations" / invocation_id
        return AttemptReservation(
            summary=self._read_summary(attempt_dir, expected_task=task),
            attempt_dir=attempt_dir,
            invocation_dir=published_invocation,
            reused=False,
            resume=False,
            checkpoint=None,
        )

    def _begin_invocation(
        self,
        task: TaskKey,
        raw: dict[str, object],
        attempt_dir: Path,
        *,
        resume: bool,
        checkpoint: Path | None,
        compatibility: BackendCompatibility,
    ) -> AttemptReservation:
        state = AttemptState(str(raw["state"]))
        if state.terminal:
            raise WorkspaceError("cannot append an invocation to a sealed attempt")
        invocation_root = attempt_dir / "invocations"
        self._reconcile_uncommitted_invocations(
            task, raw, attempt_dir
        )
        observed_numbers = [
            int(item.name)
            for item in invocation_root.iterdir()
            if item.is_dir()
            and item.name.isdigit()
            and len(item.name) == 4
        ]
        invocation_number = max(
            [int(raw.get("invocation_count", 0)), *observed_numbers]
        ) + 1
        invocation_id = f"{invocation_number:04d}"
        invocation_dir = invocation_root / invocation_id
        now = utc_now()
        with atomic_directory(invocation_dir) as staging:
            atomic_write_json(
                staging / "invocation.json",
                self._invocation_record(
                    invocation_id,
                    resume=resume,
                    started_at=now,
                    compatibility=compatibility,
                ),
            )
        raw["state"] = AttemptState.RUNNING.value
        raw["invocation_count"] = invocation_number
        raw["current_invocation_id"] = invocation_id
        raw["invocation_terminal"] = None
        raw["started_at"] = raw.get("started_at") or now
        raw["finished_at"] = None
        raw["error"] = None
        # Minor/patch changes are intentionally invocation provenance only.
        raw["last_invocation_backend"] = compatibility.as_dict()
        self._write_state(task, str(raw["attempt_id"]), raw)
        summary = self._read_summary(attempt_dir, expected_task=task)
        return AttemptReservation(
            summary=summary,
            attempt_dir=attempt_dir,
            invocation_dir=invocation_dir,
            reused=False,
            resume=resume,
            checkpoint=checkpoint,
        )

    @classmethod
    def _invocation_record(
        cls,
        invocation_id: str,
        *,
        resume: bool,
        started_at: str,
        compatibility: BackendCompatibility,
    ) -> dict[str, object]:
        return {
            "schema_version": INVOCATION_SCHEMA,
            "invocation_id": invocation_id,
            "resume": resume,
            "started_at": started_at,
            "finished_at": None,
            "outcome": "running",
            "backend": compatibility.as_dict(),
            "owner": cls._process_owner(),
            "runtime": None,
            "provenance": {},
            "pending_terminal": None,
            "error": None,
        }

    @staticmethod
    def _process_start_token(pid: int) -> str | None:
        """Return Linux process start ticks to protect against PID reuse."""

        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            return fields[19]
        except (OSError, IndexError):
            return None

    @classmethod
    def _process_owner(cls) -> dict[str, object]:
        pid = os.getpid()
        return {
            "hostname": socket.gethostname(),
            "pid": pid,
            "process_start_ticks": cls._process_start_token(pid),
        }

    @classmethod
    def _owner_is_alive(cls, value: object) -> bool:
        if not isinstance(value, Mapping):
            return True
        if value.get("hostname") != socket.gethostname():
            return True
        try:
            pid = int(value["pid"])
        except (KeyError, TypeError, ValueError):
            return True
        observed_start = cls._process_start_token(pid)
        if observed_start is None:
            return False
        expected_start = value.get("process_start_ticks")
        return expected_start is None or str(expected_start) == observed_start

    @staticmethod
    def _processes_within(root: Path) -> tuple[int, ...]:
        """Find live local subprocesses whose cwd is inside an attempt."""

        proc = Path("/proc")
        if not proc.is_dir():
            return ()
        expected = root.resolve()
        result: list[int] = []
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            try:
                cwd = (entry / "cwd").resolve(strict=True)
                cwd.relative_to(expected)
            except (OSError, ValueError):
                continue
            result.append(pid)
        return tuple(sorted(result))

    @staticmethod
    def _current_invocation_id(raw: Mapping[str, object]) -> str:
        value = raw.get("current_invocation_id")
        if value is not None:
            invocation_id = str(value)
        else:
            invocation_id = f"{int(raw.get('invocation_count', 0)):04d}"
        if not (
            invocation_id.isdigit()
            and len(invocation_id) == 4
            and invocation_id != "0000"
        ):
            raise WorkspaceError("attempt current invocation id is invalid")
        return invocation_id

    def _current_invocation(
        self, raw: Mapping[str, object], attempt_dir: Path
    ) -> dict[str, object]:
        invocation_id = self._current_invocation_id(raw)
        path = (
            attempt_dir
            / "invocations"
            / invocation_id
            / "invocation.json"
        )
        return dict(self._read_json(path, "current invocation"))

    def _reconcile_task(self, task: TaskKey) -> None:
        """Repair invocation ledgers from authoritative attempt snapshots."""

        for attempt_dir in self._attempt_dirs(task):
            raw = dict(
                self._read_json(attempt_dir / "state.json", "attempt state")
            )
            self._sync_invocation_from_state(
                task, attempt_dir, raw, tolerate_failure=False
            )

    def _reconcile_uncommitted_invocations(
        self,
        task: TaskKey,
        raw: Mapping[str, object],
        attempt_dir: Path,
    ) -> None:
        """Seal invocation directories published before their state commit."""

        committed_count = int(raw.get("invocation_count", 0))
        invocation_root = attempt_dir / "invocations"
        observed = sorted(
            item
            for item in invocation_root.iterdir()
            if item.is_dir()
            and item.name.isdigit()
            and len(item.name) == 4
            and int(item.name) > committed_count
        )
        for invocation_dir in observed:
            path = invocation_dir / "invocation.json"
            invocation = dict(
                self._read_json(path, "uncommitted invocation")
            )
            if invocation.get("outcome") != "running":
                continue
            if self._owner_is_alive(invocation.get("owner")):
                raise ActiveAttemptError(
                    f"{task.label} has live invocation "
                    f"{invocation_dir.name} whose state commit is incomplete"
                )
            now = utc_now()
            error = {
                "type": "AbandonedBeforeStateCommit",
                "message": (
                    "invocation directory was published but its attempt "
                    "state was not committed"
                ),
            }
            invocation["finished_at"] = now
            invocation["outcome"] = "abandoned_before_state_commit"
            invocation["pending_terminal"] = None
            invocation["error"] = error
            atomic_write_json(path, invocation)

    def _prepare_invocation_terminal(
        self,
        reservation: AttemptReservation,
        terminal: Mapping[str, object],
    ) -> None:
        assert reservation.invocation_dir is not None
        path = reservation.invocation_dir / "invocation.json"
        raw = dict(self._read_json(path, "invocation record"))
        if raw.get("outcome") != "running":
            raise WorkspaceError("only a running invocation can be committed")
        raw["pending_terminal"] = dict(terminal)
        atomic_write_json(path, raw)

    def _sync_invocation_from_state(
        self,
        task: TaskKey,
        attempt_dir: Path,
        raw: Mapping[str, object],
        *,
        tolerate_failure: bool,
    ) -> None:
        terminal = raw.get("invocation_terminal")
        if terminal is None:
            return
        if not isinstance(terminal, Mapping):
            raise WorkspaceError(
                f"{task.label} invocation terminal record is invalid"
            )
        try:
            invocation_id = self._current_invocation_id(raw)
            path = (
                attempt_dir
                / "invocations"
                / invocation_id
                / "invocation.json"
            )
            invocation = dict(
                self._read_json(path, "invocation record")
            )
            invocation["finished_at"] = terminal.get("finished_at")
            invocation["outcome"] = terminal.get("outcome")
            provenance = terminal.get("provenance", {})
            error = terminal.get("error")
            if not isinstance(provenance, Mapping):
                raise WorkspaceError(
                    f"{task.label} invocation provenance is invalid"
                )
            if error is not None and not isinstance(error, Mapping):
                raise WorkspaceError(
                    f"{task.label} invocation error is invalid"
                )
            invocation["provenance"] = dict(provenance)
            invocation["error"] = None if error is None else dict(error)
            invocation["pending_terminal"] = None
            atomic_write_json(path, invocation)
        except Exception:
            if not tolerate_failure:
                raise

    def _recover_abandoned_invocation(
        self,
        task: TaskKey,
        raw: dict[str, object],
        attempt_dir: Path,
    ) -> None:
        invocation_id = self._current_invocation_id(raw)
        invocation = self._current_invocation(raw, attempt_dir)
        if invocation.get("outcome") != "running":
            raise WorkspaceError(
                f"{task.label} has inconsistent running attempt state"
            )
        if self._owner_is_alive(invocation.get("owner")):
            raise ActiveAttemptError(
                f"{task.label} invocation {invocation_id} is still running; "
                "resume would write the same attempt concurrently"
            )
        child_pids = self._processes_within(attempt_dir)
        if child_pids:
            raise ActiveAttemptError(
                f"{task.label} still has live subprocesses in its attempt "
                f"directory: {', '.join(str(pid) for pid in child_pids)}"
            )
        now = utc_now()
        error = {
            "type": "AbandonedInvocation",
            "message": "recorded owner process is no longer alive",
        }
        raw["state"] = AttemptState.INTERRUPTED.value
        raw["finished_at"] = None
        raw["error"] = {
            "type": "AbandonedInvocation",
            "message": "previous invocation owner is no longer alive",
        }
        raw["current_invocation_id"] = invocation_id
        raw["invocation_terminal"] = {
            "outcome": "abandoned",
            "finished_at": now,
            "provenance": {},
            "error": error,
        }
        self._write_state(task, str(raw["attempt_id"]), raw)
        self._sync_invocation_from_state(
            task, attempt_dir, raw, tolerate_failure=False
        )

    def record_invocation_runtime(
        self,
        reservation: AttemptReservation,
        provenance: Mapping[str, object],
    ) -> None:
        """Persist the effective, secret-safe runtime identity before execution."""

        if reservation.reused or reservation.invocation_dir is None:
            raise WorkspaceError("cannot record runtime for a reused attempt")
        task = reservation.summary.task
        with self._lock(task):
            path = reservation.invocation_dir / "invocation.json"
            raw = dict(self._read_json(path, "invocation record"))
            if raw.get("outcome") != "running":
                raise WorkspaceError(
                    "runtime can only be recorded for a running invocation"
                )
            raw["runtime"] = dict(provenance)
            atomic_write_json(path, raw)

    @staticmethod
    def _infer_media_type(path: Path) -> str:
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"

    @staticmethod
    def _artifact_ref(
        attempt_dir: Path, role: str, path: Path
    ) -> ArtifactRef:
        try:
            candidate = Path(path)
            member = (
                candidate.resolve(strict=True).relative_to(attempt_dir.resolve())
                if candidate.is_absolute()
                else candidate
            )
            resolved = resolve_within(attempt_dir, member, must_exist=True)
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"stage output escapes attempt directory: {path}"
            ) from exc
        if not resolved.is_file() or resolved.stat().st_size < 1:
            raise ArtifactIntegrityError(
                f"stage output is missing or empty for {role!r}: {resolved}"
            )
        return ArtifactRef(
            path=resolved.relative_to(attempt_dir),
            sha256=sha256_file(resolved),
            size=resolved.stat().st_size,
            role=role,
            media_type=Workspace._infer_media_type(resolved),
        )

    @staticmethod
    def _checkpoint_ref(
        attempt_dir: Path, checkpoint: Path | None
    ) -> Mapping[str, object] | None:
        if checkpoint is None:
            return None
        try:
            candidate = Path(checkpoint)
            member = (
                candidate.resolve(strict=True).relative_to(attempt_dir.resolve())
                if candidate.is_absolute()
                else candidate
            )
            resolved = resolve_within(
                attempt_dir, member, must_exist=True
            )
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"checkpoint escapes attempt directory: {checkpoint}"
            ) from exc
        if not resolved.is_file() or resolved.stat().st_size < 1:
            raise ArtifactIntegrityError(
                f"checkpoint is missing or empty: {resolved}"
            )
        return {
            "path": resolved.relative_to(attempt_dir).as_posix(),
            "sha256": sha256_file(resolved),
            "size": resolved.stat().st_size,
        }

    def complete(
        self,
        reservation: AttemptReservation,
        result: StageResult,
    ) -> AttemptSummary:
        if reservation.reused or reservation.invocation_dir is None:
            raise WorkspaceError("cannot complete a reused attempt")
        task = reservation.summary.task
        with self._lock(task):
            raw = self._raw_state(task, reservation.summary.attempt_id)
            if AttemptState(str(raw["state"])) is not AttemptState.RUNNING:
                raise WorkspaceError("only a running attempt can complete")
            artifacts = {
                str(role): self._artifact_ref(
                    reservation.attempt_dir, str(role), Path(path)
                )
                for role, path in result.artifacts.items()
            }
            if not artifacts:
                raise ArtifactIntegrityError(
                    f"{task.label} returned no output artifacts"
                )
            raw["artifacts"] = {
                role: artifact.as_dict()
                for role, artifact in sorted(artifacts.items())
            }
            raw["artifact_state"] = ArtifactState.COMPLETE.value
            raw["payload"] = dict(result.payload)
            raw["qc_verdict"] = (
                None
                if result.qc_verdict is None
                else result.qc_verdict.value
            )
            raw["checkpoint"] = self._checkpoint_ref(
                reservation.attempt_dir, result.checkpoint
            )
            finished_at = utc_now()
            terminal = {
                "outcome": "completed",
                "finished_at": finished_at,
                "provenance": dict(result.invocation_provenance),
                "error": None,
            }
            self._prepare_invocation_terminal(reservation, terminal)
            raw["state"] = AttemptState.COMPLETED.value
            raw["finished_at"] = finished_at
            raw["error"] = None
            raw["current_invocation_id"] = reservation.invocation_dir.name
            raw["invocation_terminal"] = terminal
            self._write_state(task, reservation.summary.attempt_id, raw)
            self._sync_invocation_from_state(
                task,
                reservation.attempt_dir,
                raw,
                tolerate_failure=True,
            )
            return self._read_summary(
                reservation.attempt_dir,
                expected_task=reservation.summary.task,
            )

    def fail(
        self,
        reservation: AttemptReservation,
        error: BaseException,
        *,
        interrupted: bool = False,
        checkpoint: Path | None = None,
    ) -> AttemptSummary:
        if reservation.reused or reservation.invocation_dir is None:
            raise WorkspaceError("cannot fail a reused attempt")
        task = reservation.summary.task
        with self._lock(task):
            raw = self._raw_state(task, reservation.summary.attempt_id)
            state = AttemptState(str(raw["state"]))
            if state is not AttemptState.RUNNING:
                return self._read_summary(
                    reservation.attempt_dir,
                    expected_task=reservation.summary.task,
                )
            raw["state"] = (
                AttemptState.INTERRUPTED.value
                if interrupted
                else AttemptState.FAILED.value
            )
            raw["artifact_state"] = ArtifactState.PARTIAL.value
            finished_at = utc_now()
            raw["finished_at"] = None if interrupted else finished_at
            raw["checkpoint"] = self._checkpoint_ref(
                reservation.attempt_dir, checkpoint
            )
            error_record = {
                "type": type(error).__name__,
                "message": str(error),
            }
            raw["error"] = error_record
            terminal = {
                "outcome": "interrupted" if interrupted else "failed",
                "finished_at": finished_at,
                "provenance": {},
                "error": error_record,
            }
            self._prepare_invocation_terminal(reservation, terminal)
            raw["current_invocation_id"] = reservation.invocation_dir.name
            raw["invocation_terminal"] = terminal
            self._write_state(task, reservation.summary.attempt_id, raw)
            self._sync_invocation_from_state(
                task,
                reservation.attempt_dir,
                raw,
                tolerate_failure=True,
            )
            return self._read_summary(
                reservation.attempt_dir,
                expected_task=reservation.summary.task,
            )

    def verify_artifacts(self, summary: AttemptSummary) -> Mapping[str, Path]:
        attempt_dir = (
            self.task_dir(summary.task) / "attempts" / summary.attempt_id
        )
        resolved: dict[str, Path] = {}
        for role, artifact in summary.artifacts.items():
            try:
                path = resolve_within(
                    attempt_dir, artifact.path, must_exist=True
                )
            except (OSError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    f"attempt artifact is unavailable: {artifact.path}"
                ) from exc
            if (
                not path.is_file()
                or path.stat().st_size != artifact.size
                or sha256_file(path) != artifact.sha256
            ):
                raise ArtifactIntegrityError(
                    f"attempt artifact changed for {role!r}: {path}"
                )
            resolved[role] = path
        if summary.state is AttemptState.COMPLETED and not resolved:
            raise ArtifactIntegrityError(
                f"completed attempt has no artifacts: {summary.task.label}"
            )
        return resolved

    def as_stage_input(self, summary: AttemptSummary) -> StageInput:
        if summary.state is not AttemptState.COMPLETED:
            raise WorkspaceError(
                f"upstream attempt is not completed: {summary.task.label}"
            )
        attempt_dir = (
            self.task_dir(summary.task) / "attempts" / summary.attempt_id
        )
        return StageInput(
            task=summary.task,
            attempt_id=summary.attempt_id,
            root=attempt_dir,
            artifacts=self.verify_artifacts(summary),
            science_hash=summary.science_hash,
            payload=summary.payload,
            qc_verdict=summary.qc_verdict,
        )

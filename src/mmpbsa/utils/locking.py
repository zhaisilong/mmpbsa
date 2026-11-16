"""Portable lock-file ownership with explicit stale-lock inspection."""

from __future__ import annotations

import errno
import json
import math
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from .errors import LockOwnershipError, LockTimeoutError
from .filesystem import atomic_write_json
from .serialization import jsonable
from .time import new_id, utc_now

_SCHEMA = "mmpbsa.utils.file-lock.v1"


@dataclass(frozen=True)
class LockMetadata:
    """Ownership metadata stored inside an acquired lock file."""

    owner: str
    token: str
    pid: int
    hostname: str
    acquired_at: str
    acquired_epoch: float
    process_identity: str | None
    metadata: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA,
            "owner": self.owner,
            "token": self.token,
            "pid": self.pid,
            "hostname": self.hostname,
            "acquired_at": self.acquired_at,
            "acquired_epoch": self.acquired_epoch,
            "process_identity": self.process_identity,
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> LockMetadata:
        if record.get("schema_version") != _SCHEMA:
            raise ValueError("Unsupported lock metadata schema")
        owner = record.get("owner")
        token = record.get("token")
        pid = record.get("pid")
        hostname = record.get("hostname")
        acquired_at = record.get("acquired_at")
        acquired_epoch = record.get("acquired_epoch")
        process_identity = record.get("process_identity")
        metadata = record.get("metadata", {})
        if not isinstance(owner, str) or not owner:
            raise ValueError("Lock owner must be a non-empty string")
        if not isinstance(token, str) or not token:
            raise ValueError("Lock token must be a non-empty string")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("Lock PID must be a positive integer")
        if not isinstance(hostname, str) or not hostname:
            raise ValueError("Lock hostname must be a non-empty string")
        if not isinstance(acquired_at, str) or not acquired_at:
            raise ValueError("Lock acquisition timestamp is missing")
        if isinstance(acquired_epoch, bool) or not isinstance(
            acquired_epoch, (int, float)
        ):
            raise ValueError("Lock acquisition epoch is invalid")
        if process_identity is not None and not isinstance(process_identity, str):
            raise ValueError("Lock process identity is invalid")
        if not isinstance(metadata, Mapping):
            raise ValueError("Lock custom metadata must be a mapping")
        return cls(
            owner=owner,
            token=token,
            pid=pid,
            hostname=hostname,
            acquired_at=acquired_at,
            acquired_epoch=float(acquired_epoch),
            process_identity=process_identity,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class LockInspection:
    """A snapshot describing an existing, absent, or stale lock."""

    path: Path
    exists: bool
    metadata: LockMetadata | None
    age_seconds: float | None
    stale: bool
    reason: str | None
    inode: int | None
    modified_ns: int | None


def inspect_file_lock(
    path: str | PurePath,
    *,
    stale_after_seconds: float | None = None,
) -> LockInspection:
    """Inspect lock metadata and conservatively determine whether it is stale."""

    threshold = _validate_nonnegative_optional(
        stale_after_seconds, "stale_after_seconds"
    )
    lock_path = Path(path)
    try:
        stat = lock_path.stat()
    except FileNotFoundError:
        return LockInspection(lock_path, False, None, None, False, None, None, None)

    now = time.time()
    file_age = max(0.0, now - stat.st_mtime)
    try:
        record = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("Lock metadata root is not an object")
        metadata = LockMetadata.from_record(record)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        stale = threshold is not None and file_age >= threshold
        return LockInspection(
            lock_path,
            True,
            None,
            file_age,
            stale,
            "malformed_metadata" if stale else "metadata_unreadable",
            stat.st_ino,
            stat.st_mtime_ns,
        )

    age = max(0.0, now - metadata.acquired_epoch)
    local_host = socket.gethostname()
    if metadata.hostname == local_host:
        alive = _process_is_owner(metadata.pid, metadata.process_identity)
        if not alive:
            return LockInspection(
                lock_path,
                True,
                metadata,
                age,
                True,
                "owner_process_not_running",
                stat.st_ino,
                stat.st_mtime_ns,
            )
        reason = "owner_process_running"
    else:
        reason = "owner_host_not_local"

    if (
        metadata.hostname != local_host
        and threshold is not None
        and age >= threshold
    ):
        return LockInspection(
            lock_path,
            True,
            metadata,
            age,
            True,
            "remote_lock_age_exceeded",
            stat.st_ino,
            stat.st_mtime_ns,
        )
    return LockInspection(
        lock_path,
        True,
        metadata,
        age,
        False,
        reason,
        stat.st_ino,
        stat.st_mtime_ns,
    )


class FileLock:
    """An exclusive interprocess lock represented by an owned JSON file.

    Acquisition uses exclusive file creation. Dead local owners are reclaimed;
    malformed and remote locks are reclaimed only after ``stale_after_seconds``.
    A remote owner should refresh its lock before that lease duration elapses.
    """

    def __init__(
        self,
        path: str | PurePath,
        *,
        owner: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = 0,
        poll_interval_seconds: float = 0.1,
        stale_after_seconds: float | None = None,
        break_stale: bool = True,
    ) -> None:
        self.path = Path(path)
        self.owner = owner or f"pid-{os.getpid()}"
        if not isinstance(self.owner, str) or not self.owner.strip():
            raise ValueError("owner must be a non-empty string")
        converted_metadata = jsonable(dict(metadata or {}))
        if not isinstance(converted_metadata, dict):
            raise TypeError("metadata must be a mapping")
        self.custom_metadata = converted_metadata
        self.timeout_seconds = _validate_nonnegative_optional(
            timeout_seconds, "timeout_seconds"
        )
        self.poll_interval_seconds = _validate_positive(
            poll_interval_seconds, "poll_interval_seconds"
        )
        self.stale_after_seconds = _validate_nonnegative_optional(
            stale_after_seconds, "stale_after_seconds"
        )
        self.break_stale = bool(break_stale)
        self._metadata: LockMetadata | None = None

    @property
    def held(self) -> bool:
        return self._metadata is not None

    @property
    def lock_metadata(self) -> LockMetadata | None:
        return self._metadata

    @property
    def metadata(self) -> LockMetadata | None:
        """Return current ownership metadata, or ``None`` while not held."""

        return self._metadata

    def acquire(self) -> FileLock:
        """Acquire the lock or raise :class:`LockTimeoutError`."""

        if self.held:
            raise LockOwnershipError(f"Lock is already held by this object: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = (
            None
            if self.timeout_seconds is None
            else time.monotonic() + self.timeout_seconds
        )
        while True:
            metadata = LockMetadata(
                owner=self.owner,
                token=new_id("lock"),
                pid=os.getpid(),
                hostname=socket.gethostname(),
                acquired_at=utc_now(),
                acquired_epoch=time.time(),
                process_identity=_process_identity(os.getpid()),
                metadata=self.custom_metadata,
            )
            if not _create_lock_file(self.path, metadata):
                inspection = inspect_file_lock(
                    self.path,
                    stale_after_seconds=self.stale_after_seconds,
                )
                if (
                    self.break_stale
                    and inspection.stale
                    and _unlink_snapshot(inspection)
                ):
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    owner = (
                        inspection.metadata.owner
                        if inspection.metadata is not None
                        else "unknown"
                    )
                    raise LockTimeoutError(
                        f"Could not acquire lock {self.path}; held by {owner}"
                    ) from None
                remaining = (
                    self.poll_interval_seconds
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                )
                time.sleep(min(self.poll_interval_seconds, remaining))
                continue

            self._metadata = metadata
            return self

    def refresh(self) -> None:
        """Refresh the lease timestamp after confirming ownership."""

        current = self._require_owned()
        refreshed = LockMetadata(
            owner=current.owner,
            token=current.token,
            pid=current.pid,
            hostname=current.hostname,
            acquired_at=current.acquired_at,
            acquired_epoch=time.time(),
            process_identity=current.process_identity,
            metadata=current.metadata,
        )
        atomic_write_json(self.path, refreshed.to_record())
        self._metadata = refreshed

    def release(self) -> None:
        """Release this object's lock after confirming its ownership token."""

        self._require_owned()
        try:
            self.path.unlink()
        except FileNotFoundError as error:
            self._metadata = None
            raise LockOwnershipError(
                f"Lock disappeared before release: {self.path}"
            ) from error
        self._metadata = None

    def _require_owned(self) -> LockMetadata:
        expected = self._metadata
        if expected is None:
            raise LockOwnershipError(f"Lock is not held by this object: {self.path}")
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
            actual = LockMetadata.from_record(record)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise LockOwnershipError(
                f"Cannot verify ownership of lock: {self.path}"
            ) from error
        if actual.token != expected.token:
            raise LockOwnershipError(
                f"Lock owner changed before operation: {self.path}"
            )
        return expected

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(self, *_: object) -> bool:
        self.release()
        return False


def _unlink_snapshot(inspection: LockInspection) -> bool:
    if not inspection.exists or not inspection.stale:
        return False
    try:
        current = inspection.path.stat()
    except FileNotFoundError:
        return True
    if (
        current.st_ino != inspection.inode
        or current.st_mtime_ns != inspection.modified_ns
    ):
        return False
    try:
        inspection.path.unlink()
    except FileNotFoundError:
        pass
    return True


def _process_is_owner(pid: int, expected_identity: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        return error.errno == errno.EPERM
    if expected_identity is None:
        return True
    actual_identity = _process_identity(pid)
    return actual_identity is None or actual_identity == expected_identity


def _process_identity(pid: int) -> str | None:
    """Return Linux process start ticks to guard against PID reuse."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = stat[closing_parenthesis + 2 :].split()
    # /proc/<pid>/stat field 22; the slice begins at field 3.
    return fields[19] if len(fields) > 19 else None


def _validate_nonnegative_optional(
    value: float | None,
    label: str,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be non-negative or None")
    return float(value)


def _validate_positive(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be positive")
    return float(value)


InterProcessFileLock = FileLock


def _create_lock_file(path: Path, metadata: LockMetadata) -> bool:
    """Publish complete metadata with no partially written lock-file window."""

    pending = path.parent / f".{path.name}.{metadata.token}.pending"
    atomic_write_json(pending, metadata.to_record())
    try:
        try:
            os.link(pending, path)
        except FileExistsError:
            return False
        return True
    finally:
        pending.unlink(missing_ok=True)

"""Public dependency-free helpers for the MMPBSA package family."""

from .errors import (
    AtomicReplaceError,
    CommandError,
    JsonEncodingError,
    LockError,
    LockOwnershipError,
    LockTimeoutError,
    MmpbsaUtilsError,
    UnsafePathError,
)
from .filesystem import (
    atomic_directory,
    atomic_replace_directory,
    atomic_write_json,
    atomic_write_text,
    materialize_immutable_file,
    resolve_within,
    safe_relative_path,
)
from .locking import (
    FileLock,
    InterProcessFileLock,
    LockInspection,
    LockMetadata,
    inspect_file_lock,
)
from .report_audit import (
    LinkAuditError,
    LinkAuditIssue,
    audit_local_links,
    audit_portable_manifests,
    collect_link_issues,
)
from .serialization import (
    canonical_json,
    canonical_json_bytes,
    file_sha256,
    jsonable,
    sha256_bytes,
    sha256_digest,
    sha256_file,
)
from .subprocess import (
    CommandExecutionError,
    CommandReceipt,
    CommandResult,
    CommandStartError,
    CommandTimeoutError,
    probe_version,
    run_command,
)
from .time import new_id, utc_now, utc_now_datetime

__version__ = "0.7.0"

__all__ = [
    "AtomicReplaceError",
    "CommandError",
    "CommandExecutionError",
    "CommandReceipt",
    "CommandResult",
    "CommandStartError",
    "CommandTimeoutError",
    "FileLock",
    "InterProcessFileLock",
    "JsonEncodingError",
    "LockError",
    "LockInspection",
    "LockMetadata",
    "LockOwnershipError",
    "LockTimeoutError",
    "LinkAuditError",
    "LinkAuditIssue",
    "MmpbsaUtilsError",
    "UnsafePathError",
    "atomic_directory",
    "atomic_replace_directory",
    "atomic_write_json",
    "atomic_write_text",
    "audit_local_links",
    "audit_portable_manifests",
    "materialize_immutable_file",
    "canonical_json",
    "canonical_json_bytes",
    "collect_link_issues",
    "file_sha256",
    "inspect_file_lock",
    "jsonable",
    "new_id",
    "probe_version",
    "resolve_within",
    "run_command",
    "safe_relative_path",
    "sha256_bytes",
    "sha256_digest",
    "sha256_file",
    "utc_now",
    "utc_now_datetime",
]

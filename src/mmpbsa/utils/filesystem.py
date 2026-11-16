"""Safe paths and transactional filesystem publication."""

from __future__ import annotations

import codecs
import ctypes
import fcntl
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePath
from typing import Any

from .errors import AtomicReplaceError, UnsafePathError
from .serialization import jsonable, sha256_file

_FICLONE = 0x40049409
_AT_FDCWD = -100
_RENAME_EXCHANGE = 0x2


def safe_relative_path(member: str | PurePath) -> Path:
    """Validate and return a non-absolute path without parent traversal."""

    path = Path(member)
    if path.is_absolute():
        raise UnsafePathError(f"Path must be relative: {member!s}")
    if any(part == ".." for part in path.parts):
        raise UnsafePathError(f"Path cannot contain parent traversal: {member!s}")
    return path


def resolve_within(
    root: str | PurePath,
    member: str | PurePath,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve ``member`` beneath ``root``, rejecting lexical and symlink escapes."""

    relative = safe_relative_path(member)
    root_path = Path(root).resolve(strict=must_exist)
    try:
        candidate = (root_path / relative).resolve(strict=must_exist)
    except (FileNotFoundError, RuntimeError) as error:
        if isinstance(error, RuntimeError):
            raise UnsafePathError(
                f"Path cannot be resolved safely beneath {root_path}: {member!s}"
            ) from error
        raise
    try:
        candidate.relative_to(root_path)
    except ValueError as error:
        raise UnsafePathError(
            f"Path resolves outside root {root_path}: {member!s}"
        ) from error
    return candidate


def atomic_write_text(
    path: str | PurePath,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
) -> Path:
    """Durably write text and atomically replace a file in the same directory."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    codecs.lookup(encoding)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0:
                raise ValueError("mode must be a non-negative integer")
            os.chmod(temporary, mode)
        elif target.is_file():
            os.chmod(temporary, target.stat().st_mode & 0o7777)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(
    path: str | PurePath,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    trailing_newline: bool = True,
) -> Path:
    """Serialize strict JSON and atomically replace ``path``."""

    text = json.dumps(
        jsonable(payload),
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=sort_keys,
        separators=(",", ":") if indent is None else None,
    )
    if trailing_newline:
        text += "\n"
    return atomic_write_text(path, text)


def materialize_immutable_file(
    source: str | PurePath,
    destination: str | PurePath,
    *,
    allow_hardlink: bool = False,
) -> Path:
    """Materialize an immutable file with verified clone/link/copy fallback.

    Reflinks are preferred because they are space-efficient and remain
    independently mutable. Hard links are only attempted when the caller
    explicitly confirms that both locations are immutable internal artifacts.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file() or source_path.is_symlink():
        raise FileNotFoundError(f"Source must be a regular file: {source_path}")
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    materialized = False
    try:
        try:
            with source_path.open("rb") as source_handle, destination_path.open(
                "xb"
            ) as destination_handle:
                fcntl.ioctl(destination_handle.fileno(), _FICLONE, source_handle.fileno())
            shutil.copystat(source_path, destination_path, follow_symlinks=False)
            materialized = True
        except OSError:
            destination_path.unlink(missing_ok=True)

        if not materialized and allow_hardlink:
            try:
                os.link(source_path, destination_path, follow_symlinks=False)
                materialized = True
            except OSError:
                destination_path.unlink(missing_ok=True)

        if not materialized:
            shutil.copy2(source_path, destination_path, follow_symlinks=False)

        if (
            destination_path.stat().st_size != source_path.stat().st_size
            or sha256_file(destination_path) != sha256_file(source_path)
        ):
            raise OSError(
                f"Materialized file failed integrity verification: {destination_path}"
            )
    except BaseException:
        destination_path.unlink(missing_ok=True)
        raise
    return destination_path


def atomic_replace_directory(
    staging: str | PurePath,
    target: str | PurePath,
) -> Path:
    """Install a complete tree without making an existing target disappear.

    ``staging`` is consumed on success. Both paths must be distinct, non-symlink
    directories on the same filesystem. Callers coordinating concurrent
    publishers must hold a lock around this operation. Replacing an existing
    target requires Linux ``renameat2(RENAME_EXCHANGE)``; unsupported systems
    fail without moving the old target.
    """

    staging_path = Path(staging).absolute()
    target_path = Path(target).absolute()
    if staging_path == target_path:
        raise AtomicReplaceError("Staging and target directories must differ")
    if not staging_path.is_dir() or staging_path.is_symlink():
        raise AtomicReplaceError(
            f"Staging path must be a real directory: {staging_path}"
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.is_symlink() or (
        target_path.exists() and not target_path.is_dir()
    ):
        raise AtomicReplaceError(
            f"Target path must be absent or a real directory: {target_path}"
        )
    if _paths_overlap(staging_path, target_path):
        raise AtomicReplaceError("Staging and target directories cannot be nested")
    if staging_path.stat().st_dev != target_path.parent.stat().st_dev:
        raise AtomicReplaceError(
            "Staging and target must reside on the same filesystem"
        )

    if target_path.exists():
        _exchange_directories(staging_path, target_path)
        _fsync_directory(target_path.parent)
        try:
            shutil.rmtree(staging_path)
            _fsync_directory(target_path.parent)
        except OSError as error:
            raise AtomicReplaceError(
                f"Published {target_path}, but could not remove previous tree "
                f"at {staging_path}"
            ) from error
    else:
        try:
            os.replace(staging_path, target_path)
            _fsync_directory(target_path.parent)
        except OSError as error:
            raise AtomicReplaceError(
                f"Failed to publish staging directory {staging_path} "
                f"to {target_path}"
            ) from error
    return target_path


def _exchange_directories(left: Path, right: Path) -> None:
    """Atomically exchange two existing directory names on Linux."""

    if sys.platform != "linux":
        raise AtomicReplaceError(
            "Replacing an existing directory without a reader gap requires "
            "Linux renameat2(RENAME_EXCHANGE)"
        )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise AtomicReplaceError(
            "Linux renameat2(RENAME_EXCHANGE) is unavailable; old target was preserved"
        ) from error
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AtomicReplaceError(
            "Linux renameat2(RENAME_EXCHANGE) is unavailable; old target was preserved"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        _AT_FDCWD,
        os.fsencode(left),
        _AT_FDCWD,
        os.fsencode(right),
        _RENAME_EXCHANGE,
    ) != 0:
        error_number = ctypes.get_errno()
        raise AtomicReplaceError(
            error_number,
            f"Could not exchange directories {left} and {right}: "
            f"{os.strerror(error_number)}; old target was preserved",
        )


@contextmanager
def atomic_directory(target: str | PurePath) -> Iterator[Path]:
    """Yield an empty sibling staging directory and publish it on success."""

    target_path = Path(target).absolute()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target_path.name}.staging-",
            dir=target_path.parent,
        )
    )
    try:
        yield staging
        atomic_replace_directory(staging, target_path)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)

"""Shell-free subprocess execution with serializable receipts."""

from __future__ import annotations

import math
import os
import subprocess as _subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import IO, Any

from .errors import CommandError
from .time import new_id, utc_now


@dataclass(frozen=True)
class CommandReceipt:
    """Immutable record of a command at the moment it starts."""

    command_id: str
    argv: tuple[str, ...]
    cwd: Path
    started_at: str
    environment_keys: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": "mmpbsa.utils.command-receipt.v1",
            "command_id": self.command_id,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "started_at": self.started_at,
            "environment_keys": list(self.environment_keys),
        }


@dataclass(frozen=True)
class CommandResult:
    """Completed command result and captured-output locations."""

    receipt: CommandReceipt
    returncode: int
    finished_at: str
    duration_seconds: float
    stdout: str | None
    stderr: str | None
    stdout_path: Path | None
    stderr_path: Path | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def argv(self) -> tuple[str, ...]:
        return self.receipt.argv

    @property
    def cwd(self) -> Path:
        return self.receipt.cwd

    @property
    def started_at(self) -> str:
        return self.receipt.started_at

    @property
    def log_path(self) -> Path | None:
        if (
            self.stdout_path is not None
            and self.stdout_path == self.stderr_path
        ):
            return self.stdout_path
        return self.stdout_path

    def to_record(self, *, include_output: bool = False) -> dict[str, Any]:
        record = self.receipt.to_record()
        record.update(
            {
                "schema_version": "mmpbsa.utils.command-result.v1",
                "returncode": self.returncode,
                "finished_at": self.finished_at,
                "duration_seconds": self.duration_seconds,
                "stdout_path": (
                    str(self.stdout_path) if self.stdout_path is not None else None
                ),
                "stderr_path": (
                    str(self.stderr_path) if self.stderr_path is not None else None
                ),
            }
        )
        if include_output:
            record["stdout"] = self.stdout
            record["stderr"] = self.stderr
        return record


class CommandExecutionError(CommandError):
    """A command completed with a non-zero status."""

    def __init__(self, result: CommandResult) -> None:
        self.result = result
        super().__init__(
            f"Command {result.receipt.command_id} exited with status "
            f"{result.returncode}: {list(result.argv)!r}"
        )


class CommandTimeoutError(CommandError, TimeoutError):
    """A command exceeded its configured timeout."""

    def __init__(
        self,
        receipt: CommandReceipt,
        timeout_seconds: float,
        *,
        stdout: str | bytes | None = None,
        stderr: str | bytes | None = None,
    ) -> None:
        self.receipt = receipt
        self.timeout_seconds = timeout_seconds
        self.stdout = _decode_output(stdout)
        self.stderr = _decode_output(stderr)
        super().__init__(
            f"Command {receipt.command_id} exceeded {timeout_seconds} seconds: "
            f"{list(receipt.argv)!r}"
        )


class CommandStartError(CommandError):
    """The operating system could not start a command."""

    def __init__(self, receipt: CommandReceipt, error: OSError) -> None:
        self.receipt = receipt
        self.os_error = error
        super().__init__(
            f"Could not start command {receipt.command_id}: "
            f"{list(receipt.argv)!r}: {error}"
        )


def run_command(
    argv: Sequence[str | PurePath],
    *,
    cwd: str | PurePath | None = None,
    environment: Mapping[str, str] | None = None,
    inherit_environment: bool = True,
    stdin: str | None = None,
    timeout_seconds: float | None = None,
    check: bool = True,
    stdout_path: str | PurePath | None = None,
    stderr_path: str | PurePath | None = None,
    log_path: str | PurePath | None = None,
    command_id: str | None = None,
) -> CommandResult:
    """Run an argument vector directly, never through a command shell.

    Output is captured in the returned result unless a corresponding output
    path is supplied. ``log_path`` is shorthand for one combined stdout/stderr
    stream and is mutually exclusive with the two explicit output paths.
    Environment values are intentionally omitted from receipts to avoid
    persisting secrets; only the explicitly supplied key names are recorded.
    """

    command = _normalize_argv(argv)
    working_directory = Path(cwd or os.getcwd()).resolve(strict=True)
    if not working_directory.is_dir():
        raise NotADirectoryError(working_directory)
    overrides = _normalize_environment(environment)
    process_environment = os.environ.copy() if inherit_environment else {}
    process_environment.update(overrides)
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive or None")
    if log_path is not None and (
        stdout_path is not None or stderr_path is not None
    ):
        raise ValueError(
            "log_path is mutually exclusive with stdout_path and stderr_path"
        )
    if log_path is not None:
        stdout_path = log_path
        stderr_path = log_path

    normalized_stdout = (
        Path(stdout_path).resolve() if stdout_path is not None else None
    )
    normalized_stderr = (
        Path(stderr_path).resolve() if stderr_path is not None else None
    )
    receipt = CommandReceipt(
        command_id=command_id or new_id("cmd"),
        argv=command,
        cwd=working_directory,
        started_at=utc_now(),
        environment_keys=tuple(sorted(overrides)),
    )

    stdout_handle: IO[str] | None = None
    stderr_handle: IO[str] | None = None
    stdout_target: int | IO[str] = _subprocess.PIPE
    stderr_target: int | IO[str] = _subprocess.PIPE
    try:
        if normalized_stdout is not None:
            normalized_stdout.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = normalized_stdout.open("w", encoding="utf-8")
            stdout_target = stdout_handle
        if normalized_stderr is not None:
            if (
                normalized_stdout is not None
                and normalized_stderr == normalized_stdout
            ):
                stderr_target = _subprocess.STDOUT
            else:
                normalized_stderr.parent.mkdir(parents=True, exist_ok=True)
                stderr_handle = normalized_stderr.open("w", encoding="utf-8")
                stderr_target = stderr_handle

        monotonic_start = time.monotonic()
        try:
            completed = _subprocess.run(
                list(command),
                cwd=working_directory,
                env=process_environment,
                input=stdin,
                stdout=stdout_target,
                stderr=stderr_target,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except OSError as error:
            raise CommandStartError(receipt, error) from error
        except _subprocess.TimeoutExpired as error:
            raise CommandTimeoutError(
                receipt,
                float(timeout_seconds),
                stdout=error.stdout,
                stderr=error.stderr,
            ) from error
        duration = time.monotonic() - monotonic_start
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()

    result = CommandResult(
        receipt=receipt,
        returncode=completed.returncode,
        finished_at=utc_now(),
        duration_seconds=duration,
        stdout=completed.stdout if normalized_stdout is None else None,
        stderr=completed.stderr if normalized_stderr is None else None,
        stdout_path=normalized_stdout,
        stderr_path=normalized_stderr,
    )
    if check and not result.ok:
        raise CommandExecutionError(result)
    return result


def probe_version(
    argv: Sequence[str | PurePath],
    *,
    cwd: str | PurePath | None = None,
    environment: Mapping[str, str] | None = None,
    markers: Sequence[str] = (),
) -> str:
    """Return a best-effort version line without raising command errors."""

    try:
        result = run_command(
            argv,
            cwd=cwd,
            environment=environment,
            check=False,
        )
    except (CommandError, ValueError):
        return "unknown"
    text = f"{result.stdout or ''}\n{result.stderr or ''}"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for marker in markers:
        for line in lines:
            if marker in line:
                return line.split(":", 1)[-1].strip() if ":" in line else line
    return lines[0] if result.ok and lines else "unknown"


def _normalize_argv(argv: Sequence[str | PurePath]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes, bytearray)):
        raise TypeError("argv must be a sequence of arguments, not a string")
    command = tuple(os.fspath(item) for item in argv)
    if not command:
        raise ValueError("argv cannot be empty")
    if any(not item or "\x00" in item for item in command):
        raise ValueError("command arguments cannot be empty or contain NUL")
    return command


def _normalize_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (environment or {}).items():
        if not isinstance(key, str) or not key or "=" in key or "\x00" in key:
            raise ValueError(f"Invalid environment key: {key!r}")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"Invalid environment value for {key!r}")
        result[key] = value
    return result


def _decode_output(value: str | bytes | None) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value

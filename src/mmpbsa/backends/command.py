"""Safe command validation and injectable subprocess execution."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path
from shutil import which

from mmpbsa.utils.gpu_visibility import GpuVisibilityError
from mmpbsa.utils.process_identity import (
    current_process_observer,
    identity_alive,
    lease_process_identities,
    process_group_alive,
    process_identity,
    session_process_groups,
)

from .api import CommandRecord, CommandSpec
from .errors import BackendConfigurationError, BackendExecutionError


def validate_command(command: CommandSpec) -> None:
    if any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in command.cpu_ids):
        raise BackendConfigurationError("command CPU IDs must be non-negative integers")
    if len(set(command.cpu_ids)) != len(command.cpu_ids):
        raise BackendConfigurationError("command CPU IDs must be unique")
    if not command.argv:
        raise BackendConfigurationError("command argv cannot be empty")
    if any(not isinstance(item, str) or not item or "\x00" in item for item in command.argv):
        raise BackendConfigurationError("command arguments must be non-empty strings without NUL")
    cwd = Path(command.cwd)
    if not str(cwd) or "\x00" in str(cwd):
        raise BackendConfigurationError("command cwd cannot be empty")
    if command.timeout_seconds is not None:
        timeout = float(command.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise BackendConfigurationError("command timeout must be finite and positive")
    for key, value in command.environment.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "\x00" in key
            or "=" in key
            or "\x00" in value
        ):
            raise BackendConfigurationError("command environment contains an invalid entry")


class SubprocessCommandRunner:
    """Default runner; it never invokes a shell."""

    def run(self, command: CommandSpec) -> CommandRecord:
        validate_command(command)
        cwd = Path(command.cwd).resolve()
        if not cwd.is_dir():
            raise BackendConfigurationError(f"command cwd does not exist: {cwd}")
        environment = os.environ.copy()
        environment.update({str(key): str(value) for key, value in command.environment.items()})
        observer = current_process_observer()
        if observer is not None:
            guard = getattr(observer, "validate_environment", None)
            if guard is not None:
                try:
                    guard(environment)
                except GpuVisibilityError as exc:
                    raise BackendConfigurationError(str(exc)) from exc
            environment.update(observer.environment)
        argv = command.argv
        if command.cpu_ids:
            taskset = which("taskset")
            if taskset is None:
                raise BackendConfigurationError("physical CPU execution requires taskset")
            argv = (taskset, "--cpu-list", ",".join(map(str, command.cpu_ids)), *argv)
        invocation_id = uuid.uuid4().hex
        if observer is not None:
            observer.launching(invocation_id)
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                text=True,
                stdin=subprocess.PIPE if command.stdin is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            if observer is not None:
                observer.launch_failed(invocation_id)
            raise BackendExecutionError(f"could not execute {command.argv[0]!r}: {exc}") from exc
        identity = None
        try:
            try:
                identity = process_identity(process.pid)
            except (OSError, AttributeError):
                if observer is not None:
                    raise
            if observer is not None:
                assert identity is not None
                observer.started(invocation_id, identity)
            stdout, stderr = process.communicate(command.stdin, timeout=command.timeout_seconds)
        except BaseException as exc:
            _terminate_session(
                process,
                identity,
                environment.get("MMPBSA_RESOURCE_LEASE_TOKEN") if observer is not None else None,
                int(environment.get("MMPBSA_RESOURCE_LEASE_START_TICKS", "0")),
            )
            if isinstance(exc, (OSError, subprocess.SubprocessError)):
                raise BackendExecutionError(
                    f"could not execute {command.argv[0]!r}: {exc}"
                ) from exc
            raise
        finally:
            if observer is not None:
                observer.finished(invocation_id)
        record = CommandRecord(
            argv=argv,
            cwd=cwd,
            returncode=int(process.returncode),
            stdout=stdout,
            stderr=stderr,
            cpu_ids=command.cpu_ids,
        )
        if record.returncode != 0:
            detail = (record.stderr or record.stdout).strip()
            raise BackendExecutionError(
                f"command failed with exit code {record.returncode}: "
                f"{' '.join(record.argv)}" + (f"\n{detail}" if detail else "")
            )
        return record


def _terminate_session(
    process: subprocess.Popen,
    identity: dict[str, object] | None,
    lease_token: str | None = None,
    created_boot_ticks: int = 0,
) -> None:
    """Cancel only this launch's verified process group, never another job."""
    if identity is None:
        # Popen still owns this exact direct child; no group identity is inferred.
        if process.poll() is None:
            process.kill()
    else:
        for sig, grace in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
            groups = set(session_process_groups(identity) or ())
            if lease_token is not None:
                descendants = lease_process_identities(
                    lease_token,
                    created_boot_ticks=created_boot_ticks,
                )
                for descendant in descendants or ():
                    if identity_alive(descendant) is True:
                        groups.add(int(descendant["pgid"]))
            if not groups:
                break
            for pgid in groups:
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    continue
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                process.poll()
                descendants = (
                    lease_process_identities(
                        lease_token,
                        created_boot_ticks=created_boot_ticks,
                    )
                    if lease_token
                    else ()
                )
                if process_group_alive(identity) is not True and not descendants:
                    break
                time.sleep(0.02)
    if process.poll() is None:
        # Even if descendant/session evidence is unknown, this direct Popen
        # child cannot have been PID-reused before it is reaped.
        process.kill()
    try:
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        # Descendants with unverifiable identity remain held in the registry.
        pass


class AffinityCommandRunner:
    """Immutable invocation-scoped decorator, including injectable fake runners."""

    def __init__(self, cpu_ids: tuple[int, ...], runner=None) -> None:
        self.cpu_ids = cpu_ids
        self.runner = runner or SubprocessCommandRunner()

    def run(self, command: CommandSpec) -> CommandRecord:
        if command.cpu_ids and command.cpu_ids != self.cpu_ids:
            raise BackendConfigurationError("command CPU mask conflicts with stage allocation")
        return self.runner.run(replace(command, cpu_ids=self.cpu_ids))

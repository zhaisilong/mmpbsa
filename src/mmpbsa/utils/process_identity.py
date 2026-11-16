"""Linux process identities and invocation-local native launch observation.

PID numbers and heartbeat age alone are never proofs of process termination.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Protocol


class ProcessObserver(Protocol):
    @property
    def environment(self) -> Mapping[str, str]: ...

    def launching(self, invocation_id: str) -> None: ...

    def started(self, invocation_id: str, identity: Mapping[str, object]) -> None: ...

    def launch_failed(self, invocation_id: str) -> None: ...

    def finished(self, invocation_id: str) -> None: ...


_OBSERVER: ContextVar[ProcessObserver | None] = ContextVar("mmpbsa_process_observer", default=None)


def current_process_observer() -> ProcessObserver | None:
    return _OBSERVER.get()


@contextmanager
def observe_processes(observer: ProcessObserver) -> Iterator[None]:
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def boot_ticks() -> int:
    return int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))


def process_identity(pid: int | None = None) -> dict[str, object]:
    pid = os.getpid() if pid is None else pid
    # comm may contain whitespace or parentheses; split after its final ')'.
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "pgid": int(fields[2]),
        "session_id": int(fields[3]),
        "boot_id": boot_id(),
        "state": fields[0],
    }


def identity_alive(identity: Mapping[str, object]) -> bool | None:
    """False only with positive evidence; None means inaccessible/unknown."""
    try:
        if identity["boot_id"] != boot_id():
            return False
        current = process_identity(int(identity["pid"]))
    except FileNotFoundError:
        return False
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None
    return current["start_ticks"] == identity["start_ticks"] and current["state"] not in {"Z", "X"}


def process_group_alive(identity: Mapping[str, object]) -> bool | None:
    """Include the launch session, including children that change process group."""
    groups = session_process_groups(identity)
    return None if groups is None else bool(groups)


def session_process_groups(identity: Mapping[str, object]) -> tuple[int, ...] | None:
    """Prove the private session still belongs to this recorded launch identity."""
    try:
        if identity["boot_id"] != boot_id():
            return ()
        session_id = int(identity["session_id"])
        try:
            leader = process_identity(int(identity["pid"]))
        except FileNotFoundError:
            leader = None
        if leader is not None and leader["start_ticks"] != identity["start_ticks"]:
            # A recycled group/session leader must never authorize signalling.
            return None
        saw_unknown = False
        groups: set[int] = set()
        for member in Path("/proc").iterdir():
            if not member.name.isdecimal():
                continue
            try:
                stat = (member / "stat").read_text()
                fields = stat[stat.rfind(")") + 2 :].split()
                if int(fields[3]) == session_id and fields[0] not in {"Z", "X"}:
                    groups.add(int(fields[2]))
            except FileNotFoundError:
                continue
            except (OSError, ValueError, IndexError):
                saw_unknown = True
        return None if saw_unknown else tuple(sorted(groups))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def lease_process_identities(
    lease_token: str,
    *,
    created_boot_ticks: int = 0,
) -> tuple[dict[str, object], ...] | None:
    """Find same-UID native descendants even after setsid/leader termination.

    The opaque execution token is inherited in native environments, never in
    science identity. Unreadable same-user processes or a disappearing matching
    process make termination evidence inconclusive instead of freeing a lease.
    No environment values are retained or logged.
    """
    marker = f"MMPBSA_RESOURCE_LEASE_TOKEN={lease_token}".encode()
    identities = []
    unknown = False
    try:
        for member in Path("/proc").iterdir():
            if not member.name.isdecimal():
                continue
            try:
                if member.stat().st_uid != os.getuid():
                    continue
                identity = process_identity(int(member.name))
                if int(identity["start_ticks"]) < created_boot_ticks - 1:
                    # A pre-existing process cannot inherit a token born later.
                    continue
                if identity["state"] in {"Z", "X"}:
                    continue
                environment = (member / "environ").read_bytes()
                if marker not in environment.split(b"\x00"):
                    continue
                # Verify the PID did not change between identity and environ.
                if identity_alive(identity) is not True:
                    unknown = True
                    continue
                identities.append(identity)
            except FileNotFoundError:
                continue
            except (OSError, ValueError, TypeError, IndexError):
                unknown = True
        return None if unknown else tuple(identities)
    except OSError:
        return None

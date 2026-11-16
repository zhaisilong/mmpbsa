"""Explicit host pool policy, fail-closed validation and process-backed recovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import socket
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mmpbsa.utils.cpu import PhysicalCpu, physical_cpu_inventory
from mmpbsa.utils.filesystem import atomic_write_json
from mmpbsa.utils.gpu_visibility import (
    FULL_GPU_UUID_PATTERN,
    GpuVisibilityError,
    canonical_gpu_uuid,
)
from mmpbsa.utils.process_identity import (
    boot_id,
    identity_alive,
    lease_process_identities,
    process_group_alive,
)

from .errors import ConfigurationError

_SCHEMA = "mmpbsa.host-resources.v1"
_NETWORK_FILESYSTEMS = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb",
        "smb3",
        "smbfs",
        "fuse.sshfs",
        "ceph",
        "fuse.ceph",
        "glusterfs",
        "fuse.glusterfs",
        "lustre",
        "afs",
        "9p",
        "gpfs",
        "beegfs",
        "orangefs",
        "pvfs2",
        "davfs",
        "fuse.davfs",
        "gfs2",
        "ocfs2",
    }
)


def _host_identity() -> dict[str, object]:
    try:
        return {
            "uid": os.getuid(),
            "hostname": socket.gethostname(),
            "boot_id": boot_id(),
            "pid_namespace_inode": Path("/proc/self/ns/pid").stat().st_ino,
        }
    except (AttributeError, OSError) as exc:
        raise ConfigurationError("host resource coordination requires a Linux /proc host") from exc


def _path(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ConfigurationError("resource registry must be an absolute, non-symlink file path")
    _require_local_filesystem(path)
    return path


def _mountinfo_text() -> str:
    return Path("/proc/self/mountinfo").read_text()


def _require_local_filesystem(path: Path) -> None:
    """Reject known network/cluster filesystems before registry or lock access."""
    try:
        resolved = path.resolve()
        candidates = []
        for index, line in enumerate(_mountinfo_text().splitlines()):
            before, separator, after = line.partition(" - ")
            fields, filesystem = before.split(), after.split()
            if not separator or len(fields) < 6 or len(filesystem) < 3:
                raise ValueError("malformed mountinfo record")
            mountpoint = Path(
                re.sub(
                    r"\\([0-7]{3})",
                    lambda match: chr(int(match.group(1), 8)),
                    fields[4],
                )
            )
            if not mountpoint.is_absolute():
                raise ValueError("mountinfo contains a relative mountpoint")
            if resolved.is_relative_to(mountpoint):
                candidates.append((len(mountpoint.parts), index, filesystem[0].lower()))
        if not candidates:
            raise ValueError("no visible mount covers the registry path")
        filesystem_type = max(candidates)[2]
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigurationError(f"cannot verify local registry filesystem: {path}: {exc}") from exc
    if filesystem_type in _NETWORK_FILESYSTEMS:
        raise ConfigurationError(
            f"host resource registry requires a local filesystem, not {filesystem_type}: {path}"
        )


def _read(path: Path) -> dict[str, Any]:
    path = _path(path)
    try:
        if path.stat().st_uid != os.getuid():
            raise ConfigurationError("resource registry is owned by another UID")
        state = json.loads(path.read_text())
        if not isinstance(state, dict) or state.get("integrity_sha256") != _state_digest(state):
            raise ConfigurationError("resource registry integrity seal is missing or corrupt")
        _validate_state(state)
        return state
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        raise ConfigurationError(
            f"resource registry is unavailable or corrupt: {path}: {exc}"
        ) from exc


def _state_digest(state: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {key: value for key, value in state.items() if key != "integrity_sha256"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_state(state: object) -> None:
    if not isinstance(state, dict) or state.get("schema_version") != _SCHEMA:
        raise ConfigurationError("unsupported or corrupt host resource registry")
    if state.get("host") != _host_identity():
        raise ConfigurationError(
            "resource registry UID, host, boot, or PID namespace identity differs"
        )
    policy = state["policy"]
    if not isinstance(policy, dict) or not isinstance(policy.get("cpus"), list):
        raise ConfigurationError("resource registry lacks an explicit CPU pool policy")
    cpus = [PhysicalCpu(**item) for item in policy["cpus"]]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for cpu in cpus
        for value in asdict(cpu).values()
    ):
        raise ConfigurationError("corrupt CPU topology counters")
    if not cpus or len({(cpu.package_id, cpu.core_id) for cpu in cpus}) != len(cpus):
        raise ConfigurationError("resource policy duplicates physical cores or has no cores")
    if len({cpu.cpu_id for cpu in cpus}) != len(cpus):
        raise ConfigurationError("resource policy duplicates CPU representatives")
    if any(_topology_key(cpu.cpu_id) != (cpu.package_id, cpu.core_id) for cpu in cpus):
        raise ConfigurationError("resource policy CPU IDs do not match host physical topology")
    gpus = policy["gpu_ids"]
    if not isinstance(gpus, list) or any(
        not isinstance(g, str)
        or not FULL_GPU_UUID_PATTERN.fullmatch(g)
        or canonical_gpu_uuid(g) != g
        for g in gpus
    ):
        raise ConfigurationError(
            "host GPU policy must contain full GPU UUIDs, not indices or MIG IDs"
        )
    if len(set(gpus)) != len(gpus):
        raise ConfigurationError("host GPU policy contains duplicate UUIDs")
    for name in ("leases", "waiters"):
        if not isinstance(state[name], dict):
            raise ConfigurationError(f"corrupt registry {name}")
    order = state["project_order"]
    if (
        not isinstance(order, list)
        or any(not isinstance(x, str) for x in order)
        or len(set(order)) != len(order)
    ):
        raise ConfigurationError("corrupt project fairness order")
    core_keys = {(cpu.package_id, cpu.core_id) for cpu in cpus}
    used_cores: set[tuple[int, int]] = set()
    used_gpus: set[str] = set()
    project_usage: dict[str, int] = {}
    project_policies: dict[str, tuple[int, tuple[str, ...]]] = {}
    for token, lease in state["leases"].items():
        if lease["token"] != token or not isinstance(lease["project_id"], str):
            raise ConfigurationError("corrupt lease identity")
        _validate_identity(lease["controller"])
        if (
            isinstance(lease["created_boot_ticks"], bool)
            or not isinstance(lease["created_boot_ticks"], int)
            or lease["created_boot_ticks"] < 0
        ):
            raise ConfigurationError("corrupt lease birth clock evidence")
        if lease.get("attempt") is not None and (
            not isinstance(lease["attempt"], dict)
            or any(
                not isinstance(lease["attempt"].get(key), str) or not lease["attempt"][key]
                for key in ("task", "attempt_id", "invocation_id")
            )
        ):
            raise ConfigurationError("corrupt lease attempt/invocation context")
        if lease["state"] not in {"active", "orphan-held", "unknown-held"}:
            raise ConfigurationError("corrupt lease state")
        keys = {tuple(k) for k in lease["core_keys"]}
        if (
            not keys
            or len(keys) != len(lease["cpu_ids"])
            or len(set(lease["cpu_ids"])) != len(lease["cpu_ids"])
            or not keys <= core_keys
            or keys & used_cores
        ):
            raise ConfigurationError("corrupt or overlapping CPU leases")
        if any(
            _topology_key(cpu_id) != tuple(core_key)
            for cpu_id, core_key in zip(lease["cpu_ids"], lease["core_keys"], strict=True)
        ):
            raise ConfigurationError(
                "lease CPU ID aliases do not match their physical core identities"
            )
        leased_gpus = set(lease["gpu_ids"])
        if (
            len(leased_gpus) != len(lease["gpu_ids"])
            or not leased_gpus <= set(gpus)
            or leased_gpus & used_gpus
        ):
            raise ConfigurationError("corrupt or overlapping GPU leases")
        used_cores.update(keys)
        used_gpus.update(leased_gpus)
        if not isinstance(lease["launches"], dict):
            raise ConfigurationError("corrupt launch receipts")
        for launch in lease["launches"].values():
            if launch["state"] not in {"pending", "running", "finished"}:
                raise ConfigurationError("corrupt launch state")
            if launch["state"] != "pending":
                _validate_identity(launch["identity"])
        if (
            not isinstance(lease["cpu_cap"], int)
            or lease["cpu_cap"] < len(keys)
            or not isinstance(lease["gpu_policy"], list)
            or not leased_gpus <= set(lease["gpu_policy"]) <= set(gpus)
        ):
            raise ConfigurationError("corrupt project cap in lease")
        project = lease["project_id"]
        project_policy = (lease["cpu_cap"], tuple(lease["gpu_policy"]))
        if project in project_policies and project_policies[project] != project_policy:
            raise ConfigurationError("inconsistent project cap across host leases")
        project_policies[project] = project_policy
        project_usage[project] = project_usage.get(project, 0) + len(keys)
        if project_usage[project] > lease["cpu_cap"]:
            raise ConfigurationError("host leases exceed their project CPU cap")
    for waiter in state["waiters"].values():
        _validate_identity(waiter["controller"])
        for name in ("cpu_slots", "cpu_cap"):
            if (
                not isinstance(waiter[name], int)
                or isinstance(waiter[name], bool)
                or waiter[name] < 1
            ):
                raise ConfigurationError("corrupt waiting request CPU budget")
        if not isinstance(waiter["gpu_slots"], int) or waiter["gpu_slots"] < 0:
            raise ConfigurationError("corrupt waiting request GPU budget")
        if not isinstance(waiter["reserved"], int) or waiter["reserved"] < 0:
            raise ConfigurationError("corrupt waiting request CPU reservation")
        if (
            isinstance(waiter["priority"], bool)
            or not isinstance(waiter["priority"], int)
            or waiter["priority"] < 0
        ):
            raise ConfigurationError("corrupt waiting request priority")
        if (
            not isinstance(waiter["allowed_core_keys"], list)
            or not {tuple(k) for k in waiter["allowed_core_keys"]} <= core_keys
        ):
            raise ConfigurationError("corrupt waiting request CPU affinity")
        if not isinstance(waiter["gpu_ids"], list) or not set(waiter["gpu_ids"]) <= set(gpus):
            raise ConfigurationError("corrupt waiting request GPU policy")
        if not isinstance(waiter["project_id"], str) or not waiter["project_id"]:
            raise ConfigurationError("corrupt waiting project identity")
        if (
            waiter["cpu_slots"] > waiter["cpu_cap"]
            or waiter["gpu_slots"] > len(waiter["gpu_ids"])
            or len({tuple(k) for k in waiter["allowed_core_keys"]}) < waiter["cpu_cap"]
        ):
            raise ConfigurationError("waiting request cannot fit its declared resource policy")
        project_policy = (waiter["cpu_cap"], tuple(waiter["gpu_ids"]))
        if waiter["project_id"] in project_policies and (
            project_policies[waiter["project_id"]] != project_policy
        ):
            raise ConfigurationError("inconsistent waiting project resource policy")


def _validate_identity(identity: object) -> None:
    if not isinstance(identity, dict) or not isinstance(identity.get("boot_id"), str):
        raise ConfigurationError("corrupt process identity")
    if any(
        isinstance(identity.get(key), bool)
        or not isinstance(identity.get(key), int)
        or identity[key] < 1
        for key in ("pid", "start_ticks", "pgid", "session_id")
    ):
        raise ConfigurationError("corrupt process identity counters")


def _topology_key(cpu_id: int) -> tuple[int, int]:
    if isinstance(cpu_id, bool) or not isinstance(cpu_id, int) or cpu_id < 0:
        raise ConfigurationError("physical CPU ID must be a non-negative integer")
    root = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
    try:
        return (
            int((root / "physical_package_id").read_text()),
            int((root / "core_id").read_text()),
        )
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot verify physical CPU topology for {cpu_id}") from exc


@contextmanager
def _locked(path: Path, *, initializing: bool = False) -> Iterator[dict[str, Any]]:
    path = _path(path)
    lock_path = path.with_name(path.name + ".lock")
    if not initializing:
        _read(path)  # No accidental registry/parent creation on miss or corruption.
    elif not path.parent.is_dir():
        raise ConfigurationError("registry parent must be an explicitly created directory")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(descriptor).st_uid != os.getuid():
            raise ConfigurationError("resource registry lock is owned by another UID")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        state = {} if initializing else _read(path)
        yield state
        if state:
            _validate_state(state)
            state["integrity_sha256"] = _state_digest(state)
            atomic_write_json(path, state)
            os.chmod(path, 0o600)
    finally:
        os.close(descriptor)


def initialize_registry(
    registry_path: Path, *, cpu_ids: Sequence[int], gpu_ids: Sequence[str] = ()
) -> dict[str, Any]:
    """Explicitly seal a CPU/GPU policy; never overwrite an existing registry."""
    path = _path(registry_path)
    try:
        normalized_gpu_ids = [canonical_gpu_uuid(gpu_id) for gpu_id in gpu_ids]
    except GpuVisibilityError as exc:
        raise ConfigurationError(str(exc)) from exc
    if (
        not cpu_ids
        or len(set(cpu_ids)) != len(cpu_ids)
        or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in cpu_ids)
    ):
        raise ConfigurationError("pool initialization requires explicit unique CPU IDs")
    inventory = physical_cpu_inventory()
    # Accept any allowed SMT representative, then normalize to physical keys.
    available = _allowed_topology(inventory)
    missing = set(cpu_ids) - set(available)
    if missing:
        raise ConfigurationError(f"CPU IDs are unavailable in allowed affinity: {sorted(missing)}")
    cpus = [available[cpu] for cpu in cpu_ids]
    if len({(cpu.package_id, cpu.core_id) for cpu in cpus}) != len(cpus):
        raise ConfigurationError("pool initialization cannot include SMT siblings of the same core")
    with _locked(path, initializing=True) as state:
        if path.exists() or path.is_symlink():
            raise ConfigurationError(
                "resource registry already exists; inspect/reconcile it instead"
            )
        state.update(
            {
                "schema_version": _SCHEMA,
                "host": _host_identity(),
                "policy": {
                    "cpus": [asdict(cpu) for cpu in cpus],
                    "gpu_ids": normalized_gpu_ids,
                },
                "leases": {},
                "waiters": {},
                "project_order": [],
            }
        )
    return inspect_registry(path)


def _allowed_topology(inventory: Sequence[PhysicalCpu]) -> dict[int, PhysicalCpu]:
    # Inventory provides representatives; inspect other allowed siblings too.
    result = {item.cpu_id: item for item in inventory}
    for cpu_id in os.sched_getaffinity(0):
        if cpu_id in result:
            continue
        root = Path(f"/sys/devices/system/cpu/cpu{cpu_id}")
        try:
            package = int((root / "topology/physical_package_id").read_text())
            core = int((root / "topology/core_id").read_text())
            nodes = sorted(root.glob("node[0-9]*"))
            numa = int(nodes[0].name[4:]) if nodes else package
            result[cpu_id] = PhysicalCpu(cpu_id, package, core, numa)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot verify allowed CPU topology: {cpu_id}") from exc
    return result


def _launch_evidence(lease: Mapping[str, Any]) -> tuple[bool, bool]:
    alive = unknown = False
    for launch in lease["launches"].values():
        if launch["state"] == "pending":
            unknown = True
        else:
            evidence = process_group_alive(launch["identity"])
            alive |= evidence is True
            unknown |= evidence is None
    if lease["launches"]:
        native_processes = lease_process_identities(
            lease["token"],
            created_boot_ticks=lease["created_boot_ticks"],
        )
        alive |= bool(native_processes)
        unknown |= native_processes is None
    return alive, unknown


def _reclamation(state: dict[str, Any]) -> dict[str, list[str]]:
    actions: dict[str, list[str]] = {"reclaimable": [], "orphan_held": [], "unknown_held": []}
    for token, lease in state["leases"].items():
        alive, unknown = _launch_evidence(lease)
        controller = identity_alive(lease["controller"])
        if lease["state"] == "active" and controller is True:
            continue
        if unknown or controller is None:
            actions["unknown_held"].append(token)
        elif alive:
            actions["orphan_held"].append(token)
        else:
            actions["reclaimable"].append(token)
    return actions


def _clean(state: dict[str, Any]) -> None:
    actions = _reclamation(state)
    for token in actions["reclaimable"]:
        del state["leases"][token]
    for token in actions["orphan_held"]:
        state["leases"][token]["state"] = "orphan-held"
    for token in actions["unknown_held"]:
        state["leases"][token]["state"] = "unknown-held"
    for ticket, waiter in list(state["waiters"].items()):
        if identity_alive(waiter["controller"]) is False:
            del state["waiters"][ticket]
    projects = {w["project_id"] for w in state["waiters"].values()}
    state["project_order"] = [p for p in state["project_order"] if p in projects]


def inspect_registry(registry_path: Path) -> dict[str, Any]:
    """Read-only snapshot with process-backed recovery evidence; no lock file."""
    state = _read(registry_path)
    return {**state, "reconciliation": _reclamation(state)}


def reconcile_registry(registry_path: Path, *, confirm: bool = False) -> dict[str, Any]:
    """Dry-run by default. Never force-release living or unverifiable work."""
    if not confirm:
        return inspect_registry(registry_path)
    with _locked(registry_path) as state:
        _clean(state)
    return inspect_registry(registry_path)

"""Atomic same-host allocations, fair project grants and native launch observers."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmpbsa.utils.cpu import PhysicalCpu, physical_cpu_inventory
from mmpbsa.utils.gpu_visibility import (
    GpuVisibilityError,
    normalize_gpu_ids,
    require_gpu_subset,
    visible_gpu_ids,
)
from mmpbsa.utils.process_identity import (
    boot_ticks,
    identity_alive,
    observe_processes,
    process_group_alive,
    process_identity,
)

from .errors import ConfigurationError, ContractError
from .host_registry import (
    _allowed_topology,
    _clean,
    _launch_evidence,
    _locked,
    _path,
    _read,
)
from .host_registry import (
    initialize_registry as initialize_registry,
)
from .host_registry import (
    inspect_registry as inspect_registry,
)
from .host_registry import (
    reconcile_registry as reconcile_registry,
)
from .models import ResourceAllocation, ResourceRequest, TaskKey


@dataclass(frozen=True, slots=True)
class ResourceWait:
    reason: str
    since: float
    detail: str


def gpu_activity(gpu_ids: Sequence[str]) -> Mapping[str, Sequence[int]]:
    """Fail closed unless UUID inventory and native process observation succeed."""
    inventory = subprocess.run(
        ("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"),
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )
    observed = {line.strip() for line in inventory.stdout.splitlines() if line.strip()}
    if not set(gpu_ids) <= observed:
        raise RuntimeError("configured GPU UUID not present in observed inventory")
    result: dict[str, list[int]] = {gpu: [] for gpu in gpu_ids}
    process_result = subprocess.run(
        ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"),
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )
    for line in process_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[1].isdecimal():
            raise RuntimeError("GPU process observation is unavailable or ambiguous")
        if fields[0] in result:
            result[fields[0]].append(int(fields[1]))
    return result


class HostResourcePool:
    is_host_pool = True

    def __init__(
        self,
        *,
        registry_path: Path,
        project_id: str,
        cpu_slots: int,
        gpu_ids: Sequence[str],
        gpu_probe: Callable = gpu_activity,
    ) -> None:
        self.path = _path(registry_path)
        state = _read(self.path)
        if not isinstance(project_id, str) or not project_id:
            raise ConfigurationError("host resource project_id must be non-empty")
        if isinstance(cpu_slots, bool) or not isinstance(cpu_slots, int) or cpu_slots < 1:
            raise ConfigurationError("host resource project CPU budget must be positive")
        self.project_id = project_id
        self.cpu_cap = cpu_slots
        self._parent_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._parent_gpu_visibility = None
        try:
            self.gpu_ids = normalize_gpu_ids(tuple(gpu_ids))
            if self.gpu_ids:
                self._parent_gpu_visibility = visible_gpu_ids(self._parent_cuda_visible_devices)
                require_gpu_subset(self.gpu_ids, self._parent_gpu_visibility, label="parent")
        except GpuVisibilityError as exc:
            raise ConfigurationError(str(exc)) from exc
        if len(set(self.gpu_ids)) != len(self.gpu_ids) or not set(self.gpu_ids) <= set(
            state["policy"]["gpu_ids"]
        ):
            raise ConfigurationError(
                "project GPUs must be full UUIDs within the initialized host pool"
            )
        allowed = _allowed_topology(physical_cpu_inventory())
        pool_keys = {(c["package_id"], c["core_id"]) for c in state["policy"]["cpus"]}
        representatives: dict[tuple[int, int], PhysicalCpu] = {}
        for cpu in sorted(allowed.values(), key=lambda cpu: cpu.cpu_id):
            key = (cpu.package_id, cpu.core_id)
            if key in pool_keys:
                representatives.setdefault(key, cpu)
        self._inventory = tuple(representatives.values())
        if cpu_slots > len(self._inventory):
            raise ConfigurationError(
                "project CPU budget exceeds allowed initialized physical-core pool"
            )
        self._controller = process_identity()
        self._controller_token = uuid.uuid4().hex
        self._owned: set[str] = set()
        self._probe = gpu_probe
        self.last_wait: ResourceWait | None = None

    @property
    def cpu_ids(self) -> tuple[int, ...]:
        """All eligible candidates, not an arbitrary cap-sized common prefix."""
        return tuple(cpu.cpu_id for cpu in self._inventory)

    def validate(self, request: ResourceRequest) -> None:
        if request.cpu_slots > self.cpu_cap:
            raise ContractError("task CPU request exceeds project cap")
        if request.gpu_slots > len(self.gpu_ids):
            raise ContractError("task GPU request exceeds project GPU policy")

    def _wait(self, reason: str, detail: str, since: float) -> None:
        self.last_wait = ResourceWait(reason, since, detail)

    def try_acquire(
        self,
        request: ResourceRequest,
        *,
        reserved: int = 0,
        request_key: str | None = None,
        priority: int = 0,
    ) -> ResourceAllocation | None:
        self.validate(request)
        if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved < 0:
            raise ContractError("reserved CPU slots must be a non-negative integer")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise ContractError("host waiting priority must be a non-negative integer")
        request_key = request_key or f"cpu={request.cpu_slots},gpu={request.gpu_slots}"
        ticket = f"{self._controller_token}:{request_key}"
        with _locked(self.path) as state:
            _clean(state)
            for lease in state["leases"].values():
                if lease["project_id"] == self.project_id and (
                    lease["cpu_cap"] != self.cpu_cap or lease["gpu_policy"] != list(self.gpu_ids)
                ):
                    raise ConfigurationError(
                        "project resource policy changed while host leases remain"
                    )
            for queued in state["waiters"].values():
                if queued["project_id"] == self.project_id and (
                    queued["cpu_cap"] != self.cpu_cap or queued["gpu_ids"] != list(self.gpu_ids)
                ):
                    raise ConfigurationError(
                        "project resource policy changed while host waiters remain"
                    )
            existing = state["waiters"].get(ticket)
            since = existing["since"] if existing else time.time()
            waiter = {
                "project_id": self.project_id,
                "controller": self._controller,
                "controller_token": self._controller_token,
                "request_key": request_key,
                "cpu_slots": request.cpu_slots,
                "gpu_slots": request.gpu_slots,
                "cpu_cap": self.cpu_cap,
                "reserved": reserved,
                "priority": priority,
                "gpu_ids": list(self.gpu_ids),
                "allowed_core_keys": [[c.package_id, c.core_id] for c in self._inventory],
                "since": since,
            }
            state["waiters"][ticket] = waiter
            if self.project_id not in state["project_order"]:
                state["project_order"].append(self.project_id)
            occupied = {tuple(k) for lease in state["leases"].values() for k in lease["core_keys"]}
            occupied_gpus = {g for lease in state["leases"].values() for g in lease["gpu_ids"]}
            candidate_gpus = {
                g
                for w in state["waiters"].values()
                if w["gpu_slots"]
                for g in w["gpu_ids"]
                if g not in occupied_gpus
            }
            unavailable_gpus: set[str] = set()
            unknown_gpu = False
            if candidate_gpus:
                try:
                    activity = self._probe(tuple(sorted(candidate_gpus)))
                    if not isinstance(activity, Mapping) or not candidate_gpus <= set(activity):
                        raise RuntimeError("incomplete GPU process observation")
                    unavailable_gpus = {g for g in candidate_gpus if activity[g]}
                except (OSError, subprocess.SubprocessError, RuntimeError, ValueError):
                    unknown_gpu = True
                    unavailable_gpus = candidate_gpus
            eligible: dict[str, tuple[str, dict[str, Any]]] = {}
            for project in state["project_order"]:
                for waiting_ticket, queued in sorted(
                    state["waiters"].items(),
                    key=lambda item: (item[1]["priority"], item[1]["since"], item[0]),
                ):
                    if queued["project_id"] != project:
                        continue
                    if identity_alive(queued["controller"]) is not True:
                        continue
                    usage = sum(
                        len(lease["cpu_ids"])
                        for lease in state["leases"].values()
                        if lease["project_id"] == project
                    )
                    free_keys = {tuple(k) for k in queued["allowed_core_keys"]} - occupied
                    free_gpus = set(queued["gpu_ids"]) - occupied_gpus - unavailable_gpus
                    if (
                        usage + queued["reserved"] + queued["cpu_slots"] <= queued["cpu_cap"]
                        and len(free_keys) >= queued["cpu_slots"]
                        and len(free_gpus) >= queued["gpu_slots"]
                    ):
                        eligible[project] = (waiting_ticket, queued)
                        break
            chosen = next((eligible[p][0] for p in state["project_order"] if p in eligible), None)
            if chosen != ticket:
                own_usage = sum(
                    len(lease["cpu_ids"])
                    for lease in state["leases"].values()
                    if lease["project_id"] == self.project_id
                )
                free_cpu = {(c.package_id, c.core_id) for c in self._inventory} - occupied
                free_gpu = set(self.gpu_ids) - occupied_gpus - unavailable_gpus
                if own_usage + reserved + request.cpu_slots > self.cpu_cap:
                    reason = "project_capacity"
                elif len(free_cpu) < request.cpu_slots:
                    reason = "host_cpu_capacity"
                elif len(free_gpu) < request.gpu_slots:
                    if unknown_gpu:
                        reason = "gpu_observation_unavailable"
                    elif unavailable_gpus & set(self.gpu_ids):
                        reason = "external_gpu_process"
                    else:
                        reason = "host_gpu_capacity"
                else:
                    reason = "host_fairness"
                self._wait(reason, "waiting for an atomic host CPU/GPU grant", since)
                return None
            free = [c for c in self._inventory if (c.package_id, c.core_id) not in occupied]
            local = next(
                (
                    [c for c in free if c.numa_id == node]
                    for node in sorted({c.numa_id for c in free})
                    if sum(c.numa_id == node for c in free) >= request.cpu_slots
                ),
                free,
            )
            cpus = local[: request.cpu_slots]
            gpus = tuple(g for g in self.gpu_ids if g not in occupied_gpus | unavailable_gpus)[
                : request.gpu_slots
            ]
            token = uuid.uuid4().hex
            state["leases"][token] = {
                "token": token,
                "project_id": self.project_id,
                "controller": self._controller,
                "controller_token": self._controller_token,
                "state": "active",
                "heartbeat_at": time.time(),
                "created_boot_ticks": boot_ticks(),
                "cpu_cap": self.cpu_cap,
                "gpu_policy": list(self.gpu_ids),
                "cpu_ids": [c.cpu_id for c in cpus],
                "core_keys": [[c.package_id, c.core_id] for c in cpus],
                "gpu_ids": list(gpus),
                "launches": {},
            }
            del state["waiters"][ticket]
            state["project_order"].remove(self.project_id)
            if any(w["project_id"] == self.project_id for w in state["waiters"].values()):
                state["project_order"].append(self.project_id)
            self._owned.add(token)
            self.last_wait = None
            return ResourceAllocation(
                request.cpu_slots,
                tuple(c.cpu_id for c in cpus),
                tuple(str(i) for i in range(len(gpus))),
                gpus,
                token,
            )

    def release(self, allocation: ResourceAllocation) -> None:
        token = allocation.lease_token
        if token not in self._owned:
            raise ContractError("host allocation is not owned by this pool or was already released")
        with _locked(self.path) as state:
            lease = state["leases"].get(token)
            if lease is None or lease["controller_token"] != self._controller_token:
                raise ContractError("host lease token does not match the owning controller")
            alive, unknown = _launch_evidence(lease)
            if unknown:
                lease["state"] = "unknown-held"
            elif alive:
                lease["state"] = "orphan-held"
            else:
                del state["leases"][token]
            self._owned.remove(token)

    def heartbeat(self) -> None:
        with _locked(self.path) as state:
            for token in self._owned:
                if token in state["leases"]:
                    state["leases"][token]["heartbeat_at"] = time.time()

    def attach_attempt(
        self,
        allocation: ResourceAllocation,
        task: TaskKey | str,
        attempt_id: str,
        invocation_id: str,
    ) -> None:
        """Seal the one attempt/invocation authorized by this stage lease."""
        context = {
            "task": task.label if isinstance(task, TaskKey) else task,
            "attempt_id": attempt_id,
            "invocation_id": invocation_id,
        }
        if any(not isinstance(value, str) or not value for value in context.values()):
            raise ContractError("lease attempt context must contain non-empty identifiers")
        if isinstance(task, TaskKey):
            context.update(
                case_id=task.case_id,
                replica_id=task.replica_id,
                stage=task.stage.value,
                model_id=task.model_id,
            )
        if allocation.lease_token not in self._owned:
            raise ContractError("cannot attach attempt without its owned host lease")
        with _locked(self.path) as state:
            lease = state["leases"].get(allocation.lease_token)
            if lease is None or lease["controller_token"] != self._controller_token:
                raise ContractError("attempt context belongs to another host controller")
            if lease.get("attempt") is not None and lease["attempt"] != context:
                raise ContractError("host lease is already attached to another attempt/invocation")
            lease["attempt"] = context

    def cancel_request(self, request_key: str) -> None:
        """Remove a withdrawn task ticket without dropping other queued stages."""
        with _locked(self.path) as state:
            state["waiters"].pop(f"{self._controller_token}:{request_key}", None)
            _clean(state)

    def cancel_pending(self) -> None:
        with _locked(self.path) as state:
            state["waiters"] = {
                key: waiter
                for key, waiter in state["waiters"].items()
                if waiter["controller_token"] != self._controller_token
            }
            _clean(state)

    @contextmanager
    def lease(self, request: ResourceRequest) -> Iterator[ResourceAllocation]:
        allocation = self.try_acquire(request)
        try:
            while allocation is None:
                time.sleep(5)
                allocation = self.try_acquire(request)
            yield allocation
        finally:
            if allocation is not None:
                self.release(allocation)
            else:
                self.cancel_pending()

    @contextmanager
    def observe(self, allocation: ResourceAllocation, invocation_id: str) -> Iterator[None]:
        if allocation.lease_token not in self._owned:
            raise ContractError("cannot observe processes without an owned host lease")
        with observe_processes(_LeaseObserver(self, allocation, str(invocation_id))):
            yield


class _LeaseObserver:
    def __init__(
        self, pool: HostResourcePool, allocation: ResourceAllocation, invocation_id: str
    ) -> None:
        self.pool = pool
        self.token = allocation.lease_token
        self.invocation_id = invocation_id
        state = _read(pool.path)
        lease = state["leases"][self.token]
        if (
            tuple(lease["cpu_ids"]) != allocation.cpu_ids
            or len(lease["cpu_ids"]) != allocation.cpu_slots
            or tuple(lease["gpu_ids"]) != allocation.physical_gpu_ids
            or allocation.gpu_ids != tuple(str(index) for index in range(len(lease["gpu_ids"])))
        ):
            raise ContractError("process observer allocation differs from its sealed host lease")
        created_boot_ticks = lease["created_boot_ticks"]
        self.environment = (
            {"CUDA_VISIBLE_DEVICES": ",".join(allocation.physical_gpu_ids)}
            if allocation.physical_gpu_ids
            else {}
        )
        self.environment["MMPBSA_RESOURCE_LEASE_TOKEN"] = str(self.token)
        self.environment["MMPBSA_RESOURCE_LEASE_START_TICKS"] = str(created_boot_ticks)

    def validate_environment(self, environment: Mapping[str, str]) -> None:
        allocated = tuple(self.environment.get("CUDA_VISIBLE_DEVICES", "").split(","))
        if not self.environment.get("CUDA_VISIBLE_DEVICES"):
            return  # CPU stages need no GPU inventory or CUDA mask interpretation.
        require_gpu_subset(allocated, self.pool._parent_gpu_visibility, label="original parent")
        current = os.environ.get("CUDA_VISIBLE_DEVICES")
        if current != self.pool._parent_cuda_visible_devices:
            require_gpu_subset(allocated, visible_gpu_ids(current), label="current parent")
        supplied = environment.get("CUDA_VISIBLE_DEVICES")
        visible = (
            self.pool._parent_gpu_visibility
            if supplied == self.pool._parent_cuda_visible_devices
            else visible_gpu_ids(supplied)
        )
        require_gpu_subset(allocated, visible, label="command")

    @contextmanager
    def _launches(self) -> Iterator[dict[str, Any]]:
        with _locked(self.pool.path) as state:
            lease = state["leases"].get(self.token)
            if lease is None or lease["controller_token"] != self.pool._controller_token:
                raise ContractError("native launch cannot be recorded without its owning lease")
            yield lease["launches"]

    def launching(self, invocation_id: str) -> None:
        with self._launches() as launches:
            launches[invocation_id] = {
                "state": "pending",
                "invocation": self.invocation_id,
                "created_at": time.time(),
            }

    def started(self, invocation_id: str, identity: Mapping[str, object]) -> None:
        with self._launches() as launches:
            launches[invocation_id].update(state="running", identity=dict(identity))

    def launch_failed(self, invocation_id: str) -> None:
        with self._launches() as launches:
            # Popen positively reported failure before handing out a child.
            del launches[invocation_id]

    def finished(self, invocation_id: str) -> None:
        with self._launches() as launches:
            launch = launches[invocation_id]
            if launch["state"] != "pending" and process_group_alive(launch["identity"]) is False:
                launch["state"] = "finished"

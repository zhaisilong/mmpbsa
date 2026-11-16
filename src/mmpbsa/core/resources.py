"""One process-local pool with concrete, non-overlapping stage allocations."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from mmpbsa.utils.cpu import PhysicalCpu, physical_cpu_inventory

from .errors import ConfigurationError, ContractError
from .models import ResourceAllocation, ResourceRequest, TaskKey


class ResourcePool:
    is_host_pool = False
    last_wait = None

    def __init__(
        self, *, cpu_slots: int, gpu_ids: Sequence[str], cpu_affinity: str = "none"
    ) -> None:
        self._cpu_total = cpu_slots
        self._cpu_available = cpu_slots
        self._gpu_ids = tuple(gpu_ids)
        self._gpu_available = list(gpu_ids)
        self._condition = threading.Condition()
        self._leased: set[int] = set()
        self._cpu_inventory: tuple[PhysicalCpu, ...] = ()
        if cpu_affinity == "physical":
            try:
                inventory = physical_cpu_inventory()
            except (OSError, ValueError, RuntimeError) as exc:
                raise ConfigurationError(f"cannot discover physical CPUs: {exc}") from exc
            if len(inventory) < cpu_slots:
                raise ConfigurationError(
                    f"requested {cpu_slots} physical cores; only {len(inventory)} allowed"
                )
            self._cpu_inventory = inventory[:cpu_slots]
        self._cpu_free = {item.cpu_id: item for item in self._cpu_inventory}

    @property
    def cpu_ids(self) -> tuple[int, ...]:
        return tuple(item.cpu_id for item in self._cpu_inventory)

    def validate(self, request: ResourceRequest) -> None:
        if request.cpu_slots > self._cpu_total:
            raise ContractError(
                f"task requests {request.cpu_slots} CPU slots, "
                f"but scheduler provides {self._cpu_total}"
            )
        if request.gpu_slots > len(self._gpu_ids):
            raise ContractError(
                f"task requests {request.gpu_slots} GPUs, "
                f"but scheduler provides {len(self._gpu_ids)}"
            )

    def try_acquire(
        self, request: ResourceRequest, *, reserved: int = 0,
        request_key: str | None = None, priority: int = 0,
    ) -> ResourceAllocation | None:
        self.validate(request)
        with self._condition:
            if (
                self._cpu_available - reserved < request.cpu_slots
                or len(self._gpu_available) < request.gpu_slots
            ):
                return None
            ids: tuple[int, ...] = ()
            if self._cpu_inventory:
                groups: dict[int, list[int]] = defaultdict(list)
                for cpu_id, item in sorted(self._cpu_free.items()):
                    groups[item.numa_id].append(cpu_id)
                local = next(
                    (
                        groups[node]
                        for node in sorted(groups)
                        if len(groups[node]) >= request.cpu_slots
                    ),
                    None,
                )
                ids = tuple((local or sorted(self._cpu_free))[: request.cpu_slots])
                for cpu_id in ids:
                    del self._cpu_free[cpu_id]
            gpus = tuple(self._gpu_available.pop(0) for _ in range(request.gpu_slots))
            allocation = ResourceAllocation(request.cpu_slots, ids, gpus)
            self._cpu_available -= request.cpu_slots
            self._leased.add(id(allocation))
            return allocation

    def release(self, allocation: ResourceAllocation) -> None:
        with self._condition:
            if id(allocation) not in self._leased:
                raise ContractError(
                    "resource allocation is not owned by this pool or was already released"
                )
            self._leased.remove(id(allocation))
            self._cpu_available += allocation.cpu_slots
            inventory = {item.cpu_id: item for item in self._cpu_inventory}
            self._cpu_free.update({cpu: inventory[cpu] for cpu in allocation.cpu_ids})
            self._gpu_available.extend(allocation.gpu_ids)
            self._gpu_available.sort(key=self._gpu_ids.index)
            self._condition.notify_all()

    def heartbeat(self) -> None:
        """Local allocations have no persistent controller liveness record."""

    def attach_attempt(
        self, allocation: ResourceAllocation, task: TaskKey | str,
        attempt_id: str, invocation_id: str,
    ) -> None:
        """Local allocations have no persistent per-attempt lease record."""

    def cancel_pending(self) -> None:
        """Local allocations have no persistent wait tickets."""

    def cancel_request(self, request_key: str) -> None:
        """Local allocations have no persistent per-task wait tickets."""

    @contextmanager
    def observe(self, allocation: ResourceAllocation, invocation_id: str) -> Iterator[None]:
        yield

    @contextmanager
    def lease(self, request: ResourceRequest) -> Iterator[ResourceAllocation]:
        self.validate(request)
        with self._condition:
            allocation = self.try_acquire(request)
            while allocation is None:
                self._condition.wait()
                allocation = self.try_acquire(request)
        try:
            yield allocation
        finally:
            self.release(allocation)

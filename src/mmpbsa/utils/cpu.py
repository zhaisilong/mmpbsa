"""Linux CPU topology and invocation-local affinity (no global environment edits)."""

from __future__ import annotations

import os
import threading
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PhysicalCpu:
    cpu_id: int
    package_id: int
    core_id: int
    numa_id: int


def physical_cpu_inventory() -> tuple[PhysicalCpu, ...]:
    """One allowed hardware thread per physical core, NUMA-round-robin order."""
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("physical CPU allocation requires Linux sched affinity")
    representatives: dict[tuple[int, int], PhysicalCpu] = {}
    for cpu_id in sorted(os.sched_getaffinity(0)):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu_id}")
        package = int((root / "topology/physical_package_id").read_text())
        core = int((root / "topology/core_id").read_text())
        nodes = sorted(root.glob("node[0-9]*"))
        node = int(nodes[0].name[4:]) if nodes else package
        representatives.setdefault((package, core), PhysicalCpu(cpu_id, package, core, node))
    groups: dict[int, deque[PhysicalCpu]] = defaultdict(deque)
    for item in sorted(representatives.values(), key=lambda item: item.cpu_id):
        groups[item.numa_id].append(item)
    result = []
    while any(groups.values()):
        for node in sorted(groups):
            if groups[node]:
                result.append(groups[node].popleft())
    return tuple(result)


@contextmanager
def thread_cpu_affinity(cpu_ids: Sequence[int]) -> Iterator[None]:
    """Bind only the calling worker; restore it before thread-pool reuse."""
    if not cpu_ids:
        yield
        return
    native_id = threading.get_native_id()
    previous = os.sched_getaffinity(native_id)
    try:
        os.sched_setaffinity(native_id, cpu_ids)
        if os.sched_getaffinity(native_id) != set(cpu_ids):
            raise RuntimeError("worker CPU affinity differs from its resource lease")
        yield
    finally:
        os.sched_setaffinity(native_id, previous)

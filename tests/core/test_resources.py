from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from mmpbsa.backends.api import CommandSpec
from mmpbsa.backends.command import AffinityCommandRunner
from mmpbsa.core.config import SchedulerConfig, load_runtime
from mmpbsa.core.errors import ConfigurationError, ContractError
from mmpbsa.core.models import ResourceRequest
from mmpbsa.core.resources import ResourcePool
from mmpbsa.utils.cpu import PhysicalCpu, physical_cpu_inventory, thread_cpu_affinity


def test_physical_leases_are_disjoint_and_release_on_failure(monkeypatch):
    inventory = tuple(PhysicalCpu(i, i // 16, i % 16, i // 16) for i in range(32))
    monkeypatch.setattr("mmpbsa.core.resources.physical_cpu_inventory", lambda: inventory)
    pool = ResourcePool(cpu_slots=32, gpu_ids=("5", "6", "7"), cpu_affinity="physical")
    allocations = [pool.try_acquire(ResourceRequest(4)) for _ in range(8)]
    assert all(item is not None for item in allocations)
    assert len({cpu for item in allocations for cpu in item.cpu_ids}) == 32
    assert pool.try_acquire(ResourceRequest()) is None
    for item in allocations:
        pool.release(item)
    with pytest.raises(RuntimeError), pool.lease(ResourceRequest(4, 1)) as item:
        assert item.gpu_ids == ("5",)
        raise RuntimeError("failed stage")
    with pool.lease(ResourceRequest(32, 3)) as item:
        assert len(item.cpu_ids) == 32
        assert item.gpu_ids == ("5", "6", "7")
    with pytest.raises(ContractError, match="already released"):
        pool.release(item)


def test_capacity_and_reserved_cores(monkeypatch):
    monkeypatch.setattr("mmpbsa.core.resources.physical_cpu_inventory", lambda: ())
    with pytest.raises(ConfigurationError, match="only 0"):
        ResourcePool(cpu_slots=32, gpu_ids=(), cpu_affinity="physical")
    pool = ResourcePool(cpu_slots=32, gpu_ids=())
    allocations = [pool.try_acquire(ResourceRequest(4), reserved=14) for _ in range(4)]
    assert pool.try_acquire(ResourceRequest(4), reserved=14) is None
    for item in allocations:
        pool.release(item)
    with pytest.raises(ContractError):
        pool.try_acquire(ResourceRequest(33))
    assert pool.try_acquire(ResourceRequest(4), reserved=28) is not None


def test_worker_and_child_affinity_restored(tmp_path):
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux CPU affinity")
    pool = ResourcePool(cpu_slots=2, gpu_ids=(), cpu_affinity="physical")
    previous = os.sched_getaffinity(0)
    with pool.lease(ResourceRequest(1)) as first, pool.lease(ResourceRequest(1)) as second:

        def probe(allocation):
            with thread_cpu_affinity(allocation.cpu_ids):
                assert os.sched_getaffinity(0) == set(allocation.cpu_ids)
                runner = AffinityCommandRunner(allocation.cpu_ids)
                record = runner.run(
                    CommandSpec(
                        (
                            sys.executable,
                            "-c",
                            "import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))",
                        ),
                        Path(tmp_path),
                    )
                )
                assert tuple(json.loads(record.stdout)) == allocation.cpu_ids
                assert record.cpu_ids == allocation.cpu_ids
                assert "taskset" in record.argv[0]
                return record.cpu_ids

        with ThreadPoolExecutor(max_workers=2) as executor:
            a = executor.submit(probe, first)
            b = executor.submit(probe, second)
            assert set(a.result()).isdisjoint(b.result())
    assert os.sched_getaffinity(0) == previous
    inventory = physical_cpu_inventory()
    assert len({(item.package_id, item.core_id) for item in inventory}) == len(inventory)


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_mode": "invalid"},
        {"cpu_affinity": "invalid"},
        {"stage_limits": {"md": 0}},
        {"stage_limits": {"other": 1}},
        {"cpu_reservations": {"md": True}},
        {"cpu_reservations": {"md": 2}},
    ],
)
def test_invalid_scheduler_fields(changes):
    with pytest.raises(ConfigurationError):
        replace(SchedulerConfig(), **changes)


def test_optional_scheduler_fields_roundtrip(tmp_path):
    path = tmp_path / "runtime.yaml"
    path.write_text("""schema_version: mmpbsa.runtime.v1
providers: {}
backends: {}
scheduler:
  execution_mode: pipeline
  cpu_affinity: physical
  cpu_slots: 32
  max_workers: 13
  stage_limits: {md: 3, postprocess: 2, scoring: 8}
  cpu_reservations: {md: 12, postprocess: 2}
""")
    config = load_runtime(path).scheduler
    assert config.execution_mode == "pipeline"
    assert config.stage_limits["scoring"] == 8
    assert config.cpu_reservations["md"] == 12

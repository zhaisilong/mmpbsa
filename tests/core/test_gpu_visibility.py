from __future__ import annotations

import json
import os
import sys
from subprocess import CompletedProcess

import pytest

from mmpbsa.backends.api import CommandSpec
from mmpbsa.backends.command import SubprocessCommandRunner
from mmpbsa.backends.errors import BackendConfigurationError
from mmpbsa.core.errors import ConfigurationError
from mmpbsa.core.host_resources import HostResourcePool, initialize_registry, inspect_registry
from mmpbsa.core.models import ResourceRequest
from mmpbsa.utils.cpu import physical_cpu_inventory
from mmpbsa.utils.gpu_visibility import (
    GpuVisibilityError,
    gpu_index_uuid_inventory,
    normalize_gpu_ids,
    visible_gpu_ids,
)

GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "mmpbsa.utils.gpu_visibility.gpu_index_uuid_inventory",
        lambda: {"0": GPU_A, "1": GPU_B},
    )
    path = tmp_path / "pool.json"
    initialize_registry(path, cpu_ids=(physical_cpu_inventory()[0].cpu_id,), gpu_ids=(GPU_A, GPU_B))
    return path


def _pool(registry, gpu_ids):
    return HostResourcePool(
        registry_path=registry,
        project_id="test",
        cpu_slots=1,
        gpu_ids=gpu_ids,
        gpu_probe=lambda ids: {gpu: () for gpu in ids},
    )


def test_full_uuids_and_cpu_only_never_query_gpu_inventory(monkeypatch):
    def unexpected_query():
        raise AssertionError("must not query a GPU for full UUID or CPU-only input")

    monkeypatch.setattr("mmpbsa.utils.gpu_visibility.gpu_index_uuid_inventory", unexpected_query)
    assert normalize_gpu_ids((GPU_A.upper(),)) == (GPU_A,)
    assert normalize_gpu_ids(()) == ()
    assert visible_gpu_ids(None) is None
    assert visible_gpu_ids("") == ()
    assert visible_gpu_ids("-1") == ()


def test_integer_ids_resolve_physical_inventory_and_duplicate_aliases_fail(registry):
    assert normalize_gpu_ids(("0", "1")) == (GPU_A, GPU_B)
    assert visible_gpu_ids("0, 1") == (GPU_A, GPU_B)
    with pytest.raises(GpuVisibilityError, match="absent"):
        normalize_gpu_ids(("2",))
    with pytest.raises(GpuVisibilityError, match="duplicate"):
        normalize_gpu_ids(("0", GPU_A))


@pytest.mark.parametrize("value", ["incorrect-index", "0,", "GPU-aaaa", "MIG-GPU-aaaa", "all"])
def test_invalid_cuda_masks_fail_closed(value):
    with pytest.raises(GpuVisibilityError):
        visible_gpu_ids(value)


def test_inventory_lookup_is_read_only_and_uuid_index_based(monkeypatch):
    invocations = []

    def probe(argv, **options):
        invocations.append((argv, options))
        return CompletedProcess(argv, 0, f"0, {GPU_A}\n1, {GPU_B}\n", "")

    monkeypatch.setattr("mmpbsa.utils.gpu_visibility.subprocess.run", probe)
    assert gpu_index_uuid_inventory() == {"0": GPU_A, "1": GPU_B}
    assert invocations[0][0] == (
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    )
    assert invocations[0][1]["check"] is True
    assert invocations[0][1]["timeout"] == 5


@pytest.mark.parametrize(
    "output", ["", f"0,{GPU_A}\n0,{GPU_B}", f"0,{GPU_A}\n1,{GPU_A}", "not-an-index,not-a-uuid"]
)
def test_malformed_inventory_fails_closed(monkeypatch, output):
    monkeypatch.setattr(
        "mmpbsa.utils.gpu_visibility.subprocess.run",
        lambda argv, **_: CompletedProcess(argv, 0, output, ""),
    )
    with pytest.raises(GpuVisibilityError):
        gpu_index_uuid_inventory()


def test_host_constructor_normalizes_legacy_indices_read_only(registry, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in registry.parent.iterdir()}
    resource_pool = _pool(registry, ("0",))
    assert resource_pool.gpu_ids == (GPU_A,)
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in registry.parent.iterdir()}
    assert after == before


@pytest.mark.parametrize("parent_mask", ["", "-1", "incorrect-index", GPU_B, "1"])
def test_constructor_cannot_expand_parent_visible_authority(registry, monkeypatch, parent_mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", parent_mask)
    before = registry.read_bytes(), registry.stat().st_mtime_ns
    with pytest.raises(ConfigurationError):
        _pool(registry, (GPU_A,))
    assert (registry.read_bytes(), registry.stat().st_mtime_ns) == before


@pytest.mark.parametrize("command_mask", ["", "-1", "incorrect-index", GPU_B, "1"])
def test_command_visibility_conflict_rejected_before_native_launch(
    registry, monkeypatch, command_mask
):
    resource_pool = _pool(registry, (GPU_A,))
    allocation = resource_pool.try_acquire(ResourceRequest(1, 1))
    with resource_pool.observe(allocation, "native"):
        with pytest.raises(BackendConfigurationError):
            SubprocessCommandRunner().run(
                CommandSpec(
                    (sys.executable, "-c", "raise AssertionError('must not launch')"),
                    registry.parent,
                    environment={"CUDA_VISIBLE_DEVICES": command_mask},
                )
            )
    assert not inspect_registry(registry)["leases"][allocation.lease_token]["launches"]
    resource_pool.release(allocation)


def test_command_mask_cannot_hide_changed_parent_restriction(registry, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_A)
    resource_pool = _pool(registry, (GPU_A,))
    allocation = resource_pool.try_acquire(ResourceRequest(1, 1))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_B)
    with resource_pool.observe(allocation, "native"):
        with pytest.raises(BackendConfigurationError, match="current parent"):
            SubprocessCommandRunner().run(
                CommandSpec(
                    (sys.executable, "-c", "raise AssertionError('must not launch')"),
                    registry.parent,
                    environment={"CUDA_VISIBLE_DEVICES": GPU_A},
                )
            )
    assert not inspect_registry(registry)["leases"][allocation.lease_token]["launches"]
    resource_pool.release(allocation)


def test_valid_numeric_command_mask_becomes_child_uuid_without_mutating_parent(
    registry, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    resource_pool = _pool(registry, ("0",))
    allocation = resource_pool.try_acquire(ResourceRequest(1, 1))
    with resource_pool.observe(allocation, "native"):
        record = SubprocessCommandRunner().run(
            CommandSpec(
                (
                    sys.executable,
                    "-c",
                    "import os,json;print(json.dumps(os.environ['CUDA_VISIBLE_DEVICES']))",
                ),
                registry.parent,
                environment={"CUDA_VISIBLE_DEVICES": "0"},
            )
        )
    assert json.loads(record.stdout) == GPU_A
    assert allocation.gpu_ids == ("0",)
    resource_pool.release(allocation)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_cpu_only_pool_and_commands_ignore_invalid_gpu_mask_without_query(registry, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "incorrect-index")

    def unexpected_query():
        raise AssertionError("CPU stages must not query GPUs")

    monkeypatch.setattr("mmpbsa.utils.gpu_visibility.gpu_index_uuid_inventory", unexpected_query)
    resource_pool = _pool(registry, ())
    allocation = resource_pool.try_acquire(ResourceRequest(1))
    with resource_pool.observe(allocation, "cpu-native"):
        SubprocessCommandRunner().run(
            CommandSpec((sys.executable, "-c", "print('cpu')"), registry.parent)
        )
    resource_pool.release(allocation)

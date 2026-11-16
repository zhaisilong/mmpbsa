from __future__ import annotations

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mmpbsa.backends.api import CommandSpec
from mmpbsa.backends.command import SubprocessCommandRunner
from mmpbsa.backends.errors import BackendExecutionError
from mmpbsa.core.config import ResourceCoordinationConfig, SchedulerConfig, load_runtime
from mmpbsa.core.errors import ConfigurationError, ContractError
from mmpbsa.core.host_registry import _locked, _state_digest
from mmpbsa.core.host_resources import (
    HostResourcePool,
    initialize_registry,
    inspect_registry,
    reconcile_registry,
)
from mmpbsa.core.models import ResourceRequest, TaskKey
from mmpbsa.core.state import StageId
from mmpbsa.utils.cpu import PhysicalCpu, physical_cpu_inventory
from mmpbsa.utils.process_identity import (
    current_process_observer,
    process_group_alive,
    process_identity,
)

GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
def policy(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    inventory = tuple(PhysicalCpu(i, 0, i, i // 2) for i in range(4))
    monkeypatch.setattr("mmpbsa.core.host_resources.physical_cpu_inventory", lambda: inventory)
    monkeypatch.setattr("mmpbsa.core.host_registry.physical_cpu_inventory", lambda: inventory)
    monkeypatch.setattr("mmpbsa.core.host_registry._topology_key", lambda cpu: (0, cpu % 4))
    monkeypatch.setattr(
        "mmpbsa.core.host_resources._allowed_topology", lambda _: {c.cpu_id: c for c in inventory}
    )
    monkeypatch.setattr(
        "mmpbsa.core.host_registry._allowed_topology", lambda _: {c.cpu_id: c for c in inventory}
    )
    path = tmp_path / "pool.json"
    initialize_registry(path, cpu_ids=(0, 1, 2, 3), gpu_ids=(GPU_A, GPU_B))
    return path


def pool(path, project="one", *, cpus=4, gpus=(), probe=lambda ids: {gpu: () for gpu in ids}):
    return HostResourcePool(
        registry_path=path,
        project_id=project,
        cpu_slots=cpus,
        gpu_ids=gpus,
        gpu_probe=probe,
    )


def test_read_only_construction_inspect_and_default_reconcile(policy):
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in policy.parent.iterdir()}
    resource_pool = pool(policy)
    assert resource_pool.cpu_ids == (0, 1, 2, 3)
    assert inspect_registry(policy)["leases"] == {}
    assert reconcile_registry(policy)["reconciliation"]["reclaimable"] == []
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in policy.parent.iterdir()}
    assert before == after
    with pytest.raises(ConfigurationError, match="already exists"):
        initialize_registry(policy, cpu_ids=(0, 1))


def test_missing_registry_is_not_initialized_or_locked(tmp_path):
    path = tmp_path / "missing" / "pool.json"
    with pytest.raises(ConfigurationError):
        pool(path)
    with pytest.raises(ConfigurationError):
        inspect_registry(path)
    assert not path.parent.exists()


def test_full_candidate_mask_project_caps_and_atomic_cpu_gpu(policy):
    first = pool(policy, "one", cpus=2, gpus=(GPU_A,))
    second = pool(policy, "two", cpus=2, gpus=(GPU_A, GPU_B))
    allocation = first.try_acquire(ResourceRequest(2, 1), request_key="md-one")
    assert allocation.physical_gpu_ids == (GPU_A,)
    assert allocation.gpu_ids == ("0",)
    assert allocation.cpu_ids == (0, 1)
    assert first.try_acquire(ResourceRequest(1), request_key="scoring") is None
    other = second.try_acquire(ResourceRequest(2, 1), request_key="md-two")
    assert other.physical_gpu_ids == (GPU_B,)
    assert other.cpu_ids == (2, 3)
    assert set(allocation.cpu_ids).isdisjoint(other.cpu_ids)
    assert len(inspect_registry(policy)["leases"]) == 2
    first.release(allocation)
    second.release(other)
    first.cancel_pending()
    assert inspect_registry(policy)["leases"] == {}
    with pytest.raises(ContractError, match="already released"):
        first.release(allocation)


def test_cpu_not_taken_while_gpu_busy_or_observation_unknown(policy):
    busy = pool(policy, gpus=(GPU_A,), probe=lambda ids: {GPU_A: (9999,)})
    assert busy.try_acquire(ResourceRequest(2, 1)) is None
    assert busy.last_wait.reason == "external_gpu_process"
    assert not inspect_registry(policy)["leases"]
    cpu = pool(
        policy, "cpu-only", probe=lambda ids: {gpu: (9999,) if gpu == GPU_A else () for gpu in ids}
    )
    allocation = cpu.try_acquire(ResourceRequest(4))
    assert allocation is not None
    cpu.release(allocation)
    busy.cancel_pending()

    def unavailable(ids):
        raise RuntimeError("cannot observe")

    unavailable_pool = pool(policy, "unavailable", gpus=(GPU_A,), probe=unavailable)
    assert unavailable_pool.try_acquire(ResourceRequest(1, 1)) is None
    assert unavailable_pool.last_wait.reason == "gpu_observation_unavailable"
    assert not inspect_registry(policy)["leases"]


def test_round_robin_projects_without_preempting(policy):
    first, second = pool(policy, "one"), pool(policy, "two")
    occupied = first.try_acquire(ResourceRequest(4), request_key="first")
    assert second.try_acquire(ResourceRequest(4), request_key="second") is None
    assert first.try_acquire(ResourceRequest(4), request_key="next") is None
    first.release(occupied)
    assert first.try_acquire(ResourceRequest(4), request_key="next") is None
    assert first.last_wait.reason == "host_fairness"
    granted = second.try_acquire(ResourceRequest(4), request_key="second")
    assert granted is not None
    assert first.try_acquire(ResourceRequest(4), request_key="next") is None
    second.release(granted)
    granted = first.try_acquire(ResourceRequest(4), request_key="next")
    assert granted is not None
    first.release(granted)


def test_unknown_controller_waiter_does_not_monopolize_grants(policy, monkeypatch):
    first, unknown = pool(policy, "one"), pool(policy, "unknown")
    allocation = first.try_acquire(ResourceRequest(4))
    assert unknown.try_acquire(ResourceRequest(4), request_key="unknown-stage") is None
    inaccessible_pid = 987654321
    with _locked(policy) as state:
        next(iter(state["waiters"].values()))["controller"]["pid"] = inaccessible_pid
    real_identity_alive = __import__(
        "mmpbsa.utils.process_identity", fromlist=["identity_alive"]
    ).identity_alive

    def evidence(identity):
        return None if identity["pid"] == inaccessible_pid else real_identity_alive(identity)

    monkeypatch.setattr("mmpbsa.core.host_registry.identity_alive", evidence)
    monkeypatch.setattr("mmpbsa.core.host_resources.identity_alive", evidence)
    first.release(allocation)
    other = pool(policy, "other")
    granted = other.try_acquire(ResourceRequest(4))
    assert granted is not None
    assert inspect_registry(policy)["waiters"]
    other.release(granted)


def test_semantically_wrong_cpu_alias_rejected_even_with_valid_state_seal(policy):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(2))
    state = json.loads(policy.read_text())
    state["leases"][allocation.lease_token]["cpu_ids"] = [1, 0]
    state["integrity_sha256"] = _state_digest(state)
    policy.write_text(json.dumps(state))
    with pytest.raises(ConfigurationError, match="aliases"):
        inspect_registry(policy)
    with pytest.raises(ConfigurationError, match="aliases"):
        resource_pool.try_acquire(ResourceRequest(1))


def test_same_project_aggregate_cap_and_policy_cannot_expand_while_active(policy):
    first = pool(policy, "shared", cpus=2)
    second = pool(policy, "shared", cpus=2)
    allocation = first.try_acquire(ResourceRequest(2))
    assert second.try_acquire(ResourceRequest(1), request_key="wait") is None
    with pytest.raises(ConfigurationError, match="policy changed"):
        pool(policy, "shared", cpus=4).try_acquire(ResourceRequest(1))
    second.cancel_request("wait")
    assert not inspect_registry(policy)["waiters"]
    first.release(allocation)


def test_project_stage_priority_and_fifo_survive_sorted_json(policy):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(4), request_key="running")
    # Reverse lexical task order; serialized dictionary order is irrelevant.
    assert resource_pool.try_acquire(ResourceRequest(4), request_key="zz-gb", priority=2) is None
    assert resource_pool.try_acquire(ResourceRequest(4), request_key="aa-pb", priority=2) is None
    assert resource_pool.try_acquire(ResourceRequest(4), request_key="md", priority=0) is None
    resource_pool.release(allocation)
    assert resource_pool.try_acquire(ResourceRequest(4), request_key="zz-gb", priority=2) is None
    allocation = resource_pool.try_acquire(ResourceRequest(4), request_key="md", priority=0)
    assert allocation is not None
    resource_pool.release(allocation)
    assert resource_pool.try_acquire(ResourceRequest(4), request_key="aa-pb", priority=2) is None
    allocation = resource_pool.try_acquire(ResourceRequest(4), request_key="zz-gb", priority=2)
    assert allocation is not None
    resource_pool.release(allocation)
    resource_pool.cancel_pending()


def test_smt_sibling_affinity_uses_core_key_not_logical_id(policy, monkeypatch):
    first = pool(policy, "one", cpus=2)
    allocation = first.try_acquire(ResourceRequest(2))
    sibling_inventory = tuple(PhysicalCpu(i + 4, 0, i, i // 2) for i in range(4))
    monkeypatch.setattr(
        "mmpbsa.core.host_resources._allowed_topology",
        lambda _: {c.cpu_id: c for c in sibling_inventory},
    )
    other = pool(policy, "two", cpus=2)
    sibling_allocation = other.try_acquire(ResourceRequest(2))
    assert sibling_allocation.cpu_ids == (6, 7)
    assert not set(allocation.cpu_ids) & set(sibling_allocation.cpu_ids)
    state = inspect_registry(policy)
    core_keys = [tuple(k) for lease in state["leases"].values() for k in lease["core_keys"]]
    assert len(set(core_keys)) == 4
    first.release(allocation)
    other.release(sibling_allocation)


def test_pid_reuse_and_inaccessible_identity_never_release(policy, monkeypatch):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(1))
    with resource_pool.observe(allocation, "unverifiable"):
        observer = current_process_observer()
        observer.launching("child")
        observer.started("child", process_identity())
    monkeypatch.setattr("mmpbsa.core.host_resources.process_group_alive", lambda _: None)
    monkeypatch.setattr("mmpbsa.core.host_registry.process_group_alive", lambda _: None)
    resource_pool.release(allocation)
    reconcile_registry(policy, confirm=True)
    assert inspect_registry(policy)["leases"][allocation.lease_token]["state"] == "unknown-held"


def test_pending_launch_is_quarantined_even_with_ancient_heartbeat(policy):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(4))
    with resource_pool.observe(allocation, "unknown-invocation"):
        current_process_observer().launching("pending-before-fork")
    resource_pool.release(allocation)
    state = inspect_registry(policy)
    assert state["leases"][allocation.lease_token]["state"] == "unknown-held"
    with _locked(policy) as raw:
        raw["leases"][allocation.lease_token]["heartbeat_at"] = 0
    reconcile_registry(policy, confirm=True)
    assert allocation.lease_token in inspect_registry(policy)["leases"]
    assert pool(policy, "other").try_acquire(ResourceRequest(1)) is None


def test_observed_command_persists_identity_and_uuid_environment(policy, tmp_path):
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    resource_pool = pool(policy, gpus=(GPU_A,))
    allocation = resource_pool.try_acquire(ResourceRequest(1, 1))
    with resource_pool.observe(allocation, "completed-command"):
        record = SubprocessCommandRunner().run(
            CommandSpec(
                (sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"),
                tmp_path,
                environment={"CUDA_VISIBLE_DEVICES": GPU_A},
            )
        )
    assert record.stdout.strip() == GPU_A
    launch = next(
        iter(inspect_registry(policy)["leases"][allocation.lease_token]["launches"].values())
    )
    assert launch["state"] == "finished"
    assert launch["identity"]["pid"] > 0
    assert launch["identity"]["boot_id"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous_cuda
    resource_pool.release(allocation)
    assert not inspect_registry(policy)["leases"]


def test_lease_attaches_exact_attempt_invocation_once(policy):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(1))
    task = TaskKey(stage=StageId.MD, case_id="compound", replica_id="rep01")
    resource_pool.attach_attempt(allocation, task, "attempt001", "/workspace/invocation001")
    context = inspect_registry(policy)["leases"][allocation.lease_token]["attempt"]
    assert context["task"] == task.label
    assert context["stage"] == "md"
    assert context["replica_id"] == "rep01"
    assert context["attempt_id"] == "attempt001"
    assert context["invocation_id"] == "/workspace/invocation001"
    resource_pool.attach_attempt(allocation, task, "attempt001", "/workspace/invocation001")
    with pytest.raises(ContractError, match="already attached"):
        resource_pool.attach_attempt(allocation, task, "attempt002", "/workspace/invocation002")
    resource_pool.release(allocation)


def test_timeout_also_terminates_descendant_with_changed_process_group(policy, tmp_path):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(1))
    with resource_pool.observe(allocation, "changed-group"):
        with pytest.raises(BackendExecutionError):
            SubprocessCommandRunner().run(
                CommandSpec(
                    (
                        sys.executable,
                        "-c",
                        "import subprocess,sys,time; "
                        "subprocess.Popen([sys.executable,'-c',"
                        "'import os,time;os.setpgid(0,0);time.sleep(30)']); time.sleep(30)",
                    ),
                    tmp_path,
                    timeout_seconds=0.1,
                )
            )
    launch = next(
        iter(inspect_registry(policy)["leases"][allocation.lease_token]["launches"].values())
    )
    assert process_group_alive(launch["identity"]) is False
    resource_pool.release(allocation)
    assert not inspect_registry(policy)["leases"]


def test_detached_setsid_child_retains_lease_after_launcher_exits(policy, tmp_path):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(4))
    with resource_pool.observe(allocation, "detached-launcher"):
        record = SubprocessCommandRunner().run(
            CommandSpec(
                (
                    sys.executable,
                    "-c",
                    "import subprocess,sys,time; "
                    "p=subprocess.Popen([sys.executable,'-c',"
                    "'import os,time;os.setsid();time.sleep(30)'],"
                    "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                    "print(p.pid,flush=True);time.sleep(0.05)",
                ),
                tmp_path,
            )
        )
    child_identity = process_identity(int(record.stdout.strip()))
    try:
        launch = next(
            iter(inspect_registry(policy)["leases"][allocation.lease_token]["launches"].values())
        )
        assert process_group_alive(launch["identity"]) is False
        assert child_identity["session_id"] != launch["identity"]["session_id"]
        resource_pool.release(allocation)
        assert inspect_registry(policy)["leases"][allocation.lease_token]["state"] == "orphan-held"
        reconcile_registry(policy, confirm=True)
        assert allocation.lease_token in inspect_registry(policy)["leases"]
        assert pool(policy, "next").try_acquire(ResourceRequest(1)) is None
    finally:
        if process_group_alive(child_identity) is True:
            os.killpg(child_identity["pgid"], signal.SIGTERM)
        for _ in range(100):
            if process_group_alive(child_identity) is False:
                break
            time.sleep(0.01)
        reconcile_registry(policy, confirm=True)
    assert not inspect_registry(policy)["leases"]


def test_timeout_terminates_token_child_that_detaches_to_new_session(policy, tmp_path):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(1))
    with resource_pool.observe(allocation, "timeout-detached"):
        with pytest.raises(BackendExecutionError):
            SubprocessCommandRunner().run(
                CommandSpec(
                    (
                        sys.executable,
                        "-c",
                        "import subprocess,sys,time; "
                        "subprocess.Popen([sys.executable,'-c',"
                        "'import os,time;os.setsid();time.sleep(30)'],"
                        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(30)",
                    ),
                    tmp_path,
                    timeout_seconds=0.15,
                )
            )
    resource_pool.release(allocation)
    assert not inspect_registry(policy)["leases"]


def test_failed_launch_has_positive_no_child_evidence(policy, tmp_path):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(1))
    with resource_pool.observe(allocation, "failed-exec"):
        with pytest.raises(BackendExecutionError):
            SubprocessCommandRunner().run(
                CommandSpec(("/definitely/missing/executable",), tmp_path)
            )
    assert not inspect_registry(policy)["leases"][allocation.lease_token]["launches"]
    resource_pool.release(allocation)
    assert not inspect_registry(policy)["leases"]


def test_timeout_terminates_own_group_and_not_unrelated_process(policy, tmp_path):
    unrelated = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(30)"), start_new_session=True
    )
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(1))
    try:
        with resource_pool.observe(allocation, "timed-out"):
            with pytest.raises(BackendExecutionError):
                SubprocessCommandRunner().run(
                    CommandSpec(
                        (
                            sys.executable,
                            "-c",
                            "import subprocess,sys,time; "
                            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                            "time.sleep(30)",
                        ),
                        tmp_path,
                        timeout_seconds=0.1,
                    )
                )
        assert unrelated.poll() is None
        launch = next(
            iter(inspect_registry(policy)["leases"][allocation.lease_token]["launches"].values())
        )
        assert process_group_alive(launch["identity"]) is False
        resource_pool.release(allocation)
        assert not inspect_registry(policy)["leases"]
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_live_native_child_retained_on_release_and_reconciled_after_exit(policy):
    resource_pool = pool(policy)
    allocation = resource_pool.try_acquire(ResourceRequest(4))
    process = subprocess.Popen(
        (sys.executable, "-c", "import time;time.sleep(30)"), start_new_session=True
    )
    try:
        with resource_pool.observe(allocation, "orphan"):
            observer = current_process_observer()
            observer.launching("native-child")
            observer.started("native-child", process_identity(process.pid))
        resource_pool.release(allocation)
        reconcile_registry(policy, confirm=True)
        assert inspect_registry(policy)["leases"][allocation.lease_token]["state"] == "orphan-held"
        assert pool(policy, "other").try_acquire(ResourceRequest(1)) is None
        process.terminate()
        process.wait(timeout=5)
        reconcile_registry(policy, confirm=True)
        assert not inspect_registry(policy)["leases"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s.update(schema_version="bad"),
        lambda s: s["host"].update(uid=-1),
        lambda s: s["host"].update(boot_id="different-boot"),
        lambda s: s["policy"]["gpu_ids"].append("0"),
        lambda s: s.update(leases=[]),
    ],
)
def test_corrupt_registry_fails_closed_without_rewriting(policy, mutator):
    state = json.loads(policy.read_text())
    mutator(state)
    policy.write_text(json.dumps(state))
    before = policy.read_bytes()
    with pytest.raises(ConfigurationError):
        pool(policy)
    with pytest.raises(ConfigurationError):
        reconcile_registry(policy, confirm=True)
    assert policy.read_bytes() == before


def test_smt_policy_duplicate_rejected(tmp_path, monkeypatch):
    topology = {0: PhysicalCpu(0, 0, 0, 0), 4: PhysicalCpu(4, 0, 0, 0)}
    monkeypatch.setattr(
        "mmpbsa.core.host_resources.physical_cpu_inventory", lambda: tuple(topology.values())
    )
    monkeypatch.setattr(
        "mmpbsa.core.host_registry.physical_cpu_inventory", lambda: tuple(topology.values())
    )
    monkeypatch.setattr("mmpbsa.core.host_resources._allowed_topology", lambda _: topology)
    monkeypatch.setattr("mmpbsa.core.host_registry._allowed_topology", lambda _: topology)
    with pytest.raises(ConfigurationError, match="SMT"):
        initialize_registry(tmp_path / "pool.json", cpu_ids=(0, 4))


def test_host_runtime_contract_and_legacy_default(tmp_path):
    assert SchedulerConfig().resource_coordination.mode == "local"
    with pytest.raises(ConfigurationError, match="absolute"):
        ResourceCoordinationConfig(mode="host", registry_path=Path("relative.json"))
    with pytest.raises(ConfigurationError, match="pipeline"):
        SchedulerConfig(
            resource_coordination=ResourceCoordinationConfig(
                mode="host", registry_path=tmp_path / "pool.json"
            )
        )
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(f"""schema_version: mmpbsa.runtime.v1
providers: {{}}
backends: {{}}
scheduler:
  execution_mode: pipeline
  cpu_affinity: physical
  resource_coordination:
    mode: host
    registry_path: {tmp_path / "pool.json"}
""")
    scheduler = load_runtime(runtime).scheduler
    assert scheduler.resource_coordination.mode == "host"
    assert scheduler.resource_coordination.registry_path == tmp_path / "pool.json"


def _multiprocess_lease(path, project, cpus, granted, release):
    resource_pool = pool(Path(path), project, cpus=cpus)
    allocation = None
    deadline = time.monotonic() + 10
    while allocation is None and time.monotonic() < deadline:
        allocation = resource_pool.try_acquire(ResourceRequest(cpus), request_key="stage")
        if allocation is None:
            time.sleep(0.01)
    if allocation is None:
        granted.put((project, None))
        return
    granted.put((project, allocation.cpu_ids))
    release.wait(10)
    resource_pool.release(allocation)
    resource_pool.cancel_pending()


def test_real_multiprocess_atomic_grants_do_not_overlap(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux host leases")
    cores = physical_cpu_inventory()
    if len(cores) < 2:
        pytest.skip("two physical cores required")
    path = tmp_path / "pool.json"
    initialize_registry(path, cpu_ids=tuple(c.cpu_id for c in cores[:2]))
    context = multiprocessing.get_context("fork")
    granted, release = context.Queue(), context.Event()
    workers = [
        context.Process(target=_multiprocess_lease, args=(str(path), f"p{i}", 1, granted, release))
        for i in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        allocations = [granted.get(timeout=10) for _ in workers]
        assert all(ids for _, ids in allocations)
        assert set(allocations[0][1]).isdisjoint(allocations[1][1])
        assert len(inspect_registry(path)["leases"]) == 2
        release.set()
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert not inspect_registry(path)["leases"]
    finally:
        release.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def _controller_with_orphan(path, granted):
    resource_pool = pool(Path(path), "dead-controller", cpus=1)
    allocation = resource_pool.try_acquire(ResourceRequest(1))
    child = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    identity = process_identity(child.pid)
    with resource_pool.observe(allocation, "native"):
        observer = current_process_observer()
        observer.launching("child")
        observer.started("child", identity)
    granted.put((allocation.lease_token, identity))


def test_dead_controller_does_not_release_live_native_child(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux host leases")
    cpu = physical_cpu_inventory()[0].cpu_id
    path = tmp_path / "pool.json"
    initialize_registry(path, cpu_ids=(cpu,))
    context = multiprocessing.get_context("fork")
    granted = context.Queue()
    controller = context.Process(target=_controller_with_orphan, args=(str(path), granted))
    controller.start()
    token, identity = granted.get(timeout=10)
    try:
        controller.join(timeout=10)
        assert controller.exitcode == 0
        reconcile_registry(path, confirm=True)
        assert inspect_registry(path)["leases"][token]["state"] == "orphan-held"
        assert pool(path, "next", cpus=1).try_acquire(ResourceRequest(1)) is None
    finally:
        if process_group_alive(identity) is True:
            os.killpg(identity["pgid"], signal.SIGTERM)
        for _ in range(100):
            if process_group_alive(identity) is False:
                break
            time.sleep(0.01)
        reconcile_registry(path, confirm=True)
    assert not inspect_registry(path)["leases"]

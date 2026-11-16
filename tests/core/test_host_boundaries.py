"""Declared host/namespace/local-filesystem boundaries are enforced read-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mmpbsa.core.errors import ConfigurationError
from mmpbsa.core.host_registry import _host_identity, _require_local_filesystem, _state_digest
from mmpbsa.core.host_resources import initialize_registry, inspect_registry, reconcile_registry
from mmpbsa.utils.cpu import physical_cpu_inventory

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux host registry")


def _escape(path: Path) -> str:
    return str(path).replace("\\", "\\134").replace(" ", "\\040").replace("\t", "\\011")


def _mounts(path: Path, filesystem: str, *, root_filesystem: str = "ext4") -> str:
    return (
        f"1 0 0:1 / / rw - {root_filesystem} root-device rw\n"
        f"2 1 0:2 / {_escape(path)} rw - {filesystem} mounted-device rw\n"
    )


def _inventory(directory: Path):
    return {
        item.name: (item.stat().st_mtime_ns, item.read_bytes())
        for item in directory.iterdir()
        if item.is_file()
    }


def test_registry_seals_actual_pid_namespace(tmp_path):
    path = tmp_path / "pool.json"
    initialize_registry(path, cpu_ids=(physical_cpu_inventory()[0].cpu_id,))
    host = inspect_registry(path)["host"]
    assert host["pid_namespace_inode"] == Path("/proc/self/ns/pid").stat().st_ino
    assert host == _host_identity()


def test_different_pid_namespace_rejected_even_same_host_uid_boot(tmp_path):
    path = tmp_path / "pool.json"
    initialize_registry(path, cpu_ids=(physical_cpu_inventory()[0].cpu_id,))
    state = json.loads(path.read_text())
    state["host"]["pid_namespace_inode"] += 1
    state["integrity_sha256"] = _state_digest(state)
    path.write_text(json.dumps(state))
    before = _inventory(tmp_path)
    with pytest.raises(ConfigurationError, match="PID namespace"):
        inspect_registry(path)
    with pytest.raises(ConfigurationError, match="PID namespace"):
        reconcile_registry(path, confirm=True)
    assert _inventory(tmp_path) == before


@pytest.mark.parametrize(
    "filesystem",
    ["nfs", "nfs4", "cifs", "smb", "fuse.sshfs", "ceph", "lustre", "gpfs", "beegfs", "9p"],
)
def test_network_registry_initialization_rejected_before_any_lock_or_file(
    tmp_path,
    monkeypatch,
    filesystem,
):
    monkeypatch.setattr(
        "mmpbsa.core.host_registry._mountinfo_text", lambda: _mounts(tmp_path, filesystem)
    )
    with pytest.raises(ConfigurationError, match=f"local filesystem, not {filesystem}"):
        initialize_registry(tmp_path / "pool.json", cpu_ids=(0,))
    assert list(tmp_path.iterdir()) == []


def test_network_inspect_and_confirmed_reconcile_are_nonmutating(tmp_path, monkeypatch):
    path = tmp_path / "pool.json"
    initialize_registry(path, cpu_ids=(physical_cpu_inventory()[0].cpu_id,))
    before = _inventory(tmp_path)
    monkeypatch.setattr(
        "mmpbsa.core.host_registry._mountinfo_text", lambda: _mounts(tmp_path, "nfs4")
    )
    with pytest.raises(ConfigurationError, match="local filesystem"):
        inspect_registry(path)
    with pytest.raises(ConfigurationError, match="local filesystem"):
        reconcile_registry(path, confirm=True)
    assert _inventory(tmp_path) == before


def test_longest_mount_and_escaped_mountpoint_choose_local_submount(tmp_path, monkeypatch):
    directory = tmp_path / "local pool with spaces"
    directory.mkdir()
    monkeypatch.setattr(
        "mmpbsa.core.host_registry._mountinfo_text",
        lambda: _mounts(directory, "tmpfs", root_filesystem="nfs4"),
    )
    path = directory / "pool.json"
    initialize_registry(path, cpu_ids=(physical_cpu_inventory()[0].cpu_id,))
    before = _inventory(directory)
    assert inspect_registry(path)["leases"] == {}
    assert _inventory(directory) == before


def test_parent_symlink_cannot_disguise_network_mount(tmp_path, monkeypatch):
    target = tmp_path / "network"
    target.mkdir()
    alias = tmp_path / "local-alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        "mmpbsa.core.host_registry._mountinfo_text", lambda: _mounts(target, "cifs")
    )
    with pytest.raises(ConfigurationError, match="not cifs"):
        initialize_registry(alias / "pool.json", cpu_ids=(0,))
    assert list(target.iterdir()) == []


def test_lexical_prefix_is_not_a_network_mount_ancestor(tmp_path, monkeypatch):
    mountpoint = tmp_path / "network"
    candidate = tmp_path / "network-actually-local" / "pool.json"
    monkeypatch.setattr(
        "mmpbsa.core.host_registry._mountinfo_text", lambda: _mounts(mountpoint, "nfs")
    )
    _require_local_filesystem(candidate)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mountinfo", ["", "malformed", "1 0 0:1 / / rw - ext4"])
def test_unverifiable_mountinfo_fails_closed_without_writes(tmp_path, monkeypatch, mountinfo):
    monkeypatch.setattr("mmpbsa.core.host_registry._mountinfo_text", lambda: mountinfo)
    with pytest.raises(ConfigurationError, match="cannot verify local registry filesystem"):
        initialize_registry(tmp_path / "pool.json", cpu_ids=(0,))
    assert list(tmp_path.iterdir()) == []

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from mmpbsa.utils import (
    FileLock,
    LockOwnershipError,
    LockTimeoutError,
    inspect_file_lock,
)

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


class FileLockTests(unittest.TestCase):
    def test_context_manager_records_metadata_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.lock"
            lock = FileLock(
                path,
                owner="test-owner",
                metadata={"case": "case-a"},
            )
            with lock as acquired:
                self.assertIs(acquired, lock)
                self.assertTrue(lock.held)
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(record["owner"], "test-owner")
                self.assertEqual(record["metadata"], {"case": "case-a"})
                inspection = inspect_file_lock(path)
                self.assertTrue(inspection.exists)
                self.assertFalse(inspection.stale)
                self.assertEqual(inspection.reason, "owner_process_running")
            self.assertFalse(lock.held)
            self.assertFalse(path.exists())

    def test_contended_lock_times_out_without_breaking_live_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.lock"
            with FileLock(path, owner="first"):
                with self.assertRaises(LockTimeoutError):
                    FileLock(
                        path,
                        owner="second",
                        timeout_seconds=0.03,
                        poll_interval_seconds=0.005,
                        stale_after_seconds=0,
                    ).acquire()

    def test_lock_excludes_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.lock"
            script = """
import sys
from mmpbsa.utils import FileLock, LockTimeoutError
try:
    FileLock(sys.argv[1], timeout_seconds=0.03, poll_interval_seconds=0.005).acquire()
except LockTimeoutError:
    raise SystemExit(0)
raise SystemExit(9)
"""
            environment = os.environ.copy()
            inherited_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(SOURCE_ROOT)
                if not inherited_pythonpath
                else os.pathsep.join((str(SOURCE_ROOT), inherited_pythonpath))
            )
            with FileLock(path, owner="parent"):
                completed = subprocess.run(
                    [sys.executable, "-c", script, str(path)],
                    check=False,
                    env=environment,
                )
            self.assertEqual(completed.returncode, 0)

    def test_dead_local_owner_is_detected_and_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.lock"
            record = {
                "schema_version": "mmpbsa.utils.file-lock.v1",
                "owner": "dead",
                "token": "old-token",
                "pid": 2_000_000_000,
                "hostname": socket.gethostname(),
                "acquired_at": "2020-01-01T00:00:00.000000Z",
                "acquired_epoch": 1.0,
                "process_identity": None,
                "metadata": {},
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            inspection = inspect_file_lock(path)
            self.assertTrue(inspection.stale)
            self.assertEqual(inspection.reason, "owner_process_not_running")
            with FileLock(path, owner="replacement") as lock:
                self.assertEqual(lock.lock_metadata.owner, "replacement")

    def test_malformed_lock_requires_stale_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.lock"
            path.write_text("{", encoding="utf-8")
            self.assertFalse(inspect_file_lock(path).stale)
            old = time.time() - 10
            os.utime(path, (old, old))
            self.assertTrue(
                inspect_file_lock(path, stale_after_seconds=1).stale
            )

    def test_refresh_keeps_token_and_updates_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.lock"
            lock = FileLock(path).acquire()
            original = lock.lock_metadata
            time.sleep(0.001)
            lock.refresh()
            refreshed = lock.lock_metadata
            self.assertEqual(original.token, refreshed.token)
            self.assertGreater(refreshed.acquired_epoch, original.acquired_epoch)
            lock.release()

    def test_release_refuses_changed_owner_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.lock"
            lock = FileLock(path).acquire()
            record = json.loads(path.read_text(encoding="utf-8"))
            record["token"] = "replacement-token"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(LockOwnershipError):
                lock.release()
            path.unlink()

    def test_rejects_non_finite_timing_values(self) -> None:
        with self.assertRaises(ValueError):
            FileLock("unused.lock", timeout_seconds=float("nan"))
        with self.assertRaises(ValueError):
            FileLock("unused.lock", poll_interval_seconds=float("inf"))


if __name__ == "__main__":
    unittest.main()

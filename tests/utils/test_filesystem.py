from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import mmpbsa.utils.filesystem as filesystem
from mmpbsa.utils import (
    AtomicReplaceError,
    UnsafePathError,
    atomic_directory,
    atomic_replace_directory,
    atomic_write_json,
    atomic_write_text,
    materialize_immutable_file,
    resolve_within,
    safe_relative_path,
)


class SafePathTests(unittest.TestCase):
    def test_resolve_within_accepts_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "a" / "value.txt"
            child.parent.mkdir()
            child.write_text("ok", encoding="utf-8")
            self.assertEqual(
                resolve_within(root, "a/value.txt", must_exist=True),
                child.resolve(),
            )

    def test_resolve_within_rejects_absolute_and_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(UnsafePathError):
                resolve_within(root, "../outside")
            with self.assertRaises(UnsafePathError):
                resolve_within(root, root / "absolute")
            with self.assertRaises(UnsafePathError):
                safe_relative_path("inside/../../outside")

    def test_resolve_within_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                (root / "link").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")
            with self.assertRaises(UnsafePathError):
                resolve_within(root, "link/value.txt")


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_text_and_json_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = atomic_write_text(root / "nested" / "value.txt", "hello")
            self.assertEqual(text_path.read_text(encoding="utf-8"), "hello")
            json_path = atomic_write_json(root / "state.json", {"b": 2, "a": 1})
            self.assertEqual(json.loads(json_path.read_text()), {"a": 1, "b": 2})
            self.assertTrue(json_path.read_text(encoding="utf-8").endswith("\n"))

    def test_failed_file_replace_preserves_old_target_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "value.txt"
            target.write_text("old", encoding="utf-8")
            with mock.patch.object(
                filesystem.os,
                "replace",
                side_effect=OSError("injected"),
            ):
                with self.assertRaises(OSError):
                    atomic_write_text(target, "new")
            self.assertEqual(target.read_text(encoding="utf-8"), "old")
            self.assertEqual(
                [path for path in target.parent.iterdir() if path != target],
                [],
            )

    def test_immutable_materialization_is_verified_and_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "nested" / "destination.bin"
            source.write_bytes(b"immutable payload")

            self.assertEqual(
                materialize_immutable_file(source, destination),
                destination,
            )
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            destination.write_bytes(b"changed")
            self.assertEqual(source.read_bytes(), b"immutable payload")


class AtomicDirectoryTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "directory exchange requires Linux")
    def test_replace_consumes_staging_and_replaces_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "published"
            staging = root / "staging"
            target.mkdir()
            staging.mkdir()
            (target / "value").write_text("old", encoding="utf-8")
            (staging / "value").write_text("new", encoding="utf-8")
            self.assertEqual(atomic_replace_directory(staging, target), target)
            self.assertFalse(staging.exists())
            self.assertEqual((target / "value").read_text(), "new")
            self.assertEqual(tuple(root.iterdir()), (target,))

    @unittest.skipUnless(sys.platform == "linux", "directory exchange requires Linux")
    def test_publish_failure_rolls_back_previous_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "published"
            staging = root / "staging"
            target.mkdir()
            staging.mkdir()
            (target / "value").write_text("old", encoding="utf-8")
            (staging / "value").write_text("new", encoding="utf-8")
            with mock.patch.object(
                filesystem,
                "_exchange_directories",
                side_effect=AtomicReplaceError("injected publication failure"),
            ):
                with self.assertRaises(AtomicReplaceError):
                    atomic_replace_directory(staging, target)
            self.assertEqual((target / "value").read_text(), "old")
            self.assertEqual((staging / "value").read_text(), "new")

    @unittest.skipUnless(sys.platform == "linux", "directory exchange requires Linux")
    def test_concurrent_reader_never_observes_a_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "published"
            target.mkdir()
            (target / "value").write_text("old", encoding="utf-8")
            stopped = threading.Event()
            missing: list[Exception] = []

            def read_target() -> None:
                while not stopped.is_set():
                    try:
                        (target / "value").read_text(encoding="utf-8")
                    except FileNotFoundError as error:
                        missing.append(error)
                        stopped.set()

            reader = threading.Thread(target=read_target)
            reader.start()
            try:
                for index in range(40):
                    staging = root / f"staging-{index}"
                    staging.mkdir()
                    (staging / "value").write_text(str(index), encoding="utf-8")
                    atomic_replace_directory(staging, target)
            finally:
                stopped.set()
                reader.join(timeout=5)
            self.assertFalse(reader.is_alive())
            self.assertEqual(missing, [])

    @unittest.skipUnless(sys.platform == "linux", "directory exchange requires Linux")
    def test_atomic_directory_commits_or_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "published"
            with atomic_directory(target) as staging:
                (staging / "value").write_text("complete", encoding="utf-8")
            self.assertEqual((target / "value").read_text(), "complete")
            with self.assertRaises(RuntimeError):
                with atomic_directory(target) as staging:
                    (staging / "incomplete").write_text("no", encoding="utf-8")
                    raise RuntimeError("abort")
            self.assertEqual((target / "value").read_text(), "complete")
            self.assertEqual(list(Path(directory).glob(".published.staging-*")), [])

    def test_nested_or_symlink_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            nested = target / "staging"
            nested.mkdir()
            with self.assertRaises(AtomicReplaceError):
                atomic_replace_directory(nested, target)
            with self.assertRaises(AtomicReplaceError):
                atomic_replace_directory(target / ".." / "target", target)


if __name__ == "__main__":
    unittest.main()

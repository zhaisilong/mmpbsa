from __future__ import annotations

import hashlib
import re
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from mmpbsa.utils import (
    JsonEncodingError,
    canonical_json,
    jsonable,
    new_id,
    sha256_bytes,
    sha256_digest,
    sha256_file,
    utc_now,
)


class Color(Enum):
    BLUE = "blue"


@dataclass(frozen=True)
class Payload:
    path: Path
    color: Color
    values: tuple[int, ...]


class SerializationTests(unittest.TestCase):
    def test_canonical_json_is_sorted_and_unicode_preserving(self) -> None:
        left = {"β": 2, "a": [3, 1]}
        right = {"a": [3, 1], "β": 2}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(canonical_json(left), '{"a":[3,1],"β":2}')

    def test_jsonable_supports_typed_standard_values(self) -> None:
        value = Payload(Path("a/b"), Color.BLUE, (1, 2))
        self.assertEqual(
            jsonable(value),
            {"path": "a/b", "color": "blue", "values": [1, 2]},
        )
        instant = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
        self.assertEqual(jsonable(instant), "2026-07-30T12:00:00.000000Z")

    def test_jsonable_rejects_ambiguous_values(self) -> None:
        for value in ({1: "integer key"}, {1, 2}, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(JsonEncodingError):
                    canonical_json(value)
        with self.assertRaises(JsonEncodingError):
            jsonable(datetime(2026, 1, 1))

    def test_digest_dispatch_distinguishes_raw_bytes_and_json_string(self) -> None:
        raw = sha256_digest(b"abc")
        text = sha256_digest("abc")
        self.assertEqual(raw, hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(raw, sha256_bytes(b"abc"))
        self.assertNotEqual(raw, text)
        self.assertEqual(text, hashlib.sha256(b'"abc"').hexdigest())

    def test_file_hash_streams_in_configurable_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(b"abcdef" * 100)
            self.assertEqual(
                sha256_file(path, chunk_size=7),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        with self.assertRaises(ValueError):
            sha256_file("unused", chunk_size=0)


class TimeTests(unittest.TestCase):
    def test_utc_now_is_rfc3339_utc(self) -> None:
        timestamp = utc_now()
        self.assertRegex(
            timestamp,
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$",
        )
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        self.assertEqual(parsed.utcoffset(), timezone.utc.utcoffset(parsed))

    def test_new_id_is_unique_sortable_and_path_safe(self) -> None:
        identifiers = [new_id("attempt") for _ in range(20)]
        self.assertEqual(len(set(identifiers)), len(identifiers))
        for identifier in identifiers:
            self.assertTrue(
                re.fullmatch(r"attempt-\d{8}T\d{12}Z-[0-9a-f]{32}", identifier)
            )
        with self.assertRaises(ValueError):
            new_id("../escape")


if __name__ == "__main__":
    unittest.main()

"""Integrity checks for the stable report publication boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from mmpbsa.utils import (
    audit_local_links,
    audit_portable_manifests,
    sha256_file,
)

from .errors import ArtifactIntegrityError


def audit_stable_report_staging(
    staging: Path,
    inventory: Sequence[Mapping[str, object]],
) -> None:
    """Audit the exact, independent file tree immediately before publication."""

    expected: dict[Path, Mapping[str, object]] = {}
    roles: set[str] = set()
    for record in inventory:
        relative = Path(str(record.get("path", "")))
        role = str(record.get("role", ""))
        if (
            relative.is_absolute()
            or relative == Path(".")
            or ".." in relative.parts
            or relative in expected
            or not role
            or role in roles
        ):
            raise ArtifactIntegrityError(
                "stable report publication inventory is ambiguous or unsafe"
            )
        expected[relative] = record
        roles.add(role)

    metadata_paths = {
        Path("publication.json"),
        Path("publication_inventory.json"),
    }
    required_paths = set(expected).union(metadata_paths)
    actual: set[Path] = set()
    for candidate in staging.rglob("*"):
        if candidate.is_symlink():
            raise ArtifactIntegrityError(
                f"stable report publication contains a symbolic link: {candidate}"
            )
        if candidate.is_file():
            relative = candidate.relative_to(staging)
            actual.add(relative)
            if candidate.stat().st_nlink != 1:
                raise ArtifactIntegrityError(
                    f"stable report publication contains a hard-linked file: {relative}"
                )
        elif not candidate.is_dir():
            raise ArtifactIntegrityError(
                f"stable report publication contains a non-file asset: {candidate}"
            )
    if actual != required_paths:
        missing = sorted(
            path.as_posix() for path in required_paths.difference(actual)
        )
        unexpected = sorted(
            path.as_posix() for path in actual.difference(required_paths)
        )
        raise ArtifactIntegrityError(
            "stable report publication tree differs from its inventory: "
            f"missing={missing}, unexpected={unexpected}"
        )

    try:
        published_inventory = json.loads(
            (staging / "publication_inventory.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            "stable report publication inventory is unreadable"
        ) from exc
    if (
        not isinstance(published_inventory, Mapping)
        or published_inventory.get("artifacts") != list(inventory)
    ):
        raise ArtifactIntegrityError(
            "stable report publication inventory does not describe the staged tree"
        )

    for relative, record in expected.items():
        candidate = staging / relative
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or not _is_sha256(digest)
            or candidate.stat().st_size != size
            or sha256_file(candidate) != digest
        ):
            raise ArtifactIntegrityError(
                f"stable report artifact differs from its inventory: {relative}"
            )

    audit_local_links(staging)
    audit_portable_manifests(staging)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

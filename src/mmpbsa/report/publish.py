"""Staging, local-link audit, and atomic report-tree publication."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable

from mmpbsa.utils import (
    LinkAuditError,
    LinkAuditIssue,
    atomic_replace_directory,
    audit_local_links,
    audit_portable_manifests,
    collect_link_issues,
)

from .models import AssetRequest, DocumentSpec
from .render import render_document

__all__ = [
    "AssetMaterializer",
    "AssetOrchestrator",
    "LinkAuditError",
    "LinkAuditIssue",
    "PublishResult",
    "ReportPublisher",
    "audit_local_links",
    "audit_portable_manifests",
    "collect_link_issues",
    "publish_report",
]


@runtime_checkable
class AssetMaterializer(Protocol):
    """Structural boundary implemented by display-asset providers."""

    def materialize(self, request: AssetRequest, staging_root: Path) -> None:
        """Create every declared request output below ``staging_root``."""


@dataclass(frozen=True, slots=True)
class PublishResult:
    destination: Path
    pages: tuple[PurePosixPath, ...]
    assets: tuple[PurePosixPath, ...]


class AssetOrchestrator:
    """Validate and fulfill typed display-asset requests in staging."""

    def materialize(
        self,
        requests: Sequence[AssetRequest],
        *,
        staging_root: Path,
        materializer: AssetMaterializer | None,
        reserved_paths: Sequence[PurePosixPath] = (),
    ) -> tuple[PurePosixPath, ...]:
        if not requests:
            return ()
        if materializer is None:
            raise ValueError("An AssetMaterializer is required for asset requests")
        request_ids = tuple(request.request_id for request in requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Asset request IDs must be unique")
        outputs = tuple(output for request in requests for output in request.outputs)
        if len(outputs) != len(set(outputs)):
            raise ValueError("Asset outputs must be globally unique")
        collisions = set(outputs).intersection(reserved_paths)
        if collisions:
            raise ValueError(
                "Asset outputs collide with report pages: "
                f"{sorted(map(str, collisions))}"
            )

        root = staging_root.resolve()
        for request in sorted(requests, key=lambda item: item.request_id):
            materializer.materialize(request, staging_root)
            for output in request.outputs:
                candidate = staging_root.joinpath(*output.parts)
                if not candidate.exists():
                    raise FileNotFoundError(
                        f"Asset materializer did not create {output} for {request.request_id}"
                    )
                if candidate.is_symlink():
                    raise ValueError(f"Display assets cannot be symbolic links: {output}")
                if not candidate.is_file():
                    raise ValueError(f"Display asset outputs must be files: {output}")
                resolved = candidate.resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(f"Display asset escapes staging root: {output}")
        allowed = set(outputs).union(reserved_paths)
        actual = {
            PurePosixPath(path.relative_to(staging_root).as_posix())
            for path in staging_root.rglob("*")
            if path.is_file()
        }
        undeclared = actual.difference(allowed)
        if undeclared:
            raise ValueError(
                "Asset materializer created undeclared files: "
                f"{sorted(map(str, undeclared))}"
            )
        return outputs


class ReportPublisher:
    """Publish a complete report tree without exposing partial replacements."""

    def __init__(self, *, asset_orchestrator: AssetOrchestrator | None = None) -> None:
        self.asset_orchestrator = asset_orchestrator or AssetOrchestrator()

    def publish(
        self,
        document: DocumentSpec,
        destination: Path | str,
        *,
        asset_requests: Sequence[AssetRequest] = (),
        asset_materializer: AssetMaterializer | None = None,
    ) -> PublishResult:
        destination_path = Path(destination).absolute()
        parent = destination_path.parent
        if destination_path.is_symlink():
            raise ValueError("Report destination cannot be a symbolic link")
        if destination_path.exists() and not destination_path.is_dir():
            raise NotADirectoryError(destination_path)
        parent.mkdir(parents=True, exist_ok=True)

        lock_path = parent / f".{destination_path.name}.publish.lock"
        lock_fd = _acquire_lock(lock_path)
        staging: Path | None = None
        try:
            staging = Path(
                tempfile.mkdtemp(prefix=f".{destination_path.name}.staging-", dir=parent)
            )
            rendered = render_document(document)
            _write_rendered_files(staging, rendered)
            assets = self.asset_orchestrator.materialize(
                asset_requests,
                staging_root=staging,
                materializer=asset_materializer,
                reserved_paths=tuple(rendered),
            )
            _verify_rendered_files(staging, rendered)
            audit_local_links(staging)
            audit_portable_manifests(staging)

            atomic_replace_directory(staging, destination_path)
            return PublishResult(
                destination=destination_path,
                pages=tuple(rendered),
                assets=assets,
            )
        finally:
            try:
                if staging is not None and staging.exists():
                    shutil.rmtree(staging)
            finally:
                try:
                    os.close(lock_fd)
                finally:
                    lock_path.unlink(missing_ok=True)


def publish_report(
    document: DocumentSpec,
    destination: Path | str,
    *,
    asset_requests: Sequence[AssetRequest] = (),
    asset_materializer: AssetMaterializer | None = None,
) -> PublishResult:
    return ReportPublisher().publish(
        document,
        destination,
        asset_requests=asset_requests,
        asset_materializer=asset_materializer,
    )


def _write_rendered_files(
    staging: Path,
    rendered: Mapping[PurePosixPath, str],
) -> None:
    root = staging.resolve()
    for relative, content in rendered.items():
        candidate = staging.joinpath(*relative.parts)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if not candidate.resolve().is_relative_to(root):
            raise ValueError(f"Rendered page escapes staging root: {relative}")
        candidate.write_text(content, encoding="utf-8", newline="\n")


def _verify_rendered_files(
    staging: Path,
    rendered: Mapping[PurePosixPath, str],
) -> None:
    for relative, expected in rendered.items():
        candidate = staging.joinpath(*relative.parts)
        if candidate.is_symlink() or candidate.read_text(encoding="utf-8") != expected:
            raise ValueError(f"Asset materialization modified report page: {relative}")


def _acquire_lock(path: Path) -> int:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Another report publication owns {path}") from error
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    return descriptor

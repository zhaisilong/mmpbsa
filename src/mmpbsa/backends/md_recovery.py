"""Copy native MD state into a fresh attempt without rewriting failed history."""

from __future__ import annotations

import shutil
from pathlib import Path

from mmpbsa.utils import sha256_file

from .errors import BackendConfigurationError, BackendOutputError


def clone_native_output(source_attempt: Path, destination_attempt: Path) -> dict[str, object]:
    source_attempt = source_attempt.resolve()
    destination_attempt = destination_attempt.resolve()
    if source_attempt.parent != destination_attempt.parent or source_attempt == destination_attempt:
        raise BackendConfigurationError('Failed MD recovery must use a fresh sibling attempt')
    source = source_attempt / 'output'
    destination = destination_attempt / 'output'
    if source.is_symlink() or destination.is_symlink():
        raise BackendConfigurationError('Native recovery output roots cannot be symlinks')
    if destination.exists() and any(destination.iterdir()):
        raise BackendConfigurationError('Failed MD recovery destination is not empty')
    files = [p for p in source.rglob('*') if p.is_file()]
    if not files or any(p.is_symlink() for p in source.rglob('*')):
        raise BackendConfigurationError('Native recovery output is empty or contains symlinks')
    identities = {str(p.relative_to(source)): sha256_file(p) for p in files}
    shutil.copytree(source, destination, dirs_exist_ok=True)
    if any(sha256_file(destination / name) != digest for name, digest in identities.items()):
        raise BackendOutputError('Native MD recovery copy differs from failed source')
    return {'schema_version': 'mmpbsa.failed-md-recovery.v1',
            'source_attempt': str(source_attempt), 'source_attempt_id': source_attempt.name,
            'copied_artifact_sha256': identities, 'source_is_unchanged': True}

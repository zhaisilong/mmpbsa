"""Read-only inspection of legacy workflow and workspace documents."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import yaml

from .errors import ConfigurationError


def inspect_legacy_workflow(path: Path) -> dict[str, object]:
    """Describe a legacy workflow without importing profiles or executing it."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"legacy workflow does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read legacy workflow: {source}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError("legacy workflow must contain a mapping")
    cases = raw.get("cases")
    replicas = raw.get("replicas")
    return {
        "path": str(source),
        "schema_version": raw.get("schema_version"),
        "project_id": raw.get("project_id"),
        "protocol_id": raw.get("protocol_id"),
        "case_count": len(cases) if isinstance(cases, list) else None,
        "replica_count": len(replicas) if isinstance(replicas, list) else None,
        "executable": False,
        "writes": False,
    }


def inspect_legacy_workspace(path: Path) -> dict[str, object]:
    """Inventory legacy state documents without validating or rewriting them."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"legacy workspace does not exist: {root}")
    manifest = _read_json_mapping(root / "workspace.json")
    state_schemas: Counter[str] = Counter()
    states: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    unreadable: list[str] = []
    state_files = tuple(
        item
        for item in root.rglob("state.json")
        if item.is_file() and not item.is_symlink()
    )
    for state_path in state_files:
        try:
            state = _read_json_mapping(state_path)
        except ConfigurationError:
            unreadable.append(state_path.relative_to(root).as_posix())
            continue
        state_schemas[str(state.get("schema_version", "(missing)"))] += 1
        states[str(state.get("state", "(missing)"))] += 1
        task = state.get("task")
        if isinstance(task, Mapping):
            stages[str(task.get("stage", "(missing)"))] += 1
    return {
        "path": str(root),
        "workspace_schema": manifest.get("schema_version"),
        "project_id": manifest.get("project_id"),
        "protocol_id": manifest.get("protocol_id"),
        "state_documents": len(state_files),
        "state_schemas": dict(sorted(state_schemas.items())),
        "states": dict(sorted(states.items())),
        "stages": dict(sorted(stages.items())),
        "unreadable_state_documents": sorted(unreadable),
        "executable": False,
        "writes": False,
    }


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ConfigurationError(f"legacy JSON document does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read legacy JSON document: {path}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"legacy JSON document must be a mapping: {path}")
    return raw


__all__ = ["inspect_legacy_workflow", "inspect_legacy_workspace"]

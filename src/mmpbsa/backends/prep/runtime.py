"""Read the explicit non-stage AmberTools runtime section."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from .errors import PreparationError


def load_preparation_runtime(path: Path) -> Mapping[str, object]:
    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreparationError(f"cannot read runtime configuration: {source}") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "mmpbsa.runtime.v1":
        raise PreparationError("preparation requires mmpbsa.runtime.v1")
    backends = raw.get("backends")
    if not isinstance(backends, Mapping):
        raise PreparationError("runtime.backends must be a mapping")
    builtin = backends.get("backends")
    if not isinstance(builtin, Mapping):
        raise PreparationError("runtime.backends.backends must be a mapping")
    preparation = builtin.get("preparation")
    if not isinstance(preparation, Mapping):
        raise PreparationError(
            "runtime.backends.backends.preparation must configure AmberTools"
        )
    global_environment = raw.get("environment", {})
    if not isinstance(global_environment, Mapping):
        raise PreparationError("runtime.environment must be a mapping")
    merged: dict[str, object] = dict(preparation)
    for tool_name in ("antechamber", "parmchk2", "tleap", "reduce"):
        raw_tool = merged.get(tool_name, {})
        if not isinstance(raw_tool, Mapping):
            raise PreparationError(
                f"runtime preparation tool {tool_name!r} must be a mapping"
            )
        tool = dict(raw_tool)
        local_environment = tool.get("environment", {})
        if not isinstance(local_environment, Mapping):
            raise PreparationError(
                f"runtime preparation tool {tool_name!r} environment "
                "must be a mapping"
            )
        tool["environment"] = {
            **{str(key): str(value) for key, value in global_environment.items()},
            **{str(key): str(value) for key, value in local_environment.items()},
        }
        merged[tool_name] = tool
    return merged


__all__ = ["load_preparation_runtime"]

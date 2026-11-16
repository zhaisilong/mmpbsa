"""Shared filesystem and configuration helpers for concrete adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .api import CommandRecord, CommandRunner, CommandSpec, GeneratedInput
from .command import SubprocessCommandRunner
from .errors import (
    BackendConfigurationError,
    BackendOutputError,
)
from .versioning import ToolConfig


def require_file(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise BackendConfigurationError(f"{label} is missing or empty: {resolved}")
    return resolved


def require_files(paths: Sequence[Path], label: str) -> tuple[Path, ...]:
    if not paths:
        raise BackendConfigurationError(f"{label} cannot be empty")
    return tuple(require_file(path, label) for path in paths)


def clean_output_dir(path: Path) -> Path:
    root = Path(path).resolve()
    if root.exists():
        if not root.is_dir():
            raise BackendConfigurationError(f"output path is not a directory: {root}")
        if any(root.iterdir()):
            raise BackendConfigurationError(f"output directory is not empty: {root}")
    else:
        root.mkdir(parents=True)
    return root


def prepare_output_dir(path: Path, *, allow_nonempty: bool = False) -> Path:
    """Create an output directory, optionally preserving resumable state."""

    root = Path(path).resolve()
    if root.exists() and not root.is_dir():
        raise BackendConfigurationError(
            f"output path is not a directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    if not allow_nonempty and any(root.iterdir()):
        raise BackendConfigurationError(
            f"output directory is not empty: {root}"
        )
    return root


def materialize_inputs(root: Path, inputs: Sequence[GeneratedInput]) -> None:
    for item in inputs:
        relative = Path(item.relative_path)
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise BackendConfigurationError(
                f"generated input must use a safe relative path: {relative}"
            )
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise BackendConfigurationError(
                f"generated input escapes output directory: {relative}"
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8")


def require_artifacts(artifacts: Mapping[str, Path]) -> None:
    failures = [
        f"{label}={Path(path).resolve()}"
        for label, path in artifacts.items()
        if not Path(path).resolve().is_file() or Path(path).resolve().stat().st_size == 0
    ]
    if failures:
        raise BackendOutputError(
            "backend did not produce declared artifacts: " + ", ".join(failures)
        )


def execute_commands(
    commands: Sequence[CommandSpec],
    *,
    runner: CommandRunner | None,
) -> tuple[CommandRecord, ...]:
    selected = runner or SubprocessCommandRunner()
    return tuple(selected.run(command) for command in commands)


def tool_from_config(
    config: Mapping[str, object],
    *,
    default_executable: str,
    default_version_arguments: tuple[str, ...] = ("--version",),
    key: str = "tool",
) -> ToolConfig:
    nested = config.get(key)
    if nested is None:
        nested = config
    if not isinstance(nested, Mapping):
        raise BackendConfigurationError(f"{key} configuration must be a mapping")
    value = dict(nested)
    if nested is not config:
        inherited_environment = config.get("environment", {})
        nested_environment = nested.get("environment", {})
        if not isinstance(inherited_environment, Mapping):
            raise BackendConfigurationError(
                "capability environment must be a mapping"
            )
        if not isinstance(nested_environment, Mapping):
            raise BackendConfigurationError(
                f"{key} environment must be a mapping"
            )
        value["environment"] = {
            **dict(inherited_environment),
            **dict(nested_environment),
        }
    return ToolConfig.from_mapping(
        value,
        default_executable=default_executable,
        default_version_arguments=default_version_arguments,
    )


def string_tuple(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BackendConfigurationError(f"{label} must be a sequence")
    if any(not isinstance(item, str) for item in value):
        raise BackendConfigurationError(f"{label} items must be strings")
    result = tuple(value)
    if any(not item or "\x00" in item for item in result):
        raise BackendConfigurationError(f"{label} contains an invalid argument")
    return result

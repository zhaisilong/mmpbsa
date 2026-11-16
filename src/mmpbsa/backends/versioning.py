"""Executable probing and explicit major-version compatibility."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .api import CommandRunner, CommandSpec
from .command import SubprocessCommandRunner
from .errors import (
    BackendConfigurationError,
    IncompatibleToolVersion,
    ToolProbeError,
)

_VERSION = re.compile(r"(?<!\d)(\d+)(?:\.(\d+))?(?:\.(\d+))?")


@dataclass(frozen=True, slots=True)
class MajorVersionPolicy:
    """An explicit set or inclusive range of supported tool majors."""

    accepted: frozenset[int] = frozenset()
    minimum: int | None = None
    maximum: int | None = None

    def validate(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in self.accepted
        ):
            raise BackendConfigurationError(
                "accepted tool majors must be non-negative integers"
            )
        if self.minimum is not None and (
            isinstance(self.minimum, bool)
            or not isinstance(self.minimum, int)
            or self.minimum < 0
        ):
            raise BackendConfigurationError(
                "minimum tool major must be a non-negative integer"
            )
        if self.maximum is not None and (
            isinstance(self.maximum, bool)
            or not isinstance(self.maximum, int)
            or self.maximum < 0
        ):
            raise BackendConfigurationError(
                "maximum tool major must be a non-negative integer"
            )
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise BackendConfigurationError("minimum tool major exceeds maximum")
        if self.accepted and (self.minimum is not None or self.maximum is not None):
            raise BackendConfigurationError(
                "use either accepted majors or a major range, not both"
            )

    def accepts(self, major: int) -> bool:
        self.validate()
        if self.accepted:
            return major in self.accepted
        if self.minimum is not None and major < self.minimum:
            return False
        if self.maximum is not None and major > self.maximum:
            return False
        return True

    @classmethod
    def from_value(cls, value: object) -> MajorVersionPolicy:
        if isinstance(value, cls):
            value.validate()
            return value
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            unknown = set(value).difference({"accepted", "minimum", "maximum"})
            if unknown:
                raise BackendConfigurationError(
                    "unknown major-version policy fields: "
                    + ", ".join(sorted(str(item) for item in unknown))
                )
            raw_accepted = value.get("accepted", ())
            if isinstance(raw_accepted, (str, bytes)) or not isinstance(
                raw_accepted, Iterable
            ):
                raise BackendConfigurationError("version accepted must be an iterable")
            result = cls(
                accepted=frozenset(
                    _major_integer(item, "accepted tool major")
                    for item in raw_accepted
                ),
                minimum=(
                    None
                    if value.get("minimum") is None
                    else _major_integer(value["minimum"], "minimum tool major")
                ),
                maximum=(
                    None
                    if value.get("maximum") is None
                    else _major_integer(value["maximum"], "maximum tool major")
                ),
            )
            result.validate()
            return result
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result = cls(
                accepted=frozenset(
                    _major_integer(item, "accepted tool major")
                    for item in value
                )
            )
            result.validate()
            return result
        raise BackendConfigurationError("invalid major-version policy")


@dataclass(frozen=True, slots=True)
class ToolConfig:
    executable: str
    version_arguments: tuple[str, ...] = ("--version",)
    version_pattern: str | None = None
    major_policy: MajorVersionPolicy = field(default_factory=MajorVersionPolicy)
    environment: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if (
            not isinstance(self.executable, str)
            or not self.executable.strip()
            or "\x00" in self.executable
        ):
            raise BackendConfigurationError("tool executable cannot be empty")
        if not self.version_arguments:
            raise BackendConfigurationError("tool version arguments cannot be empty")
        if any(
            not isinstance(item, str)
            or not item
            or "\x00" in item
            for item in self.version_arguments
        ):
            raise BackendConfigurationError("tool version arguments are invalid")
        if self.version_pattern is not None:
            try:
                re.compile(self.version_pattern)
            except re.error as exc:
                raise BackendConfigurationError("invalid tool version pattern") from exc
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "=" in key
            or "\x00" in key
            or "\x00" in value
            for key, value in self.environment.items()
        ):
            raise BackendConfigurationError(
                "tool environment contains an invalid entry"
            )
        self.major_policy.validate()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        default_executable: str,
        default_version_arguments: tuple[str, ...] = ("--version",),
    ) -> ToolConfig:
        arguments = value.get("version_arguments", default_version_arguments)
        if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
            raise BackendConfigurationError("version_arguments must be a sequence")
        if any(not isinstance(item, str) for item in arguments):
            raise BackendConfigurationError(
                "version_arguments items must be strings"
            )
        environment = value.get("environment", {})
        if not isinstance(environment, Mapping):
            raise BackendConfigurationError("tool environment must be a mapping")
        if any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in environment.items()
        ):
            raise BackendConfigurationError(
                "tool environment keys and values must be strings"
            )
        executable = value.get("executable", default_executable)
        if not isinstance(executable, str):
            raise BackendConfigurationError("tool executable must be a string")
        pattern = value.get("version_pattern")
        if pattern is not None and not isinstance(pattern, str):
            raise BackendConfigurationError(
                "tool version_pattern must be a string"
            )
        result = cls(
            executable=executable,
            version_arguments=tuple(arguments),
            version_pattern=pattern,
            major_policy=MajorVersionPolicy.from_value(value.get("major_policy")),
            environment=dict(environment),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class ToolIdentity:
    executable: str
    raw_version: str
    version: tuple[int, ...]

    @property
    def major(self) -> int:
        return self.version[0]


def parse_version(raw: str, *, pattern: str | None = None) -> tuple[int, ...]:
    expression = re.compile(pattern) if pattern is not None else _VERSION
    match = expression.search(raw)
    if match is None:
        raise ToolProbeError(f"could not parse tool version from: {raw.strip()!r}")
    groups = match.groups() or (match.group(0),)
    values: list[int] = []
    for value in groups:
        if value is None:
            continue
        if str(value).isdigit():
            values.append(int(value))
    if not values:
        nested = _VERSION.search(match.group(0))
        if nested is None:
            raise ToolProbeError(f"version pattern did not capture numeric fields: {raw!r}")
        values = [int(value) for value in nested.groups() if value is not None]
    return tuple(values)


def _major_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise BackendConfigurationError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise BackendConfigurationError(f"{label} must be an integer")


def probe_tool(
    config: ToolConfig,
    *,
    cwd: Path,
    runner: CommandRunner | None = None,
) -> ToolIdentity:
    config.validate()
    selected_runner = runner or SubprocessCommandRunner()
    record = selected_runner.run(
        CommandSpec(
            argv=(config.executable, *config.version_arguments),
            cwd=Path(cwd),
            environment=config.environment,
        )
    )
    raw = "\n".join(part for part in (record.stdout, record.stderr) if part).strip()
    version = parse_version(raw, pattern=config.version_pattern)
    identity = ToolIdentity(
        executable=config.executable,
        raw_version=raw,
        version=version,
    )
    if not config.major_policy.accepts(identity.major):
        raise IncompatibleToolVersion(
            f"{config.executable} major {identity.major} is incompatible with "
            f"{config.major_policy}"
        )
    return identity

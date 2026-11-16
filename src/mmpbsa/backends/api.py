"""Neutral command value objects used inside the 0.7 backend provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


class Capability(str, Enum):
    ADMISSION = "admission"
    MD = "md"
    TRAJECTORY = "trajectory"
    SCORING = "scoring"
    TRAJECTORY_QC = "trajectory_qc"
    REPORT_ASSETS = "report.assets"


class ReceiptState(str, Enum):
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """A completely materialized external process invocation."""

    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str] = field(default_factory=dict)
    stdin: str | None = None
    timeout_seconds: float | None = None
    cpu_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandRecord:
    argv: tuple[str, ...]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str
    cpu_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GeneratedInput:
    relative_path: Path
    content: str


@dataclass(frozen=True, slots=True)
class BackendReceipt:
    """Common result shape consumed by the workflow application layer."""

    state: ReceiptState
    commands: tuple[CommandRecord | CommandSpec, ...]
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parsed: Mapping[str, Any] = field(default_factory=dict)
    generated_inputs: tuple[GeneratedInput, ...] = ()


@runtime_checkable
class CommandRunner(Protocol):
    def run(self, command: CommandSpec) -> CommandRecord: ...


@runtime_checkable
class BackendFactory(Protocol):
    def __call__(
        self,
        config: Mapping[str, object],
        *,
        runner: CommandRunner | None = None,
    ) -> object: ...

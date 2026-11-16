"""cpptraj QC execution with caller-owned scripts and strict table parsing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._support import (
    clean_output_dir,
    execute_commands,
    materialize_inputs,
    require_artifacts,
    require_files,
    tool_from_config,
)
from .api import (
    BackendReceipt,
    CommandRunner,
    CommandSpec,
    GeneratedInput,
    ReceiptState,
)
from .errors import BackendConfigurationError, BackendOutputError
from .versioning import ToolConfig, probe_tool


@dataclass(frozen=True, slots=True)
class CpptrajQcRequest:
    input_text: str
    declared_inputs: tuple[Path, ...]
    output_tables: Mapping[str, Path]
    output_dir: Path
    expected_frames: int | None = None

    def validate(self) -> None:
        if not self.input_text.strip() or "\x00" in self.input_text:
            raise BackendConfigurationError("cpptraj QC input text cannot be empty")
        require_files(self.declared_inputs, "cpptraj declared inputs")
        if not self.output_tables:
            raise BackendConfigurationError("cpptraj QC must declare output tables")
        for label, path in self.output_tables.items():
            if not str(label).strip():
                raise BackendConfigurationError("cpptraj output label cannot be empty")
            relative = Path(path)
            if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
                raise BackendConfigurationError(
                    f"cpptraj output table must be a safe relative path: {relative}"
                )
        if len(set(Path(path) for path in self.output_tables.values())) != len(
            self.output_tables
        ):
            raise BackendConfigurationError(
                "cpptraj output table paths must be unique"
            )
        if self.expected_frames is not None and (
            isinstance(self.expected_frames, bool)
            or not isinstance(self.expected_frames, int)
            or self.expected_frames < 1
        ):
            raise BackendConfigurationError("cpptraj expected_frames must be positive")


@dataclass(frozen=True, slots=True)
class CpptrajQcBackend:
    tool: ToolConfig
    runner: CommandRunner | None = None

    def build_commands(
        self,
        request: CpptrajQcRequest,
    ) -> tuple[tuple[GeneratedInput, ...], tuple[CommandSpec, ...], Mapping[str, Path]]:
        request.validate()
        root = Path(request.output_dir).resolve()
        script = root / "qc.in"
        artifacts = {
            label: root / Path(relative)
            for label, relative in request.output_tables.items()
        }
        artifacts["cpptraj_input"] = script
        return (
            (GeneratedInput(Path(script.name), _terminal_newline(request.input_text)),),
            (
                CommandSpec(
                    argv=(self.tool.executable, "-i", str(script)),
                    cwd=root,
                    environment=self.tool.environment,
                ),
            ),
            artifacts,
        )

    def run(self, request: CpptrajQcRequest, *, dry_run: bool = False) -> BackendReceipt:
        generated, commands, artifacts = self.build_commands(request)
        if dry_run:
            return BackendReceipt(
                state=ReceiptState.PLANNED,
                commands=commands,
                artifacts=artifacts,
                generated_inputs=generated,
            )
        root = clean_output_dir(request.output_dir)
        identity = probe_tool(self.tool, cwd=root, runner=self.runner)
        materialize_inputs(root, generated)
        records = execute_commands(commands, runner=self.runner)
        require_artifacts(artifacts)
        parsed = {
            label: parse_cpptraj_table(path)
            for label, path in artifacts.items()
            if label != "cpptraj_input"
        }
        axes = {
            label: tuple(row[0] for row in table["rows"])
            for label, table in parsed.items()
        }
        if axes:
            reference_label = next(iter(axes))
            mismatched_axes = tuple(
                label
                for label, axis in axes.items()
                if axis != axes[reference_label]
            )
            if mismatched_axes:
                raise BackendOutputError(
                    "cpptraj QC tables use different frame/time axes: "
                    + ", ".join(mismatched_axes)
                )
        if request.expected_frames is not None:
            mismatched = {
                label: len(table["rows"])
                for label, table in parsed.items()
                if len(table["rows"]) != request.expected_frames
            }
            if mismatched:
                raise BackendOutputError(
                    f"cpptraj QC frame counts differ from "
                    f"{request.expected_frames}: {mismatched}"
                )
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=records,
            artifacts=artifacts,
            metadata={"tool": identity},
            parsed=parsed,
            generated_inputs=generated,
        )


def parse_cpptraj_table(path: Path) -> Mapping[str, Any]:
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise BackendOutputError(f"cpptraj table is missing or empty: {source}")
    columns: tuple[str, ...] | None = None
    rows: list[tuple[float, ...]] = []
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            candidate = tuple(line.lstrip("#").split())
            if candidate and candidate[0].casefold() in {"frame", "time"}:
                columns = candidate
            continue
        if columns is None:
            raise BackendOutputError(
                f"cpptraj data precedes a #Frame/#Time header at {source}:{line_number}"
            )
        fields = line.split()
        if len(fields) != len(columns):
            raise BackendOutputError(
                f"cpptraj row width differs from header at {source}:{line_number}"
            )
        try:
            row = tuple(float(field) for field in fields)
        except ValueError as exc:
            raise BackendOutputError(
                f"cpptraj table has non-numeric data at {source}:{line_number}"
            ) from exc
        if not all(math.isfinite(value) for value in row):
            raise BackendOutputError(
                f"cpptraj table has non-finite data at {source}:{line_number}"
            )
        rows.append(row)
    if columns is None or not rows:
        raise BackendOutputError(f"cpptraj table has no header/data: {source}")
    if len(columns) != len(set(columns)):
        raise BackendOutputError(
            f"cpptraj table has duplicate columns: {source}"
        )
    axes = tuple(row[0] for row in rows)
    if len(axes) != len(set(axes)) or any(
        right <= left for left, right in zip(axes, axes[1:], strict=False)
    ):
        raise BackendOutputError(f"cpptraj axis must be strictly increasing: {source}")
    return {"columns": columns, "rows": tuple(rows)}


def create_cpptraj_qc_backend(
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None = None,
) -> CpptrajQcBackend:
    return CpptrajQcBackend(
        tool=tool_from_config(
            config,
            default_executable="cpptraj",
            default_version_arguments=("-V",),
        ),
        runner=runner,
    )


def _terminal_newline(value: str) -> str:
    return value.rstrip("\n") + "\n"

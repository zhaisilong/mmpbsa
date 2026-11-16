"""Typed GROMACS MD stage adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ._support import (
    clean_output_dir,
    execute_commands,
    materialize_inputs,
    prepare_output_dir,
    require_artifacts,
    require_file,
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

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_GROMPP_RESERVED = frozenset(
    {"-f", "-c", "-p", "-o", "-n", "-t", "-r", "-pp"}
)
_MDRUN_RESERVED = frozenset(
    {"-deffnm", "-s", "-ntomp", "-gpu_id", "-cpi", "-append", "-noappend"}
)
_MDRUN_SCIENCE_OVERRIDES = frozenset(
    {
        "-awh",
        "-ei",
        "-imd",
        "-membed",
        "-maxh",
        "-multidir",
        "-nsteps",
        "-plumed",
        "-replex",
        "-rerun",
        "-table",
        "-tableb",
        "-tablep",
    }
)


@dataclass(frozen=True, slots=True)
class GromacsMdRequest:
    stage_id: str
    mdp_text: str
    coordinates: Path
    topology: Path
    output_dir: Path
    index: Path | None = None
    state_input: Path | None = None
    reference: Path | None = None
    checkpoint_input: Path | None = None
    append: bool = False
    require_unconstrained_topology: bool = False
    maxwarn: int = 0
    threads: int = 1
    gpu_id: str | None = None
    grompp_arguments: tuple[str, ...] = ()
    mdrun_arguments: tuple[str, ...] = ()
    expected_suffixes: tuple[str, ...] = (
        "tpr",
        "log",
        "edr",
        "gro",
        "xtc",
        "cpt",
    )

    def validate(self) -> None:
        if not _ID.fullmatch(self.stage_id):
            raise BackendConfigurationError(f"invalid GROMACS stage_id: {self.stage_id!r}")
        if not self.mdp_text.strip() or "\x00" in self.mdp_text:
            raise BackendConfigurationError("GROMACS MDP text cannot be empty")
        require_file(self.coordinates, "GROMACS coordinates")
        require_file(self.topology, "GROMACS topology")
        for label, path in (
            ("GROMACS index", self.index),
            ("GROMACS state input", self.state_input),
            ("GROMACS reference", self.reference),
            ("GROMACS checkpoint input", self.checkpoint_input),
        ):
            if path is not None:
                require_file(path, label)
        if (
            isinstance(self.maxwarn, bool)
            or not isinstance(self.maxwarn, int)
            or self.maxwarn < 0
        ):
            raise BackendConfigurationError("GROMACS maxwarn must be a non-negative integer")
        if not isinstance(self.require_unconstrained_topology, bool):
            raise BackendConfigurationError(
                "GROMACS require_unconstrained_topology must be boolean"
            )
        if (
            isinstance(self.threads, bool)
            or not isinstance(self.threads, int)
            or self.threads < 1
        ):
            raise BackendConfigurationError("GROMACS threads must be a positive integer")
        _reject_options(self.grompp_arguments, _GROMPP_RESERVED, "grompp_arguments")
        _reject_options(
            self.mdrun_arguments,
            _MDRUN_RESERVED | _MDRUN_SCIENCE_OVERRIDES,
            "mdrun_arguments",
        )
        if not self.expected_suffixes or len(set(self.expected_suffixes)) != len(
            self.expected_suffixes
        ):
            raise BackendConfigurationError(
                "GROMACS expected_suffixes must be non-empty and unique"
            )
        if any(not _ID.fullmatch(suffix) for suffix in self.expected_suffixes):
            raise BackendConfigurationError("GROMACS expected artifact suffix is invalid")
        if "mdp_input" in self.expected_suffixes:
            raise BackendConfigurationError(
                "GROMACS mdp_input is a reserved artifact label"
            )


@dataclass(frozen=True, slots=True)
class GromacsMdBackend:
    tool: ToolConfig
    runner: CommandRunner | None = None

    def probe(self, *, cwd: Path) -> object:
        return probe_tool(self.tool, cwd=cwd, runner=self.runner)

    def build_commands(
        self,
        request: GromacsMdRequest,
        *,
        resume: bool = False,
    ) -> tuple[tuple[GeneratedInput, ...], tuple[CommandSpec, ...], Mapping[str, Path]]:
        request.validate()
        root = Path(request.output_dir).resolve()
        mdp = root / f"{request.stage_id}.mdp"
        tpr = root / f"{request.stage_id}.tpr"
        grompp = [
            self.tool.executable,
            "grompp",
            "-f",
            str(mdp),
            "-c",
            str(Path(request.coordinates).resolve()),
            "-p",
            str(Path(request.topology).resolve()),
            "-o",
            str(tpr),
        ]
        if request.index is not None:
            grompp.extend(("-n", str(Path(request.index).resolve())))
        if request.state_input is not None:
            grompp.extend(("-t", str(Path(request.state_input).resolve())))
        if request.reference is not None:
            grompp.extend(("-r", str(Path(request.reference).resolve())))
        if request.require_unconstrained_topology:
            grompp.extend(
                ("-pp", str(root / f"{request.stage_id}.processed.top"))
            )
        if request.maxwarn:
            grompp.extend(("-maxwarn", str(request.maxwarn)))
        grompp.extend(request.grompp_arguments)

        mdrun = [
            self.tool.executable,
            "mdrun",
            "-deffnm",
            request.stage_id,
            "-s",
            str(tpr),
            "-ntomp",
            str(request.threads),
        ]
        if request.gpu_id is not None:
            if (
                not isinstance(request.gpu_id, str)
                or not request.gpu_id.strip()
                or "\x00" in request.gpu_id
            ):
                raise BackendConfigurationError("GROMACS gpu_id cannot be empty")
            mdrun.extend(("-gpu_id", str(request.gpu_id)))
        if request.checkpoint_input is not None:
            mdrun.extend(
                ("-cpi", str(Path(request.checkpoint_input).resolve()))
            )
            mdrun.append("-append" if request.append else "-noappend")
        mdrun.extend(request.mdrun_arguments)
        environment = self.tool.environment
        if resume:
            require_file(tpr, "GROMACS resume TPR")
            commands = (CommandSpec(tuple(mdrun), root, environment),)
            generated: tuple[GeneratedInput, ...] = (
                GeneratedInput(
                    Path(mdp.name),
                    _terminal_newline(request.mdp_text),
                ),
            )
        else:
            commands = (
                CommandSpec(tuple(grompp), root, environment),
                CommandSpec(tuple(mdrun), root, environment),
            )
            generated = (
                GeneratedInput(
                    Path(mdp.name),
                    _terminal_newline(request.mdp_text),
                ),
            )
        artifacts = {
            suffix: root / f"{request.stage_id}.{suffix}"
            for suffix in request.expected_suffixes
        }
        artifacts["mdp_input"] = mdp
        return generated, commands, artifacts

    def run(
        self,
        request: GromacsMdRequest,
        *,
        dry_run: bool = False,
        resume: bool = False,
    ) -> BackendReceipt:
        if resume and request.checkpoint_input is None:
            raise BackendConfigurationError(
                "GROMACS resume requires checkpoint_input"
            )
        generated, commands, artifacts = self.build_commands(
            request,
            resume=resume,
        )
        if dry_run:
            return BackendReceipt(
                state=ReceiptState.PLANNED,
                commands=commands,
                artifacts=artifacts,
                generated_inputs=generated,
            )
        root = (
            prepare_output_dir(request.output_dir, allow_nonempty=True)
            if resume
            else clean_output_dir(request.output_dir)
        )
        identity = self.probe(cwd=root)
        materialize_inputs(root, generated)
        if request.require_unconstrained_topology:
            processed = root / f"{request.stage_id}.processed.top"
            if resume:
                require_file(processed, "GROMACS preprocessed topology")
                _validate_unconstrained_topology(processed)
                records = execute_commands(commands, runner=self.runner)
            else:
                preprocessing = execute_commands(
                    commands[:1], runner=self.runner
                )
                require_file(processed, "GROMACS preprocessed topology")
                _validate_unconstrained_topology(processed)
                records = preprocessing + execute_commands(
                    commands[1:], runner=self.runner
                )
        else:
            records = execute_commands(commands, runner=self.runner)
        require_artifacts(artifacts)
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=records,
            artifacts=artifacts,
            metadata={"tool": identity},
            generated_inputs=generated,
        )


def create_gromacs_md_backend(
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None = None,
) -> GromacsMdBackend:
    return GromacsMdBackend(
        tool=tool_from_config(config, default_executable="gmx"),
        runner=runner,
    )


def _reject_options(
    arguments: Sequence[str],
    forbidden: frozenset[str],
    label: str,
) -> None:
    for argument in arguments:
        option = str(argument).split("=", 1)[0]
        if option in forbidden:
            raise BackendConfigurationError(f"{label} cannot override reserved {option}")


def _terminal_newline(value: str) -> str:
    return value.rstrip("\n") + "\n"


def _validate_unconstrained_topology(path: Path) -> None:
    section = ""
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8", errors="strict").splitlines(),
        start=1,
    ):
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        header = re.fullmatch(r"\[\s*([^\]]+?)\s*\]", line)
        if header is not None:
            section = header.group(1).strip().lower()
            continue
        if section in {"constraints", "settles"}:
            raise BackendOutputError(
                "GROMACS EM preprocessed topology retains an explicit "
                f"[{section}] term at line {line_number}"
            )

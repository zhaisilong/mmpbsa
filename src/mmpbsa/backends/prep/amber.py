"""AmberTools preparation commands kept separate from workflow stages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .._support import (
    clean_output_dir,
    execute_commands,
    materialize_inputs,
    require_artifacts,
    require_file,
    require_files,
    tool_from_config,
)
from ..api import (
    BackendReceipt,
    CommandRunner,
    CommandSpec,
    GeneratedInput,
    ReceiptState,
)
from ..errors import BackendConfigurationError
from ..versioning import ToolConfig, probe_tool

_FORMAT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_RESIDUE = re.compile(r"^[A-Za-z0-9]{1,4}$")
_ANTECHAMBER_RESERVED = frozenset(
    {"-i", "-fi", "-o", "-fo", "-rn", "-nc", "-c", "-at"}
)
_PARMCHK_RESERVED = frozenset({"-i", "-f", "-o", "-s"})


@dataclass(frozen=True, slots=True)
class SmallMoleculeParameterizationRequest:
    """Caller-owned choices for antechamber followed by parmchk2."""

    source: Path
    output_dir: Path
    input_format: str
    residue_name: str
    formal_charge: int
    charge_method: str
    atom_type: str
    force_field: str
    molecule_filename: str = "molecule.mol2"
    parameter_filename: str = "molecule.frcmod"
    antechamber_arguments: tuple[str, ...] = ()
    parmchk_arguments: tuple[str, ...] = ()

    def validate(self) -> None:
        require_file(self.source, "small-molecule source")
        for label, value in (
            ("input_format", self.input_format),
            ("charge_method", self.charge_method),
            ("atom_type", self.atom_type),
            ("force_field", self.force_field),
        ):
            if not _FORMAT.fullmatch(value):
                raise BackendConfigurationError(
                    f"small-molecule {label} is invalid: {value!r}"
                )
        if not _RESIDUE.fullmatch(self.residue_name):
            raise BackendConfigurationError(
                "small-molecule residue_name must contain 1-4 letters or numbers"
            )
        if isinstance(self.formal_charge, bool) or not isinstance(
            self.formal_charge, int
        ):
            raise BackendConfigurationError(
                "small-molecule formal_charge must be an integer"
            )
        for label, filename in (
            ("molecule_filename", self.molecule_filename),
            ("parameter_filename", self.parameter_filename),
        ):
            _safe_filename(filename, label)
        _reject_options(
            self.antechamber_arguments,
            _ANTECHAMBER_RESERVED,
            "antechamber_arguments",
        )
        _reject_options(
            self.parmchk_arguments,
            _PARMCHK_RESERVED,
            "parmchk_arguments",
        )


@dataclass(frozen=True, slots=True)
class AmberAssemblyRequest:
    """A caller-authored tleap program with a declared artifact contract."""

    input_text: str
    declared_inputs: tuple[Path, ...]
    output_artifacts: Mapping[str, Path]
    output_dir: Path
    extra_arguments: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.input_text.strip() or "\x00" in self.input_text:
            raise BackendConfigurationError("tleap input text cannot be empty")
        require_files(self.declared_inputs, "tleap declared inputs")
        if not self.output_artifacts:
            raise BackendConfigurationError(
                "tleap output_artifacts cannot be empty"
            )
        for label, relative in self.output_artifacts.items():
            if not str(label).strip():
                raise BackendConfigurationError(
                    "tleap artifact label cannot be empty"
                )
            path = Path(relative)
            if path.is_absolute() or path == Path(".") or ".." in path.parts:
                raise BackendConfigurationError(
                    f"tleap output must be a safe relative path: {path}"
                )
        if len(set(Path(path) for path in self.output_artifacts.values())) != len(
            self.output_artifacts
        ):
            raise BackendConfigurationError(
                "tleap output artifact paths must be unique"
            )
        _reject_options(
            self.extra_arguments,
            frozenset({"-f"}),
            "tleap extra_arguments",
        )


@dataclass(frozen=True, slots=True)
class AmberPreparationTools:
    """Standalone parameterization and topology-assembly operations."""

    antechamber: ToolConfig
    parmchk2: ToolConfig
    tleap: ToolConfig
    runner: CommandRunner | None = None

    def parameterize(
        self,
        request: SmallMoleculeParameterizationRequest,
        *,
        dry_run: bool = False,
    ) -> BackendReceipt:
        generated, commands, artifacts = self.build_parameterization(request)
        if dry_run:
            return BackendReceipt(
                state=ReceiptState.PLANNED,
                commands=commands,
                artifacts=artifacts,
                generated_inputs=generated,
            )
        root = clean_output_dir(request.output_dir)
        tools = {
            "antechamber": probe_tool(
                self.antechamber,
                cwd=root,
                runner=self.runner,
            ),
            "parmchk2": probe_tool(
                self.parmchk2,
                cwd=root,
                runner=self.runner,
            ),
        }
        records = execute_commands(commands, runner=self.runner)
        require_artifacts(artifacts)
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=records,
            artifacts=artifacts,
            metadata={
                "tools": tools,
                "charge_method": request.charge_method,
                "atom_type": request.atom_type,
                "force_field": request.force_field,
                "formal_charge": request.formal_charge,
            },
        )

    def build_parameterization(
        self,
        request: SmallMoleculeParameterizationRequest,
    ) -> tuple[
        tuple[GeneratedInput, ...],
        tuple[CommandSpec, ...],
        Mapping[str, Path],
    ]:
        request.validate()
        root = Path(request.output_dir).resolve()
        molecule = root / request.molecule_filename
        parameters = root / request.parameter_filename
        first = CommandSpec(
            argv=(
                self.antechamber.executable,
                "-i",
                str(Path(request.source).resolve()),
                "-fi",
                request.input_format,
                "-o",
                str(molecule),
                "-fo",
                "mol2",
                "-rn",
                request.residue_name,
                "-nc",
                str(request.formal_charge),
                "-c",
                request.charge_method,
                "-at",
                request.atom_type,
                *request.antechamber_arguments,
            ),
            cwd=root,
            environment=self.antechamber.environment,
        )
        second = CommandSpec(
            argv=(
                self.parmchk2.executable,
                "-i",
                str(molecule),
                "-f",
                "mol2",
                "-o",
                str(parameters),
                "-s",
                request.force_field,
                *request.parmchk_arguments,
            ),
            cwd=root,
            environment=self.parmchk2.environment,
        )
        return (
            (),
            (first, second),
            {"molecule": molecule, "parameters": parameters},
        )

    def assemble(
        self,
        request: AmberAssemblyRequest,
        *,
        dry_run: bool = False,
    ) -> BackendReceipt:
        generated, commands, artifacts = self.build_assembly(request)
        if dry_run:
            return BackendReceipt(
                state=ReceiptState.PLANNED,
                commands=commands,
                artifacts=artifacts,
                generated_inputs=generated,
            )
        root = clean_output_dir(request.output_dir)
        identity = probe_tool(self.tleap, cwd=root, runner=self.runner)
        materialize_inputs(root, generated)
        records = execute_commands(commands, runner=self.runner)
        require_artifacts(artifacts)
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=records,
            artifacts=artifacts,
            metadata={"tool": identity},
            generated_inputs=generated,
        )

    def build_assembly(
        self,
        request: AmberAssemblyRequest,
    ) -> tuple[
        tuple[GeneratedInput, ...],
        tuple[CommandSpec, ...],
        Mapping[str, Path],
    ]:
        request.validate()
        root = Path(request.output_dir).resolve()
        script = root / "tleap.in"
        artifacts = {
            label: root / Path(relative)
            for label, relative in request.output_artifacts.items()
        }
        artifacts["tleap_input"] = script
        return (
            (
                GeneratedInput(
                    Path(script.name),
                    request.input_text.rstrip("\n") + "\n",
                ),
            ),
            (
                CommandSpec(
                    argv=(
                        self.tleap.executable,
                        "-f",
                        str(script),
                        *request.extra_arguments,
                    ),
                    cwd=root,
                    environment=self.tleap.environment,
                ),
            ),
            artifacts,
        )


def create_amber_preparation_tools(
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None = None,
) -> AmberPreparationTools:
    return AmberPreparationTools(
        antechamber=tool_from_config(
            config,
            key="antechamber",
            default_executable="antechamber",
            default_version_arguments=("-h",),
        ),
        parmchk2=tool_from_config(
            config,
            key="parmchk2",
            default_executable="parmchk2",
            default_version_arguments=("-l",),
        ),
        tleap=tool_from_config(
            config,
            key="tleap",
            default_executable="tleap",
            default_version_arguments=("-h",),
        ),
        runner=runner,
    )


def _safe_filename(value: str, label: str) -> None:
    path = Path(value)
    if (
        path.is_absolute()
        or path == Path(".")
        or len(path.parts) != 1
        or value in {"", ".", ".."}
        or "\x00" in value
    ):
        raise BackendConfigurationError(f"{label} must be a safe filename")


def _reject_options(
    arguments: tuple[str, ...],
    forbidden: frozenset[str],
    label: str,
) -> None:
    for argument in arguments:
        if (
            not isinstance(argument, str)
            or not argument
            or "\x00" in argument
        ):
            raise BackendConfigurationError(
                f"{label} contains an invalid argument"
            )
        option = str(argument).split("=", 1)[0]
        if option in forbidden:
            raise BackendConfigurationError(
                f"{label} cannot override reserved {option}"
            )

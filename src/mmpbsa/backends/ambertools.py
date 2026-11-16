"""AmberTools MMPBSA execution and neutral result parsers."""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mmpbsa.utils import sha256_digest

from ._support import (
    clean_output_dir,
    execute_commands,
    materialize_inputs,
    require_artifacts,
    require_file,
    string_tuple,
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

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_SUMMARY_ROW = re.compile(
    rf"^\s*(?P<label>.+?)\s+(?P<average>{_NUMBER})\s+"
    rf"(?P<stddev>{_NUMBER})\s+(?P<stderr>{_NUMBER})\s*$"
)
_RESERVED = frozenset(
    {
        "-O",
        "--overwrite",
        "-i",
        "-o",
        "-eo",
        "-cp",
        "-rp",
        "-lp",
        "-sp",
        "-y",
        "-prefix",
        "-make-mdins",
        "-use-mdins",
        "-rewrite-output",
        "--clean",
    }
)
_MODEL_SECTIONS = {
    "gb": "GENERALIZED BORN",
    "pb": "POISSON BOLTZMANN",
}
_SECTION_MODELS = {value.casefold(): key for key, value in _MODEL_SECTIONS.items()}
_COMPONENT_SUBSECTIONS = {
    "complex energy terms": "complex",
    "receptor energy terms": "receptor",
    "ligand energy terms": "ligand",
    "delta energy terms": "delta",
}
_SUMMARY_COMPONENTS = {
    "complex": "complex",
    "receptor": "receptor",
    "ligand": "ligand",
    "differences (complex - receptor - ligand)": "delta",
}
_NATIVE_INPUT_LAYOUTS = {
    "gb": (
        (
            "main",
            Path("_MMPBSA_gb.mdin"),
            Path("inputs/gb.mdin"),
            "native_main_input",
        ),
    ),
    "pb": (
        (
            "complex_receptor",
            Path("_MMPBSA_pb.mdin"),
            Path("inputs/pb.mdin"),
            "native_complex_receptor_input",
        ),
        (
            "ligand",
            Path("_MMPBSA_pb.mdin2"),
            Path("inputs/pb.mdin2"),
            "native_ligand_input",
        ),
    ),
}
_PER_FRAME_SELECTOR_KEYS = frozenset({"section", "subsection"})
_DMM_EVALUATOR = "v1_dmm_v1"
_DMM_LABEL = "DMM"


@dataclass(frozen=True, slots=True)
class AmberToolsScoringRequest:
    model_id: str
    control_text: str
    complex_topology: Path
    receptor_topology: Path
    binder_topology: Path
    trajectory: Path
    output_dir: Path
    native_inputs: Mapping[str, str] | None = None
    expected_frames: int = 1
    extra_arguments: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.model_id not in _MODEL_SECTIONS:
            raise BackendConfigurationError(
                "MMPBSA model_id must be one of: " + ", ".join(_MODEL_SECTIONS)
            )
        if not self.control_text.strip() or "\x00" in self.control_text:
            raise BackendConfigurationError("MMPBSA control text cannot be empty")
        require_file(self.complex_topology, "MMPBSA complex topology")
        require_file(self.receptor_topology, "MMPBSA receptor topology")
        require_file(self.binder_topology, "MMPBSA binder topology")
        require_file(self.trajectory, "MMPBSA trajectory")
        selected_group = re.search(
            rf"(?im)^\s*&{re.escape(self.model_id)}\b",
            self.control_text,
        )
        other_model = "pb" if self.model_id == "gb" else "gb"
        other_group = re.search(
            rf"(?im)^\s*&{other_model}\b",
            self.control_text,
        )
        if selected_group is None or other_group is not None:
            raise BackendConfigurationError(
                "MMPBSA control_text must contain exactly the selected "
                f"&{self.model_id} model block"
            )
        _validate_native_inputs(self.model_id, self.native_inputs)
        if (
            isinstance(self.expected_frames, bool)
            or not isinstance(self.expected_frames, int)
            or self.expected_frames < 1
        ):
            raise BackendConfigurationError("MMPBSA expected_frames must be positive")
        for argument in self.extra_arguments:
            if (
                not isinstance(argument, str)
                or not argument
                or "\x00" in argument
            ):
                raise BackendConfigurationError(
                    "MMPBSA extra_arguments contains an invalid argument"
                )
            if str(argument).split("=", 1)[0] in _RESERVED:
                raise BackendConfigurationError(
                    f"MMPBSA extra_arguments cannot override {argument!r}"
                )


@dataclass(frozen=True, slots=True)
class AmberToolsScoringBackend:
    tool: ToolConfig
    sander: ToolConfig
    mpi_launcher: tuple[str, ...] = ()
    ranks: int = 1
    runner: CommandRunner | None = None

    def __post_init__(self) -> None:
        self.tool.validate()
        self.sander.validate()
        _validate_bound_sander(self.tool, self.sander)
        if (
            isinstance(self.ranks, bool)
            or not isinstance(self.ranks, int)
            or self.ranks < 1
        ):
            raise BackendConfigurationError("MMPBSA ranks must be positive")
        if self.ranks > 1 and not self.mpi_launcher:
            raise BackendConfigurationError(
                "MMPBSA ranks > 1 requires an explicit MPI launcher"
            )

    def build_commands(
        self,
        request: AmberToolsScoringRequest,
    ) -> tuple[tuple[GeneratedInput, ...], tuple[CommandSpec, ...], Mapping[str, Path]]:
        request.validate()
        root = Path(request.output_dir).resolve()
        control = root / "mmpbsa.in"
        final = root / "FINAL_RESULTS_MMPBSA.dat"
        per_frame = root / "per_frame_energy.csv"
        delta = root / "delta_energy.csv"
        dmm = root / "dmm_energy.csv"
        arguments = [
            self.tool.executable,
            "-O",
            "-i",
            str(control),
            "-o",
            str(final),
            "-eo",
            str(per_frame),
            "-cp",
            str(Path(request.complex_topology).resolve()),
            "-rp",
            str(Path(request.receptor_topology).resolve()),
            "-lp",
            str(Path(request.binder_topology).resolve()),
            "-y",
            str(Path(request.trajectory).resolve()),
        ]
        generated = [
            GeneratedInput(
                Path(control.name),
                _terminal_newline(request.control_text),
            )
        ]
        artifacts = {
            "final_output": final,
            "per_frame_output": per_frame,
            "delta_output": delta,
            "dmm_output": dmm,
            "control_input": control,
        }
        if request.native_inputs is not None:
            arguments.append("-use-mdins")
            for key, working_path, persisted_path, artifact_role in _NATIVE_INPUT_LAYOUTS[
                request.model_id
            ]:
                content = _terminal_newline(request.native_inputs[key])
                generated.extend(
                    (
                        GeneratedInput(working_path, content),
                        GeneratedInput(persisted_path, content),
                    )
                )
                artifacts[artifact_role] = root / persisted_path
        arguments.extend(request.extra_arguments)
        argv = (
            (
                *self.mpi_launcher,
                "-np",
                str(self.ranks),
                *arguments,
            )
            if self.ranks > 1
            else tuple(arguments)
        )
        return (
            tuple(generated),
            (
                CommandSpec(
                    argv=tuple(argv),
                    cwd=root,
                    environment=self.tool.environment,
                ),
            ),
            artifacts,
        )

    def run(
        self,
        request: AmberToolsScoringRequest,
        *,
        dry_run: bool = False,
    ) -> BackendReceipt:
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
        sander_identity = probe_tool(
            self.sander,
            cwd=root,
            runner=self.runner,
        )
        materialize_inputs(root, generated)
        records = execute_commands(commands, runner=self.runner)
        # Derived outputs are written only after the raw tool outputs have passed
        # the model/component/frame contract.
        require_artifacts(
            {
                role: path
                for role, path in artifacts.items()
                if role not in {"delta_output", "dmm_output"}
            }
        )
        summary = parse_mmpbsa_summary(
            artifacts["final_output"],
            model_id=request.model_id,
        )
        per_frame = parse_mmpbsa_models_csv(
            artifacts["per_frame_output"],
            model_id=request.model_id,
            expected_frames=request.expected_frames,
        )
        _write_delta_csv(artifacts["delta_output"], per_frame["components"]["delta"])
        dmm = derive_v1_dmm(
            request.model_id,
            per_frame["components"]["delta"],
        )
        _write_delta_csv(artifacts["dmm_output"], dmm)
        require_artifacts(
            {
                "delta_output": artifacts["delta_output"],
                "dmm_output": artifacts["dmm_output"],
            }
        )
        frame_ids = per_frame["frame_ids"]
        observed_frame_ids = {
            "count": len(frame_ids),
            "first": frame_ids[0],
            "last": frame_ids[-1],
            # Hash the canonical JSON array of integer frame IDs.  This keeps
            # the complete observed identity auditable without copying a long
            # sequence into every stage payload.
            "sha256": sha256_digest(frame_ids),
        }
        dmm_values = tuple(float(row[_DMM_LABEL]) for row in dmm["rows"])
        dmm_stddev = statistics.stdev(dmm_values) if len(dmm_values) > 1 else 0.0
        derived_summary = {
            "model_id": request.model_id,
            "rows": (
                {
                    "model_id": request.model_id,
                    "component": "delta",
                    "label": _DMM_LABEL,
                    "average": statistics.fmean(dmm_values),
                    "stddev": dmm_stddev,
                    "stderr": dmm_stddev / math.sqrt(len(dmm_values)),
                    "evaluator": _DMM_EVALUATOR,
                },
            ),
        }
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=records,
            artifacts=artifacts,
            metadata={
                "tool": identity,
                "sander": sander_identity,
                "ranks": self.ranks,
                "model_id": request.model_id,
            },
            parsed={
                "model_id": request.model_id,
                "summary": summary,
                "derived_summary": derived_summary,
                "frame_count": len(frame_ids),
                "frame_ids": observed_frame_ids,
                "frame_id_normalization": per_frame[
                    "frame_id_normalization"
                ],
                "delta": {
                    "columns": per_frame["components"]["delta"]["columns"],
                    "row_count": len(frame_ids),
                },
                "derived": {
                    "evaluator": _DMM_EVALUATOR,
                    "label": _DMM_LABEL,
                    "source_model_id": request.model_id,
                    "columns": dmm["columns"],
                    "row_count": len(frame_ids),
                },
            },
            generated_inputs=generated,
        )


def _normalized_energy_term(value: object) -> str:
    return "".join(
        character
        for character in str(value).upper()
        if character.isalnum()
    )


def derive_v1_dmm(
    model_id: str,
    delta_table: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Derive the historical v1 dMM estimator from one validated delta table.

    This is deliberately strict: every term in the declared formula must be
    present exactly once.  Missing PB dispersion therefore cannot silently
    become zero.
    """

    formulas = {
        "gb": (("VDWAALS", 1.0), ("EEL", 0.2), ("EGB", 0.2), ("ESURF", 1.0)),
        "pb": (
            ("VDWAALS", 1.0),
            ("EEL", 0.2),
            ("EPB", 0.2),
            ("ENPOLAR", 1.0),
            ("EDISPER", 1.0),
        ),
    }
    try:
        formula = formulas[model_id]
    except KeyError as exc:
        raise BackendOutputError(
            f"dMM evaluator does not support model {model_id!r}"
        ) from exc
    raw_columns = delta_table.get("columns")
    raw_rows = delta_table.get("rows")
    if not isinstance(raw_columns, (tuple, list)) or not isinstance(
        raw_rows, (tuple, list)
    ):
        raise BackendOutputError("dMM source must be a parsed numeric table")
    by_normalized: dict[str, list[str]] = {}
    for raw_column in raw_columns:
        column = str(raw_column)
        by_normalized.setdefault(_normalized_energy_term(column), []).append(column)
    frame_matches = [
        column
        for key, columns in by_normalized.items()
        if key in {"FRAME", "FRAMENUMBER", "FRAMEINDEX"}
        for column in columns
    ]
    if len(frame_matches) != 1:
        raise BackendOutputError("dMM source must contain exactly one frame column")
    selected: dict[str, str] = {}
    for term, _ in formula:
        matches = by_normalized.get(_normalized_energy_term(term), ())
        if len(matches) != 1:
            raise BackendOutputError(
                f"dMM {model_id} requires exactly one {term} column; "
                f"observed {len(matches)}"
            )
        selected[term] = matches[0]
    frame_column = frame_matches[0]
    rows: list[dict[str, float]] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise BackendOutputError(f"dMM source row {index} is not a mapping")
        try:
            frame = float(raw_row[frame_column])
            value = sum(
                coefficient * float(raw_row[selected[term]])
                for term, coefficient in formula
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendOutputError(
                f"dMM source row {index} is missing a finite numeric term"
            ) from exc
        if not math.isfinite(frame) or not math.isfinite(value):
            raise BackendOutputError(f"dMM source row {index} is non-finite")
        rows.append({frame_column: frame, _DMM_LABEL: value})
    if not rows:
        raise BackendOutputError("dMM source contains no frames")
    return {
        "columns": (frame_column, _DMM_LABEL),
        "rows": tuple(rows),
        "evaluator": _DMM_EVALUATOR,
        "source_model_id": model_id,
        "unit": "kcal/mol",
    }


def parse_mmpbsa_summary(
    path: Path,
    *,
    model_id: str | None = None,
) -> Mapping[str, Any]:
    """Parse numeric summary rows and optionally select one canonical model."""

    source = require_file(path, "MMPBSA final output")
    rows: list[dict[str, object]] = []
    section = ""
    subsection = ""
    for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":") and not _SUMMARY_ROW.match(line):
            if line.casefold().startswith(("differences", "complex", "receptor", "ligand")):
                subsection = line.removesuffix(":").strip()
            else:
                section = line.removesuffix(":").strip()
                subsection = ""
            continue
        match = _SUMMARY_ROW.match(line)
        if match is None:
            continue
        values = tuple(float(match.group(name)) for name in ("average", "stddev", "stderr"))
        if not all(math.isfinite(value) for value in values):
            raise BackendOutputError(f"MMPBSA summary has non-finite row: {line}")
        rows.append(
            {
                "section": section,
                "subsection": subsection,
                "label": match.group("label").strip(),
                "average": values[0],
                "stddev": values[1],
                "stderr": values[2],
            }
        )
    if not rows:
        raise BackendOutputError(f"MMPBSA final output contains no numeric summary rows: {source}")
    canonical: list[dict[str, object]] = []
    for row in rows:
        observed_model = _SECTION_MODELS.get(str(row["section"]).casefold())
        observed_component = _SUMMARY_COMPONENTS.get(
            str(row["subsection"]).casefold()
        )
        if observed_model is None or observed_component is None:
            continue
        if model_id is not None and observed_model != model_id:
            continue
        canonical.append(
            {
                "model_id": observed_model,
                "component": observed_component,
                "label": row["label"],
                "average": row["average"],
                "stddev": row["stddev"],
                "stderr": row["stderr"],
            }
        )
    if model_id is not None and not canonical:
        raise BackendOutputError(
            f"MMPBSA final output contains no summary rows for model {model_id!r}: "
            f"{source}"
        )
    return {"model_id": model_id, "rows": tuple(canonical)}


def parse_numeric_csv(
    path: Path,
    *,
    selector: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Parse either one numeric CSV table or MMPBSA's named CSV tables."""

    _validate_per_frame_selector(selector)
    source = require_file(path, "numeric CSV")
    lines = tuple(
        (line_number, raw.strip())
        for line_number, raw in enumerate(
            source.read_text(encoding="utf-8", errors="strict").splitlines(),
            start=1,
        )
        if raw.strip() and not raw.lstrip().startswith(("#", "|"))
    )
    if not lines:
        raise BackendOutputError(f"numeric CSV is empty: {source}")

    if not _is_section_heading(lines[0][1]):
        if selector is not None:
            raise BackendOutputError(
                "numeric CSV is a single unnamed table and cannot satisfy "
                "per_frame_selector"
            )
        return _parse_numeric_table(
            source,
            header=lines[0],
            data=lines[1:],
            label="numeric CSV",
        )

    tables = _parse_named_numeric_tables(source, lines)
    if selector is None:
        if len(tables) != 1:
            raise BackendOutputError(
                f"numeric CSV contains {len(tables)} named tables; "
                "per_frame_selector is required"
            )
        selected = tables[0]
    else:
        matches = [
            table
            for table in tables
            if table["section"] == selector["section"]
            and table["subsection"] == selector["subsection"]
        ]
        if len(matches) != 1:
            available = ", ".join(
                f"{table['section']} / {table['subsection']}"
                for table in tables
            )
            raise BackendOutputError(
                "per_frame_selector did not match exactly one numeric CSV table: "
                f"{selector['section']} / {selector['subsection']}; "
                f"available: {available}"
            )
        selected = matches[0]
    return {
        "columns": selected["columns"],
        "rows": selected["rows"],
        "selected": {
            "section": selected["section"],
            "subsection": selected["subsection"],
        },
        "tables": tables,
    }


def parse_mmpbsa_models_csv(
    path: Path,
    *,
    model_id: str | None = None,
    expected_frames: int | None = None,
) -> Mapping[str, Any]:
    """Return canonical model/component tables after strict frame validation.

    ``model_id=None`` is intentionally supported for the v1 workspace
    migrator, which must split one historical combined GB/PB output without
    re-running the scientific calculation. New scoring attempts always pass a
    concrete model and therefore reject a CSV containing any other model.
    """

    if model_id is not None and model_id not in _MODEL_SECTIONS:
        raise BackendConfigurationError(
            "MMPBSA model_id must be one of: " + ", ".join(_MODEL_SECTIONS)
        )
    if expected_frames is not None and (
        isinstance(expected_frames, bool)
        or not isinstance(expected_frames, int)
        or expected_frames < 1
    ):
        raise BackendConfigurationError(
            "MMPBSA expected_frames must be a positive integer"
        )
    source = require_file(path, "MMPBSA per-frame CSV")
    lines = tuple(
        (line_number, raw.strip())
        for line_number, raw in enumerate(
            source.read_text(encoding="utf-8", errors="strict").splitlines(),
            start=1,
        )
        if raw.strip() and not raw.lstrip().startswith(("#", "|"))
    )
    if not lines or not _is_section_heading(lines[0][1]):
        raise BackendOutputError(
            f"MMPBSA per-frame CSV must contain named model tables: {source}"
        )
    raw_tables = _parse_named_numeric_tables(source, lines)
    by_model: dict[str, dict[str, Mapping[str, Any]]] = {}
    unknown: list[str] = []
    for table in raw_tables:
        observed_model = _SECTION_MODELS.get(str(table["section"]).casefold())
        observed_component = _COMPONENT_SUBSECTIONS.get(
            str(table["subsection"]).casefold()
        )
        if observed_model is None or observed_component is None:
            unknown.append(f"{table['section']} / {table['subsection']}")
            continue
        components = by_model.setdefault(observed_model, {})
        if observed_component in components:
            raise BackendOutputError(
                "MMPBSA per-frame CSV repeats canonical table "
                f"{observed_model}/{observed_component}: {source}"
            )
        components[observed_component] = {
            "columns": table["columns"],
            "rows": table["rows"],
        }
    if unknown:
        raise BackendOutputError(
            "MMPBSA per-frame CSV contains unsupported tables: "
            + ", ".join(unknown)
        )
    if model_id is not None:
        observed = set(by_model)
        if observed != {model_id}:
            raise BackendOutputError(
                "model-specific MMPBSA output must contain exactly "
                f"{model_id!r}; observed {', '.join(sorted(observed)) or '(none)'}"
            )
        return _validated_model_tables(
            source,
            model_id,
            by_model[model_id],
            expected_frames=expected_frames,
        )
    if not by_model:
        raise BackendOutputError(
            f"MMPBSA per-frame CSV contains no supported models: {source}"
        )
    return {
        "models": {
            observed_model: _validated_model_tables(
                source,
                observed_model,
                components,
                expected_frames=expected_frames,
            )
            for observed_model, components in sorted(by_model.items())
        }
    }


def _validated_model_tables(
    source: Path,
    model_id: str,
    components: Mapping[str, Mapping[str, Any]],
    *,
    expected_frames: int | None,
) -> Mapping[str, Any]:
    required = {"complex", "receptor", "ligand", "delta"}
    observed = set(components)
    if observed != required:
        missing = sorted(required.difference(observed))
        unexpected = sorted(observed.difference(required))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise BackendOutputError(
            f"MMPBSA {model_id} component tables are incomplete: "
            + "; ".join(details)
        )
    frame_ids_by_component: dict[str, tuple[int, ...]] = {}
    frame_columns_by_component: dict[str, str] = {}
    for component, table in sorted(components.items()):
        columns = table["columns"]
        rows = table["rows"]
        frame_columns = [
            column
            for column in columns
            if _normalized_heading(str(column)) in {"frame", "framenumber"}
        ]
        if len(frame_columns) != 1:
            raise BackendOutputError(
                f"MMPBSA {model_id}/{component} must have exactly one frame column: "
                f"{source}"
            )
        frame_column = frame_columns[0]
        frame_columns_by_component[component] = frame_column
        frame_ids: list[int] = []
        for row in rows:
            raw_frame = row[frame_column]
            if not float(raw_frame).is_integer():
                raise BackendOutputError(
                    f"MMPBSA {model_id}/{component} has a non-integer frame ID: "
                    f"{raw_frame}"
                )
            frame_ids.append(int(raw_frame))
        if len(frame_ids) != len(set(frame_ids)):
            raise BackendOutputError(
                f"MMPBSA {model_id}/{component} repeats frame IDs: {source}"
            )
        if frame_ids != sorted(frame_ids):
            raise BackendOutputError(
                f"MMPBSA {model_id}/{component} frame IDs are not increasing: "
                f"{source}"
            )
        frame_ids_by_component[component] = tuple(frame_ids)
    reference = frame_ids_by_component["complex"]
    if not reference:
        raise BackendOutputError(
            f"MMPBSA {model_id} component tables contain no frames: {source}"
        )
    mismatched = [
        component
        for component, frame_ids in frame_ids_by_component.items()
        if frame_ids != reference
    ]
    if mismatched:
        raise BackendOutputError(
            f"MMPBSA {model_id} component frame IDs differ: "
            + ", ".join(sorted(mismatched))
        )
    frame_id_offset = 0
    canonical_ids = reference
    if expected_frames is not None:
        if len(reference) != expected_frames:
            raise BackendOutputError(
                f"MMPBSA {model_id} produced {len(reference)} frames; "
                f"expected exactly {expected_frames}"
            )
        one_based = tuple(range(1, expected_frames + 1))
        zero_based = tuple(range(expected_frames))
        if reference == zero_based:
            frame_id_offset = 1
            canonical_ids = one_based
        elif reference == one_based:
            canonical_ids = one_based
        else:
            raise BackendOutputError(
                f"MMPBSA {model_id} frame IDs must equal 0..{expected_frames - 1} "
                f"or 1..{expected_frames}: "
                f"{source}"
            )
    elif reference == tuple(range(len(reference))):
        frame_id_offset = 1
        canonical_ids = tuple(range(1, len(reference) + 1))

    normalized_components: dict[str, Mapping[str, Any]] = {}
    for component in sorted(required):
        table = components[component]
        frame_column = frame_columns_by_component[component]
        normalized_components[component] = {
            "columns": table["columns"],
            "rows": tuple(
                {
                    **row,
                    frame_column: int(row[frame_column]) + frame_id_offset,
                }
                for row in table["rows"]
            ),
        }
    return {
        "model_id": model_id,
        "frame_ids": canonical_ids,
        "frame_id_normalization": {
            "source_first": reference[0],
            "source_last": reference[-1],
            "offset": frame_id_offset,
            "canonical_first": canonical_ids[0],
            "canonical_last": canonical_ids[-1],
        },
        "components": normalized_components,
    }


def _write_delta_csv(path: Path, table: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(str(column) for column in table["columns"])
    rows = table["rows"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(tuple(row[column] for column in columns))
    return destination


def _normalized_heading(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _parse_named_numeric_tables(
    source: Path,
    lines: tuple[tuple[int, str], ...],
) -> tuple[Mapping[str, Any], ...]:
    tables: list[Mapping[str, Any]] = []
    table_keys: set[tuple[str, str]] = set()
    sections: set[str] = set()
    index = 0
    while index < len(lines):
        section_line, raw_section = lines[index]
        if not _is_section_heading(raw_section):
            raise BackendOutputError(
                f"numeric CSV expected a section heading at line {section_line}: "
                f"{source}"
            )
        section = raw_section.removesuffix(":").strip()
        if not section:
            raise BackendOutputError(
                f"numeric CSV has an empty section at line {section_line}: {source}"
            )
        if section in sections:
            raise BackendOutputError(
                f"numeric CSV repeats section {section!r} at line "
                f"{section_line}: {source}"
            )
        sections.add(section)
        index += 1
        section_table_count = 0

        while index < len(lines) and not _is_section_heading(lines[index][1]):
            subsection_line, subsection = lines[index]
            if "," in subsection or subsection.endswith(":"):
                raise BackendOutputError(
                    f"numeric CSV expected a subsection heading at line "
                    f"{subsection_line}: {source}"
                )
            index += 1
            if index >= len(lines):
                raise BackendOutputError(
                    f"numeric CSV subsection {subsection!r} has no header: {source}"
                )
            header = lines[index]
            if "," not in header[1]:
                raise BackendOutputError(
                    f"numeric CSV subsection {subsection!r} has no CSV header "
                    f"at line {header[0]}: {source}"
                )
            index += 1
            data_start = index
            while index < len(lines) and "," in lines[index][1]:
                index += 1
            table = _parse_numeric_table(
                source,
                header=header,
                data=lines[data_start:index],
                label=f"{section} / {subsection}",
            )
            if len(table["columns"]) < 2:
                raise BackendOutputError(
                    f"numeric CSV named table has fewer than two columns: "
                    f"{section} / {subsection}"
                )
            key = (section, subsection)
            if key in table_keys:
                raise BackendOutputError(
                    f"numeric CSV repeats table {section} / {subsection}: {source}"
                )
            table_keys.add(key)
            tables.append(
                {
                    "section": section,
                    "subsection": subsection,
                    **table,
                }
            )
            section_table_count += 1

        if section_table_count == 0:
            raise BackendOutputError(
                f"numeric CSV section {section!r} has no tables: {source}"
            )

    return tuple(tables)


def _parse_numeric_table(
    source: Path,
    *,
    header: tuple[int, str],
    data: tuple[tuple[int, str], ...],
    label: str,
) -> Mapping[str, Any]:
    columns = _parse_csv_record(source, header, label=label)
    if not columns or any(not column.strip() for column in columns):
        raise BackendOutputError(
            f"{label} has no valid header at line {header[0]}: {source}"
        )
    if len(columns) != len(set(columns)):
        raise BackendOutputError(
            f"{label} has duplicate columns at line {header[0]}: {source}"
        )

    rows: list[dict[str, float]] = []
    for record in data:
        values = _parse_csv_record(source, record, label=label)
        if len(values) != len(columns):
            raise BackendOutputError(
                f"{label} has {len(values)} fields at line {record[0]}; "
                f"expected {len(columns)}: {source}"
            )
        parsed: dict[str, float] = {}
        for column, raw in zip(columns, values, strict=True):
            try:
                value = float(raw)
            except ValueError as exc:
                raise BackendOutputError(
                    f"{label} has non-numeric {column!r} at line {record[0]}"
                ) from exc
            if not math.isfinite(value):
                raise BackendOutputError(
                    f"{label} has non-finite {column!r} at line {record[0]}"
                )
            parsed[column] = value
        rows.append(parsed)
    if not rows:
        raise BackendOutputError(f"{label} has no data rows: {source}")
    return {"columns": columns, "rows": tuple(rows)}


def _parse_csv_record(
    source: Path,
    record: tuple[int, str],
    *,
    label: str,
) -> tuple[str, ...]:
    try:
        parsed = tuple(csv.reader((record[1],), strict=True))
    except csv.Error as exc:
        raise BackendOutputError(
            f"{label} has invalid CSV syntax at line {record[0]}: {source}"
        ) from exc
    if len(parsed) != 1:
        raise BackendOutputError(
            f"{label} has an invalid record at line {record[0]}: {source}"
        )
    return tuple(parsed[0])


def _is_section_heading(value: str) -> bool:
    return "," not in value and value.endswith(":")


def create_ambertools_scoring_backend(
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None = None,
) -> AmberToolsScoringBackend:
    launcher = string_tuple(config.get("mpi_launcher"), "mpi_launcher")
    ranks_raw = config.get("ranks", 1)
    if isinstance(ranks_raw, bool):
        raise BackendConfigurationError("MMPBSA ranks must be an integer")
    try:
        ranks = int(ranks_raw)
    except (TypeError, ValueError) as exc:
        raise BackendConfigurationError("MMPBSA ranks must be an integer") from exc
    if isinstance(ranks_raw, float) and not ranks_raw.is_integer():
        raise BackendConfigurationError("MMPBSA ranks must be an integer")
    tool = tool_from_config(
        config,
        default_executable="MMPBSA.py",
        default_version_arguments=("--version",),
    )
    sander = _bind_configured_sander(
        tool,
        tool_from_config(
            config,
            key="sander",
            default_executable="sander",
            default_version_arguments=("--version",),
        ),
    )
    return AmberToolsScoringBackend(
        tool=tool,
        sander=sander,
        mpi_launcher=launcher,
        ranks=ranks,
        runner=runner,
    )


def _expected_sander(tool: ToolConfig) -> Path:
    amberhome = tool.environment.get("AMBERHOME")
    if not isinstance(amberhome, str) or not amberhome.strip():
        raise BackendConfigurationError(
            "MMPBSA use_sander execution requires an explicit AMBERHOME "
            "in the scoring environment"
        )
    root = Path(amberhome).expanduser()
    if not root.is_absolute():
        raise BackendConfigurationError(
            "scoring environment AMBERHOME must be an absolute path"
        )
    return (root / "bin" / "sander").resolve()


def _bind_configured_sander(
    tool: ToolConfig,
    sander: ToolConfig,
) -> ToolConfig:
    """Bind the probe to the path MMPBSA.py 14 resolves at execution."""

    expected = _expected_sander(tool)
    configured = Path(sander.executable).expanduser()
    if configured.is_absolute() or configured.parent != Path("."):
        if configured.resolve() != expected:
            raise BackendConfigurationError(
                "configured sander differs from AMBERHOME/bin/sander used "
                f"by MMPBSA: {configured} != {expected}"
            )
    elif configured.name != "sander":
        raise BackendConfigurationError(
            "relative scoring sander executable must be exactly 'sander'"
        )
    return replace(
        sander,
        executable=str(expected),
        environment={
            **dict(sander.environment),
            "AMBERHOME": str(expected.parent.parent),
        },
    )


def _validate_bound_sander(
    tool: ToolConfig,
    sander: ToolConfig,
) -> None:
    expected = _expected_sander(tool)
    configured = Path(sander.executable).expanduser()
    if not configured.is_absolute() or configured.resolve() != expected:
        raise BackendConfigurationError(
            "AmberToolsScoringBackend.sander must be the absolute "
            "AMBERHOME/bin/sander path"
        )
    if sander.environment.get("AMBERHOME") != str(expected.parent.parent):
        raise BackendConfigurationError(
            "MMPBSA and sander probes must use the same AMBERHOME"
        )


def _terminal_newline(value: str) -> str:
    return value.rstrip("\n") + "\n"


def _validate_native_inputs(
    model_id: str,
    value: Mapping[str, str] | None,
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise BackendConfigurationError(
            "MMPBSA native_inputs must be a mapping with string keys"
        )
    expected = frozenset(item[0] for item in _NATIVE_INPUT_LAYOUTS[model_id])
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise BackendConfigurationError(
            f"MMPBSA native_inputs for {model_id!r} require exactly "
            + ", ".join(sorted(expected))
            + (": " + "; ".join(details) if details else "")
        )
    for key in sorted(expected):
        text = value[key]
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise BackendConfigurationError(
                f"MMPBSA native_inputs.{key} must be a non-empty string"
            )


def _validate_per_frame_selector(value: Mapping[str, str] | None) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise BackendConfigurationError(
            "MMPBSA per_frame_selector must be a mapping with string keys"
        )
    actual = frozenset(value)
    if actual != _PER_FRAME_SELECTOR_KEYS:
        raise BackendConfigurationError(
            "MMPBSA per_frame_selector requires exactly section, subsection"
        )
    for key in sorted(_PER_FRAME_SELECTOR_KEYS):
        text = value[key]
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise BackendConfigurationError(
                f"MMPBSA per_frame_selector.{key} must be a non-empty string"
            )

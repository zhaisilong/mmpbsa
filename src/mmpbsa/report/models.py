"""Typed contracts for report assembly, rendering, and display assets."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TypeAlias
from urllib.parse import urlsplit

Scalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = Scalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_SCHEMES = frozenset({"http", "https", "mailto"})
PREFLIGHT_STATUSES = frozenset({"preflight_rejected", "preflight_pending", "preflight_skipped"})


class NamedSlot(str, Enum):
    """Stable placement points shared by standard and project contributors."""

    OVERVIEW = "overview"
    PROJECT_AFTER_SUMMARY = "project.after_summary"
    CASE_AFTER_SUMMARY = "case.after_summary"
    PROTOCOL = "protocol"
    INVENTORY = "inventory"
    SCORES = "scores"
    PROJECT_AFTER_SCORES = "project.after_scores"
    QC = "qc"
    CASE_AFTER_QC = "case.after_qc"
    TRAJECTORY = "trajectory"
    CONVERGENCE = "convergence"
    CASE_AFTER_ENERGY = "case.after_energy"
    CORRELATION = "correlation"
    DECOMPOSITION = "decomposition"
    ACTIONS = "actions"
    ARTIFACTS = "artifacts"
    PROJECT_BEFORE_PROVENANCE = "project.before_provenance"
    CASE_BEFORE_PROVENANCE = "case.before_provenance"
    PROVENANCE = "provenance"


SLOT_ORDER: tuple[NamedSlot, ...] = tuple(NamedSlot)


class NoticeLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKED = "blocked"


class PlotKind(str, Enum):
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"


class PageKind(str, Enum):
    PROJECT = "project"
    CASE = "case"


@dataclass(frozen=True, slots=True)
class Link:
    label: str
    href: str
    description: str = ""

    def __post_init__(self) -> None:
        _require_text(self.label, "Link label")
        _validate_href(self.href)


Cell: TypeAlias = Scalar | Link


@dataclass(frozen=True, slots=True)
class TableColumn:
    key: str
    label: str
    unit: str = ""
    numeric: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.key, "Table column key")
        _require_text(self.label, "Table column label")


@dataclass(frozen=True, slots=True)
class PlotPoint:
    x: float
    y: float
    label: str = ""
    emphasized: bool = False
    break_before: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("Plot coordinates must be finite")


@dataclass(frozen=True, slots=True)
class PlotSeries:
    series_id: str
    label: str
    points: tuple[PlotPoint, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.series_id, "Plot series ID")
        _require_text(self.label, "Plot series label")
        if not self.points:
            raise ValueError("Plot series must contain at least one point")


@dataclass(frozen=True, slots=True)
class PlotSpec:
    kind: PlotKind
    x_label: str
    y_label: str
    series: tuple[PlotSeries, ...]
    zero_line: bool = False
    x_min: float | None = None
    y_min: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.x_label, "Plot x-axis label")
        _require_text(self.y_label, "Plot y-axis label")
        if not self.series:
            raise ValueError("Plot spec must contain at least one series")
        ids = tuple(item.series_id for item in self.series)
        if len(ids) != len(set(ids)):
            raise ValueError("Plot series IDs must be unique")
        if self.x_min is not None:
            if not math.isfinite(self.x_min):
                raise ValueError("Plot x-axis minimum must be finite")
            if any(
                point.x < self.x_min
                for series in self.series
                for point in series.points
            ):
                raise ValueError("Plot points cannot fall below x-axis minimum")
        if self.y_min is not None:
            if not math.isfinite(self.y_min):
                raise ValueError("Plot y-axis minimum must be finite")
            if any(
                point.y < self.y_min
                for series in self.series
                for point in series.points
            ):
                raise ValueError("Plot points cannot fall below y-axis minimum")


@dataclass(frozen=True, slots=True)
class NoticeBlock:
    block_id: str
    title: str
    body: str
    level: NoticeLevel = NoticeLevel.INFO

    def __post_init__(self) -> None:
        _validate_block_header(self.block_id, self.title)
        _require_text(self.body, "Notice body")


@dataclass(frozen=True, slots=True)
class TableBlock:
    block_id: str
    title: str
    columns: tuple[TableColumn, ...]
    rows: tuple[tuple[Cell, ...], ...]
    caption: str = ""

    def __post_init__(self) -> None:
        _validate_block_header(self.block_id, self.title)
        if not self.columns:
            raise ValueError("Table block requires at least one column")
        keys = tuple(column.key for column in self.columns)
        if len(keys) != len(set(keys)):
            raise ValueError("Table column keys must be unique")
        expected = len(self.columns)
        if any(len(row) != expected for row in self.rows):
            raise ValueError("Every table row must match the declared columns")
        for row in self.rows:
            for cell in row:
                if not isinstance(cell, (str, int, float, bool, Link, type(None))):
                    raise TypeError(f"Unsupported typed table cell: {type(cell).__name__}")
                if isinstance(cell, float) and not math.isfinite(cell):
                    raise ValueError("Numeric table cells must be finite")


@dataclass(frozen=True, slots=True)
class PlotBlock:
    block_id: str
    title: str
    plot: PlotSpec
    caption: str = ""

    def __post_init__(self) -> None:
        _validate_block_header(self.block_id, self.title)


@dataclass(frozen=True, slots=True)
class PlotPanel:
    """One titled plot inside a responsive small-multiple group."""

    panel_id: str
    title: str
    plot: PlotSpec
    caption: str = ""

    def __post_init__(self) -> None:
        _validate_block_header(self.panel_id, self.title)


@dataclass(frozen=True, slots=True)
class PlotGroupBlock:
    block_id: str
    title: str
    panels: tuple[PlotPanel, ...]
    caption: str = ""

    def __post_init__(self) -> None:
        _validate_block_header(self.block_id, self.title)
        if not self.panels:
            raise ValueError("Plot group requires at least one panel")
        panel_ids = tuple(panel.panel_id for panel in self.panels)
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError("Plot panel IDs must be unique within a group")


@dataclass(frozen=True, slots=True)
class AssetBlock:
    block_id: str
    title: str
    links: tuple[Link, ...]
    caption: str = ""

    def __post_init__(self) -> None:
        _validate_block_header(self.block_id, self.title)
        if not self.links:
            raise ValueError("Asset block requires at least one link")


ReportBlock: TypeAlias = (
    NoticeBlock | TableBlock | PlotBlock | PlotGroupBlock | AssetBlock
)
REPORT_BLOCK_TYPES = (
    NoticeBlock,
    TableBlock,
    PlotBlock,
    PlotGroupBlock,
    AssetBlock,
)


@dataclass(frozen=True, slots=True)
class SlotContent:
    slot: NamedSlot
    blocks: tuple[ReportBlock, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("Empty report slots must be omitted")
        for block in self.blocks:
            if not isinstance(block, REPORT_BLOCK_TYPES):
                raise TypeError("Only typed report blocks can be placed in a slot")


@dataclass(frozen=True, slots=True)
class PageTarget:
    kind: PageKind
    case_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind is PageKind.PROJECT:
            if self.case_id is not None:
                raise ValueError("Project page target cannot carry a case ID")
        elif self.case_id is None:
            raise ValueError("Case page target requires a case ID")
        else:
            _require_path_identifier(self.case_id, "Case ID")

    @classmethod
    def project(cls) -> PageTarget:
        return cls(PageKind.PROJECT)

    @classmethod
    def case(cls, case_id: str) -> PageTarget:
        return cls(PageKind.CASE, case_id)

    @property
    def page_id(self) -> str:
        return "project" if self.kind is PageKind.PROJECT else f"case:{self.case_id}"


@dataclass(frozen=True, slots=True)
class PageSpec:
    target: PageTarget
    path: PurePosixPath
    title: str
    slots: tuple[SlotContent, ...]

    def __post_init__(self) -> None:
        _require_text(self.title, "Page title")
        _validate_output_path(self.path)
        names = tuple(content.slot for content in self.slots)
        if len(names) != len(set(names)):
            raise ValueError("A named slot may occur only once per page")
        expected_order = sorted(names, key=SLOT_ORDER.index)
        if list(names) != expected_order:
            raise ValueError("Page slots must use canonical slot order")
        ids = tuple(block.block_id for slot in self.slots for block in slot.blocks)
        if len(ids) != len(set(ids)):
            raise ValueError("Block IDs must be unique within a page")


@dataclass(frozen=True, slots=True)
class DocumentSpec:
    title: str
    pages: tuple[PageSpec, ...]
    schema_version: str = "mmpbsa.report.document.v1"

    def __post_init__(self) -> None:
        _require_text(self.title, "Document title")
        if not self.pages or self.pages[0].path != PurePosixPath("index.html"):
            raise ValueError("The first report page must be index.html")
        paths = tuple(page.path for page in self.pages)
        targets = tuple(page.target.page_id for page in self.pages)
        if len(paths) != len(set(paths)) or len(targets) != len(set(targets)):
            raise ValueError("Report page paths and targets must be unique")
        for page in self.pages[1:]:
            expected = PurePosixPath("cases") / str(page.target.case_id) / "index.html"
            if page.target.kind is not PageKind.CASE or page.path != expected:
                raise ValueError("Case pages must use cases/<case-id>/index.html")


@dataclass(frozen=True, slots=True)
class ScoreColumn:
    key: str
    label: str
    unit: str = ""
    precision: int = 2

    def __post_init__(self) -> None:
        _require_identifier(self.key, "Score key")
        _require_text(self.label, "Score label")
        if not 0 <= self.precision <= 8:
            raise ValueError("Score precision must be between 0 and 8")


@dataclass(frozen=True, slots=True)
class QcIssue:
    code: str
    message: str
    verdict: str
    replica_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.code, "QC issue code")
        _require_text(self.message, "QC issue message")
        _require_text(self.verdict, "QC verdict")
        if self.replica_id is not None:
            _require_path_identifier(self.replica_id, "Replica ID")


@dataclass(frozen=True, slots=True)
class CorrelationRecord:
    reference: str
    score_key: str
    n: int
    pearson_r: float | None = None
    spearman_r: float | None = None
    kendall_tau_b: float | None = None
    slope: float | None = None
    intercept: float | None = None
    mae: float | None = None
    rmse: float | None = None
    mean_signed_error: float | None = None
    status: str = "available"
    reference_mode: str | None = None
    notes: tuple[str, ...] = ()
    comparison_group: str | None = None
    resolved_internal_dielectric: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.reference, "Correlation reference")
        _require_identifier(self.score_key, "Correlation score key")
        _require_text(self.status, "Correlation status")
        if self.reference_mode is not None:
            _require_identifier(self.reference_mode, "Correlation reference mode")
        if self.comparison_group is not None:
            _require_text(self.comparison_group, "Correlation comparison group")
        if self.resolved_internal_dielectric is not None and (
            not math.isfinite(self.resolved_internal_dielectric)
            or self.resolved_internal_dielectric <= 0
        ):
            raise ValueError("Correlation internal dielectric must be positive and finite")
        for note in self.notes:
            _require_text(note, "Correlation note")
        if self.n < 0:
            raise ValueError("Correlation sample count cannot be negative")
        for value in (
            self.pearson_r,
            self.spearman_r,
            self.kendall_tau_b,
            self.slope,
            self.intercept,
            self.mae,
            self.rmse,
            self.mean_signed_error,
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError("Correlation statistics must be finite")


@dataclass(frozen=True, slots=True)
class DecompositionRecord:
    residue: str
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        _require_text(self.residue, "Decomposition residue")
        normalized = _numeric_mapping(self.values, "Decomposition value", allow_none=False)
        object.__setattr__(self, "values", normalized)


@dataclass(frozen=True, slots=True)
class ReplicaRecord:
    replica_id: str
    status: str
    seed: int | None = None
    frame_count: int | None = None
    qc_verdict: str | None = None
    scores: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_path_identifier(self.replica_id, "Replica ID")
        _require_text(self.status, "Replica status")
        if self.frame_count is not None and self.frame_count < 0:
            raise ValueError("Replica frame count cannot be negative")
        if self.qc_verdict is not None:
            _require_text(self.qc_verdict, "Replica QC verdict")
        object.__setattr__(
            self,
            "scores",
            _numeric_mapping(self.scores, "Replica score", allow_none=True),
        )


@dataclass(frozen=True, slots=True)
class QcContactContext:
    """Descriptive BLOCK context only; none of these fields creates a QC gate."""

    replica_id: str
    sample_count: int
    native_zero_fraction: float
    longest_native_zero_samples: int
    nonnative_at_native_zero_samples: int | None = None
    total_contacts_min: float | None = None

    def __post_init__(self) -> None:
        _require_path_identifier(self.replica_id, "Replica ID")
        if self.sample_count < 1 or not 0 <= self.native_zero_fraction <= 1:
            raise ValueError("Invalid contact context sample count/fraction")
        if not 0 <= self.longest_native_zero_samples <= self.sample_count:
            raise ValueError("Invalid contact context zero run")
        if self.nonnative_at_native_zero_samples is not None and not (
            0 <= self.nonnative_at_native_zero_samples <= self.sample_count
        ):
            raise ValueError("Invalid nonnative contact context sample count")
        if self.total_contacts_min is not None and (
            not math.isfinite(self.total_contacts_min) or self.total_contacts_min < 0
        ):
            raise ValueError("Invalid total contact context")


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    eligible: bool
    reasons: tuple[str, ...] = ()
    group: str | None = None
    dielectric: float | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        if self.eligible and self.reasons:
            raise ValueError("Eligible comparison cannot carry exclusion reasons")
        if not self.eligible and not self.reasons:
            raise ValueError("Excluded comparison needs an explicit reason")
        if self.eligible and (
            self.group is None or self.dielectric is None or self.rank is None
        ):
            raise ValueError("Eligible comparison needs group, dielectric, and rank")
        if self.rank is not None and (not self.eligible or self.rank < 1):
            raise ValueError("Only eligible comparisons can have a positive rank")
        if self.dielectric is not None and (
            not math.isfinite(self.dielectric) or self.dielectric <= 0
        ):
            raise ValueError("Comparison internal dielectric must be positive and finite")
        if self.dielectric is not None and self.group is not None:
            if self.group != f"epsilon={self.dielectric:g}":
                raise ValueError("Comparison group disagrees with internal dielectric")


@dataclass(frozen=True, slots=True)
class CaseRecord:
    case_id: str
    label: str
    expected_replica_ids: tuple[str, ...]
    replicas: tuple[ReplicaRecord, ...] = ()
    status: str = "planned"
    outcome: str = "pending"
    qc_verdict: str | None = None
    metadata: Mapping[str, Scalar] = field(default_factory=dict)
    protocol: Mapping[str, Scalar] = field(default_factory=dict)
    scores: Mapping[str, float | None] = field(default_factory=dict)
    qc_issues: tuple[QcIssue, ...] = ()
    trajectory_plots: tuple[PlotPanel, ...] = ()
    energy_plots: tuple[PlotPanel, ...] = ()
    convergence_plots: tuple[PlotPanel, ...] = ()
    trajectory: PlotSpec | None = None
    energy: PlotSpec | None = None
    convergence: PlotSpec | None = None
    occupancy_basin: PlotSpec | None = None
    occupancy_basin_note: str = ""
    correlations: tuple[CorrelationRecord, ...] = ()
    correlation_plot: PlotSpec | None = None
    decomposition: tuple[DecompositionRecord, ...] = ()
    actions: tuple[Link, ...] = ()
    artifacts: tuple[Link, ...] = ()
    provenance: Mapping[str, Scalar] = field(default_factory=dict)
    comparisons: Mapping[str, ComparisonRecord] = field(default_factory=dict)
    qc_contact_context: tuple[QcContactContext, ...] = ()

    @property
    def is_preflight(self) -> bool:
        """A report-only admission state, with no simulated replica evidence."""
        return self.status in PREFLIGHT_STATUSES

    def __post_init__(self) -> None:
        _require_path_identifier(self.case_id, "Case ID")
        _require_text(self.label, "Case label")
        _require_text(self.status, "Case status")
        _require_text(self.outcome, "Case outcome")
        _validate_expected_replicas(self.expected_replica_ids)
        ids = tuple(replica.replica_id for replica in self.replicas)
        if len(ids) != len(set(ids)):
            raise ValueError("Replica records must be unique within a case")
        if self.qc_verdict is not None:
            _require_text(self.qc_verdict, "Case QC verdict")
        if self.occupancy_basin_note and not self.occupancy_basin_note.strip():
            raise ValueError("Occupancy-basin note cannot contain only whitespace")
        for label, panels in (
            ("trajectory", self.trajectory_plots),
            ("energy", self.energy_plots),
            ("convergence", self.convergence_plots),
        ):
            panel_ids = tuple(panel.panel_id for panel in panels)
            if len(panel_ids) != len(set(panel_ids)):
                raise ValueError(f"Case {label} plot panel IDs must be unique")
        object.__setattr__(self, "metadata", _scalar_mapping(self.metadata))
        object.__setattr__(self, "protocol", _scalar_mapping(self.protocol))
        object.__setattr__(
            self,
            "scores",
            _numeric_mapping(self.scores, "Case score", allow_none=True),
        )
        if any(not isinstance(value, ComparisonRecord) for value in self.comparisons.values()):
            raise TypeError("Case comparisons must contain ComparisonRecord values")
        object.__setattr__(self, "comparisons", dict(self.comparisons))
        object.__setattr__(self, "provenance", _scalar_mapping(self.provenance))


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    project_id: str
    title: str
    cases: tuple[CaseRecord, ...]
    score_columns: tuple[ScoreColumn, ...] = ()
    rank_by: str | None = None
    generated_at: str = ""
    metadata: Mapping[str, Scalar] = field(default_factory=dict)
    protocol: Mapping[str, Scalar] = field(default_factory=dict)
    correlations: tuple[CorrelationRecord, ...] = ()
    correlation_plot: PlotSpec | None = None
    artifacts: tuple[Link, ...] = ()
    provenance: Mapping[str, Scalar] = field(default_factory=dict)
    comparison_contract: bool = False

    def __post_init__(self) -> None:
        _require_path_identifier(self.project_id, "Project ID")
        _require_text(self.title, "Project title")
        ids = tuple(case.case_id for case in self.cases)
        if len(ids) != len(set(ids)):
            raise ValueError("Project case IDs must be unique")
        keys = tuple(column.key for column in self.score_columns)
        if len(keys) != len(set(keys)):
            raise ValueError("Score columns must have unique keys")
        if self.rank_by is not None and self.rank_by not in set(keys):
            raise ValueError("rank_by must identify a declared score column")
        object.__setattr__(self, "metadata", _scalar_mapping(self.metadata))
        object.__setattr__(self, "protocol", _scalar_mapping(self.protocol))
        object.__setattr__(self, "provenance", _scalar_mapping(self.provenance))


@dataclass(frozen=True, slots=True)
class Contribution:
    """One safe, typed addition to a named slot."""

    page: PageTarget
    slot: NamedSlot
    block: ReportBlock
    contributor_id: str
    priority: int = 100

    def __post_init__(self) -> None:
        _require_identifier(self.contributor_id, "Contributor ID")
        if not isinstance(self.block, REPORT_BLOCK_TYPES):
            raise TypeError("Contributors may return only typed report blocks")


@dataclass(frozen=True, slots=True)
class AssetRequest:
    """A display-only request fulfilled by an injected structural provider."""

    request_id: str
    kind: str
    outputs: tuple[PurePosixPath, ...]
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.request_id, "Asset request ID")
        _require_identifier(self.kind, "Asset request kind")
        if not self.outputs:
            raise ValueError("Asset request must declare at least one output")
        for output in self.outputs:
            _validate_output_path(output)
            if output.suffix.lower() == ".html":
                raise ValueError("Asset requests cannot own report HTML pages")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("Asset request outputs must be unique")
        object.__setattr__(self, "parameters", _json_mapping(self.parameters))


def case_page_path(case_id: str) -> PurePosixPath:
    _require_path_identifier(case_id, "Case ID")
    return PurePosixPath("cases") / case_id / "index.html"


def _validate_block_header(block_id: str, title: str) -> None:
    _require_identifier(block_id, "Block ID")
    _require_text(title, "Block title")


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")


def _require_identifier(value: str, label: str) -> None:
    _require_text(value, label)
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, digits, '.', '_' or '-'")


def _require_path_identifier(value: str, label: str) -> None:
    _require_identifier(value, label)
    if value in {".", ".."}:
        raise ValueError(f"{label} cannot be '.' or '..'")


def _validate_expected_replicas(replica_ids: tuple[str, ...]) -> None:
    if len(replica_ids) != len(set(replica_ids)):
        raise ValueError("Expected replica IDs must be unique")
    for replica_id in replica_ids:
        _require_path_identifier(replica_id, "Expected replica ID")


def _validate_href(href: str) -> None:
    _require_text(href, "Link href")
    if any(character in href for character in ("\x00", "\r", "\n")):
        raise ValueError("Link href contains control characters")
    if "\\" in href:
        raise ValueError("Backslashes are not supported in report links")
    parsed = urlsplit(href)
    scheme = parsed.scheme.lower()
    if scheme and scheme not in _SAFE_SCHEMES:
        raise ValueError(f"Unsafe link scheme: {scheme}")
    if not scheme and href.startswith("//"):
        raise ValueError("Protocol-relative links are not supported")
    if not scheme and parsed.path.startswith(("/", "\\")):
        raise ValueError("Local report links must be relative")


def _validate_output_path(path: PurePosixPath) -> None:
    if not isinstance(path, PurePosixPath):
        raise TypeError("Report output paths must be PurePosixPath values")
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe report output path: {path}")
    if "\\" in path.as_posix():
        raise ValueError(f"Unsafe report output path: {path}")


def _scalar_mapping(values: Mapping[str, Scalar]) -> Mapping[str, Scalar]:
    normalized: dict[str, Scalar] = {}
    for key, value in values.items():
        _require_text(str(key), "Metadata key")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"Unsupported scalar value for {key!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Metadata value for {key!r} must be finite")
        normalized[str(key)] = value
    return MappingProxyType(normalized)


def _numeric_mapping(
    values: Mapping[str, float | None],
    label: str,
    *,
    allow_none: bool,
) -> Mapping[str, float | None]:
    normalized: dict[str, float | None] = {}
    for key, value in values.items():
        _require_identifier(str(key), f"{label} key")
        if value is None:
            if not allow_none:
                raise ValueError(f"{label} cannot be missing")
            normalized[str(key)] = None
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        normalized[str(key)] = number
    return MappingProxyType(normalized)


def _json_mapping(values: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    normalized: dict[str, JsonValue] = {}
    for key, value in values.items():
        _require_text(str(key), "Asset parameter key")
        normalized[str(key)] = _json_value(value)
    return MappingProxyType(normalized)


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Asset parameters must be finite")
        return value
    if isinstance(value, tuple):
        return tuple(_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return _json_mapping(value)
    raise TypeError("Asset parameters must contain only immutable JSON-shaped values")

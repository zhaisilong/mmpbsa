"""Provider entry point exposed through ``mmpbsa.providers.v2``."""

from __future__ import annotations

import gzip
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath

from mmpbsa.utils import canonical_json, materialize_immutable_file

from .adapters import adapt_analysis_input
from .builders import ReportContributor, build_document
from .models import AssetRequest, DocumentSpec, Link, ProjectRecord
from .publish import (
    AssetMaterializer,
    AssetOrchestrator,
    PublishResult,
    ReportPublisher,
)
from .render import render_document

__version__ = "0.7.0"

_QUOTED_HOST_PATH = re.compile(
    r"(?P<quote>['\"])(?P<path>(?:file:///|[A-Za-z]:[\\/]|\\\\|/)[^'\"\r\n]+)(?P=quote)",
    re.IGNORECASE,
)
_EMBEDDED_HOST_PATH = re.compile(
    r"file:///[^\s\"'<>|,;]+"
    r"|(?<![A-Za-z0-9:/\\])"
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|,;]+"
    r"|\\\\[^\\\s\"'<>|,;]+\\[^\s\"'<>|,;]+"
    r"|/[^\s\"'<>|,;]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _FallbackBackendCompatibility:
    provider_id: str
    adapter_contract: str
    engine: str
    engine_major: str
    engine_version: str

    @property
    def key(self) -> str:
        return (
            f"{self.provider_id}:{self.adapter_contract}:"
            f"{self.engine}:{self.engine_major}"
        )


@dataclass(frozen=True, slots=True)
class _FallbackStageResult:
    artifacts: dict[str, Path]
    payload: dict[str, object] = field(default_factory=dict)
    qc_verdict: object | None = None
    checkpoint: Path | None = None
    invocation_provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _FallbackResourceRequest:
    cpu_slots: int = 1
    gpu_slots: int = 0


class _CopyAssetMaterializer:
    def __init__(self, sources: dict[str, Path]) -> None:
        self.sources = sources

    def materialize(self, request: AssetRequest, staging_root: Path) -> None:
        source = self.sources[request.request_id]
        if not source.is_file():
            raise FileNotFoundError(f"Upstream display asset is not a file: {source}")
        if len(request.outputs) != 1:
            raise ValueError("Copy asset requests must declare exactly one output")
        destination = staging_root.joinpath(*request.outputs[0].parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Report trees are user-facing mutable publications.  A reflink or
        # copy keeps them independent from sealed upstream attempt artifacts;
        # a hardlink would let either pathname mutate the other after audit.
        materialize_immutable_file(source, destination)


class ReportProvider:
    """Structural provider; importing it does not require core to be installed."""

    provider_id = "report"
    api_version = "2"
    package_version = __version__
    capabilities = ("report",)

    def __init__(self) -> None:
        self._assets = AssetOrchestrator()
        self._publisher = ReportPublisher(asset_orchestrator=self._assets)

    def get_service(self, capability: str) -> object | None:
        if capability == "report":
            return self
        return None

    def resource_request(self, request):
        """Reports use one CPU slot and no GPU."""

        try:
            from mmpbsa.core import ResourceRequest
        except (ImportError, AttributeError):
            return _FallbackResourceRequest()
        return ResourceRequest()

    def build_document(
        self,
        project: ProjectRecord,
        *,
        contributors: Sequence[ReportContributor] = (),
    ) -> DocumentSpec:
        return build_document(project, contributors=contributors)

    def render(self, document: DocumentSpec):
        return render_document(document)

    def publish(
        self,
        document: DocumentSpec,
        destination: Path | str,
        *,
        asset_requests: Sequence[AssetRequest] = (),
        asset_materializer: AssetMaterializer | None = None,
    ) -> PublishResult:
        return self._publisher.publish(
            document,
            destination,
            asset_requests=asset_requests,
            asset_materializer=asset_materializer,
        )

    def compatibility(self, runtime):
        """Return the core contract when available, or its structural twin."""

        compatibility_type, _ = _core_types()
        return compatibility_type(
            provider_id=self.provider_id,
            adapter_contract="mmpbsa.report.stage.v2",
            engine="mmpbsa-report",
            engine_major="0.7",
            engine_version=self.package_version,
        )

    def execute(self, request):
        """Render one project report from the explicit analysis StageInput."""

        upstream = getattr(request, "upstream", None)
        if not isinstance(upstream, dict) and not hasattr(upstream, "get"):
            raise TypeError("Report StageRequest.upstream must be a mapping")
        analysis_input = upstream.get("analysis")
        if analysis_input is None:
            raise ValueError("Report stage requires upstream['analysis']")
        analysis_payload = getattr(analysis_input, "payload", None)
        if not isinstance(analysis_payload, dict) and not hasattr(
            analysis_payload, "get"
        ):
            raise TypeError("Analysis StageInput.payload must be a mapping")
        science = getattr(request, "science", {})
        if not isinstance(science, dict) and not hasattr(science, "get"):
            raise TypeError("Report StageRequest.science must be a mapping")
        assets_input = upstream.get("assets")
        asset_artifacts = (
            getattr(assets_input, "artifacts", {}) if assets_input is not None else {}
        )
        asset_payload = (
            getattr(assets_input, "payload", None) if assets_input is not None else None
        )
        asset_root = (
            Path(assets_input.root) / "output" if assets_input is not None else None
        )
        adapted = adapt_analysis_input(
            analysis_payload,
            report_configuration=science,
            asset_artifacts=asset_artifacts,
            asset_root=asset_root,
            asset_payload=asset_payload,
        )
        attempt_dir = Path(request.attempt_dir)
        archive_source = attempt_dir / "report_inputs" / "analysis.json.gz"
        archive_source.parent.mkdir(parents=True, exist_ok=True)
        archive_source.write_bytes(
            gzip.compress(
                canonical_json(_portable_analysis_value(analysis_payload)).encode(
                    "utf-8"
                ),
                compresslevel=9,
                mtime=0,
            )
        )
        archive_request = AssetRequest(
            "full-analysis-data",
            "copy",
            (PurePosixPath("data/analysis.json.gz"),),
            {"source_role": "full_analysis_data"},
        )
        project = replace(
            adapted.project,
            artifacts=adapted.project.artifacts
            + (
                Link(
                    "Full analysis data (JSON.gz)",
                    "data/analysis.json.gz",
                    "Complete, non-downsampled report input; host paths are redacted.",
                ),
            ),
        )
        document = self.build_document(
            project,
            contributors=adapted.contributors,
        )
        output_dir = attempt_dir / "output"
        asset_sources = dict(adapted.asset_sources or {})
        asset_sources[archive_request.request_id] = archive_source
        asset_requests = adapted.asset_requests + (archive_request,)
        materializer = _CopyAssetMaterializer(asset_sources)
        result = self.publish(
            document,
            output_dir,
            asset_requests=asset_requests,
            asset_materializer=materializer,
        )
        artifacts = _stage_artifacts(output_dir, result.pages, result.assets)
        _, result_type = _core_types()
        return result_type(
            artifacts=artifacts,
            payload={
                "schema_version": "mmpbsa.report.result.v1",
                "input_schema_version": adapted.schema_version,
                "project_id": adapted.project.project_id,
                "pages": [path.as_posix() for path in result.pages],
                "assets": [path.as_posix() for path in result.assets],
            },
            invocation_provenance={
                "provider_id": self.provider_id,
                "package_version": self.package_version,
                "renderer": "mmpbsa.report.html-svg.v2",
            },
        )

    def resume(self, request):
        """Reports are reproducible display trees; resume is a fresh execute."""

        return self.execute(request)


def get_provider() -> ReportProvider:
    """Entry-point factory used by core provider discovery."""

    return ReportProvider()


def _portable_analysis_value(value: object) -> object:
    """Retain analysis data while removing standalone and embedded host paths."""

    if isinstance(value, Mapping):
        portable: dict[str, object] = {}
        for key, item in value.items():
            portable_key = _portable_analysis_text(str(key))
            if portable_key in portable:
                raise ValueError(
                    "portable analysis mapping keys collide after redaction: "
                    f"{portable_key!r}"
                )
            portable[portable_key] = _portable_analysis_value(item)
        return portable
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_portable_analysis_value(item) for item in value]
    if isinstance(value, str):
        return _portable_analysis_text(value)
    return value


def _portable_analysis_text(value: str) -> str:
    posix = Path(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute():
        return f"host-local://{_portable_basename(value)}"
    without_quoted_paths = _QUOTED_HOST_PATH.sub(_redact_quoted_path, value)
    if _has_ambiguous_unquoted_path(without_quoted_paths):
        return "host-local://redacted"
    return _EMBEDDED_HOST_PATH.sub(_redact_embedded_path, without_quoted_paths)


def _has_ambiguous_unquoted_path(text: str) -> bool:
    """Whitespace after a bare path may still be part of its filename."""

    for match in _EMBEDDED_HOST_PATH.finditer(text):
        token = match.group(0)
        if (
            match.end() < len(text)
            and text[match.end()].isspace()
            and token[-1] not in ".,:;!?)]}"
        ):
            return True
    return False


def _portable_basename(path: str) -> str:
    windows = PureWindowsPath(path)
    name = windows.name if windows.is_absolute() else Path(path).name
    return name or "redacted"


def _redact_quoted_path(match: re.Match[str]) -> str:
    quote = match.group("quote")
    return f"{quote}host-local://{_portable_basename(match.group('path'))}{quote}"


def _redact_embedded_path(match: re.Match[str]) -> str:
    token = match.group(0)
    path = token.rstrip(".,:)]}")
    suffix = token[len(path) :]
    return f"host-local://{_portable_basename(path)}{suffix}"


def _stage_artifacts(
    output_dir: Path,
    pages,
    assets,
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    case_index = 0
    for page in pages:
        path = output_dir.joinpath(*page.parts)
        if page.as_posix() == "index.html":
            role = "report_index"
        else:
            case_index += 1
            case_id = page.parts[1]
            role = f"report_case_{case_id}"
        if role in artifacts:
            role = f"{role}_{case_index:04d}"
        artifacts[role] = path
    for index, asset in enumerate(assets, 1):
        artifacts[f"report_asset_{index:04d}"] = output_dir.joinpath(*asset.parts)
    return artifacts


def _core_types():
    """Import core lazily so report model/render imports remain standalone."""

    try:
        from mmpbsa.core import BackendCompatibility, StageResult
    except (ImportError, AttributeError):
        return _FallbackBackendCompatibility, _FallbackStageResult
    return BackendCompatibility, StageResult

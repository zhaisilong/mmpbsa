from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from mmpbsa.report import (
    AssetRequest,
    CaseRecord,
    Link,
    LinkAuditError,
    ProjectRecord,
    ReportPublisher,
    audit_local_links,
    audit_portable_manifests,
    build_document,
    collect_link_issues,
)
from mmpbsa.utils import AtomicReplaceError


def _document(*, artifact: Link | None = None):
    case = CaseRecord(
        case_id="case1",
        label="Case 1",
        expected_replica_ids=("rep01",),
        artifacts=(artifact,) if artifact else (),
    )
    return build_document(ProjectRecord("demo", "Demo", (case,)))


def test_local_link_audit_checks_missing_escape_fragment_and_active_content(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text(
        """
        <a href="missing.csv">missing</a>
        <a href="../outside.txt">escape</a>
        <a href="#absent">fragment</a>
        <script src="thing.js"></script>
        """,
        encoding="utf-8",
    )
    issues = collect_link_issues(tmp_path)
    reasons = {issue.reason for issue in issues}
    assert "target does not exist" in reasons
    assert "target escapes report root" in reasons
    assert "fragment ID does not exist" in reasons
    assert "active content is forbidden" in reasons
    with pytest.raises(LinkAuditError):
        audit_local_links(tmp_path)


def test_failed_preflight_preserves_existing_report(tmp_path: Path) -> None:
    destination = tmp_path / "report"
    destination.mkdir()
    (destination / "index.html").write_text("old report", encoding="utf-8")
    document = _document(artifact=Link("missing", "assets/missing.csv"))

    with pytest.raises(LinkAuditError):
        ReportPublisher().publish(document, destination)

    assert (destination / "index.html").read_text(encoding="utf-8") == "old report"
    assert not list(tmp_path.glob(".report.staging-*"))
    assert not list(tmp_path.glob(".report.backup-*"))


def test_atomic_install_failure_rolls_back_previous_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "report"
    destination.mkdir()
    (destination / "index.html").write_text("old report", encoding="utf-8")
    def fail_staging_install(source: Path, target: Path) -> None:
        assert source.name.startswith(".report.staging-")
        assert target == destination
        raise AtomicReplaceError("simulated install failure")

    monkeypatch.setattr(
        "mmpbsa.report.publish.atomic_replace_directory", fail_staging_install
    )
    with pytest.raises(AtomicReplaceError, match="simulated"):
        ReportPublisher().publish(_document(), destination)

    assert (destination / "index.html").read_text(encoding="utf-8") == "old report"
    assert not list(tmp_path.glob(".report.backup-*"))


def test_staging_creation_failure_releases_publication_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_staging(*args: object, **kwargs: object) -> None:
        raise OSError("simulated staging failure")

    monkeypatch.setattr("mmpbsa.report.publish.tempfile.mkdtemp", fail_staging)
    with pytest.raises(OSError, match="simulated staging failure"):
        ReportPublisher().publish(_document(), tmp_path / "report")

    assert not (tmp_path / ".report.publish.lock").exists()


class _Assets:
    def materialize(self, request: AssetRequest, staging_root: Path) -> None:
        for output in request.outputs:
            path = staging_root.joinpath(*output.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"asset:{request.request_id}", encoding="utf-8")


def test_assets_are_structural_and_audited_before_publish(tmp_path: Path) -> None:
    document = _document(artifact=Link("scores", "assets/scores.csv"))
    request = AssetRequest(
        "scores",
        "copy",
        (PurePosixPath("cases/case1/assets/scores.csv"),),
        {"source": "artifact:score-summary"},
    )
    result = ReportPublisher().publish(
        document,
        tmp_path / "report",
        asset_requests=(request,),
        asset_materializer=_Assets(),
    )

    assert result.assets == (PurePosixPath("cases/case1/assets/scores.csv"),)
    assert (
        tmp_path / "report" / "cases" / "case1" / "assets" / "scores.csv"
    ).read_text(encoding="utf-8") == "asset:scores"


def test_asset_materializer_cannot_create_undeclared_files(tmp_path: Path) -> None:
    class ExtraFile(_Assets):
        def materialize(self, request: AssetRequest, staging_root: Path) -> None:
            super().materialize(request, staging_root)
            (staging_root / "undeclared.txt").write_text("extra", encoding="utf-8")

    request = AssetRequest(
        "scores",
        "copy",
        (PurePosixPath("scores.csv"),),
    )
    with pytest.raises(ValueError, match="undeclared files"):
        ReportPublisher().publish(
            _document(),
            tmp_path / "report",
            asset_requests=(request,),
            asset_materializer=ExtraFile(),
        )


def test_portable_manifest_audit_requires_every_declared_asset(tmp_path: Path) -> None:
    bundle = tmp_path / "assets" / "case1"
    (bundle / "structures").mkdir(parents=True)
    (bundle / "trajectories").mkdir()
    (bundle / "structures" / "rep01.pdb").write_text("ATOM\n", encoding="utf-8")
    manifest = bundle / "display_assets.json"
    manifest.write_text(
        """{
          "schema_version": "mmpbsa.display_assets.v4",
          "topologies": {"rep01": "structures/rep01.pdb"},
          "trajectories": {"rep01": "trajectories/rep01.xtc"},
          "pymol_scripts": {}
        }\n""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing path"):
        audit_portable_manifests(tmp_path)

    (bundle / "trajectories" / "rep01.xtc").write_bytes(b"xtc")
    audit_portable_manifests(tmp_path)

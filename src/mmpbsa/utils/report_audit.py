"""Dependency-free audits for portable static report trees."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

_EXTERNAL_SCHEMES = frozenset({"http", "https", "mailto"})
_LINK_ATTRIBUTES = {
    "a": "href",
    "img": "src",
    "link": "href",
    "source": "src",
}


@dataclass(frozen=True, slots=True)
class LinkAuditIssue:
    page: PurePosixPath
    target: str
    reason: str


class LinkAuditError(ValueError):
    def __init__(self, issues: Sequence[LinkAuditIssue]) -> None:
        self.issues = tuple(issues)
        detail = "; ".join(
            f"{issue.page}: {issue.target!r} ({issue.reason})" for issue in self.issues
        )
        super().__init__(f"Report link audit failed: {detail}")


def collect_link_issues(root: Path | str) -> tuple[LinkAuditIssue, ...]:
    """Return all unsafe, escaping, missing, or fragment-invalid local links."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    issues: list[LinkAuditIssue] = []
    parsed_pages: dict[Path, _ParsedPage] = {}
    for candidate in sorted(root_path.rglob("*")):
        relative = PurePosixPath(candidate.relative_to(root_path).as_posix())
        if candidate.is_symlink():
            issues.append(
                LinkAuditIssue(relative, str(relative), "symbolic link is not portable")
            )
            continue
        if candidate.is_file() and candidate.suffix.lower() == ".html":
            parser = _PageParser()
            parser.feed(candidate.read_text(encoding="utf-8"))
            parser.close()
            parsed_pages[candidate.resolve()] = parser.result
            for tag in parser.result.forbidden_tags:
                issues.append(
                    LinkAuditIssue(relative, f"<{tag}>", "active content is forbidden")
                )

    for page_path, parsed in parsed_pages.items():
        page_relative = PurePosixPath(page_path.relative_to(root_path).as_posix())
        for target in parsed.links:
            _audit_target(
                root_path=root_path,
                page_path=page_path,
                page_relative=page_relative,
                target=target,
                parsed_pages=parsed_pages,
                issues=issues,
            )
    return tuple(issues)


def audit_local_links(root: Path | str) -> None:
    issues = collect_link_issues(root)
    if issues:
        raise LinkAuditError(issues)


def audit_portable_manifests(root: Path | str) -> None:
    """Require every path declared by a portable report manifest to exist."""

    root_path = Path(root).resolve()
    problems: list[str] = []
    for manifest in sorted(root_path.rglob("*.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            problems.append(f"{manifest.relative_to(root_path)}: invalid JSON ({error})")
            continue
        if not isinstance(payload, Mapping):
            continue
        schema = payload.get("schema_version")
        references: list[str] = []
        if schema == "mmpbsa.report_assets.v3":
            bundles = payload.get("bundles")
            if isinstance(bundles, Mapping):
                for entries in bundles.values():
                    if isinstance(entries, Mapping):
                        references.extend(
                            value for value in entries.values() if isinstance(value, str)
                        )
        elif schema == "mmpbsa.display_assets.v4":
            for key in ("topologies", "trajectories", "pymol_scripts"):
                entries = payload.get(key)
                if isinstance(entries, Mapping):
                    references.extend(
                        value for value in entries.values() if isinstance(value, str)
                    )
            diagnostic = payload.get("equilibrated_pose_rmsd")
            if isinstance(diagnostic, Mapping):
                entries = diagnostic.get("replicas")
                if isinstance(entries, Mapping):
                    references.extend(
                        value for value in entries.values() if isinstance(value, str)
                    )
        else:
            continue
        for reference in references:
            relative = PurePosixPath(reference)
            if relative.is_absolute() or ".." in relative.parts:
                problems.append(
                    f"{manifest.relative_to(root_path)}: unsafe path {reference!r}"
                )
                continue
            candidate = manifest.parent.joinpath(*relative.parts)
            if not candidate.is_file() or candidate.is_symlink():
                problems.append(
                    f"{manifest.relative_to(root_path)}: missing path {reference!r}"
                )
    if problems:
        raise ValueError("Portable manifest audit failed: " + "; ".join(problems))


def _audit_target(
    *,
    root_path: Path,
    page_path: Path,
    page_relative: PurePosixPath,
    target: str,
    parsed_pages: Mapping[Path, _ParsedPage],
    issues: list[LinkAuditIssue],
) -> None:
    decoded = unescape(target).strip()
    parsed = urlsplit(decoded)
    scheme = parsed.scheme.lower()
    if scheme:
        if scheme not in _EXTERNAL_SCHEMES:
            issues.append(LinkAuditIssue(page_relative, target, f"unsafe scheme {scheme}"))
        return
    if decoded.startswith("//"):
        issues.append(LinkAuditIssue(page_relative, target, "protocol-relative target"))
        return
    local_path = unquote(parsed.path)
    if local_path.startswith(("/", "\\")):
        issues.append(LinkAuditIssue(page_relative, target, "absolute local target"))
        return
    candidate = page_path if not local_path else page_path.parent / local_path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root_path):
        issues.append(LinkAuditIssue(page_relative, target, "target escapes report root"))
        return
    if resolved.is_dir():
        resolved = (resolved / "index.html").resolve()
    if not resolved.exists():
        issues.append(LinkAuditIssue(page_relative, target, "target does not exist"))
        return
    if parsed.fragment:
        fragment = unquote(parsed.fragment)
        target_page = parsed_pages.get(resolved)
        if target_page is None:
            issues.append(LinkAuditIssue(page_relative, target, "fragment target is not HTML"))
        elif fragment not in target_page.ids:
            issues.append(LinkAuditIssue(page_relative, target, "fragment ID does not exist"))


@dataclass(frozen=True, slots=True)
class _ParsedPage:
    links: tuple[str, ...]
    ids: frozenset[str]
    forbidden_tags: tuple[str, ...]


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._links: list[str] = []
        self._ids: set[str] = set()
        self._forbidden: list[str] = []

    @property
    def result(self) -> _ParsedPage:
        return _ParsedPage(tuple(self._links), frozenset(self._ids), tuple(self._forbidden))

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized = tag.lower()
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self._ids.add(element_id)
        link_attribute = _LINK_ATTRIBUTES.get(normalized)
        if link_attribute and attributes.get(link_attribute):
            self._links.append(str(attributes[link_attribute]))
        if normalized in {"script", "iframe", "object", "embed", "form"}:
            self._forbidden.append(normalized)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)

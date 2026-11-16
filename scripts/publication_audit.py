#!/usr/bin/env python3
"""Known-risk publication checks; hooks may be bypassed and do not replace review."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

TRUSTED_BASE = "d20d474cc6b8c2a2ba7e1b0e05fa33ffbdfe13e1"
MAX_FILE_BYTES = 10 * 1024 * 1024
_ZERO = "0" * 40
_OID = re.compile(r"^[0-9a-f]{40}$")
_PRIVATE_PATH = re.compile(r"/(?:home|Users|data2)/[A-Za-z0-9_.-]+/", re.I)
_SECRETS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(
        r"(?i)(?:password|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*[A-Za-z0-9_/-]{20,}"
    ),
    re.compile(
        r"(?i)(?:password|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*[\"\x27][^\"\x27\s]{16,}[\"\x27]"
    ),
)
_OUTPUT_DIRS = {"_runs", "_workspace", "_integration_build", "prepared"}
_OUTPUT_SUFFIXES = {
    ".xtc",
    ".trr",
    ".dcd",
    ".nc",
    ".edr",
    ".cpt",
    ".tpr",
    ".log",
    ".gz",
    ".zip",
    ".tar",
    ".bz2",
    ".xz",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".parquet",
    ".7z",
    ".zst",
}
_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".sdf", ".mol2", ".gro", ".prmtop", ".rst7"}
_HOOK_MARKER = "# mmpbsa-publication-guard-v1"
_HOOK_TEXT = (
    "#!/bin/sh\n"
    + _HOOK_MARKER
    + "\n"
    + "repo=$(git rev-parse --show-toplevel) || exit 1\n"
    + 'exec python3 "$repo/scripts/publication_audit.py" pre-push "$@"\n'
)


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def _ancestor(root: Path, parent: str, child: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", parent, child],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _tree(root: Path, commit: str) -> dict[str, tuple[str, str, str]]:
    result = {}
    for entry in _git(root, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if entry:
            meta, path = entry.split(b"\t", 1)
            mode, kind, oid = meta.decode("ascii").split(" ")
            result[os.fsdecode(path)] = (mode, kind, oid)
    return result


def _path_issue(path: str) -> str | None:
    lower = [part.lower() for part in path.split("/")]
    name = lower[-1]
    if any(part in _OUTPUT_DIRS or part.startswith("._integration_build.") for part in lower):
        return "runtime/output directory"
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "environment secret file"
    if re.fullmatch(r"runtime(?:\..*)?\.local\.ya?ml", name):
        return "local runtime configuration"
    if name in {"id_rsa", "id_ed25519", "id_ecdsa", ".netrc", "credentials"}:
        return "credential filename"
    if name in {
        "mdinfo",
        "analysis.json",
        "results.csv",
        "scores.csv",
        "ranking.csv",
        "final_results_mmpbsa.dat",
        "final_decomp_mmpbsa.dat",
    }:
        return "generated result filename"
    if Path(name).suffix in _OUTPUT_SUFFIXES:
        return "run output or archive"
    return None


def audit_push(
    root: Path, records: str, *, remote_name: str = "origin", trusted_base: str = TRUSTED_BASE
) -> list[str]:
    """Return violations for every outgoing commit and complete tree."""
    errors: list[str] = []
    if remote_name != "origin":
        return ["only the configured origin remote may be pushed"]
    updates = [line.split() for line in records.splitlines() if line.strip()]
    if not updates:
        return []
    if len(updates) != 1 or len(updates[0]) != 4:
        return ["push must update exactly one main ref"]
    local_ref, local_oid, remote_ref, remote_oid = updates[0]
    if local_ref != "refs/heads/main" or remote_ref != "refs/heads/main":
        return ["only refs/heads/main may be updated"]
    if not _OID.fullmatch(local_oid) or not _OID.fullmatch(remote_oid):
        return ["invalid object identifier in push request"]
    if local_oid == _ZERO or remote_oid == _ZERO:
        return ["creating or deleting a remote ref requires manual review"]
    try:
        if _git(root, "rev-parse", "refs/heads/main").decode().strip() != local_oid:
            return ["push does not match local main"]
        if not _ancestor(root, trusted_base, remote_oid) or not _ancestor(
            root, trusted_base, local_oid
        ):
            return ["push contains untrusted history or the public base is unavailable"]
        if not _ancestor(root, remote_oid, local_oid):
            return ["non-fast-forward push or missing remote object"]
        roots = _git(root, "rev-list", "--max-parents=0", local_oid).decode().splitlines()
        trusted_roots = (
            _git(root, "rev-list", "--max-parents=0", trusted_base).decode().splitlines()
        )
        if len(roots) != 1 or roots != trusted_roots:
            return ["push has additional unrelated history"]
        baseline = _tree(root, trusted_base)
        commits = _git(root, "rev-list", local_oid, "^" + remote_oid).decode().splitlines()
        examined: set[tuple[str, str]] = set()
        for commit in commits:
            message = _git(root, "cat-file", "commit", commit).decode("utf-8")
            if _PRIVATE_PATH.search(message) or any(p.search(message) for p in _SECRETS):
                errors.append(f"{commit[:12]}: sensitive commit metadata")
            for path, (mode, kind, oid) in _tree(root, commit).items():
                if _PRIVATE_PATH.search(path) or any(pattern.search(path) for pattern in _SECRETS):
                    errors.append(f"{commit[:12]}: <redacted>: sensitive filename")
                    continue
                safe_path = repr(path)
                issue = _path_issue(path)
                if issue:
                    errors.append(f"{commit[:12]}: {safe_path}: {issue}")
                    continue
                if kind != "blob" or mode not in {"100644", "100755"}:
                    errors.append(
                        f"{commit[:12]}: {safe_path}: symlink, submodule or unsupported mode"
                    )
                    continue
                entry = (path, oid)
                if entry in examined:
                    continue
                examined.add(entry)
                if baseline.get(path) == (mode, kind, oid):
                    continue
                if Path(path).suffix.lower() in _STRUCTURE_SUFFIXES:
                    errors.append(f"{commit[:12]}: {safe_path}: unreviewed structure file")
                    continue
                if int(_git(root, "cat-file", "-s", oid)) > MAX_FILE_BYTES:
                    errors.append(f"{commit[:12]}: {safe_path}: file exceeds audit size limit")
                    continue
                content = _git(root, "cat-file", "blob", oid)
                if b"\0" in content:
                    errors.append(f"{commit[:12]}: {safe_path}: unreviewed binary file")
                    continue
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    errors.append(f"{commit[:12]}: {safe_path}: unreviewed non-UTF-8 file")
                    continue
                if _PRIVATE_PATH.search(text):
                    errors.append(f"{commit[:12]}: {safe_path}: private absolute path")
                if any(pattern.search(text) for pattern in _SECRETS):
                    errors.append(f"{commit[:12]}: {safe_path}: possible credential")
    except (OSError, subprocess.CalledProcessError, ValueError, UnicodeError):
        return ["Git audit failed or required objects are unavailable"]
    return errors


def install_hook(root: Path) -> None:
    """Install without overwriting an existing hook or custom hook location."""
    custom = subprocess.run(
        ["git", "-C", str(root), "config", "--get", "core.hooksPath"],
        text=True,
        capture_output=True,
    )
    if custom.returncode == 0:
        raise ValueError("custom core.hooksPath is configured; install manually")
    if custom.returncode != 1:
        raise ValueError("cannot inspect Git hooks configuration")
    hook_path = Path(os.fsdecode(_git(root, "rev-parse", "--git-path", "hooks/pre-push").strip()))
    if not hook_path.is_absolute():
        hook_path = root / hook_path
    if hook_path.is_symlink():
        raise ValueError("existing pre-push hook must not be overwritten")
    if hook_path.exists():
        if hook_path.read_text(encoding="utf-8") != _HOOK_TEXT:
            raise ValueError("existing pre-push hook must not be overwritten")
        hook_path.chmod(0o755)
        return
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    with hook_path.open("x", encoding="utf-8") as stream:
        stream.write(_HOOK_TEXT)
    hook_path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    hook = sub.add_parser("pre-push")
    hook.add_argument("remote")
    hook.add_argument("url")
    sub.add_parser("install-hook")
    sub.add_parser("check")
    args = parser.parse_args()
    try:
        root = Path(os.fsdecode(_git(Path.cwd(), "rev-parse", "--show-toplevel").strip()))
        if args.command == "install-hook":
            install_hook(root)
            print("Installed local pre-push guard; hooks can be bypassed.")
            return 0
        if args.command == "pre-push":
            errors = audit_push(root, sys.stdin.read(), remote_name=args.remote)
        else:
            local = _git(root, "rev-parse", "refs/heads/main").decode().strip()
            remote = _git(root, "rev-parse", "refs/remotes/origin/main").decode().strip()
            errors = audit_push(root, f"refs/heads/main {local} refs/heads/main {remote}\n")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"publication audit: {exc}", file=sys.stderr)
        return 1
    for error in errors:
        print(f"publication audit: {error}", file=sys.stderr)
    if errors:
        return 1
    print("Publication audit passed for known patterns; manual review remains necessary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

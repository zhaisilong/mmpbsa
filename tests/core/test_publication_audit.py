"""Publication checks inspect every outgoing tree, including intermediate commits."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts.publication_audit import TRUSTED_BASE, audit_push, install_hook


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _write(root: Path, name: str, value: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "remote", "add", "origin", "https://example.invalid/synthetic.git")
    _write(root, "README.md", "Synthetic example.\n")
    _write(root, "validations/example/data/receptor.pdb", "HEADER synthetic\n")
    base = _commit(root, "create synthetic baseline")
    return root, base


def _record(local: str, remote: str, *, local_ref: str = "refs/heads/main",
            remote_ref: str = "refs/heads/main") -> str:
    return f"{local_ref} {local} {remote_ref} {remote}\n"


def _check(root: Path, base: str, local: str, remote: str, **refs: str) -> list[str]:
    return audit_push(root, _record(local, remote, **refs), trusted_base=base)


def test_clean_fast_forward_allows_unchanged_public_structure(repository):
    root, base = repository
    _write(root, "README.md", "Public framework documentation.\n")
    head = _commit(root, "update docs")
    assert _check(root, base, head, base) == []


@pytest.mark.parametrize(
    "local_ref,remote_ref,deleted",
    [
        ("refs/heads/main", "refs/heads/other", False),
        ("refs/heads/other", "refs/heads/main", False),
        ("refs/tags/example", "refs/tags/example", False),
        ("(delete)", "refs/heads/main", True),
    ],
)
def test_non_main_and_deletion_are_rejected(repository, local_ref, remote_ref, deleted):
    root, base = repository
    _write(root, "README.md", "A new change.\n")
    head = _commit(root, "change")
    assert _check(
        root, base, "0" * 40 if deleted else head, base,
        local_ref=local_ref, remote_ref=remote_ref,
    )


def test_unknown_remote_non_origin_and_non_fast_forward_are_rejected(repository):
    root, base = repository
    _write(root, "README.md", "New public change.\n")
    head = _commit(root, "change")
    assert _check(root, base, head, "0" * 40)
    assert _check(root, base, head, "f" * 40)
    assert _check(root, base, base, head)
    assert audit_push(root, _record(head, base), remote_name="other", trusted_base=base)


def test_local_ref_must_match_main_head(repository):
    root, base = repository
    _write(root, "README.md", "A newer public change.\n")
    _commit(root, "change")
    assert _check(root, base, base, base)


def test_missing_trusted_baseline_and_malformed_updates_fail_closed(repository):
    root, base = repository
    _write(root, "README.md", "Public change.\n")
    head = _commit(root, "change")
    assert audit_push(root, _record(head, base), trusted_base="f" * 40)
    assert audit_push(root, "malformed input\n", trusted_base=base)


def test_unrelated_history_is_rejected_even_when_merged(repository):
    root, base = repository
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    unrelated = _git(root, "commit-tree", tree, "-m", "unrelated synthetic root")
    _git(root, "merge", "--allow-unrelated-histories", "--no-edit", unrelated)
    head = _git(root, "rev-parse", "HEAD")
    assert _check(root, base, head, base)


def test_intermediate_commit_with_result_cannot_be_hidden_by_later_deletion(repository):
    root, base = repository
    _write(root, "validations/example/_runs/run1/result.txt", "synthetic data\n")
    _commit(root, "add result")
    _git(root, "rm", "-q", "validations/example/_runs/run1/result.txt")
    head = _commit(root, "remove result")
    assert _check(root, base, head, base)


def test_same_blob_under_an_allowed_and_forbidden_path_is_checked_per_path(repository):
    root, base = repository
    _write(root, "notes.txt", "synthetic data\n")
    _write(root, "validations/example/_workspace/data.txt", "synthetic data\n")
    head = _commit(root, "add duplicate blob")
    assert _check(root, base, head, base)


@pytest.mark.parametrize(
    "path",
    [
        "validations/example/runtime.local.yaml",
        "validations/example/runtime.smoke.local.yaml",
        "run/trajectory.xtc",
        "run/checkpoint.cpt",
        "run/output.log",
        "run/archive.zip",
        "run/mdinfo",
        "run/FINAL_RESULTS_MMPBSA.dat",
        "run/FINAL_DECOMP_MMPBSA.dat",
        "validations/example/prepared/state.json",
        "validations/example/_integration_build/output.json",
    ],
)
def test_runtime_or_generated_artifacts_are_rejected(repository, path):
    root, base = repository
    _write(root, path, "synthetic data\n")
    head = _commit(root, "add unsafe artifact")
    assert _check(root, base, head, base)


def test_public_structure_requires_approved_path_and_exact_baseline_content(repository):
    root, base = repository
    _write(root, "validations/example/data/receptor.pdb", "HEADER altered synthetic\n")
    head = _commit(root, "change structure")
    assert _check(root, base, head, base)


def test_structure_cannot_be_added_elsewhere_using_approved_blob(repository):
    root, base = repository
    _write(root, "validations/example/data/other.pdb", "HEADER synthetic\n")
    head = _commit(root, "add structure elsewhere")
    assert _check(root, base, head, base)


def test_secret_is_rejected_without_echoing_its_value(repository):
    root, base = repository
    token = "gh" + "p_" + "a" * 36
    _write(root, "example.txt", f"token={token}\n")
    head = _commit(root, "add synthetic credential")
    errors = _check(root, base, head, base)
    assert errors
    assert token not in "\n".join(errors)


def test_commit_message_secret_is_rejected_without_echo(repository):
    root, base = repository
    token = "gh" + "p_" + "b" * 36
    _write(root, "README.md", "Synthetic change.\n")
    head = _commit(root, f"synthetic credential {token}")
    errors = _check(root, base, head, base)
    assert errors
    assert token not in "\n".join(errors)


def test_secret_in_filename_is_rejected_without_echo(repository):
    root, base = repository
    token = "gh" + "p_" + "c" * 36
    _write(root, f"example_{token}.txt", "synthetic data\n")
    head = _commit(root, "add sensitive filename")
    errors = _check(root, base, head, base)
    assert errors
    assert token not in "\n".join(errors)


def test_unquoted_yaml_credential_is_rejected_without_echo(repository):
    root, base = repository
    token = "example" + "z" * 24
    _write(root, "settings.yaml", f"access_token: {token}\n")
    head = _commit(root, "add configuration")
    errors = _check(root, base, head, base)
    assert errors
    assert token not in "\n".join(errors)


def test_renamed_binary_content_is_rejected(repository):
    root, base = repository
    _write(root, "hidden.txt", "synthetic\x00binary\n")
    head = _commit(root, "add hidden binary")
    assert _check(root, base, head, base)


def test_large_text_with_innocent_extension_is_rejected(repository, monkeypatch):
    root, base = repository
    monkeypatch.setitem(audit_push.__globals__, "MAX_FILE_BYTES", 128)
    _write(root, "notes.txt", "x" * 129)
    head = _commit(root, "add oversized file")
    assert _check(root, base, head, base)


def test_symlink_is_not_publishable(repository):
    root, base = repository
    os.symlink("README.md", root / "link")
    head = _commit(root, "add symbolic link")
    assert _check(root, base, head, base)


def test_submodule_gitlink_is_not_publishable(repository):
    root, base = repository
    _git(root, "update-index", "--add", "--cacheinfo", "160000", base, "vendor")
    _git(root, "commit", "-q", "-m", "add synthetic gitlink")
    head = _git(root, "rev-parse", "HEAD")
    assert _check(root, base, head, base)


def test_multiple_ref_updates_are_all_validated(repository):
    root, base = repository
    _write(root, "README.md", "A public change.\n")
    head = _commit(root, "change")
    records = _record(head, base) + _record(head, base, remote_ref="refs/heads/other")
    assert audit_push(root, records, trusted_base=base)


def test_install_hook_does_not_overwrite_existing_hook(repository):
    root, _ = repository
    hook = root / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    with pytest.raises(ValueError):
        install_hook(root)
    assert hook.read_text(encoding="utf-8") == "#!/bin/sh\nexit 7\n"


def test_install_hook_does_not_overwrite_custom_content_with_marker(repository):
    root, _ = repository
    hook = root / ".git" / "hooks" / "pre-push"
    custom = "#!/bin/sh\n# mmpbsa-publication-guard-v1\nexit 7\n"
    hook.write_text(custom, encoding="utf-8")
    with pytest.raises(ValueError):
        install_hook(root)
    assert hook.read_text(encoding="utf-8") == custom


def test_install_hook_rejects_nondefault_hooks_path(repository):
    root, _ = repository
    _git(root, "config", "core.hooksPath", ".custom-hooks")
    with pytest.raises(ValueError):
        install_hook(root)
    assert not (root / ".git" / "hooks" / "pre-push").exists()


def test_installed_hook_guards_real_push_to_local_bare_repository(repository, tmp_path):
    root, base = repository
    bare = tmp_path / "local bare.git"
    _git(tmp_path, "init", "--bare", "-q", str(bare))
    _git(root, "remote", "set-url", "origin", str(bare))
    _git(root, "push", "-q", "origin", "refs/heads/main:refs/heads/main")

    source = Path(__file__).resolve().parents[2] / "scripts" / "publication_audit.py"
    _write(root, "scripts/publication_audit.py", source.read_text().replace(TRUSTED_BASE, base))
    _write(root, "README.md", "Synthetic public change.\n")
    safe_head = _commit(root, "add publication guard")
    install_hook(root)
    _git(root, "push", "-q", "origin", "refs/heads/main:refs/heads/main")
    assert _git(bare, "rev-parse", "refs/heads/main") == safe_head

    _write(root, "example/_runs/result.txt", "synthetic data\n")
    _commit(root, "add prohibited output")
    result = subprocess.run(
        ["git", "push", "origin", "refs/heads/main:refs/heads/main"],
        cwd=root, check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert _git(bare, "rev-parse", "refs/heads/main") == safe_head

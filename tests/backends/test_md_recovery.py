import pytest

from mmpbsa.backends.errors import BackendConfigurationError
from mmpbsa.backends.md_recovery import clone_native_output


def test_failed_native_output_is_copied_and_old_history_is_preserved(tmp_path):
    source = tmp_path / "attempts/0001"
    target = tmp_path / "attempts/0002"
    (source / "output/production").mkdir(parents=True)
    target.mkdir()
    (source / "state.json").write_text('{"state":"failed"}')
    checkpoint = source / "output/production/production.cpt"
    checkpoint.write_bytes(b"native checkpoint bytes")
    audit = clone_native_output(source, target)
    assert (target / "output/production/production.cpt").read_bytes() == checkpoint.read_bytes()
    assert (source / "state.json").read_text() == '{"state":"failed"}'
    assert audit["source_attempt_id"] == "0001"
    assert "production/production.cpt" in audit["copied_artifact_sha256"]


def test_recovery_cannot_overwrite_or_escape_an_attempt(tmp_path):
    source = tmp_path / "attempts/0001"
    (source / "output").mkdir(parents=True)
    (source / "output/state.cpt").write_bytes(b"checkpoint")
    with pytest.raises(BackendConfigurationError, match="fresh sibling"):
        clone_native_output(source, tmp_path / "other/0002")
    target = tmp_path / "attempts/0002"
    (target / "output").mkdir(parents=True)
    (target / "output/existing").write_bytes(b"do not overwrite")
    with pytest.raises(BackendConfigurationError, match="not empty"):
        clone_native_output(source, target)
    assert (target / "output/existing").read_bytes() == b"do not overwrite"


def test_recovery_does_not_follow_symlinks(tmp_path):
    source = tmp_path / "attempts/0001"
    target = tmp_path / "attempts/0002"
    (source / "output").mkdir(parents=True)
    target.mkdir()
    (source / "output/checkpoint").symlink_to(tmp_path / "private")
    with pytest.raises(BackendConfigurationError, match="empty or contains symlinks"):
        clone_native_output(source, target)

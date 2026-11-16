from __future__ import annotations

import builtins
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.common.preflight import preflight_case  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_nineteen_preflights_reach_their_expected_outcome() -> None:
    catalog = load(ROOT / "catalog" / "cases.json")
    observed = {}
    for case in catalog["cases"]:
        result = preflight_case(ROOT / case["model_file"])
        assert result["errors"] == []
        assert result["preflight_status"] == case["expected_outcome"]
        assert result["source_evidence"]["status"] == "derived"
        assert result["source_evidence"]["schema_version"] == "sp2016.source_evidence.v1"
        for check in result["checks"]:
            if check["status"] == "pass":
                assert "unsupported eligibility" not in check["detail"]
                assert "unsafe source path" not in check["detail"]
        observed[result["preflight_status"]] = observed.get(result["preflight_status"], 0) + 1
    assert observed == {"eligible_preflight_passed": 9, "rejected_expected": 10}


def test_every_case_prepare_script_is_executable_as_a_python_entrypoint() -> None:
    catalog = load(ROOT / "catalog" / "cases.json")
    for case in catalog["cases"]:
        completed = subprocess.run(
            [sys.executable, str(ROOT / case["prepare_file"]), "--json"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["case_id"] == case["case_id"]
        assert payload["preflight_status"] == case["expected_outcome"]


def test_prepare_all_json_emits_one_complete_structured_summary() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "prepare_all.py"), "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["summary"] == {
        "eligible_catalog": 9,
        "preflight_failed": 0,
        "production_topologies_present": 0,
        "rejected_catalog": 10,
        "selected": 19,
        "status_counts": {
            "eligible_preflight_passed": 9,
            "rejected_expected": 10,
        },
    }
    assert len(payload["results"]) == 19


def test_prepare_all_output_writes_json_but_prints_only_one_line(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "prepare_all.py"), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        f"sp2016_3x5ns: selected=19 eligible=9 rejected=10 failed=0 output={output}\n"
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 19
    assert payload["summary"]["preflight_failed"] == 0


def test_source_hash_tampering_fails_closed(tmp_path: Path) -> None:
    model = ROOT / "modeling" / "cases" / "1D4T" / "model.yaml"
    source_root = tmp_path / "mmcif"
    source_root.mkdir()
    copied = source_root / "1D4T.cif"
    shutil.copyfile(ROOT / "sources" / "mmcif" / "1D4T.cif", copied)
    copied.write_bytes(copied.read_bytes() + b"\n# tampered\n")
    result = preflight_case(model, source_root=source_root)
    assert result["preflight_status"] == "failed"
    assert any(error.startswith("source_size:") for error in result["errors"])
    assert any(error.startswith("source_sha256:") for error in result["errors"])


@pytest.mark.parametrize("path_form", ["absolute", "traversal"])
def test_unsafe_source_path_is_rejected_before_file_access(
    path_form: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = ROOT / "modeling" / "cases" / "2HPL" / "model.yaml"
    model = copy.deepcopy(load(model_path))
    official = ROOT / "sources" / "mmcif" / "2HPL.cif"
    model["source"]["path"] = (
        str(official)
        if path_form == "absolute"
        else str(Path("..") / ROOT.name / "sources" / "mmcif" / "2HPL.cif")
    )
    monkeypatch.setattr(
        "modeling.common.preflight._sha256",
        lambda path: pytest.fail(f"unsafe source was read: {path}"),
    )

    result = preflight_case(model_path, model_override=model)

    assert result["preflight_status"] == "failed"
    assert result["source_evidence"]["status"] == "not_derived"
    assert any(error.startswith("source_path_is_relative:") for error in result["errors"])
    assert not any(check["name"] == "source_sha256" for check in result["checks"])


def test_symlinked_source_cannot_escape_source_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = ROOT / "modeling" / "cases" / "2HPL" / "model.yaml"
    source_root = tmp_path / "mmcif"
    source_root.mkdir()
    (source_root / "2HPL.cif").symlink_to(ROOT / "sources" / "mmcif" / "2HPL.cif")
    monkeypatch.setattr(
        "modeling.common.preflight._sha256",
        lambda path: pytest.fail(f"outside-root source was read: {path}"),
    )

    result = preflight_case(model_path, source_root=source_root)

    assert result["preflight_status"] == "failed"
    assert result["source_evidence"]["status"] == "not_derived"
    assert any(error.startswith("source_path_within_root:") for error in result["errors"])
    assert not any(check["name"] == "source_sha256" for check in result["checks"])


def test_require_eligible_distinguishes_expected_scientific_rejection() -> None:
    rejected = ROOT / "modeling" / "cases" / "1CKA" / "prepare.py"
    completed = subprocess.run(
        [sys.executable, str(rejected), "--require-eligible", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    assert json.loads(completed.stdout)["preflight_status"] == "rejected_expected"


def test_source_evidence_preserves_insertions_heterogens_altlocs_connections_and_assembly() -> None:
    insertion = preflight_case(ROOT / "modeling" / "cases" / "1PZ5" / "model.yaml")
    receptor_segments = insertion["source_evidence"]["polymer_segments"]["receptor"]
    assert receptor_segments[0]["derived_observed_residue_selector"] == (
        "1-27,27A,27B,27C,27D,27E,28-95,97-211"
    )
    assert len(insertion["source_evidence"]["struct_conn"]["all_source_records"]) == 4

    modified = preflight_case(ROOT / "modeling" / "cases" / "2CCH" / "model.yaml")
    evidence = modified["source_evidence"]
    assert [item["component_id"] for item in evidence["selected_chain_non_water_heterogens"]] == [
        "ATP",
        "SO4",
        "GOL",
        "GOL",
    ]
    assert evidence["selected_chain_nonstandard_polymer_residues"] == [
        {
            "atom_count": 11,
            "auth_chain": "A",
            "auth_residue_id": "A:160",
            "auth_seq_id": 160,
            "component_id": "TPO",
            "entity_type": "polymer",
            "gemmi_is_standard": False,
            "gemmi_one_letter_code": "t",
            "gemmi_residue_info_found": True,
            "insertion_code": None,
            "label_asym_id": "A",
        }
    ]
    assert any(
        item["component_id"] == "ATP" and item["auth_residue_id"] == "A:1297"
        for item in evidence["selected_chain_altloc_residues"]
    )
    assert evidence["struct_conn"]["selected_connection_ids"] == ["covale1", "covale2"]

    assembly = preflight_case(ROOT / "modeling" / "cases" / "2HPL" / "model.yaml")
    assembly_evidence = assembly["source_evidence"]["biological_assemblies"]
    assert assembly_evidence[0]["assembly_id"] == "1"
    assert assembly_evidence[0]["generators"][0]["operators"][0]["is_identity"] is True

    two_ak5 = preflight_case(ROOT / "modeling" / "cases" / "2AK5" / "model.yaml")
    assert (
        next(
            check
            for check in two_ak5["checks"]
            if check["name"] == "source_biological_assembly_declaration"
        )["status"]
        == "pass"
    )
    operators = two_ak5["source_evidence"]["biological_assemblies"][0]["generators"][0]["operators"]
    assert [(item["operator_id"], item["is_identity"]) for item in operators] == [
        ("1", True),
        ("2", False),
    ]

    rhodamine = preflight_case(ROOT / "modeling" / "cases" / "3D1E" / "model.yaml")
    ligand_entity = rhodamine["source_evidence"]["selected_polymer_entities"]["ligand"][0]
    assert ligand_entity["entity_id"] == "2"
    assert ligand_entity["pdbx_description"] == "decamer from polymerase II C-terminal"
    assert "Rhodamine 6G" in ligand_entity["details"]
    assert "TLMT" in ligand_entity["details"]


def test_mutated_observed_residue_selector_fails_against_pinned_source() -> None:
    path = ROOT / "modeling" / "cases" / "2HPL" / "model.yaml"
    model = copy.deepcopy(load(path))
    model["entities"]["receptor"]["segments"][0]["observed_residues"] = "12-110"
    result = preflight_case(path, model_override=model)
    assert result["preflight_status"] == "failed"
    assert any(
        error.startswith("source_receptor_segment_1_observed_selector:")
        for error in result["errors"]
    )


@pytest.mark.parametrize(
    ("field", "wrong_value", "expected_check"),
    [
        ("declared_sequence", "ADLYG", "source_ligand_declared_sequence:"),
        ("observed_sequence", "DDLYA", "source_ligand_observed_sequence:"),
    ],
)
def test_mutated_ligand_sequence_fails_against_pinned_source(
    field: str, wrong_value: str, expected_check: str
) -> None:
    path = ROOT / "modeling" / "cases" / "2HPL" / "model.yaml"
    model = copy.deepcopy(load(path))
    model["entities"]["ligand"][field] = wrong_value
    result = preflight_case(path, model_override=model)
    assert result["preflight_status"] == "failed"
    assert any(error.startswith(expected_check) for error in result["errors"])


def test_missing_gemmi_fails_with_actionable_preparation_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def without_gemmi(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gemmi":
            raise ModuleNotFoundError("simulated missing gemmi")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_gemmi)
    result = preflight_case(ROOT / "modeling" / "cases" / "2HPL" / "model.yaml")
    assert result["preflight_status"] == "failed"
    assert result["source_evidence"]["status"] == "failed"
    assert ".[preparation]" in result["source_evidence"]["error"]
    assert any(error.startswith("source_evidence_derivation:") for error in result["errors"])

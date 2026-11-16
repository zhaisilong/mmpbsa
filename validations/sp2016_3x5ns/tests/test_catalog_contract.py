from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ELIGIBLE = {
    "1D4T",
    "1MFG",
    "1PZ5",
    "1T4F",
    "1X2R",
    "2AK5",
    "2HO2",
    "2HPL",
    "2O9V",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_catalog_has_fixed_nine_ten_partition() -> None:
    catalog = load(ROOT / "catalog" / "cases.json")
    cases = catalog["cases"]
    assert len(cases) == 19
    assert len({case["case_id"] for case in cases}) == 19
    assert len({case["pdb_id"] for case in cases}) == 19
    assert {case["pdb_id"] for case in cases if case["status"] == "eligible"} == ELIGIBLE
    assert sum(case["status"] == "eligible" for case in cases) == 9
    assert sum(case["status"] == "rejected" for case in cases) == 10
    assert catalog["eligibility_contract"]["eligible_count"] == 9
    assert catalog["eligibility_contract"]["rejected_count"] == 10


def test_every_case_has_explicit_model_wrapper_readme_and_hash_bound_source() -> None:
    catalog = load(ROOT / "catalog" / "cases.json")
    outcomes = load(ROOT / "catalog" / "expected_outcomes.json")["cases"]
    for case in catalog["cases"]:
        model_path = ROOT / case["model_file"]
        prepare_path = ROOT / case["prepare_file"]
        readme_path = model_path.with_name("README.md")
        assert model_path.is_file()
        assert prepare_path.is_file()
        assert readme_path.is_file()

        model = load(model_path)
        assert model["case_id"] == case["case_id"]
        assert model["pdb_id"] == case["pdb_id"]
        assert model["eligibility"]["status"] == case["status"]
        assert model["eligibility"]["reason_codes"] == case["reason_codes"]
        assert model["entities"]["ligand"]["kind"] == "peptide"
        assert model["entities"]["ligand"]["qc_rmsd_selection"] == "backbone"
        assert model["structure_decisions"]["include"]
        assert model["structure_decisions"]["explicit_exclude"]
        assert model["build"]["production_topology_present"] is False
        expected_build = outcomes[case["case_id"]]["build_status"]
        assert model["build"]["status"] == expected_build

        source = ROOT / model["source"]["path"]
        assert source.is_file()
        assert source.stat().st_size == model["source"]["bytes"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == model["source"]["sha256"]


def test_rejections_have_specific_stable_reasons() -> None:
    catalog = load(ROOT / "catalog" / "cases.json")
    rejected = [case for case in catalog["cases"] if case["status"] == "rejected"]
    assert len(rejected) == 10
    for case in rejected:
        assert case["reason_codes"]
        assert all(code == code.upper() and " " not in code for code in case["reason_codes"])
        model = load(ROOT / case["model_file"])
        assert len(model["eligibility"]["reasons"]) == len(case["reason_codes"])
        assert all(len(reason) >= 20 for reason in model["eligibility"]["reasons"])


def test_machine_readable_outcomes_provenance_and_licenses() -> None:
    outcomes = load(ROOT / "catalog" / "expected_outcomes.json")
    provenance = load(ROOT / "catalog" / "provenance.json")
    licenses = load(ROOT / "catalog" / "licenses.json")
    assert len(outcomes["cases"]) == 19
    assert outcomes["policy"]["no_production_claim"] is True
    assert provenance["coordinate_sources"]["license_id"] == "CC0-1.0"
    assert provenance["excluded_legacy_material"]["copied_trajectories"] is False
    assert {item["spdx_id"] for item in licenses["datasets"]} == {"CC0-1.0", "CC-BY-4.0"}


def test_validation_contains_no_production_or_legacy_simulation_artifacts() -> None:
    forbidden_suffixes = {".nc", ".xtc", ".trr", ".prmtop", ".top", ".rst7", ".mdcrd"}
    artifacts = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in forbidden_suffixes
        and not any(
            part == "_integration_build"
            or part.startswith("._integration_build.failed.")
            or part.startswith(".build-proof-")
            for part in path.relative_to(ROOT).parts
        )
    ]
    assert artifacts == []


def test_only_2hpl_declares_a_validation_local_build_proof() -> None:
    outcomes = load(ROOT / "catalog" / "expected_outcomes.json")["cases"]
    implemented = {
        item["pdb_id"]
        for item in outcomes.values()
        if item["build_status"] == "validation_local_proof_implemented"
    }
    assert implemented == {"2HPL"}

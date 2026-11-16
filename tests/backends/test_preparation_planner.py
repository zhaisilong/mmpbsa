from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmpbsa.backends.prep import (
    PreparationError,
    build_preparation_plan,
    inspect_sources,
    load_preparation_runtime,
    load_preparation_spec,
    write_receptor_views,
)
from mmpbsa.backends.prep.builder import _map_scoring_atoms, _partition

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "validations/tyk2_3x5ns/preparation.yaml"


def test_tyk2_sources_resolve_explicit_entities_and_minimal_qc() -> None:
    spec = load_preparation_spec(CONFIG)
    inspection = inspect_sources(spec)

    assert spec.ligand.entity_id == "ligand"
    assert spec.ligand.formal_charge == 0
    assert tuple(item.entity_id for item in spec.crystal_waters) == (
        "crystal_water_A_1318",
        "crystal_water_A_1325",
    )
    assert len(inspection.interface_residue_keys) == 19
    assert inspection.interface_chemistry_classes == (
        "charged",
        "nonpolar",
        "polar",
    )


def test_receptor_views_keep_crystal_water_only_in_md(tmp_path: Path) -> None:
    spec = load_preparation_spec(CONFIG)
    md_path = tmp_path / "receptor_md.pdb"
    dry_path = tmp_path / "receptor_dry.pdb"

    receipt = write_receptor_views(spec, md_path=md_path, dry_path=dry_path)

    assert receipt["md_crystal_water_count"] == 2
    assert "HOH A1318" in md_path.read_text(encoding="utf-8")
    assert "HOH A1325" in md_path.read_text(encoding="utf-8")
    assert " HOH " not in dry_path.read_text(encoding="utf-8")
    assert "HID" in md_path.read_text(encoding="utf-8")
    assert "HIE" in md_path.read_text(encoding="utf-8")


def test_dry_run_is_non_writing_and_names_ligand_topology(tmp_path: Path) -> None:
    spec = load_preparation_spec(CONFIG)
    destination = tmp_path / "prepared"

    plan = build_preparation_plan(spec)

    assert plan["dry_run"] is True
    assert plan["writes_destination"] is False
    assert not destination.exists()
    assert plan["systems"]["md"]["crystal_waters_retained"] is True
    assert plan["systems"]["scoring"]["all_water_excluded"] is True
    assert plan["systems"]["scoring"]["topology_roles"] == [
        "complex_topology",
        "receptor_topology",
        "ligand_topology",
    ]
    assert plan["entities"]["ligand"]["qc_selection"] == "heavy_atoms"
    assert plan["deferred_two_phase"] == {
        "step": "assemble_salted_md",
        "state": "deferred_until_noions_inventory",
        "reason": (
            "Na+/Cl- counts require the actual noions water count and integral "
            "solute charge"
        ),
        "target_salt_concentration_molar": 0.15,
        "salt_pair_formula": "round(N_water * target_molar / 55.5)",
        "neutralization_policy": (
            "add counterions for integral solute charge, then add equal Na+/Cl- "
            "salt pairs"
        ),
        "resolved_counts_recorded_by": "materialized preparation receipt",
    }
    assert "assemble_salted_md" not in {
        command["step"] for command in plan["commands"]
    }
    initial = plan["generated_tleap_inputs"]["initial"]
    assert "DRY_REC" in initial
    assert "MD_REC" in initial
    assert "DRY = combine { DRY_REC LIG }" in initial
    assert "NOIONS = combine { MD_REC LIG }" in initial

    second = build_preparation_plan(spec)
    assert second["plan_digest"] == plan["plan_digest"]


def test_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    shutil.copytree(CONFIG.parent / "data", tmp_path / "data")
    document = CONFIG.read_text(encoding="utf-8").replace(
        "9090684f4bdae90afbe5f2698a14c778396c024c19ceb6333de4808d9e29fae6",
        "a" * 64,
    )
    config = tmp_path / "preparation.yaml"
    config.write_text(document, encoding="utf-8")

    with pytest.raises(PreparationError, match="receptor source digest differs"):
        inspect_sources(load_preparation_spec(config))


def _atom(residue: str, name: str, atomic_number: int) -> SimpleNamespace:
    return SimpleNamespace(
        residue=SimpleNamespace(name=residue),
        name=name,
        atomic_number=atomic_number,
    )


def test_scoring_projection_is_ordered_unique_and_resname_independent() -> None:
    scoring = SimpleNamespace(
        atoms=(
            _atom("AAA", "N", 7),
            _atom("AAA", "CA", 6),
            _atom("not_lig", "Q1", 6),
        )
    )
    md = SimpleNamespace(
        atoms=(
            _atom("AAA", "N", 7),
            _atom("AAA", "CA", 6),
            _atom("water", "O", 8),
            _atom("not_lig", "Q1", 6),
            _atom("ion", "NA", 11),
        )
    )

    receptor, ligand = _partition(
        scoring, receptor_atom_count=2, ligand_atom_count=1
    )

    assert receptor == (1, 2)
    assert ligand == (3,)
    assert _map_scoring_atoms(scoring, md) == (1, 2, 4)


def test_scoring_projection_rejects_an_ambiguous_embedding() -> None:
    scoring = SimpleNamespace(atoms=(_atom("wat", "O", 8),))
    md = SimpleNamespace(
        atoms=(_atom("wat", "O", 8), _atom("wat", "O", 8))
    )

    with pytest.raises(PreparationError, match="ambiguous ordered projection"):
        _map_scoring_atoms(scoring, md)


def test_preparation_runtime_reads_its_section_from_complete_runtime(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        """\
schema_version: mmpbsa.runtime.v1
providers: {md: backends}
scheduler: {max_workers: 1, threads_per_job: 1, cpu_slots: 1, gpu_ids: []}
backends:
  backends:
    preparation:
      antechamber: {executable: antechamber}
      parmchk2: {executable: parmchk2}
      tleap: {executable: tleap}
environment: {AMBERHOME: /opt/amber24}
""",
        encoding="utf-8",
    )

    resolved = load_preparation_runtime(runtime)

    assert resolved["antechamber"]["environment"] == {
        "AMBERHOME": "/opt/amber24"
    }

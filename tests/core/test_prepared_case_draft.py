from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mmpbsa.core import cli
from mmpbsa.core.errors import ArtifactIntegrityError, ContractError
from mmpbsa.core.prepared_draft import (
    inspect_prepared_case_draft,
    load_prepared_case_draft,
    seal_prepared_case_draft,
)
from mmpbsa.utils import InterProcessFileLock


@dataclass
class _Residue:
    idx: int
    name: str


@dataclass
class _Atom:
    idx: int
    residue: _Residue
    name: str
    atomic_number: int
    charge: float = 0.0
    mass: float = 12.0
    type: str = "CT"
    epsilon: float = 0.1
    rmin: float = 1.5


@dataclass
class _Bond:
    atom1: _Atom
    atom2: _Atom


@dataclass
class _Structure:
    atoms: list[_Atom]
    residues: list[_Residue]
    bonds: list[_Bond]
    box: list[float] | None = None
    parm_data: dict[str, list[object]] | None = None


def _structures() -> tuple[_Structure, _Structure]:
    residues = [
        _Residue(0, "REC"),
        _Residue(1, "LIG"),
        _Residue(2, "SOL"),
        _Residue(3, "NA"),
    ]
    md_atoms = [
        _Atom(0, residues[0], "CA", 6),
        _Atom(1, residues[0], "N", 7),
        _Atom(2, residues[1], "C1", 6),
        _Atom(3, residues[1], "O1", 8),
        _Atom(4, residues[2], "OW", 8),
        _Atom(5, residues[3], "NA", 11),
    ]
    scoring_residues = [_Residue(0, "REC"), _Residue(1, "LIG")]
    scoring_atoms = [
        _Atom(0, scoring_residues[0], "CA", 6),
        _Atom(1, scoring_residues[0], "N", 7),
        _Atom(2, scoring_residues[1], "C1", 6),
        _Atom(3, scoring_residues[1], "O1", 8),
    ]
    return (
        _Structure(
            md_atoms,
            residues,
            [_Bond(md_atoms[0], md_atoms[1]), _Bond(md_atoms[2], md_atoms[3])],
            [30.0, 30.0, 30.0, 90.0, 90.0, 90.0],
        ),
        _Structure(
            scoring_atoms,
            scoring_residues,
            [
                _Bond(scoring_atoms[0], scoring_atoms[1]),
                _Bond(scoring_atoms[2], scoring_atoms[3]),
            ],
        ),
    )


def _selection(md: list[int], scoring: list[int]) -> dict[str, object]:
    return {"md_atom_indices": md, "scoring_atom_indices": scoring}


def _artifact(
    root: Path,
    source: str,
    role: str,
    target: str,
    content: str | None = None,
) -> dict[str, object]:
    path = root / source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or f"{role}\n", encoding="utf-8")
    return {
        "role": role,
        "source": source,
        "path": target,
        "media_type": "application/octet-stream",
    }


def _draft(root: Path) -> tuple[Path, dict[str, object]]:
    index = """[ System ]
1 2 3 4 5 6
[ Complex ]
1 2 3 4
[ Receptor ]
1 2
[ Ligand ]
3 4
"""
    md_artifacts = [
        _artifact(root, "sources/topol.top", "md_topology", "md/topol.top"),
        _artifact(root, "sources/system.gro", "md_coordinates", "md/system.gro"),
        _artifact(root, "sources/index.ndx", "index", "md/index.ndx", index),
        _artifact(
            root,
            "sources/preparation.json",
            "preparation_receipt",
            "provenance/preparation.json",
            '{"schema_version":"example.preparation.v1"}\n',
        ),
    ]
    scoring_artifacts = [
        _artifact(
            root,
            "sources/complex.prmtop",
            "complex_topology",
            "scoring/complex.prmtop",
        ),
        _artifact(
            root,
            "sources/receptor.prmtop",
            "receptor_topology",
            "scoring/receptor.prmtop",
        ),
        _artifact(
            root,
            "sources/ligand.prmtop",
            "ligand_topology",
            "scoring/ligand.prmtop",
        ),
        _artifact(
            root,
            "sources/coordinates.inpcrd",
            "coordinates",
            "scoring/coordinates.inpcrd",
        ),
    ]
    raw: dict[str, object] = {
        "schema_version": "mmpbsa.prepared_case_draft.v1",
        "case_id": "wrapped_case",
        "md": {
            "backend": "gromacs",
            "artifacts": md_artifacts,
            "metadata": {"source": "existing-md"},
        },
        "scoring": {
            "backend": "ambertools",
            "radii_model": "mbondi2",
            "source_md_atom_indices": [1, 2, 3, 4],
            "artifacts": scoring_artifacts,
            "metadata": {"dry_complex": True},
        },
        "entities": [
            {
                "entity_id": "receptor",
                "kind": "protein",
                "md_atom_indices": [1, 2],
                "scoring_atom_indices": [1, 2],
                "residues": [
                    {
                        "residue_id": "R_1",
                        "residue_name": "REC",
                        "md_residue_index": 1,
                        "scoring_residue_index": 1,
                    }
                ],
                "named_selections": {
                    "all": _selection([1, 2], [1, 2]),
                    "heavy_atoms": _selection([1, 2], [1, 2]),
                    "backbone": _selection([1, 2], [1, 2]),
                    "interface_backbone": _selection([1], [1]),
                },
            },
            {
                "entity_id": "ligand",
                "kind": "small_molecule",
                "md_atom_indices": [3, 4],
                "scoring_atom_indices": [3, 4],
                "residues": [
                    {
                        "residue_id": "L_1",
                        "residue_name": "LIG",
                        "md_residue_index": 2,
                        "scoring_residue_index": 2,
                    }
                ],
                "named_selections": {
                    "all": _selection([3, 4], [3, 4]),
                    "heavy_atoms": _selection([3, 4], [3, 4]),
                },
            },
            {
                "entity_id": "structural_water",
                "kind": "water",
                "md_atom_indices": [5],
                "scoring_atom_indices": [],
                "residues": [
                    {
                        "residue_id": "W_1",
                        "residue_name": "SOL",
                        "md_residue_index": 3,
                        "scoring_residue_index": None,
                    }
                ],
                "named_selections": {
                    "all": _selection([5], []),
                    "heavy_atoms": _selection([5], []),
                },
            },
        ],
        "primary_binding": {
            "binding_id": "primary",
            "receptor_entity_ids": ["receptor"],
            "ligand_entity_ids": ["ligand"],
            "excluded_entity_ids": ["structural_water"],
            "topology_artifact_roles": {
                "complex": "complex_topology",
                "receptor": "receptor_topology",
                "ligand": "ligand_topology",
            },
            "cross_entity_bonds": [],
        },
        "provenance": {"owner": "test"},
    }
    path = root / "draft.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, raw


@pytest.fixture
def mechanical_stubs(monkeypatch):
    import mmpbsa.core.prepared_draft as module

    monkeypatch.setattr(module, "_load_parmed_structures", lambda _resolved: _structures())
    monkeypatch.setattr(
        module,
        "_run_projection_admission",
        lambda _manifest, _resolved: {
            "accepted": True,
            "issues": [],
            "complex_signature": "c" * 64,
            "component_signatures": {
                "receptor": "r" * 64,
                "ligand": "l" * 64,
            },
            "tool": {"name": "FakeParmEd", "version": [4, 3, 1]},
        },
    )


def test_inspection_derives_counts_and_v2_contract(tmp_path: Path, mechanical_stubs) -> None:
    path, _ = _draft(tmp_path)

    inspection = inspect_prepared_case_draft(path)

    assert inspection.validation_summary()["valid"] is True
    assert inspection.prepared_manifest.md.atom_count == 6
    assert inspection.prepared_manifest.scoring.atom_count == 4
    assert inspection.index_group_counts == {
        "System": 6,
        "Complex": 4,
        "Receptor": 2,
        "Ligand": 2,
    }
    assert inspection.prepared_manifest.entity("receptor").selection_signature
    seal = inspection.prepared_manifest.provenance["prepared_case_draft"]
    assert seal["source_structure_policy"] == "not_read"
    assert seal["modeling_policy"] == "none"


def test_seal_copies_and_atomically_publishes_v2(tmp_path: Path, mechanical_stubs) -> None:
    path, _ = _draft(tmp_path)
    output = tmp_path / "sealed"

    prepared = seal_prepared_case_draft(path, output)

    assert prepared.case_id == "wrapped_case"
    assert prepared.manifest.schema_version == "mmpbsa.prepared_case.v2"
    assert prepared.artifact("md_topology").read_text(encoding="utf-8") == ("md_topology\n")
    (tmp_path / "sources/topol.top").write_text("changed\n", encoding="utf-8")
    assert prepared.artifact("md_topology").read_text(encoding="utf-8") == ("md_topology\n")


def test_existing_seal_destination_is_refused(tmp_path: Path, mechanical_stubs) -> None:
    path, _ = _draft(tmp_path)
    output = tmp_path / "sealed"
    output.mkdir()

    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        seal_prepared_case_draft(path, output)


def test_draft_rejects_unknown_fields(tmp_path: Path) -> None:
    path, raw = _draft(tmp_path)
    raw["source_structure"] = "input.pdb"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="unknown fields: source_structure"):
        load_prepared_case_draft(path)


def test_yaml_and_json_use_the_same_strict_contract(tmp_path: Path) -> None:
    _, raw = _draft(tmp_path)
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    draft = load_prepared_case_draft(path)

    assert draft.manifest.case_id == "wrapped_case"


def test_modeling_receipt_artifact_is_mandatory(tmp_path: Path) -> None:
    path, raw = _draft(tmp_path)
    md = raw["md"]
    assert isinstance(md, dict)
    artifacts = md["artifacts"]
    assert isinstance(artifacts, list)
    md["artifacts"] = [
        item
        for item in artifacts
        if isinstance(item, dict) and item["role"] != "preparation_receipt"
    ]
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="preparation_receipt"):
        load_prepared_case_draft(path)


def test_draft_requires_the_standard_topology_roles(tmp_path: Path) -> None:
    path, raw = _draft(tmp_path)
    binding = raw["primary_binding"]
    assert isinstance(binding, dict)
    roles = binding["topology_artifact_roles"]
    assert isinstance(roles, dict)
    roles["complex"] = "coordinates"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="standard topology artifact roles"):
        load_prepared_case_draft(path)


def test_explicit_scoring_to_md_identity_is_checked(tmp_path: Path, mechanical_stubs) -> None:
    path, raw = _draft(tmp_path)
    scoring = raw["scoring"]
    assert isinstance(scoring, dict)
    scoring["source_md_atom_indices"] = [2, 1, 3, 4]
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="atom identity differs"):
        inspect_prepared_case_draft(path)


def test_explicit_scoring_to_md_charge_is_checked(
    tmp_path: Path, mechanical_stubs, monkeypatch
) -> None:
    import mmpbsa.core.prepared_draft as module

    path, _ = _draft(tmp_path)
    md, scoring = _structures()
    scoring.atoms[0].charge = 0.01
    monkeypatch.setattr(module, "_load_parmed_structures", lambda _resolved: (md, scoring))

    with pytest.raises(ContractError, match="atom charge differs"):
        inspect_prepared_case_draft(path)


def test_entity_residue_name_is_checked_against_topologies(
    tmp_path: Path, mechanical_stubs
) -> None:
    path, raw = _draft(tmp_path)
    entities = raw["entities"]
    assert isinstance(entities, list)
    receptor = entities[0]
    assert isinstance(receptor, dict)
    residues = receptor["residues"]
    assert isinstance(residues, list)
    assert isinstance(residues[0], dict)
    residues[0]["residue_name"] = "WRONG"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="name differs in MD topology"):
        inspect_prepared_case_draft(path)


def test_entity_residue_index_must_exist(tmp_path: Path, mechanical_stubs) -> None:
    path, raw = _draft(tmp_path)
    entities = raw["entities"]
    assert isinstance(entities, list)
    water = entities[2]
    assert isinstance(water, dict)
    residues = water["residues"]
    assert isinstance(residues, list)
    assert isinstance(residues[0], dict)
    residues[0]["md_residue_index"] = 99
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="exceeds the GROMACS residue count"):
        inspect_prepared_case_draft(path)


def test_entity_atom_inventory_must_cover_complete_declared_residues(
    tmp_path: Path, mechanical_stubs, monkeypatch
) -> None:
    import mmpbsa.core.prepared_draft as module

    path, _ = _draft(tmp_path)
    md, scoring = _structures()
    md.atoms.append(_Atom(6, md.residues[2], "HW1", 1, mass=1.0))
    monkeypatch.setattr(module, "_load_parmed_structures", lambda _resolved: (md, scoring))

    with pytest.raises(ContractError, match="exactly contain its declared residues"):
        inspect_prepared_case_draft(path)


def test_heavy_atom_selection_has_exact_mechanical_semantics(
    tmp_path: Path, mechanical_stubs
) -> None:
    path, raw = _draft(tmp_path)
    entities = raw["entities"]
    assert isinstance(entities, list)
    receptor = entities[0]
    assert isinstance(receptor, dict)
    selections = receptor["named_selections"]
    assert isinstance(selections, dict)
    selections["heavy_atoms"] = _selection([1], [1])
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="atomic number greater than one"):
        inspect_prepared_case_draft(path)


def test_backbone_selection_has_exact_mechanical_semantics(
    tmp_path: Path, mechanical_stubs
) -> None:
    path, raw = _draft(tmp_path)
    entities = raw["entities"]
    assert isinstance(entities, list)
    receptor = entities[0]
    assert isinstance(receptor, dict)
    selections = receptor["named_selections"]
    assert isinstance(selections, dict)
    selections["backbone"] = _selection([1], [1])
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="exactly N, CA, C, and O atoms"):
        inspect_prepared_case_draft(path)


def test_core_projection_contract_checks_parameters_and_internal_terms() -> None:
    from mmpbsa.core.topology_projection import component_projection_issues

    _, complex_structure = _structures()
    residue = _Residue(0, "REC")
    atoms = [_Atom(0, residue, "CA", 6), _Atom(1, residue, "N", 7)]
    receptor = _Structure(
        atoms=atoms,
        residues=[residue],
        bonds=[_Bond(atoms[0], atoms[1])],
    )
    assert component_projection_issues(complex_structure, receptor, (1, 2), "receptor") == ()

    receptor.atoms[0].charge = 0.25
    assert component_projection_issues(complex_structure, receptor, (1, 2), "receptor") == (
        "receptor: charge differs at component atom 1",
    )
    receptor.atoms[0].charge = 0.0
    receptor.bonds.clear()
    assert component_projection_issues(complex_structure, receptor, (1, 2), "receptor") == (
        "receptor: internal term count differs",
    )


def test_core_projection_normalizes_terms_after_nonmonotonic_projection() -> None:
    from mmpbsa.core.topology_projection import component_projection_issues

    _, complex_structure = _structures()
    residue = _Residue(0, "REC")
    atoms = [_Atom(0, residue, "N", 7), _Atom(1, residue, "CA", 6)]
    receptor = _Structure(
        atoms=atoms,
        residues=[residue],
        bonds=[_Bond(atoms[0], atoms[1])],
    )

    assert component_projection_issues(complex_structure, receptor, (2, 1), "receptor") == ()


def test_md_to_scoring_projection_checks_lj_and_internal_terms() -> None:
    from mmpbsa.core.topology_projection import mapped_system_projection_issues

    md, scoring = _structures()
    assert mapped_system_projection_issues(md, scoring, (1, 2, 3, 4), "MD-to-scoring") == ()

    scoring.atoms[0].epsilon = 0.2
    assert mapped_system_projection_issues(md, scoring, (1, 2, 3, 4), "MD-to-scoring") == (
        "MD-to-scoring: epsilon differs at target atom 1",
    )
    scoring.atoms[0].epsilon = float("nan")
    assert mapped_system_projection_issues(md, scoring, (1, 2, 3, 4), "MD-to-scoring") == (
        "MD-to-scoring: epsilon differs at target atom 1",
    )
    scoring.atoms[0].epsilon = 0.1
    scoring.bonds.clear()
    assert mapped_system_projection_issues(md, scoring, (1, 2, 3, 4), "MD-to-scoring") == (
        "MD-to-scoring: internal term identities differ",
    )


def test_improper_projection_accepts_only_full_reversal_without_changing_signatures() -> None:
    from mmpbsa.backends.parmed import (
        AtomRecord,
        ProjectionTolerances,
        TermRecord,
        TopologyView,
        _projection_issues,
    )
    from mmpbsa.core.topology_projection import (
        component_projection_issues,
        mapped_system_projection_issues,
        topology_signature,
    )

    md, scoring = _structures()
    term_type = SimpleNamespace(phi_k=1.1, per=2, phase=180, scee=0, scnb=0)

    def term(atoms, order):
        return SimpleNamespace(
            **{f"atom{i}": atoms[j] for i, j in enumerate(order, 1)}, improper=True, type=term_type
        )

    md.dihedrals = [term(md.atoms, (0, 1, 2, 3))]
    scoring.dihedrals = [term(scoring.atoms, (3, 2, 1, 0))]
    signature = topology_signature(scoring)
    assert component_projection_issues(md, scoring, (1, 2, 3, 4), "component") == ()
    assert mapped_system_projection_issues(md, scoring, (1, 2, 3, 4), "mapped") == ()
    assert topology_signature(scoring) == signature

    atoms = tuple(AtomRecord(i, 1, "MOL", f"C{i}", 6, "CT", 0, 12, 0.1, 1.5) for i in range(1, 5))
    source = TopologyView(atoms, (TermRecord("improper", (1, 2, 3, 4), ("phi_k", 1.1)),))
    target = TopologyView(atoms, (TermRecord("improper", (4, 3, 2, 1), ("phi_k", 1.1)),))
    assert not _projection_issues("component", source, (1, 2, 3, 4), target, ProjectionTolerances())
    target = TopologyView(atoms, (TermRecord("improper", (4, 3, 2, 1), ("phi_k", 2.2)),))
    assert _projection_issues("component", source, (1, 2, 3, 4), target, ProjectionTolerances())

    scoring.dihedrals = [term(scoring.atoms, (1, 0, 2, 3))]
    assert component_projection_issues(md, scoring, (1, 2, 3, 4), "component")
    assert mapped_system_projection_issues(md, scoring, (1, 2, 3, 4), "mapped")


def test_md_topology_cannot_hide_a_binding_boundary_bond(
    tmp_path: Path, mechanical_stubs, monkeypatch
) -> None:
    import mmpbsa.core.prepared_draft as module

    path, _ = _draft(tmp_path)
    md, scoring = _structures()
    md.bonds.append(_Bond(md.atoms[1], md.atoms[2]))
    monkeypatch.setattr(module, "_load_parmed_structures", lambda _resolved: (md, scoring))

    with pytest.raises(ContractError, match="MD binding-boundary admission failed"):
        inspect_prepared_case_draft(path)


def test_amber_radii_metadata_matches_the_declared_model() -> None:
    from mmpbsa.core.prepared_draft import _amber_radii_inventory

    _, structure = _structures()
    structure.parm_data = {
        "RADIUS_SET": ["H(N)-modified Bondi radii (mbondi2)"],
        "RADII": [1.5, 1.2, 1.7, 1.4],
    }
    assert _amber_radii_inventory(structure, "mbondi2", "complex topology") == (1.5, 1.2, 1.7, 1.4)

    with pytest.raises(ContractError, match="does not match declared radii_model"):
        _amber_radii_inventory(structure, "bondi", "complex topology")

    assert structure.parm_data is not None
    structure.parm_data["RADII"] = [1.5, 1.2, 1.7, 0.0]
    with pytest.raises(ContractError, match="finite and positive"):
        _amber_radii_inventory(structure, "mbondi2", "complex topology")


def test_index_groups_must_match_declared_binding(tmp_path: Path, mechanical_stubs) -> None:
    path, _ = _draft(tmp_path)
    (tmp_path / "sources/index.ndx").write_text(
        "[ System ]\n1 2 3 4 5 6\n[ Complex ]\n1 2 3 4\n[ Receptor ]\n1 3\n[ Ligand ]\n2 4\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="Receptor.*differs"):
        inspect_prepared_case_draft(path)


def test_complex_index_group_must_preserve_scoring_atom_order(
    tmp_path: Path, mechanical_stubs
) -> None:
    path, _ = _draft(tmp_path)
    (tmp_path / "sources/index.ndx").write_text(
        "[ System ]\n1 2 3 4 5 6\n[ Complex ]\n2 1 3 4\n[ Receptor ]\n1 2\n[ Ligand ]\n3 4\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="Complex.*required order"):
        inspect_prepared_case_draft(path)


def test_complex_topology_cannot_hide_a_binding_boundary_bond() -> None:
    from mmpbsa.core.topology_projection import binding_boundary_bond_issues

    _, complex_structure = _structures()
    complex_structure.bonds.append(_Bond(complex_structure.atoms[1], complex_structure.atoms[2]))

    assert binding_boundary_bond_issues(
        complex_structure,
        (1, 2),
        (3, 4),
    ) == ("complex topology contains a receptor-ligand covalent bond: 2--3",)


def test_relocation_is_revalidated_before_atomic_publication(
    tmp_path: Path, mechanical_stubs, monkeypatch
) -> None:
    import mmpbsa.core.prepared_draft as module

    path, _ = _draft(tmp_path)
    calls: list[dict[str, Path]] = []

    def load(resolved):
        calls.append(dict(resolved))
        if len(calls) == 2:
            raise ContractError("relocated topology cannot be loaded")
        return _structures()

    monkeypatch.setattr(module, "_load_parmed_structures", load)
    output = tmp_path / "sealed"

    with pytest.raises(ContractError, match="relocated topology"):
        seal_prepared_case_draft(path, output)

    assert len(calls) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".sealed.staging-*"))


def test_concurrent_seal_lock_cannot_overwrite_destination(
    tmp_path: Path, mechanical_stubs
) -> None:
    path, _ = _draft(tmp_path)
    output = tmp_path / "sealed"
    lock_path = tmp_path / ".sealed.seal.lock"

    with InterProcessFileLock(lock_path, owner="test-owner"):
        with pytest.raises(ArtifactIntegrityError, match="is locked"):
            seal_prepared_case_draft(path, output)

    assert not output.exists()


def test_prepared_case_cli_is_independent_of_workflow_config(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    inspection = SimpleNamespace(
        validation_summary=lambda: {
            "valid": True,
            "case_id": "wrapped_case",
        }
    )
    monkeypatch.setattr(cli, "validate_prepared_case_draft", lambda _path: inspection)

    result = cli.main(["prepared-case", "validate", "--draft", str(tmp_path / "draft.json")])

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "case_id": "wrapped_case",
        "valid": True,
    }


def test_prepared_case_cli_validates_a_sealed_manifest(monkeypatch, capsys, tmp_path: Path) -> None:
    manifest = (tmp_path / "prepared_case.json").resolve()
    prepared = SimpleNamespace(
        case_id="wrapped_case",
        manifest=SimpleNamespace(schema_version="mmpbsa.prepared_case.v2"),
        manifest_path=manifest,
        resolved_artifacts={"md_topology": tmp_path / "topol.top"},
    )
    monkeypatch.setattr(cli, "load_prepared_case", lambda _path: prepared)

    result = cli.main(["prepared-case", "validate", "--manifest", str(manifest)])

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "artifact_count": 1,
        "case_id": "wrapped_case",
        "manifest": str(manifest),
        "schema_version": "mmpbsa.prepared_case.v2",
        "valid": True,
    }

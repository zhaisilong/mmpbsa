from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mmpbsa.core.errors import ArtifactIntegrityError, ContractError
from mmpbsa.core.md_inputs import AmberPairValidation
from mmpbsa.core.models import TaskKey
from mmpbsa.core.prepared import (
    ENTITY_KINDS,
    EntityFeature,
    EntitySelection,
    PreparedEntity,
    ResidueIdentity,
    load_prepared_case,
)
from mmpbsa.core.prepared_projection import prepared_case_projection
from mmpbsa.core.state import StageId


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(
    root: Path, path: str, role: str, data: bytes
) -> dict[str, object]:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {
        "path": path,
        "sha256": _digest(data),
        "size": len(data),
        "role": role,
        "media_type": "application/octet-stream",
    }


def _selection(
    md_atoms: tuple[int, ...],
    scoring_atoms: tuple[int, ...],
) -> EntitySelection:
    return EntitySelection(
        md_atom_indices=md_atoms,
        scoring_atom_indices=scoring_atoms,
    )


def _protein_entity(*, include_interface: bool = True) -> PreparedEntity:
    named = {
        "all": _selection((1, 2), (1, 2)),
        "heavy_atoms": _selection((1,), (1,)),
        "backbone": _selection((1, 2), (1, 2)),
    }
    if include_interface:
        named["interface_backbone"] = _selection((1,), (1,))
    return PreparedEntity(
        entity_id="receptor_protein",
        kind="protein",
        md_atom_indices=(1, 2),
        scoring_atom_indices=(1, 2),
        residues=(
            ResidueIdentity(
                residue_id="A_1",
                residue_name="ALA",
                md_residue_index=1,
                scoring_residue_index=1,
            ),
        ),
        named_selections=named,
    )


def _small_molecule_entity() -> PreparedEntity:
    return PreparedEntity(
        entity_id="ligand",
        kind="small_molecule",
        md_atom_indices=(3, 4),
        scoring_atom_indices=(3, 4),
        residues=(
            ResidueIdentity(
                residue_id="L_1",
                residue_name="LIG",
                md_residue_index=2,
                scoring_residue_index=2,
            ),
        ),
        named_selections={
            "all": _selection((3, 4), (3, 4)),
            "heavy_atoms": _selection((3,), (3,)),
        },
    )


def _peptide_entity() -> PreparedEntity:
    return PreparedEntity(
        entity_id="ligand",
        kind="peptide",
        md_atom_indices=(3, 4),
        scoring_atom_indices=(3, 4),
        residues=(
            ResidueIdentity(
                residue_id="P_1",
                residue_name="NME",
                md_residue_index=2,
                scoring_residue_index=2,
            ),
        ),
        named_selections={
            "all": _selection((3, 4), (3, 4)),
            "heavy_atoms": _selection((3, 4), (3, 4)),
            "backbone": _selection((3,), (3,)),
        },
        features=(
            EntityFeature(
                feature_id="c_terminal_cap",
                kind="cap",
                selection=_selection((4,), (4,)),
                residue_ids=("P_1",),
                metadata={"chemistry": "N-methylamide"},
            ),
        ),
    )


def _water_entity() -> PreparedEntity:
    return PreparedEntity(
        entity_id="structural_water_1",
        kind="water",
        md_atom_indices=(5,),
        scoring_atom_indices=(),
        residues=(
            ResidueIdentity(
                residue_id="W_1",
                residue_name="HOH",
                md_residue_index=3,
                scoring_residue_index=None,
            ),
        ),
        named_selections={
            "all": _selection((5,), ()),
            "heavy_atoms": _selection((5,), ()),
        },
    )


def _manifest(
    root: Path,
    *,
    peptide: bool = False,
    include_interface: bool = True,
    include_solvated_topology: bool = False,
) -> dict[str, object]:
    artifacts = {
        "md": _artifact(root, "md/system.tpr", "md_topology", b"md-system-v2"),
        "md_coordinates": _artifact(
            root,
            "md/system.gro",
            "md_coordinates",
            b"md-coordinates-v2",
        ),
        "index": _artifact(
            root,
            "md/index.ndx",
            "index",
            b"md-index-v2",
        ),
        "solvated_topology": _artifact(
            root,
            "md/solvated.prmtop",
            "solvated_topology",
            b"solvated-topology-v2",
        ),
        "complex": _artifact(
            root,
            "scoring/complex.prmtop",
            "complex_topology",
            b"complex-v2",
        ),
        "receptor": _artifact(
            root,
            "scoring/receptor.prmtop",
            "receptor_topology",
            b"receptor-v2",
        ),
        "ligand": _artifact(
            root,
            "scoring/ligand.prmtop",
            "ligand_topology",
            b"ligand-v2",
        ),
        "coordinates": _artifact(
            root,
            "scoring/coordinates.inpcrd",
            "coordinates",
            b"coordinates-v2",
        ),
    }
    ligand = _peptide_entity() if peptide else _small_molecule_entity()
    return {
        "schema_version": "mmpbsa.prepared_case.v2",
        "case_id": "lig_ejm_31",
        "md": {
            "backend": "gromacs",
            "atom_count": 6,
            "system_signature": "b" * 64,
            "artifacts": [
                artifacts["md"],
                artifacts["md_coordinates"],
                artifacts["index"],
            ]
            + (
                [artifacts["solvated_topology"]]
                if include_solvated_topology
                else []
            ),
            "metadata": {"water_model": "tip3p"},
        },
        "scoring": {
            "backend": "ambertools",
            "atom_count": 4,
            "atom_order_signature": "a" * 64,
            "source_md_atom_indices": [1, 2, 3, 4],
            "radii_model": "mbondi2",
            "artifacts": [
                artifacts["complex"],
                artifacts["receptor"],
                artifacts["ligand"],
                artifacts["coordinates"],
            ],
        },
        "entities": [
            _protein_entity(include_interface=include_interface).as_dict(),
            ligand.as_dict(),
            _water_entity().as_dict(),
        ],
        "primary_binding": {
            "binding_id": "primary",
            "receptor_entity_ids": ["receptor_protein"],
            "ligand_entity_ids": ["ligand"],
            "excluded_entity_ids": ["structural_water_1"],
            "topology_artifact_roles": {
                "complex": "complex_topology",
                "receptor": "receptor_topology",
                "ligand": "ligand_topology",
            },
            "cross_entity_bonds": [],
        },
        "provenance": {"preparer": "test-fixture"},
    }


def _write_prepared_case(
    root: Path,
    *,
    peptide: bool = False,
) -> Path:
    manifest = _manifest(root, peptide=peptide)
    path = root / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.mark.parametrize("peptide", [False, True])
def test_scientific_context_preserves_sealed_entity_selection_and_modifications(tmp_path, peptide):
    from mmpbsa.core.scientific_context import build_scientific_context

    prepared = load_prepared_case(_write_prepared_case(tmp_path, peptide=peptide))
    original = prepared.manifest_path.read_bytes()
    context = build_scientific_context(prepared)
    ligand = next(row for row in context["prepared_case"]["entities"]
                  if row["entity_id"] == "ligand")
    assert context["status"] == "available"
    assert context["prepared_case"]["modeling_provenance"] == {"preparer": "test-fixture"}
    assert ligand["kind"] == ("peptide" if peptide else "small_molecule")
    selection = ligand["named_selections"]["backbone" if peptide else "heavy_atoms"]
    assert selection["scoring_atom_indices"] == [3]
    if peptide:
        assert ligand["features"][0]["kind"] == "cap"
    assert prepared.manifest_path.read_bytes() == original


def test_prepared_case_v2_keeps_verified_external_references(
    tmp_path: Path,
) -> None:
    prepared = load_prepared_case(
        _write_prepared_case(tmp_path),
        expected_case_id="lig_ejm_31",
    )

    assert prepared.root == tmp_path.resolve()
    assert prepared.md_view.artifact("md_topology") == (
        tmp_path / "md/system.tpr"
    ).resolve()
    assert set(prepared.md_view.resolved_artifacts) == {
        "md_coordinates",
        "md_topology",
    }
    assert {item.role for item in prepared.md_view.system.artifacts} == {
        "md_coordinates",
        "md_topology",
    }
    metadata_case = replace(
        prepared,
        manifest=replace(
            prepared.manifest,
            md=replace(
                prepared.manifest.md,
                metadata={"ligand_entity_id": "ligand"},
            ),
        ),
    )
    assert metadata_case.md_view.system.metadata == {}
    assert set(prepared.binding_view.resolved_artifacts) == {
        "complex_topology",
        "receptor_topology",
        "ligand_topology",
        "coordinates",
        "index",
    }
    assert not hasattr(prepared.md_view, "binding")
    assert not hasattr(prepared.md_view, "entities")


def test_openmm_v2_selects_only_sealed_amber_md_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path, include_solvated_topology=True)
    manifest["md"]["artifacts"].append(
        _artifact(
            tmp_path,
            "md/solvated.inpcrd",
            "solvated_coordinates",
            b"solvated-coordinates-v2",
        )
    )
    path = tmp_path / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    prepared = load_prepared_case(path)
    monkeypatch.setattr(
        "mmpbsa.core.prepared.validate_amber_md_pair",
        lambda *_: AmberPairValidation(6, 4, 0.0, 0.0, 0),
    )

    amber = prepared.md_view_for_profile("openmm_8_5@2")
    assert amber.input_format == "amber"
    assert set(amber.resolved_artifacts) == {
        "solvated_topology",
        "solvated_coordinates",
    }
    assert amber.system.metadata == {}
    assert prepared.md_view.input_format == "gromacs"
    assert set(prepared.md_view.resolved_artifacts) == {
        "md_topology",
        "md_coordinates",
    }
    task = TaskKey(StageId.MD, case_id="lig_ejm_31", replica_id="rep01")
    science = {"md_engine_profile": "openmm_8_5@2"}
    baseline = prepared_case_projection(task, science, prepared)
    changed_gromacs_signature = replace(
        prepared,
        manifest=replace(
            prepared.manifest,
            md=replace(prepared.manifest.md, system_signature="f" * 64),
        ),
    )
    assert prepared_case_projection(task, science, changed_gromacs_signature) == baseline
    assert baseline["md"]["artifacts"] == [
        item.as_dict() for item in amber.system.artifacts
    ]
    changed_gromacs_artifacts = replace(
        prepared,
        manifest=replace(
            prepared.manifest,
            md=replace(
                prepared.manifest.md,
                artifacts=tuple(
                    replace(item, sha256="e" * 64)
                    if item.role == "md_topology"
                    else item
                    for item in prepared.manifest.md.artifacts
                ),
            ),
        ),
    )
    assert prepared_case_projection(task, science, changed_gromacs_artifacts) == baseline
    changed_amber_artifacts = replace(
        prepared,
        manifest=replace(
            prepared.manifest,
            md=replace(
                prepared.manifest.md,
                artifacts=tuple(
                    replace(item, sha256="e" * 64)
                    if item.role == "solvated_topology"
                    else item
                    for item in prepared.manifest.md.artifacts
                ),
            ),
        ),
    )
    assert prepared_case_projection(task, science, changed_amber_artifacts) != baseline


def test_direct_md_view_cannot_expose_unreferenced_artifacts(tmp_path: Path) -> None:
    prepared = load_prepared_case(_write_prepared_case(tmp_path))
    view = prepared.md_view
    index = next(item for item in prepared.manifest.md.artifacts if item.role == "index")
    with pytest.raises(ContractError, match="outside its declared topology include closure"):
        replace(
            view,
            system=replace(view.system, artifacts=(*view.system.artifacts, index)),
            resolved_artifacts={
                **view.resolved_artifacts,
                "index": prepared.artifact("index"),
            },
        )


def test_trajectory_view_exposes_projection_without_preparation_metadata(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, include_solvated_topology=True)
    path = tmp_path / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    view = load_prepared_case(path).trajectory_view

    assert view.source_md_atom_indices == (1, 2, 3, 4)
    assert view.atom_order_signature == "a" * 64
    assert view.residue_crosswalk == (
        (1, 1, "ALA"),
        (2, 2, "LIG"),
    )
    assert set(view.resolved_artifacts) == {
        "solvated_topology",
        "md_topology",
        "md_coordinates",
        "complex_topology",
        "index",
    }
    assert set(view.binding_selections) == {
        "receptor_all",
        "ligand_all",
        "receptor_interface_backbone",
        "ligand_qc",
        "receptor_heavy",
        "ligand_heavy",
    }
    assert not hasattr(view, "md")
    assert not hasattr(view, "scoring")
    assert not hasattr(view, "entities")
    assert not hasattr(view, "binding")
    assert view.binding_selection("ligand_qc").scoring_atom_indices == (3,)


def test_trajectory_view_rejects_malformed_direct_construction(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, include_solvated_topology=True)
    path = tmp_path / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    view = load_prepared_case(path).trajectory_view

    with pytest.raises(ContractError, match="positive MD atom indices"):
        replace(view, source_md_atom_indices=(True,))
    with pytest.raises(ContractError, match="contain triples"):
        replace(view, residue_crosswalk=((1, 1),))
    with pytest.raises(ContractError, match="unique and scoring-ordered"):
        replace(
            view,
            residue_crosswalk=((1, 2, "ALA"), (2, 1, "LIG")),
        )
    with pytest.raises(ContractError, match="topology role"):
        replace(view, complex_topology_role="")
    for reserved_role in ("index", "solvated_topology", "md_topology", "md_coordinates"):
        with pytest.raises(ContractError, match="reserved trajectory"):
            replace(view, complex_topology_role=reserved_role)
    with pytest.raises(ContractError, match="typed binding selection"):
        replace(
            view,
            binding_selections={
                **view.binding_selections,
                "ligand_qc": object(),
            },
        )
    with pytest.raises(ContractError, match="cannot be empty"):
        replace(
            view,
            binding_selections={
                **view.binding_selections,
                "ligand_qc": EntitySelection((), ()),
            },
        )
    with pytest.raises(ContractError, match="inconsistent MD/scoring"):
        replace(
            view,
            binding_selections={
                **view.binding_selections,
                "ligand_qc": EntitySelection((999,), (3,)),
            },
        )


def test_primary_binding_exposes_nonredundant_qc_and_contact_selections(
    tmp_path: Path,
) -> None:
    prepared = load_prepared_case(_write_prepared_case(tmp_path))
    manifest = prepared.manifest

    assert manifest.binding_selection("receptor_all").scoring_atom_indices == (
        1,
        2,
    )
    assert manifest.binding_selection("ligand_all").scoring_atom_indices == (
        3,
        4,
    )
    assert manifest.binding_selection(
        "receptor_interface_backbone"
    ).scoring_atom_indices == (1,)
    assert manifest.binding_selection(
        "ligand_qc"
    ).scoring_atom_indices == (3,)
    assert manifest.binding_selection(
        "receptor_heavy"
    ).scoring_atom_indices == (1,)
    assert manifest.binding_selection(
        "ligand_heavy"
    ).scoring_atom_indices == (3,)
    assert manifest.entity("ligand").default_qc_selection == "heavy_atoms"
    assert prepared.binding_view.binding_selection(
        "ligand_qc"
    ).scoring_atom_indices == (3,)
    assert prepared.binding_view.topology_artifact_role("complex") == (
        "complex_topology"
    )
    assert prepared.binding_view.topology_artifact("ligand") == (
        tmp_path / "scoring/ligand.prmtop"
    ).resolve()


def test_modified_peptide_is_one_ligand_and_uses_backbone_qc(
    tmp_path: Path,
) -> None:
    prepared = load_prepared_case(
        _write_prepared_case(tmp_path, peptide=True)
    )
    ligand = prepared.manifest.entity("ligand")

    assert ligand.kind == "peptide"
    assert ligand.default_qc_selection == "backbone"
    assert ligand.features[0].feature_id == "c_terminal_cap"
    assert ligand.features[0].selection.scoring_atom_indices == (4,)
    assert prepared.manifest.binding_selection(
        "ligand_qc"
    ).scoring_atom_indices == (3,)


def test_md_only_structural_water_is_explicit_but_bulk_solvent_is_not(
    tmp_path: Path,
) -> None:
    prepared = load_prepared_case(_write_prepared_case(tmp_path))
    water = prepared.manifest.entity("structural_water_1")

    assert water.md_atom_indices == (5,)
    assert water.scoring_atom_indices == ()
    assert 6 not in {
        atom
        for entity in prepared.manifest.entities
        for atom in entity.md_atom_indices
    }
    assert prepared.manifest.primary_binding.excluded_entity_ids == (
        "structural_water_1",
    )


def test_scoring_projection_must_map_back_to_the_same_entity(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    scoring = manifest["scoring"]
    assert isinstance(scoring, dict)
    scoring["source_md_atom_indices"] = [1, 3, 2, 4]
    path = tmp_path / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ContractError,
        match="MD/scoring projections differ",
    ):
        load_prepared_case(path)


def test_primary_binding_must_cover_every_dry_scoring_atom(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    binding = manifest["primary_binding"]
    assert isinstance(binding, dict)
    binding["excluded_entity_ids"] = [
        "ligand",
        "structural_water_1",
    ]
    binding["ligand_entity_ids"] = ["receptor_protein"]
    binding["receptor_entity_ids"] = []
    path = tmp_path / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ContractError, match="cannot be empty"):
        load_prepared_case(path)


def test_receptor_interface_backbone_is_required(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, include_interface=False)
    path = tmp_path / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ContractError,
        match="must declare a non-empty interface_backbone",
    ):
        load_prepared_case(path)


def test_cross_receptor_ligand_covalent_bond_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    binding = manifest["primary_binding"]
    assert isinstance(binding, dict)
    binding["cross_entity_bonds"] = [
        {
            "entity_a_id": "receptor_protein",
            "md_atom_a": 1,
            "entity_b_id": "ligand",
            "md_atom_b": 3,
        }
    ]
    path = tmp_path / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ContractError,
        match="covalent bond across the receptor-ligand boundary",
    ):
        load_prepared_case(path)


def test_ligand_role_accepts_only_the_two_supported_ligand_forms(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    entities = manifest["entities"]
    assert isinstance(entities, list)
    ligand = entities[1]
    assert isinstance(ligand, dict)
    ligand["kind"] = "cofactor"
    path = tmp_path / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ContractError,
        match="ligand entities must be small_molecule or peptide",
    ):
        load_prepared_case(path)


def test_all_supported_entity_kinds_are_public() -> None:
    assert ENTITY_KINDS == {
        "protein",
        "small_molecule",
        "peptide",
        "water",
        "cofactor",
        "ion",
        "other",
    }


def test_prepared_case_revalidates_digest_and_detects_same_size_tamper(
    tmp_path: Path,
) -> None:
    manifest_path = _write_prepared_case(tmp_path)
    target = tmp_path / "scoring/complex.prmtop"
    assert len(b"complex-v2") == len(b"tampered!!")
    target.write_bytes(b"tampered!!")

    with pytest.raises(
        ArtifactIntegrityError,
        match="artifact digest changed for complex_topology",
    ):
        load_prepared_case(manifest_path)


def test_prepared_case_identity_is_bound_to_workflow_case(
    tmp_path: Path,
) -> None:
    manifest_path = _write_prepared_case(tmp_path)

    with pytest.raises(ContractError, match="does not match workflow case"):
        load_prepared_case(manifest_path, expected_case_id="different_case")


def test_v1_manifest_is_not_implicitly_executed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["schema_version"] = "mmpbsa.prepared_case.v1"
    path = tmp_path / "prepared_case.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ContractError,
        match="unsupported PreparedCase schema",
    ):
        load_prepared_case(path)

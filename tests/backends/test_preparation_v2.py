from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mmpbsa.backends.prep import PreparationSpecV2, load_preparation_spec
from mmpbsa.backends.prep.errors import PreparationError
from mmpbsa.backends.prep.models_v2 import ResidueRef
from mmpbsa.backends.prep.peptide_backbone import _peptide_backbone_indices
from mmpbsa.utils import sha256_file


def _document(source_digest: str, *, feature_residue: int = 2) -> str:
    return f"""\
schema_version: mmpbsa.preparation.v2
case_id: peptide_case
source:
  path: ../inputs/source.cif
  sha256: {source_digest}
entities:
  receptor:
    entity_id: receptor
    kind: protein
    ranges: [{{chain_id: A, start: 1, end: 2}}]
  ligand:
    entity_id: ligand
    kind: peptide
    formal_charge: -2
    ranges: [{{chain_id: B, start: 1, end: 1}}]
    features:
      - feature_id: phosphosite
        kind: ptm
        residues: [{{chain_id: B, residue_number: {feature_residue}}}]
force_fields:
  leaprcs: [leaprc.protein.ff14SB, leaprc.phosaa14SB, leaprc.water.tip3p]
  water: TIP3P
  water_box: TIP3PBOX
  pb_radii: mbondi2
solvation:
  shape: truncated_octahedron
  padding_angstrom: 10.0
  salt_concentration_molar: 0.15
interface: {{cutoff_angstrom: 5.0}}
expectations:
  receptor_atoms: 20
  receptor_residues: 2
  ligand_atoms: 10
  ligand_residues: 1
  ligand_charge_e: -2
"""


def test_v2_loads_sha_bound_mmcif_and_explicit_peptide_entity(tmp_path: Path) -> None:
    source = tmp_path / "inputs" / "source.cif"
    source.parent.mkdir()
    source.write_text("data_test\n", encoding="utf-8")
    config = tmp_path / "preparation" / "case.yaml"
    config.parent.mkdir()
    config.write_text(_document(sha256_file(source), feature_residue=1), encoding="utf-8")

    spec = load_preparation_spec(config)

    assert isinstance(spec, PreparationSpecV2)
    spec.verify_sources()
    assert spec.ligand.entity_id == "ligand"
    assert spec.ligand.formal_charge == -2
    assert spec.ligand.features[0].kind == "ptm"


def test_v2_rejects_feature_that_escapes_ligand_entity(tmp_path: Path) -> None:
    source = tmp_path / "inputs" / "source.cif"
    source.parent.mkdir()
    source.write_text("data_test\n", encoding="utf-8")
    config = tmp_path / "preparation" / "case.yaml"
    config.parent.mkdir()
    config.write_text(_document(sha256_file(source)), encoding="utf-8")

    with pytest.raises(PreparationError, match="escape selected ranges"):
        load_preparation_spec(config)


def test_v2_rejects_source_path_outside_project_root(tmp_path: Path) -> None:
    source = tmp_path / "outside.cif"
    source.write_text("data_test\n", encoding="utf-8")
    config = tmp_path / "project" / "preparation" / "case.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        _document(sha256_file(source), feature_residue=1).replace(
            "../inputs/source.cif", "../../../outside.cif"
        ),
        encoding="utf-8",
    )

    with pytest.raises(PreparationError, match="below the project root"):
        load_preparation_spec(config)


def _residue(first_atom_index: int, names: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        atoms=tuple(
            SimpleNamespace(idx=first_atom_index + offset - 1, name=name)
            for offset, name in enumerate(names)
        )
    )


@pytest.mark.parametrize("glycan_first", [False, True])
def test_v2_peptide_backbone_ignores_ligand_range_order(glycan_first: bool) -> None:
    # Invented peptide and glycan residues test order-independent selection.
    peptide = [
        (ResidueRef("B", number), _residue((number - 1) * 4 + 1, ("N", "CA", "C", "O")))
        for number in range(1, 5)
    ]
    glycan = (ResidueRef("G", 1), _residue(17, ("C1", "O1")))
    pairs = [glycan, *peptide] if glycan_first else [*peptide, glycan]

    assert _peptide_backbone_indices(
        tuple(key for key, _ in pairs),
        tuple(residue for _, residue in pairs),
        tuple(range(1, 19)),
    ) == tuple(range(1, 17))


@pytest.mark.parametrize(
    "second_residue_names",
    [
        ("N", "CA", "C"),
        ("N", "CA", "C", "O", "O"),
        ("C1", "O1"),
    ],
)
def test_v2_peptide_backbone_rejects_incomplete_selected_chain(
    second_residue_names: tuple[str, ...],
) -> None:
    keys = (ResidueRef("B", 1), ResidueRef("B", 2), ResidueRef("G", 1))
    residues = (
        _residue(1, ("N", "CA", "C", "O")),
        _residue(5, second_residue_names),
        _residue(5 + len(second_residue_names), ("C1", "O1")),
    )

    with pytest.raises(PreparationError, match="incomplete for residue B_2"):
        _peptide_backbone_indices(keys, residues, tuple(range(1, 7 + len(second_residue_names))))


def test_v2_peptide_backbone_rejects_multiple_candidate_chains() -> None:
    keys = (ResidueRef("B", 1), ResidueRef("G", 1), ResidueRef("C", 1))
    residues = (
        _residue(1, ("N", "CA", "C", "O")),
        _residue(5, ("C1", "O1")),
        _residue(7, ("N", "CA", "C", "O")),
    )

    with pytest.raises(PreparationError, match="exactly one chain.*'B', 'C'"):
        _peptide_backbone_indices(keys, residues, tuple(range(1, 11)))


def test_v2_peptide_backbone_rejects_no_candidate_chain() -> None:
    with pytest.raises(PreparationError, match="exactly one chain.*found \\[\\]"):
        _peptide_backbone_indices(
            (ResidueRef("B", 1), ResidueRef("G", 1)),
            (_residue(1, ("N", "CA", "C")), _residue(4, ("C1", "O1"))),
            tuple(range(1, 6)),
        )

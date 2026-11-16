"""Fail-closed backbone selection for a modified peptide ligand."""

from __future__ import annotations

from collections.abc import Sequence

from .errors import PreparationError
from .models_v2 import ResidueRef
from .planner import BACKBONE_NAMES


def _peptide_backbone_indices(
    ligand_keys: Sequence[ResidueRef],
    ligand_residues: Sequence[object],
    ligand_indices: Sequence[int],
) -> tuple[int, ...]:
    """Select one complete peptide chain without relying on range order."""

    if len(ligand_keys) != len(ligand_residues):
        raise PreparationError("peptide residue inventory differs from source selection")
    by_chain: dict[str, list[tuple[ResidueRef, object]]] = {}
    candidate_chains: set[str] = set()
    for key, residue in zip(ligand_keys, ligand_residues, strict=True):
        by_chain.setdefault(key.chain_id, []).append((key, residue))
        names = [str(atom.name) for atom in residue.atoms if str(atom.name) in BACKBONE_NAMES]
        if len(names) == len(BACKBONE_NAMES) and set(names) == BACKBONE_NAMES:
            candidate_chains.add(key.chain_id)
    if len(candidate_chains) != 1:
        raise PreparationError(
            "peptide QC backbone requires exactly one chain with complete "
            f"N/CA/C/O residues; found {sorted(candidate_chains)!r}"
        )

    chain_id = next(iter(candidate_chains))
    selected: set[int] = set()
    for key, residue in by_chain[chain_id]:
        backbone_atoms = [atom for atom in residue.atoms if str(atom.name) in BACKBONE_NAMES]
        if (
            len(backbone_atoms) != len(BACKBONE_NAMES)
            or {str(atom.name) for atom in backbone_atoms} != BACKBONE_NAMES
        ):
            raise PreparationError(
                "peptide QC backbone is incomplete for residue "
                f"{key.chain_id}_{key.residue_number}; every residue on the "
                "selected chain must contain exactly one N, CA, C, and O"
            )
        selected.update(int(atom.idx) + 1 for atom in backbone_atoms)
    if not selected.issubset(ligand_indices):
        raise PreparationError("peptide QC backbone escapes the ligand atom inventory")
    return tuple(index for index in ligand_indices if index in selected)

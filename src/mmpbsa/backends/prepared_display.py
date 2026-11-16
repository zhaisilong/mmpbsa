"""Explicit PreparedCase entities projected into dry display components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mmpbsa.core import ContractError, PreparedCaseManifest, StageInput
from mmpbsa.core.models import require_identifier

from .errors import BackendConfigurationError


def _display_identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise BackendConfigurationError(f"{label} must be a non-empty string")
    try:
        return require_identifier(value, label)
    except ContractError as exc:
        raise BackendConfigurationError(str(exc)) from exc


def prepared_components(
    prepared: StageInput,
    raw_selected: object,
    case_id: str,
) -> tuple[
    Mapping[str, tuple[int, ...]],
    Mapping[str, object],
    Mapping[str, tuple[int, ...]],
]:
    """Project the v2 primary binding onto the dry display atom order.

    Automatic report assets consume the dry complex topology and the centered
    dry trajectory.  Component indices must therefore use the scoring
    projection, even when an entity also exists in the role-neutral MD system.
    MD-only excluded entities (for example structural waters) are recorded as
    unavailable instead of being assigned invalid indices in the dry topology.
    """

    if prepared.payload.get("schema_version") != "mmpbsa.prepared_case.v2":
        raise BackendConfigurationError(
            f"prepared:{case_id} report assets require a "
            "mmpbsa.prepared_case.v2 payload; legacy partition payloads are "
            "not executable"
        )
    try:
        manifest = PreparedCaseManifest.from_mapping(prepared.payload)
    except ContractError as exc:
        raise BackendConfigurationError(
            f"prepared:{case_id} has an invalid PreparedCase v2 payload: {exc}"
        ) from exc
    if manifest.case_id != case_id:
        raise BackendConfigurationError(
            f"prepared:{case_id} payload declares case_id {manifest.case_id!r}"
        )

    if raw_selected is None:
        selected = ("receptor", "ligand")
    else:
        if not isinstance(raw_selected, Sequence) or isinstance(
            raw_selected, (str, bytes, bytearray)
        ):
            raise BackendConfigurationError(
                f"report.assets.{case_id}.components must be a sequence"
            )
        selected = tuple(
            _display_identifier(
                item,
                f"report.assets.{case_id}.components item",
            )
            for item in raw_selected
        )
    if len(selected) != len(set(selected)):
        raise BackendConfigurationError(
            f"report.assets.{case_id}.components entries must be unique"
        )
    # Role groups remain the backwards-compatible default.  Named entities
    # allow a multi-component receptor to be presented without inferring
    # chemical identity from residue names or atom order.
    entity_ids = {entity.entity_id for entity in manifest.entities}
    supported = {"receptor", "ligand", "excluded"} | entity_ids
    unknown = set(selected).difference(supported)
    if unknown:
        raise BackendConfigurationError(
            f"report.assets.{case_id}.components contains unsupported "
            "PreparedCase v2 components: " + ", ".join(sorted(unknown))
        )

    binding = manifest.primary_binding
    entity_groups = {
        "receptor": binding.receptor_entity_ids,
        "ligand": binding.ligand_entity_ids,
        "excluded": binding.excluded_entity_ids,
    }
    entity_groups.update(
        {entity_id: (entity_id,) for entity_id in entity_ids if entity_id not in entity_groups}
    )
    result: dict[str, tuple[int, ...]] = {}
    component_availability: dict[str, object] = {
        "projection": "dry_scoring",
    }
    for component in selected:
        entity_ids = entity_groups[component]
        entities = tuple(manifest.entity(entity_id) for entity_id in entity_ids)
        indices = tuple(sorted(atom for entity in entities for atom in entity.scoring_atom_indices))
        if indices:
            result[component] = indices
            component_availability[component] = {
                "status": "available",
                "entity_ids": list(entity_ids),
                "atom_count": len(indices),
            }
        elif component == "excluded" or all(
            entity.entity_id in binding.excluded_entity_ids for entity in entities
        ):
            component_availability[component] = {
                "status": "unavailable" if entity_ids else "not_applicable",
                "reason": (
                    "md_only_entities_excluded_from_dry_display"
                    if entity_ids
                    else "no_excluded_entities"
                ),
                "entity_ids": list(entity_ids),
                "entity_kinds": [entity.kind for entity in entities],
            }
        else:  # The typed PreparedCase contract should make this unreachable.
            raise BackendConfigurationError(
                f"prepared:{case_id} primary binding component {component!r} "
                "has no atoms in the dry scoring projection"
            )

    selected_atoms: set[int] = set()
    for component, indices in result.items():
        if selected_atoms.intersection(indices):
            raise BackendConfigurationError(
                f"report.assets.{case_id}.components overlap in the dry scoring "
                f"projection at component {component!r}; select role groups or "
                "their entities, not both"
            )
        selected_atoms.update(indices)

    if "excluded" not in component_availability:
        excluded = tuple(manifest.entity(entity_id) for entity_id in binding.excluded_entity_ids)
        component_availability["excluded"] = {
            "status": "unavailable" if excluded else "not_applicable",
            "reason": (
                "md_only_entities_excluded_from_dry_display" if excluded else "no_excluded_entities"
            ),
            "entity_ids": [entity.entity_id for entity in excluded],
            "entity_kinds": [entity.kind for entity in excluded],
        }
    if not result:
        raise BackendConfigurationError(
            f"report.assets case {case_id!r} selected no components present "
            "in the dry scoring projection"
        )
    diagnostic_selections = {
        name: manifest.binding_selection(name).scoring_atom_indices
        for name in (
            "receptor_all",
            "ligand_all",
            "receptor_interface_backbone",
            "ligand_qc",
        )
    }
    return result, component_availability, diagnostic_selections

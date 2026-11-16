"""Stage-scoped projections of the sealed PreparedCase contract."""

from __future__ import annotations

from collections.abc import Mapping

from .errors import ContractError
from .models import TaskKey
from .prepared import (
    PreparedBindingView,
    PreparedCase,
    PreparedMdView,
    PreparedTrajectoryView,
)
from .state import StageId

PREPARED_CASE_CONSUMERS = frozenset(
    {
        StageId.ADMIT,
        StageId.MD,
        StageId.TRAJECTORY,
        StageId.TRAJECTORY_QC,
        StageId.SCORING,
    }
)


def prepared_request_view(
    task: TaskKey,
    prepared_case: PreparedCase | None,
    science: Mapping[str, object] | None = None,
) -> PreparedMdView | PreparedTrajectoryView | PreparedBindingView | None:
    """Expose only the prepared contract consumed by a provider stage."""

    if prepared_case is None or task.stage not in PREPARED_CASE_CONSUMERS:
        return None
    if task.stage is StageId.MD:
        profile = (
            str(science.get("md_engine_profile", "gromacs_canonical@1"))
            if science is not None
            else "gromacs_canonical@1"
        )
        return prepared_case.md_view_for_profile(profile)
    if task.stage is StageId.TRAJECTORY:
        return prepared_case.trajectory_view
    return prepared_case.binding_view


def _scoring_entity_projection(
    prepared_case: PreparedCase,
    *,
    include_kind: bool,
    include_metadata: bool,
) -> list[dict[str, object]]:
    """Return binding endpoint entities expressed only in scoring order."""

    manifest = prepared_case.manifest
    endpoint_ids = set(
        manifest.primary_binding.receptor_entity_ids
        + manifest.primary_binding.ligand_entity_ids
    )
    result: list[dict[str, object]] = []
    for entity in sorted(
        (item for item in manifest.entities if item.entity_id in endpoint_ids),
        key=lambda item: item.entity_id,
    ):
        projected: dict[str, object] = {
            "entity_id": entity.entity_id,
            "scoring_atom_indices": list(entity.scoring_atom_indices),
            "residues": [
                {
                    "residue_id": residue.residue_id,
                    "residue_name": residue.residue_name,
                    "scoring_residue_index": residue.scoring_residue_index,
                }
                for residue in sorted(
                    entity.residues,
                    key=lambda item: (
                        item.scoring_residue_index
                        if item.scoring_residue_index is not None
                        else 0,
                        item.residue_id,
                    ),
                )
            ],
            "named_selections": {
                name: {
                    "scoring_atom_indices": list(
                        entity.named_selections[name].scoring_atom_indices
                    )
                }
                for name in sorted(entity.named_selections)
            },
            "features": [
                {
                    "feature_id": feature.feature_id,
                    "kind": feature.kind,
                    "scoring_atom_indices": list(
                        feature.selection.scoring_atom_indices
                    ),
                    "residue_ids": list(feature.residue_ids),
                    **(
                        {"metadata": dict(feature.metadata)}
                        if include_metadata
                        else {}
                    ),
                }
                for feature in sorted(
                    entity.features,
                    key=lambda item: item.feature_id,
                )
            ],
        }
        if include_kind:
            projected["kind"] = entity.kind
        if include_metadata:
            projected["metadata"] = dict(entity.metadata)
        result.append(projected)
    return result


def _binding_projection(prepared_case: PreparedCase) -> dict[str, object]:
    """Project the dry receptor-ligand boundary, excluding MD-only entities."""

    binding = prepared_case.manifest.primary_binding
    excluded = set(binding.excluded_entity_ids)
    return {
        "binding_id": binding.binding_id,
        "receptor_entity_ids": list(binding.receptor_entity_ids),
        "ligand_entity_ids": list(binding.ligand_entity_ids),
        "topology_artifact_roles": binding.topology_artifact_roles.as_dict(),
        "cross_entity_bonds": [
            item.as_dict()
            for item in sorted(
                binding.cross_entity_bonds,
                key=lambda item: item.canonical_key,
            )
            if item.entity_a_id not in excluded and item.entity_b_id not in excluded
        ],
    }


def _artifact_projection(
    prepared_case: PreparedCase,
    roles: set[str],
) -> dict[str, list[dict[str, object]]]:
    """Select artifact references by consumed role and retain their system."""

    manifest = prepared_case.manifest
    systems = (("md", manifest.md), ("scoring", manifest.scoring))
    available = {item.role for _, system in systems for item in system.artifacts}
    missing = roles.difference(available)
    if missing:
        raise ContractError(
            "PreparedCase is missing stage-required artifact roles: "
            + ", ".join(sorted(missing))
        )
    return {
        label: [
            artifact.as_dict()
            for artifact in sorted(
                (item for item in system.artifacts if item.role in roles),
                key=lambda item: item.role,
            )
        ]
        for label, system in systems
        if any(item.role in roles for item in system.artifacts)
    }


def prepared_case_projection(
    task: TaskKey,
    science: Mapping[str, object],
    prepared_case: PreparedCase | None,
) -> dict[str, object] | None:
    """Project PreparedCase onto the exact scientific stage boundary."""

    if prepared_case is None or task.stage not in PREPARED_CASE_CONSUMERS:
        return None

    manifest = prepared_case.manifest
    common: dict[str, object] = {
        "schema_version": manifest.schema_version,
        "case_id": manifest.case_id,
    }
    if task.stage is StageId.ADMIT:
        admission_binding = manifest.primary_binding
        topology_roles = admission_binding.topology_artifact_roles
        admission_roles = set(
            prepared_case.binding_view.resolved_artifacts
        )
        return {
            **common,
            "scoring_projection": {"atom_count": manifest.scoring.atom_count},
            "primary_binding": {
                "receptor_entity_ids": list(admission_binding.receptor_entity_ids),
                "ligand_entity_ids": list(admission_binding.ligand_entity_ids),
                "topology_artifact_roles": topology_roles.as_dict(),
            },
            "artifacts": _artifact_projection(
                prepared_case,
                admission_roles,
            ),
            "binding_selections": {
                name: list(
                    manifest.binding_selection(name).scoring_atom_indices
                )
                for name in ("receptor_all", "ligand_all")
            },
        }
    if task.stage is StageId.MD:
        view = prepared_request_view(task, prepared_case, science)
        assert isinstance(view, PreparedMdView)
        if view.input_format == "amber":
            return {
                **common,
                "input_format": view.input_format,
                "md": {
                    "atom_count": view.system.atom_count,
                    "artifacts": [item.as_dict() for item in view.system.artifacts],
                },
            }
        return {
            **common,
            "md": view.system.as_dict(),
            "input_format": view.input_format,
        }

    if task.stage is StageId.TRAJECTORY:
        trajectory_view = prepared_case.trajectory_view
        v3 = science.get("adapter_contract") == "mmpbsa.trajectory.v3"
        if v3:
            required_roles = {
                "full_system_topology_role": "md_topology",
                "full_system_coordinates_role": "md_coordinates",
            }
            for field, role in required_roles.items():
                if science.get(field, role) != role:
                    raise ContractError(
                        f"trajectory v3 {field} must be {role!r}"
                    )
            exposed_roles = {
                "md_topology",
                "md_coordinates",
                trajectory_view.complex_topology_role,
            }
        else:
            source_topology_role = str(science.get("index_role", "index"))
            exposed_roles = {
                "index",
                trajectory_view.complex_topology_role,
                *(
                    {"solvated_topology"}
                    if "solvated_topology" in trajectory_view.resolved_artifacts
                    else set()
                ),
            }
        missing_roles = exposed_roles.difference(trajectory_view.resolved_artifacts)
        if not v3 and source_topology_role not in trajectory_view.resolved_artifacts:
            missing_roles.add(source_topology_role)
        if missing_roles:
            raise ContractError(
                "PreparedCase is missing stage-required artifact roles: "
                + ", ".join(sorted(missing_roles))
            )
        return {
            **common,
            "scoring_projection": {
                "atom_count": manifest.scoring.atom_count,
                "atom_order_signature": manifest.scoring.atom_order_signature,
                "source_md_atom_indices": list(
                    manifest.scoring.source_md_atom_indices
                ),
                **(
                    {
                        "residue_crosswalk": [
                            {
                                "md_residue_index": md_index,
                                "scoring_residue_index": scoring_index,
                                "residue_name": name,
                            }
                            for md_index, scoring_index, name in (
                                trajectory_view.residue_crosswalk
                            )
                        ]
                    }
                    if v3
                    else {}
                ),
            },
            "artifacts": _artifact_projection(
                prepared_case,
                exposed_roles,
            ),
            "binding_selections": {
                name: list(
                    trajectory_view.binding_selection(name).scoring_atom_indices
                )
                for name in sorted(trajectory_view.binding_selections)
            },
        }

    binding_projection = _binding_projection(prepared_case)
    complex_role = manifest.primary_binding.topology_artifact_roles.complex
    if task.stage is StageId.TRAJECTORY_QC:
        return {
            **common,
            "scoring_projection": {
                "atom_count": manifest.scoring.atom_count,
                "atom_order_signature": manifest.scoring.atom_order_signature,
            },
            "entities": _scoring_entity_projection(
                prepared_case,
                include_kind=True,
                include_metadata=False,
            ),
            "primary_binding": binding_projection,
            "artifacts": _artifact_projection(prepared_case, {complex_role}),
        }
    if task.stage is StageId.SCORING:
        return {
            **common,
            "scoring": manifest.scoring.as_dict(),
            "entities": _scoring_entity_projection(
                prepared_case,
                include_kind=True,
                include_metadata=True,
            ),
            "primary_binding": binding_projection,
        }
    raise AssertionError(f"unhandled PreparedCase consumer: {task.stage}")

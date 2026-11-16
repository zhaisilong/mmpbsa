"""PreparedCase v2 entity, binding, and artifact-admission contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from mmpbsa.utils import resolve_within, sha256_file

from .errors import ArtifactIntegrityError, ContractError
from .md_inputs import (
    AmberPairValidation,
    topology_include_closure,
    validate_amber_md_pair,
)
from .models import ArtifactRef, require_identifier

PREPARED_CASE_SCHEMA = "mmpbsa.prepared_case.v2"
ENTITY_KINDS = frozenset(
    {
        "protein",
        "small_molecule",
        "peptide",
        "water",
        "cofactor",
        "ion",
        "other",
    }
)
LIGAND_ENTITY_KINDS = frozenset({"small_molecule", "peptide"})
BINDING_SELECTION_NAMES = frozenset(
    {
        "receptor_all",
        "ligand_all",
        "receptor_interface_backbone",
        "ligand_qc",
        "receptor_heavy",
        "ligand_heavy",
    }
)
_MD_EXECUTION_ARTIFACT_ROLES = frozenset(
    {"md_coordinates", "md_topology"}
)
_AMBER_MD_EXECUTION_ARTIFACT_ROLES = frozenset(
    {"solvated_coordinates", "solvated_topology"}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PEPTIDE_FEATURE_KINDS = frozenset(
    {
        "cap",
        "cyclization",
        "glycan",
        "modification",
        "ptm",
        "staple",
        "other",
    }
)


def _strict_keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> None:
    missing = required.difference(value)
    unknown = set(value).difference(required | optional)
    if missing:
        raise ContractError(f"{label} is missing: " + ", ".join(sorted(missing)))
    if unknown:
        raise ContractError(
            f"{label} contains unknown fields: " + ", ".join(sorted(unknown))
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ContractError(f"{label} keys must be strings")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ContractError(f"{label} must be a sequence")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _indices(
    value: object,
    label: str,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    result = tuple(
        _positive_int(item, f"{label} item")
        for item in _sequence(value, label)
    )
    if not result and not allow_empty:
        raise ContractError(f"{label} cannot be empty")
    if tuple(sorted(set(result))) != result:
        raise ContractError(f"{label} must be sorted and unique")
    return result


def _identifiers(
    value: object,
    label: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    result = tuple(str(item) for item in _sequence(value, label))
    if not result and not allow_empty:
        raise ContractError(f"{label} cannot be empty")
    for item in result:
        require_identifier(item, f"{label} item")
    if tuple(sorted(set(result))) != result:
        raise ContractError(f"{label} must be sorted and unique")
    return result


def _artifact_inventory(
    value: object, label: str
) -> tuple[ArtifactRef, ...]:
    result = tuple(
        ArtifactRef.from_mapping(_mapping(item, f"{label} artifact"))
        for item in _sequence(value, label)
    )
    if not result:
        raise ContractError(f"{label} cannot be empty")
    roles = tuple(item.role for item in result)
    paths = tuple(item.path for item in result)
    if len(roles) != len(set(roles)):
        raise ContractError(f"{label} artifact roles must be unique")
    if len(paths) != len(set(paths)):
        raise ContractError(f"{label} artifact paths must be unique")
    return result


@dataclass(frozen=True, slots=True)
class EntitySelection:
    """The same explicit atom selection projected into MD and scoring orders."""

    md_atom_indices: tuple[int, ...]
    scoring_atom_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for label, values in (
            ("selection md_atom_indices", self.md_atom_indices),
            ("selection scoring_atom_indices", self.scoring_atom_indices),
        ):
            if any(isinstance(item, bool) or item < 1 for item in values):
                raise ContractError(f"{label} must contain positive integers")
            if tuple(sorted(set(values))) != values:
                raise ContractError(f"{label} must be sorted and unique")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> EntitySelection:
        _strict_keys(
            value,
            required={"md_atom_indices", "scoring_atom_indices"},
            label="entity selection",
        )
        return cls(
            md_atom_indices=_indices(
                value["md_atom_indices"],
                "selection md_atom_indices",
                allow_empty=True,
            ),
            scoring_atom_indices=_indices(
                value["scoring_atom_indices"],
                "selection scoring_atom_indices",
                allow_empty=True,
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "md_atom_indices": list(self.md_atom_indices),
            "scoring_atom_indices": list(self.scoring_atom_indices),
        }


@dataclass(frozen=True, slots=True)
class ResidueIdentity:
    """Stable residue identity and its positions in the two atom orders."""

    residue_id: str
    residue_name: str
    md_residue_index: int
    scoring_residue_index: int | None

    def __post_init__(self) -> None:
        require_identifier(self.residue_id, "residue_id")
        if not self.residue_name.strip():
            raise ContractError("residue_name cannot be empty")
        _positive_int(self.md_residue_index, "md_residue_index")
        if self.scoring_residue_index is not None:
            _positive_int(
                self.scoring_residue_index,
                "scoring_residue_index",
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> ResidueIdentity:
        _strict_keys(
            value,
            required={
                "residue_id",
                "residue_name",
                "md_residue_index",
                "scoring_residue_index",
            },
            label="residue identity",
        )
        raw_scoring_index = value["scoring_residue_index"]
        return cls(
            residue_id=str(value["residue_id"]),
            residue_name=str(value["residue_name"]),
            md_residue_index=_positive_int(
                value["md_residue_index"],
                "md_residue_index",
            ),
            scoring_residue_index=(
                None
                if raw_scoring_index is None
                else _positive_int(
                    raw_scoring_index,
                    "scoring_residue_index",
                )
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "residue_id": self.residue_id,
            "residue_name": self.residue_name,
            "md_residue_index": self.md_residue_index,
            "scoring_residue_index": self.scoring_residue_index,
        }


@dataclass(frozen=True, slots=True)
class EntityFeature:
    """An explicit modified-peptide feature contained by one peptide entity."""

    feature_id: str
    kind: str
    selection: EntitySelection
    residue_ids: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier(self.feature_id, "feature_id")
        if self.kind not in _PEPTIDE_FEATURE_KINDS:
            raise ContractError(
                f"unsupported peptide feature kind: {self.kind!r}"
            )
        if not self.selection.md_atom_indices:
            raise ContractError("feature MD atom selection cannot be empty")
        for residue_id in self.residue_ids:
            require_identifier(residue_id, "feature residue_id")
        if tuple(sorted(set(self.residue_ids))) != self.residue_ids:
            raise ContractError(
                "feature residue_ids must be sorted and unique"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> EntityFeature:
        _strict_keys(
            value,
            required={"feature_id", "kind", "selection", "residue_ids"},
            optional={"metadata"},
            label="entity feature",
        )
        return cls(
            feature_id=str(value["feature_id"]),
            kind=str(value["kind"]),
            selection=EntitySelection.from_mapping(
                _mapping(value["selection"], "feature selection")
            ),
            residue_ids=_identifiers(
                value["residue_ids"],
                "feature residue_ids",
                allow_empty=True,
            ),
            metadata=_mapping(
                value.get("metadata", {}),
                "feature metadata",
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "selection": self.selection.as_dict(),
            "residue_ids": list(self.residue_ids),
            "metadata": dict(self.metadata),
        }


def _entity_selection_digest(
    *,
    md_atom_indices: tuple[int, ...],
    scoring_atom_indices: tuple[int, ...],
    named_selections: Mapping[str, EntitySelection],
    features: tuple[EntityFeature, ...],
) -> str:
    canonical = {
        "md_atom_indices": list(md_atom_indices),
        "scoring_atom_indices": list(scoring_atom_indices),
        "named_selections": {
            name: named_selections[name].as_dict()
            for name in sorted(named_selections)
        },
        "features": [
            {
                "feature_id": item.feature_id,
                "kind": item.kind,
                "selection": item.selection.as_dict(),
                "residue_ids": list(item.residue_ids),
            }
            for item in sorted(features, key=lambda item: item.feature_id)
        ],
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedEntity:
    """A stable chemical entity, independent of its binding role."""

    entity_id: str
    kind: str
    md_atom_indices: tuple[int, ...]
    scoring_atom_indices: tuple[int, ...]
    residues: tuple[ResidueIdentity, ...]
    named_selections: Mapping[str, EntitySelection]
    selection_signature: str = ""
    features: tuple[EntityFeature, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier(self.entity_id, "entity_id")
        if self.kind not in ENTITY_KINDS:
            raise ContractError(f"unsupported entity kind: {self.kind!r}")
        for label, values in (
            ("entity md_atom_indices", self.md_atom_indices),
            ("entity scoring_atom_indices", self.scoring_atom_indices),
        ):
            if any(isinstance(item, bool) or item < 1 for item in values):
                raise ContractError(f"{label} must contain positive integers")
            if tuple(sorted(set(values))) != values:
                raise ContractError(f"{label} must be sorted and unique")
        if not self.md_atom_indices:
            raise ContractError("entity md_atom_indices cannot be empty")
        if not self.residues:
            raise ContractError("entity residues cannot be empty")

        residue_ids = tuple(item.residue_id for item in self.residues)
        md_residue_indices = tuple(
            item.md_residue_index for item in self.residues
        )
        scoring_residue_indices = tuple(
            item.scoring_residue_index
            for item in self.residues
            if item.scoring_residue_index is not None
        )
        if len(residue_ids) != len(set(residue_ids)):
            raise ContractError("entity residue_ids must be unique")
        if len(md_residue_indices) != len(set(md_residue_indices)):
            raise ContractError("entity MD residue indices must be unique")
        if len(scoring_residue_indices) != len(
            set(scoring_residue_indices)
        ):
            raise ContractError(
                "entity scoring residue indices must be unique"
            )
        if self.scoring_atom_indices and len(scoring_residue_indices) != len(
            self.residues
        ):
            raise ContractError(
                "a scored entity must map every residue into scoring order"
            )
        if not self.scoring_atom_indices and scoring_residue_indices:
            raise ContractError(
                "an MD-only entity cannot have scoring residue indices"
            )

        required_selections = {"all", "heavy_atoms"}
        if self.kind in {"protein", "peptide"}:
            required_selections.add("backbone")
        missing = required_selections.difference(self.named_selections)
        if missing:
            raise ContractError(
                f"entity {self.entity_id!r} is missing named selections: "
                + ", ".join(sorted(missing))
            )
        for name, selection in self.named_selections.items():
            require_identifier(name, "named selection")
            if not isinstance(selection, EntitySelection):
                raise ContractError(
                    f"named selection {name!r} must be an EntitySelection"
                )
            if not selection.md_atom_indices:
                raise ContractError(
                    f"named selection {name!r} MD atoms cannot be empty"
                )
            if not set(selection.md_atom_indices).issubset(
                self.md_atom_indices
            ):
                raise ContractError(
                    f"named selection {name!r} escapes entity MD atoms"
                )
            if not set(selection.scoring_atom_indices).issubset(
                self.scoring_atom_indices
            ):
                raise ContractError(
                    f"named selection {name!r} escapes entity scoring atoms"
                )
        all_selection = self.named_selections["all"]
        if (
            all_selection.md_atom_indices != self.md_atom_indices
            or all_selection.scoring_atom_indices
            != self.scoring_atom_indices
        ):
            raise ContractError(
                "named selection 'all' must equal the entity atom inventory"
            )
        interface_backbone = self.named_selections.get(
            "interface_backbone"
        )
        if interface_backbone is not None:
            if self.kind not in {"protein", "peptide"}:
                raise ContractError(
                    "only protein or peptide entities may declare "
                    "interface_backbone"
                )
            backbone = self.named_selections["backbone"]
            if (
                not set(interface_backbone.md_atom_indices).issubset(
                    backbone.md_atom_indices
                )
                or not set(
                    interface_backbone.scoring_atom_indices
                ).issubset(backbone.scoring_atom_indices)
            ):
                raise ContractError(
                    "interface_backbone must be a subset of backbone"
                )

        if self.features and self.kind != "peptide":
            raise ContractError(
                "only peptide entities may declare modification features"
            )
        feature_ids = tuple(item.feature_id for item in self.features)
        if len(feature_ids) != len(set(feature_ids)):
            raise ContractError("entity feature_ids must be unique")
        residue_id_set = set(residue_ids)
        for feature in self.features:
            if not set(feature.selection.md_atom_indices).issubset(
                self.md_atom_indices
            ):
                raise ContractError(
                    f"feature {feature.feature_id!r} escapes entity MD atoms"
                )
            if not set(feature.selection.scoring_atom_indices).issubset(
                self.scoring_atom_indices
            ):
                raise ContractError(
                    f"feature {feature.feature_id!r} escapes entity scoring atoms"
                )
            if not set(feature.residue_ids).issubset(residue_id_set):
                raise ContractError(
                    f"feature {feature.feature_id!r} references unknown residues"
                )

        expected_signature = _entity_selection_digest(
            md_atom_indices=self.md_atom_indices,
            scoring_atom_indices=self.scoring_atom_indices,
            named_selections=self.named_selections,
            features=self.features,
        )
        if not self.selection_signature:
            object.__setattr__(
                self,
                "selection_signature",
                expected_signature,
            )
        elif (
            not _SHA256.fullmatch(self.selection_signature)
            or self.selection_signature != expected_signature
        ):
            raise ContractError(
                f"entity {self.entity_id!r} selection_signature is invalid"
            )

    @property
    def default_qc_selection(self) -> str | None:
        if self.kind == "small_molecule":
            return "heavy_atoms"
        if self.kind == "peptide":
            return "backbone"
        return None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PreparedEntity:
        _strict_keys(
            value,
            required={
                "entity_id",
                "kind",
                "md_atom_indices",
                "scoring_atom_indices",
                "residues",
                "named_selections",
                "selection_signature",
            },
            optional={"features", "metadata"},
            label="prepared entity",
        )
        raw_selections = _mapping(
            value["named_selections"],
            "entity named_selections",
        )
        return cls(
            entity_id=str(value["entity_id"]),
            kind=str(value["kind"]),
            md_atom_indices=_indices(
                value["md_atom_indices"],
                "entity md_atom_indices",
                allow_empty=False,
            ),
            scoring_atom_indices=_indices(
                value["scoring_atom_indices"],
                "entity scoring_atom_indices",
                allow_empty=True,
            ),
            residues=tuple(
                ResidueIdentity.from_mapping(
                    _mapping(item, "entity residue")
                )
                for item in _sequence(value["residues"], "entity residues")
            ),
            named_selections={
                name: EntitySelection.from_mapping(
                    _mapping(selection, f"named selection {name!r}")
                )
                for name, selection in raw_selections.items()
            },
            selection_signature=str(value["selection_signature"]),
            features=tuple(
                EntityFeature.from_mapping(
                    _mapping(item, "entity feature")
                )
                for item in _sequence(
                    value.get("features", ()),
                    "entity features",
                )
            ),
            metadata=_mapping(
                value.get("metadata", {}),
                "entity metadata",
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "md_atom_indices": list(self.md_atom_indices),
            "scoring_atom_indices": list(self.scoring_atom_indices),
            "residues": [item.as_dict() for item in self.residues],
            "named_selections": {
                name: selection.as_dict()
                for name, selection in self.named_selections.items()
            },
            "selection_signature": self.selection_signature,
            "features": [item.as_dict() for item in self.features],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CrossEntityBond:
    """A covalent bond whose endpoints belong to different entities."""

    entity_a_id: str
    md_atom_a: int
    entity_b_id: str
    md_atom_b: int

    def __post_init__(self) -> None:
        require_identifier(self.entity_a_id, "bond entity_a_id")
        require_identifier(self.entity_b_id, "bond entity_b_id")
        if self.entity_a_id == self.entity_b_id:
            raise ContractError(
                "cross-entity bond endpoints must use different entities"
            )
        _positive_int(self.md_atom_a, "bond md_atom_a")
        _positive_int(self.md_atom_b, "bond md_atom_b")

    @property
    def canonical_key(self) -> tuple[tuple[str, int], tuple[str, int]]:
        return tuple(
            sorted(
                (
                    (self.entity_a_id, self.md_atom_a),
                    (self.entity_b_id, self.md_atom_b),
                )
            )
        )  # type: ignore[return-value]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CrossEntityBond:
        _strict_keys(
            value,
            required={
                "entity_a_id",
                "md_atom_a",
                "entity_b_id",
                "md_atom_b",
            },
            label="cross-entity bond",
        )
        return cls(
            entity_a_id=str(value["entity_a_id"]),
            md_atom_a=_positive_int(value["md_atom_a"], "bond md_atom_a"),
            entity_b_id=str(value["entity_b_id"]),
            md_atom_b=_positive_int(value["md_atom_b"], "bond md_atom_b"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "entity_a_id": self.entity_a_id,
            "md_atom_a": self.md_atom_a,
            "entity_b_id": self.entity_b_id,
            "md_atom_b": self.md_atom_b,
        }


@dataclass(frozen=True, slots=True)
class BindingTopologyRoles:
    """Artifact roles for the three dry-scoring endpoint topologies."""

    complex: str
    receptor: str
    ligand: str

    def __post_init__(self) -> None:
        roles = (self.complex, self.receptor, self.ligand)
        for role in roles:
            require_identifier(role, "binding topology artifact role")
        if len(set(roles)) != len(roles):
            raise ContractError(
                "binding topology artifact roles must be distinct"
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> BindingTopologyRoles:
        _strict_keys(
            value,
            required={"complex", "receptor", "ligand"},
            label="binding topology artifact roles",
        )
        return cls(
            complex=str(value["complex"]),
            receptor=str(value["receptor"]),
            ligand=str(value["ligand"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "complex": self.complex,
            "receptor": self.receptor,
            "ligand": self.ligand,
        }


@dataclass(frozen=True, slots=True)
class PrimaryBinding:
    """The sole thermodynamic receptor/ligand interpretation of a case."""

    binding_id: str
    receptor_entity_ids: tuple[str, ...]
    ligand_entity_ids: tuple[str, ...]
    excluded_entity_ids: tuple[str, ...]
    topology_artifact_roles: BindingTopologyRoles
    cross_entity_bonds: tuple[CrossEntityBond, ...] = ()

    def __post_init__(self) -> None:
        if self.binding_id != "primary":
            raise ContractError("the sole binding_id must be 'primary'")
        for label, values, allow_empty in (
            ("receptor_entity_ids", self.receptor_entity_ids, False),
            ("ligand_entity_ids", self.ligand_entity_ids, False),
            ("excluded_entity_ids", self.excluded_entity_ids, True),
        ):
            if not values and not allow_empty:
                raise ContractError(f"{label} cannot be empty")
            for value in values:
                require_identifier(value, f"{label} item")
            if tuple(sorted(set(values))) != values:
                raise ContractError(f"{label} must be sorted and unique")
        all_ids = (
            self.receptor_entity_ids
            + self.ligand_entity_ids
            + self.excluded_entity_ids
        )
        if len(all_ids) != len(set(all_ids)):
            raise ContractError("primary binding entity roles overlap")
        bond_keys = tuple(item.canonical_key for item in self.cross_entity_bonds)
        if len(bond_keys) != len(set(bond_keys)):
            raise ContractError("primary binding cross_entity_bonds repeat")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PrimaryBinding:
        _strict_keys(
            value,
            required={
                "binding_id",
                "receptor_entity_ids",
                "ligand_entity_ids",
                "excluded_entity_ids",
                "topology_artifact_roles",
            },
            optional={"cross_entity_bonds"},
            label="primary binding",
        )
        return cls(
            binding_id=str(value["binding_id"]),
            receptor_entity_ids=_identifiers(
                value["receptor_entity_ids"],
                "receptor_entity_ids",
                allow_empty=False,
            ),
            ligand_entity_ids=_identifiers(
                value["ligand_entity_ids"],
                "ligand_entity_ids",
                allow_empty=False,
            ),
            excluded_entity_ids=_identifiers(
                value["excluded_entity_ids"],
                "excluded_entity_ids",
                allow_empty=True,
            ),
            topology_artifact_roles=BindingTopologyRoles.from_mapping(
                _mapping(
                    value["topology_artifact_roles"],
                    "binding topology artifact roles",
                )
            ),
            cross_entity_bonds=tuple(
                CrossEntityBond.from_mapping(
                    _mapping(item, "cross-entity bond")
                )
                for item in _sequence(
                    value.get("cross_entity_bonds", ()),
                    "cross_entity_bonds",
                )
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "receptor_entity_ids": list(self.receptor_entity_ids),
            "ligand_entity_ids": list(self.ligand_entity_ids),
            "excluded_entity_ids": list(self.excluded_entity_ids),
            "topology_artifact_roles": self.topology_artifact_roles.as_dict(),
            "cross_entity_bonds": [
                item.as_dict() for item in self.cross_entity_bonds
            ],
        }


@dataclass(frozen=True, slots=True)
class PreparedMdSystem:
    """Role-neutral physical MD system."""

    backend: str
    atom_count: int
    system_signature: str
    artifacts: tuple[ArtifactRef, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier(self.backend, "MD backend")
        if self.atom_count < 1:
            raise ContractError("MD atom_count must be positive")
        if not _SHA256.fullmatch(self.system_signature):
            raise ContractError("MD system_signature must be a SHA-256 digest")
        if not self.artifacts:
            raise ContractError("MD artifacts cannot be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PreparedMdSystem:
        _strict_keys(
            value,
            required={"backend", "atom_count", "system_signature", "artifacts"},
            optional={"metadata"},
            label="prepared MD system",
        )
        return cls(
            backend=str(value["backend"]),
            atom_count=_positive_int(value["atom_count"], "MD atom_count"),
            system_signature=str(value["system_signature"]),
            artifacts=_artifact_inventory(value["artifacts"], "MD artifacts"),
            metadata=_mapping(value.get("metadata", {}), "MD metadata"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "atom_count": self.atom_count,
            "system_signature": self.system_signature,
            "artifacts": [item.as_dict() for item in self.artifacts],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PreparedScoringSystem:
    """Dry scoring atom order and its exact projection from MD."""

    backend: str
    atom_count: int
    atom_order_signature: str
    source_md_atom_indices: tuple[int, ...]
    radii_model: str
    artifacts: tuple[ArtifactRef, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier(self.backend, "scoring backend")
        if self.atom_count < 1:
            raise ContractError("scoring atom_count must be positive")
        if not _SHA256.fullmatch(self.atom_order_signature):
            raise ContractError(
                "scoring atom_order_signature must be a SHA-256 digest"
            )
        if len(self.source_md_atom_indices) != self.atom_count:
            raise ContractError(
                "source_md_atom_indices length must equal scoring atom_count"
            )
        if (
            any(
                isinstance(index, bool) or index < 1
                for index in self.source_md_atom_indices
            )
            or len(set(self.source_md_atom_indices))
            != len(self.source_md_atom_indices)
        ):
            raise ContractError(
                "source_md_atom_indices must be positive and unique"
            )
        if not self.radii_model.strip():
            raise ContractError("radii_model cannot be empty")
        observed_roles = {item.role for item in self.artifacts}
        if "coordinates" not in observed_roles:
            raise ContractError(
                "scoring artifacts are missing role: coordinates"
            )

    def mapped_md_atoms(
        self, scoring_atom_indices: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.source_md_atom_indices[index - 1]
                for index in scoring_atom_indices
            )
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> PreparedScoringSystem:
        _strict_keys(
            value,
            required={
                "backend",
                "atom_count",
                "atom_order_signature",
                "source_md_atom_indices",
                "radii_model",
                "artifacts",
            },
            optional={"metadata"},
            label="prepared scoring system",
        )
        return cls(
            backend=str(value["backend"]),
            atom_count=_positive_int(
                value["atom_count"], "scoring atom_count"
            ),
            atom_order_signature=str(value["atom_order_signature"]),
            source_md_atom_indices=tuple(
                _positive_int(item, "source MD atom index")
                for item in _sequence(
                    value["source_md_atom_indices"],
                    "source_md_atom_indices",
                )
            ),
            radii_model=str(value["radii_model"]),
            artifacts=_artifact_inventory(
                value["artifacts"], "scoring artifacts"
            ),
            metadata=_mapping(value.get("metadata", {}), "scoring metadata"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "atom_count": self.atom_count,
            "atom_order_signature": self.atom_order_signature,
            "source_md_atom_indices": list(self.source_md_atom_indices),
            "radii_model": self.radii_model,
            "artifacts": [item.as_dict() for item in self.artifacts],
            "metadata": dict(self.metadata),
        }


def _binding_selection(
    entities: tuple[PreparedEntity, ...],
    binding: PrimaryBinding,
    name: str,
) -> EntitySelection:
    if name not in BINDING_SELECTION_NAMES:
        raise KeyError(f"unknown primary binding selection {name!r}")
    entity_by_id = {item.entity_id: item for item in entities}
    selected_entities = tuple(
        entity_by_id[item]
        for item in (
            binding.receptor_entity_ids
            if name.startswith("receptor_")
            else binding.ligand_entity_ids
        )
    )
    if name in {"receptor_all", "ligand_all"}:
        selections = tuple(
            EntitySelection(
                md_atom_indices=entity.md_atom_indices,
                scoring_atom_indices=entity.scoring_atom_indices,
            )
            for entity in selected_entities
        )
    elif name in {"receptor_heavy", "ligand_heavy"}:
        selections = tuple(
            entity.named_selections["heavy_atoms"]
            for entity in selected_entities
        )
    elif name == "ligand_qc":
        selections = tuple(
            entity.named_selections[
                "heavy_atoms" if entity.kind == "small_molecule" else "backbone"
            ]
            for entity in selected_entities
        )
    else:
        selections = tuple(
            entity.named_selections["interface_backbone"]
            for entity in selected_entities
            if "interface_backbone" in entity.named_selections
        )
        if not selections:
            raise ContractError(
                "primary receptor must declare a non-empty interface_backbone selection"
            )
    return EntitySelection(
        md_atom_indices=tuple(
            sorted(
                atom
                for selection in selections
                for atom in selection.md_atom_indices
            )
        ),
        scoring_atom_indices=tuple(
            sorted(
                atom
                for selection in selections
                for atom in selection.scoring_atom_indices
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class PreparedCaseManifest:
    case_id: str
    md: PreparedMdSystem
    scoring: PreparedScoringSystem
    entities: tuple[PreparedEntity, ...]
    primary_binding: PrimaryBinding
    provenance: Mapping[str, object]
    schema_version: str = PREPARED_CASE_SCHEMA

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        if self.schema_version != PREPARED_CASE_SCHEMA:
            raise ContractError(
                f"unsupported PreparedCase schema: {self.schema_version!r}"
            )
        if max(self.scoring.source_md_atom_indices) > self.md.atom_count:
            raise ContractError(
                "scoring source atom mapping exceeds MD atom count"
            )
        if not self.entities:
            raise ContractError("PreparedCase entities cannot be empty")
        entity_by_id = {item.entity_id: item for item in self.entities}
        if len(entity_by_id) != len(self.entities):
            raise ContractError("PreparedCase entity_ids must be unique")

        md_atoms: list[int] = []
        scoring_atoms: list[int] = []
        md_residues: list[int] = []
        scoring_residues: list[int] = []
        for entity in self.entities:
            if max(entity.md_atom_indices) > self.md.atom_count:
                raise ContractError(
                    f"entity {entity.entity_id!r} exceeds MD atom count"
                )
            if (
                entity.scoring_atom_indices
                and max(entity.scoring_atom_indices)
                > self.scoring.atom_count
            ):
                raise ContractError(
                    f"entity {entity.entity_id!r} exceeds scoring atom count"
                )
            md_atoms.extend(entity.md_atom_indices)
            scoring_atoms.extend(entity.scoring_atom_indices)
            md_residues.extend(
                item.md_residue_index for item in entity.residues
            )
            scoring_residues.extend(
                item.scoring_residue_index
                for item in entity.residues
                if item.scoring_residue_index is not None
            )
            self._validate_entity_projection(entity)

        if len(md_atoms) != len(set(md_atoms)):
            raise ContractError("entity MD atom inventories overlap")
        if len(scoring_atoms) != len(set(scoring_atoms)):
            raise ContractError("entity scoring atom inventories overlap")
        if len(md_residues) != len(set(md_residues)):
            raise ContractError("entity MD residue inventories overlap")
        if len(scoring_residues) != len(set(scoring_residues)):
            raise ContractError("entity scoring residue inventories overlap")

        binding = self.primary_binding
        role_ids = (
            binding.receptor_entity_ids
            + binding.ligand_entity_ids
            + binding.excluded_entity_ids
        )
        unknown = set(role_ids).difference(entity_by_id)
        if unknown:
            raise ContractError(
                "primary binding references unknown entities: "
                + ", ".join(sorted(unknown))
            )
        missing = set(entity_by_id).difference(role_ids)
        if missing:
            raise ContractError(
                "entities must be assigned to receptor, ligand, or excluded: "
                + ", ".join(sorted(missing))
            )
        for entity_id in (
            binding.receptor_entity_ids + binding.ligand_entity_ids
        ):
            if not entity_by_id[entity_id].scoring_atom_indices:
                raise ContractError(
                    f"binding endpoint entity {entity_id!r} has no scoring atoms"
                )
        for entity_id in binding.excluded_entity_ids:
            if entity_by_id[entity_id].scoring_atom_indices:
                raise ContractError(
                    f"excluded entity {entity_id!r} has scoring atoms"
                )
        invalid_ligands = tuple(
            entity_id
            for entity_id in binding.ligand_entity_ids
            if entity_by_id[entity_id].kind not in LIGAND_ENTITY_KINDS
        )
        if invalid_ligands:
            raise ContractError(
                "ligand entities must be small_molecule or peptide: "
                + ", ".join(invalid_ligands)
            )

        endpoint_atoms = sorted(
            atom
            for entity_id in (
                binding.receptor_entity_ids + binding.ligand_entity_ids
            )
            for atom in entity_by_id[entity_id].scoring_atom_indices
        )
        if endpoint_atoms != list(range(1, self.scoring.atom_count + 1)):
            raise ContractError(
                "receptor and ligand entities must cover every dry scoring "
                "atom exactly once"
            )

        scoring_roles = {item.role for item in self.scoring.artifacts}
        topology_roles = binding.topology_artifact_roles
        required_topologies = {
            topology_roles.complex,
            topology_roles.receptor,
            topology_roles.ligand,
        }
        missing_topologies = required_topologies.difference(scoring_roles)
        if missing_topologies:
            raise ContractError(
                "primary binding topology artifacts are missing roles: "
                + ", ".join(sorted(missing_topologies))
            )
        self.binding_selection("receptor_interface_backbone")
        self._validate_cross_entity_bonds(entity_by_id)

    def _validate_entity_projection(self, entity: PreparedEntity) -> None:
        mapped_entity_atoms = self.scoring.mapped_md_atoms(
            entity.scoring_atom_indices
        )
        if entity.scoring_atom_indices:
            if mapped_entity_atoms != entity.md_atom_indices:
                raise ContractError(
                    f"entity {entity.entity_id!r} MD/scoring projections differ"
                )
        for name, selection in entity.named_selections.items():
            mapped_atoms = self.scoring.mapped_md_atoms(
                selection.scoring_atom_indices
            )
            if entity.scoring_atom_indices:
                if mapped_atoms != selection.md_atom_indices:
                    raise ContractError(
                        f"entity {entity.entity_id!r} named selection "
                        f"{name!r} has inconsistent MD/scoring projections"
                    )
            elif selection.scoring_atom_indices:
                raise ContractError(
                    f"MD-only entity {entity.entity_id!r} selection "
                    f"{name!r} has scoring atoms"
                )
        for feature in entity.features:
            mapped_atoms = self.scoring.mapped_md_atoms(
                feature.selection.scoring_atom_indices
            )
            if mapped_atoms != feature.selection.md_atom_indices:
                raise ContractError(
                    f"entity {entity.entity_id!r} feature "
                    f"{feature.feature_id!r} has inconsistent "
                    "MD/scoring projections"
                )

    def _validate_cross_entity_bonds(
        self, entity_by_id: Mapping[str, PreparedEntity]
    ) -> None:
        receptor_ids = set(self.primary_binding.receptor_entity_ids)
        ligand_ids = set(self.primary_binding.ligand_entity_ids)
        for bond in self.primary_binding.cross_entity_bonds:
            try:
                entity_a = entity_by_id[bond.entity_a_id]
                entity_b = entity_by_id[bond.entity_b_id]
            except KeyError as exc:
                raise ContractError(
                    "cross-entity bond references an unknown entity"
                ) from exc
            if bond.md_atom_a not in entity_a.md_atom_indices:
                raise ContractError(
                    f"cross-entity bond atom {bond.md_atom_a} does not belong "
                    f"to {bond.entity_a_id!r}"
                )
            if bond.md_atom_b not in entity_b.md_atom_indices:
                raise ContractError(
                    f"cross-entity bond atom {bond.md_atom_b} does not belong "
                    f"to {bond.entity_b_id!r}"
                )
            crosses_binding_boundary = (
                bond.entity_a_id in receptor_ids
                and bond.entity_b_id in ligand_ids
            ) or (
                bond.entity_b_id in receptor_ids
                and bond.entity_a_id in ligand_ids
            )
            if crosses_binding_boundary:
                raise ContractError(
                    "standard MM/PB(GB)SA cannot split a covalent bond across "
                    "the receptor-ligand boundary: "
                    f"{bond.entity_a_id}:{bond.md_atom_a}--"
                    f"{bond.entity_b_id}:{bond.md_atom_b}"
                )

    def entity(self, entity_id: str) -> PreparedEntity:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        raise KeyError(f"PreparedCase has no entity {entity_id!r}")

    def binding_selection(self, name: str) -> EntitySelection:
        """Resolve one non-redundant primary-binding selection placeholder."""

        return _binding_selection(self.entities, self.primary_binding, name)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> PreparedCaseManifest:
        _strict_keys(
            value,
            required={
                "schema_version",
                "case_id",
                "md",
                "scoring",
                "entities",
                "primary_binding",
                "provenance",
            },
            label="PreparedCase manifest",
        )
        return cls(
            schema_version=str(value["schema_version"]),
            case_id=str(value["case_id"]),
            md=PreparedMdSystem.from_mapping(
                _mapping(value["md"], "prepared MD system")
            ),
            scoring=PreparedScoringSystem.from_mapping(
                _mapping(value["scoring"], "prepared scoring system")
            ),
            entities=tuple(
                PreparedEntity.from_mapping(
                    _mapping(item, "prepared entity")
                )
                for item in _sequence(value["entities"], "prepared entities")
            ),
            primary_binding=PrimaryBinding.from_mapping(
                _mapping(value["primary_binding"], "primary binding")
            ),
            provenance=_mapping(value["provenance"], "provenance"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "md": self.md.as_dict(),
            "scoring": self.scoring.as_dict(),
            "entities": [item.as_dict() for item in self.entities],
            "primary_binding": self.primary_binding.as_dict(),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class PreparedMdView:
    """The role-neutral projection passed to MD execution."""

    case_id: str
    system: PreparedMdSystem
    resolved_artifacts: Mapping[str, Path]
    input_format: str = "gromacs"
    preflight: AmberPairValidation | None = None

    def __post_init__(self) -> None:
        system_roles = {item.role for item in self.system.artifacts}
        resolved_roles = set(self.resolved_artifacts)
        if resolved_roles != system_roles:
            raise ContractError(
                "Prepared MD view resolved artifacts must match selected system artifacts"
            )
        if self.input_format == "amber":
            expected_roles = _AMBER_MD_EXECUTION_ARTIFACT_ROLES
            if system_roles != expected_roles:
                raise ContractError(
                    "Amber Prepared MD view must contain exactly solvated "
                    "topology and coordinates"
                )
            if self.preflight is not None and not isinstance(
                self.preflight, AmberPairValidation
            ):
                raise ContractError("Amber Prepared MD preflight has an invalid result")
        elif self.input_format == "gromacs":
            if self.preflight is not None:
                raise ContractError("GROMACS Prepared MD view cannot carry Amber preflight")
            expected_roles = _MD_EXECUTION_ARTIFACT_ROLES
            if not expected_roles.issubset(system_roles):
                raise ContractError(
                    "GROMACS Prepared MD view requires MD topology and coordinates"
                )
            refs_by_role = {item.role: item for item in self.system.artifacts}
            refs_by_path = {item.path: item for item in self.system.artifacts}
            if len(refs_by_path) != len(self.system.artifacts):
                raise ContractError("GROMACS MD view artifact paths must be unique")
            closure = topology_include_closure(
                refs_by_role["md_topology"].path,
                {
                    ref.path: self.resolved_artifacts[ref.role]
                    for ref in self.system.artifacts
                },
            )
            closure_roles = {refs_by_path[path].role for path in closure}
            if system_roles != expected_roles | closure_roles:
                raise ContractError(
                    "GROMACS Prepared MD view exposes an artifact outside its "
                    "declared topology include closure"
                )
        else:
            raise ContractError(f"unsupported Prepared MD input format: {self.input_format}")
        if self.system.metadata:
            raise ContractError(
                "Prepared MD view cannot expose preparation or entity metadata"
            )

    def artifact(self, role: str) -> Path:
        try:
            return self.resolved_artifacts[role]
        except KeyError as exc:
            raise KeyError(f"Prepared MD view has no artifact role {role!r}") from exc

    @property
    def preflight_warning_count(self) -> int:
        return (
            0
            if self.preflight is None
            else self.preflight.suppressed_pair_function_warnings
        )


@dataclass(frozen=True, slots=True)
class PreparedTrajectoryView:
    """Post-MD projection combining physical-system and binding identities.

    Unlike :class:`PreparedMdView`, this view is only exposed after dynamics.
    It therefore carries the explicit scoring-to-MD atom projection and binding
    selections needed to construct a dry, scoring-ordered trajectory without
    leaking receptor or ligand semantics into the MD engine.
    """

    case_id: str
    source_md_atom_indices: tuple[int, ...]
    atom_order_signature: str
    residue_crosswalk: tuple[tuple[int, int, str], ...]
    complex_topology_role: str
    binding_selections: Mapping[str, EntitySelection]
    resolved_artifacts: Mapping[str, Path]

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        if (
            not self.source_md_atom_indices
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 1
                for index in self.source_md_atom_indices
            )
            or len(set(self.source_md_atom_indices))
            != len(self.source_md_atom_indices)
        ):
            raise ContractError(
                "Prepared trajectory view requires unique positive MD atom indices"
            )
        if (
            not isinstance(self.atom_order_signature, str)
            or _SHA256.fullmatch(self.atom_order_signature) is None
        ):
            raise ContractError(
                "Prepared trajectory view atom_order_signature must be SHA-256"
            )
        if not self.residue_crosswalk or any(
            not isinstance(item, tuple) or len(item) != 3
            for item in self.residue_crosswalk
        ):
            raise ContractError(
                "Prepared trajectory view residue_crosswalk must contain triples"
            )
        if any(
            isinstance(md_index, bool)
            or not isinstance(md_index, int)
            or md_index < 1
            or isinstance(scoring_index, bool)
            or not isinstance(scoring_index, int)
            or scoring_index < 1
            or not isinstance(name, str)
            or not name.strip()
            for md_index, scoring_index, name in self.residue_crosswalk
        ):
            raise ContractError(
                "Prepared trajectory view residue_crosswalk entries are invalid"
            )
        md_residues = tuple(item[0] for item in self.residue_crosswalk)
        scoring_residues = tuple(item[1] for item in self.residue_crosswalk)
        if (
            len(set(md_residues)) != len(md_residues)
            or len(set(scoring_residues)) != len(scoring_residues)
            or scoring_residues != tuple(sorted(scoring_residues))
        ):
            raise ContractError(
                "Prepared trajectory view residue_crosswalk must be unique "
                "and scoring-ordered"
            )
        if not isinstance(self.complex_topology_role, str):
            raise ContractError(
                "Prepared trajectory view complex topology role must be a string"
            )
        require_identifier(
            self.complex_topology_role,
            "Prepared trajectory view complex topology role",
        )
        if self.complex_topology_role in {
            "index",
            "solvated_topology",
            "md_topology",
            "md_coordinates",
        }:
            raise ContractError(
                "Prepared trajectory view complex topology role must be "
                "distinct from reserved trajectory artifact roles"
            )
        if (
            not isinstance(self.binding_selections, Mapping)
            or set(self.binding_selections) != set(BINDING_SELECTION_NAMES)
            or any(
                not isinstance(selection, EntitySelection)
                for selection in self.binding_selections.values()
            )
        ):
            raise ContractError(
                "Prepared trajectory view must expose every typed binding selection"
            )
        scoring_atom_count = len(self.source_md_atom_indices)
        if any(
            atom > scoring_atom_count
            for selection in self.binding_selections.values()
            for atom in selection.scoring_atom_indices
        ):
            raise ContractError(
                "Prepared trajectory view binding selection exceeds scoring atom count"
            )
        for name, selection in self.binding_selections.items():
            if (
                not selection.md_atom_indices
                or not selection.scoring_atom_indices
            ):
                raise ContractError(
                    f"Prepared trajectory view binding selection {name!r} "
                    "cannot be empty"
                )
            mapped_md_atoms = tuple(
                sorted(
                    self.source_md_atom_indices[index - 1]
                    for index in selection.scoring_atom_indices
                )
            )
            if mapped_md_atoms != selection.md_atom_indices:
                raise ContractError(
                    f"Prepared trajectory view binding selection {name!r} "
                    "has inconsistent MD/scoring projections"
                )
        if not isinstance(self.resolved_artifacts, Mapping) or any(
            not isinstance(role, str) or not isinstance(path, Path)
            for role, path in self.resolved_artifacts.items()
        ):
            raise ContractError(
                "Prepared trajectory view artifacts must map roles to paths"
            )
        required_roles = {
            "index",
            "md_topology",
            "md_coordinates",
            self.complex_topology_role,
        }
        exposed_roles = set(self.resolved_artifacts)
        if not required_roles.issubset(exposed_roles) or not exposed_roles.issubset(
            required_roles | {"solvated_topology"}
        ):
            raise ContractError(
                "Prepared trajectory view artifacts must be index, MD topology, "
                "MD coordinates, dry complex, and optional solvated topology"
            )

    def artifact(self, role: str) -> Path:
        try:
            return self.resolved_artifacts[role]
        except KeyError as exc:
            raise KeyError(
                f"Prepared trajectory view has no artifact role {role!r}"
            ) from exc

    def binding_selection(self, name: str) -> EntitySelection:
        try:
            return self.binding_selections[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown primary binding selection {name!r}"
            ) from exc

    def topology_artifact_role(self, component: str) -> str:
        if component == "complex":
            return self.complex_topology_role
        raise KeyError(f"unknown binding topology component {component!r}")

    def topology_artifact(self, component: str) -> Path:
        return self.artifact(self.topology_artifact_role(component))


@dataclass(frozen=True, slots=True)
class PreparedBindingView:
    """The explicit primary binding projection used after MD."""

    case_id: str
    scoring: PreparedScoringSystem
    entities: tuple[PreparedEntity, ...]
    binding: PrimaryBinding
    resolved_artifacts: Mapping[str, Path]

    def __post_init__(self) -> None:
        expected_roles = {
            *(item.role for item in self.scoring.artifacts),
            "index",
        }
        if set(self.resolved_artifacts) != expected_roles:
            raise ContractError(
                "Prepared binding view resolved artifacts must be exactly "
                "the scoring artifacts plus the MD index"
            )

    def entity(self, entity_id: str) -> PreparedEntity:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        raise KeyError(f"Prepared binding view has no entity {entity_id!r}")

    def artifact(self, role: str) -> Path:
        try:
            return self.resolved_artifacts[role]
        except KeyError as exc:
            raise KeyError(
                f"Prepared binding view has no artifact role {role!r}"
            ) from exc

    def binding_selection(self, name: str) -> EntitySelection:
        """Resolve a selection from the sole declared primary binding."""

        return _binding_selection(self.entities, self.binding, name)

    def topology_artifact_role(self, component: str) -> str:
        """Resolve one topology role solely from ``PrimaryBinding``."""

        roles = self.binding.topology_artifact_roles
        if component == "complex":
            return roles.complex
        if component == "receptor":
            return roles.receptor
        if component == "ligand":
            return roles.ligand
        raise KeyError(f"unknown binding topology component {component!r}")

    def topology_artifact(self, component: str) -> Path:
        return self.artifact(self.topology_artifact_role(component))


@dataclass(frozen=True, slots=True)
class PreparedCase:
    """A v2 manifest whose externally referenced artifacts were verified."""

    manifest_path: Path
    root: Path
    manifest: PreparedCaseManifest
    resolved_artifacts: Mapping[str, Path]

    @property
    def case_id(self) -> str:
        return self.manifest.case_id

    @property
    def md_view(self) -> PreparedMdView:
        return self.md_view_for_profile("gromacs_canonical@1")

    def md_view_for_profile(self, md_engine_profile: str) -> PreparedMdView:
        """Select only physical inputs consumed by the requested MD engine."""

        amber_input = md_engine_profile == "openmm_8_5@2"
        if md_engine_profile not in {
            "gromacs_canonical@1",
            "openmm_8_5@1",
            "openmm_8_5@2",
        }:
            raise ContractError(f"unsupported MD engine profile: {md_engine_profile}")
        roles = set(
            _AMBER_MD_EXECUTION_ARTIFACT_ROLES
            if amber_input
            else _MD_EXECUTION_ARTIFACT_ROLES
        )
        artifact_by_role = {item.role: item for item in self.manifest.md.artifacts}
        missing = roles.difference(artifact_by_role)
        if missing:
            raise ContractError(
                "PreparedCase is missing MD input artifact roles: "
                + ", ".join(sorted(missing))
            )
        topology = artifact_by_role.get("md_topology")
        if topology is None:
            raise ContractError("PreparedCase is missing md_topology for MD preflight")
        sealed_files = {
            item.path: self.resolved_artifacts[item.role]
            for item in self.manifest.md.artifacts
        }
        if len(sealed_files) != len(self.manifest.md.artifacts):
            raise ContractError("PreparedCase MD artifact paths must be unique")
        closure = topology_include_closure(topology.path, sealed_files)
        if not amber_input:
            by_path = {item.path: item for item in self.manifest.md.artifacts}
            roles.update(by_path[path].role for path in closure)
        preflight = None
        if amber_input:
            required_pair_roles = (
                "md_topology",
                "md_coordinates",
                "solvated_topology",
                "solvated_coordinates",
            )
            missing_pair = set(required_pair_roles).difference(artifact_by_role)
            if missing_pair:
                raise ContractError(
                    "OpenMM Amber input preflight requires GROMACS and Amber pairs: "
                    + ", ".join(sorted(missing_pair))
                )
            preflight = validate_amber_md_pair(
                *(
                    (
                        self.resolved_artifacts[role],
                        artifact_by_role[role].sha256,
                    )
                    for role in required_pair_roles
                )
            )
            if preflight.atom_count != self.manifest.md.atom_count:
                raise ContractError(
                    "OpenMM Amber atom count differs from PreparedCase MD atom count"
                )
        artifacts = tuple(
            item
            for item in self.manifest.md.artifacts
            if item.role in roles
        )
        return PreparedMdView(
            case_id=self.case_id,
            system=replace(
                self.manifest.md,
                artifacts=artifacts,
                metadata={},
            ),
            resolved_artifacts={
                role: path
                for role, path in self.resolved_artifacts.items()
                if role in roles
            },
            input_format="amber" if amber_input else "gromacs",
            preflight=preflight,
        )

    @property
    def trajectory_view(self) -> PreparedTrajectoryView:
        manifest = self.manifest
        binding = manifest.primary_binding
        endpoint_ids = set(
            binding.receptor_entity_ids + binding.ligand_entity_ids
        )
        residue_values = (
            (
                residue.md_residue_index,
                residue.scoring_residue_index,
                residue.residue_name,
            )
            for entity in manifest.entities
            if entity.entity_id in endpoint_ids
            for residue in entity.residues
        )
        residue_crosswalk = tuple(
            (md_index, scoring_index, name)
            for md_index, scoring_index, name in sorted(
                residue_values,
                key=lambda item: (
                    item[1] if item[1] is not None else 0,
                    item[0],
                ),
            )
            if scoring_index is not None
        )
        complex_role = binding.topology_artifact_roles.complex
        roles = {
            "index",
            "md_topology",
            "md_coordinates",
            "solvated_topology",
            complex_role,
        }
        return PreparedTrajectoryView(
            case_id=self.case_id,
            source_md_atom_indices=manifest.scoring.source_md_atom_indices,
            atom_order_signature=manifest.scoring.atom_order_signature,
            residue_crosswalk=residue_crosswalk,
            complex_topology_role=complex_role,
            binding_selections={
                name: manifest.binding_selection(name)
                for name in sorted(BINDING_SELECTION_NAMES)
            },
            resolved_artifacts={
                role: path
                for role, path in self.resolved_artifacts.items()
                if role in roles
            },
        )

    @property
    def binding_view(self) -> PreparedBindingView:
        roles = {
            *(item.role for item in self.manifest.scoring.artifacts),
            "index",
        }
        return PreparedBindingView(
            case_id=self.case_id,
            scoring=self.manifest.scoring,
            entities=self.manifest.entities,
            binding=self.manifest.primary_binding,
            resolved_artifacts={
                role: path
                for role, path in self.resolved_artifacts.items()
                if role in roles
            },
        )

    def artifact(self, role: str) -> Path:
        try:
            return self.resolved_artifacts[role]
        except KeyError as exc:
            raise KeyError(f"PreparedCase has no artifact role {role!r}") from exc


def _verify_artifact(root: Path, artifact: ArtifactRef) -> Path:
    try:
        path = resolve_within(root, artifact.path)
    except (OSError, ValueError) as exc:
        raise ArtifactIntegrityError(
            f"artifact path escapes PreparedCase root: {artifact.path}"
        ) from exc
    if not path.is_file():
        raise ArtifactIntegrityError(f"artifact is missing: {path}")
    size = path.stat().st_size
    if size != artifact.size:
        raise ArtifactIntegrityError(
            f"artifact size changed for {artifact.role}: "
            f"expected {artifact.size}, observed {size}"
        )
    observed = sha256_file(path)
    if observed != artifact.sha256:
        raise ArtifactIntegrityError(
            f"artifact digest changed for {artifact.role}: {path}"
        )
    return path


def load_prepared_case(
    path: Path, *, expected_case_id: str | None = None
) -> PreparedCase:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(
            f"PreparedCase manifest does not exist: {manifest_path}"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"cannot read PreparedCase manifest: {manifest_path}"
        ) from exc
    manifest = PreparedCaseManifest.from_mapping(
        _mapping(raw, "PreparedCase manifest")
    )
    if expected_case_id is not None and manifest.case_id != expected_case_id:
        raise ContractError(
            f"PreparedCase case_id {manifest.case_id!r} does not match "
            f"workflow case {expected_case_id!r}"
        )
    root = manifest_path.parent.resolve()
    resolved: dict[str, Path] = {}
    for artifact in (*manifest.md.artifacts, *manifest.scoring.artifacts):
        if artifact.role in resolved:
            raise ContractError(
                "duplicate artifact role across PreparedCase systems: "
                f"{artifact.role}"
            )
        resolved[artifact.role] = _verify_artifact(root, artifact)
    return PreparedCase(
        manifest_path=manifest_path,
        root=root,
        manifest=manifest,
        resolved_artifacts=resolved,
    )


def write_prepared_case_manifest(
    path: Path, manifest: PreparedCaseManifest
) -> None:
    """Write a manifest without copying any scientific input files."""

    from mmpbsa.utils import atomic_write_json

    atomic_write_json(Path(path), manifest.as_dict())

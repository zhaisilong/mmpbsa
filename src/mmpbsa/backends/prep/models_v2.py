"""Strict mmCIF preparation inputs for proteins and modified peptide ligands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import PreparationError
from .models import (
    _SHA256,
    SolvationSettings,
    SourceInput,
    _identifier,
    _mapping,
    _nonnegative_int,
    _positive_float,
    _sequence,
    _strict_keys,
)

PREPARATION_SCHEMA_V2 = "mmpbsa.preparation.v2"


def _source_input_v2(value: Mapping[str, object], *, config_path: Path) -> SourceInput:
    _strict_keys(value, required={"path", "sha256"}, label="source")
    relative = Path(str(value["path"]))
    if relative.is_absolute():
        raise PreparationError("source.path must be relative to the preparation file")
    resolved = (config_path.parent / relative).resolve()
    project_root = config_path.parent.parent.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise PreparationError("source.path must stay below the project root") from exc
    digest = str(value["sha256"])
    if not _SHA256.fullmatch(digest):
        raise PreparationError("source.sha256 must be a SHA-256 digest")
    return SourceInput(path=resolved, sha256=digest)


@dataclass(frozen=True, slots=True, order=True)
class ResidueRef:
    chain_id: str
    residue_number: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], label: str) -> ResidueRef:
        _strict_keys(value, required={"chain_id", "residue_number"}, label=label)
        chain = str(value["chain_id"])
        number = value["residue_number"]
        if not chain or len(chain) > 4:
            raise PreparationError(f"{label}.chain_id must contain 1-4 characters")
        if isinstance(number, bool) or not isinstance(number, int):
            raise PreparationError(f"{label}.residue_number must be an integer")
        return cls(chain, number)


@dataclass(frozen=True, slots=True)
class ResidueRange:
    chain_id: str
    start: int
    end: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], label: str) -> ResidueRange:
        _strict_keys(value, required={"chain_id", "start", "end"}, label=label)
        chain = str(value["chain_id"])
        start = value["start"]
        end = value["end"]
        if not chain or len(chain) > 4:
            raise PreparationError(f"{label}.chain_id must contain 1-4 characters")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end)):
            raise PreparationError(f"{label} bounds must be integers")
        if start > end:
            raise PreparationError(f"{label}.start must not exceed end")
        return cls(chain, start, end)


@dataclass(frozen=True, slots=True)
class CapTransform:
    source: ResidueRef
    expected_name: str
    prepared_name: str
    atom_renames: Mapping[str, str]
    retain_atoms: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CapTransform:
        _strict_keys(
            value,
            required={
                "source",
                "expected_name",
                "prepared_name",
                "atom_renames",
                "retain_atoms",
            },
            label="entities.receptor.cap",
        )
        renames = _mapping(value["atom_renames"], "receptor cap atom_renames")
        retained = tuple(str(item) for item in _sequence(value["retain_atoms"], "cap retain_atoms"))
        if not retained or len(retained) != len(set(retained)):
            raise PreparationError("cap retain_atoms must be non-empty and unique")
        return cls(
            source=ResidueRef.from_mapping(_mapping(value["source"], "cap source"), "cap source"),
            expected_name=str(value["expected_name"]),
            prepared_name=str(value["prepared_name"]),
            atom_renames={str(key): str(item) for key, item in renames.items()},
            retain_atoms=retained,
        )


@dataclass(frozen=True, slots=True)
class ReceptorInputV2:
    entity_id: str
    ranges: tuple[ResidueRange, ...]
    cap: CapTransform | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ReceptorInputV2:
        _strict_keys(
            value,
            required={"entity_id", "kind", "ranges"},
            optional={"cap"},
            label="entities.receptor",
        )
        if value["kind"] != "protein":
            raise PreparationError("entities.receptor.kind must be protein")
        ranges = tuple(
            ResidueRange.from_mapping(_mapping(item, "receptor range"), "receptor range")
            for item in _sequence(value["ranges"], "entities.receptor.ranges")
        )
        if not ranges:
            raise PreparationError("entities.receptor.ranges cannot be empty")
        raw_cap = value.get("cap")
        return cls(
            entity_id=_identifier(value["entity_id"], "receptor entity_id"),
            ranges=ranges,
            cap=None
            if raw_cap is None
            else CapTransform.from_mapping(_mapping(raw_cap, "receptor cap")),
        )


@dataclass(frozen=True, slots=True)
class ResidueTransform:
    source: ResidueRef
    expected_name: str
    prepared_name: str
    atom_renames: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], label: str) -> ResidueTransform:
        _strict_keys(
            value,
            required={"source", "expected_name", "prepared_name"},
            optional={"atom_renames"},
            label=label,
        )
        renames = _mapping(value.get("atom_renames", {}), f"{label}.atom_renames")
        return cls(
            source=ResidueRef.from_mapping(
                _mapping(value["source"], f"{label}.source"), f"{label}.source"
            ),
            expected_name=str(value["expected_name"]),
            prepared_name=str(value["prepared_name"]),
            atom_renames={str(key): str(item) for key, item in renames.items()},
        )


@dataclass(frozen=True, slots=True)
class BondEndpoint:
    residue: ResidueRef
    atom_name: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], label: str) -> BondEndpoint:
        _strict_keys(value, required={"chain_id", "residue_number", "atom_name"}, label=label)
        residue = ResidueRef.from_mapping(
            {"chain_id": value["chain_id"], "residue_number": value["residue_number"]},
            label,
        )
        atom = str(value["atom_name"])
        if not atom or len(atom) > 4:
            raise PreparationError(f"{label}.atom_name must contain 1-4 characters")
        return cls(residue=residue, atom_name=atom)


@dataclass(frozen=True, slots=True)
class IntraEntityBond:
    first: BondEndpoint
    second: BondEndpoint

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], label: str) -> IntraEntityBond:
        _strict_keys(value, required={"first", "second"}, label=label)
        first = BondEndpoint.from_mapping(
            _mapping(value["first"], f"{label}.first"), f"{label}.first"
        )
        second = BondEndpoint.from_mapping(
            _mapping(value["second"], f"{label}.second"), f"{label}.second"
        )
        if first == second:
            raise PreparationError(f"{label} endpoints must differ")
        return cls(first, second)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    feature_id: str
    kind: str
    residues: tuple[ResidueRef, ...]
    metadata: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], label: str) -> FeatureSpec:
        _strict_keys(
            value, required={"feature_id", "kind", "residues"}, optional={"metadata"}, label=label
        )
        residues = tuple(
            sorted(
                ResidueRef.from_mapping(_mapping(item, f"{label}.residue"), f"{label}.residue")
                for item in _sequence(value["residues"], f"{label}.residues")
            )
        )
        if not residues or len(residues) != len(set(residues)):
            raise PreparationError(f"{label}.residues must be non-empty and unique")
        metadata = _mapping(value.get("metadata", {}), f"{label}.metadata")
        return cls(
            feature_id=_identifier(value["feature_id"], f"{label}.feature_id"),
            kind=str(value["kind"]),
            residues=residues,
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class PeptideInputV2:
    entity_id: str
    ranges: tuple[ResidueRange, ...]
    transforms: tuple[ResidueTransform, ...]
    bonds: tuple[IntraEntityBond, ...]
    features: tuple[FeatureSpec, ...]
    formal_charge: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PeptideInputV2:
        _strict_keys(
            value,
            required={"entity_id", "kind", "ranges", "formal_charge"},
            optional={"residue_transforms", "bonds", "features"},
            label="entities.ligand",
        )
        if value["kind"] != "peptide":
            raise PreparationError("preparation v2 currently requires a peptide ligand")
        charge = value["formal_charge"]
        if isinstance(charge, bool) or not isinstance(charge, int):
            raise PreparationError("entities.ligand.formal_charge must be an integer")
        ranges = tuple(
            ResidueRange.from_mapping(_mapping(item, "ligand range"), "ligand range")
            for item in _sequence(value["ranges"], "entities.ligand.ranges")
        )
        if not ranges:
            raise PreparationError("entities.ligand.ranges cannot be empty")
        transforms = tuple(
            ResidueTransform.from_mapping(_mapping(item, "residue transform"), "residue transform")
            for item in _sequence(value.get("residue_transforms", ()), "ligand residue_transforms")
        )
        bonds = tuple(
            IntraEntityBond.from_mapping(_mapping(item, "ligand bond"), "ligand bond")
            for item in _sequence(value.get("bonds", ()), "ligand bonds")
        )
        features = tuple(
            FeatureSpec.from_mapping(_mapping(item, "ligand feature"), "ligand feature")
            for item in _sequence(value.get("features", ()), "ligand features")
        )
        return cls(
            entity_id=_identifier(value["entity_id"], "ligand entity_id"),
            ranges=ranges,
            transforms=transforms,
            bonds=bonds,
            features=features,
            formal_charge=charge,
        )


@dataclass(frozen=True, slots=True)
class ForceFieldSettingsV2:
    leaprcs: tuple[str, ...]
    water: str
    water_box: str
    pb_radii: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ForceFieldSettingsV2:
        _strict_keys(
            value, required={"leaprcs", "water", "water_box", "pb_radii"}, label="force_fields"
        )
        leaprcs = tuple(str(item) for item in _sequence(value["leaprcs"], "force_fields.leaprcs"))
        if not leaprcs or len(leaprcs) != len(set(leaprcs)) or any(not item for item in leaprcs):
            raise PreparationError("force_fields.leaprcs must be non-empty and unique")
        return cls(leaprcs, str(value["water"]), str(value["water_box"]), str(value["pb_radii"]))


@dataclass(frozen=True, slots=True)
class PreparationExpectationsV2:
    receptor_atoms: int
    receptor_residues: int
    ligand_atoms: int
    ligand_residues: int
    ligand_charge_e: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PreparationExpectationsV2:
        _strict_keys(
            value,
            required={
                "receptor_atoms",
                "receptor_residues",
                "ligand_atoms",
                "ligand_residues",
                "ligand_charge_e",
            },
            label="expectations",
        )
        charge = value["ligand_charge_e"]
        if isinstance(charge, bool) or not isinstance(charge, int):
            raise PreparationError("expectations.ligand_charge_e must be an integer")
        return cls(
            _nonnegative_int(value["receptor_atoms"], "receptor_atoms"),
            _nonnegative_int(value["receptor_residues"], "receptor_residues"),
            _nonnegative_int(value["ligand_atoms"], "ligand_atoms"),
            _nonnegative_int(value["ligand_residues"], "ligand_residues"),
            charge,
        )


@dataclass(frozen=True, slots=True)
class PreparationSpecV2:
    case_id: str
    source: SourceInput
    receptor: ReceptorInputV2
    ligand: PeptideInputV2
    force_fields: ForceFieldSettingsV2
    solvation: SolvationSettings
    interface_cutoff_angstrom: float
    expectations: PreparationExpectationsV2
    config_path: Path
    schema_version: str = PREPARATION_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.schema_version != PREPARATION_SCHEMA_V2:
            raise PreparationError(f"unsupported preparation schema: {self.schema_version!r}")
        if self.receptor.entity_id == self.ligand.entity_id:
            raise PreparationError("receptor and ligand entity_ids must differ")
        selected_ligand = {
            ResidueRef(item.chain_id, number)
            for item in self.ligand.ranges
            for number in range(item.start, item.end + 1)
        }
        referenced = {
            *(item.source for item in self.ligand.transforms),
            *(
                endpoint.residue
                for bond in self.ligand.bonds
                for endpoint in (bond.first, bond.second)
            ),
            *(residue for feature in self.ligand.features for residue in feature.residues),
        }
        unknown = referenced.difference(selected_ligand)
        if unknown:
            raise PreparationError(
                f"ligand transforms/features/bonds escape selected ranges: {sorted(unknown)!r}"
            )
        transform_refs = tuple(item.source for item in self.ligand.transforms)
        if len(transform_refs) != len(set(transform_refs)):
            raise PreparationError("ligand residue transforms repeat a source residue")
        feature_ids = tuple(item.feature_id for item in self.ligand.features)
        if len(feature_ids) != len(set(feature_ids)):
            raise PreparationError("ligand feature_ids must be unique")
        if self.ligand.formal_charge != self.expectations.ligand_charge_e:
            raise PreparationError("ligand formal charge and expected charge differ")

    def verify_sources(self) -> None:
        self.source.verify("mmCIF source")


def load_preparation_spec_v2(path: Path) -> PreparationSpecV2:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreparationError(f"cannot read preparation config: {config_path}") from exc
    document = _mapping(raw, "preparation document")
    _strict_keys(
        document,
        required={
            "schema_version",
            "case_id",
            "source",
            "entities",
            "force_fields",
            "solvation",
            "interface",
            "expectations",
        },
        label="preparation document",
    )
    if document["schema_version"] != PREPARATION_SCHEMA_V2:
        raise PreparationError(f"unsupported preparation schema: {document['schema_version']!r}")
    entities = _mapping(document["entities"], "entities")
    _strict_keys(entities, required={"receptor", "ligand"}, label="entities")
    interface = _mapping(document["interface"], "interface")
    _strict_keys(interface, required={"cutoff_angstrom"}, label="interface")
    return PreparationSpecV2(
        case_id=_identifier(document["case_id"], "case_id"),
        source=_source_input_v2(_mapping(document["source"], "source"), config_path=config_path),
        receptor=ReceptorInputV2.from_mapping(_mapping(entities["receptor"], "entities.receptor")),
        ligand=PeptideInputV2.from_mapping(_mapping(entities["ligand"], "entities.ligand")),
        force_fields=ForceFieldSettingsV2.from_mapping(
            _mapping(document["force_fields"], "force_fields")
        ),
        solvation=SolvationSettings.from_mapping(_mapping(document["solvation"], "solvation")),
        interface_cutoff_angstrom=_positive_float(
            interface["cutoff_angstrom"], "interface.cutoff_angstrom"
        ),
        expectations=PreparationExpectationsV2.from_mapping(
            _mapping(document["expectations"], "expectations")
        ),
        config_path=config_path,
    )


__all__ = ["PREPARATION_SCHEMA_V2", "PreparationSpecV2", "load_preparation_spec_v2"]

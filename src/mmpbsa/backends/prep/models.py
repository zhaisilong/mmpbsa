"""Strict, case-level inputs for reproducible small-molecule preparation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import yaml

from mmpbsa.utils import sha256_file

from .errors import PreparationError

if TYPE_CHECKING:
    from .models_v2 import PreparationSpecV2

PREPARATION_SCHEMA = "mmpbsa.preparation.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESIDUE_NAME = re.compile(r"^[A-Za-z0-9]{1,4}$")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise PreparationError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise PreparationError(f"{label} must be a sequence")
    return value


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
        raise PreparationError(
            f"{label} is missing: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise PreparationError(
            f"{label} contains unknown fields: "
            + ", ".join(sorted(unknown))
        )


def _identifier(value: object, label: str) -> str:
    result = str(value)
    if not _IDENTIFIER.fullmatch(result):
        raise PreparationError(f"{label} is not a safe identifier: {result!r}")
    return result


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreparationError(f"{label} must be numeric")
    result = float(value)
    if result <= 0.0:
        raise PreparationError(f"{label} must be positive")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreparationError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class SourceInput:
    path: Path
    sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        base_dir: Path,
        label: str,
    ) -> SourceInput:
        _strict_keys(value, required={"path", "sha256"}, label=label)
        relative = Path(str(value["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise PreparationError(f"{label}.path must stay below the case root")
        digest = str(value["sha256"])
        if not _SHA256.fullmatch(digest):
            raise PreparationError(f"{label}.sha256 must be a SHA-256 digest")
        return cls(path=(base_dir / relative).resolve(), sha256=digest)

    def verify(self, label: str) -> None:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise PreparationError(f"{label} is missing or empty: {self.path}")
        observed = sha256_file(self.path)
        if observed != self.sha256:
            raise PreparationError(
                f"{label} digest differs: expected {self.sha256}, "
                f"observed {observed}"
            )


@dataclass(frozen=True, slots=True)
class ReceptorInput:
    entity_id: str
    source: SourceInput

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, base_dir: Path
    ) -> ReceptorInput:
        _strict_keys(
            value,
            required={"entity_id", "kind", "source"},
            label="entities.receptor",
        )
        if value["kind"] != "protein":
            raise PreparationError("entities.receptor.kind must be protein")
        return cls(
            entity_id=_identifier(value["entity_id"], "receptor entity_id"),
            source=SourceInput.from_mapping(
                _mapping(value["source"], "receptor source"),
                base_dir=base_dir,
                label="entities.receptor.source",
            ),
        )


@dataclass(frozen=True, slots=True)
class LigandInput:
    entity_id: str
    source: SourceInput
    residue_name: str
    formal_charge: int
    charge_method: str
    atom_type: str
    force_field: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, base_dir: Path
    ) -> LigandInput:
        _strict_keys(
            value,
            required={
                "entity_id",
                "kind",
                "source",
                "residue_name",
                "formal_charge",
                "charge_method",
                "atom_type",
                "force_field",
            },
            label="entities.ligand",
        )
        if value["kind"] != "small_molecule":
            raise PreparationError(
                "this preparation adapter requires a small_molecule ligand"
            )
        residue_name = str(value["residue_name"])
        if not _RESIDUE_NAME.fullmatch(residue_name):
            raise PreparationError("ligand residue_name must contain 1-4 alphanumerics")
        charge = value["formal_charge"]
        if isinstance(charge, bool) or not isinstance(charge, int):
            raise PreparationError("ligand formal_charge must be an integer")
        return cls(
            entity_id=_identifier(value["entity_id"], "ligand entity_id"),
            source=SourceInput.from_mapping(
                _mapping(value["source"], "ligand source"),
                base_dir=base_dir,
                label="entities.ligand.source",
            ),
            residue_name=residue_name,
            formal_charge=charge,
            charge_method=_identifier(value["charge_method"], "charge_method"),
            atom_type=_identifier(value["atom_type"], "atom_type"),
            force_field=_identifier(value["force_field"], "force_field"),
        )


@dataclass(frozen=True, slots=True)
class CrystalWaterInput:
    entity_id: str
    chain_id: str
    residue_number: int
    insertion_code: str = ""
    residue_name: str = "HOH"

    @property
    def source_key(self) -> tuple[str, int, str, str]:
        return (
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.residue_name,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CrystalWaterInput:
        _strict_keys(
            value,
            required={"entity_id", "chain_id", "residue_number", "residue_name"},
            optional={"insertion_code"},
            label="crystal water",
        )
        chain_id = str(value["chain_id"])
        insertion_code = str(value.get("insertion_code", ""))
        if len(chain_id) != 1 or len(insertion_code) > 1:
            raise PreparationError("water chain_id/insertion_code must fit PDB fields")
        number = value["residue_number"]
        if isinstance(number, bool) or not isinstance(number, int):
            raise PreparationError("water residue_number must be an integer")
        residue_name = str(value["residue_name"]).upper()
        if not _RESIDUE_NAME.fullmatch(residue_name):
            raise PreparationError("water residue_name is invalid")
        return cls(
            entity_id=_identifier(value["entity_id"], "water entity_id"),
            chain_id=chain_id,
            residue_number=number,
            insertion_code=insertion_code,
            residue_name=residue_name,
        )


@dataclass(frozen=True, slots=True)
class ForceFieldSettings:
    protein: str
    water: str
    pb_radii: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ForceFieldSettings:
        _strict_keys(
            value,
            required={"protein", "water", "pb_radii"},
            label="force_fields",
        )
        return cls(
            protein=_identifier(value["protein"], "protein force field"),
            water=_identifier(value["water"], "water force field"),
            pb_radii=_identifier(value["pb_radii"], "PB radii"),
        )


@dataclass(frozen=True, slots=True)
class SolvationSettings:
    shape: str
    padding_angstrom: float
    salt_concentration_molar: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SolvationSettings:
        _strict_keys(
            value,
            required={"shape", "padding_angstrom", "salt_concentration_molar"},
            label="solvation",
        )
        if value["shape"] != "truncated_octahedron":
            raise PreparationError("only truncated_octahedron solvation is supported")
        return cls(
            shape="truncated_octahedron",
            padding_angstrom=_positive_float(
                value["padding_angstrom"], "solvation.padding_angstrom"
            ),
            salt_concentration_molar=_positive_float(
                value["salt_concentration_molar"],
                "solvation.salt_concentration_molar",
            ),
        )


@dataclass(frozen=True, slots=True)
class PreparationExpectations:
    receptor_protein_atoms: int
    receptor_protein_residues: int
    ligand_atoms: int
    crystal_water_atoms: int
    histidine_states: Mapping[str, int]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> PreparationExpectations:
        _strict_keys(
            value,
            required={
                "receptor_protein_atoms",
                "receptor_protein_residues",
                "ligand_atoms",
                "crystal_water_atoms",
                "histidine_states",
            },
            label="expectations",
        )
        states_raw = _mapping(value["histidine_states"], "histidine_states")
        states = {
            str(key): _nonnegative_int(item, f"histidine_states.{key}")
            for key, item in states_raw.items()
        }
        return cls(
            receptor_protein_atoms=_nonnegative_int(
                value["receptor_protein_atoms"], "receptor_protein_atoms"
            ),
            receptor_protein_residues=_nonnegative_int(
                value["receptor_protein_residues"], "receptor_protein_residues"
            ),
            ligand_atoms=_nonnegative_int(value["ligand_atoms"], "ligand_atoms"),
            crystal_water_atoms=_nonnegative_int(
                value["crystal_water_atoms"], "crystal_water_atoms"
            ),
            histidine_states=states,
        )


@dataclass(frozen=True, slots=True)
class PreparationSpec:
    case_id: str
    receptor: ReceptorInput
    ligand: LigandInput
    crystal_waters: tuple[CrystalWaterInput, ...]
    force_fields: ForceFieldSettings
    solvation: SolvationSettings
    interface_cutoff_angstrom: float
    expectations: PreparationExpectations
    config_path: Path
    schema_version: str = PREPARATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PREPARATION_SCHEMA:
            raise PreparationError(
                f"unsupported preparation schema: {self.schema_version!r}"
            )
        ids = (
            self.receptor.entity_id,
            self.ligand.entity_id,
            *(item.entity_id for item in self.crystal_waters),
        )
        if len(ids) != len(set(ids)):
            raise PreparationError("preparation entity_ids must be unique")
        water_keys = tuple(item.source_key for item in self.crystal_waters)
        if len(water_keys) != len(set(water_keys)):
            raise PreparationError("crystal water source identities must be unique")
        if not self.crystal_waters:
            raise PreparationError(
                "explicit crystal_waters cannot be empty; declare retained waters"
            )

    def verify_sources(self) -> None:
        self.receptor.source.verify("receptor source")
        self.ligand.source.verify("ligand source")


PreparationIntent: TypeAlias = "PreparationSpec | PreparationSpecV2"


def load_preparation_spec(path: Path) -> PreparationIntent:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreparationError(f"cannot read preparation config: {config_path}") from exc
    document = _mapping(raw, "preparation document")
    if document.get("schema_version") == "mmpbsa.preparation.v2":
        from .models_v2 import load_preparation_spec_v2

        return load_preparation_spec_v2(config_path)
    _strict_keys(
        document,
        required={
            "schema_version",
            "case_id",
            "entities",
            "force_fields",
            "solvation",
            "interface",
            "expectations",
        },
        label="preparation document",
    )
    entities = _mapping(document["entities"], "entities")
    _strict_keys(
        entities,
        required={"receptor", "ligand", "crystal_waters"},
        label="entities",
    )
    interface = _mapping(document["interface"], "interface")
    _strict_keys(
        interface, required={"cutoff_angstrom"}, label="interface"
    )
    base_dir = config_path.parent
    return PreparationSpec(
        schema_version=str(document["schema_version"]),
        case_id=_identifier(document["case_id"], "case_id"),
        receptor=ReceptorInput.from_mapping(
            _mapping(entities["receptor"], "entities.receptor"),
            base_dir=base_dir,
        ),
        ligand=LigandInput.from_mapping(
            _mapping(entities["ligand"], "entities.ligand"),
            base_dir=base_dir,
        ),
        crystal_waters=tuple(
            CrystalWaterInput.from_mapping(_mapping(item, "crystal water"))
            for item in _sequence(
                entities["crystal_waters"], "entities.crystal_waters"
            )
        ),
        force_fields=ForceFieldSettings.from_mapping(
            _mapping(document["force_fields"], "force_fields")
        ),
        solvation=SolvationSettings.from_mapping(
            _mapping(document["solvation"], "solvation")
        ),
        interface_cutoff_angstrom=_positive_float(
            interface["cutoff_angstrom"], "interface.cutoff_angstrom"
        ),
        expectations=PreparationExpectations.from_mapping(
            _mapping(document["expectations"], "expectations")
        ),
        config_path=config_path,
    )


__all__ = [
    "CrystalWaterInput",
    "ForceFieldSettings",
    "LigandInput",
    "PREPARATION_SCHEMA",
    "PreparationExpectations",
    "PreparationIntent",
    "PreparationSpec",
    "ReceptorInput",
    "SolvationSettings",
    "SourceInput",
    "load_preparation_spec",
]

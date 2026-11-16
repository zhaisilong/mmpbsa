"""Strict mechanical wrapping of existing artifacts into PreparedCase v2.

The draft contract deliberately has no structure-preparation vocabulary.  It
accepts existing GROMACS MD and Amber scoring artifacts together with explicit
atom/entity semantics, verifies them mechanically, and seals immutable copies
into the already authoritative :mod:`mmpbsa.core.prepared` v2 contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path

import yaml

from mmpbsa.utils import (
    InterProcessFileLock,
    LockTimeoutError,
    atomic_directory,
    materialize_immutable_file,
    resolve_within,
    sha256_file,
)

from .errors import ArtifactIntegrityError, ContractError
from .md_inputs import topology_include_closure
from .models import ArtifactRef, require_identifier
from .prepared import (
    PREPARED_CASE_SCHEMA,
    PreparedCase,
    PreparedCaseManifest,
    PreparedEntity,
    PreparedMdSystem,
    PreparedScoringSystem,
    PrimaryBinding,
    load_prepared_case,
    write_prepared_case_manifest,
)
from .topology_projection import (
    binding_boundary_bond_issues,
    component_projection_issues,
    mapped_system_projection_issues,
    topology_signature,
)

PREPARED_CASE_DRAFT_SCHEMA = "mmpbsa.prepared_case_draft.v1"
_SEAL_PROVENANCE_KEY = "prepared_case_draft"
_REQUIRED_MD_ROLES = frozenset({"md_topology", "md_coordinates", "index", "preparation_receipt"})
_REQUIRED_SCORING_ROLES = frozenset(
    {"complex_topology", "receptor_topology", "ligand_topology", "coordinates"}
)
_INDEX_GROUPS = ("System", "Complex", "Receptor", "Ligand")
_STANDARD_TOPOLOGY_ROLES = {
    "complex": "complex_topology",
    "receptor": "receptor_topology",
    "ligand": "ligand_topology",
}
_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O"})


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
        raise ContractError(f"{label} contains unknown fields: " + ", ".join(sorted(unknown)))


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractError(f"{label} must be a sequence")
    return value


def _safe_relative_path(value: object, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ContractError(f"{label} must be a safe relative path")
    return path


def _positive_unique_indices(value: object, label: str) -> tuple[int, ...]:
    indices = tuple(_sequence(value, label))
    if not indices:
        raise ContractError(f"{label} cannot be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in indices):
        raise ContractError(f"{label} must contain positive integers")
    if len(indices) != len(set(indices)):
        raise ContractError(f"{label} must contain unique integers")
    return indices  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DraftArtifact:
    """One source file and its safe path in the sealed package."""

    role: str
    source: Path
    path: Path
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        require_identifier(self.role, "draft artifact role")
        object.__setattr__(
            self, "source", _safe_relative_path(self.source, "draft artifact source")
        )
        object.__setattr__(self, "path", _safe_relative_path(self.path, "draft artifact path"))
        if self.path == Path("prepared_case.json"):
            raise ContractError("draft artifact path is reserved: prepared_case.json")
        if not self.media_type.strip():
            raise ContractError("draft artifact media_type cannot be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DraftArtifact:
        _strict_keys(
            value,
            required={"role", "source", "path"},
            optional={"media_type"},
            label="draft artifact",
        )
        return cls(
            role=str(value["role"]),
            source=Path(str(value["source"])),
            path=Path(str(value["path"])),
            media_type=str(value.get("media_type") or "application/octet-stream"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "source": self.source.as_posix(),
            "path": self.path.as_posix(),
            "media_type": self.media_type,
        }


def _draft_artifacts(value: object, label: str) -> tuple[DraftArtifact, ...]:
    artifacts = tuple(
        DraftArtifact.from_mapping(_mapping(item, f"{label} artifact"))
        for item in _sequence(value, label)
    )
    if not artifacts:
        raise ContractError(f"{label} cannot be empty")
    roles = tuple(item.role for item in artifacts)
    sources = tuple(item.source for item in artifacts)
    targets = tuple(item.path for item in artifacts)
    if len(roles) != len(set(roles)):
        raise ContractError(f"{label} roles must be unique")
    if len(sources) != len(set(sources)):
        raise ContractError(f"{label} source paths must be unique")
    if len(targets) != len(set(targets)):
        raise ContractError(f"{label} publication paths must be unique")
    return artifacts


@dataclass(frozen=True, slots=True)
class PreparedMdDraft:
    backend: str
    artifacts: tuple[DraftArtifact, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend != "gromacs":
            raise ContractError("draft v1 MD backend must be 'gromacs'")
        roles = {item.role for item in self.artifacts}
        missing = _REQUIRED_MD_ROLES.difference(roles)
        if missing:
            raise ContractError(
                "draft MD artifacts are missing roles: " + ", ".join(sorted(missing))
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PreparedMdDraft:
        _strict_keys(
            value,
            required={"backend", "artifacts"},
            optional={"metadata"},
            label="draft MD system",
        )
        return cls(
            backend=str(value["backend"]),
            artifacts=_draft_artifacts(value["artifacts"], "draft MD artifacts"),
            metadata=dict(_mapping(value.get("metadata", {}), "draft MD metadata")),
        )


@dataclass(frozen=True, slots=True)
class PreparedScoringDraft:
    backend: str
    radii_model: str
    source_md_atom_indices: tuple[int, ...]
    artifacts: tuple[DraftArtifact, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend != "ambertools":
            raise ContractError("draft v1 scoring backend must be 'ambertools'")
        if not self.radii_model.strip():
            raise ContractError("draft scoring radii_model cannot be empty")
        roles = {item.role for item in self.artifacts}
        missing = _REQUIRED_SCORING_ROLES.difference(roles)
        if missing:
            raise ContractError(
                "draft scoring artifacts are missing roles: " + ", ".join(sorted(missing))
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PreparedScoringDraft:
        _strict_keys(
            value,
            required={
                "backend",
                "radii_model",
                "source_md_atom_indices",
                "artifacts",
            },
            optional={"metadata"},
            label="draft scoring system",
        )
        return cls(
            backend=str(value["backend"]),
            radii_model=str(value["radii_model"]),
            source_md_atom_indices=_positive_unique_indices(
                value["source_md_atom_indices"], "source_md_atom_indices"
            ),
            artifacts=_draft_artifacts(value["artifacts"], "draft scoring artifacts"),
            metadata=dict(_mapping(value.get("metadata", {}), "draft scoring metadata")),
        )


def _draft_entity(value: Mapping[str, object]) -> PreparedEntity:
    _strict_keys(
        value,
        required={
            "entity_id",
            "kind",
            "md_atom_indices",
            "scoring_atom_indices",
            "residues",
            "named_selections",
        },
        optional={"selection_signature", "features", "metadata"},
        label="draft entity",
    )
    normalized = dict(value)
    normalized.setdefault("selection_signature", "")
    return PreparedEntity.from_mapping(normalized)


@dataclass(frozen=True, slots=True)
class PreparedCaseDraftManifest:
    case_id: str
    md: PreparedMdDraft
    scoring: PreparedScoringDraft
    entities: tuple[PreparedEntity, ...]
    primary_binding: PrimaryBinding
    provenance: Mapping[str, object]
    schema_version: str = PREPARED_CASE_DRAFT_SCHEMA

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "draft case_id")
        if self.schema_version != PREPARED_CASE_DRAFT_SCHEMA:
            raise ContractError(f"unsupported PreparedCase draft schema: {self.schema_version!r}")
        if not self.entities:
            raise ContractError("draft entities cannot be empty")
        if not self.provenance:
            raise ContractError("draft provenance cannot be empty")
        if _SEAL_PROVENANCE_KEY in self.provenance:
            raise ContractError(f"draft provenance key {_SEAL_PROVENANCE_KEY!r} is reserved")
        observed_topology_roles = self.primary_binding.topology_artifact_roles.as_dict()
        if observed_topology_roles != _STANDARD_TOPOLOGY_ROLES:
            raise ContractError(
                "PreparedCase draft primary binding must use the standard "
                "topology artifact roles: complex_topology, receptor_topology, "
                "ligand_topology"
            )
        artifacts = (*self.md.artifacts, *self.scoring.artifacts)
        roles = tuple(item.role for item in artifacts)
        sources = tuple(item.source for item in artifacts)
        targets = tuple(item.path for item in artifacts)
        if len(roles) != len(set(roles)):
            raise ContractError("draft artifact roles must be unique across systems")
        if len(sources) != len(set(sources)):
            raise ContractError("draft artifact sources must be unique across systems")
        if len(targets) != len(set(targets)):
            raise ContractError("draft artifact publication paths must be unique across systems")

    @property
    def artifacts(self) -> tuple[DraftArtifact, ...]:
        return (*self.md.artifacts, *self.scoring.artifacts)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PreparedCaseDraftManifest:
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
            label="PreparedCase draft manifest",
        )
        return cls(
            schema_version=str(value["schema_version"]),
            case_id=str(value["case_id"]),
            md=PreparedMdDraft.from_mapping(_mapping(value["md"], "draft MD system")),
            scoring=PreparedScoringDraft.from_mapping(
                _mapping(value["scoring"], "draft scoring system")
            ),
            entities=tuple(
                _draft_entity(_mapping(item, "draft entity"))
                for item in _sequence(value["entities"], "draft entities")
            ),
            primary_binding=PrimaryBinding.from_mapping(
                _mapping(value["primary_binding"], "draft primary binding")
            ),
            provenance=dict(_mapping(value["provenance"], "draft provenance")),
        )


@dataclass(frozen=True, slots=True)
class PreparedCaseDraft:
    manifest_path: Path
    root: Path
    manifest: PreparedCaseDraftManifest
    resolved_artifacts: Mapping[str, Path]

    def artifact(self, role: str) -> Path:
        try:
            return self.resolved_artifacts[role]
        except KeyError as exc:
            raise KeyError(f"PreparedCase draft has no artifact role {role!r}") from exc


@dataclass(frozen=True, slots=True)
class DraftArtifactSnapshot:
    role: str
    source: Path
    path: Path
    media_type: str
    size: int
    sha256: str

    def artifact_ref(self) -> ArtifactRef:
        return ArtifactRef(
            path=self.path,
            sha256=self.sha256,
            size=self.size,
            role=self.role,
            media_type=self.media_type,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "source": str(self.source),
            "path": self.path.as_posix(),
            "size": self.size,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }


@dataclass(frozen=True, slots=True)
class PreparedCaseDraftInspection:
    draft_path: Path
    prepared_manifest: PreparedCaseManifest
    artifacts: tuple[DraftArtifactSnapshot, ...]
    index_group_counts: Mapping[str, int]
    projection_admission: Mapping[str, object]

    def validation_summary(self) -> dict[str, object]:
        return {
            "valid": True,
            "schema_version": PREPARED_CASE_DRAFT_SCHEMA,
            "case_id": self.prepared_manifest.case_id,
            "draft": str(self.draft_path),
            "md_atom_count": self.prepared_manifest.md.atom_count,
            "scoring_atom_count": self.prepared_manifest.scoring.atom_count,
            "artifact_count": len(self.artifacts),
            "projection_admission": {
                "accepted": bool(self.projection_admission["accepted"]),
                "issues": list(self.projection_admission["issues"]),
            },
        }

    def as_dict(self) -> dict[str, object]:
        manifest = self.prepared_manifest
        return {
            **self.validation_summary(),
            "output_schema_version": PREPARED_CASE_SCHEMA,
            "md": {
                "backend": manifest.md.backend,
                "atom_count": manifest.md.atom_count,
                "system_signature": manifest.md.system_signature,
                "artifact_roles": [item.role for item in manifest.md.artifacts],
            },
            "scoring": {
                "backend": manifest.scoring.backend,
                "atom_count": manifest.scoring.atom_count,
                "atom_order_signature": manifest.scoring.atom_order_signature,
                "radii_model": manifest.scoring.radii_model,
                "source_md_atom_count": len(manifest.scoring.source_md_atom_indices),
                "artifact_roles": [item.role for item in manifest.scoring.artifacts],
            },
            "entities": [
                {
                    "entity_id": item.entity_id,
                    "kind": item.kind,
                    "md_atom_count": len(item.md_atom_indices),
                    "scoring_atom_count": len(item.scoring_atom_indices),
                    "named_selections": sorted(item.named_selections),
                    "feature_ids": [feature.feature_id for feature in item.features],
                }
                for item in manifest.entities
            ],
            "primary_binding": manifest.primary_binding.as_dict(),
            "index_group_counts": dict(self.index_group_counts),
            "artifacts": [item.as_dict() for item in self.artifacts],
            "projection_admission": dict(self.projection_admission),
            "provenance": dict(manifest.provenance),
        }


def load_prepared_case_draft(path: Path) -> PreparedCaseDraft:
    """Load a strict draft and resolve only its declared existing artifacts."""

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"PreparedCase draft manifest does not exist: {manifest_path}")
    suffix = manifest_path.suffix.lower()
    try:
        text = manifest_path.read_text(encoding="utf-8")
        if suffix == ".json":
            raw = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(text)
        else:
            raise ContractError("PreparedCase draft manifest must use .json, .yaml, or .yml")
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read PreparedCase draft manifest: {manifest_path}") from exc
    manifest = PreparedCaseDraftManifest.from_mapping(_mapping(raw, "PreparedCase draft manifest"))
    root = manifest_path.parent.resolve()
    resolved: dict[str, Path] = {}
    resolved_paths: set[Path] = set()
    for artifact in manifest.artifacts:
        try:
            source = resolve_within(root, artifact.source, must_exist=True)
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"draft artifact source is unsafe or missing: {artifact.source}"
            ) from exc
        if not source.is_file() or source.stat().st_size < 1:
            raise ArtifactIntegrityError(
                f"draft artifact source must be a non-empty file: {source}"
            )
        if source in resolved_paths:
            raise ContractError("draft artifact sources resolve to the same file")
        resolved[artifact.role] = source
        resolved_paths.add(source)
    return PreparedCaseDraft(
        manifest_path=manifest_path,
        root=root,
        manifest=manifest,
        resolved_artifacts=resolved,
    )


def _snapshot_artifacts(
    draft: PreparedCaseDraft, resolved: Mapping[str, Path]
) -> tuple[DraftArtifactSnapshot, ...]:
    snapshots: list[DraftArtifactSnapshot] = []
    for artifact in draft.manifest.artifacts:
        source = Path(resolved[artifact.role]).resolve()
        if not source.is_file() or source.stat().st_size < 1:
            raise ArtifactIntegrityError(
                f"draft artifact source must be a non-empty file: {source}"
            )
        snapshots.append(
            DraftArtifactSnapshot(
                role=artifact.role,
                source=source,
                path=artifact.path,
                media_type=artifact.media_type,
                size=source.stat().st_size,
                sha256=sha256_file(source),
            )
        )
    return tuple(snapshots)


def _load_parmed_structures(
    resolved: Mapping[str, Path],
) -> tuple[object, object]:
    """Load existing topology/coordinate pairs; never infer or alter chemistry."""

    try:
        import parmed as pmd
    except ImportError as exc:
        raise ContractError(
            "PreparedCase draft validation requires the 'parmed' package extra"
        ) from exc
    try:
        md = pmd.load_file(str(resolved["md_topology"]), xyz=str(resolved["md_coordinates"]))
        scoring = pmd.load_file(str(resolved["complex_topology"]), xyz=str(resolved["coordinates"]))
    except Exception as exc:
        raise ContractError(
            "ParmEd could not load the declared GROMACS/Amber artifact pairs"
        ) from exc
    return md, scoring


def _atom_identity(atom: object) -> tuple[str, str, int]:
    residue = getattr(atom, "residue", None)
    return (
        str(getattr(residue, "name", "")).upper(),
        str(getattr(atom, "name", "")),
        int(getattr(atom, "atomic_number", 0) or 0),
    )


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atom_order_signature(structure: object) -> str:
    return _sha256_json(
        {
            "schema": "mmpbsa.atom_order.v1",
            "atoms": [
                {
                    "residue_index": int(atom.residue.idx) + 1,
                    "residue_name": str(atom.residue.name),
                    "atom_name": str(atom.name),
                    "atomic_number": int(getattr(atom, "atomic_number", 0) or 0),
                }
                for atom in structure.atoms
            ],
        }
    )


def _md_system_signature(structure: object) -> str:
    bonds = sorted(
        (
            min(int(bond.atom1.idx), int(bond.atom2.idx)) + 1,
            max(int(bond.atom1.idx), int(bond.atom2.idx)) + 1,
        )
        for bond in structure.bonds
    )
    return _sha256_json(
        {
            "schema": "mmpbsa.md_system.v1",
            "atom_order": _atom_order_signature(structure),
            "atom_count": len(structure.atoms),
            "residue_count": len(structure.residues),
            "bonds": bonds,
            "box": (
                None
                if getattr(structure, "box", None) is None
                else [float(value) for value in structure.box]
            ),
        }
    )


def _verify_explicit_atom_mapping(
    md: object, scoring: object, source_md_atom_indices: tuple[int, ...]
) -> None:
    if len(source_md_atom_indices) != len(scoring.atoms):
        raise ContractError("source_md_atom_indices length differs from Amber scoring atom count")
    if max(source_md_atom_indices) > len(md.atoms):
        raise ContractError("source_md_atom_indices exceeds GROMACS MD atom count")
    for scoring_index, md_index in enumerate(source_md_atom_indices, start=1):
        scoring_identity = _atom_identity(scoring.atoms[scoring_index - 1])
        md_identity = _atom_identity(md.atoms[md_index - 1])
        if scoring_identity != md_identity:
            raise ContractError(
                "explicit scoring-to-MD atom identity differs at scoring atom "
                f"{scoring_index} -> MD atom {md_index}: "
                f"{scoring_identity!r} != {md_identity!r}"
            )
        for label, tolerance in (("charge", 1.0e-8), ("mass", 1.0e-8)):
            scoring_value = float(getattr(scoring.atoms[scoring_index - 1], label))
            md_value = float(getattr(md.atoms[md_index - 1], label))
            if (
                not math.isfinite(scoring_value)
                or not math.isfinite(md_value)
                or abs(scoring_value - md_value) > tolerance
            ):
                raise ContractError(
                    "explicit scoring-to-MD atom "
                    f"{label} differs at scoring atom {scoring_index} -> "
                    f"MD atom {md_index}: {scoring_value!r} != {md_value!r}"
                )


def _residue_atom_indices(structure: object, residue_indices: set[int]) -> tuple[int, ...]:
    return tuple(
        ordinal
        for ordinal, atom in enumerate(structure.atoms, start=1)
        if int(atom.residue.idx) + 1 in residue_indices
    )


def _selected_atom_indices(
    structure: object,
    atom_indices: tuple[int, ...],
    predicate: Callable[[object], bool],
) -> tuple[int, ...]:
    return tuple(index for index in atom_indices if predicate(structure.atoms[index - 1]))


def _verify_entity_semantics(md: object, scoring: object, manifest: PreparedCaseManifest) -> None:
    """Verify declared entity facts against both serialized atom orders."""

    for entity in manifest.entities:
        md_residue_indices: set[int] = set()
        scoring_residue_indices: set[int] = set()
        for residue in entity.residues:
            md_index = residue.md_residue_index
            if md_index > len(md.residues):
                raise ContractError(
                    f"entity {entity.entity_id!r} MD residue index {md_index} "
                    "exceeds the GROMACS residue count"
                )
            md_name = str(md.residues[md_index - 1].name)
            if md_name != residue.residue_name:
                raise ContractError(
                    f"entity {entity.entity_id!r} residue {residue.residue_id!r} "
                    f"name differs in MD topology: {residue.residue_name!r} != "
                    f"{md_name!r}"
                )
            md_residue_indices.add(md_index)

            scoring_index = residue.scoring_residue_index
            if scoring_index is None:
                continue
            if scoring_index > len(scoring.residues):
                raise ContractError(
                    f"entity {entity.entity_id!r} scoring residue index "
                    f"{scoring_index} exceeds the Amber residue count"
                )
            scoring_name = str(scoring.residues[scoring_index - 1].name)
            if scoring_name != residue.residue_name:
                raise ContractError(
                    f"entity {entity.entity_id!r} residue {residue.residue_id!r} "
                    f"name differs in scoring topology: {residue.residue_name!r} "
                    f"!= {scoring_name!r}"
                )
            scoring_residue_indices.add(scoring_index)

        expected_md_atoms = _residue_atom_indices(md, md_residue_indices)
        if entity.md_atom_indices != expected_md_atoms:
            raise ContractError(
                f"entity {entity.entity_id!r} MD atom inventory does not exactly "
                "contain its declared residues"
            )
        expected_scoring_atoms = _residue_atom_indices(scoring, scoring_residue_indices)
        if entity.scoring_atom_indices != expected_scoring_atoms:
            raise ContractError(
                f"entity {entity.entity_id!r} scoring atom inventory does not "
                "exactly contain its declared residues"
            )

        expected_md_heavy = _selected_atom_indices(
            md,
            entity.md_atom_indices,
            lambda atom: int(getattr(atom, "atomic_number", 0) or 0) > 1,
        )
        expected_scoring_heavy = _selected_atom_indices(
            scoring,
            entity.scoring_atom_indices,
            lambda atom: int(getattr(atom, "atomic_number", 0) or 0) > 1,
        )
        heavy = entity.named_selections["heavy_atoms"]
        if (
            heavy.md_atom_indices != expected_md_heavy
            or heavy.scoring_atom_indices != expected_scoring_heavy
        ):
            raise ContractError(
                f"entity {entity.entity_id!r} selection 'heavy_atoms' must "
                "contain exactly atoms with atomic number greater than one"
            )

        if entity.kind in {"protein", "peptide"}:
            expected_md_backbone = _selected_atom_indices(
                md,
                entity.md_atom_indices,
                lambda atom: str(atom.name) in _BACKBONE_ATOM_NAMES,
            )
            expected_scoring_backbone = _selected_atom_indices(
                scoring,
                entity.scoring_atom_indices,
                lambda atom: str(atom.name) in _BACKBONE_ATOM_NAMES,
            )
            backbone = entity.named_selections["backbone"]
            if (
                backbone.md_atom_indices != expected_md_backbone
                or backbone.scoring_atom_indices != expected_scoring_backbone
            ):
                raise ContractError(
                    f"entity {entity.entity_id!r} selection 'backbone' must "
                    "contain exactly N, CA, C, and O atoms"
                )


def _parse_index(path: Path) -> Mapping[str, tuple[int, ...]]:
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
    ):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            if not name or name in groups:
                raise ContractError(
                    f"GROMACS index has an empty or duplicate group at line {line_number}"
                )
            groups[name] = []
            current = name
            continue
        if current is None:
            raise ContractError(f"GROMACS index has atoms before a group at line {line_number}")
        try:
            values = [int(item) for item in line.split()]
        except ValueError as exc:
            raise ContractError(
                f"GROMACS index has a non-integer atom at line {line_number}"
            ) from exc
        if any(item < 1 for item in values):
            raise ContractError(f"GROMACS index has a non-positive atom at line {line_number}")
        groups[current].extend(values)
    return {name: tuple(values) for name, values in groups.items()}


def _validate_index(path: Path, manifest: PreparedCaseManifest) -> Mapping[str, int]:
    groups = _parse_index(path)
    missing = set(_INDEX_GROUPS).difference(groups)
    if missing:
        raise ContractError("GROMACS index is missing groups: " + ", ".join(sorted(missing)))
    expected = {
        "System": tuple(range(1, manifest.md.atom_count + 1)),
        # GROMACS writes selected coordinates in index-group order.  The
        # Complex group therefore has to preserve the exact Amber scoring
        # order, not merely contain the same atom set.
        "Complex": manifest.scoring.source_md_atom_indices,
        "Receptor": manifest.binding_selection("receptor_all").md_atom_indices,
        "Ligand": manifest.binding_selection("ligand_all").md_atom_indices,
    }
    for name, wanted in expected.items():
        observed = groups[name]
        if len(observed) != len(set(observed)):
            raise ContractError(f"GROMACS index group {name!r} repeats atoms")
        matches = (
            observed == wanted
            if name == "Complex"
            else tuple(sorted(observed)) == tuple(sorted(wanted))
        )
        if not matches:
            raise ContractError(
                f"GROMACS index group {name!r} differs from the declared atom "
                "inventory or required order"
            )
    return {name: len(groups[name]) for name in _INDEX_GROUPS}


def _radii_model_token(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _amber_radii_inventory(structure: object, declared_model: str, label: str) -> tuple[float, ...]:
    """Read and validate the radii declaration embedded in an Amber topology."""

    parm_data = getattr(structure, "parm_data", None)
    if not isinstance(parm_data, Mapping):
        raise ContractError(f"{label} is not an Amber topology with radii metadata")
    raw_radius_set = parm_data.get("RADIUS_SET")
    if isinstance(raw_radius_set, str):
        radius_set_values = [raw_radius_set]
    else:
        try:
            radius_set_values = list(raw_radius_set)
        except TypeError as exc:
            raise ContractError(f"{label} has no valid RADIUS_SET") from exc
    if len(radius_set_values) != 1 or not str(radius_set_values[0]).strip():
        raise ContractError(f"{label} has no unambiguous RADIUS_SET")
    radius_set = str(radius_set_values[0]).strip()
    declared_token = _radii_model_token(declared_model)
    observed_tokens = {
        _radii_model_token(radius_set),
        *(_radii_model_token(item) for item in re.findall(r"\(([^()]*)\)", radius_set)),
    }
    if declared_token not in observed_tokens:
        raise ContractError(
            f"{label} RADIUS_SET {radius_set!r} does not match declared "
            f"radii_model {declared_model!r}"
        )

    raw_radii = parm_data.get("RADII")
    try:
        radii = tuple(float(value) for value in raw_radii)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} has no valid RADII array") from exc
    if len(radii) != len(structure.atoms):
        raise ContractError(f"{label} RADII count differs from its atom count")
    if any(not math.isfinite(value) or value <= 0 for value in radii):
        raise ContractError(f"{label} RADII must be finite and positive")
    return radii


def _run_projection_admission(
    manifest: PreparedCaseManifest, resolved: Mapping[str, Path]
) -> Mapping[str, object]:
    try:
        import parmed as pmd

        raw_version = metadata.version("ParmEd")
        if int(raw_version.partition(".")[0]) != 4:
            raise ContractError(f"PreparedCase draft requires ParmEd major 4: {raw_version}")
        roles = manifest.primary_binding.topology_artifact_roles
        complex_structure = pmd.load_file(str(resolved[roles.complex]))
        component_structures = {
            "receptor": pmd.load_file(str(resolved[roles.receptor])),
            "ligand": pmd.load_file(str(resolved[roles.ligand])),
        }
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"ParmEd topology projection admission failed: {exc}") from exc
    indices = {
        "receptor": manifest.binding_selection("receptor_all").scoring_atom_indices,
        "ligand": manifest.binding_selection("ligand_all").scoring_atom_indices,
    }
    radii = {
        "complex": _amber_radii_inventory(
            complex_structure, manifest.scoring.radii_model, "complex topology"
        ),
        **{
            component: _amber_radii_inventory(
                structure,
                manifest.scoring.radii_model,
                f"{component} topology",
            )
            for component, structure in component_structures.items()
        },
    }
    partition = tuple(sorted(atom for values in indices.values() for atom in values))
    issues = []
    if partition != tuple(range(1, len(complex_structure.atoms) + 1)):
        issues.append("component selections do not exactly partition complex atoms")
    issues.extend(
        binding_boundary_bond_issues(
            complex_structure,
            indices["receptor"],
            indices["ligand"],
        )
    )
    for component in ("receptor", "ligand"):
        issues.extend(
            component_projection_issues(
                complex_structure,
                component_structures[component],
                indices[component],
                component,
            )
        )
        projected_radii = tuple(radii["complex"][index - 1] for index in indices[component])
        if len(projected_radii) != len(radii[component]) or any(
            abs(source - target) > 1.0e-8
            for source, target in zip(projected_radii, radii[component], strict=False)
        ):
            issues.append(f"{component}: projected Amber radii differ")
    if issues:
        raise ContractError("ParmEd topology projection admission failed: " + "; ".join(issues))
    return {
        "accepted": True,
        "issues": [],
        "complex_signature": topology_signature(complex_structure),
        "component_signatures": {
            key: topology_signature(value) for key, value in component_structures.items()
        },
        "radii": {
            "model": manifest.scoring.radii_model,
            "complex_atom_count": len(radii["complex"]),
            "component_atom_counts": {
                component: len(radii[component]) for component in ("receptor", "ligand")
            },
        },
        "tool": {"name": "ParmEd", "raw_version": raw_version},
    }


def _prepared_manifest(
    draft: PreparedCaseDraft,
    snapshots: tuple[DraftArtifactSnapshot, ...],
    md: object,
    scoring: object,
) -> PreparedCaseManifest:
    snapshot_by_role = {item.role: item for item in snapshots}
    seal_provenance = {
        "schema_version": PREPARED_CASE_DRAFT_SCHEMA,
        "draft_sha256": sha256_file(draft.manifest_path),
        "mode": "mechanical_artifact_wrap",
        "source_structure_policy": "not_read",
        "modeling_policy": "none",
    }
    return PreparedCaseManifest(
        case_id=draft.manifest.case_id,
        md=PreparedMdSystem(
            backend=draft.manifest.md.backend,
            atom_count=len(md.atoms),
            system_signature=_md_system_signature(md),
            artifacts=tuple(
                snapshot_by_role[item.role].artifact_ref() for item in draft.manifest.md.artifacts
            ),
            metadata=dict(draft.manifest.md.metadata),
        ),
        scoring=PreparedScoringSystem(
            backend=draft.manifest.scoring.backend,
            atom_count=len(scoring.atoms),
            atom_order_signature=_atom_order_signature(scoring),
            source_md_atom_indices=draft.manifest.scoring.source_md_atom_indices,
            radii_model=draft.manifest.scoring.radii_model,
            artifacts=tuple(
                snapshot_by_role[item.role].artifact_ref()
                for item in draft.manifest.scoring.artifacts
            ),
            metadata=dict(draft.manifest.scoring.metadata),
        ),
        entities=draft.manifest.entities,
        primary_binding=draft.manifest.primary_binding,
        provenance={
            **dict(draft.manifest.provenance),
            _SEAL_PROVENANCE_KEY: seal_provenance,
        },
    )


def _inspect_resolved(
    draft: PreparedCaseDraft, resolved: Mapping[str, Path]
) -> PreparedCaseDraftInspection:
    md_files = {
        artifact.path: resolved[artifact.role]
        for artifact in draft.manifest.md.artifacts
    }
    md_topology = next(
        artifact.path
        for artifact in draft.manifest.md.artifacts
        if artifact.role == "md_topology"
    )
    topology_include_closure(md_topology, md_files)
    snapshots = _snapshot_artifacts(draft, resolved)
    md, scoring = _load_parmed_structures(resolved)
    if not getattr(md, "atoms", None) or not getattr(scoring, "atoms", None):
        raise ContractError("declared MD and scoring systems must contain atoms")
    _verify_explicit_atom_mapping(md, scoring, draft.manifest.scoring.source_md_atom_indices)
    manifest = _prepared_manifest(draft, snapshots, md, scoring)
    _verify_entity_semantics(md, scoring, manifest)
    md_boundary_issues = binding_boundary_bond_issues(
        md,
        manifest.binding_selection("receptor_all").md_atom_indices,
        manifest.binding_selection("ligand_all").md_atom_indices,
    )
    if md_boundary_issues:
        raise ContractError(
            "MD binding-boundary admission failed: " + "; ".join(md_boundary_issues)
        )
    projection_issues = mapped_system_projection_issues(
        md,
        scoring,
        draft.manifest.scoring.source_md_atom_indices,
        "MD-to-scoring projection",
    )
    if projection_issues:
        raise ContractError(
            "MD-to-scoring topology projection failed: " + "; ".join(projection_issues)
        )
    index_counts = _validate_index(resolved["index"], manifest)
    admission = _run_projection_admission(manifest, resolved)
    seal = dict(manifest.provenance[_SEAL_PROVENANCE_KEY])
    seal["projection_admission"] = admission
    manifest = replace(
        manifest,
        provenance={
            **dict(draft.manifest.provenance),
            _SEAL_PROVENANCE_KEY: seal,
        },
    )
    return PreparedCaseDraftInspection(
        draft_path=draft.manifest_path,
        prepared_manifest=manifest,
        artifacts=snapshots,
        index_group_counts=index_counts,
        projection_admission=admission,
    )


def inspect_prepared_case_draft(path: Path) -> PreparedCaseDraftInspection:
    """Perform read-only contract, topology, mapping, and index inspection."""

    draft = load_prepared_case_draft(path)
    return _inspect_resolved(draft, draft.resolved_artifacts)


def validate_prepared_case_draft(path: Path) -> PreparedCaseDraftInspection:
    """Validate a draft and return the same typed evidence used by ``inspect``."""

    return inspect_prepared_case_draft(path)


@contextmanager
def _locked_destination(target: Path, case_id: str) -> Iterator[None]:
    try:
        with InterProcessFileLock(
            target.parent / f".{target.name}.seal.lock",
            owner="prepared-case-seal",
            metadata={"case_id": case_id, "target": str(target)},
            timeout_seconds=0,
            stale_after_seconds=24 * 60 * 60,
        ):
            yield
    except LockTimeoutError as exc:
        raise ArtifactIntegrityError(f"PreparedCase seal destination is locked: {target}") from exc


def seal_prepared_case_draft(path: Path, destination: Path) -> PreparedCase:
    """Validate, copy, and atomically publish an immutable PreparedCase v2.

    Existing destinations are refused.  Every copied byte is re-hashed and the
    relocated GROMACS/Amber systems are re-admitted before publication.
    """

    draft = load_prepared_case_draft(path)
    source_inspection = _inspect_resolved(draft, draft.resolved_artifacts)
    requested_target = Path(destination).absolute()
    target = requested_target.parent.resolve() / requested_target.name
    source_snapshot = {item.role: item for item in source_inspection.artifacts}
    with _locked_destination(target, draft.manifest.case_id):
        if target.exists() or target.is_symlink():
            raise ArtifactIntegrityError(f"PreparedCase seal destination already exists: {target}")
        with atomic_directory(target) as staging:
            staged: dict[str, Path] = {}
            for artifact in draft.manifest.artifacts:
                destination_path = resolve_within(staging, artifact.path)
                materialize_immutable_file(
                    draft.resolved_artifacts[artifact.role], destination_path
                )
                expected = source_snapshot[artifact.role]
                if (
                    destination_path.stat().st_size != expected.size
                    or sha256_file(destination_path) != expected.sha256
                ):
                    raise ArtifactIntegrityError(
                        f"sealed copy differs for artifact role {artifact.role!r}"
                    )
                staged[artifact.role] = destination_path

            staged_inspection = _inspect_resolved(draft, staged)
            if (
                staged_inspection.prepared_manifest.as_dict()
                != source_inspection.prepared_manifest.as_dict()
            ):
                raise ArtifactIntegrityError(
                    "relocated artifacts changed the PreparedCase mechanical admission"
                )
            manifest_path = staging / "prepared_case.json"
            write_prepared_case_manifest(manifest_path, staged_inspection.prepared_manifest)
            load_prepared_case(manifest_path, expected_case_id=draft.manifest.case_id)

    return load_prepared_case(
        target / "prepared_case.json", expected_case_id=draft.manifest.case_id
    )


__all__ = [
    "DraftArtifact",
    "DraftArtifactSnapshot",
    "PREPARED_CASE_DRAFT_SCHEMA",
    "PreparedCaseDraft",
    "PreparedCaseDraftInspection",
    "PreparedCaseDraftManifest",
    "PreparedMdDraft",
    "PreparedScoringDraft",
    "inspect_prepared_case_draft",
    "load_prepared_case_draft",
    "seal_prepared_case_draft",
    "validate_prepared_case_draft",
]

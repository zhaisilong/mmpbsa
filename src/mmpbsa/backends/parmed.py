"""ParmEd-backed topology admission and component parameter projection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from ._support import require_file
from .api import BackendReceipt, CommandRunner, ReceiptState
from .errors import BackendConfigurationError, BackendOutputError
from .versioning import MajorVersionPolicy, parse_version


@dataclass(frozen=True, slots=True)
class AtomRecord:
    ordinal: int
    residue_ordinal: int
    residue_name: str
    atom_name: str
    atomic_number: int
    atom_type: str
    charge: float
    mass: float
    epsilon: float
    rmin: float


@dataclass(frozen=True, slots=True)
class TermRecord:
    kind: str
    atom_ordinals: tuple[int, ...]
    parameters: tuple[str | int | float, ...]


@dataclass(frozen=True, slots=True)
class TopologyView:
    atoms: tuple[AtomRecord, ...]
    terms: tuple[TermRecord, ...] = ()

    def validate(self) -> None:
        if not self.atoms:
            raise BackendOutputError("topology contains no atoms")
        if tuple(atom.ordinal for atom in self.atoms) != tuple(range(1, len(self.atoms) + 1)):
            raise BackendOutputError("topology atom ordinals must be contiguous and one-based")
        for atom in self.atoms:
            if atom.residue_ordinal < 1 or not atom.residue_name or not atom.atom_name:
                raise BackendOutputError("topology contains an invalid atom identity")
            numeric = (atom.charge, atom.mass, atom.epsilon, atom.rmin)
            if not all(math.isfinite(float(value)) for value in numeric):
                raise BackendOutputError("topology contains non-finite atom parameters")
        for term in self.terms:
            if not term.kind or not term.atom_ordinals:
                raise BackendOutputError("topology contains a malformed parameter term")
            if any(index < 1 or index > len(self.atoms) for index in term.atom_ordinals):
                raise BackendOutputError("topology term references an invalid atom")


class TopologyLoader(Protocol):
    def probe(self) -> Mapping[str, object]: ...

    def load(self, path: Path) -> TopologyView: ...


@dataclass(frozen=True, slots=True)
class ProjectionTolerances:
    charge: float = 1.0e-8
    mass: float = 1.0e-8
    parameter: float = 1.0e-8

    def validate(self) -> None:
        for label, value in (
            ("charge", self.charge),
            ("mass", self.mass),
            ("parameter", self.parameter),
        ):
            if not math.isfinite(value) or value < 0:
                raise BackendConfigurationError(
                    f"ParmEd projection {label} tolerance must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class ParmEdAdmissionRequest:
    complex_topology: Path
    component_topologies: Mapping[str, Path]
    component_atom_indices: Mapping[str, tuple[int, ...]]
    tolerances: ProjectionTolerances = ProjectionTolerances()
    require_complete_partition: bool = True

    def validate(self) -> None:
        require_file(self.complex_topology, "complex topology")
        if not self.component_topologies:
            raise BackendConfigurationError("component topologies cannot be empty")
        if set(self.component_topologies) != set(self.component_atom_indices):
            raise BackendConfigurationError("component topology and atom-index inventories differ")
        for component, path in self.component_topologies.items():
            if not str(component).strip():
                raise BackendConfigurationError("component name cannot be empty")
            require_file(path, f"{component} topology")
            indices = self.component_atom_indices[component]
            if (
                not indices
                or any(isinstance(index, bool) or index < 1 for index in indices)
                or tuple(sorted(set(indices))) != indices
            ):
                raise BackendConfigurationError(
                    f"{component} atom indices must be sorted, unique, and positive"
                )
        flattened = tuple(
            index
            for component in self.component_atom_indices
            for index in self.component_atom_indices[component]
        )
        if len(flattened) != len(set(flattened)):
            raise BackendConfigurationError("component atom selections overlap")
        self.tolerances.validate()


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    accepted: bool
    issues: tuple[str, ...]
    complex_signature: str
    component_signatures: Mapping[str, str]
    tool: Mapping[str, object]

    def require_accepted(self) -> None:
        if not self.accepted:
            raise BackendOutputError(
                "topology parameter projection was rejected: " + "; ".join(self.issues)
            )


@dataclass(frozen=True, slots=True)
class ParmEdAdmissionBackend:
    loader: TopologyLoader

    def admit(self, request: ParmEdAdmissionRequest) -> AdmissionResult:
        request.validate()
        complex_view = self.loader.load(request.complex_topology)
        complex_view.validate()
        all_indices = tuple(
            sorted(
                index for indices in request.component_atom_indices.values() for index in indices
            )
        )
        issues: list[str] = []
        if all_indices and all_indices[-1] > len(complex_view.atoms):
            issues.append("component selection exceeds complex atom count")
        if request.require_complete_partition and all_indices != tuple(
            range(1, len(complex_view.atoms) + 1)
        ):
            issues.append("component selections do not exactly partition complex atoms")

        signatures: dict[str, str] = {}
        for component, path in request.component_topologies.items():
            component_view = self.loader.load(path)
            component_view.validate()
            signatures[component] = topology_signature(component_view)
            indices = request.component_atom_indices[component]
            if indices and indices[-1] <= len(complex_view.atoms):
                issues.extend(
                    _projection_issues(
                        component,
                        complex_view,
                        indices,
                        component_view,
                        request.tolerances,
                    )
                )
        return AdmissionResult(
            accepted=not issues,
            issues=tuple(issues),
            complex_signature=topology_signature(complex_view),
            component_signatures=signatures,
            tool=self.loader.probe(),
        )

    def run(
        self,
        request: ParmEdAdmissionRequest,
        *,
        dry_run: bool = False,
    ) -> BackendReceipt:
        request.validate()
        if dry_run:
            return BackendReceipt(
                state=ReceiptState.PLANNED,
                commands=(),
                metadata={
                    "components": tuple(request.component_topologies),
                    "require_complete_partition": request.require_complete_partition,
                },
            )
        result = self.admit(request)
        return BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=(),
            metadata={"tool": result.tool},
            parsed={
                "accepted": result.accepted,
                "issues": result.issues,
                "complex_signature": result.complex_signature,
                "component_signatures": result.component_signatures,
            },
        )


@dataclass(frozen=True, slots=True)
class DefaultParmEdLoader:
    major_policy: MajorVersionPolicy

    def probe(self) -> Mapping[str, object]:
        try:
            raw = metadata.version("ParmEd")
        except metadata.PackageNotFoundError as exc:
            raise BackendConfigurationError(
                "ParmEd admission requires the 'parmed' package extra"
            ) from exc
        version = parse_version(raw)
        if not self.major_policy.accepts(version[0]):
            raise BackendConfigurationError(
                f"ParmEd major {version[0]} is incompatible with {self.major_policy}"
            )
        return {"name": "ParmEd", "raw_version": raw, "version": version}

    def load(self, path: Path) -> TopologyView:
        self.probe()
        try:
            import parmed as pmd
        except ImportError as exc:
            raise BackendConfigurationError(
                "ParmEd admission requires the 'parmed' package extra"
            ) from exc
        structure = pmd.load_file(str(require_file(path, "ParmEd topology")))
        atoms = tuple(_atom_record(atom, index) for index, atom in enumerate(structure.atoms, 1))
        terms = tuple(
            sorted(
                (
                    *(_term_record("bond", term) for term in getattr(structure, "bonds", ())),
                    *(_term_record("angle", term) for term in getattr(structure, "angles", ())),
                    *(
                        _term_record(
                            ("improper" if bool(getattr(term, "improper", False)) else "dihedral"),
                            term,
                        )
                        for term in getattr(structure, "dihedrals", ())
                    ),
                    *(
                        _term_record("improper", term)
                        for term in getattr(structure, "impropers", ())
                    ),
                ),
                key=lambda item: (item.kind, item.atom_ordinals, item.parameters),
            )
        )
        return TopologyView(atoms=atoms, terms=terms)


def topology_signature(view: TopologyView) -> str:
    view.validate()
    payload = {
        "atoms": [
            {
                "residue_ordinal": atom.residue_ordinal,
                "residue_name": atom.residue_name,
                "atom_name": atom.atom_name,
                "atomic_number": atom.atomic_number,
                "atom_type": atom.atom_type,
                "charge": atom.charge,
                "mass": atom.mass,
                "epsilon": atom.epsilon,
                "rmin": atom.rmin,
            }
            for atom in view.atoms
        ],
        "terms": [
            {
                "kind": term.kind,
                "atoms": term.atom_ordinals,
                "parameters": term.parameters,
            }
            for term in view.terms
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def create_parmed_admission_backend(
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None = None,
) -> ParmEdAdmissionBackend:
    del runner
    return ParmEdAdmissionBackend(
        loader=DefaultParmEdLoader(
            major_policy=MajorVersionPolicy.from_value(config.get("major_policy"))
        )
    )


def _projection_issues(
    component: str,
    complex_view: TopologyView,
    indices: tuple[int, ...],
    component_view: TopologyView,
    tolerances: ProjectionTolerances,
) -> list[str]:
    issues: list[str] = []
    projected_atoms = tuple(complex_view.atoms[index - 1] for index in indices)
    if len(projected_atoms) != len(component_view.atoms):
        return [
            f"{component}: atom count differs "
            f"({len(projected_atoms)} != {len(component_view.atoms)})"
        ]
    for ordinal, (source, target) in enumerate(
        zip(projected_atoms, component_view.atoms, strict=True),
        start=1,
    ):
        identity = (
            source.residue_name,
            source.atom_name,
            source.atomic_number,
            source.atom_type,
        )
        observed = (
            target.residue_name,
            target.atom_name,
            target.atomic_number,
            target.atom_type,
        )
        if identity != observed:
            issues.append(f"{component}: atom identity differs at component atom {ordinal}")
            continue
        for label, left, right, tolerance in (
            ("charge", source.charge, target.charge, tolerances.charge),
            ("mass", source.mass, target.mass, tolerances.mass),
            ("epsilon", source.epsilon, target.epsilon, tolerances.parameter),
            ("rmin", source.rmin, target.rmin, tolerances.parameter),
        ):
            if abs(float(left) - float(right)) > tolerance:
                issues.append(f"{component}: {label} differs at component atom {ordinal}")
    local_ordinals = {source: local for local, source in enumerate(indices, start=1)}
    projected_terms = tuple(
        sorted(
            (
                TermRecord(
                    kind=term.kind,
                    atom_ordinals=_projection_term_atoms(
                        term.kind, tuple(local_ordinals[index] for index in term.atom_ordinals)
                    ),
                    parameters=term.parameters,
                )
                for term in complex_view.terms
                if all(index in local_ordinals for index in term.atom_ordinals)
            ),
            key=lambda item: (item.kind, item.atom_ordinals, item.parameters),
        )
    )
    target_terms = tuple(
        sorted(
            (
                TermRecord(
                    term.kind,
                    _projection_term_atoms(term.kind, term.atom_ordinals),
                    term.parameters,
                )
                for term in component_view.terms
            ),
            key=lambda item: (item.kind, item.atom_ordinals, item.parameters),
        )
    )
    if len(projected_terms) != len(target_terms):
        issues.append(
            f"{component}: internal term count differs "
            f"({len(projected_terms)} != {len(target_terms)})"
        )
    else:
        for index, (source, target) in enumerate(
            zip(projected_terms, target_terms, strict=True),
            start=1,
        ):
            if source.kind != target.kind or source.atom_ordinals != target.atom_ordinals:
                issues.append(f"{component}: internal term identity differs at term {index}")
                continue
            if not _parameter_tuple_close(
                source.parameters,
                target.parameters,
                tolerances.parameter,
            ):
                issues.append(f"{component}: internal term parameters differ at term {index}")
    return issues


def _projection_term_atoms(kind: str, atoms: tuple[int, ...]) -> tuple[int, ...]:
    # Normalize exact reversal only, including Amber zero-index improper
    # serialization. Other improper permutations and force terms stay strict.
    # This does not change persisted topology signatures.
    if kind in {"bond", "angle", "dihedral", "improper"}:
        return min(atoms, tuple(reversed(atoms)))
    return atoms


def _parameter_tuple_close(
    left: Sequence[str | int | float],
    right: Sequence[str | int | float],
    tolerance: float,
) -> bool:
    if len(left) != len(right):
        return False
    for first, second in zip(left, right, strict=True):
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            if abs(float(first) - float(second)) > tolerance:
                return False
        elif first != second:
            return False
    return True


def _atom_record(atom: Any, ordinal: int) -> AtomRecord:
    atom_type = getattr(atom, "atom_type", None)
    epsilon = getattr(atom_type, "epsilon", getattr(atom, "epsilon", 0.0))
    rmin = getattr(atom_type, "rmin", getattr(atom, "rmin", 0.0))
    return AtomRecord(
        ordinal=ordinal,
        residue_ordinal=int(atom.residue.idx) + 1,
        residue_name=str(atom.residue.name),
        atom_name=str(atom.name),
        atomic_number=int(getattr(atom, "atomic_number", 0) or 0),
        atom_type=str(getattr(atom, "type", "")),
        charge=float(getattr(atom, "charge", 0.0)),
        mass=float(getattr(atom, "mass", 0.0)),
        epsilon=float(epsilon or 0.0),
        rmin=float(rmin or 0.0),
    )


def _term_record(kind: str, term: Any) -> TermRecord:
    atoms = tuple(
        int(getattr(term, name).idx) + 1
        for name in ("atom1", "atom2", "atom3", "atom4")
        if hasattr(term, name)
    )
    reverse = tuple(reversed(atoms))
    if kind in {"bond", "angle", "dihedral"} and reverse < atoms:
        atoms = reverse
    term_type = getattr(term, "type", None)
    values = _term_parameters(term_type)
    return TermRecord(
        kind=kind,
        atom_ordinals=atoms,
        parameters=values,
    )


def _term_parameters(value: Any) -> tuple[str | int | float, ...]:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        result: list[str | int | float] = ["terms", len(value)]
        for index, item in enumerate(value):
            result.extend(("term", index, *_term_parameters(item)))
        return tuple(result)
    result: list[str | int | float] = []
    for attribute in (
        "k",
        "req",
        "theteq",
        "phi_k",
        "per",
        "phase",
        "scee",
        "scnb",
    ):
        if hasattr(value, attribute):
            item = getattr(value, attribute)
            result.extend(
                (
                    attribute,
                    (float(item) if isinstance(item, (int, float)) else str(item)),
                )
            )
    return tuple(result)

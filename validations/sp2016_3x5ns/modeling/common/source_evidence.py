"""Derive auditable SP2016 facts from one hash-pinned official mmCIF.

This module observes coordinate/deposition facts only.  It deliberately does
not infer protonation, parameterization, inclusion, exclusion, or eligibility.
Those scientific decisions remain case-local model data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "sp2016.source_evidence.v1"


class SourceEvidenceError(ValueError):
    """The pinned source cannot support the requested factual audit."""


def _require_gemmi():
    try:
        import gemmi
    except ModuleNotFoundError as exc:
        raise SourceEvidenceError(
            "gemmi is required for source-derived SP2016 preflight evidence; "
            "install the project with `pip install -e '.[preparation]'`"
        ) from exc
    return gemmi


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SourceEvidenceError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SourceEvidenceError(f"{label} must be a sequence")
    return value


def _icode(value: object) -> str:
    normalized = str(value).replace("\x00", "").strip()
    return "" if normalized in {".", "?"} else normalized


def _altloc(value: object) -> str | None:
    normalized = str(value).replace("\x00", "").strip()
    return None if normalized in {"", ".", "?"} else normalized


def _residue_token(number: int, insertion_code: str) -> str:
    return f"{number}{insertion_code}"


def _selector(residue_ids: Sequence[tuple[int, str]]) -> str:
    """Render ordered author residue IDs in the catalog's canonical notation."""

    tokens: list[str] = []
    run_start: int | None = None
    run_end: int | None = None

    def flush() -> None:
        nonlocal run_start, run_end
        if run_start is not None and run_end is not None:
            tokens.append(str(run_start) if run_start == run_end else f"{run_start}-{run_end}")
        run_start = None
        run_end = None

    seen: set[tuple[int, str]] = set()
    for number, insertion_code in residue_ids:
        identity = (number, insertion_code)
        if identity in seen:
            continue
        seen.add(identity)
        if insertion_code:
            flush()
            tokens.append(_residue_token(number, insertion_code))
        elif run_start is None:
            run_start = run_end = number
        elif run_end is not None and number == run_end + 1:
            run_end = number
        else:
            flush()
            run_start = run_end = number
    flush()
    if not tokens:
        raise SourceEvidenceError("selected polymer segment has no observed residues")
    return ",".join(tokens)


def _one_letter(gemmi: object, residue_names: Sequence[str]) -> str:
    letters: list[str] = []
    for name in residue_names:
        letter = str(gemmi.find_tabulated_residue(name).one_letter_code).strip()
        letters.append((letter or "X").upper())
    return "".join(letters)


def _polymer_segment_evidence(
    gemmi: object,
    structure: object,
    *,
    role: str,
    ordinal: int,
    segment: Mapping[str, Any],
) -> dict[str, object]:
    auth_chain = str(segment.get("auth_chain", ""))
    label_asym_id = str(segment.get("label_asym_id", ""))
    chain = structure[0].find_chain(auth_chain)
    if chain is None:
        raise SourceEvidenceError(
            f"{role} segment {ordinal} auth chain {auth_chain!r} is absent from model 1"
        )
    residues = [
        residue
        for residue in chain
        if residue.entity_type == gemmi.EntityType.Polymer
        and str(residue.subchain) == label_asym_id
    ]
    if not residues:
        raise SourceEvidenceError(
            f"{role} segment {ordinal} auth/label pair "
            f"{auth_chain!r}/{label_asym_id!r} has no observed polymer residues"
        )
    residue_ids = [(int(residue.seqid.num), _icode(residue.seqid.icode)) for residue in residues]
    evidence: dict[str, object] = {
        "segment_ordinal": ordinal,
        "auth_chain": auth_chain,
        "label_asym_id": label_asym_id,
        "derived_observed_residue_selector": _selector(residue_ids),
        "observed_residue_count": len(residue_ids),
        "first_observed_residue": _residue_token(*residue_ids[0]),
        "last_observed_residue": _residue_token(*residue_ids[-1]),
    }
    if role == "ligand":
        evidence["derived_observed_sequence"] = _one_letter(
            gemmi, [str(residue.name) for residue in residues]
        )
    return evidence


def _declared_ligand_sequences(
    gemmi: object, structure: object, ligand_segments: Sequence[Mapping[str, Any]]
) -> tuple[str, list[dict[str, object]]]:
    entities_by_subchain: dict[str, list[object]] = {}
    for entity in structure.entities:
        for subchain in entity.subchains:
            entities_by_subchain.setdefault(str(subchain), []).append(entity)

    records: list[dict[str, object]] = []
    seen_entities: set[str] = set()
    combined: list[str] = []
    for segment in ligand_segments:
        label_asym_id = str(segment.get("label_asym_id", ""))
        entities = entities_by_subchain.get(label_asym_id, [])
        if len(entities) != 1:
            raise SourceEvidenceError(
                f"ligand label asym {label_asym_id!r} maps to {len(entities)} entities"
            )
        entity = entities[0]
        entity_id = str(entity.name)
        if entity_id in seen_entities:
            continue
        seen_entities.add(entity_id)
        residue_names = [str(name) for name in entity.full_sequence]
        if not residue_names:
            raise SourceEvidenceError(f"ligand entity {entity_id!r} has no declared sequence")
        sequence = _one_letter(gemmi, residue_names)
        combined.append(sequence)
        records.append(
            {
                "entity_id": entity_id,
                "label_asym_ids": sorted(str(item) for item in entity.subchains),
                "declared_residue_count": len(residue_names),
                "derived_declared_sequence": sequence,
                "source_category": "_entity_poly_seq via gemmi Structure.entities.full_sequence",
            }
        )
    if not combined:
        raise SourceEvidenceError("ligand has no source-declared polymer entity sequence")
    return "".join(combined), records


def _assembly_evidence(structure: object) -> list[dict[str, object]]:
    assemblies: list[dict[str, object]] = []
    for assembly in structure.assemblies:
        generators: list[dict[str, object]] = []
        for generator in assembly.generators:
            operators = [
                {
                    "operator_id": str(operator.name),
                    "type": str(operator.type),
                    "is_identity": bool(operator.transform.is_identity()),
                    "matrix": [
                        [float(item) for item in row] for row in operator.transform.mat.tolist()
                    ],
                    "vector": [float(item) for item in operator.transform.vec.tolist()],
                }
                for operator in generator.operators
            ]
            generators.append(
                {
                    "auth_chain_ids": sorted(str(item) for item in generator.chains),
                    "label_asym_ids": sorted(str(item) for item in generator.subchains),
                    "operators": operators,
                }
            )
        assemblies.append(
            {
                "assembly_id": str(assembly.name),
                "author_defined": bool(assembly.author_determined),
                "software_defined": bool(assembly.software_determined),
                "oligomeric_details": str(assembly.oligomeric_details),
                "generators": generators,
            }
        )
    return assemblies


def _residue_base_record(chain: object, residue: object) -> dict[str, object]:
    insertion_code = _icode(residue.seqid.icode)
    return {
        "auth_chain": str(chain.name),
        "auth_seq_id": int(residue.seqid.num),
        "insertion_code": insertion_code or None,
        "auth_residue_id": (
            f"{chain.name}:{_residue_token(int(residue.seqid.num), insertion_code)}"
        ),
        "label_asym_id": str(residue.subchain),
        "component_id": str(residue.name),
        "entity_type": str(residue.entity_type).rsplit(".", maxsplit=1)[-1].lower(),
    }


def _selected_non_water_heterogens(
    gemmi: object, structure: object, selected_auth_chains: set[str]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for chain in structure[0]:
        if str(chain.name) not in selected_auth_chains:
            continue
        for residue in chain:
            if residue.entity_type in {
                gemmi.EntityType.Water,
                gemmi.EntityType.Polymer,
            }:
                continue
            atoms = list(residue)
            record = _residue_base_record(chain, residue)
            occupancies = [float(atom.occ) for atom in atoms]
            record.update(
                {
                    "group_pdb": "HETATM",
                    "atom_count": len(atoms),
                    "atom_names": sorted({str(atom.name) for atom in atoms}),
                    "altloc_ids": sorted(
                        {altloc for atom in atoms if (altloc := _altloc(atom.altloc))}
                    ),
                    "occupancy_min": min(occupancies),
                    "occupancy_max": max(occupancies),
                }
            )
            records.append(record)
    return records


def _selected_nonstandard_polymer_residues(
    gemmi: object, structure: object, selected_auth_chains: set[str]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for chain in structure[0]:
        if str(chain.name) not in selected_auth_chains:
            continue
        for residue in chain:
            if residue.entity_type != gemmi.EntityType.Polymer:
                continue
            residue_info = gemmi.find_tabulated_residue(str(residue.name))
            if residue_info.is_standard():
                continue
            record = _residue_base_record(chain, residue)
            record.update(
                {
                    "atom_count": len(residue),
                    "gemmi_residue_info_found": bool(residue_info.found()),
                    "gemmi_is_standard": False,
                    "gemmi_one_letter_code": (str(residue_info.one_letter_code).strip() or "X"),
                }
            )
            records.append(record)
    return records


def _selected_altlocs(
    gemmi: object, structure: object, selected_auth_chains: set[str]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for chain in structure[0]:
        if str(chain.name) not in selected_auth_chains:
            continue
        for residue in chain:
            if residue.entity_type == gemmi.EntityType.Water:
                continue
            alternate_atoms = [
                (atom, altloc) for atom in residue if (altloc := _altloc(atom.altloc)) is not None
            ]
            if not alternate_atoms:
                continue
            conformers: list[dict[str, object]] = []
            for altloc in sorted({item[1] for item in alternate_atoms}):
                atoms = [
                    atom for atom, observed_altloc in alternate_atoms if observed_altloc == altloc
                ]
                conformers.append(
                    {
                        "altloc_id": altloc,
                        "atom_count": len(atoms),
                        "atom_names": sorted(str(atom.name) for atom in atoms),
                        "occupancies": sorted({float(atom.occ) for atom in atoms}),
                    }
                )
            record = _residue_base_record(chain, residue)
            record.update(
                {
                    "conformers": conformers,
                    "shared_blank_altloc_atom_count": sum(
                        _altloc(atom.altloc) is None for atom in residue
                    ),
                }
            )
            records.append(record)
    return records


def _cif_value(row: object, columns: Mapping[str, int], name: str) -> str | None:
    index = columns.get(name)
    if index is None:
        return None
    value = str(row.str(index)).strip()
    return None if value in {"", ".", "?"} else value


def _selected_polymer_entity_metadata(
    block: object,
    structure: object,
    model_segments: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, object]]]:
    table = block.find_mmcif_category("_entity.")
    columns = {str(tag).removeprefix("_entity."): index for index, tag in enumerate(table.tags)}
    source_rows: dict[str, object] = {}
    for row in table:
        entity_id = _cif_value(row, columns, "id")
        if entity_id is not None:
            source_rows[entity_id] = row

    result: dict[str, list[dict[str, object]]] = {"receptor": [], "ligand": []}
    for role in ("receptor", "ligand"):
        selected_labels = {
            str(segment.get("label_asym_id", "")) for segment in model_segments[role]
        }
        for entity in structure.entities:
            all_labels = sorted(str(item) for item in entity.subchains)
            matched_labels = sorted(selected_labels.intersection(all_labels))
            if not matched_labels:
                continue
            entity_id = str(entity.name)
            row = source_rows.get(entity_id)
            if row is None:
                raise SourceEvidenceError(
                    f"selected {role} polymer entity {entity_id!r} is absent from _entity"
                )
            entity_type = _cif_value(row, columns, "type")
            if entity_type != "polymer":
                raise SourceEvidenceError(
                    f"selected {role} entity {entity_id!r} has _entity.type={entity_type!r}"
                )
            result[role].append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "source_method": _cif_value(row, columns, "src_method"),
                    "pdbx_description": _cif_value(row, columns, "pdbx_description"),
                    "details": _cif_value(row, columns, "details"),
                    "selected_label_asym_ids": matched_labels,
                    "all_entity_label_asym_ids": all_labels,
                }
            )
        if not result[role]:
            raise SourceEvidenceError(f"no selected {role} polymer entity metadata derived")
    return result


def _connection_partner(
    row: object, columns: Mapping[str, int], ordinal: int
) -> dict[str, object] | None:
    prefix = f"ptnr{ordinal}"
    label_asym = _cif_value(row, columns, f"{prefix}_label_asym_id")
    auth_asym = _cif_value(row, columns, f"{prefix}_auth_asym_id")
    if label_asym is None and auth_asym is None:
        return None
    return {
        "auth_chain": auth_asym,
        "auth_seq_id": _cif_value(row, columns, f"{prefix}_auth_seq_id"),
        "auth_component_id": _cif_value(row, columns, f"{prefix}_auth_comp_id"),
        "label_asym_id": label_asym,
        "label_seq_id": _cif_value(row, columns, f"{prefix}_label_seq_id"),
        "label_component_id": _cif_value(row, columns, f"{prefix}_label_comp_id"),
        "atom_id": _cif_value(row, columns, f"{prefix}_label_atom_id"),
        "altloc_id": _cif_value(row, columns, f"pdbx_{prefix}_label_alt_id"),
        "insertion_code": _cif_value(row, columns, f"pdbx_{prefix}_PDB_ins_code"),
        "symmetry": _cif_value(row, columns, f"{prefix}_symmetry"),
    }


def _struct_conn_evidence(
    block: object, selected_auth_chains: set[str], selected_label_asym_ids: set[str]
) -> list[dict[str, object]]:
    table = block.find_mmcif_category("_struct_conn.")
    columns = {
        str(tag).removeprefix("_struct_conn."): index for index, tag in enumerate(table.tags)
    }
    records: list[dict[str, object]] = []
    for row in table:
        partners = [
            partner
            for ordinal in (1, 2, 3)
            if (partner := _connection_partner(row, columns, ordinal)) is not None
        ]
        involves_selected = any(
            partner["auth_chain"] in selected_auth_chains
            or partner["label_asym_id"] in selected_label_asym_ids
            for partner in partners
        )
        records.append(
            {
                "connection_id": _cif_value(row, columns, "id"),
                "connection_type": _cif_value(row, columns, "conn_type_id"),
                "partners": partners,
                "distance_angstrom": _cif_value(row, columns, "pdbx_dist_value"),
                "value_order": _cif_value(row, columns, "pdbx_value_order"),
                "role": _cif_value(row, columns, "pdbx_role"),
                "details": _cif_value(row, columns, "details"),
                "involves_selected_chain": involves_selected,
            }
        )
    return records


def derive_source_evidence(source_path: Path, model: Mapping[str, Any]) -> dict[str, object]:
    """Derive coordinate/deposition evidence without making chemistry decisions."""

    gemmi = _require_gemmi()
    try:
        document = gemmi.cif.read(str(source_path))
        block = document.sole_block()
        structure = gemmi.make_structure_from_block(block)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SourceEvidenceError(f"cannot parse official mmCIF with gemmi: {exc}") from exc
    if len(structure) != 1:
        raise SourceEvidenceError(f"expected one coordinate model, observed {len(structure)}")

    entities = _mapping(model.get("entities"), "model entities")
    segment_evidence: dict[str, list[dict[str, object]]] = {"receptor": [], "ligand": []}
    selected_auth_chains: set[str] = set()
    selected_label_asym_ids: set[str] = set()
    model_segments: dict[str, list[Mapping[str, Any]]] = {"receptor": [], "ligand": []}
    for role in ("receptor", "ligand"):
        entity = _mapping(entities.get(role), f"{role} entity")
        segments = [
            _mapping(item, f"{role} segment")
            for item in _sequence(entity.get("segments"), f"{role} segments")
        ]
        if not segments:
            raise SourceEvidenceError(f"{role} must declare at least one segment")
        model_segments[role] = segments
        for ordinal, segment in enumerate(segments, start=1):
            evidence = _polymer_segment_evidence(
                gemmi,
                structure,
                role=role,
                ordinal=ordinal,
                segment=segment,
            )
            segment_evidence[role].append(evidence)
            selected_auth_chains.add(str(segment.get("auth_chain", "")))
            selected_label_asym_ids.add(str(segment.get("label_asym_id", "")))

    declared_sequence, declared_entities = _declared_ligand_sequences(
        gemmi, structure, model_segments["ligand"]
    )
    observed_sequence = "".join(
        str(segment["derived_observed_sequence"]) for segment in segment_evidence["ligand"]
    )
    struct_conn = _struct_conn_evidence(block, selected_auth_chains, selected_label_asym_ids)
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "parser": {"name": "gemmi", "version": str(gemmi.__version__)},
        "model": {"count": len(structure), "selected_model_number": 1},
        "selected_chains": {
            "auth_chain_ids": sorted(selected_auth_chains),
            "polymer_label_asym_ids": sorted(selected_label_asym_ids),
        },
        "selected_polymer_entities": _selected_polymer_entity_metadata(
            block, structure, model_segments
        ),
        "polymer_segments": segment_evidence,
        "ligand_sequence": {
            "derived_declared_sequence": declared_sequence,
            "derived_observed_sequence": observed_sequence,
            "declared_entities": declared_entities,
        },
        "biological_assemblies": _assembly_evidence(structure),
        "selected_chain_non_water_heterogens": _selected_non_water_heterogens(
            gemmi, structure, selected_auth_chains
        ),
        "selected_chain_nonstandard_polymer_residues": (
            _selected_nonstandard_polymer_residues(gemmi, structure, selected_auth_chains)
        ),
        "selected_chain_altloc_residues": _selected_altlocs(gemmi, structure, selected_auth_chains),
        "struct_conn": {
            "all_source_records": struct_conn,
            "selected_connection_ids": [
                record["connection_id"]
                for record in struct_conn
                if record["involves_selected_chain"]
            ],
        },
        "interpretation_boundary": (
            "Source facts only; no chemistry, inclusion/exclusion, force-field, "
            "or eligibility decision is inferred."
        ),
    }


def declared_assembly_matches(
    evidence: Mapping[str, object], declaration: Mapping[str, object]
) -> bool:
    """Check an explicit case declaration against derived assembly/operator facts."""

    assembly_id = str(declaration.get("assembly_id", ""))
    operator_id = str(declaration.get("operator_id", ""))
    expected_labels = sorted(str(item) for item in declaration.get("label_asym_ids", []))
    for assembly_value in evidence.get("biological_assemblies", []):
        assembly = _mapping(assembly_value, "derived assembly")
        if str(assembly.get("assembly_id")) != assembly_id:
            continue
        if declaration.get("author_defined") is not None and bool(
            assembly.get("author_defined")
        ) != bool(declaration.get("author_defined")):
            return False
        for generator_value in assembly.get("generators", []):
            generator = _mapping(generator_value, "derived assembly generator")
            if sorted(str(item) for item in generator.get("label_asym_ids", [])) != expected_labels:
                continue
            for operator_value in generator.get("operators", []):
                operator = _mapping(operator_value, "derived assembly operator")
                if str(operator.get("operator_id")) != operator_id:
                    continue
                if declaration.get("operator") == "identity" and not operator.get("is_identity"):
                    return False
                return True
    return False


__all__ = [
    "EVIDENCE_SCHEMA",
    "SourceEvidenceError",
    "declared_assembly_matches",
    "derive_source_evidence",
]

#!/usr/bin/env python3
"""Strict, validation-local gemmi source audit for one SP2016 modeling entry.

The ``model.yaml`` files intentionally use the JSON subset of YAML.  Source
facts are independently derived from the pinned official mmCIF with gemmi,
provided by the project's ``.[preparation]`` extra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modeling.common.source_evidence import (
    SourceEvidenceError,
    declared_assembly_matches,
    derive_source_evidence,
)

SCHEMA_VERSION = 1
ELIGIBLE_OUTCOME = "eligible_preflight_passed"
REJECTED_OUTCOME = "rejected_expected"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON-compatible YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(
    checks: list[dict[str, str]],
    errors: list[str],
    name: str,
    condition: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "status": "pass" if condition else "fail", "detail": detail})
    if not condition:
        errors.append(f"{name}: {detail}")


def _validation_root(model_path: Path) -> Path:
    resolved = model_path.resolve()
    try:
        root = resolved.parents[3]
    except IndexError as exc:
        raise ValueError(f"model path is not under modeling/cases/<PDB>: {resolved}") from exc
    expected = root / "modeling" / "cases" / resolved.parent.name / "model.yaml"
    if resolved != expected:
        raise ValueError(f"unexpected model path layout: {resolved}")
    return root


def preflight_case(
    model_path: Path,
    source_root: Path | None = None,
    *,
    model_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one immutable source and its modeling decision contract."""

    model_path = model_path.resolve()
    root = _validation_root(model_path)
    checks: list[dict[str, str]] = []
    errors: list[str] = []
    warnings: list[str] = []

    try:
        model = _load_object(model_path) if model_override is None else dict(model_override)
        catalog = _load_object(root / "catalog" / "cases.json")
        outcomes = _load_object(root / "catalog" / "expected_outcomes.json")
    except ValueError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "preflight_status": "failed",
            "case_id": model_path.parent.name,
            "errors": [str(exc)],
            "checks": checks,
            "warnings": warnings,
        }

    case_id = str(model.get("case_id", ""))
    pdb_id = str(model.get("pdb_id", "")).upper()
    entries = {entry["case_id"]: entry for entry in catalog.get("cases", [])}
    expected_by_case = outcomes.get("cases", {})
    catalog_entry = entries.get(case_id)
    expected = expected_by_case.get(case_id)

    _check(
        checks,
        errors,
        "schema_version",
        model.get("schema_version") == SCHEMA_VERSION,
        f"expected {SCHEMA_VERSION}, observed {model.get('schema_version')!r}",
    )
    _check(
        checks,
        errors,
        "case_directory",
        model_path.parent.name == pdb_id,
        f"directory {model_path.parent.name!r} must equal PDB ID {pdb_id!r}",
    )
    _check(
        checks,
        errors,
        "catalog_membership",
        catalog_entry is not None,
        f"case {case_id!r} must occur exactly once in catalog",
    )
    _check(
        checks,
        errors,
        "expected_outcome_membership",
        expected is not None,
        f"case {case_id!r} must have an expected outcome",
    )

    status = str(model.get("eligibility", {}).get("status", ""))
    reason_codes = model.get("eligibility", {}).get("reason_codes", [])
    build = model.get("build", {})
    source = model.get("source", {})
    entities = model.get("entities", {})
    receptor = entities.get("receptor", {})
    ligand = entities.get("ligand", {})

    if catalog_entry is not None:
        _check(
            checks,
            errors,
            "catalog_pdb_id",
            catalog_entry.get("pdb_id") == pdb_id,
            "model and catalog PDB IDs must match",
        )
        _check(
            checks,
            errors,
            "catalog_status",
            catalog_entry.get("status") == status,
            "model and catalog eligibility must match",
        )
        _check(
            checks,
            errors,
            "catalog_reason_codes",
            catalog_entry.get("reason_codes") == reason_codes,
            "model and catalog reason codes must match in order",
        )
        _check(
            checks,
            errors,
            "catalog_source_hash",
            catalog_entry.get("source", {}).get("sha256") == source.get("sha256"),
            "model and catalog SHA-256 must match",
        )

    _check(
        checks,
        errors,
        "eligibility_value",
        status in {"eligible", "rejected"},
        f"observed {status!r}; allowed values are 'eligible' and 'rejected'",
    )
    _check(
        checks,
        errors,
        "peptide_identity",
        ligand.get("kind") == "peptide",
        "scored ligand must be explicitly typed as peptide",
    )
    _check(
        checks,
        errors,
        "entity_roles",
        bool(receptor.get("auth_chains")) and bool(ligand.get("auth_chains")),
        "receptor and ligand auth chains must be explicit",
    )
    _check(
        checks,
        errors,
        "entity_separation",
        set(receptor.get("auth_chains", [])).isdisjoint(ligand.get("auth_chains", [])),
        "receptor and peptide auth chains must be disjoint",
    )
    _check(
        checks,
        errors,
        "residue_selectors",
        bool(receptor.get("segments")) and bool(ligand.get("segments")),
        "declared and observed residue selectors must be explicit",
    )
    _check(
        checks,
        errors,
        "production_topology_claim",
        build.get("production_topology_present") is False,
        "validation must not claim that a production topology exists",
    )

    default_build = "not_implemented" if status == "eligible" else "blocked"
    expected_build = (
        str(expected.get("build_status", default_build))
        if isinstance(expected, dict)
        else default_build
    )
    allowed_builds = (
        {"not_implemented", "validation_local_proof_implemented"}
        if status == "eligible"
        else {"blocked"}
    )
    _check(
        checks,
        errors,
        "build_status_value",
        expected_build in allowed_builds,
        f"observed expected build status {expected_build!r}; "
        f"allowed for {status!r}: {sorted(allowed_builds)}",
    )
    _check(
        checks,
        errors,
        "build_boundary",
        build.get("status") == expected_build,
        f"observed {build.get('status')!r}; catalog expects {expected_build!r}",
    )
    if status == "rejected":
        _check(
            checks,
            errors,
            "rejection_reasons",
            bool(reason_codes),
            "rejected case requires at least one stable reason code",
        )

    if expected is not None:
        expected_preflight = expected.get("preflight_status")
        desired = ELIGIBLE_OUTCOME if status == "eligible" else REJECTED_OUTCOME
        _check(
            checks,
            errors,
            "expected_preflight",
            expected_preflight == desired,
            f"expected outcome must be {desired!r}",
        )

    relative_source = Path(str(source.get("path", "")))
    safe_relative = (
        bool(relative_source.parts)
        and not relative_source.is_absolute()
        and ".." not in relative_source.parts
    )
    _check(
        checks,
        errors,
        "source_path_is_relative",
        safe_relative,
        f"observed {str(relative_source)!r}; source paths must be relative and traversal-free",
    )
    source_base = root if source_root is None else source_root.resolve()
    source_candidate = source_base / (
        relative_source if source_root is None else relative_source.name
    )
    resolve_error: OSError | RuntimeError | None = None
    if safe_relative:
        try:
            source_path = source_candidate.resolve()
        except (OSError, RuntimeError) as exc:
            source_path = source_candidate
            resolve_error = exc
    else:
        source_path = source_candidate
    source_within_root = (
        safe_relative
        and resolve_error is None
        and source_path.is_relative_to(source_base)
    )
    _check(
        checks,
        errors,
        "source_path_within_root",
        source_within_root,
        f"resolved source {source_path} must remain within {source_base}"
        + (f"; resolution failed: {resolve_error}" if resolve_error else ""),
    )

    source_evidence: dict[str, object] = {
        "status": "not_derived",
        "reason": "source identity has not been admitted",
    }
    source_exists = source_within_root and source_path.is_file()
    if source_within_root:
        _check(checks, errors, "source_exists", source_exists, str(source_path))
    if source_exists:
        actual_size = source_path.stat().st_size
        actual_hash = _sha256(source_path)
        size_matches = actual_size == source.get("bytes")
        hash_matches = actual_hash == source.get("sha256")
        _check(
            checks,
            errors,
            "source_size",
            size_matches,
            f"expected {source.get('bytes')}, observed {actual_size}",
        )
        _check(
            checks,
            errors,
            "source_sha256",
            hash_matches,
            f"expected {source.get('sha256')}, observed {actual_hash}",
        )
        if size_matches and hash_matches:
            try:
                derived = derive_source_evidence(source_path, model)
            except SourceEvidenceError as exc:
                source_evidence = {"status": "failed", "error": str(exc)}
                _check(checks, errors, "source_evidence_derivation", False, str(exc))
            else:
                source_evidence = {"status": "derived", **derived}
                _check(
                    checks,
                    errors,
                    "source_evidence_derivation",
                    True,
                    f"derived with gemmi {derived['parser']['version']}",
                )
                derived_segments = derived["polymer_segments"]
                for role, entity in (("receptor", receptor), ("ligand", ligand)):
                    model_segments = entity.get("segments", [])
                    source_segments = derived_segments[role]
                    _check(
                        checks,
                        errors,
                        f"source_{role}_segment_count",
                        len(model_segments) == len(source_segments),
                        f"model declares {len(model_segments)}, "
                        f"source derives {len(source_segments)}",
                    )
                    for ordinal, (model_segment, source_segment) in enumerate(
                        zip(model_segments, source_segments, strict=False), start=1
                    ):
                        source_selector = source_segment["derived_observed_residue_selector"]
                        _check(
                            checks,
                            errors,
                            f"source_{role}_segment_{ordinal}_observed_selector",
                            model_segment.get("observed_residues") == source_selector,
                            f"model declares {model_segment.get('observed_residues')!r}; "
                            f"source derives {source_selector!r}",
                        )
                        source_complete = model_segment.get("declared_residues") == source_selector
                        _check(
                            checks,
                            errors,
                            f"source_{role}_segment_{ordinal}_completeness",
                            model_segment.get("complete") is source_complete,
                            f"model declares complete={model_segment.get('complete')!r}; "
                            f"source/model selectors imply {source_complete}",
                        )
                ligand_sequence = derived["ligand_sequence"]
                source_declared_sequence = ligand_sequence["derived_declared_sequence"]
                source_observed_sequence = ligand_sequence["derived_observed_sequence"]
                _check(
                    checks,
                    errors,
                    "source_ligand_declared_sequence",
                    ligand.get("declared_sequence") == source_declared_sequence,
                    f"model declares {ligand.get('declared_sequence')!r}; "
                    f"source derives {source_declared_sequence!r}",
                )
                _check(
                    checks,
                    errors,
                    "source_ligand_observed_sequence",
                    ligand.get("observed_sequence") == source_observed_sequence,
                    f"model declares {ligand.get('observed_sequence')!r}; "
                    f"source derives {source_observed_sequence!r}",
                )
                source_ligand_complete = source_declared_sequence == source_observed_sequence
                _check(
                    checks,
                    errors,
                    "source_ligand_sequence_completeness",
                    ligand.get("complete") is source_ligand_complete,
                    f"model declares complete={ligand.get('complete')!r}; "
                    f"source sequences imply {source_ligand_complete}",
                )
                assemblies = derived["biological_assemblies"]
                _check(
                    checks,
                    errors,
                    "source_biological_assembly_evidence",
                    bool(assemblies),
                    f"source derives {len(assemblies)} biological assembly record(s)",
                )
                declared_assembly = source.get("biological_assembly")
                if isinstance(declared_assembly, Mapping):
                    _check(
                        checks,
                        errors,
                        "source_biological_assembly_declaration",
                        declared_assembly_matches(derived, declared_assembly),
                        "case-declared assembly/operator must match source-derived evidence",
                    )
                selected_polymer_entities = derived["selected_polymer_entities"]
                _check(
                    checks,
                    errors,
                    "source_selected_polymer_entity_metadata",
                    bool(selected_polymer_entities["receptor"])
                    and bool(selected_polymer_entities["ligand"]),
                    "derived _entity id/description/details for selected receptor and ligand",
                )
                heterogens = derived["selected_chain_non_water_heterogens"]
                nonstandard_polymer = derived["selected_chain_nonstandard_polymer_residues"]
                altlocs = derived["selected_chain_altloc_residues"]
                connections = derived["struct_conn"]["all_source_records"]
                _check(
                    checks,
                    errors,
                    "source_selected_chain_heterogen_evidence",
                    isinstance(heterogens, list),
                    f"derived {len(heterogens)} selected-chain non-water non-polymer residue(s)",
                )
                _check(
                    checks,
                    errors,
                    "source_selected_chain_nonstandard_polymer_evidence",
                    isinstance(nonstandard_polymer, list),
                    f"derived {len(nonstandard_polymer)} selected-chain "
                    "nonstandard polymer residue(s)",
                )
                _check(
                    checks,
                    errors,
                    "source_selected_chain_altloc_evidence",
                    isinstance(altlocs, list),
                    f"derived {len(altlocs)} selected-chain non-water altloc residue(s)",
                )
                _check(
                    checks,
                    errors,
                    "source_struct_conn_evidence",
                    isinstance(connections, list),
                    f"derived {len(connections)} source struct_conn record(s)",
                )
        else:
            source_evidence = {
                "status": "not_derived",
                "reason": "source size/SHA-256 identity checks failed",
            }
    elif not source_within_root:
        source_evidence = {
            "status": "not_derived",
            "reason": "source path failed lexical or resolved-root admission",
        }

    outcome = "failed"
    if not errors:
        outcome = ELIGIBLE_OUTCOME if status == "eligible" else REJECTED_OUTCOME
    if status == "eligible" and build.get("status") == "not_implemented":
        warnings.append(
            "Source/model preflight passed; case-specific topology build is not implemented."
        )
    elif status == "eligible":
        warnings.append(
            "Source/model preflight passed; a validation-local builder and sealing "
            "workflow are implemented, but no production MD is claimed."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "validation_id": catalog.get("validation_id"),
        "case_id": case_id,
        "pdb_id": pdb_id,
        "eligibility": status,
        "preflight_status": outcome,
        "source": {
            "path": str(source_path),
            "sha256": source.get("sha256"),
            "revision_date": source.get("revision_date"),
        },
        "source_evidence": source_evidence,
        "modeling": {
            "build_status": build.get("status"),
            "production_topology_present": build.get("production_topology_present"),
        },
        "reason_codes": reason_codes,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="JSON-compatible model.yaml")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Optional directory containing immutable <PDB>.cif files",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON")
    parser.add_argument("--json-output", type=Path, help="Also write structured JSON")
    parser.add_argument(
        "--require-eligible",
        action="store_true",
        help="Return exit code 3 for a scientifically rejected case",
    )
    return parser


def cli(default_model: Path | None = None, argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model_path = args.model or default_model
    if model_path is None:
        raise SystemExit("--model is required")
    result = preflight_case(model_path, args.source_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload, encoding="utf-8")
    if args.json:
        sys.stdout.write(payload)
    else:
        print(f"{result.get('case_id')}: {result['preflight_status']}")
        for error in result.get("errors", []):
            print(f"  ERROR: {error}", file=sys.stderr)
        for warning in result.get("warnings", []):
            print(f"  NOTE: {warning}")
    if result["preflight_status"] == "failed":
        return 1
    if args.require_eligible and result.get("eligibility") != "eligible":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

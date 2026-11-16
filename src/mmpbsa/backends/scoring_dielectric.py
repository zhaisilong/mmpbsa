"""Resolve scoring dielectric inputs and preserve the applied value for analysis."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from mmpbsa.core import (
    AUTO_DIELECTRIC_TOKEN,
    PreparedBindingView,
    resolve_internal_dielectric,
)

from .errors import BackendConfigurationError


def resolve_scoring_dielectric(
    prepared: PreparedBindingView,
    *,
    control_text: str,
    native_inputs: Mapping[str, str] | None,
) -> tuple[str, Mapping[str, str] | None, Mapping[str, object]]:
    """Resolve the automatic 2/3/4 policy from explicit entity metadata."""

    native = None if native_inputs is None else dict(native_inputs)
    contains_token = AUTO_DIELECTRIC_TOKEN in control_text or any(
        AUTO_DIELECTRIC_TOKEN in text for text in (() if native is None else native.values())
    )
    if not contains_token:
        values = [
            float(match.group(1))
            for text in (control_text, *((native or {}).values()))
            for match in re.finditer(
                r"\b(?:intdiel|indi)\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)\b",
                text,
                flags=re.IGNORECASE,
            )
        ]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise BackendConfigurationError(
                "scoring internal dielectric must be positive and finite"
            )
        if values and any(value != values[0] for value in values[1:]):
            raise BackendConfigurationError(
                "scoring control and native inputs disagree on internal dielectric"
            )
        return (
            control_text,
            native,
            {
                "mode": "fixed",
                "resolved_internal_dielectric": values[0] if values else None,
                "fallback": {"value": 4.0, "used": False},
                "evidence": {
                    "scope": "not_evaluated_for_fixed_value",
                    "ligand_formal_charge": None,
                    "interface_chemistry_classes": None,
                },
                "warnings": (
                    () if values else ("fixed internal dielectric was not found in scoring inputs",)
                ),
            },
        )

    binding = prepared.binding
    ligand_entities = tuple(
        prepared.entity(entity_id)
        for entity_id in binding.ligand_entity_ids
    )
    receptor_entities = tuple(
        prepared.entity(entity_id)
        for entity_id in binding.receptor_entity_ids
    )
    include_ligand_interface = any(entity.kind == "peptide" for entity in ligand_entities)
    interface_entities = (
        receptor_entities + ligand_entities if include_ligand_interface else receptor_entities
    )

    evidence_warnings: list[str] = []
    charge_values: list[int] = []
    for entity in ligand_entities:
        raw_charge = entity.metadata.get("formal_charge")
        if isinstance(raw_charge, bool) or not isinstance(raw_charge, int):
            evidence_warnings.append(
                "automatic dielectric ligand-charge scope "
                f"{entity.entity_id!r}: metadata.formal_charge is missing "
                "or is not an integer"
            )
            continue
        charge_values.append(raw_charge)
    ligand_formal_charge = (
        sum(charge_values) if len(charge_values) == len(ligand_entities) else None
    )

    interface_classes_list: list[str] = []
    interface_complete = True
    for entity in interface_entities:
        raw_classes = entity.metadata.get("interface_chemistry_classes")
        if (
            not isinstance(raw_classes, Sequence)
            or isinstance(raw_classes, (str, bytes, bytearray))
            or not raw_classes
            or any(not isinstance(item, str) or not item.strip() for item in raw_classes)
        ):
            interface_complete = False
            evidence_warnings.append(
                "automatic dielectric interface scope "
                f"{entity.entity_id!r}: "
                "metadata.interface_chemistry_classes is missing or is not "
                "a non-empty string sequence"
            )
            continue
        interface_classes_list.extend(str(item).strip().lower() for item in raw_classes)
    interface_classes = tuple(interface_classes_list) if interface_complete else None
    resolved, policy_warnings = resolve_internal_dielectric(
        ligand_formal_charge=ligand_formal_charge,
        interface_classes=interface_classes,
    )
    warnings = tuple(dict.fromkeys((*evidence_warnings, *policy_warnings)))
    replacement = f"{resolved:g}"
    rendered_control = control_text.replace(
        AUTO_DIELECTRIC_TOKEN,
        replacement,
    )
    rendered_native = (
        None
        if native is None
        else {key: text.replace(AUTO_DIELECTRIC_TOKEN, replacement) for key, text in native.items()}
    )
    fallback_used = bool(policy_warnings)
    audit: Mapping[str, object] = {
        "mode": "automatic",
        "resolved_internal_dielectric": resolved,
        "fallback": {"value": 4.0, "used": fallback_used},
        "evidence": {
            "scope": {
                "ligand_charge_entity_ids": tuple(entity.entity_id for entity in ligand_entities),
                "interface_entity_ids": tuple(entity.entity_id for entity in interface_entities),
                "interface_policy": (
                    "receptor_and_ligand_for_peptide"
                    if include_ligand_interface
                    else "receptor_only_for_small_molecule"
                ),
            },
            "ligand_formal_charge": ligand_formal_charge,
            "interface_chemistry_classes": interface_classes,
        },
        "warnings": warnings,
    }
    return rendered_control, rendered_native, audit

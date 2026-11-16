"""Stage-level routing for legacy and engine-neutral trajectory adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from mmpbsa.core import PreparedBindingView, PreparedTrajectoryView, StageInput

from .api import BackendReceipt, CommandRunner
from .errors import BackendConfigurationError
from .trajectory import TrajectoryPreparationRequest, create_trajectory_backend
from .trajectory_geometry import PREPROCESSING_REVISION
from .trajectory_v3 import (
    TrajectoryPreparationV3Request,
    create_trajectory_v3_backend,
)

_V3_CONTRACT = "mmpbsa.trajectory.v3"


def execute_trajectory_adapter(
    *,
    science: Mapping[str, object],
    config: Mapping[str, object],
    prepared: PreparedBindingView | PreparedTrajectoryView,
    upstream: StageInput,
    output_dir: Path,
    runner: CommandRunner | None,
) -> BackendReceipt:
    """Select the adapter from executable science, never runtime hints alone."""

    contract = science.get("adapter_contract")
    if contract == _V3_CONTRACT:
        if science.get("preprocessing_revision", PREPROCESSING_REVISION) != PREPROCESSING_REVISION:
            raise BackendConfigurationError("unsupported trajectory preprocessing_revision")
        if not isinstance(prepared, PreparedTrajectoryView):
            raise BackendConfigurationError(
                "trajectory v3 requires a verified PreparedTrajectoryView"
            )
        topology_role = _text(
            science.get("full_system_topology_role", "md_topology"),
            "trajectory.full_system_topology_role",
        )
        coordinates_role = _text(
            science.get("full_system_coordinates_role", "md_coordinates"),
            "trajectory.full_system_coordinates_role",
        )
        if topology_role != "md_topology" or coordinates_role != "md_coordinates":
            raise BackendConfigurationError(
                "trajectory v3 requires md_topology and md_coordinates roles"
            )
        try:
            full_topology = prepared.artifact(topology_role)
            full_coordinates = prepared.artifact(coordinates_role)
        except KeyError as exc:
            raise BackendConfigurationError(
                "trajectory v3 requires prepared MD topology and coordinates"
            ) from exc
        v3_backend = create_trajectory_v3_backend(config, runner=runner)
        source_contract = _mapping(
            science.get("source_trajectory_contract"),
            "trajectory.source_trajectory_contract",
        )
        first_time_ps = _number(
            source_contract.get("first_time_ps"),
            "source_trajectory_contract.first_time_ps",
        )
        return v3_backend.run(
            TrajectoryPreparationV3Request(
                full_trajectory=_artifact(
                    upstream,
                    _text(
                        science.get("source_trajectory_artifact", "full_trajectory"),
                        "trajectory.source_trajectory_artifact",
                    ),
                ),
                full_system_topology=full_topology,
                full_system_coordinates=full_coordinates,
                dry_complex_topology=prepared.topology_artifact("complex"),
                source_md_atom_indices=prepared.source_md_atom_indices,
                expected_atom_order_signature=prepared.atom_order_signature,
                residue_crosswalk=prepared.residue_crosswalk,
                output_dir=output_dir,
                start_frame=_integer(science.get("start_frame", 1), "start_frame"),
                end_frame=_optional_integer(science.get("end_frame"), "end_frame"),
                stride=_integer(science.get("stride", 1), "stride"),
                source_first_time_ps=first_time_ps,
                source_interval_ps=_number(
                    source_contract.get("interval_ps"),
                    "source_trajectory_contract.interval_ps",
                ),
                source_interval_steps=_integer(
                    source_contract.get("interval_steps"),
                    "source_trajectory_contract.interval_steps",
                ),
                expected_source_frames=_integer(
                    source_contract.get("expected_frames"),
                    "source_trajectory_contract.expected_frames",
                ),
                autoimage_anchor_mask=_selection(
                    science.get("autoimage_anchor_mask"), prepared, "autoimage_anchor_mask"
                ),
                autoimage_fixed_mask=_selection(
                    science.get("autoimage_fixed_mask"), prepared, "autoimage_fixed_mask"
                ),
                fit_mask=_selection(science.get("fit_mask"), prepared, "fit_mask"),
                expected_frames=_optional_integer(
                    science.get("expected_frames"), "expected_frames"
                ),
            )
        )
    if contract is not None:
        raise BackendConfigurationError(f"unsupported trajectory adapter contract: {contract!r}")
    legacy_backend = create_trajectory_backend(config, runner=runner)
    return legacy_backend.run(
        TrajectoryPreparationRequest(
            source_tpr=_artifact(
                upstream,
                _text(science.get("source_tpr_artifact", "tpr"), "source_tpr_artifact"),
            ),
            source_trajectory=_artifact(
                upstream,
                _text(
                    science.get("source_trajectory_artifact", "xtc"),
                    "source_trajectory_artifact",
                ),
            ),
            index=prepared.artifact(_text(science.get("index_role", "index"), "index_role")),
            complex_topology=prepared.topology_artifact("complex"),
            output_dir=output_dir,
            group_name=_text(science.get("group_name"), "group_name"),
            start_frame=_integer(science.get("start_frame", 1), "start_frame"),
            end_frame=_optional_integer(science.get("end_frame"), "end_frame"),
            stride=_integer(science.get("stride", 1), "stride"),
            autoimage_anchor_mask=_selection(
                science.get("autoimage_anchor_mask"), prepared, "autoimage_anchor_mask"
            ),
            autoimage_fixed_mask=_selection(
                science.get("autoimage_fixed_mask"), prepared, "autoimage_fixed_mask"
            ),
            fit_mask=_selection(science.get("fit_mask"), prepared, "fit_mask"),
            expected_frames=_optional_integer(science.get("expected_frames"), "expected_frames"),
        )
    )


def _selection(
    value: object,
    prepared: PreparedBindingView | PreparedTrajectoryView,
    label: str,
) -> str | None:
    if value is None:
        return None
    text = _text(value, f"trajectory.{label}")

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        try:
            indices = prepared.binding_selection(name).scoring_atom_indices
        except KeyError as exc:
            raise BackendConfigurationError(f"unknown primary-binding selection: {name!r}") from exc
        if not indices:
            raise BackendConfigurationError(
                f"primary-binding selection {name!r} has no scoring atoms"
            )
        return "@" + ",".join(str(index) for index in indices)

    result = re.sub(r"\{\{selection:(?P<name>[A-Za-z0-9_.-]+)\}\}", replace, text)
    if "{{" in result or "}}" in result:
        raise BackendConfigurationError(
            f"trajectory.{label} contains an invalid selection placeholder"
        )
    return result


def _artifact(upstream: StageInput, role: str) -> Path:
    try:
        path = Path(upstream.artifacts[role]).resolve()
    except KeyError as exc:
        raise BackendConfigurationError(
            f"upstream {upstream.task.label} has no artifact {role!r}"
        ) from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise BackendConfigurationError(f"upstream artifact is missing or empty: {path}")
    return path


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise BackendConfigurationError(f"trajectory.{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BackendConfigurationError(f"trajectory.{label} must be a mapping")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendConfigurationError(f"trajectory.{label} must be a number")
    return float(value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BackendConfigurationError(f"trajectory.{label} must be a positive integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label)

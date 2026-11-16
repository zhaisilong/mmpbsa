"""Strict, runtime-free validation of OpenMM worker receipts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import BackendOutputError

_AMBER_INPUT_FORMAT = "mmpbsa.openmm.input.amber.v1"

_MAX_SEED = 2**31 - 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _request_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _derive_seed(master_seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _MAX_SEED + 1


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BackendOutputError(f"{label} must be a string-keyed mapping")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise BackendOutputError(f"{label} fields differ from the contract")


def _validate_artifacts(receipt: Mapping[str, object], expected: Mapping[str, Path]) -> None:
    raw = _mapping(receipt.get("artifacts"), "OpenMM receipt artifacts")
    roles = set(expected).difference({"worker_receipt"})
    if set(raw) != roles:
        raise BackendOutputError("OpenMM worker receipt artifact roles differ from contract")
    for role in roles:
        item = _mapping(raw[role], f"OpenMM receipt artifact {role}")
        _exact_fields(item, {"path", "size", "sha256"}, f"artifact {role}")
        raw_path, size, digest = item["path"], item["size"], item["sha256"]
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or "\x00" in raw_path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise BackendOutputError(
                f"OpenMM worker receipt artifact record is invalid for {role!r}"
            )
        path = expected[role]
        if path.is_symlink() or Path(raw_path).resolve() != path.resolve():
            raise BackendOutputError(f"OpenMM worker receipt path differs for artifact {role!r}")
        if not path.is_file() or size != path.stat().st_size or digest != _sha256_file(path):
            raise BackendOutputError(
                f"OpenMM worker receipt integrity differs for artifact {role!r}"
            )


def _expected_platform_properties(request: Any) -> Mapping[str, str]:
    if request.platform.name == "CUDA":
        return {
            "DeviceIndex": str(request.platform.device_index),
            "Precision": str(request.platform.precision),
            "DeterministicForces": "true",
        }
    if request.platform.name == "CPU":
        return {
            "Threads": str(request.platform.threads),
            "DeterministicForces": ("true" if request.platform.deterministic_forces else "false"),
        }
    return {}


def _validate_provenance(receipt: Mapping[str, object], request: Any) -> None:
    identity = _mapping(receipt.get("openmm"), "OpenMM receipt identity")
    if (
        set(identity) != {"version"}
        or not isinstance(identity.get("version"), str)
        or not str(identity["version"]).strip()
    ):
        raise BackendOutputError("OpenMM worker receipt identity is invalid")
    if (
        request.expected_openmm_version is not None
        and identity["version"] != request.expected_openmm_version
    ):
        raise BackendOutputError("OpenMM worker version differs from reserved probe")
    if (
        request.expected_probe_sha256 is not None
        and receipt.get("probe_sha256") != request.expected_probe_sha256
    ):
        raise BackendOutputError("OpenMM worker probe differs from reserved probe")

    platform = _mapping(receipt.get("platform"), "OpenMM receipt platform")
    _exact_fields(platform, {"name", "properties", "diagnostics"}, "platform")
    if platform["name"] != request.platform.name:
        raise BackendOutputError("OpenMM worker receipt platform differs from request")
    properties = _mapping(platform["properties"], "OpenMM platform properties")
    expected = _expected_platform_properties(request)
    if set(properties) != set(expected) or any(
        not isinstance(properties.get(key), str)
        or (
            properties[key] != value
            if key == "DeviceIndex"
            else str(properties[key]).lower() != value.lower()
        )
        for key, value in expected.items()
    ):
        raise BackendOutputError("OpenMM worker receipt platform properties differ from request")
    diagnostics = _mapping(platform["diagnostics"], "OpenMM platform diagnostics")
    if set(diagnostics).difference({"DeviceName"}) or any(
        not isinstance(value, str) or not value.strip() for value in diagnostics.values()
    ):
        raise BackendOutputError("OpenMM worker receipt diagnostics are invalid")

    system_digest = receipt.get("system_sha256")
    if not isinstance(system_digest, str) or re.fullmatch(r"[0-9a-f]{64}", system_digest) is None:
        raise BackendOutputError("OpenMM worker receipt System digest is invalid")
    checkpoint = _mapping(receipt.get("checkpoint_provenance"), "OpenMM checkpoint provenance")
    _exact_fields(
        checkpoint,
        {"openmm_version", "system_sha256", "platform"},
        "checkpoint provenance",
    )
    checkpoint_platform = _mapping(checkpoint["platform"], "OpenMM checkpoint platform")
    if (
        checkpoint["openmm_version"] != identity["version"]
        or checkpoint["system_sha256"] != system_digest
        or dict(checkpoint_platform) != {"name": platform["name"], "properties": dict(properties)}
    ):
        raise BackendOutputError("OpenMM worker receipt checkpoint provenance is inconsistent")

    seeds = _mapping(receipt.get("seeds"), "OpenMM receipt seeds")
    expected_seeds = {
        "master": request.seed,
        "integrator": _derive_seed(request.seed, "integrator"),
        "barostat": _derive_seed(request.seed, "barostat"),
        "velocities": _derive_seed(request.seed, "velocities"),
    }
    if set(seeds) != set(expected_seeds) or any(
        isinstance(seeds.get(key), bool)
        or not isinstance(seeds.get(key), int)
        or seeds[key] != value
        for key, value in expected_seeds.items()
    ):
        raise BackendOutputError("OpenMM worker receipt seeds differ from request")

    system = _mapping(receipt.get("system"), "OpenMM receipt System")
    _exact_fields(system, {"particles", "constraints", "forces"}, "System")
    particles, constraints, forces = (
        system["particles"],
        system["constraints"],
        system["forces"],
    )
    if (
        isinstance(particles, bool)
        or not isinstance(particles, int)
        or particles < 1
        or isinstance(constraints, bool)
        or not isinstance(constraints, int)
        or constraints < 0
        or isinstance(forces, (str, bytes, bytearray))
        or not isinstance(forces, Sequence)
        or not forces
        or any(not isinstance(force, str) or not force.strip() for force in forces)
    ):
        raise BackendOutputError("OpenMM worker receipt System is invalid")
    if request.phase.kind == "em" and constraints != 0:
        raise BackendOutputError("OpenMM EM worker receipt must report an unconstrained System")


def _validate_input(receipt: Mapping[str, object], request: Any) -> None:
    input_identity = _mapping(receipt.get("input"), "OpenMM Amber receipt input")
    _exact_fields(
        input_identity, {"format", "topology_sha256", "coordinates_sha256"}, "Amber input"
    )
    if dict(input_identity) != {
        "format": _AMBER_INPUT_FORMAT,
        "topology_sha256": request.topology_sha256,
        "coordinates_sha256": request.coordinates_sha256,
    }:
        raise BackendOutputError("OpenMM Amber receipt input differs from request")
    if (
        _sha256_file(request.topology) != request.topology_sha256
        or _sha256_file(request.coordinates) != request.coordinates_sha256
    ):
        raise BackendOutputError("OpenMM Amber receipt input snapshot changed")


def validate_md_receipt(
    receipt: Mapping[str, object],
    request: Any,
    *,
    schema: str,
    artifacts: Mapping[str, Path],
) -> None:
    required = {
        "schema_version",
        "status",
        "request_sha256",
        "openmm",
        "platform",
        "checkpoint_provenance",
        "system_sha256",
        "system",
        "phase",
        "seeds",
        "artifacts",
    }
    if request.input_format == _AMBER_INPUT_FORMAT:
        required.update({"input", "probe_sha256"})
    _exact_fields(receipt, required, "OpenMM worker receipt")
    if receipt["schema_version"] != schema or receipt["status"] != "completed":
        raise BackendOutputError("OpenMM worker receipt is not completed")
    if receipt["request_sha256"] != _request_digest(request.as_dict()):
        raise BackendOutputError("OpenMM worker receipt belongs to a different MD request")
    _validate_provenance(receipt, request)
    if request.input_format == _AMBER_INPUT_FORMAT:
        _validate_input(receipt, request)

    phase = _mapping(receipt["phase"], "OpenMM worker receipt phase")
    _exact_fields(
        phase,
        {
            "phase_id",
            "kind",
            "chunk_id",
            "start_step",
            "end_step",
            "executed_steps",
            "emit_initial_frame",
            "frame_count",
            "maximum_force_kj_mol_nm",
        },
        "OpenMM worker receipt phase",
    )
    exact = {
        "phase_id": request.phase.phase_id,
        "kind": request.phase.kind,
        "chunk_id": request.chunk_id,
        "executed_steps": request.phase.steps,
        "emit_initial_frame": request.emit_initial_frame,
    }
    if any(phase[key] != value for key, value in exact.items()):
        raise BackendOutputError("OpenMM worker receipt phase differs from request")
    start, end = phase["start_step"], phase["end_step"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end - start != request.phase.steps
        or (request.checkpoint_input is None and start != 0)
    ):
        raise BackendOutputError("OpenMM worker receipt step interval differs from request")
    expected_frames = (
        1
        if request.phase.kind == "em"
        else request.phase.steps // request.phase.trajectory_interval_steps
        + int(request.emit_initial_frame)
    )
    frames = phase["frame_count"]
    if isinstance(frames, bool) or not isinstance(frames, int) or frames != expected_frames:
        raise BackendOutputError("OpenMM worker receipt frame count differs from request")
    maximum_force = phase["maximum_force_kj_mol_nm"]
    if request.phase.kind == "em":
        if (
            isinstance(maximum_force, bool)
            or not isinstance(maximum_force, (int, float))
            or not math.isfinite(float(maximum_force))
            or not 0
            <= float(maximum_force)
            <= float(request.phase.maximum_force_threshold_kj_mol_nm)
        ):
            raise BackendOutputError(
                "OpenMM EM receipt maximum force exceeds the request threshold"
            )
    elif maximum_force is not None:
        raise BackendOutputError("dynamic OpenMM receipt cannot report EM force")
    _validate_artifacts(receipt, artifacts)


def validate_merge_receipt(
    receipt: Mapping[str, object],
    request: Any,
    *,
    schema: str,
    artifacts: Mapping[str, Path],
) -> None:
    required = {
        "schema_version",
        "status",
        "request_sha256",
        "frame_count",
        "energy_rows",
        "artifacts",
    }
    if request.input_format == _AMBER_INPUT_FORMAT:
        required.update({"openmm", "input", "probe_sha256"})
    _exact_fields(receipt, required, "OpenMM merge receipt")
    if receipt["schema_version"] != schema or receipt["status"] != "completed":
        raise BackendOutputError("OpenMM merge receipt is not completed")
    if receipt["request_sha256"] != _request_digest(request.as_dict()):
        raise BackendOutputError("OpenMM merge receipt belongs to a different merge request")
    if request.input_format == _AMBER_INPUT_FORMAT:
        identity = _mapping(receipt["openmm"], "OpenMM merge identity")
        _exact_fields(identity, {"version"}, "merge identity")
        if identity["version"] != request.expected_openmm_version:
            raise BackendOutputError("OpenMM merge worker version differs from reserved probe")
        if receipt["probe_sha256"] != request.expected_probe_sha256:
            raise BackendOutputError("OpenMM merge worker probe differs from reserved probe")
        _validate_input(receipt, request)
    frames, rows = receipt["frame_count"], receipt["energy_rows"]
    if isinstance(frames, bool) or not isinstance(frames, int) or frames != request.expected_frames:
        raise BackendOutputError("OpenMM merge receipt has an invalid frame count")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise BackendOutputError("OpenMM merge receipt has an invalid energy row count")
    _validate_artifacts(receipt, artifacts)

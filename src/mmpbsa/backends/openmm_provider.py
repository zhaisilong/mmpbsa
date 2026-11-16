"""OpenMM-only API-v2 provider and phase/chunk StageService."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from mmpbsa.core import (
    BackendCompatibility,
    PreparedMdView,
    ResourceRequest,
    StageRequest,
    StageResult,
)

from ._support import clean_output_dir
from .api import CommandRecord, CommandRunner
from .errors import BackendConfigurationError, BackendOutputError
from .openmm import (
    OPENMM_AMBER_INPUT_FORMAT,
    OPENMM_MERGE_RECEIPT_SCHEMA,
    OPENMM_RECEIPT_SCHEMA,
    OpenMMIdentity,
    OpenMMMdRequest,
    OpenMMMergeRequest,
    OpenMMPhaseSpec,
    OpenMMPlatformSpec,
    create_openmm_md_backend,
    openmm_artifact_paths,
    openmm_merge_artifact_paths,
    validate_openmm_md_worker_receipt,
    validate_openmm_merge_worker_receipt,
)

PROVIDER_ID = "openmm"
PACKAGE_VERSION = "0.7.0"
MD_PHASE_PLAN_SCHEMA = "mmpbsa.md-phase-plan.v2"
MD_ENGINE_PROFILE = "openmm_8_5@2"
ADAPTER_CONTRACT = "mmpbsa.backends.openmm.md.v2"
_EM_MINIMIZATION_TOLERANCE_KJ_MOL_NM = 10.0

_EXPECTED_SYSTEM = {
    "nonbonded_method": "particle_mesh_ewald",
    "nonbonded_cutoff_nm": 1.0,
    "ewald_error_tolerance": 1.0e-5,
    "dispersion_correction": True,
    "constraints": "hydrogen_bonds",
    "rigid_water": True,
    "constraint_tolerance": 1.0e-5,
    "center_of_mass_removal_interval_steps": 100,
}

# Importing preparation dependencies may populate unrelated process variables
# (for example RDKit sets RDBASE).  Only inherited values that can select the
# Python/OpenMM runtime or alter numerical execution belong to worker identity.
# Every explicitly configured worker environment entry is always included.
_WORKER_INHERITED_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONNOUSERSITE",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "LD_AUDIT",
        "DYLD_LIBRARY_PATH",
        "GOTO_NUM_THREADS",
        "OPENMM_PLUGIN_DIR",
        "OPENMM_CPU_THREADS",
        "OPENMM_DEFAULT_PLATFORM",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CUDA_PATH",
        "OPENCL_VENDOR_PATH",
        "OCL_ICD_VENDORS",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
    }
)
_WORKER_INHERITED_ENV_PREFIXES = (
    "OMP_",
    "MKL_",
    "KMP_",
    "GOMP_",
    "OPENBLAS_",
    "BLIS_",
    "NUMEXPR_",
    "VECLIB_",
)


@dataclass(frozen=True, slots=True)
class _PhasePlan:
    phase_id: str
    kind: str
    source: Mapping[str, object]
    phase: OpenMMPhaseSpec
    transaction_chunk_steps: int | None = None
    emit_initial_frame: bool = True
    expected_trajectory_frames: int | None = None


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BackendConfigurationError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BackendConfigurationError(f"{label} must be a sequence")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BackendConfigurationError(f"{label} must be a positive integer")
    return value


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendConfigurationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise BackendConfigurationError(f"{label} must be finite and positive")
    return result


def _strict_fields(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    missing = required.difference(value)
    unknown = set(value).difference(required | optional)
    details = []
    if missing:
        details.append("missing " + ", ".join(sorted(missing)))
    if unknown:
        details.append("unknown " + ", ".join(sorted(unknown)))
    if details:
        raise BackendConfigurationError(f"invalid {label}: " + "; ".join(details))


def _effective_config(runtime: Mapping[str, object]) -> dict[str, object]:
    nested = runtime.get("md")
    if nested is None:
        return dict(runtime)
    result = dict(_mapping(nested, "runtime.md"))
    for key in (
        "environment",
        "threads_per_job",
        "gpu_id",
        "gpu_enabled",
        "probe_cwd",
        "_stage_contract",
        "_md_engine_profile",
    ):
        if key in runtime and key not in result:
            result[key] = runtime[key]
    return result


def _platform_name(config: Mapping[str, object]) -> str:
    raw = config.get("platform")
    if isinstance(raw, str):
        name = raw
    elif isinstance(raw, Mapping):
        name = str(raw.get("name", ""))
    else:
        raise BackendConfigurationError(
            "OpenMM runtime must explicitly select platform CUDA, CPU, or Reference"
        )
    if name not in {"CUDA", "CPU", "Reference"}:
        raise BackendConfigurationError("OpenMM runtime platform must be CUDA, CPU, or Reference")
    return name


def _platform_spec(config: Mapping[str, object], *, require_lease: bool) -> OpenMMPlatformSpec:
    name = _platform_name(config)
    nested = config.get("platform")
    values = dict(nested) if isinstance(nested, Mapping) else {}
    threads = _positive_integer(
        config.get("threads_per_job", values.get("threads", 1)),
        "OpenMM threads_per_job",
    )
    deterministic = values.get(
        "deterministic_forces",
        config.get("deterministic_forces", True),
    )
    if not isinstance(deterministic, bool):
        raise BackendConfigurationError("OpenMM deterministic_forces must be boolean")
    if name == "CUDA":
        precision = values.get("precision", config.get("precision"))
        if precision is None:
            raise BackendConfigurationError(
                "OpenMM CUDA requires explicit precision=single|mixed|double"
            )
        configured_device = values.get("device_index")
        leased_device = config.get("gpu_id")
        if configured_device is not None:
            raise BackendConfigurationError(
                "OpenMM CUDA device_index is scheduler-owned; configure scheduler.gpu_ids"
            )
        if require_lease and leased_device is None:
            raise BackendConfigurationError(
                "OpenMM CUDA execution requires a scheduler-leased gpu_id"
            )
        # Compatibility validation happens before the scheduler lease.  A
        # sentinel is used only to validate non-device platform controls.
        device = "__scheduler_lease__" if leased_device is None else str(leased_device)
        return OpenMMPlatformSpec(
            name=name,
            device_index=device,
            precision=str(precision),
            deterministic_forces=deterministic,
            threads=threads,
        )
    if config.get("gpu_id") is not None:
        raise BackendConfigurationError(f"OpenMM {name} cannot consume a scheduler GPU lease")
    return OpenMMPlatformSpec(
        name=name,
        deterministic_forces=deterministic,
        threads=(1 if name == "Reference" else threads),
    )


def _validate_runtime_contract(config: Mapping[str, object]) -> None:
    contract = config.get("_stage_contract")
    if contract != MD_PHASE_PLAN_SCHEMA:
        raise BackendConfigurationError(
            f"OpenMM provider requires stage contract {MD_PHASE_PLAN_SCHEMA!r}"
        )
    profile = config.get("_md_engine_profile")
    if profile != MD_ENGINE_PROFILE:
        raise BackendConfigurationError(
            f"OpenMM provider requires md_engine_profile={MD_ENGINE_PROFILE!r}"
        )


def _validate_identity(identity: OpenMMIdentity, platform: OpenMMPlatformSpec) -> None:
    match = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:[.+-][0-9A-Za-z.-]+)?",
        identity.version,
    )
    if match is None:
        raise BackendOutputError(f"cannot parse OpenMM worker version {identity.version!r}")
    major, minor, patch = (int(item) for item in match.groups()[:3])
    if (major, minor) != (8, 5) or patch < 1:
        raise BackendConfigurationError(
            f"openmm_8_5@2 requires OpenMM >=8.5.1,<8.6, found {identity.version}"
        )
    if platform.name not in identity.platforms:
        raise BackendConfigurationError(
            f"requested OpenMM platform {platform.name!r} is unavailable; "
            f"available: {', '.join(identity.platforms) or '(none)'}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _execution_environment_digest(backend: object) -> str:
    worker = backend.worker  # type: ignore[attr-defined]
    executable = shutil.which(worker.python_executable)
    if executable is None:
        raise BackendConfigurationError(
            f"OpenMM worker executable is unavailable: {worker.python_executable}"
        )
    interpreter = Path(executable).resolve()
    effective_environment = {
        name: value
        for name, value in os.environ.items()
        if name in _WORKER_INHERITED_ENV_NAMES or name.startswith(_WORKER_INHERITED_ENV_PREFIXES)
    }
    effective_environment.update(worker.environment)
    return _sha256_json(
        {
            "python": str(interpreter),
            "python_sha256": _sha256_file(interpreter),
            "environment": effective_environment,
        }
    )


def _compatibility_from_probe(
    identity: OpenMMIdentity,
    platform: OpenMMPlatformSpec,
    backend: object,
) -> BackendCompatibility:
    _validate_identity(identity, platform)
    precision = platform.precision or "native"
    deterministic = str(platform.deterministic_forces).lower()
    probe_digest = _sha256_json(identity.raw)
    environment_digest = _execution_environment_digest(backend)
    return BackendCompatibility(
        provider_id=PROVIDER_ID,
        adapter_contract=ADAPTER_CONTRACT,
        engine=(
            f"OpenMM-{platform.name}-{precision}-"
            f"deterministic={deterministic}-threads={platform.threads}-"
            f"probe={probe_digest}-env={environment_digest}"
        ),
        engine_major=identity.version,
        engine_version=identity.version,
    )


def _prepared_input_digest(prepared: PreparedMdView, role: str) -> str:
    matching = [item for item in prepared.system.artifacts if item.role == role]
    if len(matching) != 1:
        raise BackendConfigurationError(f"OpenMM PreparedMdView lacks {role}")
    path = prepared.artifact(role)
    digest = matching[0].sha256
    if _sha256_file(path) != digest:
        raise BackendConfigurationError(f"OpenMM PreparedMdView {role} snapshot changed")
    return digest


def _require_request(request: StageRequest) -> PreparedMdView:
    if not isinstance(request, StageRequest):
        raise TypeError("OpenMM StageService requires mmpbsa.core.StageRequest")
    if not isinstance(request.prepared_case, PreparedMdView):
        raise BackendConfigurationError(f"{request.task.label} requires a verified PreparedMdView")
    return request.prepared_case


def _validate_reference(
    value: object,
    *,
    expected_kind: str,
    expected_phase: str | None = None,
    expected_role: str | None = None,
    expected_artifact: str | None = None,
    label: str,
) -> None:
    reference = _mapping(value, label)
    if reference.get("kind") != expected_kind:
        raise BackendConfigurationError(f"{label} has an invalid reference kind")
    expected = {"kind": expected_kind}
    if expected_phase is not None:
        expected["phase_id"] = expected_phase
    if expected_role is not None:
        expected["role"] = expected_role
    if expected_artifact is not None:
        expected["artifact"] = expected_artifact
    if dict(reference) != expected:
        raise BackendConfigurationError(f"{label} must be the canonical reference {expected!r}")


def _phase_plans(science: Mapping[str, object]) -> tuple[_PhasePlan, ...]:
    public_fields = {
        "schema_version",
        "md_engine_profile",
        "topology_role",
        "primary_phase",
        "replica_seed",
        "seed_derivation",
        "system",
        "phases",
    }
    unknown = set(science).difference(public_fields | {"_replica", "_prepared_case_schema"})
    missing = public_fields.difference(science)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise BackendConfigurationError("invalid OpenMM MD science: " + "; ".join(details))
    if science.get("schema_version") != MD_PHASE_PLAN_SCHEMA:
        raise BackendConfigurationError(f"OpenMM MD science must use {MD_PHASE_PLAN_SCHEMA}")
    if science.get("md_engine_profile") != MD_ENGINE_PROFILE:
        raise BackendConfigurationError(f"OpenMM MD science requires {MD_ENGINE_PROFILE}")
    if science.get("topology_role") != "solvated_topology":
        raise BackendConfigurationError("OpenMM topology_role must be solvated_topology")
    if science.get("primary_phase") != "production":
        raise BackendConfigurationError("OpenMM primary_phase must be production")
    if science.get("replica_seed") != "{{replica.seed}}":
        raise BackendConfigurationError(
            "OpenMM replica_seed must use the canonical replica placeholder"
        )
    if science.get("seed_derivation") != "sha256-phase-stream-v1":
        raise BackendConfigurationError("OpenMM seed_derivation must be sha256-phase-stream-v1")
    system = _mapping(science.get("system"), "md.system")
    if dict(system) != _EXPECTED_SYSTEM:
        raise BackendConfigurationError("OpenMM system controls differ from openmm_8_5@2")
    raw_phases = _sequence(science.get("phases"), "md.phases")
    if len(raw_phases) != 4:
        raise BackendConfigurationError("OpenMM MD requires exactly four phases")
    entries = tuple(
        _mapping(value, f"md.phases[{index}]") for index, value in enumerate(raw_phases)
    )
    if [entry.get("phase_id") for entry in entries] != [
        "em",
        "nvt",
        "npt",
        "production",
    ]:
        raise BackendConfigurationError("OpenMM MD phases must be ordered em, nvt, npt, production")
    plans: list[_PhasePlan] = []
    for entry in entries:
        phase_id = str(entry["phase_id"])
        kind = str(entry.get("kind"))
        if kind != phase_id:
            raise BackendConfigurationError(
                f"OpenMM phase {phase_id!r} must declare kind={phase_id!r}"
            )
        input_value = _mapping(entry.get("input"), f"md phase {phase_id}.input")
        if phase_id == "em":
            _strict_fields(
                entry,
                required={"phase_id", "kind", "input", "minimization"},
                optional=set(),
                label="OpenMM EM phase",
            )
            _validate_reference(
                input_value.get("coordinates"),
                expected_kind="prepared",
                expected_role="solvated_coordinates",
                label="OpenMM EM coordinates",
            )
            if set(input_value) != {"coordinates"}:
                raise BackendConfigurationError("OpenMM EM input contains unknown fields")
            minimization = _mapping(entry["minimization"], "OpenMM minimization")
            _strict_fields(
                minimization,
                required={
                    "maximum_force_threshold_kj_mol_nm",
                    "max_iterations",
                },
                optional=set(),
                label="OpenMM minimization",
            )
            phase = OpenMMPhaseSpec(
                phase_id=phase_id,
                kind=kind,
                minimization_tolerance_kj_mol_nm=(_EM_MINIMIZATION_TOLERANCE_KJ_MOL_NM),
                maximum_force_threshold_kj_mol_nm=_finite_positive(
                    minimization["maximum_force_threshold_kj_mol_nm"],
                    "OpenMM maximum force threshold",
                ),
                max_iterations=_positive_integer(
                    minimization["max_iterations"], "OpenMM max_iterations"
                ),
            )
            phase.validate()
            plans.append(_PhasePlan(phase_id, kind, input_value, phase))
            continue

        required = {
            "phase_id",
            "kind",
            "input",
            "duration_ps",
            "steps",
            "timestep_fs",
            "temperature_kelvin",
            "initialize_velocities",
            "output",
        }
        if phase_id in {"npt", "production"}:
            required.add("pressure_bar")
        _strict_fields(
            entry,
            required=required,
            optional=set(),
            label=f"OpenMM {phase_id} phase",
        )
        previous = {"nvt": "em", "npt": "nvt", "production": "npt"}[phase_id]
        _validate_reference(
            input_value.get("coordinates"),
            expected_kind="phase",
            expected_phase=previous,
            expected_artifact="final_coordinates",
            label=f"OpenMM {phase_id} coordinates",
        )
        expected_inputs = {"coordinates"}
        if phase_id in {"npt", "production"}:
            _validate_reference(
                input_value.get("state"),
                expected_kind="phase",
                expected_phase=previous,
                expected_artifact="portable_state",
                label=f"OpenMM {phase_id} state",
            )
            expected_inputs.add("state")
        if set(input_value) != expected_inputs:
            raise BackendConfigurationError(
                f"OpenMM {phase_id} input contains unknown/missing fields"
            )
        steps = _positive_integer(entry["steps"], f"OpenMM {phase_id} steps")
        timestep = _finite_positive(entry["timestep_fs"], f"OpenMM {phase_id} timestep_fs")
        duration = _finite_positive(entry["duration_ps"], f"OpenMM {phase_id} duration_ps")
        if not math.isclose(
            duration * 1000.0,
            steps * timestep,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise BackendConfigurationError(
                f"OpenMM {phase_id} duration does not equal steps*timestep"
            )
        output = _mapping(entry["output"], f"OpenMM {phase_id}.output")
        output_required = {"report_interval_steps", "checkpoint_interval_steps"}
        if phase_id == "production":
            output_required |= {
                "trajectory_interval_steps",
                "transaction_chunk_steps",
                "emit_initial_frame",
                "expected_trajectory_frames",
            }
        _strict_fields(
            output,
            required=output_required,
            optional=set(),
            label=f"OpenMM {phase_id}.output",
        )
        report_interval = _positive_integer(
            output["report_interval_steps"],
            f"OpenMM {phase_id} report interval",
        )
        checkpoint_interval = _positive_integer(
            output["checkpoint_interval_steps"],
            f"OpenMM {phase_id} checkpoint interval",
        )
        trajectory_interval = (
            _positive_integer(
                output["trajectory_interval_steps"],
                "OpenMM production trajectory interval",
            )
            if phase_id == "production"
            else steps
        )
        initialize = entry["initialize_velocities"]
        if not isinstance(initialize, bool):
            raise BackendConfigurationError(
                f"OpenMM {phase_id} initialize_velocities must be boolean"
            )
        if (phase_id == "nvt") != initialize:
            raise BackendConfigurationError(
                "OpenMM NVT must initialize velocities and later phases must preserve them"
            )
        phase = OpenMMPhaseSpec(
            phase_id=phase_id,
            kind=kind,
            steps=steps,
            timestep_fs=timestep,
            temperature_kelvin=_finite_positive(
                entry["temperature_kelvin"],
                f"OpenMM {phase_id} temperature",
            ),
            pressure_bar=(
                None
                if phase_id == "nvt"
                else _finite_positive(entry["pressure_bar"], f"OpenMM {phase_id} pressure")
            ),
            initialize_velocities=initialize,
            report_interval_steps=report_interval,
            trajectory_interval_steps=trajectory_interval,
            checkpoint_interval_steps=checkpoint_interval,
        )
        phase.validate()
        if phase_id == "production":
            chunk_steps = _positive_integer(
                output["transaction_chunk_steps"],
                "OpenMM production transaction chunk",
            )
            if chunk_steps > steps or steps % chunk_steps:
                raise BackendConfigurationError(
                    "OpenMM production steps must be divisible by transaction_chunk_steps"
                )
            if chunk_steps % report_interval or chunk_steps % trajectory_interval:
                raise BackendConfigurationError(
                    "OpenMM production chunks must align to report/trajectory intervals"
                )
            emit_initial = output["emit_initial_frame"]
            if emit_initial is not True:
                raise BackendConfigurationError(
                    "OpenMM production must emit exactly one initial frame"
                )
            expected_frames = _positive_integer(
                output["expected_trajectory_frames"],
                "OpenMM production expected frames",
            )
            if expected_frames != steps // trajectory_interval + 1:
                raise BackendConfigurationError(
                    "OpenMM production expected frame count is inconsistent"
                )
            plans.append(
                _PhasePlan(
                    phase_id,
                    kind,
                    input_value,
                    phase,
                    transaction_chunk_steps=chunk_steps,
                    emit_initial_frame=True,
                    expected_trajectory_frames=expected_frames,
                )
            )
        else:
            plans.append(_PhasePlan(phase_id, kind, input_value, phase))
    return tuple(plans)


def _replica_seed(science: Mapping[str, object]) -> int:
    replica = _mapping(science.get("_replica"), "OpenMM science._replica")
    seed = replica.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed < 2**31:
        raise BackendConfigurationError("OpenMM replica seed must be a non-zero 31-bit integer")
    return seed


def _phase_seed(master: int, phase_id: str) -> int:
    digest = hashlib.sha256(f"{master}:{phase_id}:phase".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


def _checkpoint_provenance(receipt: Mapping[str, object]) -> Mapping[str, object]:
    value = receipt.get("checkpoint_provenance")
    return _mapping(value, "OpenMM checkpoint provenance")


def _read_json(path: Path) -> Mapping[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _sealed_md_receipt(request: OpenMMMdRequest) -> Mapping[str, object] | None:
    artifacts = openmm_artifact_paths(request)
    request_path = request.output_dir.resolve() / "openmm_request.json"
    if any(not path.is_file() or path.stat().st_size == 0 for path in artifacts.values()):
        return None
    raw_request = _read_json(request_path)
    receipt = _read_json(artifacts["worker_receipt"])
    if raw_request != request.as_dict() or receipt is None:
        return None
    if (
        receipt.get("schema_version") != OPENMM_RECEIPT_SCHEMA
        or receipt.get("status") != "completed"
    ):
        return None
    try:
        validate_openmm_md_worker_receipt(receipt, request)
    except BackendOutputError:
        return None
    return receipt


def _sealed_merge_receipt(request: OpenMMMergeRequest) -> Mapping[str, object] | None:
    artifacts = openmm_merge_artifact_paths(request)
    request_path = request.output_dir.resolve() / "openmm_merge_request.json"
    if any(not path.is_file() or path.stat().st_size == 0 for path in artifacts.values()):
        return None
    raw_request = _read_json(request_path)
    receipt = _read_json(artifacts["worker_receipt"])
    if raw_request != request.as_dict() or receipt is None:
        return None
    if (
        receipt.get("schema_version") != OPENMM_MERGE_RECEIPT_SCHEMA
        or receipt.get("status") != "completed"
    ):
        return None
    try:
        validate_openmm_merge_worker_receipt(receipt, request)
    except BackendOutputError:
        return None
    return receipt


def _attach_last_checkpoint(
    exc: BaseException,
    checkpoint: Path | None,
    root: Path,
) -> None:
    if checkpoint is None:
        return
    resolved = checkpoint.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return
    if not resolved.is_file() or resolved.stat().st_size == 0:
        return
    try:
        exc.checkpoint = resolved  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass


def _run_md_with_checkpoint(
    backend: object,
    request: OpenMMMdRequest,
    *,
    last_checkpoint: Path | None,
    root: Path,
):
    try:
        return backend.run(request)  # type: ignore[attr-defined]
    except BaseException as exc:
        sealed = _sealed_md_receipt(request)
        candidate = (
            openmm_artifact_paths(request)["checkpoint"] if sealed is not None else last_checkpoint
        )
        _attach_last_checkpoint(exc, candidate, root)
        raise


def _run_merge_with_checkpoint(
    backend: object,
    request: OpenMMMergeRequest,
    *,
    last_checkpoint: Path | None,
    root: Path,
):
    try:
        return backend.merge(request)  # type: ignore[attr-defined]
    except BaseException as exc:
        _attach_last_checkpoint(exc, last_checkpoint, root)
        raise


def _reset_output(path: Path, root: Path) -> None:
    lexical_root = Path(os.path.abspath(root))
    lexical_target = Path(os.path.abspath(path))
    if lexical_root.is_symlink():
        raise BackendConfigurationError(
            f"OpenMM recovery refuses a symlink attempt root: {lexical_root}"
        )
    try:
        relative = lexical_target.relative_to(lexical_root)
    except ValueError as exc:
        raise BackendConfigurationError(
            f"OpenMM recovery target escapes attempt output: {lexical_target}"
        ) from exc
    if relative == Path("."):
        raise BackendConfigurationError(
            f"OpenMM recovery refuses broad deletion target: {lexical_target}"
        )
    expected = (
        relative.parts in {("em",), ("nvt",), ("npt",)}
        or relative.parts == ("production", "final")
        or (
            len(relative.parts) == 2
            and relative.parts[0] == "production"
            and re.fullmatch(r"chunk_[0-9]{4}", relative.parts[1]) is not None
        )
    )
    if not expected:
        raise BackendConfigurationError(
            f"OpenMM recovery target is not a declared phase/chunk: {relative}"
        )
    cursor = lexical_root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise BackendConfigurationError(
                f"OpenMM recovery refuses symlink deletion target: {cursor}"
            )
    resolved_root = root.resolve()
    resolved = lexical_target.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise BackendConfigurationError(
            f"OpenMM recovery target resolves outside attempt output: {resolved}"
        ) from exc
    if lexical_target.exists():
        if not lexical_target.is_dir():
            raise BackendConfigurationError(
                f"OpenMM recovery output is not a directory: {lexical_target}"
            )
        shutil.rmtree(lexical_target)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _command_dict(record: CommandRecord) -> dict[str, object]:
    return {
        "argv": list(record.argv),
        "cwd": str(record.cwd),
        "returncode": record.returncode,
        "stdout": record.stdout,
        "stderr": record.stderr,
    }


class OpenMMStageService:
    """API-v2 MD StageService with sealed production transaction chunks."""

    capability = "md"
    supports_host_resources = True

    from .openmm_allocation import validate_resume_allocation as _resume_allocation_guard

    validate_resume_allocation = staticmethod(_resume_allocation_guard)

    def __init__(self, *, runner: CommandRunner | None = None) -> None:
        self.runner = runner

    def resource_request(self, request: StageRequest) -> ResourceRequest:
        _require_request(request)
        config = _effective_config(request.runtime)
        platform = _platform_name(config)
        threads = _positive_integer(config.get("threads_per_job", 1), "OpenMM threads_per_job")
        return ResourceRequest(
            cpu_slots=threads,
            gpu_slots=1 if platform == "CUDA" else 0,
        )

    def compatibility(self, runtime: Mapping[str, object]) -> BackendCompatibility:
        config = _effective_config(runtime)
        _validate_runtime_contract(config)
        platform = _platform_spec(config, require_lease=False)
        backend = create_openmm_md_backend(config, runner=self.runner)
        cwd = Path(str(config.get("probe_cwd", Path.cwd()))).resolve()
        if not cwd.is_dir():
            raise BackendConfigurationError(f"OpenMM probe_cwd is not a directory: {cwd}")
        identity = backend.probe(cwd=cwd)
        return _compatibility_from_probe(identity, platform, backend)

    def execute(self, request: StageRequest) -> StageResult:
        return self._execute_pipeline(request, resume=False)

    def resume(self, request: StageRequest) -> StageResult:
        return self._execute_pipeline(request, resume=True)

    def _execute_pipeline(self, request: StageRequest, *, resume: bool) -> StageResult:
        prepared = _require_request(request)
        plans = _phase_plans(request.science)
        master_seed = _replica_seed(request.science)
        config = _effective_config(request.runtime)
        _validate_runtime_contract(config)
        platform = _platform_spec(config, require_lease=True)
        runner = self.runner
        if request.allocation is not None and request.allocation.cpu_ids:
            from .command import AffinityCommandRunner

            runner = AffinityCommandRunner(request.allocation.cpu_ids, runner)
        backend = create_openmm_md_backend(config, runner=runner)
        cwd = Path(str(config.get("probe_cwd", Path.cwd()))).resolve()
        identity = backend.probe(cwd=cwd)
        observed_compatibility = _compatibility_from_probe(identity, platform, backend)
        reserved_compatibility = getattr(request, "compatibility", None)
        if reserved_compatibility is None or observed_compatibility != reserved_compatibility:
            raise BackendConfigurationError(
                "OpenMM worker probe/interpreter/environment differs from reserved attempt"
            )
        probe_digest = _sha256_json(identity.raw)
        if prepared.input_format != "amber":
            raise BackendConfigurationError("openmm_8_5@2 requires an Amber PreparedMdView")
        topology = prepared.artifact("solvated_topology")
        prepared_coordinates = prepared.artifact("solvated_coordinates")
        topology_digest = _prepared_input_digest(prepared, "solvated_topology")
        prepared_coordinates_digest = _prepared_input_digest(prepared, "solvated_coordinates")
        root = (request.attempt_dir / "output").resolve()
        preflight_request = OpenMMMdRequest(
            topology=topology,
            coordinates=prepared_coordinates,
            output_dir=root / "em",
            phase=plans[0].phase,
            platform=platform,
            seed=_phase_seed(master_seed, "em"),
            input_format=OPENMM_AMBER_INPUT_FORMAT,
            topology_sha256=topology_digest,
            coordinates_sha256=prepared_coordinates_digest,
            expected_openmm_version=identity.version,
            expected_probe_sha256=probe_digest,
        )
        preflight = backend.preflight(preflight_request, invocation_dir=request.invocation_dir)
        if resume:
            root.mkdir(parents=True, exist_ok=True)
        else:
            clean_output_dir(root)

        phase_artifacts: dict[str, Mapping[str, Path]] = {}
        phase_receipts: dict[str, object] = {}
        all_commands: list[CommandRecord] = []
        skipped: list[str] = []
        restarted: list[str] = []
        production_requests: list[OpenMMMdRequest] = []
        production_receipts: list[Mapping[str, object]] = []
        found_incomplete = not resume
        last_sealed_checkpoint: Path | None = None

        for plan in plans:
            if plan.phase_id == "em":
                coordinates = prepared_coordinates
                state_input = None
            else:
                previous = {"nvt": "em", "npt": "nvt", "production": "npt"}[plan.phase_id]
                coordinates = phase_artifacts[previous]["final_coordinates"]
                state_input = (
                    phase_artifacts[previous]["portable_state"]
                    if plan.phase_id in {"npt", "production"}
                    else None
                )
            if plan.phase_id != "production":
                output_dir = root / plan.phase_id
                md_request = OpenMMMdRequest(
                    topology=topology,
                    coordinates=coordinates,
                    output_dir=output_dir,
                    phase=plan.phase,
                    platform=platform,
                    seed=_phase_seed(master_seed, plan.phase_id),
                    state_input=state_input,
                    emit_initial_frame=True,
                    input_format=OPENMM_AMBER_INPUT_FORMAT,
                    topology_sha256=topology_digest,
                    coordinates_sha256=_sha256_file(coordinates),
                    expected_openmm_version=identity.version,
                    expected_probe_sha256=probe_digest,
                )
                receipt = (
                    _sealed_md_receipt(md_request) if resume and not found_incomplete else None
                )
                if receipt is None:
                    found_incomplete = True
                    if resume and output_dir.exists():
                        _reset_output(output_dir, root)
                        restarted.append(plan.phase_id)
                    backend_receipt = _run_md_with_checkpoint(
                        backend,
                        md_request,
                        last_checkpoint=last_sealed_checkpoint,
                        root=root,
                    )
                    all_commands.extend(
                        command
                        for command in backend_receipt.commands
                        if isinstance(command, CommandRecord)
                    )
                    receipt = backend_receipt.parsed
                else:
                    skipped.append(plan.phase_id)
                artifacts = openmm_artifact_paths(md_request)
                phase_artifacts[plan.phase_id] = artifacts
                phase_receipts[plan.phase_id] = receipt
                last_sealed_checkpoint = artifacts["checkpoint"]
                continue

            assert plan.transaction_chunk_steps is not None
            assert plan.expected_trajectory_frames is not None
            production_root = root / "production"
            chunk_count = plan.phase.steps // plan.transaction_chunk_steps
            previous_checkpoint: Path | None = None
            previous_provenance: Mapping[str, object] | None = None
            last_artifacts: Mapping[str, Path] | None = None
            for chunk_index in range(1, chunk_count + 1):
                chunk_id = f"chunk_{chunk_index:04d}"
                chunk_root = production_root / chunk_id
                chunk_phase = replace(plan.phase, steps=plan.transaction_chunk_steps)
                md_request = OpenMMMdRequest(
                    topology=topology,
                    coordinates=coordinates,
                    output_dir=chunk_root,
                    phase=chunk_phase,
                    platform=platform,
                    seed=_phase_seed(master_seed, plan.phase_id),
                    state_input=(state_input if chunk_index == 1 else None),
                    checkpoint_input=previous_checkpoint,
                    checkpoint_provenance=previous_provenance,
                    chunk_id=chunk_id,
                    emit_initial_frame=(chunk_index == 1),
                    input_format=OPENMM_AMBER_INPUT_FORMAT,
                    topology_sha256=topology_digest,
                    coordinates_sha256=_sha256_file(coordinates),
                    expected_openmm_version=identity.version,
                    expected_probe_sha256=probe_digest,
                )
                receipt = (
                    _sealed_md_receipt(md_request) if resume and not found_incomplete else None
                )
                label = f"production/{chunk_id}"
                if receipt is None:
                    found_incomplete = True
                    if resume and chunk_root.exists():
                        _reset_output(chunk_root, root)
                        restarted.append(label)
                    backend_receipt = _run_md_with_checkpoint(
                        backend,
                        md_request,
                        last_checkpoint=last_sealed_checkpoint,
                        root=root,
                    )
                    all_commands.extend(
                        command
                        for command in backend_receipt.commands
                        if isinstance(command, CommandRecord)
                    )
                    receipt = backend_receipt.parsed
                else:
                    skipped.append(label)
                artifacts = openmm_artifact_paths(md_request)
                production_requests.append(md_request)
                production_receipts.append(receipt)
                previous_checkpoint = artifacts["checkpoint"]
                previous_provenance = _checkpoint_provenance(receipt)
                last_artifacts = artifacts
                last_sealed_checkpoint = artifacts["checkpoint"]
            assert last_artifacts is not None
            merge_request = OpenMMMergeRequest(
                trajectories=tuple(
                    openmm_artifact_paths(item)["trajectory"] for item in production_requests
                ),
                energy_series=tuple(
                    openmm_artifact_paths(item)["energy_series"] for item in production_requests
                ),
                topology=topology,
                coordinates=prepared_coordinates,
                output_dir=production_root / "final",
                expected_frames=plan.expected_trajectory_frames,
                trajectory_interval_steps=plan.phase.trajectory_interval_steps,
                timestep_fs=plan.phase.timestep_fs,
                input_format=OPENMM_AMBER_INPUT_FORMAT,
                topology_sha256=topology_digest,
                coordinates_sha256=prepared_coordinates_digest,
                expected_openmm_version=identity.version,
                expected_probe_sha256=probe_digest,
            )
            merge_receipt = (
                _sealed_merge_receipt(merge_request) if resume and not found_incomplete else None
            )
            if merge_receipt is None:
                if resume and merge_request.output_dir.exists():
                    _reset_output(merge_request.output_dir, root)
                    restarted.append("production/finalize")
                backend_receipt = _run_merge_with_checkpoint(
                    backend,
                    merge_request,
                    last_checkpoint=last_sealed_checkpoint,
                    root=root,
                )
                all_commands.extend(
                    command
                    for command in backend_receipt.commands
                    if isinstance(command, CommandRecord)
                )
                merge_receipt = backend_receipt.parsed
            else:
                skipped.append("production/finalize")
            merge_artifacts = openmm_merge_artifact_paths(merge_request)
            phase_artifacts[plan.phase_id] = {
                **last_artifacts,
                **merge_artifacts,
            }
            phase_receipts[plan.phase_id] = {
                "chunks": production_receipts,
                "merge": merge_receipt,
            }

        production = phase_artifacts["production"]
        em_receipt = _mapping(phase_receipts["em"], "OpenMM EM worker receipt")
        openmm_identity = _mapping(em_receipt.get("openmm"), "OpenMM worker identity")
        actual_platform = _mapping(em_receipt.get("platform"), "OpenMM worker platform")
        manifest_path = root / "openmm_md_manifest.json"
        manifest = {
            "schema_version": "mmpbsa.openmm.md-manifest.v1",
            "provider_id": PROVIDER_ID,
            "adapter_contract": ADAPTER_CONTRACT,
            "engine_profile": MD_ENGINE_PROFILE,
            "input_format": OPENMM_AMBER_INPUT_FORMAT,
            "input": {
                "topology_sha256": topology_digest,
                "coordinates_sha256": prepared_coordinates_digest,
            },
            "probe_sha256": probe_digest,
            "preflight": dict(preflight),
            "input_equivalence_warning_count": prepared.preflight_warning_count,
            "openmm": dict(openmm_identity),
            "platform": dict(actual_platform),
            "replica_seed": master_seed,
            "phase_receipts": phase_receipts,
            "recovery": {
                "requested": resume,
                "skipped": skipped,
                "restarted": restarted,
            },
            "public_artifacts": {
                "full_trajectory": str(production["full_trajectory"]),
                "coordinates": str(production["final_coordinates"]),
                "state": str(production["portable_state"]),
                "checkpoint": str(production["checkpoint"]),
                "energy_log": str(production["energy_log"]),
            },
        }
        _atomic_json(manifest_path, manifest)
        invocation_receipt = request.invocation_dir / "backend_receipt.json"
        _atomic_json(
            invocation_receipt,
            {
                "schema_version": "mmpbsa.openmm.stage-receipt.v1",
                "provider_id": PROVIDER_ID,
                "adapter_contract": ADAPTER_CONTRACT,
                "commands": [_command_dict(command) for command in all_commands],
                "manifest": str(manifest_path),
                "recovery": manifest["recovery"],
            },
        )
        artifacts = {
            "full_trajectory": production["full_trajectory"],
            "coordinates": production["final_coordinates"],
            "state": production["portable_state"],
            "checkpoint": production["checkpoint"],
            "energy_log": production["energy_log"],
        }
        return StageResult(
            artifacts=artifacts,
            payload={
                "schema_version": "mmpbsa.openmm.md-result.v1",
                "provider_id": PROVIDER_ID,
                "md_engine_profile": MD_ENGINE_PROFILE,
                "state_format": "openmm_xml_state",
                "checkpoint_format": "openmm_binary_checkpoint",
                "expected_trajectory_frames": phase_receipts["production"][  # type: ignore[index]
                    "merge"
                ]["frame_count"],
                "recovery": manifest["recovery"],
            },
            checkpoint=production["checkpoint"],
            invocation_provenance={
                "provider_id": PROVIDER_ID,
                "package_version": PACKAGE_VERSION,
                "adapter_contract": ADAPTER_CONTRACT,
                "openmm": dict(openmm_identity),
                "platform": dict(actual_platform),
                "requested_platform": platform.as_dict(),
                "backend_receipt": str(invocation_receipt),
                "private_artifacts": {
                    "stage_manifest": str(manifest_path),
                    "trajectory_manifest": str(production["trajectory_manifest"]),
                    "backend_receipt": str(invocation_receipt),
                },
            },
        )


@dataclass(frozen=True, slots=True)
class OpenMMProvider:
    _service: OpenMMStageService

    provider_id = PROVIDER_ID
    api_version = "2"
    package_version = PACKAGE_VERSION
    capabilities = ("md",)

    def get_service(self, capability: str) -> OpenMMStageService | None:
        return self._service if capability == "md" else None


def create_provider(*, runner: CommandRunner | None = None) -> OpenMMProvider:
    return OpenMMProvider(OpenMMStageService(runner=runner))


def get_provider() -> OpenMMProvider:
    """Entry-point factory; it imports no OpenMM runtime module."""

    return create_provider()

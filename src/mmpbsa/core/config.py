"""Strict loaders for the versioned science and runtime documents."""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ConfigurationError, ContractError
from .models import (
    ReplicaSpec,
    WorkflowProfile,
    require_identifier,
)
from .protocols import (
    AUTO_DIELECTRIC_TOKEN,
    DEFAULT_MD_ENGINE_PROFILE,
    ResolvedProtocol,
    resolve_protocol,
)
from .state import StageId

WORKFLOW_SCHEMA = "mmpbsa.workflow.v3"
RUNTIME_SCHEMA = "mmpbsa.runtime.v1"
_REQUIRED_SCIENCE_STAGES = {
    StageId.MD.value,
    StageId.TRAJECTORY.value,
    StageId.TRAJECTORY_QC.value,
    StageId.SCORING.value,
}
_ALLOWED_SCIENCE_STAGES = _REQUIRED_SCIENCE_STAGES | {StageId.ADMIT.value}
_KNOWN_CAPABILITIES = frozenset(
    {
        "admission",
        "md",
        "trajectory",
        "trajectory_qc",
        "scoring",
        "analysis",
        "report",
        "report.assets",
    }
)


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
        raise ConfigurationError(
            f"{label} is missing: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise ConfigurationError(
            f"{label} contains unknown fields: " + ", ".join(sorted(unknown))
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ConfigurationError(f"{label} must be a mapping with string keys")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ConfigurationError(f"{label} must be a sequence")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ConfigurationError(f"{label} must be an integer >= {minimum}")
    return value


def _load_yaml(path: Path, label: str) -> tuple[Path, Mapping[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"{label} does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read {label}: {source}") from exc
    return source, _mapping(raw, label)


def _json_value(value: object, label: str) -> object:
    """Return a detached JSON-shaped value and reject implicit Python objects."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ConfigurationError(f"{label} keys must be strings")
        return {
            str(key): _json_value(item, f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ConfigurationError(f"{label} must contain only JSON-compatible values")


def _reject_environment_interpolation(value: object, label: str) -> None:
    """Science inputs must not silently change with the launching shell."""

    if isinstance(value, str) and "${" in value:
        raise ConfigurationError(
            f"{label} contains environment interpolation; put host-specific "
            "values in runtime.yaml"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_environment_interpolation(item, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _reject_environment_interpolation(item, f"{label}[{index}]")


def _validate_scoring_config(
    value: Mapping[str, object],
) -> tuple[str, ...]:
    label = "workflow.stages.scoring"
    _json_value(value, label)
    _strict_keys(
        value,
        required={"completion_policy", "models"},
        optional={"decomposition"},
        label=label,
    )
    decomposition_raw = value.get("decomposition")
    if decomposition_raw is not None:
        decomposition = _mapping(decomposition_raw, f"{label}.decomposition")
        _strict_keys(
            decomposition,
            required={"enabled", "scope", "note"},
            label=f"{label}.decomposition",
        )
        if not isinstance(decomposition["enabled"], bool):
            raise ConfigurationError(
                f"{label}.decomposition.enabled must be boolean"
            )
        if decomposition["scope"] != "interface_residues":
            raise ConfigurationError(
                f"{label}.decomposition.scope must be 'interface_residues'"
            )
    completion_policy = value["completion_policy"]
    if (
        not isinstance(completion_policy, str)
        or completion_policy not in {"any", "all"}
    ):
        raise ConfigurationError(
            "workflow.stages.scoring.completion_policy must be 'any' or 'all'"
        )
    models = _mapping(value["models"], f"{label}.models")
    if not models:
        raise ConfigurationError(
            "workflow.stages.scoring.models cannot be empty"
        )
    model_ids: list[str] = []
    for model_id, raw_model in models.items():
        try:
            require_identifier(model_id, "scoring model_id")
        except ContractError as exc:
            raise ConfigurationError(str(exc)) from exc
        model_label = f"{label}.models.{model_id}"
        model = _mapping(raw_model, model_label)
        required_model_keys = {
            "expected_frames",
            "control_text",
            "native_inputs",
        }
        allowed_model_keys = required_model_keys | {
            "trajectory_artifact",
            "dielectric_policy",
            "frame_manifest_digest",
        }
        missing = required_model_keys.difference(model)
        unknown = set(model).difference(allowed_model_keys)
        if missing:
            raise ConfigurationError(
                f"{model_label} is missing: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise ConfigurationError(
                f"{model_label} contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        _integer(
            model["expected_frames"],
            f"{model_label}.expected_frames",
            minimum=1,
        )
        if (
            not isinstance(model["control_text"], str)
            or not model["control_text"].strip()
            or "\x00" in model["control_text"]
        ):
            raise ConfigurationError(
                f"{model_label}.control_text must be a non-empty string"
            )
        native_inputs = _mapping(
            model["native_inputs"],
            f"{model_label}.native_inputs",
        )
        if not native_inputs or any(
            not isinstance(item, str)
            or not item.strip()
            or "\x00" in item
            for item in native_inputs.values()
        ):
            raise ConfigurationError(
                f"{model_label}.native_inputs must contain non-empty strings"
            )
        trajectory_artifact = model.get("trajectory_artifact")
        if "trajectory_artifact" in model and (
            not isinstance(trajectory_artifact, str)
            or not trajectory_artifact.strip()
            or "\x00" in trajectory_artifact
        ):
            raise ConfigurationError(
                f"{model_label}.trajectory_artifact must be a non-empty string"
            )
        digest = model.get("frame_manifest_digest")
        if digest is not None and (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ConfigurationError(
                f"{model_label}.frame_manifest_digest must be a lowercase "
                "SHA-256 digest"
            )
        policy_raw = model.get("dielectric_policy")
        if policy_raw is not None:
            policy = _mapping(
                policy_raw,
                f"{model_label}.dielectric_policy",
            )
            _strict_keys(
                policy,
                required={"mode", "fallback", "fixed_value"},
                label=f"{model_label}.dielectric_policy",
            )
            mode = policy["mode"]
            if mode not in {"automatic", "fixed"}:
                raise ConfigurationError(
                    f"{model_label}.dielectric_policy.mode must be "
                    "'automatic' or 'fixed'"
                )
            fallback = policy["fallback"]
            if (
                isinstance(fallback, bool)
                or not isinstance(fallback, (int, float))
                or not math.isfinite(float(fallback))
                or float(fallback) <= 0
            ):
                raise ConfigurationError(
                    f"{model_label}.dielectric_policy.fallback must be "
                    "finite and positive"
                )
            fixed = policy["fixed_value"]
            if mode == "automatic":
                if fixed is not None:
                    raise ConfigurationError(
                        f"{model_label}.dielectric_policy.fixed_value must "
                        "be null in automatic mode"
                    )
                generated_text = "\n".join(
                    (str(model["control_text"]), *native_inputs.values())
                )
                if AUTO_DIELECTRIC_TOKEN not in generated_text:
                    raise ConfigurationError(
                        f"{model_label} automatic dielectric policy is "
                        "missing its generated resolution token"
                    )
            elif (
                isinstance(fixed, bool)
                or not isinstance(fixed, (int, float))
                or float(fixed) not in {2.0, 3.0, 4.0}
            ):
                raise ConfigurationError(
                    f"{model_label}.dielectric_policy.fixed_value must be "
                    "2, 3, or 4 in fixed mode"
                )
        model_ids.append(model_id)
    return tuple(model_ids)


def _validate_analysis_config(
    value: Mapping[str, object],
    *,
    scoring_model_ids: tuple[str, ...],
) -> None:
    label = "workflow.analysis"
    _json_value(value, label)
    if "models" not in value:
        raise ConfigurationError("workflow.analysis is missing: models")
    models = _mapping(value["models"], f"{label}.models")
    if not models:
        raise ConfigurationError("workflow.analysis.models cannot be empty")
    reference_mode = value.get("reference_mode", "quantitative_comparison")
    if reference_mode not in {
        "quantitative_comparison",
        "descriptive_association",
    }:
        raise ConfigurationError(
            "workflow.analysis.reference_mode must be "
            "'quantitative_comparison' or 'descriptive_association'"
        )
    for model_id, raw_model in models.items():
        model_label = f"{label}.models.{model_id}"
        model = _mapping(raw_model, model_label)
        unknown_model_fields = set(model).difference(
            {"source_model_id", "evaluator", "score_selector", "score_transform"}
        )
        if unknown_model_fields:
            raise ConfigurationError(
                f"{model_label} contains unknown fields: "
                + ", ".join(sorted(unknown_model_fields))
            )
        missing = {"score_selector", "score_transform"}.difference(model)
        if missing:
            raise ConfigurationError(
                f"{model_label} is missing: "
                + ", ".join(sorted(missing))
            )
        source_model_id = model.get("source_model_id", model_id)
        if source_model_id not in scoring_model_ids:
            raise ConfigurationError(
                f"{model_label}.source_model_id must select a configured "
                "scoring model"
            )
        evaluator = model.get("evaluator", "direct")
        if evaluator not in {"direct", "v1_dmm_v1"}:
            raise ConfigurationError(
                f"{model_label}.evaluator must be 'direct' or 'v1_dmm_v1'"
            )
        if evaluator == "direct" and model_id != source_model_id:
            raise ConfigurationError(
                f"{model_label} direct evaluator must use the source model ID"
            )
        if evaluator == "v1_dmm_v1" and model_id == source_model_id:
            raise ConfigurationError(
                f"{model_label} derived evaluator requires a distinct model ID"
            )
        selector = _mapping(
            model["score_selector"],
            f"{model_label}.score_selector",
        )
        _strict_keys(
            selector,
            required={"component", "label", "value_key"},
            label=f"{model_label}.score_selector",
        )
        if any(
            not isinstance(item, str) or not item.strip()
            for item in selector.values()
        ):
            raise ConfigurationError(
                f"{model_label}.score_selector values must be non-empty strings"
            )
        if selector["component"] != "delta":
            raise ConfigurationError(
                f"{model_label}.score_selector.component must be 'delta'"
            )
        if selector["value_key"] not in {
            "average",
            "mean",
            "value",
            "score",
        }:
            raise ConfigurationError(
                f"{model_label}.score_selector.value_key is unsupported"
            )
        transform = _mapping(
            model["score_transform"],
            f"{model_label}.score_transform",
        )
        _strict_keys(
            transform,
            required={"scale", "offset", "input_unit", "output_unit"},
            label=f"{model_label}.score_transform",
        )
        for field_name in ("scale", "offset"):
            number = transform[field_name]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
            ):
                raise ConfigurationError(
                    f"{model_label}.score_transform.{field_name} "
                    "must be a finite number"
                )
        for field_name in ("input_unit", "output_unit"):
            unit = transform[field_name]
            if not isinstance(unit, str) or not unit.strip():
                raise ConfigurationError(
                    f"{model_label}.score_transform.{field_name} "
                    "must be a non-empty string"
                )


def _validate_report_models(
    value: Mapping[str, object],
    *,
    analysis_model_ids: tuple[str, ...],
) -> None:
    label = "workflow.report.model_order"
    if "model_order" not in value:
        return
    raw_order = _sequence(value["model_order"], label)
    if any(not isinstance(item, str) or not item for item in raw_order):
        raise ConfigurationError(
            f"{label} entries must be non-empty model ID strings"
        )
    model_order = tuple(str(item) for item in raw_order)
    if len(model_order) != len(set(model_order)):
        raise ConfigurationError(f"{label} entries must be unique")
    expected = set(analysis_model_ids)
    observed = set(model_order)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        unknown = sorted(observed.difference(expected))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ConfigurationError(
            f"{label} must list every analysis model: " + "; ".join(details)
        )


def _validate_report_assets(value: Mapping[str, object]) -> None:
    """Separate generation settings from renderer copy/link descriptors."""

    for key in ("assets", "asset_links"):
        configured = value.get(key)
        if configured is None:
            continue
        label = f"workflow.report.{key}"
        if key == "assets" and isinstance(configured, Mapping):
            _mapping(configured, label)
            continue
        descriptors = _sequence(configured, label)
        for index, descriptor in enumerate(descriptors):
            _mapping(descriptor, f"{label}[{index}]")


@dataclass(frozen=True, slots=True)
class CaseConfig:
    case_id: str
    prepared_case: Path
    label: str | None = None
    reference: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier(self.case_id, "case_id")
        if self.label is not None and not self.label.strip():
            raise ContractError("case label cannot be empty")


@dataclass(frozen=True, slots=True)
class ResourceCoordinationConfig:
    """Execution-only, explicitly initialized same-user host coordination."""

    mode: str = "local"
    registry_path: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"local", "host"}:
            raise ConfigurationError("resource_coordination.mode must be local or host")
        if self.mode == "host":
            if not sys.platform.startswith("linux"):
                raise ConfigurationError("host resource coordination requires Linux")
            if self.registry_path is None or not Path(self.registry_path).is_absolute():
                raise ConfigurationError(
                    "host resource coordination requires an absolute registry_path"
                )
        elif self.registry_path is not None:
            raise ConfigurationError("local resource coordination does not accept registry_path")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    gpu_ids: tuple[str, ...] = ()
    max_workers: int = 1
    threads_per_job: int = 1
    cpu_slots: int = 1
    execution_mode: str = "legacy"
    cpu_affinity: str = "none"
    stage_limits: Mapping[str, int] = field(default_factory=dict)
    cpu_reservations: Mapping[str, int] = field(default_factory=dict)
    resource_coordination: ResourceCoordinationConfig = field(
        default_factory=ResourceCoordinationConfig
    )

    def __post_init__(self) -> None:
        if self.execution_mode not in {"legacy", "pipeline"}:
            raise ConfigurationError("scheduler.execution_mode must be legacy or pipeline")
        if self.cpu_affinity not in {"none", "physical"}:
            raise ConfigurationError("scheduler.cpu_affinity must be none or physical")
        if not isinstance(self.resource_coordination, ResourceCoordinationConfig):
            raise ConfigurationError("scheduler.resource_coordination has an invalid value")
        if self.resource_coordination.mode == "host" and (
            self.execution_mode != "pipeline" or self.cpu_affinity != "physical"
        ):
            raise ConfigurationError(
                "host resource coordination requires pipeline and physical affinity"
            )
        for label, mapping, minimum in (
            ("stage_limits", self.stage_limits, 1),
            ("cpu_reservations", self.cpu_reservations, 0),
        ):
            if not isinstance(mapping, Mapping) or set(mapping) - {"md", "postprocess", "scoring"}:
                raise ConfigurationError(f"scheduler.{label} has unknown resource groups")
            if any(
                isinstance(v, bool) or not isinstance(v, int) or v < minimum
                for v in mapping.values()
            ):
                raise ConfigurationError(f"scheduler.{label} values must be integers >= {minimum}")
        if any(
            not isinstance(gpu_id, str)
            or not gpu_id
            or gpu_id != gpu_id.strip()
            or "," in gpu_id
            or "\x00" in gpu_id
            or any(character.isspace() for character in gpu_id)
            for gpu_id in self.gpu_ids
        ):
            raise ConfigurationError(
                "scheduler.gpu_ids entries must each name one non-empty device"
            )
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ConfigurationError("scheduler.gpu_ids must be unique")
        for label, value in (
            ("max_workers", self.max_workers),
            ("threads_per_job", self.threads_per_job),
            ("cpu_slots", self.cpu_slots),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"scheduler.{label} must be an integer >= 1")
        if sum(self.cpu_reservations.values()) > self.cpu_slots:
            raise ConfigurationError("scheduler CPU reservations exceed cpu_slots")


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    source_path: Path
    project_id: str
    protocol_id: str
    profile_factory: str
    profile_configuration: Mapping[str, object]
    profile: WorkflowProfile
    cases: tuple[CaseConfig, ...]
    replicas: tuple[ReplicaSpec, ...]
    stages: Mapping[str, Mapping[str, object]]
    analysis: Mapping[str, object]
    report: Mapping[str, object]
    resolved_protocol: ResolvedProtocol | None = None
    schema_version: str = WORKFLOW_SCHEMA

    def __post_init__(self) -> None:
        require_identifier(self.project_id, "project_id")
        require_identifier(self.protocol_id, "protocol_id")
        if not self.cases:
            raise ConfigurationError("workflow must declare at least one case")
        if not self.replicas:
            raise ConfigurationError(
                "workflow must declare at least one replica"
            )
        case_ids = [item.case_id for item in self.cases]
        replica_ids = [item.replica_id for item in self.replicas]
        replica_indices = [item.replica_index for item in self.replicas]
        if len(set(case_ids)) != len(case_ids):
            raise ConfigurationError("case_id values must be unique")
        if len(set(replica_ids)) != len(replica_ids):
            raise ConfigurationError("replica_id values must be unique")
        if len(set(replica_indices)) != len(replica_indices):
            raise ConfigurationError("replica_index values must be unique")
        missing_stages = _REQUIRED_SCIENCE_STAGES.difference(self.stages)
        unknown_stages = set(self.stages).difference(_ALLOWED_SCIENCE_STAGES)
        if missing_stages:
            raise ConfigurationError(
                "workflow.stages is missing: "
                + ", ".join(sorted(missing_stages))
            )
        if unknown_stages:
            raise ConfigurationError(
                "workflow.stages contains unknown stages: "
                + ", ".join(sorted(unknown_stages))
            )
        scoring_model_ids = _validate_scoring_config(
            _mapping(
                self.stages[StageId.SCORING.value],
                "workflow.stages.scoring",
            )
        )
        _validate_analysis_config(
            self.analysis,
            scoring_model_ids=scoring_model_ids,
        )
        analysis_models = _mapping(
            self.analysis["models"],
            "workflow.analysis.models",
        )
        _validate_report_models(
            self.report,
            analysis_model_ids=tuple(analysis_models),
        )
        _validate_report_assets(self.report)

    def case(self, case_id: str) -> CaseConfig:
        for item in self.cases:
            if item.case_id == case_id:
                return item
        raise ConfigurationError(f"unknown workflow case: {case_id}")

    def replica(self, replica_id: str) -> ReplicaSpec:
        for item in self.replicas:
            if item.replica_id == replica_id:
                return item
        raise ConfigurationError(f"unknown workflow replica: {replica_id}")

    def stage_science(self, stage: StageId) -> Mapping[str, object]:
        if stage is StageId.FINALIZE_CASE:
            return {}
        if stage is StageId.ANALYSIS:
            return self.analysis
        if stage is StageId.ASSETS:
            # Nested generation settings override historical top-level settings.
            # The original report mapping stays intact for rendering and hashing.
            generation = self.report.get("assets")
            if isinstance(generation, Mapping):
                return {**self.report, **generation}
            return self.report
        if stage is StageId.REPORT:
            return self.report
        return self.stages.get(stage.value, {})

    @property
    def scoring_model_ids(self) -> tuple[str, ...]:
        scoring = _mapping(
            self.stages.get(StageId.SCORING.value),
            "workflow.stages.scoring",
        )
        models = _mapping(
            scoring.get("models"),
            "workflow.stages.scoring.models",
        )
        return tuple(models)

    def scoring_model(self, model_id: str) -> Mapping[str, object]:
        require_identifier(model_id, "model_id")
        scoring = _mapping(
            self.stages.get(StageId.SCORING.value),
            "workflow.stages.scoring",
        )
        models = _mapping(
            scoring.get("models"),
            "workflow.stages.scoring.models",
        )
        try:
            value = models[model_id]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown scoring model: {model_id}"
            ) from exc
        return _mapping(
            value,
            f"workflow.stages.scoring.models.{model_id}",
        )

    @property
    def completion_policy(self) -> str:
        scoring = _mapping(
            self.stages.get(StageId.SCORING.value),
            "workflow.stages.scoring",
        )
        return str(scoring["completion_policy"])

    def analysis_model(self, model_id: str) -> Mapping[str, object]:
        require_identifier(model_id, "model_id")
        models = _mapping(
            self.analysis.get("models"),
            "workflow.analysis.models",
        )
        try:
            value = models[model_id]
        except KeyError as exc:
            raise ConfigurationError(
                f"analysis does not select scoring model: {model_id}"
            ) from exc
        return _mapping(value, f"workflow.analysis.models.{model_id}")

    def normalized_science(self) -> dict[str, object]:
        normalized = {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "protocol_id": self.protocol_id,
            "profile": {
                "factory": self.profile_factory,
                "profile_id": self.profile.profile_id,
                "profile_version": self.profile.profile_version,
                "configuration": dict(self.profile_configuration),
                "normalized": dict(self.profile.normalized),
                "extensions": dict(self.profile.extensions),
            },
            "cases": [
                {
                    "case_id": case.case_id,
                    "prepared_case": str(case.prepared_case),
                    "label": case.label,
                    "reference": dict(case.reference),
                    "metadata": dict(case.metadata),
                }
                for case in self.cases
            ],
            "replicas": [
                {
                    "replica_id": replica.replica_id,
                    "replica_index": replica.replica_index,
                    "seed": replica.seed,
                }
                for replica in self.replicas
            ],
            "stages": {
                key: dict(value) for key, value in self.stages.items()
            },
            "analysis": dict(self.analysis),
            "report": dict(self.report),
        }
        if self.resolved_protocol is not None:
            normalized["resolved_protocol"] = self.resolved_protocol.as_dict()
        return normalized


@dataclass(frozen=True, slots=True)
class RuntimeDefinition:
    source_path: Path
    providers: Mapping[str, str]
    scheduler: SchedulerConfig
    backends: Mapping[str, Mapping[str, object]]
    environment: Mapping[str, str]
    schema_version: str = RUNTIME_SCHEMA

    def provider_for(self, capability: str) -> str | None:
        return self.providers.get(capability)

    def stage_runtime(
        self,
        capability: str,
        *,
        gpu_id: str | None = None,
    ) -> dict[str, object]:
        provider_id = self.provider_for(capability)
        result: dict[str, object] = {}
        common_environment: Mapping[str, object] = {}
        stage_environment: Mapping[str, object] = {}
        if provider_id is not None:
            provider_config = self.backends.get(provider_id, {})
            common = provider_config.get("common")
            selected = provider_config.get(capability)
            if isinstance(common, Mapping):
                result.update(common)
                common_environment = _mapping(
                    common.get("environment", {}),
                    f"runtime.backends.{provider_id}.common.environment",
                )
            if isinstance(selected, Mapping):
                result.update(selected)
                stage_environment = _mapping(
                    selected.get("environment", {}),
                    (
                        f"runtime.backends.{provider_id}."
                        f"{capability}.environment"
                    ),
                )
            elif common is None and not any(
                isinstance(provider_config.get(key), Mapping)
                for key in _KNOWN_CAPABILITIES
            ):
                # A flat provider mapping is convenient for a provider that
                # exposes only one capability.
                result.update(provider_config)
                stage_environment = _mapping(
                    provider_config.get("environment", {}),
                    f"runtime.backends.{provider_id}.environment",
                )
        merged_environment: dict[str, str] = dict(self.environment)
        for layer, label in (
            (common_environment, "provider common environment"),
            (stage_environment, f"{capability} environment"),
        ):
            if any(
                not isinstance(name, str) or not isinstance(value, str)
                for name, value in layer.items()
            ):
                raise ConfigurationError(
                    f"{label} keys and values must be strings"
                )
            merged_environment.update(layer)
        result["environment"] = merged_environment
        result["threads_per_job"] = self.scheduler.threads_per_job
        result["cpu_slots_total"] = self.scheduler.cpu_slots
        result["gpu_enabled"] = bool(self.scheduler.gpu_ids)
        # GPU selection is scheduler-owned. Provider defaults and per-stage
        # overrides are configuration hints from the pre-lease document and
        # must never supersede the concrete ID granted by the resource lease.
        result.pop("gpu_id", None)
        stage_overrides = result.get("stage_overrides")
        if isinstance(stage_overrides, Mapping):
            sanitized_overrides: dict[str, object] = {}
            for stage_id, raw_override in stage_overrides.items():
                if isinstance(raw_override, Mapping):
                    override = dict(raw_override)
                    override.pop("gpu_id", None)
                    sanitized_overrides[str(stage_id)] = override
                else:
                    sanitized_overrides[str(stage_id)] = raw_override
            result["stage_overrides"] = sanitized_overrides
        if gpu_id is not None:
            result["gpu_id"] = gpu_id
        return result


def load_workflow(path: Path) -> WorkflowDefinition:
    source, raw = _load_yaml(path, "workflow configuration")
    schema = raw.get("schema_version")
    if schema == WORKFLOW_SCHEMA:
        return _load_workflow_v3(source, raw)
    if schema == "mmpbsa.workflow.v2":
        raise ConfigurationError(
            "mmpbsa.workflow.v2 is read-only legacy configuration; use "
            "'mmpbsa workspace inspect --legacy-workflow ...' instead"
        )
    raise ConfigurationError(f"unsupported workflow schema: {schema!r}")


def _load_workflow_v3(
    source: Path,
    raw: Mapping[str, object],
) -> WorkflowDefinition:
    _strict_keys(
        raw,
        required={
            "schema_version",
            "project_id",
            "protocol",
            "cases",
            "replicas",
        },
        optional={"analysis", "report"},
        label="workflow configuration",
    )
    _reject_environment_interpolation(raw, "workflow")
    project_id = str(raw["project_id"])
    try:
        require_identifier(project_id, "project_id")
    except ContractError as exc:
        raise ConfigurationError(str(exc)) from exc

    protocol_raw = _mapping(raw["protocol"], "workflow.protocol")
    _strict_keys(
        protocol_raw,
        required={"preset"},
        optional={
            "md_engine_profile",
            "models",
            "completion_policy",
            "overrides",
            "engine_overrides",
        },
        label="workflow.protocol",
    )
    models_raw = protocol_raw.get("models")
    models = (
        None
        if models_raw is None
        else tuple(
            str(item)
            for item in _sequence(models_raw, "workflow.protocol.models")
        )
    )
    completion_policy = protocol_raw.get("completion_policy", "all")
    if not isinstance(completion_policy, str):
        raise ConfigurationError(
            "workflow.protocol.completion_policy must be a string"
        )
    resolved = resolve_protocol(
        str(protocol_raw["preset"]),
        md_engine_profile=str(
            protocol_raw.get("md_engine_profile", DEFAULT_MD_ENGINE_PROFILE)
        ),
        models=models,
        completion_policy=completion_policy,
        overrides=_mapping(
            protocol_raw.get("overrides", {}),
            "workflow.protocol.overrides",
        ),
        engine_overrides=_mapping(
            protocol_raw.get("engine_overrides", {}),
            "workflow.protocol.engine_overrides",
        ),
    )

    cases: list[CaseConfig] = []
    for index, item in enumerate(_sequence(raw["cases"], "workflow.cases")):
        case_raw = _mapping(item, f"workflow.cases[{index}]")
        _strict_keys(
            case_raw,
            required={"case_id", "prepared_case"},
            optional={"label", "reference", "metadata"},
            label=f"workflow.cases[{index}]",
        )
        prepared = Path(str(case_raw["prepared_case"])).expanduser()
        if not prepared.is_absolute():
            prepared = source.parent / prepared
        cases.append(
            CaseConfig(
                case_id=str(case_raw["case_id"]),
                prepared_case=prepared.resolve(),
                label=(
                    None
                    if case_raw.get("label") is None
                    else str(case_raw["label"])
                ),
                reference=_mapping(
                    _json_value(
                        case_raw.get("reference", {}),
                        f"workflow.cases[{index}].reference",
                    ),
                    f"workflow.cases[{index}].reference",
                ),
                metadata=_mapping(
                    _json_value(
                        case_raw.get("metadata", {}),
                        f"workflow.cases[{index}].metadata",
                    ),
                    f"workflow.cases[{index}].metadata",
                ),
            )
        )

    replicas: list[ReplicaSpec] = []
    for index, item in enumerate(
        _sequence(raw["replicas"], "workflow.replicas")
    ):
        replica_raw = _mapping(item, f"workflow.replicas[{index}]")
        _strict_keys(
            replica_raw,
            required={"replica_id", "replica_index", "seed"},
            label=f"workflow.replicas[{index}]",
        )
        try:
            replicas.append(
                ReplicaSpec(
                    replica_id=str(replica_raw["replica_id"]),
                    replica_index=_integer(
                        replica_raw["replica_index"],
                        f"workflow.replicas[{index}].replica_index",
                        minimum=1,
                    ),
                    seed=_integer(
                        replica_raw["seed"],
                        f"workflow.replicas[{index}].seed",
                        minimum=0,
                    ),
                )
            )
        except (TypeError, ValueError, ContractError) as exc:
            raise ConfigurationError(
                f"invalid workflow.replicas[{index}]"
            ) from exc
    if len(replicas) != resolved.preset.replicas:
        raise ConfigurationError(
            f"{resolved.preset.preset_id} requires exactly "
            f"{resolved.preset.replicas} replicas"
        )
    expected_indices = tuple(range(1, resolved.preset.replicas + 1))
    if tuple(item.replica_index for item in replicas) != expected_indices:
        raise ConfigurationError(
            "workflow.replicas must be ordered with replica_index 1, 2, 3"
        )
    seeds = tuple(item.seed for item in replicas)
    if len(set(seeds)) != len(seeds):
        raise ConfigurationError("workflow.replicas seeds must be unique")

    analysis_overrides = _mapping(
        _json_value(raw.get("analysis", {}), "workflow.analysis"),
        "workflow.analysis",
    )
    _strict_keys(
        analysis_overrides,
        required=set(),
        optional={"reference_key", "aggregation", "reference_mode"},
        label="workflow.analysis",
    )
    analysis = {**resolved.analysis, **analysis_overrides}
    report_overrides = _mapping(
        _json_value(raw.get("report", {}), "workflow.report"),
        "workflow.report",
    )
    report = _merge_report_defaults(resolved.report_defaults, report_overrides)

    preset_name, preset_version = resolved.preset.preset_id.rsplit("@", 1)
    protocol_id = f"{preset_name}_v{preset_version}"
    profile = WorkflowProfile(
        profile_id=preset_name,
        profile_version=preset_version,
        normalized=resolved.as_dict(),
        extensions={},
    )
    try:
        return WorkflowDefinition(
            source_path=source,
            project_id=project_id,
            protocol_id=protocol_id,
            profile_factory="mmpbsa.core.protocols:resolve_protocol",
            profile_configuration=dict(protocol_raw),
            profile=profile,
            cases=tuple(cases),
            replicas=tuple(replicas),
            stages=resolved.stages,
            analysis=analysis,
            report=report,
            resolved_protocol=resolved,
            schema_version=WORKFLOW_SCHEMA,
        )
    except (ContractError, ConfigurationError) as exc:
        raise ConfigurationError(str(exc)) from exc


def _merge_report_defaults(
    defaults: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    result = dict(defaults)
    for key, value in overrides.items():
        if (
            key in {"metadata", "protocol", "provenance"}
            and isinstance(value, Mapping)
            and isinstance(result.get(key), Mapping)
        ):
            result[key] = {**dict(result[key]), **dict(value)}
        else:
            result[key] = value
    return result


def load_runtime(path: Path) -> RuntimeDefinition:
    source, raw = _load_yaml(path, "runtime configuration")
    _strict_keys(
        raw,
        required={"schema_version", "providers", "scheduler", "backends"},
        optional={"environment"},
        label="runtime configuration",
    )
    if raw["schema_version"] != RUNTIME_SCHEMA:
        raise ConfigurationError(f"unsupported runtime schema: {raw['schema_version']!r}")
    providers_raw = _mapping(raw["providers"], "runtime.providers")
    unknown_capabilities = set(providers_raw).difference(_KNOWN_CAPABILITIES)
    if unknown_capabilities:
        raise ConfigurationError(
            "runtime.providers contains unknown capabilities: "
            + ", ".join(sorted(unknown_capabilities))
        )
    providers: dict[str, str] = {}
    for capability, value in providers_raw.items():
        provider_id = str(value)
        try:
            require_identifier(provider_id, f"provider for {capability}")
        except ContractError as exc:
            raise ConfigurationError(str(exc)) from exc
        providers[capability] = provider_id

    scheduler_raw = _mapping(raw["scheduler"], "runtime.scheduler")
    _strict_keys(
        scheduler_raw,
        required=set(),
        optional={
            "gpu_ids",
            "max_workers",
            "threads_per_job",
            "cpu_slots",
            "execution_mode",
            "cpu_affinity",
            "stage_limits",
            "cpu_reservations",
            "resource_coordination",
        },
        label="runtime.scheduler",
    )
    gpu_ids = tuple(
        str(item)
        for item in _sequence(scheduler_raw.get("gpu_ids", ()), "runtime.scheduler.gpu_ids")
    )
    coordination_raw = _mapping(
        scheduler_raw.get("resource_coordination", {}),
        "runtime.scheduler.resource_coordination",
    )
    _strict_keys(
        coordination_raw, required=set(), optional={"mode", "registry_path"},
        label="runtime.scheduler.resource_coordination",
    )
    registry_path = coordination_raw.get("registry_path")
    if registry_path is not None and (
        not isinstance(registry_path, str) or not registry_path.strip()
    ):
        raise ConfigurationError(
            "resource_coordination.registry_path must be a non-empty absolute path"
        )
    coordination = ResourceCoordinationConfig(
        mode=str(coordination_raw.get("mode", "local")),
        registry_path=Path(registry_path) if registry_path is not None else None,
    )
    try:
        max_workers = _integer(
            scheduler_raw.get("max_workers", max(1, len(gpu_ids))),
            "runtime.scheduler.max_workers",
            minimum=1,
        )
        scheduler = SchedulerConfig(
            gpu_ids=gpu_ids,
            max_workers=max_workers,
            threads_per_job=_integer(
                scheduler_raw.get("threads_per_job", 1),
                "runtime.scheduler.threads_per_job",
                minimum=1,
            ),
            cpu_slots=_integer(
                scheduler_raw.get("cpu_slots", max_workers),
                "runtime.scheduler.cpu_slots",
                minimum=1,
            ),
            execution_mode=str(scheduler_raw.get("execution_mode", "legacy")),
            cpu_affinity=str(scheduler_raw.get("cpu_affinity", "none")),
            stage_limits=dict(
                _mapping(scheduler_raw.get("stage_limits", {}), "scheduler.stage_limits")
            ),
            cpu_reservations=dict(
                _mapping(scheduler_raw.get("cpu_reservations", {}), "scheduler.cpu_reservations")
            ),
            resource_coordination=coordination,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("invalid runtime.scheduler") from exc

    backends_raw = _mapping(raw["backends"], "runtime.backends")
    backends: dict[str, Mapping[str, object]] = {}
    for provider_id, value in backends_raw.items():
        try:
            require_identifier(provider_id, "runtime backend provider_id")
        except ContractError as exc:
            raise ConfigurationError(str(exc)) from exc
        backends[provider_id] = _mapping(
            _json_value(value, f"runtime.backends.{provider_id}"),
            f"runtime.backends.{provider_id}",
        )

    environment_raw = _mapping(raw.get("environment", {}), "runtime.environment")
    environment: dict[str, str] = {}
    for name, value in environment_raw.items():
        if not name or "\x00" in name or "=" in name:
            raise ConfigurationError(f"invalid environment variable name: {name!r}")
        environment[name] = str(value)
    return RuntimeDefinition(
        source_path=source,
        providers=providers,
        scheduler=scheduler,
        backends=backends,
        environment=environment,
    )

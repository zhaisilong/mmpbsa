"""Nominal adapters from the core StageService ABI to executable backends."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from mmpbsa.core import (
    BackendCompatibility,
    PreparedBindingView,
    PreparedMdView,
    PreparedTrajectoryView,
    QcVerdict,
    ResourceRequest,
    StageInput,
    StageRequest,
    StageResult,
    StageService,
)

from ._support import clean_output_dir, require_artifacts
from .ambertools import AmberToolsScoringRequest
from .api import (
    BackendReceipt,
    CommandRecord,
    CommandRunner,
    CommandSpec,
    GeneratedInput,
    ReceiptState,
)
from .cpptraj import CpptrajQcRequest
from .display import DisplayAssetRequest
from .errors import BackendConfigurationError, BackendOutputError
from .gromacs import GromacsMdRequest
from .gromacs_phase import (
    GROMACS_CANONICAL_PROFILE,
    MD_PHASE_PLAN_V2,
    gromacs_operational_mapping,
    is_md_phase_plan_v2,
    materialize_gromacs_public_artifacts,
    render_gromacs_phase_plan,
    validate_gromacs_phase_output,
)
from .parmed import ParmEdAdmissionRequest, ProjectionTolerances
from .prepared_display import prepared_components as _prepared_components
from .scoring_dielectric import resolve_scoring_dielectric as _resolve_scoring_dielectric
from .versioning import ToolIdentity, probe_tool

PROVIDER_ID = "backends"
_CAPABILITIES = (
    "admission",
    "md",
    "trajectory",
    "scoring",
    "trajectory_qc",
    "report.assets",
)
_PLACEHOLDER = re.compile(
    r"\{\{(?:(?P<prepared>prepared):(?P<prepared_role>[A-Za-z0-9_.-]+)"
    r"|binding_topology:(?P<binding_component>complex|receptor|ligand)"
    r"|upstream:(?P<stage>[A-Za-z0-9_.-]+):"
    r"(?P<artifact>[A-Za-z0-9_.-]+)"
    r"|selection:(?P<selection>[A-Za-z0-9_.-]+))\}\}"
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MD_COMPLETION_FILE = "stage_complete.json"
_MD_COMPLETION_SCHEMA = "mmpbsa.md_stage_completion.v1"


@dataclass(frozen=True, slots=True)
class _MdInputReference:
    kind: str
    role: str | None = None
    stage_id: str | None = None
    artifact: str | None = None


@dataclass(frozen=True, slots=True)
class _MdStagePlan:
    stage_id: str
    mdp_text: str
    coordinates: _MdInputReference
    state_input: _MdInputReference | None
    reference: _MdInputReference | None
    expected_suffixes: tuple[str, ...]
    checkpoint: bool
    primary: bool


class BuiltinStageService(StageService):
    """One stateless service instance for a provider capability."""

    supports_host_resources = True

    def __init__(
        self,
        capability: str,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        if capability not in _CAPABILITIES:
            raise BackendConfigurationError(f"unknown stage capability: {capability!r}")
        self.capability = capability
        self.runner = runner

    def resource_request(self, request: StageRequest) -> ResourceRequest:
        _require_request(request)
        if self.capability == "md":
            config = _capability_config(request.runtime, self.capability)
            return ResourceRequest(
                cpu_slots=_md_resource_threads(request, config),
                gpu_slots=1 if bool(request.runtime.get("gpu_enabled")) else 0,
            )
        if self.capability == "scoring":
            config = _capability_config(request.runtime, self.capability)
            return ResourceRequest(
                cpu_slots=_integer(
                    config.get("ranks", 1),
                    "scoring ranks",
                    minimum=1,
                )
            )
        return ResourceRequest()

    def compatibility(
        self,
        runtime: Mapping[str, object],
    ) -> BackendCompatibility:
        config = _capability_config(runtime, self.capability)
        backend = _create_backend(self.capability, config, runner=self.runner)
        cwd = Path(str(runtime.get("probe_cwd", Path.cwd()))).resolve()
        if not cwd.is_dir():
            raise BackendConfigurationError(f"probe_cwd is not a directory: {cwd}")
        identities = _probe_backend(self.capability, backend, cwd)
        if (
            self.capability == "md"
            and config.get("_stage_contract") == MD_PHASE_PLAN_V2
        ):
            from .trajectory_v3 import probe_mdtraj_identity

            identities = {**identities, "mdtraj": probe_mdtraj_identity()}
        return _compatibility(
            self.capability,
            identities,
            stage_contract=_optional_string(
                config.get("_stage_contract"),
                f"runtime.{self.capability}._stage_contract",
            ),
        )

    def execute(self, request: StageRequest) -> StageResult:
        return self._allocated_service(request)._execute(request)

    def _allocated_service(self, request: StageRequest) -> BuiltinStageService:
        if request.allocation is None or not request.allocation.cpu_ids:
            return self
        from .command import AffinityCommandRunner

        return BuiltinStageService(
            self.capability,
            runner=AffinityCommandRunner(request.allocation.cpu_ids, self.runner),
        )

    def _execute(self, request: StageRequest) -> StageResult:
        _require_request(request)
        if self.capability == "admission":
            return self._execute_admission(request)
        if self.capability == "md":
            return self._execute_md(request, resume=False)
        if self.capability == "trajectory":
            return self._execute_trajectory(request)
        if self.capability == "scoring":
            return self._execute_scoring(request)
        if self.capability == "trajectory_qc":
            return self._execute_trajectory_qc(request)
        return self._execute_assets(request)

    def resume(self, request: StageRequest) -> StageResult:
        return self._allocated_service(request)._resume(request)

    def _resume(self, request: StageRequest) -> StageResult:
        _require_request(request)
        if self.capability == "md":
            return self._execute_md(request, resume=True)
        output = _output_root(request)
        if output.exists() and any(output.iterdir()):
            raise BackendConfigurationError(
                f"{self.capability} has no in-place resume protocol; retry it in a new attempt"
            )
        return self._execute(request)

    def recover_failed_md(self, request: StageRequest, source_attempt: Path) -> StageResult:
        """Optional explicit retry hook; the old terminal attempt stays untouched."""
        from dataclasses import replace

        from .md_recovery import clone_native_output

        if self.capability != "md":
            raise BackendConfigurationError("native checkpoint recovery is MD-only")
        _require_request(request)
        science = _science(request)
        rendered = render_gromacs_phase_plan(science) if is_md_phase_plan_v2(science) else None
        plans = _md_stage_plans(science if rendered is None else rendered.pipeline_science, request)
        incomplete = None
        for plan in plans:
            phase_root = source_attempt / "output" / plan.stage_id
            declared = {
                suffix: phase_root / f"{plan.stage_id}.{suffix}"
                for suffix in plan.expected_suffixes
            }
            declared["mdp_input"] = phase_root / f"{plan.stage_id}.mdp"
            if _read_md_stage_completion(phase_root, plan, declared) is None:
                incomplete = plan
                break
        if incomplete is None:
            raise BackendConfigurationError("failed MD has no incomplete phase to resume")
        phase_root = source_attempt / "output" / incomplete.stage_id
        if (not incomplete.checkpoint
                or not _nonempty_file(phase_root / f"{incomplete.stage_id}.cpt")
                or not _nonempty_file(phase_root / f"{incomplete.stage_id}.tpr")):
            raise BackendConfigurationError(
                "failed MD has no matching native checkpoint; cold restart withheld"
            )
        audit = clone_native_output(source_attempt, request.attempt_dir)
        result = self._allocated_service(request)._execute_md(request, resume=True)
        return replace(result, invocation_provenance={**dict(result.invocation_provenance),
                                                     "failed_md_recovery": audit})

    def _execute_admission(self, request: StageRequest) -> StageResult:
        prepared = _binding_view(request)
        science = _science(request)
        config = _capability_config(request.runtime, self.capability)
        backend = _create_backend(self.capability, config, runner=self.runner)
        output = clean_output_dir(_output_root(request))

        binding = prepared.binding
        topology_roles = binding.topology_artifact_roles
        resolved_roles = {
            "receptor": topology_roles.receptor,
            "ligand": topology_roles.ligand,
        }
        component_topologies = {
            component: prepared.topology_artifact(component)
            for component in resolved_roles
        }
        component_indices = {
            "receptor": prepared.binding_selection(
                "receptor_all"
            ).scoring_atom_indices,
            "ligand": prepared.binding_selection(
                "ligand_all"
            ).scoring_atom_indices,
        }
        tolerance_raw = science.get("tolerances", {})
        tolerance = _mapping(tolerance_raw, "admission.tolerances")
        unknown_tolerances = set(tolerance).difference({"charge", "mass", "parameter"})
        if unknown_tolerances:
            raise BackendConfigurationError(
                "admission.tolerances has unknown fields: " + ", ".join(sorted(unknown_tolerances))
            )
        projection = ProjectionTolerances(
            charge=_finite_float(
                tolerance.get("charge", 1.0e-8),
                "admission.tolerances.charge",
            ),
            mass=_finite_float(
                tolerance.get("mass", 1.0e-8),
                "admission.tolerances.mass",
            ),
            parameter=_finite_float(
                tolerance.get("parameter", 1.0e-8),
                "admission.tolerances.parameter",
            ),
        )
        complete = tuple(
            sorted(index for values in component_indices.values() for index in values)
        ) == tuple(range(1, prepared.scoring.atom_count + 1))
        result = backend.admit(
            ParmEdAdmissionRequest(
                complex_topology=prepared.topology_artifact("complex"),
                component_topologies=component_topologies,
                component_atom_indices=component_indices,
                tolerances=projection,
                require_complete_partition=True,
            )
        )
        result.require_accepted()
        artifact = output / "admission.json"
        payload = {
            "accepted": result.accepted,
            "issues": result.issues,
            "complex_signature": result.complex_signature,
            "component_signatures": result.component_signatures,
            "complete_projection": complete,
            "topology_artifact_roles": topology_roles.as_dict(),
            "tool": result.tool,
        }
        _write_json(artifact, payload)
        receipt = BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=(),
            artifacts={"admission": artifact},
            metadata={"tool": result.tool},
            parsed=payload,
        )
        return _stage_result(
            self.capability,
            request,
            receipt,
        )

    def _execute_md(
        self,
        request: StageRequest,
        *,
        resume: bool,
    ) -> StageResult:
        return self._execute_md_pipeline(
            request,
            _science(request),
            resume=resume,
        )

    def _execute_md_pipeline(
        self,
        request: StageRequest,
        science: Mapping[str, object],
        *,
        resume: bool,
    ) -> StageResult:
        prepared = _md_view(request)
        rendered = (
            render_gromacs_phase_plan(science)
            if is_md_phase_plan_v2(science)
            else None
        )
        pipeline_science = (
            science if rendered is None else rendered.pipeline_science
        )
        plans = _md_stage_plans(pipeline_science, request)
        config = _capability_config(request.runtime, self.capability)
        backend = _create_backend(self.capability, config, runner=self.runner)
        root = _output_root(request)
        if resume:
            root.mkdir(parents=True, exist_ok=True)
        else:
            clean_output_dir(root)

        topology = prepared.artifact(
            _string(
                pipeline_science.get("topology_role", "topology"),
                "md.topology_role",
            )
        )
        by_stage: dict[str, Mapping[str, Path]] = {}
        combined_artifacts: dict[str, Path] = {}
        commands = []
        generated: list[GeneratedInput] = []
        tools: dict[str, object] = {}
        stage_provenance: dict[str, object] = {}
        skipped: list[str] = []
        recovery: dict[str, str] = {}
        scientific_validation: dict[str, object] = {}
        found_incomplete = False

        for plan in plans:
            stage_root = _md_stage_root(root, plan.stage_id)
            declared = {
                suffix: stage_root / f"{plan.stage_id}.{suffix}"
                for suffix in plan.expected_suffixes
            }
            mdp_path = stage_root / f"{plan.stage_id}.mdp"
            declared_with_input = {**declared, "mdp_input": mdp_path}
            completion = _read_md_stage_completion(
                stage_root,
                plan,
                declared_with_input,
            )
            if resume and not found_incomplete and completion is not None:
                stage_artifacts = declared_with_input
                by_stage[plan.stage_id] = stage_artifacts
                skipped.append(plan.stage_id)
                tools[plan.stage_id] = completion["tool"]
                stage_provenance[plan.stage_id] = {
                    "source": "completion_marker",
                    "commands": completion["commands"],
                    "tool": completion["tool"],
                }
                _merge_md_stage_artifacts(
                    combined_artifacts,
                    plan,
                    stage_artifacts,
                )
                if rendered is not None:
                    scientific_validation[plan.stage_id] = (
                        validate_gromacs_phase_output(
                            science, plan.stage_id, stage_artifacts
                        )
                    )
                continue

            stage_resume = False
            if resume:
                first_incomplete = not found_incomplete
                found_incomplete = True
                checkpoint = stage_root / f"{plan.stage_id}.cpt"
                tpr = stage_root / f"{plan.stage_id}.tpr"
                if (
                    first_incomplete
                    and plan.checkpoint
                    and _nonempty_file(checkpoint)
                    and _nonempty_file(tpr)
                ):
                    stage_resume = True
                    recovery[plan.stage_id] = "checkpoint_append"
                else:
                    _clear_md_stage_root(root, plan.stage_id)
                    recovery[plan.stage_id] = "restart_stage"
            coordinates = _resolve_md_reference(
                plan.coordinates,
                prepared,
                by_stage,
                plan.stage_id,
            )
            state_input = (
                None
                if plan.state_input is None
                else _resolve_md_reference(
                    plan.state_input,
                    prepared,
                    by_stage,
                    plan.stage_id,
                )
            )
            reference = (
                None
                if plan.reference is None
                else _resolve_md_reference(
                    plan.reference,
                    prepared,
                    by_stage,
                    plan.stage_id,
                )
            )
            stage_config = _md_stage_runtime(config, plan.stage_id)
            checkpoint_input = stage_root / f"{plan.stage_id}.cpt" if stage_resume else None
            backend_request = GromacsMdRequest(
                stage_id=plan.stage_id,
                mdp_text=plan.mdp_text,
                coordinates=coordinates,
                topology=topology,
                output_dir=stage_root,
                index=None,
                state_input=state_input,
                reference=reference,
                checkpoint_input=checkpoint_input,
                append=stage_resume,
                require_unconstrained_topology=(
                    rendered is not None and plan.stage_id == "em"
                ),
                maxwarn=_integer(
                    stage_config.get("maxwarn", config.get("maxwarn", 0)),
                    f"md.{plan.stage_id}.maxwarn",
                    minimum=0,
                ),
                threads=_integer(
                    stage_config.get(
                        "threads_per_job",
                        config.get("threads_per_job", 1),
                    ),
                    f"md.{plan.stage_id}.threads_per_job",
                    minimum=1,
                ),
                gpu_id=_optional_string(
                    config.get("gpu_id"),
                    f"md.{plan.stage_id}.gpu_id",
                ),
                grompp_arguments=_string_tuple(
                    stage_config.get(
                        "grompp_arguments",
                        config.get("grompp_arguments", ()),
                    ),
                    f"md.{plan.stage_id}.grompp_arguments",
                ),
                mdrun_arguments=_string_tuple(
                    stage_config.get(
                        "mdrun_arguments",
                        config.get("mdrun_arguments", ()),
                    ),
                    f"md.{plan.stage_id}.mdrun_arguments",
                ),
                expected_suffixes=plan.expected_suffixes,
            )
            _invalidate_md_stage_completion(stage_root)
            try:
                stage_receipt = backend.run(
                    backend_request,
                    resume=stage_resume,
                )
                if rendered is not None:
                    scientific_validation[plan.stage_id] = (
                        validate_gromacs_phase_output(
                            science, plan.stage_id, stage_receipt.artifacts
                        )
                    )
                _write_md_stage_completion(
                    stage_root,
                    plan,
                    stage_receipt,
                )
            except BaseException as exc:
                checkpoint = stage_root / f"{plan.stage_id}.cpt"
                if _nonempty_file(checkpoint):
                    try:
                        exc.checkpoint = checkpoint  # type: ignore[attr-defined]
                    except (AttributeError, TypeError):
                        pass
                raise
            stage_artifacts = dict(stage_receipt.artifacts)
            by_stage[plan.stage_id] = stage_artifacts
            _merge_md_stage_artifacts(
                combined_artifacts,
                plan,
                stage_artifacts,
            )
            commands.extend(stage_receipt.commands)
            generated.extend(
                GeneratedInput(
                    Path(plan.stage_id) / item.relative_path,
                    item.content,
                )
                for item in stage_receipt.generated_inputs
            )
            tools[plan.stage_id] = stage_receipt.metadata.get("tool")
            stage_provenance[plan.stage_id] = {
                "source": "current_invocation",
                "commands": tuple(_command_payload(command) for command in stage_receipt.commands),
                "tool": stage_receipt.metadata.get("tool"),
            }

        primary = next(plan for plan in plans if plan.primary)
        public_artifacts = None if rendered is None else (
            materialize_gromacs_public_artifacts(root, combined_artifacts)
        )
        checkpoint_role = f"{primary.stage_id}.cpt" if rendered else "cpt"
        checkpoint_label = (
            checkpoint_role if "cpt" in combined_artifacts and primary.checkpoint else None
        )
        receipt_artifacts = combined_artifacts if rendered is None else {
            role: path for role, path in combined_artifacts.items() if "." in role
        }
        contract_audit = {} if rendered is None else dict(rendered.audit)
        if rendered is not None:
            primary_runtime = _md_stage_runtime(config, primary.stage_id)
            production_arguments = _string_tuple(
                primary_runtime.get(
                    "mdrun_arguments", config.get("mdrun_arguments", ())
                ),
                "md.production.mdrun_arguments",
            )
            contract_audit["operational_mapping"] = (
                gromacs_operational_mapping(science, production_arguments)
            )
            contract_audit["native_artifacts"] = {
                role: str(path)
                for role, path in sorted(receipt_artifacts.items())
            }
        receipt = BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=tuple(commands),
            artifacts=receipt_artifacts,
            metadata={
                "tools": tools,
                "pipeline": tuple(plan.stage_id for plan in plans),
                "primary_stage": primary.stage_id,
                "skipped_complete_stages": tuple(skipped),
                "recovery": recovery,
                "stage_provenance": stage_provenance,
                "rendered_mdp": {plan.stage_id: plan.mdp_text for plan in plans},
                **({"phase_plan": contract_audit} if contract_audit else {}),
                **(
                    {"scientific_validation": scientific_validation}
                    if scientific_validation
                    else {}
                ),
            },
            parsed={
                "pipeline": tuple(plan.stage_id for plan in plans),
                "primary_stage": primary.stage_id,
                "skipped_complete_stages": tuple(skipped),
                "recovery": recovery,
                "stage_provenance": stage_provenance,
                **({"phase_plan": contract_audit} if contract_audit else {}),
                **(
                    {"scientific_validation": scientific_validation}
                    if scientific_validation
                    else {}
                ),
            },
            generated_inputs=tuple(generated),
        )
        result = _stage_result(
            self.capability,
            request,
            receipt,
            checkpoint_label=checkpoint_label,
        )
        if public_artifacts is None:
            return result
        return StageResult(
            artifacts=public_artifacts,
            payload=result.payload,
            checkpoint=public_artifacts["checkpoint"],
            invocation_provenance=result.invocation_provenance,
        )

    def _execute_trajectory(self, request: StageRequest) -> StageResult:
        science = _science(request)
        from .trajectory_dispatch import execute_trajectory_adapter

        receipt = execute_trajectory_adapter(
            science=science,
            config=_capability_config(request.runtime, self.capability),
            prepared=_binding_view(request),
            upstream=_upstream(request, "md"),
            output_dir=_output_root(request),
            runner=self.runner,
        )
        frame_manifest = _frame_manifest_audit(science)
        if frame_manifest:
            receipt = BackendReceipt(
                state=receipt.state,
                commands=receipt.commands,
                artifacts=receipt.artifacts,
                metadata={
                    **dict(receipt.metadata),
                    "frame_manifest": frame_manifest,
                },
                parsed={
                    **dict(receipt.parsed),
                    "frame_manifest": frame_manifest,
                },
                generated_inputs=receipt.generated_inputs,
            )
        return _stage_result(self.capability, request, receipt)

    def _execute_scoring(self, request: StageRequest) -> StageResult:
        prepared = _binding_view(request)
        science = _science(request)
        upstream = _upstream(request, "trajectory")
        _require_trajectory_geometry(upstream)
        config = _capability_config(request.runtime, self.capability)
        backend = _create_backend(self.capability, config, runner=self.runner)
        model_id = request.task.model_id
        if model_id is None:
            raise BackendConfigurationError("scoring task requires a concrete model_id")
        native_inputs_raw = science.get("native_inputs")
        native_inputs = (
            None
            if native_inputs_raw is None
            else _string_mapping(
                native_inputs_raw,
                f"scoring.models.{model_id}.native_inputs",
            )
        )
        control_text = _required_string(
            science,
            "control_text",
            f"scoring.models.{model_id}",
        )
        control_text, native_inputs, dielectric_audit = _resolve_scoring_dielectric(
            prepared,
            control_text=control_text,
            native_inputs=native_inputs,
        )
        receipt = backend.run(
            AmberToolsScoringRequest(
                model_id=model_id,
                control_text=control_text,
                complex_topology=prepared.topology_artifact("complex"),
                receptor_topology=prepared.topology_artifact("receptor"),
                binder_topology=prepared.topology_artifact("ligand"),
                trajectory=_upstream_artifact(
                    upstream,
                    _string(
                        science.get(
                            "trajectory_artifact",
                            "prepared_trajectory",
                        ),
                        "scoring.trajectory_artifact",
                    ),
                ),
                output_dir=_output_root(request),
                native_inputs=native_inputs,
                expected_frames=_integer(
                    science.get("expected_frames"),
                    f"scoring.models.{model_id}.expected_frames",
                    minimum=1,
                ),
                extra_arguments=_string_tuple(
                    config.get("extra_arguments", ()),
                    "scoring.extra_arguments",
                ),
            )
        )
        parsed = dict(receipt.parsed)
        parsed["dielectric_resolution"] = dielectric_audit
        frame_manifest = _frame_manifest_audit(science)
        if frame_manifest:
            parsed["frame_manifest"] = frame_manifest
        warnings = tuple(dielectric_audit["warnings"])
        if warnings:
            parsed["warnings"] = tuple(
                dict.fromkeys(
                    (
                        *_string_tuple(
                            parsed.get("warnings", ()),
                            "scoring parsed warnings",
                        ),
                        *warnings,
                    )
                )
            )
        metadata = dict(receipt.metadata)
        metadata["dielectric_resolution"] = dielectric_audit
        if frame_manifest:
            metadata["frame_manifest"] = frame_manifest
        if warnings:
            metadata["warnings"] = warnings
        receipt = BackendReceipt(
            state=receipt.state,
            commands=receipt.commands,
            artifacts=receipt.artifacts,
            metadata=metadata,
            parsed=parsed,
            generated_inputs=receipt.generated_inputs,
        )
        return _stage_result(self.capability, request, receipt)

    def _execute_trajectory_qc(self, request: StageRequest) -> StageResult:
        science = _science(request)
        config = _capability_config(request.runtime, self.capability)
        backend = _create_backend(self.capability, config, runner=self.runner)
        reference_audit = _qc_reference_audit(science, request)
        input_text, referenced = _render_stage_input(
            _required_string(science, "input_text", "trajectory_qc"),
            request,
        )
        declared = list(referenced)
        raw_declared = science.get("declared_inputs", ())
        if not _is_sequence(raw_declared):
            raise BackendConfigurationError("trajectory_qc.declared_inputs must be a sequence")
        for descriptor in raw_declared:
            declared.append(_artifact_descriptor(descriptor, request))
        if not declared:
            for stage in sorted(request.upstream):
                declared.extend(
                    request.upstream[stage].artifacts[label]
                    for label in sorted(request.upstream[stage].artifacts)
                )
        if not declared and request.prepared_case is not None:
            prepared = _binding_view(request)
            declared.append(prepared.topology_artifact("complex"))
        output_tables_raw = _mapping(
            science.get("output_tables"),
            "trajectory_qc.output_tables",
        )
        output_tables = {
            _string(label, "qc output label"): Path(
                _string(value, f"trajectory_qc.output_tables.{label}")
            )
            for label, value in output_tables_raw.items()
        }
        receipt = backend.run(
            CpptrajQcRequest(
                input_text=input_text,
                declared_inputs=tuple(dict.fromkeys(declared)),
                output_tables=output_tables,
                output_dir=_output_root(request),
                expected_frames=_optional_integer(
                    science.get("expected_frames"),
                    "trajectory_qc.expected_frames",
                    minimum=1,
                ),
            )
        )
        verdict, metrics, check_results = _evaluate_qc(
            receipt.parsed,
            science.get("checks", ()),
        )
        issues = tuple(
            {
                "code": str(result["id"]),
                "message": _qc_issue_message(result),
                "verdict": str(result["verdict"]),
            }
            for result in check_results
            if result["verdict"] in {"WARN", "BLOCK"}
        )
        parsed = dict(receipt.parsed)
        if reference_audit:
            parsed["reference_definition"] = reference_audit
        parsed.update(
            {
                "metrics": metrics,
                "check_results": check_results,
                "verdict": verdict.value,
                "issues": issues,
            }
        )
        receipt = BackendReceipt(
            state=receipt.state,
            commands=receipt.commands,
            artifacts=receipt.artifacts,
            metadata=receipt.metadata,
            parsed=parsed,
            generated_inputs=receipt.generated_inputs,
        )
        return _stage_result(
            self.capability,
            request,
            receipt,
            qc_verdict=verdict,
        )

    def _execute_assets(self, request: StageRequest) -> StageResult:
        science = _science(request)
        raw_bundles = science.get("bundles")
        if raw_bundles is None:
            bundle_specs, availability = _automatic_asset_bundles(
                request,
                science,
            )
        else:
            if not _is_sequence(raw_bundles) or not raw_bundles:
                raise BackendConfigurationError(
                    "report.assets.bundles must be a non-empty sequence"
                )
            bundle_specs, availability = _explicit_asset_bundles(
                request,
                raw_bundles,
            )
        config = _capability_config(request.runtime, self.capability)
        backend = _create_backend(self.capability, config, runner=self.runner)
        root = _output_root(request)
        seen: set[str] = set()
        commands = []
        artifacts: dict[str, Path] = {}
        generated: list[GeneratedInput] = []
        bundle_metadata: dict[str, object] = {}
        diagnostic_records: dict[str, object] = {}
        asset_records: list[Mapping[str, object]] = []
        for bundle in bundle_specs:
            bundle_id = _identifier(
                bundle["id"],
                "report.assets bundle id",
            )
            if bundle_id in seen:
                raise BackendConfigurationError(f"duplicate report.assets bundle id: {bundle_id}")
            seen.add(bundle_id)
            receipt = backend.run(
                DisplayAssetRequest(
                    reference_artifact=str(bundle.get("reference_artifact", "selected_trajectory")),
                    complex_topology=Path(bundle["complex_topology"]),
                    trajectories=bundle["trajectories"],
                    component_atom_indices=bundle["component_atom_indices"],
                    output_dir=root / bundle_id,
                    alignment_mask=_optional_string(
                        bundle.get("alignment_mask"),
                        f"report.assets.{bundle_id}.alignment_mask",
                    ),
                    start_frame=_integer(
                        bundle.get("start_frame", 1),
                        f"report.assets.{bundle_id}.start_frame",
                        minimum=1,
                    ),
                    end_frame=_optional_integer(
                        bundle.get("end_frame"),
                        f"report.assets.{bundle_id}.end_frame",
                        minimum=1,
                    ),
                    stride=_integer(
                        bundle.get("stride", 1),
                        f"report.assets.{bundle_id}.stride",
                        minimum=1,
                    ),
                    default_replica=_optional_string(
                        bundle.get("default_replica"),
                        f"report.assets.{bundle_id}.default_replica",
                    ),
                    default_state=_integer(
                        bundle.get("default_state", 1),
                        f"report.assets.{bundle_id}.default_state",
                        minimum=1,
                    ),
                    reference_trajectories=bundle.get(
                        "reference_trajectories"
                    ),
                    reference_frame=_integer(
                        bundle.get("reference_frame", 1),
                        f"report.assets.{bundle_id}.reference_frame",
                        minimum=1,
                    ),
                    reference_autoimage_anchor_atom_indices=(
                        None
                        if bundle.get(
                            "reference_autoimage_anchor_atom_indices"
                        )
                        is None
                        else _integer_tuple(
                            bundle[
                                "reference_autoimage_anchor_atom_indices"
                            ],
                            f"report.assets.{bundle_id}."
                            "reference_autoimage_anchor_atom_indices",
                        )
                    ),
                    reference_autoimage_fixed_atom_indices=(
                        None
                        if bundle.get(
                            "reference_autoimage_fixed_atom_indices"
                        )
                        is None
                        else _integer_tuple(
                            bundle[
                                "reference_autoimage_fixed_atom_indices"
                            ],
                            f"report.assets.{bundle_id}."
                            "reference_autoimage_fixed_atom_indices",
                        )
                    ),
                    reference_alignment_atom_indices=(
                        None
                        if bundle.get("reference_alignment_atom_indices") is None
                        else _integer_tuple(
                            bundle["reference_alignment_atom_indices"],
                            f"report.assets.{bundle_id}.reference_alignment_atom_indices",
                        )
                    ),
                    ligand_rmsd_atom_indices=(
                        None
                        if bundle.get("ligand_rmsd_atom_indices") is None
                        else _integer_tuple(
                            bundle["ligand_rmsd_atom_indices"],
                            f"report.assets.{bundle_id}.ligand_rmsd_atom_indices",
                        )
                    ),
                )
            )
            commands.extend(receipt.commands)
            generated.extend(
                GeneratedInput(
                    Path(bundle_id) / item.relative_path,
                    item.content,
                )
                for item in receipt.generated_inputs
            )
            for label, path in receipt.artifacts.items():
                artifact_role = f"{bundle_id}.{label}"
                artifacts[artifact_role] = path
                if label.startswith("structure."):
                    kind = "topology"
                    replica_id: str | None = label.removeprefix("structure.")
                    visibility = "visible"
                elif label.startswith("trajectory."):
                    kind = "trajectory"
                    replica_id = label.removeprefix("trajectory.")
                    visibility = "visible"
                elif label.startswith("cpptraj_input."):
                    # Generated command inputs contain host-local absolute paths.
                    # They remain sealed attempt artifacts but are not portable
                    # report assets.
                    continue
                elif label.startswith("pymol_script."):
                    kind = "pymol_script"
                    replica_id = label.removeprefix("pymol_script.")
                    visibility = "visible"
                elif label.startswith("ligand_equilibrated_rmsd."):
                    kind = "diagnostic"
                    replica_id = label.removeprefix(
                        "ligand_equilibrated_rmsd."
                    )
                    visibility = "hidden"
                elif label.startswith("equilibrated_reference."):
                    kind = "diagnostic"
                    replica_id = label.removeprefix("equilibrated_reference.")
                    visibility = "hidden"
                elif label.startswith("reference_input."):
                    continue
                else:
                    kind = "manifest"
                    replica_id = None
                    visibility = "visible"
                asset_records.append(
                    {
                        "artifact_role": artifact_role,
                        "case_id": bundle_id,
                        "replica_id": replica_id,
                        "kind": kind,
                        "visibility": visibility,
                        "label": (
                            f"{bundle_id} {replica_id or ''} {kind.replace('_', ' ')}"
                        ).strip(),
                    }
                )
            bundle_metadata[bundle_id] = receipt.metadata
            rmsd_tables = receipt.parsed.get("ligand_equilibrated_rmsd")
            if isinstance(rmsd_tables, Mapping) and rmsd_tables:
                diagnostic_records[bundle_id] = {
                    "ligand_rmsd": {
                        "reference_artifact": str(
                            bundle.get("reference_artifact", "selected_trajectory")
                        ),
                        "reference_frame": _integer(
                            bundle.get("reference_frame", 1),
                            f"report.assets.{bundle_id}.reference_frame",
                            minimum=1,
                        ),
                        "reference_time_ps": 0.0,
                        "reference_semantics": "post_npt_production_start",
                        "reference_preprocessing": (
                            "whole_molecules_then_autoimage_ligand_anchor_receptor_fixed"
                            if bundle.get("reference_artifact") == "prepared_reference_trajectory"
                            else "autoimage_ligand_anchor_receptor_fixed"
                        ),
                        "alignment_selection": str(
                            bundle.get(
                                "reference_alignment_selection",
                                "receptor_interface_backbone",
                            )
                        ),
                        "rmsd_selection": str(
                            bundle.get("ligand_rmsd_selection", "ligand_qc")
                        ),
                        "replicas": dict(rmsd_tables),
                    }
                }

        index = root / "asset_index.json"
        portable_roles = {
            str(record["artifact_role"])
            for record in asset_records
        }
        _write_json(
            index,
            {
                "schema_version": "mmpbsa.report_assets.v3",
                "bundles": {
                    bundle_id: {
                        label.removeprefix(bundle_id + "."): str(path.relative_to(root))
                        for label, path in sorted(artifacts.items())
                        if label.startswith(bundle_id + ".")
                        and label in portable_roles
                    }
                    for bundle_id in sorted(seen)
                },
                "availability": availability,
                "diagnostics": diagnostic_records,
            },
        )
        artifacts["asset_index"] = index
        asset_records.append(
            {
                "artifact_role": "asset_index",
                "case_id": None,
                "replica_id": None,
                "kind": "manifest",
                "visibility": "visible",
                "label": "Display asset availability",
            }
        )
        require_artifacts(artifacts)
        receipt = BackendReceipt(
            state=ReceiptState.COMPLETED,
            commands=tuple(commands),
            artifacts=artifacts,
            metadata={"bundles": bundle_metadata},
            parsed={
                "schema_version": "mmpbsa.report_assets.v3",
                "availability": availability,
                "assets": tuple(asset_records),
                "diagnostics": diagnostic_records,
            },
            generated_inputs=tuple(generated),
        )
        return _stage_result(self.capability, request, receipt)


def create_stage_service(
    capability: str,
    *,
    runner: CommandRunner | None = None,
) -> BuiltinStageService:
    return BuiltinStageService(capability, runner=runner)


def _md_stage_plans(
    science: Mapping[str, object],
    request: StageRequest,
) -> tuple[_MdStagePlan, ...]:
    raw_stages = science.get("stages")
    if not _is_sequence(raw_stages) or not raw_stages:
        raise BackendConfigurationError("md.stages must be a non-empty sequence")
    allowed = {
        "stage_id",
        "mdp_text",
        "coordinates",
        "state_input",
        "reference",
        "expected_suffixes",
        "checkpoint",
        "primary",
    }
    entries: list[Mapping[str, object]] = []
    stage_ids: list[str] = []
    item_primaries: list[str] = []
    for index, raw in enumerate(raw_stages):
        entry = _mapping(raw, f"md.stages[{index}]")
        unknown = set(entry).difference(allowed)
        missing = {"stage_id", "mdp_text", "coordinates"}.difference(entry)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise BackendConfigurationError(f"invalid md.stages[{index}]: " + "; ".join(details))
        stage_id = _identifier(
            entry["stage_id"],
            f"md.stages[{index}].stage_id",
        )
        if stage_id in stage_ids:
            raise BackendConfigurationError(f"duplicate MD pipeline stage_id: {stage_id}")
        primary_raw = entry.get("primary", False)
        if not isinstance(primary_raw, bool):
            raise BackendConfigurationError(f"md.stages[{index}].primary must be boolean")
        if primary_raw:
            item_primaries.append(stage_id)
        stage_ids.append(stage_id)
        entries.append(entry)
    if len(item_primaries) > 1:
        raise BackendConfigurationError("MD pipeline declares more than one primary stage")
    primary_raw = science.get("primary_stage")
    if primary_raw is not None:
        primary_id = _identifier(primary_raw, "md.primary_stage")
        if primary_id not in stage_ids:
            raise BackendConfigurationError(f"md.primary_stage is not declared: {primary_id}")
        if item_primaries and item_primaries[0] != primary_id:
            raise BackendConfigurationError("md.primary_stage conflicts with a stage primary flag")
    elif item_primaries:
        primary_id = item_primaries[0]
    else:
        primary_id = stage_ids[-1]

    resolved: list[_MdStagePlan] = []
    expected_by_stage: dict[str, tuple[str, ...]] = {}
    for index, (stage_id, entry) in enumerate(zip(stage_ids, entries, strict=True)):
        primary = stage_id == primary_id
        checkpoint_raw = entry.get("checkpoint")
        if checkpoint_raw is not None and not isinstance(
            checkpoint_raw,
            bool,
        ):
            raise BackendConfigurationError(f"md.stages[{index}].checkpoint must be boolean")
        expected_raw = entry.get("expected_suffixes")
        if expected_raw is None:
            checkpoint = primary if checkpoint_raw is None else checkpoint_raw
            expected_values = ["tpr", "log", "edr", "gro"]
            if primary:
                expected_values.append("xtc")
            if checkpoint:
                expected_values.append("cpt")
            expected = tuple(expected_values)
        else:
            expected = _string_tuple(
                expected_raw,
                f"md.stages[{index}].expected_suffixes",
            )
            checkpoint = "cpt" in expected if checkpoint_raw is None else checkpoint_raw
        if len(set(expected)) != len(expected):
            raise BackendConfigurationError(
                f"md.stages[{index}].expected_suffixes contains duplicates"
            )
        if any(not _IDENTIFIER.fullmatch(item) for item in expected):
            raise BackendConfigurationError(f"md.stages[{index}].expected_suffixes is invalid")
        required = {"tpr", "gro"}
        if primary:
            required.add("xtc")
        if checkpoint:
            required.add("cpt")
        missing_required = required.difference(expected)
        if missing_required:
            raise BackendConfigurationError(
                f"md.stages[{index}] expected_suffixes is missing: "
                + ", ".join(sorted(missing_required))
            )
        previous = tuple(stage_ids[:index])
        coordinates = _parse_md_reference(
            entry["coordinates"],
            f"md.stages[{index}].coordinates",
            previous,
            expected_by_stage,
        )
        state_input = (
            None
            if entry.get("state_input") is None
            else _parse_md_reference(
                entry["state_input"],
                f"md.stages[{index}].state_input",
                previous,
                expected_by_stage,
            )
        )
        reference = (
            None
            if entry.get("reference") is None
            else _parse_md_reference(
                entry["reference"],
                f"md.stages[{index}].reference",
                previous,
                expected_by_stage,
            )
        )
        plan = _MdStagePlan(
            stage_id=stage_id,
            mdp_text=_render_mdp_text(
                _required_string(
                    entry,
                    "mdp_text",
                    f"md.stages[{index}]",
                ),
                request,
            ),
            coordinates=coordinates,
            state_input=state_input,
            reference=reference,
            expected_suffixes=expected,
            checkpoint=checkpoint,
            primary=primary,
        )
        resolved.append(plan)
        expected_by_stage[stage_id] = expected
    return tuple(resolved)


def _parse_md_reference(
    value: object,
    label: str,
    previous_stages: tuple[str, ...],
    expected_by_stage: Mapping[str, tuple[str, ...]],
) -> _MdInputReference:
    reference = _mapping(value, label)
    kind = _string(reference.get("kind"), f"{label}.kind")
    if kind == "prepared":
        if set(reference) != {"kind", "role"}:
            raise BackendConfigurationError(
                f"{label} prepared reference requires exactly kind/role"
            )
        return _MdInputReference(
            kind=kind,
            role=_string(reference["role"], f"{label}.role"),
        )
    if kind != "stage":
        raise BackendConfigurationError(f"{label}.kind must be 'prepared' or 'stage'")
    if set(reference).difference({"kind", "stage_id", "artifact"}):
        raise BackendConfigurationError(f"{label} stage reference has unknown fields")
    if "artifact" not in reference:
        raise BackendConfigurationError(f"{label} stage reference requires artifact")
    if reference.get("stage_id") is None:
        if not previous_stages:
            raise BackendConfigurationError(f"{label} cannot infer a previous stage")
        stage_id = previous_stages[-1]
    else:
        stage_id = _identifier(reference["stage_id"], f"{label}.stage_id")
    if stage_id not in previous_stages:
        raise BackendConfigurationError(f"{label} must reference an earlier declared stage")
    artifact = _identifier(reference["artifact"], f"{label}.artifact")
    if artifact not in expected_by_stage[stage_id]:
        raise BackendConfigurationError(f"{label} references undeclared {stage_id}.{artifact}")
    return _MdInputReference(
        kind=kind,
        stage_id=stage_id,
        artifact=artifact,
    )


def _resolve_md_reference(
    reference: _MdInputReference,
    prepared: PreparedMdView,
    by_stage: Mapping[str, Mapping[str, Path]],
    current_stage: str,
) -> Path:
    if reference.kind == "prepared":
        assert reference.role is not None
        try:
            return prepared.artifact(reference.role)
        except KeyError as exc:
            raise BackendConfigurationError(
                f"MD stage {current_stage!r} references missing PreparedCase "
                f"role {reference.role!r}"
            ) from exc
    assert reference.stage_id is not None
    assert reference.artifact is not None
    try:
        path = Path(by_stage[reference.stage_id][reference.artifact]).resolve()
    except KeyError as exc:
        raise BackendConfigurationError(
            f"MD stage {current_stage!r} cannot resolve {reference.stage_id}.{reference.artifact}"
        ) from exc
    if not _nonempty_file(path):
        raise BackendConfigurationError(f"MD stage input is missing or empty: {path}")
    return path


def _render_mdp_text(text: str, request: StageRequest) -> str:
    raw_replica = request.science.get("_replica")
    values: Mapping[str, object] = {}
    if raw_replica is not None:
        replica = _mapping(raw_replica, "science._replica")
        values = {
            "seed": replica.get("seed"),
            "id": replica.get("replica_id"),
            "index": replica.get("replica_index"),
        }
    pattern = re.compile(r"\{\{replica\.(seed|id|index)\}\}")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = values.get(key)
        if value is None:
            raise BackendConfigurationError(f"MDP placeholder replica.{key} is unavailable")
        return str(value)

    rendered = pattern.sub(replace, text)
    if "{{" in rendered or "}}" in rendered:
        raise BackendConfigurationError("MDP text contains an unknown or malformed placeholder")
    return rendered


def _md_stage_runtime(
    config: Mapping[str, object],
    stage_id: str,
) -> Mapping[str, object]:
    raw = config.get("stage_overrides")
    if raw is None:
        return {}
    overrides = _mapping(raw, "runtime.md.stage_overrides")
    value = overrides.get(stage_id)
    if value is None:
        return {}
    return _mapping(value, f"runtime.md.stage_overrides.{stage_id}")


def _md_resource_threads(
    request: StageRequest,
    config: Mapping[str, object],
) -> int:
    base_threads = _integer(
        config.get("threads_per_job", 1),
        "md.threads_per_job",
        minimum=1,
    )
    science = _science(request)
    if is_md_phase_plan_v2(science):
        science = dict(render_gromacs_phase_plan(science).pipeline_science)
    effective_threads: list[int] = []
    for plan in _md_stage_plans(science, request):
        stage_config = _md_stage_runtime(config, plan.stage_id)
        effective_threads.append(
            _integer(
                stage_config.get("threads_per_job", base_threads),
                f"md.{plan.stage_id}.threads_per_job",
                minimum=1,
            )
        )
    return max(effective_threads)


def _md_stage_root(root: Path, stage_id: str) -> Path:
    _identifier(stage_id, "MD stage_id")
    resolved_root = Path(root).resolve()
    raw = resolved_root / stage_id
    if raw.is_symlink():
        raise BackendConfigurationError(f"MD stage output cannot be a symlink: {raw}")
    stage_root = raw.resolve()
    try:
        stage_root.relative_to(resolved_root)
    except ValueError as exc:  # pragma: no cover - identifier is already safe
        raise BackendConfigurationError(f"MD stage output escapes output root: {stage_id}") from exc
    return stage_root


def _clear_md_stage_root(root: Path, stage_id: str) -> None:
    stage_root = _md_stage_root(root, stage_id)
    if not stage_root.exists():
        return
    if not stage_root.is_dir():
        raise BackendConfigurationError(f"MD stage output is not a directory: {stage_root}")
    shutil.rmtree(stage_root)


def _nonempty_file(path: Path) -> bool:
    candidate = Path(path)
    return candidate.is_file() and candidate.stat().st_size > 0


def _md_completion_path(stage_root: Path) -> Path:
    return Path(stage_root).resolve() / _MD_COMPLETION_FILE


def _invalidate_md_stage_completion(stage_root: Path) -> None:
    marker = _md_completion_path(stage_root)
    if marker.is_dir() and not marker.is_symlink():
        shutil.rmtree(marker)
    else:
        marker.unlink(missing_ok=True)


def _write_md_stage_completion(
    stage_root: Path,
    plan: _MdStagePlan,
    receipt: BackendReceipt,
) -> None:
    expected_roles = (*plan.expected_suffixes, "mdp_input")
    records: dict[str, object] = {}
    resolved_root = Path(stage_root).resolve()
    for role in expected_roles:
        try:
            artifact = Path(receipt.artifacts[role]).resolve(strict=True)
        except KeyError as exc:
            raise BackendOutputError(
                f"MD stage {plan.stage_id!r} did not return artifact {role!r}"
            ) from exc
        expected_name = f"{plan.stage_id}.mdp" if role == "mdp_input" else f"{plan.stage_id}.{role}"
        expected = (resolved_root / expected_name).resolve()
        if artifact != expected or not _nonempty_file(artifact):
            raise BackendOutputError(
                f"MD stage {plan.stage_id!r} completion artifact is invalid: {role}"
            )
        stat = artifact.stat()
        records[role] = {
            "relative_path": expected_name,
            "size": stat.st_size,
            "sha256": _file_sha256(artifact),
        }
    command_records = tuple(_command_payload(command) for command in receipt.commands)
    tool = _json_value(receipt.metadata.get("tool"))
    if not _valid_completion_provenance(command_records, tool):
        raise BackendOutputError(
            f"MD stage {plan.stage_id!r} returned invalid execution provenance"
        )
    _write_json(
        _md_completion_path(resolved_root),
        {
            "schema_version": _MD_COMPLETION_SCHEMA,
            "stage_id": plan.stage_id,
            "mdp_text_sha256": _text_sha256(_terminal_newline(plan.mdp_text)),
            "expected_suffixes": plan.expected_suffixes,
            "artifacts": records,
            "commands": command_records,
            "tool": tool,
        },
    )


def _read_md_stage_completion(
    stage_root: Path,
    plan: _MdStagePlan,
    artifacts: Mapping[str, Path],
) -> Mapping[str, object] | None:
    marker = _md_completion_path(stage_root)
    if marker.is_symlink() or not _nonempty_file(marker):
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    raw_suffixes = payload.get("expected_suffixes")
    if not _is_sequence(raw_suffixes) or not all(isinstance(item, str) for item in raw_suffixes):
        return None
    if (
        payload.get("schema_version") != _MD_COMPLETION_SCHEMA
        or payload.get("stage_id") != plan.stage_id
        or tuple(raw_suffixes) != plan.expected_suffixes
        or payload.get("mdp_text_sha256") != _text_sha256(_terminal_newline(plan.mdp_text))
    ):
        return None
    if not _valid_completion_provenance(
        payload.get("commands"),
        payload.get("tool"),
    ):
        return None
    raw_records = payload.get("artifacts")
    if not isinstance(raw_records, Mapping):
        return None
    expected_roles = (*plan.expected_suffixes, "mdp_input")
    if set(raw_records) != set(expected_roles):
        return None
    resolved_root = Path(stage_root).resolve()
    for role in expected_roles:
        record = raw_records.get(role)
        if not isinstance(record, Mapping):
            return None
        expected_name = f"{plan.stage_id}.mdp" if role == "mdp_input" else f"{plan.stage_id}.{role}"
        artifact = Path(artifacts[role])
        try:
            resolved = artifact.resolve(strict=True)
        except OSError:
            return None
        if (
            artifact.is_symlink()
            or resolved != (resolved_root / expected_name).resolve()
            or not _nonempty_file(resolved)
            or record.get("relative_path") != expected_name
        ):
            return None
        size = record.get("size")
        digest = record.get("sha256")
        try:
            observed_size = resolved.stat().st_size
            observed_digest = _file_sha256(resolved)
        except OSError:
            return None
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or observed_size != size
            or not isinstance(digest, str)
            or observed_digest != digest
        ):
            return None
    return payload


def _valid_completion_provenance(
    raw_commands: object,
    raw_tool: object,
) -> bool:
    if not _is_sequence(raw_commands) or not raw_commands:
        return False
    for raw_command in raw_commands:
        if not isinstance(raw_command, Mapping):
            return False
        argv = raw_command.get("argv")
        returncode = raw_command.get("returncode")
        if (
            set(raw_command) != {"argv", "cwd", "returncode", "stdout", "stderr"}
            or not _is_sequence(argv)
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
            or not isinstance(raw_command.get("cwd"), str)
            or not raw_command["cwd"]
            or isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or returncode != 0
            or not isinstance(raw_command.get("stdout"), str)
            or not isinstance(raw_command.get("stderr"), str)
        ):
            return False
    if not isinstance(raw_tool, Mapping):
        return False
    version = raw_tool.get("version")
    return (
        isinstance(raw_tool.get("executable"), str)
        and bool(raw_tool["executable"])
        and isinstance(raw_tool.get("raw_version"), str)
        and bool(raw_tool["raw_version"])
        and _is_sequence(version)
        and bool(version)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in version)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _terminal_newline(value: str) -> str:
    return value.rstrip("\n") + "\n"


def _command_payload(
    command: CommandRecord | CommandSpec,
) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "argv": tuple(command.argv),
        "cwd": str(command.cwd),
    }
    if isinstance(command, CommandRecord):
        payload.update(
            {
                "returncode": command.returncode,
                "stdout": command.stdout,
                "stderr": command.stderr,
            }
        )
    return payload


def _merge_md_stage_artifacts(
    combined: dict[str, Path],
    plan: _MdStagePlan,
    stage_artifacts: Mapping[str, Path],
) -> None:
    for artifact, path in stage_artifacts.items():
        combined[f"{plan.stage_id}.{artifact}"] = path
        if plan.primary:
            combined[artifact] = path


def _explicit_asset_bundles(
    request: StageRequest,
    raw_bundles: Sequence[object],
) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, object]]:
    specs: list[Mapping[str, object]] = []
    availability: dict[str, object] = {}
    for raw_bundle in raw_bundles:
        bundle = _mapping(raw_bundle, "report.assets bundle")
        bundle_id = _identifier(
            bundle.get("id"),
            "report.assets bundle id",
        )
        trajectories_raw = _mapping(
            bundle.get("trajectories"),
            f"report.assets.{bundle_id}.trajectories",
        )
        trajectories = {
            _identifier(replica, "report.assets replica"): (
                _artifact_descriptor(descriptor, request)
            )
            for replica, descriptor in trajectories_raw.items()
        }
        components_raw = _mapping(
            bundle.get("component_atom_indices"),
            f"report.assets.{bundle_id}.component_atom_indices",
        )
        components = {
            _identifier(component, "report.assets component"): _integer_tuple(
                indices,
                f"report.assets.{bundle_id}.{component}",
            )
            for component, indices in components_raw.items()
        }
        specs.append(
            {
                "id": bundle_id,
                "complex_topology": _artifact_descriptor(
                    bundle.get("complex_topology"),
                    request,
                ),
                "trajectories": trajectories,
                "component_atom_indices": components,
                **{
                    key: bundle[key]
                    for key in (
                        "alignment_mask",
                        "start_frame",
                        "end_frame",
                        "stride",
                        "default_replica",
                        "default_state",
                    )
                    if key in bundle
                },
            }
        )
        availability[bundle_id] = {
            "status": "available",
            "replicas": {replica: {"status": "available"} for replica in sorted(trajectories)},
        }
    return tuple(specs), availability


def _automatic_asset_bundles(
    request: StageRequest,
    science: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, object]]:
    """Derive bundles only from the verified StageInputs supplied by core."""

    analysis = _upstream(request, "analysis")
    expected_by_case = _analysis_expected_replicas(analysis)
    prepared_by_case = {
        key.removeprefix("prepared:"): value
        for key, value in request.upstream.items()
        if key.startswith("prepared:")
    }
    trajectory_by_case: dict[str, dict[str, StageInput]] = {}
    for key, value in request.upstream.items():
        if not key.startswith("trajectory:"):
            continue
        parts = key.split(":")
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise BackendConfigurationError(f"invalid assets trajectory upstream key: {key!r}")
        trajectory_by_case.setdefault(parts[1], {})[parts[2]] = value
    case_ids = tuple(
        sorted(set(expected_by_case) | set(prepared_by_case) | set(trajectory_by_case))
    )
    specs: list[Mapping[str, object]] = []
    availability: dict[str, object] = {}
    for case_id in case_ids:
        settings = _asset_case_settings(science, case_id)
        expected = tuple(expected_by_case.get(case_id, ()))
        observed = trajectory_by_case.get(case_id, {})
        replicas = tuple(sorted(set(expected) | set(observed)))
        prepared = prepared_by_case.get(case_id)
        if prepared is None:
            availability[case_id] = {
                "status": "unavailable",
                "reason": "prepared_case_unavailable",
                "replicas": {
                    replica: {
                        "status": "unavailable",
                        "reason": "prepared_case_unavailable",
                    }
                    for replica in replicas
                },
            }
            continue
        topology_role = _string(
            settings.get("complex_topology_role", "complex_topology"),
            f"report.assets.{case_id}.complex_topology_role",
        )
        topology = prepared.artifacts.get(topology_role)
        if topology is None:
            availability[case_id] = {
                "status": "unavailable",
                "reason": f"prepared_artifact_missing:{topology_role}",
                "replicas": {
                    replica: {
                        "status": "unavailable",
                        "reason": f"prepared_artifact_missing:{topology_role}",
                    }
                    for replica in replicas
                },
            }
            continue
        topology = _upstream_artifact(prepared, topology_role)
        (
            components,
            component_availability,
            diagnostic_selections,
        ) = _prepared_components(
            prepared,
            settings.get("components"),
            case_id,
        )
        trajectory_artifact = _string(
            settings.get("trajectory_artifact", "centered_trajectory"),
            f"report.assets.{case_id}.trajectory_artifact",
        )
        engine_neutral = any(
            isinstance(item.payload.get("backend_metadata"), Mapping)
            and item.payload["backend_metadata"].get("contract") == "mmpbsa.trajectory.v3"
            for item in observed.values()
        )
        reference_artifact = (
            "prepared_reference_trajectory" if engine_neutral else "selected_trajectory"
        )
        available: dict[str, Path] = {}
        references: dict[str, Path] = {}
        replica_availability: dict[str, object] = {}
        for replica in replicas:
            stage_input = observed.get(replica)
            if stage_input is None:
                replica_availability[replica] = {
                    "status": "unavailable",
                    "reason": "trajectory_stage_unavailable",
                }
                continue
            try:
                _require_trajectory_geometry(stage_input)
            except BackendConfigurationError:
                replica_availability[replica] = {
                    "status": "unavailable", "reason": "unverified_periodic_geometry"
                }
                continue
            if trajectory_artifact not in stage_input.artifacts:
                replica_availability[replica] = {
                    "status": "unavailable",
                    "reason": (f"trajectory_artifact_missing:{trajectory_artifact}"),
                }
                continue
            if reference_artifact not in stage_input.artifacts:
                replica_availability[replica] = {
                    "status": "unavailable",
                    "reason": (
                        f"trajectory_artifact_missing:{reference_artifact}"
                    ),
                }
                continue
            available[replica] = _upstream_artifact(
                stage_input,
                trajectory_artifact,
            )
            references[replica] = _upstream_artifact(
                stage_input,
                reference_artifact,
            )
            replica_availability[replica] = {"status": "available"}
        unavailable = tuple(
            replica
            for replica, state in replica_availability.items()
            if state["status"] == "unavailable"
        )
        if not available:
            availability[case_id] = {
                "status": "unavailable",
                "reason": "no_trajectory_available",
                "replicas": replica_availability,
                "components": component_availability,
            }
            continue
        availability[case_id] = {
            "status": "partial" if unavailable else "available",
            "replicas": replica_availability,
            "components": component_availability,
        }
        default_replica = _optional_string(
            settings.get("default_replica"),
            f"report.assets.{case_id}.default_replica",
        )
        if default_replica not in available:
            default_replica = sorted(available)[0]
        specs.append(
            {
                "id": case_id,
                "complex_topology": topology,
                "trajectories": available,
                "component_atom_indices": components,
                "reference_trajectories": references,
                "reference_artifact": reference_artifact,
                "reference_frame": 1,
                "reference_autoimage_anchor_atom_indices": (
                    diagnostic_selections["ligand_all"]
                ),
                "reference_autoimage_fixed_atom_indices": (
                    diagnostic_selections["receptor_all"]
                ),
                "reference_alignment_atom_indices": diagnostic_selections[
                    "receptor_interface_backbone"
                ],
                "ligand_rmsd_atom_indices": diagnostic_selections[
                    "ligand_qc"
                ],
                "reference_alignment_selection": "receptor_interface_backbone",
                "ligand_rmsd_selection": "ligand_qc",
                "alignment_mask": settings.get("alignment_mask"),
                "start_frame": settings.get("start_frame", 1),
                "end_frame": settings.get("end_frame"),
                "stride": settings.get("stride", 1),
                "default_replica": default_replica,
                "default_state": settings.get("default_state", 1),
            }
        )
    return tuple(specs), availability


def _analysis_expected_replicas(
    analysis: StageInput,
) -> Mapping[str, tuple[str, ...]]:
    raw_cases = analysis.payload.get("cases", ())
    if not _is_sequence(raw_cases):
        raise BackendConfigurationError("analysis payload cases must be a sequence")
    result: dict[str, tuple[str, ...]] = {}
    for index, raw_case in enumerate(raw_cases):
        case = _mapping(raw_case, f"analysis.cases[{index}]")
        case_id = _identifier(
            case.get("case_id"),
            f"analysis.cases[{index}].case_id",
        )
        record_raw = case.get("record")
        if record_raw is None:
            result[case_id] = ()
            continue
        record = _mapping(
            record_raw,
            f"analysis.cases[{index}].record",
        )
        raw_replicas = record.get("expected_replicas", ())
        if not _is_sequence(raw_replicas):
            raise BackendConfigurationError(
                f"analysis case {case_id!r} expected_replicas must be a sequence"
            )
        result[case_id] = tuple(
            _identifier(
                replica,
                f"analysis case {case_id!r} replica",
            )
            for replica in raw_replicas
        )
    return result


def _asset_case_settings(
    science: Mapping[str, object],
    case_id: str,
) -> dict[str, object]:
    supported = {
        "complex_topology_role",
        "trajectory_artifact",
        "components",
        "alignment_mask",
        "start_frame",
        "end_frame",
        "stride",
        "default_replica",
        "default_state",
    }
    settings = {key: science[key] for key in supported if key in science}
    display_raw = science.get("display")
    if display_raw is not None:
        display = _mapping(display_raw, "report.assets.display")
        settings.update({key: display[key] for key in supported if key in display})
    cases_raw = science.get("case_settings")
    if cases_raw is not None:
        cases = _mapping(cases_raw, "report.assets.case_settings")
        if case_id in cases:
            case_settings = _mapping(
                cases[case_id],
                f"report.assets.case_settings.{case_id}",
            )
            settings.update({key: case_settings[key] for key in supported if key in case_settings})
    return settings


def _create_backend(
    capability: str,
    config: Mapping[str, object],
    *,
    runner: CommandRunner | None,
) -> object:
    if capability == "admission":
        from .parmed import create_parmed_admission_backend

        return create_parmed_admission_backend(config, runner=runner)
    if capability == "md":
        profile = config.get("_md_engine_profile")
        if profile is not None and profile != GROMACS_CANONICAL_PROFILE:
            raise BackendConfigurationError(
                "built-in GROMACS MD requires md_engine_profile "
                f"{GROMACS_CANONICAL_PROFILE!r}; got {profile!r}"
            )
        from .gromacs import create_gromacs_md_backend

        return create_gromacs_md_backend(config, runner=runner)
    if capability == "trajectory":
        contract = config.get("_stage_contract")
        if contract == "mmpbsa.trajectory.v3":
            from .trajectory_v3 import create_trajectory_v3_backend

            return create_trajectory_v3_backend(config, runner=runner)
        if contract is not None:
            raise BackendConfigurationError(
                f"unsupported trajectory stage contract: {contract!r}"
            )
        from .trajectory import create_trajectory_backend

        return create_trajectory_backend(config, runner=runner)
    if capability == "scoring":
        from .ambertools import create_ambertools_scoring_backend

        return create_ambertools_scoring_backend(config, runner=runner)
    if capability == "trajectory_qc":
        from .cpptraj import create_cpptraj_qc_backend

        return create_cpptraj_qc_backend(config, runner=runner)
    if capability == "report.assets":
        from .display import create_display_asset_backend

        return create_display_asset_backend(config, runner=runner)
    raise BackendConfigurationError(f"unknown capability: {capability!r}")


def _probe_backend(
    capability: str,
    backend: object,
    cwd: Path,
) -> Mapping[str, ToolIdentity | Mapping[str, object]]:
    if capability == "admission":
        return {"parmed": backend.loader.probe()}
    if capability == "md":
        return {"gromacs": backend.probe(cwd=cwd)}
    if capability == "trajectory":
        identities = {
            "cpptraj": probe_tool(
                backend.cpptraj, cwd=cwd, runner=backend.runner
            )
        }
        if hasattr(backend, "gromacs"):
            identities["gromacs"] = probe_tool(
                backend.gromacs, cwd=cwd, runner=backend.runner
            )
        else:
            from .trajectory_v3 import (
                probe_mdtraj_identity,
                probe_parmed_identity,
            )

            identities["mdtraj"] = probe_mdtraj_identity()
            identities["parmed"] = probe_parmed_identity()
        return identities
    if capability == "scoring":
        return {
            "mmpbsa": probe_tool(
                backend.tool,
                cwd=cwd,
                runner=backend.runner,
            ),
            "sander": probe_tool(
                backend.sander,
                cwd=cwd,
                runner=backend.runner,
            ),
        }
    if capability == "trajectory_qc":
        return {
            "cpptraj": probe_tool(
                backend.tool,
                cwd=cwd,
                runner=backend.runner,
            )
        }
    return {
        "cpptraj": probe_tool(
            backend.cpptraj,
            cwd=cwd,
            runner=backend.runner,
        )
    }


def _compatibility(
    capability: str,
    identities: Mapping[str, ToolIdentity | Mapping[str, object]],
    *,
    stage_contract: str | None = None,
) -> BackendCompatibility:
    majors: list[str] = []
    versions: list[str] = []
    for name, identity in sorted(identities.items()):
        if isinstance(identity, ToolIdentity):
            major = identity.major
            raw = identity.raw_version
        else:
            raw_version = identity.get("raw_version")
            version = identity.get("version")
            if raw_version is None or not _is_sequence(version) or not version:
                raise BackendOutputError(f"{name} probe did not return raw_version/version")
            major = int(version[0])
            raw = str(raw_version)
        majors.append(f"{name}={major}")
        versions.append(f"{name}={raw}")
    contract_version = "v3" if (
        capability in {"scoring", "report.assets"}
        or stage_contract
        in {MD_PHASE_PLAN_V2, "mmpbsa.trajectory.v3"}
    ) else "v2"
    if capability == "trajectory" and stage_contract == "mmpbsa.trajectory.v3":
        contract_version = "v4"  # Geometry fix only; MD compatibility is unchanged.
    return BackendCompatibility(
        provider_id=PROVIDER_ID,
        adapter_contract=f"mmpbsa.backends.{capability}.{contract_version}",
        engine="+".join(sorted(identities)),
        engine_major=";".join(majors),
        engine_version="; ".join(versions),
    )


def _capability_config(
    runtime: Mapping[str, object],
    capability: str,
) -> dict[str, object]:
    raw = runtime.get(capability)
    if raw is None:
        result = dict(runtime)
    else:
        result = dict(_mapping(raw, f"runtime.{capability}"))
        for key in ("environment", "threads_per_job", "gpu_id", "probe_cwd"):
            if key in runtime and key not in result:
                result[key] = runtime[key]
    global_environment = _string_mapping(
        result.get("environment", {}),
        f"runtime.{capability}.environment",
    )
    tool_keys = {
        "admission": (),
        "md": ("tool",),
        "trajectory": ("gromacs", "cpptraj"),
        "scoring": ("tool", "sander"),
        "trajectory_qc": ("tool",),
        "report.assets": ("cpptraj",),
    }[capability]
    for key in tool_keys:
        nested_raw = result.get(key)
        if nested_raw is None:
            if len(tool_keys) == 1:
                existing = _string_mapping(
                    result.get("environment", {}),
                    f"runtime.{capability}.environment",
                )
                result["environment"] = {**global_environment, **existing}
            continue
        nested = dict(_mapping(nested_raw, f"runtime.{capability}.{key}"))
        local_environment = _string_mapping(
            nested.get("environment", {}),
            f"runtime.{capability}.{key}.environment",
        )
        nested["environment"] = {
            **global_environment,
            **local_environment,
        }
        result[key] = nested
    return result


def _require_request(request: StageRequest) -> None:
    if not isinstance(request, StageRequest):
        raise TypeError("stage services require mmpbsa.core.StageRequest")
    if not isinstance(request.upstream, Mapping) or any(
        not isinstance(value, StageInput) for value in request.upstream.values()
    ):
        raise TypeError("StageRequest.upstream must contain mmpbsa.core.StageInput values")


def _md_view(request: StageRequest) -> PreparedMdView:
    if not isinstance(request.prepared_case, PreparedMdView):
        raise BackendConfigurationError(
            f"{request.task.label} requires a verified PreparedMdView"
        )
    return request.prepared_case


def _binding_view(
    request: StageRequest,
) -> PreparedBindingView | PreparedTrajectoryView:
    if not isinstance(
        request.prepared_case,
        (PreparedBindingView, PreparedTrajectoryView),
    ):
        raise BackendConfigurationError(
            f"{request.task.label} requires a verified prepared binding view"
        )
    return request.prepared_case


def _amber_atom_mask(
    prepared: PreparedBindingView | PreparedTrajectoryView,
    selection_name: str,
) -> str:
    try:
        atom_indices = prepared.binding_selection(
            selection_name
        ).scoring_atom_indices
    except KeyError as exc:
        raise BackendConfigurationError(
            f"unknown primary-binding selection: {selection_name!r}"
        ) from exc
    if not atom_indices:
        raise BackendConfigurationError(
            f"primary-binding selection {selection_name!r} has no scoring atoms"
        )
    return "@" + ",".join(str(index) for index in atom_indices)


def _frame_manifest_audit(
    science: Mapping[str, object],
) -> Mapping[str, object]:
    raw_digest = science.get("frame_manifest_digest")
    raw_manifest = science.get("frame_manifest")
    if raw_digest is None and raw_manifest is None:
        return {}
    digest = _string(raw_digest, "frame_manifest_digest")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise BackendConfigurationError(
            "frame_manifest_digest must be a lowercase SHA-256 digest"
        )
    result: dict[str, object] = {"digest": digest}
    if raw_manifest is not None:
        result["selection"] = dict(
            _mapping(raw_manifest, "frame_manifest")
        )
    return result


def _upstream(request: StageRequest, stage: str) -> StageInput:
    try:
        value = request.upstream[stage]
    except KeyError as exc:
        raise BackendConfigurationError(
            f"{request.task.label} requires upstream stage {stage!r}"
        ) from exc
    if not isinstance(value, StageInput):
        raise TypeError(f"upstream {stage!r} is not a core StageInput")
    return value


def _upstream_artifact(value: StageInput, label: str) -> Path:
    try:
        path = Path(value.artifacts[label]).resolve()
    except KeyError as exc:
        raise BackendConfigurationError(
            f"upstream {value.task.label} has no artifact {label!r}"
        ) from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise BackendConfigurationError(f"upstream artifact is missing or empty: {path}")
    return path


def _science(request: StageRequest) -> dict[str, object]:
    """Return only the stage's explicit executable science mapping."""

    return {
        str(key): value
        for key, value in request.science.items()
        if key != "_profile"
    }


def _output_root(request: StageRequest) -> Path:
    attempt = Path(request.attempt_dir).resolve()
    root = (attempt / "output").resolve()
    try:
        root.relative_to(attempt)
    except ValueError as exc:  # pragma: no cover - defensive path invariant
        raise BackendConfigurationError("stage output escapes attempt_dir") from exc
    return root


def _stage_result(
    capability: str,
    request: StageRequest,
    receipt: BackendReceipt,
    *,
    checkpoint_label: str | None = None,
    qc_verdict: QcVerdict | None = None,
) -> StageResult:
    attempt = Path(request.attempt_dir).resolve()
    normalized_artifacts: dict[str, Path] = {}
    for label, raw_path in receipt.artifacts.items():
        path = Path(raw_path).resolve()
        try:
            path.relative_to(attempt)
        except ValueError as exc:
            raise BackendOutputError(
                f"backend artifact escapes attempt_dir: {label}={path}"
            ) from exc
        if not path.is_file() or path.stat().st_size == 0:
            raise BackendOutputError(f"backend artifact is missing or empty: {label}={path}")
        normalized_artifacts[str(label)] = path

    invocation = Path(request.invocation_dir).resolve()
    try:
        invocation.relative_to(attempt)
    except ValueError as exc:
        raise BackendOutputError(f"invocation_dir escapes attempt_dir: {invocation}") from exc
    invocation.mkdir(parents=True, exist_ok=True)
    receipt_path = invocation / "backend_receipt.json"
    receipt_payload = {
        "schema_version": "mmpbsa.backend_receipt.v1",
        "provider_id": PROVIDER_ID,
        "capability": capability,
        "state": receipt.state.value,
        "commands": [_command_payload(command) for command in receipt.commands],
        "artifacts": {label: str(path) for label, path in normalized_artifacts.items()},
        "metadata": _json_value(receipt.metadata),
        "parsed": _json_value(receipt.parsed),
    }
    _write_json(receipt_path, receipt_payload)
    payload = dict(_json_value(receipt.parsed))
    payload["backend_metadata"] = _json_value(receipt.metadata)
    checkpoint = (
        normalized_artifacts.get(checkpoint_label) if checkpoint_label is not None else None
    )
    return StageResult(
        artifacts=normalized_artifacts,
        payload=payload,
        qc_verdict=qc_verdict,
        checkpoint=checkpoint,
        invocation_provenance={
            "provider_id": PROVIDER_ID,
            "capability": capability,
            "receipt": str(receipt_path),
            "command_count": len(receipt.commands),
            "metadata": _json_value(receipt.metadata),
        },
    )


def _render_stage_input(
    text: str,
    request: StageRequest,
) -> tuple[str, tuple[Path, ...]]:
    referenced: list[Path] = []

    def replace(match: re.Match[str]) -> str:
        if match.group("binding_component") is not None:
            path = _binding_view(request).topology_artifact(
                str(match.group("binding_component"))
            )
        elif match.group("prepared") is not None:
            path = _binding_view(request).artifact(
                str(match.group("prepared_role"))
            )
        elif match.group("selection") is not None:
            return _amber_atom_mask(
                _binding_view(request),
                str(match.group("selection")),
            )
        else:
            path = _upstream_artifact(
                _upstream(request, str(match.group("stage"))),
                str(match.group("artifact")),
            )
        referenced.append(path)
        return '"' + str(Path(path).resolve()).replace('"', '\\"') + '"'

    rendered = _PLACEHOLDER.sub(replace, text)
    if "{{" in rendered or "}}" in rendered:
        raise BackendConfigurationError("QC input contains an invalid or unresolved placeholder")
    return rendered, tuple(dict.fromkeys(referenced))


def _artifact_descriptor(value: object, request: StageRequest) -> Path:
    descriptor = _mapping(value, "artifact descriptor")
    expected = {"stage", "artifact"}
    if set(descriptor) != expected:
        raise BackendConfigurationError(
            "artifact descriptor requires exactly 'stage' and 'artifact'"
        )
    stage = _string(descriptor["stage"], "artifact descriptor stage")
    artifact = _string(
        descriptor["artifact"],
        "artifact descriptor artifact",
    )
    return _upstream_artifact(_upstream(request, stage), artifact)


def _require_trajectory_geometry(trajectory: StageInput) -> None:
    metadata = trajectory.payload.get("backend_metadata", {})
    if not isinstance(metadata, Mapping) or metadata.get("contract") != "mmpbsa.trajectory.v3":
        return  # The legacy @1 adapter has a separate GROMACS PBC contract.
    geometry = metadata.get("geometry")
    if (metadata.get("preprocessing_revision") != "whole_molecules_v1"
            or not isinstance(geometry, Mapping)
            or geometry.get("revision") != "whole_molecules_v1"
            or geometry.get("status") != "passed"):
        raise BackendConfigurationError(
            "trajectory lacks verified whole-molecule geometry; regenerate trajectory "
            "and QC in new attempts before scoring"
        )
    artifacts = geometry.get("artifacts", {})
    if not isinstance(artifacts, Mapping) or any(
        not isinstance(artifacts.get(role), Mapping)
        or artifacts[role].get("status") != "passed"
        for role in ("prepared_trajectory", "prepared_reference_trajectory")
    ):
        raise BackendConfigurationError("trajectory geometry audit is incomplete")


def _qc_reference_audit(
    science: Mapping[str, object], request: StageRequest
) -> Mapping[str, object]:
    artifact = science.get("reference_artifact")
    if artifact is None:
        return {}
    artifact = _string(artifact, "trajectory_qc.reference_artifact")
    frame = _integer(
        science.get("reference_frame"),
        "trajectory_qc.reference_frame",
        minimum=1,
    )
    semantics = _string(
        science.get("reference_semantics"),
        "trajectory_qc.reference_semantics",
    )
    input_text = _required_string(science, "input_text", "trajectory_qc")
    reference_line = f"reference {{{{upstream:trajectory:{artifact}}}}} {frame}"
    if reference_line not in input_text.splitlines():
        raise BackendConfigurationError(
            "trajectory_qc reference metadata disagrees with the input text"
        )
    audit: dict[str, object] = {
        "artifact": artifact,
        "frame": frame,
        "semantics": semantics,
    }
    if semantics == "post_npt_production_start":
        if artifact != "prepared_reference_trajectory" or frame != 1:
            raise BackendConfigurationError(
                "post-NPT production start must use prepared_reference_trajectory "
                "frame 1"
            )
        trajectory = _upstream(request, "trajectory")
        backend_metadata = _mapping(
            trajectory.payload.get("backend_metadata"),
            "trajectory_qc.trajectory.backend_metadata",
        )
        provenance = _mapping(
            backend_metadata.get("reference"),
            "trajectory_qc.trajectory.reference",
        )
        preprocessing = "autoimage_ligand_anchor_receptor_fixed_then_receptor_fit"
        if backend_metadata.get("preprocessing_revision") == "whole_molecules_v1":
            _require_trajectory_geometry(trajectory)
            preprocessing = "whole_molecules_then_autoimage_ligand_anchor_receptor_fixed"
        expected_provenance = {
            "artifact": artifact,
            "frame": frame,
            "semantics": semantics,
            "source_artifact": "selected_trajectory",
            "preprocessing": preprocessing,
        }
        for key, expected in expected_provenance.items():
            if provenance.get(key) != expected:
                raise BackendConfigurationError(
                    f"trajectory_qc reference provenance {key} disagrees with "
                    "the verified trajectory output"
                )
        validation = _mapping(
            backend_metadata.get("validation"),
            "trajectory_qc.trajectory.validation",
        )
        reference_validation = _mapping(
            validation.get(artifact),
            "trajectory_qc.trajectory.prepared_reference_trajectory",
        )
        source_validation = _mapping(
            validation.get("selected_trajectory"),
            "trajectory_qc.trajectory.selected_trajectory",
        )
        t0_time = _finite_float(
            reference_validation.get("first_time_ps"),
            "trajectory_qc.trajectory.reference.first_time_ps",
        )
        last_time = _finite_float(
            reference_validation.get("last_time_ps"),
            "trajectory_qc.trajectory.reference.last_time_ps",
        )
        source_t0_time = _finite_float(
            source_validation.get("first_time_ps"),
            "trajectory_qc.trajectory.source.first_time_ps",
        )
        if (
            _integer(
                reference_validation.get("frame_count"),
                "trajectory_qc.trajectory.reference.frame_count",
                minimum=1,
            )
            != 1
            or _integer(
                reference_validation.get("atom_count"),
                "trajectory_qc.trajectory.reference.atom_count",
                minimum=1,
            )
            != _integer(
                source_validation.get("atom_count"),
                "trajectory_qc.trajectory.source.atom_count",
                minimum=1,
            )
            or not math.isclose(t0_time, 0.0, abs_tol=1.0e-3)
            or not math.isclose(last_time, t0_time, abs_tol=1.0e-3)
            or not math.isclose(source_t0_time, t0_time, abs_tol=1.0e-3)
        ):
            raise BackendConfigurationError(
                "trajectory_qc t0 reference must be one frame at production t=0"
            )
        audit.update(
            {
                "source_artifact": provenance["source_artifact"],
                "preprocessing": provenance["preprocessing"],
                "time_ps": t0_time,
            }
        )
    return audit


def _evaluate_qc(
    parsed: Mapping[str, object],
    raw_checks: object,
) -> tuple[QcVerdict, Mapping[str, float], tuple[Mapping[str, object], ...]]:
    if not _is_sequence(raw_checks):
        raise BackendConfigurationError("trajectory_qc.checks must be a sequence")
    metrics: dict[str, float] = {}
    results: list[Mapping[str, object]] = []
    overall = QcVerdict.PASS
    seen: set[str] = set()
    for index, raw_check in enumerate(raw_checks):
        check = _mapping(raw_check, f"trajectory_qc.checks[{index}]")
        allowed = {
            "id",
            "table",
            "column",
            "reducer",
            "warn_above",
            "block_above",
            "warn_below",
            "block_below",
        }
        missing = {"id", "table", "column", "reducer"}.difference(check)
        unknown = set(check).difference(allowed)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise BackendConfigurationError(
                f"invalid trajectory_qc.checks[{index}]: " + "; ".join(details)
            )
        check_id = _string(check.get("id"), f"trajectory_qc.checks[{index}].id")
        if check_id in seen:
            raise BackendConfigurationError(f"duplicate QC check id: {check_id}")
        seen.add(check_id)
        table_name = _string(
            check.get("table"),
            f"trajectory_qc.checks[{index}].table",
        )
        column_name = _string(
            check.get("column"),
            f"trajectory_qc.checks[{index}].column",
        )
        reducer = _string(
            check.get("reducer"),
            f"trajectory_qc.checks[{index}].reducer",
        )
        if reducer not in {"mean", "max", "min", "last", "p95"}:
            raise BackendConfigurationError(f"QC reducer is unsupported: {reducer!r}")
        try:
            table = _mapping(parsed[table_name], f"QC table {table_name}")
        except KeyError as exc:
            raise BackendConfigurationError(
                f"QC check {check_id!r} names unknown table {table_name!r}"
            ) from exc
        columns_raw = table.get("columns")
        rows_raw = table.get("rows")
        if not _is_sequence(columns_raw) or not _is_sequence(rows_raw):
            raise BackendOutputError(f"QC table {table_name!r} has an invalid parsed shape")
        columns = tuple(str(item) for item in columns_raw)
        try:
            column_index = columns.index(column_name)
        except ValueError as exc:
            raise BackendConfigurationError(
                f"QC check {check_id!r} names unknown column {column_name!r}"
            ) from exc
        values_list: list[float] = []
        for row_index, row in enumerate(rows_raw):
            if not _is_sequence(row) or len(row) != len(columns):
                raise BackendOutputError(f"QC table {table_name!r} row {row_index} is invalid")
            values_list.append(float(row[column_index]))
        values = tuple(values_list)
        if not values or not all(math.isfinite(value) for value in values):
            raise BackendOutputError(f"QC check {check_id!r} has no finite values")
        if reducer == "mean":
            value = sum(values) / len(values)
        elif reducer == "max":
            value = max(values)
        elif reducer == "min":
            value = min(values)
        elif reducer == "p95":
            value = _linear_percentile(values, 0.95)
        else:
            value = values[-1]
        thresholds = {
            key: (
                None
                if check.get(key) is None
                else _finite_float(check[key], f"trajectory_qc.{check_id}.{key}")
            )
            for key in (
                "warn_above",
                "block_above",
                "warn_below",
                "block_below",
            )
        }
        _validate_threshold_order(check_id, thresholds)
        verdict = QcVerdict.PASS
        if (thresholds["block_above"] is not None and value > thresholds["block_above"]) or (
            thresholds["block_below"] is not None and value < thresholds["block_below"]
        ):
            verdict = QcVerdict.BLOCK
        elif (thresholds["warn_above"] is not None and value > thresholds["warn_above"]) or (
            thresholds["warn_below"] is not None and value < thresholds["warn_below"]
        ):
            verdict = QcVerdict.WARN
        if verdict is QcVerdict.BLOCK:
            overall = QcVerdict.BLOCK
        elif verdict is QcVerdict.WARN and overall is QcVerdict.PASS:
            overall = QcVerdict.WARN
        metrics[check_id] = value
        results.append(
            {
                "id": check_id,
                "table": table_name,
                "column": column_name,
                "reducer": reducer,
                "value": value,
                "verdict": verdict.value,
                **thresholds,
            }
        )
    return overall, metrics, tuple(results)


def _linear_percentile(values: Sequence[float], fraction: float) -> float:
    """Match the v1 linearly interpolated percentile definition."""

    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise ValueError("cannot calculate a percentile for an empty series")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be between zero and one")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _qc_issue_message(result: Mapping[str, object]) -> str:
    thresholds = ", ".join(
        f"{key}={result[key]}"
        for key in (
            "warn_above",
            "block_above",
            "warn_below",
            "block_below",
        )
        if result.get(key) is not None
    )
    return (
        f"{result['table']}.{result['column']} {result['reducer']}="
        f"{result['value']} triggered {result['verdict']}"
        + (f" ({thresholds})" if thresholds else "")
    )


def _validate_threshold_order(
    check_id: str,
    thresholds: Mapping[str, float | None],
) -> None:
    warn_above = thresholds["warn_above"]
    block_above = thresholds["block_above"]
    warn_below = thresholds["warn_below"]
    block_below = thresholds["block_below"]
    if warn_above is not None and block_above is not None and warn_above > block_above:
        raise BackendConfigurationError(f"QC check {check_id!r} warn_above exceeds block_above")
    if warn_below is not None and block_below is not None and warn_below < block_below:
        raise BackendConfigurationError(f"QC check {check_id!r} warn_below is below block_below")


def _write_json(path: Path, value: object) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_value(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if _is_sequence(value):
        return [_json_value(item) for item in value]
    return repr(value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise BackendConfigurationError(f"{label} must be a mapping with string keys")
    return value


def _string_mapping(value: object, label: str) -> dict[str, str]:
    mapping = _mapping(value, label)
    if any(not isinstance(item, str) for item in mapping.values()):
        raise BackendConfigurationError(f"{label} values must be strings")
    result = {str(key): item for key, item in mapping.items()}
    if any(not key or not item or "\x00" in key + item for key, item in result.items()):
        raise BackendConfigurationError(f"{label} contains an invalid entry")
    return result


def _required_string(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> str:
    if key not in mapping:
        raise BackendConfigurationError(f"{label} requires {key!r}")
    return _string(mapping[key], f"{label}.{key}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise BackendConfigurationError(f"{label} must be a non-empty string")
    return value


def _identifier(value: object, label: str) -> str:
    result = _string(value, label)
    if not _IDENTIFIER.fullmatch(result):
        raise BackendConfigurationError(
            f"{label} must be an identifier containing only letters, numbers, '.', '_' or '-'"
        )
    return result


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise BackendConfigurationError(f"{label} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value):
        result = int(value)
    else:
        raise BackendConfigurationError(f"{label} must be an integer")
    if result < minimum:
        raise BackendConfigurationError(f"{label} must be >= {minimum}")
    return result


def _optional_integer(
    value: object,
    label: str,
    *,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    return _integer(value, label, minimum=minimum)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise BackendConfigurationError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BackendConfigurationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise BackendConfigurationError(f"{label} must be finite")
    return result


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not _is_sequence(value):
        raise BackendConfigurationError(f"{label} must be a sequence")
    return tuple(_string(item, f"{label} item") for item in value)


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not _is_sequence(value):
        raise BackendConfigurationError(f"{label} must be a sequence")
    return tuple(_integer(item, f"{label} item", minimum=1) for item in value)


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )

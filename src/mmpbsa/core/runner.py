"""Composition root for replica-addressable execution and explicit finalizers."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from time import monotonic

from mmpbsa.utils import (
    FileLock,
    atomic_directory,
    atomic_write_json,
    materialize_immutable_file,
    resolve_within,
    sha256_digest,
    sha256_file,
    utc_now,
)
from mmpbsa.utils.cpu import thread_cpu_affinity

from .analysis import build_project_analysis, write_project_analysis
from .config import WORKFLOW_SCHEMA, RuntimeDefinition, WorkflowDefinition
from .errors import (
    ActiveAttemptError,
    ArtifactIntegrityError,
    ConfigurationError,
    ContractError,
    ResumeError,
    RetryError,
    TaskExecutionError,
)
from .frame_identity import require_gb_pb_frame_parity
from .models import (
    AttemptSummary,
    BackendCompatibility,
    JobKey,
    JobResult,
    ResourceAllocation,
    ResourceRequest,
    StageInput,
    StageRequest,
    StageResult,
    StageService,
    TaskKey,
)
from .prepared import (
    PreparedBindingView,
    PreparedCase,
    PreparedMdView,
    PreparedTrajectoryView,
    load_prepared_case,
)
from .prepared_projection import (
    prepared_case_projection,
    prepared_request_view,
)
from .providers import ProviderRegistry, discover_providers
from .publication import audit_stable_report_staging
from .resources import ResourcePool as _ResourcePool
from .state import (
    REPLICA_STAGES,
    AttemptState,
    QcVerdict,
    StageId,
    WorkflowOutcome,
)
from .workspace import ExecutionAction, Workspace

_CORE_VERSION = "0.7.0"
_STAGE_CAPABILITY = {
    StageId.ADMIT: "admission",
    StageId.MD: "md",
    StageId.TRAJECTORY: "trajectory",
    StageId.SCORING: "scoring",
    StageId.TRAJECTORY_QC: "trajectory_qc",
    StageId.ASSETS: "report.assets",
    StageId.REPORT: "report",
}


def effective_stage_runtime(
    runtime_definition: RuntimeDefinition,
    capability: str,
    science: Mapping[str, object],
    *,
    gpu_id: str | None = None,
) -> dict[str, object]:
    """Attach core-owned stage-contract context to a runtime projection."""

    runtime = runtime_definition.stage_runtime(capability, gpu_id=gpu_id)
    if capability == "md" and isinstance(science.get("schema_version"), str):
        runtime["_stage_contract"] = science["schema_version"]
        runtime["_md_engine_profile"] = science.get("md_engine_profile")
    elif capability == "trajectory" and isinstance(
        science.get("adapter_contract"), str
    ):
        runtime["_stage_contract"] = science["adapter_contract"]
    return runtime


@dataclass(frozen=True, slots=True)
class BatchFailure:
    job: JobKey
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class BatchResult:
    results: Mapping[JobKey, JobResult]
    failures: Mapping[JobKey, BatchFailure]

    @property
    def outcome(self) -> WorkflowOutcome:
        if self.failures and self.results:
            return WorkflowOutcome.PARTIAL
        if self.failures:
            return WorkflowOutcome.EXECUTION_FAILED
        if any(
            attempt.task.stage is StageId.TRAJECTORY_QC
            and attempt.qc_verdict is QcVerdict.BLOCK
            for result in self.results.values()
            for attempt in result.attempts
        ):
            return WorkflowOutcome.SCIENTIFICALLY_BLOCKED
        if any(
            attempt.task.stage is StageId.TRAJECTORY_QC
            and attempt.qc_verdict is QcVerdict.WARN
            for result in self.results.values()
            for attempt in result.attempts
        ):
            return WorkflowOutcome.COMPLETE_WITH_WARNINGS
        if any(result.model_failures for result in self.results.values()):
            return WorkflowOutcome.COMPLETE_WITH_WARNINGS
        if any(not result.completed for result in self.results.values()):
            return WorkflowOutcome.PARTIAL
        return WorkflowOutcome.COMPLETE

    @property
    def completed(self) -> bool:
        return not self.failures and all(
            result.completed for result in self.results.values()
        )


@dataclass(frozen=True, slots=True)
class ProjectResult:
    analysis: AttemptSummary
    assets: AttemptSummary | None
    report: AttemptSummary
    publication: Path


class WorkflowRunner:
    """Execute the fixed stage graph against one canonical workspace."""

    def __init__(
        self,
        workflow: WorkflowDefinition,
        runtime: RuntimeDefinition,
        workspace: Path,
        *,
        providers: ProviderRegistry | None = None,
    ) -> None:
        if workflow.schema_version != WORKFLOW_SCHEMA:
            raise ConfigurationError(
                f"{workflow.schema_version} is read-only legacy configuration; "
                f"execution requires {WORKFLOW_SCHEMA}"
            )
        if workflow.resolved_protocol is None:
            raise ConfigurationError(
                f"{WORKFLOW_SCHEMA} execution requires a resolved package protocol"
            )
        self.workflow = workflow
        self.runtime = runtime
        self.providers = providers or discover_providers()
        self.workspace = Workspace(
            workspace,
            project_id=workflow.project_id,
            protocol_id=workflow.protocol_id,
        )
        self._resource_pool = None
        self._resource_pool_lock = Lock()

    @property
    def _resources(self):
        """Delay machine resource discovery until actual stage execution."""
        with self._resource_pool_lock:
            if self._resource_pool is None:
                self._resource_pool = self._create_resources()
            return self._resource_pool

    def _create_resources(self):
        runtime = self.runtime
        coordination = getattr(runtime.scheduler, "resource_coordination", None)
        if coordination is not None and coordination.mode == "host":
            from .host_resources import HostResourcePool

            return HostResourcePool(
                registry_path=coordination.registry_path,
                project_id=self.workflow.project_id,
                cpu_slots=runtime.scheduler.cpu_slots,
                gpu_ids=runtime.scheduler.gpu_ids,
            )
        return _ResourcePool(
            cpu_slots=runtime.scheduler.cpu_slots,
            gpu_ids=runtime.scheduler.gpu_ids,
            cpu_affinity=runtime.scheduler.cpu_affinity,
        )

    @staticmethod
    def _core_compatibility(stage: StageId) -> BackendCompatibility:
        # Observed GB/PB frame-identity parity is mandatory in finalized cases;
        # v2 finalizer attempts predate that fail-closed wire contract.
        contract_version = "v3" if stage is StageId.FINALIZE_CASE else "v2"
        return BackendCompatibility(
            provider_id="core",
            adapter_contract=f"stage.{stage.value}.{contract_version}",
            engine="mmpbsa-core",
            engine_major="0.5",
            engine_version=_CORE_VERSION,
        )

    def _prepared(self, case_id: str) -> PreparedCase:
        case = self.workflow.case(case_id)
        return load_prepared_case(
            case.prepared_case, expected_case_id=case.case_id
        )

    @staticmethod
    def _snapshot_prepared_view(
        prepared_case: PreparedCase | None,
        view: PreparedMdView | PreparedTrajectoryView | PreparedBindingView | None,
        attempt_dir: Path,
        *,
        resume: bool,
    ) -> PreparedMdView | PreparedTrajectoryView | PreparedBindingView | None:
        """Give one attempt independent, verified copies of its declared inputs."""

        if prepared_case is None or view is None:
            return view
        refs = {
            item.role: item
            for item in (
                *prepared_case.manifest.md.artifacts,
                *prepared_case.manifest.scoring.artifacts,
            )
        }
        input_root = attempt_dir / "prepared_inputs"
        if input_root.is_symlink():
            raise ArtifactIntegrityError("attempt input directory cannot be a symlink")
        if resume and not input_root.is_dir():
            raise ArtifactIntegrityError("resumed attempt is missing its prepared input snapshot")
        selected: dict[str, Path] = {}
        for role, source in sorted(view.resolved_artifacts.items()):
            ref = refs[role]
            if source.stat().st_size != ref.size or sha256_file(source) != ref.sha256:
                raise ArtifactIntegrityError(f"PreparedCase source changed for {role}")
            parent = input_root
            for component in ref.path.parts[:-1]:
                parent = parent / component
                if parent.is_symlink():
                    raise ArtifactIntegrityError(
                        f"attempt input snapshot contains a symlinked parent for {role}"
                    )
            target = resolve_within(input_root, ref.path)
            if resume:
                if target.is_symlink() or not target.is_file():
                    raise ArtifactIntegrityError(f"resumed attempt input is missing: {role}")
            else:
                materialize_immutable_file(source, target, allow_hardlink=False)
            if target.stat().st_size != ref.size or sha256_file(target) != ref.sha256:
                raise ArtifactIntegrityError(f"attempt input snapshot differs for {role}")
            if source.stat().st_size != ref.size or sha256_file(source) != ref.sha256:
                raise ArtifactIntegrityError(
                    f"PreparedCase source changed while snapshotting {role}"
                )
            selected[role] = target
        return replace(view, resolved_artifacts=selected)

    def _science(
        self,
        task: TaskKey,
        prepared_case: PreparedCase | None,
    ) -> dict[str, object]:
        from .execution_preflight import task_science

        science = task_science(self.workflow, task, prepared_case)
        if task.stage is StageId.REPORT:
            science["_profile"] = {
                "profile_id": self.workflow.profile.profile_id,
                "profile_version": self.workflow.profile.profile_version,
                "extensions": dict(self.workflow.profile.extensions),
            }
            science["_protocol_settings"] = {
                "temperature_k": self.workflow.resolved_protocol.settings[
                    "temperature_k"
                ],
            }
        if task.stage is StageId.FINALIZE_CASE:
            assert task.case_id is not None
            case = self.workflow.case(task.case_id)
            science["_case"] = {
                "case_id": case.case_id,
                "label": case.label,
                "reference": dict(case.reference),
                "metadata": dict(case.metadata),
            }
        if task.stage is StageId.FINALIZE_CASE:
            science["_expected_replicas"] = [
                {
                    "replica_id": replica.replica_id,
                    "replica_index": replica.replica_index,
                    "seed": replica.seed,
                }
                for replica in self.workflow.replicas
            ]
            science["_scoring"] = {
                "completion_policy": self.workflow.completion_policy,
                "model_ids": list(self.workflow.scoring_model_ids),
            }
        if task.stage in {
            StageId.ANALYSIS,
            StageId.ASSETS,
            StageId.REPORT,
        }:
            science["_project"] = {
                "project_id": self.workflow.project_id,
                "protocol_id": self.workflow.protocol_id,
                "cases": [
                    {
                        "case_id": case.case_id,
                        "label": case.label,
                        "reference": dict(case.reference),
                        "metadata": dict(case.metadata),
                    }
                    for case in self.workflow.cases
                ],
                "replicas": [
                    {
                        "replica_id": replica.replica_id,
                        "replica_index": replica.replica_index,
                        "seed": replica.seed,
                    }
                    for replica in self.workflow.replicas
                ],
            }
        return science

    def _science_hash(
        self,
        task: TaskKey,
        science: Mapping[str, object],
        prepared_case: PreparedCase | None,
        upstream: Mapping[str, StageInput],
    ) -> str:
        return sha256_digest(
            self._science_identity(
                task,
                science,
                prepared_case,
                upstream,
            )
        )

    def _science_identity(
        self,
        task: TaskKey,
        science: Mapping[str, object],
        prepared_case: PreparedCase | None,
        upstream: Mapping[str, StageInput],
    ) -> dict[str, object]:
        science_identity = dict(science)
        profile_raw = science_identity.get("_profile")
        if isinstance(profile_raw, Mapping):
            # The effective normalized values and typed extensions are
            # scientific inputs. Human-facing factory identity is provenance,
            # not a reason to invalidate an otherwise identical MD attempt.
            science_identity["_profile"] = {
                key: value
                for key, value in profile_raw.items()
                if key not in {"profile_id", "profile_version"}
            }
        upstream_identity: dict[str, object] = {}
        for name, item in sorted(upstream.items()):
            upstream_identity[name] = {
                "task": item.task.label,
                "attempt_id": item.attempt_id,
                "science_hash": item.science_hash,
                "payload_sha256": sha256_digest(item.payload),
                "artifacts": {
                    role: {
                        "sha256": sha256_file(path),
                        "size": path.stat().st_size,
                    }
                    for role, path in sorted(item.artifacts.items())
                },
                "qc_verdict": (
                    None
                    if item.qc_verdict is None
                    else item.qc_verdict.value
                ),
            }
        return {
            "contract": "mmpbsa.science-hash.v2",
            "project_id": self.workflow.project_id,
            "protocol_id": self.workflow.protocol_id,
            "task": task.label,
            "science": science_identity,
            "prepared_case": prepared_case_projection(
                task,
                science_identity,
                prepared_case,
            ),
            "upstream": upstream_identity,
        }

    def _provider_service(
        self, stage: StageId
    ) -> tuple[StageService | object | None, str | None]:
        capability = _STAGE_CAPABILITY.get(stage)
        if capability is None:
            return None, None
        provider_id = self.runtime.provider_for(capability)
        if provider_id is None:
            if stage is StageId.ADMIT:
                return None, capability
            if stage is StageId.ASSETS:
                return None, capability
            raise ContractError(
                f"runtime does not select a provider for {capability!r}"
            )
        service = self.providers.service(provider_id, capability)
        from .execution_preflight import require_host_cooperation

        require_host_cooperation(service, self.runtime, capability)
        return service, capability

    @staticmethod
    def _require_service(value: object, capability: str) -> StageService:
        for method in ("resource_request", "compatibility", "execute", "resume"):
            if not callable(getattr(value, method, None)):
                raise ContractError(
                    f"{capability!r} service is missing {method}()"
                )
        return value  # type: ignore[return-value]

    def _stage_execution_contract(
        self,
        stage: StageId,
        science: Mapping[str, object],
    ) -> tuple[
        StageService | None,
        str | None,
        Mapping[str, object],
        BackendCompatibility,
    ]:
        """Resolve the same backend identity for execution and availability."""

        service_object, capability = self._provider_service(stage)
        if service_object is None:
            return None, capability, {}, self._core_compatibility(stage)
        assert capability is not None
        service = self._require_service(service_object, capability)
        runtime = self._effective_stage_runtime(capability, science)
        compatibility = service.compatibility(runtime)
        if not isinstance(compatibility, BackendCompatibility):
            raise ContractError(
                f"{capability!r} compatibility() returned the wrong type"
            )
        return service, capability, runtime, compatibility

    def _runtime_provenance(
        self, runtime: Mapping[str, object]
    ) -> dict[str, object]:
        environment = runtime.get("environment", {})
        environment_names = (
            sorted(str(name) for name in environment)
            if isinstance(environment, Mapping)
            else []
        )
        executables: set[str] = set()

        def collect(value: object) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if key == "executable" and isinstance(item, str):
                        executables.add(item)
                    elif key != "environment":
                        collect(item)
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for item in value:
                    collect(item)

        collect(runtime)
        source = self.runtime.source_path
        return {
            "effective_runtime_sha256": sha256_digest(runtime),
            "runtime_document": {
                "path": str(source),
                "sha256": sha256_file(source),
            },
            "gpu_id": runtime.get("gpu_id"),
            "threads_per_job": runtime.get("threads_per_job"),
            "probe_cwd": runtime.get("probe_cwd"),
            "environment_variables": environment_names,
            "executables": sorted(executables),
            "configuration_keys": sorted(str(key) for key in runtime),
        }

    def _effective_stage_runtime(
        self,
        capability: str,
        science: Mapping[str, object],
        *,
        gpu_id: str | None = None,
    ) -> dict[str, object]:
        """Add core-owned contract context to each scheduler runtime view."""

        return effective_stage_runtime(
            self.runtime,
            capability,
            science,
            gpu_id=gpu_id,
        )

    def _execute_task(
        self,
        task: TaskKey,
        *,
        action: ExecutionAction,
        prepared_case: PreparedCase | None,
        upstream: Mapping[str, StageInput],
        new_on_conflict: bool = False,
        allocation: ResourceAllocation | None = None,
        queued_at: str | None = None,
        queued_monotonic: float | None = None,
        resume_failed_md: bool = False,
    ) -> AttemptSummary:
        science = self._science(task, prepared_case)
        service, capability, runtime, compatibility = self._stage_execution_contract(
            task.stage, science
        )
        science_hash = self._science_hash(task, science, prepared_case, upstream)
        recovery_source = None
        if resume_failed_md:
            previous = self.workspace.latest(task)
            if (task.stage is not StageId.MD or action != "retry" or previous is None
                    or previous.state is not AttemptState.FAILED
                    or previous.science_hash != science_hash
                    or previous.compatibility_key != compatibility.key
                    or not callable(getattr(service, "recover_failed_md", None))):
                raise ContractError("failed MD checkpoint recovery requires matching "
                                    "science/backend and explicit retry")
            recovery_source = self.workspace.task_dir(task) / "attempts" / previous.attempt_id
            if self.workspace._processes_within(recovery_source):
                raise ActiveAttemptError("failed MD still has live subprocesses; "
                                         "checkpoint recovery withheld")
        try:
            reservation = self.workspace.reserve(
                task,
                action=action,
                science_hash=science_hash,
                compatibility=compatibility,
            )
        except (ActiveAttemptError, ArtifactIntegrityError, RetryError):
            if not new_on_conflict or action != "run":
                raise
            reservation = self.workspace.reserve(
                task,
                action="new",
                science_hash=science_hash,
                compatibility=compatibility,
            )
        if reservation.reused:
            reused_view = prepared_request_view(task, prepared_case, science)
            self._snapshot_prepared_view(
                prepared_case,
                reused_view,
                reservation.attempt_dir,
                resume=True,
            )
            return reservation.summary
        assert reservation.invocation_dir is not None
        try:
            view = prepared_request_view(task, prepared_case, science)
            view = self._snapshot_prepared_view(
                prepared_case,
                view,
                reservation.attempt_dir,
                resume=reservation.resume,
            )
            request = StageRequest(
                task=task,
                science=science,
                runtime=runtime,
                prepared_case=view,
                upstream=upstream,
                attempt_dir=reservation.attempt_dir,
                invocation_dir=reservation.invocation_dir,
                resume=reservation.resume,
                checkpoint=(reservation.checkpoint if reservation.resume else None),
                compatibility=compatibility,
            )
            resource_request = (
                ResourceRequest() if service is None else service.resource_request(request)
            )
            if not isinstance(resource_request, ResourceRequest):
                raise ContractError(f"{task.label} resource_request() returned the wrong type")
            wait_started = queued_monotonic if queued_monotonic is not None else monotonic()
            enqueued_at = queued_at or utc_now()
            lease = (
                nullcontext(allocation)
                if allocation is not None
                else self._resources.lease(resource_request)
            )
            with lease as acquired:
                assert acquired is not None
                if (
                    acquired.cpu_slots != resource_request.cpu_slots
                    or len(acquired.gpu_ids) != resource_request.gpu_slots
                ):
                    raise ContractError("preallocated resources differ from stage resource request")
                gpu_ids = acquired.gpu_ids
                request = replace(request, allocation=acquired)
                if len(gpu_ids) > 1:
                    raise ContractError("current local runtime supports at most one GPU per task")
                effective_runtime = runtime
                if service is not None and capability is not None and gpu_ids:
                    effective_runtime = self._effective_stage_runtime(
                        capability,
                        science,
                        gpu_id=gpu_ids[0],
                    )
                    request = replace(request, runtime=effective_runtime)
                from .md_allocation_guard import validate_resume_allocation

                validate_resume_allocation(service, request)
                physical_gpu_ids = getattr(acquired, "physical_gpu_ids", ())
                provenance = self._runtime_provenance(effective_runtime)
                provenance["resource_request"] = {
                    "cpu_slots": resource_request.cpu_slots,
                    "gpu_slots": resource_request.gpu_slots,
                }
                provenance["allocation"] = {
                    "cpu_ids": list(acquired.cpu_ids),
                    "gpu_ids": list(acquired.gpu_ids),
                    "cpu_slots": acquired.cpu_slots,
                    "physical_gpu_ids": list(physical_gpu_ids),
                    "lease_token": getattr(acquired, "lease_token", None),
                }
                provenance["timing"] = {
                    "queued_at": enqueued_at,
                    "execution_started_at": utc_now(),
                    "queue_wait_seconds": max(0.0, monotonic() - wait_started),
                }
                attach_attempt = getattr(self._resources, "attach_attempt", None)
                if attach_attempt is not None:
                    attach_attempt(acquired, task, reservation.summary.attempt_id,
                                   str(request.invocation_dir))
                self.workspace.record_invocation_runtime(reservation, provenance)
                execution_started = monotonic()
                try:
                    observer = getattr(self._resources, "observe", None)
                    observation = (observer(acquired, str(request.invocation_dir))
                                   if observer else nullcontext())
                    with thread_cpu_affinity(acquired.cpu_ids), observation:
                        if service is None:
                            result = self._execute_core(request)
                        elif recovery_source is not None:
                            recovery_service = self._require_service(
                                service, capability or task.stage.value)
                            result = recovery_service.recover_failed_md(request, recovery_source)
                        elif reservation.resume:
                            result = service.resume(request)
                        else:
                            result = service.execute(request)
                finally:
                    provenance["timing"].update(
                        {
                            "execution_finished_at": utc_now(),
                            "execution_seconds": monotonic() - execution_started,
                        }
                    )
                    self.workspace.record_invocation_runtime(reservation, provenance)
            if not isinstance(result, StageResult):
                raise ContractError(
                    f"{task.label} returned {type(result).__name__}, expected StageResult"
                )
            if task.stage is StageId.TRAJECTORY_QC and result.qc_verdict is None:
                raise ContractError("trajectory QC service did not return a verdict")
            return self.workspace.complete(reservation, result)
        except (KeyboardInterrupt, SystemExit) as exc:
            checkpoint = getattr(exc, "checkpoint", None)
            self.workspace.fail(
                reservation,
                exc,
                interrupted=True,
                checkpoint=(Path(checkpoint) if checkpoint is not None else None),
            )
            raise
        except BaseException as exc:
            checkpoint = getattr(exc, "checkpoint", None)
            self.workspace.fail(
                reservation,
                exc,
                checkpoint=(Path(checkpoint) if checkpoint is not None else None),
            )
            if isinstance(exc, (ContractError, TaskExecutionError)):
                raise
            raise TaskExecutionError(
                f"{task.label} failed in attempt {reservation.summary.attempt_id}: {exc}"
            ) from exc

    def _execute_core(self, request: StageRequest) -> StageResult:
        output = request.attempt_dir / "output"
        if request.task.stage is StageId.ADMIT:
            prepared = request.prepared_case
            if not isinstance(prepared, PreparedBindingView):
                raise ContractError(
                    "admission requires a PreparedBindingView"
                )
            receipt = {
                "schema_version": "mmpbsa.admission.v1",
                "case_id": prepared.case_id,
                "artifact_count": len(prepared.resolved_artifacts),
                "artifacts": {
                    role: str(path)
                    for role, path in sorted(
                        prepared.resolved_artifacts.items()
                    )
                },
                "verified_at": utc_now(),
            }
            artifact = atomic_write_json(output / "admission.json", receipt)
            return StageResult(
                artifacts={"admission": artifact},
                payload=receipt,
                invocation_provenance={
                    "implementation": "mmpbsa.core.integrity-admission"
                },
            )
        if request.task.stage is StageId.FINALIZE_CASE:
            record = self._case_record(request.task.case_id, request.upstream)
            artifact = atomic_write_json(output / "case.json", record)
            verdict = record["qc_verdict"]
            return StageResult(
                artifacts={"case_record": artifact},
                payload=record,
                qc_verdict=(
                    None if verdict is None else QcVerdict(str(verdict))
                ),
                invocation_provenance={
                    "implementation": "mmpbsa.core.case-finalizer"
                },
            )
        if request.task.stage is StageId.ANALYSIS:
            records = [
                item.payload
                for name, item in sorted(request.upstream.items())
                if name.startswith("case:")
            ]
            analysis = build_project_analysis(self.workflow, records)
            artifacts = write_project_analysis(output, analysis)
            return StageResult(
                artifacts=artifacts,
                payload=analysis,
                invocation_provenance={
                    "implementation": "mmpbsa.core.project-analysis"
                },
            )
        raise ContractError(
            f"stage has no core implementation: {request.task.stage.value}"
        )

    def _attempt_record(self, item: StageInput) -> dict[str, object]:
        payload = dict(item.payload)
        if item.task.stage is StageId.SCORING:
            per_frame_models: dict[str, object] = {}
            analysis_models = self.workflow.analysis.get("models", {})
            if isinstance(analysis_models, Mapping):
                for analysis_model_id, raw_analysis in analysis_models.items():
                    if not isinstance(raw_analysis, Mapping):
                        continue
                    source_model_id = str(
                        raw_analysis.get("source_model_id", analysis_model_id)
                    )
                    if source_model_id != item.task.model_id:
                        continue
                    per_frame = self._report_per_frame(
                        item,
                        analysis_model_id=str(analysis_model_id),
                        analysis=raw_analysis,
                    )
                    if per_frame is not None:
                        per_frame_models[str(analysis_model_id)] = per_frame
            if per_frame_models:
                payload["per_frame_models"] = per_frame_models
                direct = per_frame_models.get(str(item.task.model_id))
                if isinstance(direct, Mapping):
                    payload["per_frame"] = dict(direct)
        return {
            "task": {
                "stage": item.task.stage.value,
                "case_id": item.task.case_id,
                "replica_id": item.task.replica_id,
                "model_id": item.task.model_id,
            },
            "model_id": item.task.model_id,
            "attempt_id": item.attempt_id,
            "artifacts": {
                role: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
                for role, path in sorted(item.artifacts.items())
            },
            "payload": payload,
            "qc_verdict": (
                None if item.qc_verdict is None else item.qc_verdict.value
            ),
        }

    def _report_per_frame(
        self,
        item: StageInput,
        *,
        analysis_model_id: str,
        analysis: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        """Read only the selected delta series from a verified artifact."""

        evaluator = str(analysis.get("evaluator", "direct"))
        artifact_role = "dmm_output" if evaluator == "v1_dmm_v1" else "delta_output"
        source = item.artifacts.get(artifact_role)
        model_id = item.task.model_id
        if source is None or model_id is None:
            return None
        selector = analysis.get("score_selector")
        if not isinstance(selector, Mapping):
            raise ContractError(
                f"analysis model {analysis_model_id!r} has no score_selector"
            )
        selected_label = selector.get("label")
        if not isinstance(selected_label, str) or not selected_label.strip():
            raise ContractError(
                f"analysis model {analysis_model_id!r} has no score label"
            )

        def normalized(value: str) -> str:
            return "".join(
                character
                for character in value.casefold()
                if character.isalnum()
            )

        try:
            with source.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames
                if (
                    columns is None
                    or any(not column for column in columns)
                    or len(columns) != len(set(columns))
                ):
                    raise ContractError(
                        f"delta artifact has invalid columns: {source}"
                    )
                frame_matches = [
                    column
                    for column in columns
                    if normalized(column)
                    in {"frame", "frameindex", "framenumber"}
                ]
                value_matches = [
                    column
                    for column in columns
                    if normalized(column) == normalized(selected_label)
                ]
                if len(frame_matches) != 1 or len(value_matches) != 1:
                    raise ContractError(
                        "delta artifact must contain exactly one frame and "
                        f"{selected_label!r} column: {source}"
                    )
                frame_column = frame_matches[0]
                value_column = value_matches[0]
                rows: list[dict[str, float]] = []
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ContractError(
                            f"delta artifact has a malformed row: {source}"
                        )
                    frame = float(str(row[frame_column]))
                    value = float(str(row[value_column]))
                    if not math.isfinite(frame) or not math.isfinite(value):
                        raise ContractError(
                            f"delta artifact has a non-finite value: {source}"
                        )
                    rows.append(
                        {
                            frame_column: frame,
                            value_column: value,
                        }
                    )
        except (OSError, UnicodeError, ValueError) as exc:
            raise ContractError(
                f"cannot read delta artifact for {item.task.label}: {source}"
            ) from exc
        if not rows:
            raise ContractError(
                f"delta artifact contains no frames: {source}"
            )
        return {
            "columns": [frame_column, value_column],
            "rows": rows,
        }

    def _case_record(
        self,
        case_id: str | None,
        upstream: Mapping[str, StageInput],
    ) -> dict[str, object]:
        if case_id is None:
            raise ContractError("case finalization requires case_id")
        replicas: list[dict[str, object]] = []
        missing: list[dict[str, object]] = []
        verdicts: list[QcVerdict] = []
        model_warnings = False
        for replica in self.workflow.replicas:
            md_status = self._missing_stage_record(
                upstream.get(f"inventory:md:{replica.replica_id}")
            )
            trajectory_status = self._missing_stage_record(
                upstream.get(f"inventory:trajectory:{replica.replica_id}")
            )
            qc = upstream.get(f"trajectory_qc:{replica.replica_id}")
            scoring_models = {
                model_id: item
                for model_id in self.workflow.scoring_model_ids
                if (
                    item := upstream.get(
                        f"scoring:{replica.replica_id}:{model_id}"
                    )
                )
                is not None
            }
            model_statuses = {
                model_id: self._missing_stage_record(
                    upstream.get(
                        f"inventory:scoring:{replica.replica_id}:{model_id}"
                    )
                )
                for model_id in self.workflow.scoring_model_ids
            }
            model_result_statuses = {}
            for model_id, model_status in model_statuses.items():
                if qc is not None and qc.qc_verdict is QcVerdict.BLOCK:
                    result_status = "skipped_qc_block"
                elif model_id in scoring_models:
                    result_status = "completed"
                elif model_status.get("state") == "failed":
                    result_status = "execution_failed"
                elif model_status.get("state") == "interrupted":
                    result_status = "cancelled"
                else:
                    result_status = "not_started"
                model_result_statuses[model_id] = result_status
            if qc is None:
                qc_status = self._missing_stage_record(
                    upstream.get(
                        f"inventory:trajectory_qc:{replica.replica_id}"
                    )
                )
                missing.append(
                    {
                        "replica_id": replica.replica_id,
                        "reason": (
                            "no current completed trajectory_qc"
                            if qc_status.get("usable") is False
                            else "missing completed trajectory_qc"
                        ),
                        "stages": {
                            "md": md_status,
                            "trajectory": trajectory_status,
                            "trajectory_qc": qc_status,
                            "scoring_models": model_statuses,
                        },
                    }
                )
                continue
            verdict = qc.qc_verdict
            if verdict is None:
                raise ContractError(
                    f"completed trajectory QC attempt has no verdict: "
                    f"{case_id}/{replica.replica_id}"
                )
            verdicts.append(verdict)
            completed_models = (
                ()
                if verdict is QcVerdict.BLOCK
                else tuple(
                    model_id
                    for model_id in self.workflow.scoring_model_ids
                    if model_id in scoring_models
                )
            )
            require_gb_pb_frame_parity(
                self.workflow,
                case_id,
                replica.replica_id,
                {
                    model_id: scoring_models[model_id]
                    for model_id in completed_models
                    if model_id in {"gb", "pb"}
                },
            )
            scoring_acceptable = (
                bool(completed_models)
                if self.workflow.completion_policy == "any"
                else len(completed_models) == len(
                    self.workflow.scoring_model_ids
                )
            )
            if verdict is not QcVerdict.BLOCK and not scoring_acceptable:
                has_unusable_model = any(
                    status.get("usable") is False
                    for status in model_statuses.values()
                )
                missing.append(
                    {
                        "replica_id": replica.replica_id,
                        "reason": (
                            "no acceptable current scoring model"
                            if has_unusable_model
                            else "no acceptable completed scoring model"
                        ),
                        "stages": {
                            "md": md_status,
                            "trajectory": trajectory_status,
                            "trajectory_qc": self._attempt_record(qc),
                            "scoring_models": model_statuses,
                        },
                    }
                )
                continue
            if len(completed_models) < len(self.workflow.scoring_model_ids):
                model_warnings = model_warnings or verdict is not QcVerdict.BLOCK
            trajectory_receipt = self.workspace.latest(TaskKey(
                StageId.TRAJECTORY, case_id, replica.replica_id))
            trajectory_manifest = (trajectory_receipt.payload.get("frame_manifest")
                                   if trajectory_receipt is not None
                                   and trajectory_status.get("usable") is True
                                   and trajectory_receipt.attempt_id
                                   == trajectory_status.get("attempt_id")
                                   else None)
            replicas.append(
                {
                    "replica_id": replica.replica_id,
                    "replica_index": replica.replica_index,
                    "seed": replica.seed,
                    "scoring_models": {
                        model_id: self._attempt_record(scoring_models[model_id])
                        for model_id in completed_models
                    },
                    "completed_models": completed_models,
                    "model_statuses": model_statuses,
                    "model_result_statuses": model_result_statuses,
                    "trajectory_qc": self._attempt_record(qc),
                    "trajectory_frame_manifest": trajectory_manifest,
                    "qc_verdict": verdict.value,
                }
            )
        overall: QcVerdict | None = None
        if QcVerdict.BLOCK in verdicts:
            overall = QcVerdict.BLOCK
        elif QcVerdict.WARN in verdicts:
            overall = QcVerdict.WARN
        elif QcVerdict.PASS in verdicts:
            overall = QcVerdict.PASS
        if missing:
            outcome = WorkflowOutcome.PARTIAL
        elif overall is QcVerdict.BLOCK:
            outcome = WorkflowOutcome.SCIENTIFICALLY_BLOCKED
        elif overall is QcVerdict.WARN or model_warnings:
            outcome = WorkflowOutcome.COMPLETE_WITH_WARNINGS
        else:
            outcome = WorkflowOutcome.COMPLETE
        case = self.workflow.case(case_id)
        from .scientific_context import build_scientific_context

        scientific_context = build_scientific_context(
            self._prepared(case_id),
            frame_selection=self.workflow.stage_science(StageId.TRAJECTORY).get("frame_manifest"),
            replicas=replicas,
        )
        return {
            "schema_version": "mmpbsa.finalized-case.v2",
            "project_id": self.workflow.project_id,
            "protocol_id": self.workflow.protocol_id,
            "case_id": case_id,
            "label": case.label or case_id,
            "reference": dict(case.reference),
            "expected_replicas": [
                replica.replica_id for replica in self.workflow.replicas
            ],
            "expected_models": list(self.workflow.scoring_model_ids),
            "scoring_completion_policy": self.workflow.completion_policy,
            "complete_replicas": [
                str(item["replica_id"]) for item in replicas
            ],
            "missing_replicas": missing,
            "partial": bool(missing),
            "replicas": replicas,
            "qc_evaluated_replicas": len(verdicts),
            "qc_verdict": None if overall is None else overall.value,
            "outcome": outcome.value,
            "scientific_context": scientific_context,
        }

    @staticmethod
    def _missing_stage_record(
        value: StageInput | None,
    ) -> Mapping[str, object]:
        if value is None:
            return {"state": "absent", "attempt_id": None}
        result = {
            "state": value.payload.get("state", "unknown"),
            "artifact_state": value.payload.get(
                "artifact_state", "unknown"
            ),
            "attempt_id": value.attempt_id,
        }
        for key in (
            "usable",
            "reason",
            "identity",
            "observed_science_hash",
            "expected_science_hash",
            "observed_compatibility_key",
            "expected_compatibility_key",
            "backend_error",
        ):
            if key in value.payload:
                result[key] = value.payload[key]
        return result

    def _job_tasks(
        self,
        job: JobKey,
        through: StageId,
        *,
        model_ids: Sequence[str] | None = None,
    ) -> tuple[TaskKey, ...]:
        if through is StageId.ADMIT:
            return (TaskKey(StageId.ADMIT, case_id=job.case_id),)
        if through not in REPLICA_STAGES:
            raise ContractError(
                "job through-stage must be admit, md, trajectory, "
                "trajectory_qc or scoring"
            )
        tasks = [TaskKey(StageId.ADMIT, case_id=job.case_id)]
        for stage in REPLICA_STAGES:
            if stage is StageId.SCORING:
                selected = self._selected_models(model_ids)
                tasks.extend(
                    TaskKey(
                        stage,
                        case_id=job.case_id,
                        replica_id=job.replica_id,
                        model_id=model_id,
                    )
                    for model_id in selected
                )
            else:
                tasks.append(
                    TaskKey(
                        stage,
                        case_id=job.case_id,
                        replica_id=job.replica_id,
                    )
                )
            if stage is through:
                break
        return tuple(tasks)

    def _selected_models(
        self, model_ids: Sequence[str] | None
    ) -> tuple[str, ...]:
        selected = (
            tuple(dict.fromkeys(model_ids))
            if model_ids is not None
            else self.workflow.scoring_model_ids
        )
        if not selected:
            raise ContractError("at least one scoring model must be selected")
        unknown = set(selected).difference(self.workflow.scoring_model_ids)
        if unknown:
            raise ContractError(
                "unknown scoring models: " + ", ".join(sorted(unknown))
            )
        return selected

    def _actions_for_tasks(
        self,
        tasks: Sequence[TaskKey],
        action: ExecutionAction,
        *,
        strict: bool,
    ) -> tuple[ExecutionAction, ...]:
        actions: list[ExecutionAction] = ["run"] * len(tasks)
        if action == "run":
            return tuple(actions)
        if action == "new":
            terminal_stage = tasks[-1].stage
            for index, task in enumerate(tasks):
                if task.stage is terminal_stage:
                    actions[index] = "new"
            return tuple(actions)
        def desired_state(value: AttemptSummary | None) -> bool:
            if action == "resume":
                return value is not None and value.state.active
            return value is not None and value.state is AttemptState.FAILED
        selected = None
        for index, task in enumerate(tasks):
            if desired_state(self.workspace.latest(task)):
                selected = index
                break
        if selected is None:
            if strict:
                error = ResumeError if action == "resume" else RetryError
                raise error(
                    f"no task in the requested job can {action}"
                )
            return tuple(actions)
        actions[selected] = action
        return tuple(actions)

    @staticmethod
    def _job_upstream(
        stage: StageId, completed: Mapping[StageId, StageInput]
    ) -> dict[str, StageInput]:
        dependencies = {
            StageId.ADMIT: (),
            StageId.MD: (),
            StageId.TRAJECTORY: (StageId.MD,),
            StageId.TRAJECTORY_QC: (StageId.TRAJECTORY,),
            StageId.SCORING: (
                StageId.TRAJECTORY,
                StageId.TRAJECTORY_QC,
            ),
        }[stage]
        return {
            dependency.value: completed[dependency]
            for dependency in dependencies
        }

    def run_job(
        self,
        job: JobKey,
        *,
        through: StageId = StageId.SCORING,
        action: ExecutionAction = "run",
        model_ids: Sequence[str] | None = None,
        strict_action: bool = True,
    ) -> JobResult:
        self.workflow.case(job.case_id)
        self.workflow.replica(job.replica_id)
        from .execution_preflight import preflight_execution

        preflight_execution(self.workflow, self.runtime, self.providers, (job,),
                            through=through, model_ids=model_ids)
        if through is not StageId.SCORING and model_ids is not None:
            raise ContractError(
                "model_ids is only valid when through is scoring"
            )
        prepared = self._prepared(job.case_id)
        selected_models = (
            self._selected_models(model_ids)
            if through is StageId.SCORING
            else ()
        )
        if (
            through is StageId.SCORING
            and action in {"resume", "retry"}
            and len(selected_models) != 1
        ):
            raise ContractError(
                f"multi-model {action} requires exactly one explicit model"
            )
        tasks = self._job_tasks(
            job,
            through,
            model_ids=selected_models or None,
        )
        actions = self._actions_for_tasks(
            tasks, action, strict=strict_action
        )
        completed: dict[StageId, StageInput] = {}
        attempts: list[AttemptSummary] = []
        model_failures: dict[str, str] = {}
        model_statuses = {model: "not_started" for model in selected_models}
        task_actions = tuple(zip(tasks, actions, strict=True))
        prerequisite_actions = tuple(
            item for item in task_actions if item[0].stage is not StageId.SCORING
        )
        scoring_actions = tuple(
            item for item in task_actions if item[0].stage is StageId.SCORING
        )
        for task, stage_action in prerequisite_actions:
            # PreparedCase files remain external and read-only. Re-load at every
            # stage boundary so a long MD invocation cannot leave later stages
            # consuming inputs that changed after the job's initial admission.
            prepared = self._prepared(job.case_id)
            upstream = self._job_upstream(task.stage, completed)
            summary = self._execute_task(
                task,
                action=stage_action,
                prepared_case=prepared,
                upstream=upstream,
                new_on_conflict=(
                    action == "new" and stage_action == "run"
                ),
            )
            attempts.append(summary)
            completed[task.stage] = self.workspace.as_stage_input(summary)

        if scoring_actions:
            qc = completed[StageId.TRAJECTORY_QC]
            if qc.qc_verdict is QcVerdict.BLOCK:
                for task, _ in scoring_actions:
                    assert task.model_id is not None
                    model_failures[task.model_id] = (
                        "skipped: trajectory_qc verdict is BLOCK"
                    )
                    model_statuses[task.model_id] = "skipped_qc_block"
            else:
                scoring_upstream = self._job_upstream(
                    StageId.SCORING,
                    completed,
                )

                def execute_model(
                    task_action: tuple[TaskKey, ExecutionAction],
                ) -> AttemptSummary:
                    task, stage_action = task_action
                    return self._execute_task(
                        task,
                        action=stage_action,
                        prepared_case=self._prepared(job.case_id),
                        upstream=scoring_upstream,
                        new_on_conflict=(
                            action == "new" and stage_action == "run"
                        ),
                    )

                worker_count = min(
                    len(scoring_actions),
                    self.runtime.scheduler.max_workers,
                )
                scoring_results: dict[TaskKey, AttemptSummary] = {}
                scoring_errors: dict[TaskKey, Exception] = {}
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="mmpbsa-model",
                ) as executor:
                    futures = {
                        executor.submit(execute_model, item): item[0]
                        for item in scoring_actions
                    }
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            scoring_results[task] = future.result()
                        except Exception as exc:
                            scoring_errors[task] = exc
                # Preserve declared model order in JobResult even though the
                # independent engine invocations complete out of order.
                for task, _ in scoring_actions:
                    assert task.model_id is not None
                    if task in scoring_errors:
                        exc = scoring_errors[task]
                        model_failures[task.model_id] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        model_statuses[task.model_id] = "execution_failed"
                        failed = self.workspace.latest(task)
                        if failed is not None:
                            attempts.append(failed)
                    else:
                        attempts.append(scoring_results[task])
                        model_statuses[task.model_id] = "completed"
        if through is StageId.SCORING:
            completed_models = {
                item.task.model_id
                for item in attempts
                if (
                    item.task.stage is StageId.SCORING
                    and item.state is AttemptState.COMPLETED
                )
            }
            acceptable = (
                bool(completed_models)
                if self.workflow.completion_policy == "any"
                else set(selected_models).issubset(completed_models)
            )
            if (
                not acceptable
                and completed[StageId.TRAJECTORY_QC].qc_verdict
                is not QcVerdict.BLOCK
            ):
                raise TaskExecutionError(
                    "no acceptable scoring result: "
                    + "; ".join(
                        f"{model}={message}"
                        for model, message in sorted(model_failures.items())
                    )
                )
        return JobResult(
            job=job,
            through=through,
            attempts=tuple(attempts),
            completion_policy=(
                self.workflow.completion_policy
                if through is StageId.SCORING
                else "all"
            ),
            expected_model_ids=selected_models,
            model_failures=model_failures,
            model_statuses=model_statuses,
        )

    def run_batch(
        self,
        *,
        case_ids: Sequence[str] | None = None,
        replica_ids: Sequence[str] | None = None,
        model_ids: Sequence[str] | None = None,
        through: StageId = StageId.SCORING,
        action: ExecutionAction = "run",
        resume_md: bool = False,
    ) -> BatchResult:
        from .execution_preflight import preflight_execution

        cases = (tuple(dict.fromkeys(case_ids)) if case_ids is not None
                 else tuple(c.case_id for c in self.workflow.cases))
        replicas = (tuple(dict.fromkeys(replica_ids)) if replica_ids is not None
                    else tuple(r.replica_id for r in self.workflow.replicas))
        if not cases:
            raise ContractError("batch requires at least one case")
        if not replicas:
            raise ContractError("batch requires at least one replica")
        jobs = tuple(JobKey(case, replica) for case in cases for replica in replicas)
        preflight_execution(self.workflow, self.runtime, self.providers, jobs,
                            through=through, model_ids=model_ids)
        # Ownership covers admission as well as dispatch: a rejected second
        # controller cannot materialize an admission or a native invocation.
        with FileLock(self.workspace.root / ".batch.lock", timeout_seconds=0):
            try:
                return self._run_batch_owned(case_ids=cases, replica_ids=replicas,
                                             model_ids=model_ids, through=through,
                                             action=action, resume_md=resume_md)
            finally:
                cancel_pending = getattr(self._resources, "cancel_pending", None)
                if cancel_pending is not None:
                    cancel_pending()

    def _run_batch_owned(
        self,
        *,
        case_ids: Sequence[str] | None = None,
        replica_ids: Sequence[str] | None = None,
        model_ids: Sequence[str] | None = None,
        through: StageId = StageId.SCORING,
        action: ExecutionAction = "run",
        resume_md: bool = False,
    ) -> BatchResult:
        if resume_md:
            from .pipeline import validate_recovery_mode
            validate_recovery_mode(through, action, self.runtime.scheduler.execution_mode)
        if through is not StageId.SCORING and model_ids is not None:
            raise ContractError("model_ids is only valid when through is scoring")
        selected_cases = (
            tuple(dict.fromkeys(case_ids))
            if case_ids is not None
            else tuple(item.case_id for item in self.workflow.cases)
        )
        selected_replicas = (
            tuple(dict.fromkeys(replica_ids))
            if replica_ids is not None
            else tuple(item.replica_id for item in self.workflow.replicas)
        )
        if not selected_cases:
            raise ContractError("batch requires at least one case")
        if not selected_replicas:
            raise ContractError("batch requires at least one replica")
        jobs = tuple(
            JobKey(case_id, replica_id)
            for case_id in selected_cases
            for replica_id in selected_replicas
        )
        for job in jobs:
            self.workflow.case(job.case_id)
            self.workflow.replica(job.replica_id)
        results: dict[JobKey, JobResult] = {}
        failures: dict[JobKey, BatchFailure] = {}
        # Admission is case-scoped. Reserve it once before replica threads so
        # jobs for the same case do not mistake their sibling's active
        # admission attempt for an externally interrupted attempt.
        admitted_cases: set[str] = set()
        for case_id in selected_cases:
            task = TaskKey(StageId.ADMIT, case_id=case_id)
            latest = self.workspace.latest(task)
            admission_action: ExecutionAction = "run"
            if action == "resume" and latest is not None and latest.state.active:
                admission_action = "resume"
            elif action == "retry" and latest is not None and latest.state is AttemptState.FAILED:
                admission_action = "retry"
            try:
                prepared = self._prepared(case_id)
                self._execute_task(
                    task,
                    action=admission_action,
                    prepared_case=prepared,
                    upstream={},
                    new_on_conflict=(action == "new" and admission_action == "run"),
                )
                admitted_cases.add(case_id)
            except Exception as exc:
                for job in jobs:
                    if job.case_id == case_id:
                        failures[job] = BatchFailure(
                            job=job,
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
        runnable_jobs = tuple(job for job in jobs if job.case_id in admitted_cases)
        scheduler_limit = self.runtime.scheduler.max_workers
        if self.runtime.scheduler.execution_mode == "pipeline":
            from .pipeline import run_pipeline

            pipelined = run_pipeline(
                self, runnable_jobs, through=through, action=action, model_ids=model_ids,
                resume_md=resume_md,
            )
            return BatchResult(
                results=pipelined.results,
                failures={**failures, **pipelined.failures},
            )
        worker_count = min(scheduler_limit, max(1, len(runnable_jobs)))

        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="mmpbsa-job"
        ) as executor:
            futures = {
                executor.submit(
                    self.run_job,
                    job,
                    through=through,
                    action=action,
                    model_ids=model_ids,
                    strict_action=False,
                ): job
                for job in runnable_jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    results[job] = future.result()
                except Exception as exc:
                    failures[job] = BatchFailure(
                        job=job,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
        return BatchResult(results=results, failures=failures)

    def _inventory_input(
        self,
        summary: AttemptSummary,
        payload: Mapping[str, object],
    ) -> StageInput:
        attempt_root = (
            self.workspace.task_dir(summary.task)
            / "attempts"
            / summary.attempt_id
        )
        return StageInput(
            task=summary.task,
            attempt_id=summary.attempt_id,
            root=attempt_root,
            artifacts={},
            science_hash=sha256_digest(
                {
                    "science_hash": summary.science_hash,
                    "attempt_id": summary.attempt_id,
                    **dict(payload),
                }
            ),
            payload=dict(payload),
        )

    def _attempt_availability(
        self,
        task: TaskKey,
        *,
        prepared_case: PreparedCase,
        upstream: Mapping[str, StageInput],
        dependencies_available: bool,
    ) -> tuple[StageInput | None, StageInput | None]:
        """Return a completed input only when science and backend are current."""

        summary = self.workspace.latest(task)
        if summary is None:
            return None, None
        expected_hash: str | None = None
        expected_compatibility_key: str | None = None
        backend_error: dict[str, str] | None = None
        if dependencies_available:
            science = self._science(task, prepared_case)
            expected_hash = self._science_hash(
                task, science, prepared_case, upstream
            )
            try:
                _, _, _, compatibility = self._stage_execution_contract(
                    task.stage, science
                )
                expected_compatibility_key = compatibility.key
            except ArtifactIntegrityError:
                raise
            except Exception as exc:
                backend_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        native_science = (
            expected_hash is not None
            and summary.science_hash == expected_hash
        )
        native_backend = (
            expected_compatibility_key is not None
            and summary.compatibility_key == expected_compatibility_key
        )
        native_identity = native_science and native_backend
        usable = (
            summary.state is AttemptState.COMPLETED
            and native_identity
        )
        if summary.state is not AttemptState.COMPLETED:
            reason = "attempt_not_completed"
        elif usable:
            reason = None
        elif not dependencies_available:
            reason = "upstream_unavailable"
        elif backend_error is not None:
            reason = "backend_unavailable"
        elif not native_science:
            reason = "science_hash_mismatch"
        else:
            reason = "backend_compatibility_mismatch"
        inventory_payload: dict[str, object] = {
            "state": summary.state.value,
            "artifact_state": summary.artifact_state.value,
            "usable": usable,
            "reason": reason,
            "identity": (
                "native"
                if native_identity
                else None
            ),
            "observed_science_hash": summary.science_hash,
            "expected_science_hash": expected_hash,
            "observed_compatibility_key": summary.compatibility_key,
            "expected_compatibility_key": expected_compatibility_key,
        }
        if backend_error is not None:
            inventory_payload["backend_error"] = backend_error
        inventory = self._inventory_input(summary, inventory_payload)
        return (
            self.workspace.as_stage_input(summary) if usable else None,
            inventory,
        )

    def _case_finalization_inputs(
        self,
        case_id: str,
    ) -> dict[str, StageInput]:
        prepared = self._prepared(case_id)
        result: dict[str, StageInput] = {}
        for replica in self.workflow.replicas:
            md_task = TaskKey(
                StageId.MD,
                case_id=case_id,
                replica_id=replica.replica_id,
            )
            md, md_inventory = self._attempt_availability(
                md_task,
                prepared_case=prepared,
                upstream={},
                dependencies_available=True,
            )
            if md_inventory is not None:
                result[f"inventory:md:{replica.replica_id}"] = md_inventory
            trajectory_task = TaskKey(
                StageId.TRAJECTORY,
                case_id=case_id,
                replica_id=replica.replica_id,
            )
            trajectory_upstream = {} if md is None else {"md": md}
            trajectory, trajectory_inventory = self._attempt_availability(
                trajectory_task,
                prepared_case=prepared,
                upstream=trajectory_upstream,
                dependencies_available=md is not None,
            )
            if trajectory_inventory is not None:
                result[
                    f"inventory:trajectory:{replica.replica_id}"
                ] = trajectory_inventory
            qc_task = TaskKey(
                StageId.TRAJECTORY_QC,
                case_id=case_id,
                replica_id=replica.replica_id,
            )
            qc_upstream = (
                {} if trajectory is None else {"trajectory": trajectory}
            )
            qc, qc_inventory = self._attempt_availability(
                qc_task,
                prepared_case=prepared,
                upstream=qc_upstream,
                dependencies_available=trajectory is not None,
            )
            if qc_inventory is not None:
                result[
                    f"inventory:trajectory_qc:{replica.replica_id}"
                ] = qc_inventory
            if qc is not None:
                result[f"trajectory_qc:{replica.replica_id}"] = qc

            for model_id in self.workflow.scoring_model_ids:
                scoring_task = TaskKey(
                    StageId.SCORING,
                    case_id=case_id,
                    replica_id=replica.replica_id,
                    model_id=model_id,
                )
                scoring_dependencies = (
                    trajectory is not None and qc is not None
                )
                scoring_upstream = (
                    {
                        "trajectory": trajectory,
                        "trajectory_qc": qc,
                    }
                    if scoring_dependencies
                    else {}
                )
                scoring, scoring_inventory = self._attempt_availability(
                    scoring_task,
                    prepared_case=prepared,
                    upstream=scoring_upstream,
                    dependencies_available=scoring_dependencies,
                )
                suffix = f"{replica.replica_id}:{model_id}"
                if scoring_inventory is not None:
                    result[
                        f"inventory:scoring:{suffix}"
                    ] = scoring_inventory
                if scoring is not None:
                    result[f"scoring:{suffix}"] = scoring
        return result

    def finalize_case(
        self,
        case_id: str,
        *,
        action: ExecutionAction = "run",
    ) -> AttemptSummary:
        self.workflow.case(case_id)
        upstream = self._case_finalization_inputs(case_id)
        return self._execute_task(
            TaskKey(StageId.FINALIZE_CASE, case_id=case_id),
            action=action,
            prepared_case=self._prepared(case_id),
            upstream=upstream,
        )

    def _finalized_cases(self) -> dict[str, StageInput]:
        result: dict[str, StageInput] = {}
        for case in self.workflow.cases:
            task = TaskKey(StageId.FINALIZE_CASE, case_id=case.case_id)
            prepared = self._prepared(case.case_id)
            current, _ = self._attempt_availability(
                task,
                prepared_case=prepared,
                upstream=self._case_finalization_inputs(case.case_id),
                dependencies_available=True,
            )
            if current is not None:
                result[f"case:{case.case_id}"] = current
        if not result:
            raise ContractError(
                "project finalization requires at least one finalized case"
            )
        return result

    def _asset_inputs(
        self, analysis: StageInput
    ) -> dict[str, StageInput]:
        """Expose only revalidated inputs; the assets provider never scans."""

        result = {"analysis": analysis}
        for case in self.workflow.cases:
            prepared = self._prepared(case.case_id)
            admit_task = TaskKey(StageId.ADMIT, case_id=case.case_id)
            admit, _ = self._attempt_availability(
                admit_task,
                prepared_case=prepared,
                upstream={},
                dependencies_available=True,
            )
            if admit is not None:
                result[f"prepared:{case.case_id}"] = StageInput(
                    task=admit_task,
                    attempt_id=admit.attempt_id,
                    root=prepared.root,
                    artifacts=prepared.resolved_artifacts,
                    science_hash=sha256_digest(
                        prepared.manifest.as_dict()
                    ),
                    payload=prepared.manifest.as_dict(),
                )
            for replica in self.workflow.replicas:
                md_task = TaskKey(
                    StageId.MD,
                    case_id=case.case_id,
                    replica_id=replica.replica_id,
                )
                md, _ = self._attempt_availability(
                    md_task,
                    prepared_case=prepared,
                    upstream={},
                    dependencies_available=True,
                )
                trajectory_task = TaskKey(
                    StageId.TRAJECTORY,
                    case_id=case.case_id,
                    replica_id=replica.replica_id,
                )
                trajectory, _ = self._attempt_availability(
                    trajectory_task,
                    prepared_case=prepared,
                    upstream={} if md is None else {"md": md},
                    dependencies_available=md is not None,
                )
                if trajectory is not None:
                    result[
                        f"trajectory:{case.case_id}:{replica.replica_id}"
                    ] = trajectory
        return result

    def finalize_project(
        self,
        *,
        action: ExecutionAction = "run",
    ) -> ProjectResult:
        cases = self._finalized_cases()
        project_tasks = [TaskKey(StageId.ANALYSIS)]
        with_assets = self.runtime.provider_for("report.assets") is not None
        if with_assets:
            project_tasks.append(TaskKey(StageId.ASSETS))
        project_tasks.append(TaskKey(StageId.REPORT))
        if action == "new":
            project_actions: tuple[ExecutionAction, ...] = tuple(
                "new" for _ in project_tasks
            )
        else:
            project_actions = self._actions_for_tasks(
                project_tasks, action, strict=action != "run"
            )
        selected_actions = {
            task.stage: stage_action
            for task, stage_action in zip(
                project_tasks, project_actions, strict=True
            )
        }
        analysis = self._execute_task(
            TaskKey(StageId.ANALYSIS),
            action=selected_actions[StageId.ANALYSIS],
            prepared_case=None,
            upstream=cases,
        )
        analysis_input = self.workspace.as_stage_input(analysis)
        assets: AttemptSummary | None = None
        if with_assets:
            assets = self._execute_task(
                TaskKey(StageId.ASSETS),
                action=selected_actions[StageId.ASSETS],
                prepared_case=None,
                upstream=self._asset_inputs(analysis_input),
            )
        report_upstream = {"analysis": analysis_input}
        if assets is not None:
            report_upstream["assets"] = self.workspace.as_stage_input(assets)
        report = self._execute_task(
            TaskKey(StageId.REPORT),
            action=selected_actions[StageId.REPORT],
            prepared_case=None,
            upstream=report_upstream,
        )
        publication = self.publish_report(report)
        return ProjectResult(
            analysis=analysis,
            assets=assets,
            report=report,
            publication=publication,
        )

    def rebuild_report(self, *, rebuild_assets: bool = True) -> ProjectResult:
        """Rebuild assets and HTML while reusing an unchanged analysis."""

        cases = self._finalized_cases()
        analysis = self._execute_task(
            TaskKey(StageId.ANALYSIS),
            action="run",
            prepared_case=None,
            upstream=cases,
        )
        analysis_input = self.workspace.as_stage_input(analysis)
        assets: AttemptSummary | None = None
        if self.runtime.provider_for("report.assets") is not None:
            assets = self._execute_task(
                TaskKey(StageId.ASSETS),
                action="new" if rebuild_assets else "run",
                prepared_case=None,
                upstream=self._asset_inputs(analysis_input),
            )
        report_upstream = {"analysis": analysis_input}
        if assets is not None:
            report_upstream["assets"] = self.workspace.as_stage_input(assets)
        report = self._execute_task(
            TaskKey(StageId.REPORT),
            action="new",
            prepared_case=None,
            upstream=report_upstream,
        )
        publication = self.publish_report(report)
        return ProjectResult(
            analysis=analysis,
            assets=assets,
            report=report,
            publication=publication,
        )

    def publish_report(self, summary: AttemptSummary) -> Path:
        if summary.task.stage is not StageId.REPORT:
            raise ContractError("only a report attempt can be published")
        source = (
            self.workspace.task_dir(summary.task)
            / "attempts"
            / summary.attempt_id
            / "output"
        )
        target = self.workspace.root / "report"
        lock = FileLock(
            self.workspace.root / ".report-publication.lock",
            owner=f"mmpbsa:publish:{summary.attempt_id}",
            timeout_seconds=60,
            stale_after_seconds=24 * 60 * 60,
        )
        with lock:
            # Verify under the publication lock, then materialize only files in
            # the report attempt's declared, content-addressed inventory.
            artifacts = self.workspace.verify_artifacts(summary)
            index = artifacts.get("report_index")
            if index is None:
                raise ContractError(
                    "report attempt has no report_index artifact"
                )
            declared: dict[Path, tuple[str, Path]] = {}
            for role, path in artifacts.items():
                try:
                    relative = path.relative_to(source)
                except ValueError as exc:
                    raise ContractError(
                        f"report artifact is outside output/: {path}"
                    ) from exc
                if relative == Path("publication.json"):
                    raise ContractError(
                        "report provider cannot declare publication.json"
                    )
                if relative == Path("publication_inventory.json"):
                    raise ContractError(
                        "report provider cannot declare publication_inventory.json"
                    )
                if relative in declared:
                    raise ContractError(
                        f"report provider declared duplicate artifact path: {relative}"
                    )
                declared[relative] = (role, path)
            with atomic_directory(target) as staging:
                inventory: list[dict[str, object]] = []
                for relative, (role, path) in sorted(
                    declared.items(), key=lambda item: item[0].as_posix()
                ):
                    destination = staging / relative
                    materialize_immutable_file(path, destination)
                    sealed = summary.artifacts[role]
                    inventory.append(
                        {
                            "role": role,
                            "path": relative.as_posix(),
                            "sha256": sealed.sha256,
                            "size_bytes": sealed.size,
                        }
                    )
                atomic_write_json(
                    staging / "publication.json",
                    {
                        "schema_version": "mmpbsa.report-publication.v1",
                        "attempt_id": summary.attempt_id,
                        "science_hash": summary.science_hash,
                        "published_at": utc_now(),
                    },
                )
                atomic_write_json(
                    staging / "publication_inventory.json",
                    {
                        "schema_version": "mmpbsa.report-publication-inventory.v1",
                        "attempt_id": summary.attempt_id,
                        "artifacts": inventory,
                    },
                )
                # The report attempt was audited before it was sealed, but the
                # stable tree is a second publication boundary.  Re-run every
                # audit on the exact final staging tree, after publication
                # metadata is present and immediately before atomic replace.
                audit_stable_report_staging(staging, inventory)
            # Keep verification of the atomically installed tree inside the
            # same publication lock.  Otherwise a concurrent publisher could
            # replace ``target`` between installation and these hash checks.
            for record in inventory:
                published = target / str(record["path"])
                if (
                    not published.is_file()
                    or published.stat().st_size != record["size_bytes"]
                    or sha256_file(published) != record["sha256"]
                ):
                    raise ArtifactIntegrityError(
                        "published report artifact failed verification: "
                        f"{published}"
                    )
        return target / index.relative_to(source)

"""Thin, explicit orchestration for new projects; chemistry stays outside core."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from mmpbsa.utils import FileLock, atomic_write_json, sha256_digest, sha256_file
from mmpbsa.utils.filesystem import resolve_within

from .config import SchedulerConfig, load_runtime, load_workflow
from .errors import ContractError
from .execution_preflight import preflight_execution
from .inspection import read_workspace_status
from .models import JobKey
from .prepared import load_prepared_case
from .project_preflight import inspect_execution_environment
from .providers import ProviderRegistry, discover_providers
from .runner import WorkflowRunner

CAMPAIGN_SCHEMA = "mmpbsa.campaign.v1"
GATE_SCHEMA = "mmpbsa.smoke-gate.v1"


@dataclass(frozen=True)
class CampaignSpec:
    source_path: Path
    workflow: Path
    runtime: Path
    workspace: Path
    smoke_workflow: Path
    smoke_workspace: Path
    gate: Path
    exports: Path

    @classmethod
    def load(cls, path: Path) -> CampaignSpec:
        source = Path(path).resolve()
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ContractError(f"cannot read campaign: {source}") from exc
        names = {
            "workflow",
            "runtime",
            "workspace",
            "smoke_workflow",
            "smoke_workspace",
            "gate",
            "exports",
        }
        if not isinstance(raw, Mapping) or set(raw) != names | {"schema_version"}:
            raise ContractError("campaign requires its explicit input/output paths")
        if raw["schema_version"] != CAMPAIGN_SCHEMA:
            raise ContractError("unsupported campaign schema")
        paths: dict[str, Path] = {}
        for name in names:
            value = raw[name]
            if not isinstance(value, str) or not value:
                raise ContractError(f"campaign.{name} must be a path")
            target = Path(value)
            paths[name] = (
                (source.parent / target).resolve() if not target.is_absolute() else target.resolve()
            )
        workspaces = (paths["workspace"], paths["smoke_workspace"])
        if any(a.is_relative_to(b) for a, b in (workspaces, workspaces[::-1])):
            raise ContractError(
                "smoke and production workspaces must be separate, non-nested trees"
            )
        for target in (
            source,
            paths["workflow"],
            paths["runtime"],
            paths["smoke_workflow"],
            paths["gate"],
            paths["exports"],
        ):
            if any(target.is_relative_to(root) for root in workspaces):
                raise ContractError(
                    "campaign inputs/gate/exports cannot be inside execution workspaces"
                )
        return cls(source, **paths)


def postprocessing_runtime(runtime):
    """Offline finalization/reporting cannot be blocked by an MD GPU budget."""
    return replace(
        runtime,
        scheduler=SchedulerConfig(cpu_slots=1, threads_per_job=1, max_workers=1, gpu_ids=()),
    )


class CampaignRunner:
    """Public workflow-only port; construction and preflight are read-only.

    An optional project gate is a trusted local callable, not shell code or a
    framework chemical decision. It receives smoke evidence and returns a
    JSON-shaped {passed: bool, checks: [...]} receipt.
    """

    def __init__(
        self,
        spec: CampaignSpec,
        *,
        providers: ProviderRegistry | None = None,
        runner_factory: Callable = WorkflowRunner,
        project_gate: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    ):
        self.spec = spec
        self.providers = providers
        self.runner_factory = runner_factory
        self.project_gate = project_gate

    def _inputs(self, *, smoke: bool = False):
        workflow = load_workflow(self.spec.smoke_workflow if smoke else self.spec.workflow)
        runtime = load_runtime(self.spec.runtime)
        return workflow, runtime

    def _registry(self):
        return self.providers if self.providers is not None else discover_providers()

    def preflight(self, *, smoke: bool = False) -> dict[str, object]:
        workflow, runtime = self._inputs(smoke=smoke)
        jobs = [
            JobKey(case.case_id, replica.replica_id)
            for case in workflow.cases
            for replica in workflow.replicas
        ]
        registry = self._registry()
        report = preflight_execution(workflow, runtime, registry, jobs)
        report.update(
            inspect_execution_environment(
                workflow,
                runtime,
                registry,
                [self.spec.smoke_workspace if smoke else self.spec.workspace, self.spec.exports],
            )
        )
        return report

    def status(self) -> dict[str, object]:
        workflow, _ = self._inputs()
        return read_workspace_status(workflow, self.spec.workspace, summary=True)

    def _bindings(self) -> dict[str, object]:
        workflow, runtime = self._inputs()
        smoke, _ = self._inputs(smoke=True)
        if {case.case_id for case in smoke.cases} != {case.case_id for case in workflow.cases}:
            raise ContractError("smoke must cover exactly the declared production cases")
        if not smoke.resolved_protocol or not smoke.resolved_protocol.preset.smoke:
            raise ContractError("campaign smoke workflow must select an explicit smoke preset")
        if workflow.resolved_protocol is None or workflow.resolved_protocol.preset.smoke:
            raise ContractError("production workflow must select a production preset")
        if smoke.scoring_model_ids != workflow.scoring_model_ids:
            raise ContractError("smoke and production must select the same scoring models")
        if smoke.completion_policy != "all":
            raise ContractError("smoke gate requires every selected scoring model (policy all)")
        production_protocol = workflow.resolved_protocol
        smoke_protocol = smoke.resolved_protocol
        if (
            smoke_protocol.settings != production_protocol.settings
            or smoke_protocol.engine_overrides != production_protocol.engine_overrides
            or smoke_protocol.md_engine_profile != production_protocol.md_engine_profile
        ):
            raise ContractError("smoke and production engine/parameters must match")
        prepared = {}
        for case in workflow.cases:
            item = load_prepared_case(case.prepared_case, expected_case_id=case.case_id)
            if smoke.case(case.case_id).prepared_case.resolve() != item.manifest_path:
                raise ContractError("smoke and production must use the same sealed PreparedCase")
            prepared[case.case_id] = {
                "manifest_sha256": sha256_file(item.manifest_path),
                "artifacts": {
                    role: sha256_file(path) for role, path in item.resolved_artifacts.items()
                },
            }
        # Code and inputs are fixed, while harmless resource/GPU allocation changes
        # are excluded. Neither a stale gate nor an edited source tree is accepted.
        namespace = Path(__file__).resolve().parents[1]
        code = {
            str(path.relative_to(namespace)): sha256_file(path)
            for path in sorted(namespace.rglob("*.py"))
        }
        inherited = {
            name: value
            for name, value in os.environ.items()
            if name
            in {
                "PATH",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "OPENMM_PLUGIN_DIR",
                "OPENMM_CPU_THREADS",
            }
            or name.startswith(("OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_"))
        }
        hooks = None
        if self.project_gate is not None:
            try:
                source = inspect.getsourcefile(self.project_gate)
            except (TypeError, OSError) as exc:
                raise ContractError(
                    "project gate must have a verifiable local source file"
                ) from exc
            if source is None or not Path(source).is_file():
                raise ContractError("project gate must have a verifiable local source file")
            hooks = {
                "source_sha256": sha256_file(Path(source)),
                "name": f"{self.project_gate.__module__}.{self.project_gate.__qualname__}",
            }
        environment = inspect_execution_environment(
            workflow, runtime, self._registry(), [self.spec.workspace]
        )
        return {
            "workflow_sha256": sha256_file(self.spec.workflow),
            "smoke_workflow_sha256": sha256_file(self.spec.smoke_workflow),
            "prepared": prepared,
            "runtime_execution_sha256": sha256_digest(
                {
                    "providers": runtime.providers,
                    "backends": runtime.backends,
                    "environment": runtime.environment,
                    "threads_per_job": runtime.scheduler.threads_per_job,
                }
            ),
            "code_sha256": sha256_digest(code),
            "inherited_environment_sha256": sha256_digest(inherited),
            "dependency_probes": environment["dependency_probes"],
            "project_gate": hooks,
        }

    def _runner(self, *, smoke: bool = False, offline: bool = False):
        workflow, runtime = self._inputs(smoke=smoke)
        if offline:
            runtime = postprocessing_runtime(runtime)
        return self.runner_factory(
            workflow,
            runtime,
            self.spec.smoke_workspace if smoke else self.spec.workspace,
            providers=self._registry(),
        )

    def run_smoke(self) -> Mapping[str, object]:
        self.preflight(smoke=True)
        bindings = self._bindings()
        with FileLock(self.spec.source_path.parent / ".campaign.lock", timeout_seconds=0):
            runner = self._runner(smoke=True)
            # Presets remain frozen at three declared replicas; smoke exercises
            # only rep01 per case in its separate workspace.
            batch = runner.run_batch(replica_ids=(runner.workflow.replicas[0].replica_id,))
            passed = (
                not batch.failures
                and bool(batch.results)
                and all(job.completed for job in batch.results.values())
            )
            proof: dict[str, str] = {}
            for result in batch.results.values():
                for attempt in result.attempts:
                    state = (
                        runner.workspace.task_dir(attempt.task)
                        / "attempts"
                        / attempt.attempt_id
                        / "state.json"
                    )
                    proof[str(state.relative_to(self.spec.smoke_workspace))] = sha256_file(state)
                    for artifact in runner.workspace.verify_artifacts(attempt).values():
                        proof[str(artifact.relative_to(self.spec.smoke_workspace))] = sha256_file(
                            artifact
                        )
            evidence: dict[str, object] = {
                "schema_version": GATE_SCHEMA,
                "passed": passed,
                "bindings": bindings,
                "smoke_proof": proof,
                "project_checks": None,
            }
            if passed and self.project_gate is not None:
                project = dict(self.project_gate(evidence))
                if not isinstance(project.get("passed"), bool) or not isinstance(
                    project.get("checks"), list
                ):
                    raise ContractError("project gate must return passed: bool and checks: list")
                evidence["project_checks"] = project
                evidence["passed"] = project["passed"]
            if self._bindings() != bindings:
                raise ContractError("campaign source changed during smoke; gate was not published")
            atomic_write_json(self.spec.gate, evidence)
            return evidence

    def verify_gate(self) -> Mapping[str, object]:
        try:
            receipt = json.loads(self.spec.gate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ContractError("production requires a matching, passed smoke gate") from exc
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("schema_version") != GATE_SCHEMA
            or receipt.get("passed") is not True
            or receipt.get("bindings") != self._bindings()
        ):
            raise ContractError("smoke gate is failed/stale or belongs to different inputs/code")
        proof = receipt.get("smoke_proof")
        if not isinstance(proof, Mapping) or not proof:
            raise ContractError("smoke gate has no sealed execution evidence")
        for relative, digest in proof.items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise ContractError("invalid smoke proof")
            path = resolve_within(self.spec.smoke_workspace, relative, must_exist=True)
            if sha256_file(path) != digest:
                raise ContractError("smoke execution evidence changed")
        return receipt

    def run_production(self):
        # Gate failure occurs before workspace/registry ownership or native launch.
        self.verify_gate()
        self.preflight()
        with FileLock(self.spec.source_path.parent / ".campaign.lock", timeout_seconds=0):
            self.verify_gate()
            return self._runner().run_batch()

    def finalize(self):
        with FileLock(self.spec.source_path.parent / ".campaign.lock", timeout_seconds=0):
            runner = self._runner(offline=True)
            for case in runner.workflow.cases:
                runner.finalize_case(case.case_id)
            return runner.finalize_project()

    def report(self):
        with FileLock(self.spec.source_path.parent / ".campaign.lock", timeout_seconds=0):
            return self._runner(offline=True).rebuild_report(rebuild_assets=False)

    def export(self) -> Mapping[str, Path]:
        from .export import export_analysis
        from .inspection import WorkspaceReader
        from .models import TaskKey
        from .state import StageId

        # Portable analysis is read directly; no native provider/GPU admission.
        archive = self.spec.workspace / "report/data/analysis.json.gz"
        if not archive.is_file():
            reader = WorkspaceReader(self.spec.workspace)
            latest = reader.latest(TaskKey(StageId.ANALYSIS))
            if latest is None:
                raise ContractError("sealed analysis unavailable; finalize explicitly first")
            artifacts = reader.verify_artifacts(latest)
            if "analysis" not in artifacts:
                raise ContractError("completed analysis does not declare its canonical artifact")
            archive = artifacts["analysis"]
        return export_analysis(archive, self.spec.exports)

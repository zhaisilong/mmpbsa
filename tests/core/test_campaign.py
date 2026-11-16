from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_runner import FakeStageService, _project, _write_prepared_case

from mmpbsa.core import cli
from mmpbsa.core.campaign import GATE_SCHEMA, CampaignRunner, CampaignSpec
from mmpbsa.core.config import load_runtime, load_workflow
from mmpbsa.core.errors import ContractError
from mmpbsa.core.project_template import create_project
from mmpbsa.utils import sha256_file


def _runtime(path: Path) -> Path:
    target = path / "runtime.source.yaml"
    target.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mmpbsa.runtime.v1",
                "providers": {"md": "backends"},
                "scheduler": {"cpu_slots": 2, "threads_per_job": 1, "gpu_ids": ["0"]},
                "backends": {},
                "environment": {},
            }
        )
    )
    return target


def _spec(root: Path) -> CampaignSpec:
    document = root / "campaign.yaml"
    document.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mmpbsa.campaign.v1",
                "workflow": "workflow.yaml",
                "runtime": "runtime.source.yaml",
                "workspace": "_production",
                "smoke_workflow": "smoke.yaml",
                "smoke_workspace": "_smoke",
                "gate": "gate.json",
                "exports": "exports",
            }
        )
    )
    return CampaignSpec.load(document)


def test_project_init_explicit_new_inputs_and_no_submission(tmp_path):
    runtime = _runtime(tmp_path)
    prepared = _write_prepared_case(tmp_path / "prepared", "case_a")
    output = tmp_path / "new_project"
    files = create_project(
        output,
        project_id="future",
        preset="production_3x15ns@2",
        cases={"case_a": prepared},
        runtime_template=runtime,
        registry_path=tmp_path / "host.json",
    )
    assert len(files) == 5
    assert not (output / "_production").exists()
    assert not (output / "_smoke").exists()
    assert not (tmp_path / "host.json").exists()
    campaign = CampaignSpec.load(output / "campaign.yaml")
    production = load_workflow(campaign.workflow)
    smoke = load_workflow(campaign.smoke_workflow)
    configured = load_runtime(campaign.runtime)
    assert production.scoring_model_ids == ("gb", "pb")
    assert production.completion_policy == "all"
    assert len(production.replicas) == 3 and len(smoke.replicas) == 3
    assert configured.providers["md"] == "backends"
    assert configured.scheduler.execution_mode == "pipeline"
    assert configured.scheduler.cpu_affinity == "physical"
    assert configured.scheduler.resource_coordination.mode == "host"
    script = files["run_project.py"].read_text()
    assert script.index('os.environ[name] = "1"') < script.index("from mmpbsa.core.cli import")
    assert 'if __name__ == "__main__"' in script
    with pytest.raises(ContractError, match="existing"):
        create_project(
            output,
            project_id="future",
            preset="production_3x15ns@2",
            cases={"case_a": prepared},
            runtime_template=runtime,
            registry_path=tmp_path / "host.json",
        )


def test_campaign_rejects_nested_smoke_and_production(tmp_path):
    spec = _spec(tmp_path)
    document = yaml.safe_load(spec.source_path.read_text())
    document["smoke_workspace"] = "_production/smoke"
    spec.source_path.write_text(yaml.safe_dump(document))
    with pytest.raises(ContractError, match="non-nested"):
        CampaignSpec.load(spec.source_path)


def test_gate_missing_or_stale_never_constructs_runner(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    calls = []
    campaign = CampaignRunner(spec, runner_factory=lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(campaign, "_bindings", lambda: {"prepared": "new"})
    with pytest.raises(ContractError, match="requires"):
        campaign.run_production()
    spec.gate.write_text(
        json.dumps({"schema_version": GATE_SCHEMA, "passed": True, "bindings": {"prepared": "old"}})
    )
    with pytest.raises(ContractError, match="stale"):
        campaign.run_production()
    assert calls == [] and not spec.workspace.exists()


def test_gate_verifies_smoke_evidence_and_refuses_path_escape(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec.smoke_workspace.mkdir()
    evidence = spec.smoke_workspace / "state.json"
    evidence.write_text("sealed")
    campaign = CampaignRunner(spec)
    monkeypatch.setattr(campaign, "_bindings", lambda: {"sealed": 1})
    receipt = {
        "schema_version": GATE_SCHEMA,
        "passed": True,
        "bindings": {"sealed": 1},
        "smoke_proof": {"state.json": sha256_file(evidence)},
    }
    spec.gate.write_text(json.dumps(receipt))
    assert campaign.verify_gate()["passed"] is True
    evidence.write_text("modified")
    with pytest.raises(ContractError, match="changed"):
        campaign.verify_gate()
    receipt["smoke_proof"] = {"../campaign.yaml": sha256_file(spec.source_path)}
    spec.gate.write_text(json.dumps(receipt))
    with pytest.raises(Exception, match="relative|traversal|beneath"):
        campaign.verify_gate()


def test_report_only_scopes_out_gpu_registry_and_never_runs_md(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    runtime = load_runtime(_runtime(tmp_path))
    calls = []
    workflow = SimpleNamespace(cases=[])

    def factory(w, r, root, **kwargs):
        calls.append((r, root))
        return SimpleNamespace(rebuild_report=lambda **kw: kw)

    campaign = CampaignRunner(spec, providers=object(), runner_factory=factory)
    monkeypatch.setattr(campaign, "_inputs", lambda **kw: (workflow, runtime))
    assert campaign.report() == {"rebuild_assets": False}
    assert calls[0][0].scheduler.gpu_ids == ()
    assert calls[0][0].scheduler.resource_coordination.mode == "local"
    assert calls[0][0].scheduler.cpu_slots == 1
    assert not spec.smoke_workspace.exists() and not spec.workspace.exists()


def test_project_status_does_not_discover_providers_or_create_workspace(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    runtime = load_runtime(_runtime(tmp_path))
    campaign = CampaignRunner(spec)
    monkeypatch.setattr(campaign, "_inputs", lambda **kw: (SimpleNamespace(), runtime))
    monkeypatch.setattr(
        "mmpbsa.core.campaign.read_workspace_status",
        lambda *a, **kw: {"workspace_state": "missing"},
    )
    monkeypatch.setattr(campaign, "_registry", lambda: pytest.fail("provider discovery"))
    assert campaign.status()["workspace_state"] == "missing"
    assert not spec.workspace.exists()


def test_cli_status_does_not_load_runtime_or_construct_runner(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "load_workflow", lambda path: SimpleNamespace(schema_version="mmpbsa.workflow.v3")
    )
    monkeypatch.setattr(cli, "_runner", lambda args: pytest.fail("runner constructed"))
    monkeypatch.setattr(cli, "load_runtime", lambda path: pytest.fail("runtime loaded"))
    monkeypatch.setattr(
        "mmpbsa.core.inspection.read_workspace_status",
        lambda *a, **kw: {"workspace_state": "missing", "summary": {}},
    )
    assert cli.main(["--workspace", str(tmp_path / "absent"), "status", "--summary"]) == 0
    assert json.loads(capsys.readouterr().out)["workspace_state"] == "missing"
    assert not (tmp_path / "absent").exists()


def test_resource_and_export_cli_require_explicit_targets():
    parser = cli._parser()
    args = parser.parse_args(["resources", "reconcile", "--registry", "/tmp/pool.json"])
    assert args.confirm is False
    args = parser.parse_args(["export", "--analysis", "analysis.json.gz", "--output", "exports"])
    assert args.analysis == Path("analysis.json.gz")
    with pytest.raises(SystemExit):
        parser.parse_args(["resources", "init", "--registry", "/tmp/pool.json"])


def test_fake_campaign_smoke_production_finalize_export_end_to_end(tmp_path):
    """Real framework state machine with fake tools; no simulation is launched."""
    fixture = _project(
        tmp_path / "fixtures",
        replica_ids=("rep01", "rep02", "rep03"),
        completion_policy="all",
        with_assets=True,
    )

    class CanonicalFakeScoring(FakeStageService):
        def _invoke(self, request, method):
            # The compact runner fixture expects filenames. Here verify native
            # protocol controls, then adapt only the fake service's input format.
            assert all(
                "Generated Sander" in value for value in request.science["native_inputs"].values()
            )
            model = request.task.model_id
            key = (request.task.case_id, request.task.replica_id, model)
            self.controls.scoring_frame_ids[key] = tuple(
                range(1, request.science["expected_frames"] + 1)
            )
            native = (
                {"main": "gb.mdin"}
                if model == "gb"
                else {"complex_receptor": "pb.mdin", "ligand": "pb_ligand.mdin"}
            )
            adjusted = replace(request, science={**request.science, "native_inputs": native})
            result = super()._invoke(adjusted, method)
            payload = dict(result.payload)
            payload["tables"] = [{**row, "label": "DELTA TOTAL"} for row in payload["tables"]]
            payload["tables"].append({**payload["tables"][0], "label": "DMM"})
            return replace(result, payload=payload)

    provider = fixture.registry.service("fake", "scoring")
    provider.__class__ = CanonicalFakeScoring
    spec = _spec(tmp_path)
    runtime = yaml.safe_load(fixture.runtime_path.read_text())
    runtime["scheduler"]["execution_mode"] = "pipeline"
    spec.runtime.write_text(yaml.safe_dump(runtime))
    workflow = {
        "schema_version": "mmpbsa.workflow.v3",
        "project_id": "future_fake",
        "protocol": {
            "preset": "production_3x5ns@1",
            "models": ["gb", "pb"],
            "completion_policy": "all",
        },
        "cases": [
            {"case_id": "case_a", "prepared_case": str(fixture.workflow.cases[0].prepared_case)}
        ],
        "replicas": [
            {"replica_id": f"rep{i:02d}", "replica_index": i, "seed": 41000 + i}
            for i in range(1, 4)
        ],
    }
    spec.workflow.write_text(yaml.safe_dump(workflow))
    smoke = {
        **workflow,
        "project_id": "future_fake_smoke",
        "protocol": {**workflow["protocol"], "preset": "smoke_20ps@1"},
    }
    spec.smoke_workflow.write_text(yaml.safe_dump(smoke))
    campaign = CampaignRunner(spec, providers=fixture.registry)
    assert campaign.preflight()["workspace_modified"] is False
    assert not spec.workspace.exists() and not spec.smoke_workspace.exists()
    assert campaign.run_smoke()["passed"] is True
    smoke_md = [event for event in fixture.controls.events if event.capability == "md"]
    assert {event.replica_id for event in smoke_md} == {"rep01"}
    assert campaign.verify_gate()["passed"] is True
    assert campaign.run_production().completed
    assert len([e for e in fixture.controls.events if e.capability == "md"]) == 4
    campaign.finalize()
    before = len(fixture.controls.events)
    files = campaign.export()
    assert set(files) == {"case_table", "replica_table", "field_dictionary", "validation_summary"}
    assert all(path.is_file() for path in files.values())
    assert len(fixture.controls.events) == before
    assert campaign.status()["summary"]["md"]["completed"] == 3


def test_gate_rejects_changed_scientific_settings_before_smoke(tmp_path):
    fixture = _project(tmp_path / "fixtures", replica_ids=("rep01", "rep02", "rep03"))
    spec = _spec(tmp_path)
    spec.runtime.write_bytes(fixture.runtime_path.read_bytes())
    base = {
        "schema_version": "mmpbsa.workflow.v3",
        "project_id": "future_fake",
        "protocol": {"preset": "production_3x5ns@1", "models": ["gb", "pb"]},
        "cases": [
            {"case_id": "case_a", "prepared_case": str(fixture.workflow.cases[0].prepared_case)}
        ],
        "replicas": [
            {"replica_id": f"rep{i:02d}", "replica_index": i, "seed": 41000 + i}
            for i in range(1, 4)
        ],
    }
    spec.workflow.write_text(yaml.safe_dump(base))
    smoke = {
        **base,
        "protocol": {
            **base["protocol"],
            "preset": "smoke_20ps@1",
            "overrides": {"internal_dielectric": 4},
        },
    }
    spec.smoke_workflow.write_text(yaml.safe_dump(smoke))
    campaign = CampaignRunner(spec, providers=fixture.registry)
    with pytest.raises(ContractError, match="parameters must match"):
        campaign._bindings()
    smoke["protocol"].pop("overrides")
    smoke["protocol"]["completion_policy"] = "any"
    spec.smoke_workflow.write_text(yaml.safe_dump(smoke))
    with pytest.raises(ContractError, match="policy all"):
        campaign._bindings()
    assert not spec.smoke_workspace.exists()


def test_eta_cannot_claim_zero_remaining_for_still_running_overlong_task():
    from mmpbsa.core.execution_telemetry import TimingSamples

    samples = TimingSamples()
    samples.observe("pb", 60)
    samples.observe("pb", 80)
    estimate = samples.estimate([("pb", 90)])
    assert estimate["available"] is False
    assert "exceeds" in estimate["reason"]

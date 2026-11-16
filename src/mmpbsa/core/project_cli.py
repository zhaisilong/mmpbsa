"""Small CLI ports for future-project orchestration and explicit host resources."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from .campaign import CampaignRunner, CampaignSpec
from .errors import ContractError
from .project_template import create_project


def add_project_commands(commands) -> None:
    project = commands.add_parser(
        "project", help="explicit workflow-only future-project operations"
    )
    actions = project.add_subparsers(dest="project_action", required=True)
    init = actions.add_parser(
        "init", help="create a new project; never prepare chemistry or submit jobs"
    )
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--id", dest="project_id", required=True)
    init.add_argument("--preset", required=True)
    init.add_argument(
        "--case", action="append", required=True, help="CASE_ID=sealed/prepared_case.json"
    )
    init.add_argument("--runtime-template", type=Path, required=True)
    init.add_argument("--registry-path", type=Path, required=True)
    init.add_argument("--seeds", nargs=3, type=int, default=[41001, 41002, 41003])
    for action in ("preflight", "status", "run"):
        item = actions.add_parser(action)
        item.add_argument("--campaign", type=Path, required=True)
        if action == "preflight":
            item.add_argument("--smoke", action="store_true")
        if action == "run":
            item.add_argument(
                "--phase",
                required=True,
                choices=("smoke", "production", "finalize", "report", "export"),
            )
    resources = commands.add_parser("resources", help="explicit cooperative local host pool")
    modes = resources.add_subparsers(dest="resource_action", required=True)
    for action in ("init", "inspect", "reconcile"):
        item = modes.add_parser(action)
        item.add_argument("--registry", type=Path, required=True)
        if action == "init":
            item.add_argument(
                "--cpu", required=True, help="explicit logical representatives, e.g. 0,2,4-7"
            )
            item.add_argument(
                "--gpu", action="append", default=[], help="full GPU UUID; repeat per GPU"
            )
        if action == "reconcile":
            selection = item.add_mutually_exclusive_group()
            selection.add_argument("--dry-run", action="store_true")
            selection.add_argument("--confirm", action="store_true")
    export = commands.add_parser("export", help="portable scientific tables from a single analysis")
    export.add_argument("--analysis", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)


def _cpu_ids(value: str) -> tuple[int, ...]:
    result = []
    try:
        for token in value.split(","):
            parts = token.split("-")
            if len(parts) == 1:
                result.append(int(parts[0]))
            elif len(parts) == 2 and int(parts[0]) <= int(parts[1]):
                result.extend(range(int(parts[0]), int(parts[1]) + 1))
            else:
                raise ValueError
    except ValueError as exc:
        raise ContractError("--cpu must contain explicit non-negative CPU IDs/ranges") from exc
    if not result or min(result) < 0 or len(result) != len(set(result)):
        raise ContractError("--cpu must contain unique non-negative CPU IDs")
    return tuple(result)


def dispatch_project(
    arguments: argparse.Namespace,
    emit: Callable[[object], None],
    batch_json: Callable,
    project_json: Callable,
) -> int:
    if arguments.command == "export":
        from .export import export_analysis

        emit(
            {
                name: str(path)
                for name, path in export_analysis(arguments.analysis, arguments.output).items()
            }
        )
        return 0
    if arguments.command == "resources":
        from .host_resources import initialize_registry, inspect_registry, reconcile_registry

        if arguments.resource_action == "init":
            initialize_registry(
                arguments.registry, cpu_ids=_cpu_ids(arguments.cpu), gpu_ids=arguments.gpu
            )
            result = inspect_registry(arguments.registry)
        elif arguments.resource_action == "inspect":
            result = inspect_registry(arguments.registry)
        else:
            result = reconcile_registry(arguments.registry, confirm=arguments.confirm)
        emit(result)
        return 0
    if arguments.project_action == "init":
        cases = {}
        for declaration in arguments.case:
            if "=" not in declaration:
                raise ContractError("--case requires CASE_ID=sealed/prepared_case.json")
            case_id, path = declaration.split("=", 1)
            if case_id in cases:
                raise ContractError("duplicate --case identifier")
            cases[case_id] = Path(path)
        files = create_project(
            arguments.output,
            project_id=arguments.project_id,
            preset=arguments.preset,
            cases=cases,
            runtime_template=arguments.runtime_template,
            registry_path=arguments.registry_path,
            seeds=tuple(arguments.seeds),
        )
        emit({"submitted": False, "files": {name: str(path) for name, path in files.items()}})
        return 0
    campaign = CampaignRunner(CampaignSpec.load(arguments.campaign))
    if arguments.project_action == "preflight":
        emit(campaign.preflight(smoke=arguments.smoke))
        return 0
    if arguments.project_action == "status":
        value = campaign.status()
        emit(value)
        return 1 if value.get("workspace_state") == "corrupt" else 0
    if arguments.phase == "smoke":
        value = campaign.run_smoke()
        emit(value)
        return 0 if value["passed"] else 1
    if arguments.phase == "production":
        value = campaign.run_production()
        emit(batch_json(value))
        return 0 if not value.failures else 1
    if arguments.phase in {"finalize", "report"}:
        value = campaign.finalize() if arguments.phase == "finalize" else campaign.report()
        emit(project_json(value))
        return 0
    emit({name: str(path) for name, path in campaign.export().items()})
    return 0

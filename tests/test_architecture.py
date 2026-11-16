from __future__ import annotations

import ast
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE_ROOT = ROOT / "src" / "mmpbsa"
PACKAGES = ("utils", "core", "backends", "report")


def _python_files(package: str) -> tuple[Path, ...]:
    return tuple(sorted((NAMESPACE_ROOT / package).rglob("*.py")))


def _mmpbsa_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.name
                for alias in node.names
                if alias.name == "mmpbsa" or alias.name.startswith("mmpbsa.")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "mmpbsa" or node.module.startswith("mmpbsa."))
        ):
            modules.add(node.module)
    return modules


def test_single_distribution_preserves_pep420_namespace() -> None:
    assert NAMESPACE_ROOT.is_dir()
    assert not (NAMESPACE_ROOT / "__init__.py").exists()
    for package in PACKAGES:
        package_root = NAMESPACE_ROOT / package
        assert package_root.is_dir()
        assert (package_root / "__init__.py").is_file()
        assert (package_root / "py.typed").is_file()


def test_logical_package_dependency_dag() -> None:
    forbidden = {
        "utils": ("mmpbsa.",),
        "core": ("mmpbsa.backends", "mmpbsa.report"),
        "backends": ("mmpbsa.report",),
        "report": ("mmpbsa.backends",),
    }
    violations: list[str] = []
    for package, prefixes in forbidden.items():
        for path in _python_files(package):
            for module in _mmpbsa_imports(path):
                if any(module.startswith(prefix) for prefix in prefixes):
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")
    assert violations == []


def test_root_is_the_only_distribution_configuration() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["name"] == "mmpbsa"
    assert project["project"]["version"] == "0.7.0"
    assert project["project"]["scripts"] == {"mmpbsa": "mmpbsa.core.cli:main"}
    assert project["project"]["entry-points"]["mmpbsa.providers.v2"] == {
        "builtin": "mmpbsa.backends.provider:get_provider",
        "openmm": "mmpbsa.backends.openmm_provider:get_provider",
        "report": "mmpbsa.report.provider:get_provider",
    }
    assert project["tool"]["setuptools"]["packages"]["find"] == {
        "where": ["src"],
        "include": ["mmpbsa.*"],
        "namespaces": True,
    }
    assert not (ROOT / "packages").exists()


def test_core_is_the_only_provider_registry_owner() -> None:
    backend_init = (NAMESPACE_ROOT / "backends" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert (NAMESPACE_ROOT / "core" / "providers.py").is_file()
    assert not (NAMESPACE_ROOT / "backends" / "registry.py").exists()
    assert "ProviderRegistry" not in backend_init
    assert "ProviderProtocol" not in backend_init


def test_public_package_versions_match_distribution() -> None:
    for package in PACKAGES:
        text = (NAMESPACE_ROOT / package / "__init__.py").read_text(encoding="utf-8")
        match = re.search(r'^__version__ = "([^"]+)"$', text, re.MULTILINE)
        assert match is not None
        assert match.group(1) == "0.7.0"


def test_known_composition_roots_do_not_exceed_refactoring_budgets() -> None:
    budgets = {
        "src/mmpbsa/backends/services.py": 2850,
        "src/mmpbsa/backends/gromacs_phase.py": 900,
        "src/mmpbsa/backends/openmm.py": 1050,
        "src/mmpbsa/backends/openmm_receipt.py": 400,
        "src/mmpbsa/backends/openmm_provider.py": 1200,
        "src/mmpbsa/backends/openmm_worker.py": 850,
        "src/mmpbsa/backends/trajectory_dispatch.py": 250,
        "src/mmpbsa/backends/trajectory_v3.py": 1250,
        "src/mmpbsa/core/runner.py": 2100,
        "src/mmpbsa/core/prepared_projection.py": 350,
        "src/mmpbsa/backends/prep/builder.py": 1350,
        "src/mmpbsa/backends/prep/builder_v2.py": 1000,
    }
    observed = {
        relative: len((ROOT / relative).read_text(encoding="utf-8").splitlines())
        for relative in budgets
    }

    assert {
        relative: (observed[relative], budget)
        for relative, budget in budgets.items()
        if observed[relative] > budget
    } == {}

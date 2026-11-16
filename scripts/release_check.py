#!/usr/bin/env python3
"""Run reproducibility checks and audit a wheel built from repository sources."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def _run(arguments: list[str], *, cwd: Path, capture: bool = False) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def _source_files(
    root: Path,
    *,
    include_untracked: bool,
) -> tuple[Path, ...]:
    arguments = ["git", "ls-files", "-z", "--cached"]
    if include_untracked:
        arguments.extend(("--others", "--exclude-standard"))
    output = subprocess.run(
        arguments,
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return tuple(
        Path(item.decode("utf-8"))
        for item in output.split(b"\0")
        if item
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development-only: permit tracked or untracked working-tree changes",
    )
    parser.add_argument("--skip-tests", action="store_true")
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    status = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture=True,
    )
    if status and not arguments.allow_dirty:
        raise SystemExit("release check requires a clean Git working tree")
    if (root / "packages").exists():
        raise SystemExit("legacy packages/ nesting is forbidden")

    if not arguments.skip_tests:
        _run([sys.executable, "-m", "ruff", "check", "src", "tests"], cwd=root)
        _run([sys.executable, "-m", "pytest", "-q"], cwd=root)

    source_files = _source_files(
        root,
        include_untracked=arguments.allow_dirty,
    )
    with tempfile.TemporaryDirectory(prefix="mmpbsa-release-") as directory:
        temporary = Path(directory)
        source = temporary / "source"
        wheels = temporary / "wheels"
        for relative in source_files:
            origin = root / relative
            if not origin.is_file():
                continue
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
        wheels.mkdir()
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheels),
                str(source),
            ],
            cwd=temporary,
        )
        built = tuple(wheels.glob("mmpbsa-*.whl"))
        if len(built) != 1:
            raise SystemExit(f"expected one wheel, observed {len(built)}")
        with zipfile.ZipFile(built[0]) as archive:
            names = set(archive.namelist())
        expected_modules = {
            path.relative_to(source / "src").as_posix()
            for path in (source / "src" / "mmpbsa").rglob("*")
            if path.is_file()
        }
        wheel_modules = {name for name in names if name.startswith("mmpbsa/")}
        missing = expected_modules.difference(wheel_modules)
        unexpected = wheel_modules.difference(expected_modules)
        if missing or unexpected:
            raise SystemExit(
                "wheel source inventory mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if any(name.startswith(("build/", "packages/")) for name in names):
            raise SystemExit("wheel contains stale build/packages content")
        installed = temporary / "wheel-import"
        with zipfile.ZipFile(built[0]) as archive:
            archive.extractall(installed)
        _run(
            [
                sys.executable,
                "-I",
                "-c",
                "import pathlib, sys; "
                f"root = pathlib.Path({str(installed)!r}); "
                "sys.path.insert(0, str(root)); "
                "import mmpbsa.core as core; "
                "assert pathlib.Path(core.__file__).resolve().is_relative_to(root); "
                "from mmpbsa.core.cli import _parser; "
                "assert _parser().parse_args(['status', '--summary']).summary; "
                "from mmpbsa.core.providers import discover_providers; "
                "assert discover_providers().describe(); "
                "assert 'openmm' not in sys.modules and 'mdtraj' not in sys.modules; "
                "print('wheel-only CLI/provider import checked; optional engines remain lazy')",
            ],
            cwd=temporary,
        )
        print(f"clean wheel audited: {built[0].name} ({len(names)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

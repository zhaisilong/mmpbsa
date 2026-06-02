from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar, Iterable

from .common import DEFAULT_PROFILE, load_profile, read_json, remove_paths, require_files, utc_now, write_text_atomic


VALID_MODES = {"full", "prepare", "md", "analysis", "report"}


@dataclass(frozen=True)
class JobContext:
    job_id: str
    job_dir: Path
    config_path: Path
    config: dict[str, Any]
    protocol_path: Path
    protocol: dict[str, Any]

    def resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.job_dir / path
        return path.resolve()


def apply_env_overrides(protocol: dict[str, Any]) -> dict[str, Any]:
    profile = json.loads(json.dumps(protocol))
    runtime = profile.setdefault("runtime", {})
    md = profile.setdefault("md", {})
    mmpbsa = profile.setdefault("mmpbsa", {})
    if os.environ.get("MAMBA_ENV"):
        runtime["mamba_env"] = os.environ["MAMBA_ENV"]
    if os.environ.get("GMXRC"):
        runtime["gmxrc"] = os.environ["GMXRC"]
    if os.environ.get("GMX_BIN"):
        runtime["gmx_bin"] = os.environ["GMX_BIN"]
    if os.environ.get("GPU_ID"):
        runtime["gpu_id"] = os.environ["GPU_ID"]
    if os.environ.get("NTOMP"):
        md["ntomp"] = int(os.environ["NTOMP"])
    if os.environ.get("MMPBSA_NP"):
        mmpbsa["np"] = int(os.environ["MMPBSA_NP"])
    return profile


def discover_job_contexts(run_dir: Path, protocol: Path | None, job_id: str | None = None) -> list[JobContext]:
    root = run_dir.resolve()
    if not root.exists():
        raise SystemExit(f"RUN_DIR does not exist: {root}")
    protocol_path = (protocol or DEFAULT_PROFILE).resolve()
    profile = apply_env_overrides(load_profile(protocol_path))

    job_dirs = [root / job_id] if job_id else sorted(path for path in root.iterdir() if path.is_dir())
    contexts: list[JobContext] = []
    for directory in job_dirs:
        if not directory.exists():
            raise SystemExit(f"Job directory does not exist: {directory}")
        config_path = directory / f"{directory.name}.json"
        if not config_path.exists():
            if job_id:
                raise SystemExit(f"Missing job config: {config_path}")
            continue
        config = read_json(config_path)
        declared = str(config.get("job_id") or "").strip()
        if declared != directory.name:
            raise SystemExit(f"{config_path} has job_id={declared!r}; expected {directory.name!r}")
        if config_path.stem != directory.name:
            raise SystemExit(f"Job config filename must match directory name: {config_path}")
        contexts.append(
            JobContext(
                job_id=directory.name,
                job_dir=directory.resolve(),
                config_path=config_path.resolve(),
                config=config,
                protocol_path=protocol_path,
                protocol=profile,
            )
        )
    if not contexts:
        raise SystemExit(f"No job configs found under {root}")
    return contexts


class DoneFileRunner:
    STEPS: ClassVar[list[str]] = []
    MODE_STEPS: ClassVar[dict[str, list[str]]] = {}

    def __init__(self, context: JobContext) -> None:
        self.context = context
        self.config = context.config
        self.profile = context.protocol

    def mode_steps(self, mode: str) -> list[str]:
        if mode not in VALID_MODES:
            raise SystemExit(f"Unknown mode {mode!r}; choose one of: {', '.join(sorted(VALID_MODES))}")
        steps = self.MODE_STEPS.get(mode)
        if steps is None:
            raise SystemExit(f"{self.__class__.__name__} does not define mode {mode!r}")
        return steps

    def run(self, mode: str = "full", resume: bool = False, force: bool = False) -> None:
        if resume and force:
            raise SystemExit("--resume and --force are mutually exclusive")
        self.ensure_dirs()
        selected = self.mode_steps(mode)
        if force:
            self.clear_from_step(selected[0])
        self.ensure_previous_steps_done(selected[0])
        for step in selected:
            if self.done_file(step).exists():
                if resume:
                    print(f"{self.context.job_id}: skip {step} (.done exists)")
                    continue
                raise SystemExit(f"{self.context.job_id}: {self.done_file(step).name} exists; use --resume or --force")
            self.run_one_step(step)

    def ensure_previous_steps_done(self, first_step: str) -> None:
        first_index = self.STEPS.index(first_step)
        missing = [step for step in self.STEPS[:first_index] if not self.done_file(step).exists()]
        if missing:
            raise SystemExit(
                f"{self.context.job_id}: cannot start at {first_step}; missing previous done files: "
                + ", ".join(f".{step}_done" for step in missing)
            )

    def clear_from_step(self, first_step: str) -> None:
        first_index = self.STEPS.index(first_step)
        for step in self.STEPS[first_index:]:
            remove_paths([self.done_file(step), self.failed_file(step), *self.required_outputs(step)])
            self.cleanup_for_step(step)

    def run_one_step(self, step: str) -> None:
        self.ensure_dirs()
        started = monotonic()
        print(f"{self.context.job_id}: start {step}")
        try:
            getattr(self, f"step_{step}")()
            if not self.outputs_exist(step):
                raise RuntimeError(f"{step} finished but required outputs are missing")
        except Exception as exc:
            write_text_atomic(self.failed_file(step), f"{utc_now()}\n{exc}\n")
            raise
        write_text_atomic(self.done_file(step), f"{utc_now()}\nelapsed_seconds={monotonic() - started:.3f}\n")
        remove_paths([self.failed_file(step)])
        print(f"{self.context.job_id}: done {step}")

    def done_file(self, step: str) -> Path:
        return self.context.job_dir / f".{step}_done"

    def failed_file(self, step: str) -> Path:
        return self.context.job_dir / f".{step}_failed"

    def outputs_exist(self, step: str) -> bool:
        return require_files(self.required_outputs(step))

    def cleanup_for_step(self, step: str) -> None:
        return None

    def remove_dirs(self, directories: Iterable[Path]) -> None:
        for directory in directories:
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)

    def ensure_dirs(self) -> None:
        raise NotImplementedError

    def required_outputs(self, step: str) -> list[Path]:
        raise NotImplementedError

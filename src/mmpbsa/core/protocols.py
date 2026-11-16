"""Versioned, typed scientific protocol presets.

The YAML workflow selects a preset and a deliberately small set of typed
overrides.  Native GROMACS and Amber inputs are generated from that resolved
document; hand-written engine input is not part of the public workflow
contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from .errors import ConfigurationError

PROTOCOL_PRESET_SCHEMA: Final = "mmpbsa.protocol-preset.v1"
MD_PHASE_PLAN_SCHEMA: Final = "mmpbsa.md-phase-plan.v2"
AUTO_DIELECTRIC_TOKEN: Final = "__MMPBSA_AUTO_INTERNAL_DIELECTRIC__"
DEFAULT_MD_ENGINE_PROFILE: Final = "gromacs_canonical@1"
_MD_TIMESTEP_FS: Final = 2

_SUPPORTED_MD_ENGINE_PROFILES = frozenset(
    {DEFAULT_MD_ENGINE_PROFILE, "openmm_8_5@2"}
)

_SUPPORTED_MODELS = ("gb", "pb")
_MODEL_SET = frozenset(_SUPPORTED_MODELS)
_TYPED_OVERRIDE_KEYS = frozenset(
    {
        "temperature_k",
        "pressure_bar",
        "salt_concentration_molar",
        "internal_dielectric",
        "gb_external_dielectric",
        "pb_external_dielectric",
        "interface_cutoff_angstrom",
        "native_contact_cutoff_angstrom",
        "interface_rmsd_p95_block_angstrom",
        "ligand_rmsd_p95_warn_angstrom",
        "interface_distance_block_angstrom",
    }
)
_GROMACS_OVERRIDE_KEYS = {
    "em": frozenset({"emtol", "emstep"}),
    "nvt": frozenset({"tau_t", "lincs_iter", "lincs_order"}),
    "npt": frozenset({"tau_t", "tau_p", "compressibility", "lincs_iter", "lincs_order"}),
    "production": frozenset(
        {"tau_t", "tau_p", "compressibility", "lincs_iter", "lincs_order"}
    ),
}
_AMBER_OVERRIDE_KEYS = {
    "gb": frozenset({"surften"}),
    "pb": frozenset({"fillratio", "maxitn", "dprob"}),
}


@dataclass(frozen=True, slots=True)
class FramePlan:
    """An inclusive, trajectory-aligned analysis window."""

    production_ps: int
    trajectory_interval_ps: int
    analysis_start_ps: int
    analysis_end_ps: int
    scoring_interval_ps: int

    def __post_init__(self) -> None:
        values = {
            "production_ps": self.production_ps,
            "trajectory_interval_ps": self.trajectory_interval_ps,
            "analysis_start_ps": self.analysis_start_ps,
            "analysis_end_ps": self.analysis_end_ps,
            "scoring_interval_ps": self.scoring_interval_ps,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
            raise ConfigurationError("frame-plan values must be integer picoseconds")
        if self.production_ps < 1 or self.trajectory_interval_ps < 1:
            raise ConfigurationError("production and trajectory intervals must be positive")
        if not 0 <= self.analysis_start_ps < self.analysis_end_ps <= self.production_ps:
            raise ConfigurationError("analysis window must lie inside production")
        for label, value in (
            ("production_ps", self.production_ps),
            ("analysis_start_ps", self.analysis_start_ps),
            ("analysis_end_ps", self.analysis_end_ps),
            ("scoring_interval_ps", self.scoring_interval_ps),
        ):
            if value % self.trajectory_interval_ps:
                raise ConfigurationError(
                    f"{label} must align to trajectory_interval_ps"
                )

    @property
    def stored_frames(self) -> int:
        return self.production_ps // self.trajectory_interval_ps + 1

    @property
    def start_frame(self) -> int:
        return self.analysis_start_ps // self.trajectory_interval_ps + 1

    @property
    def end_frame(self) -> int:
        return self.analysis_end_ps // self.trajectory_interval_ps + 1

    @property
    def trajectory_stride(self) -> int:
        return self.scoring_interval_ps // self.trajectory_interval_ps

    @property
    def expected_frames(self) -> int:
        return (
            (self.analysis_end_ps - self.analysis_start_ps)
            // self.scoring_interval_ps
            + 1
        )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": "mmpbsa.frame-manifest.v1",
            "source_stage": "production",
            "source_trajectory_interval_ps": self.trajectory_interval_ps,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "stride": self.trajectory_stride,
            "expected_frames": self.expected_frames,
        }

    @property
    def manifest_digest(self) -> str:
        encoded = json.dumps(
            self.manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "production_ps": self.production_ps,
            "trajectory_interval_ps": self.trajectory_interval_ps,
            "stored_frames": self.stored_frames,
            "analysis_start_ps": self.analysis_start_ps,
            "analysis_end_ps": self.analysis_end_ps,
            "scoring_interval_ps": self.scoring_interval_ps,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "trajectory_stride": self.trajectory_stride,
            "expected_frames_per_replica": self.expected_frames,
            "frame_manifest": self.manifest,
            "frame_manifest_digest": self.manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class ProtocolPreset:
    preset_id: str
    frame_plan: FramePlan
    replicas: int = 3
    smoke: bool = False

    def __post_init__(self) -> None:
        if not self.preset_id or "@" not in self.preset_id:
            raise ConfigurationError("preset_id must carry an @version suffix")
        if self.replicas != 3:
            raise ConfigurationError("production presets require three replicas")


@dataclass(frozen=True, slots=True)
class ResolvedProtocol:
    """Canonical protocol consumed by ``WorkflowDefinition``."""

    preset: ProtocolPreset
    md_engine_profile: str
    models: tuple[str, ...]
    completion_policy: str
    settings: Mapping[str, object]
    engine_overrides: Mapping[str, Mapping[str, Mapping[str, object]]]
    stages: Mapping[str, Mapping[str, object]]
    analysis: Mapping[str, object]
    report_defaults: Mapping[str, object]

    @property
    def protocol_id(self) -> str:
        return self.preset.preset_id

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": PROTOCOL_PRESET_SCHEMA,
            "preset": self.preset.preset_id,
            "md_engine_profile": self.md_engine_profile,
            "replicas": self.preset.replicas,
            "frame_plan": self.preset.frame_plan.as_dict(),
            "models": list(self.models),
            "completion_policy": self.completion_policy,
            "settings": dict(self.settings),
            "engine_overrides": {
                engine: {
                    section: dict(values)
                    for section, values in sections.items()
                }
                for engine, sections in self.engine_overrides.items()
            },
            "stages": {
                stage: dict(science) for stage, science in self.stages.items()
            },
            "analysis": dict(self.analysis),
            "report_defaults": dict(self.report_defaults),
        }


_PRESETS: Final[dict[str, ProtocolPreset]] = {
    "production_3x5ns@1": ProtocolPreset(
        "production_3x5ns@1",
        FramePlan(
            production_ps=5_000,
            trajectory_interval_ps=20,
            analysis_start_ps=3_000,
            analysis_end_ps=5_000,
            scoring_interval_ps=20,
        ),
    ),
    "production_3x15ns@1": ProtocolPreset(
        "production_3x15ns@1",
        FramePlan(
            production_ps=15_000,
            trajectory_interval_ps=20,
            analysis_start_ps=5_000,
            analysis_end_ps=15_000,
            scoring_interval_ps=20,
        ),
    ),
    "production_3x50ns@1": ProtocolPreset(
        "production_3x50ns@1",
        FramePlan(
            production_ps=50_000,
            trajectory_interval_ps=20,
            analysis_start_ps=10_000,
            analysis_end_ps=50_000,
            scoring_interval_ps=40,
        ),
    ),
    "production_3x100ns@1": ProtocolPreset(
        "production_3x100ns@1",
        FramePlan(
            production_ps=100_000,
            trajectory_interval_ps=20,
            analysis_start_ps=20_000,
            analysis_end_ps=100_000,
            scoring_interval_ps=40,
        ),
    ),
    "production_3x5ns@2": ProtocolPreset(
        "production_3x5ns@2",
        FramePlan(
            production_ps=5_000,
            trajectory_interval_ps=20,
            analysis_start_ps=3_000,
            analysis_end_ps=5_000,
            scoring_interval_ps=20,
        ),
    ),
    "production_3x15ns@2": ProtocolPreset(
        "production_3x15ns@2",
        FramePlan(
            production_ps=15_000,
            trajectory_interval_ps=20,
            analysis_start_ps=5_000,
            analysis_end_ps=15_000,
            scoring_interval_ps=20,
        ),
    ),
    "production_3x50ns@2": ProtocolPreset(
        "production_3x50ns@2",
        FramePlan(
            production_ps=50_000,
            trajectory_interval_ps=20,
            analysis_start_ps=10_000,
            analysis_end_ps=50_000,
            scoring_interval_ps=40,
        ),
    ),
    "production_3x100ns@2": ProtocolPreset(
        "production_3x100ns@2",
        FramePlan(
            production_ps=100_000,
            trajectory_interval_ps=20,
            analysis_start_ps=20_000,
            analysis_end_ps=100_000,
            scoring_interval_ps=40,
        ),
    ),
}
_SMOKE_PRESETS: Final[dict[str, ProtocolPreset]] = {
    "smoke_20ps@1": ProtocolPreset(
        "smoke_20ps@1",
        FramePlan(
            production_ps=20,
            trajectory_interval_ps=2,
            analysis_start_ps=0,
            analysis_end_ps=20,
            scoring_interval_ps=2,
        ),
        smoke=True,
    ),
    "smoke_20ps@2": ProtocolPreset(
        "smoke_20ps@2",
        FramePlan(
            production_ps=20,
            trajectory_interval_ps=2,
            analysis_start_ps=0,
            analysis_end_ps=20,
            scoring_interval_ps=2,
        ),
        smoke=True,
    ),
}


def available_protocol_presets(
    *, include_smoke: bool = False
) -> tuple[str, ...]:
    """Return stable public production preset IDs."""

    values = tuple(_PRESETS)
    if include_smoke:
        values += tuple(_SMOKE_PRESETS)
    return values


def protocol_preset(preset_id: str) -> ProtocolPreset:
    try:
        return {**_PRESETS, **_SMOKE_PRESETS}[preset_id]
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown protocol preset {preset_id!r}; supported presets: "
            + ", ".join(available_protocol_presets(include_smoke=True))
        ) from exc


def resolve_protocol(
    preset_id: str,
    *,
    md_engine_profile: str = DEFAULT_MD_ENGINE_PROFILE,
    models: Sequence[str] | None = None,
    completion_policy: str = "all",
    overrides: Mapping[str, object] | None = None,
    engine_overrides: Mapping[str, object] | None = None,
) -> ResolvedProtocol:
    """Resolve a preset into executable, canonical stage science."""

    preset = protocol_preset(preset_id)
    selected_md_profile = _validate_md_engine_profile(preset, md_engine_profile)
    selected_models = _validate_models(models)
    if completion_policy not in {"all", "any"}:
        raise ConfigurationError("protocol.completion_policy must be 'all' or 'any'")
    settings = _resolve_settings(overrides or {})
    native_overrides = _validate_engine_overrides(engine_overrides or {})
    if _preset_version(preset) == 2 and native_overrides["gromacs"]:
        raise ConfigurationError(
            "protocol.engine_overrides.gromacs is not supported by engine-neutral "
            "@2 MD presets"
        )
    stages = _stage_science(
        preset,
        md_engine_profile=selected_md_profile,
        models=selected_models,
        completion_policy=completion_policy,
        settings=settings,
        engine_overrides=native_overrides,
    )
    analysis = _analysis_science(selected_models)
    report_defaults = _report_defaults(preset, selected_models, settings)
    return ResolvedProtocol(
        preset=preset,
        md_engine_profile=selected_md_profile,
        models=selected_models,
        completion_policy=completion_policy,
        settings=settings,
        engine_overrides=native_overrides,
        stages=stages,
        analysis=analysis,
        report_defaults=report_defaults,
    )


def _preset_version(preset: ProtocolPreset) -> int:
    _, _, raw_version = preset.preset_id.rpartition("@")
    try:
        return int(raw_version)
    except ValueError as exc:  # pragma: no cover - guarded by the preset registry
        raise ConfigurationError(
            f"protocol preset has an invalid version: {preset.preset_id!r}"
        ) from exc


def _validate_md_engine_profile(
    preset: ProtocolPreset,
    value: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError("protocol.md_engine_profile must be a non-empty string")
    if value == "openmm_8_5@1":
        if _preset_version(preset) == 1:
            raise ConfigurationError(
                f"protocol preset {preset.preset_id!r} requires "
                f"md_engine_profile={DEFAULT_MD_ENGINE_PROFILE!r}"
            )
        raise ConfigurationError(
            "openmm_8_5@1 is archived read-only; new execution requires "
            "md_engine_profile='openmm_8_5@2'"
        )
    if value not in _SUPPORTED_MD_ENGINE_PROFILES:
        raise ConfigurationError(
            f"unknown protocol.md_engine_profile {value!r}; supported profiles: "
            + ", ".join(sorted(_SUPPORTED_MD_ENGINE_PROFILES))
        )
    if _preset_version(preset) == 1 and value != DEFAULT_MD_ENGINE_PROFILE:
        raise ConfigurationError(
            f"protocol preset {preset.preset_id!r} requires "
            f"md_engine_profile={DEFAULT_MD_ENGINE_PROFILE!r}"
        )
    return value


def _validate_models(models: Sequence[str] | None) -> tuple[str, ...]:
    if models is None:
        return _SUPPORTED_MODELS
    if isinstance(models, (str, bytes, bytearray)):
        raise ConfigurationError("protocol.models must be a sequence")
    selected = tuple(models)
    if not selected:
        raise ConfigurationError("protocol.models cannot be empty")
    if any(not isinstance(model, str) or not model for model in selected):
        raise ConfigurationError("protocol.models entries must be non-empty strings")
    if len(set(selected)) != len(selected):
        raise ConfigurationError("protocol.models entries must be unique")
    unknown = set(selected).difference(_MODEL_SET)
    if unknown:
        raise ConfigurationError(
            "protocol.models contains unknown models: " + ", ".join(sorted(unknown))
        )
    return selected


def _resolve_settings(overrides: Mapping[str, object]) -> dict[str, object]:
    unknown = set(overrides).difference(_TYPED_OVERRIDE_KEYS)
    if unknown:
        raise ConfigurationError(
            "protocol.overrides contains unknown fields: " + ", ".join(sorted(unknown))
        )
    settings: dict[str, object] = {
        "temperature_k": 300.0,
        "pressure_bar": 1.0,
        "salt_concentration_molar": 0.15,
        "internal_dielectric": "auto",
        "gb_external_dielectric": 78.5,
        "pb_external_dielectric": 80.0,
        "interface_cutoff_angstrom": 5.0,
        "native_contact_cutoff_angstrom": 4.5,
        "interface_rmsd_p95_block_angstrom": 3.0,
        "ligand_rmsd_p95_warn_angstrom": 5.0,
        "interface_distance_block_angstrom": 5.0,
    }
    settings.update(overrides)
    for key in _TYPED_OVERRIDE_KEYS.difference({"internal_dielectric"}):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"protocol.overrides.{key} must be a number")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ConfigurationError(
                f"protocol.overrides.{key} must be finite and positive"
            )
        settings[key] = number
    dielectric = settings["internal_dielectric"]
    if dielectric != "auto":
        if isinstance(dielectric, bool) or not isinstance(dielectric, (int, float)):
            raise ConfigurationError(
                "protocol.overrides.internal_dielectric must be auto, 2, 3, or 4"
            )
        dielectric = float(dielectric)
        if dielectric not in {2.0, 3.0, 4.0}:
            raise ConfigurationError(
                "protocol.overrides.internal_dielectric must be auto, 2, 3, or 4"
            )
        settings["internal_dielectric"] = dielectric
    return settings


def _validate_engine_overrides(
    value: Mapping[str, object],
) -> dict[str, dict[str, dict[str, object]]]:
    unknown_engines = set(value).difference({"gromacs", "amber"})
    if unknown_engines:
        raise ConfigurationError(
            "protocol.engine_overrides contains unknown engines: "
            + ", ".join(sorted(unknown_engines))
        )
    result: dict[str, dict[str, dict[str, object]]] = {}
    for engine, allowed_sections in (
        ("gromacs", _GROMACS_OVERRIDE_KEYS),
        ("amber", _AMBER_OVERRIDE_KEYS),
    ):
        raw_sections = value.get(engine, {})
        if not isinstance(raw_sections, Mapping):
            raise ConfigurationError(f"protocol.engine_overrides.{engine} must be a mapping")
        unknown_sections = set(raw_sections).difference(allowed_sections)
        if unknown_sections:
            raise ConfigurationError(
                f"protocol.engine_overrides.{engine} contains unknown sections: "
                + ", ".join(sorted(str(item) for item in unknown_sections))
            )
        sections: dict[str, dict[str, object]] = {}
        for section, raw_values in raw_sections.items():
            if not isinstance(raw_values, Mapping):
                raise ConfigurationError(
                    f"protocol.engine_overrides.{engine}.{section} must be a mapping"
                )
            unknown = set(raw_values).difference(allowed_sections[str(section)])
            if unknown:
                raise ConfigurationError(
                    f"protocol.engine_overrides.{engine}.{section} contains "
                    "unknown fields: " + ", ".join(sorted(str(item) for item in unknown))
                )
            resolved: dict[str, object] = {}
            for key, item in raw_values.items():
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or float(item) <= 0
                ):
                    raise ConfigurationError(
                        f"protocol.engine_overrides.{engine}.{section}.{key} "
                        "must be finite and positive"
                    )
                if key in {"lincs_iter", "lincs_order", "maxitn"} and (
                    not isinstance(item, int)
                ):
                    raise ConfigurationError(
                        f"protocol.engine_overrides.{engine}.{section}.{key} "
                        "must be a positive integer"
                    )
                resolved[str(key)] = item
            sections[str(section)] = resolved
        result[engine] = sections
    return result


def _stage_science(
    preset: ProtocolPreset,
    *,
    md_engine_profile: str,
    models: tuple[str, ...],
    completion_policy: str,
    settings: Mapping[str, object],
    engine_overrides: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, Mapping[str, object]]:
    frame_plan = preset.frame_plan
    return {
        "admit": {
            "tolerances": {"charge": 1.0e-8, "mass": 1.0e-8, "parameter": 1.0e-8},
        },
        "md": (
            _md_science_v1(preset, settings, engine_overrides["gromacs"])
            if _preset_version(preset) == 1
            else _md_science_v2(preset, settings, md_engine_profile)
        ),
        "trajectory": _trajectory_science(preset),
        "trajectory_qc": _qc_science(
            frame_plan,
            settings,
            production_t0_reference=_preset_version(preset) == 2,
        ),
        "scoring": {
            "completion_policy": completion_policy,
            "decomposition": {
                "enabled": False,
                "scope": "interface_residues",
                "note": "optional analysis; disabled by default",
            },
            "models": {
                model: _scoring_science(
                    model,
                    frame_plan,
                    settings,
                    engine_overrides["amber"].get(model, {}),
                )
                for model in models
            },
        },
    }


def _trajectory_science(preset: ProtocolPreset) -> Mapping[str, object]:
    frame_plan = preset.frame_plan
    shared: dict[str, object] = {
        "start_frame": frame_plan.start_frame,
        "end_frame": frame_plan.end_frame,
        "stride": frame_plan.trajectory_stride,
        "expected_frames": frame_plan.expected_frames,
        "frame_manifest": frame_plan.manifest,
        "frame_manifest_digest": frame_plan.manifest_digest,
        "autoimage_anchor_mask": "{{selection:ligand_all}}",
        "autoimage_fixed_mask": "{{selection:receptor_all}}",
        "fit_mask": "{{selection:receptor_interface_backbone}}",
    }
    if _preset_version(preset) == 1:
        return {
            "source_tpr_artifact": "tpr",
            "source_trajectory_artifact": "xtc",
            "index_role": "index",
            "group_name": "Complex",
            **shared,
        }
    return {
        "adapter_contract": "mmpbsa.trajectory.v3",
        "preprocessing_revision": "whole_molecules_v1",
        "source_trajectory_artifact": "full_trajectory",
        "full_system_topology_role": "md_topology",
        "full_system_coordinates_role": "md_coordinates",
        "source_trajectory_contract": {
            "first_time_ps": 0.0,
            "interval_ps": frame_plan.trajectory_interval_ps,
            "interval_steps": (
                frame_plan.trajectory_interval_ps * 1_000 // _MD_TIMESTEP_FS
            ),
            "expected_frames": frame_plan.stored_frames,
        },
        **shared,
    }


def _md_science_v1(
    preset: ProtocolPreset,
    settings: Mapping[str, object],
    overrides: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    frame_plan = preset.frame_plan
    dt_ps = 0.002
    em_steps = 200 if preset.smoke else 50_000
    nvt_steps = 500 if preset.smoke else 50_000
    npt_steps = 500 if preset.smoke else 100_000
    common = {
        "integrator": "md",
        "dt": dt_ps,
        "constraints": "h-bonds",
        "constraint-algorithm": "lincs",
        "lincs-iter": 1,
        "lincs-order": 4,
        "cutoff-scheme": "Verlet",
        "nstlist": 20,
        "rlist": 1.0,
        "coulombtype": "PME",
        "rcoulomb": 1.0,
        "vdwtype": "Cut-off",
        "rvdw": 1.0,
        "DispCorr": "EnerPres",
        "tcoupl": "V-rescale",
        "tc-grps": "System",
        "tau_t": 1.0,
        "ref_t": settings["temperature_k"],
        "pbc": "xyz",
    }
    em = {
        "title": "MMPBSA energy minimization",
        "integrator": "steep",
        "emtol": 1000.0,
        "emstep": 0.001,
        "nsteps": em_steps,
        "cutoff-scheme": "Verlet",
        "nstlist": 20,
        "rlist": 1.0,
        "coulombtype": "PME",
        "rcoulomb": 1.0,
        "vdwtype": "Cut-off",
        "rvdw": 1.0,
        "DispCorr": "EnerPres",
        "constraints": "none",
        "pbc": "xyz",
    }
    nvt = {
        **common,
        "title": f"MMPBSA NVT {nvt_steps * dt_ps:g} ps",
        "nsteps": nvt_steps,
        "continuation": "no",
        "pcoupl": "no",
        "gen_vel": "yes",
        "gen_temp": settings["temperature_k"],
        "gen_seed": "{{replica.seed}}",
        "nstxout": 0,
        "nstvout": 0,
        "nstfout": 0,
        "nstenergy": 500,
        "nstlog": 500,
        "nstxout-compressed": 0,
    }
    npt = {
        **common,
        "title": f"MMPBSA NPT {npt_steps * dt_ps:g} ps",
        "nsteps": npt_steps,
        "continuation": "yes",
        "pcoupl": "C-rescale",
        "pcoupltype": "isotropic",
        "tau_p": 5.0,
        "ref_p": settings["pressure_bar"],
        "compressibility": 4.5e-5,
        "gen_vel": "no",
        "nstxout": 0,
        "nstvout": 0,
        "nstfout": 0,
        "nstenergy": 500,
        "nstlog": 500,
        "nstxout-compressed": 0,
    }
    production = {
        **common,
        "title": f"MMPBSA production {frame_plan.production_ps / 1000:g} ns",
        "nsteps": int(frame_plan.production_ps / dt_ps),
        "continuation": "yes",
        "pcoupl": "C-rescale",
        "pcoupltype": "isotropic",
        "tau_p": 5.0,
        "ref_p": settings["pressure_bar"],
        "compressibility": 4.5e-5,
        "gen_vel": "no",
        "nstxout": 0,
        "nstvout": 0,
        "nstfout": 0,
        "nstcalcenergy": 100,
        "nstenergy": 5000,
        "nstlog": 5000,
        "nstxout-compressed": int(frame_plan.trajectory_interval_ps / dt_ps),
        "compressed-x-precision": 1000,
    }
    stage_values = {"em": em, "nvt": nvt, "npt": npt, "production": production}
    for stage_id, stage_overrides in overrides.items():
        stage_values[stage_id].update(stage_overrides)
    inputs = {
        "em": {
            "coordinates": {"kind": "prepared", "role": "md_coordinates"},
            "expected_suffixes": ["tpr", "log", "edr", "gro"],
            "checkpoint": False,
        },
        "nvt": {
            "coordinates": {"kind": "stage", "stage_id": "em", "artifact": "gro"},
            "expected_suffixes": ["tpr", "log", "edr", "gro", "cpt"],
            "checkpoint": True,
        },
        "npt": {
            "coordinates": {"kind": "stage", "stage_id": "nvt", "artifact": "gro"},
            "state_input": {"kind": "stage", "stage_id": "nvt", "artifact": "cpt"},
            "expected_suffixes": ["tpr", "log", "edr", "gro", "cpt"],
            "checkpoint": True,
        },
        "production": {
            "coordinates": {"kind": "stage", "stage_id": "npt", "artifact": "gro"},
            "state_input": {"kind": "stage", "stage_id": "npt", "artifact": "cpt"},
            "expected_suffixes": ["tpr", "log", "edr", "gro", "xtc", "cpt"],
            "checkpoint": True,
            "primary": True,
        },
    }
    return {
        "topology_role": "md_topology",
        "primary_stage": "production",
        "stages": [
            {
                "stage_id": stage_id,
                **inputs[stage_id],
                "mdp_text": _render_namelist(stage_values[stage_id]),
            }
            for stage_id in ("em", "nvt", "npt", "production")
        ],
    }


def _md_science_v2(
    preset: ProtocolPreset,
    settings: Mapping[str, object],
    md_engine_profile: str,
) -> Mapping[str, object]:
    """Build an engine-neutral, phase-oriented MD execution contract."""

    frame_plan = preset.frame_plan
    timestep_fs = _MD_TIMESTEP_FS
    # Minimizers stop when converged, so a low smoke-only ceiling creates
    # false failures on realistic solvated systems without making the gate
    # scientifically cheaper. Keep the production ceiling and retain the
    # independent maximum-force acceptance threshold below.
    em_iterations = 50_000
    nvt_steps = 500 if preset.smoke else 50_000
    npt_steps = 500 if preset.smoke else 100_000
    production_steps = frame_plan.production_ps * 1_000 // timestep_fs
    trajectory_interval_steps = frame_plan.trajectory_interval_ps * 1_000 // timestep_fs
    transaction_chunk_steps = min(500_000, production_steps)

    amber_input = md_engine_profile == "openmm_8_5@2"
    prepared_coordinates = {
        "kind": "prepared",
        "role": "solvated_coordinates" if amber_input else "md_coordinates",
    }

    def phase_artifact(phase_id: str, artifact: str) -> dict[str, str]:
        return {
            "kind": "phase",
            "phase_id": phase_id,
            "artifact": artifact,
        }

    phases: list[dict[str, object]] = [
        {
            "phase_id": "em",
            "kind": "em",
            "input": {"coordinates": prepared_coordinates},
            "minimization": {
                "maximum_force_threshold_kj_mol_nm": 1_000.0,
                "max_iterations": em_iterations,
            },
        },
        {
            "phase_id": "nvt",
            "kind": "nvt",
            "input": {
                "coordinates": phase_artifact("em", "final_coordinates"),
            },
            "duration_ps": nvt_steps * timestep_fs / 1_000,
            "steps": nvt_steps,
            "timestep_fs": timestep_fs,
            "temperature_kelvin": settings["temperature_k"],
            "initialize_velocities": True,
            "output": {
                "report_interval_steps": 500,
                "checkpoint_interval_steps": 500,
            },
        },
        {
            "phase_id": "npt",
            "kind": "npt",
            "input": {
                "coordinates": phase_artifact("nvt", "final_coordinates"),
                "state": phase_artifact("nvt", "portable_state"),
            },
            "duration_ps": npt_steps * timestep_fs / 1_000,
            "steps": npt_steps,
            "timestep_fs": timestep_fs,
            "temperature_kelvin": settings["temperature_k"],
            "pressure_bar": settings["pressure_bar"],
            "initialize_velocities": False,
            "output": {
                "report_interval_steps": 500,
                "checkpoint_interval_steps": 500,
            },
        },
        {
            "phase_id": "production",
            "kind": "production",
            "input": {
                "coordinates": phase_artifact("npt", "final_coordinates"),
                "state": phase_artifact("npt", "portable_state"),
            },
            "duration_ps": frame_plan.production_ps,
            "steps": production_steps,
            "timestep_fs": timestep_fs,
            "temperature_kelvin": settings["temperature_k"],
            "pressure_bar": settings["pressure_bar"],
            "initialize_velocities": False,
            "output": {
                "report_interval_steps": 5_000,
                "trajectory_interval_steps": trajectory_interval_steps,
                "checkpoint_interval_steps": transaction_chunk_steps,
                "transaction_chunk_steps": transaction_chunk_steps,
                "emit_initial_frame": True,
                "expected_trajectory_frames": frame_plan.stored_frames,
            },
        },
    ]
    return {
        "schema_version": MD_PHASE_PLAN_SCHEMA,
        "md_engine_profile": md_engine_profile,
        "topology_role": "solvated_topology" if amber_input else "md_topology",
        "primary_phase": "production",
        "replica_seed": "{{replica.seed}}",
        "seed_derivation": "sha256-phase-stream-v1",
        "system": {
            "nonbonded_method": "particle_mesh_ewald",
            "nonbonded_cutoff_nm": 1.0,
            "ewald_error_tolerance": 1.0e-5,
            "dispersion_correction": True,
            "constraints": "hydrogen_bonds",
            "rigid_water": True,
            "constraint_tolerance": 1.0e-5,
            "center_of_mass_removal_interval_steps": 100,
        },
        "phases": phases,
    }


def _render_namelist(values: Mapping[str, object]) -> str:
    return "".join(f"{key:<25} = {_engine_scalar(value)}\n" for key, value in values.items())


def _engine_scalar(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _qc_science(
    frame_plan: FramePlan,
    settings: Mapping[str, object],
    *,
    production_t0_reference: bool = False,
) -> Mapping[str, object]:
    contact_cutoff = settings["native_contact_cutoff_angstrom"]
    reference_artifact = (
        "prepared_reference_trajectory"
        if production_t0_reference
        else "prepared_trajectory"
    )
    contact_imaging = " noimage" if production_t0_reference else ""
    input_text = (
        "parm {{binding_topology:complex}}\n"
        f"trajin {{{{upstream:trajectory:prepared_trajectory}}}} "
        f"1 {frame_plan.expected_frames} 1\n"
        f"reference {{{{upstream:trajectory:{reference_artifact}}}}} 1\n"
        "rms interface_backbone {{selection:receptor_interface_backbone}} "
        "reference mass out interface_rmsd.dat\n"
        "rms ligand_primary {{selection:ligand_qc}} reference nofit mass "
        "out ligand_rmsd.dat\n"
        "nativecontacts name primary_binding {{selection:receptor_heavy}} "
        "{{selection:ligand_heavy}} "
        f"distance {contact_cutoff:g} reference{contact_imaging} out native_contacts.dat "
        "mindist maxdist\n"
        "run\n"
    )
    return {
        "expected_frames": frame_plan.expected_frames,
        "input_text": input_text,
        **(
            {
                "reference_artifact": reference_artifact,
                "reference_frame": 1,
                "reference_semantics": "post_npt_production_start",
            }
            if production_t0_reference
            else {}
        ),
        "output_tables": {
            "interface_rmsd": "interface_rmsd.dat",
            "ligand_rmsd": "ligand_rmsd.dat",
            "native_contacts": "native_contacts.dat",
        },
        "checks": [
            {
                "id": "interface_backbone_rmsd_p95_angstrom",
                "table": "interface_rmsd",
                "column": "interface_backbone",
                "reducer": "p95",
                "block_above": settings["interface_rmsd_p95_block_angstrom"],
            },
            {
                "id": "ligand_primary_rmsd_p95_angstrom",
                "table": "ligand_rmsd",
                "column": "ligand_primary",
                "reducer": "p95",
                "warn_above": settings["ligand_rmsd_p95_warn_angstrom"],
            },
            {
                "id": "native_contacts_min",
                "table": "native_contacts",
                "column": "primary_binding[native]",
                "reducer": "min",
                "block_below": 1.0,
            },
            {
                "id": "interface_distance_max_angstrom",
                "table": "native_contacts",
                "column": "primary_binding[mindist]",
                "reducer": "max",
                "block_above": settings["interface_distance_block_angstrom"],
            },
        ],
    }


def _scoring_science(
    model: str,
    frame_plan: FramePlan,
    settings: Mapping[str, object],
    overrides: Mapping[str, object],
) -> Mapping[str, object]:
    internal = settings["internal_dielectric"]
    internal_text = (
        AUTO_DIELECTRIC_TOKEN if internal == "auto" else _engine_scalar(internal)
    )
    frames = frame_plan.expected_frames
    common_control = (
        f"Generated {model.upper()} input for {frames} aligned frames\n"
        "&general\n"
        f"  startframe=1, endframe={frames}, interval=1,\n"
        "  verbose=1, keep_files=0,\n"
        "  netcdf=1, use_sander=1,\n"
        "/\n"
    )
    if model == "gb":
        surften = overrides.get("surften", 0.0072)
        control = (
            common_control
            + "&gb\n"
            f"  igb=5, saltcon={settings['salt_concentration_molar']:.3f},\n"
            "/\n"
        )
        native = (
            "Generated Sander GB input\n"
            "&cntrl\n"
            "  ntb=0, nsnb=99999, cut=999.0,\n"
            "  ioutfm=1, imin=5, ncyc=0, igb=5,\n"
            f"  saltcon={settings['salt_concentration_molar']:.3f}, "
            f"surften={float(surften):g},\n"
            f"  intdiel={internal_text}, "
            f"extdiel={settings['gb_external_dielectric']:g}, dec_verbose=0,\n"
            "/\n"
        )
        native_inputs = {"main": native}
    else:
        fillratio = float(overrides.get("fillratio", 4.0))
        maxitn = int(overrides.get("maxitn", 1000))
        dprob = float(overrides.get("dprob", 1.4))
        ionic_strength_mm = float(settings["salt_concentration_molar"]) * 1000.0
        control = (
            common_control
            + "&pb\n"
            f"  istrng={settings['salt_concentration_molar']:.3f}, radiopt=0,\n"
            f"  indi={internal_text}, exdi={settings['pb_external_dielectric']:g},\n"
            f"  inp=2, prbrad={dprob:g}, fillratio={fillratio:g},\n"
            "/\n"
        )
        native = (
            "Generated Sander PB input\n"
            "&cntrl\n"
            "  ntb=0, nsnb=99999, cut=999.0,\n"
            "  ioutfm=1, imin=5, igb=10, ipb=2,\n"
            "  inp=2, dec_verbose=0,\n"
            "/\n"
            "&pb\n"
            f"  epsin={internal_text}, epsout={settings['pb_external_dielectric']:g}, "
            f"istrng={ionic_strength_mm:g},\n"
            f"  radiopt=0, dprob={dprob:g}, solvopt=2,\n"
            f"  maxitn={maxitn}, fillratio={fillratio:g},\n"
            "  mthick=-9999.0, mctrdz=-9999.0,\n"
            "/\n"
        )
        native_inputs = {"complex_receptor": native, "ligand": native}
    return {
        "expected_frames": frames,
        "frame_manifest_digest": frame_plan.manifest_digest,
        "trajectory_artifact": "prepared_trajectory",
        "dielectric_policy": {
            "mode": "automatic" if internal == "auto" else "fixed",
            "fallback": 4.0,
            "fixed_value": None if internal == "auto" else internal,
        },
        "control_text": control,
        "native_inputs": native_inputs,
    }


def _analysis_science(models: tuple[str, ...]) -> Mapping[str, object]:
    analysis_models: dict[str, object] = {}
    for model in models:
        common_transform = {
            "scale": 4.184,
            "offset": 0.0,
            "input_unit": "kcal/mol",
            "output_unit": "kJ/mol",
        }
        analysis_models[model] = {
            "source_model_id": model,
            "evaluator": "direct",
            "score_selector": {
                "component": "delta",
                "label": "DELTA TOTAL",
                "value_key": "average",
            },
            "score_transform": dict(common_transform),
        }
        analysis_models[f"{model}_dmm"] = {
            "source_model_id": model,
            "evaluator": "v1_dmm_v1",
            "score_selector": {
                "component": "delta",
                "label": "DMM",
                "value_key": "average",
            },
            "score_transform": dict(common_transform),
        }
    return {
        "reference_key": "experimental_dg_kj_mol",
        "aggregation": "replica_mean",
        "reference_mode": "quantitative_comparison",
        "models": analysis_models,
    }


def _report_defaults(
    preset: ProtocolPreset,
    models: tuple[str, ...],
    settings: Mapping[str, object],
) -> Mapping[str, object]:
    columns: list[dict[str, object]] = []
    analysis_models = tuple(
        model_id
        for model in models
        for model_id in (model, f"{model}_dmm")
    )
    for model in analysis_models:
        label = model.upper()
        columns.extend(
            (
                {
                    "key": f"{model}_mean",
                    "label": f"{label} mean",
                    "unit": "kJ/mol",
                    "precision": 2,
                },
                {
                    "key": f"{model}_stddev",
                    "label": f"{label} replica SD",
                    "unit": "kJ/mol",
                    "precision": 2,
                },
                {
                    "key": f"{model}_sem",
                    "label": f"{label} replica SEM",
                    "unit": "kJ/mol",
                    "precision": 2,
                },
            )
        )
    frame_plan = preset.frame_plan
    return {
        "title": f"MMPBSA {preset.preset_id}",
        "reference_label": "Experimental ΔG (kJ/mol)",
        "rank_by": f"{analysis_models[0]}_mean",
        "score_columns": columns,
        "metadata": {
            "Replicas": preset.replicas,
            "Production per replica": f"{frame_plan.production_ps / 1000:g} ns",
            "Frames per model and replica": frame_plan.expected_frames,
        },
        "protocol": {
            "Preset": preset.preset_id,
            "Temperature": f"{settings['temperature_k']:g} K",
            "Salt concentration": f"{settings['salt_concentration_molar']:g} M",
            "Scoring models": ", ".join(model.upper() for model in analysis_models),
            "Analysis window": (
                f"{frame_plan.analysis_start_ps / 1000:g}–"
                f"{frame_plan.analysis_end_ps / 1000:g} ns; "
                f"{frame_plan.scoring_interval_ps} ps stride"
            ),
        },
        "model_order": list(analysis_models),
    }


def resolve_internal_dielectric(
    *,
    ligand_formal_charge: int | None,
    interface_classes: Sequence[str] | None,
) -> tuple[float, tuple[str, ...]]:
    """Apply the v1 2/3/4 dielectric policy to explicit entity evidence.

    ``interface_classes`` contains normalized residue chemistry labels from
    both members of the primary binding for peptide ligands, and from the
    receptor member for small molecules.  Missing or unknown evidence is
    intentionally conservative: emit a warning and use epsilon-in 4.
    """

    warnings: list[str] = []
    if ligand_formal_charge is None or interface_classes is None:
        return 4.0, ("dielectric evidence is incomplete; using epsilon-in 4",)
    if ligand_formal_charge != 0:
        return 4.0, ()
    normalized = tuple(str(item).strip().lower() for item in interface_classes)
    known = {"charged", "polar", "nonpolar"}
    unknown = tuple(item for item in normalized if item not in known)
    if unknown or not normalized:
        warnings.append(
            "dielectric interface chemistry is unknown; using epsilon-in 4"
        )
        return 4.0, tuple(warnings)
    if "charged" in normalized:
        return 4.0, ()
    if "polar" in normalized:
        return 3.0, ()
    return 2.0, ()


__all__ = [
    "AUTO_DIELECTRIC_TOKEN",
    "DEFAULT_MD_ENGINE_PROFILE",
    "FramePlan",
    "MD_PHASE_PLAN_SCHEMA",
    "PROTOCOL_PRESET_SCHEMA",
    "ProtocolPreset",
    "ResolvedProtocol",
    "available_protocol_presets",
    "protocol_preset",
    "resolve_internal_dielectric",
    "resolve_protocol",
]

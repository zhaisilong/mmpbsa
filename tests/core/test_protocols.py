from __future__ import annotations

import hashlib
import json

import pytest

from mmpbsa.core.errors import ConfigurationError
from mmpbsa.core.protocols import (
    AUTO_DIELECTRIC_TOKEN,
    DEFAULT_MD_ENGINE_PROFILE,
    MD_PHASE_PLAN_SCHEMA,
    available_protocol_presets,
    resolve_internal_dielectric,
    resolve_protocol,
)


@pytest.mark.parametrize(
    (
        "preset_id",
        "production_ps",
        "start_frame",
        "end_frame",
        "stride",
        "frames",
    ),
    (
        ("production_3x5ns@1", 5_000, 151, 251, 1, 101),
        ("production_3x15ns@1", 15_000, 251, 751, 1, 501),
        ("production_3x50ns@1", 50_000, 501, 2501, 2, 1001),
        ("production_3x100ns@1", 100_000, 1001, 5001, 2, 2001),
        ("production_3x5ns@2", 5_000, 151, 251, 1, 101),
        ("production_3x15ns@2", 15_000, 251, 751, 1, 501),
        ("production_3x50ns@2", 50_000, 501, 2501, 2, 1001),
        ("production_3x100ns@2", 100_000, 1001, 5001, 2, 2001),
    ),
)
def test_public_presets_have_aligned_inclusive_frame_manifests(
    preset_id: str,
    production_ps: int,
    start_frame: int,
    end_frame: int,
    stride: int,
    frames: int,
) -> None:
    protocol = resolve_protocol(preset_id)
    frame_plan = protocol.preset.frame_plan

    assert frame_plan.production_ps == production_ps
    assert frame_plan.stored_frames == production_ps // 20 + 1
    assert frame_plan.start_frame == start_frame
    assert frame_plan.end_frame == end_frame
    assert frame_plan.trajectory_stride == stride
    assert frame_plan.expected_frames == frames
    assert protocol.stages["trajectory"]["expected_frames"] == frames
    assert protocol.stages["trajectory_qc"]["expected_frames"] == frames
    assert {
        model["expected_frames"]
        for model in protocol.stages["scoring"]["models"].values()
    } == {frames}


def test_both_versioned_production_protocol_families_are_public() -> None:
    assert available_protocol_presets() == (
        "production_3x5ns@1",
        "production_3x15ns@1",
        "production_3x50ns@1",
        "production_3x100ns@1",
        "production_3x5ns@2",
        "production_3x15ns@2",
        "production_3x50ns@2",
        "production_3x100ns@2",
    )


def test_nonproduction_smoke_preset_is_explicit_and_fast() -> None:
    protocol = resolve_protocol("smoke_20ps@1")
    stages = protocol.stages["md"]["stages"]

    assert protocol.preset.smoke is True
    assert protocol.preset.frame_plan.expected_frames == 11
    assert [
        next(
            line
            for line in stage["mdp_text"].splitlines()
            if line.startswith("nsteps")
        ).split("=", 1)[1].strip()
        for stage in stages
    ] == ["200", "500", "500", "10000"]


@pytest.mark.parametrize(
    ("preset_id", "expected_digest"),
    (
        (
            "production_3x5ns@1",
            "f43dc3d6bf9e0dab7e41b8a47e77bedca90534cb13f0846f245c6aac91d75f38",
        ),
        (
            "production_3x15ns@1",
            "8fe06e880f52143b01e3722a4799cba017b906ac69ec8a3b58b9094cbad09eec",
        ),
        (
            "production_3x50ns@1",
            "26b4f975922dfe2f74f4c434525aa3c3b7124ec5096efd229bb1713fe4168399",
        ),
        (
            "production_3x100ns@1",
            "f434fb9d87b17c5164fad4a4d18e1ca3f62a5287d65984bbab5e260b5cbcc954",
        ),
        (
            "smoke_20ps@1",
            "5ee00a11ab88758e508f07c51bed2591697c985b6505727d6939110cffaf2f77",
        ),
    ),
)
def test_v1_md_stage_contract_is_byte_stable(
    preset_id: str,
    expected_digest: str,
) -> None:
    md_science = resolve_protocol(preset_id).stages["md"]
    encoded = json.dumps(
        md_science,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert hashlib.sha256(encoded).hexdigest() == expected_digest


@pytest.mark.parametrize(
    ("preset_id", "production_steps", "trajectory_interval_steps", "frames"),
    (
        ("production_3x5ns@2", 2_500_000, 10_000, 251),
        ("production_3x15ns@2", 7_500_000, 10_000, 751),
        ("production_3x50ns@2", 25_000_000, 10_000, 2_501),
        ("production_3x100ns@2", 50_000_000, 10_000, 5_001),
        ("smoke_20ps@2", 10_000, 1_000, 11),
    ),
)
def test_v2_md_phase_plan_is_engine_neutral_and_frame_aligned(
    preset_id: str,
    production_steps: int,
    trajectory_interval_steps: int,
    frames: int,
) -> None:
    protocol = resolve_protocol(
        preset_id,
        md_engine_profile="openmm_8_5@2",
    )
    md_science = protocol.stages["md"]
    phases = md_science["phases"]
    production = phases[-1]

    assert md_science["schema_version"] == MD_PHASE_PLAN_SCHEMA
    assert md_science["md_engine_profile"] == "openmm_8_5@2"
    assert [phase["phase_id"] for phase in phases] == [
        "em",
        "nvt",
        "npt",
        "production",
    ]
    assert production["steps"] == production_steps
    assert production["duration_ps"] * 1_000 == (
        production["steps"] * production["timestep_fs"]
    )
    assert production["output"]["trajectory_interval_steps"] == (
        trajectory_interval_steps
    )
    assert production["output"]["expected_trajectory_frames"] == frames
    assert production["output"]["emit_initial_frame"] is True

    serialized = json.dumps(md_science, sort_keys=True).lower()
    for forbidden in ('"mdp_text"', '"expected_suffixes"', '"tpr"', '"cpt"', '"gro"', '"xtc"'):
        assert forbidden not in serialized


def test_v2_md_phase_durations_and_inputs_are_explicit() -> None:
    md_science = resolve_protocol("production_3x5ns@2").stages["md"]
    em, nvt, npt, production = md_science["phases"]

    assert em["input"]["coordinates"] == {
        "kind": "prepared",
        "role": "md_coordinates",
    }
    assert set(em["minimization"]) == {
        "maximum_force_threshold_kj_mol_nm",
        "max_iterations",
    }
    assert em["minimization"]["maximum_force_threshold_kj_mol_nm"] == 1_000.0
    assert em["minimization"]["max_iterations"] == 50_000
    assert (nvt["duration_ps"], nvt["steps"]) == (100.0, 50_000)
    assert (npt["duration_ps"], npt["steps"]) == (200.0, 100_000)
    assert nvt["initialize_velocities"] is True
    assert npt["input"]["state"]["artifact"] == "portable_state"
    assert production["input"]["state"]["artifact"] == "portable_state"
    assert production["output"]["transaction_chunk_steps"] == 500_000


def test_v2_trajectory_contract_is_engine_neutral() -> None:
    trajectory = resolve_protocol("production_3x5ns@2").stages["trajectory"]

    assert trajectory["adapter_contract"] == "mmpbsa.trajectory.v3"
    assert trajectory["source_trajectory_artifact"] == "full_trajectory"
    assert trajectory["full_system_topology_role"] == "md_topology"
    assert trajectory["full_system_coordinates_role"] == "md_coordinates"
    assert trajectory["expected_frames"] == 101
    assert trajectory["source_trajectory_contract"] == {
        "first_time_ps": 0.0,
        "interval_ps": 20,
        "interval_steps": 10_000,
        "expected_frames": 251,
    }
    assert {
        "source_tpr_artifact",
        "index_role",
        "group_name",
    }.isdisjoint(trajectory)


def test_smoke_v2_retains_short_equilibration_and_single_chunk() -> None:
    protocol = resolve_protocol(
        "smoke_20ps@2",
        md_engine_profile="openmm_8_5@2",
    )
    em, nvt, npt, production = protocol.stages["md"]["phases"]

    assert em["minimization"]["max_iterations"] == 50_000
    assert [nvt["steps"], npt["steps"], production["steps"]] == [
        500,
        500,
        10_000,
    ]
    assert production["output"]["transaction_chunk_steps"] == 10_000


def test_md_engine_profile_is_explicit_in_resolved_identity() -> None:
    protocol = resolve_protocol("production_3x5ns@2")

    assert protocol.md_engine_profile == DEFAULT_MD_ENGINE_PROFILE
    assert protocol.as_dict()["md_engine_profile"] == DEFAULT_MD_ENGINE_PROFILE
    assert protocol.stages["md"]["md_engine_profile"] == DEFAULT_MD_ENGINE_PROFILE


@pytest.mark.parametrize(
    ("preset_id", "profile", "message"),
    (
        (
            "production_3x5ns@1",
            "openmm_8_5@1",
            "requires md_engine_profile",
        ),
        (
            "production_3x5ns@2",
            "openmm_8_5@1",
            "archived read-only",
        ),
        ("production_3x5ns@2", "openmm@latest", "unknown"),
    ),
)
def test_md_engine_profile_compatibility_is_strict(
    preset_id: str,
    profile: str,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        resolve_protocol(preset_id, md_engine_profile=profile)


def test_v2_openmm_amber_inputs_and_qc_reference_production_start() -> None:
    protocol = resolve_protocol(
        "production_3x5ns@2", md_engine_profile="openmm_8_5@2"
    )
    md = protocol.stages["md"]
    qc = protocol.stages["trajectory_qc"]
    assert md["topology_role"] == "solvated_topology"
    assert md["phases"][0]["input"]["coordinates"] == {
        "kind": "prepared",
        "role": "solvated_coordinates",
    }
    assert qc["reference_artifact"] == "prepared_reference_trajectory"
    assert qc["reference_frame"] == 1
    assert qc["reference_semantics"] == "post_npt_production_start"
    assert "trajin {{upstream:trajectory:prepared_trajectory}}" in qc["input_text"]
    assert (
        "reference {{upstream:trajectory:prepared_reference_trajectory}} 1"
        in qc["input_text"]
    )


def test_v1_qc_keeps_analysis_window_first_frame_reference() -> None:
    qc = resolve_protocol("production_3x5ns@1").stages["trajectory_qc"]
    assert "reference_artifact" not in qc
    assert "reference {{upstream:trajectory:prepared_trajectory}} 1" in qc["input_text"]


def test_v2_rejects_native_gromacs_science_overrides() -> None:
    with pytest.raises(ConfigurationError, match="engine-neutral"):
        resolve_protocol(
            "production_3x5ns@2",
            engine_overrides={"gromacs": {"production": {"tau_p": 4.0}}},
        )


@pytest.mark.parametrize("models", (("gb",), ("pb",), ("gb", "pb")))
def test_gb_and_pb_are_independently_selectable(models: tuple[str, ...]) -> None:
    protocol = resolve_protocol("production_3x5ns@1", models=models)

    assert protocol.models == models
    assert tuple(protocol.stages["scoring"]["models"]) == models
    expected_analysis = tuple(
        model_id
        for model in models
        for model_id in (model, f"{model}_dmm")
    )
    assert tuple(protocol.analysis["models"]) == expected_analysis
    assert tuple(protocol.report_defaults["model_order"]) == expected_analysis
    assert protocol.stages["scoring"]["decomposition"]["enabled"] is False
    assert protocol.report_defaults["model_order"] == list(expected_analysis)


def test_default_scoring_requires_both_models_and_uses_paired_frames() -> None:
    protocol = resolve_protocol("production_3x100ns@1")
    scoring = protocol.stages["scoring"]

    assert scoring["completion_policy"] == "all"
    assert tuple(scoring["models"]) == ("gb", "pb")
    assert scoring["models"]["gb"]["expected_frames"] == 2001
    assert scoring["models"]["pb"]["expected_frames"] == 2001
    assert scoring["models"]["gb"]["frame_manifest_digest"] == (
        scoring["models"]["pb"]["frame_manifest_digest"]
    )
    assert scoring["models"]["gb"]["frame_manifest_digest"] == (
        protocol.stages["trajectory"]["frame_manifest_digest"]
    )
    assert "endframe=2001, interval=1" in scoring["models"]["gb"]["control_text"]
    assert "endframe=2001, interval=1" in scoring["models"]["pb"]["control_text"]


def test_native_inputs_are_generated_with_correct_salt_units() -> None:
    protocol = resolve_protocol("production_3x5ns@1")
    scoring = protocol.stages["scoring"]["models"]

    assert "saltcon=0.150" in scoring["gb"]["control_text"]
    assert "istrng=0.150" in scoring["pb"]["control_text"]
    assert "istrng=150" in scoring["pb"]["native_inputs"]["ligand"]
    assert AUTO_DIELECTRIC_TOKEN in scoring["gb"]["native_inputs"]["main"]
    assert AUTO_DIELECTRIC_TOKEN in scoring["pb"]["control_text"]


def test_typed_and_engine_overrides_are_allowlisted_and_canonical() -> None:
    protocol = resolve_protocol(
        "production_3x15ns@1",
        models=("gb",),
        overrides={
            "temperature_k": 310,
            "salt_concentration_molar": 0.1,
            "internal_dielectric": 3,
        },
        engine_overrides={
            "gromacs": {"production": {"tau_p": 4.0}},
            "amber": {"gb": {"surften": 0.008}},
        },
    )
    production = protocol.stages["md"]["stages"][-1]["mdp_text"]
    gb = protocol.stages["scoring"]["models"]["gb"]

    assert "ref_t                     = 310" in production
    assert "tau_p                     = 4" in production
    assert "saltcon=0.100" in gb["control_text"]
    assert "surften=0.008" in gb["native_inputs"]["main"]
    assert "intdiel=3" in gb["native_inputs"]["main"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"models": ("sa",)}, "unknown models"),
        ({"completion_policy": "first"}, "must be 'all' or 'any'"),
        ({"overrides": {"production_ns": 20}}, "unknown fields"),
        (
            {"engine_overrides": {"gromacs": {"production": {"nsteps": 10}}}},
            "unknown fields: nsteps",
        ),
        (
            {"engine_overrides": {"amber": {"pb": {"raw_mdin": "unsafe"}}}},
            "unknown fields: raw_mdin",
        ),
        (
            {"engine_overrides": {"amber": {"pb": {"maxitn": 2.5}}}},
            "must be a positive integer",
        ),
    ),
)
def test_protocol_rejects_unversioned_or_raw_science_knobs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        resolve_protocol("production_3x5ns@1", **kwargs)


@pytest.mark.parametrize(
    ("charge", "classes", "expected", "warning"),
    (
        (1, ("nonpolar",), 4.0, False),
        (0, ("charged", "polar"), 4.0, False),
        (0, ("polar", "nonpolar"), 3.0, False),
        (0, ("nonpolar",), 2.0, False),
        (None, None, 4.0, True),
        (0, ("unknown",), 4.0, True),
    ),
)
def test_automatic_dielectric_policy_is_conservative(
    charge: int | None,
    classes: tuple[str, ...] | None,
    expected: float,
    warning: bool,
) -> None:
    value, warnings = resolve_internal_dielectric(
        ligand_formal_charge=charge,
        interface_classes=classes,
    )

    assert value == expected
    assert bool(warnings) is warning

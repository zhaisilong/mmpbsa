from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from mmpbsa.backends.errors import BackendConfigurationError, BackendOutputError
from mmpbsa.backends.gromacs_phase import (
    gromacs_operational_mapping,
    render_gromacs_phase_plan,
    validate_gromacs_phase_output,
)
from mmpbsa.core import resolve_protocol


def _science() -> dict[str, object]:
    science = dict(resolve_protocol("smoke_20ps@2").stages["md"])
    science["_replica"] = {
        "replica_id": "rep01",
        "replica_index": 1,
        "seed": 4101,
    }
    return science


def _stage(rendered: object, phase_id: str) -> dict[str, object]:
    stages = rendered.pipeline_science["stages"]
    return next(dict(item) for item in stages if item["stage_id"] == phase_id)


def test_canonical_phase_plan_lowers_to_explicit_gromacs_mdp() -> None:
    rendered = render_gromacs_phase_plan(_science())
    em = _stage(rendered, "em")
    production = _stage(rendered, "production")

    assert "integrator                = steep" in em["mdp_text"]
    assert "emtol                     = 1000" in em["mdp_text"]
    assert "define                    = -DFLEXIBLE" in em["mdp_text"]
    assert "ewald-rtol                = 1e-05" in production["mdp_text"]
    assert "DispCorr                  = EnerPres" in production["mdp_text"]
    assert "constraints               = h-bonds" in production["mdp_text"]
    assert "nstcomm                   = 100" in production["mdp_text"]
    assert "tinit                     = 0" in production["mdp_text"]
    assert "init-step                 = 0" in production["mdp_text"]
    assert "nstxout-compressed        = 1000" in production["mdp_text"]
    assert "ld-seed" in production["mdp_text"]
    assert production["state_input"] == {
        "kind": "stage",
        "stage_id": "npt",
        "artifact": "cpt",
    }
    assert rendered.audit["expected_trajectory_frames"] == 11
    assert rendered.audit["emit_initial_frame"] is True
    assert rendered.audit["state_format"] == "gromacs_checkpoint"
    assert (
        rendered.audit["constraint_mapping"]["status"]
        == "profile_approximation"
    )
    assert rendered.audit["constraint_mapping"]["direct_numeric_mapping"] is False
    assert rendered.audit["minimization_mapping"] == {
        "native_algorithm": "steepest_descent",
        "native_force_threshold_kj_mol_nm": 1_000.0,
        "constraints": "none",
        "rigid_water": False,
        "topology_define": "-DFLEXIBLE",
    }
    assert rendered.audit["random_seeds"]["nvt"]["velocities"] > 0


def test_phase_plan_rejects_wrong_profile_or_ambiguous_extension() -> None:
    wrong_profile = _science()
    wrong_profile["md_engine_profile"] = "openmm_8_5@1"
    with pytest.raises(BackendConfigurationError, match="gromacs_canonical"):
        render_gromacs_phase_plan(wrong_profile)

    extension = _science()
    extension["native_mdp"] = {"nstxout-compressed": 1}
    with pytest.raises(BackendConfigurationError, match="unknown native_mdp"):
        render_gromacs_phase_plan(extension)


def test_checkpoint_and_transaction_fields_are_audited_as_operational_hints() -> None:
    default = gromacs_operational_mapping(_science(), ())
    checkpoint = default["checkpoint"]

    assert default["classification"] == "operational_resilience_not_ensemble_science"
    assert checkpoint["requested_interval_steps_by_phase"] == {
        "nvt": 500,
        "npt": 500,
        "production": 10_000,
    }
    assert checkpoint["mapping_status"] == "profile_approximation"
    assert checkpoint["native_control"] == "gmx mdrun -cpt"
    assert checkpoint["native_interval_minutes"] == 15.0
    assert checkpoint["exact_step_schedule"] is False
    assert checkpoint["phase_end_checkpoint_guaranteed"] is True
    assert "no nstcheckpoint" in checkpoint["reason"]
    assert default["transaction_chunk"] == {
        "requested_steps": 10_000,
        "mapping_status": "openmm_journal_hint",
        "applied_by_gromacs": False,
        "gromacs_strategy": "single mdrun with native checkpoint append",
    }

    explicit = gromacs_operational_mapping(
        _science(), ("-cpt", "2.5")
    )
    assert explicit["checkpoint"]["native_interval_minutes"] == 2.5
    assert explicit["checkpoint"]["native_control_source"] == "runtime.mdrun_arguments"


def test_em_gate_uses_optimizer_target_and_separate_force_acceptance(tmp_path: Path) -> None:
    rendered = render_gromacs_phase_plan(_science())
    assert "emtol                     = 1000" in _stage(rendered, "em")["mdp_text"]
    log = tmp_path / "em.log"
    log.write_text("Maximum force     =  9.5000000e+02 on atom 42\n", encoding="utf-8")

    audit = validate_gromacs_phase_output(_science(), "em", {"log": log})

    assert audit == {
        "maximum_force_kj_mol_nm": 950.0,
        "acceptance_threshold_kj_mol_nm": 1000.0,
    }
    log.write_text("Maximum force = 1.001e+03 on atom 42\n", encoding="utf-8")
    with pytest.raises(BackendOutputError, match="exceeds acceptance"):
        validate_gromacs_phase_output(_science(), "em", {"log": log})


def test_production_gate_requires_t0_and_exact_declared_interval(tmp_path: Path) -> None:
    trajectory = tmp_path / "production.xtc"
    trajectory.write_bytes(b"parsable fake XTC\n")

    class Coordinates:
        def __init__(self, frames: int) -> None:
            self.shape = (frames, 1, 3)

    class FakeXTC:
        times = tuple(float(index * 2) for index in range(11))

        def __init__(self, _path: str, _mode: str) -> None:
            self.finished = False

        def __enter__(self) -> FakeXTC:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, *, n_frames: int) -> tuple[object, ...]:
            assert n_frames == 100
            if self.finished:
                return Coordinates(0), (), (), ()
            self.finished = True
            return (
                Coordinates(11),
                self.times,
                tuple(index * 1000 for index in range(11)),
                (None,) * 11,
            )

    mdtraj = types.ModuleType("mdtraj")
    formats = types.ModuleType("mdtraj.formats")
    formats.XTCTrajectoryFile = FakeXTC
    mdtraj.formats = formats
    with patch.dict(sys.modules, {"mdtraj": mdtraj, "mdtraj.formats": formats}):
        audit = validate_gromacs_phase_output(
            _science(), "production", {"xtc": trajectory}
        )

    assert audit["frame_count"] == 11
    assert audit["first_time_ps"] == 0.0
    assert audit["last_time_ps"] == 20.0
    FakeXTC.times = tuple(float(index * 2 + 1) for index in range(11))
    with patch.dict(sys.modules, {"mdtraj": mdtraj, "mdtraj.formats": formats}):
        with pytest.raises(BackendOutputError, match="declared t=0 timeline"):
            validate_gromacs_phase_output(
                _science(), "production", {"xtc": trajectory}
            )

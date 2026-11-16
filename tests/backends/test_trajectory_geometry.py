from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mmpbsa.backends.errors import BackendOutputError
from mmpbsa.backends.trajectory_geometry import (
    chemical_topology,
    reconstruct,
    validate,
    validate_rigid_geometry,
)

pytestmark = pytest.mark.skipif(
    not importlib.util.find_spec("mdtraj") or not importlib.util.find_spec("parmed"),
    reason="MDTraj and ParmEd required",
)


def _topology(path: Path) -> None:
    # Non-contiguous HEM-His connection, peptide backbone, and a cyclic ligand.
    atoms = [
        ("N", "HIS", 7),
        ("CA", "HIS", 6),
        ("C", "HIS", 6),
        ("C1", "LIG", 6),
        ("C2", "LIG", 6),
        ("C3", "LIG", 6),
        ("FE", "HEM", 26),
    ]
    import parmed as pmd

    s = pmd.Structure()
    for index, (name, residue, element) in enumerate(atoms):
        atom = pmd.Atom(name=name, type=f"T{index}", atomic_number=element, mass=12.0, charge=0.0)
        atom_type = pmd.topologyobjects.AtomType(atom.type, index + 1, 12.0, element)
        atom_type.set_lj_params(0.1, 1.5)
        atom.atom_type = atom_type
        s.add_atom(atom, residue, 1 if residue == "HIS" else 2 if residue == "LIG" else 3)
    bond_type = pmd.BondType(200.0, 1.5)
    s.bond_types.append(bond_type)
    for a, b in ((0, 1), (1, 2), (0, 6), (3, 4), (4, 5), (5, 3)):
        s.bonds.append(pmd.Bond(s.atoms[a], s.atoms[b], type=bond_type))
    s.save(str(path), format="amber", overwrite=True)


@pytest.mark.parametrize("triclinic", [False, True])
def test_streaming_whole_molecules_preserves_bonds_frame_identity_and_cycles(tmp_path, triclinic):
    import mdtraj as md
    import numpy as np
    from mdtraj.formats import XTCTrajectoryFile

    topology = tmp_path / "complex.prmtop"
    _topology(topology)
    box = np.array(
        [
            [3.0, 0, 0],
            [0.7 if triclinic else 0, 3.1, 0],
            [0.3 if triclinic else 0, 0.5 if triclinic else 0, 3.2],
        ],
        dtype=np.float32,
    )
    xyz = np.array(
        [
            [
                [0.05, 0.05, 0.05],
                [0.19, 0.05, 0.05],
                [0.33, 0.05, 0.05],
                [0.5, 0.5, 0.5],
                [0.64, 0.5, 0.5],
                [0.57, 0.62, 0.5],
                [0.05, 0.05, 0.27],
            ]
        ],
        dtype=np.float32,
    )
    xyz = np.repeat(xyz, 5, axis=0)
    xyz[1:, 1] += box[0]
    xyz[2:, 2] -= box[1]
    xyz[3:, 6] += box[2]
    xyz[4:, 4] += box[0] - box[1]
    times = np.arange(5, dtype=np.float32) * 20
    steps = np.arange(5, dtype=np.int32) * 10000
    boxes = np.repeat(box[None], 5, axis=0)
    raw, whole = tmp_path / "raw.xtc", tmp_path / "whole.xtc"
    with XTCTrajectoryFile(str(raw), "w") as stream:
        stream.write(xyz, time=times, step=steps, box=boxes)
    with pytest.raises(BackendOutputError, match="periodic bond integrity.*frame 2.*atoms"):
        validate(raw, topology, chunk_frames=2)
    audit = reconstruct(raw, whole, topology, chunk_frames=2)
    assert audit["frame_count"] == 5 and audit["bond_count"] == 6
    assert audit["raw_maximum_bond_angstrom"] > 30
    assert audit["maximum_bond_vector_error_angstrom"] < 0.05
    valid = validate(whole, topology, chunk_frames=2)
    assert valid["maximum_bond_angstrom"] < 2.3
    with XTCTrajectoryFile(str(whole), "r") as stream:
        _, observed_times, observed_steps, observed_boxes = stream.read()
    np.testing.assert_array_equal(observed_times, times)
    np.testing.assert_array_equal(observed_steps, steps)
    np.testing.assert_allclose(observed_boxes, boxes, atol=1e-6)

    corrected = md.load_xtc(str(whole), top=chemical_topology(topology))
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    fitted = tmp_path / "fitted.xtc"
    with XTCTrajectoryFile(str(fitted), "w") as stream:
        stream.write(corrected.xyz @ rotation, time=times, step=steps, box=boxes @ rotation)
    assert validate_rigid_geometry(whole, fitted, topology, chunk_frames=2)["status"] == "passed"


def test_fitted_geometry_cannot_hide_a_broken_bond(tmp_path):
    import numpy as np
    from mdtraj.formats import XTCTrajectoryFile

    topology = tmp_path / "complex.prmtop"
    _topology(topology)
    xyz = np.zeros((1, 7, 3), dtype=np.float32)
    box = np.eye(3, dtype=np.float32)[None] * 3
    physical, broken = tmp_path / "physical.xtc", tmp_path / "broken.xtc"
    with XTCTrajectoryFile(str(physical), "w") as stream:
        stream.write(xyz, time=np.array([0.0]), step=np.array([0]), box=box)
    xyz[:, 6, 0] = 3
    with XTCTrajectoryFile(str(broken), "w") as stream:
        stream.write(xyz, time=np.array([0.0]), step=np.array([0]), box=box)
    with pytest.raises(BackendOutputError, match="fitted geometry changed.*atoms"):
        validate_rigid_geometry(physical, broken, topology, chunk_frames=1)


def test_unverified_old_v3_trajectory_is_not_admitted_to_scoring():
    from mmpbsa.backends.errors import BackendConfigurationError
    from mmpbsa.backends.services import _require_trajectory_geometry
    from mmpbsa.core import StageId, StageInput, TaskKey

    old = StageInput(
        TaskKey(StageId.TRAJECTORY, case_id="test", replica_id="rep01"),
        "0001",
        Path("/tmp"),
        artifacts={},
        payload={"backend_metadata": {"contract": "mmpbsa.trajectory.v3"}},
    )
    with pytest.raises(BackendConfigurationError, match="regenerate trajectory"):
        _require_trajectory_geometry(old)


def test_display_reference_reuses_physical_t0_without_second_imaging(tmp_path):
    import json

    from mmpbsa.backends.display import (
        DisplayAssetRequest,
        _render_manifest,
        _render_reference_cpptraj,
    )

    request = DisplayAssetRequest(
        complex_topology=tmp_path / "complex.prmtop",
        trajectories={"rep01": tmp_path / "centered.xtc"},
        component_atom_indices={"ligand": (1,), "receptor": (2,)},
        output_dir=tmp_path,
        reference_trajectories={"rep01": tmp_path / "prepared_reference.nc"},
        reference_autoimage_anchor_atom_indices=(1,),
        reference_autoimage_fixed_atom_indices=(2,),
        reference_alignment_atom_indices=(2,),
        ligand_rmsd_atom_indices=(1,),
        reference_artifact="prepared_reference_trajectory",
    )
    script = _render_reference_cpptraj(request, "rep01", tmp_path / "reference.pdb")
    assert "prepared_reference.nc" in script and "autoimage" not in script
    provenance = json.loads(_render_manifest(request))["equilibrated_pose_rmsd"]
    assert provenance["reference_artifact"] == "prepared_reference_trajectory"
    assert provenance["reference_preprocessing"].startswith("whole_molecules_then_")

from __future__ import annotations

import hashlib
import json

import pytest

from mmpbsa.core.scientific_context import build_scientific_context, verified_frame_selection


def _selection():
    return {
        "schema_version": "mmpbsa.frame-manifest.v1", "source_stage": "production",
        "source_trajectory_interval_ps": 20, "start_frame": 251, "end_frame": 751,
        "stride": 1, "expected_frames": 501,
    }


def _receipt(selection):
    return {
        "selection": selection,
        "digest": hashlib.sha256(json.dumps(
            selection, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
    }


def test_verified_selection_retains_canonical_digest_and_absolute_production_time():
    selection = _selection()
    result = verified_frame_selection(selection, _receipt(selection))
    assert result["status"] == "verified"
    assert result["analysis_start_ps"] == 5000
    assert result["analysis_interval_ps"] == 20
    assert result["expected_frames"] == 501


@pytest.mark.parametrize("mutation", ["missing", "digest", "selection", "geometry", "nan"])
def test_unknown_selection_is_not_inferred_or_silently_repaired(mutation):
    selection = _selection()
    receipt = _receipt(selection)
    if mutation == "missing":
        receipt = None
    elif mutation == "digest":
        receipt["digest"] = "0" * 64
    elif mutation == "selection":
        receipt["selection"] = {**selection, "start_frame": 1}
    elif mutation == "geometry":
        selection["stride"] = 2
        receipt = _receipt(selection)
    else:
        selection["source_trajectory_interval_ps"] = float("nan")
    assert verified_frame_selection(selection, receipt)["status"] == "unknown"


def test_legacy_and_blocked_context_does_not_fabricate_entity_or_scoring_evidence():
    result = build_scientific_context(None, replicas=[{
        "replica_id": "rep01", "qc_verdict": "BLOCK",
        "trajectory_qc": {"payload": {"reference_definition": {
            "semantics": "post_npt_production_start",
        }}},
        "scoring_models": {},
    }])
    assert result["status"] == "unknown"
    assert result["prepared_case"] == {"status": "unknown"}
    replica = result["replicas"]["rep01"]
    assert replica["qc_frame_selection"]["status"] == "unknown"
    assert replica["scoring"] == {}
    assert replica["qc_reference"]["semantics"] == "post_npt_production_start"


def test_actual_trajectory_and_scoring_receipts_are_separate_evidence():
    selection = _selection()
    result = build_scientific_context(None, frame_selection=selection, replicas=[{
        "replica_id": "rep01", "trajectory_frame_manifest": _receipt(selection),
        "scoring_models": {
            "gb": {"payload": {"frame_manifest": _receipt(selection)}},
            "pb": {"payload": {"frame_manifest": {"digest": "0" * 64}}},
        },
    }])
    replica = result["replicas"]["rep01"]
    assert replica["qc_frame_selection"]["status"] == "verified"
    assert replica["scoring"]["gb"]["frame_selection"]["status"] == "verified"
    assert replica["scoring"]["pb"]["frame_selection"]["status"] == "unknown"

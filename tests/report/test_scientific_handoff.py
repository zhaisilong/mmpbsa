from __future__ import annotations

import copy
import hashlib
import json

from mmpbsa.report.adapters import _energy_plots, _occupancy_profile, _trajectory_plots
from mmpbsa.report.builders import build_document
from mmpbsa.report.models import CaseRecord, PlotKind, PlotSeries, PlotSpec, ProjectRecord
from mmpbsa.report.render import render_document, render_plot_svg


def _receipt():
    selection = {
        "schema_version": "mmpbsa.frame-manifest.v1", "source_stage": "production",
        "source_trajectory_interval_ps": 20, "start_frame": 251, "end_frame": 253,
        "stride": 1, "expected_frames": 3,
    }
    digest = hashlib.sha256(json.dumps(
        selection, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return {"selection": selection, "digest": digest}


def _row():
    receipt = _receipt()
    return {
        "replica_id": "rep01", "qc_verdict": "PASS", "trajectory_frame_manifest": receipt,
        "trajectory_qc": {"payload": {"ligand_rmsd": {
            "columns": ["Frame", "ligand_primary"], "rows": [[1, 0.2], [2, 0.3], [3, 0.4]],
        }}},
        "scoring_models": {"gb": {"payload": {
            "frame_manifest": receipt, "frame_count": 3,
            "per_frame": {"columns": ["Frame", "DELTA TOTAL"],
                          "rows": [[1, -10], [2, -11], [3, -12]]},
        }}},
    }


def _models():
    return {"gb": {"score_selection": {"transform": {
        "scale": 4.184, "offset": 0, "input_unit": "kcal/mol", "output_unit": "kJ/mol",
    }}}}


def test_verified_actual_frame_receipts_map_scoring_and_qc_to_absolute_ns():
    row = _row()
    context = {"planned_frame_selection": _receipt()["selection"]}
    energy, convergence = _energy_plots((row,), ("gb",), _models(), scientific_context=context)
    qc = _trajectory_plots((row,), scientific_context=context)
    for panel in (*energy, *convergence, *qc):
        assert panel.plot.x_label == "Time (ns)"
        assert [point.x for point in panel.plot.series[0].points] == [5, 5.02, 5.04]
    assert energy[0].plot.series[0].points[0].y == -41.84


def test_legacy_or_mismatched_evidence_keeps_analysis_frame_and_never_guesses_time():
    row = _row()
    for context in (None, {"planned_frame_selection": {**_receipt()["selection"], "stride": 2}}):
        energy, _ = _energy_plots((row,), ("gb",), _models(), scientific_context=context)
        qc = _trajectory_plots((row,), scientific_context=context)
        assert energy[0].plot.x_label == qc[0].plot.x_label == "Analysis frame"
        assert [point.x for point in energy[0].plot.series[0].points] == [1, 2, 3]


def test_one_unverified_replica_prevents_mixing_ns_and_ordinal_frame_on_same_panel():
    first = _row()
    second = copy.deepcopy(first)
    second["replica_id"] = "rep02"
    second["trajectory_frame_manifest"]["digest"] = "0" * 64
    second["scoring_models"]["gb"]["payload"]["frame_manifest"]["digest"] = "0" * 64
    context = {"planned_frame_selection": _receipt()["selection"]}
    energy, _ = _energy_plots((first, second), ("gb",), _models(), scientific_context=context)
    qc = _trajectory_plots((first, second), scientific_context=context)
    assert energy[0].plot.x_label == qc[0].plot.x_label == "Analysis frame"
    for panel in (*energy, *qc):
        assert len(panel.plot.series) == 2
        assert all([point.x for point in series.points] == [1, 2, 3]
                   for series in panel.plot.series)


def test_zero_occupancy_bins_produce_disconnected_segments_and_visible_samples():
    points = _occupancy_profile(
        (0.1, 0.2, 2.1, 2.2), 300, minimum=0, maximum=3, bin_count=6,
    )
    assert len(points) == 2 and all(point.break_before for point in points)
    plot = PlotSpec(PlotKind.LINE, "RMSD", "occupancy", (PlotSeries("rep01", "rep01", points),))
    svg = render_plot_svg(plot)
    assert "polyline" not in svg
    assert svg.count("<circle") == 2


def test_block_native_loss_context_is_descriptive_and_preserves_existing_gates():
    from mmpbsa.report.adapters import _qc_contact_context

    row = {
        "replica_id": "rep01", "qc_verdict": "BLOCK", "trajectory_qc": {"payload": {
            "native_contacts": {
                "columns": [
                    "Frame", "Contacts[native]", "Contacts[nonnative]", "Contacts[mindist]",
                ],
                "rows": [[1, 2, 1, 3], [2, 0, 4, 3], [3, 0, 5, 3], [4, 1, 4, 3]],
            },
        }},
    }
    initial = copy.deepcopy(row)
    context = _qc_contact_context((row,))[0]
    assert context.native_zero_fraction == 0.5
    assert context.longest_native_zero_samples == 2
    assert context.nonnative_at_native_zero_samples == 2
    assert context.total_contacts_min == 3
    case = CaseRecord("case1", "Case", ("rep01",), qc_contact_context=(context,))
    html = render_document(build_document(ProjectRecord("demo", "Demo", (case,))))
    case_html = next(content for path, content in html.items() if str(path) != "index.html")
    assert "not additional QC gates" in case_html
    assert "does not mean losing all receptor contacts" in case_html
    assert row == initial
    row["trajectory_qc"]["payload"]["native_contacts"]["columns"].pop(2)
    for samples in row["trajectory_qc"]["payload"]["native_contacts"]["rows"]:
        samples.pop(2)
    missing_other = _qc_contact_context((row,))[0]
    assert missing_other.total_contacts_min is None
    assert missing_other.nonnative_at_native_zero_samples is None


def test_binding_score_titles_and_cumulative_mean_warning_are_explicit():
    energy, convergence = _energy_plots((_row(),), ("gb",), _models())
    case = CaseRecord("case1", "Case", ("rep01",), energy_plots=energy,
                      convergence_plots=convergence)
    html = render_document(build_document(ProjectRecord("demo", "Demo", (case,))))
    text = "".join(html.values())
    assert "Per-frame binding score" in text
    assert "Per-frame total energy" not in text
    assert "flattened cumulative mean alone does not establish" in text

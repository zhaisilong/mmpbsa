from __future__ import annotations

import pytest

from mmpbsa.backends.ambertools import derive_v1_dmm
from mmpbsa.backends.errors import BackendOutputError


def test_gb_dmm_matches_v1_formula_per_frame() -> None:
    derived = derive_v1_dmm(
        "gb",
        {
            "columns": ("Frame #", "VDWAALS", "EEL", "EGB", "ESURF"),
            "rows": (
                {
                    "Frame #": 1.0,
                    "VDWAALS": -5.0,
                    "EEL": -10.0,
                    "EGB": -3.0,
                    "ESURF": 0.5,
                },
            ),
        },
    )

    assert derived["evaluator"] == "v1_dmm_v1"
    assert derived["rows"][0]["DMM"] == pytest.approx(-7.1)


def test_pb_dmm_requires_dispersion_instead_of_defaulting_to_zero() -> None:
    with pytest.raises(BackendOutputError, match="EDISPER"):
        derive_v1_dmm(
            "pb",
            {
                "columns": ("Frame", "VDWAALS", "EEL", "EPB", "ENPOLAR"),
                "rows": (
                    {
                        "Frame": 1.0,
                        "VDWAALS": -5.0,
                        "EEL": -10.0,
                        "EPB": -3.0,
                        "ENPOLAR": 0.5,
                    },
                ),
            },
        )


def test_pb_dmm_matches_v1_formula() -> None:
    derived = derive_v1_dmm(
        "pb",
        {
            "columns": (
                "Frame",
                "VDWAALS",
                "EEL",
                "EPB",
                "ENPOLAR",
                "EDISPER",
            ),
            "rows": (
                {
                    "Frame": 2.0,
                    "VDWAALS": -5.0,
                    "EEL": -10.0,
                    "EPB": -3.0,
                    "ENPOLAR": 0.5,
                    "EDISPER": -0.2,
                },
            ),
        },
    )

    assert derived["rows"][0]["DMM"] == pytest.approx(-7.3)

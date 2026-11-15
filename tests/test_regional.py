from datetime import date

import numpy as np
import pandas as pd
import pytest

from agri_nova.regional import (
    cell_anomalies,
    cell_bounds,
    cell_label,
    grid_cells,
    regional_season_metrics,
    severity_color,
    snap_to_cell,
)
from agri_nova.scenarios import apply_scenario, describe_scenario


# ------------------------------------------------------------------ grid
def test_snap_to_cell_returns_the_centre_of_the_merra2_cell():
    assert snap_to_cell(44.88, -98.52) == (45.0, -98.75)


def test_grid_cells_is_a_block_of_distinct_cells_around_the_field():
    cells = grid_cells(44.88, -98.52, radius=1)
    assert len(cells) == 9 == len(set(cells))
    assert (45.0, -98.75) in cells                      # the field's own cell
    lats = sorted({c[0] for c in cells})
    assert lats == [44.5, 45.0, 45.5]
    assert len(grid_cells(44.88, -98.52, radius=2)) == 25


def test_cell_bounds_span_one_cell():
    (south, west), (north, east) = cell_bounds(45.0, -98.75)
    assert north - south == pytest.approx(0.5)
    assert east - west == pytest.approx(0.625)


# ------------------------------------------------------------- anomalies
def _long() -> pd.DataFrame:
    rows = []
    for (la, lo), values in {(45.0, -98.75): [10, 12, 11, 40], (45.5, -98.75): [5, 5, 5, 5]}.items():
        for year, v in zip([2014, 2015, 2016, 2017], values):
            rows.append({"lat": la, "lon": lo, "year": year, "water_stress_days": float(v)})
    return pd.DataFrame(rows)


def test_cell_anomalies_use_each_cells_own_record():
    out = cell_anomalies(_long(), "water_stress_days", 2017).set_index(["lat", "lon"])
    assert out.loc[(45.0, -98.75), "z"] > 1.0
    assert out.loc[(45.0, -98.75), "value"] == 40.0
    assert out.loc[(45.5, -98.75), "z"] == 0.0           # constant record -> no anomaly, no division by zero


def test_cell_anomalies_skip_cells_without_that_year_and_handle_empty_input():
    assert cell_anomalies(_long(), "water_stress_days", 1999).empty
    assert cell_anomalies(pd.DataFrame(), "water_stress_days", 2017).empty


def test_severity_color_flips_with_direction():
    assert severity_color(2.0, higher_is_worse=True) == severity_color(-2.0, higher_is_worse=False)   # both "bad"
    assert severity_color(0.0, True) == severity_color(0.0, False)                                    # normal
    assert severity_color(2.0, True) != severity_color(-2.0, True)


# ------------------------------------------------------------ downloading
def _fake_power(lat, lon, start, end) -> pd.DataFrame:
    if lat < 44.6:
        raise RuntimeError("rate limited")
    idx = pd.date_range(start, end)
    rng = np.random.RandomState(1)
    n = len(idx)
    return pd.DataFrame(
        {
            "PRECTOTCORR": rng.gamma(0.5, 4.0, n),
            "T2M": rng.normal(15, 8, n),
            "T2M_MAX": rng.normal(22, 8, n),
            "T2M_MIN": rng.normal(8, 8, n),
            "RH2M": rng.uniform(40, 90, n),
            "GWETROOT": rng.uniform(0.2, 0.7, n),
        },
        index=idx,
    )


def test_regional_season_metrics_reports_failed_cells_instead_of_raising():
    long, errors = regional_season_metrics(
        44.88, -98.52, date(2015, 1, 1), date(2016, 12, 31), fetch=_fake_power, max_workers=1
    )
    assert len(errors) == 3                              # the whole southern row (lat 44.5) failed
    assert all("rate limited" in msg for msg in errors.values())
    assert long[["lat", "lon"]].drop_duplicates().shape[0] == 6
    assert sorted(long["year"].unique()) == [2015, 2016]
    assert cell_label((45.0, -98.75)) == "45.00, -98.750"


# --------------------------------------------------------------- scenarios
def _raw() -> pd.DataFrame:
    idx = pd.date_range("2017-07-01", periods=3)
    return pd.DataFrame(
        {
            "PRECTOTCORR": [10.0, 0.0, 5.0],
            "T2M": [25.0, 26.0, 24.0],
            "T2M_MAX": [33.0, 34.0, 31.0],
            "T2M_MIN": [15.0, 16.0, 14.0],
            "TS_MAX": [36.0, 37.0, 34.0],
            "GWETROOT": [0.5, 0.5, 0.5],
        },
        index=idx,
    )


def test_scenario_scales_rain_shifts_temperature_and_leaves_the_input_alone():
    raw = _raw()
    out = apply_scenario(raw, rain_change_pct=-20, temp_shift_c=2.0)
    assert out["PRECTOTCORR"].tolist() == pytest.approx([8.0, 0.0, 4.0])
    assert out["T2M_MAX"].tolist() == pytest.approx([35.0, 36.0, 33.0])
    assert (out["GWETROOT"] < raw["GWETROOT"]).all()        # drier and hotter -> lower wetness
    assert raw["PRECTOTCORR"].tolist() == [10.0, 0.0, 5.0]   # original untouched


def test_scenario_with_no_change_is_the_identity_and_wetness_stays_in_range():
    raw = _raw()
    pd.testing.assert_frame_equal(apply_scenario(raw), raw)
    wet = apply_scenario(raw, rain_change_pct=40, temp_shift_c=-2.0)
    assert wet["GWETROOT"].between(0.0, 1.0).all()
    assert (wet["GWETROOT"] > raw["GWETROOT"]).all()


def test_describe_scenario_mentions_the_assumption():
    assert "no scenario" in describe_scenario(0, 0).lower()
    text = describe_scenario(-20, 2.0)
    assert "-20%" in text and "+2.0" in text and "not a hydrology model" in text

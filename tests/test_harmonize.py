import numpy as np
import pandas as pd
import pytest

from agri_nova.harmonize import harmonize_daily, to_common_grid, to_daily


def _daily(start="2020-06-01", end="2020-06-11") -> pd.DatetimeIndex:
    return pd.date_range(start, end)


def test_to_daily_interpolates_short_gaps_and_records_age():
    obs = pd.Series([0.0, 10.0], index=pd.to_datetime(["2020-06-01", "2020-06-11"]))
    out = to_daily(obs, _daily(), max_gap_days=5)

    assert out.loc["2020-06-04", "value"] == pytest.approx(3.0)
    assert out.loc["2020-06-04", "age_days"] == 3
    assert out.loc["2020-06-06", "value"] == pytest.approx(5.0)         # age 5 is still allowed
    assert out.loc["2020-06-01", "age_days"] == 0 and out.loc["2020-06-11", "age_days"] == 0
    assert out.loc["2020-06-11", "value"] == pytest.approx(10.0)


def test_to_daily_leaves_long_gaps_empty_instead_of_inventing_values():
    obs = pd.Series([0.0, 10.0], index=pd.to_datetime(["2020-06-01", "2020-06-11"]))
    out = to_daily(obs, _daily(), max_gap_days=5)
    assert out.loc["2020-06-08", "value"] != out.loc["2020-06-08", "value"]   # NaN
    assert pd.isna(out.loc["2020-06-08", "age_days"])


def test_to_daily_with_no_observations_is_all_nan():
    out = to_daily(pd.Series(dtype=float), _daily())
    assert out["value"].isna().all()


def test_harmonize_daily_adds_value_and_age_columns_per_layer():
    base = pd.DataFrame({"PRECTOTCORR": 1.0}, index=_daily())
    ndvi = pd.Series([0.2, 0.4], index=pd.to_datetime(["2020-06-01", "2020-06-05"]))
    out = harmonize_daily(base, {"ndvi": ndvi}, {"ndvi": 20})
    assert {"ndvi", "ndvi_age_days"} <= set(out.columns)
    assert out.loc["2020-06-03", "ndvi"] == pytest.approx(0.3)
    assert out.loc["2020-06-05", "ndvi_age_days"] == 0
    assert "PRECTOTCORR" in out.columns          # base columns are untouched


def test_to_common_grid_block_averages_fine_data_onto_the_coarse_grid():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("scipy")

    fine = xr.DataArray(
        np.ones((20, 20)),
        coords={"lat": 40 + 0.05 * np.arange(20), "lon": -100 + 0.05 * np.arange(20)},
        dims=("lat", "lon"),
    )
    coarse = xr.DataArray(
        np.full((4, 4), 2.0),
        coords={"lat": 40 + 0.25 * np.arange(4), "lon": -100 + 0.25 * np.arange(4)},
        dims=("lat", "lon"),
    )
    out = to_common_grid({"fine": fine, "coarse": coarse}, resolution_deg=0.25)

    assert out["fine"].shape == out["coarse"].shape == (4, 4)
    assert float(out["fine"].sel(lat=40.5, lon=-99.5, method="nearest")) == pytest.approx(1.0)
    assert float(out["coarse"].sel(lat=40.5, lon=-99.5, method="nearest")) == pytest.approx(2.0)


def test_to_common_grid_rejects_layers_that_do_not_overlap():
    xr = pytest.importorskip("xarray")
    a = xr.DataArray(np.ones((2, 2)), coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]}, dims=("lat", "lon"))
    b = xr.DataArray(np.ones((2, 2)), coords={"lat": [5.0, 6.0], "lon": [5.0, 6.0]}, dims=("lat", "lon"))
    with pytest.raises(ValueError):
        to_common_grid({"a": a, "b": b}, resolution_deg=0.5)

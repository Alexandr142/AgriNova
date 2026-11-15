import numpy as np
import pandas as pd
import pytest

from agri_nova.history import (
    anomalies,
    available_years,
    detect_flash_drought_events,
    frost_table,
    period_comparison,
    season_metrics,
    support_summary,
    wilson_interval,
    year_vs_record,
    yearly_signal_table,
)
from agri_nova.indicators import add_indicators


def _multi_year(first=2012, last=2022) -> pd.DataFrame:
    idx = pd.date_range(f"{first}-01-01", f"{last}-12-31")
    rng = np.random.RandomState(0)
    n = len(idx)
    df = pd.DataFrame(
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
    return add_indicators(df)


# ------------------------------------------------------------- season metrics
def test_season_metrics_has_one_row_per_year_and_the_expected_columns():
    df = _multi_year(2015, 2019)
    m = season_metrics(df)
    assert list(m.index) == [2015, 2016, 2017, 2018, 2019]
    for col in ("summer_rain_mm", "water_stress_days", "peak_heat_days", "min_root_wetness"):
        assert col in m.columns
    assert available_years(df) == [2015, 2016, 2017, 2018, 2019]


def test_incomplete_years_are_skipped():
    df = _multi_year(2015, 2016).loc["2015-01-01":"2016-05-01"]   # 2016 season is far too short
    assert available_years(df) == [2015]


def test_anomalies_are_centred_and_constant_columns_give_zero():
    m = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]}, index=[2015, 2016, 2017])
    z = anomalies(m)
    assert z["a"].mean() == pytest.approx(0.0)
    assert (z["b"] == 0.0).all()


def test_year_vs_record_ranks_the_worst_year_first():
    m = pd.DataFrame(
        {"water_stress_days": [1.0, 2.0, 30.0, 3.0], "summer_rain_mm": [300.0, 280.0, 90.0, 310.0]},
        index=[2014, 2015, 2017, 2018],
    )
    cmp_df = year_vs_record(m, 2017).set_index("metric")
    assert cmp_df.loc["Water-stress days (Jun-Aug)", "worst_rank"] == 1   # most stress days = worst
    assert cmp_df.loc["Summer rainfall (mm)", "worst_rank"] == 1          # least rain = worst
    assert cmp_df.loc["Water-stress days (Jun-Aug)", "z_score"] > 1


# ------------------------------------------------------------- flash drought
def test_flash_drought_detector_finds_a_rapid_drop_and_ignores_steady_dryness():
    idx = pd.date_range("2017-06-01", "2017-08-31")
    wet = np.full(len(idx), 0.6)
    crash = wet.copy()
    start = (pd.Timestamp("2017-07-01") - idx[0]).days
    for k in range(15):                       # 0.6 -> 0.3 over 14 days, then stays low
        crash[start + k] = 0.6 - 0.3 * k / 14
    crash[start + 15:] = 0.3

    events = detect_flash_drought_events(pd.DataFrame({"GWETROOT": crash}, index=idx))
    assert len(events) == 1
    assert int(events.iloc[0]["year"]) == 2017
    assert events.iloc[0]["min_wetness"] == pytest.approx(0.3, abs=0.01)

    steady = pd.DataFrame({"GWETROOT": np.full(len(idx), 0.25)}, index=idx)
    assert detect_flash_drought_events(steady).empty


# ------------------------------------------------------------------ statistics
def test_wilson_interval_behaves_for_tiny_samples():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(5, 10)
    assert 0.2 < lo < 0.3 and 0.7 < hi < 0.8
    lo, hi = wilson_interval(0, 3)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi > 0.3   # zero successes in 3 seasons is still very uncertain


def test_support_summary_counts_only_valid_seasons_and_warns_about_small_samples():
    table = pd.DataFrame({"cover_crop": ["supports", "supports", "mixed", "insufficient_data", "limited"]})
    s = support_summary(table, "cover_crop")
    assert (s["k"], s["mixed"], s["n"]) == (2, 1, 4)
    assert s["share"] == pytest.approx(0.5)
    assert "2 of 4" in s["text"]
    assert "rough guide" in s["text"]


def test_yearly_signal_table_returns_a_signal_for_each_complete_season():
    table = yearly_signal_table(_multi_year(2015, 2017))
    assert list(table.index) == [2015, 2016, 2017]
    assert set(table.columns) >= {"stress_level", "baseline", "cover_crop", "diversified"}
    assert (table["baseline"] == "reference").all()


# ----------------------------------------------------------------------- frost
def test_frost_table_finds_last_spring_and_first_fall_frost():
    idx = pd.date_range("2020-01-01", "2021-12-31")
    tmin = pd.Series(5.0, index=idx)
    tmin.loc["2020-04-20"] = -1.0
    tmin.loc["2020-10-05"] = -2.0
    df = pd.DataFrame({"T2M_MIN": tmin})
    t = frost_table(df)
    assert str(t.loc[2020, "last_spring_frost"]) == "2020-04-20"
    assert str(t.loc[2020, "first_fall_frost"]) == "2020-10-05"
    assert t.loc[2020, "frost_free_days"] == 168
    assert pd.isna(t.loc[2021, "frost_free_days"])      # no frost at all in 2021


def test_frost_table_ignores_incomplete_years():
    idx = pd.date_range("2020-01-01", "2020-06-30")
    t = frost_table(pd.DataFrame({"T2M_MIN": 5.0}, index=idx))
    assert t.empty


# ----------------------------------------------------------- before / after 2017
def test_period_comparison_splits_around_the_event_year():
    years = list(range(2012, 2023))
    values = [10.0 if y < 2017 else (30.0 if y == 2017 else 20.0) for y in years]
    m = pd.DataFrame({"water_stress_days": values}, index=years)
    row = period_comparison(m, 2017).set_index("metric").loc["Water-stress days (Jun-Aug)"]
    assert row["before_mean"] == 10.0
    assert row["event_year"] == 30.0
    assert row["after_mean"] == 20.0
    assert (row["n_before"], row["n_after"]) == (5, 5)

"""Multi-year statistics: how unusual is a season, and how often would each rotation have helped?

Everything here is plain, explainable statistics on the 2012-2022 record:

* per-season stress metrics and their z-scores against the whole record,
* a simple flash-drought detector (rapid drop in root-zone wetness),
* the share of historical seasons in which each rotation's signal was "supports",
  with a Wilson 95% interval so the (small!) sample size stays visible.
"""
from __future__ import annotations

import math

import pandas as pd

from agri_nova.rotations import ClimateSignals, strategy_signals, summarize_signals

SEASON_START = "04-01"
SEASON_END = "11-15"
MIN_SEASON_DAYS = 200
SMALL_SAMPLE_YEARS = 15

# metric key -> (display label, higher_is_worse)
METRIC_INFO: dict[str, tuple[str, bool]] = {
    "summer_rain_mm": ("Summer rainfall (mm)", False),
    "summer_tmax_mean_c": ("Mean summer daily max temp (C)", True),
    "peak_heat_days": ("Hot days in Jul-Aug", True),
    "water_stress_days": ("Water-stress days (Jun-Aug)", True),
    "longest_dry_spell_days": ("Longest dry spell (days)", True),
    "min_root_wetness": ("Lowest root-zone wetness (0-1)", False),
    "max_14d_wetness_drop": ("Fastest 14-day wetness drop", True),
}


def season_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df.loc[f"{year}-{SEASON_START}":f"{year}-{SEASON_END}"]


def available_years(df: pd.DataFrame) -> list[int]:
    years = sorted(set(int(y) for y in df.index.year))
    return [y for y in years if len(season_slice(df, y)) >= MIN_SEASON_DAYS]


def season_metrics(df: pd.DataFrame, heat_threshold_c: float = 35.0) -> pd.DataFrame:
    """One row per year with the metrics in ``METRIC_INFO``. ``df`` must have indicators."""
    drop_full = df["GWETROOT"].shift(14) - df["GWETROOT"]
    rows: dict[int, dict[str, float]] = {}
    for y in available_years(df):
        sub = season_slice(df, y)
        month = sub.index.month
        summer = sub[month.isin([6, 7, 8])]
        peak = sub[month.isin([7, 8])]
        drop = drop_full.loc[sub.index][month.isin([5, 6, 7, 8, 9])].dropna()
        rows[y] = {
            "summer_rain_mm": float(summer["PRECTOTCORR"].sum()),
            "summer_tmax_mean_c": float(summer["T2M_MAX"].mean()),
            "peak_heat_days": float((peak["T2M_MAX"] >= heat_threshold_c).sum()),
            "water_stress_days": float(summer["water_stress"].sum()),
            "longest_dry_spell_days": float(summer["dry_spell_days"].max()),
            "min_root_wetness": float(summer["GWETROOT"].min()),
            "max_14d_wetness_drop": float(drop.max()) if len(drop) else 0.0,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "year"
    return out


def anomalies(metrics: pd.DataFrame) -> pd.DataFrame:
    """Z-scores against the whole record. Constant columns get 0."""
    std = metrics.std(ddof=1).replace(0, float("nan"))
    return ((metrics - metrics.mean()) / std).fillna(0.0)


def year_vs_record(metrics: pd.DataFrame, year: int) -> pd.DataFrame:
    """Where does ``year`` sit within the record? ``worst_rank`` 1 = the worst season on record."""
    z = anomalies(metrics)
    rows = []
    for col, (label, higher_is_worse) in METRIC_INFO.items():
        if col not in metrics.columns or year not in metrics.index:
            continue
        ranks = metrics[col].rank(ascending=not higher_is_worse, method="min")
        rows.append(
            {
                "metric": label,
                "value": float(metrics.loc[year, col]),
                "record_mean": float(metrics[col].mean()),
                "z_score": float(z.loc[year, col]),
                "worst_rank": int(ranks.loc[year]),
                "n_years": int(len(metrics)),
            }
        )
    return pd.DataFrame(rows)


def detect_flash_drought_events(
    df: pd.DataFrame,
    window: int = 14,
    drop_threshold: float = 0.12,
    max_wetness: float = 0.45,
    merge_gap_days: int = 7,
    months: tuple[int, ...] = (5, 6, 7, 8, 9),
) -> pd.DataFrame:
    """Flag rapid-onset drying: root-zone wetness falling fast *and* ending up low.

    Heuristic inspired by the "rapid intensification" idea behind flash-drought research;
    it is not one of the formal indices and thresholds should be tuned locally.
    """
    cols = ["start", "end", "duration_days", "peak_drop", "min_wetness", "year"]
    w = df["GWETROOT"]
    drop = w.shift(window) - w
    cond = (drop >= drop_threshold) & (w <= max_wetness) & pd.Series(df.index.month, index=df.index).isin(months)
    days = list(df.index[cond.fillna(False)])
    if not days:
        return pd.DataFrame(columns=cols)

    groups: list[list[pd.Timestamp]] = [[days[0]]]
    for d in days[1:]:
        if (d - groups[-1][-1]).days > merge_gap_days:
            groups.append([d])
        else:
            groups[-1].append(d)

    rows = []
    for g in groups:
        start, end = g[0], g[-1]
        rows.append(
            {
                "start": start.date(),
                "end": end.date(),
                "duration_days": (end - start).days + 1,
                "peak_drop": float(drop.loc[start:end].max()),
                "min_wetness": float(w.loc[start:end].min()),
                "year": int(start.year),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def yearly_signal_table(df: pd.DataFrame, heat_threshold_c: float = 35.0) -> pd.DataFrame:
    """Signal each rotation would have received in every historical season."""
    rows = []
    for y in available_years(df):
        s: ClimateSignals = summarize_signals(season_slice(df, y), heat_threshold_c)
        rows.append({"year": y, "stress_level": s.stress_level, **strategy_signals(s)})
    return pd.DataFrame(rows).set_index("year")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion - behaves sensibly for tiny samples."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def support_summary(table: pd.DataFrame, key: str) -> dict:
    """How often did ``key`` get a 'supports' signal across historical seasons?"""
    valid = table[key][table[key] != "insufficient_data"]
    n = int(len(valid))
    k = int((valid == "supports").sum())
    m = int((valid == "mixed").sum())
    lo, hi = wilson_interval(k, n)
    if n == 0:
        text = "No complete historical seasons were available."
    else:
        text = (
            f"Data pattern supported this option in {k} of {n} historical seasons ({k / n:.0%}; "
            f"95% interval {lo:.0%}-{hi:.0%}) and was mixed in {m}."
        )
        if n < SMALL_SAMPLE_YEARS:
            text += f" With only {n} seasons this is a rough guide, not a forecast."
    return {"key": key, "k": k, "mixed": m, "n": n, "share": (k / n if n else None), "low": lo, "high": hi, "text": text}


def attach_history(assessments: list, table: pd.DataFrame) -> list:
    """Write the historical-support sentence into the cover-crop and diversified assessments."""
    for a in assessments:
        if a.key in ("cover_crop", "diversified") and a.key in table.columns:
            a.historical_support = support_summary(table, a.key)["text"]
    return assessments


def frost_table(df: pd.DataFrame, threshold_c: float = 0.0) -> pd.DataFrame:
    """Last spring frost, first fall frost and frost-free days for every complete calendar year.

    A frost day is a day whose POWER daily minimum air temperature (2 m, gridded) is at or
    below ``threshold_c``. Gridded minima run warmer than a frost hollow in a real field, so
    treat these dates as a regional guide. Spring = Jan-Jun, fall = Aug-Dec.
    """
    cols = ["year", "last_spring_frost", "first_fall_frost", "frost_free_days"]
    if "T2M_MIN" not in df.columns:
        return pd.DataFrame(columns=cols).set_index("year")

    rows = []
    for y in sorted(set(int(v) for v in df.index.year)):
        sub = df.loc[f"{y}-01-01":f"{y}-12-31"]
        if len(sub) < 330:
            continue
        frost = (sub["T2M_MIN"] <= threshold_c).to_numpy()
        month = sub.index.month.to_numpy()
        spring = sub.index[frost & (month <= 6)]
        fall = sub.index[frost & (month >= 8)]
        last_spring = spring.max().date() if len(spring) else None
        first_fall = fall.min().date() if len(fall) else None
        free = (first_fall - last_spring).days if (last_spring and first_fall) else None
        rows.append(
            {"year": y, "last_spring_frost": last_spring, "first_fall_frost": first_fall, "frost_free_days": free}
        )
    return pd.DataFrame(rows, columns=cols).set_index("year")


def period_comparison(metrics: pd.DataFrame, event_year: int = 2017) -> pd.DataFrame:
    """Average of every metric before vs after ``event_year``, next to the event year itself.

    Purely descriptive: with ~5 seasons on each side no significance test is meaningful, and
    the ``n_before`` / ``n_after`` columns keep that visible.
    """
    before = metrics[metrics.index < event_year]
    after = metrics[metrics.index > event_year]
    rows = []
    for col, (label, _) in METRIC_INFO.items():
        if col not in metrics.columns:
            continue
        rows.append(
            {
                "metric": label,
                "before_mean": float(before[col].mean()),
                "event_year": float(metrics.loc[event_year, col]) if event_year in metrics.index else float("nan"),
                "after_mean": float(after[col].mean()),
                "n_before": int(len(before)),
                "n_after": int(len(after)),
            }
        )
    return pd.DataFrame(rows)

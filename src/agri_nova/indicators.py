"""Simple, explainable farm-risk indicators built from daily weather data.

These are heuristics for a prototype. Tune thresholds per crop and region.
"""
from __future__ import annotations

import pandas as pd


def add_indicators(
    df: pd.DataFrame,
    heat_threshold_c: float = 35.0,
    dry_window_days: int = 14,
) -> pd.DataFrame:
    out = df.copy()

    # Water stress: low recent rainfall plus dry root zone
    out["rain_14d_mm"] = out["PRECTOTCORR"].rolling(dry_window_days, min_periods=1).sum()
    out["dry_spell_days"] = (
        (out["PRECTOTCORR"] < 1.0)
        .astype(int)
        .groupby((out["PRECTOTCORR"] >= 1.0).cumsum())
        .cumsum()
    )
    out["water_stress"] = (
        (out["GWETROOT"] < 0.35) & (out["rain_14d_mm"] < 10)
    ).astype(int)

    # Heat stress
    out["heat_stress"] = (out["T2M_MAX"] >= heat_threshold_c).astype(int)

    # Fungal disease pressure: warm + humid conditions persisting over a week
    humid_warm = (out["RH2M"] >= 80) & out["T2M"].between(15, 28)
    out["disease_risk_7d"] = humid_warm.rolling(7, min_periods=1).mean()

    # Rough pest proxy: growing degree days (base 10 C), accumulated
    out["gdd"] = ((out["T2M_MAX"] + out["T2M_MIN"]) / 2 - 10).clip(lower=0)
    out["gdd_cum"] = out["gdd"].cumsum()

    return out

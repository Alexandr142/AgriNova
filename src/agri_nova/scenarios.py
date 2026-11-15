"""What-if scenarios: shift rainfall and temperature, then re-run the indicators.

This is a deliberately simple *sensitivity* tool, not a climate projection or a
hydrology model. It answers: "if every season were a bit drier / hotter, would the
rotation signals change?"
"""
from __future__ import annotations

import pandas as pd

TEMP_COLUMNS = ("T2M", "T2M_MAX", "T2M_MIN", "TS_MAX")


def apply_scenario(
    df: pd.DataFrame,
    rain_change_pct: float = 0.0,
    temp_shift_c: float = 0.0,
    wetness_rain_elasticity: float = 0.4,
    wetness_heat_penalty_per_c: float = 0.03,
) -> pd.DataFrame:
    """Return a copy of the *raw* POWER frame with rainfall/temperature perturbed.

    Assumptions (stated in the app too):
    * rainfall is scaled by ``rain_change_pct`` percent;
    * all temperature columns are shifted by ``temp_shift_c`` degrees;
    * root-zone wetness (``GWETROOT``) responds with a rough elasticity:
      ``wetness *= (1 + elasticity * rain_fraction) * (1 - penalty * temp_shift)``,
      clipped to [0, 1]. This is an assumption, not a measured relationship.
    """
    out = df.copy()
    rain_frac = rain_change_pct / 100.0

    if "PRECTOTCORR" in out.columns:
        out["PRECTOTCORR"] = (out["PRECTOTCORR"] * (1.0 + rain_frac)).clip(lower=0.0)

    for col in TEMP_COLUMNS:
        if col in out.columns:
            out[col] = out[col] + temp_shift_c

    if "GWETROOT" in out.columns:
        factor = (1.0 + wetness_rain_elasticity * rain_frac) * (1.0 - wetness_heat_penalty_per_c * temp_shift_c)
        out["GWETROOT"] = (out["GWETROOT"] * max(factor, 0.0)).clip(0.0, 1.0)

    return out


def describe_scenario(rain_change_pct: float, temp_shift_c: float) -> str:
    if rain_change_pct == 0 and temp_shift_c == 0:
        return "Observed conditions (no scenario applied)."
    return (
        f"What-if: rainfall {rain_change_pct:+.0f}% and temperature {temp_shift_c:+.1f} C applied to every day of the record. "
        "Soil wetness response uses a simple assumed sensitivity, not a hydrology model."
    )

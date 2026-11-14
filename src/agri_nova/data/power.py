"""NASA POWER client: daily weather, solar and soil-wetness data for any lat/lon.

Docs: https://power.larc.nasa.gov/docs/services/api/
No API key required, which makes it ideal for a hackathon baseline.
"""
from __future__ import annotations

from datetime import date

import httpx
import numpy as np
import pandas as pd

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# PRECTOTCORR: precipitation (mm/day)   T2M / T2M_MAX / T2M_MIN: temperature (C)
# RH2M: relative humidity (%)           ALLSKY_SFC_SW_DWN: solar radiation (kWh/m2/day)
# GWETROOT: root-zone soil wetness (0-1)
DEFAULT_PARAMETERS = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "GWETROOT",
]


def fetch_daily(
    lat: float,
    lon: float,
    start: date,
    end: date,
    parameters: list[str] | None = None,
) -> pd.DataFrame:
    """Return a date-indexed DataFrame with one column per POWER parameter."""
    params = {
        "parameters": ",".join(parameters or DEFAULT_PARAMETERS),
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    resp = httpx.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df.replace(-999, np.nan)  # POWER uses -999 for missing values

"""NASA POWER client: daily weather, solar and soil-wetness data for any lat/lon.

Docs: https://power.larc.nasa.gov/docs  /services/api/
API Key not required
"""
from __future__ import annotations

from datetime import date

import httpx
import numpy as np
import pandas as pd

from agri_nova.paths import CACHE_DIR

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# PRECTOTCORR: precipitation (mm/day)   T2M / T2M_MAX / T2M_MIN: temperature (C)
# RH2M: relative humidity (%)           ALLSKY_SFC_SW_DWN: solar radiation (kWh/m2/day)
# GWETROOT: root-zone soil wetness (0-1)   TS_MAX: max earth skin temperature (C)
DEFAULT_PARAMETERS = [
    "PRECTOTCORR",
    "T2M",
    "TS_MAX",
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
        "time-standard": "UTC"
    }
    resp = httpx.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df.replace(-999, np.nan)  # POWER uses -999 for missing values


def fetch_daily_cached(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Like ``fetch_daily`` but keeps a CSV copy so the demo survives flaky Wi-Fi.

    POWER's grid is 0.5 degrees, so rounding the coordinates to 2 decimals in the cache key
    loses nothing. A cached copy is always preferred; delete the CSV in data/processed to refresh.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"power_{lat:.2f}_{lon:.2f}_{start:%Y%m%d}_{end:%Y%m%d}.csv"

    if path.exists():
        return pd.read_csv(path, index_col="date", parse_dates=True)

    df = fetch_daily(lat, lon, start, end)
    df.to_csv(path)
    return df

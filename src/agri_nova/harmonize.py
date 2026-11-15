"""Put datasets with different resolutions onto one common analysis grid.

Not used by the dashboard at the moment (POWER is the only time series it loads), but kept as a
tested utility for when another layer is added. Two kinds of harmonisation, both explicit about
what they invent:

* **Time axis** (``harmonize_daily``): each series is interpolated onto a daily axis, but
  only across gaps up to ``max_gap_days`` - beyond that it stays NaN. A ``<name>_age_days``
  column records how many days old the nearest earlier real observation is, so downstream
  code (and the farmer) can tell measurement from interpolation.
* **Space** (``to_common_grid``): raster layers of different pixel size are block-averaged /
  interpolated to one grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MAX_GAP_DAYS = 5


def to_daily(series: pd.Series, index: pd.DatetimeIndex, max_gap_days: int = DEFAULT_MAX_GAP_DAYS) -> pd.DataFrame:
    """Interpolate ``series`` onto the daily ``index``; return columns ``value`` and ``age_days``."""
    obs = series.dropna().sort_index()
    obs.index = pd.DatetimeIndex(obs.index).normalize()
    obs = obs[~obs.index.duplicated(keep="last")]

    if obs.empty:
        return pd.DataFrame({"value": np.nan, "age_days": np.nan}, index=index)

    union = index.union(obs.index)
    interp = obs.reindex(union).interpolate(method="time", limit_area="inside")

    last_obs_date = pd.Series(obs.index, index=obs.index).reindex(union).ffill()
    age = (pd.Series(union, index=union) - last_obs_date).dt.days

    value = interp.reindex(index)
    age = age.reindex(index)
    value[age > max_gap_days] = np.nan
    age[value.isna()] = np.nan
    return pd.DataFrame({"value": value, "age_days": age}, index=index)


def harmonize_daily(
    base: pd.DataFrame,
    layers: dict[str, pd.Series],
    max_gap_days: dict[str, int] | int = DEFAULT_MAX_GAP_DAYS,
) -> pd.DataFrame:
    """Join each layer onto ``base`` (a daily frame) as ``<name>`` and ``<name>_age_days``."""
    out = base.copy()
    for name, series in layers.items():
        gap = max_gap_days.get(name, DEFAULT_MAX_GAP_DAYS) if isinstance(max_gap_days, dict) else max_gap_days
        daily = to_daily(series, out.index, gap)
        out[name] = daily["value"]
        out[f"{name}_age_days"] = daily["age_days"]
    return out


def to_common_grid(arrays: dict, resolution_deg: float) -> dict:
    """Resample 2-D (lat, lon) ``xarray.DataArray`` layers onto one regular grid.

    The grid covers the *intersection* of all layers' extents. Finer layers are block-averaged
    first (so a 250 m NDVI pixel is not just point-sampled), then every layer is linearly
    interpolated onto the shared coordinates.
    """
    if not arrays:
        return {}

    south = max(float(a.lat.min()) for a in arrays.values())
    north = min(float(a.lat.max()) for a in arrays.values())
    west = max(float(a.lon.min()) for a in arrays.values())
    east = min(float(a.lon.max()) for a in arrays.values())
    if south >= north or west >= east:
        raise ValueError("The layers do not overlap spatially.")

    lat = np.arange(south, north + 1e-9, resolution_deg)
    lon = np.arange(west, east + 1e-9, resolution_deg)

    out = {}
    for name, da in arrays.items():
        res_lat = abs(float(da.lat[1] - da.lat[0])) if da.lat.size > 1 else resolution_deg
        res_lon = abs(float(da.lon[1] - da.lon[0])) if da.lon.size > 1 else resolution_deg
        f_lat = max(1, int(round(resolution_deg / res_lat)))
        f_lon = max(1, int(round(resolution_deg / res_lon)))
        if f_lat > 1 or f_lon > 1:
            da = da.coarsen(lat=f_lat, lon=f_lon, boundary="trim").mean()
        out[name] = da.interp(lat=lat, lon=lon)
    return out

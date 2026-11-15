"""Regional context for the map tab: how did conditions vary across the cells around the field?

NASA POWER's weather and soil-wetness fields come from MERRA-2, whose grid cells are about
0.5 deg (latitude) x 0.625 deg (longitude), roughly 55 x 50 km in the Northern Plains. This
module downloads the 2012-2022 record for a small block of neighbouring cells, computes the
same season metrics used elsewhere (``history.season_metrics``) and expresses each cell's
chosen year as a z-score against that cell's *own* record.

It is a regional picture, not a field-level one: nothing here resolves differences inside a
single cell. Field-level slope and soil come from SRTM and SSURGO instead.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from agri_nova.data.power import fetch_daily_cached
from agri_nova.history import season_metrics
from agri_nova.indicators import add_indicators

LAT_STEP = 0.5
LON_STEP = 0.625


def snap_to_cell(lat: float, lon: float) -> tuple[float, float]:
    """Centre of the MERRA-2 cell that contains the point."""
    return (
        round(round(lat / LAT_STEP) * LAT_STEP, 3),
        round(round(lon / LON_STEP) * LON_STEP, 3),
    )


def grid_cells(lat: float, lon: float, radius: int = 1) -> list[tuple[float, float]]:
    """(2*radius+1)^2 cell centres around the cell containing the point, north-west first."""
    clat, clon = snap_to_cell(lat, lon)
    return [
        (round(clat + i * LAT_STEP, 3), round(clon + j * LON_STEP, 3))
        for i in range(radius, -radius - 1, -1)
        for j in range(-radius, radius + 1)
    ]


def cell_label(cell: tuple[float, float]) -> str:
    return f"{cell[0]:.2f}, {cell[1]:.3f}"


def cell_bounds(lat: float, lon: float) -> list[list[float]]:
    """[[south, west], [north, east]] of the cell centred on (lat, lon) - folium rectangle format."""
    return [[lat - LAT_STEP / 2, lon - LON_STEP / 2], [lat + LAT_STEP / 2, lon + LON_STEP / 2]]


def regional_season_metrics(
    lat: float,
    lon: float,
    start: date,
    end: date,
    radius: int = 1,
    fetch=None,
    max_workers: int = 3,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Season metrics for every cell in the block.

    Returns ``(long_frame, errors)``. ``long_frame`` has one row per (cell, year) with the
    ``history.METRIC_INFO`` columns plus ``lat`` and ``lon``; a cell that fails to download is
    skipped and reported in ``errors`` (label -> message) instead of raising.
    """
    fetch = fetch or fetch_daily_cached

    def one(cell: tuple[float, float]) -> pd.DataFrame:
        la, lo = cell
        metrics = season_metrics(add_indicators(fetch(la, lo, start, end))).reset_index()
        metrics["lat"], metrics["lon"] = la, lo
        return metrics

    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    cells = grid_cells(lat, lon, radius)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {cell: pool.submit(one, cell) for cell in cells}
        for cell, fut in futures.items():
            try:
                frames.append(fut.result())
            except Exception as exc:  # network, rate limit, bad payload
                errors[cell_label(cell)] = str(exc)

    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return long, errors


def cell_anomalies(long: pd.DataFrame, metric: str, year: int) -> pd.DataFrame:
    """Per cell: the metric in ``year``, the cell's record mean, and the z-score against its own record."""
    rows = []
    if long.empty or metric not in long.columns:
        return pd.DataFrame(columns=["lat", "lon", "value", "record_mean", "z"])
    for (la, lo), g in long.groupby(["lat", "lon"]):
        vals = g.set_index("year")[metric]
        if year not in vals.index:
            continue
        std = vals.std(ddof=1)
        value = float(vals.loc[year])
        z = 0.0 if (pd.isna(std) or std == 0) else float((value - vals.mean()) / std)
        rows.append({"lat": la, "lon": lo, "value": value, "record_mean": float(vals.mean()), "z": z})
    return pd.DataFrame(rows, columns=["lat", "lon", "value", "record_mean", "z"])


def severity_color(z: float, higher_is_worse: bool) -> str:
    """Red = worse than that cell usually sees, blue = better, grey = about normal."""
    s = z if higher_is_worse else -z
    if s >= 1.5:
        return "#b2182b"
    if s >= 0.5:
        return "#ef8a62"
    if s > -0.5:
        return "#bdbdbd"
    if s > -1.5:
        return "#67a9cf"
    return "#2166ac"

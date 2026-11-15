"""Topographic analysis from NASA SRTM elevation, via public lookup APIs.

Slope is computed with central differences from four elevation samples around the field
centre. The sampling distance is intentionally coarse (~250 m) because SRTM pixels are
30-90 m and the lookup services interpolate; a smaller step would mostly measure noise.

If no elevation service answers, ``available`` is False and the rest of the app treats
slope as *unknown* - never as "flat".
"""
from __future__ import annotations

import math

import httpx

SPACING_M = 250.0
M_PER_DEG_LAT = 111_320.0


def _points(lat: float, lon: float, spacing_m: float) -> list[tuple[float, float]]:
    dlat = spacing_m / M_PER_DEG_LAT
    dlon = spacing_m / (M_PER_DEG_LAT * max(math.cos(math.radians(lat)), 0.1))
    # order: centre, north, south, east, west
    return [(lat, lon), (lat + dlat, lon), (lat - dlat, lon), (lat, lon + dlon), (lat, lon - dlon)]


def _elevations_opentopodata(points: list[tuple[float, float]]) -> list[float]:
    locs = "|".join(f"{la},{lo}" for la, lo in points)
    resp = httpx.get("https://api.opentopodata.org/v1/srtm30m", params={"locations": locs}, timeout=15.0)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [r["elevation"] for r in results]


def _elevations_open_elevation(points: list[tuple[float, float]]) -> list[float]:
    locs = "|".join(f"{la},{lo}" for la, lo in points)
    resp = httpx.get("https://api.open-elevation.com/api/v1/lookup", params={"locations": locs}, timeout=15.0)
    resp.raise_for_status()
    return [r["elevation"] for r in resp.json().get("results", [])]


def slope_from_elevations(z: list[float], spacing_m: float) -> float:
    """Slope in percent from [centre, north, south, east, west] elevations (central differences)."""
    _, z_n, z_s, z_e, z_w = z
    dz_dy = (z_n - z_s) / (2 * spacing_m)
    dz_dx = (z_e - z_w) / (2 * spacing_m)
    return math.hypot(dz_dx, dz_dy) * 100.0


def risk_label(slope_percent: float) -> str:
    if slope_percent > 8.0:
        return "Critical (High Runoff)"
    if slope_percent > 3.0:
        return "Moderate"
    return "Low"


def calculate_slope_gradient(lat: float, lon: float) -> dict:
    """Return elevation, slope and a risk label; ``available`` says whether it is real data."""
    pts = _points(lat, lon, SPACING_M)
    last_error = ""
    for provider in (_elevations_opentopodata, _elevations_open_elevation):
        try:
            z = provider(pts)
            if len(z) != 5 or any(v is None for v in z):
                raise ValueError("Incomplete elevation data returned.")
            slope = slope_from_elevations(z, SPACING_M)
            return {
                "elevation_m": round(float(z[0]), 1),
                "slope_percent": round(slope, 2),
                "risk_level": risk_label(slope),
                "available": True,
            }
        except Exception as exc:  # network, rate limit, bad payload - try the next provider
            last_error = f"{provider.__name__}: {exc}"

    return {
        "elevation_m": 0.0,
        "slope_percent": 0.0,
        "risk_level": "Data Unavailable",
        "available": False,
        "message": last_error,
    }

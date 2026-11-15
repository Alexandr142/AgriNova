"""ISRIC SoilGrids: global soil texture, used as a fallback when USDA SSURGO has no answer."""
from __future__ import annotations

import httpx


def fetch_soil_profile(lat: float, lon: float) -> dict[str, float | str | bool]:
    """Sand and clay (%) for 0-15 cm from the SoilGrids REST API.

    Returns ``{"available": False}`` on any failure - no made-up default values.
    """
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lat": lat,
        "lon": lon,
        "property": ["sand", "clay"],
        "depth": "0-15cm",
        "value": "mean",
    }

    try:
        resp = httpx.get(url, params=params, timeout=15.0)
        resp.raise_for_status()
        layers = resp.json().get("properties", {}).get("layers", [])

        info: dict[str, float | str | bool] = {}
        for layer in layers:
            name = layer.get("name")
            val = layer.get("depths", [{}])[0].get("values", {}).get("mean")
            if name and val is not None:
                info[name] = val / 10.0  # SoilGrids stores g/kg; /10 gives percent
    except Exception:
        return {"available": False}

    if "sand" not in info or "clay" not in info:
        return {"available": False}

    sand, clay = float(info["sand"]), float(info["clay"])
    if sand > 60:
        profile = "High drainage (low retention, high drought risk)"
    elif clay > 40:
        profile = "Poor drainage (high runoff risk)"
    else:
        profile = "Well-balanced ground drainage"

    return {"available": True, "sand": sand, "clay": clay, "profile_type": profile, "source": "ISRIC SoilGrids"}

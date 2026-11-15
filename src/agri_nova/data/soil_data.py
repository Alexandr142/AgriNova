"""Fetch soil properties from ISRIC SoilGrids to fuse with NASA climate data for better stress modeling."""
from __future__ import annotations

import httpx


def fetch_soil_profile(lat: float, lon: float) -> dict[str, float | str]:
    """
    Fetch sand and clay percentages from ISRIC SoilGrids REST API.
    Fallback provided if API fails.
    """
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lat": lat,
        "lon": lon,
        "property": ["sand", "clay"],
        "depth": "0-15cm",
        "value": "mean"
    }

    try:
        resp = httpx.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()

        layers = data.get("properties", {}).get("layers", [])
        soil_info = {}

        for layer in layers:
            prop_name = layer.get("name")
            # ISRIC values are multiplied by 10 so divide to get actual percentages
            val = layer.get("depths", [{}])[0].get("values", {}).get("mean")
            if val is not None:
                soil_info[prop_name] = val / 10.0

    except Exception:
        # Fallback values if API fails
        soil_info = {"sand": 35.0, "clay": 25.0}

    # Derive agronomic insight
    sand = soil_info.get("sand", 0)
    clay = soil_info.get("clay", 0)

    if sand > 60:
        drainage_status = "High drainage (low retention, high drought risk)"
    elif clay > 40:
        drainage_status = "Poor drainage (high runoff risk)"
    else:
        drainage_status = "Well-balanced ground drainage"

    soil_info["profile_type"] = drainage_status

    return soil_info
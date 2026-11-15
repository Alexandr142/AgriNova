"""Topographic analysis using multi-point elevation sampling for slope calculation(NASA SRTM)."""
import math
import httpx


def calculate_slope_gradient(lat: float, lon: float) -> dict:
    """
    Calculates field slope percentage by sampling elevation at the target
    point and adjacent points (approx 100m away) to determine erosion risk.
    """
    # 0.0009 degrees is roughly 100 meters at the equator
    offset = 0.0009
    locations = f"{lat},{lon}|{lat + offset},{lon}|{lat},{lon + offset}"
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

    try:
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        results = resp.json().get("results", [])

        if len(results) != 3:
            raise ValueError("Incomplete elevation data returned.")

        z_center = results[0]["elevation"]
        z_north = results[1]["elevation"]
        z_east = results[2]["elevation"]

        # Calculate gradients (dz/dx and dz/dy)
        dz_dx = (z_east - z_center) / 100.0
        dz_dy = (z_north - z_center) / 100.0

        # Magnitude of the gradient vector
        slope_percent = math.sqrt(dz_dx ** 2 + dz_dy ** 2) * 100.0

        erosion_risk = "Low"
        if slope_percent > 8.0:
            erosion_risk = "Critical (High Runoff)"
        elif slope_percent > 3.0:
            erosion_risk = "Moderate"

        return {
            "elevation_m": round(z_center, 1),
            "slope_percent": round(slope_percent, 2),
            "risk_level": erosion_risk
        }
    except Exception:
        return {
            "elevation_m": 0.0,
            "slope_percent": 0.0,
            "risk_level": "Data Unavailable"
        }
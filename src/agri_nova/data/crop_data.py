"""USDA CDL (Cropland Data Layer) API integration with resilient raw-text parsing."""
from __future__ import annotations
import httpx
import re
from pyproj import Transformer

def fetch_crop_history(lat: float, lon: float, start_year: int = 2019, end_year: int = 2023) -> list[dict]:
    """
    Fetches actual historical crop data for a specific coordinate from USDA CropScape.
    Bypasses standard JSON parsers to handle malformed USDA responses via Regex.
    """
    # Transform coordinates from Lat/Lon (EPSG:4326) to meters (EPSG:5070)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x_meters, y_meters = transformer.transform(lon, lat)

    base_url = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLValue"
    history = []

    with httpx.Client(timeout=15.0) as client:
        for year in range(start_year, end_year + 1):
            params = {"year": year, "x": x_meters, "y": y_meters}
            try:
                resp = client.get(base_url, params=params)
                resp.raise_for_status()

                # USDA API often returns invalid JSON strings inside XML.
                # Example: <return>{Value:"1",Category:"Corn"}</return>
                # We use regex to extract the value next to 'Category' directly.
                match = re.search(r'"?[Cc]ategory"?\s*:\s*"([^"]+)"', resp.text)

                if match:
                    crop_name = match.group(1).strip()
                    if crop_name and crop_name.lower() != "unknown":
                        history.append({"year": year, "crop": crop_name})
                    else:
                        history.append({"year": year, "crop": "No Data"})
                else:
                    history.append({"year": year, "crop": "No Data"})
            except Exception as e:
                print(f"Failed to fetch USDA CDL for {year}: {e}")
                history.append({"year": year, "crop": "API Error"})

    return history

def analyze_exhaustion(history: list[dict], crop_database: dict) -> dict:
    """
    Analyzes field history mathematically without hardcoded crop names.
    Calculates monoculture index (max consecutive years) and diversity index.
    """
    valid_crops = [item["crop"] for item in history if item["crop"] not in ("Unknown", "No Data", "API Error")]

    if not valid_crops:
        return {
            "nitrogen_level": "Unknown",
            "exhaustion_status": "No valid crop history found. API unavailable.",
            "baseline_crop": "Corn",  # Safe fallback
            "raw_history": history
        }

    latest_crop = valid_crops[-1]
    unique_crops = set(valid_crops)

    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(valid_crops)):
        if valid_crops[i] == valid_crops[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    n_fixer_count = 0
    for crop_name in valid_crops:
        for db_id, trait_obj in crop_database.items():
            if db_id.lower() in crop_name.lower():
                if getattr(trait_obj, 'nitrogen_fixer', False):
                    n_fixer_count += 1
                break

    if max_consecutive >= 3:
        exhaustion_status = f"Critical (Monoculture: {max_consecutive} consecutive years)"
        nitrogen_level = "Severely Depleted"
    elif max_consecutive == 2:
        exhaustion_status = f"Moderate (Short rotation: {max_consecutive} years)"
        nitrogen_level = "Partially Depleted"
    elif len(unique_crops) >= 3:
        exhaustion_status = f"Healthy (Diverse: {len(unique_crops)} distinct crops)"
        nitrogen_level = "Stable"
    else:
        exhaustion_status = "Average"
        nitrogen_level = "Depleted"

    if n_fixer_count > 0 and nitrogen_level != "Stable":
        nitrogen_level = f"Restoring ({n_fixer_count} N-fixing cycles detected)"

    return {
        "nitrogen_level": nitrogen_level,
        "exhaustion_status": exhaustion_status,
        "baseline_crop": latest_crop,
        "raw_history": history
    }
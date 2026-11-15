"""USDA CDL (Cropland Data Layer) API integration with resilient raw-text parsing."""
from __future__ import annotations
import httpx
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pyproj import Transformer

BASE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLValue"


def parse_cdl_category(text: str) -> str:
    """Extract the crop name from a CropScape response; 'No Data' if absent/unknown."""
    # USDA API often returns invalid JSON strings inside XML.
    # Example: <return>{Value:"1",Category:"Corn"}</return>
    match = re.search(r'"?[Cc]ategory"?\s*:\s*"([^"]+)"', text)
    if not match:
        return "No Data"
    crop = match.group(1).strip()
    return crop if crop and crop.lower() != "unknown" else "No Data"


def fetch_crop_history(lat: float, lon: float, start_year: int = 2012, end_year: int = 2022) -> list[dict]:
    """
    Fetches actual historical crop data for a specific coordinate from USDA CropScape.
    Bypasses standard JSON parsers to handle malformed USDA responses via Regex.
    Years are requested in parallel; a failed year is reported as 'API Error'.
    """
    # Transform coordinates from Lat/Lon (EPSG:4326) to meters (EPSG:5070)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x_meters, y_meters = transformer.transform(lon, lat)

    def one_year(year: int) -> dict:
        try:
            resp = httpx.get(BASE_URL, params={"year": year, "x": x_meters, "y": y_meters}, timeout=20.0)
            resp.raise_for_status()
            return {"year": year, "crop": parse_cdl_category(resp.text)}
        except Exception:
            return {"year": year, "crop": "API Error"}

    years = list(range(start_year, end_year + 1))
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one_year, years))

# CDL classes that are not row-crop/small-grain cropland: they say nothing about rotation health.
NON_CROP_KEYWORDS = (
    "grass", "pasture", "developed", "water", "wetland", "forest", "shrub", "barren", "fallow", "idle",
)


def analyze_exhaustion(history: list[dict], crop_database: dict) -> dict:
    """
    Calculates monoculture index (max consecutive years) and diversity index.
    """
    ignored = ("Unknown", "No Data", "API Error")
    named = [item["crop"] for item in history if item["crop"] not in ignored]
    valid_crops = [c for c in named if not any(k in c.lower() for k in NON_CROP_KEYWORDS)]

    if named and not valid_crops:
        return {
            "nitrogen_level": "Not applicable",
            "exhaustion_status": "CDL shows no cropland at this point (grass, water, developed or fallow). Pick a field.",
            "baseline_crop": "Corn",
            "raw_history": history,
        }

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


def suggest_baseline(history: list[dict]) -> tuple[str, str]:
    """Pick the existing-rotation preset (a key of ``rotations.BASELINES``) that fits the CDL history.

    Returns ``(preset_key, explanation)``. Only a default: the farmer can always change it.
    """
    ignored = ("Unknown", "No Data", "API Error")
    crops = [
        item["crop"] for item in history
        if item["crop"] not in ignored and not any(k in item["crop"].lower() for k in NON_CROP_KEYWORDS)
    ]
    recent = crops[-3:]
    if len(recent) == 3 and all("corn" in c.lower() for c in recent):
        return "continuous_corn", "USDA CDL shows corn in each of the last 3 mapped years, so Continuous Corn was pre-selected."
    if recent:
        return (
            "corn_soy",
            f"Latest CDL crops here: {', '.join(recent)}. Closest preset is Corn / Soybean - change it if that is not your rotation.",
        )
    return "corn_soy", "No usable CDL crop history for this point, so Corn / Soybean is assumed."


# USDA CDL class (lower case) -> id in rotations.CROP_DATABASE. Only classes the model knows about.
CDL_TO_CROP: dict[str, str] = {
    "corn": "Corn",
    "soybeans": "Soybean",
    "winter wheat": "Winter Wheat",
    "spring wheat": "Spring Wheat",
    "durum wheat": "Spring Wheat",
    "sorghum": "Sorghum",
    "sunflower": "Sunflower",
    "peas": "Peas",
    "alfalfa": "Alfalfa",
}


def suggest_rotation(history: list[dict], window: int = 6, max_crops: int = 3) -> tuple[list[str], str]:
    """Pre-select the crops of the existing rotation from the CDL history.

    Returns ``(crop ids from rotations.CROP_DATABASE, explanation)``. Only a default: the farmer
    can change it. Unknown CDL classes (grass, fallow, double-crop codes, ...) are skipped.
    """
    mapped = [
        (item["crop"], CDL_TO_CROP[item["crop"].strip().lower()])
        for item in history
        if item["crop"].strip().lower() in CDL_TO_CROP
    ]
    recent = mapped[-window:]
    if not recent:
        return ["Corn", "Soybean"], "No usable CDL crop history for this point, so Corn / Soybean is assumed - pick your real crops."

    ids = [crop_id for _, crop_id in recent]
    if len(ids) >= 3 and len(set(ids[-3:])) == 1:
        return [ids[-1]], f"USDA CDL shows {recent[-1][0]} in each of the last 3 mapped years, so a single-crop rotation was pre-selected."

    counts = Counter(ids)
    unique: list[str] = []
    for crop_id in ids:                       # order of first appearance
        if crop_id not in unique:
            unique.append(crop_id)
    if len(unique) > max_crops:
        keep = {c for c, _ in counts.most_common(max_crops)}
        unique = [c for c in unique if c in keep]

    raw_names: list[str] = []
    for raw, _ in recent:
        if raw not in raw_names:
            raw_names.append(raw)
    return unique, (
        f"USDA CDL mapped these crops here in the last {len(ids)} mapped years: {', '.join(raw_names)}. "
        "Change the selection to match your real rotation."
    )
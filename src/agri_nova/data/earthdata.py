"""NASA Earthdata access via `earthaccess` (needs a free account: https://urs.earthdata.nasa.gov).

Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD in .env, or run `earthaccess.login()` interactively.
Useful collections for farming:
  SPL3SMP_E   SMAP enhanced daily soil moisture
  MOD13Q1     MODIS 16-day NDVI (vegetation health)
  GPM_3IMERGDL GPM IMERG daily precipitation
  ECO_L2T_LSTE ECOSTRESS land surface temperature / evapotranspiration proxies
"""
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

import earthaccess

load_dotenv()


def search_granules(
    short_name: str,
    bbox: tuple[float, float, float, float],  # (west, south, east, north)
    start: str,  # "YYYY-MM-DD"
    end: str,
    count: int = 10,
):
    earthaccess.login(strategy="environment")
    return earthaccess.search_data(
        short_name=short_name,
        bounding_box=bbox,
        temporal=(start, end),
        count=count,
    )

def download(granules, out_dir: str = "data/raw"):
    """Download satellite data."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if not granules:
        print("No satellite data found.")
        return []

    files = earthaccess.download(granules, out_dir)

    print(f"Downloaded {len(files)} files")

    return files

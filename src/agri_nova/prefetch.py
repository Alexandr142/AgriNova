"""Pre-download everything the dashboard needs into data/processed so a demo works offline.

Usage:
    uv run python -m agri_nova.prefetch
    uv run python -m agri_nova.prefetch --lat 44.88 --lon -98.52
    uv run python -m agri_nova.prefetch --region         # also the 3x3 block for the regional map
    uv run python -m agri_nova.prefetch --force          # re-fetch terrain / soil / crop history
"""
from __future__ import annotations

import argparse
from datetime import date

from agri_nova.data.power import fetch_daily_cached
from agri_nova.data.site import load_site_context
from agri_nova.regional import regional_season_metrics

DEFAULT_LAT, DEFAULT_LON = 44.88, -98.52  # Spink County, SD


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pre-fetch NASA and USDA data into the local cache.")
    p.add_argument("--lat", type=float, default=DEFAULT_LAT)
    p.add_argument("--lon", type=float, default=DEFAULT_LON)
    p.add_argument("--force", action="store_true", help="ignore the cached site file")
    p.add_argument("--region", action="store_true", help="also cache the 3x3 POWER block for the regional map")
    args = p.parse_args(argv)

    start, end = date(2012, 1, 1), date(2022, 12, 31)

    print(f"NASA POWER 2012-2022 at {args.lat:.2f}, {args.lon:.2f} ...")
    df = fetch_daily_cached(args.lat, args.lon, start, end)
    print(f"  {len(df)} daily rows cached")

    print("Terrain, soil survey and crop history ...")
    ctx = load_site_context(args.lat, args.lon, use_cache=not args.force)
    print(
        f"  terrain: {'OK' if ctx.terrain else 'FAIL'}, soil: {'OK' if ctx.soil else 'FAIL'}, "
        f"crop history: {len(ctx.crop_history)} years"
        + (" (from cache)" if ctx.from_cache else "")
    )

    if args.region:
        print("Neighbouring POWER cells for the regional map ...")
        long, errors = regional_season_metrics(args.lat, args.lon, start, end)
        print(f"  {long[['lat', 'lon']].drop_duplicates().shape[0] if not long.empty else 0} cells cached")
        for label, msg in errors.items():
            print(f"  FAIL {label}: {msg}")


if __name__ == "__main__":
    main()

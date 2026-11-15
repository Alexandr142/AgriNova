"""Everything we know about the *place* (as opposed to the weather): terrain, soil, crop history.

A complete result is cached as JSON in ``data/processed`` so a demo survives a rate-limited
or offline API. Partial results (a service failed) are never cached, so a transient outage
does not become permanent.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agri_nova.data.crop_data import analyze_exhaustion, fetch_crop_history
from agri_nova.data.soil_data import fetch_soil_profile
from agri_nova.data.ssurgo import fetch_ssurgo
from agri_nova.data.terrain import calculate_slope_gradient
from agri_nova.paths import CACHE_DIR
from agri_nova.rotations import CROP_DATABASE, SoilProfile, TerrainProfile


@dataclass
class SiteContext:
    terrain: TerrainProfile | None
    terrain_message: str
    soil: SoilProfile | None
    wss: bool                      # SSURGO answered
    soilgrids: bool                # fallback answered
    soil_extra: dict = field(default_factory=dict)   # component name, hydrologic group, drainage text
    crop_history: list[dict] = field(default_factory=list)
    exhaustion: dict = field(default_factory=dict)
    from_cache: bool = False


def _cache_path(lat: float, lon: float):
    return CACHE_DIR / f"site_{lat:.4f}_{lon:.4f}.json"


def _is_complete(ctx: SiteContext) -> bool:
    """Only cache when every service answered (terrain, a soil source and a crop history without errors)."""
    return (
        ctx.terrain is not None
        and ctx.soil is not None
        and bool(ctx.crop_history)
        and all(item["crop"] != "API Error" for item in ctx.crop_history)
    )


def _to_json(ctx: SiteContext) -> dict:
    return {
        "terrain": ctx.terrain.model_dump() if ctx.terrain else None,
        "terrain_message": ctx.terrain_message,
        "soil": ctx.soil.model_dump() if ctx.soil else None,
        "wss": ctx.wss,
        "soilgrids": ctx.soilgrids,
        "soil_extra": ctx.soil_extra,
        "crop_history": ctx.crop_history,
    }


def _from_json(data: dict) -> SiteContext:
    history = data.get("crop_history", [])
    return SiteContext(
        terrain=TerrainProfile(**data["terrain"]) if data.get("terrain") else None,
        terrain_message=data.get("terrain_message", ""),
        soil=SoilProfile(**data["soil"]) if data.get("soil") else None,
        wss=bool(data.get("wss")),
        soilgrids=bool(data.get("soilgrids")),
        soil_extra=data.get("soil_extra", {}),
        crop_history=history,
        exhaustion=analyze_exhaustion(history, CROP_DATABASE),
        from_cache=True,
    )


def load_site_context(lat: float, lon: float, use_cache: bool = True) -> SiteContext:
    path = _cache_path(lat, lon)
    if use_cache and path.exists():
        try:
            return _from_json(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass  # corrupt cache file: fall through and fetch again

    ctx = _fetch_site_context(lat, lon)
    if _is_complete(ctx):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_to_json(ctx), indent=2), encoding="utf-8")
        except OSError:
            pass
    return ctx


def _fetch_site_context(lat: float, lon: float) -> SiteContext:
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_terrain = pool.submit(calculate_slope_gradient, lat, lon)
        f_ssurgo = pool.submit(fetch_ssurgo, lat, lon)
        f_grids = pool.submit(fetch_soil_profile, lat, lon)
        f_hist = pool.submit(fetch_crop_history, lat, lon)
        terrain_raw, ssurgo, grids, history = (
            f_terrain.result(), f_ssurgo.result(), f_grids.result(), f_hist.result()
        )

    terrain = None
    if terrain_raw.get("available"):
        terrain = TerrainProfile(
            elevation_m=terrain_raw["elevation_m"],
            slope_percent=terrain_raw["slope_percent"],
            risk_level=terrain_raw["risk_level"],
        )

    soil, wss, soilgrids, extra = None, False, False, {}
    if ssurgo:
        soil = SoilProfile(
            awc_mm_100cm=ssurgo["awc_mm_100cm"],
            om_pct_topsoil=ssurgo["om_pct_topsoil"],
            sand_pct=ssurgo["sand_pct"],
            clay_pct=ssurgo["clay_pct"],
            source=ssurgo["source"],
        )
        wss = True
        extra = {"component": ssurgo.get("component"), "hydrologic_group": ssurgo.get("hydrologic_group")}
    if grids.get("available"):
        if soil is None:
            soil = SoilProfile(sand_pct=grids["sand"], clay_pct=grids["clay"], source=grids["source"])
        soilgrids = True
        extra["drainage"] = grids["profile_type"]

    return SiteContext(
        terrain=terrain,
        terrain_message=terrain_raw.get("message", ""),
        soil=soil,
        wss=wss,
        soilgrids=soilgrids,
        soil_extra=extra,
        crop_history=history,
        exhaustion=analyze_exhaustion(history, CROP_DATABASE),
    )

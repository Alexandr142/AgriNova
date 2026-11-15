"""USDA SSURGO soil properties via the Soil Data Access (SDA) web service.

Free, no key. Gives field-specific plant-available water, organic matter, texture and
infiltration class for the dominant soil component at a point - the "local soil
information" layer that satellites cannot provide.
"""
from __future__ import annotations

import httpx

SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/tabular/post.rest"

_SQL = """
SELECT c.cokey, c.compname, c.comppct_r, c.hydgrp,
       ch.hzdept_r, ch.hzdepb_r, ch.awc_r, ch.om_r, ch.sandtotal_r, ch.claytotal_r, ch.ksat_r
FROM mapunit AS mu
INNER JOIN component AS c ON c.mukey = mu.mukey
INNER JOIN chorizon AS ch ON ch.cokey = c.cokey
WHERE mu.mukey IN (SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point ({lon} {lat})'))
  AND c.majcompflag = 'Yes'
ORDER BY c.comppct_r DESC, c.cokey, ch.hzdept_r
"""


def _f(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _overlap(row: dict, lo: float, hi: float) -> float:
    top, bot = _f(row.get("hzdept_r")), _f(row.get("hzdepb_r"))
    if top is None or bot is None:
        return 0.0
    return max(0.0, min(bot, hi) - max(top, lo))


def _weighted_mean(rows: list[dict], field: str, lo: float, hi: float) -> float | None:
    num = den = 0.0
    for r in rows:
        v, w = _f(r.get(field)), _overlap(r, lo, hi)
        if v is not None and w > 0:
            num += v * w
            den += w
    return num / den if den else None


def summarize_horizons(rows: list[dict]) -> dict | None:
    """Reduce SDA rows to the dominant component's key numbers.

    * ``awc_mm_100cm``: plant-available water in the top metre (mm) = sum(AWC cm/cm x thickness cm x 10)
    * ``om_pct_topsoil`` / sand / clay: thickness-weighted means over 0-30 cm
    """
    if not rows:
        return None

    by_component: dict[str, list[dict]] = {}
    for r in rows:
        by_component.setdefault(str(r.get("cokey")), []).append(r)
    cokey = max(by_component, key=lambda k: _f(by_component[k][0].get("comppct_r")) or 0.0)
    horizons = sorted(by_component[cokey], key=lambda r: _f(r.get("hzdept_r")) or 0.0)

    with_awc = [r for r in horizons if _f(r.get("awc_r")) is not None]
    awc_mm = None
    if with_awc:
        awc_mm = sum(_f(r["awc_r"]) * _overlap(r, 0, 100) * 10.0 for r in with_awc)

    return {
        "awc_mm_100cm": awc_mm,
        "om_pct_topsoil": _weighted_mean(horizons, "om_r", 0, 30),
        "sand_pct": _weighted_mean(horizons, "sandtotal_r", 0, 30),
        "clay_pct": _weighted_mean(horizons, "claytotal_r", 0, 30),
        "component": horizons[0].get("compname"),
        "hydrologic_group": horizons[0].get("hydgrp"),
        "source": "USDA SSURGO (Soil Data Access)",
    }


def fetch_ssurgo(lat: float, lon: float) -> dict | None:
    """Query SDA for the point. Returns ``None`` on any failure or if the point has no survey."""
    try:
        resp = httpx.post(
            SDA_URL,
            json={"query": _SQL.format(lat=lat, lon=lon), "format": "JSON+COLUMNNAME"},
            timeout=30.0,
        )
        resp.raise_for_status()
        table = resp.json().get("Table", [])
        if len(table) < 2:
            return None
        columns = table[0]
        rows = [dict(zip(columns, values)) for values in table[1:]]
        return summarize_horizons(rows)
    except Exception:
        return None

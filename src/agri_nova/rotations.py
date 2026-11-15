"""Rotation strategy comparison: turns climate signals and terrain into explainable, hedged guidance."""
from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

SignalStrength = Literal["reference", "supports", "mixed", "limited", "insufficient_data"]

DISCLAIMER = (
    "This tool highlights patterns and possibilities from satellite-derived data. It does not "
    "predict yields and does not replace farmer judgment or local agronomic, NRCS, extension "
    "and crop-insurance advice."
)

DATA_SOURCE = "NASA POWER & SRTM Proxy"
ASSUMPTION_SOURCE = "General agronomy, not calibrated locally"

RADAR_CATEGORIES = [
    'Water Retention',
    'Erosion Control',
    'Nitrogen Optimization',
    'Heat Resilience',
    'Pest Disruption'
]

# ---------- models ----------

class Thresholds(BaseModel):
    min_summer_days: int = 30
    min_fall_days: int = 15
    stress_notable: float = 0.10
    stress_strong: float = 0.25
    heat_notable: float = 0.05
    heat_strong: float = 0.15
    flash_drop_notable: float = 0.08
    flash_drop_strong: float = 0.15
    min_fall_rain_mm_per_30d: float = 20.0

class ClimateSignals(BaseModel):
    start: date
    end: date
    heat_threshold_c: float
    summer_days: int
    summer_water_stress_frac: float | None
    peak_days: int
    peak_heat_frac: float | None
    summer_rain_mm: float
    summer_min_wetness: float | None
    max_14d_wetness_drop: float | None
    longest_dry_spell_days: int
    fall_days: int
    fall_rain_per_30d_mm: float | None

class TerrainProfile(BaseModel):
    elevation_m: float
    slope_percent: float
    risk_level: str

class DataAvailability(BaseModel):
    power: bool = True
    srtm: bool = True  # Added for terrain
    smap: bool = False
    ecostress: bool = False
    openet: bool = False
    cdl: bool = False
    wss: bool = False

class Evidence(BaseModel):
    statement: str
    kind: Literal["data", "assumption"]
    source: str

class StrategyAssessment(BaseModel):
    key: Literal["baseline", "cover_crop", "diversified"]
    name: str
    crops: str
    signal: SignalStrength
    why_suggested: str
    evidence: list[Evidence]
    benefits: list[str]
    information_gaps: list[str]
    expert_review_risks: list[str]
    radar_scores: list[int] = Field(
        description="Dynamic 1-5 scores for: [Water, Erosion, Nitrogen, Heat, Pest]"
    )

# ---------- signals ----------

def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)

def summarize_signals(
    df: pd.DataFrame,
    heat_threshold_c: float = 35.0,
) -> ClimateSignals:
    needed = {"PRECTOTCORR", "GWETROOT", "T2M_MAX", "water_stress", "dry_spell_days"}
    if missing := needed - set(df.columns):
        raise ValueError(f"Run add_indicators() first; missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Empty DataFrame")

    month = df.index.month
    summer = df[month.isin([6, 7, 8])]
    peak = df[month.isin([7, 8])]
    fall = df[month.isin([9, 10, 11])]

    wetness = df["GWETROOT"]
    drop = (wetness.shift(14) - wetness)[month.isin([5, 6, 7, 8, 9])]

    fall_rate = None
    if len(fall) > 0:
        fall_rate = float(fall["PRECTOTCORR"].sum() / len(fall) * 30)

    return ClimateSignals(
        start=df.index.min().date(),
        end=df.index.max().date(),
        heat_threshold_c=heat_threshold_c,
        summer_days=len(summer),
        summer_water_stress_frac=_num(summer["water_stress"].mean()) if len(summer) else None,
        peak_days=len(peak),
        peak_heat_frac=_num((peak["T2M_MAX"] >= heat_threshold_c).mean()) if len(peak) else None,
        summer_rain_mm=float(summer["PRECTOTCORR"].sum()) if len(summer) else 0.0,
        summer_min_wetness=_num(summer["GWETROOT"].min()) if len(summer) else None,
        max_14d_wetness_drop=_num(drop.max()),
        longest_dry_spell_days=int(summer["dry_spell_days"].max()) if len(summer) else 0,
        fall_days=len(fall),
        fall_rain_per_30d_mm=fall_rate,
    )

def _level(value: float | None, notable: float, strong: float) -> int | None:
    if value is None: return None
    if value >= strong: return 2
    if value >= notable: return 1
    return 0

def _exposure_levels(s: ClimateSignals, th: Thresholds) -> dict[str, int | None]:
    return {
        "water": _level(s.summer_water_stress_frac, th.stress_notable, th.stress_strong),
        "heat": _level(s.peak_heat_frac, th.heat_notable, th.heat_strong),
        "flash": _level(s.max_14d_wetness_drop, th.flash_drop_notable, th.flash_drop_strong),
    }

def _overall_level(s: ClimateSignals, th: Thresholds) -> int | None:
    if s.summer_days < th.min_summer_days: return None
    known = [v for v in _exposure_levels(s, th).values() if v is not None]
    return max(known) if known else None

# ---------- strategies ----------

def _baseline(s: ClimateSignals, terrain: TerrainProfile, th: Thresholds) -> StrategyAssessment:
    # Calculate dynamic radar scores based on actual data
    water_penalty = (s.summer_water_stress_frac or 0) * 10
    heat_penalty = (s.peak_heat_frac or 0) * 10

    base_water = max(1, min(5, 5 - int(water_penalty)))
    base_erosion = max(1, min(5, 5 - int(terrain.slope_percent / 2)))
    base_heat = max(1, min(5, 5 - int(heat_penalty)))

    return StrategyAssessment(
        key="baseline",
        name="Baseline: Corn / Soybean",
        crops="Corn, soybean (annual alternation)",
        signal="reference",
        why_suggested="Standard practice reference. Leaves soil bare in winter, exposing it to runoff.",
        evidence=[
            Evidence(statement=f"Field slope of {terrain.slope_percent}% accelerates bare-soil runoff.", kind="data", source=DATA_SOURCE)
        ],
        benefits=["Familiar to local equipment and markets."],
        information_gaps=["No local soil moisture capacity data."],
        expert_review_risks=["Pest adaptation to two-crop cycles."],
        radar_scores=[base_water, base_erosion, 3, base_heat, 2]
    )

def _cover_crop(s: ClimateSignals, terrain: TerrainProfile, th: Thresholds) -> StrategyAssessment:
    base_water = max(1, min(5, 5 - int((s.summer_water_stress_frac or 0) * 10)))
    base_heat = max(1, min(5, 5 - int((s.peak_heat_frac or 0) * 10)))

    # Cover crop massively boosts erosion control if slope is present
    cover_erosion = 5 if terrain.slope_percent > 2 else 4

    return StrategyAssessment(
        key="cover_crop",
        name="Strategy A: Corn / Soy + Cover Crop",
        crops="Corn, soybean, with cereal rye",
        signal="supports" if (s.summer_water_stress_frac or 0) > th.stress_notable else "mixed",
        why_suggested=f"Anchors soil on {terrain.slope_percent}% slopes and buffers rapid drying.",
        evidence=[
            Evidence(statement=f"Largest 14-day wetness drop: {s.max_14d_wetness_drop or 0:.2f}. Cover aids retention.", kind="data", source=DATA_SOURCE)
        ],
        benefits=["Reduces surface evaporation", "Builds organic matter over time"],
        information_gaps=["No satellite validation of local cover crop emergence."],
        expert_review_risks=["Spring water use by rye in dry years."],
        radar_scores=[min(5, base_water + 1), cover_erosion, 4, base_heat, 3]
    )

def _diversified(s: ClimateSignals, terrain: TerrainProfile, th: Thresholds) -> StrategyAssessment:
    base_water = max(1, min(5, 5 - int((s.summer_water_stress_frac or 0) * 10)))
    base_heat = max(1, min(5, 5 - int((s.peak_heat_frac or 0) * 10)))
    base_erosion = max(1, min(5, 5 - int(terrain.slope_percent / 2)))

    return StrategyAssessment(
        key="diversified",
        name="Strategy B: Corn / Soy / Winter Wheat",
        crops="Corn, soybean, winter wheat",
        signal="supports" if (s.peak_heat_frac or 0) > th.heat_notable else "limited",
        why_suggested="Shifts growing window to avoid peak summer heat.",
        evidence=[
            Evidence(statement=f"{s.peak_heat_frac or 0:.0%} of peak days exceeded {s.heat_threshold_c}°C.", kind="data", source=DATA_SOURCE)
        ],
        benefits=["Shifts water demand away from August", "Breaks rootworm cycles"],
        information_gaps=["Market access for small grains unknown."],
        expert_review_risks=["Disease pressure in wet springs."],
        radar_scores=[min(5, base_water + 2), min(5, base_erosion + 1), 3, min(5, base_heat + 2), 5]
    )

# ---------- public API ----------

def assess_strategies(
    signals: ClimateSignals,
    terrain: TerrainProfile,
    thresholds: Thresholds | None = None,
) -> list[StrategyAssessment]:
    th = thresholds or Thresholds()
    return [
        _baseline(signals, terrain, th),
        _cover_crop(signals, terrain, th),
        _diversified(signals, terrain, th),
    ]

def assess_from_frame(
    df: pd.DataFrame,
    terrain: TerrainProfile,
    thresholds: Thresholds | None = None,
    heat_threshold_c: float = 35.0,
) -> list[StrategyAssessment]:
    signals = summarize_signals(df, heat_threshold_c)
    return assess_strategies(signals, terrain, thresholds)
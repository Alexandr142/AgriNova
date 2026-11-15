"""Dynamic Recommendation Engine for crop rotations using climate and terrain data."""
from __future__ import annotations
from datetime import date
from typing import Literal
import pandas as pd
from pydantic import BaseModel, Field

# --- Pydantic Models for Architecture ---
class CropTrait(BaseModel):
    id: str
    name: str
    water_demand: int      # 1 (low) to 5 (high)
    heat_tolerance: int    # 1 (poor) to 5 (excellent)
    erosion_control: int   # 1 (poor) to 5 (excellent root structure)
    nitrogen_fixer: bool

class ClimateSignals(BaseModel):
    start: date
    end: date
    heat_threshold_c: float
    summer_days: int
    summer_water_stress_frac: float
    peak_heat_frac: float
    summer_rain_mm: float
    max_14d_wetness_drop: float
    longest_dry_spell_days: int

class TerrainProfile(BaseModel):
    elevation_m: float
    slope_percent: float
    risk_level: str

class StrategyAssessment(BaseModel):
    name: str
    crops: str
    why_suggested: str
    benefits: list[str]
    radar_scores: list[int] = Field(description="[Water, Erosion, Nitrogen, Heat, Pest]")

RADAR_CATEGORIES = ['Water Retention', 'Erosion Control', 'Nitrogen Optimization', 'Heat Resilience', 'Pest Disruption']

# --- Crop Database ---
CROP_DATABASE = {
    "Corn": CropTrait(id="Corn", name="Corn", water_demand=4, heat_tolerance=3, erosion_control=2, nitrogen_fixer=False),
    "Soybean": CropTrait(id="Soybean", name="Soybean", water_demand=3, heat_tolerance=3, erosion_control=2, nitrogen_fixer=True),
    "Sunflower": CropTrait(id="Sunflower", name="Sunflower", water_demand=5, heat_tolerance=4, erosion_control=2, nitrogen_fixer=False),
    "Sorghum": CropTrait(id="Sorghum", name="Grain Sorghum", water_demand=2, heat_tolerance=5, erosion_control=3, nitrogen_fixer=False),
    "Winter Wheat": CropTrait(id="Winter Wheat", name="Winter Wheat", water_demand=3, heat_tolerance=4, erosion_control=4, nitrogen_fixer=False),
    "Alfalfa": CropTrait(id="Alfalfa", name="Alfalfa (Perennial)", water_demand=3, heat_tolerance=4, erosion_control=5, nitrogen_fixer=True),
    "Cover Rye": CropTrait(id="Cover Rye", name="Winter Rye Cover", water_demand=2, heat_tolerance=3, erosion_control=5, nitrogen_fixer=False),
    "Peas": CropTrait(id="Peas", name="Field Peas", water_demand=2, heat_tolerance=2, erosion_control=2, nitrogen_fixer=True)
}

# --- Signal Processing ---
def summarize_signals(df: pd.DataFrame, heat_threshold_c: float = 35.0) -> ClimateSignals:
    """Extracts critical risk vectors from the NASA timeline."""
    month = df.index.month
    summer = df[month.isin([6, 7, 8])]
    peak = df[month.isin([7, 8])]
    wetness = df["GWETROOT"]
    drop = (wetness.shift(14) - wetness)[month.isin([5, 6, 7, 8, 9])]

    return ClimateSignals(
        start=df.index.min().date(),
        end=df.index.max().date(),
        heat_threshold_c=heat_threshold_c,
        summer_days=len(summer),
        summer_water_stress_frac=float(summer["water_stress"].mean()) if len(summer) else 0.0,
        peak_heat_frac=float((peak["T2M_MAX"] >= heat_threshold_c).mean()) if len(peak) else 0.0,
        summer_rain_mm=float(summer["PRECTOTCORR"].sum()) if len(summer) else 0.0,
        max_14d_wetness_drop=float(drop.max()) if not drop.dropna().empty else 0.0,
        longest_dry_spell_days=int(summer["dry_spell_days"].max()) if len(summer) else 0
    )

# --- Dynamic Assessment Engine ---
def _compute_score(base_trait: int, climate_penalty: float, reverse: bool = False) -> int:
    """Calculates a 1-5 score. If reverse=True, lower trait value is better (e.g. water demand)."""
    if reverse:
        raw = 5.0 - (base_trait - 1) - (climate_penalty * 3.0)
    else:
        raw = base_trait - (climate_penalty * 2.0)
    return max(1, min(5, round(raw)))

def build_strategies(baseline_id: str, signals: ClimateSignals, terrain: TerrainProfile) -> list[StrategyAssessment]:
    """Generates context-aware strategies based on the current crop and satellite data."""
    base_crop = CROP_DATABASE.get(baseline_id, CROP_DATABASE["Corn"])
    strategies = []

    # Penalties derived from NASA & SRTM data (0.0 to ~1.0+)
    water_penalty = signals.summer_water_stress_frac
    heat_penalty = signals.peak_heat_frac
    erosion_penalty = terrain.slope_percent / 5.0  # e.g., 5% slope = 1.0 penalty

    # 1. BASELINE EVALUATION
    base_water_score = _compute_score(base_crop.water_demand, water_penalty, reverse=True)
    base_erosion_score = _compute_score(base_crop.erosion_control, erosion_penalty)
    base_n_score = 4 if base_crop.nitrogen_fixer else 2
    base_heat_score = _compute_score(base_crop.heat_tolerance, heat_penalty)

    strategies.append(StrategyAssessment(
        name=f"Baseline: Continuous {base_crop.name}",
        crops=base_crop.name,
        why_suggested="Current operational baseline derived from user selection.",
        benefits=["Familiar operational workflow and market access."],
        radar_scores=[base_water_score, base_erosion_score, base_n_score, base_heat_score, 1]
    ))

    # 2. EROSION INTERVENTION (Triggered if slope > 2% or baseline erosion control is weak)
    if terrain.slope_percent > 2.0 or base_crop.erosion_control <= 2:
        cover = CROP_DATABASE["Cover Rye"]
        strategies.append(StrategyAssessment(
            name=f"Soil Armor: {base_crop.name} + {cover.name}",
            crops=f"{base_crop.name} followed by {cover.name}",
            why_suggested=f"Topographic slope is {terrain.slope_percent}%. Winter cover anchors the soil and reduces runoff.",
            benefits=["Drastically reduces topsoil loss.", "Builds long-term organic matter."],
            radar_scores=[
                min(5, base_water_score + 1),
                _compute_score(cover.erosion_control, 0), # Cover crop negates slope penalty
                base_n_score,
                base_heat_score,
                2
            ]
        ))

    # 3. DROUGHT/HEAT ADAPTATION (Triggered if water/heat stress is high)
    if signals.summer_water_stress_frac > 0.15 or signals.peak_heat_frac > 0.10:
        # Select Sorghum or Wheat based on what the baseline is NOT
        alt_crop = CROP_DATABASE["Sorghum"] if baseline_id != "Sorghum" else CROP_DATABASE["Winter Wheat"]
        strategies.append(StrategyAssessment(
            name=f"Drought Pivot: Rotate to {alt_crop.name}",
            crops=f"{base_crop.name} / {alt_crop.name}",
            why_suggested=f"Detected {signals.longest_dry_spell_days} days dry spell. {alt_crop.name} has vastly superior heat/drought tolerance.",
            benefits=["Evades peak summer stress.", "Reduces total irrigation/water demand."],
            radar_scores=[
                _compute_score(alt_crop.water_demand, water_penalty * 0.5, reverse=True),
                _compute_score(alt_crop.erosion_control, erosion_penalty),
                base_n_score,
                _compute_score(alt_crop.heat_tolerance, heat_penalty * 0.5),
                4
            ]
        ))

    # 4. SOIL REGENERATION (Triggered if baseline doesn't fix nitrogen)
    if not base_crop.nitrogen_fixer:
        legume = CROP_DATABASE["Alfalfa"] if terrain.slope_percent > 3 else CROP_DATABASE["Soybean"]
        strategies.append(StrategyAssessment(
            name=f"Nitrogen Fix: {base_crop.name} / {legume.name}",
            crops=f"{base_crop.name} rotated with {legume.name}",
            why_suggested="Baseline crop depletes soil nitrogen. Integrating a legume reduces synthetic fertilizer dependency.",
            benefits=["Fixes atmospheric nitrogen.", "Breaks specific pest cycles."],
            radar_scores=[
                _compute_score(legume.water_demand, water_penalty, reverse=True),
                _compute_score(legume.erosion_control, erosion_penalty),
                5, # Max nitrogen score
                _compute_score(legume.heat_tolerance, heat_penalty),
                4
            ]
        ))

    return strategies

def assess_from_frame(df: pd.DataFrame, terrain: TerrainProfile, baseline_id: str) -> list[StrategyAssessment]:
    """Public API for generating the strategy dashboard."""
    signals = summarize_signals(df)
    return build_strategies(baseline_id, signals, terrain)
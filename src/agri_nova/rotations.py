"""Rotation comparison engine.

Always compares exactly three strategies, in a fixed order:

1. ``baseline``     - the farmer's existing rotation (detected from USDA CDL history, editable)
2. ``cover_crop``   - the existing rotation + a winter cereal-rye cover crop
3. ``diversified``  - the existing rotation + one added crop (picked automatically for the
                      farmer's priorities, or chosen by the farmer)

Trait scores (1-5, higher is better on every axis) are *derived from per-crop traits* in
``CROP_DATABASE`` instead of being hard-coded per preset, so any crop combination can be
scored the same way. The trait values are rules of thumb from general agronomy, not measured
data: they are meant to be reviewed with local extension staff.

Every strategy comes with the five sections the challenge asks for: why it was suggested, the
evidence, the benefits, the information gaps and the risks that need local review. Signals are
transparent threshold rules - not a black box and not agronomic advice.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from agri_nova.priorities import CRITERIA, CRITERIA_LABELS, Constraints, Priorities, priority_fit

Signal = Literal["reference", "supports", "mixed", "limited", "insufficient_data"]
StressLevel = Literal["low", "moderate", "high"]

RADAR_CATEGORIES = [CRITERIA_LABELS[c] for c in CRITERIA]

MIN_SUMMER_DAYS = 60   # Jun-Aug days needed before we judge summer stress
MIN_FALL_DAYS = 30     # Sep-Nov days needed before we judge cover-crop establishment


# --------------------------------------------------------------------------- models
class CropTrait(BaseModel):
    id: str
    name: str
    water_demand: int            # 1 (low) to 5 (high)
    heat_tolerance: int          # 1 (poor) to 5 (excellent)
    erosion_control: int         # 1 (poor) to 5 (excellent root structure / residue)
    nitrogen_fixer: bool
    summer_exposure: int = 3     # 1 (done before Jul-Aug) to 5 (most sensitive stage sits in Jul-Aug)
    income: int = 3              # 1-5 typical market strength / stability in the Northern Plains
    simplicity: int = 3          # 1-5, how routine it is on a typical row-crop farm
    pest_group: str = "other"    # crops in the same group share pests and diseases
    needs_small_grain_equipment: bool = False
    needs_elevator: bool = True
    forage: bool = False
    critical_note: str = ""
    story: str = ""
    benefits: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


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
    fall_days: int = 0
    fall_mean_rain_mm_day: float = 0.0
    fall_mean_wetness: float = 0.0
    stress_level: StressLevel = "low"


class TerrainProfile(BaseModel):
    elevation_m: float
    slope_percent: float
    risk_level: str
    available: bool = True


class SoilProfile(BaseModel):
    awc_mm_100cm: float | None = None      # plant-available water in the top metre
    om_pct_topsoil: float | None = None    # organic matter, top 30 cm
    sand_pct: float | None = None
    clay_pct: float | None = None
    source: str = ""


class DataAvailability(BaseModel):
    """Which data layers were actually loaded. Drives the 'information gaps' lists."""

    power: bool = True
    terrain: bool = False
    wss: bool = False        # USDA SSURGO / Web Soil Survey
    soilgrids: bool = False


class StrategyAssessment(BaseModel):
    key: Literal["baseline", "cover_crop", "diversified"]
    name: str
    crops: str
    signal: Signal
    why_suggested: str
    evidence: list[str]
    benefits: list[str]
    information_gaps: list[str]
    expert_review_risks: list[str]
    trait_scores: dict[str, int]
    radar_scores: list[int] = Field(description="Scores in CRITERIA order, for the radar chart")
    historical_support: str = ""
    priority_fit: float | None = None
    rotation: list[str] = Field(default_factory=list)   # cash crops modeled for this strategy
    addon: str | None = None                            # crop added in Strategy B


class _Option(BaseModel):
    """Light stand-in used to rank add-on crops with ``priority_fit``."""

    key: str
    trait_scores: dict[str, int]


# ------------------------------------------------------------------ static knowledge
# Values are judgement-based starting points from general agronomy for the Northern Plains.
CROP_DATABASE: dict[str, CropTrait] = {
    "Corn": CropTrait(
        id="Corn", name="Corn", water_demand=4, heat_tolerance=3, erosion_control=2, nitrogen_fixer=False,
        summer_exposure=5, income=4, simplicity=5, pest_group="warm_grass",
        critical_note="Corn pollination (mid-July) is the most heat- and water-sensitive stage - check local yield history.",
    ),
    "Soybean": CropTrait(
        id="Soybean", name="Soybean", water_demand=3, heat_tolerance=3, erosion_control=2, nitrogen_fixer=True,
        summer_exposure=5, income=4, simplicity=5, pest_group="legume",
        critical_note="Soybean pod set (August) is the most heat- and water-sensitive stage - check local yield history.",
    ),
    "Winter Wheat": CropTrait(
        id="Winter Wheat", name="Winter Wheat", water_demand=3, heat_tolerance=4, erosion_control=4, nitrogen_fixer=False,
        summer_exposure=1, income=3, simplicity=3, pest_group="cool_grass", needs_small_grain_equipment=True,
        critical_note="Winter wheat survival depends on variety, seeding date and snow cover; June grain fill is heat-sensitive.",
        story=(
            "Winter wheat grows mostly in cooler months and is harvested around July, before the peak Jul-Aug heat; "
            "its deep roots protect soil and it breaks corn/soybean pest cycles."
        ),
        benefits=[
            "Uses water in cooler months and is harvested before peak summer heat.",
            "Adds a deep-rooted, soil-protecting crop and an extra income stream.",
            "Breaks corn and soybean pest and disease cycles; wheat after soybean can lower nitrogen needs.",
        ],
        risks=[
            "Winter survival depends on variety, seeding date and snow cover - confirm with local extension trials.",
            "Local wheat price, elevator access and harvest timing may change the economics.",
            "Wet seasons raise wheat disease pressure (for example head blight).",
        ],
    ),
    "Spring Wheat": CropTrait(
        id="Spring Wheat", name="Spring Wheat", water_demand=3, heat_tolerance=2, erosion_control=3, nitrogen_fixer=False,
        summer_exposure=2, income=3, simplicity=3, pest_group="cool_grass", needs_small_grain_equipment=True,
        critical_note="Spring wheat flowering and grain fill (June-July) are heat-sensitive.",
        story=(
            "Spring wheat is seeded early and mostly finishes before the Jul-Aug peak, spreading weather risk and "
            "breaking corn/soybean pest cycles; it is less heat-tolerant than winter wheat at flowering."
        ),
        benefits=[
            "Early-season crop: moves field work and water use away from the Jul-Aug peak.",
            "Breaks corn and soybean pest and disease cycles.",
            "Familiar crop with established markets across the Northern Plains.",
        ],
        risks=[
            "Hot spells during June-July flowering and grain fill can cut yield sharply - check local variety trials.",
            "Seeding date matters: a late spring shortens the cool growing window.",
            "Local wheat price and elevator access may change the economics.",
        ],
    ),
    "Sorghum": CropTrait(
        id="Sorghum", name="Grain Sorghum", water_demand=2, heat_tolerance=5, erosion_control=3, nitrogen_fixer=False,
        summer_exposure=4, income=3, simplicity=4, pest_group="warm_grass",
        critical_note="Sorghum flowering and grain fill (July-August) need enough season length; late planting is risky in the north.",
        story=(
            "Grain sorghum tolerates heat and dry spells better than corn and can pause growth during drought, "
            "then recover when rain returns."
        ),
        benefits=[
            "Heat- and drought-tolerant grain crop that can use less water than corn.",
            "Fits row-crop planting equipment.",
            "Adds a different crop to the sequence (though it shares the warm-season grass family with corn).",
        ],
        risks=[
            "Short growing season in northern South Dakota: hybrid maturity, planting date and frost risk need local guidance.",
            "Cool, slow springs hurt establishment.",
            "Shares some pest pressure with corn, so the pest-cycle benefit is smaller.",
            "Local sorghum market and elevator access may be limited.",
        ],
    ),
    "Sunflower": CropTrait(
        id="Sunflower", name="Sunflower", water_demand=3, heat_tolerance=4, erosion_control=3, nitrogen_fixer=False,
        summer_exposure=4, income=3, simplicity=3, pest_group="broadleaf",
        critical_note="Sunflower flowering (July-August) is the sensitive stage; head diseases rise in wet late seasons.",
        story=(
            "Sunflower has a deep taproot that reaches stored water and tolerates dry spells better than many row "
            "crops, while breaking grass-crop pest cycles."
        ),
        benefits=[
            "Deep taproot can use subsoil water.",
            "Broadleaf crop that breaks corn pest cycles.",
            "Fits row-crop planting equipment.",
        ],
        risks=[
            "Birds, head diseases and wet late seasons need local management.",
            "Needs a sunflower-capable header and a nearby market or crusher.",
            "Sunflower should not follow itself or soybean too closely (shared disease pressure) - check local guidance.",
        ],
    ),
    "Peas": CropTrait(
        id="Peas", name="Field Peas", water_demand=2, heat_tolerance=2, erosion_control=2, nitrogen_fixer=True,
        summer_exposure=2, income=3, simplicity=3, pest_group="legume", needs_small_grain_equipment=True,
        critical_note="Field peas are heat-sensitive at flowering (June).",
        story=(
            "Field peas are an early, cool-season legume harvested around July, so they avoid peak heat and can "
            "add nitrogen for the next crop."
        ),
        benefits=[
            "Harvested around July, before the hottest weeks.",
            "Legume that can cut the nitrogen needs of the next crop.",
            "Spreads workload and weather risk across the season.",
        ],
        risks=[
            "Heat at flowering (June-July) strongly reduces yield.",
            "Shares disease pressure with soybean (both legumes) - avoid a close sequence.",
            "Needs a small-grain-type harvest setup and a local market.",
        ],
    ),
    "Alfalfa": CropTrait(
        id="Alfalfa", name="Alfalfa (Perennial)", water_demand=3, heat_tolerance=4, erosion_control=5, nitrogen_fixer=True,
        summer_exposure=3, income=3, simplicity=2, pest_group="perennial", needs_elevator=False, forage=True,
        critical_note="Alfalfa stand establishment and winter injury decide whether the stand pays off.",
        story=(
            "Alfalfa is a deep-rooted perennial: it protects soil all year, fixes nitrogen and can tap deeper water "
            "in dry spells."
        ),
        benefits=[
            "Perennial cover protects soil from erosion all year.",
            "Fixes nitrogen that can cut fertilizer for the crops that follow.",
            "Hay or forage adds income when you have livestock or a hay market.",
        ],
        risks=[
            "Stand establishment and winter injury risk; a stand usually lasts several years, so it is a longer commitment.",
            "Needs a hay or forage use (livestock or a hay market) to pay off.",
            "Termination and the following crop's planting need local guidance.",
        ],
    ),
    "Cover Rye": CropTrait(
        id="Cover Rye", name="Winter Rye Cover", water_demand=2, heat_tolerance=3, erosion_control=5, nitrogen_fixer=False,
        summer_exposure=1, income=1, simplicity=3, pest_group="cool_grass",
    ),
}

DEFAULT_ROTATION = ["Corn", "Soybean"]
ROTATION_CROPS = [k for k in CROP_DATABASE if k != "Cover Rye"]
ADDON_CANDIDATES = ("Winter Wheat", "Spring Wheat", "Sorghum", "Peas", "Sunflower", "Alfalfa")


def sanitize_rotation(rotation: list[str] | None) -> list[str]:
    """Known cash crops only, no duplicates; falls back to Corn / Soybean when nothing usable is left."""
    out: list[str] = []
    for c in rotation or []:
        if c in CROP_DATABASE and c != "Cover Rye" and c not in out:
            out.append(c)
    return out or list(DEFAULT_ROTATION)


def addon_candidates(existing: list[str]) -> list[str]:
    """Crops that could be added to ``existing`` (a crop already in the rotation is not new)."""
    return [c for c in ADDON_CANDIDATES if c not in existing]


def rotation_label(crops: list[str]) -> str:
    return " / ".join(CROP_DATABASE[c].name for c in crops)


# --------------------------------------------------------------- trait derivation
def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _clamp(v: int) -> int:
    return max(1, min(5, int(v)))


def mean_exposure(crops: list[str]) -> float:
    return sum(CROP_DATABASE[c].summer_exposure for c in crops) / len(crops)


def rotation_traits(crops: list[str]) -> dict[str, int]:
    """Derive the seven 1-5 trait scores of a rotation from its crops (higher is always better).

    * water / heat: low water demand, low Jul-Aug exposure and high heat tolerance score well
    * soil: mean erosion control (residue, rooting, perennials)
    * nitrogen: share of nitrogen-fixing crops, plus a bonus for a diverse rotation with a legume
    * pest: one point per extra crop family (crops of the same family share pests)
    * income / simplicity: mean crop values, minus a penalty for monoculture or extra crops
    """
    crops = sanitize_rotation(crops)
    traits = [CROP_DATABASE[c] for c in crops]
    n = len(traits)
    exposure = sum(t.summer_exposure for t in traits) / n
    demand = sum(t.water_demand for t in traits) / n
    tolerance = sum(t.heat_tolerance for t in traits) / n
    legume_share = sum(t.nitrogen_fixer for t in traits) / n
    groups = {t.pest_group for t in traits}

    nitrogen = 1 + 4 * legume_share
    if n >= 3 and legume_share > 0:
        nitrogen += 1
    income = sum(t.income for t in traits) / n - (1 if n == 1 else 0)
    simplicity = sum(t.simplicity for t in traits) / n - max(0, n - 2)

    raw = {
        "water": 0.5 * (6 - demand) + 0.5 * (6 - exposure),
        "soil": sum(t.erosion_control for t in traits) / n,
        "nitrogen": nitrogen,
        "heat": 0.5 * (6 - exposure) + 0.5 * tolerance,
        "pest": 1 + 2 * (len(groups) - 1),
        "income": income,
        "simplicity": simplicity,
    }
    return {c: _clamp(_round_half_up(raw[c])) for c in CRITERIA}


# --------------------------------------------------------------- signal extraction
def _safe_mean(s: pd.Series) -> float:
    v = s.mean() if len(s) else float("nan")
    return 0.0 if pd.isna(v) else float(v)


def _stress_level(water_frac: float, heat_frac: float) -> StressLevel:
    if water_frac >= 0.15 or heat_frac >= 0.10:
        return "high"
    if water_frac < 0.05 and heat_frac < 0.02:
        return "low"
    return "moderate"


def summarize_signals(df: pd.DataFrame, heat_threshold_c: float = 35.0) -> ClimateSignals:
    """Extract the risk signals from a date-indexed frame that already has indicators."""
    month = df.index.month
    summer = df[month.isin([6, 7, 8])]
    peak = df[month.isin([7, 8])]
    fall = df[month.isin([9, 10, 11])]
    wetness = df["GWETROOT"]
    drop = (wetness.shift(14) - wetness)[month.isin([5, 6, 7, 8, 9])].dropna()

    water_frac = float(summer["water_stress"].mean()) if len(summer) else 0.0
    heat_frac = float((peak["T2M_MAX"] >= heat_threshold_c).mean()) if len(peak) else 0.0

    return ClimateSignals(
        start=df.index.min().date(),
        end=df.index.max().date(),
        heat_threshold_c=heat_threshold_c,
        summer_days=len(summer),
        summer_water_stress_frac=water_frac,
        peak_heat_frac=heat_frac,
        summer_rain_mm=float(summer["PRECTOTCORR"].sum()) if len(summer) else 0.0,
        max_14d_wetness_drop=float(drop.max()) if len(drop) else 0.0,
        longest_dry_spell_days=int(summer["dry_spell_days"].max()) if len(summer) else 0,
        fall_days=len(fall),
        fall_mean_rain_mm_day=_safe_mean(fall["PRECTOTCORR"]) if len(fall) else 0.0,
        fall_mean_wetness=_safe_mean(fall["GWETROOT"]) if len(fall) else 0.0,
        stress_level=_stress_level(water_frac, heat_frac),
    )


def fall_moisture_adequate(s: ClimateSignals) -> bool:
    """Is there likely enough Sep-Nov moisture to establish a winter cover crop?"""
    return (
        s.fall_days >= MIN_FALL_DAYS
        and s.fall_mean_wetness >= 0.30
        and s.fall_mean_rain_mm_day >= 0.8
    )


def _signal_cover(s: ClimateSignals) -> Signal:
    if s.summer_days < MIN_SUMMER_DAYS:
        return "insufficient_data"
    if s.stress_level == "low":
        return "limited"
    return "supports" if fall_moisture_adequate(s) else "mixed"


def _signal_diversified(s: ClimateSignals) -> Signal:
    if s.summer_days < MIN_SUMMER_DAYS:
        return "insufficient_data"
    return {"low": "limited", "moderate": "mixed", "high": "supports"}[s.stress_level]


def strategy_signals(s: ClimateSignals) -> dict[str, Signal]:
    """Signals only (no text) - cheap enough to run for every historical season."""
    return {
        "baseline": "reference",
        "cover_crop": _signal_cover(s),
        "diversified": _signal_diversified(s),
    }


# ------------------------------------------------------------------- scoring rules
def _score_traits(
    key: str,
    existing: list[str],
    addon: str | None,
    s: ClimateSignals,
    terrain: TerrainProfile | None,
    soil: SoilProfile | None,
    constraints: Constraints,
) -> dict[str, int]:
    cash = existing + ([addon] if key == "diversified" and addon else [])
    t = rotation_traits(cash)
    exposure = mean_exposure(cash)

    high = s.stress_level == "high"
    steep = bool(terrain and terrain.available and terrain.slope_percent > 3.0)
    low_awc = bool(soil and soil.awc_mm_100cm is not None and soil.awc_mm_100cm < 120)
    low_om = bool(soil and soil.om_pct_topsoil is not None and soil.om_pct_topsoil < 2.5)
    irrigated = constraints.irrigation_available   # water-saving benefits count for less when water can be added

    # Under high stress, a rotation whose sensitive stages sit in Jul-Aug loses points and one that
    # mostly avoids them gains (the cover crop is judged on its own rules below).
    if high and key != "cover_crop":
        if exposure >= 4.5:
            t["heat"] -= 1
            if not irrigated:
                t["water"] -= 1
        elif exposure <= 3.8 and not irrigated:
            t["water"] += 1

    if key == "baseline":
        if steep:
            t["soil"] -= 1
        if low_awc:
            t["water"] -= 1

    elif key == "cover_crop":
        t["water"] += 1
        t["soil"] += 2
        t["income"] -= 1
        t["simplicity"] -= 1
        if steep:
            t["soil"] += 1
        if low_om:
            t["soil"] += 1
        if low_awc:
            t["water"] += 1
        if high and not fall_moisture_adequate(s):
            t["water"] -= 1          # a poorly established cover crop mainly costs water
        if constraints.livestock_for_forage:
            t["income"] += 1         # grazing / haying the cover crop
        if not constraints.can_seed_cover_crops:
            t["simplicity"] -= 2

    else:  # diversified
        if low_om:
            t["soil"] += 1
        crop = CROP_DATABASE.get(addon) if addon else None
        if crop:
            if crop.needs_small_grain_equipment and not constraints.can_harvest_small_grains:
                t["simplicity"] -= 2
            if crop.needs_elevator and not constraints.wheat_market_nearby:
                t["income"] -= 2
            if crop.forage:
                t["income"] += 1 if constraints.livestock_for_forage else -1

    return {c: _clamp(v) for c, v in t.items()}


def rank_addons(
    existing: list[str],
    s: ClimateSignals,
    terrain: TerrainProfile | None,
    soil: SoilProfile | None,
    constraints: Constraints,
    priorities: Priorities | None,
) -> list[tuple[str, float | None]]:
    """Add-on crops ordered by alignment with the farmer's priorities (0-100). Ties keep list order."""
    existing = sanitize_rotation(existing)
    cands = addon_candidates(existing)
    options = [
        _Option(key=c, trait_scores=_score_traits("diversified", existing, c, s, terrain, soil, constraints))
        for c in cands
    ]
    fits = priority_fit(options, priorities or Priorities())
    return sorted(((c, fits[c]) for c in cands), key=lambda kv: -(kv[1] if kv[1] is not None else 0.0))


# ----------------------------------------------------------------- text generation
def _pct(x: float) -> str:
    return f"{x:.0%}"


def _common_evidence(s: ClimateSignals) -> list[str]:
    if s.summer_days < MIN_SUMMER_DAYS:
        return [f"Only {s.summer_days} summer (Jun-Aug) days are in the selected window - too few to judge summer stress."]
    return [
        f"Water-stress flag on {_pct(s.summer_water_stress_frac)} of {s.summer_days} summer days "
        "(root-zone wetness < 0.35 and < 10 mm rain in the last 14 days).",
        f"Jul-Aug days at or above {s.heat_threshold_c:.0f} C: {_pct(s.peak_heat_frac)}.",
        f"Summer rainfall {s.summer_rain_mm:.0f} mm; longest dry spell {s.longest_dry_spell_days} days; "
        f"largest 14-day drop in root-zone wetness {s.max_14d_wetness_drop:.2f}.",
    ]


def _terrain_soil_evidence(terrain: TerrainProfile | None, soil: SoilProfile | None) -> list[str]:
    out: list[str] = []
    if terrain and terrain.available:
        out.append(f"Terrain (SRTM): slope about {terrain.slope_percent:.1f}% - erosion risk {terrain.risk_level}.")
    if soil:
        parts = []
        if soil.awc_mm_100cm is not None:
            parts.append(f"plant-available water about {soil.awc_mm_100cm:.0f} mm in the top metre")
        if soil.om_pct_topsoil is not None:
            parts.append(f"topsoil organic matter about {soil.om_pct_topsoil:.1f}%")
        if soil.sand_pct is not None and soil.clay_pct is not None:
            parts.append(f"sand {soil.sand_pct:.0f}% / clay {soil.clay_pct:.0f}%")
        if parts:
            out.append(f"Soil ({soil.source or 'soil survey'}): " + "; ".join(parts) + ".")
    return out


def _cover_evidence(s: ClimateSignals) -> list[str]:
    if s.fall_days >= MIN_FALL_DAYS:
        verdict = "enough" if fall_moisture_adequate(s) else "marginal"
        return [
            f"Sep-Nov: {s.fall_mean_rain_mm_day:.1f} mm/day rain and root-zone wetness {s.fall_mean_wetness:.2f} "
            f"- {verdict} moisture to establish a cover crop."
        ]
    return [f"Only {s.fall_days} Sep-Nov days in the window - fall establishment moisture could not be judged."]


def _addon_evidence(
    addon: str | None, chosen_by: str, ranking: list[tuple[str, float | None]], existing: list[str]
) -> list[str]:
    if not addon:
        return ["Your rotation already contains every add-on crop that is modeled, so no extra crop was added."]
    out = [f"Rotation modeled: {rotation_label(existing + [addon])} (added crop: {CROP_DATABASE[addon].name}, "
           f"{'chosen by you' if chosen_by == 'you' else 'picked automatically for your priorities'})."]
    if ranking:
        if all(v is None for _, v in ranking):
            out.append("All priorities are 0, so the add-on crop could not be ranked; the first option was used.")
        else:
            compared = ", ".join(f"{CROP_DATABASE[c].name} {v:.0f}" for c, v in ranking if v is not None)
            out.append(f"Add-on crops compared (alignment with your priorities, 0-100): {compared}.")
    return out


def _why(key: str, signal: Signal, s: ClimateSignals, existing_label: str, addon: str | None) -> str:
    if key == "baseline":
        return f"This is your existing rotation ({existing_label}). It is the reference the other two are compared against."

    if signal == "insufficient_data":
        return "Not enough summer data in the selected window to say whether this option is supported by the observations."

    if key == "cover_crop":
        if signal == "supports":
            return (
                "Summer heat/dryness was detected and fall moisture looks sufficient to establish a winter cover crop. "
                "Residue and roots can slow runoff, keep soil cooler and help bank water for the next dry spell."
            )
        if signal == "mixed":
            if s.fall_days < MIN_FALL_DAYS:
                clause = "but the window has no real Sep-Nov record, so fall establishment is untested."
            else:
                clause = "but fall moisture looks marginal, so the cover crop could struggle to establish or use scarce water."
            return f"Summer stress was detected, which is when a cover crop's soil protection matters, {clause}"
        return (
            "Little summer stress showed up in this window, so the water-banking benefit is weak here. "
            "Erosion and organic-matter benefits may still apply on sloping or low-residue fields."
        )

    # diversified
    if not addon:
        return "Your rotation already includes every add-on crop that is modeled, so no further diversification was tested."
    crop = CROP_DATABASE[addon]
    if signal == "supports":
        return f"Hot/dry conditions hit the corn and soybean peak-water window. {crop.story}"
    if signal == "mixed":
        return (
            f"Moderate summer stress. Adding {crop.name} may spread weather risk, but the observed signal is not "
            f"strong enough to call it clearly better. {crop.story}"
        )
    return (
        f"Little summer stress in this window, so avoiding the Jul-Aug peak is worth less here. "
        f"Diversification benefits (pests, nitrogen, risk spreading) may still apply with {crop.name}."
    )


def _benefits(key: str, constraints: Constraints, addon: str | None) -> list[str]:
    if key == "baseline":
        return [
            "Familiar workflow, equipment and market access.",
            "No extra seed, termination or harvest operations.",
        ]
    if key == "cover_crop":
        out = [
            "Living roots and residue protect soil from erosion and crusting.",
            "Residue mulch cools the soil surface and can reduce evaporation.",
            "Builds organic matter and infiltration over several years.",
        ]
        if constraints.livestock_for_forage:
            out.append("Can be grazed or hayed, adding forage value.")
        return out
    if not addon:
        return ["No additional crop was available to add to your rotation."]
    return list(CROP_DATABASE[addon].benefits)


def _risks(key: str, constraints: Constraints, existing: list[str], addon: str | None) -> list[str]:
    out = _risks_base(key, constraints, existing, addon)
    if constraints.irrigation_available:
        out.append(
            "You indicated irrigation is available: water rights, pumping capacity and cost are not modeled, "
            "and the water-saving scores are reduced because water can be added."
        )
    return out


def _risks_base(key: str, constraints: Constraints, existing: list[str], addon: str | None) -> list[str]:
    if key == "baseline":
        notes = [CROP_DATABASE[c].critical_note for c in existing if CROP_DATABASE[c].critical_note]
        return notes + ["Review crop-insurance coverage and drought contingency with your local agent."]
    if key == "cover_crop":
        risks = [
            "Cereal rye can tie up nitrogen and use spring soil water; termination timing needs local guidance, especially in dry springs.",
            "Seeding date after harvest and fall moisture decide establishment in the Northern Plains.",
            "Check crop-insurance and program rules for cover-crop termination dates and herbicide carry-over.",
        ]
        if not constraints.can_seed_cover_crops:
            risks.append("You indicated no suitable seeding equipment - custom seeding or aerial seeding would be needed.")
        return risks
    if not addon:
        return ["Review your rotation with local extension staff; no additional crop was modeled."]
    crop = CROP_DATABASE[addon]
    risks = list(crop.risks)
    if crop.needs_small_grain_equipment and not constraints.can_harvest_small_grains:
        risks.append(f"You indicated no small-grain harvest capacity - custom harvest would be needed for {crop.name}.")
    if crop.needs_elevator and not constraints.wheat_market_nearby:
        risks.append(f"You indicated no nearby market or elevator - hauling cost could remove the income benefit of {crop.name}.")
    if crop.forage and not constraints.livestock_for_forage:
        risks.append("You indicated no livestock for forage - a hay or forage market would be needed.")
    return risks


# gap = (data flag that closes it | None for gaps no dataset can close, text)
_GAPS: dict[str, list[tuple[str | None, str]]] = {
    "baseline": [
        (None, "Field yield history and insurance records are not included."),
        ("wss", "Field-specific soil water-holding capacity and organic matter (SSURGO) were not loaded."),
        (None, "Soil wetness is modeled (POWER / MERRA-2), not measured by a satellite or a probe in your field."),
        (None, "Field-scale evapotranspiration is not included, so crop water demand is estimated from crop traits, not observed."),
        (None, "The existing rotation comes from a coarse USDA map of one point; check it against your own field records."),
    ],
    "cover_crop": [
        (None, "Local seeding windows, termination dates and spring soil-water effects of the cover crop are not modeled."),
        ("terrain", "Slope could not be measured, so the erosion benefit is unquantified."),
        (None, "Bare-soil periods are not measured (no satellite vegetation data), so ground cover is not observed."),
        ("wss", "Soil organic matter and infiltration class (SSURGO) were not loaded."),
    ],
    "diversified": [
        (None, "Local {addon} survival, variety and seeding-date performance are not modeled."),
        (None, "Local prices, elevator access and input costs for {addon} are not included."),
        (None, "The summer water-demand benefit is inferred from crop traits, not measured with field-scale evapotranspiration."),
        ("wss", "Soil water-holding capacity (SSURGO) was not loaded, so the buffer against dry spells is unknown."),
    ],
}


def _gaps(key: str, available: DataAvailability, addon: str | None) -> list[str]:
    label = CROP_DATABASE[addon].name if addon else "the added crop"
    return [
        text.replace("{addon}", label)
        for flag, text in _GAPS[key]
        if flag is None or not getattr(available, flag)
    ]


# ------------------------------------------------------------------------ assembly
def build_strategies(
    signals: ClimateSignals,
    available: DataAvailability,
    terrain: TerrainProfile | None = None,
    soil: SoilProfile | None = None,
    rotation: list[str] | None = None,
    constraints: Constraints | None = None,
    priorities: Priorities | None = None,
    addon: str | None = "auto",
) -> list[StrategyAssessment]:
    """Three fully explained strategies for ``rotation`` (a list of ``CROP_DATABASE`` ids).

    ``addon`` is the crop added in Strategy B: ``"auto"`` picks the one that best matches
    ``priorities``; a crop id uses that crop (ignored if the rotation already contains it).
    """
    constraints = constraints or Constraints()
    existing = sanitize_rotation(rotation)
    sigs = strategy_signals(signals)

    ranking = rank_addons(existing, signals, terrain, soil, constraints, priorities)
    if addon in (None, "auto") or addon in existing or addon not in ADDON_CANDIDATES:
        chosen = ranking[0][0] if ranking else None
        chosen_by = "auto"
    else:
        chosen, chosen_by = addon, "you"

    existing_label = rotation_label(existing)
    div_crops = existing + ([chosen] if chosen else [])
    names = {
        "baseline": f"Existing: {existing_label}",
        "cover_crop": f"Strategy A: {existing_label} + cereal rye cover crop",
        "diversified": f"Strategy B: {rotation_label(div_crops)}" if chosen
        else f"Strategy B: {existing_label} (no extra crop to add)",
    }
    crops = {
        "baseline": ", ".join(CROP_DATABASE[c].name for c in existing),
        "cover_crop": ", ".join(CROP_DATABASE[c].name for c in existing) + " + Cereal Rye (winter cover)",
        "diversified": ", ".join(CROP_DATABASE[c].name for c in div_crops),
    }
    rotations = {"baseline": existing, "cover_crop": existing, "diversified": div_crops}

    out: list[StrategyAssessment] = []
    for key in ("baseline", "cover_crop", "diversified"):
        evidence = _common_evidence(signals)
        if key == "baseline":
            evidence.append(f"Rotation modeled: {existing_label}.")
        if key == "cover_crop":
            evidence += _cover_evidence(signals)
        if key == "diversified":
            evidence += _addon_evidence(chosen, chosen_by, ranking, existing)
        evidence += _terrain_soil_evidence(terrain, soil)

        addon_for = chosen if key == "diversified" else None
        traits = _score_traits(key, existing, addon_for, signals, terrain, soil, constraints)
        out.append(
            StrategyAssessment(
                key=key,
                name=names[key],
                crops=crops[key],
                signal=sigs[key],
                why_suggested=_why(key, sigs[key], signals, existing_label, addon_for),
                evidence=evidence,
                benefits=_benefits(key, constraints, addon_for),
                information_gaps=_gaps(key, available, addon_for),
                expert_review_risks=_risks(key, constraints, existing, addon_for),
                trait_scores=traits,
                radar_scores=[traits[c] for c in CRITERIA],
                rotation=rotations[key],
                addon=addon_for,
            )
        )
    return out


def assess_from_frame(
    df: pd.DataFrame,
    available: DataAvailability | None = None,
    terrain: TerrainProfile | None = None,
    soil: SoilProfile | None = None,
    rotation: list[str] | None = None,
    constraints: Constraints | None = None,
    heat_threshold_c: float = 35.0,
    priorities: Priorities | None = None,
    addon: str | None = "auto",
) -> list[StrategyAssessment]:
    """Public API: indicator frame in, three fully-explained strategies out."""
    signals = summarize_signals(df, heat_threshold_c)
    return build_strategies(
        signals, available or DataAvailability(), terrain, soil, rotation, constraints, priorities, addon
    )

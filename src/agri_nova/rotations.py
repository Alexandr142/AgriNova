"""Rotation strategy comparison: turns climate signals into explainable, hedged guidance.

Design rules
------------
* Every strategy gets the same five sections the challenge requires: why it was
  suggested, supporting evidence, potential benefits, information gaps, and risks
  that need local expert review.
* Evidence is tagged ``data`` (computed from the loaded dataset) or ``assumption``
  (general agronomic knowledge that has NOT been calibrated for Spink County).
* The "signal" describes how well the observed climate pattern matches the *rationale*
  for a strategy. It is not a yield forecast and not a recommendation to act.
* Information gaps shrink automatically as data sources are marked available in
  ``DataAvailability`` (SMAP, OpenET, CDL, WSS, ...), so the dashboard stays honest
  as the pipeline grows.

Input is the DataFrame returned by ``indicators.add_indicators``.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

SignalStrength = Literal["reference", "supports", "mixed", "limited", "insufficient_data"]
Rating = Literal["low", "moderate", "high"]

DISCLAIMER = (
    "This tool highlights patterns and possibilities from satellite-derived data. It does not "
    "predict yields and does not replace farmer judgment or local agronomic, NRCS, extension "
    "and crop-insurance advice."
)

DATA_SOURCE = "NASA POWER (loaded period)"
ASSUMPTION_SOURCE = "General agronomy, not calibrated for Spink County"


# --------------------------------------------------------------------------- models
class Thresholds(BaseModel):
    """Tunable heuristics. All shares are fractions of observed days (0-1)."""

    min_summer_days: int = 30  # Jun-Aug days needed before we say anything
    min_fall_days: int = 15  # Sep-Nov days needed to judge cover-crop establishment moisture
    stress_notable: float = 0.10  # share of summer days meeting the water-stress rule
    stress_strong: float = 0.25
    heat_notable: float = 0.05  # share of Jul-Aug days at/above the heat threshold
    heat_strong: float = 0.15
    flash_drop_notable: float = 0.08  # 14-day drop in root-zone wetness (0-1 scale)
    flash_drop_strong: float = 0.15
    min_fall_rain_mm_per_30d: float = 20.0  # rough floor for rye establishment moisture


class ClimateSignals(BaseModel):
    start: date
    end: date
    heat_threshold_c: float
    summer_days: int  # observed Jun-Aug days
    summer_water_stress_frac: float | None
    peak_days: int  # observed Jul-Aug days
    peak_heat_frac: float | None
    summer_rain_mm: float
    summer_min_wetness: float | None
    max_14d_wetness_drop: float | None  # growing-season rapid-drying proxy (May-Sep)
    longest_dry_spell_days: int
    fall_days: int  # observed Sep-Nov days
    fall_rain_per_30d_mm: float | None


class DataAvailability(BaseModel):
    """Which datasets the pipeline actually uses. Flip to True as you add them."""

    power: bool = True
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
    ratings: dict[str, Rating] = Field(
        description="Qualitative, general-knowledge ratings; not calibrated locally."
    )


# --------------------------------------------------------------------------- signals
def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def summarize_signals(
    df: pd.DataFrame,
    heat_threshold_c: float = 35.0,
) -> ClimateSignals:
    """Condense an indicator DataFrame into the few numbers the rotation logic uses."""
    needed = {"PRECTOTCORR", "GWETROOT", "T2M_MAX", "water_stress", "dry_spell_days"}
    missing = needed - set(df.columns)
    if missing:
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
        peak_heat_frac=(
            _num((peak["T2M_MAX"] >= heat_threshold_c).mean()) if len(peak) else None
        ),
        summer_rain_mm=float(summer["PRECTOTCORR"].sum()) if len(summer) else 0.0,
        summer_min_wetness=_num(summer["GWETROOT"].min()) if len(summer) else None,
        max_14d_wetness_drop=_num(drop.max()),
        longest_dry_spell_days=int(summer["dry_spell_days"].max()) if len(summer) else 0,
        fall_days=len(fall),
        fall_rain_per_30d_mm=fall_rate,
    )


def _level(value: float | None, notable: float, strong: float) -> int | None:
    if value is None:
        return None
    if value >= strong:
        return 2
    if value >= notable:
        return 1
    return 0


def _exposure_levels(s: ClimateSignals, th: Thresholds) -> dict[str, int | None]:
    return {
        "water": _level(s.summer_water_stress_frac, th.stress_notable, th.stress_strong),
        "heat": _level(s.peak_heat_frac, th.heat_notable, th.heat_strong),
        "flash": _level(s.max_14d_wetness_drop, th.flash_drop_notable, th.flash_drop_strong),
    }


def _overall_level(s: ClimateSignals, th: Thresholds) -> int | None:
    if s.summer_days < th.min_summer_days:
        return None
    known = [v for v in _exposure_levels(s, th).values() if v is not None]
    return max(known) if known else None


def _exposure_summary(s: ClimateSignals, th: Thresholds) -> str:
    lv = _exposure_levels(s, th)
    parts = []
    if lv["water"]:
        parts.append(f"{s.summer_water_stress_frac:.0%} of summer days met the water-stress rule")
    if lv["heat"]:
        parts.append(
            f"{s.peak_heat_frac:.0%} of Jul-Aug days reached {s.heat_threshold_c:.0f} C or more"
        )
    if lv["flash"]:
        parts.append(f"root-zone wetness fell {s.max_14d_wetness_drop:.2f} within 14 days")
    return "; ".join(parts)


def _fall_moisture_ok(s: ClimateSignals, th: Thresholds) -> bool | None:
    if s.fall_days < th.min_fall_days or s.fall_rain_per_30d_mm is None:
        return None
    return s.fall_rain_per_30d_mm >= th.min_fall_rain_mm_per_30d


# --------------------------------------------------------------------------- evidence
def _data_evidence(s: ClimateSignals, th: Thresholds, which: set[str]) -> list[Evidence]:
    out: list[Evidence] = []

    def add(text: str) -> None:
        out.append(Evidence(statement=text, kind="data", source=DATA_SOURCE))

    if "water" in which and s.summer_days >= th.min_summer_days:
        add(
            f"{s.summer_water_stress_frac:.0%} of {s.summer_days} observed Jun-Aug days met the "
            "water-stress rule (dry root zone and under 10 mm rain in the previous 14 days)."
        )
        add(
            f"Longest summer dry spell: {s.longest_dry_spell_days} days under 1 mm rain; "
            f"lowest root-zone wetness: "
            f"{'n/a' if s.summer_min_wetness is None else f'{s.summer_min_wetness:.2f}'} "
            "(0 dry, 1 saturated)."
        )
    if "heat" in which and s.peak_days > 0 and s.peak_heat_frac is not None:
        add(
            f"{s.peak_heat_frac:.0%} of {s.peak_days} observed Jul-Aug days reached "
            f"{s.heat_threshold_c:.0f} C or more."
        )
    if "flash" in which and s.max_14d_wetness_drop is not None:
        note = ""
        if s.max_14d_wetness_drop >= th.flash_drop_notable:
            note = " This rapid-drying pattern is consistent with flash-drought onset (heuristic, not a formal detection)."
        add(
            f"Largest 14-day drop in root-zone wetness during May-Sep: "
            f"{s.max_14d_wetness_drop:.2f}.{note}"
        )
    if "fall" in which:
        if s.fall_rain_per_30d_mm is not None and s.fall_days >= th.min_fall_days:
            add(
                f"Sep-Nov rainfall averaged {s.fall_rain_per_30d_mm:.0f} mm per 30 days over "
                f"{s.fall_days} observed days (rough floor used for rye establishment: "
                f"{th.min_fall_rain_mm_per_30d:.0f} mm)."
            )
        else:
            add("Too few Sep-Nov days in the loaded period to judge fall moisture for seeding.")
    if not out:
        add("Not enough summer days in the loaded period to compute stress signals.")
    return out


def _assumption(text: str) -> Evidence:
    return Evidence(statement=text, kind="assumption", source=ASSUMPTION_SOURCE)


# --------------------------------------------------------------------------- gaps
_GAP_TEXT = {
    "smap": (
        "No SMAP surface soil moisture yet: root-zone wetness here comes from a model-derived "
        "product (NASA POWER), not satellite-observed moisture."
    ),
    "ecostress": (
        "No ECOSTRESS plant-stress overlay. Note ECOSTRESS data only begins in 2018, so it "
        "cannot cover the 2017 flash drought."
    ),
    "openet": (
        "No OpenET field-scale evapotranspiration, so crop-specific summer water demand "
        "differences are assumed rather than measured."
    ),
    "cdl": (
        "No USDA Cropland Data Layer history, so we do not know what has actually been grown "
        "on this field or whether a rotation is already in place."
    ),
    "wss": (
        "No Web Soil Survey soil properties (water-holding capacity, organic matter, "
        "infiltration), which largely determine how much each strategy could help."
    ),
}

_RELEVANT_GAPS = {
    "baseline": ["smap", "openet", "cdl"],
    "cover_crop": ["wss", "smap", "cdl"],
    "diversified": ["openet", "wss", "ecostress", "cdl"],
}


def _gaps(key: str, avail: DataAvailability, s: ClimateSignals) -> list[str]:
    gaps = [_GAP_TEXT[k] for k in _RELEVANT_GAPS[key] if not getattr(avail, k)]
    gaps.append(
        "Weather data is on a coarse regional grid (tens of km), not field-level conditions."
    )
    gaps.append(
        "No economic inputs (seed and input costs, prices, elevator access, crop insurance "
        "rules), which strongly affect whether a rotation is practical."
    )
    if (s.end - s.start).days < 365 * 3:
        gaps.append(
            "The loaded period is short. Judging what is typical versus unusual needs a "
            "longer record, such as 2012-2022."
        )
    return gaps


# --------------------------------------------------------------------------- strategies
def _baseline(s: ClimateSignals, th: Thresholds, av: DataAvailability) -> StrategyAssessment:
    summary = _exposure_summary(s, th)
    why = (
        "Included as the reference point: it is the common practice other options are "
        "compared against, not a recommendation to keep or change anything."
    )
    if summary:
        why += f" In the loaded period, {summary}, which shows how exposed this rotation was."
    return StrategyAssessment(
        key="baseline",
        name="Baseline: Corn / Soybean",
        crops="Corn, soybean (annual alternation)",
        signal="reference",
        why_suggested=why,
        evidence=_data_evidence(s, th, {"water", "heat", "flash"})
        + [
            _assumption(
                "Corn pollination and grain fill fall in roughly July-August, the same window "
                "this tool uses to flag heat and water stress."
            ),
            _assumption(
                "Winter soils are typically bare after corn or soybean harvest, leaving less "
                "cover to reduce evaporation or help water soak in."
            ),
        ],
        benefits=[
            "Familiar to local equipment, markets, insurance and agronomic advice.",
            "Alternating corn and soybean already breaks some pest and disease cycles compared "
            "with continuous corn.",
            "Gives a consistent baseline to judge other options against.",
        ],
        information_gaps=_gaps("baseline", av, s),
        expert_review_risks=[
            "Whether local pest pressure (for example rootworm variants or soybean cyst "
            "nematode) is already adapting to the two-crop rotation.",
            "Sensitivity of the corn crop to heat and dry spells at your planting dates and "
            "hybrid maturity.",
        ],
        ratings={
            "summer_water_demand": "high",
            "organic_matter_input": "moderate",
            "pest_cycle_disruption": "moderate",
            "infiltration_benefit": "low",
        },
    )


def _cover_crop(s: ClimateSignals, th: Thresholds, av: DataAvailability) -> StrategyAssessment:
    levels = _exposure_levels(s, th)
    stress = max(
        [v for k, v in levels.items() if k in ("water", "flash") and v is not None],
        default=None,
    )
    fall_ok = _fall_moisture_ok(s, th)

    if s.summer_days < th.min_summer_days or stress is None:
        signal: SignalStrength = "insufficient_data"
        why = (
            f"Fewer than {th.min_summer_days} Jun-Aug days are loaded, so there is not enough "
            "to judge. Extend the date range to cover a full summer."
        )
    elif stress == 0:
        signal = "limited"
        why = (
            "Water-stress and rapid-drying signals were low in this period, so the data does "
            "not on its own argue for adding a cover crop. Its case is mainly long-term soil "
            "health, which this dataset cannot show."
        )
    elif stress == 2 and fall_ok:
        signal = "supports"
        why = (
            f"Dry-soil signals were strong ({_exposure_summary(s, th)}) and fall moisture looks "
            "adequate for establishment. Soil cover and residue are one way to slow "
            "evaporation and help rainfall soak in, building a buffer for the next dry spell."
        )
    else:
        signal = "mixed"
        reason = (
            "fall moisture for seeding looks low"
            if fall_ok is False
            else "fall moisture for seeding cannot be judged from the loaded period"
            if fall_ok is None
            else "dry-soil signals were present but not extreme"
        )
        why = (
            f"There is some dry-soil signal ({_exposure_summary(s, th) or 'modest'}), but "
            f"{reason}, so the case is plausible rather than clear."
        )

    return StrategyAssessment(
        key="cover_crop",
        name="Strategy A: Corn / Soybean + Winter Cover Crop",
        crops="Corn, soybean, with cereal rye between cash crops",
        signal=signal,
        why_suggested=why,
        evidence=_data_evidence(s, th, {"water", "flash", "fall"})
        + [
            _assumption(
                "Cover crops are widely reported to improve soil cover and infiltration, but "
                "organic-matter gains are gradual and cash-crop yield effects vary by region "
                "and year."
            ),
            _assumption(
                "Rye can use spring soil water and tie up nitrogen ahead of corn if termination "
                "timing is poor."
            ),
        ],
        benefits=[
            "Living roots and residue cover soil outside the cash-crop season, which can reduce "
            "erosion and surface evaporation and support infiltration over several years.",
            "Adds biomass and root inputs that can contribute slowly to soil organic matter.",
            "Keeps the existing cash crops, equipment and markets, so it is a smaller change "
            "than altering the rotation.",
        ],
        information_gaps=_gaps("cover_crop", av, s),
        expert_review_risks=[
            "Seeding window after harvest in the northern plains is short and cold, so "
            "establishment can be poor in some years.",
            "Spring water use by rye before corn planting in dry years, and termination timing.",
            "Nitrogen management for corn following rye.",
            "Crop-insurance and program rules on cover crop termination (check with your agent "
            "and NRCS).",
            "Seed and seeding cost versus expected benefit on your soils.",
        ],
        ratings={
            "summer_water_demand": "high",
            "organic_matter_input": "moderate",
            "pest_cycle_disruption": "moderate",
            "infiltration_benefit": "moderate",
        },
    )


def _diversified(s: ClimateSignals, th: Thresholds, av: DataAvailability) -> StrategyAssessment:
    level = _overall_level(s, th)
    summary = _exposure_summary(s, th)

    if level is None:
        signal: SignalStrength = "insufficient_data"
        why = (
            f"Fewer than {th.min_summer_days} Jun-Aug days are loaded, so there is not enough "
            "to judge. Extend the date range to cover a full summer."
        )
    elif level == 2:
        signal = "supports"
        why = (
            f"Summer stress signals were strong ({summary}). A winter crop that finishes "
            "before peak summer is one option to put less of the rotation in the hottest, "
            "driest window and to break pest and disease cycles."
        )
    elif level == 1:
        signal = "mixed"
        why = (
            f"Some stress signals appeared ({summary}), but not consistently strong, so "
            "wheat's case is plausible rather than compelling for this period."
        )
    else:
        signal = "limited"
        why = (
            "Stress signals were low in this period, so this data does not on its own argue "
            "for changing the rotation. A longer record that includes the 2017 flash drought "
            "may tell a different story."
        )

    return StrategyAssessment(
        key="diversified",
        name="Strategy B: Corn / Soybean / Winter Wheat",
        crops="Corn, soybean, winter wheat (three-year rotation)",
        signal=signal,
        why_suggested=why,
        evidence=_data_evidence(s, th, {"water", "heat", "flash"})
        + [
            _assumption(
                "Winter wheat typically matures and is harvested by mid-summer, before the "
                "late-July and August heat peak that stresses corn at pollination and grain fill."
            ),
            _assumption(
                "A longer gap between host crops can interrupt some pest, weed and disease "
                "cycles, but the benefit depends on the specific pests present."
            ),
        ],
        benefits=[
            "Wheat's main water use occurs earlier in the season, which can shift part of the "
            "rotation away from late-summer heat and dry spells.",
            "A third crop adds a different root system and residue and spreads workload and "
            "weather risk across planting and harvest windows.",
            "May interrupt pest, weed and disease cycles that build up in a two-crop rotation.",
        ],
        information_gaps=_gaps("diversified", av, s),
        expert_review_risks=[
            "Winter survival of wheat in this area, including snow cover and variety "
            "adaptation.",
            "Disease pressure in wet springs (for example Fusarium head blight).",
            "Seeding after soybean harvest, plus equipment and storage needs for a new crop.",
            "Local market access and price for wheat, and effects on crop insurance.",
            "Which specific pests your fields face, since rotation benefits vary by pest.",
        ],
        ratings={
            "summer_water_demand": "moderate",
            "organic_matter_input": "moderate",
            "pest_cycle_disruption": "high",
            "infiltration_benefit": "moderate",
        },
    )


# --------------------------------------------------------------------------- public API
def assess_strategies(
    signals: ClimateSignals,
    available: DataAvailability | None = None,
    thresholds: Thresholds | None = None,
) -> list[StrategyAssessment]:
    """Return the three assessments in a fixed order: baseline, cover crop, diversified."""
    th = thresholds or Thresholds()
    av = available or DataAvailability()
    return [
        _baseline(signals, th, av),
        _cover_crop(signals, th, av),
        _diversified(signals, th, av),
    ]


def assess_from_frame(
    df: pd.DataFrame,
    available: DataAvailability | None = None,
    thresholds: Thresholds | None = None,
    heat_threshold_c: float = 35.0,
) -> list[StrategyAssessment]:
    return assess_strategies(summarize_signals(df, heat_threshold_c), available, thresholds)

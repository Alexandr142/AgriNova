"""Farmer priorities and practical constraints.

The decision-support tool never decides *for* the farmer. Farmers tell it what
matters to them (water, soil, nitrogen cost, income, ...) and what their farm can
physically do (equipment, markets, livestock). These inputs then:

* weight the "alignment with your priorities" score for each rotation, and
* adjust practicality scores / add warnings (e.g. no small-grain combine).
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

# Shared scoring criteria (each strategy gets a 1-5 score on every one of them).
CRITERIA: tuple[str, ...] = (
    "water",
    "soil",
    "nitrogen",
    "heat",
    "pest",
    "income",
    "simplicity",
)

CRITERIA_LABELS: dict[str, str] = {
    "water": "Water conservation",
    "soil": "Soil health & erosion control",
    "nitrogen": "Nitrogen / fertilizer savings",
    "heat": "Heat & drought resilience",
    "pest": "Pest & disease cycle break",
    "income": "Income & market stability",
    "simplicity": "Workload & equipment fit",
}

CRITERIA_HELP: dict[str, str] = {
    "water": "Lower summer water demand, better infiltration and water banking.",
    "soil": "Residue cover, deep roots, organic matter, less topsoil loss.",
    "nitrogen": "Legumes or diversity that reduce purchased nitrogen.",
    "heat": "Crop stages that avoid or tolerate peak-summer heat.",
    "pest": "Rotations that interrupt weed, insect and disease cycles.",
    "income": "Revenue per acre and spreading market/weather risk.",
    "simplicity": "Fits your existing machines, labor and timing.",
}


class Priorities(BaseModel):
    """0 = don't care, 5 = top priority."""

    water: int = Field(3, ge=0, le=5)
    soil: int = Field(3, ge=0, le=5)
    nitrogen: int = Field(3, ge=0, le=5)
    heat: int = Field(3, ge=0, le=5)
    pest: int = Field(3, ge=0, le=5)
    income: int = Field(3, ge=0, le=5)
    simplicity: int = Field(3, ge=0, le=5)

    def weights(self) -> dict[str, int]:
        return {c: int(getattr(self, c)) for c in CRITERIA}


class Constraints(BaseModel):
    """What the farm can practically do. Defaults assume a typical row-crop farm."""

    can_seed_cover_crops: bool = True
    can_harvest_small_grains: bool = True
    wheat_market_nearby: bool = True
    livestock_for_forage: bool = False
    irrigation_available: bool = False


class _Scored(Protocol):
    key: str
    trait_scores: dict[str, int]


def priority_fit(assessments: list[_Scored], priorities: Priorities) -> dict[str, float | None]:
    """Weighted alignment (0-100) between each strategy's trait scores and the farmer's weights.

    Returns ``None`` for every strategy if all weights are zero.
    """
    weights = priorities.weights()
    total = sum(weights.values())
    if total == 0:
        return {a.key: None for a in assessments}

    out: dict[str, float | None] = {}
    for a in assessments:
        score = sum(weights[c] * (a.trait_scores[c] - 1) / 4 for c in CRITERIA)
        out[a.key] = round(100.0 * score / total, 1)
    return out


def top_matches(fit: dict[str, float | None]) -> list[str]:
    """Keys ordered from best to worst alignment (ties keep input order)."""
    scored = [(k, v) for k, v in fit.items() if v is not None]
    return [k for k, _ in sorted(scored, key=lambda kv: -kv[1])]

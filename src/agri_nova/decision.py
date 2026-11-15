"""Turn the three scored strategies into something a farmer can act on.

The scoring engine (``rotations.py``) says *how each option scores*. This module answers the
questions a farmer actually asks next:

* Which option fits what I care about, and what do I give up for it?   -> ``build_verdict``
* Does that answer survive a drier or hotter year?                      -> ``stress_test``
* Can *my farm* actually do it?                                         -> ``practical_checks``
* What do I do on Monday?                                               -> ``next_steps``

Everything here is plain, explainable arithmetic on the scores that are already shown. Nothing
is a yield or profit forecast, and nothing here replaces local advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import pandas as pd

from agri_nova.priorities import CRITERIA, CRITERIA_LABELS, Constraints, Priorities, top_matches

# Below this many alignment points (0-100) two options are treated as a tie: the scores are
# whole-number rules of thumb, so a gap this small is within their noise.
CLOSE_CALL_POINTS = 5.0

SHORT_NAMES: dict[str, str] = {
    "baseline": "Existing rotation",
    "cover_crop": "Strategy A (cover crop)",
    "diversified": "Strategy B (add a crop)",
}


# ----------------------------------------------------------------------------- verdict
@dataclass
class Verdict:
    best_key: str
    runner_up_key: str | None
    gap_to_runner_up: float | None
    close_call: bool
    headline: str
    weather_note: str = ""
    helps: list[str] = field(default_factory=list)   # where the best option beats your current rotation
    costs: list[str] = field(default_factory=list)   # where it is worse than your current rotation


def _weather_note(signal: str) -> str:
    return {
        "supports": "The weather pattern in the season you selected also supports this option.",
        "mixed": "The weather signal for the selected season is mixed, so the fit score matters more than the season.",
        "limited": "The selected season showed little stress, so the water-saving case is weaker than the score suggests.",
        "insufficient_data": "There is not enough summer data in the selected season to check this against the weather.",
    }.get(signal, "")


def build_verdict(assessments: Sequence, priorities: Priorities) -> Verdict | None:
    """Plain-language answer to "which option fits my priorities, and what does it cost me?".

    ``assessments`` must already carry ``priority_fit``. Returns ``None`` when every priority is 0.
    """
    fit = {a.key: a.priority_fit for a in assessments}
    order = top_matches(fit)
    if not order:
        return None

    by_key = {a.key: a for a in assessments}
    best = by_key[order[0]]
    base = by_key.get("baseline")
    runner = by_key[order[1]] if len(order) > 1 else None
    gap = (fit[order[0]] - fit[order[1]]) if runner is not None else None
    close = gap is not None and gap < CLOSE_CALL_POINTS

    helps: list[str] = []
    costs: list[str] = []
    if base is not None and best.key != "baseline":
        weights = priorities.weights()
        diffs = []
        for c in CRITERIA:
            if weights[c] == 0:      # you said you do not care about this one
                continue
            d = best.trait_scores[c] - base.trait_scores[c]
            if d:
                diffs.append((weights[c] * abs(d), c, d))
        diffs.sort(key=lambda t: -t[0])
        for _, c, d in diffs:
            line = f"{CRITERIA_LABELS[c]}: {base.trait_scores[c]} to {best.trait_scores[c]} (out of 5)"
            (helps if d > 0 else costs).append(line)
        helps, costs = helps[:3], costs[:3]

    if best.key == "baseline":
        headline = "Keeping your current rotation fits your priorities best."
        if close and runner is not None:
            headline += f" {runner.name} is close behind, so it is still worth a look."
        weather = ""
    else:
        headline = f"Best fit for your priorities: {best.name}."
        if close and runner is not None:
            headline += (
                f" It is a close call with {runner.name} - the gap is within the noise of these rules of thumb."
            )
        weather = _weather_note(best.signal)

    return Verdict(
        best_key=best.key,
        runner_up_key=runner.key if runner is not None else None,
        gap_to_runner_up=gap,
        close_call=close,
        headline=headline,
        weather_note=weather,
        helps=helps,
        costs=costs,
    )


# --------------------------------------------------------------------------- scorecard
def scorecard(assessments: Sequence, priorities: Priorities) -> pd.DataFrame:
    """One row per criterion (your most important first), one column per strategy, scores 1-5."""
    weights = priorities.weights()
    rows = []
    for c in sorted(CRITERIA, key=lambda c: -weights[c]):
        row: dict[str, object] = {"What matters": CRITERIA_LABELS[c], "Your priority (0-5)": weights[c]}
        for a in assessments:
            row[SHORT_NAMES.get(a.key, a.name)] = a.trait_scores[c]
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- stress test
@dataclass(frozen=True)
class Scenario:
    label: str
    rain_pct: float
    temp_c: float


DEFAULT_STRESS_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("As observed", 0, 0),
    Scenario("Drier (-20% rain)", -20, 0),
    Scenario("Hotter (+2 C)", 0, 2),
    Scenario("Drier and hotter (-20% rain, +2 C)", -20, 2),
    Scenario("Wetter (+20% rain)", 20, 0),
)


def stress_test(
    assess_fn: Callable[[float, float], Sequence],
    scenarios: Sequence[Scenario] = DEFAULT_STRESS_SCENARIOS,
) -> list[dict]:
    """Re-rank the options under several weather cases.

    ``assess_fn(rain_pct, temp_c)`` must return the three assessments with ``priority_fit`` set.
    The first scenario should be the observed weather.
    """
    rows = []
    for sc in scenarios:
        items = assess_fn(sc.rain_pct, sc.temp_c)
        fit = {a.key: a.priority_fit for a in items}
        order = top_matches(fit)
        addon = next((a.addon for a in items if a.key == "diversified"), None)
        rows.append({"label": sc.label, "best_key": order[0] if order else None, "fit": fit, "addon": addon})
    return rows


def stress_table(rows: Sequence[dict], addon_names: Mapping[str, str] | None = None) -> pd.DataFrame:
    addon_names = addon_names or {}
    out = []
    for r in rows:
        row: dict[str, object] = {"Weather case": r["label"]}
        for key, short in SHORT_NAMES.items():
            v = r["fit"].get(key)
            row[short] = None if v is None else round(float(v), 1)
        row["Best fit"] = SHORT_NAMES.get(r["best_key"], "-") if r["best_key"] else "-"
        row["Strategy B adds"] = addon_names.get(r["addon"], r["addon"]) if r["addon"] else "-"
        out.append(row)
    return pd.DataFrame(out)


def stress_summary(rows: Sequence[dict]) -> str:
    ranked = [r for r in rows if r["best_key"]]
    if not ranked:
        return "No ranking is available because every priority is set to 0."
    first = ranked[0]["best_key"]
    flips = [r for r in ranked if r["best_key"] != first]
    if not flips:
        return (
            f"{SHORT_NAMES[first]} stays the best fit in all {len(ranked)} weather cases tested, "
            "so this advice does not hinge on one kind of year."
        )
    detail = "; ".join(f"{SHORT_NAMES[r['best_key']]} under '{r['label']}'" for r in flips)
    return (
        f"The best fit changes with the weather: {SHORT_NAMES[first]} as observed, but {detail}. "
        "Treat this choice as sensitive and talk it through locally."
    )


# ---------------------------------------------------------------------- practical checks
Check = tuple[str, str]   # (status, text); status is "ok", "warn" or "info"


def practical_checks(assessment, constraints: Constraints, crop_db: Mapping | None = None) -> list[Check]:
    """What this option needs from *your* farm, checked against what you ticked in the sidebar."""
    if crop_db is None:
        from agri_nova.rotations import CROP_DATABASE   # local import keeps this module light for tests

        crop_db = CROP_DATABASE

    key = assessment.key
    if key == "baseline":
        out: list[Check] = [("ok", "No new equipment, seed or market is needed - you already run this.")]
        if len(assessment.rotation) == 1:
            out.append(("warn", "One crop every year lets pests and disease build up; this is the weakest option for breaking those cycles."))
        return out

    if key == "cover_crop":
        out = [
            ("ok", "You can seed a cover crop after harvest.")
            if constraints.can_seed_cover_crops
            else ("warn", "No seeding equipment: you would need custom or aerial seeding, which adds cost and timing risk."),
            ("info", "Plan the spring termination before you seed: in a dry spring the cover crop competes with your cash crop for water."),
        ]
        out.append(
            ("ok", "Your livestock can graze or hay it for extra value.")
            if constraints.livestock_for_forage
            else ("info", "With no livestock it is purely a soil and water investment, not a feed source.")
        )
        return out

    crop = crop_db.get(assessment.addon) if assessment.addon else None
    if crop is None:
        return [("info", "Your rotation already includes every add-on crop that is modeled.")]
    out = []
    if crop.needs_small_grain_equipment:
        out.append(
            ("ok", f"You can harvest small grains, which {crop.name} needs.")
            if constraints.can_harvest_small_grains
            else ("warn", f"{crop.name} needs small-grain harvest equipment: budget for custom harvest.")
        )
    if crop.needs_elevator:
        out.append(
            ("ok", f"A market or elevator for {crop.name} is within reach.")
            if constraints.wheat_market_nearby
            else ("warn", f"No market or elevator within reach: hauling could cancel the income benefit of {crop.name}.")
        )
    if crop.forage:
        out.append(
            ("ok", "Your livestock can use the hay or forage.")
            if constraints.livestock_for_forage
            else ("warn", "No livestock: you would need a hay or forage buyer for this crop.")
        )
    out.append(("info", f"Ask your extension agent which {crop.name} variety and seeding date have worked locally."))
    return out


# -------------------------------------------------------------------------- next steps
def next_steps(best_key: str | None) -> list[str]:
    """Generic, safe next actions. They are about checking and trialing, never about committing."""
    if best_key is None:
        return ["Set at least one priority above 0 so the options can be compared."]
    if best_key == "baseline":
        return [
            "Keep your rotation, and ask your insurance agent how a flash drought would be handled for these crops.",
            "Re-run this with the drier and hotter what-if sliders to see how fragile that choice is.",
        ]
    return [
        "Try it on one field first rather than the whole farm, so a wrong guess stays small.",
        "Take this page to your county extension agent, crop consultant or NRCS office and go through the red flags together.",
        "Check insurance and program rules (seeding and termination dates, eligible crops) before you buy seed.",
        "Compare against your own yield history: the data here cannot see your fields.",
    ]

from types import SimpleNamespace

import pandas as pd

from agri_nova.decision import (
    CLOSE_CALL_POINTS,
    Scenario,
    build_verdict,
    next_steps,
    practical_checks,
    scorecard,
    stress_summary,
    stress_table,
    stress_test,
)
from agri_nova.indicators import add_indicators
from agri_nova.priorities import Constraints, Priorities, priority_fit
from agri_nova.rotations import assess_from_frame
from agri_nova.scenarios import apply_scenario


def _raw(rain, wetness, tmax):
    idx = pd.date_range("2017-06-01", "2017-08-31")
    n = len(idx)
    return pd.DataFrame(
        {"PRECTOTCORR": [rain] * n, "T2M": [tmax - 8] * n, "T2M_MAX": [tmax] * n,
         "T2M_MIN": [tmax - 16] * n, "RH2M": [50] * n, "GWETROOT": [wetness] * n},
        index=idx,
    )


def _frame(rain, wetness, tmax):
    return add_indicators(_raw(rain, wetness, tmax))


def _fit(items, priorities):
    fit = priority_fit(items, priorities)
    for a in items:
        a.priority_fit = fit[a.key]
    return items


def _only(**weights):
    base = dict(water=0, soil=0, nitrogen=0, heat=0, pest=0, income=0, simplicity=0)
    base.update(weights)
    return Priorities(**base)


# ----------------------------------------------------------------------- verdict
def test_verdict_names_the_best_option_and_what_it_gains_and_costs():
    p = _only(soil=5, simplicity=5)
    items = _fit(assess_from_frame(_frame(0.0, 0.2, 36), priorities=p), p)
    v = build_verdict(items, p)
    assert v.best_key == "cover_crop"                     # soil +2 outweighs simplicity -1 here
    assert any("Soil health" in line for line in v.helps)
    assert any("Workload" in line for line in v.costs)
    assert v.headline.startswith("Best fit for your priorities: Strategy A")


def test_verdict_ignores_criteria_the_farmer_gave_zero_weight():
    p = _only(soil=5)                                      # simplicity weight 0
    items = _fit(assess_from_frame(_frame(0.0, 0.2, 36), priorities=p), p)
    v = build_verdict(items, p)
    assert not any("Workload" in line for line in v.costs)


def test_verdict_can_recommend_keeping_the_current_rotation():
    p = _only(simplicity=5)                                # the baseline is the simplest
    items = _fit(assess_from_frame(_frame(5.0, 0.6, 25), priorities=p), p)
    v = build_verdict(items, p)
    assert v.best_key == "baseline"
    assert "current rotation" in v.headline
    assert v.helps == [] and v.costs == []


def test_verdict_is_none_when_every_priority_is_zero():
    p = _only()
    items = _fit(assess_from_frame(_frame(5.0, 0.6, 25)), p)
    assert build_verdict(items, p) is None


def test_close_call_is_flagged_when_the_gap_is_small():
    items = [
        SimpleNamespace(key="baseline", name="Existing", priority_fit=50.0, trait_scores={}, signal="reference"),
        SimpleNamespace(key="cover_crop", name="Strategy A", priority_fit=60.0, trait_scores={}, signal="supports"),
        SimpleNamespace(key="diversified", name="Strategy B", priority_fit=60.0 - CLOSE_CALL_POINTS + 1, trait_scores={}, signal="mixed"),
    ]
    for a in items:
        a.trait_scores = {c: 3 for c in Priorities().weights()}
    v = build_verdict(items, Priorities())
    assert v.close_call and "close call" in v.headline


def test_weather_note_warns_when_the_season_showed_little_stress():
    p = _only(water=5)
    items = _fit(assess_from_frame(_frame(5.0, 0.6, 25), priorities=p), p)
    v = build_verdict(items, p)
    assert v.best_key != "baseline"
    assert "little stress" in v.weather_note


# --------------------------------------------------------------------- scorecard
def test_scorecard_lists_the_most_important_criteria_first():
    p = _only(income=5, pest=2)
    items = _fit(assess_from_frame(_frame(0.0, 0.2, 36), priorities=p), p)
    sc = scorecard(items, p)
    assert sc.iloc[0]["What matters"] == "Income & market stability"
    assert sc.iloc[1]["What matters"] == "Pest & disease cycle break"
    assert list(sc.columns) == ["What matters", "Your priority (0-5)", "Existing rotation",
                                "Strategy A (cover crop)", "Strategy B (add a crop)"]
    assert len(sc) == 7


# ------------------------------------------------------------------- stress test
def test_stress_test_reports_a_winner_per_scenario_and_flags_flips():
    def fake(rain, temp):
        best = "cover_crop" if (rain < 0 or temp > 0) else "baseline"
        fit = {"baseline": 40.0, "cover_crop": 40.0, "diversified": 30.0}
        fit[best] = 70.0
        return [SimpleNamespace(key=k, priority_fit=v, addon="Sorghum" if k == "diversified" else None) for k, v in fit.items()]

    rows = stress_test(fake, (Scenario("As observed", 0, 0), Scenario("Drier", -20, 0)))
    assert [r["best_key"] for r in rows] == ["baseline", "cover_crop"]
    text = stress_summary(rows)
    assert "changes with the weather" in text and "Drier" in text
    table = stress_table(rows, {"Sorghum": "Grain Sorghum"})
    assert list(table["Best fit"]) == ["Existing rotation", "Strategy A (cover crop)"]
    assert set(table["Strategy B adds"]) == {"Grain Sorghum"}


def test_stress_summary_says_robust_when_the_winner_never_changes():
    rows = [{"label": s, "best_key": "diversified", "fit": {}, "addon": None} for s in ("a", "b", "c")]
    assert "stays the best fit in all 3" in stress_summary(rows)


def test_stress_test_with_the_real_engine_runs_end_to_end():
    p = Priorities()
    raw = _raw(1.0, 0.45, 32)

    def assess(rain, temp):
        frame = add_indicators(apply_scenario(raw, rain, temp))
        return _fit(assess_from_frame(frame, priorities=p), p)

    rows = stress_test(assess)
    assert len(rows) == 5 and all(r["best_key"] for r in rows)


# --------------------------------------------------------------- practical checks
def _a(key, rotation=("Corn", "Soybean"), addon=None):
    return SimpleNamespace(key=key, rotation=list(rotation), addon=addon)


def test_cover_crop_check_warns_without_a_seeder():
    checks = practical_checks(_a("cover_crop"), Constraints(can_seed_cover_crops=False))
    assert checks[0][0] == "warn" and "seeding" in checks[0][1]
    ok = practical_checks(_a("cover_crop"), Constraints())
    assert ok[0][0] == "ok"


def test_addon_checks_follow_the_crop_and_the_constraints():
    wheat = practical_checks(_a("diversified", addon="Winter Wheat"),
                             Constraints(can_harvest_small_grains=False, wheat_market_nearby=False))
    statuses = [s for s, _ in wheat]
    assert statuses.count("warn") == 2
    alfalfa = practical_checks(_a("diversified", addon="Alfalfa"), Constraints())
    assert any(s == "warn" and "livestock" in t for s, t in alfalfa)          # no livestock by default
    assert not any("market or elevator" in t for _, t in alfalfa)             # alfalfa needs no elevator
    with_cows = practical_checks(_a("diversified", addon="Alfalfa"), Constraints(livestock_for_forage=True))
    assert not any(s == "warn" for s, _ in with_cows)


def test_baseline_check_flags_a_single_crop_rotation():
    assert practical_checks(_a("baseline", rotation=("Corn",)), Constraints())[-1][0] == "warn"
    assert all(s == "ok" for s, _ in practical_checks(_a("baseline"), Constraints()))


def test_next_steps_differ_for_keeping_versus_changing():
    assert any("one field first" in s for s in next_steps("cover_crop"))
    assert not any("one field first" in s for s in next_steps("baseline"))
    assert next_steps(None)

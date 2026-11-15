import pandas as pd

from agri_nova.indicators import add_indicators
from agri_nova.priorities import Constraints, Priorities, priority_fit, top_matches
from agri_nova.rotations import (
    DataAvailability,
    TerrainProfile,
    assess_from_frame,
    summarize_signals,
)


def _frame(start: str, end: str, rain: float, wetness: float, tmax: float) -> pd.DataFrame:
    idx = pd.date_range(start, end)
    n = len(idx)
    df = pd.DataFrame(
        {
            "PRECTOTCORR": [rain] * n,
            "T2M": [tmax - 8] * n,
            "T2M_MAX": [tmax] * n,
            "T2M_MIN": [tmax - 16] * n,
            "RH2M": [50] * n,
            "GWETROOT": [wetness] * n,
        },
        index=idx,
    )
    return add_indicators(df)


def _hot_dry() -> pd.DataFrame:
    return _frame("2017-06-01", "2017-08-31", rain=0.0, wetness=0.2, tmax=36)


def _wet_mild() -> pd.DataFrame:
    return _frame("2017-06-01", "2017-08-31", rain=5.0, wetness=0.6, tmax=25)


def _terrain(slope: float = 0.5) -> TerrainProfile:
    return TerrainProfile(elevation_m=100, slope_percent=slope, risk_level="Low")


# ---------------------------------------------------------------- signals
def test_hot_dry_summer_supports_diversified_but_cover_crop_is_mixed_without_fall_data():
    baseline, cover, diversified = assess_from_frame(_hot_dry(), terrain=_terrain())
    assert baseline.signal == "reference"
    assert diversified.signal == "supports"
    assert cover.signal == "mixed"  # no Sep-Nov record, so fall establishment is untested


def test_hot_dry_summer_with_adequate_fall_rain_supports_cover_crop():
    fall = _frame("2017-09-01", "2017-11-15", rain=2.0, wetness=0.4, tmax=15)
    df = pd.concat([_hot_dry(), fall])
    cover = assess_from_frame(df, terrain=_terrain())[1]
    assert cover.signal == "supports"


def test_hot_dry_summer_with_dry_fall_leaves_cover_crop_mixed():
    fall = _frame("2017-09-01", "2017-11-15", rain=0.1, wetness=0.2, tmax=15)
    df = pd.concat([_hot_dry(), fall])
    cover = assess_from_frame(df, terrain=_terrain())[1]
    assert cover.signal == "mixed"


def test_wet_mild_summer_gives_limited_signal():
    df = _wet_mild()
    assert summarize_signals(df).summer_water_stress_frac == 0
    _, cover, diversified = assess_from_frame(df, terrain=_terrain())
    assert cover.signal == "limited"
    assert diversified.signal == "limited"


def test_too_few_summer_days_is_insufficient_data():
    df = _frame("2017-06-01", "2017-06-10", rain=0.0, wetness=0.2, tmax=36)
    _, cover, diversified = assess_from_frame(df, terrain=_terrain())
    assert cover.signal == "insufficient_data"
    assert diversified.signal == "insufficient_data"


# --------------------------------------------------------------- structure
def test_every_strategy_has_all_required_sections_in_fixed_order():
    result = assess_from_frame(_hot_dry(), terrain=_terrain())
    assert [a.key for a in result] == ["baseline", "cover_crop", "diversified"]
    for a in result:
        assert a.why_suggested
        assert a.evidence
        assert a.benefits
        assert a.information_gaps
        assert a.expert_review_risks
        assert len(a.radar_scores) == 7


def test_exactly_three_strategies_even_for_a_mild_site():
    result = assess_from_frame(_wet_mild())
    assert [a.key for a in result] == ["baseline", "cover_crop", "diversified"]


def test_information_gaps_shrink_as_data_sources_are_added():
    df = _hot_dry()
    before = assess_from_frame(df, terrain=_terrain())[2]
    after = assess_from_frame(
        df, available=DataAvailability(wss=True), terrain=_terrain()
    )[2]
    assert len(after.information_gaps) == len(before.information_gaps) - 1


# ------------------------------------------------------- terrain and scoring
def test_unavailable_terrain_is_not_treated_as_flat():
    unknown = TerrainProfile(elevation_m=0, slope_percent=0, risk_level="Data Unavailable", available=False)
    cover = assess_from_frame(_hot_dry(), terrain=unknown)[1]
    assert not any("slope" in line.lower() for line in cover.evidence)


def test_steep_slope_raises_cover_crop_soil_score_and_lowers_baseline():
    flat = assess_from_frame(_hot_dry(), terrain=_terrain(0.5))
    steep = assess_from_frame(_hot_dry(), terrain=_terrain(6.0))
    assert steep[1].trait_scores["soil"] > flat[1].trait_scores["soil"]
    assert steep[0].trait_scores["soil"] < flat[0].trait_scores["soil"]


def test_no_cover_crop_seeder_lowers_simplicity_and_adds_a_risk():
    default = assess_from_frame(_hot_dry())[1]
    limited = assess_from_frame(_hot_dry(), constraints=Constraints(can_seed_cover_crops=False))[1]
    assert limited.trait_scores["simplicity"] < default.trait_scores["simplicity"]
    assert any("seeding equipment" in r for r in limited.expert_review_risks)


def test_no_wheat_market_lowers_diversified_income_and_adds_a_risk():
    # Pin the crop: with "auto" the picker would simply steer away from crops that need an elevator.
    default = assess_from_frame(_hot_dry(), addon="Winter Wheat")[2]
    limited = assess_from_frame(_hot_dry(), addon="Winter Wheat", constraints=Constraints(wheat_market_nearby=False))[2]
    assert limited.trait_scores["income"] < default.trait_scores["income"]
    assert any("market or elevator" in r for r in limited.expert_review_risks)


def test_irrigation_softens_water_scores_and_adds_a_caveat():
    dry = _hot_dry()
    rain_fed = assess_from_frame(dry)
    irrigated = assess_from_frame(dry, constraints=Constraints(irrigation_available=True))
    assert irrigated[0].trait_scores["water"] > rain_fed[0].trait_scores["water"]   # baseline penalty removed
    assert irrigated[2].trait_scores["water"] < rain_fed[2].trait_scores["water"]   # wheat benefit smaller
    assert any("irrigation" in r.lower() for r in irrigated[2].expert_review_risks)
    assert not any("irrigation" in r.lower() for r in rain_fed[2].expert_review_risks)


def test_trait_scores_stay_between_one_and_five():
    for df in (_hot_dry(), _wet_mild()):
        for a in assess_from_frame(df, constraints=Constraints(can_seed_cover_crops=False, can_harvest_small_grains=False, wheat_market_nearby=False)):
            assert all(1 <= v <= 5 for v in a.trait_scores.values())


# ------------------------------------------------------------- priorities
def test_priority_fit_follows_the_farmers_weights():
    # Wet mild summer, Winter Wheat as the added crop: no stress adjustments, so the water scores are
    # existing Corn/Soybean 2, + cover crop 3, + Winter Wheat 3 (both beat the existing rotation).
    result = assess_from_frame(_wet_mild(), addon="Winter Wheat")
    fit = priority_fit(result, Priorities(water=5, soil=0, nitrogen=0, heat=0, pest=0, income=0, simplicity=0))
    assert fit == {"baseline": 25.0, "cover_crop": 50.0, "diversified": 50.0}
    assert top_matches(fit)[-1] == "baseline"


def test_priority_fit_is_none_when_every_weight_is_zero():
    result = assess_from_frame(_wet_mild())
    fit = priority_fit(result, Priorities(water=0, soil=0, nitrogen=0, heat=0, pest=0, income=0, simplicity=0))
    assert all(v is None for v in fit.values())
    assert top_matches(fit) == []


# ------------------------------------------------------- rotation and add-on choice
def test_addon_is_never_a_crop_already_in_the_rotation():
    for rot in (["Corn", "Soybean"], ["Corn", "Soybean", "Winter Wheat"], ["Sunflower"]):
        div = assess_from_frame(_hot_dry(), rotation=rot)[2]
        assert div.addon not in rot
        assert div.rotation == rot + [div.addon]


def test_chosen_addon_is_used_and_one_already_in_the_rotation_is_ignored():
    chosen = assess_from_frame(_hot_dry(), addon="Sorghum")[2]
    assert chosen.addon == "Sorghum" and "Grain Sorghum" in chosen.name
    ignored = assess_from_frame(_hot_dry(), rotation=["Corn", "Sorghum"], addon="Sorghum")[2]
    assert ignored.addon != "Sorghum"


def test_a_custom_rotation_flows_into_every_strategy():
    base, cover, _ = assess_from_frame(_hot_dry(), rotation=["Corn", "Spring Wheat", "Sunflower"])
    assert base.rotation == ["Corn", "Spring Wheat", "Sunflower"]
    assert cover.rotation == base.rotation
    assert "Spring Wheat" in base.name


def test_unknown_or_empty_rotation_falls_back_to_corn_and_soybean():
    assert assess_from_frame(_hot_dry(), rotation=[])[0].rotation == ["Corn", "Soybean"]
    assert assess_from_frame(_hot_dry(), rotation=["Banana"])[0].rotation == ["Corn", "Soybean"]


def test_priorities_steer_the_automatic_addon_choice():
    heat = Priorities(water=0, soil=0, nitrogen=0, heat=5, pest=0, income=0, simplicity=0)
    simple = Priorities(water=0, soil=0, nitrogen=0, heat=0, pest=0, income=0, simplicity=5)
    assert assess_from_frame(_hot_dry(), priorities=heat)[2].addon != assess_from_frame(_hot_dry(), priorities=simple)[2].addon


def test_automatic_addon_adapts_when_there_is_no_market_for_new_crops():
    stocked = assess_from_frame(_hot_dry(), priorities=Priorities(income=5))[2]
    no_market = assess_from_frame(
        _hot_dry(), priorities=Priorities(income=5), constraints=Constraints(wheat_market_nearby=False)
    )[2]
    assert no_market.trait_scores["income"] >= stocked.trait_scores["income"] - 1   # it dodged the -2 penalty
    assert not any("market or elevator" in r for r in no_market.expert_review_risks)

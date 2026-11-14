import pandas as pd

from agri_nova.indicators import add_indicators
from agri_nova.rotations import (
    DataAvailability,
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


def test_dry_hot_summer_supports_diversified_and_cover_crop_is_mixed_without_fall_data():
    df = _frame("2017-06-01", "2017-08-31", rain=0.0, wetness=0.2, tmax=36)
    baseline, cover, diversified = assess_from_frame(df)
    assert baseline.signal == "reference"
    assert diversified.signal == "supports"
    # strong stress but no Sep-Nov data, so fall moisture cannot be judged
    assert cover.signal == "mixed"


def test_dry_summer_with_adequate_fall_rain_supports_cover_crop():
    summer = _frame("2017-06-01", "2017-08-31", rain=0.0, wetness=0.2, tmax=36)
    fall = _frame("2017-09-01", "2017-11-15", rain=2.0, wetness=0.4, tmax=15)
    df = pd.concat([summer, fall])
    cover = assess_from_frame(df)[1]
    assert cover.signal == "supports"


def test_wet_mild_summer_gives_limited_signal():
    df = _frame("2017-06-01", "2017-08-31", rain=5.0, wetness=0.6, tmax=25)
    signals = summarize_signals(df)
    assert signals.summer_water_stress_frac == 0
    _, cover, diversified = assess_from_frame(df)
    assert cover.signal == "limited"
    assert diversified.signal == "limited"


def test_too_few_summer_days_is_insufficient_data():
    df = _frame("2017-06-01", "2017-06-10", rain=0.0, wetness=0.2, tmax=36)
    _, cover, diversified = assess_from_frame(df)
    assert cover.signal == "insufficient_data"
    assert diversified.signal == "insufficient_data"


def test_every_strategy_has_all_required_sections_in_fixed_order():
    df = _frame("2017-06-01", "2017-08-31", rain=0.0, wetness=0.2, tmax=36)
    result = assess_from_frame(df)
    assert [a.key for a in result] == ["baseline", "cover_crop", "diversified"]
    for a in result:
        assert a.why_suggested
        assert a.evidence
        assert a.benefits
        assert a.information_gaps
        assert a.expert_review_risks


def test_information_gaps_shrink_as_data_sources_are_added():
    df = _frame("2017-06-01", "2017-08-31", rain=0.0, wetness=0.2, tmax=36)
    before = assess_from_frame(df)[2]
    after = assess_from_frame(df, available=DataAvailability(openet=True, wss=True))[2]
    assert len(after.information_gaps) == len(before.information_gaps) - 2

import numpy as np
import pandas as pd

from agri_nova.indicators import add_indicators


def test_heat_stress_flag():
    idx = pd.date_range("2026-07-01", periods=3)
    df = pd.DataFrame(
        {
            "PRECTOTCORR": [0.0, 0.0, 5.0],
            "T2M": [30, 31, 20],
            "T2M_MAX": [36, 37, 25],
            "T2M_MIN": [20, 21, 15],
            "RH2M": [40, 40, 90],
            "GWETROOT": [0.2, 0.2, 0.5],
        },
        index=idx,
    )
    out = add_indicators(df)
    assert out["heat_stress"].tolist() == [1, 1, 0]
    assert not np.isnan(out["gdd_cum"]).any()

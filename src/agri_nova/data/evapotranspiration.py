"""Plant heat-load chart.

``plot_heat_load_proxy`` is a *model-based* proxy from NASA POWER: how much hotter the
land-surface skin is than the air at the daily maximum. When plants transpire freely the
surface stays cooler; a large positive gap is a hint of water stress. It is NOT a canopy
temperature measurement and the +3 C line is a rule of thumb, not a calibrated threshold.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

HEAT_LOAD_THRESHOLD_C = 3.0


def plot_heat_load_proxy(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or "TS_MAX" not in df.columns or "T2M_MAX" not in df.columns:
        fig.update_layout(title="Skin temperature (TS_MAX) not available")
        return fig

    gap = df["TS_MAX"] - df["T2M_MAX"]
    over = gap.where(gap > HEAT_LOAD_THRESHOLD_C)

    fig.add_hline(
        y=HEAT_LOAD_THRESHOLD_C,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Rule-of-thumb line (+{HEAT_LOAD_THRESHOLD_C:.0f} C)",
    )
    fig.add_trace(go.Scatter(x=gap.index, y=gap, mode="lines", line=dict(color="#444", width=1), name="Skin - air (C)"))
    fig.add_trace(go.Scatter(x=over.index, y=over, mode="markers", marker=dict(color="crimson", size=5), name="Above the line"))
    fig.update_layout(
        title="Heat-load proxy: surface skin temperature minus air temperature (POWER model)",
        yaxis_title="Difference at daily max (C)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

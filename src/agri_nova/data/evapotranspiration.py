"""Evaporative stress detection using Canopy Temperature Depression (CTD) physics."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go


def plot_evaporative_stress(df: pd.DataFrame) -> go.Figure:
    if df.empty or "TS_MAX" not in df.columns or "T2M_MAX" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Awaiting Skin Temperature (TS_MAX) Data")
        return fig

    df_stress = df.copy()
    df_stress["thermal_diff"] = df_stress["TS_MAX"] - df_stress["T2M_MAX"]

    # Калібрування: допускаємо нагрівання ґрунту на 3 градуси вище повітря
    # Все, що вище +3.0 - це вже критичне закриття продихів рослини
    STRESS_THRESHOLD = 3.0

    fig = go.Figure()

    fig.add_hline(y=STRESS_THRESHOLD, line_dash="dash", line_color="gray",
                  annotation_text=f"Stress Threshold (+{STRESS_THRESHOLD}°C)")

    # Здорова транспірація (до порогу стресу)
    healthy = df_stress["thermal_diff"].copy()
    healthy[healthy > STRESS_THRESHOLD] = STRESS_THRESHOLD

    fig.add_trace(go.Scatter(
        x=healthy.index, y=healthy,
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 200, 100, 0.4)',
        name="Normal ET / Cooling"
    ))

    # Зона стресу (вище порогу)
    stress = df_stress["thermal_diff"].copy()
    stress[stress <= STRESS_THRESHOLD] = STRESS_THRESHOLD

    fig.add_trace(go.Scatter(
        x=stress.index, y=stress,
        fill='tonexty',
        mode='none',
        fillcolor='rgba(255, 50, 50, 0.6)',
        name="Severe Stress (Stomata Closed)"
    ))

    fig.add_trace(go.Scatter(
        x=df_stress.index, y=df_stress["thermal_diff"],
        mode='lines',
        line=dict(color="black", width=1),
        name="Skin - Air Temp (°C)"
    ))

    fig.update_layout(
        title="Evaporative Stress Proxy (Calibrated Canopy Temp Depression)",
        yaxis_title="Temp Difference (°C)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
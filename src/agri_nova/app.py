"""Run with: uv run streamlit run src/agri_nova/app.py"""
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import streamlit as st
from streamlit_folium import st_folium

from agri_nova.data.crop_data import fetch_crop_history, analyze_exhaustion
from agri_nova.data.power import fetch_daily
from agri_nova.data.terrain import calculate_slope_gradient
from agri_nova.indicators import add_indicators
from agri_nova.rotations import StrategyAssessment, assess_from_frame, RADAR_CATEGORIES, TerrainProfile, CROP_DATABASE

SIGNAL_LABELS = {
    "reference": ("info", "Reference point"),
    "supports": ("success", "Data pattern supports considering this"),
    "mixed": ("warning", "Mixed signal"),
    "limited": ("info", "Limited signal in this period"),
    "insufficient_data": ("warning", "Not enough data in this period"),
}

CROP_PROFILES = {
    "Corn": {
        "critical_start": "07-15",
        "critical_end": "08-10",
        "heat_threshold": 35.0,
        "phase_name": "Pollination"
    },
    "Soybean": {
        "critical_start": "08-01",
        "critical_end": "08-25",
        "heat_threshold": 32.0,
        "phase_name": "Pod Set"
    },
    "Winter Wheat": {
        "critical_start": "05-15",
        "critical_end": "06-15",
        "heat_threshold": 30.0,
        "phase_name": "Flowering & Grain Fill"
    }
}


st.set_page_config(page_title="Agri Nova", page_icon="🌾", layout="wide")
st.title("🌾 Agri Nova: The Flash Drought Pivot")
st.caption("Built on NASA POWER data so far. Prototype heuristics, not agronomic advice.")

with st.sidebar:
    st.header("📍 Choose field")
    st.caption("Click on map to update location")

    # Make map and center about USA Midwest
    m = folium.Map(location=[41.5000, -93.5000], zoom_start=5)

    # Add satellite layer from Esri data
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Display map in sidebar and get coords
    map_data = st_folium(m, height=300, width="content")

    # Handle coords from map
    if map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
    else:
        lat, lon = 41.5000, -93.5000  # Default for USA Midwest

    st.success(f"Coordinates: {lat:.4f}, {lon:.4f}")

    selected_crop = st.selectbox("Current Target Crop", list(CROP_PROFILES.keys()))

    # Demo dates for drought in Iowa 2017
    start = st.date_input("Start date", value=date(2017, 4, 1))
    end = st.date_input("End date", value=date(2017, 11, 15))


@st.cache_data(show_spinner="Running Earth Observation Pipeline...", ttl=3600)
def execute_pipeline(lat, lon, start, end):
    # 1. Fetch climate (NASA POWER)
    weather_df = fetch_daily(lat, lon, start, end)
    processed_df = add_indicators(weather_df)

    # 2. Fetch terrain (SRTM proxy)
    terrain_data = calculate_slope_gradient(lat, lon)

    # 3. Fetch crop data (CDL)
    crop_history = fetch_crop_history(lat, lon)
    soil_analysis = analyze_exhaustion(crop_history, CROP_DATABASE)

    return processed_df, terrain_data, soil_analysis


try:
    df, terrain_data, soil_status = execute_pipeline(lat, lon, start, end)
except Exception as exc:  # Network / API errors
    st.error(f"Could not load data: {exc}")
    st.stop()


def plot_radar_chart(assessments: list[StrategyAssessment]):
    fig = go.Figure()
    colors = ['#EF553B', '#00CC96', '#AB63FA']

    for i, strat in enumerate(assessments):
        fig.add_trace(go.Scatterpolar(
            r=strat.radar_scores,
            theta=RADAR_CATEGORIES,
            fill='toself',
            name=strat.name,
            line_color=colors[i % len(colors)]
        ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])))
    return fig


def plot_vulnerability_window(df, profile):
    """Plots temperature with critical crop phase highlighting."""
    fig = px.line(df, y=["T2M_MAX", "T2M_MIN"], title=f"Temperature Stress: {profile['phase_name']}")
    if df.empty:
        return fig

    target_year = df.index.year.value_counts().idxmax()
    start_window = f"{target_year}-{profile['critical_start']}"
    end_window = f"{target_year}-{profile['critical_end']}"

    fig.add_vrect(
        x0=start_window, x1=end_window, fillcolor="red", opacity=0.15, layer="below", line_width=0,
        annotation_text="Critical Phase", annotation_position="top left"
    )
    fig.add_hline(
        y=profile["heat_threshold"], line_dash="dot", line_color="red",
        annotation_text=f"Heat Limit ({profile['heat_threshold']}°C)"
    )
    return fig


c1, c2, c3 = st.columns(3)
c1.metric("Water-Stress Days", int(df["water_stress"].sum()))
c2.metric("Heat-Stress Days", int(df["heat_stress"].sum()))

if "disease_risk_7d" in df.columns and not df.empty:
    c3.metric("Disease Risk (Last 7d)", f"{df['disease_risk_7d'].iloc[-1]:.0%}")
else:
    c3.metric("Disease Risk", "N/A")

col_hist, col_diag = st.columns(2)
with col_hist:
    st.markdown("#### USDA Satellite Crop History (Last 5 Years)")
    history_df = pd.DataFrame(soil_status.get("raw_history", []))
    if not history_df.empty:
        st.dataframe(history_df, width="content", hide_index=True)

with col_diag:
    st.markdown("#### Automated Soil Diagnostics")
    st.write(f"**Detected Baseline Crop:** {soil_status['baseline_crop']}")
    st.write(f"**Nitrogen Status:** {soil_status['nitrogen_level']}")
    st.write(f"**Soil Exhaustion Level:** {soil_status['exhaustion_status']}")

st.plotly_chart(plot_vulnerability_window(df, CROP_PROFILES[selected_crop]), width="content")

st.header("Rotation Options Assessment")
terrain_profile = TerrainProfile(**terrain_data)
assessments = assess_from_frame(df, terrain_profile, selected_crop)
st.plotly_chart(plot_radar_chart(assessments), width="content")

st.header("Climate & Vegetation Context")

c_left, c_right = st.columns(2)
with c_left:
    st.plotly_chart(
        px.bar(df, y="PRECTOTCORR", title="Daily Precipitation (mm)"),
        width="content"
    )
with c_right:
    st.plotly_chart(
        px.line(df, y="GWETROOT", title="Root-Zone Soil Wetness (0 = Dry, 1 = Saturated)"),
        width="content"
    )

with st.expander("Explore Raw Data"):
    st.dataframe(df)

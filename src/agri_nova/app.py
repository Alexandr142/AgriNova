"""Run with: uv run streamlit run src/agri_nova/app.py"""
from datetime import date

import plotly.express as px
import folium
import streamlit as st
from streamlit_folium import st_folium

from agri_nova.data.power import fetch_daily
from agri_nova.indicators import add_indicators
from agri_nova.rotations import DISCLAIMER, StrategyAssessment, assess_from_frame

SIGNAL_LABELS = {
    "reference": ("info", "Reference point"),
    "supports": ("success", "Data pattern supports considering this"),
    "mixed": ("warning", "Mixed signal"),
    "limited": ("info", "Limited signal in this period"),
    "insufficient_data": ("warning", "Not enough data in this period"),
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
    map_data = st_folium(m, height=300, use_container_width=True)

    # Handle coords from map
    if map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
    else:
        lat, lon = 41.5000, -93.5000  # Default for USA Midwest

    st.success(f"Coordinates: {lat:.4f}, {lon:.4f}")

    # Demo dates
    start = st.date_input("Start date", value=date(2017, 4, 1))
    end = st.date_input("End date", value=date(2017, 11, 15))


@st.cache_data(show_spinner="Fetching NASA POWER data...", ttl=3600)
def load(lat: float, lon: float, start: date, end: date):
    return add_indicators(fetch_daily(lat, lon, start, end))


try:
    df = load(lat, lon, start, end)
except Exception as exc:  # Network / API errors
    st.error(f"Could not load data: {exc}")
    st.stop()


def render_assessment(a: StrategyAssessment) -> None:
    level, label = SIGNAL_LABELS[a.signal]
    getattr(st, level)(f"**{label}**")

    st.markdown("#### Why this was suggested")
    st.write(a.why_suggested)

    st.markdown("#### Supporting evidence")
    for e in a.evidence:
        icon = "📊" if e.kind == "data" else "📘"
        st.markdown(f"- {icon} {e.statement}  \n  <sub>{e.source}</sub>", unsafe_allow_html=True)

    st.markdown("#### Potential benefits")
    for b in a.benefits:
        st.markdown(f"- {b}")

    st.markdown("#### Information gaps")
    for g in a.information_gaps:
        st.markdown(f"- {g}")

    st.markdown("#### Risks needing local expert review")
    for r in a.expert_review_risks:
        st.markdown(f"- {r}")

    st.markdown("#### Qualitative profile")
    cols = st.columns(len(a.ratings))
    for col, (name, rating) in zip(cols, a.ratings.items(), strict=True):
        col.metric(name.replace("_", " ").capitalize(), rating.capitalize())
    st.caption("General-knowledge ratings, not calibrated for Spink County.")


c1, c2, c3 = st.columns(3)
c1.metric("Water-stress days", int(df["water_stress"].sum()))
c2.metric("Heat-stress days", int(df["heat_stress"].sum()))
c3.metric("Disease risk (last 7d)", f"{df['disease_risk_7d'].iloc[-1]:.0%}")

st.header("Rotation options")
st.info(DISCLAIMER)

assessments = assess_from_frame(df)
tabs = st.tabs([a.name for a in assessments])
for tab, assessment in zip(tabs, assessments, strict=True):
    with tab:
        render_assessment(assessment)

st.header("Climate context")
st.plotly_chart(
    px.bar(df, y="PRECTOTCORR", title="Daily precipitation (mm)"), use_container_width=True
)
st.plotly_chart(
    px.line(df, y="GWETROOT", title="Root-zone soil wetness (0 dry, 1 saturated)"),
    use_container_width=True,
)
st.plotly_chart(
    px.line(df, y=["T2M_MAX", "T2M_MIN"], title="Temperature (C)"), use_container_width=True
)

with st.expander("Raw data"):
    st.dataframe(df)

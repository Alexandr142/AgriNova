"""Run with: uv run streamlit run src/agri_nova/app.py"""
from datetime import date, timedelta

import plotly.express as px
import streamlit as st

from agri_nova.data.power import fetch_daily
from agri_nova.indicators import add_indicators

st.set_page_config(page_title="Field Shift", page_icon="🌾", layout="wide")
st.title("🌾 Field Shift: farm climate dashboard")
st.caption("Built on NASA POWER data. Prototype heuristics, not agronomic advice.")

with st.sidebar:
    st.header("Field")
    lat = st.number_input("Latitude", value=50.15, format="%.4f")
    lon = st.number_input("Longitude", value=30.05, format="%.4f")
    end = st.date_input("End date", value=date.today() - timedelta(days=3))
    start = st.date_input("Start date", value=end - timedelta(days=120))


@st.cache_data(show_spinner="Fetching NASA POWER data...", ttl=3600)
def load(lat: float, lon: float, start: date, end: date):
    return add_indicators(fetch_daily(lat, lon, start, end))


try:
    df = load(lat, lon, start, end)
except Exception as exc:  # network / API errors
    st.error(f"Could not load data: {exc}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Water-stress days", int(df["water_stress"].sum()))
c2.metric("Heat-stress days", int(df["heat_stress"].sum()))
c3.metric("Disease risk (last 7d)", f"{df['disease_risk_7d'].iloc[-1]:.0%}")

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

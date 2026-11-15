"""Agri Nova - "Flash Drought" Pivot. Decision support for crop rotations.

Run with: uv run streamlit run src/agri_nova/app.py
"""
from __future__ import annotations

from datetime import date

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from agri_nova.data.crop_data import CDL_TO_CROP, suggest_rotation
from agri_nova.data.evapotranspiration import plot_heat_load_proxy
from agri_nova.data.power import fetch_daily_cached
from agri_nova.data.site import load_site_context
from agri_nova.glossary import GLOSSARY, HUMAN_JUDGMENT, LIMITS, SATELLITE_VS_GROUND
from agri_nova.history import (
    METRIC_INFO,
    anomalies,
    attach_history,
    detect_flash_drought_events,
    frost_table,
    period_comparison,
    season_metrics,
    season_slice,
    support_summary,
    year_vs_record,
    yearly_signal_table,
)
from agri_nova.indicators import add_indicators
from agri_nova.priorities import (
    CRITERIA,
    CRITERIA_HELP,
    CRITERIA_LABELS,
    Constraints,
    Priorities,
    priority_fit,
)
from agri_nova.decision import (
    build_verdict,
    next_steps,
    practical_checks,
    scorecard,
    stress_summary,
    stress_table,
    stress_test,
)
from agri_nova.rotations import (
    CROP_DATABASE,
    RADAR_CATEGORIES,
    ROTATION_CROPS,
    DataAvailability,
    StrategyAssessment,
    addon_candidates,
    assess_from_frame,
    sanitize_rotation,
)
from agri_nova.regional import cell_anomalies, cell_bounds, cell_label, regional_season_metrics, severity_color
from agri_nova.scenarios import apply_scenario, describe_scenario

# Spink County, South Dakota (near Redfield) - the project's study area.
DEFAULT_LAT, DEFAULT_LON = 44.88, -98.52
RECORD_START, RECORD_END = date(2012, 1, 1), date(2022, 12, 31)
YEARS = list(range(2022, 2011, -1))

SIGNAL_LABELS = {
    "reference": ("info", "Reference point - your existing rotation"),
    "supports": ("success", "Data pattern supports considering this"),
    "mixed": ("warning", "Mixed signal"),
    "limited": ("info", "Limited signal in this period"),
    "insufficient_data": ("warning", "Not enough data in this period"),
}

STRATEGY_COLORS = {"baseline": "#7f7f7f", "cover_crop": "#00a676", "diversified": "#7b4bd8"}
CHECK_ICONS = {"ok": "✅", "warn": "⚠️", "info": "ℹ️"}

CROP_PROFILES = {
    "Corn": {"critical_start": "07-15", "critical_end": "08-10", "heat_threshold": 35.0, "phase_name": "Pollination"},
    "Soybean": {"critical_start": "08-01", "critical_end": "08-25", "heat_threshold": 32.0, "phase_name": "Pod Set"},
    "Winter Wheat": {"critical_start": "05-15", "critical_end": "06-15", "heat_threshold": 30.0, "phase_name": "Flowering & Grain Fill"},
    # The crops below are approximate rules of thumb matching the notes in rotations.CROP_DATABASE; confirm with local extension.
    "Spring Wheat": {"critical_start": "06-15", "critical_end": "07-15", "heat_threshold": 30.0, "phase_name": "Flowering & Grain Fill"},
    "Sorghum": {"critical_start": "07-20", "critical_end": "08-20", "heat_threshold": 38.0, "phase_name": "Flowering & Grain Fill"},
    "Sunflower": {"critical_start": "07-10", "critical_end": "08-10", "heat_threshold": 35.0, "phase_name": "Flowering"},
    "Peas": {"critical_start": "06-01", "critical_end": "06-30", "heat_threshold": 28.0, "phase_name": "Flowering"},
    "Alfalfa": {"critical_start": "07-01", "critical_end": "08-15", "heat_threshold": 35.0, "phase_name": "Summer Regrowth"},
}

st.set_page_config(page_title="Agri Nova", page_icon="🌾", layout="wide")
st.title("🌾 Agri Nova: The Flash Drought Pivot")
st.caption(
    "Compares crop-rotation options using NASA Earth observations plus USDA soil and crop records. "
    "A decision-support aid - it highlights patterns and possibilities, it does not replace your judgment or local expertise."
)


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner="Downloading the NASA POWER climate record (2012-2022)...", ttl=24 * 3600)
def load_climate(lat: float, lon: float) -> pd.DataFrame:
    return fetch_daily_cached(lat, lon, RECORD_START, RECORD_END)


@st.cache_data(show_spinner="Reading terrain, soil survey and crop history...", ttl=24 * 3600)
def load_site(lat: float, lon: float):
    return load_site_context(lat, lon)


@st.cache_data(show_spinner="Downloading NASA POWER for the surrounding cells (cached afterwards)...", ttl=24 * 3600)
def load_region(lat: float, lon: float):
    return regional_season_metrics(lat, lon, RECORD_START, RECORD_END)


# ----------------------------------------------------------------------- sidebar
if "lat" not in st.session_state:
    st.session_state.lat, st.session_state.lon = DEFAULT_LAT, DEFAULT_LON

with st.sidebar:
    st.header("📍 Field")
    st.caption("Click the map to move the field. Default: Spink County, SD.")
    fmap = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=8)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(fmap)
    folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(fmap)
    map_data = st_folium(fmap, height=260, width=300, key="field_map", returned_objects=["last_clicked"])
    clicked = map_data.get("last_clicked") if map_data else None
    if clicked:
        st.session_state.lat, st.session_state.lon = clicked["lat"], clicked["lng"]
    lat, lon = st.session_state.lat, st.session_state.lon
    st.success(f"{lat:.4f}, {lon:.4f}")

    focus_year = st.selectbox("Season to examine", YEARS, index=YEARS.index(2017), help="2017 was the Northern Plains flash drought.")
    default_rotation, rotation_reason = suggest_rotation(load_site(lat, lon).crop_history)
    picked = st.multiselect(
        "Crops in your current rotation",
        ROTATION_CROPS,
        default=default_rotation,
        format_func=lambda c: CROP_DATABASE[c].name,
        key=f"rotation_{lat:.3f}_{lon:.3f}",   # new field -> new default from its CDL history
        help="Pick every crop you grow in sequence. One crop means the same crop every year.",
    )
    if not picked:
        st.warning("No crop selected - Corn / Soybean is assumed.")
    rotation = sanitize_rotation(picked)
    st.caption(rotation_reason)
    addon_options = ["auto"] + addon_candidates(rotation)
    addon_choice = st.selectbox(
        "Crop to test adding (Strategy B)",
        addon_options,
        format_func=lambda c: "Pick the best one for my priorities" if c == "auto" else CROP_DATABASE[c].name,
        key=f"addon_{lat:.3f}_{lon:.3f}_{'-'.join(rotation)}",
        help="Strategy B adds this crop to your rotation. 'Pick the best one' ranks every candidate against your priorities.",
    )
    _latest = str(load_site(lat, lon).exhaustion.get("baseline_crop", ""))
    if _latest and _latest.strip().lower() not in CDL_TO_CROP:
        st.warning(
            f"The latest USDA-mapped crop here is **{_latest}**, which this tool does not model. "
            "Choose the crops you really grow in the box above so the comparison starts from your real rotation."
        )
    inspect_options = rotation + [c for c in CROP_PROFILES if c not in rotation]
    selected_crop = st.selectbox(
        "Crop to inspect (critical stage)",
        inspect_options,
        format_func=lambda c: CROP_DATABASE[c].name + (" (in your rotation)" if c in rotation else ""),
        key=f"inspect_{lat:.3f}_{lon:.3f}_{'-'.join(rotation)}",
        help="Shows the heat chart for this crop's most sensitive stage on the Field overview tab. Your rotation's crops are listed first. Dates and heat limits are approximate rules of thumb.",
    )

    with st.expander("🎯 Your priorities", expanded=True):
        st.caption("0 = don't care, 5 = top priority")
        weights = {c: st.slider(CRITERIA_LABELS[c], 0, 5, 3, help=CRITERIA_HELP[c], key=f"prio_{c}") for c in CRITERIA}
    priorities = Priorities(**weights)

    with st.expander("🚜 What your farm can do"):
        constraints = Constraints(
            can_seed_cover_crops=st.checkbox("I can seed a cover crop after harvest", True),
            can_harvest_small_grains=st.checkbox("I can harvest small grains (combine/header)", True),
            wheat_market_nearby=st.checkbox("A market/elevator for wheat or other new crops is within reach", True),
            livestock_for_forage=st.checkbox("I have livestock that could use forage", False),
            irrigation_available=st.checkbox("I can irrigate", False, help="When ticked, the water-saving advantage of the cover crop and wheat is scored lower because water can be added. Irrigation itself is not modeled."),
        )

    with st.expander("🌦️ What-if scenario"):
        rain_pct = st.slider("Rainfall change (%)", -40, 40, 0, step=5)
        temp_c = st.slider("Temperature shift (°C)", -2.0, 6.0, 0.0, step=0.5)
        st.caption("Applied to every day of 2012-2022 before the indicators are recomputed.")


# ------------------------------------------------------------------- data loading
try:
    raw = load_climate(lat, lon)
except Exception as exc:  # network / API errors
    st.error(f"Could not load NASA POWER data: {exc}")
    st.stop()

site = load_site(lat, lon)

profile = CROP_PROFILES[selected_crop]
obs = add_indicators(raw)                                              # what actually happened
scn = add_indicators(apply_scenario(raw, rain_pct, temp_c))            # what-if version
focus_scn = season_slice(scn, focus_year)

available = DataAvailability(
    terrain=site.terrain is not None,
    wss=site.wss,
    soilgrids=site.soilgrids,
)

assess_kwargs = dict(
    available=available, terrain=site.terrain, soil=site.soil, rotation=rotation,
    constraints=constraints, priorities=priorities, addon=addon_choice,
)
assessments = assess_from_frame(focus_scn, **assess_kwargs)
hist_table = yearly_signal_table(scn)
attach_history(assessments, hist_table)
fit = priority_fit(assessments, priorities)
for a in assessments:
    a.priority_fit = fit[a.key]

metrics_obs = season_metrics(obs)
frost_obs = frost_table(obs)
frost_scn = frost_table(scn)


# ------------------------------------------------------------------------- charts
def plot_vulnerability_window(df: pd.DataFrame, crop: dict, year: int) -> go.Figure:
    fig = px.line(df, y=["T2M_MAX", "T2M_MIN"], title=f"Temperature stress around {crop['phase_name'].lower()} ({year})")
    fig.add_vrect(
        x0=f"{year}-{crop['critical_start']}", x1=f"{year}-{crop['critical_end']}",
        fillcolor="red", opacity=0.15, layer="below", line_width=0,
        annotation_text=crop["phase_name"], annotation_position="top left",
    )
    fig.add_hline(
        y=crop["heat_threshold"], line_dash="dot", line_color="red",
        annotation_text=f"Heat limit ({crop['heat_threshold']:.0f} °C)",
    )
    fig.update_layout(yaxis_title="°C", legend_title_text="")
    return fig


def plot_radar(items: list[StrategyAssessment]) -> go.Figure:
    fig = go.Figure()
    for a in items:
        r = list(a.radar_scores)
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1], theta=RADAR_CATEGORIES + RADAR_CATEGORIES[:1], fill="toself", name=a.name,
            line_color=STRATEGY_COLORS[a.key], opacity=0.7,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), legend=dict(orientation="h", y=-0.15))
    return fig


def plot_fit(items: list[StrategyAssessment]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[a.priority_fit for a in items],
        y=[a.name for a in items],
        orientation="h",
        marker_color=[STRATEGY_COLORS[a.key] for a in items],
        text=[f"{a.priority_fit:.0f}" for a in items],
    ))
    fig.update_layout(xaxis=dict(range=[0, 100], title="Alignment with your priorities (0-100)"), yaxis=dict(autorange="reversed"))
    return fig


def critical_heat_days(df: pd.DataFrame, crop: dict, year: int) -> int:
    window = df.loc[f"{year}-{crop['critical_start']}":f"{year}-{crop['critical_end']}"]
    return int((window["T2M_MAX"] >= crop["heat_threshold"]).sum())


def bullets(items: list[str]) -> None:
    st.markdown("\n".join(f"- {i}" for i in items))


tab_overview, tab_rot, tab_hist, tab_map, tab_learn = st.tabs(
    ["📊 Field overview", "🔄 Rotation options", "📈 History & anomalies", "🗺️ Regional map", "📚 Learn & limits"]
)

# ----------------------------------------------------------------------- overview
with tab_overview:
    st.caption(describe_scenario(rain_pct, temp_c))
    summer = focus_scn[focus_scn.index.month.isin([6, 7, 8])]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Water-stress days (season)", int(focus_scn["water_stress"].sum()))
    c2.metric(f"Hot days during {profile['phase_name'].lower()}", critical_heat_days(focus_scn, profile, focus_year),
              help=f"Days at or above {profile['heat_threshold']:.0f} °C in the {selected_crop} critical window.")
    c3.metric("Longest summer dry spell (days)", int(summer["dry_spell_days"].max()) if len(summer) else 0)
    c4.metric("Disease risk (last 7 days)", f"{focus_scn['disease_risk_7d'].iloc[-1]:.0%}" if len(focus_scn) else "N/A")

    frost_row = frost_scn.loc[focus_year] if focus_year in frost_scn.index else None

    def _frost(col: str, suffix: str = "") -> str:
        v = None if frost_row is None else frost_row[col]
        if v is None or pd.isna(v):
            return "none found" if frost_row is not None else "n/a"
        return f"{int(v)}{suffix}" if col == "frost_free_days" else f"{v}"

    fr1, fr2, fr3 = st.columns(3)
    frost_help = "Frost day = daily minimum air temperature at or below 0 °C (NASA POWER, gridded). Frost hollows in a real field can be colder."
    fr1.metric("Last spring frost", _frost("last_spring_frost"), help=frost_help)
    fr2.metric("First fall frost", _frost("first_fall_frost"), help=frost_help)
    fr3.metric("Frost-free days", _frost("frost_free_days"), help="Days between the last spring frost and the first fall frost. It sets the window for fall cover-crop and winter-wheat seeding.")

    st.markdown("#### Heat during each crop's most sensitive stage")
    st.dataframe(
        pd.DataFrame([
            {
                "Crop": CROP_DATABASE[c].name,
                "Sensitive stage": CROP_PROFILES[c]["phase_name"],
                "Window (month-day)": f"{CROP_PROFILES[c]['critical_start']} to {CROP_PROFILES[c]['critical_end']}",
                "Heat limit (°C)": CROP_PROFILES[c]["heat_threshold"],
                f"Hot days in {focus_year}": critical_heat_days(focus_scn, CROP_PROFILES[c], focus_year),
            }
            for c in rotation
        ]),
        width="stretch", hide_index=True,
    )
    st.caption("Crops in your current rotation. Windows and heat limits are approximate rules of thumb; "
               "pick a crop in the sidebar to see its temperature chart below.")
    st.plotly_chart(plot_vulnerability_window(focus_scn, profile, focus_year), width="stretch")

    col_site, col_hist = st.columns(2)
    with col_site:
        st.markdown("#### Terrain & soil")
        if site.terrain:
            st.write(f"**Elevation:** {site.terrain.elevation_m:.0f} m  |  **Slope:** {site.terrain.slope_percent:.1f}% "
                     f"({site.terrain.risk_level} erosion risk) - NASA SRTM")
        else:
            st.warning(f"Slope could not be measured ({site.terrain_message or 'elevation service unavailable'}). "
                       "Erosion risk is treated as unknown, not flat.")
        if site.soil:
            s = site.soil
            rows = {
                "Plant-available water, top metre": f"{s.awc_mm_100cm:.0f} mm" if s.awc_mm_100cm is not None else "n/a",
                "Organic matter, topsoil": f"{s.om_pct_topsoil:.1f}%" if s.om_pct_topsoil is not None else "n/a",
                "Sand / clay, topsoil": f"{s.sand_pct:.0f}% / {s.clay_pct:.0f}%" if s.sand_pct is not None else "n/a",
                "Dominant soil": site.soil_extra.get("component") or "n/a",
                "Source": s.source,
            }
            st.table(pd.Series(rows, name="Value"))
        else:
            st.warning("No soil data returned for this point.")

    with col_hist:
        st.markdown("#### USDA crop history (Cropland Data Layer)")
        hist_df = pd.DataFrame(site.crop_history)
        if not hist_df.empty:
            st.dataframe(hist_df.set_index("year").T, width="stretch")
        ex = site.exhaustion
        st.write(f"**Detected latest crop:** {ex.get('baseline_crop', 'n/a')}")
        st.write(f"**Nitrogen status:** {ex.get('nitrogen_level', 'n/a')}")
        st.write(f"**Rotation health:** {ex.get('exhaustion_status', 'n/a')}")

    st.markdown("#### Climate context")
    left, right = st.columns(2)
    left.plotly_chart(px.bar(focus_scn, y="PRECTOTCORR", title="Daily precipitation (mm)"), width="stretch")
    right.plotly_chart(px.line(focus_scn, y="GWETROOT", title="Root-zone soil wetness (0 = dry, 1 = saturated; model)"), width="stretch")
    st.plotly_chart(plot_heat_load_proxy(focus_scn), width="stretch")
    with st.expander("Explore raw data"):
        st.dataframe(focus_scn)

# ---------------------------------------------------------------------- rotations
with tab_rot:
    st.caption(describe_scenario(rain_pct, temp_c))
    by_key = {a.key: a for a in assessments}
    _addon = by_key["diversified"].addon
    st.caption(
        f"Compared: your current rotation (**{by_key['baseline'].crops}**), the same plus a cereal-rye cover crop (A), "
        f"and the same plus {CROP_DATABASE[_addon].name if _addon else 'no extra crop'} (B). "
        "Signals come from the season you selected; 'seasons supported' repeats the test on every season from 2012 to 2022."
    )

    if sum(weights.values()) == 0:
        st.warning("All priorities are set to 0, so no alignment score can be calculated. Raise at least one priority slider in the sidebar.")

    verdict = build_verdict(assessments, priorities)
    if verdict is not None:
        with st.container(border=True):
            st.markdown(f"### {verdict.headline}")
            if verdict.helps or verdict.costs:
                gain_col, cost_col = st.columns(2)
                with gain_col:
                    st.markdown("**Better than your current rotation on**")
                    bullets(verdict.helps or ["Nothing you rated as a priority."])
                with cost_col:
                    st.markdown("**Worse than your current rotation on**")
                    bullets(verdict.costs or ["Nothing you rated as a priority."])
            if verdict.weather_note:
                st.caption(verdict.weather_note)
            st.caption("Fit means alignment with *your* priorities, from rules of thumb. It is not a yield or profit forecast.")

    summary = pd.DataFrame(
        {
            "Strategy": [a.name for a in assessments],
            "Signal this season": [SIGNAL_LABELS[a.signal][1] for a in assessments],
            "Alignment with your priorities": [a.priority_fit for a in assessments],
            "Seasons supported (2012-2022)": [
                (lambda s: f"{s['k']} of {s['n']}")(support_summary(hist_table, a.key)) if a.key != "baseline" else "-"
                for a in assessments
            ],
        }
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    rc, fc = st.columns(2)
    rc.plotly_chart(plot_radar(assessments), width="stretch")
    if sum(weights.values()) > 0:
        fc.plotly_chart(plot_fit(assessments), width="stretch")
        fc.caption("Alignment weights each 1-5 trait score by the priorities you set. It ranks fit with *your* goals; "
                   "it is not a prediction of profit or yield.")

    with st.expander("Scores behind the ranking (1-5, higher is better)"):
        st.dataframe(scorecard(assessments, priorities), width="stretch", hide_index=True)

    st.markdown("#### Would this still be the best choice in a drier or hotter year?")
    if sum(weights.values()) == 0:
        st.info("Set at least one priority above 0 to run this check.")
    else:
        def _assess_under(rain: float, temp: float):
            frame = season_slice(add_indicators(apply_scenario(raw, rain, temp)), focus_year)
            items = assess_from_frame(frame, **assess_kwargs)
            item_fit = priority_fit(items, priorities)
            for item in items:
                item.priority_fit = item_fit[item.key]
            return items

        stress_rows = stress_test(_assess_under)
        st.info(stress_summary(stress_rows))
        st.dataframe(
            stress_table(stress_rows, {k: v.name for k, v in CROP_DATABASE.items()}),
            width="stretch", hide_index=True,
        )
        st.caption(
            f"Each row re-runs the {focus_year} season with changed rain and temperature, starting from observed weather "
            "(the sidebar what-if sliders are not added on top). Numbers are alignment with your priorities, 0-100."
        )

    st.markdown("#### The three options in detail")
    for a in assessments:
        is_best = verdict is not None and a.key == verdict.best_key
        with st.expander(("⭐ " if is_best else "") + a.name, expanded=is_best):
            kind, label = SIGNAL_LABELS[a.signal]
            getattr(st, kind)(label)
            st.markdown(f"**Why it's suggested:** {a.why_suggested}")
            if a.historical_support:
                st.markdown(f"**Across 2012-2022:** {a.historical_support}")
            st.markdown("**Can your farm do it?**")
            for status, text in practical_checks(a, constraints):
                st.markdown(f"{CHECK_ICONS[status]} {text}")
            e_col, b_col = st.columns(2)
            with e_col:
                st.markdown("**Evidence**")
                bullets(a.evidence)
            with b_col:
                st.markdown("**Benefits it may offer**")
                bullets(a.benefits)
            g_col, r_col = st.columns(2)
            with g_col:
                st.markdown("**Information gaps**")
                bullets(a.information_gaps)
            with r_col:
                st.markdown("**Risks that need local review**")
                bullets(a.expert_review_risks)

    st.markdown("#### What to do next")
    bullets(next_steps(verdict.best_key if verdict is not None else None))

    st.warning("**Where your judgment matters most**\n\n" + "\n".join(f"- {h}" for h in HUMAN_JUDGMENT))

# --------------------------------------------------------------------- history
with tab_hist:
    st.markdown(f"### How unusual was {focus_year}?")
    st.caption("Observed conditions only (scenario sliders do not change this tab's anomaly figures). "
               f"Compared with all {len(metrics_obs)} seasons in 2012-2022.")
    if focus_year in metrics_obs.index:
        cmp_df = year_vs_record(metrics_obs, focus_year)
        view = cmp_df.rename(columns={
            "metric": "Metric", "value": f"{focus_year}", "record_mean": "Record average",
            "z_score": "Z-score", "worst_rank": "Rank (1 = worst)", "n_years": "Seasons",
        })
        st.dataframe(view.round(2), width="stretch", hide_index=True)
    else:
        st.info("This season has too little data for a comparison.")

    metric_key = st.selectbox("Metric by year", list(METRIC_INFO), format_func=lambda k: METRIC_INFO[k][0])
    z = anomalies(metrics_obs)[metric_key]
    bar = go.Figure(go.Bar(
        x=[str(y) for y in z.index], y=z.values,
        marker_color=["crimson" if y == focus_year else "#9aa5b1" for y in z.index],
    ))
    bar.update_layout(title=f"{METRIC_INFO[metric_key][0]} - anomaly vs 2012-2022 (z-score)", yaxis_title="Z-score")
    st.plotly_chart(bar, width="stretch")

    st.markdown("#### Root-zone wetness: this season vs every other season")
    ov = go.Figure()
    for y in metrics_obs.index:
        s = season_slice(obs, y)["GWETROOT"]
        ov.add_trace(go.Scatter(
            x=s.index.dayofyear, y=s.values, mode="lines", name=str(y),
            line=dict(color="crimson" if y == focus_year else "#c8ced6", width=3 if y == focus_year else 1),
            showlegend=(y == focus_year),
        ))
    ov.update_layout(xaxis_title="Day of year", yaxis_title="Wetness (0-1)", title="Gray = other seasons")
    st.plotly_chart(ov, width="stretch")

    st.markdown("#### Detected flash-drought events (rapid drying of the root zone)")
    events = detect_flash_drought_events(obs)
    if events.empty:
        st.info("No events crossed the detection thresholds.")
    else:
        st.dataframe(events, width="stretch", hide_index=True)
    st.caption("Heuristic: root-zone wetness falls >= 0.12 within 14 days and ends below 0.45 (May-Sep). "
               "Tune the thresholds with local knowledge.")

    st.markdown("#### Before vs after the 2017 flash drought")
    if 2017 in metrics_obs.index:
        pc = period_comparison(metrics_obs, 2017)
        st.dataframe(
            pc.rename(columns={
                "metric": "Metric", "before_mean": "Average 2012-2016", "event_year": "2017",
                "after_mean": "Average 2018-2022", "n_before": "Seasons before", "n_after": "Seasons after",
            }).round(2),
            width="stretch", hide_index=True,
        )
        st.caption("Descriptive only. Five seasons on each side are too few for a significance test, so read this as a pattern "
                   "to discuss locally, not as proof that conditions changed.")
    else:
        st.info("2017 has too little data in this record for a before/after view.")

    with st.expander("Frost dates by year (observed)"):
        st.dataframe(frost_obs.rename(columns={
            "last_spring_frost": "Last spring frost", "first_fall_frost": "First fall frost", "frost_free_days": "Frost-free days",
        }), width="stretch")
        st.caption("Frost day = POWER daily minimum at or below 0 °C. Gridded minima run warmer than a frost-prone low spot, "
                   "so use these dates as a regional guide.")

    st.markdown("#### How often would each rotation have been supported?")
    for key in ("cover_crop", "diversified"):
        name = next(a.name for a in assessments if a.key == key)
        st.write(f"**{name}** - {support_summary(hist_table, key)['text']}")
    st.dataframe(hist_table.rename(columns={
        "stress_level": "Stress level", "baseline": "Existing", "cover_crop": "Cover crop", "diversified": "Diversified",
    }), width="stretch")

# -------------------------------------------------------------------------- map
with tab_map:
    st.markdown("### How did conditions vary around your field?")
    st.caption(
        "Each box is one NASA POWER (MERRA-2) grid cell, about 0.5° × 0.625° (roughly 50 km). Colour shows how the chosen "
        "year compares with that same cell's own 2012-2022 record: red = worse than usual, blue = better, grey = about normal. "
        "This is a regional picture - it cannot show differences between fields inside one cell."
    )
    map_metric_keys = ["water_stress_days", "peak_heat_days", "summer_rain_mm", "longest_dry_spell_days", "min_root_wetness"]
    mc1, mc2 = st.columns(2)
    map_metric = mc1.selectbox("What to map", map_metric_keys, format_func=lambda k: METRIC_INFO[k][0], key="map_metric")
    map_year = mc2.selectbox("Year", YEARS, index=YEARS.index(focus_year), key="map_year")

    region_key = (round(lat, 2), round(lon, 2))
    if st.button("Build / refresh regional map", help="Downloads 9 cells of NASA POWER data: about a minute the first time, instant afterwards."):
        st.session_state.region_key = region_key

    if st.session_state.get("region_key") != region_key:
        st.info("Press the button to download the 3 × 3 block of grid cells around your field. Observed weather only - "
                "the what-if sliders do not change this map.")
    else:
        region, region_errors = load_region(lat, lon)
        for label, msg in region_errors.items():
            st.warning(f"Cell {label} could not be loaded: {msg}")
        cells = cell_anomalies(region, map_metric, map_year)
        if cells.empty:
            st.info("No cell has a complete season for that year.")
        else:
            metric_label, higher_is_worse = METRIC_INFO[map_metric]
            rmap = folium.Map(location=[lat, lon], zoom_start=7)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", name="Satellite", overlay=False, control=True,
            ).add_to(rmap)
            for row in cells.itertuples():
                folium.Rectangle(
                    bounds=cell_bounds(row.lat, row.lon),
                    color="#333333", weight=1, fill=True,
                    fill_color=severity_color(row.z, higher_is_worse), fill_opacity=0.55,
                    tooltip=f"{metric_label}: {row.value:.1f} (usual {row.record_mean:.1f}, z = {row.z:+.1f})",
                ).add_to(rmap)
            folium.Marker([lat, lon], tooltip="Your field").add_to(rmap)
            st_folium(rmap, height=460, width=900, key="region_map", returned_objects=[])

            shown = cells.assign(cell=[cell_label((a, b)) for a, b in zip(cells["lat"], cells["lon"])])
            st.dataframe(
                shown[["cell", "value", "record_mean", "z"]].rename(columns={
                    "cell": "Cell centre (lat, lon)", "value": f"{map_year}", "record_mean": "Cell's 2012-2022 average", "z": "Z-score",
                }).round(2),
                width="stretch", hide_index=True,
            )

            trend = region.assign(cell=[cell_label((a, b)) for a, b in zip(region["lat"], region["lon"])])
            st.plotly_chart(
                px.line(trend, x="year", y=map_metric, color="cell", markers=True,
                        title=f"{metric_label} by year, every cell in the block"),
                width="stretch",
            )

# ------------------------------------------------------------------------ learn
with tab_learn:
    st.markdown("### Satellite vs. ground measurements")
    st.markdown(SATELLITE_VS_GROUND)

    st.markdown("### Data in this tool")
    sources = pd.DataFrame([
        ("NASA POWER (MERRA-2 / CERES)", "Daily rain, temperature, humidity, root-zone wetness", "Loaded"),
        ("NASA SRTM elevation", "Slope and erosion risk", "Loaded" if available.terrain else "Not available"),
        ("USDA SSURGO soil survey", "Water-holding capacity, organic matter", "Loaded" if available.wss else "Not available"),
        ("ISRIC SoilGrids", "Soil texture fallback", "Loaded" if available.soilgrids else "Not available"),
        ("USDA Cropland Data Layer", "What was actually planted 2012-2022", "Loaded" if site.crop_history else "Not available"),
    ], columns=["Dataset", "Used for", "Status"])
    st.dataframe(sources, width="stretch", hide_index=True)

    st.markdown("### Glossary")
    for term, text in GLOSSARY:
        with st.expander(term):
            st.write(text)

    st.markdown("### Limits of this tool")
    bullets(LIMITS)
    st.warning("**Where your judgment matters most**\n\n" + "\n".join(f"- {h}" for h in HUMAN_JUDGMENT))

# 🌾 Agri Nova: The "Flash Drought" Pivot

**NASA Space Apps Challenge 2026** | **Challenge:** Field Shift: Adapting Farms with NASA Data

A decision-support dashboard that helps a farmer compare crop-rotation options under flash-drought stress. It combines NASA Earth observations with USDA soil and crop records, explains *why* each option is suggested, what evidence supports it, what is still unknown, and what a local expert must check. It **supports, and does not replace, farmer judgment**.

---

## 🎯 Scope

* **Study area:** Spink County, South Dakota (default field 44.88 N, 98.52 W). Click the map to move to any other point; SSURGO and the Cropland Data Layer cover the US.
* **Record:** 2012-2022, which contains the 2017 Northern Plains flash drought, so the app can show a before/after picture and rank 2017 against every other season.

---

## 🔄 The three strategies compared (always exactly three)

1. **Existing rotation** (crops pre-selected from the USDA Cropland Data Layer history; pick any mix of Corn, Soybean, Winter/Spring Wheat, Sorghum, Sunflower, Peas, Alfalfa): the reference point.
2. **Strategy A:** existing rotation + winter cereal-rye cover crop.
3. **Strategy B:** existing rotation + one added crop, either chosen by the farmer or picked automatically as the best match for their priorities.

Each one is shown with **why it is suggested, evidence, benefits, information gaps, and risks that need local review**. Signals are transparent threshold rules on climate indicators, not a black box. The farmer's priorities (7 sliders) and practical constraints (seeding equipment, small-grain harvest, wheat market, livestock, irrigation) re-weight the scores and add warnings.

---

## 🛰️ Data, and what is actually used

| Source | Used for | Needs |
|---|---|---|
| **NASA POWER** (MERRA-2 / CERES) | Daily rain, temperature, humidity, **modeled** root-zone wetness, skin temperature | nothing |
| **NASA SRTM** via OpenTopoData / Open-Elevation | Slope and erosion risk | nothing |
| **USDA SSURGO** (Soil Data Access) | Plant-available water, organic matter, texture | nothing |
| **ISRIC SoilGrids** | Texture fallback when SSURGO has no answer | nothing |
| **USDA Cropland Data Layer** | What was planted 2012-2022, monoculture check, default baseline | nothing |

No logins or API keys are needed. The soil wetness used is **modeled** (POWER/MERRA-2), not measured by a satellite or a probe. Things the app does not have (for example satellite soil moisture, vegetation index or field-scale evapotranspiration) appear as *information gaps* in the strategy cards instead of being silently assumed.

> **Removed:** optional NASA satellite layers (SMAP, GPM IMERG, MODIS NDVI, ECOSTRESS, OpenET) were taken out because the AppEEARS / OpenET download path was never verified end to end.

### Time grid
POWER is the only time series the dashboard loads, and it is already daily. `harmonize.py` (interpolate onto a daily axis across short gaps with an `_age_days` marker; block-average rasters to a common grid) is unit-tested but not used by the dashboard.

---

## 🧭 What the dashboard shows

| Tab | Contents |
|---|---|
| **Field overview** | Stress metrics, frost dates and frost-free days, crop critical-window temperature chart, terrain/soil/crop history, precipitation, soil wetness, heat-load proxy |
| **Rotation options** | A plain-language verdict first (best fit for your priorities, what it gains and costs versus your current rotation), summary table, radar and alignment charts, a **drier/hotter stress test** (does the winner change?), then per strategy a "can your farm do it?" checklist plus the why / evidence / benefits / gaps / risks card, next steps, and "where your judgment matters most" |
| **History & anomalies** | Z-scores and worst-rank for the chosen season, per-year anomaly bars, wetness overlay of all seasons, detected flash-drought events, **before vs after 2017**, frost table, and how often each rotation would have been supported (Wilson 95% interval, with a small-sample warning) |
| **Regional map** | 3 × 3 block of POWER cells around the field, coloured by how the chosen year compares with each cell's own record; metric and year selectors; trend lines |
| **Learn & limits** | Satellite vs ground data, data inventory, glossary (ET, soil moisture anomaly, frost, ...), limits of the tool |

Sidebar: field map, season, current rotation, priorities, practical constraints, **what-if sliders** (rainfall -40..+40 %, temperature -2..+6 C).

---

## 🚀 Setup

This project uses [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev).

```bash
task setup               # uv sync
task run                 # Streamlit dashboard
task test                # pytest
task prefetch            # cache POWER, terrain, soil, crop history and the regional block
```

No credentials are needed. Downloaded data is cached under `data/processed/`. The `.gitignore` lets the default Spink County files through so a fresh clone can run offline: the POWER file already exists, and `task prefetch` creates the soil/terrain/crop-history file (run it once, then commit). Anything else is downloaded on first use.

---

## ⚠️ Honest limits

* Scores and signals are rules of thumb from general agronomy: not a yield forecast, not agronomic advice.
* Eleven seasons is a short record; percentages across seasons carry wide uncertainty.
* POWER values describe a ~50 km grid cell, not a field. Frost dates from gridded minima run warmer than a frost hollow.
* The heat-load chart is POWER's modeled skin-minus-air temperature, **not** measured canopy temperature; its +3 C line is a rule of thumb.
* Scenario sliders use a simple assumed wetness sensitivity: a sensitivity test, not a climate projection.
* Local seeding dates, varieties, winter survival, prices, equipment, insurance and program rules are not modeled; talk to your county extension agent, crop consultant or NRCS office before changing a rotation.

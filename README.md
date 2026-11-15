# 🌾 Agri Nova: The "Flash Drought" Pivot

**NASA Space Apps Challenge 2026** | **Challenge:** [Field Shift: Adapting Farms with NASA Data]

## 📖 Project Overview

Farmers in the American Midwest and Northern Plains are increasingly facing **flash droughts**—rapid-onset environmental events where spiking temperatures and dry winds quickly strip moisture from the topsoil. During these events, traditional crop rotations (like continuous corn and soybeans) suffer severe heat stress and pollination failure. 

**The "Flash Drought" Pivot** is a decision-support dashboard designed to help farmers adapt to these rapid evapotranspiration events. By fusing NASA Earth observations with hyper-local USDA agricultural data, this tool visualizes field-level vulnerabilities and evaluates drought-resilient crop rotation strategies, explicitly acting as a supportive aid for farmer judgment.

---

## 🎯 Scope & Focus
* **Target Location:** Spink County, South Dakota (Covered by OpenET for field-scale water demand).
* **Target Timeframe:** 2012–2022. This captures the catastrophic 2017 Northern Plains flash drought, providing a clear "before and after" environmental signal for our statistical models.

---

## 🔄 Rotation Strategies Evaluated
To meet the challenge requirements, this tool compares three specific cropping strategies, evaluating water use, organic matter contributions, and soil resilience:

1. **The Baseline: Traditional Corn / Soybean**
   * *Vulnerability:* High water demand mid-summer; bare winter soils lead to poor water infiltration and rapid evaporation when flash droughts hit.
2. **Strategy A: Corn / Soybean + Winter Cover Crop (Cereal Rye)**
   * *The Pivot:* Introduces residue mulch to cool the soil, trap moisture, and improve water infiltration during heavy rain to "bank" water for dry spells.
3. **Strategy B: Diversified Pivot (Corn / Soybean / Winter Wheat)**
   * *The Pivot:* Adds a deep-rooting winter crop that grows during cooler months and is harvested before peak July/August heat, reducing total summer water demand and breaking pest cycles.

---

## 🛰️ Data Strategy & Sources

Harmonizing disparate spatial and temporal resolutions onto a common analysis grid is a core technical feature of this project.

### NASA & Space Agency Data (Macro-Level Climate & Stress)
* **SMAP (Soil Moisture Active Passive):** Primary input (2015-2022) for tracking surface soil moisture and drought stress.
* **NLDAS / GLDAS:** Used to bridge the pre-SMAP historical gap (2012-2014) for soil moisture and temperature proxy data.
* **ECOSTRESS:** Maps high-resolution hotspots of plant heat exposure and crop-water demand.

### USDA & Local Data (Micro-Level Ground Truth)
* **OpenET:** Provides highly accurate, field-scale evapotranspiration estimates for western U.S. states to measure true water demand.
* **USDA Web Soil Survey (WSS) / SSURGO:** Provides field-specific baseline soil properties (water-holding capacity, organic matter, infiltration rates).
* **USDA Cropland Data Layer (CDL):** Tracks the actual historical crop types planted in our target county to train our baseline model.

---

## 💻 Technical Architecture

The project is built in a two-phase pipeline to balance complex Statistical Inference (SI) with a farmer-centered design.

1. **Data Fusion & Analysis (Jupyter / Python)**
   * Fetches and downsamples satellite datasets (NetCDF/GeoTIFF) to a common grid.
   * Calculates a unified "Drought Vulnerability Score" based on historical evapotranspiration and soil moisture anomalies.
2. **Decision-Support Dashboard (Streamlit)**
   * **Interactive Map:** Built with Folium/Plotly to visualize vulnerable field zones.
   * **Actionable Insights UI:** Explains *why* a rotation is suggested, the supporting evidence, benefits, information gaps, and risks requiring local agronomist review.

---

## 🚀 Local Setup & Installation

*(Coming soon: Instructions on how to clone this repository, install the `requirements.txt`, and launch the Streamlit app locally during the hackathon.)*

```bash
# Example setup (To be updated)
git clone https://github.com/yourusername/field-shift-flash-drought.git
cd field-shift-flash-drought
uv sync
uv run streamlit run src/agri_nova/app.py
uv run pytest
uv run ruff check .
```
# farm-shift

Field Shift: Adapting Farms with NASA Data (NASA Space Apps 2026).

## Quickstart
```bash
uv sync
uv run streamlit run src/agri_nova/app.py
uv run pytest
uv run ruff check .
```

## Data sources
- NASA POWER (no key): daily rain, temperature, humidity, solar, soil wetness
- Earthdata via `earthaccess` (free login): SMAP soil moisture, MODIS NDVI, GPM rainfall

import pytest

from agri_nova.data.crop_data import analyze_exhaustion, parse_cdl_category, suggest_baseline, suggest_rotation
from agri_nova.data.ssurgo import summarize_horizons
from agri_nova.data.terrain import _points, risk_label, slope_from_elevations
from agri_nova.rotations import CROP_DATABASE


# ------------------------------------------------------------------- SSURGO
def test_summarize_horizons_uses_the_dominant_component_and_depth_weights():
    rows = [
        {"cokey": "1", "compname": "Houdek", "comppct_r": "80", "hydgrp": "B", "hzdept_r": "0", "hzdepb_r": "30",
         "awc_r": "0.2", "om_r": "3.0", "sandtotal_r": "30", "claytotal_r": "20", "ksat_r": "10"},
        {"cokey": "1", "compname": "Houdek", "comppct_r": "80", "hydgrp": "B", "hzdept_r": "30", "hzdepb_r": "100",
         "awc_r": "0.1", "om_r": "1.0", "sandtotal_r": "40", "claytotal_r": "25", "ksat_r": "10"},
        {"cokey": "2", "compname": "Other", "comppct_r": "20", "hydgrp": "C", "hzdept_r": "0", "hzdepb_r": "100",
         "awc_r": "0.05", "om_r": "9.0", "sandtotal_r": "90", "claytotal_r": "5", "ksat_r": "10"},
    ]
    s = summarize_horizons(rows)
    assert s["component"] == "Houdek"
    assert s["awc_mm_100cm"] == pytest.approx(0.2 * 30 * 10 + 0.1 * 70 * 10)   # 130 mm
    assert s["om_pct_topsoil"] == pytest.approx(3.0)
    assert s["sand_pct"] == pytest.approx(30.0)
    assert summarize_horizons([]) is None


# ------------------------------------------------------------------ terrain
def test_slope_from_elevations_uses_central_differences():
    # [centre, north, south, east, west]: 10 m rise over 500 m north-south = 2 %
    assert slope_from_elevations([100, 105, 95, 100, 100], 250.0) == pytest.approx(2.0)


def test_risk_labels():
    assert risk_label(1.0) == "Low"
    assert risk_label(5.0) == "Moderate"
    assert risk_label(9.0).startswith("Critical")


def test_longitude_offset_is_corrected_for_latitude():
    pts = _points(45.0, -98.0, 250.0)
    d_lat = pts[1][0] - pts[0][0]
    d_lon = pts[3][1] - pts[0][1]
    assert d_lon > d_lat                      # a degree of longitude is shorter at 45 N
    assert d_lon / d_lat == pytest.approx(1.414, abs=0.01)


# ---------------------------------------------------------------------- CDL
def test_parse_cdl_category_handles_malformed_json_and_unknowns():
    assert parse_cdl_category('<return>{Value:"1",Category:"Corn"}</return>') == "Corn"
    assert parse_cdl_category('<return>{Value:"0",Category:"Unknown"}</return>') == "No Data"
    assert parse_cdl_category("<return></return>") == "No Data"


def _history(*crops: str) -> list[dict]:
    return [{"year": 2012 + i, "crop": c} for i, c in enumerate(crops)]


def test_analyze_exhaustion_flags_monoculture():
    out = analyze_exhaustion(_history("Corn", "Corn", "Corn", "Soybeans"), CROP_DATABASE)
    assert "Critical" in out["exhaustion_status"]
    assert out["baseline_crop"] == "Soybeans"


def test_analyze_exhaustion_says_not_applicable_for_non_cropland():
    out = analyze_exhaustion(_history("Grassland/Pasture", "Grassland/Pasture"), CROP_DATABASE)
    assert out["nitrogen_level"] == "Not applicable"


def test_suggest_baseline_follows_the_cdl_history():
    assert suggest_baseline(_history("Soybeans", "Corn", "Corn", "Corn"))[0] == "continuous_corn"
    assert suggest_baseline(_history("Corn", "Soybeans", "Corn", "Soybeans"))[0] == "corn_soy"
    key, why = suggest_baseline(_history("API Error", "No Data"))
    assert key == "corn_soy" and "No usable" in why


# ------------------------------------------------------------ suggested rotation
def test_suggest_rotation_detects_continuous_corn_and_corn_soy():
    ids, why = suggest_rotation(_history("Soybeans", "Corn", "Corn", "Corn"))
    assert ids == ["Corn"] and "single-crop" in why
    assert suggest_rotation(_history("Corn", "Soybeans", "Corn", "Soybeans"))[0] == ["Corn", "Soybean"]


def test_suggest_rotation_keeps_small_grains_and_caps_at_three_crops():
    ids, why = suggest_rotation(_history("Corn", "Soybeans", "Spring Wheat", "Corn", "Soybeans", "Sunflower"))
    assert len(ids) == 3 and "Corn" in ids and "Soybean" in ids
    assert "Spring Wheat" in why and "Sunflower" in why


def test_suggest_rotation_skips_unmodeled_classes_and_falls_back():
    ids, _ = suggest_rotation(_history("Grassland/Pasture", "Corn", "Dbl Crop WinWht/Soybeans", "Soybeans", "API Error"))
    assert ids == ["Corn", "Soybean"]
    ids, why = suggest_rotation(_history("API Error", "No Data"))
    assert ids == ["Corn", "Soybean"] and "No usable" in why


def test_suggest_rotation_only_returns_crops_the_model_knows():
    ids, _ = suggest_rotation(_history("Durum Wheat", "Peas", "Alfalfa", "Winter Wheat"))
    assert all(i in CROP_DATABASE for i in ids)

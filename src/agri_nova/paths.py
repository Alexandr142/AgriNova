"""Shared filesystem locations."""
from pathlib import Path

# src/agri_nova/paths.py -> project root is three levels up
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

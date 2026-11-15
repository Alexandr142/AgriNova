"""Command-line entry point: ``agri-nova`` launches the Streamlit dashboard."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).with_name("app.py")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]))


if __name__ == "__main__":
    main()

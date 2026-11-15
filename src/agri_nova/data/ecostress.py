"""NASA AppEEARS API integration for ECOSTRESS Evapotranspiration data."""
from __future__ import annotations
import httpx
import pandas as pd
import time
import io
from datetime import date

from agri_nova.data.settings import data_settings


def fetch_ecostress(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """
    Authenticates with NASA EarthData, submits a point-extraction task for
    ECOSTRESS ET data, polls for completion, and returns the ET time series.
    """
    # ECOSTRESS data is only available after July 9, 2018.
    eco_launch_date = date(2018, 7, 9)
    if end < eco_launch_date:
        raise ValueError("ECOSTRESS data is not available before July 9, 2018.")

    # Adjust start date if it's before launch
    safe_start = max(start, eco_launch_date)

    base_url = "https://appeears.earthdatacloud.nasa.gov/api"

    with httpx.Client(timeout=90.0) as client:
        # 1. Authenticate
        login_resp = client.post(f"{base_url}/login", auth=(data_settings.earthdata_username, data_settings.earthdata_password))
        if login_resp.status_code in (401, 403):
            raise PermissionError("Invalid EarthData credentials.")
        login_resp.raise_for_status()
        token = login_resp.json().get("token")
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Submit Point Task
        task_payload = {
            "task_type": "point",
            "task_name": "ECOSTRESS_ET_Extraction",
            "params": {
                "dates": [{
                    "startDate": safe_start.strftime("%m-%d-%Y"),
                    "endDate": end.strftime("%m-%d-%Y")
                }],
                "layers": [{
                    "layer": "ETdaily",
                    "product": "ECO_L3T_JET.002"
                }],
                "coordinates": [{
                    "latitude": lat,
                    "longitude": lon,
                    "id": "target_field"
                }]
            }
        }

        task_resp = client.post(f"{base_url}/task", json=task_payload, headers=headers)

        # Capture precise error if 400 occurs
        if task_resp.status_code == 400:
            error_details = task_resp.json()
            raise ValueError(f"AppEEARS rejected the task: {error_details}")

        task_resp.raise_for_status()
        task_id = task_resp.json().get("task_id")

        # 3. Poll for Completion
        max_retries = 9
        for _ in range(max_retries):
            time.sleep(10)
            status_resp = client.get(f"{base_url}/task/{task_id}", headers=headers)
            status_resp.raise_for_status()
            status = status_resp.json().get("status")
            if status == "done":
                break
            elif status == "error":
                raise RuntimeError("AppEEARS task failed on the server.")
        else:
            raise TimeoutError("ECOSTRESS extraction timed out.")

        # 4. Fetch Bundle
        bundle_resp = client.get(f"{base_url}/bundle/{task_id}", headers=headers)
        bundle_resp.raise_for_status()
        files = bundle_resp.json().get("files", [])

        csv_file_id = next((f["file_id"] for f in files if f["file_name"].endswith(".csv")), None)
        if not csv_file_id:
            raise ValueError("No CSV output found in task bundle.")

        # 5. Download Data
        data_resp = client.get(f"{base_url}/bundle/{task_id}/{csv_file_id}", headers=headers)
        data_resp.raise_for_status()

        df = pd.read_csv(io.StringIO(data_resp.text))

        if df.empty or "Date" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # Find the ET column dynamically to avoid naming mismatches
        et_col = [col for col in df.columns if "EVAPOTRANSPIRATION" in col.upper()]
        if et_col:
            df.rename(columns={et_col[0]: "ecostress_et"}, inplace=True)
            return df[["ecostress_et"]].resample("D").interpolate(method="time")

        return pd.DataFrame()
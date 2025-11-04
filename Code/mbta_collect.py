"""
put your MBTA API key in .env as:
MBTA_API_KEY=zzzz
"""
# Code/mbta_collect.py
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "Data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

MBTA_BASE_URL = "https://api.mbta.com"

# Route types: 0–4 (light rail, subway, commuter rail, bus, ferry)
ROUTE_TYPES = [0, 1]  # rail/subway; adjust if you want bus/CR/etc.

def fetch_alerts(start_dt: datetime, end_dt: datetime):
    """
    Fetch alerts between start_dt and end_dt.
    We use the MBTA v3 /alerts endpoint with date filters.
    """
    api_key = os.getenv("MBTA_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}

    params = {
        "filter[route_type]": ",".join(str(t) for t in ROUTE_TYPES),
        "filter[datetime]": f"{start_dt.isoformat()},{end_dt.isoformat()}",
        "page[limit]": 100,
    }

    url = f"{MBTA_BASE_URL}/alerts"
    all_rows = []

    while url:
        print(f"🔍 GET {url}")
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        for item in tqdm(data.get("data", [])):
            attrs = item["attributes"]
            relationships = item.get("relationships", {})

            routes = []
            included_routes = relationships.get("routes", {}).get("data", [])
            for r in included_routes:
                routes.append(r.get("id"))

            all_rows.append(
                {
                    "alert_id": item["id"],
                    "route_ids": ",".join(routes),
                    "effect": attrs.get("effect"),
                    "severity": attrs.get("severity"),
                    "header": attrs.get("header"),
                    "description": attrs.get("description"),
                    "short_header": attrs.get("short_header"),
                    "service_effect": attrs.get("service_effect"),
                    "start_time": attrs.get("active_period", [{}])[0].get("start"),
                    "end_time": attrs.get("active_period", [{}])[0].get("end"),
                }
            )

        # Pagination
        links = data.get("links", {})
        next_link = links.get("next")
        if next_link:
            url = next_link
            params = {}  # next link already has params embedded
        else:
            url = None

    return pd.DataFrame(all_rows)

def main():
    # Example: same 12-month window as Reddit
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt.replace(year=end_dt.year - 1)

    df = fetch_alerts(start_dt, end_dt)
    out_path = RAW_DATA_DIR / "mbta_alerts_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df)} alerts to {out_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()

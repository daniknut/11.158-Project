# Code/clean_data.py
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "Data" / "raw"
PROC_DIR = PROJECT_ROOT / "Data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def to_local(dt_str: str) -> datetime | None:
    """
    Convert ISO or timestamp-like string to naive local-time datetime.
    Assume America/New_York (you can refine later).
    """
    if pd.isna(dt_str):
        return None
    try:
        # If it's ISO with timezone, let pandas handle it then convert
        dt = pd.to_datetime(dt_str, utc=True)
    except Exception:
        # Maybe it's a unix timestamp
        try:
            dt = pd.to_datetime(float(dt_str), unit="s", utc=True)
        except Exception:
            return None
    # Convert to US/Eastern and drop tz info (for simplicity)
    return dt.tz_convert("America/New_York").tz_localize(None)

def clean_reddit():
    in_path = RAW_DIR / "reddit_mbta_raw.csv"
    if not in_path.exists():
        print("⚠️ No reddit_mbta_raw.csv found, skipping Reddit cleaning.")
        return

    df = pd.read_csv(in_path)

    # Drop posts with no title and no body
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df = df[(df["title"].str.strip() != "") | (df["body"].str.strip() != "")]

    # Convert created_utc (float) to local datetime
    df["created_dt_local"] = (
        pd.to_datetime(df["created_utc"], unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

    out_path = PROC_DIR / "reddit_mbta_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned Reddit to {out_path.relative_to(PROJECT_ROOT)}")

def clean_mbta_alerts():
    in_path = RAW_DIR / "mbta_alerts_raw.csv"
    if not in_path.exists():
        print("⚠️ No mbta_alerts_raw.csv found, skipping MBTA cleaning.")
        return

    df = pd.read_csv(in_path)

    # Parse start/end times if present
    if "start_time" in df.columns:
        df["start_dt_local"] = (
            pd.to_datetime(df["start_time"], utc=True, errors="coerce")
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )
    if "end_time" in df.columns:
        df["end_dt_local"] = (
            pd.to_datetime(df["end_time"], utc=True, errors="coerce")
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

    out_path = PROC_DIR / "mbta_alerts_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned MBTA alerts to {out_path.relative_to(PROJECT_ROOT)}")

def clean_weather():
    in_path = RAW_DIR / "weather_boston_raw.csv"
    if not in_path.exists():
        print("⚠️ No weather_boston_raw.csv found, skipping weather cleaning.")
        return

    df = pd.read_csv(in_path)

    # Expect a 'timestamp' column; parse to local
    if "timestamp" in df.columns:
        df["timestamp_local"] = (
            pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

    out_path = PROC_DIR / "weather_boston_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned weather to {out_path.relative_to(PROJECT_ROOT)}")

def main():
    clean_reddit()
    clean_mbta_alerts()
    clean_weather()

if __name__ == "__main__":
    main()

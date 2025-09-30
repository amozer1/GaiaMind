import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent / "data" / "sensors"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = DATA_DIR / "river_levels.csv"

def fetch_river_level(site="07032000"):  
    """
    Fetch river level for a given USGS site code.
    Example: 07032000 = Mississippi River at Memphis, TN
    """
    url = (
        f"https://waterservices.usgs.gov/nwis/iv/?sites={site}&parameterCd=00065"
        "&format=json"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    try:
        value = data["value"]["timeSeries"][0]["values"][0]["value"][-1]["value"]
    except Exception:
        return None

    df = pd.DataFrame([{
        "site": site,
        "river_level_ft": float(value),
        "timestamp": datetime.utcnow().isoformat()
    }])
    df.to_csv(OUT_FILE, index=False)
    return df

if __name__ == "__main__":
    print(fetch_river_level())

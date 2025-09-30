import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent / "data" / "sensors"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = DATA_DIR / "air_quality.csv"

def fetch_air_quality(city=None, location=None, limit=1):
    """Fetch latest PM2.5 values from OpenAQ using /latest endpoint."""
    base = "https://api.openaq.org/v2/latest"
    params = {"parameter": "pm25", "limit": limit}
    if city:
        params["city"] = city
    if location:
        params["location"] = location

    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"OpenAQ HTTP error for city={city}, location={location}: {e}")
        return None
    except Exception as e:
        print(f"Network error: {e}")
        return None

    data = r.json()
    if "results" not in data or not data["results"]:
        print("OpenAQ latest returned no results for", params)
        return None

    # pick first result
    res = data["results"][0]
    measurements = res.get("measurements", [])
    if not measurements:
        print("No measurements in results[0]", res)
        return None

    # find measurement for pm25
    for m in measurements:
        if m.get("parameter") == "pm25":
            val = m.get("value")
            unit = m.get("unit")
            loc_name = res.get("city") or res.get("location") or "unknown"
            df = pd.DataFrame([{
                "location": loc_name,
                "parameter": "pm25",
                "value": val,
                "unit": unit,
                "timestamp": datetime.utcnow().isoformat()
            }])
            df.to_csv(OUT_FILE, index=False)
            return df

    print("No pm25 in measurements list", measurements)
    return None

if __name__ == "__main__":
    result = fetch_air_quality(city="Los Angeles")
    if result is None:
        # optionally try location name instead
        result = fetch_air_quality(location="Los Angeles Airport")
    print("Air quality fetch result:", result)

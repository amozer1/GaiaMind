# fetch_fires_firms.py
import requests
import pandas as pd
from io import StringIO

# Your FIRMS API Key
FIRMS_API_KEY = "7314f250dfb40658685fd53a309a1146"

def fetch_wildfire_data(city, lat, lon):
    """
    Fetch active wildfire data from NASA FIRMS for given city coordinates.
    Returns dict with number of active fires and latest fire date.
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{FIRMS_API_KEY}?lat={lat}&lon={lon}&radius_km=50"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        active_fires = len(df)
        latest_date = df['acq_date'].max() if active_fires > 0 else "N/A"
        return {"city": city, "active_fires": active_fires, "latest_fire_date": latest_date}
    except Exception as e:
        print(f"FIRMS fetch error for {city}: {e}")
        return {"city": city, "active_fires": 0, "latest_fire_date": "N/A"}


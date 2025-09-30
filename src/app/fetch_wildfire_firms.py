# src/app/fetch_wildfire_firms.py
import requests
import pandas as pd
from datetime import datetime

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/csv/MODIS_C6_Global_24h.csv"  # Example CSV

def fetch_firms():
    """Fetch active fire hotspots from NASA FIRMS."""
    try:
        r = requests.get(FIRMS_URL, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(r.text))
        df = df[['latitude', 'longitude', 'brightness']]
        df['timestamp'] = datetime.utcnow()
        return df
    except Exception as e:
        print(f"FIRMS fetch error: {e}")
        return None

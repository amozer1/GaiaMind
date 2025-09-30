"""
fetch_data.py
-------------
Fetch live environmental risk data from multiple APIs:
- Flood risk (NASA GFMS / Tethys)
- Wildfire hotspots (NASA FIRMS)
- Air Quality Index (WAQI)
- News hazard signals (NewsAPI)

Returns scores for aggregation in the main dashboard.
"""

import requests
import pandas as pd

# ================= API KEYS =================
WAQI_TOKEN = "08c63f08a86912f29978ecfb43f0bb6e92897d7e"  # Your WAQI token
FIRMS_KEY = "7314f250dfb40658685fd53a309a1146"          # Your NASA FIRMS key
NEWSAPI_KEY = "3b7046f1708e4bab807a572076209dbd"        # Your NewsAPI key


# ================= Flood Risk =================
def fetch_floods(lat, lon):
    """Fetch flood severity for a given lat/lon from NASA GFMS (Tethys API)."""
    try:
        url = f"http://tethys.icimod.org/apps/floodviewer/api/GetFloodNowcast/?lat={lat}&lon={lon}"
        r = requests.get(url, timeout=10).json()
        if "flood_nowcast" in r:
            return float(r["flood_nowcast"]["severity"])
        return 0
    except Exception as e:
        print(f"Flood fetch error: {e}")
        return 0


# ================= Wildfire Risk =================
def fetch_wildfires(lat, lon, api_key=FIRMS_KEY):
    """Fetch wildfire hotspots near a location from NASA FIRMS."""
    try:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/world/1"
        df = pd.read_csv(url)

        # Filter within ~1 degree box around lat/lon
        nearby = df[(df["latitude"].between(lat - 1, lat + 1)) &
                    (df["longitude"].between(lon - 1, lon + 1))]

        # Score = number of fires capped at 100
        return min(len(nearby) * 10, 100)
    except Exception as e:
        print(f"Wildfire fetch error: {e}")
        return 0


# ================= Air Quality =================
def fetch_air_quality(city, token=WAQI_TOKEN):
    """Fetch AQI for a given city from WAQI API."""
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    try:
        r = requests.get(url, timeout=10).json()
        if r.get("status") == "ok":
            return float(r["data"]["aqi"])
        else:
            print(f"WAQI error for {city}: {r}")
            return 0
    except Exception as e:
        print(f"Air quality fetch error: {e}")
        return 0


# ================= News Risk =================
def fetch_news(keyword="flood OR wildfire OR pollution OR disaster"):
    """Fetch hazard-related headlines from NewsAPI."""
    try:
        url = f"https://newsapi.org/v2/everything?q={keyword}&sortBy=publishedAt&language=en&apiKey={NEWSAPI_KEY}"
        r = requests.get(url, timeout=10).json()
        articles = r.get("articles", [])
        # Score = #articles * 3 (capped at 100)
        return min(len(articles) * 3, 100)
    except Exception as e:
        print(f"News fetch error: {e}")
        return 0

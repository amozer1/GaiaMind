# fetchers.py
import requests
import pandas as pd
from datetime import datetime

# --------------------
# 1. Air Quality (WAQI)
# --------------------
WAQI_TOKEN = "08c63f08a86912f29978ecfb43f0bb6e92897d7e"

def fetch_air_quality(city):
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
        res = requests.get(url, timeout=10)
        data = res.json()
        if data["status"] == "ok":
            aqi = data["data"]["aqi"]
            return {"city": city, "aqi": aqi}
        else:
            return {"city": city, "aqi": None}
    except Exception as e:
        print(f"Air fetch error for {city}: {e}")
        return {"city": city, "aqi": None}

# --------------------
# 2. Wildfires (NASA FIRMS)
# --------------------
FIRMS_MAP_KEY = "7314f250dfb40658685fd53a309a1146"

def fetch_wildfires(lat, lon):
    try:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{FIRMS_MAP_KEY}/USA/"
        # For simplicity, we fetch sample CSV; in production, adjust for lat/lon bounding box
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(res.text))
            # Filter nearby fires
            nearby = df[(df['latitude'] - lat).abs() < 1.0 & (df['longitude'] - lon).abs() < 1.0]
            total = len(nearby)
            latest = nearby['acq_date'].max() if total > 0 else "N/A"
            return {"lat": lat, "lon": lon, "active_fires": total, "latest": latest}
        else:
            return {"lat": lat, "lon": lon, "active_fires": 0, "latest": "N/A"}
    except Exception as e:
        print(f"FIRMS fetch error for ({lat}, {lon}): {e}")
        return {"lat": lat, "lon": lon, "active_fires": 0, "latest": "N/A"}

# --------------------
# 3. Flood Risk (NASA MODAPS Flood API)
# --------------------
def fetch_flood_risk(lat, lon):
    try:
        url = f"https://floodmap.modaps.eosdis.nasa.gov/floods/api?lat={lat}&lon={lon}"
        res = requests.get(url, timeout=10)
        data = res.json()
        risk_score = data.get("risk_score", None)
        return {"lat": lat, "lon": lon, "risk_score": risk_score}
    except Exception as e:
        print(f"Flood fetch error for ({lat}, {lon}): {e}")
        return {"lat": lat, "lon": lon, "risk_score": None}

# --------------------
# 4. Environmental News (NewsAPI)
# --------------------
NEWS_API_KEY = "3b7046f1708e4bab807a572076209dbd"

def fetch_news(query="environment", page_size=5):
    try:
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&"
            f"pageSize={page_size}&"
            f"sortBy=publishedAt&"
            f"language=en&"
            f"apiKey={NEWS_API_KEY}"
        )
        res = requests.get(url, timeout=10)
        data = res.json()
        articles = []
        if data.get("status") == "ok":
            for art in data.get("articles", []):
                articles.append({
                    "title": art.get("title", "No Title"),
                    "url": art.get("url", "#"),
                    "description": art.get("description", ""),
                    "source": {"name": art.get("source", {}).get("name", "Unknown")},
                    "publishedAt": art.get("publishedAt", "")
                })
        return articles
    except Exception as e:
        print(f"News fetch error: {e}")
        return []

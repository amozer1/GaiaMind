# src/app/risk_calculator.py
import pandas as pd
from datetime import datetime
from fetch_air_waqi import fetch_air_waqi
from fetch_flood_gfms import fetch_gfms
from fetch_wildfire_firms import fetch_firms
from fetch_news_gdelt import fetch_gdelt

REGIONS = [
    {"name": "Los Angeles", "country": "USA", "lat": 34.0522, "lon": -118.2437},
    {"name": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
    {"name": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
    {"name": "Paris", "country": "FR", "lat": 48.8566, "lon": 2.3522},
    {"name": "Tokyo", "country": "JP", "lat": 35.6762, "lon": 139.6503}
]

def risk_color(risk):
    if risk < 40: return [0,200,0]
    elif risk < 70: return [255,165,0]
    else: return [255,0,0]

def get_risk_dataframe():
    rows = []
    flood_df = fetch_gfms()
    wildfire_df = fetch_firms()
    news_df = fetch_gdelt()

    for region in REGIONS:
        city = region["name"]

        # Flood score: nearest GFMS point
        if flood_df is not None and not flood_df.empty:
            flood_point = flood_df.iloc[(flood_df['latitude'] - region['lat']).abs().argsort()[:1]]
            flood_score = float(flood_point['flood_depth'])
        else:
            flood_score = 0

        # Wildfire score: nearest FIRMS brightness
        if wildfire_df is not None and not wildfire_df.empty:
            fire_point = wildfire_df.iloc[(wildfire_df['latitude'] - region['lat']).abs().argsort()[:1]]
            wildfire_score = float(fire_point['brightness'])
        else:
            wildfire_score = 0

        # Air quality
        air_df = fetch_air_waqi(city)
        air_score = float(air_df["value"].values[0]) if air_df is not None else 0

        # News score: number of GDELT articles mentioning city
        news_score = len(news_df[news_df['title'].str.contains(city, case=False)]) if not news_df.empty else 0

        # Composite risk index
        risk_index = round((flood_score + wildfire_score + air_score + news_score)/4, 1)

        rows.append({
            "region": city,
            "country": region["country"],
            "lat": region["lat"],
            "lon": region["lon"],
            "flood_score": flood_score,
            "wildfire_score": wildfire_score,
            "air_score": air_score,
            "news_score": news_score,
            "risk_index": risk_index,
            "color": risk_color(risk_index),
            "timestamp": datetime.utcnow().isoformat()
        })
    return pd.DataFrame(rows)

import requests
import pandas as pd
import plotly.express as px

WAQI_TOKEN = "08c63f08a86912f29978ecfb43f0bb6e92897d7e"

def fetch_air_quality(city):
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("status") != "ok":
            return {"city": city, "aqi": None}
        return {"city": city, "aqi": data["data"]["aqi"]}
    except Exception as e:
        return {"city": city, "aqi": None, "error": str(e)}

def air_quality_summary(df):
    stats = {
        "Average AQI": df["aqi"].mean(skipna=True),
        "Worst AQI": df["aqi"].max(skipna=True),
        "Best AQI": df["aqi"].min(skipna=True),
    }
    return stats

def plot_air_quality(df):
    fig = px.bar(df, x="city", y="aqi", title="Air Quality Index by City",
                 labels={"aqi": "AQI", "city": "City"}, color="aqi")
    return fig

# fetch_air_waqi.py
import requests

# Your WAQI API token
WAQI_TOKEN = "08c63f08a86912f29978ecfb43f0bb6e92897d7e"

def fetch_air_quality(city):
    """
    Fetch air quality data for a given city from WAQI API.
    Returns a dictionary with AQI and main pollutants.
    """
    url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            print(f"Air fetch error for {city}: WAQI fetch failed for {city}")
            return {"city": city, "aqi": -1, "pm2_5": None, "pm10": None, "co": None, "o3": None}
        iaqi = data['data'].get('iaqi', {})
        return {
            "city": city,
            "aqi": data['data'].get('aqi', -1),
            "pm2_5": iaqi.get('pm25', {}).get('v'),
            "pm10": iaqi.get('pm10', {}).get('v'),
            "co": iaqi.get('co', {}).get('v'),
            "o3": iaqi.get('o3', {}).get('v')
        }
    except Exception as e:
        print(f"Air fetch error for {city}: {e}")
        return {"city": city, "aqi": -1, "pm2_5": None, "pm10": None, "co": None, "o3": None}

# fetch_flood_gfms.py
import requests

def fetch_flood_risk(city, lat, lon):
    """
    Fetch flood risk for a given city and coordinates.
    Uses NASA GFMS free data (Earthdata login not needed for demo).
    Returns dict with risk_score and location_name.
    """
    url = f"https://floodmap.modaps.eosdis.nasa.gov/floods/api?lat={lat}&lon={lon}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        risk_score = data.get("risk_score", -1)
        return {"city": city, "risk_score": risk_score, "location_name": city}
    except Exception as e:
        print(f"Flood fetch error for {city}: {e}")
        return {"city": city, "risk_score": -1, "location_name": city}


import requests
import pandas as pd
import plotly.express as px

# NASA FIRMS API Key
FIRMS_API_KEY = "7314f250dfb40658685fd53a309a1146"

# ----------------------
# Fetch Wildfire Data
# ----------------------
def fetch_wildfires(lat, lon, radius_km=100):
    """
    Fetch wildfire data from NASA FIRMS API for a given location.
    Returns a DataFrame with fire details.
    """
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/"
        f"{FIRMS_API_KEY}/VIIRS_SNPP_NRT/world/1"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        # Convert CSV text into DataFrame
        df = pd.read_csv(pd.compat.StringIO(response.text))

        # Filter around given location within radius
        df["distance"] = ((df["latitude"] - lat) ** 2 + (df["longitude"] - lon) ** 2) ** 0.5 * 111  # km approx
        nearby_fires = df[df["distance"] <= radius_km].copy()

        if not nearby_fires.empty:
            nearby_fires = nearby_fires[["latitude", "longitude", "acq_date", "confidence"]]

        return nearby_fires

    except Exception as e:
        print(f"FIRMS fetch error for ({lat}, {lon}): {e}")
        return pd.DataFrame()

# ----------------------
# Plot Wildfires
# ----------------------
def plot_wildfires(df, city="Selected Area"):
    """
    Create a scatter mapbox plot of wildfire locations.
    """
    if df.empty:
        return None

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="confidence",
        size_max=15,
        zoom=4,
        hover_name="acq_date",
        color_continuous_scale="Reds",
        title=f"🔥 Wildfires near {city}",
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

    return fig

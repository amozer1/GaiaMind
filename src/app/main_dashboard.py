# main_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
from fetchers import fetch_air_quality, fetch_wildfires, fetch_flood_risk, fetch_news

# --------------------
# Page Config
# --------------------
st.set_page_config(
    page_title="🌍 GaiaMind Environmental Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------
# Sidebar - Instructions & Info
# --------------------
st.sidebar.title("GaiaMind 🌍")
st.sidebar.info(
    """
    **GaiaMind Environmental Dashboard**

    This application provides real-time monitoring of:
    - Air Quality (AQI)
    - Active Wildfires
    - Flood Risk
    - Environmental News

    **Instructions:**
    1. Select the cities you want to monitor.
    2. Explore the metrics, charts, and maps.
    3. Hover on graphs for detailed info.
    """
)

# --------------------
# Cities and Coordinates
# --------------------
cities = {
    "Los Angeles": (34.05, -118.25),
    "New York": (40.71, -74.01),
    "London": (51.51, -0.13),
    "Paris": (48.85, 2.35),
    "Tokyo": (35.68, 139.76)
}

selected_cities = st.sidebar.multiselect(
    "Select Cities", list(cities.keys()), default=list(cities.keys())
)


# --------------------
# Cache wrapper
# --------------------
@st.cache_data
def cache_fetch(_func, *args):
    try:
        return _func(*args)
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
        return None


# --------------------
# Fetch Data
# --------------------
aq_data = [cache_fetch(fetch_air_quality, city) for city in selected_cities]
wf_data = [cache_fetch(fetch_wildfires, lat, lon) for city, (lat, lon) in cities.items() if city in selected_cities]
fl_data = [cache_fetch(fetch_flood_risk, lat, lon) for city, (lat, lon) in cities.items() if city in selected_cities]
news_data = cache_fetch(fetch_news)


# --------------------
# Convert to DataFrames
# --------------------
def to_dataframe(data, columns):
    return pd.DataFrame([d for d in data if d], columns=columns)


aq_df = to_dataframe(aq_data, ["city", "aqi"])
wf_df = to_dataframe(wf_data, ["lat", "lon", "active_fires", "latest"])
fl_df = to_dataframe(fl_data, ["lat", "lon", "risk_score"])

# Ensure numeric for metrics
if not aq_df.empty:
    aq_df['aqi'] = pd.to_numeric(aq_df['aqi'], errors='coerce')
if not fl_df.empty:
    fl_df['risk_score'] = pd.to_numeric(fl_df['risk_score'], errors='coerce').fillna(0)
if not wf_df.empty:
    wf_df['active_fires'] = pd.to_numeric(wf_df['active_fires'], errors='coerce').fillna(0)

# --------------------
# Dashboard Header
# --------------------
st.title("🌍 GaiaMind Environmental Dashboard")
st.markdown(
    "Real-time monitoring of Air Quality, Wildfires, Flood Risk, and Environmental News"
)

st.divider()

# --------------------
# Air Quality Metrics
# --------------------
st.subheader("🌬️ Air Quality")
avg_aqi = aq_df['aqi'].mean() if not aq_df.empty and not aq_df['aqi'].isna().all() else "N/A"
st.metric(label="Average AQI", value=f"{avg_aqi:.1f}" if isinstance(avg_aqi, (int, float)) else avg_aqi)

if not aq_df.empty:
    fig_aq = px.bar(aq_df, x='city', y='aqi', color='aqi', color_continuous_scale='RdYlGn_r',
                    title="AQI by City")
    st.plotly_chart(fig_aq, use_container_width=True)

st.divider()

# --------------------
# Wildfires Metrics
# --------------------
st.subheader("🔥 Wildfires")
total_fires = wf_df['active_fires'].sum() if not wf_df.empty else 0
st.metric(label="Total Active Fires", value=int(total_fires))

if not wf_df.empty:
    fig_wf = px.scatter_mapbox(
        wf_df, lat='lat', lon='lon', size='active_fires', color='active_fires',
        color_continuous_scale='Oranges', size_max=25, zoom=1,
        hover_data={'active_fires': True, 'latest': True},
        title="Active Wildfires Map"
    )
    fig_wf.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_wf, use_container_width=True)

st.divider()

# --------------------
# Flood Risk Metrics
# --------------------
st.subheader("🌊 Flood Risk")
avg_risk = fl_df['risk_score'].mean() if not fl_df.empty else "N/A"
st.metric(label="Average Flood Risk", value=f"{avg_risk:.1f}" if isinstance(avg_risk, (int, float)) else avg_risk)

if not fl_df.empty:
    fig_fl = px.scatter_mapbox(
        fl_df, lat='lat', lon='lon', size='risk_score', color='risk_score',
        color_continuous_scale='Blues', size_max=25, zoom=1,
        title="Flood Risk Map"
    )
    fig_fl.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_fl, use_container_width=True)

st.divider()

# --------------------
# Environmental News
# --------------------
st.subheader("📰 Environmental News")
if news_data and isinstance(news_data, list):
    for article in news_data[:5]:
        title = article.get('title', 'No Title')
        url = article.get('url', '#')
        source = article.get('source', {}).get('name', 'Unknown')
        desc = article.get('description', '')
        st.markdown(f"**[{title}]({url})** - Source: {source}")
        st.caption(desc)
        st.divider()
else:
    st.info("No news data available at the moment.")

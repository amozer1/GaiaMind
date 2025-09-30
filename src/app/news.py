import requests
import pandas as pd
import plotly.express as px

# Your News API key
NEWS_API_KEY = "3b7046f1708e4bab807a572076209dbd"

# ----------------------
# Fetch Environmental News
# ----------------------
def fetch_news(query="environment", language="en", page_size=10):
    """
    Fetches latest environmental news using NewsAPI.
    Returns a DataFrame with title, source, url, and published date.
    """
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language={language}&pageSize={page_size}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "title": a["title"],
                "source": a["source"]["name"],
                "url": a["url"],
                "published": a["publishedAt"][:10]
            }
            for a in articles
        ])

        return df

    except Exception as e:
        print(f"News fetch error: {e}")
        return pd.DataFrame()

# ----------------------
# Plot News Timeline
# ----------------------
def plot_news(df):
    """
    Plots a bar chart showing number of articles by date.
    """
    if df.empty:
        return None

    count_by_date = df.groupby("published").size().reset_index(name="articles")

    fig = px.bar(
        count_by_date,
        x="published",
        y="articles",
        title="📰 Environmental News Over Time",
        labels={"published": "Date", "articles": "Number of Articles"},
        color="articles",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_tickangle=-45)

    return fig

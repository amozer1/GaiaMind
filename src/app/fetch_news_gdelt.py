# fetch_news_gdelt.py
import requests

# Your News API key
NEWS_API_KEY = "3b7046f1708e4bab807a572076209dbd"

def fetch_env_news(query="environment", max_results=5):
    """
    Fetch latest environment-related news articles.
    Returns a list of dicts with title, url, and publishedAt.
    """
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={max_results}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        return [{"title": a["title"], "url": a["url"], "publishedAt": a["publishedAt"]} for a in articles]
    except Exception as e:
        print(f"GDELT fetch error: {e}")
        return []

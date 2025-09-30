import requests

NEWS_API_KEY = "3b7046f1708e4bab807a572076209dbd"

def fetch_env_news():
    url = f"https://newsapi.org/v2/everything?q=environment&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data.get("articles", [])[:5]
    except Exception as e:
        return [{"title": "Error fetching news", "description": str(e), "url": ""}]

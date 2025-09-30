import requests
import json
from pathlib import Path

news_path = Path(r"C:\Users\adane\GaiaMind\data\social_news\latest_news.json")

# Example: pull JSON from Twitter API or news API (replace with your keys & endpoint)
news_url = "https://example.com/api/fake_news.json"
r = requests.get(news_url)
news_path.parent.mkdir(exist_ok=True)
with open(news_path, "w") as f:
    f.write(r.text)

print("News updated:", news_path)

from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd, json

ROOT = Path(__file__).resolve().parent
SAT = ROOT / "src" / "app" / "data" / "satellite"
SENS = ROOT / "src" / "app" / "data" / "sensors"
NEWS = ROOT / "src" / "app" / "data" / "social_news"

SAT.mkdir(parents=True, exist_ok=True)
SENS.mkdir(parents=True, exist_ok=True)
NEWS.mkdir(parents=True, exist_ok=True)

# 1) Satellite image (placeholder)
img = Image.new("RGB", (256, 256), color=(173, 216, 230))
d = ImageDraw.Draw(img)
d.text((10, 10), "GaiaMind Sample Satellite", fill=(0, 0, 0))
img.save(SAT / "sample_region.jpg")

# 2) Sensor data
river_df = pd.DataFrame({
    "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=12, freq="H"),
    "river_level": [2.0, 2.1, 2.3, 2.5, 2.8, 3.0, 3.5, 3.9, 4.1, 4.5, 4.8, 5.2]
})
river_df.to_csv(SENS / "river_levels.csv", index=False)

aqi_df = pd.DataFrame({
    "date": pd.date_range(end=pd.Timestamp.now(), periods=10, freq="D"),
    "PM2.5": [10, 12, 18, 25, 30, 22, 28, 35, 40, 20]
})
aqi_df.to_csv(SENS / "daily_aqi_by_cbsa_2025.csv", index=False)

# 3) Social/news JSON
news = {"tweets": [
    {"text": "Flood risk rising near Memphis"},
    {"text": "Air quality worsening in region X"},
    {"text": "Wildfire risk remains low"}
]}
with open(NEWS / "sample_news.json", "w", encoding="utf8") as f:
    json.dump(news, f, ensure_ascii=False, indent=2)

print("✅ Sample data created at", SAT, SENS, NEWS)

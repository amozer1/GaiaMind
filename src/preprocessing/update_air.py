import requests
import pandas as pd
from pathlib import Path

air_csv_path = Path(r"C:\Users\adane\GaiaMind\data\sensors\air_quality.csv")

# Example: download a sample PM2.5 CSV from EPA (replace URL with actual dataset)
air_url = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2024.csv"
r = requests.get(air_url)
with open(air_csv_path, "wb") as f:
    f.write(r.content)

print("Air quality updated:", air_csv_path)

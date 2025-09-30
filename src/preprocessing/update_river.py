import requests
import json
import pandas as pd
from pathlib import Path

river_json_path = Path(r"C:\Users\adane\GaiaMind\data\sensors\mississippi_river.json")
river_csv_path = Path(r"C:\Users\adane\GaiaMind\data\sensors\mississippi_river.csv")

# Download USGS JSON
river_url = "https://waterservices.usgs.gov/nwis/iv/?format=json&sites=01646500&parameterCd=00060&period=P7D"
r = requests.get(river_url)
with open(river_json_path, "w") as f:
    f.write(r.text)

# Convert JSON -> CSV
with open(river_json_path) as f:
    data = json.load(f)

ts = []
for value in data['value']['timeSeries'][0]['values'][0]['value']:
    ts.append([value['dateTime'], float(value['value'])])

df = pd.DataFrame(ts, columns=['datetime','river_level'])
df.to_csv(river_csv_path, index=False)

print("River levels updated:", river_csv_path)

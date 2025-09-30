import requests
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# --- Step 1: Download USGS River Level JSON ---
river_json_path = Path(r"C:\Users\adane\GaiaMind\data\sensors\mississippi_river.json")
river_csv_path = Path(r"C:\Users\adane\GaiaMind\data\sensors\mississippi_river.csv")

print("Downloading USGS river level data...")
river_url = "https://waterservices.usgs.gov/nwis/iv/?format=json&sites=01646500&parameterCd=00060&period=P7D"
r = requests.get(river_url)
with open(river_json_path, "w") as f:
    f.write(r.text)
print("Saved JSON:", river_json_path)

# --- Step 2: Convert JSON -> CSV ---
with open(river_json_path) as f:
    data = pd.json_normalize(pd.read_json(f).loc[0]) if False else None  # placeholder

import json
with open(river_json_path) as f:
    data = json.load(f)

ts = []
for value in data['value']['timeSeries'][0]['values'][0]['value']:
    ts.append([value['dateTime'], float(value['value'])])

river_df = pd.DataFrame(ts, columns=['datetime','river_level'])
river_df.to_csv(river_csv_path, index=False)
print("Converted to CSV:", river_csv_path)

# --- Step 3: Load EPA Air Quality CSV ---
# You need to download a CSV manually first, e.g., PM2.5 for a city
air_csv_path = Path(r"C:\Users\adane\GaiaMind\data\sensors\air_quality.csv")
try:
    air_df = pd.read_csv(air_csv_path)
    print("Loaded air quality CSV:", air_csv_path)
except:
    print("Air quality CSV not found. Place a CSV at:", air_csv_path)
    air_df = None

# --- Step 4: LSTM Model Definition ---
class SensorLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Step 5: Run LSTM on River Levels ---
series = torch.tensor(river_df['river_level'].values, dtype=torch.float32).unsqueeze(-1)
input_seq = series.unsqueeze(0)  # batch dimension

model = SensorLSTM()
model.eval()
predicted_river_level = model(input_seq)
print("Predicted next river level:", predicted_river_level.item())

# --- Step 6: Run LSTM on Air Quality (Optional) ---
if air_df is not None:
    # Example: use first numeric column after datetime
    air_series = torch.tensor(air_df.iloc[:,1].values, dtype=torch.float32).unsqueeze(-1)
    air_input = air_series.unsqueeze(0)
    predicted_air = model(air_input)
    print("Predicted next air quality value:", predicted_air.item())

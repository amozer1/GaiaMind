from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
import folium
from pathlib import Path
import torch
import torch.nn as nn

app = FastAPI(title="GaiaMind Environmental Dashboard")

# --- LSTM Model (reuse from sensors pipeline) ---
class SensorLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Function to load data and predict ---
def get_predictions():
    predictions = []

    # River levels
    river_csv = Path(r"C:\Users\adane\GaiaMind\data\sensors\mississippi_river.csv")
    if river_csv.exists():
        df = pd.read_csv(river_csv)
        series = torch.tensor(df['river_level'].values, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
        model = SensorLSTM()
        model.eval()
        pred = model(series).item()
        predictions.append({
            "name": "Mississippi River Level",
            "value": round(pred,2),
            "lat": 35.15,   # example coords for Memphis, TN
            "lon": -90.05
        })

    # Air quality
    air_csv = Path(r"C:\Users\adane\GaiaMind\data\sensors\daily_aqi_by_cbsa_2025.csv")
    if air_csv.exists():
        df = pd.read_csv(air_csv)
        series = torch.tensor(df.iloc[:,1].values, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
        model = SensorLSTM()
        model.eval()
        pred = model(series).item()
        predictions.append({
            "name": "Air Quality PM2.5",
            "value": round(pred,2),
            "lat": 38.90,   # example coords for Washington DC
            "lon": -77.03
        })

    return predictions

# --- Route for dashboard ---
@app.get("/", response_class=HTMLResponse)
def dashboard():
    predictions = get_predictions()

    # Create Folium map
    m = folium.Map(location=[37, -95], zoom_start=4)

    # Add markers for each prediction
    for p in predictions:
        folium.Marker(
            location=[p["lat"], p["lon"]],
            popup=f"{p['name']}: {p['value']}",
            icon=folium.Icon(color="red" if "River" in p['name'] else "green")
        ).add_to(m)

    return m._repr_html_()

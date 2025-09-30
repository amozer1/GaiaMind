from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import folium
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms

app = FastAPI(title="GaiaMind Full Dashboard")

# -----------------------------
# --- Satellite CNN Module ---
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32*32*32, 2)  # adjust for 128x128 input

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def predict_satellite():
    data_dir = Path(r"C:\Users\adane\GaiaMind\data\satellite")
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    cnn = SimpleCNN()
    cnn.eval()
    predictions = []
    for img_path in data_dir.glob("*.png"):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        out = cnn(tensor)
        pred_class = torch.argmax(out, dim=1).item()
        predictions.append({
            "name": img_path.stem,
            "value": f"Class {pred_class}",
            "lat": 40.0,  # placeholder lat/lon
            "lon": -100.0
        })
    return predictions

# -----------------------------
# --- Sensor LSTM Module ---
# -----------------------------
class SensorLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def predict_sensors():
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
            "lat": 35.15,
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
            "lat": 38.90,
            "lon": -77.03
        })
    return predictions

# -----------------------------
# --- NLP News Module ---
# -----------------------------
def predict_news():
    predictions = []
    news_dir = Path(r"C:\Users\adane\GaiaMind\data\social_news")
    for json_file in news_dir.glob("*.json"):
        # Example: flag any news with "flood" or "wildfire"
        import json
        with open(json_file) as f:
            data = json.load(f)
        for item in data.get("tweets", []):
            text = item.get("text","").lower()
            if "flood" in text or "wildfire" in text:
                predictions.append({
                    "name": "News Alert",
                    "value": text[:50]+"...",
                    "lat": 39.0,   # placeholder coordinates
                    "lon": -95.0
                })
    return predictions

# -----------------------------
# --- Dashboard Route ---
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    all_predictions = predict_satellite() + predict_sensors() + predict_news()
    m = folium.Map(location=[37, -95], zoom_start=4)
    for p in all_predictions:
        folium.Marker(
            location=[p["lat"], p["lon"]],
            popup=f"{p['name']}: {p['value']}",
            icon=folium.Icon(color="red" if "River" in p['name'] or "News" in p['name'] else "green")
        ).add_to(m)
    return m._repr_html_()

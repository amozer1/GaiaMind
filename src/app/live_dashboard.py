from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import folium
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms
import json

app = FastAPI(title="GaiaMind Live Dashboard")

# -----------------------------
# --- Satellite CNN Module ---
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32*32*32, 2)  # for 128x128 images

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def predict_satellite():
    predictions = []
    try:
        data_dir = Path(r"C:\Users\adane\GaiaMind\data\satellite")
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])
        cnn = SimpleCNN()
        cnn.eval()
        for img_path in data_dir.glob("*.png"):
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            out = cnn(tensor)
            pred_class = torch.argmax(out, dim=1).item()
            predictions.append({
                "name": img_path.stem,
                "value": f"Class {pred_class}",
                "lat": 40.0,
                "lon": -100.0
            })
    except Exception as e:
        print(f"Error in predict_satellite: {e}")
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

    # --- River Levels ---
    river_csv = Path(r"C:\Users\adane\GaiaMind\data\sensors\mississippi_river.csv")
    if river_csv.exists():
        try:
            df = pd.read_csv(river_csv)
            if 'river_level' in df.columns:
                series = torch.tensor(df['river_level'].values, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
                model = SensorLSTM()
                model.eval()
                pred = model(series).item()
                predictions.append({
                    "name": "Mississippi River Level",
                    "value": round(pred, 2),
                    "lat": 35.15,
                    "lon": -90.05
                })
            else:
                print(f"Warning: 'river_level' column not found in {river_csv}")
        except Exception as e:
            print(f"Error processing {river_csv}: {e}")

    # --- Air Quality using daily_aqi_by_cbsa_2025.csv ---
    air_csv = Path(r"C:\Users\adane\GaiaMind\data\sensors\daily_aqi_by_cbsa_2025.csv")
    if air_csv.exists():
        try:
            df = pd.read_csv(air_csv)
            # Detect PM2.5 or AQI column
            aqi_col = None
            for col in df.columns:
                if "PM2.5" in col or "AQI" in col:
                    aqi_col = col
                    break

            if aqi_col:
                series = torch.tensor(df[aqi_col].values, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
                model = SensorLSTM()
                model.eval()
                pred = model(series).item()
                predictions.append({
                    "name": f"Air Quality ({aqi_col})",
                    "value": round(pred, 2),
                    "lat": 38.90,
                    "lon": -77.03
                })
            else:
                print(f"Warning: No PM2.5 or AQI column found in {air_csv}")
        except Exception as e:
            print(f"Error processing {air_csv}: {e}")

    return predictions

# -----------------------------
# --- NLP News Module ---
# -----------------------------
def predict_news():
    predictions = []
    try:
        news_dir = Path(r"C:\Users\adane\GaiaMind\data\social_news")
        for json_file in news_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            for item in data.get("tweets", []):
                text = item.get("text","").lower()
                if "flood" in text or "wildfire" in text:
                    predictions.append({
                        "name": "News Alert",
                        "value": text[:50]+"...",
                        "lat": 39.0,
                        "lon": -95.0
                    })
    except Exception as e:
        print(f"Error in predict_news: {e}")
    return predictions

# -----------------------------
# --- Safe Predictor Wrapper ---
# -----------------------------
def safe_predict(func):
    try:
        result = func()
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        return []

# -----------------------------
# --- JSON Endpoint ---
# -----------------------------
@app.get("/api/predictions", response_class=JSONResponse)
def api_predictions():
    all_predictions = (
        safe_predict(predict_satellite) +
        safe_predict(predict_sensors) +
        safe_predict(predict_news)
    )
    return {"predictions": all_predictions}

# -----------------------------
# --- Dashboard Route ---
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GaiaMind Live Dashboard</title>
        <meta http-equiv="refresh" content="120"> <!-- refresh every 2 mins -->
    </head>
    <body>
        <h2>GaiaMind Live Environmental Map (updates every 2 min)</h2>
        <iframe src="/map" width="100%" height="700"></iframe>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# -----------------------------
# --- Map Route ---
# -----------------------------
@app.get("/map", response_class=HTMLResponse)
def map_view():
    all_predictions = (
        safe_predict(predict_satellite) +
        safe_predict(predict_sensors) +
        safe_predict(predict_news)
    )
    m = folium.Map(location=[37, -95], zoom_start=4)
    for p in all_predictions:
        folium.Marker(
            location=[p["lat"], p["lon"]],
            popup=f"{p['name']}: {p['value']}",
            icon=folium.Icon(color="red" if "River" in p['name'] or "News" in p['name'] else "green")
        ).add_to(m)
    return m._repr_html_()

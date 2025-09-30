import torch
import torch.nn as nn
import pandas as pd

# --- Load Data ---
river_df = pd.read_csv(r"C:\Users\adane\GaiaMind\data\sensors\mississippi_river.csv")
series = torch.tensor(river_df['river_level'].values, dtype=torch.float32).unsqueeze(-1)  # [time,1]

# --- Define LSTM Model ---
class SensorLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Initialize and run model ---
model = SensorLSTM()
model.eval()  # we just run forward, no training yet

# Add batch dimension: [1, time, features]
input_seq = series.unsqueeze(0)
predicted = model(input_seq)
print("Predicted next river level:", predicted.item())

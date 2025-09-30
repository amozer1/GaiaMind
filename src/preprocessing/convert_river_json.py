import pandas as pd
import json

json_path = r"C:\Users\adane\GaiaMind\data\sensors\mississippi_river.json"
csv_path = r"C:\Users\adane\GaiaMind\data\sensors\mississippi_river.csv"

with open(json_path) as f:
    data = json.load(f)

ts = []
for value in data['value']['timeSeries'][0]['values'][0]['value']:
    ts.append([value['dateTime'], float(value['value'])])

df = pd.DataFrame(ts, columns=['datetime','river_level'])
df.to_csv(csv_path, index=False)

print("River CSV created:", csv_path)

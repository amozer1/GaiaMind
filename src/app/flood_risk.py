import pandas as pd
import numpy as np
import plotly.express as px

def get_flood_risk_data(cities):
    """
    Simulate flood risk scores for the selected cities.
    Replace this with real API/ML model outputs later.
    """
    df = pd.DataFrame({"city": cities})
    # Fake risk scores between 0 and 100
    df["risk_score"] = np.random.randint(0, 100, size=len(cities))
    return df

def plot_flood_risk(df):
    """
    Create a bar chart showing flood risk per city.
    """
    fig = px.bar(
        df,
        x="city",
        y="risk_score",
        title="Flood Risk by City",
        labels={"risk_score": "Risk Score (0-100)", "city": "City"},
        color="risk_score",
        color_continuous_scale="Blues"
    )
    return fig


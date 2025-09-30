GaiaMind 🌍 Environmental Dashboard

GaiaMind is an AI-powered environmental monitoring dashboard providing real-time insights into air quality, wildfires, flood risk, and environmental news for selected cities. The platform is designed to support data-driven decision-making, urban planning, and environmental awareness.

Features

Air Quality Monitoring (AQI)
Visualise and track real-time air quality metrics for multiple cities, with average AQI and historical trends.

Wildfire Tracking
Monitor active wildfires around the world, with geospatial mapping of affected areas.

Flood Risk Assessment
Evaluate current flood risk levels using real-time data sources and predictive scoring.

Environmental News Feed
Stay updated with the latest news on environmental events, disasters, and policies from multiple sources.

Interactive Interface
Select multiple cities, explore metrics, charts, and maps, and hover over graphs for detailed information.

Demo

Live App: GaiaMind on Streamlit

Installation

Clone the repository:

git clone https://github.com/amozer1/GaiaMind.git
cd GaiaMind


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Run the app locally:

streamlit run src/app/main_dashboard.py

Data Sources & Integration

Air Quality: OpenAQ API

Wildfires: NASA FIRMS or other public APIs

Flood Risk: Global flood risk datasets and scoring algorithms

Environmental News: Aggregated from reliable news APIs

The system integrates multiple real-time APIs and processes data for live dashboard visualisation. Caching mechanisms ensure responsive performance.

Technology Stack

Frontend: Streamlit

Backend / Data Processing: Python, Pandas, NumPy, requests

Geospatial Mapping: Folium

Machine Learning / AI: Predictive scoring for flood and AQI risk (optional extensions)

Deployment: Streamlit Cloud

Usage Instructions

Select the cities you want to monitor from the dropdown menu.

Explore the Air Quality, Wildfire, and Flood Risk metrics.

View interactive charts and maps to understand trends and risk areas.

Read environmental news related to the selected regions.

Future Improvements

Implement AI-based predictive models for AQI and flood risk.

Add historical data analysis and trend forecasting.

Enable notifications for high-risk events.

Enhance news feed with NLP summarisation of critical alerts.

Author

Ebenezer Amoako – Lead Machine Learning Engineer & PhD Candidate in Computer Science
Portfolio - www.linkedin.com/in/
ebenezer-amoako


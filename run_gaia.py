import subprocess
import os

# --- Activate virtual environment ---
venv_python = r"C:\Users\adane\GaiaMind\venv\Scripts\python.exe"

# --- Step 1: Update all data feeds ---
print("Updating all sensor & news data...")
subprocess.run([venv_python, r"C:\Users\adane\GaiaMind\src\preprocessing\update_all.py"])

# --- Step 2: Launch FastAPI live dashboard ---
print("Launching GaiaMind Live Dashboard...")
os.system(f'{venv_python} -m uvicorn src.app.live_dashboard:app --reload')

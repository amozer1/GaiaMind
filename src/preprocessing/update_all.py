import subprocess

print("Updating river levels...")
subprocess.run(["python", r"C:\Users\adane\GaiaMind\src\preprocessing\update_river.py"])

print("Updating air quality...")
subprocess.run(["python", r"C:\Users\adane\GaiaMind\src\preprocessing\update_air.py"])

print("Updating news feeds...")
subprocess.run(["python", r"C:\Users\adane\GaiaMind\src\preprocessing\update_news.py"])

print("All data updated successfully!")

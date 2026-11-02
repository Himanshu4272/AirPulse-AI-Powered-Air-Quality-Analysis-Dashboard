from pathlib import Path
import sys
import pandas as pd

# Locate the actual AirPulse_AI project folder
BASE_DIR = Path(__file__).resolve().parent / "AirPulse_AI"

# Add project folder to Python's import path
sys.path.insert(0, str(BASE_DIR))

from src.visualizations import create_aircraft_map

# Dataset and output paths
DATA_FILE = BASE_DIR / "data" / "processed" / "aircraft_history.csv"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load historical aircraft data
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"Total records: {len(df)}")

# Generate geographical map
fig = create_aircraft_map(df)

fig.update_layout(
    title="Global Geographical Distribution of Observed Aircraft",
    height=650,
    width=1100
)

# Save PNG only
png_path = REPORTS_DIR / "figure_13_aircraft_geographical_map.png"

try:
    fig.write_image(
        str(png_path),
        width=1400,
        height=800,
        scale=2
    )
    print(f"PNG map saved successfully: {png_path}")

except Exception as error:
    print(f"PNG export failed: {error}")
    print('Install Kaleido using: python -m pip install --upgrade "kaleido>=1.0.0"')

# Display coordinate statistics
if {"latitude", "longitude"}.issubset(df.columns):
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    valid = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ]

    print(f"Records with valid coordinates: {len(valid)}")
    print(f"Records without valid coordinates: {len(df) - len(valid)}")

print("Geographical map generation completed.")
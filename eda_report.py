from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent / "AirPulse_AI"

DATA_FILE = BASE_DIR / "data" / "processed" / "aircraft_history.csv"

if not DATA_FILE.exists():
    print(f"Dataset not found: {DATA_FILE}")
    raise SystemExit(1)

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 60)
print("AIRPULSE AI - EDA FINDINGS")
print("=" * 60)

# 1. Dataset dimensions
print("\n1. DATASET INFORMATION")
print(f"Total number of records: {len(df)}")
print(f"Total number of columns: {len(df.columns)}")

# 2. Collection period
print("\n2. COLLECTION PERIOD")

if "snapshot_time_utc" in df.columns:
    timestamps = pd.to_datetime(
        df["snapshot_time_utc"],
        errors="coerce",
        utc=True
    ).dropna()

    if not timestamps.empty:
        print(f"Start: {timestamps.min()}")
        print(f"End:   {timestamps.max()}")
        print(f"Snapshots: {timestamps.nunique()}")
    else:
        print("No valid snapshot timestamps found.")
else:
    print("Snapshot timestamp column not found.")

# 3. Missing values
print("\n3. MISSING VALUES")

missing_total = int(df.isna().sum().sum())
print(f"Total missing cells: {missing_total}")

missing_by_column = df.isna().sum()
print(missing_by_column[missing_by_column > 0])

# 4. Exact duplicate records
print("\n4. DUPLICATE RECORDS")
print(f"Exact duplicate rows: {df.duplicated().sum()}")

# 5. Distinct aircraft identifiers
print("\n5. AIRCRAFT IDENTIFIERS")

if "icao24" in df.columns:
    identifiers = df["icao24"].dropna().astype(str).str.strip()
    identifiers = identifiers[identifiers != ""]

    print(f"Distinct valid identifiers: {identifiers.nunique()}")
else:
    print("icao24 column not found.")

# 6. Average observed velocity
print("\n6. AVERAGE VELOCITY")

if "velocity" in df.columns:
    velocity = pd.to_numeric(df["velocity"], errors="coerce")

    print(f"Average velocity: {velocity.mean():.2f} m/s")
    print(f"Average velocity: {velocity.mean() * 3.6:.2f} km/h")
    print(f"Valid velocity observations: {velocity.notna().sum()}")
else:
    print("Velocity column not found.")

# 7. Average altitude
print("\n7. AVERAGE ALTITUDE")

if "baro_altitude" in df.columns:
    altitude = pd.to_numeric(
        df["baro_altitude"],
        errors="coerce"
    )

    print(f"Average barometric altitude: {altitude.mean():.2f} metres")
    print(f"Valid altitude observations: {altitude.notna().sum()}")
else:
    print("baro_altitude column not found.")

# 8. Distribution over time
print("\n8. DISTRIBUTION OVER TIME")

if "snapshot_time_utc" in df.columns:
    df["snapshot_time_utc"] = pd.to_datetime(
        df["snapshot_time_utc"],
        errors="coerce",
        utc=True
    )

    observations = (
        df.groupby("snapshot_time_utc")
        .size()
        .sort_index()
    )

    print(f"Number of snapshots: {len(observations)}")

    if not observations.empty:
        print(f"Minimum records in a snapshot: {observations.min()}")
        print(f"Maximum records in a snapshot: {observations.max()}")
        print(f"Average records per snapshot: {observations.mean():.2f}")

        observations.rename(
            "aircraft_records"
        ).to_csv(
            BASE_DIR / "data" / "processed" / "eda_time_distribution.csv"
        )
else:
    print("Snapshot timestamp column not found.")

# 9. Geographic distribution
print("\n9. GEOGRAPHICAL DISTRIBUTION")

if {"latitude", "longitude"}.issubset(df.columns):
    latitude = pd.to_numeric(df["latitude"], errors="coerce")
    longitude = pd.to_numeric(df["longitude"], errors="coerce")

    valid_coordinates = (
        latitude.between(-90, 90)
        & longitude.between(-180, 180)
    )

    valid_coordinates &= latitude.notna() & longitude.notna()

    print(f"Records with valid coordinates: {valid_coordinates.sum()}")
    print(f"Records without valid coordinates: {(~valid_coordinates).sum()}")

    if valid_coordinates.any():
        print(
            f"Latitude range: {latitude[valid_coordinates].min():.4f} "
            f"to {latitude[valid_coordinates].max():.4f}"
        )
        print(
            f"Longitude range: {longitude[valid_coordinates].min():.4f} "
            f"to {longitude[valid_coordinates].max():.4f}"
        )
else:
    print("Latitude or longitude column not found.")

# 10. Export summary
print("\n10. EXPORTING RESULTS")

summary = {
    "total_records": len(df),
    "total_columns": len(df.columns),
    "missing_cells": missing_total,
    "exact_duplicate_rows": int(df.duplicated().sum()),
}

summary_file = BASE_DIR / "data" / "processed" / "eda_summary.csv"

pd.DataFrame(
    [summary]
).to_csv(
    summary_file,
    index=False
)

print(f"Summary saved to: {summary_file}")
print("\nEDA analysis completed.")
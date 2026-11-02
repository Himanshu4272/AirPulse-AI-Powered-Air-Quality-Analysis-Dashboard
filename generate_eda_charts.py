from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# This script is saved in the outer PRE PROJECT folder.
BASE_DIR = Path(__file__).resolve().parent / "AirPulse_AI"
DATA_FILE = BASE_DIR / "data" / "processed" / "aircraft_history.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Check that aircraft_history.csv exists in AirPulse_AI/data/processed."
    )

df = pd.read_csv(DATA_FILE)

# Normalize numeric columns used in the plots.
for col in ["velocity", "baro_altitude", "latitude", "longitude"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 1. KPI cards
total_records = len(df)
unique_aircraft = (
    df["icao24"].dropna().astype(str).str.strip().replace("", pd.NA).nunique()
    if "icao24" in df.columns else 0
)

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis("off")
ax.text(0.25, 0.62, f"{total_records:,}", ha="center", va="center",
        fontsize=28, fontweight="bold")
ax.text(0.25, 0.30, "Total Aircraft Observations", ha="center", va="center",
        fontsize=12)
ax.text(0.75, 0.62, f"{unique_aircraft:,}", ha="center", va="center",
        fontsize=28, fontweight="bold")
ax.text(0.75, 0.30, "Distinct Aircraft Identifiers", ha="center", va="center",
        fontsize=12)
fig.tight_layout()
fig.savefig(REPORTS_DIR / "figure_1_kpi_cards.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 2. Missing values bar chart: show only columns with missing values.
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5.5))
if not missing.empty:
    missing.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of Missing Cells")
    ax.set_xlabel("Dataset Column")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Missing Values by Column")
else:
    ax.text(0.5, 0.5, "No missing values found", ha="center", va="center")
    ax.axis("off")
fig.tight_layout()
fig.savefig(REPORTS_DIR / "figure_2_missing_values.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 3. Descriptive statistics table for velocity and barometric altitude.
stats_cols = [c for c in ["velocity", "baro_altitude"] if c in df.columns]
if stats_cols:
    stats = df[stats_cols].describe().T
    stats = stats.rename(columns={
        "count": "Count", "mean": "Mean", "std": "Std. Dev.",
        "min": "Minimum", "25%": "25%", "50%": "Median",
        "75%": "75%", "max": "Maximum"
    })
    stats = stats.round(2)
    stats.to_csv(REPORTS_DIR / "table_3_descriptive_statistics.csv")

    fig, ax = plt.subplots(figsize=(10, 2.5 + 0.6 * len(stats)))
    ax.axis("off")
    table = ax.table(
        cellText=stats.reset_index(names="Variable").values,
        colLabels=["Variable"] + list(stats.columns),
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("Descriptive Statistics: Velocity and Barometric Altitude", pad=18)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figure_3_descriptive_statistics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

# 4. Number of observations per snapshot.
time_col = "snapshot_time_utc" if "snapshot_time_utc" in df.columns else None
if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    counts = df.groupby(time_col).size().sort_index()
    counts.rename("aircraft_records").to_csv(
        REPORTS_DIR / "table_4_observations_by_snapshot.csv"
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(1, len(counts) + 1), counts.values, marker="o")
    ax.set_xlabel("Snapshot Number (chronological)")
    ax.set_ylabel("Number of Aircraft Observations")
    ax.set_title("Aircraft Observations Across Snapshots")
    ax.set_xticks(range(1, len(counts) + 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figure_4_observations_over_time.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

print("\nAirPulse AI EDA charts created successfully.")
print(f"Dataset used: {DATA_FILE}")
print(f"Output folder: {REPORTS_DIR}")
for filename in [
    "figure_1_kpi_cards.png",
    "figure_2_missing_values.png",
    "figure_3_descriptive_statistics.png",
    "figure_4_observations_over_time.png",
    "table_3_descriptive_statistics.csv",
    "table_4_observations_by_snapshot.csv",
]:
    path = REPORTS_DIR / filename
    if path.exists():
        print(f"Created: {path.name}")

# 5. Distinct aircraft identifiers observed over time
if "snapshot_time_utc" in df.columns and "icao24" in df.columns:
    df["snapshot_time_utc"] = pd.to_datetime(
        df["snapshot_time_utc"], errors="coerce", utc=True
    )

    df["icao24"] = df["icao24"].astype("string").str.strip()
    valid_df = df.dropna(subset=["snapshot_time_utc", "icao24"])
    valid_df = valid_df[valid_df["icao24"] != ""]

    distinct_counts = (
        valid_df.groupby("snapshot_time_utc")["icao24"]
        .nunique()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(distinct_counts.index, distinct_counts.values, marker="o")
    ax.set_xlabel("Snapshot Time (UTC)")
    ax.set_ylabel("Distinct Aircraft Identifiers")
    ax.set_title("Distinct Aircraft Identifiers Observed Over Time")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR / "figure_5_distinct_aircraft_over_time.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    distinct_counts.rename("distinct_aircraft_count").to_csv(
        REPORTS_DIR / "distinct_aircraft_by_snapshot.csv"
    )

    print("Created: figure_5_distinct_aircraft_over_time.png")

# 6. Altitude Distribution Histogram
if "baro_altitude" in df.columns:
    altitude = pd.to_numeric(df["baro_altitude"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(altitude, bins=40, edgecolor="black")
    ax.set_title("Distribution of Aircraft Altitude")
    ax.set_xlabel("Barometric Altitude (metres)")
    ax.set_ylabel("Number of Observations")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / "figure_6_altitude_histogram.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    # 7. Altitude Box Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(altitude, vert=False)
    ax.set_title("Box Plot of Aircraft Altitude")
    ax.set_xlabel("Barometric Altitude (metres)")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / "figure_7_altitude_boxplot.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    # 8. Observations by Altitude Range
    bins = [
        -float("inf"), 1000, 3000, 6000,
        9000, 12000, 15000, float("inf")
    ]

    labels = [
        "Below 1,000 m",
        "1,000–3,000 m",
        "3,000–6,000 m",
        "6,000–9,000 m",
        "9,000–12,000 m",
        "12,000–15,000 m",
        "Above 15,000 m"
    ]

    altitude_ranges = pd.cut(
        altitude,
        bins=bins,
        labels=labels,
        right=False
    )

    range_counts = altitude_ranges.value_counts().reindex(
        labels, fill_value=0
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    range_counts.plot(kind="bar", ax=ax)
    ax.set_title("Aircraft Observations by Altitude Range")
    ax.set_xlabel("Altitude Range")
    ax.set_ylabel("Number of Observations")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / "figure_8_altitude_ranges.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    range_counts.rename("observation_count").to_csv(
        REPORTS_DIR / "table_altitude_ranges.csv"
    )

    print("Created: figure_6_altitude_histogram.png")
    print("Created: figure_7_altitude_boxplot.png")
    print("Created: figure_8_altitude_ranges.png")
    print("\nAltitude range counts:")
    print(range_counts.to_string())

# 9. Velocity histogram
if "velocity" in df.columns:
    velocity = pd.to_numeric(df["velocity"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(velocity, bins=40, edgecolor="black")
    ax.set_title("Distribution of Aircraft Velocity")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Number of Observations")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figure_9_velocity_histogram.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 10. Velocity box plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(velocity, vert=False)
    ax.set_title("Box Plot of Aircraft Velocity")
    ax.set_xlabel("Velocity (m/s)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figure_10_velocity_boxplot.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 11. Velocity versus altitude scatter plot
    if "baro_altitude" in df.columns:
        scatter_df = df[["velocity", "baro_altitude"]].copy()
        scatter_df["velocity"] = pd.to_numeric(
            scatter_df["velocity"], errors="coerce"
        )
        scatter_df["baro_altitude"] = pd.to_numeric(
            scatter_df["baro_altitude"], errors="coerce"
        )
        scatter_df = scatter_df.dropna()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(scatter_df["baro_altitude"],
                   scatter_df["velocity"], alpha=0.25)
        ax.set_title("Aircraft Velocity versus Altitude")
        ax.set_xlabel("Barometric Altitude (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "figure_11_velocity_altitude_scatter.png",
                    dpi=220, bbox_inches="tight")
        plt.close(fig)

    # 12. Average velocity over time
    if "snapshot_time_utc" in df.columns:
        time_df = df[["snapshot_time_utc", "velocity"]].copy()
        time_df["snapshot_time_utc"] = pd.to_datetime(
            time_df["snapshot_time_utc"], errors="coerce", utc=True
        )
        time_df["velocity"] = pd.to_numeric(
            time_df["velocity"], errors="coerce"
        )
        time_df = time_df.dropna()

        avg_velocity = (
            time_df.groupby("snapshot_time_utc")["velocity"]
            .mean()
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(avg_velocity.index, avg_velocity.values, marker="o")
        ax.set_title("Average Aircraft Velocity over Time")
        ax.set_xlabel("Snapshot Time (UTC)")
        ax.set_ylabel("Average Velocity (m/s)")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "figure_12_average_velocity_over_time.png",
                    dpi=220, bbox_inches="tight")
        plt.close(fig)

    print("Created velocity analysis visualizations.")
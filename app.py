from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

BASE = Path(__file__).resolve().parent

DATA_FILE = BASE / "data/demo/simulated_traffic_history.csv"
OBSERVATION_FILE = BASE / "data/raw/aircraft_observations.csv"
MODEL_DIR = BASE / "models"
TABLE_DIR = BASE / "reports/tables"

st.set_page_config(
    page_title="AirPulse AI | Aviation Intelligence",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL DARK THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #08111f 0%, #0c1729 55%, #101d32 100%);
    color: #edf4ff;
}

[data-testid="stHeader"] {
    background: rgba(8, 17, 31, 0.95);
}

[data-testid="stSidebar"] {
    background: #0b1526;
    border-right: 1px solid #22334d;
}

[data-testid="stSidebar"] * {
    color: #e5edf9;
}

.block-container {
    padding-top: 1.7rem;
    padding-bottom: 2.5rem;
    max-width: 1600px;
}

.hero {
    background: linear-gradient(120deg, #142642, #102038, #152c48);
    border: 1px solid #2b4668;
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 22px;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #f3f7ff;
}

.hero-subtitle {
    color: #a9bfdc;
    font-size: 14px;
    margin-top: 7px;
}

.section-title {
    color: #eaf2ff;
    font-size: 21px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 12px;
}

.status-live, .status-demo, .status-error {
    display: inline-block;
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.status-live {
    color: #72f0b1;
    background: #10392d;
    border: 1px solid #227c5a;
}

.status-demo {
    color: #ffd782;
    background: #3b2e15;
    border: 1px solid #816326;
}

.status-error {
    color: #ffaaaa;
    background: #401f27;
    border: 1px solid #8a3947;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #13243b, #101d31);
    border: 1px solid #29415f;
    padding: 18px 18px 15px 18px;
    border-radius: 16px;
    min-height: 125px;
}

div[data-testid="stMetricLabel"] {
    color: #a9bfdc;
}

div[data-testid="stMetricValue"] {
    color: #f5f8ff;
    font-weight: 800;
}

div[data-testid="stMetricDelta"] {
    color: #72f0b1;
}

div[data-testid="stPlotlyChart"] {
    border: 1px solid #223b58;
    border-radius: 16px;
    padding: 8px;
    background: #0e1b2e;
}

.stButton > button {
    background: linear-gradient(90deg, #2878e8, #35a5e8);
    color: white;
    border: 0;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #3988f4, #48b7f2);
    color: white;
    border: 0;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #29415f;
    border-radius: 12px;
}

hr {
    border-color: #263b55;
}

.small-note {
    color: #91a9c8;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_simulation():
    if not DATA_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE)

    if "timestamp" not in df.columns or "aircraft_count" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )
    df["aircraft_count"] = pd.to_numeric(
        df["aircraft_count"], errors="coerce"
    )

    return (
        df.dropna(subset=["timestamp", "aircraft_count"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def read_demo_observations():
    if not OBSERVATION_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(OBSERVATION_FILE)
        if "data_mode" in df.columns:
            df = df[
                df["data_mode"].astype(str).str.upper().eq("SIMULATED")
            ].copy()
        return df
    except Exception:
        return pd.DataFrame()


def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    return joblib.load(path)


def fetch_opensky():
    """
    Fetch current state vectors from OpenSky.
    Data is considered LIVE only if the request succeeds.
    """
    endpoint = "https://opensky-network.org/api/states/all"

    response = requests.get(
        endpoint,
        timeout=25,
        headers={
            "User-Agent": "AirPulseAI-Academic-Dashboard"
        }
    )
    response.raise_for_status()
    payload = response.json()

    states = payload.get("states")

    if states is None:
        states = []

    columns = [
        "icao24",
        "callsign",
        "origin_country",
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "baro_altitude",
        "on_ground",
        "velocity",
        "true_track",
        "vertical_rate",
        "sensors",
        "geo_altitude",
        "squawk",
        "spi",
        "position_source",
    ]

    # Normalize records because OpenSky state vectors can vary in length.
    normalized_states = []

    for state in states:
        if state is None:
            continue

        state = list(state)

        if len(state) < len(columns):
            state.extend([None] * (len(columns) - len(state)))
        elif len(state) > len(columns):
            state = state[:len(columns)]

        normalized_states.append(state)

    # Create the DataFrame once, with exactly 17 columns.
    df = pd.DataFrame(normalized_states, columns=columns)

    if not df.empty:
        numeric_columns = [
            "longitude",
            "latitude",
            "baro_altitude",
            "velocity",
            "geo_altitude",
            "true_track",
            "vertical_rate",
            "time_position",
            "last_contact",
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df["callsign"] = (
            df["callsign"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["icao24"] = (
            df["icao24"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df, utc_now(), payload.get("time")


def traffic_features(history):
    history = history.sort_values("timestamp")
    counts = history["aircraft_count"].astype(float)

    if len(counts) < 4:
        raise ValueError(
            "At least four five-minute traffic windows are needed."
        )

    last_timestamp = history["timestamp"].iloc[-1]

    return pd.DataFrame([{
        "previous_count": counts.iloc[-1],
        "count_two_windows_ago": counts.iloc[-2],
        "count_three_windows_ago": counts.iloc[-3],
        "hour": last_timestamp.hour,
        "minute": last_timestamp.minute,
        "day_of_week": last_timestamp.dayofweek
    }])


def make_chart_layout(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0e1b2e",
        plot_bgcolor="#0e1b2e",
        font=dict(color="#dbe8fa", family="Inter"),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dbe8fa")
        ),
        xaxis=dict(
            gridcolor="#233852",
            zerolinecolor="#233852"
        ),
        yaxis=dict(
            gridcolor="#233852",
            zerolinecolor="#233852"
        )
    )
    return fig


# =========================================================
# SESSION STATE
# =========================================================

if "live_data" not in st.session_state:
    st.session_state.live_data = None

if "live_fetched_at" not in st.session_state:
    st.session_state.live_fetched_at = None

if "live_api_time" not in st.session_state:
    st.session_state.live_api_time = None

if "live_error" not in st.session_state:
    st.session_state.live_error = None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ?? AirPulse AI")
st.sidebar.caption("AVIATION INTELLIGENCE PLATFORM")
st.sidebar.markdown("---")

source_mode = st.sidebar.radio(
    "DATA SOURCE",
    ["Demo Simulation", "Live OpenSky API"],
    help="Demo uses generated data. Live mode requests current data from OpenSky."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Machine Learning")

model_name = st.sidebar.selectbox(
    "Prediction model",
    [
        "Linear Regression",
        "Gaussian Naive Bayes",
        "Decision Tree Regressor",
        "Decision Tree Classifier"
    ]
)

st.sidebar.markdown("---")

if source_mode == "Live OpenSky API":
    st.sidebar.warning(
        "Live mode makes a real API request. Access, rate limits, "
        "and available aircraft positions depend on OpenSky."
    )

    if st.sidebar.button("? Fetch Live Aircraft"):
        try:
            with st.spinner("Requesting current aircraft states from OpenSky..."):
                live_df, fetched_at, api_time = fetch_opensky()

            st.session_state.live_data = live_df
            st.session_state.live_fetched_at = fetched_at
            st.session_state.live_api_time = api_time
            st.session_state.live_error = None

        except Exception as exc:
            st.session_state.live_error = (
                f"{type(exc).__name__}: {exc}"
            )

    if st.sidebar.button("Clear Live Snapshot"):
        st.session_state.live_data = None
        st.session_state.live_fetched_at = None
        st.session_state.live_api_time = None
        st.session_state.live_error = None

else:
    st.sidebar.info(
        "Demo mode uses synthetic data. No real aircraft observations "
        "are implied."
    )


# =========================================================
# DATA SOURCE AND STATUS
# =========================================================

history = read_simulation()
demo_observations = read_demo_observations()

live_ready = (
    source_mode == "Live OpenSky API"
    and st.session_state.live_data is not None
    and st.session_state.live_error is None
)

if source_mode == "Live OpenSky API":
    if st.session_state.live_error:
        status_html = '<span class="status-error">? API ERROR</span>'
        status_message = st.session_state.live_error
        aircraft = pd.DataFrame()
        fetched_at = None

    elif live_ready:
        status_html = '<span class="status-live">? LIVE API SNAPSHOT</span>'
        status_message = "Successfully received a response from OpenSky."
        aircraft = st.session_state.live_data.copy()
        fetched_at = st.session_state.live_fetched_at

    else:
        status_html = '<span class="status-error">? WAITING FOR FETCH</span>'
        status_message = (
            "Click 'Fetch Live Aircraft' in the sidebar. "
            "Until a request succeeds, no live observation is available."
        )
        aircraft = pd.DataFrame()
        fetched_at = None

else:
    status_html = '<span class="status-demo">? DEMO SIMULATION</span>'
    status_message = (
        "Synthetic data is being displayed. It is not live aircraft data."
    )
    aircraft = demo_observations.copy()
    fetched_at = None


# =========================================================
# HERO
# =========================================================

st.markdown(f"""
<div class="hero">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap;">
        <div>
            <div class="hero-title">?? AirPulse AI</div>
            <div class="hero-subtitle">
                Air Traffic Monitoring ? Predictive Analytics ? Machine Learning
            </div>
        </div>
        <div>{status_html}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(status_message)

if source_mode == "Demo Simulation":
    st.warning(
        "DEMO MODE: all displayed traffic-history values are synthetic. "
        "They must not be presented as real aircraft observations."
    )
elif live_ready:
    st.success(
        f"Last successful API fetch: {fetched_at}. "
        "This is a snapshot from the API response, not a continuous feed."
    )
elif st.session_state.live_error:
    st.error(
        "The live API request failed. Live aircraft data is not being "
        "displayed. Switch to Demo Simulation if you want to explore "
        "the dashboard while troubleshooting the connection."
    )


# =========================================================
# KPI ROW
# =========================================================

st.markdown('<div class="section-title">Operational Overview</div>',
            unsafe_allow_html=True)

if source_mode == "Live OpenSky API" and live_ready:
    live_valid_positions = aircraft.dropna(
        subset=["longitude", "latitude"]
    )

    unique_aircraft = aircraft["icao24"].nunique()
    average_speed = aircraft["velocity"].mean() * 3.6
    average_altitude = aircraft["baro_altitude"].mean()
    observation_count = len(aircraft)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("State Records", f"{observation_count:,}")
    k2.metric("Unique Aircraft", f"{unique_aircraft:,}")
    k3.metric(
        "Average Speed",
        f"{average_speed:.1f} km/h" if pd.notna(average_speed) else "N/A"
    )
    k4.metric(
        "Average Altitude",
        f"{average_altitude:,.0f} m" if pd.notna(average_altitude) else "N/A"
    )

elif source_mode == "Demo Simulation" and not aircraft.empty:
    unique_col = "icao24" if "icao24" in aircraft.columns else None
    speed_col = "velocity" if "velocity" in aircraft.columns else None
    altitude_col = "baro_altitude" if "baro_altitude" in aircraft.columns else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Demo Observations", f"{len(aircraft):,}")
    k2.metric(
        "Unique Demo IDs",
        f"{aircraft[unique_col].nunique():,}" if unique_col else "N/A"
    )

    speed_avg = (
        pd.to_numeric(aircraft[speed_col], errors="coerce").mean() * 3.6
        if speed_col else float("nan")
    )
    altitude_avg = (
        pd.to_numeric(aircraft[altitude_col], errors="coerce").mean()
        if altitude_col else float("nan")
    )

    k3.metric(
        "Average Demo Speed",
        f"{speed_avg:.1f} km/h" if pd.notna(speed_avg) else "N/A"
    )
    k4.metric(
        "Average Demo Altitude",
        f"{altitude_avg:,.0f} m" if pd.notna(altitude_avg) else "N/A"
    )

else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("State Records", "?")
    k2.metric("Unique Aircraft", "?")
    k3.metric("Average Speed", "?")
    k4.metric("Average Altitude", "?")


# =========================================================
# MAP
# =========================================================

st.markdown('<div class="section-title">Aircraft Position Map</div>',
            unsafe_allow_html=True)

if source_mode == "Live OpenSky API" and live_ready:
    map_data = aircraft.dropna(subset=["longitude", "latitude"]).copy()
    map_data = map_data[
        map_data["longitude"].between(-180, 180)
        & map_data["latitude"].between(-90, 90)
    ]

    if not map_data.empty:
        map_data["label"] = map_data["callsign"].replace("", "Unknown")
        fig_map = px.scatter_geo(
            map_data,
            lat="latitude",
            lon="longitude",
            hover_name="label",
            hover_data={
                "icao24": True,
                "origin_country": True,
                "velocity": True,
                "baro_altitude": True,
                "latitude": ":.3f",
                "longitude": ":.3f"
            },
            projection="natural earth",
            title="Positions returned by the latest OpenSky response"
        )
        fig_map.update_traces(
            marker=dict(size=5, color="#43b7ff", opacity=0.8)
        )
        fig_map.update_geos(
            bgcolor="#0e1b2e",
            landcolor="#172b43",
            oceancolor="#091524",
            showocean=True,
            showcountries=True,
            countrycolor="#36516f",
            showcoastlines=True,
            coastlinecolor="#36516f"
        )
        fig_map.update_layout(
            paper_bgcolor="#0e1b2e",
            font=dict(color="#dbe8fa"),
            height=480,
            margin=dict(l=5, r=5, t=50, b=5)
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            f"Map uses {len(map_data):,} records with valid coordinates "
            "from the last successful API response. Some aircraft may "
            "not have position data."
        )
    else:
        st.info(
            "The API response contains no records with usable coordinates. "
            "This does not mean there are no aircraft."
        )

elif source_mode == "Demo Simulation" and not aircraft.empty:
    if {"latitude", "longitude"}.issubset(aircraft.columns):
        map_data = aircraft.dropna(subset=["latitude", "longitude"]).copy()

        if not map_data.empty:
            fig_map = px.scatter_geo(
                map_data,
                lat="latitude",
                lon="longitude",
                hover_data=[
                    c for c in ["icao24", "callsign", "data_mode"]
                    if c in map_data.columns
                ],
                projection="natural earth",
                title="Synthetic demonstration coordinates"
            )
            fig_map.update_traces(
                marker=dict(size=6, color="#ffbf69", opacity=0.75)
            )
            fig_map.update_geos(
                bgcolor="#0e1b2e",
                landcolor="#172b43",
                oceancolor="#091524",
                showocean=True,
                showcountries=True,
                countrycolor="#36516f"
            )
            fig_map.update_layout(
                paper_bgcolor="#0e1b2e",
                font=dict(color="#dbe8fa"),
                height=450
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("DEMO MAP ? coordinates are simulated, not real aircraft positions.")
        else:
            st.info("No usable demo coordinates are available.")
    else:
        st.info("The demo observation file does not contain coordinate columns.")

else:
    st.info(
        "No aircraft map is available yet. Select Demo Simulation or "
        "fetch a successful live OpenSky response."
    )


# =========================================================
# TRAFFIC HISTORY
# =========================================================

st.markdown('<div class="section-title">Traffic History & Trends</div>',
            unsafe_allow_html=True)

if not history.empty:
    fig_history = px.area(
        history,
        x="timestamp",
        y="aircraft_count",
        title="Aircraft Count by Five-Minute Window"
    )
    fig_history.update_traces(
        line=dict(color="#41b7ff", width=2),
        fillcolor="rgba(65,183,255,0.18)"
    )
    st.plotly_chart(make_chart_layout(fig_history, 390),
                    use_container_width=True)

    a, b = st.columns(2)

    with a:
        fig_histogram = px.histogram(
            history,
            x="aircraft_count",
            nbins=25,
            title="Distribution of Traffic Counts"
        )
        fig_histogram.update_traces(marker_color="#8a7dff")
        st.plotly_chart(make_chart_layout(fig_histogram),
                        use_container_width=True)

    with b:
        rolling = history.copy()
        rolling["rolling_average"] = (
            rolling["aircraft_count"].rolling(12, min_periods=1).mean()
        )
        fig_rolling = go.Figure()
        fig_rolling.add_trace(go.Scatter(
            x=rolling["timestamp"],
            y=rolling["aircraft_count"],
            mode="lines",
            name="Observed history",
            line=dict(color="#47617f", width=1)
        ))
        fig_rolling.add_trace(go.Scatter(
            x=rolling["timestamp"],
            y=rolling["rolling_average"],
            mode="lines",
            name="Rolling average",
            line=dict(color="#4ee0ad", width=3)
        ))
        fig_rolling.update_layout(title="Traffic Trend")
        st.plotly_chart(make_chart_layout(fig_rolling),
                        use_container_width=True)
else:
    st.info("Traffic-history dataset is unavailable.")


# =========================================================
# MACHINE LEARNING PREDICTIONS
# =========================================================

st.markdown('<div class="section-title">Machine Learning Predictions</div>',
            unsafe_allow_html=True)

if history.empty:
    st.warning("Prediction history is unavailable.")
elif st.button("Generate Five-Minute-Ahead Prediction", type="primary"):
    try:
        features = traffic_features(history)

        filenames = {
            "Linear Regression": "linear_regression.joblib",
            "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
            "Decision Tree Regressor": "decision_tree_regressor.joblib",
            "Decision Tree Classifier": "decision_tree_classifier.joblib"
        }

        model = load_model(filenames[model_name])

        if model is None:
            st.error(
                f"Model file not found: {filenames[model_name]}. "
                "Train or save this model first."
            )
        else:
            prediction = model.predict(features)[0]

            if model_name in [
                "Linear Regression",
                "Decision Tree Regressor"
            ]:
                st.metric(
                    "Predicted Aircraft Count",
                    f"{max(0, float(prediction)):.1f}"
                )
                st.caption(
                    "Numerical forecast from the trained regression model."
                )
            else:
                st.metric("Predicted Traffic Category", str(prediction))
                st.caption(
                    "Classification output from the trained classifier."
                )

            st.info(
                "The model is trained on the project's historical traffic "
                "series. Unless that series has been populated with real "
                "collected observations and the models retrained, this is "
                "a demonstration forecast, not a live-data forecast."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}")


# =========================================================
# MODEL METRICS
# =========================================================

st.markdown('<div class="section-title">Model Performance</div>',
            unsafe_allow_html=True)

regression_file = TABLE_DIR / "regression_metrics.csv"
classification_file = TABLE_DIR / "classification_metrics.csv"

left, right = st.columns(2)

with left:
    st.markdown("#### Regression")
    if regression_file.exists():
        st.dataframe(
            pd.read_csv(regression_file),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Regression metrics have not been exported yet.")

with right:
    st.markdown("#### Classification")
    if classification_file.exists():
        st.dataframe(
            pd.read_csv(classification_file),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Classification metrics have not been exported yet.")


# =========================================================
# RAW OBSERVATION TABLE
# =========================================================

st.markdown('<div class="section-title">Aircraft Observation Records</div>',
            unsafe_allow_html=True)

if source_mode == "Live OpenSky API" and live_ready:
    display_df = aircraft.copy()
    st.caption(
        "LIVE SNAPSHOT ? these rows are from the latest successful "
        "OpenSky API response. They are not a continuous video feed."
    )
elif source_mode == "Demo Simulation" and not aircraft.empty:
    display_df = aircraft.copy()
    st.caption("DEMO SNAPSHOT ? generated demonstration records.")
else:
    display_df = pd.DataFrame()

if not display_df.empty:
    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Showing up to 100 of {len(display_df):,} records.")
else:
    st.info("No observation records to display for the current source.")

st.markdown("---")
st.markdown(
    '<div class="small-note">AirPulse AI ? Academic project ? '
    'OpenSky observations are subject to source availability, API limits, '
    'and applicable terms of use.</div>',
    unsafe_allow_html=True
)

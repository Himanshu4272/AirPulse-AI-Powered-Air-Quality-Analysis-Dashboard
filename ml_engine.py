from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

FEATURE_COLUMNS = [
    "previous_count",
    "count_two_windows_ago",
    "count_three_windows_ago",
    "hour",
    "minute",
    "day_of_week",
]

MODEL_FILES = {
    "Linear Regression": "linear_regression.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Decision Tree Regressor": "decision_tree_regressor.joblib",
    "Decision Tree Classifier": "decision_tree_classifier.joblib",
}


def prepare_latest_features(df):
    """
    Prepare the input features for the next traffic-window prediction.

    The input DataFrame must contain:
        timestamp
        aircraft_count
    """

    required = {"timestamp", "aircraft_count"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], utc=True, errors="coerce"
    )

    data["aircraft_count"] = pd.to_numeric(
        data["aircraft_count"], errors="coerce"
    )

    data = (
        data.dropna(subset=["timestamp", "aircraft_count"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(data) < 4:
        raise ValueError(
            "At least four valid traffic windows are required."
        )

    counts = data["aircraft_count"].astype(float)
    latest_timestamp = data["timestamp"].iloc[-1]

    features = pd.DataFrame(
        [{
            "previous_count": counts.iloc[-1],
            "count_two_windows_ago": counts.iloc[-2],
            "count_three_windows_ago": counts.iloc[-3],
            "hour": latest_timestamp.hour,
            "minute": latest_timestamp.minute,
            "day_of_week": latest_timestamp.dayofweek,
        }],
        columns=FEATURE_COLUMNS,
    )

    return features, latest_timestamp


def load_model(model_name):
    """Load a previously trained model from the models directory."""

    if model_name not in MODEL_FILES:
        raise ValueError(f"Unsupported model: {model_name}")

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_traffic(df, model_name):
    """
    Generate a prediction using the selected trained model.

    Regression models return a numerical aircraft-count prediction.
    Classification models return a traffic category.
    """

    features, latest_timestamp = prepare_latest_features(df)
    model = load_model(model_name)

    prediction = model.predict(features)[0]

    if model_name in (
        "Linear Regression",
        "Decision Tree Regressor",
    ):
        prediction = max(0, float(prediction))

        return {
            "model": model_name,
            "prediction_type": "count",
            "prediction": round(prediction, 2),
            "based_on_timestamp": latest_timestamp.isoformat(),
            "forecast_timestamp": (
                latest_timestamp + pd.Timedelta(minutes=5)
            ).isoformat(),
        }

    return {
        "model": model_name,
        "prediction_type": "category",
        "prediction": str(prediction),
        "based_on_timestamp": latest_timestamp.isoformat(),
        "forecast_timestamp": (
            latest_timestamp + pd.Timedelta(minutes=5)
        ).isoformat(),
    }


def load_evaluation_table(filename):
    """Load an evaluation CSV if it exists."""

    path = BASE / "reports" / "tables" / filename

    if not path.exists():
        return None

    return pd.read_csv(path)


def get_model_names():
    """Return the models available in the dashboard."""

    return list(MODEL_FILES.keys())

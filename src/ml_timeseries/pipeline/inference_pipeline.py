# pipeline/inference_pipeline.py

import joblib
from pathlib import Path

from data.load_data import load_data
from data.preprocess import preprocess_data


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"


def inference_pipeline(forecast_steps=12):
    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------
    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # Load latest data
    # --------------------------------------------------------
    df = load_data()
    y = preprocess_data(df)

    # --------------------------------------------------------
    # Forecast future
    # --------------------------------------------------------
    forecast = model.predict(steps=forecast_steps)

    return forecast

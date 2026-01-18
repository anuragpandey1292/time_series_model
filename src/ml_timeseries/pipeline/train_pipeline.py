# pipeline/train_pipeline.py

import joblib
from pathlib import Path

from data.load_data import load_data
from data.preprocess import preprocess_data
from experiments.model_comparison import (
    compare_models,
    get_best_model,
    refit_best_model
)

from models.ARIMA_model import ARIMAModel, SARIMAModel
from models.statstics_models import HoltWintersModel, ThetaForecastModel


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def train_pipeline():
    # --------------------------------------------------------
    # Load & preprocess data
    # --------------------------------------------------------
    df = load_data()
    y = preprocess_data(df)

    # Train / test split
    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]

    # --------------------------------------------------------
    # Define candidate models
    # --------------------------------------------------------
    models = {
        "ARIMA": ARIMAModel(),
        "SARIMA": SARIMAModel(m=12),
        "HoltWinters": HoltWintersModel(season_length=12),
        "Theta": ThetaForecastModel()
    }

    # --------------------------------------------------------
    # Compare models
    # --------------------------------------------------------
    df_results = compare_models(
        models=models,
        y_train=y_train,
        y_test=y_test,
        steps=len(y_test)
    )

    print(df_results)

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------
    best_model_name, best_model = get_best_model(df_results, models)
    print(f"Best model selected: {best_model_name}")

    # --------------------------------------------------------
    # Refit on full data
    # --------------------------------------------------------
    best_model = refit_best_model(
        best_model=best_model,
        y_full=y
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------
    model_path = ARTIFACTS_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"Best model saved to {model_path}")

    return best_model_name, df_results

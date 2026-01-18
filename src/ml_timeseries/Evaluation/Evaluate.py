# Evaluation/Evaluate.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# METRICS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ============================================================
# EVALUATE SINGLE MODEL
# ============================================================

def evaluate_model(
    model,
    y_train,
    y_test,
    steps,
    exog_train=None,
    exog_test=None,
    model_name=None
):
    """
    Generic evaluator for ALL models
    """

    # Fit model
    if exog_train is not None:
        model.fit(y_train, exog_train)
        y_pred = model.predict(steps=steps, exog_future=exog_test)
    else:
        model.fit(y_train)
        y_pred = model.predict(steps=steps)

    y_pred = np.array(y_pred)
    y_true = np.array(y_test[:steps])

    results = {
        "Model": model_name or model.__class__.__name__,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred)
    }

    # Optional: store model params if available
    if hasattr(model, "best_params"):
        results["Params"] = model.best_params
    else:
        results["Params"] = None

    return results

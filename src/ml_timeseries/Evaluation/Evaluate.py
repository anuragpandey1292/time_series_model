"""
Industry-grade evaluation for time series forecasting
Model-agnostic, leakage-safe, and production-ready
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Configuration
# -------------------------------------------------
@dataclass
class EvaluationConfig:
    metrics: List[str] = ("mae", "rmse", "mape", "smape")
    horizon: int = 1


# -------------------------------------------------
# Metric functions
# -------------------------------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


METRIC_REGISTRY = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "smape": smape,
}


# -------------------------------------------------
# Core evaluation
# -------------------------------------------------
def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    config: EvaluationConfig
) -> Dict[str, float]:

    logger.info("Starting evaluation")

    # Align indices
    y_true, y_pred = y_true.align(y_pred, join="inner")

    metrics = {}
    for name in config.metrics:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unsupported metric: {name}")
        metrics[name] = METRIC_REGISTRY[name](y_true.values, y_pred.values)

    logger.info("Evaluation completed | %s", metrics)
    return metrics


# -------------------------------------------------
# Rolling backtest evaluation
# -------------------------------------------------
def backtest_evaluation(
    y_true: pd.Series,
    y_preds: List[pd.Series],
    config: EvaluationConfig
) -> pd.DataFrame:
    """
    Evaluate multiple rolling forecasts
    """

    results = []

    for i, y_pred in enumerate(y_preds):
        metrics = evaluate_forecast(y_true, y_pred, config)
        metrics["fold"] = i
        res

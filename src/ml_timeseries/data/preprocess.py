"""
Industry-grade preprocessing for time series forecasting
"""

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------
# Configuration
# ---------------------------
@dataclass
class PreprocessConfig:
    freq: str = "D"                 # Expected frequency (D, W, M)
    test_size: float = 0.2
    fill_method: str = "ffill"      # ffill | bfill | interpolate
    outlier_cap_quantile: float = 0.99
    remove_duplicates: bool = True


# ---------------------------
# Validation
# ---------------------------
def validate_series(df: pd.DataFrame, date_col: str, target_col: str):
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    if df[date_col].isna().any():
        raise ValueError("Date column contains null values")

    logger.info("Schema validation passed")


# ---------------------------
# Time index handling
# ---------------------------
def prepare_time_index(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    freq: str,
    remove_duplicates: bool
) -> pd.Series:

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    if remove_duplicates:
        df = df.drop_duplicates(subset=[date_col])

    df.set_index(date_col, inplace=True)

    # Enforce frequency
    series = df[target_col].asfreq(freq)

    logger.info(
        "Time index prepared | Start: %s | End: %s | Freq: %s",
        series.index.min(),
        series.index.max(),
        freq
    )

    return series


# ---------------------------
# Missing value handling
# ---------------------------
def handle_missing(series: pd.Series, method: str) -> pd.Series:
    missing_pct = series.isna().mean()

    if missing_pct > 0:
        logger.warning("Missing values detected: %.2f%%", missing_pct * 100)

    if method == "ffill":
        series = series.ffill()
    elif method == "bfill":
        series = series.bfill()
    elif method == "interpolate":
        series = series.interpolate()
    else:
        raise ValueError(f"Unsupported fill method: {method}")

    return series


# ---------------------------
# Outlier handling
# ---------------------------
def cap_outliers(series: pd.Series, upper_quantile: float) -> pd.Series:
    upper_cap = series.quantile(upper_quantile)
    lower_cap = series.quantile(1 - upper_quantile)

    outliers = ((series > upper_cap) | (series < lower_cap)).sum()

    if outliers > 0:
        logger.info("Capping %d outliers", outliers)

    series = series.clip(lower=lower_cap, upper=upper_cap)
    return series


# ---------------------------
# Train-test split (time-aware)
# ---------------------------
def time_series_split(
    series: pd.Series, test_size: float
) -> Tuple[pd.Series, pd.Series]:

    split_idx = int(len(series) * (1 - test_size))

    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    logger.info(
        "Train-test split | Train: %d | Test: %d",
        len(train),
        len(test)
    )

    return train, test


# ---------------------------
# Full preprocessing pipeline
# ---------------------------
def preprocess_time_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    config: PreprocessConfig
) -> Tuple[pd.Series, pd.Series]:

    validate_series(df, date_col, target_col)

    series = prepare_time_index(
        df,
        date_col,
        target_col,
        config.freq,
        config.remove_duplicates
    )

    series = handle_missing(series, config.fill_method)
    series = cap_outliers(series, config.outlier_cap_quantile)

    train, test = time_series_split(series, config.test_size)

    return train, test

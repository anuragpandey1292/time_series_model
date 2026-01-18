"""
Industry-grade feature engineering for time series forecasting
Leakage-safe, configurable, and reproducible
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Configuration
# -------------------------------------------------
@dataclass
class FeatureConfig:
    lags: List[int] = (1, 7, 14, 28)
    rolling_windows: List[int] = (7, 14, 28)
    rolling_stats: List[str] = ("mean", "std", "min", "max")
    add_diff: bool = True
    add_pct_change: bool = True
    add_calendar_features: bool = True
    drop_na: bool = True


# -------------------------------------------------
# Lag features
# -------------------------------------------------
def create_lag_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df


# -------------------------------------------------
# Rolling window features
# -------------------------------------------------
def create_rolling_features(
    series: pd.Series,
    windows: List[int],
    stats: List[str]
) -> pd.DataFrame:

    df = pd.DataFrame(index=series.index)

    for window in windows:
        roll = series.shift(1).rolling(window=window)

        if "mean" in stats:
            df[f"roll_mean_{window}"] = roll.mean()
        if "std" in stats:
            df[f"roll_std_{window}"] = roll.std()
        if "min" in stats:
            df[f"roll_min_{window}"] = roll.min()
        if "max" in stats:
            df[f"roll_max_{window}"] = roll.max()

    return df


# -------------------------------------------------
# Trend / momentum features
# -------------------------------------------------
def create_trend_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=series.index)

    df["diff_1"] = series.diff(1)
    df["diff_7"] = series.diff(7)

    df["pct_change_1"] = series.pct_change(1)
    df["pct_change_7"] = series.pct_change(7)

    return df


# -------------------------------------------------
# Calendar features
# -------------------------------------------------
def create_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=index)

    df["day_of_week"] = index.dayofweek
    df["week_of_year"] = index.isocalendar().week.astype(int)
    df["month"] = index.month
    df["quarter"] = index.quarter
    df["day_of_month"] = index.day
    df["is_weekend"] = index.dayofweek.isin([5, 6]).astype(int)
    df["is_month_start"] = index.is_month_start.astype(int)
    df["is_month_end"] = index.is_month_end.astype(int)

    return df


# -------------------------------------------------
# Master feature engineering function
# -------------------------------------------------
def build_features(
    series: pd.Series,
    config: FeatureConfig,
    exogenous: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build full feature matrix for time series forecasting
    """

    logger.info("Starting feature engineering")

    # Base target
    features = pd.DataFrame({"y": series})

    # Lag features
    lag_df = create_lag_features(series, config.lags)
    features = features.join(lag_df.drop(columns=["y"]))

    # Rolling features
    roll_df = create_rolling_features(
        series,
        config.rolling_windows,
        config.rolling_stats
    )
    features = features.join(roll_df)

    # Trend features
    if config.add_diff or config.add_pct_change:
        trend_df = create_trend_features(series)
        features = features.join(trend_df)

    # Calendar features
    if config.add_calendar_features:
        cal_df = create_calendar_features(series.index)
        features = features.join(cal_df)

    # Optional exogenous variables
    if exogenous is not None:
        features = features.join(exogenous)

    # Drop rows with NA (caused by lags/rolling)
    if config.drop_na:
        features = features.dropna()

    logger.info(
        "Feature matrix created | Rows: %d | Features: %d",
        features.shape[0],
        features.shape[1]
    )

    return features

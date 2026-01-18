"""
Facebook Prophet model wrapper
"""

from dataclasses import dataclass
import pandas as pd
from prophet import Prophet


@dataclass
class ProphetConfig:
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    seasonality_mode: str = "additive"


class ProphetModel:
    def __init__(self, config: ProphetConfig):
        self.config = config
        self.model = Prophet(
            yearly_seasonality=config.yearly_seasonality,
            weekly_seasonality=config.weekly_seasonality,
            daily_seasonality=config.daily_seasonality,
            seasonality_mode=config.seasonality_mode,
        )

    def fit(self, df: pd.DataFrame):
        """
        df must contain columns: ds, y
        """
        self.model.fit(df)

    def predict(self, future: pd.DataFrame) -> pd.Series:
        forecast = self.model.predict(future)
        return forecast["yhat"]

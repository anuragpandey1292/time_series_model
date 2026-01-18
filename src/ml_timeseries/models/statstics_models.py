# models/statistical_models.py

import numpy as np
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing
)
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.api import VAR


# ============================================================
# BASE MODEL
# ============================================================

class BaseTimeSeriesModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None

    def fit(self, y):
        raise NotImplementedError

    def predict(self, steps):
        raise NotImplementedError


# ============================================================
# NAIVE MODEL
# ============================================================

class NaiveModel(BaseTimeSeriesModel):
    def fit(self, y):
        self.last_value = y.iloc[-1]
        return self

    def predict(self, steps):
        return np.repeat(self.last_value, steps)


# ============================================================
# SEASONAL NAIVE
# ============================================================

class SeasonalNaiveModel(BaseTimeSeriesModel):
    def __init__(self, season_length):
        super().__init__()
        self.season_length = season_length

    def fit(self, y):
        self.last_season = y.iloc[-self.season_length:]
        return self

    def predict(self, steps):
        reps = int(np.ceil(steps / self.season_length))
        return np.tile(self.last_season, reps)[:steps]


# ============================================================
# SIMPLE EXPONENTIAL SMOOTHING (SES)
# ============================================================

class SESModel(BaseTimeSeriesModel):
    def fit(self, y):
        self.model = SimpleExpSmoothing(y)
        self.fitted_model = self.model.fit()
        return self

    def predict(self, steps):
        return self.fitted_model.forecast(steps)


# ============================================================
# HOLT (DOUBLE EXPONENTIAL SMOOTHING)
# ============================================================

class HoltModel(BaseTimeSeriesModel):
    def fit(self, y):
        self.model = Holt(y)
        self.fitted_model = self.model.fit()
        return self

    def predict(self, steps):
        return self.fitted_model.forecast(steps)


# ============================================================
# HOLT-WINTERS (TRIPLE ES)
# ============================================================

class HoltWintersModel(BaseTimeSeriesModel):
    def __init__(self, season_length, seasonal="additive", trend="additive"):
        super().__init__()
        self.season_length = season_length
        self.seasonal = seasonal
        self.trend = trend

    def fit(self, y):
        self.model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.season_length
        )
        self.fitted_model = self.model.fit()
        return self

    def predict(self, steps):
        return self.fitted_model.forecast(steps)


# ============================================================
# ETS (AUTOMATIC ERROR-TREND-SEASONAL)
# ============================================================

class ETSModel(BaseTimeSeriesModel):
    def __init__(self, season_length):
        super().__init__()
        self.season_length = season_length

    def fit(self, y):
        self.model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=self.season_length
        )
        self.fitted_model = self.model.fit(optimized=True)
        return self

    def predict(self, steps):
        return self.fitted_model.forecast(steps)


# ============================================================
# THETA MODEL
# ============================================================

class ThetaForecastModel(BaseTimeSeriesModel):
    def fit(self, y):
        self.model = ThetaModel(y)
        self.fitted_model = self.model.fit()
        return self

    def predict(self, steps):
        return self.fitted_model.forecast(steps)


# ============================================================
# VECTOR AUTOREGRESSION (VAR) – MULTIVARIATE
# ============================================================

class VARModel(BaseTimeSeriesModel):
    def __init__(self, maxlags=5):
        super().__init__()
        self.maxlags = maxlags

    def fit(self, y_multivariate):
        self.model = VAR(y_multivariate)
        self.fitted_model = self.model.fit(maxlags=self.maxlags, ic="aic")
        self.k_ar = self.fitted_model.k_ar
        return self

    def predict(self, steps):
        return self.fitted_model.forecast(
            self.fitted_model.endog[-self.k_ar:], steps
        )

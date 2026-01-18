# models/ARIMA_model.py

import itertools
from experiments.arima_grid_search import arima_grid_search


# ============================================================
# BASE CLASS
# ============================================================

class BaseARIMAModel:
    def __init__(self):
        self.model = None
        self.best_params = None

    def fit(self, y, exog=None):
        results = arima_grid_search(
            y=y,
            exog=exog,
            order_grid=self.order_grid,
            seasonal_order_grid=self.seasonal_order_grid
        )

        self.model = results["best_model"]
        self.best_params = {
            "order": results["best_order"],
            "seasonal_order": results["best_seasonal_order"],
            "score": results["best_score"]
        }
        return self

    def predict(self, steps, exog_future=None):
        return self.model.forecast(steps=steps, exog=exog_future)

    def summary(self):
        return self.model.summary()


# ============================================================
# AR(p) : d=0, q=0
# ============================================================

class ARModel(BaseARIMAModel):
    def __init__(self, p_range=(1, 5)):
        super().__init__()
        self.order_grid = [(p, 0, 0) for p in p_range]
        self.seasonal_order_grid = None


# ============================================================
# MA(q) : p=0, d=0
# ============================================================

class MAModel(BaseARIMAModel):
    def __init__(self, q_range=(1, 5)):
        super().__init__()
        self.order_grid = [(0, 0, q) for q in q_range]
        self.seasonal_order_grid = None


# ============================================================
# ARMA(p, q) : d=0
# ============================================================

class ARMAModel(BaseARIMAModel):
    def __init__(self, p_range=(1, 5), q_range=(1, 5)):
        super().__init__()
        self.order_grid = [(p, 0, q) for p in p_range for q in q_range]
        self.seasonal_order_grid = None


# ============================================================
# ARIMA(p, d, q) : seasonal=False
# ============================================================

class ARIMAModel(BaseARIMAModel):
    def __init__(self, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        super().__init__()
        self.order_grid = list(itertools.product(p_range, d_range, q_range))
        self.seasonal_order_grid = None


# ============================================================
# SARIMA : seasonal=True, exog=None
# ============================================================

class SARIMAModel(BaseARIMAModel):
    def __init__(
        self,
        p_range=(0, 2),
        d_range=(0, 1),
        q_range=(0, 2),
        P_range=(0, 1),
        D_range=(0, 1),
        Q_range=(0, 1),
        m=12
    ):
        super().__init__()
        self.order_grid = list(itertools.product(p_range, d_range, q_range))
        self.seasonal_order_grid = [
            (P, D, Q, m) for P in P_range for D in D_range for Q in Q_range
        ]


# ============================================================
# SARIMAX : seasonal=True, exog required
# ============================================================

class SARIMAXModel(BaseARIMAModel):
    def __init__(
        self,
        p_range=(0, 2),
        d_range=(0, 1),
        q_range=(0, 2),
        P_range=(0, 1),
        D_range=(0, 1),
        Q_range=(0, 1),
        m=12
    ):
        super().__init__()
        self.order_grid = list(itertools.product(p_range, d_range, q_range))
        self.seasonal_order_grid = [
            (P, D, Q, m) for P in P_range for D in D_range for Q in Q_range
        ]

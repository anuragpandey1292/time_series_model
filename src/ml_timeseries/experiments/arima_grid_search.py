# experiments/arima_grid_search.py

import itertools
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def grid_search(
    y,
    exog=None,
    order_grid=None,
    seasonal_order_grid=None,
    metric="aic"
):
    best_score = np.inf
    best_model = None
    best_order = None
    best_seasonal_order = None

    for order in order_grid:
        if seasonal_order_grid is None:
            try:
                model = SARIMAX(
                    y,
                    exog=exog,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                score = results.aic if metric == "aic" else results.bic

                if score < best_score:
                    best_score = score
                    best_model = results
                    best_order = order

            except:
                continue
        else:
            for seasonal_order in seasonal_order_grid:
                try:
                    model = SARIMAX(
                        y,
                        exog=exog,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    results = model.fit(disp=False)
                    score = results.aic if metric == "aic" else results.bic

                    if score < best_score:
                        best_score = score
                        best_model = results
                        best_order = order
                        best_seasonal_order = seasonal_order

                except:
                    continue

    return {
        "best_model": best_model,
        "best_order": best_order,
        "best_seasonal_order": best_seasonal_order,
        "best_score": best_score
    }

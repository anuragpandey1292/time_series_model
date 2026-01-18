# experiments/model_comparison.py

import pandas as pd
from Evaluation.Evaluate import evaluate_model


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    models,
    y_train,
    y_test,
    steps,
    exog_train=None,
    exog_test=None
):
    """
    models = {
        "ARIMA": ARIMAModel(),
        "SARIMA": SARIMAModel(),
        "HoltWinters": HoltWintersModel(...)
    }
    """

    results = []

    for model_name, model in models.items():
        print(f"Evaluating {model_name} ...")

        metrics = evaluate_model(
            model=model,
            y_train=y_train,
            y_test=y_test,
            steps=steps,
            exog_train=exog_train,
            exog_test=exog_test,
            model_name=model_name
        )

        results.append(metrics)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("RMSE").reset_index(drop=True)

    return df_results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_comparison(df, path):
    df.to_csv(path, index=False)

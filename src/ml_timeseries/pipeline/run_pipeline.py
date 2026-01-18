# pipeline/run_pipeline.py

from pipeline.train_pipeline import train_pipeline
from pipeline.inference_pipeline import inference_pipeline


def run_pipeline():
    print("Starting training pipeline...")
    best_model_name, results = train_pipeline()

    print("\nTraining completed.")
    print(f"Best model: {best_model_name}")

    print("\nRunning inference pipeline...")
    forecast = inference_pipeline(forecast_steps=12)

    print("\nFuture Forecast:")
    print(forecast)


if __name__ == "__main__":
    run_pipeline()

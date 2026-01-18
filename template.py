"""
Common utilities for time series forecasting pipelines
"""

import logging
import argparse
from pathlib import Path
import yaml

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ml-timeseries")

# ----------------------------
# Config loader
# ----------------------------
def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)

# ----------------------------
# CLI arguments
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Time Series Forecasting Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="src/ml_timeseries/config/config.yaml",
        help="Path to config file"
    )
    return parser.parse_args()
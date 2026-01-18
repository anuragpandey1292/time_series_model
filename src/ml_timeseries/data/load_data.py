# src/ml_timeseries/data/load_data.py
import pandas as pd

def load_data(path: str, date_col: str = None):
    df = pd.read_csv(path)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df.set_index(date_col, inplace=True)

    return df

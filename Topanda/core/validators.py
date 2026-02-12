

import numpy as np
import pandas as pd

from metrics import MetricFactory

def validate_numeric_dataframe(df: pd.DataFrame):
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(
            f"DataFrame contains non-numeric columns: {non_numeric}. "
            "Please use DataProcessor from topanda.preprocessing to clean your data first."
        )
    
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"DataFrame contains NaN values in columns: {nan_cols}. "
            "Use DataProcessor to handle missing values."
        )

def validate_metric_name(metric_name: str) -> None:

    if metric_name not in MetricFactory.SUPPORTED_METRICS:
        raise ValueError(
            "Unsupported metric"
            f"Supported metrics: {':'.join(MetricFactory.SUPPORTED_METRICS)}"
        )
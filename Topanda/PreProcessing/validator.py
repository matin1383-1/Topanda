"""Input validation utilities for topanda preprocessing."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def validate_dataframe(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if len(df.columns) == 0:
        raise ValueError("DataFrame has no columns")


def check_target_column(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(
            "Target column not found in DataFrame. "
        )


def identify_column_types(
    df: pd.DataFrame, exclude_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Automatically detect numeric and categorical columns.

    Numeric columns are identified via select_dtypes(include=[np.number]).
    Categorical columns are all non-numeric columns. Columns in exclude_cols
    are omitted from both lists.
    """
    exclude_cols = exclude_cols or []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    categorical_cols = [c for c in categorical_cols if c not in exclude_cols]

    return numeric_cols, categorical_cols


def resolve_column_types(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:

    exclude_cols = exclude_cols or []

    numeric_cols = list(numeric_cols) if numeric_cols is not None else []
    categorical_cols = list(categorical_cols) if categorical_cols is not None else []

    invalid_numeric = set(numeric_cols) - set(df.columns)
    invalid_categorical = set(categorical_cols) - set(df.columns)
    
    if invalid_numeric:
        raise ValueError(
            f"Invalid numeric columns: {invalid_numeric}. "
        )
    if invalid_categorical:
        raise ValueError(
            f"Invalid categorical columns: {invalid_categorical}. "
        )

    # Check for overlap
    overlap = set(numeric_cols) & set(categorical_cols)
    if overlap:
        raise ValueError(
            f"Columns cannot be both numeric and categorical: {overlap}"
        )

    # Filter out excluded columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    categorical_cols = [c for c in categorical_cols if c not in exclude_cols]

    return numeric_cols, categorical_cols

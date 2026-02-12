"""Preprocessing pipeline for topanda."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from categorical_processor import CategoricalProcessor
from numeric_processor import NumericProcessor
from validator import (
    check_target_column,
    identify_column_types,
    resolve_column_types,
    validate_dataframe,
)


class DataProcessor:
    """
    combines NumericProcessor and CategoricalProcessor,
    handles target separation, validates inputs, and merges processed results.
    """

    def __init__(
        self,
        standardize_numeric: bool = True,
        embed_categorical: bool = True,
        embedding_dim: int = 10,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ):
        self.standardize_numeric = standardize_numeric
        self.embed_categorical = embed_categorical
        self.embedding_dim = embedding_dim
        self.user_numeric_cols = numeric_cols
        self.user_categorical_cols = categorical_cols
        self._numeric_processor: Optional[NumericProcessor] = None
        self._categorical_processor: Optional[CategoricalProcessor] = None
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ):
        validate_dataframe(df)

        y: Optional[np.ndarray] = None
        work = df.copy()
        if target_col is not None:
            check_target_column(df, target_col)
            y = df[target_col].values
            work = work.drop(columns=[target_col])

        if self.user_numeric_cols is None and self.user_categorical_cols is None:
            self._numeric_cols, self._categorical_cols = identify_column_types(work)
        else:
            self._numeric_cols, self._categorical_cols = resolve_column_types(
                work,
                numeric_cols=self.user_numeric_cols,
                categorical_cols=self.user_categorical_cols,
            )

        numeric_df: pd.DataFrame
        if self._numeric_cols:
            self._numeric_processor = NumericProcessor(
                standardize=self.standardize_numeric
            )
            numeric_df = self._numeric_processor.fit_transform(
                work[self._numeric_cols]
            )
        else:
            numeric_df = pd.DataFrame(index=work.index)

        if self._categorical_cols and self.embed_categorical:
            if y is None:
                raise ValueError(
                    "Embeddings require a target column"
                )
            self._categorical_processor = CategoricalProcessor(
                embedding_dim=self.embedding_dim,
                epochs=10,
            )
            cat_df = self._categorical_processor.fit_transform(
                work[self._categorical_cols], y
            )
        else:
            cat_df = pd.DataFrame(index=work.index)

        self._fitted = True

        parts = [p for p in [numeric_df, cat_df] if not p.empty]
        if not parts:
            X_processed = pd.DataFrame(index=work.index)
        else:
            X_processed = pd.concat(parts, axis=1)

        return (X_processed, y)

    def transform(self, df: pd.DataFrame):
        if not self._fitted:
            raise ValueError(
                "you should fitt it first"
            )
        validate_dataframe(df)
        parts: List[pd.DataFrame] = []
        if self._numeric_cols and self._numeric_processor is not None:
            parts.append(
                self._numeric_processor.transform(df[self._numeric_cols])
            )
        if self._categorical_cols and self._categorical_processor is not None:
            parts.append(
                self._categorical_processor.transform(
                    df[self._categorical_cols]
                )
            )

        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)

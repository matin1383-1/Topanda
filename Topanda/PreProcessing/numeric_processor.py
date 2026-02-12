"""Numeric (ratio/interval) data processing for topanda."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class NumericProcessor:

    def __init__(self, standardize: bool = True , strategy: str = "mean"):
        self.standardize = standardize
        self._imputer = SimpleImputer(strategy=strategy)
        self._scaler = StandardScaler() if standardize else None
        self._fitted = False
        self._columns: list = []

    def fit(self, X: pd.DataFrame):
        self._columns = X.columns.tolist()
        self._imputer.fit(X)
        if self._scaler is not None:
            filled = self._imputer.transform(X)
            self._scaler.fit(filled)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame):

        if not self._fitted:
            raise ValueError("Call fitt first")
        filled = self._imputer.transform(X)
        if self._scaler is not None:
            filled = self._scaler.transform(filled)
        return pd.DataFrame(
            filled,
            index=X.index,
            columns=self._columns,
        )

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

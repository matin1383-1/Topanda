
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import Any, Dict, Optional, Union

from metrics import MetricFactory
from cache import DistanceCache
from validators import validate_numeric_dataframe, validate_metric_name


class MetricSpace:
    
    def __init__(
        self,
        data: pd.DataFrame,
        metric: str = 'euclidean',
        cache_distances: bool = True,
        metric_params: Optional[Dict[str, Any]] = None
    ):
        validate_numeric_dataframe(data)
        validate_metric_name(metric)
        
        self.data = data.copy()
        self.X = data.values
        self.n_samples, self.n_features = self.X.shape
        
        self.metric_name = metric
        metric_params = metric_params or {}
        self._metric_func = MetricFactory.get_metric(metric, **metric_params)
        self._metric_params = metric_params
        
        self._cache = DistanceCache() if cache_distances else None
        self._distances = None
    @property
    def distances(self):
        if self._distances is not None:
            return self._distances
        
        if self._cache is not None:
            cached = self._cache.get('distances')
            if cached is not None:
                self._distances = cached
                return cached
        distances_matrix = cdist(self.X, self.X, metric=self._metric_func)
        if self._cache is not None:
            self._cache.set('distances', distances_matrix)
        self._distances = distances_matrix
        return distances_matrix
    
    def invalidate_cache(self):
        self._distances = None
        if self._cache is not None:
            self._cache.clear()

    def distance_between(self, i: int, j: int):
        if i < 0 or i >= self.n_samples or j < 0 or j >= self.n_samples:
            raise IndexError(
                f"Indices must be in [0, {self.n_samples - 1}]."
            )
        if self._distances is not None:
            return float(self._distances[i, j])
        return float(self._metric_func(self.X[i], self.X[j]))

    def distance_to_points(
        self, x_new: Union[np.ndarray, pd.Series]
    ):
        x = np.asarray(x_new, dtype=np.float64)
        if x.ndim == 2:
            if x.shape[0] != 1 or x.shape[1] != self.n_features:
                raise ValueError(
                    f"x_new must have shape ({self.n_features},)"
                )
            x = x.ravel()
        elif x.ndim != 1 or x.shape[0] != self.n_features:
            raise ValueError(
                f"x_new must have length {self.n_features}. Got shape {x.shape}."
            )
        d = cdist(x.reshape(1, -1), self.X, metric=self._metric_func)
        return d.ravel()
    def __str__(self):
        return (
            f"MetricSpace(samples={self.n_samples}, "
            f"features={self.n_features}, "
            f"metric='{self.metric_name}')"
        )
    def __len__(self) -> int:
        return self.n_samples

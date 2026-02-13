import numpy as np
from typing import Optional, Union
import pandas as pd
from collections import Counter
from scipy.spatial.distance import cdist


class KNNClassifier:
    def __init__(self, metric_space, n_neighbors: int = 5, weights: str = 'uniform'):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if n_neighbors > metric_space.n_samples:
            raise ValueError("n_neighbors cannot exceed number of samples")
        if weights not in ['uniform', 'distance']:
            raise ValueError("weights must be 'uniform' or 'distance'")

        self.ms = metric_space
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.labels = None
        self.classes_ = None
        self._fitted = False

    def fit(self, labels: np.ndarray):
        labels = np.asarray(labels)
        if len(labels) != self.ms.n_samples:
            raise ValueError("Label length must match number of samples")
        self.labels = labels
        self.classes_ = np.unique(labels)
        self._fitted = True
        return self

    def _find_k_nearest(self, distances: np.ndarray):
        sorted_indices = np.argsort(distances)
        return sorted_indices[: self.n_neighbors]

    def _predict_single(self, distances: np.ndarray):
        neighbor_indices = self._find_k_nearest(distances)
        neighbor_labels = self.labels[neighbor_indices]
        neighbor_distances = distances[neighbor_indices]

        if self.weights == 'uniform':
            vote_counts = Counter(neighbor_labels)
            predicted_class = vote_counts.most_common(1)[0][0]
        else:  # distance weights
            weights = np.where(neighbor_distances == 0, 1e10, 1.0 / neighbor_distances)
            class_weights = {}
            for label, w in zip(neighbor_labels, weights):
                class_weights[label] = class_weights.get(label, 0) + w
            predicted_class = max(class_weights, key=class_weights.get)
        return predicted_class

    def predict(self, X_new: np.ndarray) -> str:
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict")
        X_new = np.asarray(X_new)
        if X_new.ndim != 1:
            raise ValueError(f"X_new must be 1D, got shape {X_new.shape}")
        if X_new.shape[0] != self.ms.n_features:
            raise ValueError(f"Expected {self.ms.n_features} features, got {X_new.shape[0]}")
        distances = self.ms.distance_to_points(X_new)
        return self._predict_single(distances)

    def score(self, true_labels: np.ndarray) -> float:
        if not self._fitted:
            raise RuntimeError("Must call fit() before score")

        predictions = []
        for i in range(self.ms.n_samples):
            point = self.ms.X[i]
            distances = self.ms.distance_to_points(point)
            distances[i] = np.inf  # ignore self-distance
            predictions.append(self._predict_single(distances))
        predictions = np.array(predictions)
        true_labels = np.asarray(true_labels)
        if len(predictions) != len(true_labels):
            raise ValueError("Prediction length mismatch with true labels")
        accuracy = np.mean(predictions == true_labels)
        return float(accuracy)

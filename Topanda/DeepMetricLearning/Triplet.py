


#-------------------------------------------------------------------------------------------------------------
"""codes of this section is based on the implementation of pytorch deep metric learning
   library in https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/triplet_margin_loss.py
"""
#--------------------------------------------------------------------------------------------------
from typing import Optional, Tuple

import numpy as np
import pandas as pd



import torch
import torch.nn as nn
import torch.optim as optim

from ..core.metric_space import MetricSpace


class _MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        hidden = max(64, 2 * embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TripletLearner:


    def __init__(
        self,
        embedding_dim: int = 16,
        margin: float = 1.0,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[_MLPEmbedding] = None
        self._fitted = False
        self._input_dim: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: np.ndarray):

        X_arr = np.asarray(X.values, dtype=np.float32)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y_arr.ndim != 1:
            raise ValueError("y must be 1D (n_samples,)")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X_arr.shape
        self._input_dim = n_features

        self.model = _MLPEmbedding(input_dim=n_features, embedding_dim=self.embedding_dim)
        self.model.to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.TripletMarginLoss(margin=self.margin, p=2)

        X_tensor = torch.from_numpy(X_arr).to(self.device)
        y_tensor = torch.from_numpy(y_arr)

        class_to_indices = self._build_class_index(y_arr)

        self.model.train()
        for epoch in range(self.epochs):
            triplets = self._sample_triplets(
                y_arr, class_to_indices, self.batch_size * max(1, n_samples // self.batch_size)
            )
            if len(triplets) == 0:
                break

            np.random.shuffle(triplets)

            batch_losses = []

            for start in range(0, len(triplets), self.batch_size):
                batch = triplets[start : start + self.batch_size]
                if len(batch) == 0:
                    continue

                anchors_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=self.device)
                positives_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=self.device)
                negatives_idx = torch.tensor([t[2] for t in batch], dtype=torch.long, device=self.device)

                anchors = X_tensor[anchors_idx]
                positives = X_tensor[positives_idx]
                negatives = X_tensor[negatives_idx]

                optimizer.zero_grad()
                emb_a = self.model(anchors)
                emb_p = self.model(positives)
                emb_n = self.model(negatives)

                loss = criterion(emb_a, emb_p, emb_n)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self.model is None or self._input_dim is None:
            raise RuntimeError("TripletLearner must be fitted before calling transform()")

        X_arr = np.asarray(X.values, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[1] != self._input_dim:
            raise ValueError(
                f"Expected X with shape (_, {self._input_dim}), got {X_arr.shape}"
            )

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_arr).to(self.device)
            embeddings = self.model(X_tensor).cpu().numpy()

        cols = [f"emb_{i}" for i in range(self.embedding_dim)]
        Z = pd.DataFrame(embeddings, columns=cols, index=X.index)
        return Z

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    

    #------------------------------------------------------------------------
    """
    this is the modified section of the code where we do the same thing in 
    """
    #------------------------------------------------------------------------
    def fit_metric_space(self, metric_space: MetricSpace, labels: np.ndarray) -> "TripletLearner":
        return self.fit(metric_space.data, labels)

    def transform_metric_space(self, metric_space: MetricSpace) -> MetricSpace:
        Z = self.transform(metric_space.data)
        return MetricSpace(Z, metric='euclidean', cache_distances=True)

    def fit_transform_metric_space(self, metric_space: MetricSpace, labels: np.ndarray) -> MetricSpace:
        self.fit_metric_space(metric_space, labels)
        return self.transform_metric_space(metric_space)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_class_index(y: np.ndarray) -> dict:
        """Build mapping: class → list of indices."""
        class_to_indices = {}
        for idx, label in enumerate(y):
            class_to_indices.setdefault(label, []).append(idx)
        return class_to_indices

    @staticmethod
    def _sample_triplets(
        y: np.ndarray,
        class_to_indices: dict,
        max_triplets: int,
    ) -> list:
        rng = np.random.default_rng()
        labels = np.unique(y)
        if len(labels) < 2:
            return []

        triplets = []

        for label in labels:
            pos_indices = class_to_indices[label]
            if len(pos_indices) < 2:
                continue

            neg_labels = [l for l in labels if l != label]
            neg_indices = np.concatenate([class_to_indices[l] for l in neg_labels])

            # Number of triplets to draw from this class
            n_pos_pairs = len(pos_indices)
            for _ in range(n_pos_pairs):
                a, p = rng.choice(pos_indices, size=2, replace=False)
                n = rng.choice(neg_indices)
                triplets.append((a, p, n))
                if len(triplets) >= max_triplets:
                    return triplets

        return triplets
"""Additive pipelines using the original Topanda encoder architecture."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..Triplet import TripletLearner, _MLPEmbedding
from .miners import balanced_anchor_positive_batches, candidate_pool, mine_negatives
from .training import TrainingConfig, set_reproducible, train_refiner, transform


def _frame(X):
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if frame.ndim != 2 or not frame.shape[1] or not len(frame):
        raise ValueError("X must be a nonempty numeric matrix")
    if not np.isfinite(frame.to_numpy(dtype=np.float32)).all():
        raise ValueError("X must contain finite numeric values")
    return frame


def _labels(y, n):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n or pd.isna(y).any():
        raise ValueError("y must contain one nonmissing label per training row")
    _, encoded = np.unique(y, return_inverse=True)
    counts = np.bincount(encoded)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Training requires at least two classes and two rows per class")
    return encoded


class HardNegativeTopanda(TripletLearner):
    """Train Topanda's encoder from scratch with nearest wrong-class negatives.

    Mining searches a balanced training candidate pool in the current embedding.
    This changes the encoder training, and does not use a residual correction.
    """

    def __init__(self, embedding_dim=16, margin=1.0, epochs=20, batch_size=128,
                 lr=1e-3, device="cpu", candidate_pool_size=256, seed=42):
        super().__init__(embedding_dim, margin, epochs, batch_size, lr, device)
        self.candidate_pool_size = candidate_pool_size
        self.seed = seed

    def fit(self, X, y):
        X = _frame(X)
        labels = _labels(y, len(X))
        if min(self.embedding_dim, self.batch_size, self.candidate_pool_size) <= 0:
            raise ValueError("Dimensions, batch size and pool size must be positive")
        if self.epochs < 0 or self.margin <= 0 or self.lr <= 0:
            raise ValueError("Epochs must be nonnegative; margin and learning rate positive")
        self._fitted = False
        set_reproducible(self.seed)
        self._input_dim = X.shape[1]
        self.model = _MLPEmbedding(self._input_dim, self.embedding_dim).to(self.device)
        inputs = torch.as_tensor(X.to_numpy(dtype=np.float32), device=self.device)
        targets = torch.as_tensor(labels, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.history_ = []
        for epoch in range(self.epochs):
            self.model.train()
            pairs = balanced_anchor_positive_batches(
                labels, self.batch_size, np.random.default_rng(self.seed + epoch * 1009))
            pool_rng = np.random.default_rng(self.seed + epoch * 1009 + 1)
            losses = []
            for a, p in pairs:
                c = candidate_pool(labels, self.candidate_pool_size, pool_rng)
                a, p, c = [torch.as_tensor(v, device=self.device) for v in (a, p, c)]
                za, zp = self.model(inputs[a]), self.model(inputs[p])
                with torch.no_grad():
                    zc = self.model(inputs[c])
                mined = mine_negatives("hard", za, zp, zc, targets[a], targets[c],
                                       c, self.margin)
                zn = self.model(inputs[mined.negative_indices])
                loss = criterion(za, zp, zn)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite hard-negative training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0,
                                               error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.history_.append({"epoch": epoch + 1, "triplet_loss": float(np.mean(losses))})
        self._fitted = True
        return self

    def transform(self, X):
        return super().transform(_frame(X))


class GPTopanda:
    """Refine an already fitted Topanda model with random negatives by default.

    fit() receives preprocessed TRAINING inputs, using the same preprocessing
    as the supplied fitted base. A frozen copy is owned by this pipeline.
    """
    strategy = "random"

    def __init__(self, base: TripletLearner, config: TrainingConfig | None = None):
        self.base = base
        self.config = config or TrainingConfig(strategy=self.strategy, preservation_lambda=1.0)

    def fit(self, X, y):
        X = _frame(X)
        labels = _labels(y, len(X))
        if not self.base._fitted or self.base.model is None:
            raise ValueError("Fit the original Topanda model before refinement")
        config = replace(self.config, strategy=self.strategy)
        if config.preservation_lambda <= 0 or config.beta <= 0:
            raise ValueError("GP requires positive preservation_lambda and beta")
        base_copy = copy.deepcopy(self.base)
        base_copy.model.eval()
        for parameter in base_copy.model.parameters():
            parameter.requires_grad_(False)
        z1 = base_copy.transform(X).to_numpy(dtype=np.float32)
        result = train_refiner(z1, labels, config)
        if result.nonfinite:
            raise FloatingPointError("Nonfinite refinement loss")
        self.base_ = base_copy
        self.result_ = result
        self.history_ = result.history
        self.embedding_dim = z1.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "result_"):
            raise RuntimeError("Fit the refinement pipeline before transform")
        X = _frame(X)
        z1 = self.base_.transform(X).to_numpy(dtype=np.float32)
        z2 = transform(self.result_.model, z1)
        return pd.DataFrame(z2, index=X.index,
                            columns=[f"emb_{i}" for i in range(self.embedding_dim)])

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class GPHTopanda(GPTopanda):
    """Geometry-preserving refinement with hard-negative mining (GP-H)."""
    strategy = "hard"

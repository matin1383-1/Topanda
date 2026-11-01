"""Residual second-stage network with an exact identity initialization."""
from __future__ import annotations

import torch
from torch import nn


class CorrectionNetwork(nn.Module):
    """Dimension -> hidden -> dimension correction branch.

    Zeroing only the last layer gives exact identity at initialization while
    retaining a trainable gradient path. Earlier layers begin receiving
    gradients after the first optimizer update.
    """

    def __init__(self, dimension: int = 32, hidden_dimension: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, dimension),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ResidualTopanda(nn.Module):
    """Refine frozen first-stage embeddings without replacing them."""

    def __init__(self, dimension: int = 32, hidden_dimension: int = 64, beta: float = 1.0):
        super().__init__()
        self.dimension = int(dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.beta = float(beta)
        self.correction = CorrectionNetwork(self.dimension, self.hidden_dimension)

    def forward(self, z1: torch.Tensor) -> torch.Tensor:
        if z1.ndim != 2 or z1.shape[1] != self.dimension:
            raise ValueError(f"Expected (_, {self.dimension}), got {tuple(z1.shape)}")
        return z1 + self.beta * self.correction(z1)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

"""Geometry-preservation objectives used by GP-Topanda."""
from __future__ import annotations

import torch


def pointwise_preservation(z2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """Mean squared Euclidean displacement per observation."""
    if z2.shape != z1.shape:
        raise ValueError("z2 and z1 must have identical shapes")
    return (z2 - z1.detach()).square().sum(dim=1).mean()


def pairwise_distance_preservation(z2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """Optional, separately recorded pairwise-distance ablation objective."""
    if z2.shape != z1.shape:
        raise ValueError("z2 and z1 must have identical shapes")
    if len(z2) < 2:
        return z2.sum() * 0.0
    old = torch.pdist(z1.detach(), p=2)
    new = torch.pdist(z2, p=2)
    return (new - old).square().mean()

"""Leakage-free training loop for geometry-preserving refinement."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from time import perf_counter
import random
import numpy as np
import torch
from torch import nn

from .miners import balanced_anchor_positive_batches, candidate_pool, mine_negatives
from .preservation_losses import pointwise_preservation
from .residual_topanda import ResidualTopanda


@dataclass(frozen=True)
class TrainingConfig:
    strategy: str
    preservation_lambda: float
    beta: float = 1.0
    epochs: int = 20
    batch_size: int = 128
    candidate_pool_size: int = 256
    learning_rate: float = 0.001
    margin: float = 1.0
    gradient_clip_norm: float = 5.0
    hidden_dimension: int = 64
    seed: int = 10042
    device: str = "cpu"


@dataclass
class TrainingResult:
    model: ResidualTopanda
    history: list[dict]
    training_seconds: float
    mining_seconds: float
    config: dict
    nonfinite: bool


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def transform(model: ResidualTopanda, z1: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(z1), batch_size):
            x = torch.as_tensor(z1[start:start + batch_size], dtype=torch.float32, device=device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def train_refiner(z1: np.ndarray, labels: np.ndarray, config: TrainingConfig) -> TrainingResult:
    """Train on first-stage training embeddings and labels only."""
    if config.strategy not in {"random", "hard", "semi_hard"}:
        raise ValueError(config.strategy)
    z1 = np.asarray(z1, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if z1.ndim != 2 or len(z1) != len(y):
        raise ValueError("Invalid training arrays")
    if y.ndim != 1 or not len(y) or not z1.shape[1] or not np.isfinite(z1).all():
        raise ValueError("Training arrays must be nonempty, finite and correctly shaped")
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("At least two classes with two rows each are required")
    if config.batch_size < len(counts) or config.candidate_pool_size < len(counts):
        raise ValueError("Batch and candidate pool must accommodate every class")
    if config.epochs < 0 or min(config.learning_rate, config.margin, config.gradient_clip_norm,
                                config.hidden_dimension) <= 0:
        raise ValueError("Invalid training configuration")
    set_reproducible(config.seed)
    device = torch.device(config.device)
    frozen = torch.as_tensor(z1, dtype=torch.float32, device=device).detach()
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    model = ResidualTopanda(z1.shape[1], config.hidden_dimension, config.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    triplet_loss = nn.TripletMarginLoss(margin=config.margin, p=2, reduction="mean")
    initial = transform(model, z1)
    if not np.array_equal(initial, z1):
        np.testing.assert_allclose(initial, z1, rtol=0.0, atol=1e-7)
    old_variance = float(np.var(z1, axis=0).mean())
    history = []
    total_mining = 0.0
    nonfinite = False
    started = perf_counter()
    for epoch in range(config.epochs):
        model.train()
        # Pair and pool RNG streams are independent of the mining strategy.
        pair_rng = np.random.default_rng(config.seed + epoch * 1009)
        pool_rng = np.random.default_rng(config.seed + epoch * 1009 + 1)
        torch_rng = torch.Generator(device=device).manual_seed(config.seed + epoch * 1009 + 2)
        batches = balanced_anchor_positive_batches(y, config.batch_size, pair_rng)
        sums = dict(total_loss=0.0, triplet_loss=0.0, preservation_loss=0.0,
                    gradient_norm=0.0, positive_distance=0.0, negative_distance=0.0,
                    active_triplet_fraction=0.0, hard_triplet_fraction=0.0,
                    semi_hard_triplet_fraction=0.0, fallback_frequency=0.0)
        observations = 0
        for anchors_np, positives_np in batches:
            pool_np = candidate_pool(y, config.candidate_pool_size, pool_rng)
            anchors = torch.as_tensor(anchors_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=device)
            pool = torch.as_tensor(pool_np, dtype=torch.long, device=device)
            za = model(frozen[anchors]); zp = model(frozen[positives]); zc = model(frozen[pool])
            mt = perf_counter()
            mined = mine_negatives(config.strategy, za, zp, zc, y_t[anchors], y_t[pool], pool,
                                   config.margin, torch_rng)
            elapsed_mining = perf_counter() - mt
            total_mining += elapsed_mining
            negatives = mined.negative_indices
            zn = model(frozen[negatives])
            tri = triplet_loss(za, zp, zn)
            # Include every unique training role in the preservation term.
            combined_idx = torch.unique(torch.cat([anchors, positives, negatives]))
            refined = model(frozen[combined_idx])
            preserve = pointwise_preservation(refined, frozen[combined_idx])
            total = tri + config.preservation_lambda * preserve
            if not bool(torch.isfinite(total)):
                nonfinite = True
                break
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm,
                                                     error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                dap = torch.linalg.vector_norm(za - zp, dim=1)
                dan = torch.linalg.vector_norm(za - zn, dim=1)
                active = (dap - dan + config.margin) > 0
                n = len(anchors)
                observations += n
                values = dict(
                    total_loss=float(total), triplet_loss=float(tri), preservation_loss=float(preserve),
                    gradient_norm=float(raw_norm), positive_distance=float(dap.mean()),
                    negative_distance=float(dan.mean()), active_triplet_fraction=float(active.float().mean()),
                    hard_triplet_fraction=mined.hard_fraction,
                    semi_hard_triplet_fraction=mined.semi_hard_fraction,
                    fallback_frequency=mined.fallback_fraction,
                )
                for key, value in values.items(): sums[key] += value * n
        if nonfinite:
            history.append(dict(epoch=epoch + 1, nonfinite=True))
            break
        current = transform(model, z1)
        variance = float(np.var(current, axis=0).mean())
        displacement = np.linalg.norm(current - z1, axis=1)
        row = {k: float(v / observations) for k, v in sums.items()}
        row.update(epoch=epoch + 1, embedding_variance=variance,
                   variance_ratio=float(variance / max(old_variance, np.finfo(float).eps)),
                   embedding_norm=float(np.linalg.norm(current, axis=1).mean()),
                   relative_displacement=float(displacement.mean() /
                                               max(np.linalg.norm(z1 - z1.mean(axis=0), axis=1).mean(),
                                                   np.finfo(float).eps)),
                   mining_seconds_cumulative=float(total_mining), nonfinite=False)
        history.append(row)
    return TrainingResult(model=model, history=history, training_seconds=perf_counter() - started,
                          mining_seconds=total_mining, config=asdict(config), nonfinite=nonfinite)


def state_dict_cpu(model: ResidualTopanda) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def unchanged_state(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)

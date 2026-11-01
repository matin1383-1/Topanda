"""Matched balanced random, hard, and semi-hard triplet mining."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class MiningResult:
    negative_indices: torch.Tensor
    hard_fraction: float
    semi_hard_fraction: float
    fallback_fraction: float
    mining_seconds: float = 0.0


def balanced_anchor_positive_batches(
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One balanced epoch of matched random anchor-positive pairs.

    Every strategy receives these exact pairs. Minority-class anchors are
    cycled so each batch is as balanced as possible; positives are sampled
    within class without using validation or test rows.
    """
    y = np.asarray(labels)
    classes = np.unique(y)
    if batch_size < len(classes):
        raise ValueError("Batch size must be at least the number of classes")
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    by_class = {c: np.flatnonzero(y == c) for c in classes}
    if any(len(v) < 2 for v in by_class.values()):
        raise ValueError("Every class needs at least two training examples")
    order = {c: rng.permutation(v) for c, v in by_class.items()}
    cursor = {c: 0 for c in classes}
    n_batches = int(np.ceil(len(y) / batch_size))
    output = []
    for _ in range(n_batches):
        anchors = []
        for j in range(batch_size):
            c = classes[j % len(classes)]
            if cursor[c] >= len(order[c]):
                order[c] = rng.permutation(by_class[c])
                cursor[c] = 0
            anchors.append(int(order[c][cursor[c]]))
            cursor[c] += 1
        anchors = np.asarray(anchors, dtype=np.int64)
        positives = np.empty_like(anchors)
        for j, a in enumerate(anchors):
            pool = by_class[y[a]]
            p = int(pool[rng.integers(len(pool))])
            if p == a:
                p = int(pool[(np.flatnonzero(pool == a)[0] + 1) % len(pool)])
            positives[j] = p
        output.append((anchors, positives))
    return output


def candidate_pool(labels: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified candidate pool shared by all mining strategies."""
    y = np.asarray(labels)
    classes = np.unique(y)
    if size < len(classes):
        raise ValueError("Candidate pool size must be at least the number of classes")
    parts = []
    for i, c in enumerate(rng.permutation(classes)):
        per = size // len(classes) + int(i < size % len(classes))
        idx = np.flatnonzero(y == c)
        parts.append(rng.choice(idx, size=per, replace=len(idx) < per))
    pool = np.concatenate(parts)
    rng.shuffle(pool)
    return pool.astype(np.int64, copy=False)


def mine_negatives(
    strategy: str,
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    anchor_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    candidate_indices: torch.Tensor,
    margin: float,
    generator: torch.Generator | None = None,
) -> MiningResult:
    """Mine from the current refined embedding with documented fallback.

    Semi-hard fallback is the closest available different-class candidate,
    matching the hard miner. The returned indices refer to the full training
    array, not positions within the candidate pool.
    """
    if strategy not in {"random", "hard", "semi_hard"}:
        raise ValueError(strategy)
    distances = torch.cdist(anchor_embeddings.detach(), candidate_embeddings.detach(), p=2)
    valid = anchor_labels[:, None] != candidate_labels[None, :]
    if not bool(valid.any(dim=1).all()):
        raise ValueError("Candidate pool lacks a different-class negative")
    inf = torch.tensor(float("inf"), device=distances.device)
    masked = torch.where(valid, distances, inf)
    closest_pos = masked.argmin(dim=1)
    dap = torch.linalg.vector_norm(anchor_embeddings.detach() - positive_embeddings.detach(), dim=1)
    if strategy == "random":
        chosen = []
        for row in valid:
            options = torch.nonzero(row, as_tuple=False).flatten()
            pick = torch.randint(len(options), (1,), generator=generator, device=options.device)
            chosen.append(options[pick].squeeze(0))
        chosen_pos = torch.stack(chosen)
        fallback = torch.zeros(len(chosen_pos), dtype=torch.bool, device=distances.device)
    elif strategy == "hard":
        chosen_pos = closest_pos
        fallback = torch.zeros(len(chosen_pos), dtype=torch.bool, device=distances.device)
    else:
        semimask = valid & (distances > dap[:, None]) & (distances < (dap + margin)[:, None])
        semid = torch.where(semimask, distances, inf)
        has = semimask.any(dim=1)
        chosen_pos = torch.where(has, semid.argmin(dim=1), closest_pos)
        fallback = ~has
    dan = distances[torch.arange(len(distances), device=distances.device), chosen_pos]
    hard = dan <= dap
    semi = (dan > dap) & (dan < dap + margin)
    return MiningResult(
        negative_indices=candidate_indices[chosen_pos],
        hard_fraction=float(hard.float().mean().cpu()),
        semi_hard_fraction=float(semi.float().mean().cpu()),
        fallback_fraction=float(fallback.float().mean().cpu()),
    )

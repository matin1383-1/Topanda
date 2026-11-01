"""Compare Topanda pipelines with 5-neighbor classification on Iris."""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Support direct execution from the repository root without installing a package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Topanda import (
    GPTopanda,
    GPHTopanda,
    HardNegativeTopanda,
    TrainingConfig,
    TripletLearner,
)


def separation_ratio(embeddings, labels):
    """Return mean between-class distance divided by mean within-class distance."""
    distances = pairwise_distances(embeddings)
    row, column = np.triu_indices(len(labels), k=1)
    same_class = labels[row] == labels[column]
    within = distances[row[same_class], column[same_class]].mean()
    between = distances[row[~same_class], column[~same_class]].mean()
    return float(between / within)


features, labels = load_iris(return_X_y=True)
train, test = train_test_split(
    np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels
)
scaler = StandardScaler().fit(features[train])
train_features = pd.DataFrame(scaler.transform(features[train]))
test_features = pd.DataFrame(scaler.transform(features[test]))

base = TripletLearner(embedding_dim=16, epochs=20, device="cpu").fit(
    train_features, labels[train]
)
config = TrainingConfig(
    strategy="random", preservation_lambda=1.0, epochs=20, seed=42
)
pipelines = {
    "Topanda": base,
    "Hard-negative Topanda": HardNegativeTopanda(epochs=20, seed=42).fit(
        train_features, labels[train]
    ),
    "GP": GPTopanda(base, config).fit(train_features, labels[train]),
    "GP-H": GPHTopanda(base, config).fit(train_features, labels[train]),
}

print("Method                    Accuracy (%)   Separation ratio")
for name, pipeline in pipelines.items():
    train_embedding = pipeline.transform(train_features)
    test_embedding = pipeline.transform(test_features)
    classifier = KNeighborsClassifier(n_neighbors=5).fit(
        train_embedding, labels[train]
    )
    accuracy = accuracy_score(labels[test], classifier.predict(test_embedding))
    ratio = separation_ratio(test_embedding, labels[test])
    print(f"{name:25s} {100 * accuracy:10.2f} {ratio:18.4f}")

# Topanda

Topanda is a lightweight toolkit for tabular data preprocessing, distance-based (metric-space) machine learning workflows, and nearest-neighbor algorithms. It provides:

- **Preprocessing Pipeline**: Numeric standardization and categorical entity embeddings
- **MetricSpace**: A unified interface for computing and querying distances in metric spaces
- **ML Algorithms**: KNN classification and radius-based neighbor search
- **Visualization**: 2D visualization utilities for neighbor relationships

---

## Features

- ✅ Numeric standardization (z-score normalization)
- ✅ Categorical entity embeddings via PyTorch
- ✅ Multiple distance metrics (Euclidean, Manhattan, Cosine, Minkowski, Mahalanobis)
- ✅ KNN Classifier with uniform and distance-weighted voting
- ✅ Radius-based neighbor search
- ✅ Optional distance caching for large datasets
- ✅ 2D visualization of neighbor relationships

---

## Quickstart

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run tests:**

```bash
# KNN tests
python Topanda/ML/test_knn_improved.py

# Preprocessing tests
python -m Topanda.PreProcessing.test_preprocessing

# Core tests (if available)
pytest Topanda/core/ -q
```

---

## Project Structure

```
Topanda/
├── core/                          # Core metric space utilities
│   ├── metric_space.py            # MetricSpace class (distances, queries)
│   ├── metrics.py                 # Metric factory and metric implementations
│   ├── cache.py                   # DistanceCache LRU cache
│   ├── validators.py              # Input validation utilities
│   └── example_metric_space.py    # Example usage
│
├── PreProcessing/                 # Data preprocessing pipeline
│   ├── pipeline.py                # DataProcessor orchestrator
│   ├── numeric_processor.py       # Numeric standardization
│   ├── categorical_processor.py   # PyTorch entity embeddings
│   ├── validator.py               # Column validation
│   └── test_*.py                  # Preprocessing tests
│
└── ML/                            # Machine learning algorithms
    ├── KNN.py                     # KNN Classifier
    ├── radius_neighbors.py        # Radius-based neighbor search + visualization
    └── test_knn_improved.py       # KNN tests
```

---

## Usage Examples

### 1. MetricSpace: Compute Distances

```python
import pandas as pd
from Topanda.core.metric_space import MetricSpace

# Create metric space from numeric data
df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2]})
space = MetricSpace(df, metric='euclidean', cache_distances=True)

# Get distance between two points
dist = space.distance_between(0, 1)

# Get distances from a new point to all training points
new_point = [1.5, 1.5]
distances = space.distance_to_points(new_point)
```

### 2. Preprocessing: Clean and Standardize Data

```python
from Topanda.PreProcessing.pipeline import DataProcessor

# Auto-detect numeric/categorical columns
processor = DataProcessor(standardize_numeric=True, embed_categorical=False)
X, y = processor.fit_transform(df, target_col='target')

# Manual column specification
processor = DataProcessor(
    numeric_cols=['age', 'income'],
    categorical_cols=['gender'],
    standardize_numeric=True,
    embed_categorical=True,
    embedding_dim=8
)
X, y = processor.fit_transform(df, target_col='target')
```

### 3. KNN Classifier: Predict with Nearest Neighbors

```python
import numpy as np
from Topanda.core.metric_space import MetricSpace
from Topanda.ML.KNN import KNNClassifier

# Create metric space and fit KNN
space = MetricSpace(X_train, metric='euclidean')
knn = KNNClassifier(space, n_neighbors=5, weights='uniform')
knn.fit(y_train)

# Predict for a single new point
prediction = knn.predict(np.array([1.0, 2.0, 3.0]))

# Get classification accuracy on training data
accuracy = knn.score(y_train)
```

### 4. Radius Search: Find Neighbors Within Distance

```python
from Topanda.ML.radius_neighbors import find_neighbors_within_radius, visualize_neighbors

# Find all neighbors within radius 0.5 from a point
point = np.array([1.5, 1.5])
neighbor_indices, distances = find_neighbors_within_radius(
    space, point, radius=0.5
)

# Visualize neighbors (2D data only)
fig = visualize_neighbors(space, point, neighbor_indices, distances, radius=0.5)
import matplotlib.pyplot as plt
plt.show()
```

---

## Core Components

### MetricSpace

**Purpose**: Encapsulates a dataset in a metric space with precomputed or on-demand distances.

**Key Methods**:
- `distance_between(i, j)` — Distance between training points i and j
- `distance_to_points(x_new)` — Distances from a new 1D point to all training points
- `distances` — Property that returns full distance matrix (cached)

**Supported Metrics**:
- `euclidean` — L2 distance
- `manhattan` — L1 distance
- `cosine` — Cosine similarity distance
- `minkowski` — Minkowski distance (configurable p)
- `mahalanobis` — Mahalanobis distance

### KNNClassifier

**Purpose**: k-Nearest Neighbors classification using a MetricSpace.

**Key Parameters**:
- `n_neighbors` (int) — Number of neighbors to consider
- `weights` (str) — `'uniform'` (equal votes) or `'distance'` (inverse distance weighting)

**Key Methods**:
- `fit(labels)` — Store class labels
- `predict(point)` — Predict label for a single new point (1D array)
- `score(true_labels)` — Classification accuracy on training data

### radius_neighbors

**Purpose**: Find neighbors within a fixed radius and visualize them.

**Key Functions**:
- `find_neighbors_within_radius(metric_space, point, radius)` — Returns sorted neighbor indices and distances
- `visualize_neighbors(metric_space, point, neighbor_indices, distances, radius)` — 2D scatter plot with radius circle

**Requirements**:
- Only supports 2D data for visualization
- Raises `ValueError` if data has ≠ 2 features

### DataProcessor (Preprocessing)

**Purpose**: Standardize numeric features and embed categorical features.

**Key Parameters**:
- `numeric_cols` (list, optional) — Columns to standardize; auto-detected if None
- `categorical_cols` (list, optional) — Columns to embed; auto-detected if None
- `standardize_numeric` (bool) — Apply z-score normalization
- `embed_categorical` (bool) — Use PyTorch entity embeddings
- `embedding_dim` (int) — Dimension of categorical embeddings

**Key Methods**:
- `fit_transform(df, target_col)` — Fit and transform; returns (X, y)
- `transform(df)` — Transform using fitted parameters

---

## Dependencies

```
numpy>=1.21
pandas>=1.3
scikit-learn>=0.24
torch>=1.9
scipy>=1.0
pytest
matplotlib
```

Install all:

```bash
pip install -r requirements.txt
```

---

## Testing

Run the test suite:

```bash
# KNN tests (comprehensive)
python Topanda/ML/test_knn_improved.py

# Preprocessing tests
python -m Topanda.PreProcessing.test_preprocessing

# All tests with pytest
pytest Topanda/ -v
```

---

## Notes & Troubleshooting

- **PyTorch**: Required only for `CategoricalProcessor` embeddings. If disabled, PyTorch is not needed.
- **MetricSpace validation**: Only numeric data is accepted. Use `DataProcessor` to clean/standardize data first.
- **KNN prediction**: Only supports single-point prediction (1D array). Pass one point at a time.
- **Visualization**: `radius_neighbors.visualize_neighbors()` requires exactly 2 features. Use preprocessing or feature selection to reduce dimensions.
- **Distance caching**: Enabled by default in `MetricSpace`. Useful for large datasets with many queries; disable with `cache_distances=False`.

---

## Contributing

Contributions welcome! Open issues or pull requests with:
- Clear problem/feature description
- Tests for new functionality
- Updates to this README if relevant

---

## License

MIT License — See LICENSE file for details



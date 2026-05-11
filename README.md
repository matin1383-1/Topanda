# Topanda

A lightweight toolkit for tabular data preprocessing, metric-space machine learning, and nearest-neighbor algorithms.

## Features

✅ **Data Preprocessing**: Numeric standardization & categorical embeddings (PyTorch)  
✅ **MetricSpace**: Unified distance computation with multiple metrics  
✅ **ML Algorithms**: KNN classification, radius-based search, triplet learning  
✅ **Distance Metrics**: Euclidean, Manhattan, Cosine, Minkowski, Mahalanobis  
✅ **Performance**: Optional LRU distance caching for large datasets  
✅ **Visualization**: 2D neighbor relationship plots  

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: numpy, pandas, scikit-learn, torch, scipy, matplotlib

---

## Quick Start

### 1. Preprocess Data

```python
from Topanda import DataProcessor
import pandas as pd

# Load your data
df = pd.DataFrame({
    'age': [25, 30, 35],
    'salary': [50000, 60000, 75000],
    'city': ['NYC', 'LA', 'NYC'],
    'target': [0, 1, 1]
})

# Automatically standardize numeric & embed categorical features
processor = DataProcessor(embed_categorical=True, embedding_dim=10)
X_processed, y = processor.fit_transform(df, target_col='target')
```

### 2. Build Metric Space

```python
from Topanda import MetricSpace

# Create metric space with your processed data
ms = MetricSpace(X_processed, metric='euclidean', cache_distances=True)

# Query distances
distance = ms.distance_between(0, 1)
distances_to_new_point = ms.distance_to_points([25, 50000, ...])
```

### 3. KNN Classification

```python
from Topanda import KNNClassifier

# Train KNN classifier
knn = KNNClassifier(ms, n_neighbors=5, weights='uniform')
knn.fit(y)

# Predict on new point
prediction = knn.predict([26, 52000, ...])
accuracy = knn.score(y)
```

### 4. Radius-based Neighbor Search

```python
from Topanda import find_neighbors_within_radius

# Find all neighbors within radius
neighbor_indices, distances = find_neighbors_within_radius(ms, point=0, radius=5.0)

# Visualize (2D data only)
visualize_neighbors(ms, point=0, neighbor_indices=neighbor_indices, 
                   distances=distances, radius=5.0)
```

### 5. Deep Metric Learning

```python
from Topanda import TripletLearner

# Learn optimized embeddings
learner = TripletLearner(embedding_dim=16, margin=1.0, epochs=20)
learner.fit(X_train, y_train)
embeddings = learner.transform(X_test)
```

---

## Package Structure

```
Topanda/
├── core/                 # MetricSpace, distance metrics, caching
├── PreProcessing/        # Data standardization & embeddings
├── ML/                   # KNN, radius search, visualization
└── DeepMetricLearning/   # Triplet loss learning
```

---

## Import Styles

**From main package:**
```python
from Topanda import MetricSpace, KNNClassifier, DataProcessor, TripletLearner
```

**From submodules:**
```python
from Topanda.core import MetricSpace, MetricFactory
from Topanda.ML import KNNClassifier, find_neighbors_within_radius
from Topanda.PreProcessing import DataProcessor
from Topanda.DeepMetricLearning import TripletLearner
```

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



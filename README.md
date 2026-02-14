# Topanda

Topanda is a lightweight toolkit for tabular data preprocessing, distance-based (metric-space) machine learning workflows, and nearest-neighbor algorithms. It provides:

- **Preprocessing Pipeline**: Numeric standardization and categorical entity embeddings
- **MetricSpace**: A unified interface for computing and querying distances in metric spaces
- **ML Algorithms**: KNN classification and radius-based neighbor search
- **Deep Metric Learning**: Triplet learning for optimized embedding spaces
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
- ✅ Deep metric learning approaches

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
|
|
|
└── Deep metric learning
```



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



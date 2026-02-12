# Topanda

Topanda is a lightweight toolkit for tabular data preprocessing and distance-based (metric-space) workflows. It contains a small preprocessing pipeline (numeric standardization and categorical entity embeddings) and core utilities for building and querying metric spaces over numeric data.

This README gives a concise overview, quickstart steps, project layout, and how to run the example/tests.

---

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the simple example (from project root):

```bash
python example_simple.py
```

3. Run tests:

```bash
# PreProcessing tests
python -m Topanda.PreProcessing.test_preprocessing

# Core tests (pytest recommended)
pytest Topanda/core/test_core.py -q
```

---

## Project layout

- `Topanda/PreProcessing/` — preprocessing pipeline and tests
  - `pipeline.py` — `DataProcessor` orchestrator
  - `numeric_processor.py` — numeric standardization
  - `categorical_processor.py` — PyTorch entity embeddings
  - `validator.py` — input validation and column resolution
  - test files: `test_preprocessing.py`, `test_sklearn_datasets.py`

- `Topanda/core/` — metric-space utilities
  - `metric_space.py` — `MetricSpace` class (distance matrix, queries)
  - `metrics.py` — metric factory and supported metrics
  - `cache.py` — `DistanceCache` LRU cache for matrices
  - `test_core.py` — core tests

- `example_simple.py` — minimal runnable example that demonstrates dataset → preprocessing → metric space → queries
- `requirements.txt` — pinned runtime/test dependencies

---

## Preprocessing (short)

- Numeric: standardization (z-score) via `NumericProcessor`.
- Categorical: supervised entity embeddings via `CategoricalProcessor` (requires PyTorch).
- `DataProcessor` combines both and accepts optional user-provided `numeric_cols` and `categorical_cols`. If neither is provided, `identify_column_types` is used to auto-detect types.

Usage (brief):
```python
from Topanda.PreProcessing.pipeline import DataProcessor
processor = DataProcessor(standardize_numeric=True, embed_categorical=False)
X, y = processor.fit_transform(df, target_col='target')
```

---

## Core: MetricSpace (short)

`MetricSpace` builds a numerical metric space from a pandas DataFrame (numeric columns required). It validates input, constructs pairwise distances using a chosen metric, and provides methods for common queries (distance_between, distance_to_points). `DistanceCache` optionally caches matrices to avoid recomputation.

Quick example:
```python
from Topanda.core.metric_space import MetricSpace
space = MetricSpace(X, metric='euclidean', cache_distances=True)
print(space.distances.shape)
print(space.distance_between(0, 1))
```

Supported metrics include `euclidean`, `manhattan`, `cosine`, `minkowski`, and `mahalanobis` via `MetricFactory`.

---

## Examples & Tests

- `example_simple.py` — minimal script that runs the Iris dataset through the preprocessing pipeline, constructs a `MetricSpace`, and demonstrates distance queries.
- `Topanda/PreProcessing/test_preprocessing.py` and `test_sklearn_datasets.py` — synthetic and scikit-learn dataset tests for preprocessing.
- `Topanda/core/test_core.py` — tests for MetricSpace, MetricFactory, DistanceCache, and validators.

Run tests with `pytest` for best output.

---

## Dependencies

Minimum recommended:

```
numpy>=1.21
pandas>=1.3
scikit-learn>=0.24
torch>=1.9
pytest
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## Notes & Troubleshooting

- PyTorch is required for categorical entity embeddings — omission will raise an ImportError when using `CategoricalProcessor`.
- Use `DataProcessor` to clean and standardize data before constructing a `MetricSpace`.

---

## Contributing

Contributions welcome — open issues or pull requests. Keep changes focused and include tests for new behavior.

---

License: MIT (or your preferred license)

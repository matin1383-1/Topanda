<p align="center">
    <img src="Topanda.png" alt="Topanda Logo" width="350" >
</p>

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
python Tests/Test.py
```

---

## Load models from the main directory

If you are running from the repository root, import from the `Topanda` package directly:

```python
from Topanda import KNNClassifier, DataProcessor, TripletLearner

processor = DataProcessor(...)
model = KNNClassifier(...)
learner = TripletLearner(...)
```

This works when your current working directory is the project root and `Topanda/` is on the Python path.

## Hard-negative and geometry-preserving pipelines

Topanda also exposes three optional deep metric-learning pipelines:

| Pipeline | What it trains | Negative selection |
|---|---|---|
| `HardNegativeTopanda` | A Topanda encoder from scratch | Closest different-class candidate |
| `GPTopanda` | A residual correction over a frozen fitted Topanda model | Random different-class candidate |
| `GPHTopanda` | A residual correction over a frozen fitted Topanda model | Closest different-class candidate |

GP and GP-H operate directly on the representation learned by the original
`TripletLearner`. Their residual layer starts as an exact identity mapping, and
their preservation loss penalizes unnecessary movement away from the original
embedding. The original model remains frozen and unchanged.

```python
from sklearn.neighbors import KNeighborsClassifier
from Topanda import GPHTopanda, TrainingConfig, TripletLearner

# X_train and X_test must use preprocessing fitted on training data only.
base = TripletLearner(embedding_dim=16).fit(X_train, y_train)
config = TrainingConfig(
    strategy="hard",
    preservation_lambda=1.0,
    epochs=20,
)
pipeline = GPHTopanda(base, config).fit(X_train, y_train)

classifier = KNeighborsClassifier(n_neighbors=5).fit(
    pipeline.transform(X_train), y_train
)
predictions = classifier.predict(pipeline.transform(X_test))
```

The strategy is fixed by the pipeline class: `GPTopanda` uses random negatives,
while `GPHTopanda` uses hard negatives. Hard mining searches a balanced candidate
pool in the current embedding. It does not require labels during `transform()`.
Every class must contain at least two training observations; batch size and
candidate-pool size must be at least the number of classes.

Run the extension checks and example from the repository root:

```bash
python -m unittest discover -s Tests -p "test_extensions.py" -v
python Examples/compare_extensions.py
```

---

## Triplet Learning Results

The test suite (`Tests/Test.py`) demonstrates the effectiveness of triplet learning for metric space optimization. Triplet learning improves KNN classification accuracy by learning embeddings where similar samples are closer together.
in this method we don't use predefined metrics , instead we build a suitable and customized metric for any labeled dataset . 

### Test Results Summary

| Dataset | Original KNN Accuracy | Embedded KNN Accuracy | Improvement |
|---------|----------------------|----------------------|-------------|
| Iris | 96.67% | 97.33% | +0.67% |
| Wine | 97.75% | 99.44% | +1.69% |
| Digits | 98.11% | 99.72% | +1.61% |
| Breast Cancer | 97.01% | 99.12% | +2.11% |

### Visualization Examples

#### Iris Dataset
![Iris Triplet Analysis](Results/iris_triplet_analysis.png)
*KNN accuracy comparison, t-SNE visualizations, and distance distributions for the Iris dataset.*

#### Wine Dataset
![Wine Triplet Analysis](Results/wine_triplet_analysis.png)
*KNN accuracy comparison, t-SNE visualizations, and distance distributions for the Wine dataset.*

#### Digits Dataset
![Digits Triplet Analysis](Results/digits_triplet_analysis.png)
*KNN accuracy comparison, t-SNE visualizations, and distance distributions for the Digits dataset.*

#### Breast Cancer Dataset
![Breast Cancer Triplet Analysis](Results/breast_cancer_triplet_analysis.png)
*KNN accuracy comparison, t-SNE visualizations, and distance distributions for the Breast Cancer dataset.*

#### CoverType Dataset
![CoverType Triplet Analysis](Results/covertype_knn_accuracy.png)



Each plot shows:
- **KNN Accuracy Comparison**: Performance improvement after triplet learning
- **t-SNE Visualizations**: 2D projections of original vs embedded spaces
- **Distance Distributions**: Within-class vs between-class distances
- **Separation Ratio**: Quantitative measure of class separability

---

## Project Structure

```
Topanda/
├── core/                          # Core metric space utilities
│   ├── metric_space.py            # MetricSpace class (distances, queries)
│   ├── metrics.py                 # Metric factory and metric 
│   ├── cache.py                   # DistanceCache LRU cache
│   ├── validators.py              # Input validation utilities
│
│
├── PreProcessing/                 # Data preprocessing pipeline
│   ├── pipeline.py                # DataProcessor orchestrator
│   ├── numeric_processor.py       # Numeric standardization
│   ├── categorical_processor.py   # PyTorch entity embeddings
│   ├── validator.py               # Column validation
│
│
└── ML/                            # Machine learning algorithms
    ├── KNN.py                     # KNN Classifier
    ├── radius_neighbors.py        # Radius-based neighbor search + visualization
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



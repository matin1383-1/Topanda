"""
Topanda: A lightweight toolkit for tabular data preprocessing, 
distance-based machine learning workflows, and nearest-neighbor algorithms.

Main Components:
  - core: MetricSpace and distance metrics
  - preprocessing: Data standardization and categorical embeddings
  - ml: KNN classification and radius-based neighbor search
  - deep_metric_learning: Triplet learning for optimized embeddings
"""

__version__ = "0.1.0"

# Core module
from .core import (
    MetricSpace,
    MetricFactory,
)

# Preprocessing module
from .PreProcessing import (
    DataProcessor,
    validate_dataframe,
)

# ML module
from .ML import (
    KNNClassifier,
    find_neighbors_within_radius,
    visualize_neighbors,
)

# Deep Metric Learning module
from .DeepMetricLearning import (
    TripletLearner,
)

__all__ = [
    # Core
    "MetricSpace",
    "MetricFactory",
    # Preprocessing
    "DataProcessor",
    "validate_dataframe",
    # ML
    "KNNClassifier",
    "find_neighbors_within_radius",
    "visualize_neighbors",
    # Deep Metric Learning
    "TripletLearner",
]

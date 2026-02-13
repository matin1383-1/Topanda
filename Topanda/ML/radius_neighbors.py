
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple


def find_neighbors_within_radius(
    metric_space,
    point: Union[int, np.ndarray],
    radius: float
):
    if isinstance(point, (int, np.integer)):
        if point < 0 or point >= metric_space.n_samples:
            raise ValueError(
                f"Point index {point} out of bounds "
                f"for dataset with {metric_space.n_samples} samples"
            )
        distances = metric_space.distances[point]
    elif isinstance(point, np.ndarray):
        if point.ndim != 1:
            raise ValueError(f"Point must be 1D array, got shape {point.shape}")
        if point.shape[0] != metric_space.n_features:
            raise ValueError(
                f"Point has {point.shape[0]} features, "
                f"expected {metric_space.n_features}"
            )
        distances = metric_space.distance_to_points(point)
    
    else:
        raise TypeError(
            f"Point must be int or numpy array, got {type(point)}"
        )
    
    neighbor_mask = distances <= radius
    neighbor_indices = np.where(neighbor_mask)[0]
    neighbor_distances = distances[neighbor_mask]
    
    sort_order = np.argsort(neighbor_distances)
    
    return neighbor_indices[sort_order], neighbor_distances[sort_order]


def visualize_neighbors(
    metric_space,
    point: Union[int, np.ndarray],
    neighbor_indices: np.ndarray,
    distances: np.ndarray,
    radius: float,
    title: str = "Radius Neighbors Visualization"
):
    if metric_space.n_features != 2:
        raise ValueError(
            f"Visualization only supports 2D data. Got {metric_space.n_features} features."
        )    
    if isinstance(point, int):
        point_coords = metric_space.X[point]
    else:
        point_coords = point
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(metric_space.X[:, 0], metric_space.X[:, 1], c='lightgray', s=30, alpha=0.5, label='Other points')
    if len(neighbor_indices) > 0:
        scatter = ax.scatter(
            metric_space.X[neighbor_indices, 0],
            metric_space.X[neighbor_indices, 1],
            c=distances,
            cmap='viridis_r',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=1,
            label='Neighbors'
        )
        plt.colorbar(scatter, ax=ax, label='Distance')
    
    ax.scatter(
        point_coords[0],
        point_coords[1],
        c='red',
        s=300,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label='Query point',
        zorder=10
    )
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'{title}\n{len(neighbor_indices)} neighbors found within radius {radius:.3f}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    return fig

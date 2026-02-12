
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from metric_space import MetricSpace


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)


space = MetricSpace(data=X, metric="euclidean", cache_distances=True)
print(space)
d01 = space.distance_between(0, 1)
print(f"Distance 0 <-> 1: {d01:.4f}")

print(f"Distance matrix shape: {space.distances.shape}")
new_point = np.zeros(X.shape[1])
d_new = space.distance_to_points(new_point)
print(f"Distances from new (first 5): {d_new[:5]}")


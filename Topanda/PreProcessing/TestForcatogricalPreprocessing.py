import numpy as np
import pandas as pd
from categorical_processor import CategoricalProcessor

X = pd.DataFrame({
    'color': ['red', 'blue', 'red', 'green', 'blue'],
    'size': ['S', 'M', 'L', 'M', 'S']
})

y = np.array([10, 20, 12, 15, 18])

print("Input data:")
print(X)
print(f"\nTarget: {y}")

processor = CategoricalProcessor(embedding_dim=1, epochs=10)
X_embedded = processor.fit_transform(X, y)

print(f"\nOutput shape: {X_embedded.shape}")
print("\nEmbedded data:")
print(X_embedded)

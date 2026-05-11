import pandas as pd
import numpy as np

# --- sample data ---
X = pd.DataFrame({
    "height": [170, 180, np.nan, 160],
    "weight": [65, np.nan, 80, 55]
})

print("Original data:")
print(X)


from .numeric_processor import NumericProcessor  

processor = NumericProcessor(standardize=True)

X_new = processor.fit_transform(X)


print("\nTransformed data:")
print(X_new)



import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

from pipeline import DataProcessor


def test_sklearn_iris_dataset():
    
    # Load iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    

    print(f"Columns: {list(df.columns)}")
    print(df.head())
    print(f"\nData types:\n{df.dtypes}")
    
    # Initialize processor with auto-detection
    processor = DataProcessor(
        standardize_numeric=True,
        embed_categorical=False,  # No categorical columns in iris
        embedding_dim=5
    )
    
    # Fit and transform
    X_processed, y = processor.fit_transform(df, target_col='target')
    
    print(f"\nAuto-detected numeric columns: {processor._numeric_cols}")
    print(f"Auto-detected categorical columns: {processor._categorical_cols}")
    print(f"\nProcessed features shape: {X_processed.shape}")
    print(f"Processed features (first 3 rows):\n{X_processed.head(3)}")

def test_sklearn_wine_dataset():
    
    # Load wine dataset
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    
    print(df.head())
    
    # Initialize processor with standardization
    processor = DataProcessor(
        standardize_numeric=True,
        embed_categorical=False,
        embedding_dim=5
    )
    
    # Fit and transform
    X_processed, y = processor.fit_transform(df, target_col='target')
    
    print(f"\nProcessed features shape: {X_processed.shape}")
    print(f"Processed features (first 3 rows):\n{X_processed.head(3)}")
    print(f"Target values (unique): {np.unique(y)}")
    print(f"Target distribution: {np.bincount(y)}")



test_sklearn_iris_dataset()
test_sklearn_wine_dataset()
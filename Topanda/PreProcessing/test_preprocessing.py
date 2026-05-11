
import numpy as np
import pandas as pd

from .pipeline import DataProcessor


def test_basic_preprocessing_auto_detection():
    
    data = {
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 75000, 80000, 95000],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'country': ['USA', 'UK', 'USA', 'Canada', 'UK'],
        'target': [0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    print(df)
    
    processor = DataProcessor(
        standardize_numeric=True,
        embed_categorical=True,
        embedding_dim=5
    )
    
    X_processed, y = processor.fit_transform(df, target_col='target')
    
    print(f"\nAuto-detected numeric columns: {processor._numeric_cols}")
    print(f"Auto-detected categorical columns: {processor._categorical_cols}")
    print(f"Processed features:\n{X_processed}")
    print(f"\nTarget: {y}")


def test_preprocessing_manual_columns():

    data = {
        'age': [25, 30, 35, 40, 45],
        'score': [88, 92, 76, 85, 90],
        'level': [1, 2, 1, 3, 2],
        'department': ['Sales', 'IT', 'HR', 'IT', 'Sales'],
        'result': [0, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    print(df)
    
    processor = DataProcessor(
        standardize_numeric=True,
        embed_categorical=True,
        embedding_dim=8,
        numeric_cols=['age', 'score'],  # Manually specify numeric
        categorical_cols=['department']  # Manually specify categorical
    )
    
    X_processed, y = processor.fit_transform(df, target_col='result')
    
    print(f"\nUser-specified numeric columns: {processor._numeric_cols}")
    print(f"User-specified categorical columns: {processor._categorical_cols}")
    print(f"\nProcessed features shape: {X_processed.shape}")
    print(f"Processed features:\n{X_processed}")
    print(f"\nTarget: {y}")



test_basic_preprocessing_auto_detection()
test_preprocessing_manual_columns()
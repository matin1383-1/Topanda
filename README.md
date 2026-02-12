# Topanda Preprocessing Module

## Overview

The Topanda preprocessing module provides a comprehensive pipeline for preparing data before machine learning tasks. It handles both **numeric** and **categorical** data types with automatic standardization and entity embeddings.

## Key Features

- **Flexible Column Type Specification**: Users can manually specify numeric and categorical columns, or rely on automatic detection
- **Numeric Processing**: Standardization (z-score normalization) using `StandardScaler`
- **Categorical Processing**: Entity embeddings via supervised learning with PyTorch
- **Modular Architecture**: Separate processors for numeric and categorical data, combined in a unified pipeline
- **PyTorch Required**: Always uses entity embeddings for categorical data (no random fallback)

## Project Structure

```
PreProcessing/
├── __init__.py
├── categorical_processor.py    # Entity embedding processor for categorical data
├── numeric_processor.py         # Standardization processor for numeric data
├── validator.py                 # Input validation and column type resolution
├── pipeline.py                  # Main orchestrator (DataProcessor class)
├── test_preprocessing.py        # Basic synthetic data tests
└── test_sklearn_datasets.py     # Tests with real scikit-learn datasets
```

## Module Details

### 1. **validator.py**

Handles input validation and column type resolution.

**Key Functions:**

- `validate_dataframe(df)` — Ensures input is a non-empty DataFrame
- `check_target_column(df, target_col)` — Verifies target column exists
- `identify_column_types(df, exclude_cols)` — Auto-detects numeric vs categorical columns
- `resolve_column_types(df, numeric_cols, categorical_cols, exclude_cols)` — Uses user-provided column specifications

**Usage:**
```python
from validator import identify_column_types, resolve_column_types

# Auto-detect column types
numeric, categorical = identify_column_types(df)

# Use user-specified columns
numeric, categorical = resolve_column_types(
    df,
    numeric_cols=['age', 'income'],
    categorical_cols=['gender', 'country']
)
```

### 2. **numeric_processor.py**

Processes numeric columns using standardization.

**Key Class:** `NumericProcessor`

- **Standardization**: Applies z-score normalization (StandardScaler)
- **Methods**:
  - `fit(X)` — Learn standardization parameters
  - `transform(X)` — Apply standardization
  - `fit_transform(X)` — Fit and transform in one step

**Usage:**
```python
from numeric_processor import NumericProcessor

processor = NumericProcessor(standardize=True)
X_standardized = processor.fit_transform(X_numeric)
```

### 3. **categorical_processor.py**

Processes categorical columns using PyTorch entity embeddings.

**Key Class:** `CategoricalProcessor`

- **Entity Embeddings**: Supervised learning with PyTorch neural networks
- **Architecture**: 
  - Embedding layers per categorical column
  - Concatenated embeddings fed through a linear layer
  - Trained using MSELoss on continuous targets
- **Methods**:
  - `fit(X, y)` — Learn embeddings from categorical features and target
  - `transform(X)` — Map categories to their learned embeddings
  - `fit_transform(X, y)` — Fit and transform in one step

**Requirements:** PyTorch must be installed; ImportError raised if unavailable

**Usage:**
```python
from categorical_processor import CategoricalProcessor

processor = CategoricalProcessor(embedding_dim=10, epochs=10)
X_embeddings = processor.fit_transform(X_categorical, y)
```

### 4. **pipeline.py**

Main orchestrator combining numeric and categorical processing.

**Key Class:** `DataProcessor`

**Initialization:**
```python
from pipeline import DataProcessor

processor = DataProcessor(
    standardize_numeric=True,           # Apply z-score normalization
    embed_categorical=True,              # Use entity embeddings
    embedding_dim=10,                    # Embedding vector dimension
    numeric_cols=['age', 'income'],     # (Optional) specify numeric columns
    categorical_cols=['gender']          # (Optional) specify categorical columns
)
```

**Methods:**

- `fit_transform(df, target_col)` — Fit on data and return processed features + target
  - Returns: `(X_processed, y)` tuple
  - If `target_col` not provided, returns `(X_processed, None)`

- `transform(df)` — Transform new data using fitted processors
  - Returns: Processed DataFrame with same structure as training data

**Workflow:**
```python
# Initialize
processor = DataProcessor(embed_categorical=True)

# Fit on training data
X_train, y_train = processor.fit_transform(df_train, target_col='target')

# Transform test data
X_test = processor.transform(df_test)
```

**Column Type Resolution:**
- If neither `numeric_cols` nor `categorical_cols` provided → **Auto-detect** using `identify_column_types`
- If both provided → **Use user-specified** via `resolve_column_types`
- Partial specification defaults to empty list (user provides explicit columns only)

## Workflow

### Step 1: Prepare Data
```python
import pandas as pd
from pipeline import DataProcessor

# Load or create DataFrame
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 75000, 80000],
    'gender': ['M', 'F', 'M', 'F'],
    'target': [0, 1, 1, 0]
})
```

### Step 2: Initialize Processor
```python
# Option A: Auto-detect columns
processor = DataProcessor(standardize_numeric=True, embed_categorical=True)

# Option B: Specify columns manually
processor = DataProcessor(
    standardize_numeric=True,
    embed_categorical=True,
    numeric_cols=['age', 'income'],
    categorical_cols=['gender']
)
```

### Step 3: Fit and Transform Training Data
```python
X_train, y_train = processor.fit_transform(df, target_col='target')
print(f"Processed features shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")
```

### Step 4: Transform New Data
```python
df_new = pd.DataFrame({
    'age': [28, 38],
    'income': [55000, 77000],
    'gender': ['F', 'M']
})

X_new = processor.transform(df_new)
```

## Example: Real Dataset (Iris)

```python
from sklearn.datasets import load_iris
from pipeline import DataProcessor

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Process (all numeric, no categorical)
processor = DataProcessor(standardize_numeric=True, embed_categorical=False)
X_processed, y = processor.fit_transform(df, target_col='target')

print(f"Shape: {X_processed.shape}")
print(f"Mean: {X_processed.mean():.6f}")  # Should be ~0
print(f"Std: {X_processed.std():.6f}")    # Should be ~1
```

## Testing

### Run Basic Tests (Synthetic Data)
```powershell
cd c:\Users\Parsian-PC\Desktop\TP\Topanda\PreProcessing
python test_preprocessing.py
```

Tests include:
1. Auto-detection of column types
2. Manual column specification
3. Partial column specification (numeric only)
4. Transform new data
5. Numeric-only processing

### Run Real Dataset Tests
```powershell
python test_sklearn_datasets.py
```

Tests include:
1. Iris dataset
2. Wine dataset
3. Breast Cancer dataset
4. Iris with manual columns
5. Wine standardization verification

## Dependencies

- **numpy** ≥ 1.21.0 — Numerical computing
- **pandas** ≥ 1.3.0 — Data manipulation
- **torch** ≥ 1.9.0 — **Required for categorical embeddings**
- **scikit-learn** ≥ 0.24.0 — Preprocessing utilities and test datasets

Install with:
```bash
pip install -r requirements.txt
```

## Error Handling

### Common Errors

1. **ImportError: PyTorch not available**
   ```
   Solution: pip install torch
   ```

2. **ValueError: Invalid numeric columns**
   ```python
   # Column name doesn't exist in DataFrame
   processor = DataProcessor(numeric_cols=['age', 'invalid_col'])
   # Fix: Use correct column names
   processor = DataProcessor(numeric_cols=['age', 'income'])
   ```

3. **ValueError: Columns cannot be both numeric and categorical**
   ```python
   # Same column specified in both lists
   processor = DataProcessor(
       numeric_cols=['age'],
       categorical_cols=['age']  # Error!
   )
   ```

4. **ValueError: Embeddings require a target column**
   ```python
   # Target needed for categorical embeddings
   X, y = processor.fit_transform(df)  # Missing target_col
   # Fix:
   X, y = processor.fit_transform(df, target_col='target')
   ```

## Design Principles

1. **Separation of Concerns** — Each processor handles its data type independently
2. **PyTorch Always** — No random fallback; entity embeddings are mandatory for categorical data
3. **Flexible Column Specification** — Users have full control over column assignments
4. **Validation First** — Comprehensive input checking before processing
5. **Reusability** — Processors can be used standalone or via the unified pipeline

## Future Enhancements

- Support for missing value imputation strategies
- Categorical encoding beyond embeddings (one-hot, ordinal)
- Feature scaling options beyond standardization
- Cross-validation utilities
- Inverse transform capabilities

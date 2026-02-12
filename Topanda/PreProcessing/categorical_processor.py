"""Categorical (nominal) data processing with entity embeddings for topanda."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn

class CategoricalProcessor:
    def __init__(self, embedding_dim: int = 10, epochs: int = 10) :
        """
        Args:
            embedding_dim: Dimension of each category embedding vector.
            epochs: Number of training epochs when using PyTorch.
        """
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self._embeddings: Dict[str, Dict[Any, np.ndarray]] = {}
        self._columns: List[str] = []
        self._fitted = False

    def _fit_neural(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._columns = list(X.columns)
        n_cols = len(self._columns)
        if n_cols == 0:
            self._fitted = True
            return

        # Integer encode each column and collect embedding sizes
        encoded_list: List[np.ndarray] = []
        categories_per_col: List[pd.Index] = []
        embedding_dims_list: List[int] = []

        for col in self._columns:
            cat = pd.Categorical(X[col])
            codes = np.array(cat.codes, dtype=np.int64)
            # -1 (NaN) -> 0, 0..n-1 -> 1..n for padding_idx=0
            codes = np.clip(codes + 1, 0, len(cat.categories))
            encoded_list.append(codes)
            categories_per_col.append(cat.categories)
            n_cats = len(cat.categories) + 1  # +1 for padding
            embedding_dims_list.append(n_cats)

        # Stack encoded inputs: (batch, n_cols)
        encoded = np.stack(encoded_list, axis=1)
        x_tensor = torch.tensor(encoded, dtype=torch.long)
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        y_tensor = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1)

        class EmbeddingModel(nn.Module):
            def __init__(
                self,
                embedding_dims: List[int],
                embedding_dim: int,
                n_cols: int,
            ) -> None:
                super().__init__()
                self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(num_embeddings=d, embedding_dim=embedding_dim, padding_idx=0)
                        for d in embedding_dims
                    ]
                )
                self.linear = nn.Linear(n_cols * embedding_dim, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (batch, n_cols)
                outs = []
                for i in range(x.size(1)):
                    out = self.embeddings[i](x[:, i])
                    outs.append(out)
                out = torch.cat(outs, dim=1)
                return self.linear(out)

        model = EmbeddingModel(embedding_dims_list, self.embedding_dim, n_cols)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = model(x_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        # Extract embedding weights into self._embeddings
        self._embeddings = {}
        for col_idx, col in enumerate(self._columns):
            weight = model.embeddings[col_idx].weight.detach().numpy()
            categories = categories_per_col[col_idx]
            self._embeddings[col] = {}
            for i, cat_val in enumerate(categories):
                self._embeddings[col][cat_val] = weight[i + 1].copy()

        self._fitted = True

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CategoricalProcessor":
        """
        Fit embeddings using PyTorch.
        Args:
            X: DataFrame of categorical columns.
            y: Target array for supervise.

        """
        if X.shape[1] == 0:
            self._columns = []
            self._embeddings = {}
            self._fitted = True
            return self
        self._fit_neural(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map each categorical value to its embedding; unseen categories get zero vector.
        Output columns are named {col}emb0, {col}emb1, ...
        """
        if not self._fitted:
            raise ValueError("error")
        if not self._columns:
            return pd.DataFrame(index=X.index)

        zero = np.zeros(self.embedding_dim, dtype=np.float64)
        parts: List[pd.DataFrame] = []
        for col in self._columns:
            if col not in X.columns:
                continue
            col_emb = self._embeddings.get(col, {})
            rows = []
            for val in X[col].values:
                vec = col_emb.get(val, zero)
                if isinstance(vec, np.ndarray):
                    rows.append(vec)
                else:
                    rows.append(np.array(vec, dtype=np.float64))
            arr = np.array(rows)
            part = pd.DataFrame(
                arr,
                index=X.index,
                columns=[f"{col}emb{i}" for i in range(self.embedding_dim)],
            )
            parts.append(part)
        if not parts:
            return pd.DataFrame(index=X.index)
        return pd.concat(parts, axis=1)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        Fit embeddings on (X, y) and transform X to embedding DataFrame.

        Args:
            X: DataFrame of categorical columns.
            y: Target array.
        Returns:
            DataFrame of concatenated embedding columns.
        """
        return self.fit(X, y).transform(X)

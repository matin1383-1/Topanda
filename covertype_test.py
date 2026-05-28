import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from Topanda.PreProcessing.pipeline import DataProcessor
from Topanda.core.metric_space import MetricSpace
from Topanda.DeepMetricLearning.Triplet import TripletLearner
from Topanda.ML.KNN import KNNClassifier

# Ensure package imports work when running from repository root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def load_covertype_dataset(sample_size: int = 1000, random_state: int = 42):
    """Load the Covertype dataset and optionally subsample it for faster experiments."""
    data = fetch_covtype()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    if sample_size is not None and sample_size < len(X):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[indices].reset_index(drop=True)
        y = y[indices]

    return X, y


def build_metric_space(X: pd.DataFrame) -> MetricSpace:
    """Build a Euclidean metric space from processed numeric data."""
    return MetricSpace(X, metric='euclidean', cache_distances=True)


def evaluate_knn(metric_space: MetricSpace, labels: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, k_values):
    """Evaluate KNN on a held-out set for several k values."""
    results = {}
    all_predictions = {}
    n_test = len(X_test)
    report_step = max(1, n_test // 10)

    for k in k_values:
        print(f"\nStarting KNN evaluation for k={k} with {n_test} test samples...")
        knn = KNNClassifier(metric_space=metric_space, n_neighbors=k, weights='uniform')
        knn.fit(labels)

        predictions = []
        for idx, x in enumerate(X_test):
            predictions.append(knn.predict(x))
            if (idx + 1) % report_step == 0 or idx == n_test - 1:
                print(
                    f"  K={k}: processed {idx + 1}/{n_test} test samples"
                )
        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y_test)
        results[k] = float(accuracy)
        all_predictions[k] = predictions
        print(f"Completed k={k}: accuracy={accuracy:.4f}")
    return results


def main():
    print("Loading Covertype dataset...")
    X, y = load_covertype_dataset(sample_size=30000)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features.")
    print(f"Classes: {np.unique(y).shape[0]}")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print(f"Train samples: {len(X_train_df)}, test samples: {len(X_test_df)}")

    processor = DataProcessor(standardize_numeric=True, embed_categorical=False)
    X_train_processed, _ = processor.fit_transform(
        pd.concat([X_train_df.reset_index(drop=True), pd.DataFrame({'target': y_train})], axis=1),
        target_col='target'
    )
    X_test_processed = processor.transform(X_test_df.reset_index(drop=True))

    print("Building original Euclidean metric space...")
    ms_train_orig = build_metric_space(X_train_processed)

    print("Training triplet learner to build a custom embedding metric...")
    learner = TripletLearner(embedding_dim=32, epochs=20, batch_size=128, lr=1e-3)
    ms_train_emb = learner.fit_transform_metric_space(ms_train_orig, y_train)

    X_test_embedded = learner.transform(X_test_processed)

    print("Evaluating KNN on original training space...")
    k_values = [5]
    original_results = evaluate_knn(
        metric_space=ms_train_orig,
        labels=y_train,
        X_test=X_test_processed.values,
        y_test=y_test,
        k_values=k_values,
    )

    print("\nEvaluating KNN on embedded training space...")
    embedded_results = evaluate_knn(
        metric_space=ms_train_emb,
        labels=y_train,
        X_test=X_test_embedded.values,
        y_test=y_test,
        k_values=k_values,
    )

    # --- Visualizations ---
    print("\nGenerating visualizations...")
    os.makedirs('Tests', exist_ok=True)

    # Accuracy comparison plot
    plt.figure(figsize=(8, 5))
    ks = k_values
    orig_acc = [original_results[k] for k in ks]
    emb_acc = [embedded_results[k] for k in ks]
    plt.plot(ks, orig_acc, 'o-', label='Original')
    plt.plot(ks, emb_acc, 's-', label='Embedded')
    plt.xlabel('k (neighbors)')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy: Original vs Embedded (Covertype sample)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'covertype_knn_accuracy.png'), dpi=150)
    plt.close()

    # t-SNE visualization on a sample (use PCA pre-reduction for speed)
    sample_n = min(3000, len(X_test_processed))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test_processed), size=sample_n, replace=False)

    X_orig_sample = X_test_processed.values[sample_idx]
    X_emb_sample = X_test_embedded.values[sample_idx]
    y_sample = y_test[sample_idx]

    # PCA to 50 dims then TSNE
    def embed_for_tsne(X):
        if X.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            Xp = pca.fit_transform(X)
        else:
            Xp = X
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        return tsne.fit_transform(Xp)

    print('Running t-SNE (this may take a while)...')
    X_tsne_orig = embed_for_tsne(X_orig_sample)
    X_tsne_emb = embed_for_tsne(X_emb_sample)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter = axes[0].scatter(X_tsne_orig[:, 0], X_tsne_orig[:, 1], c=y_sample, cmap='tab10', s=8, alpha=0.8)
    axes[0].set_title('t-SNE: Original space (sample)')
    axes[0].axis('off')
    scatter = axes[1].scatter(X_tsne_emb[:, 0], X_tsne_emb[:, 1], c=y_sample, cmap='tab10', s=8, alpha=0.8)
    axes[1].set_title('t-SNE: Embedded space (sample)')
    axes[1].axis('off')
    plt.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.6)
    plt.suptitle('Covertype: t-SNE comparison (test sample)')
    plt.tight_layout()
    plt.savefig(os.path.join('Tests', 'covertype_tsne_compare.png'), dpi=150)
    plt.close()

    # Confusion matrices for best k (based on embedded results)
    best_k_emb = max(embedded_results, key=embedded_results.get)
    best_k_orig = max(original_results, key=original_results.get)

    # Re-evaluate to obtain predictions for best ks
    def get_predictions(metric_space, labels, X_test_arr, k):
        knn = KNNClassifier(metric_space=metric_space, n_neighbors=k, weights='uniform')
        knn.fit(labels)
        preds = [knn.predict(x) for x in X_test_arr]
        return np.array(preds)

    preds_orig_best = get_predictions(ms_train_orig, y_train, X_test_processed.values, best_k_orig)
    preds_emb_best = get_predictions(ms_train_emb, y_train, X_test_embedded.values, best_k_emb)

    cm_orig = confusion_matrix(y_test, preds_orig_best)
    cm_emb = confusion_matrix(y_test, preds_emb_best)

    # Plot confusion matrices (normalized)
    def plot_cm(cm, title, fname):
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, cmap='Blues')
        plt.title(title)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.tight_layout()
        plt.savefig(os.path.join('Tests', fname), dpi=150)
        plt.close()

    plot_cm(cm_orig, f'Confusion matrix (Original) k={best_k_orig}', 'covertype_cm_original.png')
    plot_cm(cm_emb, f'Confusion matrix (Embedded) k={best_k_emb}', 'covertype_cm_embedded.png')

    print("\nSummary")
    print("k | original_acc | embedded_acc | improvement")
    for k in k_values:
        orig = original_results[k]
        emb = embedded_results[k]
        print(f"{k:2d} | {orig:.4f}       | {emb:.4f}        | {emb - orig:+.4f}")


if __name__ == '__main__':
    main()

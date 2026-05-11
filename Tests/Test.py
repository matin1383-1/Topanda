"""
Test file for Topanda package using scikit-learn datasets with KNN comparison and plotting.

This script:
1. Loads wine, digits, iris, and breast cancer datasets from scikit-learn
2. Preprocesses them using the pipeline
3. Creates metric spaces with euclidean distance
4. Runs the triplet algorithm for each dataset
5. Runs KNN on both original and embedded spaces using the KNN.py implementation
6. Creates comparison plots
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_digits, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from Topanda.PreProcessing.pipeline import DataProcessor
from Topanda.core.metric_space import MetricSpace
from Topanda.DeepMetricLearning.Triplet import TripletLearner
from Topanda.ML.KNN import KNNClassifier


def load_and_prepare_dataset(name: str, data_loader) -> tuple:
    """Load a dataset and prepare it for processing."""
    print(f"\n{'='*60}")
    print(f"Loading {name} dataset...")
    print(f"{'='*60}")
    
    # Load dataset
    data = data_loader()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {len(data.feature_names)}")
    print(f"Classes: {len(np.unique(data.target))}")
    print(f"Target distribution: {np.bincount(data.target)}")
    
    return df, data.target


def process_dataset(df: pd.DataFrame, target: np.ndarray, name: str) -> tuple:
    """Process dataset using pipeline and create metric space."""
    print(f"\nProcessing {name} dataset...")
    
    # Initialize processor with standardization
    processor = DataProcessor(
        standardize_numeric=True,
        embed_categorical=False,  # These datasets are all numeric
        embedding_dim=10
    )
    
    # Fit and transform
    X_processed, y = processor.fit_transform(df, target_col='target')
    
    print(f"Processed features shape: {X_processed.shape}")
    print(f"Auto-detected numeric columns: {len(processor._numeric_cols)}")
    print(f"Auto-detected categorical columns: {len(processor._categorical_cols)}")
    
    return X_processed, y


def create_metric_space(X_processed: pd.DataFrame, name: str) -> MetricSpace:
    """Create metric space with euclidean distance."""
    print(f"\nCreating metric space for {name}...")
    
    ms = MetricSpace(X_processed, metric='euclidean', cache_distances=True)
    print(f"MetricSpace: {ms}")
    
    return ms


def run_triplet_learning(ms: MetricSpace, y: np.ndarray, name: str, original_dim: int) -> tuple:
    """Run triplet learning on the metric space."""
    print(f"\nRunning triplet learning for {name}...")
    
    # Initialize triplet learner with embedding dimension equal to original data dimension
    learner = TripletLearner(
        embedding_dim=original_dim,
        margin=1.0,
        epochs=20,
        batch_size=32,
        lr=1e-3
    )
    
    # Fit and transform metric space
    ms_embedded = learner.fit_transform_metric_space(ms, y)
    
    print(f"Embedded MetricSpace: {ms_embedded}")
    
    return learner, ms_embedded


def run_knn_comparison(original_ms: MetricSpace, embedded_ms: MetricSpace, y: np.ndarray, name: str) -> tuple:
    """Run KNN on both original and embedded spaces using KNN.py and compare results."""
    print(f"\nRunning KNN comparison for {name}...")
    
    # Test different K values
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    orig_accuracies = []
    emb_accuracies = []
    
    for k in k_values:
        # KNN on original space
        knn_orig = KNNClassifier(metric_space=original_ms, n_neighbors=k, weights='uniform')
        knn_orig.fit(y)
        acc_orig = knn_orig.score(y)
        orig_accuracies.append(acc_orig)
        
        # KNN on embedded space
        knn_emb = KNNClassifier(metric_space=embedded_ms, n_neighbors=k, weights='uniform')
        knn_emb.fit(y)
        acc_emb = knn_emb.score(y)
        emb_accuracies.append(acc_emb)
        
        print(f"K={k:2d} - Original: {acc_orig:.4f}, Embedded: {acc_emb:.4f}, Improvement: {acc_emb-acc_orig:+.4f}")
    
    # Best K values and accuracies
    best_k_orig = k_values[np.argmax(orig_accuracies)]
    best_k_emb = k_values[np.argmax(emb_accuracies)]
    best_acc_orig = max(orig_accuracies)
    best_acc_emb = max(emb_accuracies)
    
    print(f"\nBest results for {name}:")
    print(f"Original space - Best K: {best_k_orig}, Accuracy: {best_acc_orig:.4f}")
    print(f"Embedded space - Best K: {best_k_emb}, Accuracy: {best_acc_emb:.4f}")
    print(f"Improvement: {best_acc_emb - best_acc_orig:+.4f}")
    
    return {
        'k_values': k_values,
        'orig_accuracies': orig_accuracies,
        'emb_accuracies': emb_accuracies,
        'best_k_orig': best_k_orig,
        'best_k_emb': best_k_emb,
        'best_acc_orig': best_acc_orig,
        'best_acc_emb': best_acc_emb,
        'improvement': best_acc_emb - best_acc_orig
    }


def create_visualizations(original_ms: MetricSpace, embedded_ms: MetricSpace, y: np.ndarray, name: str, knn_results: dict):
    """Create visualization plots comparing original and embedded spaces."""
    print(f"\nCreating visualizations for {name}...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Triplet Learning Analysis: {name} Dataset', fontsize=16, fontweight='bold')
    
    # 1. KNN Accuracy Comparison
    ax1 = axes[0, 0]
    k_values = knn_results['k_values']
    orig_acc = knn_results['orig_accuracies']
    emb_acc = knn_results['emb_accuracies']
    
    ax1.plot(k_values, orig_acc, 'o-', label='Original Space', linewidth=2, markersize=6)
    ax1.plot(k_values, emb_acc, 's-', label='Embedded Space', linewidth=2, markersize=6)
    ax1.axhline(y=knn_results['best_acc_orig'], color='blue', linestyle='--', alpha=0.7)
    ax1.axhline(y=knn_results['best_acc_emb'], color='orange', linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('K Value')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('KNN Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. t-SNE visualization of original space
    ax2 = axes[0, 1]
    if len(y) > 50:  # Only for larger datasets
        tsne_orig = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne_orig = tsne_orig.fit_transform(original_ms.data.values)
    else:
        # For smaller datasets, use first two dimensions
        X_tsne_orig = original_ms.data.values[:, :2]
    
    scatter = ax2.scatter(X_tsne_orig[:, 0], X_tsne_orig[:, 1], c=y, cmap='tab10', alpha=0.7, s=20)
    ax2.set_title('t-SNE: Original Space')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    # 3. t-SNE visualization of embedded space
    ax3 = axes[0, 2]
    if len(y) > 50:  # Only for larger datasets
        tsne_emb = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne_emb = tsne_emb.fit_transform(embedded_ms.data.values)
    else:
        # For smaller datasets, use first two dimensions
        X_tsne_emb = embedded_ms.data.values[:, :2]
    
    scatter = ax3.scatter(X_tsne_emb[:, 0], X_tsne_emb[:, 1], c=y, cmap='tab10', alpha=0.7, s=20)
    ax3.set_title('t-SNE: Embedded Space')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    
    # 4. Distance distribution comparison
    ax4 = axes[1, 0]
    
    # Calculate distances
    n_samples = len(y)
    within_dist_orig = []
    between_dist_orig = []
    within_dist_emb = []
    between_dist_emb = []
    
    # Sample a subset for visualization (to avoid overcrowding)
    max_samples = min(1000, n_samples)
    indices = np.random.choice(n_samples, max_samples, replace=False)
    
    for i in indices:
        for j in indices:
            if i < j:  # Avoid duplicate pairs
                if y[i] == y[j]:
                    within_dist_orig.append(original_ms.distance_between(i, j))
                    within_dist_emb.append(embedded_ms.distance_between(i, j))
                else:
                    between_dist_orig.append(original_ms.distance_between(i, j))
                    between_dist_emb.append(embedded_ms.distance_between(i, j))
    
    # Plot distance distributions
    ax4.hist(within_dist_orig, bins=30, alpha=0.7, label='Within-class (Original)', color='blue', density=True)
    ax4.hist(between_dist_orig, bins=30, alpha=0.7, label='Between-class (Original)', color='red', density=True)
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Density')
    ax4.set_title('Distance Distribution: Original Space')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Distance distribution for embedded space
    ax5 = axes[1, 1]
    ax5.hist(within_dist_emb, bins=30, alpha=0.7, label='Within-class (Embedded)', color='green', density=True)
    ax5.hist(between_dist_emb, bins=30, alpha=0.7, label='Between-class (Embedded)', color='purple', density=True)
    ax5.set_xlabel('Distance')
    ax5.set_ylabel('Density')
    ax5.set_title('Distance Distribution: Embedded Space')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Separation ratio comparison
    ax6 = axes[1, 2]
    
    # Calculate separation ratios
    within_mean_orig = np.mean(within_dist_orig) if within_dist_orig else 0
    between_mean_orig = np.mean(between_dist_orig) if between_dist_orig else 0
    ratio_orig = between_mean_orig / (within_mean_orig + 1e-8)
    
    within_mean_emb = np.mean(within_dist_emb) if within_dist_emb else 0
    between_mean_emb = np.mean(between_dist_emb) if between_dist_emb else 0
    ratio_emb = between_mean_emb / (within_mean_emb + 1e-8)
    
    categories = ['Original', 'Embedded']
    ratios = [ratio_orig, ratio_emb]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax6.bar(categories, ratios, color=colors, alpha=0.7)
    ax6.set_ylabel('Separation Ratio (Between/Within)')
    ax6.set_title('Class Separation Comparison')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ratio:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{name.lower().replace(" ", "_")}_triplet_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as: {name.lower().replace(' ', '_')}_triplet_analysis.png")


def main():
    """Main function to run the complete pipeline for all datasets."""
    print("Starting Triplet Learning Test with KNN Comparison and Visualization")
    print("=" * 80)
    
    # Define datasets to test
    datasets = [
        ("Iris", load_iris),
        ("Wine", load_wine),
        ("Digits", load_digits),
        ("Breast Cancer", load_breast_cancer)
    ]
    
    results = {}
    
    for name, loader in datasets:
        try:
            # Step 1: Load and prepare dataset
            df, target = load_and_prepare_dataset(name, loader)
            
            # Step 2: Process dataset using pipeline
            X_processed, y = process_dataset(df, target, name)
            
            # Step 3: Create metric space with euclidean distance
            ms = create_metric_space(X_processed, name)
            
            # Step 4: Run triplet learning
            learner, ms_embedded = run_triplet_learning(ms, y, name, original_dim=X_processed.shape[1])
            
            # Step 5: Run KNN comparison using KNN.py
            knn_results = run_knn_comparison(ms, ms_embedded, y, name)
            
            # Step 6: Create visualizations
            create_visualizations(ms, ms_embedded, y, name, knn_results)
            
            # Store results
            results[name] = {
                'original_space': ms,
                'embedded_space': ms_embedded,
                'learner': learner,
                'labels': y,
                'knn_results': knn_results
            }
            
            print(f"\n✓ Successfully completed {name} dataset")
            
        except Exception as e:
            print(f"\n✗ Error processing {name} dataset: {str(e)}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Dataset':<15} {'Best K (Orig)':<14} {'Best K (Emb)':<14} {'Acc (Orig)':<12} {'Acc (Emb)':<12} {'Improvement':<12}")
    print("-" * 90)
    
    for name, result in results.items():
        knn = result['knn_results']
        print(f"{name:<15} {knn['best_k_orig']:<14} {knn['best_k_emb']:<14} {knn['best_acc_orig']:<12.4f} {knn['best_acc_emb']:<12.4f} {knn['improvement']:<+12.4f}")
    
    print(f"\nTotal datasets processed successfully: {len(results)}")
    print("Test completed with KNN comparison and visualization!")


if __name__ == "__main__":
    main()
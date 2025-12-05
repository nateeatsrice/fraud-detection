"""
Unsupervised learning and dimensionality‑reduction techniques for the fraud‑detection dataset.

This script applies a variety of unsupervised algorithms to the feature space and
produces two‑dimensional scatter plots coloured by the fraud label.  These
visualisations can reveal structure, separability, or overlap between
transaction types, which is useful for interviews and portfolio projects.

Techniques included:

* Principal Component Analysis (PCA)
* Multidimensional Scaling (MDS)
* Isomap
* UMAP
* t‑Distributed Stochastic Neighbor Embedding (t‑SNE)
* Isolation Forest anomaly detection

The script samples a subset of the data for computational efficiency.  All
resulting plots are saved into a `plots/unsupervised/` directory.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
try:
    import umap
except ImportError:
    umap = None  # UMAP is optional; ensure installed via requirements


def sample_data(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    """Return a random sample of the dataframe for visualisation.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset with features and labels.
    sample_size : int
        Number of rows to sample.  If larger than the dataset, the entire
        dataset is returned.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Sampled dataframe.
    """
    if sample_size <= 0 or sample_size >= len(df):
        return df.copy()
    return df.sample(n=sample_size, random_state=random_state)


def plot_embedding(embedding: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    """Helper to create and save a scatter plot of a 2D embedding.

    Parameters
    ----------
    embedding : numpy.ndarray
        Two‑dimensional coordinates (n_samples x 2).
    labels : numpy.ndarray
        Binary fraud labels to colour the points.
    title : str
        Plot title.
    out_path : Path
        Location to save the figure.
    """
    fig, ax = plt.subplots()
    fraud_mask = labels == 1
    ax.scatter(embedding[~fraud_mask, 0], embedding[~fraud_mask, 1], alpha=0.5, label="Legitimate", s=10)
    ax.scatter(embedding[fraud_mask, 0], embedding[fraud_mask, 1], alpha=0.5, label="Fraud", s=10)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Apply unsupervised techniques to the fraud dataset and create visualisations.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/transactions.csv",
        help="Path to the CSV file containing the synthetic dataset.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=3000,
        help="Number of samples to use for unsupervised plots (0 for all samples).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    # Only use feature columns
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    fraud_labels = df["fraud_label"].values
    df_sampled = sample_data(df[feature_cols + ["fraud_label"]], args.sample_size, random_state=args.random_state)
    X = df_sampled[feature_cols].values
    y = df_sampled["fraud_label"].values

    # Standardise features for manifold learning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    plots_dir = Path("plots/unsupervised")

    # PCA
    pca = PCA(n_components=2, random_state=args.random_state)
    X_pca = pca.fit_transform(X_scaled)
    plot_embedding(X_pca, y, "PCA", plots_dir / "pca.png")
    print("Saved PCA plot")

    # MDS
    mds = MDS(n_components=2, random_state=args.random_state, dissimilarity="euclidean")
    X_mds = mds.fit_transform(X_scaled)
    plot_embedding(X_mds, y, "MDS", plots_dir / "mds.png")
    print("Saved MDS plot")

    # Isomap
    iso = Isomap(n_neighbors=15, n_components=2)
    X_iso = iso.fit_transform(X_scaled)
    plot_embedding(X_iso, y, "Isomap", plots_dir / "isomap.png")
    print("Saved Isomap plot")

    # UMAP (optional)
    if umap is not None:
        umap_model = umap.UMAP(n_components=2, random_state=args.random_state)
        X_umap = umap_model.fit_transform(X_scaled)
        plot_embedding(X_umap, y, "UMAP", plots_dir / "umap.png")
        print("Saved UMAP plot")
    else:
        print("UMAP not installed; skipping UMAP plot.  Install umap-learn to enable this.")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=args.random_state, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    plot_embedding(X_tsne, y, "t-SNE", plots_dir / "tsne.png")
    print("Saved t-SNE plot")

    # Isolation Forest anomaly score scatter (colour points by anomaly level)
    iso_forest = IsolationForest(contamination=0.1, random_state=args.random_state)
    scores = -iso_forest.fit_predict(X_scaled)
    # Normalise scores to [0, 1]
    score_norm = (scores - scores.min()) / (scores.max() - scores.min())
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=score_norm, s=10)
    ax.set_title("Isolation Forest Anomaly Scores (PCA axes)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    fig.colorbar(scatter, ax=ax, label="Anomaly Score")
    fig.tight_layout()
    out_path = plots_dir / "isolation_forest.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print("Saved Isolation Forest anomaly score plot")


if __name__ == "__main__":
    main()

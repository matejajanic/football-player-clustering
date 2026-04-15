from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.clustering.kmeans_model import run_kmeans, prepare_feature_matrix
from src.clustering.evaluation import evaluate_clustering
from src.clustering.hierarchical_model import run_hierarchical
from src.clustering.gmm_model import run_gmm
from src.clustering.dbscan_model import run_dbscan
from src.clustering.spectral_model import run_spectral


def main():
    path = PROCESSED_DATA_DIR / "player_profiles.csv"
    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)

    # === FEATURE SELECTION ===
    feature_columns = [
        "matches_played",
        "total_minutes",
        "goals_per_90",
        "assists_per_90",
        "yellow_per_90",
        "red_per_90",
        "avg_market_value",
        "max_market_value",
        "height_in_cm",
        "age",
    ]

    df = df.dropna(subset=feature_columns)

    print("Using features:", feature_columns)

    best_k = 3

    # =========================
    # KMEANS
    # =========================
    print("\n" + "="*50)
    print("KMEANS")

    labels_kmeans, model, scaler = run_kmeans(df, feature_columns, n_clusters=best_k)
    X_scaled, _ = prepare_feature_matrix(df, feature_columns)

    scores_kmeans = evaluate_clustering(X_scaled, labels_kmeans)

    print(f"Silhouette: {scores_kmeans['silhouette']:.4f}")
    print(f"Davies-Bouldin: {scores_kmeans['davies_bouldin']:.4f}")

    df["cluster_kmeans"] = labels_kmeans

    # =========================
    # HIERARCHICAL
    # =========================
    print("\n" + "="*50)
    print("HIERARCHICAL")

    labels_hier, X_scaled = run_hierarchical(df, feature_columns, best_k)
    scores_hier = evaluate_clustering(X_scaled, labels_hier)

    print(f"Silhouette: {scores_hier['silhouette']:.4f}")
    print(f"Davies-Bouldin: {scores_hier['davies_bouldin']:.4f}")

    df["cluster_hierarchical"] = labels_hier

    # =========================
    # GMM
    # =========================
    print("\n" + "="*50)
    print("GMM")

    labels_gmm, X_scaled = run_gmm(df, feature_columns, best_k)
    scores_gmm = evaluate_clustering(X_scaled, labels_gmm)

    print(f"Silhouette: {scores_gmm['silhouette']:.4f}")
    print(f"Davies-Bouldin: {scores_gmm['davies_bouldin']:.4f}")

    df["cluster_gmm"] = labels_gmm

    # =========================
    # DBSCAN
    # =========================
    print("\n" + "="*50)
    print("DBSCAN")

    labels_dbscan, X_scaled = run_dbscan(df, feature_columns, eps=0.7, min_samples=10)

    n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = list(labels_dbscan).count(-1)

    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")

    if n_clusters > 1:
        scores_dbscan = evaluate_clustering(X_scaled, labels_dbscan)
        print(f"Silhouette: {scores_dbscan['silhouette']:.4f}")
        print(f"Davies-Bouldin: {scores_dbscan['davies_bouldin']:.4f}")
    else:
        print("Not enough clusters for evaluation")

    df["cluster_dbscan"] = labels_dbscan

    # =========================
    # SPECTRAL
    # =========================
    print("\n" + "="*50)
    print("SPECTRAL")

    labels_spec, X_scaled = run_spectral(df, feature_columns, best_k)

    scores_spec = evaluate_clustering(X_scaled, labels_spec)

    print(f"Silhouette: {scores_spec['silhouette']:.4f}")
    print(f"Davies-Bouldin: {scores_spec['davies_bouldin']:.4f}")

    df["cluster_spectral"] = labels_spec

    # =========================
    # SAVE
    # =========================
    output_path = PROCESSED_DATA_DIR / "player_clusters.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved clustered dataset to: {output_path}")


if __name__ == "__main__":
    main()
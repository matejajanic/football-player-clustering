from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.clustering.kmeans_model import run_kmeans, prepare_feature_matrix
from src.clustering.evaluation import evaluate_clustering
from src.clustering.hierarchical_model import run_hierarchical
from src.clustering.gmm_model import run_gmm


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
    # SAVE
    # =========================
    output_path = PROCESSED_DATA_DIR / "player_clusters.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved clustered dataset to: {output_path}")


if __name__ == "__main__":
    main()
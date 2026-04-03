from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.clustering.kmeans_model import run_kmeans, prepare_feature_matrix
from src.clustering.evaluation import evaluate_clustering


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

    # remove missing just in case
    df = df.dropna(subset=feature_columns)

    print("Using features:", feature_columns)

    # === TEST DIFFERENT K ===
    # choose best k manually after inspection
    best_k = 3

    labels, model, scaler = run_kmeans(df, feature_columns, n_clusters=best_k)

    df["cluster"] = labels

    # save
    output_path = PROCESSED_DATA_DIR / "player_clusters.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved clustered dataset to: {output_path}")

if __name__ == "__main__":
    main()
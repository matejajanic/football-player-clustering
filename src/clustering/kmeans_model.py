from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: list[str]):
    """
    Extract and scale features for clustering.
    """
    X = df[feature_columns].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def run_kmeans(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int,
    random_state: int = 42,
):
    """
    Fit KMeans and return labels and model.
    """
    X_scaled, scaler = prepare_feature_matrix(df, feature_columns)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    return labels, model, scaler
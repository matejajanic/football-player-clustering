from __future__ import annotations

from sklearn.metrics import silhouette_score, davies_bouldin_score


def evaluate_clustering(X_scaled, labels):
    """
    Compute clustering quality metrics.
    """
    results = {}

    if len(set(labels)) > 1:
        results["silhouette"] = silhouette_score(X_scaled, labels)
        results["davies_bouldin"] = davies_bouldin_score(X_scaled, labels)
    else:
        results["silhouette"] = -1
        results["davies_bouldin"] = float("inf")

    return results
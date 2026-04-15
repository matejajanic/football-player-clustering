from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler


def run_spectral(df, feature_columns, n_clusters):
    X = df[feature_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        random_state=42,
    )

    labels = model.fit_predict(X_scaled)

    return labels, X_scaled
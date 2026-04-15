from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def run_dbscan(df, feature_columns, eps=0.7, min_samples=10):
    X = df[feature_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(eps=eps, min_samples=min_samples)

    labels = model.fit_predict(X_scaled)

    return labels, X_scaled
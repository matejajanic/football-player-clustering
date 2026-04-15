from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def run_gmm(df, feature_columns, n_components):
    X = df[feature_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianMixture(n_components=n_components, random_state=42)

    labels = model.fit_predict(X_scaled)

    return labels, X_scaled
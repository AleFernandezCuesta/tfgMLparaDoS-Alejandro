from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score, silhouette_score


def construccion_kmeans_pipeline(n_clusters=2, random_state=19):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state))
    ])

def entrenamiento_y_evaluacion_kmeans(pipeline, x, y_true):
    # Entrenamos y predecimos clusters
    pipeline.fit(x)
    y_pred = pipeline.named_steps["kmeans"].labels_

    # Metricas
    ari = adjusted_rand_score(y_true, y_pred)
    silhouette = silhouette_score(x, y_pred)
    matriz = confusion_matrix(y_true, y_pred)

    print("Evaluacion del modelo K-Means:\n")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print("\nMatriz de confusion:")
    print(matriz)

    return ari, silhouette, matriz

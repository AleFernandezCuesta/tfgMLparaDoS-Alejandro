from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, adjusted_rand_score
import numpy as np


def construccion_isolation_pipeline(random_state = 19, n_estimators=100, contamination = 0.57):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("isolation", IsolationForest(n_estimators = n_estimators, contamination = contamination, random_state = random_state))
    ])

def entrenamiento_y_evaluacion_isolation(pipeline, x_entrenamiento, x_evaluacion, y_true):
    
    pipeline.fit(x_entrenamiento)
    # Obtener predicciones: -1 = anomalia -> lo convertimos a 1 (ataque), 1 = normal -> lo convertimos a 0 (benigno)
    y_pred_raw = pipeline.named_steps["isolation"].predict(x_evaluacion)
    y_pred = np.where(y_pred_raw == -1, 1, 0)

    # Metricas
    matriz = confusion_matrix(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    reporte = classification_report(y_true, y_pred, target_names = ["BENIGN", "DoS"])

    print("Evaluacion del modelo Isolation Forest:\n")
    print(reporte)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print("\nMatriz de confusion:")
    print(matriz)

    return ari, matriz, reporte   

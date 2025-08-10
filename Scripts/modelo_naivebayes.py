from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

def construccion_nb_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("naivebayes", GaussianNB())
    ])

def entrenamiento_y_evaluacion_nb(pipeline, x_train, x_test, y_train, y_test, guardar_graficas = True, carpeta_salida = "../Notebooks/IMAGENES/Graficas_NaiveBayes"):
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    # Metricas
    reporte = classification_report(y_test, y_pred, target_names = ["BENIGN", "DoS"])
    matriz = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("Informe de clasificacion:\n")
    print(reporte)
    print("Matriz de confusion:\n")
    print(matriz)
    print(f"Area bajo la curva ROC (AUC): {auc:.4f}")

    if guardar_graficas:
        RocCurveDisplay.from_estimator(pipeline, x_test, y_test)
        plt.title("Curva ROC - Naive Bayes")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/curvaroc_nb.png")
        plt.close()

    return reporte, matriz, auc
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt


def construccion_rf_pipeline(n_estimators=100, random_state=19):
    # n_jobs = -1 para aprovechar todos los nucleos del procesador
    return Pipeline([("scaler", StandardScaler()),("rf", RandomForestClassifier(n_estimators=n_estimators, random_state = random_state, n_jobs=-1))])

# Entrenamos el modelo y evaluamos su rendimiento, ademas de generar informes y graficas ROC
def entrenamiento_y_evaluacion_rf(pipeline, x_train, x_test, y_train, y_test, guardar_graficas = True, carpeta_salida = "../Notebooks/IMAGENES/Graficas_RandomForest"):
    pipeline.fit(x_train, y_train) 
    y_pred = pipeline.predict(x_test) 
    y_prob = pipeline.predict_proba(x_test)[:,1] 
    
    # Metricas
    reporte = classification_report(y_test, y_pred, output_dict = True) 
    matriz = confusion_matrix(y_test, y_pred) 
    auc = roc_auc_score(y_test, y_prob) 

    print("Informe de clasificacion:\n")
    print(reporte)
    print("\n Matriz de confusion\n")
    print(matriz)
    print(f"\nArea bajo la curva ROC (AUC): {auc:.3f}")

    # Guardamos la curva ROC como imagen
    if guardar_graficas:
        RocCurveDisplay.from_estimator(pipeline, x_test, y_test)
        plt.title("Curva ROC - Random Forest")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/curvaroc_rf.png")
        plt.close()

    return reporte, matriz, auc
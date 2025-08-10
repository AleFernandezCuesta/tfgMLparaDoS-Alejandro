import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

#definimos el pipeline de svm:
#Kernel = rbf: kernel radial, permite separara los datos de forma no lineal
#C = 1.0: Parametro de regularizacion, cuanto mas bajo mas tolerante al error es el modelo y cuanto mas alto mas estricto (dejandonos menos margen de error)
#Gamma = scale: Define el alcance de la influencia de los puntos individuales en el modelo (scale es la opcion mas recomendada)
#Probability = True: Habilita el calculo de probabilidades necesarias para la curva ROC y AUC
def construir_svm_pipeline(kernel = 'rbf', C = 1.0, gamma = 'scale', random_state = None):

    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma = gamma, probability=True, random_state=random_state))
    ])


def entrenamiento_y_evaluacion_svm(pipeline, x_train, x_test, y_train, y_test, guardar_graficas = True, carpeta_salida = "../Notebooks/IMAGENES/Graficas_SVM"):
    pipeline.fit(x_train, y_train) 
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:,1] 

    # Metricas
    reporte = classification_report(y_test, y_pred, output_dict = True) 
    matriz = confusion_matrix(y_test, y_pred) 
    auc = roc_auc_score(y_test, y_prob) 

    
    print("Informe de clasificacion:\n")
    print(classification_report(y_test, y_pred))
    print("\n Matriz de confusion\n")
    print(matriz)
    print(f"\nArea bajo la curva ROC (AUC): {auc:.3f}")

    
    if guardar_graficas:
        disp = RocCurveDisplay.from_estimator(pipeline, x_test, y_test)
        plt.title("Curva ROC - SVM")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/curvaroc_SVM.png")
        plt.close()

    return reporte, matriz, auc


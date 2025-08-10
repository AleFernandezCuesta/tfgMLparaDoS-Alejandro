import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

def construir_decisiontree_pipeline(criterion='gini', max_depth=None, random_state=19):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('dt', DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state))
    ])

def entrenamiento_y_evaluacion_dt(pipeline, x_train, x_test, y_train, y_test, guardar_graficas = True, carpeta_salida = "../Notebooks/IMAGENES/Graficas_DecisionTree"):
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

    
    if guardar_graficas:
        RocCurveDisplay.from_estimator(pipeline, x_test, y_test)
        plt.title("Curva ROC - Decision Tree")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/curvaroc_dt.png")
        plt.close()

    return reporte, matriz, auc


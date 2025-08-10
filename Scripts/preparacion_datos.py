import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Primero cargamos los datos del csv
def cargar_datos_csv(ruta_csv: str):
    df = pd.read_csv(ruta_csv)
    return df

# Dejamos las etiquetas en modo int para asegurar que no haya problemas por tener tipos de datos diferentes en etiquetas para algunos modelos de clasificacion
def preprocesar_etiquetas(df, label_col='Label'):
    df[label_col] = df[label_col].astype(int)
    return df

# Reemplazo inf/-inf por NaN y elimino las filas con NaN (necesaria porque algunos datos nos daban errores por contener infinitos)
def limpiar_datos(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    n_antes = len(df)
    df = df.dropna()
    # Añado un capping para cada columna
    x = df.drop(columns= ["Label"])
    for col in x.select_dtypes(include = [np.number]).columns:
        # Aqui lo que hacemos es, cualquier valor que este por debajo del 1% lo subo al 1% y cualquier valor por encima del 99% lo bajo al 99%
        # Lo hacemos para asegurar que todos los valores esten dentro de un rango razonable y asi evitamos los outliers y prevenir de undefitting y overfitting
        superior = df[col].quantile(0.99)
        inferior = df[col].quantile(0.01)
        df[col] = df[col].clip(inferior, superior)

    n_despues = len(df)
    print(f"Datos limpiados: Eliminadas {n_antes - n_despues} filas que contenian inf o NaN")
    # Verificamos si queda algun infinito
    infinitos_restantes = np.isfinite(df.select_dtypes(include=[np.number])).all().all()
    print("Si sale True no quedan infinitos: ", infinitos_restantes)
    return df

# Para la separacion de datos, dejo el 20% para test (test_size=0.2), hace el capping y limpieza de infinitos y NaN
def dividir_datos(df, label_col='Label', test_size=0.2, random_state=19):
    df = limpiar_datos(df)
    x = df.drop(columns=[label_col])
    y = df[label_col]
    
    # Vamos a asegurar de que no queda ningun infinito ni NaN
    mask = np.isfinite(x).all(axis = 1)
    x = x[mask]
    y = y[mask]
    # train_test_split(...) sirve para dividir caracteristicas(x) y etiquetas (y) en conjuntos de entrenamiento y test, estratificado
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    # stratify=y mantiene la proporcion de etiquetas en ambos conjuntos, esencial si hay clases desbalanceadas
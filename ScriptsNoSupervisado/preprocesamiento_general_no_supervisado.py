# Script que procesa los CSV originales igual que para supervisado, pero sin columna Label

import pandas as pd
import os
import numpy as np

carpeta_entrada = "DataSetCICIDS2017/MachineLearningCVE"
carpeta_salida = "Datos_procesados_no_supervisado"

os.makedirs(carpeta_salida, exist_ok=True)

columnas_a_eliminar = [
    'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate'
]

for nombre_archivo in os.listdir(carpeta_entrada):
    if nombre_archivo.endswith('.csv'):
        ruta_archivo = os.path.join(carpeta_entrada, nombre_archivo)
        df = pd.read_csv(ruta_archivo)
        df.columns = df.columns.str.strip()

        columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
        df.drop(columns=columnas_existentes, inplace=True)

        df = df.replace([np.inf, -np.inf], np.nan)
        n_antes = len(df)
        df = df.dropna()
        n_despues = len(df)

        print(f"Datos limpiados: Eliminadas {n_antes - n_despues} filas con inf o NaN")
        sin_inf = np.isfinite(df.select_dtypes(include = [np.number])).all().all()
        print("Quedan infinitos?", not sin_inf)
        
        # Guardamos una copia de los archivos que solo contengan la columna label para utilizar despues del entrenamiento para evaluacion de los modelos
        if 'Label' in df.columns:
            df['Label'].to_csv(os.path.join(carpeta_salida, f"{nombre_archivo.replace('.csv', '')}_labels.csv"), index=False)
            df = df.drop(columns=['Label'])

        
        nombre_salida = f"{nombre_archivo.replace('.csv', '')}_preprocesado_no_label.csv"
        df.to_csv(os.path.join(carpeta_salida, nombre_salida), index=False)

        print(f"Archivo guardado: {nombre_salida}")

# Script para prepocesar los datos de cada uno de los datasets y los introduzco en la carpeta Datos_procesados
import pandas as pd
import os

carpeta_csv = 'DataSetCICIDS2017/MachineLearningCVE'
carpeta_csvprocesados = 'Datos_procesados'

columnas_a_eliminar = ['Fwd Header Length.1',
                       'Fwd Avg Bytes/Bulk',
                       'Fwd Avg Packets/Bulk',
                       'Fwd Avg Bulk Rate',
                       'Bwd Avg Bytes/Bulk',
                       'Bwd Avg Packets/Bulk',
                       'Bwd Avg Bulk Rate'
]

# Voy a recorrer toda la carpeta para ir procesando los datos uno a uno
for nombre_archivo in os.listdir(carpeta_csv):
    if nombre_archivo.endswith('.csv'):
        ruta_archivo = os.path.join(carpeta_csv, nombre_archivo)
        print(f"\nProcesando archivo: {nombre_archivo}")
        df = pd.read_csv(ruta_archivo)
        df.columns = df.columns.str.strip()
        # Eliminar las columnas si existen (podrian no existir en algun csv)
        columnas_existentes = []
        for col in columnas_a_eliminar:
            if col in df.columns:
                columnas_existentes.append(col)
        df.drop(columns = columnas_existentes, inplace = True)
        # Verificamos si existe la columna Label en el dataframe y lo codificamos para la maquina (algunos algoritmos necesitan trabajar con datos en formato numerico)
        if 'Label' in df.columns:
            valores_unicos = df['Label'].unique()
            print("Valores unicos de Label: ", valores_unicos)
            df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        # Guardamos el nuevo archivo prepocesado
        ruta_salida = os.path.join(carpeta_csvprocesados, f"{nombre_archivo.split('.csv')[0]}-preprocesado.csv")
        df.to_csv(ruta_salida, index=False)

        print(f"Archivo guardado: {ruta_salida}")
        print(f"Tamaño final: {df.shape}")
        print("Distribucion de etiquetas:")
        print(df['Label'].value_counts())
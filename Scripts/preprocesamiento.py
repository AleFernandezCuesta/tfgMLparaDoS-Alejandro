# Una primera vista al preprocesamiento de datos de Friday-WorkingHours-Afternoon para luego implementar otro script que sirva para todos los archivos
import pandas as pd

ruta_csv = 'DataSetCICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(ruta_csv)

# Procedo a limpiar todos los espacios que pueda haber entre nombres de columnas al leer el archivo (en el IDS de CIC vienen las columnas con un espacio inicial)
df.columns = df.columns.str.strip()
# Leo las columnas para ver que se han limpiado los espacios para que luego pueda coger las columnas que necesite sin problemas de espacios
print("Columnas reales del archivo: \n")
print(df.columns.tolist())

# Eliminamos las columnas que no aportan valor para reducir el tamaño de los datos a analizar
columnas_a_eliminar = ['Fwd Header Length.1',
                       'Fwd Avg Bytes/Bulk',
                       'Fwd Avg Packets/Bulk',
                       'Fwd Avg Bulk Rate',
                       'Bwd Avg Bytes/Bulk',
                       'Bwd Avg Packets/Bulk',
                       'Bwd Avg Bulk Rate']
df.drop(columns=columnas_a_eliminar, inplace=True)

# Codificamos las etiquetas: BENIGN = 0, DDoS = 1
print("Valores de Label antes de codificar:", df['Label'].unique())
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Guardamos el dataset limpio 
df.to_csv('Datos_procesados/Friday-DDos-preprocesado.csv', index=False)

print("Preprocesamiento completado.")
print(f"Nuevo tamaño del dataset: {df.shape}")
print("Distribución de etiquetas:")
print(df['Label'].value_counts())
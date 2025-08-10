import pandas as pd
import os
carpeta_datos_procesados = 'Datos_procesados'
# Voy a crear un array que recoja todos los dataframes de CIC-IDS para luego concatenarlos todos en uno 
dataframes = [] 

# Voy a recorrer la carpeta de datos procesados uno a uno en un bucle y guardarlo en un df de pandas para luego meterlo en el array 
for archivo in os.listdir(carpeta_datos_procesados):
    if archivo.endswith('.csv'):
        ruta_archivo = os.path.join(carpeta_datos_procesados, archivo)
        print(f"Cargando archivo: {archivo}")
        df = pd.read_csv(ruta_archivo)
        dataframes.append(df)

# Una vez finalizado el bucle, basta con concatenar todo el array resultante
df_combinado = pd.concat(dataframes, ignore_index = True)
# Lo voy a guardar en formato csv en una carpeta aparte donde esten mis datos combinados para futuros analisis
df_combinado.to_csv('Datos_procesados_combinados/dataset_combinado.csv', index=False)
print("\nTodos los archivos han sido combinados con exito")
print(f"Tamaño final del dataset combinado: {df_combinado.shape}")
print("Distribucion de etiquetas benignas(label = 0) o ataques(label =1) en el dataset combinado:\n")
cuentas = df_combinado['Label'].value_counts()
print(f"Instancias de ataques (Label=1): {cuentas[1]}")
print(f"Instancias benignas (Label=0): {cuentas[0]}")
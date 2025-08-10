import pandas as pd

ruta_csv = "DataSetCICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(ruta_csv)

print("Primeras filas del dataset:\n")
print(df.head())
print("\nTamaño del dataset (filas, columnas):", df.shape)
print("\nFrecuencia de cada clase (ataques [DDoS] o normal [BENIGN]):")
print(df[' Label'].value_counts())
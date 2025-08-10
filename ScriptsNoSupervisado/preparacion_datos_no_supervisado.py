import pandas as pd
import numpy as np

def preparar_datos_no_supervisado(df):
    
    df = df.replace([np.inf, -np.inf], np.nan)
    n_antes = len(df)
    df = df.dropna()
    n_despues = len(df)

    print(f"Datos limpiados: Eliminadas {n_antes - n_despues} filas con inf o NaN")
    sin_inf = np.isfinite(df.select_dtypes(include = [np.number])).all().all()
    print("Quedan infinitos?", not sin_inf)
    
    return df
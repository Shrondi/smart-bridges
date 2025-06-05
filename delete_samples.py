import os
import pandas as pd
import numpy as np

def clean_by_majority_sign(df):
    """
    Para cada columna de aceleración, calcula el signo mayoritario y elimina las filas cuyo signo en esa columna no coincida.
    """
    cols = ['x_accel (g)', 'y_accel (g)', 'z_accel (g)']
    data = df[cols].values
    signs = np.sign(data)
    # Calcular el signo mayoritario para cada columna
    majority_sign = np.sign(np.sum(signs, axis=0))
    # Si hay empate (suma=0), se considera positivo por defecto
    majority_sign[majority_sign == 0] = 1
    # Crear máscara: True si todos los signos coinciden con el mayoritario
    mask = (signs == majority_sign).all(axis=1)
    return df[mask].reset_index(drop=True)

def process_file(file):
    """
    Procesa un único archivo acceleration_*.csv indicado por su ruta.
    Elimina filas cuyo signo en cualquier eje no coincida con el signo mayoritario de ese eje.
    """
    try:
        df = pd.read_csv(file, sep=',', engine='c')
    except Exception as e:
        print(f"No se pudo leer {file}: {e}")
        return
    n_original = len(df)
    try:
        df_clean = clean_by_majority_sign(df)
    except Exception as e:
        print(f"No se pudo limpiar el archivo {file}: {e}")
        return
    n_final = len(df_clean)
    n_eliminadas = n_original - n_final
    if n_eliminadas > 0:
        print(f"Filas eliminadas en {file}: {n_eliminadas} de {n_original} ({n_eliminadas/n_original:.2%})")
        df_clean.to_csv(file, index=False)
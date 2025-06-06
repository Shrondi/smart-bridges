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

def remove_jumps(df, threshold=0.5):
    """
    Elimina las filas donde la diferencia absoluta entre la muestra actual y la última muestra válida
    en cualquiera de los ejes supera el umbral (por defecto 0.5).
    Al encontrar un salto brusco, avanza hasta encontrar un punto que pueda considerarse 
    válido respecto al último punto conocido como bueno.
    """
    cols = ['x_accel (g)', 'y_accel (g)', 'z_accel (g)']
    
    # Usamos numpy para acelerar los cálculos
    data = df[cols].values
    n_samples = len(data)
    result_indices = [0]  # Siempre incluimos la primera muestra
    last_good_idx = 0
    
    # Pre-extraemos el último punto bueno para evitar accesos repetidos al DataFrame
    last_good_point = data[last_good_idx]
    
    for i in range(1, n_samples):
        # Cálculo vectorizado de diferencias
        diffs = np.abs(data[i] - last_good_point)
        
        # Si todas las diferencias están dentro del umbral
        if np.all(diffs <= threshold):
            result_indices.append(i)
            last_good_idx = i
            last_good_point = data[i]  # Actualizamos el punto de referencia
    
    # Devolvemos solo las filas que pasaron el filtro
    return df.iloc[result_indices].reset_index(drop=True)

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
        df_clean = remove_jumps(df_clean, threshold=0.5)
    except Exception as e:
        print(f"No se pudo limpiar el archivo {file}: {e}")
        return
    n_final = len(df_clean)
    n_eliminadas = n_original - n_final
    if n_eliminadas > 0:
        print(f"Filas eliminadas en {file}: {n_eliminadas} de {n_original} ({n_eliminadas/n_original:.2%})")
        df_clean.to_csv(file, index=False)
import pandas as pd
import numpy as np

def clean_by_majority_sign(df):
    """
    Para cada columna de aceleración, calcula el signo mayoritario y elimina las filas cuyo signo en esa columna no coincida.
    En caso de empate (suma=0), se considera positivo por defecto.
    """
    cols = ['x_accel (g)', 'y_accel (g)', 'z_accel (g)']
    
    data = df[cols].values
    signs = np.sign(data)
    
    # Calcular el signo mayoritario para cada columna
    majority_sign = np.sign(np.sum(signs, axis=0))
    
   # Ignorar columnas con empate (suma=0) estableciendo su signo mayoritario como NaN
    majority_sign[majority_sign == 0] = np.nan
    
    # Filtrar filas: mantener solo las que coincidan con el signo mayoritario en columnas válidas
    mask = np.all((signs == majority_sign) | np.isnan(majority_sign), axis=1)
    
    return df[mask]

def remove_jumps(df, threshold=0.5):
    """
    Elimina las filas donde la diferencia absoluta entre la muestra actual y la anterior
    en cualquiera de los ejes supera el umbral (por defecto 0.5).
    """
    cols = ['x_accel (g)', 'y_accel (g)', 'z_accel (g)']
    
    # Calcular la diferencia absoluta con la fila anterior
    diff = df[cols].diff().abs()
    
    # Mantener la primera fila y aquellas donde ninguna diferencia supera el umbral
    mask = (diff <= threshold).all(axis=1)
    if not mask.empty:
        mask.iloc[0] = True  # Mantener la primera muestra
        
    return df[mask]
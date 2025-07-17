import numpy as np

def clean_by_majority_sign(df):
    """
    Para cada columna de aceleración, calcula el signo mayoritario y elimina las filas cuyo signo en esa columna no coincida.
    En caso de empate (suma=0), esa columna no se usa para filtrar filas (no se modifica).
    """
    cols = ['x_accel (g)', 'y_accel (g)', 'z_accel (g)']
    
    data = df[cols].values
    signs = np.sign(data)
    
    # Calcular el signo mayoritario por columna
    majority_sign = np.sign(np.sum(signs, axis=0))
    
    # Columnas con mayoría clara (no empate)
    valid_cols = majority_sign != 0
    
    # Filtrar filas donde el signo coincide en columnas válidas
    # Si no hay columnas válidas, no se filtra ninguna fila (mask a True)
    if valid_cols.any():
        mask = (signs[:, valid_cols] == majority_sign[valid_cols]).all(axis=1)
    else:
        mask = np.ones(len(df), dtype=bool)
    
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
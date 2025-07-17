import pandas as pd
import os
from delete_samples import clean_by_majority_sign, remove_jumps
from check_train_file import is_anomalous

def load_file(filepath):
    """
    Carga y procesa un archivo CSV, verificando que tenga las columnas esperadas y un índice basado en 'timestamp'.
    
    Args:
        filepath (str): Ruta del archivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame procesado con los datos del archivo CSV.
                     Si el archivo no cumple con las condiciones, devuelve un DataFrame vacío.
    """

    try:
        # Cargar el archivo CSV
        df = pd.read_csv(filepath, sep=',', engine='c', parse_dates=['timestamp'], 
                         index_col='timestamp', date_format='%H:%M:%S.%f', on_bad_lines='error')

        # Verificar si las columnas esperadas existen
        expected_columns = ['x_accel (g)', 'y_accel (g)', 'z_accel (g)']
        if list(df.columns) != expected_columns:
            raise ValueError(f"El archivo {filepath} no tiene las columnas esperadas: {expected_columns}. Se omite.")

    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo {filepath} no existe.")
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"El archivo {filepath} está vacío.")
    
    except pd.errors.ParserError:
        raise ValueError(f"Error al analizar el archivo {filepath}. Verifica su formato.")

    return df

def get_anomalies_filepath(filepath):
    carpeta_actual = os.path.dirname(filepath)
    carpeta_anomalias = os.path.join(carpeta_actual, 'anomalias')

    # Solo cambia la ruta si no está ya en la carpeta de anomalías
    if os.path.basename(carpeta_actual) != 'anomalias':
        os.makedirs(carpeta_anomalias, exist_ok=True)
        os.remove(filepath)  # Elimina el archivo original solo si se va a mover
        return os.path.join(carpeta_anomalias, os.path.basename(filepath))
    
    return filepath

def clean_data(df):
    """
    Limpia un DataFrame eliminando filas cuyo signo en cualquier eje no coincida con el signo mayoritario de ese eje,
    y eliminando saltos en los datos según un umbral especificado.
    Args:
        df (pd.DataFrame): DataFrame con columnas 'x_accel (g)', 'y_accel (g)', 'z_accel (g)' y un índice basado en 'timestamp'.
    Returns:
        int: Número de filas eliminadas durante el proceso de limpieza.
    """
    
    try:
        df_clean = clean_by_majority_sign(df)
        df_clean = remove_jumps(df_clean, threshold=0.5)
    except Exception as e:
        raise ValueError(f"No se pudo limpiar el DataFrame. Error: {e}")
    
    return df_clean

def process_file(filepath):
    """
    Procesa un archivo CSV, limpiándolo, verificando si es anómalo y moviéndolo a la carpeta de anomalías si es necesario.
    
    Args:
        filepath (str): Ruta del archivo CSV a procesar.
    
    Returns:
        str: Ruta final del archivo procesado (puede ser la misma o en la carpeta de anomalías).
    """
    
    df = load_file(filepath)
    
    if df.empty:
        return None
    
    df_clean = clean_data(df)
    print(f"[!] Archivo {filepath} procesado. Filas eliminadas: {len(df) - len(df_clean)} de {len(df)} ({(len(df) - len(df_clean)) / len(df):.2%})")
    
    if df_clean.empty:
        return None
    
    destino = get_anomalies_filepath(filepath) if is_anomalous(df_clean) else filepath
       
    df_clean.to_csv(destino, sep=',', index=True, header=True, index_label='timestamp', date_format='%H:%M:%S.%f')
    
    return destino
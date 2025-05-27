import os
import csv
import shutil
from datetime import datetime

def parseTimestamp(timestamp_str):
    return datetime.strptime(timestamp_str, "%H:%M:%S.%f")

def processCSV(filepath):
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # Saltar cabecera

        try:
            primera_linea = next(reader)
        except StopIteration:
            return True  # Archivo vacío o sin datos

        primera_ts = parseTimestamp(primera_linea[0])
        ultima_ts = primera_ts
        ts_anterior = primera_ts

        for row in reader:
            if not row:
                continue

            ts_actual = parseTimestamp(row[0])
            if ts_actual < ts_anterior:
                return True

            ts_anterior = ts_actual
            ultima_ts = ts_actual

        duracion = (ultima_ts - primera_ts).total_seconds()
        if duracion < 2:
            return True

        return False

def moveToFolder(filepath):
    carpeta_actual = os.path.dirname(filepath)
    carpeta_anomalias = os.path.join(carpeta_actual, 'anomalias')
    os.makedirs(carpeta_anomalias, exist_ok=True)
    destino = os.path.join(carpeta_anomalias, os.path.basename(filepath))
    shutil.move(filepath, destino)
    return destino

def verifyFile(filepath):
    """
    Verifica si un archivo CSV es anómalo.
    Devuelve una tupla (es_anomalo: bool, ruta_final: str)
    No procesa archivos que ya estén dentro de una carpeta 'anomalias'.
    """
    # Normalizar ruta para evitar problemas con separadores Windows/Linux
    normalized_path = filepath.replace('\\', '/').lower()

    # Si ya está en carpeta 'anomalias', no procesar
    if '/anomalias/' in normalized_path or normalized_path.endswith('/anomalias'):
        return True, filepath

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    es_anomalo = processCSV(filepath)
    if es_anomalo:
        nueva_ruta = moveToFolder(filepath)
        return True, nueva_ruta
    else:
        return False, filepath


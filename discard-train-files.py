import os
import csv
import shutil
import sys
from datetime import datetime

def parse_timestamp(timestamp_str):
    return datetime.strptime(timestamp_str, "%H:%M:%S.%f")

def procesar_csv(filepath):
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # Saltar cabecera

        try:
            primera_linea = next(reader)
        except StopIteration:
            return True  # Archivo vacío o sin datos

        primera_ts = parse_timestamp(primera_linea[0])
        ultima_ts = primera_ts
        ts_anterior = primera_ts

        for row in reader:
            if not row:
                continue

            ts_actual = parse_timestamp(row[0])
            if ts_actual < ts_anterior:
                return True

            ts_anterior = ts_actual
            ultima_ts = ts_actual

        duracion = (ultima_ts - primera_ts).total_seconds()
        if duracion < 2:
            return True

        return False

def mover_a_anomalias(filepath):
    carpeta_actual = os.path.dirname(filepath)
    carpeta_anomalias = os.path.join(carpeta_actual, 'anomalias')
    os.makedirs(carpeta_anomalias, exist_ok=True)
    destino = os.path.join(carpeta_anomalias, os.path.basename(filepath))
    shutil.move(filepath, destino)
    return destino

def main():
    if len(sys.argv) != 2:
        print("Uso: python verificar_archivo.py <archivo_csv>", file=sys.stderr)
        sys.exit(2)

    ruta_csv = sys.argv[1]

    if not os.path.isfile(ruta_csv):
        print(f"El archivo no existe: {ruta_csv}", file=sys.stderr)
        sys.exit(2)

    try:
        es_anomalo = procesar_csv(ruta_csv)
        if es_anomalo:
            nueva_ruta = mover_a_anomalias(ruta_csv)
            print(nueva_ruta)
            sys.exit(1)  # Indica anómalo y movido
        else:
            print(ruta_csv)
            sys.exit(0)  # Correcto, sin mover
    except Exception as e:
        print(f"Error al procesar {ruta_csv}: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()


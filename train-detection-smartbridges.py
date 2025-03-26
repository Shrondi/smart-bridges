# -*- coding: utf-8 -*-

import mysql.connector
import pandas as pd
import numpy as np
import os
import argparse
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor

"""# Definición funciones

"""## Funciones BD"""

def conectar_db(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE):
    try:
        db = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        db.start_transaction()
        cursor = db.cursor()
        return db, cursor
    except Exception as e:
        print(e)
        return None, None
    

def get_peak_data(name_bridge, datetime_init, datetime_end, UMBRAL_VIBRACION, cursor):
    """Obtener el timestap de aquellas muestras cuya magnitud de aceleración supere un cierto umbral"""
    querystr = f"""
      WITH train AS (
        SELECT
            DATE_ADD(
              time_begin,
              INTERVAL
              (
                  CASE
                      WHEN ca.frequency = 0 THEN 1/31.25
                      ELSE 1/125
                  END
              ) * (
                  SELECT COUNT(*)
                  FROM data_accelerometer da_inner
                  WHERE da_inner.data_block = da.data_block
                  AND da_inner.id <= da.id
              ) - 1
              SECOND_MICROSECOND
          ) AS incrementada,

            SQRT(
                POW(axisx * (POW(2, scale + 2) / POW(2,  word_bits)), 2) +
                POW(axisy * (POW(2, scale + 2) / POW(2,  word_bits)), 2) +
                POW(axisz * (POW(2, scale + 2) / POW(2,  word_bits)), 2)
            ) AS mag_acel

        FROM
          data_block db
          JOIN data_accelerometer da ON da.data_block = db.id
          JOIN configuration_acc ca ON ca.id = db.configuration
          JOIN accelerometer ac ON ac.sensor = ca.accelerometer
          JOIN chip ch ON ch.id = ac.chip
          JOIN sensors s ON s.id = ac.sensor
          JOIN devices d ON d.id = s.device
          JOIN bridges b ON b.id = d.bridge
        WHERE time_begin BETWEEN '{datetime_init}' AND '{datetime_end}'
        AND b.name = '{name_bridge}'
        )

        SELECT incrementada
        FROM train
        WHERE mag_acel >= {UMBRAL_VIBRACION}
    """
    cursor.execute(querystr)
    return cursor.fetchall()


def get_data_train(name_bridge, start_time, end_time, cursor):
    querystr = f"""
        SELECT
            DATE_ADD(
              time_begin,
              INTERVAL
              (
                  CASE
                      WHEN ca.frequency = 0 THEN 1/31.25
                      ELSE 1/125
                  END
              ) * (
                  SELECT COUNT(*)
                  FROM data_accelerometer da_inner
                  WHERE da_inner.data_block = da.data_block
                  AND da_inner.id <= da.id
              ) - 1
              SECOND_MICROSECOND
          ) AS incrementada,

          axisx * (POW(2, scale+2) / POW(2, word_bits)) AS x,
          axisy * (POW(2, scale+2) / POW(2, word_bits)) AS y,
          axisz * (POW(2, scale+2) / POW(2, word_bits)) AS z,
          db.accelerometer as accelerometer

        FROM
          data_block db
          STRAIGHT_JOIN data_accelerometer da ON da.data_block = db.id
          STRAIGHT_JOIN configuration_acc ca ON ca.id = db.configuration
          STRAIGHT_JOIN accelerometer ac ON ac.sensor = ca.accelerometer
          STRAIGHT_JOIN chip ch ON ch.id = ac.chip
          STRAIGHT_JOIN sensors s ON s.id = ac.sensor
          STRAIGHT_JOIN devices d ON d.id = s.device
          STRAIGHT_JOIN bridges b ON b.id = d.bridge
        WHERE time_begin BETWEEN '{start_time}' AND '{end_time}'
        AND b.name = '{name_bridge}'
    """
    cursor.execute(querystr)
    return cursor.fetchall()


"""## Funciones lógica"""

def save_train(args):
    train_number, (start_time, end_time), output_dir, db_config = args

    try:
        db, cursor = conectar_db(db_config['host'], db_config['user'], db_config['password'], db_config['database'])

        train_data = get_data_train(output_dir, start_time, end_time, cursor)

    except Exception as e:
        print(e)
        sys.exit(1)

    finally:
        if cursor:
            cursor.close()

        if db:
            db.close()

    train_data = pd.DataFrame(train_data, columns=['datetime', 'x', 'y', 'z', 'accelerometer'])
    train_data.set_index('datetime', inplace=True)
    train_data.sort_values('datetime', inplace=True)

    # Seleccionar las columnas deseadas
    train_data = train_data[['x', 'y', 'z', 'accelerometer']]

    # Create file name with bridge name and time range
    start_time_str = start_time.strftime("%Y%m%d_%H%M%S")
    end_time_str = end_time.strftime("%Y%m%d_%H%M%S")
    day_folder = start_time.strftime("%Y-%m-%d")

    day_dir = os.path.join(output_dir, day_folder)
    os.makedirs(day_dir, exist_ok=True)

    filename = os.path.join(day_dir, f"{output_dir}-{start_time_str}-{end_time_str}.csv")
    
    with open(filename, 'w') as f:
        train_data.to_csv(f, index=True, index_label='datetime')  # Write data

    print(f"Tren {train_number} guardado en {filename}")

def parallelise_save_trains(trains, name_bridge, db_config, output_dir=None):
    """Guarda los datos de las vibraciones de cada tren en archivos CSV en paralelo.
        Cada archivo CSV contiene las columnas 'datetime', 'x', 'y', 'z' y 'accelerometer'.

    Args:
        trains: Diccionario indexado por un numero de tren y con los valores de [start_time, end_time].
        name_bridge: Nombre del puente para la consulta de los datos.
        output_dir: Directorio de salida para los archivos CSV. Por defecto es el nombre del puente.
    """
    if output_dir is None:
        output_dir = name_bridge

    os.makedirs(output_dir, exist_ok=True)

    # Preparar argumentos para la función save_train
    args_list = [(train_number, (start_time, end_time), output_dir, db_config) for train_number, (start_time, end_time) in trains.items()]

    with ProcessPoolExecutor() as executor:
        executor.map(save_train, args_list)


def isolate_trains(df, WINDOWS_SECONDS_START, WINDOWS_SECONDS_END):
    """ Aislamiento de trenes

    Args:
        df: DataFrame con los datos [datetime] de franjas de tiempo donde se detectan vibraciones más altas.
        WINDOWS_SECONS: Una ventana de tiempo en segundos para agregar al inicio y al final de cada tren

    Returns:
        Un diccionario indexado por el número de tren y
        los valores son una lista con [timestamp_inicio, timestamp_fin] de cada tren.
    """

    train_number = 1
    trains = {}  # Diccionario para almacenar los timestamps de inicio y fin de cada tren
    current_train_timestamp = []  # Lista para almacenar el timestamp del tren actual

    for i in range(len(df)):
        current_train_timestamp.append(df.index[i])  # Agregar el timestamp actual

        # Verificar si la siguiente timestamp pertenece al mismo tren
        if i + 1 < len(df):
            time_diff = df.index[i + 1] - df.index[i]
            if time_diff >= pd.Timedelta(seconds=5): # Nuevo tren detectado: Suponemos que los trenes estan espaciados por al menos 5 segundos
                
                # Agregar WINDOWS_SECONS al inicio y al final del tren
                start_time = current_train_timestamp[0] - pd.Timedelta(seconds=WINDOWS_SECONDS_START)
                end_time = current_train_timestamp[-1] + pd.Timedelta(seconds=WINDOWS_SECONDS_END)

                trains[train_number] = [start_time, end_time]  # Guardar el timestamp de inicio y fin

                train_number += 1

                current_train_timestamp = []  # Reiniciar la lista para el nuevo tren

    # Guardar el último tren si hay datos aun en la lista
    if current_train_timestamp:

        start_time = current_train_timestamp[0] - pd.Timedelta(seconds=WINDOWS_SECONDS_START)
        end_time = current_train_timestamp[-1] + pd.Timedelta(seconds=WINDOWS_SECONDS_END)

        trains[train_number] = [start_time, end_time]

    return trains

def get_data(args):
    """Función para ejecutar la consulta de recogida de datos de los trenes en paralelo."""
    name_bridge, start_time, end_time, threshold, db_config = args

    try:
        # Crear una nueva conexión y cursor para cada hilo
        db, cursor = conectar_db(db_config['host'], db_config['user'], db_config['password'], db_config['database'])

        df = get_peak_data(name_bridge, start_time, end_time, threshold, cursor)

        print(f"\t - Intervalo: {start_time} - {end_time}, Muestras recuperadas: {len(df) if df else 'None'}")

        return pd.DataFrame(df, columns=['datetime']) if df else pd.DataFrame()

    except Exception as e:
        print(f"Error en intervalo {start_time} - {end_time}: {e}")
        sys.exit(1)

    finally:
        if db:
            db.close()
        if cursor:
            cursor.close()

def parallelise_get_data(name_bridge, start_time, end_time, threshold, db_config):
    """Obtener paralelamente los datos de los trenes en el intervalo de tiempo especificado para un puente"""
    
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Generar intervalos de 2 horas
    interval_duration = timedelta(hours=2)
    intervals = []
    current_time = start_time

    while current_time < end_time:
        next_time = min(current_time + interval_duration, end_time)
        intervals.append((name_bridge, current_time, next_time, threshold, db_config))
        current_time = next_time

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(get_data, intervals))

    # Eliminar los dataframes vacios antes de concatenar
    results = [df for df in results if not df.empty]

    return pd.concat(results).sort_index() if results else pd.DataFrame()


def main():

    VERSION = "2.3.0"

    """# Variables"""
    parser = argparse.ArgumentParser(description="Algoritmo para la detección de vibraciones de trenes en acelerómetros sobre puentes")
    
    # Añadir la opción --version para mostrar la versión
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')

    # Definir los argumentos
    parser.add_argument('--host', type=str, default='localhost', help='Dirección del host de la base de datos')
    parser.add_argument('--database', type=str, default='smartbridges', help='Nombre de la base de datos')
    parser.add_argument('--user', type=str, required=True, help='Usuario de la base de datos')
    parser.add_argument('--password', type=str, required=True, help='Contraseña de la base de datos')
    
    parser.add_argument('--start_time', type=str, required=True, help='Fecha y hora de inicio (formato: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, required=True, help='Fecha y hora de fin (formato: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--name_bridge', type=str, required=True, help='Nombre del puente')
    parser.add_argument('--threshold', type=float, default=0.995, help='Umbral de vibración (por defecto: 0.995)')
    parser.add_argument('--windows_seconds_start', type=float, default=1, help='Ventana de tiempo inicial extra por cada tren (por defecto: 1)')
    parser.add_argument('--windows_seconds_end', type=float, default=1.5, help='Ventana de tiempo final extra por cada tren (por defecto: 1.5)')
    parser.add_argument('--output', type=str, help='Directorio de salida para los archivos CSV (por defecto: nombre del puente)')

    # Parsear los argumentos
    args = parser.parse_args()

    # Configuración de la base de datos
    db_config = {
        'host': args.host,
        'user': args.user,
        'password': args.password,
        'database': args.database
    }

    FECHA_HORA_INICIO = args.start_time
    FECHA_HORA_FIN = args.end_time
    NAME_BRIDGE = args.name_bridge
    THERESHOLD = args.threshold
    WINDOWS_SECONDS_START = args.windows_seconds_start
    WINDOWS_SECONDS_END = args.windows_seconds_end
    OUTPUT_DIR = args.output

    """# Procesamiento de datos"""
    print("* Variables: ")
    print("\t- Fecha y hora de inicio:", FECHA_HORA_INICIO)
    print("\t- Fecha y hora de fin:", FECHA_HORA_FIN)
    print("\t- Puente:", NAME_BRIDGE)

    print("\n * Obteniendo datos de los trenes...")
    data = parallelise_get_data(NAME_BRIDGE, FECHA_HORA_INICIO, FECHA_HORA_FIN, THERESHOLD, db_config)

    print("* Aislando trenes...", end="", flush=True)
    trains = isolate_trains(data, WINDOWS_SECONDS_START, WINDOWS_SECONDS_END)
    print("ok")

    print("* Guardando trenes...")
    parallelise_save_trains(trains, NAME_BRIDGE, db_config, OUTPUT_DIR)
    
    print("* Proceso finalizado para los trenes del puente", NAME_BRIDGE, "en el intervalo", FECHA_HORA_INICIO, "-", FECHA_HORA_FIN)

if __name__ == "__main__":
    main()

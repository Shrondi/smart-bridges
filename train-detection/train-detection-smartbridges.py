# -*- coding: utf-8 -*-

import mysql.connector
import pandas as pd
import os
import argparse
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor
import sys
import shutil

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


def get_stat_data(name_bridge, datetime_init, datetime_end, cursor):
    """Obtener el timestap de aquellas muestras cuya magnitud de aceleración supere un cierto umbral"""
    querystr = f"""\
    SELECT
          time_begin,
          AVG(axisx) AS avg_x,
          AVG(axisy) AS avg_y,
          AVG(axisz) AS avg_z
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
      GROUP BY da.data_block
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
    (start_time, end_time), train_output_dir, name_bridge, db_config = args

    try:
        db, cursor = conectar_db(db_config['host'], db_config['user'], db_config['password'], db_config['database'])

        train_data = get_data_train(name_bridge, start_time, end_time, cursor)

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

    filename = os.path.join(train_output_dir, f"{name_bridge}-{start_time_str}-{end_time_str}.csv")
    
    with open(filename, 'w') as f:
        train_data.to_csv(f, index=True, index_label='datetime')  # Write data

    print(f"\t Datos guardados en {filename}")


def parallelise_save_trains(trains, name_bridge, db_config, output_dir=None):
    """Guarda los datos de las vibraciones de cada tren en archivos CSV en paralelo.
        Cada archivo CSV contiene las columnas 'datetime', 'x', 'y', 'z' y 'accelerometer'.

    Args:
        trains: Diccionario con los valores de [start_time, end_time].
        name_bridge: Nombre del puente para la consulta de los datos.
        output_dir: Directorio de salida para los archivos CSV. Por defecto es el nombre del puente.
    """
    if output_dir is None:
        output_dir = name_bridge

    os.makedirs(output_dir, exist_ok=True)

    # Crear carpetas por cada día diferente en los trenes
    unique_days = set(start_time.date() for start_time, _ in trains.values())
    for day in unique_days:
        day_dir = os.path.join(output_dir, day.strftime("%Y-%m-%d"))

        os.makedirs(day_dir, exist_ok=True)

    # Preparar argumentos para la función save_train
    args_list = [
        (
            (start_time, end_time),
            os.path.join(output_dir, start_time.date().strftime("%Y-%m-%d")),
            name_bridge,
            db_config
        )
        for start_time, end_time in trains.values()
    ]

    with ProcessPoolExecutor() as executor:
        executor.map(save_train, args_list)


def isolate_trains(df, WINDOWS_SECONDS_START, WINDOWS_SECONDS_END):
    """ Aislamiento de trenes

    Args:
        df: DataFrame con los datos [datetime] de franjas de tiempo donde se detectan vibraciones más altas.
        WINDOWS_SECONS: Una ventana de tiempo en segundos para agregar al inicio y al final de cada tren

    Returns:
        Un diccionario con los valores de [timestamp_inicio, timestamp_fin] de cada tren.
    """

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    trains = []  # Lista para almacenar los timestamps de inicio y fin de cada tren
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

                trains.append((start_time, end_time))  # Guardar el timestamp de inicio y fin

                current_train_timestamp = []  # Reiniciar la lista para el nuevo tren

    # Guardar el último tren si hay datos aun en la lista
    if current_train_timestamp:

        start_time = current_train_timestamp[0] - pd.Timedelta(seconds=WINDOWS_SECONDS_START)
        end_time = current_train_timestamp[-1] + pd.Timedelta(seconds=WINDOWS_SECONDS_END)

        trains.append((start_time, end_time))

    return {i: train for i, train in enumerate(trains)}


def peak(args):
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



def parallelise_peak(name_bridge, start_time, end_time, threshold, db_config):
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
        results = list(executor.map(peak, intervals))

    # Eliminar los dataframes vacios antes de concatenar
    results = [df for df in results if not df.empty]

    return pd.concat(results).sort_index() if results else pd.DataFrame()



def stat(args):
    """Función para ejecutar la consulta de recogida de datos de los trenes en paralelo."""
    name_bridge, start_time, end_time, db_config = args

    try:
        # Crear una nueva conexión y cursor para cada hilo
        db, cursor = conectar_db(db_config['host'], db_config['user'], db_config['password'], db_config['database'])
        
        df = get_stat_data(name_bridge, start_time, end_time, cursor)

        print(f"\t - Intervalo: {start_time} - {end_time}, Muestras recuperadas: {len(df) if df else 'None'}")

        return pd.DataFrame(df, columns=['datetime', 'avg_x', 'avg_y', 'avg_z',]) if df else pd.DataFrame()

    except Exception as e:
        print(f"Error en intervalo {start_time} - {end_time}: {e}")
        sys.exit(1)

    finally:
        if db:
            db.close()
        if cursor:
            cursor.close()



def parallelise_stat(name_bridge, start_time, end_time, db_config):
    """Obtener paralelamente los datos de los trenes en el intervalo de tiempo especificado para un puente"""
    
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Generar intervalos de 2 horas
    interval_duration = timedelta(hours=2)
    intervals = []
    current_time = start_time

    while current_time < end_time:
        next_time = min(current_time + interval_duration, end_time)
        intervals.append((name_bridge, current_time, next_time, db_config))
        current_time = next_time

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(stat, intervals))

    # Eliminar los dataframes vacios antes de concatenar
    results = [df for df in results if not df.empty]

    return pd.concat(results).sort_index() if results else pd.DataFrame()


def filter_datablocks(data, threshold):
    """ 
    Filter datablocks outside mean ± (threshold * std) and 
    return a DataFrame with 'datetime' as index and only 'datetime' column for bad datablocks.
    """

    df = pd.DataFrame(data, columns=['datetime', 'avg_x', 'avg_y', 'avg_z'])
    df.set_index('datetime', inplace=True)
    df = df.astype(float)

    # Store all initial datetimes
    all_datetimes = df.index.tolist()

    for col in ['avg_x', 'avg_y', 'avg_z']:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - threshold * std) & (df[col] <= mean + threshold * std)]

    # Get removed datetimes
    removed_datetimes = list(set(all_datetimes) - set(df.index.tolist()))

    # Create DataFrame for bad datablocks with 'datetime' as index and only 'datetime' column
    bad_datablocks_df = pd.DataFrame(removed_datetimes, columns=['datetime'])  # DataFrame with only 'datetime'

    return bad_datablocks_df

def main():

    VERSION = "3.1.0"

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
    
    parser.add_argument('--windows_seconds_start', type=float, default=1, help='Ventana de tiempo inicial extra por cada tren (por defecto: 1)')
    parser.add_argument('--windows_seconds_end', type=float, default=1.5, help='Ventana de tiempo final extra por cada tren (por defecto: 1.5)')
    parser.add_argument('--output', type=str, help='Directorio de salida para los archivos CSV (por defecto: nombre del puente)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--threshold_vibration', type=float, default=0.995, help='Umbral de vibración (por defecto: 0.995). Válido solo en modo "peak"')
    group.add_argument('--threshold_filter', type=float, default=1.3, help='Umbral de filtro (por defecto: 1.3). Válido solo en modo "filter"')

    # Opción para seleccionar el modo de ejecución
    parser.add_argument(
        '--mode',
        choices=['peak', 'filter'],
        required=True,
        type=str,
        help='Modo de ejecución: peak (por defecto) o filter'
    )

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

    THRESHOLD_VIBRATION = args.threshold_vibration
    THRESHOLD_FILTER = args.threshold_filter

    WINDOWS_SECONDS_START = args.windows_seconds_start
    WINDOWS_SECONDS_END = args.windows_seconds_end

    OUTPUT_DIR = args.output

    """# Procesamiento de datos"""
    print("* Variables: ")
    print("\t- Puente:", NAME_BRIDGE)
    print("\t- Intervalo:", FECHA_HORA_INICIO, "-", FECHA_HORA_FIN)

    print("\n\t- Modo:", args.mode)
    if args.mode == 'peak':
        print("\t- Umbral de vibración:", THRESHOLD_VIBRATION)
    if args.mode == 'filter':
        print("\t- Umbral de filtro:", THRESHOLD_FILTER)


    print("\n * Obteniendo datos...")

    if args.mode == 'peak':
        data = parallelise_peak(NAME_BRIDGE, FECHA_HORA_INICIO, FECHA_HORA_FIN, THRESHOLD_VIBRATION, db_config)

    if args.mode == 'filter':
        data = parallelise_stat(NAME_BRIDGE, FECHA_HORA_INICIO, FECHA_HORA_FIN, db_config)


    print("* Aislando trenes...", end="", flush=True)

    if args.mode == 'filter':
        data = filter_datablocks(data, THRESHOLD_FILTER)

    trains = isolate_trains(data, WINDOWS_SECONDS_START, WINDOWS_SECONDS_END)
    print("ok")

    print("* Guardando trenes...")
    parallelise_save_trains(trains, NAME_BRIDGE, db_config, OUTPUT_DIR)
    
    print("\n* Se han guardado", len(trains), "trenes del puente", NAME_BRIDGE, "en el intervalo", FECHA_HORA_INICIO, "-", FECHA_HORA_FIN)

if __name__ == "__main__":
    main()

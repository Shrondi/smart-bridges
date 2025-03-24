# -*- coding: utf-8 -*-

import mysql.connector
import pandas as pd
import numpy as np
import argparse
import datetime

"""# Definición funciones

## Funciones BD
"""

def get_stats_datablocks(accelerometer, datetime_init, datetime_end):

  querystr = f"""\
    SELECT
          da.data_block,
          AVG(axisx) AS avg_x,
          AVG(axisy) AS avg_y,
          AVG(axisz) AS avg_z
      FROM data_accelerometer da
      JOIN data_block db ON da.data_block = db.id
      JOIN configuration_acc ca ON ca.id = db.configuration
      JOIN accelerometer ac ON ac.sensor = ca.accelerometer
      JOIN chip ch ON ch.id = ac.chip
      WHERE time_begin BETWEEN '{datetime_init}' AND '{datetime_end}'
        AND ac.sensor = {accelerometer}
      GROUP BY da.data_block
      """

  cursor.execute(querystr)
  results = cursor.fetchall()
  return results

def get_data_datablocks(datablocks):

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
          da.data_block,
          da.id,
          (POW(2, scale+2) / POW(2, word_bits)) as factor
      FROM
          data_block db
          JOIN data_accelerometer da ON da.data_block = db.id
          JOIN configuration_acc ca ON ca.id = db.configuration
          JOIN accelerometer ac ON ac.sensor = ca.accelerometer
          JOIN chip ch ON ch.id = ac.chip
      WHERE
      da.data_block IN ({datablocks})
      """

    cursor.execute(querystr)
    results = cursor.fetchall()
    return results


def filter_datablocks(data, threshold):
     # Convertir los resultados a un DataFrame
    df = pd.DataFrame(data, columns=['data_block', 'avg_x', 'avg_y', 'avg_z'])
    df.set_index('data_block', inplace=True)

    df = df.astype(float)

    # Guardar todos los data_blocks iniciales
    all_datablocks = df.index.tolist()

    for col in ['avg_x', 'avg_y', 'avg_z']:
      mean = df[col].mean()
      std = df[col].std()
      df = df[(df[col] >= mean - threshold * std) & (df[col] <= mean + threshold * std)]

    # Convertir el índice a una cadena
    good_datablocks_str = ', '.join(map(str, df.index.tolist()))

    # Obtener los data_blocks eliminados
    removed_datablocks = list(set(all_datablocks) - set(df.index.tolist()))
    removed_datablocks_str = ', '.join(map(str, removed_datablocks))

    return good_datablocks_str, removed_datablocks_str

def fix_samples_datablocks(bad_data, THRESHOLD_FIX_SAMPLES):
    """
    Analiza las muestras de los datablocks intercambiando los ejes de las muestras que lo requieran.
    
    Return:
        DataFrame con los datos arreglados
    """


    # Convertir bad_data a Pandas DataFrame
    df = pd.DataFrame(bad_data, columns=['datetime', 'x', 'y', 'z', 'data_block', 'id', 'factor'])

    # Eliminar filas con baja precisión decimal (menos de 3 decimales)
    df = df[df['x'].apply(lambda x: len(str(x).split('.')[1]) >= 3) & \
            df['y'].apply(lambda x: len(str(x).split('.')[1]) >= 3) & \
            df['z'].apply(lambda x: len(str(x).split('.')[1]) >= 3)]


    # Group by data_block and apply the correction function
    def correct_axes(group):

        for i in range(1, len(group)):
            prev_row = group.iloc[i - 1]
            row = group.iloc[i]

            # Condición de ejes intercambiados
            ejes_intercambiadosA = (
                (np.abs(float(row['x'])) >= np.abs(float(prev_row['z'])) - THRESHOLD_FIX_SAMPLES) and (np.abs(float(row['x'])) <= np.abs(float(prev_row['z'])) + THRESHOLD_FIX_SAMPLES) and
                (np.abs(float(row['y'])) >= np.abs(float(prev_row['x'])) - THRESHOLD_FIX_SAMPLES) and (np.abs(float(row['y'])) <= np.abs(float(prev_row['x'])) + THRESHOLD_FIX_SAMPLES) and
                (np.abs(float(row['z'])) >= np.abs(float(prev_row['y'])) - THRESHOLD_FIX_SAMPLES) and (np.abs(float(row['z'])) <= np.abs(float(prev_row['y'])) + THRESHOLD_FIX_SAMPLES)
            )

            ejes_intercambiadosB = (
                (np.abs(float(row['x'])) >= np.abs(float(prev_row['y'])) - THRESHOLD_FIX_SAMPLES) and (np.abs(float(row['x'])) <= np.abs(float(prev_row['y'])) + THRESHOLD_FIX_SAMPLES) and
                (np.abs(float(row['y'])) >= np.abs(float(prev_row['z'])) - THRESHOLD_FIX_SAMPLES) and (np.abs(float(row['y'])) <= np.abs(float(prev_row['z'])) + THRESHOLD_FIX_SAMPLES) and
                (np.abs(float(row['z'])) >= np.abs(float(prev_row['x'])) - THRESHOLD_FIX_SAMPLES) and (np.abs(float(row['z'])) <= np.abs(float(prev_row['x'])) + THRESHOLD_FIX_SAMPLES)
            )


            row_id = row['id']

            # Intercambiar ejes solo para la muestra actual si se detecta un intercambio
            if ejes_intercambiadosA:

                # Intercambiar X y Z
                group.loc[group.index[i], ['x', 'z']] = group.loc[group.index[i], ['z', 'x']].values
                # Intercambiar X e Y
                group.loc[group.index[i], ['x', 'y']] = group.loc[group.index[i], ['y', 'x']].values

                update_query = f"""
                UPDATE data_accelerometer
                SET axisx = {row['y']/row['factor']}, axisy = {row['z']/row['factor']}, axisz = {row['x']/row['factor']}
                WHERE id = {row_id}
                """
                cursor.execute(update_query)
                db.commit()

                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{current_time}] UPDATED TO (y, z, x). SAMPLE {row_id} - DATABLOCK {row['data_block']} - DATETIME {row['datetime']}:: Old values ({row['x']}, {row['y']}, {row['z']}), New values: ({row['y']}, {row['z']}, {row['x']})")


            elif ejes_intercambiadosB:

                # Intercambiar Y y Z
                group.loc[group.index[i], ['y', 'z']] = group.loc[group.index[i], ['z', 'y']].values
                # Intercambiar Y e X
                group.loc[group.index[i], ['x', 'y']] = group.loc[group.index[i], ['y', 'x']].values

                update_query = f"""
                UPDATE data_accelerometer
                SET axisx = {row['z']/row['factor']}, axisy = {row['x']/row['factor']}, axisz = {row['y']/row['factor']}
                WHERE id = {row_id}
                """

                cursor.execute(update_query)
                db.commit()

                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{current_time}] UPDATED TO (z, x, y). SAMPLE {row_id} - DATABLOCK {row['data_block']} - DATETIME {row['datetime']}:: Old values ({row['x']}, {row['y']}, {row['z']}), New values ({row['z']}, {row['x']}, {row['y']})")


        return group

    # Apply the correction function to each data_block group
    df = df.groupby('data_block').apply(correct_axes).reset_index(drop=True)

    return df
    
def delete_samples_datablocks(accuracy, accelerometer, datetime_init, datetime_end):
	# Eliminar datos con baja precisión decimal
    delete_query = f"""
    DELETE da FROM data_accelerometer da
    JOIN data_block db ON da.data_block = db.id
    JOIN configuration_acc ca ON ca.id = db.configuration
    JOIN accelerometer ac ON ac.sensor = ca.accelerometer
    JOIN chip ch ON ch.id = ac.chip
    WHERE (LENGTH(SUBSTR(axisx * (POW(2, scale+2) / POW(2, word_bits)), INSTR(axisx * (POW(2, scale+2) / POW(2, word_bits)), '.') + 1)) < {accuracy}
          OR LENGTH(SUBSTR(axisy * (POW(2, scale+2) / POW(2, word_bits)), INSTR(axisy * (POW(2, scale+2) / POW(2, word_bits)), '.') + 1)) < {accuracy}
          OR LENGTH(SUBSTR(axisz * (POW(2, scale+2) / POW(2, word_bits)), INSTR(axisz * (POW(2, scale+2) / POW(2, word_bits)), '.') + 1)) < {accuracy})
      AND ac.sensor = {accelerometer}
      AND db.time_begin BETWEEN '{datetime_init}' AND '{datetime_end}';
    """

    cursor.execute(delete_query)
    db.commit()

def fix_datablocks(datablock_raw, THRESHOLD_FIX_DATABLOCKS):
    """
    Arregla los datablocks intercambiando sus ejes usando la media por cada eje para
    comparar con la media del bloque anterior.

    Returns:
        Un DataFrame que representan los datos de los datablocks arreglados.
    """

    # Convertir datablock_raw a Pandas DataFrame
    df_raw = pd.DataFrame(datablock_raw, columns=['data_block', 'avg_x', 'avg_y', 'avg_z'])
    df_raw.set_index('data_block', inplace=True)

    df_raw = df_raw.astype(float)

    # Get unique data_block values
    data_blocks = df_raw.index.unique()

    # Iterate over data_blocks
    for i in range(len(data_blocks)):
        current_data_block = data_blocks[i]
        prev_data_block = data_blocks[i - 1] if i > 0 else None  # Get previous block if available

        # Get current and previous block means
        current_means = df_raw.loc[current_data_block, ['avg_x', 'avg_y', 'avg_z']]
        prev_block_means = df_raw.loc[prev_data_block, ['avg_x', 'avg_y', 'avg_z']] if prev_data_block is not None and prev_data_block in df_raw.index else None

        # Compara las medias con las del bloque anterior (si está disponible)
        if prev_block_means is not None:
            # Condición de ejes intercambiados (utilizando la media)
            ejes_intercambiadosA = (
                (np.abs(float(current_means['avg_x'])) >= np.abs(float(prev_block_means['avg_z'])) - THRESHOLD_FIX_DATABLOCKS) and (np.abs(float(current_means['avg_x'])) <= np.abs(float(prev_block_means['avg_z'])) + THRESHOLD_FIX_DATABLOCKS) and
                (np.abs(float(current_means['avg_y'])) >= np.abs(float(prev_block_means['avg_x'])) - THRESHOLD_FIX_DATABLOCKS) and (np.abs(float(current_means['avg_y'])) <= np.abs(float(prev_block_means['avg_x'])) + THRESHOLD_FIX_DATABLOCKS) and
                (np.abs(float(current_means['avg_z'])) >= np.abs(float(prev_block_means['avg_y'])) - THRESHOLD_FIX_DATABLOCKS) and (np.abs(float(current_means['avg_z'])) <= np.abs(float(prev_block_means['avg_y'])) + THRESHOLD_FIX_DATABLOCKS)
            )

            ejes_intercambiadosB = (
                (np.abs(float(current_means['avg_x'])) >= np.abs(float(prev_block_means['avg_y'])) - THRESHOLD_FIX_DATABLOCKS) and (np.abs(float(current_means['avg_x'])) <= np.abs(float(prev_block_means['avg_y'])) + THRESHOLD_FIX_DATABLOCKS) and
                (np.abs(float(current_means['avg_y'])) >= np.abs(float(prev_block_means['avg_z'])) - THRESHOLD_FIX_DATABLOCKS) and (np.abs(float(current_means['avg_y'])) <= np.abs(float(prev_block_means['avg_z'])) + THRESHOLD_FIX_DATABLOCKS) and
                (np.abs(float(current_means['avg_z'])) >= np.abs(float(prev_block_means['avg_x'])) - THRESHOLD_FIX_DATABLOCKS) and (np.abs(float(current_means['avg_z'])) <= np.abs(float(prev_block_means['avg_x'])) + THRESHOLD_FIX_DATABLOCKS)
            )

            # Intercambiar los valores medios si se detecta un intercambio
            if ejes_intercambiadosA:
                # Intercambiar X y Z
                df_raw.loc[current_data_block, ['avg_x', 'avg_z']] = df_raw.loc[current_data_block, ['avg_z', 'avg_x']].values
                # Intercambiar X e Y (ahora Y es el original X)
                df_raw.loc[current_data_block, ['avg_x', 'avg_y']] = df_raw.loc[current_data_block, ['avg_y', 'avg_x']].values

                update_query = f"""
                UPDATE data_accelerometer AS d
                JOIN (
                    SELECT id, axisx AS old_axisx, axisy AS old_axisy, axisz AS old_axisz
                    FROM data_accelerometer
                    WHERE data_block = {current_data_block}
                ) AS temp
                ON d.id = temp.id
                SET d.axisx = temp.old_axisy,
                    d.axisy = temp.old_axisz,
                    d.axisz = temp.old_axisx
                WHERE d.data_block = {current_data_block};

                """

                # Ejecutar la consulta SQL
                cursor.execute(update_query)
                db.commit()  # Confirmar los cambios en la base de datos

                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{current_time}] UPDATED DATABLOCK {current_data_block} TO (y, z, x)")


            elif ejes_intercambiadosB:
                # Intercambiar Y y Z
                df_raw.loc[current_data_block, ['avg_y', 'avg_z']] = df_raw.loc[current_data_block, ['avg_z', 'avg_y']].values
                # Intercambiar X e Y (ahora Y es el original X)
                df_raw.loc[current_data_block, ['avg_x', 'avg_y']] = df_raw.loc[current_data_block, ['avg_y', 'avg_x']].values

                update_query = f"""
                UPDATE data_accelerometer AS d
                JOIN (
                    SELECT id, axisx AS old_axisx, axisy AS old_axisy, axisz AS old_axisz
                    FROM data_accelerometer
                    WHERE data_block = {current_data_block}
                ) AS temp
                ON d.id = temp.id
                SET d.axisx = temp.old_axisz,
                    d.axisy = temp.old_axisx,
                    d.axisz = temp.old_axisy
                WHERE d.data_block = {current_data_block};

                """

                # Ejecutar la consulta SQL
                cursor.execute(update_query)
                db.commit()  # Confirmar los cambios en la base de datos

                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{current_time}] UPDATED DATABLOCK {current_data_block} TO (z, x, y)")

    return df_raw  # Return the modified DataFrame


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
        print("ok")
        return db, cursor
    except Exception as e:
        print(e)
        return None
    
def main():

    VERSION = "2.0.0"

    """# Variables"""
    parser = argparse.ArgumentParser(description="Procesamiento de los datos de acelerómetros para corregir el problema del intercambio de los ejes en las muestras y datablocks \
                                                  erróneos a través de un filtrado para un posterior arreglo de los mismos en la base de datos")
    
    # Añadir la opción --version para mostrar la versión
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')

    # Definir los argumentos
    parser.add_argument('--host', type=str, default='localhost', help='Dirección del host de la base de datos')
    parser.add_argument('--database', type=str, default='smartbridges', help='Nombre de la base de datos')
    parser.add_argument('--user', type=str, required=True, help='Usuario de la base de datos')
    parser.add_argument('--password', type=str, required=True, help='Contraseña de la base de datos')
    
    parser.add_argument('--start_time', type=str, required=True, help='Fecha y hora de inicio de los datos (formato: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, required=True, help='Fecha y hora de fin de los datos (formato: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--accelerometer', type=int, required=True, help='ID del acelerómetro')
    parser.add_argument('--filter_threshold', type=int, default=2, help='Threshold para filtrar los datablocks (2 por defecto)')
    parser.add_argument('--fix_datablocks_threshold', type=int, default=50, help='Threshold para arreglar los datablocks (50 por defecto)')
    parser.add_argument('--fix_samples_threshold', type=float, default=0.2, help='Threshold para arreglar las muestras (0.2 por defecto)')
    
    # Opción para seleccionar qué ejecutar
    parser.add_argument(
        '--run',
        choices=['samples', 'datablocks', 'all'],
        default='all',
        help="Modo ejecución: 'samples' solo muestras, 'datablocks' solo bloques de datos, 'all' para todo."
    )

    # Parsear los argumentos
    args = parser.parse_args()

    # Asignar los valores de los argumentos a las variables
    MYSQL_HOST = args.host
    MYSQL_DATABASE = args.database
    MYSQL_USER = args.user
    MYSQL_PASSWORD = args.password

    FECHA_HORA_INICIO = args.start_time
    FECHA_HORA_FIN = args.end_time
    ACELEROMETRO = args.accelerometer
    THRESHOLD_FILTER = args.filter_threshold                    # Se usa para filtrar los datablocks buenos de los erroneos en la funcion filter_datablocks: media ± (threshold * desviación estándar). Elegir entre [1.5, 2, 2.5, 3]
    THRESHOLD_FIX_DATABLOCKS = args.fix_datablocks_threshold    # Diferencia entre medias de los ejes de los datablocks (Las medias estan calculadas sobre los datos raw de los sensores). Valores enteros de 50 o mas
    THRESHOLD_FIX_SAMPLES = args.fix_samples_threshold          # Diferencia entre aceleraciones de los ejes entre distintas muestras dentro de un mismo datablock. Valores flotantes (recomendado 0.2)

    """# Conexion DB"""
    global cursor, db

    print("* Conectando a la base de datos...", end="")
    db, cursor = conectar_db(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE)   

    """# Procesamiento de datos"""
    print("* DATOS: ")
    print("\t- Fecha y hora de inicio:", FECHA_HORA_INICIO)
    print("\t- Fecha y hora de fin:", FECHA_HORA_FIN)
    print("\t- Acelerómetro:", ACELEROMETRO)
    print("\t- Threshold para filtrar datablocks:", THRESHOLD_FILTER)
    print("\t- Threshold para arreglar datablocks:", THRESHOLD_FIX_DATABLOCKS)
    print("\t- Threshold para arreglar muestras:", THRESHOLD_FIX_SAMPLES)

    if args.run == 'samples' or args.run == 'all':
    
        print("* Obteniendo stats de los datablocks... ", end="", flush=True)
        datablock_stats = get_stats_datablocks(ACELEROMETRO, FECHA_HORA_INICIO, FECHA_HORA_FIN) # Obtener la media de cada eje por datablock
        print("ok")

        print("* Filtrando datablocks... ", end="", flush=True)
        _, str_datablocks_bad = filter_datablocks(datablock_stats, THRESHOLD_FILTER) # Filtrar datablocks que estén fuera de media ± (threshold * desviación estándar)
        print("ok")

        print("* Obteniendo datablocks erroneos... ", end="", flush=True)
        bad_data = get_data_datablocks(str_datablocks_bad) # Obtener los datos de los datablocks para arreglar
        print("ok")
        
        # Eliminar muestras erroneas
        print("* Eliminando muestras erróneas... ", end="", flush=True)
        delete_samples_datablocks(3, ACELEROMETRO, FECHA_HORA_INICIO, FECHA_HORA_FIN)
        print("ok")

        print("* Arreglando muestras de los datablocks erroneos...")
        # Se limpian aquellos datablocks que tenian solo algunas muestras erroneas o cambiados los ejes
        fix_samples_datablocks(bad_data, THRESHOLD_FIX_SAMPLES, ACELEROMETRO, FECHA_HORA_INICIO, FECHA_HORA_FIN) # Datablocks mejorados
        print("* Arreglando muestras de los datablocks erroneos... ok")
        
        
    
    if args.run == 'datablocks' or args.run == 'all':

        print("* Obteniendo datablocks... ", end="", flush=True)
        datablock_stats = get_stats_datablocks(ACELEROMETRO, FECHA_HORA_INICIO, FECHA_HORA_FIN) # Obtener nuevamente los datablocks para recalcular las nuevas medias
        print("ok")
        
        print("* Arreglando datablocks...")
        fix_datablocks(datablock_stats, THRESHOLD_FIX_DATABLOCKS) # Arreglar datablocks que todas sus muestras tengan los ejes intercambiados
        print("* Arreglando datablocks... ok")

    cursor.close()
    db.close()

    print("* PROCESO FINALIZADO. VARIABLES USADAS:")
    print("\t- Fecha y hora de inicio:", FECHA_HORA_INICIO)
    print("\t- Fecha y hora de fin:", FECHA_HORA_FIN)
    print("\t- Acelerómetro:", ACELEROMETRO)
    print("\t- Threshold para filtrar datablocks:", THRESHOLD_FILTER)
    print("\t- Threshold para arreglar datablocks:", THRESHOLD_FIX_DATABLOCKS)
    print("\t- Threshold para arreglar muestras:", THRESHOLD_FIX_SAMPLES)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import argparse
import calendar
import locale
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo (solo para escribir en archivos)
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType (mejor compatibilidad)
matplotlib.rcParams['pdf.use14corefonts'] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft, fftfreq
from PyPDF2 import PdfReader, PdfWriter
import io
import tempfile

# ====================
#  FUNCIONES AUXILIARES DE RUTAS Y FECHAS
# ====================

def parse_timestamp_from_filename(filename):
    """
    Extrae la hora del nombre del archivo tipo: acceleration_12-34-56.csv
    """
    match = re.match(r"acceleration_(\d{2})-(\d{2})-(\d{2})\.csv", filename)
    if match:
        h, m, s = match.groups()
        return datetime.strptime(f"{h}:{m}:{s}", "%H:%M:%S")
    return None

def get_date(date_str):
    """
    Devuelve (year, month_number, day, month_name) a partir de una fecha YYYYMMDD.
    """
    year = date_str[:4]
    month_number = int(date_str[4:6])
    day = date_str[6:]
    try:
        locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    except locale.Error:
        pass
    month_name = calendar.month_name[month_number].lower()
    return year, month_name, month_number, day

def get_report_folder_path(bridge_path, date_str):
    """Genera la ruta al archivo PDF de salida."""
    year, month_name, _, day = get_date(date_str)
    return os.path.join(bridge_path, 'report', year, month_name, day, f"train_report_{date_str}.pdf")

def get_raw_folder_path(bridge_path, date_str):
    """Genera la ruta a la carpeta de datos para una fecha específica."""
    year, month_name, _, day = get_date(date_str)
    return os.path.join(bridge_path, 'raw', year, month_name, day)

def validate_hora_format(hora):
    """Valida que la hora tenga formato HH:MM:SS"""
    return bool(re.match(r"^\d{2}:\d{2}:\d{2}$", hora))

# ====================
#  FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ====================

def load_data(input_path):
    """Loads and processes a single CSV file."""
    print(f"Cargando datos: {input_path}")
    try:
        # Cargar el archivo CSV y establecer 'timestamp' como índice
        df = pd.read_csv(input_path, index_col='timestamp', 
                         parse_dates=['timestamp'], 
                         date_format='%H:%M:%S.%f', 
                         engine='c', on_bad_lines='error')

        # Verificar si las columnas esperadas existen
        if 'timestamp' not in df.index.names:
            print(f"[x] El archivo {input_path} no tiene la columna 'timestamp' como índice. Se omite.")
            return pd.DataFrame()
        
        if list(df.columns) != ['x_accel (g)', 'y_accel (g)', 'z_accel (g)']:
            print(f"[x] El archivo {input_path} no tiene las columnas esperadas. Se omite.")
            return pd.DataFrame()
        
        # Renombrar columnas para que coincidan con el resto del código
        df.rename(columns={
            'x_accel (g)': 'x',
            'y_accel (g)': 'y',
            'z_accel (g)': 'z'
        }, inplace=True)
        
        df.sort_index(inplace=True)

    except Exception as e:
        print(f"Error al procesar el archivo {input_path}: {e}")
        return pd.DataFrame()

    return df

def calc_offsets(df):
    """Calcula y aplica offsets a los datos de aceleración."""

    offsets = {}  # Diccionario

    # Calcular y aplicar offset para cada acelerómetro
    for acc_num in df['accelerometer'].unique():
        df_acc = df[df['accelerometer'] == acc_num]

        # Calcular la media de cada eje para este acelerómetro
        mean_x = df_acc['x'].mean()
        mean_y = df_acc['y'].mean()
        mean_z = df_acc['z'].mean()

        # Ajustar los datos restando la media y sumando 1 en Z
        df.loc[df['accelerometer'] == acc_num, 'x'] = df_acc['x'] - mean_x
        df.loc[df['accelerometer'] == acc_num, 'y'] = df_acc['y'] - mean_y
        df.loc[df['accelerometer'] == acc_num, 'z'] = df_acc['z'] - mean_z + 1

        # Guardar los offsets calculados
        offsets[acc_num] = {'x': -mean_x, 'y': -mean_y, 'z': -(mean_z - 1)}

    return offsets


def fft_acc(args):
    acc_num, df = args
    df_acc = df[df['accelerometer'] == acc_num]

    L = len(df_acc)
    if L == 0:
        return acc_num, None

    # Intervalo de muestreo (Ts) en segundos
    t = df_acc.index.to_numpy()
    t = (t - t[0]) / np.timedelta64(1, 's')
    Ts = np.mean(np.diff(t))

    # FFT con todos los núcleos (-1)
    fft_x = fft(df_acc['x'].values, workers=-1)
    fft_y = fft(df_acc['y'].values, workers=-1)
    fft_z = fft(df_acc['z'].values, workers=-1)

    # Eliminar componente DC de Z
    fft_z[0] = 0

    frequencies = fftfreq(L, Ts)

    return acc_num, {
        'frequencies': frequencies[:L // 2],
        'fft_x': fft_x[:L // 2] * 2,
        'fft_y': fft_y[:L // 2] * 2,
        'fft_z': fft_z[:L // 2] * 2,
    }

def calc_fft(df):
    fft_data = {}
    accelerometers = df['accelerometer'].unique()

    # Empaquetar argumentos para pasar el DataFrame a cada proceso
    args = [(acc_num, df) for acc_num in accelerometers]

    with ProcessPoolExecutor(max_workers=None) as executor:
        results = executor.map(fft_acc, args)
        for acc_num, data in results:
            if data is not None:
                fft_data[acc_num] = data
                
    return fft_data


# ====================
#  FUNCIONES DE VISUALIZACIÓN
# ====================

def create_color_mapping(accelerometers):
    """Crea un mapeo de colores consistente para cada acelerómetro."""
    sorted_accs = sorted(accelerometers)  # Orden fijo
    num_accelerometers = len(sorted_accs)

    # Fijar semilla para paleta determinista (opcional si usas 'bright', que ya es fija)
    palette = sns.color_palette('bright', n_colors=num_accelerometers)

    colors = {
        acc_num: {
            'x': palette[i],
            'y': palette[(i + 1) % num_accelerometers],
            'z': palette[(i + 2) % num_accelerometers]
        }
        for i, acc_num in enumerate(sorted_accs)
    }
    return colors

def configure_axes(axes, titles, xlabel, ylabel, legend_loc="lower left"):
    """Configura los ejes de las gráficas con títulos, etiquetas y leyendas."""
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc=legend_loc)
        ax.grid(which='major', linestyle='--', alpha=0.5)
        ax.grid(which='minor', linestyle=':', alpha=0.3)
        ax.autoscale()

def plot_train_data(df, offsets, axes, colors):
    """Grafica los datos de acelerómetro en los ejes proporcionados."""
    accelerometers = df['accelerometer'].unique()
    accelerometers = sorted(accelerometers)

    for acc_num in accelerometers:
        df_acc = df[df['accelerometer'] == acc_num][['x', 'y', 'z']]

        time_diff = (df_acc.index - df_acc.index[0]).to_series().dt.total_seconds()

        axes[0].plot(time_diff, df_acc['x'], label=f'Acel. {acc_num} (Offset: {offsets[acc_num]["x"]:.4f})', color=colors[acc_num]['x'])
        axes[1].plot(time_diff, df_acc['y'], label=f'Acel. {acc_num} (Offset: {offsets[acc_num]["y"]:.4f})', color=colors[acc_num]['y'])
        axes[2].plot(time_diff, df_acc['z'], label=f'Acel. {acc_num} (Offset: {offsets[acc_num]["z"]:.4f})', color=colors[acc_num]['z'])

    configure_axes(axes, ['Aceleración X', 'Aceleración Y', 'Aceleración Z'], 'Tiempo [s]', 'Aceleración [g]')

def plot_fft(fft_data, axes, colors):
    """Grafica los datos FFT en los ejes proporcionados."""
    accelerometers = list(fft_data.keys())
    if not accelerometers:
        return

    bar_width = 0.8 / len(accelerometers)

    for i, acc_num in enumerate(accelerometers):
        offset = bar_width * i - bar_width * (len(accelerometers) - 1) / 2

        frequencies = fft_data[acc_num]['frequencies']

        fft_x = np.abs(fft_data[acc_num]['fft_x'])
        fft_y = np.abs(fft_data[acc_num]['fft_y'])
        fft_z = np.abs(fft_data[acc_num]['fft_z'])

        axes[0].bar(frequencies + offset, fft_x, width=bar_width, label=f'Acel. {acc_num}', color=colors[acc_num]['x'])
        axes[1].bar(frequencies + offset, fft_y, width=bar_width, label=f'Acel. {acc_num}', color=colors[acc_num]['y'])
        axes[2].bar(frequencies + offset, fft_z, width=bar_width, label=f'Acel. {acc_num}', color=colors[acc_num]['z'])

    configure_axes(axes, ['FFT Aceleración X', 'FFT Aceleración Y', 'FFT Aceleración Z'], 'Frecuencia (Hz)', 'Amplitud', 'upper right')

# ====================
#  FUNCIONES DE PROCESAMIENTO DE GRUPOS
# ====================

def process_file_group(file_group, str_date, train_number):
    """
    Procesa un grupo de archivos CSV con el mismo timestamp (± segundos) y genera una gráfica combinada.

    Args:
        file_group: Lista de tuplas (sensor_name, filepath)
        day: Día del mes
        month_number: Número del mes
        train_number: Identificador del tren

    Returns:
        fig: Figura de matplotlib con la gráfica generada
    """

    _, _, month_number, day = get_date(str_date)
    combined_df = pd.DataFrame()

    for sensor, filepath in file_group:
        df = load_data(filepath)
        if df.empty:
            print(f"Sin datos en: {filepath}")
            continue

        df['accelerometer'] = sensor
        combined_df = pd.concat([combined_df, df])

    if combined_df.empty:
        print("Todos los archivos del grupo estaban vacíos.")
        return

    # Además validar si el índice tiene valores válidos
    if combined_df.index.isnull().all():
        print("Índice de timestamps vacío o inválido en grupo de archivos.")
        return
    
    combined_df.sort_index(inplace=True)

    offsets = calc_offsets(combined_df)
    fft_data = calc_fft(combined_df)
    sensors = combined_df['accelerometer'].unique()
    colors = create_color_mapping(sensors)

    fig, axes = plt.subplots(3, 2, figsize=(20, 20), gridspec_kw={'width_ratios': [3, 3]})
    
    first_datetime = combined_df.index[0]

    figure_title = f"Train {train_number + 1} - {day}/{month_number} {first_datetime.strftime('%H:%M:%S')}"
    fig.suptitle(figure_title, fontsize=20, fontweight='bold')
    fig.subplots_adjust(hspace=0.5, top=0.92)

    plot_train_data(combined_df, offsets, axes[:, 0], colors)
    plot_fft(fft_data, axes[:, 1], colors)

    plt.close(fig)
    return fig

# ====================
#  FUNCIONES DE PROCESAMIENTO DE GRUPOS Y ARCHIVOS
# ====================

def get_acceleration_files(raw_folder_path):
    """Recupera los archivos de los sensores, excluyendo anomalías."""
    sensor_files = defaultdict(list)
    for root, dirs, files in os.walk(raw_folder_path):
        if 'anomalias' in dirs:
            dirs.remove('anomalias')
        for file in files:
            if file.endswith('.csv'):
                sensor_name = os.path.basename(root)
                full_path = os.path.join(root, file)
                sensor_files[sensor_name].append(full_path)
    return sensor_files

def group_files_by_time(sensor_files, max_diff_seconds=2):
    """
    Agrupa archivos en grupos disjuntos por timestamp, de forma que
    la diferencia máxima dentro de cada grupo sea <= max_diff_seconds.
    
    Args:
        sensor_files: dict {sensor: [filepath, ...]}
        max_diff_seconds: máximo tiempo en segundos para agrupar archivos
    
    Returns:
        List of groups, cada grupo es una lista de (sensor, filepath)
    """
    entries = []

    # Parsear timestamps y preparar lista
    for sensor, files in sensor_files.items():
        for filepath in files:
            timestamp = parse_timestamp_from_filename(os.path.basename(filepath))
            if timestamp:
                entries.append((timestamp, sensor, filepath))

    # Ordenar por timestamp ascendente
    entries.sort(key=lambda x: x[0])

    groups = []
    n = len(entries)
    start = 0

    while start < n:
        group = [(entries[start][1], entries[start][2])]
        end = start + 1

        # Avanzar end mientras timestamp[end] esté dentro del rango de timestamp[start]
        while end < n and (entries[end][0] - entries[start][0]).total_seconds() <= max_diff_seconds:
            group.append((entries[end][1], entries[end][2]))
            end += 1

        groups.append(group)
        start = end  # saltamos al siguiente grupo (sin solapamiento)

    return groups

def get_groups(sensor_files, min_sensors):
    """Obtiene grupos que cumplen con el mínimo de sensores requeridos."""
    groups = group_files_by_time(sensor_files)
    return [group for group in groups if len(set(sensor for sensor, _ in group)) >= min_sensors]

def get_group_index_by_hora(groups, hora_buscar):
    """Busca el índice del grupo que coincide con una hora específica."""
    for idx, group in enumerate(groups):
        timestamps = []
        for sensor, filepath in group:
            hora = parse_timestamp_from_filename(os.path.basename(filepath))
            if hora:
                timestamps.append(hora)
        if timestamps:
            start_time = min(timestamps).strftime("%H:%M:%S")
            print(f"[Depuración] Grupo {idx}: hora de inicio = {start_time}")
            if start_time == hora_buscar:
                return idx
    return None

# ====================
#  FUNCIONES DE REPORTES
# ====================

def create_train_report(bridge_path, date, min_sensors, workers):
    """
    Crea un informe PDF con los datos de acelerómetros agrupados por timestamp.
    Args:
        bridge_path: Ruta al directorio del puente
        date: Fecha en formato YYYYMMDD
        min_sensors: Número mínimo de sensores para que una vibración sea válida
        workers: Número de hilos para procesar archivos
        max_fig: Número máximo de figuras guardadas en memoria simultáneamente
    Returns:
        Ruta al archivo PDF generado, o None si hubo un error.
    """
    raw_folder_path = get_raw_folder_path(bridge_path, date)

    if not os.path.isdir(raw_folder_path):
        print(f"Ruta no encontrada: {raw_folder_path}")
        return None

    sensor_files = get_acceleration_files(raw_folder_path)
    groups = get_groups(sensor_files, min_sensors)

    if not groups:
        print("No se encontraron archivos para procesar.")
        return None

    output_pdf = get_report_folder_path(bridge_path, date)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    # Carpeta temporal dentro del directorio temporal del sistema
    temp_dir = os.path.join(tempfile.gettempdir(), "report_smartbridges_" + date)
    os.makedirs(temp_dir, exist_ok=True)

    total_groups = len(groups)
    temp_files = [None] * total_groups

    def producer(idx, group):
        try:
            fig = process_file_group(group, date, idx)
            if fig is not None:
                # Archivo oculto, comprimido, en la subcarpeta temporal
                tmp_path = os.path.join(temp_dir, f".report_fig_{idx}.pdf")
                fig.savefig(tmp_path, format='pdf')
                temp_files[idx] = tmp_path
                plt.close(fig)
            print(f"Grupo {idx} procesado y guardado temporalmente.")
        except Exception as exc:
            print(f"Error procesando grupo {idx}: {exc}")
            temp_files[idx] = None
            return None

    # Procesamiento paralelo y guardado temporal
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for idx, group in enumerate(groups):
            futures.append(executor.submit(producer, idx, group))
        for f in futures:
            f.result()  # Esperar a que todos terminen

    # Ensamblaje secuencial del PDF final
    writer = PdfWriter()
    for idx, temp_file in enumerate(temp_files):
        if temp_file and os.path.exists(temp_file):
            reader = PdfReader(temp_file)
            page = reader.pages[0]
            writer.add_page(page)
            os.remove(temp_file)
            print(f"Página {idx+1} añadida al PDF final.")
        else:
            print(f"Página {idx+1} no generada, se omite.")
    with open(output_pdf, "wb") as f:
        writer.write(f)

    # Elimina la carpeta temporal si queda vacía
    if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    else:
        print(f"No se han guardado correctamente todas las figuras: {temp_dir}")
        return None

    return output_pdf

def regenerate_train_report_page(bridge_path, date_str, train_date,  min_sensors):
    """
    Regenera solo la página de un grupo específico en el PDF del informe.
    """

    raw_folder_path = get_raw_folder_path(bridge_path, date_str)
    output_pdf = get_report_folder_path(bridge_path, date_str)

    if not os.path.isfile(output_pdf):
        print(f"El archivo PDF no existe: {output_pdf}")
        return

    # Recopilar archivos csv para cada sensor
    sensor_files = get_acceleration_files(raw_folder_path)
    groups = get_groups(sensor_files, min_sensors)

    group_index = get_group_index_by_hora(groups, train_date)
    print(f"[Regenerar] Índice de grupo encontrado: {group_index}")

    if group_index < 0 or group_index >= len(groups):
        print(f"Índice de grupo fuera de rango: {group_index}")
        return

    print(f"[Regenerar] Regenerando solo el grupo {group_index} en la página {group_index+1}")
    fig = process_file_group(groups[group_index], date_str, group_index)
    regenerate_pdf_page(output_pdf, fig, group_index)

def regenerate_pdf_page(output_pdf, fig, page_index):
    reader = PdfReader(output_pdf)
    writer = PdfWriter()
    for i in range(len(reader.pages)):
        if i == page_index:
            buf = io.BytesIO()
            fig.savefig(buf, format='pdf')
            buf.seek(0)
            new_reader = PdfReader(buf)
            writer.add_page(new_reader.pages[0])
            buf.close()
        else:
            writer.add_page(reader.pages[i])
    with open(output_pdf, "wb") as f:
        writer.write(f)
    print(f"Página {page_index+1} del PDF regenerada correctamente.")

def main():
    """Función principal que maneja la lógica de generación o regeneración de informes."""
    VERSION = "5.0.0"

    parser = argparse.ArgumentParser(description='Generar informe PDF de acelerómetros agrupados por timestamp')
    parser.add_argument('--bridge_path', required=True, type=str, help='Ruta a la carpeta del puente')
    parser.add_argument('--date', required=False, type=str, help='Fecha en formato YYYYMMDD')
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    parser.add_argument('--min_sensors', type=int, default=5, help='Número mínimo de sensores para que una vibración sea válida (default: 5)')
    parser.add_argument('--workers', type=int, default=2 * (os.cpu_count() or 1), help='Número de hilos para procesar archivos (default: 2 * núcleos CPU)')
    parser.add_argument('--regenerar-hora', type=str, nargs='+', default=None, help='Hora de inicio del tren a regenerar (formato HH:MM:SS)')
    
    args = parser.parse_args()

    # Establecer fecha: usar la proporcionada o el día anterior por defecto
    date_str = args.date if args.date else (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    
    # Modo regeneración: regenerar solo una página del informe
    if args.regenerar_hora:
            
       # Procesar cada hora indicada
        for hora in args.regenerar_hora:
            if not validate_hora_format(hora):
                print("El formato de la hora debe ser HH:MM:SS")
                return
            print(f"\n[INFO] Procesando regeneración para la hora: {hora}")
            # Regenerar la página correspondiente a esta hora
            regenerate_train_report_page(args.bridge_path, date_str, hora, args.min_sensors)
    else:
        # Modo normal: generar informe completo
        output = create_train_report(args.bridge_path, date_str, args.min_sensors, args.workers)
        
        if output is None:
            print("No se pudo generar el informe.")
            sys.exit(1)
        else:
            print(f"Reporte guardado en: {output}")

if __name__ == "__main__":
    main()
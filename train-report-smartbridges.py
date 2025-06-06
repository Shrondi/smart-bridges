# -*- coding: utf-8 -*-

import argparse
import calendar
import locale
import os
import re
import gc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
import threading

from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo (solo para escribir en archivos)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft, fftfreq
from PyPDF2 import PdfReader, PdfWriter
import io

# ====================
#  FUNCIONES DE UTILIDAD
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


# ====================
#  FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ====================

def load_data(input_path):
    """Loads and processes a single CSV file."""
    print(f"Cargando datos: {input_path}")
    try:
        df = pd.read_csv(input_path, index_col='timestamp', parse_dates=['timestamp'], date_format='%H:%M:%S.%f', engine='c')
        
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

def process_file_group(file_group, day, month_number, train_number):
    """
    Procesa un grupo de archivos CSV con el mismo timestamp (±1s) y genera una gráfica combinada.

    Args:
        file_group: Lista de tuplas (sensor_name, filepath)
        pdf: Objeto PdfPages donde guardar la gráfica.
    """
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


# ====================
#  FUNCIONES DE REPORTES
# ====================

def train_distribution_report(groups, bridge_path, date_str):
    horas_inicio = []
    for group in groups:
        timestamps = []
        for sensor, filepath in group:
            hora = parse_timestamp_from_filename(filepath)
            if hora:
                timestamps.append(hora)
        if timestamps:
            start_time = min(timestamps)
            horas_inicio.append(start_time)

    if not horas_inicio:
        print("No hay datos de trenes para mostrar.")
        return

    horas = [h.hour + h.minute/60.0 for h in horas_inicio]
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(25)
    counts, _, _ = ax.hist(horas, bins=bins, color='black', edgecolor='black', align='left', rwidth=0.8)
    ax.set_xticks(range(24))
    ax.set_xlim(0, 24)
    ax.set_xlabel('Hora del día')
    ax.set_ylabel('Nº de trenes')
    fecha = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]}"
    ax.set_title(f'Train Distribution - {fecha}')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ymax = int(np.ceil(max(counts))) if len(counts) > 0 else 1
    for y in range(1, ymax + 1):
        ax.axhline(y, color='gray', linestyle='dashed', linewidth=0.7, alpha=0.5)
    plt.tight_layout()

    # Parse date components
    year = date_str[:4]
    month_number = int(date_str[4:6])
    day = date_str[6:]

    # Si no quieres problemas con locale en otros sistemas, comenta la siguiente línea
    try:
        locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    except locale.Error:
        pass  # Ignorar si no se puede establecer locale

    month_name = calendar.month_name[month_number].lower()
    output_pdf = os.path.join(bridge_path, 'report', year, month_name, day, f"train_distribution_{date_str}.pdf")
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Histograma guardado en: {output_pdf}")

def create_train_report(bridge_path, date, min_sensors, workers, max_fig):
    """
    Groups accelerometer data from all sensors by timestamp (±1 second) and generates a PDF report.

    Args:
        bridge_path (str): Path to the bridge directory.
        date (str): A string like '20250522'.

    Returns:
        str: Path to the generated PDF report, or None if input folder is invalid.
    """

    # Parse date components
    year = date[:4]
    month_number = int(date[4:6])
    day = date[6:]

    # Si no quieres problemas con locale en otros sistemas, comenta la siguiente línea
    try:
        locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    except locale.Error:
        pass  # Ignorar si no se puede establecer locale

    month_name = calendar.month_name[month_number].lower()

    # Construimos la ruta con la estructura Bridge/year/month_name/day
    date_folder_path = os.path.join(bridge_path, year, month_name, day)

    if not os.path.isdir(date_folder_path):
        print(f"Ruta no encontrada: {date_folder_path}")
        return None

    # Recopilar archivos csv para cada sensor
    sensor_files = defaultdict(list)

    for root, dirs, files in os.walk(date_folder_path):
        if 'anomalias' in dirs:
            dirs.remove('anomalias')

        for file in files:
            if file.endswith('.csv'):
                sensor_name = os.path.basename(root)
                full_path = os.path.join(root, file)
                sensor_files[sensor_name].append(full_path)

    groups = group_files_by_time(sensor_files)

    # Filtrar grupos por número mínimo de sensores
    groups = [group for group in groups if len(set(sensor for sensor, _ in group)) >= min_sensors]

    if not groups:
        print("No se encontraron archivos para procesar.")
        return None

    output_pdf = os.path.join(bridge_path, 'report', year, month_name, day, f"create_train_report_{date}.pdf")
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    total_groups = len(groups)
    results = [None] * total_groups
    condition = threading.Condition()
    semaphore = threading.Semaphore(max_fig)  # Limitar a 7 grupos procesados simultáneamente

    def producer(idx, group):
        semaphore.acquire()  # Espera si hay demasiadas figuras pendientes
        try:
            fig = process_file_group(group, day, month_number, idx)
        except Exception as exc:
            print(f"Error procesando grupo {idx}: {exc}")
            fig = None
        with condition:
            results[idx] = fig
            condition.notify_all()  # Avisar al consumidor

    def consumer(pdf):
        for idx in range(total_groups):
            with condition:
                while results[idx] is None:
                    condition.wait()
                fig = results[idx]
            if fig is not None:
                # Añadir número de página en la esquina superior derecha
                fig.text(0.97, 0.97, f"{idx+2}", ha='right', va='top', fontsize=14, fontweight='bold')
                pdf.savefig(fig)
                del fig
                gc.collect()
            print(f"Grupo {idx} procesado y guardado.")
            semaphore.release()  # Libera espacio para que un productor pueda continuar

    with PdfPages(output_pdf) as pdf:
        consumer_thread = threading.Thread(target=consumer, args=(pdf,))
        consumer_thread.start()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for idx, group in enumerate(groups):
                futures.append(executor.submit(producer, idx, group))
            # Espera a que todos los productores terminen
            for f in futures:
                f.result()
        consumer_thread.join()

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
    parser.add_argument('--workers', type=int, default=5, help='Número de hilos para procesar archivos (default: 5)')
    parser.add_argument('--max_fig', type=int, default=10, help='Número máximo de figuras guardadas en memoria simultaneamente (default: 10)')
    parser.add_argument('--regenerar-hora', type=str, default=None, help='Hora de inicio del tren a regenerar (formato HH:MM:SS)')
    
    args = parser.parse_args()

    # Establecer fecha: usar la proporcionada o el día anterior por defecto
    date_str = args.date if args.date else (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    
    # Modo regeneración: regenerar solo una página del informe
    if args.regenerar_hora:
        if not validate_hora_format(args.regenerar_hora):
            print("El formato de la hora debe ser HH:MM:SS")
            return
            
        # Regenerar la página
        regenerate_train_report_page(args.bridge_path, date_str, args.regenerar_hora, args.min_sensors)
    else:
        # Modo normal: generar informe completo
        output = create_train_report(args.bridge_path, date_str, args.min_sensors, args.workers, args.max_fig)
        
        if output is None:
            print("No se pudo generar el informe.")
            return
            
        print(f"Reporte guardado en: {output}")
        train_distribution_report(groups, args.bridge_path, date_str)

if __name__ == "__main__":
    main()
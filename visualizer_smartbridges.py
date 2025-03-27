# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft, fftfreq
import argparse
import tempfile

"""# Definición funciones"""

def load_data(input_path):
    """Loads and processes a single CSV file."""
    print(f"Cargando datos: {input_path}")
    try:
        df = pd.read_csv(input_path, index_col='datetime', parse_dates=['datetime'],
                         date_format='%Y-%m-%d %H:%M:%S.%f', engine='c')

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

        # Store the calculated offsets in the dictionary
        offsets[acc_num] = {'x': -mean_x, 'y': -mean_y, 'z': -(mean_z - 1)}

    return offsets

def calc_fft(df):
    """Calcula la FFT para los datos de aceleración."""
    """
    Calcula la FFT para los datos de aceleración y devuelve
    las frecuencias y las magnitudes para cada acelerómetro.

    Args:
        df: DataFrame con los datos de aceleración.

    Returns:
        Un diccionario con las frecuencias y las magnitudes de la FFT
        para cada acelerómetro.
    """

  # https://es.mathworks.com/matlabcentral/answers/712808-how-to-remove-dc-component-in-fft

    fft_data = {}
    accelerometers = df['accelerometer'].unique()

    for acc_num in accelerometers:
        df_acc = df[df['accelerometer'] == acc_num]

        # Longitud de la señal
        L = len(df_acc)

        # Intervalo de muestreo (Ts) y frecuencia de muestreo (Fs)
        t = df_acc.index.to_numpy()  # Obtener los valores de tiempo del índice
        t = (t - t[0]) / np.timedelta64(1, 's') # restando t[0] (para empezar en 0) y convertir a segundos
        Ts = np.mean(np.diff(t))
        Fs = 1 / Ts

        # Transformada de Fourier
        fft_x = fft(df_acc['x'].values)
        fft_y = fft(df_acc['y'].values)
        fft_z = fft(df_acc['z'].values)

        # Borrar la componente continua del eje Z
        fft_z[0] = 0

        # Vector de frecuencias
        frequencies = fftfreq(L, 1/Fs)

        fft_data[acc_num] = {
            'frequencies': frequencies,
            'fft_x': fft_x,
            'fft_y': fft_y,
            'fft_z': fft_z,
        }

    return fft_data

def create_color_mapping(accelerometers):
    """Crea un mapeo de colores para cada acelerómetro."""
    """
    Crea un mapeo de colores para cada acelerómetro.

    Args:
        accelerometers: Lista de acelerómetros.

    Returns:
        Un diccionario con el mapeo de colores para cada acelerómetro.
    """
    num_accelerometers = len(accelerometers)
    palette = sns.color_palette('bright', n_colors=num_accelerometers)
    colors = {
        acc_num: {'x': palette[i], 'y': palette[(i + 1) % num_accelerometers], 'z': palette[(i + 2) % num_accelerometers]}
        for i, acc_num in enumerate(accelerometers)
    }
    return colors

def configure_axes(axes, titles, xlabel, ylabel):
    """
    Configura los ejes de las gráficas con títulos, etiquetas y leyendas.

    Args:
        axes: Array de ejes a configurar.
        titles: Lista de títulos para cada eje.
        xlabel: Etiqueta del eje X.
        ylabel: Etiqueta del eje Y.
    """
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc="lower left")
        ax.grid(which='major', linestyle='--', alpha=0.5)
        ax.grid(which='minor', linestyle=':', alpha=0.3)
        ax.autoscale()

def plot_train_data(df, offsets, axes, colors):
    """Plots accelerometer data on the provided axes."""
    """
    Plots accelerometer data on the provided axes.

    Args:
        df: DataFrame with the accelerometer data.
        offsets: Dictionary containing offsets for each accelerometer.
        axes: Array of axes to plot on.
        colors: Dictionary containing color mapping for each accelerometer.
    """

    # Obtener el número de acelerómetros únicos
    accelerometers = df['accelerometer'].unique()

    # Plotear datos
    for acc_num in accelerometers:
        df_acc = df[df['accelerometer'] == acc_num][['x', 'y', 'z']]

        # Calcular la diferencia de tiempo en segundos
        time_diff = (df_acc.index - df_acc.index[0]).to_series().dt.total_seconds()

        # Acceder a los colores usando el número de acelerómetro y el eje
        axes[0].plot(time_diff, df_acc['x'], label=f'Acel. {acc_num} (Offset: {offsets[acc_num]["x"]:.4f})', color=colors[acc_num]['x'])
        axes[1].plot(time_diff, df_acc['y'], label=f'Acel. {acc_num} (Offset: {offsets[acc_num]["y"]:.4f})', color=colors[acc_num]['y'])
        axes[2].plot(time_diff, df_acc['z'], label=f'Acel. {acc_num} (Offset: {offsets[acc_num]["z"]:.4f})', color=colors[acc_num]['z'])

    configure_axes(axes, ['Aceleración X', 'Aceleración Y', 'Aceleración Z'], 'Tiempo [s]', 'Aceleración [g]')

def plot_fft(fft_data, axes, colors):
    """Plots FFT data on the provided axes."""
    """
    Plots FFT data on the provided axes.

    Args:
        fft_data: Dictionary containing FFT data for each accelerometer.
        axes: Array of axes to plot on.
        colors: Dictionary containing color mapping for each accelerometer.
    """
    accelerometers = list(fft_data.keys())

    bar_width = 0.8 / len(accelerometers)

    for i, acc_num in enumerate(accelerometers):
        offset = bar_width * i - bar_width * (len(accelerometers) - 1) / 2 # Offset para dibujar las barras side by side

        frequencies = fft_data[acc_num]['frequencies']

        # Obtener la magnitud de la FFT
        fft_x = np.abs(fft_data[acc_num]['fft_x'])
        fft_y = np.abs(fft_data[acc_num]['fft_y'])
        fft_z = np.abs(fft_data[acc_num]['fft_z'])

        axes[0].bar(frequencies + offset, fft_x, width=bar_width, label=f'Acel. {acc_num}', color=colors[acc_num]['x'])
        axes[1].bar(frequencies + offset, fft_y, width=bar_width, label=f'Acel. {acc_num}', color=colors[acc_num]['y'])
        axes[2].bar(frequencies + offset, fft_z, width=bar_width, label=f'Acel. {acc_num}', color=colors[acc_num]['z'])

    configure_axes(axes, ['FFT Aceleración X', 'FFT Aceleración Y', 'FFT Aceleración Z'], 'Frecuencia (Hz)', 'Amplitud')

def process_file(filepath, pdf, first_date, last_date):
    """
    Procesa un archivo CSV y genera gráficos en el PDF.

    Args:
        filepath: Ruta del archivo CSV.
        pdf: Objeto PdfPages para guardar las gráficas.
        first_date: Fecha inicial del conjunto de datos.
        last_date: Fecha final del conjunto de datos.
    """
    df = load_data(filepath)
    if df.empty:
        print(f"No se encontraron datos en: {filepath}")
        return first_date, last_date

    # Actualizar fechas
    if first_date is None:
        first_date = df.index[0].strftime("%Y%m%d_%H%M%S")
    last_date = df.index[-1].strftime("%Y%m%d_%H%M%S")

    offsets = calc_offsets(df)
    fft_data = calc_fft(df)
    accelerometers = df['accelerometer'].unique()
    colors = create_color_mapping(accelerometers)

    fig, axes = plt.subplots(3, 2, figsize=(20, 20), gridspec_kw={'width_ratios': [3, 1]})
    first_datetime = df.index[0]
    
    figure_title = f"Tren {first_datetime.strftime('%d/%m')} {first_datetime.strftime('%H:%M:%S')}"
    fig.suptitle(figure_title, fontsize=20, fontweight='bold')
    
    fig.subplots_adjust(hspace=0.5, top=0.92)

    plot_train_data(df, offsets, axes[:, 0], colors)
    plot_fft(fft_data, axes[:, 1], colors)

    pdf.savefig(fig)
    plt.close()

    print(f"Datos procesados: {os.path.basename(filepath)}")

    return first_date, last_date

def process_train_file(bridge_path, output_dir='./', pdf_name=None):
    """Plotea datos de acelerómetros y FFT desde archivos CSV."""
    """
    Plots accelerometer data and FFT from a CSV file or a directory of CSV files.

    Args:
        bridge_path (str): The path to the bridge directory containing date subfolders.
        output_dir (str): Directory where the PDF will be saved.
        pdf_name (str): Name of the output PDF file. Defaults to the bridge name with the min and max dates.
    """
    # Obtener el nombre del puente
    bridge_name = os.path.basename(os.path.normpath(bridge_path))

    # Inicializar variables para las fechas
    first_date, last_date = None, None

    # Crear un archivo PDF temporal
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=output_dir) as temp_pdf:
        temp_pdf_filepath = temp_pdf.name
        pdf = PdfPages(temp_pdf_filepath)

        for date_folder in sorted(os.listdir(bridge_path)):
            date_folder_path = os.path.join(bridge_path, date_folder)

            # Check if it's a directory
            if os.path.isdir(date_folder_path):

                for filename in sorted([f for f in os.listdir(date_folder_path) if f.endswith(".csv")]):
                    filepath = os.path.join(date_folder_path, filename)
                    first_date, last_date = process_file(filepath, pdf, first_date, last_date)

        pdf.close()

    # Determinar el nombre final del PDF
    if not pdf_name:
        pdf_name = f"{bridge_name}-{first_date}-{last_date}.pdf"

    final_pdf_filepath = os.path.join(output_dir, pdf_name)

    os.rename(temp_pdf_filepath, final_pdf_filepath)

    return final_pdf_filepath

def main():
    VERSION = "1.1.0"

    parser = argparse.ArgumentParser(description="Visualizador y exportador de datos de vibraciones de trenes en acelerómetros sobre puentes")

    # Añadir la opción --version para mostrar la versión
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')

    # Definir los argumentos
    parser.add_argument('--input', type=str, required=True, help='Directorio de entrada del puente que contiene los datos (obligatorio)')
    parser.add_argument('--output', type=str, default='./', help='Directorio de salida para el archivo PDF (por defecto: ./)')
    parser.add_argument('--pdf_name', type=str, help='Nombre del archivo PDF de salida (por defecto: nombre del puente)')

    # Parsear los argumentos
    args = parser.parse_args()

    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    PDF_NAME = args.pdf_name

    print("\n* Procesando datos...")
    pdf_filepath = process_train_file(INPUT_DIR, OUTPUT_DIR, PDF_NAME)

    print("\n* Exportación completada. Archivo PDF guardado en:", pdf_filepath)

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft, fftfreq

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

def plot_train_data(df, offsets, axes):
    """
    Plots accelerometer data on the provided axes.

    Args:
        df: DataFrame with the accelerometer data.
        offsets: Dictionary containing offsets for each accelerometer.
        axes: Array of axes to plot on.
    """

    # Paleta de colores para los acelerometros
    palette = sns.color_palette('bright', n_colors=8)

    # Create a color map using the palette
    colors = {
        1: {'x': palette[0], 'y': palette[1], 'z': palette[2]},
        2: {'x': palette[3], 'y': palette[4], 'z': palette[5]},
        3: {'x': palette[6], 'y': palette[7], 'z': palette[0]},
        4: {'x': palette[1], 'y': palette[2], 'z': palette[3]},
        5: {'x': palette[4], 'y': palette[5], 'z': palette[6]},
        6: {'x': palette[7], 'y': palette[0], 'z': palette[1]},
        7: {'x': palette[2], 'y': palette[3], 'z': palette[4]},
        8: {'x': palette[5], 'y': palette[6], 'z': palette[7]}
    }

    # Plotear datos
    for acc_num in df['accelerometer'].unique():
        df_acc = df[df['accelerometer'] == acc_num]
        df_acc = df_acc[['x', 'y', 'z']]

        # Remuestrear a 1 segundo
        #df_acc = df_acc.resample('10ms').mean()

        # Calculate time difference in seconds
        time_diff = (df_acc.index - df_acc.index[0]).to_series().dt.total_seconds()

        # Access color using accelerometer number and axis
        axes[0].plot(time_diff, df_acc['x'], label=f'Acel. {acc_num} (Offset: {offsets[acc_num]["x"]:.4f})', color=colors.get(acc_num, {}).get('x'))
        axes[1].plot(time_diff, df_acc['y'], label=f'Acel. {acc_num} (Offset: {offsets[acc_num]["y"]:.4f})', color=colors.get(acc_num, {}).get('y'))
        axes[2].plot(time_diff, df_acc['z'], label=f'Acel. {acc_num} (Offset: {offsets[acc_num]["z"]:.4f})', color=colors.get(acc_num, {}).get('z'))

    # Configurar ejes para cada gráfica
    for i, ax in enumerate(axes):
        ax.set_xlabel('Tiempo [s]', fontsize=12)
        ax.set_ylabel('Aceleración [g]', fontsize=12)
        ax.legend(loc="lower left")
        ax.grid(which='major', linestyle='--', alpha=0.5)
        ax.grid(which='minor', linestyle=':', alpha=0.3)
        ax.autoscale()

        if i == 0:
            ax.set_title('Aceleración X', fontsize=14, fontweight='bold')
        elif i == 1:
            ax.set_title('Aceleración Y', fontsize=14, fontweight='bold')
        else:
            ax.set_title('Aceleración Z', fontsize=14, fontweight='bold')

def plot_fft(fft_data, axes):
    """
    Plots FFT data on the provided axes.

    Args:
        fft_data: Dictionary containing FFT data for each accelerometer.
        axes: Array of axes to plot on.
    """
    accelerometers = list(fft_data.keys())

    bar_width = 0.8 / len(accelerometers)

    palette = sns.color_palette('bright', n_colors=len(accelerometers))

    for i, acc_num in enumerate(accelerometers):
        offset = bar_width * i - bar_width * (len(accelerometers) - 1) / 2 # Offset para dibujar las barras side by side

        frequencies = fft_data[acc_num]['frequencies']

        # Obtener la magnitud de la FFT
        fft_x = np.abs(fft_data[acc_num]['fft_x'])
        fft_y = np.abs(fft_data[acc_num]['fft_y'])
        fft_z = np.abs(fft_data[acc_num]['fft_z'])

        axes[0].bar(frequencies + offset, fft_x, width=bar_width, label=f'Acel. {acc_num}', color=palette[i])
        axes[1].bar(frequencies + offset, fft_y, width=bar_width, label=f'Acel. {acc_num}', color=palette[i])
        axes[2].bar(frequencies + offset, fft_z, width=bar_width, label=f'Acel. {acc_num}', color=palette[i])

    # Configurar ejes y leyenda
    for i, ax in enumerate(axes):
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('Amplitud')
        ax.legend()

        if i == 0:
            ax.set_title('FFT Aceleración X', fontsize=14, fontweight='bold')
        elif i == 1:
            ax.set_title('FFT Aceleración Y', fontsize=14, fontweight='bold')
        else:
            ax.set_title('FFT Aceleración Z', fontsize=14, fontweight='bold')

def process_train_file(bridge_path, show=True, save=False, output_dir='./'):
    """
    Plots accelerometer data and FFT from a CSV file or a directory of CSV files.

    Args:
        bridge_path (str): The path to the bridge directory containing date subfolders.
    """

    if save:
          # Get the bridge name from the bridge_path
        bridge_name = os.path.basename(os.path.normpath(bridge_path))

        # Create PDF filename using the bridge name
        pdf_filename = f"{bridge_name}.pdf"
        pdf_filepath = os.path.join(output_dir, pdf_filename)
        pdf = PdfPages(pdf_filepath)

    for date_folder in sorted(os.listdir(bridge_path)):
        date_folder_path = os.path.join(bridge_path, date_folder)

        # Check if it's a directory
        if os.path.isdir(date_folder_path):

            for filename in sorted([f for f in os.listdir(date_folder_path) if f.endswith(".csv")]):
                filepath = os.path.join(date_folder_path, filename)

                df = load_data(filepath)

                if not df.empty:
                    offsets = calc_offsets(df)
                    fft_data = calc_fft(df)

                    # Create subplots with gridspec_kw
                    fig, axes = plt.subplots(3, 2, figsize=(20, 20), gridspec_kw={'width_ratios': [3, 1]})  # Adjust width_ratios as needed

                    # Obtener el primer datetime para usarlo como titulo
                    first_datetime = df.index[0]

                    formatted_date = first_datetime.strftime("%d/%m")  # Dia/Mes
                    formatted_time = first_datetime.strftime("%H:%M:%S")  # Hora:Minutos:Segundos

                    # Crear titulo de la figura
                    figure_title = f"Tren {formatted_date} {formatted_time}"
                    fig.suptitle(figure_title, fontsize=20, fontweight='bold')

                    fig.subplots_adjust(hspace=0.5, top=0.92)

                    # Call plot_train_data with the first column axes
                    plot_train_data(df, offsets, axes[:, 0])  # Pass axes for acceleration plots

                    # Call plot_fft with the second column axes
                    plot_fft(fft_data, axes[:, 1])  # Pass axes for FFT plots

                    if show:
                        plt.show()

                    if save and fig is not None:
                        pdf.savefig(fig)

                        print(f"Figura guardada en {pdf_filepath}")

                    plt.close()

                else:
                    print(f"No se encontraron datos en: {filepath}")

    if save:
        pdf.close()


def main(): 
    process_train_file('Guadiato', save=True, show=False)

if __name__ == '__main__':
    main()
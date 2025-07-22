import matplotlib.pyplot as plt
import matplotlib
import concurrent.futures
import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuración de matplotlib
matplotlib.use('Agg')  # Entorno sin display
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType (mejor compatibilidad)
matplotlib.rcParams['pdf.use14corefonts'] = False


def timestamps_sensor(path_sensor):
    """
    Extrae los timestamps de los archivos CSV de un sensor específico.
    """
    try:
        
        csv_files = (
            f for f in Path(path_sensor).rglob('*.csv')
            if f.is_file() and 'anomalias' not in f.name.lower()
        )

        # Extraer los timestamps de los nombres de los archivos con el formato de nombre: acceleration_HH-MM-SS.csv
        timestamps = [
            datetime.strptime(f.name.split('_')[1].replace('.csv', ''), '%H-%M-%S')
            for f in csv_files
        ]

        return timestamps
    
    except Exception as e:
        print(f"[x] Error inesperado: {e}")
        return None


def generate_matrix(timestamps, sensores, resolution):
    """
    Genera la matriz binaria que representa la actividad de los sensores a lo largo del tiempo.
    La matriz tiene una fila por sensor y una columna por intervalo de tiempo.
    La resolución define el tamaño de cada intervalo en minutos.
    """
    columnas = 1440 // resolution  # Sampling del tiempo dependiendo de la resolución

    # Crear la matriz binaria
    matriz_np = np.zeros((len(sensores), columnas), dtype=int)

    for i, sensor in enumerate(sensores):
        for timestamp in timestamps[sensor]:
            minutos = timestamp.hour * 60 + timestamp.minute
            indice = minutos // resolution
            matriz_np[i, indice] = 1

    return matriz_np


def create_schedule(path_dia, matriz_np, sensores, scale=15, resolution=2):
    """
    Crea un gráfico de programación basado en la matriz binaria y la lista de sensores.
    """
    if matriz_np.size == 0:
        print(f"[!] No hay datos para graficar en: {path_dia}")
        return

    fig, ax = plt.subplots(figsize=(45, max(6, 0.5 * len(sensores))))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    ax.imshow(matriz_np, aspect='auto', cmap='Greys', interpolation='nearest')

    # Dibujar líneas verticales para cada columna para separación visual
    y_coords, x_coords = np.where(matriz_np == 1)
    for y, x in zip(y_coords, x_coords):
        # Línea izquierda
        ax.plot([x-0.5, x-0.5], [y-0.5, y+0.5], color='white', linewidth=1.0)
        # Línea derecha
        ax.plot([x+0.5, x+0.5], [y-0.5, y+0.5], color='white', linewidth=1.0)

    # Líneas horizontales para cada sensor
    for y in range(len(sensores)):
        ax.hlines(y, xmin=-0.5, xmax=matriz_np.shape[1] - 0.5,
                  colors='gray', linestyles='dashed', linewidth=0.7, alpha=0.5)

    # Etiquetas para el eje Y
    ylabels = [
        f"Sensor {Path(s).parts[-1].split('_', 1)[1]}"
        for s in sensores
        if Path(s).name.lower().startswith('sensor_')
    ]

    ax.set_yticks(range(len(sensores)))
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.tick_params(axis='y', pad=10)

    step = scale // resolution
    x_ticks = list(range(0, matriz_np.shape[1], step))
    x_labels = [f"{(i * resolution) // 60:02d}:{(i * resolution) % 60:02d}" for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)

    ax.set_xlim(-0.5, matriz_np.shape[1] - 0.5)
    ax.set_ylim(-0.5, len(sensores) - 0.5)

    # Fecha de modificación
    fecha_mod = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax.text(1, 1.02, f"Last modification: {fecha_mod}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='gray')

    ax.set_xlabel("Time (HH:MM)")

    partes = Path(path_dia).parts
    dia, mes, año = (partes[-1], partes[-2], partes[-3]) if len(partes) >= 3 else ("?", "?", "?")

    titulo = f"Train Recording Schedule - {mes.capitalize()} {dia}, {año}"
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=20)

    output_path = Path(path_dia).joinpath("train_recording_schedule.pdf")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def process_bridge(path_dia):
    """
    Procesa un directorio de día específico, generando timestamps de eventos para cada sensor
    y creando un gráfico de programación de grabación.
    """
    sensor_timestamps = {}

    sensor_dirs = [
        d for d in Path(path_dia).glob('*/')
        if d.is_dir() and d.name.startswith('sensor')
    ]

    for sensor_dir in sensor_dirs:
        print(f"[+] Procesando {sensor_dir}")
        timestamps = timestamps_sensor(sensor_dir)

        if timestamps:
            sensor_timestamps[str(sensor_dir)] = timestamps
        else:
            print(f"[!] Sin eventos detectados en {sensor_dir}")
            return

    # Crear la matriz y la lista de sensores
    sensores = sorted(sensor_timestamps.keys(), reverse=True)
    
    matriz_np = generate_matrix(sensor_timestamps, sensores, resolution=2)

    print(f"[+] Generando gráfico para: {path_dia}")
    
    try:
        create_schedule(path_dia, matriz_np, sensores)
    except Exception as e:
        print(f"[x] Error al generar el gráfico: {e}")


def process_day(root_path):
    """
    Procesa los datos del día actual, buscando carpetas de sensores y generando gráficos de programación.
    Se espera que las carpetas sigan la estructura: root/**/año/mes/día
    """

    fecha = datetime.now()
    
    año = fecha.strftime('%Y')
    mes = fecha.strftime('%B').lower()
    dia = fecha.strftime('%d')

    # Se obtiene la ruta del día actual para cada puente
    days_path = [
        path for path in Path(root_path).rglob(f"{año}/{mes}/{dia}")
        if path.is_dir()
    ]

    if not days_path:
        print(f"[x] No se encontró el directorio del día {fecha.strftime('%Y-%m-%d')} en {root_path}")
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_bridge, day): day for day in days_path}
    
    for future in concurrent.futures.as_completed(futures):
        day = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error procesando {day}: {e}")


if __name__ == '__main__':
    
    VERSION = "3.0.0"

    parser = argparse.ArgumentParser(description="Procesar y graficar los archivos de vibraciones del día actual.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    parser.add_argument('--root', type=str, required=True, help="Ruta base que contiene las carpetas de los puentes")
    
    args = parser.parse_args()
    
    if not Path(args.root).is_dir():
        print(f"[x] La ruta proporcionada no es un directorio válido: {args.root}")
        sys.exit(1)
    
    process_day(args.root)

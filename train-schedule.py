import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Evitar problemas con entornos sin display
from datetime import datetime, timedelta
import argparse
import threading
import glob
import sys


def bins_sensor(ruta_sensor, bin_size=5):
    """
    Genera un array de bins de eventos para un sensor específico, donde cada bin representa un intervalo de tiempo.
    """
    num_bins = 24 * 60 // bin_size
    bins = [0] * num_bins

    # Buscar todos los archivos CSV en el directorio del sensor y subcarpetas,
    # evitando los que estén en rutas que contengan 'anomalias'
    csv_files = [
        f for f in glob.iglob(os.path.join(ruta_sensor, '**', '*.csv'), recursive=True)
        if os.path.isfile(f) and 'anomalias' not in f.lower()
    ]

    for archivo_csv in csv_files:

        try:
            df = pd.read_csv(archivo_csv, sep=',', engine='c')
        except Exception as e:
            print(f"[x] Error leyendo {os.path.basename(archivo_csv)}: {e}")
            continue
        
        if df.empty:
            print(f"[!] El archivo {os.path.basename(archivo_csv)} está vacío, se omite.")
            continue
        
        # Comprobar cabecera exacta
        if list(df.columns) != ['timestamp', 'x_accel (g)', 'y_accel (g)', 'z_accel (g)']:
            print(f"[x] El archivo {os.path.basename(archivo_csv)} no tiene la cabecera esperada. Se omite.")
            continue
        
        # Extraer solo la columna de timestamps. Marcar errores de conversión como NaT para eliminar después
        tiempos = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f', errors='coerce')
        tiempos = tiempos.dropna()
        
        if tiempos.empty:
            print(f"[!] No hay timestamps válidos en {os.path.basename(archivo_csv)}.")
            continue
        
        # Calcular minutos desde medianoche
        minutos = tiempos.dt.hour * 60 + tiempos.dt.minute
        
        # Calcular el índice de bin para cada evento
        indices_bin = (minutos // bin_size).astype(int, errors='ignore')
        
        valid_bins = indices_bin[(indices_bin >= 0) & (indices_bin < num_bins)].unique()
        bins_arr = pd.Series(bins)
        bins_arr.iloc[valid_bins] = 1
        bins = bins_arr.tolist()
                    
    return bins

# --- GRAFICADO ---
def create_schedule(path_dia, sensores_bins, bin_size=5, scale=15):
    """
    Crea un gráfico de la programación de grabación de eventos para los sensores en un día específico.
    Parámetros:
    - path_dia: ruta de la carpeta del día.
    - sensores_bins: diccionario con los sensores y sus respectivos bins de eventos.
    - bin_size: tamaño del bin en minutos.
    - scale: cada cuanto minutos se muestra un tick en el eje X.
    """
    sensores = sorted(sensores_bins.keys())
    matriz = [sensores_bins[s] for s in sensores]
    
    # Etiquetas genéricas: Sensor 1, Sensor 2, ...
    etiquetas = [f"Sensor {i+1}" for i in range(len(sensores))]

    partes = os.path.normpath(path_dia).split(os.sep)
    if len(partes) >= 3:
        dia, mes, año = partes[-1], partes[-2], partes[-3]
    else:
        dia = mes = año = "?"

    titulo = f"Train Recording Schedule - {mes.capitalize()} {dia}, {año}"
    output_path = os.path.join(path_dia, f"train_recording_schedule.pdf")

    fig, ax = plt.subplots(figsize=(40, max(6, 0.5 * len(sensores))))
    ax.imshow(matriz, aspect='auto', cmap='Greys', interpolation='nearest')

    # Líneas horizontales punteadas para cada sensor
    for y in range(len(sensores)):
        ax.hlines(y, xmin=0, xmax=len(matriz[0]) - 1, colors='gray', linestyles='dashed', linewidth=0.7, alpha=0.5)
    
    ax.set_yticks(range(len(sensores)))
    ax.set_yticklabels(etiquetas, fontsize=12)
    ax.tick_params(axis='y', pad=10)

    # Configurar eje X (tiempo) con ticks cada 30 minutos
    step = scale // bin_size
    x_ticks = list(range(0, 1440 // bin_size, step))
    x_labels = [f"{(i * bin_size) // 60:02d}:{(i * bin_size) % 60:02d}" for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)
    
    
    # Obtener fecha y hora última modificación del directorio del día
    fecha_mod = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax.text(1, 1.02, f"Last modification: {fecha_mod}", transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='gray')
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Sensor")
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def process_bridge(path_dia):
    """
    Procesa un directorio de día específico, generando bins de eventos para cada sensor
    y creando un gráfico de programación de grabación.
    Parámetros:
    - path_dia: ruta de la carpeta del día.
    - bin_size: tamaño del bin en minutos (default: 5).
    - scale: cada cuanto minutos se muestra un tick en el eje X (default: 15).
    """
    sensores_bins = {}
  
    sensor_dirs = [d for d in glob.glob(os.path.join(path_dia, '*/')) if os.path.isdir(d) and d.startswith(os.path.join(path_dia, 'sensor'))]

    for sensor_dir in sensor_dirs:
        
        print(f"[+] Procesando {sensor_dir}")
    
        bins = bins_sensor(sensor_dir)

        try:
            if any(bins):
                sensores_bins[sensor_dir] = bins
            else:
                print(f"[!] Sin eventos detectados en {sensor_dir}")
        except Exception as e:
            print(f"[x] Error procesando {sensor_dir}: {e}")
            continue
    
    if not sensores_bins:
        print(f"[!] Sin datos para graficar en: {path_dia}")
        return

    # Crear el gráfico de programación
    print(f"[+] Generando gráfico de programación para: {path_dia}")
    try:
        create_schedule(path_dia, sensores_bins)
    except Exception as e:
        print(f"[x] Error al generar el gráfico: {e}")


def process_day(ruta_raiz, yesterday):
    """
    Procesa los datos de vibraciones del día actual y, opcionalmente, del día anterior.
    Parámetros:
    - ruta_raiz: ruta base que contiene la carpeta Guadiato.
    - yesterday: hora (HH:MM) para procesar también el día anterior.
    Si la hora actual coincide con yesterday, se procesará también el día anterior.
    """

    ahora = datetime.now()
    fechas = [ahora]

    if yesterday and ahora.time().replace(second=0, microsecond=0) == yesterday:
        print(f"[+] Activando proceso para procesar el día anterior: {ahora.strftime('%H:%M')}")
        # Calcular la fecha de ayer a las 00:15
        ayer = ahora - timedelta(days=1)
        fechas.insert(0, ayer)  # Procesar ayer antes que hoy

    for fecha in fechas:
        print(f"[+] Procesando fecha: {fecha.strftime('%Y-%m-%d')}")

        año = fecha.strftime('%Y')
        mes = fecha.strftime('%B').lower()
        dia = fecha.strftime('%d')

        carpetas_dia = [
            path for path in glob.iglob(os.path.join(ruta_raiz, '**', año, mes, dia), recursive=True)
            if os.path.isdir(path)
        ]

        if not carpetas_dia:
            print(f"[x] No se encontró la ruta de carpetas para el día: {fecha.strftime('%Y-%m-%d')}")
            sys.exit(1)

        threads = []
        for day_path in carpetas_dia:
            print(f"[+] Iniciando hilo para datos del día: {day_path}")
            thread = threading.Thread(target=process_bridge, args=(day_path,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

if __name__ == '__main__':
    
    VERSION = "2.0.0"

    parser = argparse.ArgumentParser(description="Procesar y graficar los archivos de vibraciones del día actual.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    parser.add_argument('--root', type=str, required=True, help="Ruta base que contiene la carpeta Guadiato")
    parser.add_argument('--yesterday', default='00:15', type=str, help="Hora (HH:MM) para procesar también el día anterior para graficar los datos restantes anteriores (default: 00:15:00)" \
                                                                                "¡Atención! Este script esta gestionado por un .timer del sistema. Si se modifica esta flag, debe ser acorde a la hora de activación del timer.")
    
    args = parser.parse_args()

    try:
        yesterday = datetime.strptime(args.yesterday, "%H:%M").time()
    except ValueError:
        parser.error("El formato de --yesterday debe ser HH:MM (por ejemplo, 00:15)")

    process_day(args.root, yesterday)

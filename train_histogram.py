import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

def obtener_bins_de_sensor(ruta_sensor, bin_size=5):
    """
    Lee los CSV de un sensor y devuelve un vector binario indicando
    si hubo evento en cada intervalo de tiempo (bin) del día.
    
    Parámetros:
    - ruta_sensor: ruta de la carpeta del sensor con archivos CSV.
    - bin_size: tamaño del bin en minutos (default 5).
    
    Retorna:
    - bins: lista de 0s y 1s, tamaño 24*60/bin_size.
    """
    bins = [0] * (24 * 60 // bin_size)

    try:
        for entrada in os.scandir(ruta_sensor):
            if not entrada.is_file() or not entrada.name.endswith('.csv'):
                continue

            df = pd.read_csv(entrada.path)

            for _, fila in df.iterrows():
                try:
                    tiempo = datetime.strptime(fila['timestamp'], '%H:%M:%S.%f')
                    minutos_desde_medianoche = tiempo.hour * 60 + tiempo.minute
                    indice_bin = minutos_desde_medianoche // bin_size
                    bins[indice_bin] = 1
                except Exception:
                    continue
    except Exception as e:
        print(f"Error leyendo archivos en {ruta_sensor}: {e}")

    return bins

def generar_grafico_dia(path_dia, sensores_bins, bin_size=5):
    """
    Genera y guarda un gráfico PDF para el día, mostrando
    en el eje X el tiempo y en el eje Y los sensores con
    sus eventos en franjas de tiempo.
    
    Parámetros:
    - path_dia: ruta de la carpeta del día.
    - sensores_bins: dict {sensor: bins}.
    - bin_size: tamaño del bin en minutos.
    """
    sensores = sorted(sensores_bins.keys())
    matriz = [sensores_bins[s] for s in sensores]

    # Extraer información para título
    dia = os.path.basename(path_dia)
    mes = os.path.basename(os.path.dirname(path_dia))
    año = os.path.basename(os.path.dirname(os.path.dirname(path_dia)))
    puente = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path_dia))))

    titulo = f"Train Recording Schedule: {dia}/{mes}/{año}"
    output_path = os.path.join(path_dia, f"train_recording_schedule.pdf")

    fig, ax = plt.subplots(figsize=(25, max(6, 0.5 * len(sensores))))
    ax.imshow(matriz, aspect='auto', cmap='Greys', interpolation='nearest')

    # Líneas horizontales punteadas para cada sensor
    for y in range(len(sensores)):
        ax.hlines(y, xmin=0, xmax=len(matriz[0]) - 1,
                  colors='gray', linestyles='dashed', linewidth=0.7, alpha=0.5)

    # Configurar eje Y (sensores)
    ax.set_yticks(range(len(sensores)))
    ax.set_yticklabels(sensores, fontsize=12)
    ax.tick_params(axis='y', pad=10)

    # Configurar eje X (tiempo) con ticks cada 30 minutos
    step = 15 // bin_size
    x_ticks = list(range(0, 1440 // bin_size, step))
    x_labels = [f"{(i * bin_size) // 60:02d}:{(i * bin_size) % 60:02d}" for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)
    
    
    # Obtener fecha y hora última modificación del directorio del día
    timestamp_mod = os.path.getmtime(path_dia)
    fecha_mod = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Añadir texto en la esquina superior derecha
    ax.text(1, 1.02, f"Última mod: {fecha_mod}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='gray')

    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Sensor")
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"PDF generado: {output_path}")

def procesar_dia(path_dia, bin_size=5):
    """
    Procesa todos los sensores en un día, generando los bins
    de eventos y creando el gráfico si hay datos.
    
    Parámetros:
    - path_dia: ruta de la carpeta del día.
    - bin_size: tamaño del bin en minutos.
    """
    sensores_bins = {}

    for entrada in os.scandir(path_dia):
        if not entrada.is_dir():
            continue
        sensor = entrada.name
        if "anomalias" in sensor.lower():
            continue  # Ignorar carpetas de anomalías

        bins = obtener_bins_de_sensor(entrada.path, bin_size)
        if any(bins):
            sensores_bins[sensor] = bins

    if not sensores_bins:
        print(f"Sin datos para graficar en: {path_dia}")
        return

    generar_grafico_dia(path_dia, sensores_bins, bin_size)

def recorrer_directorios(ruta_raiz, bin_size=5):
    """
    Recorre la estructura completa de directorios:
    puente / año / mes / día / sensor
    
    Parámetros:
    - ruta_raiz: ruta base donde están los puentes.
    - bin_size: tamaño del bin en minutos.
    """
    for entrada_puente in os.scandir(ruta_raiz):
        if not entrada_puente.is_dir():
            continue
        path_puente = entrada_puente.path

        for entrada_año in os.scandir(path_puente):
            if not entrada_año.is_dir():
                continue
            path_año = entrada_año.path

            for entrada_mes in os.scandir(path_año):
                if not entrada_mes.is_dir():
                    continue
                path_mes = entrada_mes.path

                for entrada_dia in os.scandir(path_mes):
                    if not entrada_dia.is_dir():
                        continue
                    path_dia = entrada_dia.path
                    procesar_dia(path_dia, bin_size)

if __name__ == '__main__':
    ruta_base = '/srv/smartbridges/'  # Cambia por tu ruta base real
    recorrer_directorios(ruta_base, bin_size=5)


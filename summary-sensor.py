import pandas as pd
import numpy as np
import glob
import os
import json
from datetime import datetime, timedelta
import argparse
import threading

def parse_times_from_filename(filename):
    """
    Extrae la hora de inicio del nombre del archivo en formato "acceleration_HH-MM-SS.csv".
    Retorna un objeto datetime.time o None si no se puede parsear.
    """
    base = os.path.basename(filename)
    time_part = base.replace("acceleration_", "").replace(".csv", "")
    try:
        start_time = datetime.strptime(time_part, "%H-%M-%S").time()
    except ValueError:
        start_time = None
    return start_time

def load_sensor_data(sensor_folder):
    """
    Lee todos los archivos CSV del sensor_folder y retorna una lista de tuplas (archivo, df),
    asegurando que las columnas x, y, z estén presentes y renombradas si es necesario.
    """
    archivos = sorted(glob.glob(os.path.join(sensor_folder, "*.csv")))
    csv_dfs = []
    for archivo in archivos:
        df = pd.read_csv(archivo, engine='c')
        if 'timestamp' not in df.columns or len(df) < 2:
            continue
        
        if not list(df.columns) == ['timestamp', 'x_accel (g)', 'y_accel (g)', 'z_accel (g)']:
            print(f"[!] Archivo {archivo} no tiene las columnas esperadas. Se omite.")
            continue
        
        df.rename(columns={
            'x_accel (g)': 'x',
            'y_accel (g)': 'y',
            'z_accel (g)': 'z'
        }, inplace=True)
        
        csv_dfs.append((archivo, df))
        
    return csv_dfs

def calc_sampling_rates(csv_dfs):
    """
    Calcula el sampling rate de cada segmento usando los timestamps de los DataFrames.
    Retorna un diccionario {segment_id: sampling_rate}
    """

    sampling_rates = {}
    for idx, (_, df) in enumerate(csv_dfs, 1):
        try:
            times = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f', errors='coerce')
            deltas = times.diff().dt.total_seconds().dropna()
            
            # Filtrar solo deltas > 0 (ignorar huecos o saltos grandes)
            deltas = deltas[deltas > 0]
            if len(deltas) == 0:
                sampling_rates[idx] = None
            else:
                # Calcular el valor más frecuente (moda) como el sampling interval
                mode_delta = deltas.mode().iloc[0] if not deltas.mode().empty else deltas.mean()
                sampling_rates[idx] = round(1.0 / mode_delta)
        except Exception:
            sampling_rates[idx] = None
    return sampling_rates

def generate_summary(sensor_id, date_str, sampling_rates_dict, csv_dfs):
    """
    Genera un resumen de los datos del sensor, incluyendo estadísticas globales y por segmento.
    Retorna un diccionario con el resumen.
    """
    segmentos = []
    todos_x, todos_y, todos_z = [], [], []
    total_samples = 0
    segment_id = 1
    
    for (archivo, df) in csv_dfs:
        
        start_time = parse_times_from_filename(archivo)
        if start_time is None:
            raise ValueError(f"No se pudo parsear hora de archivo {archivo}")
        sampling_rate = sampling_rates_dict.get(segment_id, None)
        
        if sampling_rate is None:
            raise ValueError(f"No hay sampling rate para segmento {segment_id}")
        
        duration_seconds = len(df) / sampling_rate
        dt_start = datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=start_time.hour,
            minute=start_time.minute,
            second=start_time.second
        )
        
        dt_end = dt_start + timedelta(seconds=duration_seconds)
        end_time = dt_end.time()
        mean_vals = {k: float(np.mean(df[k])) for k in ['x', 'y', 'z']}
        std_vals = {k: float(np.std(df[k], ddof=1)) for k in ['x', 'y', 'z']}
        
        segmentos.append({
            "segment_id": segment_id,
            "file_name": os.path.basename(archivo),
            "start_time": start_time.strftime("%H:%M:%S"),
            "end_time": end_time.strftime("%H:%M:%S"),
            "duration_seconds": int(duration_seconds),
            "sampling_rate_hz": sampling_rate,
            "samples": len(df),
            "statistics": {
                "mean": mean_vals,
                "std_dev": std_vals
            }
        })
        todos_x.append(df['x'].values)
        todos_y.append(df['y'].values)
        todos_z.append(df['z'].values)
        total_samples += len(df)
        segment_id += 1
        
    x_all = np.concatenate(todos_x)
    y_all = np.concatenate(todos_y)
    z_all = np.concatenate(todos_z)
    
    statistics_global = {
        "mean": {k: float(np.mean(arr)) for k, arr in zip(['x', 'y', 'z'], [x_all, y_all, z_all])},
        "std_dev": {k: float(np.std(arr, ddof=1)) for k, arr in zip(['x', 'y', 'z'], [x_all, y_all, z_all])}
    }
    sampling_rates_list = list({rate for rate in sampling_rates_dict.values() if rate is not None})
    summary = {
        "sensor_id": sensor_id,
        "date": date_str,
        "segments": len(csv_dfs),
        "samples_total": total_samples,
        "sampling_rates": sampling_rates_list,
        "statistics": statistics_global,
        "segment_data": segmentos
    }
    return summary

def process_bridge(bridge_path):
    # Calcular fecha del día anterior
    ayer = datetime.now() - timedelta(days=1)
    year = str(ayer.year)
    month = ayer.strftime("%B").lower()
    day = ayer.strftime("%d")

    # Buscar solo sensores del día anterior
    pattern = os.path.join(bridge_path, 'raw', year, month, day, 'sensor_*')
    sensor_dirs = glob.glob(pattern)
    sensores = []
    for sensor_dir in sensor_dirs:
        partes = os.path.normpath(sensor_dir).split(os.sep)
        if len(partes) < 7:
            continue
        sensor_id = partes[-1]
        day = partes[-2]
        month = partes[-3]
        year = partes[-4]
        
        sensores.append((sensor_dir, sensor_id, year, month, day))

    print(f"[+] [{os.path.basename(bridge_path)}] {len(sensores)} sensores para procesar del día anterior ({year}-{month}-{day}).")
    for sensor_dir, sensor_id, year, month, day in sensores:
        try:
            try:
                month_number = datetime.strptime(month, "%B").month
            except ValueError:
                month_number = int(month)
                
            date_str = f"{year}-{month_number:02d}-{int(day):02d}"
            
            csv_dfs = load_sensor_data(sensor_dir)
            
            sampling_rates_dict = calc_sampling_rates(csv_dfs)
            summary = generate_summary(sensor_id, date_str, sampling_rates_dict, csv_dfs)
            
            # Extraer solo el número del sensor para el nombre del archivo
            sensor_num = sensor_id.split('_')[1] if '_' in sensor_id else sensor_id
            out_path = os.path.join(sensor_dir, f"{sensor_num}_summary.json")
            
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[+] [{os.path.basename(bridge_path)}] Resumen generado en {out_path}")
        except Exception as e:
            print(f"[!] [{os.path.basename(bridge_path)}] Error procesando {sensor_dir}: {e}")

def start_summary(root):
    # Buscar todos los puentes bajo root
    bridges = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    
    print(f"[+] Procesando {len(bridges)} puentes en paralelo...")
    threads = []
    for bridge in bridges:
        t = threading.Thread(target=process_bridge, args=(bridge,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

def main():
    VERSION = "1.0.0"
    
    parser = argparse.ArgumentParser(description="Genera summary.json solo para los sensores del día anterior en la estructura root/puente/raw/año/mes/día/sensor_XX")
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    parser.add_argument('--root', required=True, type=str, help='Ruta raíz del dataset (root)')
    
    args = parser.parse_args()
    
    start_summary(args.root)

if __name__ == "__main__":
    main()

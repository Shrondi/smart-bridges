import os
from datetime import datetime
import pytz
import threading
import argparse
import glob

def readme_content(date_str, sensor_ids, time_reference, bridge_name):
    """
    Genera el contenido del README_daily.md para un día específico.
    Parámetros:
    - date_str: Fecha en formato 'YYYY-MM-DD'.
    - sensor_ids: Lista de IDs de sensores.
    - time_reference: Referencia de tiempo (horario de verano o estándar).
    - bridge_name: Nombre del puente.
    """
    
    sampling_rate = 125  # Hz
    segment_duration = "Variable – sensors record when triggered; each file contains a continuous segment of data"
    time_format = "HH:mm:ss.SSS (hours:minutes:seconds.milliseconds)"
    
    # date_str: YYYY-MM-DD
    # Para el texto, convertir a 'Month DD, YYYY'
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_str_nice = dt.strftime("%B %d, %Y")
    except Exception:
        date_str_nice = date_str
    
    lines = []
    lines.append(f"# Acceleration Data – {date_str}\n")
    lines.append(f"This folder contains raw acceleration data recorded on **{date_str_nice}** from {len(sensor_ids)} sensors installed on the {bridge_name} bridge.\n")
    lines.append(f"- **Sampling rate**: {sampling_rate} Hz")
    lines.append(f"- **Segment duration**: {segment_duration}")
    lines.append(f"- **Time format**: {time_format}")
    lines.append(f"- **Time reference**: {time_reference}")
    lines.append("- **CSV file format**: Each CSV file is comma-separated (`,`), and all numbers use a dot (`.`) as decimal separator. The columns are:")
    lines.append("    - `timestamp`: Time of the measurement")
    lines.append("    - `x_accel (g)`: Acceleration in X axis (in g)")
    lines.append("    - `y_accel (g)`: Acceleration in Y axis (in g)")
    lines.append("    - `z_accel (g)`: Acceleration in Z axis (in g)")
    lines.append("- **Sensors**:")
    for sensor_id in sensor_ids:
        lines.append(f"  - `{sensor_id}`")
    
    return "\n".join(lines)


def is_dst(date):
    """
    Retorna True si la fecha está en horario de verano (CEST) en España.
    """
    tz = pytz.timezone("Europe/Madrid")
    aware_date = tz.localize(date)
    return bool(aware_date.dst())


def generate_readme_daily(day_path):
    """
    Genera el README_daily.md para un día específico.
    Parámetros:
    - day_path: Ruta al directorio del día con datos de aceleración.
    """
    
    sensores_dir = [d for d in os.listdir(day_path) if os.path.isdir(os.path.join(day_path, d)) and d.lower().startswith("sensor_")]
    
    if not sensores_dir:
        print(f"[!] No se encontraron sensores en {day_path}. No se generará el README.")
        return
    
    sensor_ids = [f"Sensor {d.split('_', 1)[1]}" if d.lower().startswith('sensor_') else d for d in sorted(sensores_dir)]
    
    partes = os.path.normpath(day_path).split(os.sep)
    if len(partes) >= 4:
        bridge_name = partes[-5] if partes[-4] == 'raw' else partes[-4]
    else:
        bridge_name = "UNKNOWN"
    if len(partes) >= 3:
        year, month, day = partes[-3], partes[-2], partes[-1]
        try:
            try:
                month_number = datetime.strptime(month, "%B").month
            except ValueError:
                month_number = int(month)
            date_str = f"{year}-{month_number:02d}-{int(day):02d}"
        except Exception:
            date_str = f"{year}-{month}-{day}"
    else:
        date_str = "????-??-??"
        
    now = datetime.now()
    if is_dst(now):
        time_reference = "Times correspond to local time in Spain (CEST, UTC+2 — Daylight Saving Time)"
    else:
        time_reference = "Times correspond to local time in Spain (CET, UTC+1 — Standard Time)"
        
    readme_txt = readme_content(date_str, sensor_ids, time_reference, bridge_name)
    
    readme_path = os.path.join(day_path, "README_daily.md")
    with open(readme_path, "w") as f:
        f.write(readme_txt)
    
    print(f"[+] README_daily.md generado en: {readme_path}")


def start_create_readme(root_path):
    """
    Genera el README_daily.md para los datos de aceleración del día actual.
    Busca en la ruta raíz proporcionada y crea un README para cada día con datos.
    Parámetros:
    - root_path: Ruta raíz donde se guardarán los README.
    """
    if not os.path.isdir(root_path) or not os.path.exists(root_path):
        print(f"[!] La ruta raíz '{root_path}' no es un directorio válido o no existe.")
        return
    
    today = datetime.now()
    year = str(today.year)
    month_str = today.strftime("%B").lower()
    day = today.strftime("%d")
    
    day_dirs = [os.path.join(root_path, d, "raw", year, month_str, day) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    threads = []
    for day_path in day_dirs:
        if not os.path.exists(day_path):
            os.makedirs(day_path, exist_ok=True)

        print(f"[+] Procesando directorio: {day_path}")
        t = threading.Thread(target=generate_readme_daily, args=(day_path,))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()

if __name__ == "__main__":
    
    VERSION = "1.0.0"
    
    parser = argparse.ArgumentParser(description="Genera el README_daily.md para los datos de aceleración del día actual.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    parser.add_argument('--root', required=True, type=str, help='Ruta raíz donde se guardará el README')
    
    args = parser.parse_args()
    
    start_create_readme(args.root)

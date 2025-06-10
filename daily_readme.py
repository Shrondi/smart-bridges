import os
from datetime import datetime
import pytz
import threading
import argparse

def readme_content(date_str, sensor_ids, time_reference):
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
    lines.append(f"This folder contains raw acceleration data recorded on **{date_str_nice}** from {len(sensor_ids)} sensors installed on the AVE bridge.\n")
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


def generate_readme(root_path, puente):
    now = datetime.now()
    year = str(now.year)
    month_english = now.strftime("%B")
    month_number = now.strftime("%m")
    day = now.strftime("%d")
    
    dir_path = os.path.join(root_path, puente, year, month_english, day)
    os.makedirs(dir_path, exist_ok=True)
    
    # Detectar sensores presentes en la carpeta del día
    sensores_dir = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and d.startswith("sensor_")]
    
    if not sensores_dir:
        print(f"[!] No se encontraron sensores en {dir_path}. No se generará el README.")
        return
    
    # Formatear como 'Sensor XX'
    sensor_ids = [f"Sensor {d.split('_', 1)[1]}" if d.lower().startswith('sensor_') else d for d in sorted(sensores_dir)]
    
    date_str = f"{year}-{month_number}-{day}"
    
    # Detectar horario verano/invierno para time_reference
    if is_dst(now):
        time_reference = "Times correspond to local time in Spain (CEST, UTC+2 — Daylight Saving Time)"
    else:
        time_reference = "Times correspond to local time in Spain (CET, UTC+1 — Standard Time)"
    
    readme_txt = readme_content(date_str, sensor_ids, time_reference)
    readme_path = os.path.join(dir_path, "README_daily.md")
    
    with open(readme_path, "w") as f:
        f.write(readme_txt)
    print(f"README_daily.md creado en {readme_path}")


def create_readme(root_path):
    puentes = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    threads = []
    for puente in puentes:
        t = threading.Thread(target=generate_readme, args=(root_path, puente))
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
    
    create_readme(args.root)

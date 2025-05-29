import os
import subprocess
import time
import sys
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from check_train_file import verifyFile, moveToFolder

def esperar_quitar_lock(path_lock, timeout=120):
    """Espera hasta que desaparezca el .lock o se supere el timeout (en segundos)."""
    start_time = time.time()
    print(f"[!] Lock {path_lock} detectado. Esperando... ({int(elapsed)}s): ")
    while os.path.exists(path_lock):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"[✗] Timeout esperando lock ({timeout}s): {path_lock}")
            return False
        time.sleep(8)
    return True

def transferir_archivo(path, usuario, ip, destino_dir):
    try:
        lock_path = path + ".lock"

        print(f"[+] Nuevo archivo detectado: {path}")

        if not esperar_quitar_lock(lock_path):

            print(f"[✗] Archivo considerado anómalo por timeout de lock: {path}")
            ruta_final = moveToFolder(path)
            print(f"[!] Archivo movido a carpeta de anomalías: {ruta_final}")

        else:
            print(f"[✓] Lock eliminado. Preparando...: {path}")
            es_anomalo, ruta_final = verifyFile(path)

            if es_anomalo:
                print(f"[!] Archivo anómalo movido a: {ruta_final}")


        print(f"[+] Enviando archivo: {ruta_final}")
        comando = [
            "rsync", "-avz", "-e", "ssh", "--relative", "--remove-source-files",
            ruta_final, f"{usuario}@{ip}:{destino_dir}"
        ]
        resultado = subprocess.run(comando)
        if resultado.returncode == 0:
            print(f"[✓] Transferencia completada: {ruta_final}")
        else:
            print(f"[ERROR] Error en transferencia: {ruta_final}")
    except Exception as e:
        print(f"[ERROR] Fallo en procesamiento de {path}: {e}")

class CSVHandler(FileSystemEventHandler):
    def __init__(self, args):
        self.args = args

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            transferir_archivo(
                event.src_path,
                self.args.user,
                self.args.host,
                self.args.destination
            )

def procesar_existentes(args):
    for root, _, files in os.walk(args.source):
        for name in files:
            if name.endswith(".csv"):
                full_path = os.path.join(root, name)
                print(f"Procesando archivo preexistente: {full_path}")
                transferir_archivo(
                    full_path,
                    args.user,
                    args.host,
                    args.destination
                )

def main():

    VERSION = "2.1.0"

    parser = argparse.ArgumentParser(description="Monitoriza una estructura de directorios con archivos CSV y los transfiere después de procesarlos.")
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    parser.add_argument("--source", help="Directorio local a monitorizar")
    parser.add_argument("--user", help="Usuario SSH del destino")
    parser.add_argument("--host", help="IP o hostname del destino")
    parser.add_argument("--destination", help="Ruta remota donde transferir los archivos")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"[ERROR] Directorio de origen no existe: {args.source}")
        sys.exit(1)

    print(f"Monitorizando directorio: {args.source}")
    procesar_existentes(args)

    event_handler = CSVHandler(args)
    observer = Observer()
    observer.schedule(event_handler, args.source, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()

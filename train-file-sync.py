import os
import subprocess
import time
import sys
import argparse
import threading
from delete_samples import process_file
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from check_train_file import verifyFile, moveToFolder

DEBOUNCE_SECONDS = 5

def esperar_quitar_lock(path_lock, timeout=600):  # 10 min
    """Espera hasta que desaparezca el .lock o se supere el timeout (en segundos)."""
    start_time = time.time()
    print(f"[!] Lock {path_lock} detectado. Esperando...", end="")
    while os.path.exists(path_lock):
        elapsed = time.time() - start_time
        print(f"({int(elapsed)}s)")
        if elapsed > timeout:
            print(f"[✗] Timeout esperando lock ({timeout}s): {path_lock}")
            return False
        time.sleep(10)
    return True

def transferir_archivo(path, usuario, ip, destino_dir):
    try:
        lock_path = path + ".lock"

        print(f"[+] Nuevo archivo detectado: {path}")

        if not esperar_quitar_lock(lock_path):
            print(f"[✗] Archivo considerado anómalo por timeout de lock: {path}")
            os.remove(lock_path) # Elimina el lock ya que el archivo es considerado no válido
            ruta_final = moveToFolder(path)
            print(f"[!] Archivo movido a carpeta de anomalías: {ruta_final}")
        else:
            print(f"[✓] Lock eliminado. Preparando...: {path}")
            es_anomalo, ruta_final = verifyFile(path)
            if es_anomalo:
                print(f"[!] Archivo anómalo movido a: {ruta_final}")
        
        process_file(ruta_final)
        print(f"[+] Enviando archivo: {ruta_final}")
        comando = [
            "rsync", "-avz", "-e", "ssh", "--relative", "--remove-source-files",
            ruta_final, f"{usuario}@{ip}:{destino_dir}"
        ]
        
         # Ejecuta y lanza excepción si falla
        resultado = subprocess.run(comando, check=True)
        print(f"[✓] Transferencia completada: {ruta_final}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error en transferencia: {ruta_final}: {e}")
        raise 
    except Exception as e:
        print(f"[ERROR] Fallo en procesamiento de {path}: {e}")
        raise

class CSVHandler(FileSystemEventHandler):
    def __init__(self, args, observer):
        self.args = args
        self.observer = observer
        self._lock = threading.Lock()
        self._archivos_pendientes = set()
        self._timer = None

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            with self._lock:
                self._archivos_pendientes.add(event.src_path)
            self._reiniciar_timer()

    def _reiniciar_timer(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(DEBOUNCE_SECONDS, self._procesar_pendientes)
        self._timer.daemon = True
        self._timer.start()

    def _procesar_pendientes(self):
        with self._lock:
            archivos = list(self._archivos_pendientes)
            self._archivos_pendientes.clear()

        print(f"[DEBOUNCE] Procesando {len(archivos)} archivo(s) tras {DEBOUNCE_SECONDS}s sin cambios...")

        for path in archivos:
            try:
                transferir_archivo(
                    path,
                    self.args.user,
                    self.args.host,
                    self.args.destination
                )
            except Exception as e:
                print(f"[ERROR] Excepción en procesamiento: {e}")
                self.observer.stop()
                sys.exit(1)

def procesar_existentes(args):
    threads = []
    for root, _, files in os.walk(args.source):
        for name in files:
            if name.endswith(".csv"):
                full_path = os.path.join(root, name)
                print(f"Procesando archivo preexistente: {full_path}")
                hilo = threading.Thread(target=transferir_archivo, args=(
                    full_path,
                    args.user,
                    args.host,
                    args.destination
                ), daemon=True)
                hilo.start()
                threads.append(hilo)
    # Esperar a que terminen los hilos de archivos preexistentes antes de empezar a observar
    for hilo in threads:
        hilo.join()

def main():

    VERSION = "2.3.0"

    parser = argparse.ArgumentParser(description="Monitoriza una estructura de directorios con archivos CSV y los transfiere después de procesarlos.")
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    parser.add_argument("--source", help="Directorio local a monitorizar", required=True)
    parser.add_argument("--user", help="Usuario SSH del destino", required=True)
    parser.add_argument("--host", help="IP o hostname del destino", required=True)
    parser.add_argument("--destination", help="Ruta remota donde transferir los archivos", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"[ERROR] Directorio de origen no existe: {args.source}")
        sys.exit(1)

    print(f"Monitorizando directorio: {args.source}")
    procesar_existentes(args)

    observer = Observer()
    event_handler = CSVHandler(args, observer)
    observer.schedule(event_handler, args.source, recursive=True)
    observer.start()

    try:
        observer.join()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción detectada. Deteniendo observador...")
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()

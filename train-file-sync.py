import os
import subprocess
import time
import sys
import argparse
import threading
import concurrent.futures
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from train_file_processing import process_file

DEBOUNCE_SECONDS = 5

def esperar_quitar_lock(path_lock, timeout=1200):  # 20 min
    """Espera hasta que desaparezca el .lock o se supere el timeout (en segundos)."""
    start_time = time.time()
    
    if os.path.exists(path_lock):
        print(f"[!] Lock {path_lock} detectado")
        
        # Si el lock existe, espera hasta que desaparezca
        while os.path.exists(path_lock):
            elapsed = time.time() - start_time
            print(f"[!] Esperando {elapsed:.2f}s: {path_lock}")
            if elapsed > timeout:
                print(f"[✗] Timeout esperando lock ({timeout}s): {path_lock}")
                return False
            
            if elapsed == timeout/2:
                print(f"[!] Lock {path_lock} ha estado presente por más de {timeout/2} minutos... Cambiando comprobación cada 2 minutos")
                time.sleep(120)
            else:
                time.sleep(10)  # Espera 10 segundos antes de volver a comprobar

    return True

def transferir_archivo(path, usuario, ip, destino_dir):
    try:
        lock_path = path + ".lock"

        print(f"[+] Nuevo archivo detectado: {path}")

        if not esperar_quitar_lock(lock_path):
            print(f"[✗] Archivo {path} ha superado el timeout de espera para del lock")
            os.remove(lock_path) # Eliminar el lock si ha superado el timeout
        else:
            print(f"[✓] Lock eliminado. Preparando...: {path}")
        
        ruta_final = process_file(path)
        
        # Verifica si el archivo aún existe antes de enviar
        if not os.path.exists(ruta_final):
            print(f"[!] Archivo ya no existe: {ruta_final}. Se ignora.")
            return
        
        print(f"[+] Enviando archivo: {ruta_final}")
        comando = [
            "rsync", "-avz", "-e", "ssh", "--relative", "--remove-source-files",
            ruta_final, f"{usuario}@{ip}:{destino_dir}"
        ]
        
        # Ejecuta y lanza excepción si falla
        subprocess.run(comando, check=True)
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
        self._executor = concurrent.futures.ThreadPoolExecutor()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv") and not "anomalias" in event.src_path:
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

        futures = [
            self._executor.submit(
                transferir_archivo,
                path,
                self.args.user,
                self.args.host,
                self.args.destination
            ) for path in archivos
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Excepción en procesamiento: {e}")
                self.observer.stop()
                sys.exit(1)

def procesar_existentes(args):
    archivos = []
    for root, _, files in os.walk(args.source):
        for name in files:
            if name.endswith(".csv"):
                full_path = os.path.join(root, name)
                print(f"Procesando archivo preexistente: {full_path}")
                archivos.append(full_path)
                
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                transferir_archivo,
                full_path,
                args.user,
                args.host,
                args.destination
            ) for full_path in archivos
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Excepción en procesamiento de archivo preexistente: {e}")

def main():

    VERSION = "2.6.0"

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

#!/bin/bash

# Validar parámetros
if [ "$#" -ne 4 ]; then
    echo "Uso: $0 <directorio_origen> <usuario_remoto> <ip_remota> <directorio_destino>"
    exit 1
fi

ORIGEN_DIR="$1"
DESTINO_USER="$2"
DESTINO_IP="$3"
DESTINO_DIR="$4"
ESPERA_SEGUNDOS=5

# Comprobar existencia directorio origen
if [ ! -d "$ORIGEN_DIR" ]; then
    echo "El directorio de origen $ORIGEN_DIR no existe. Abortando."
    exit 1
fi

# Comprobar conexión SSH
ssh -q "$DESTINO_USER@$DESTINO_IP" exit
if [ $? -ne 0 ]; then
    echo "No se puede conectar a la máquina remota. Abortando."
    exit 1
fi

esperar_estabilidad() {
    local archivo="$1"
    while true; do
        if [ ! -f "$archivo" ]; then
            echo "Archivo $archivo no existe. Abortando espera."
            return 1
        fi
        local mod_time=$(stat -c %Y "$archivo")
        local now=$(date +%s)
        local diff=$((now - mod_time))
        if (( diff >= ESPERA_SEGUNDOS )); then
            return 0
        fi
        sleep 1
    done
}

procesar_y_transferir() {
    local archivo="$1"

    esperar_estabilidad "$archivo" || { echo "Error esperando estabilidad de $archivo"; return 1; }

    local ruta_final
    ruta_final=$(python3 discard-train-files.py "$archivo")
    local cod=$?

    if (( cod == 2 )); then
        echo "Error al procesar archivo con Python: $archivo"
        return 1
        
    elif (( cod == 1 )); then
        echo "Archivo anómalo detectado y movido a: $ruta_final."
    fi

    if [ ! -f "$ruta_final" ]; then
        echo "Archivo no encontrado para transferir: $ruta_final"
        return 1
    fi

    rsync -avz -e ssh --relative --remove-source-files "$ruta_final" "$DESTINO_USER@$DESTINO_IP:$DESTINO_DIR"
    if [ $? -eq 0 ]; then
        echo "Archivo $ruta_final transferido con éxito."
        return 0
    else
        echo "Error al transferir $ruta_final"
        return 1
    fi
}

# Procesar archivos preexistentes
find "$ORIGEN_DIR" -type f -name '*.csv' | while read -r archivo; do
    echo "Procesando archivo preexistente: $archivo"
    procesar_y_transferir "$archivo"
done

# Monitorizar nuevos archivos
inotifywait -m -r -e create --format "%w%f" "$ORIGEN_DIR" | while read archivo; do
    if [[ "$archivo" == *.csv ]]; then
        echo "Nuevo archivo CSV detectado: $archivo"
        procesar_y_transferir "$archivo"
    fi
done

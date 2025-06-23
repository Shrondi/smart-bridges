# Train Schedule - SmartBridges

Genera un gráfico PDF tipo "schedule" que muestra la actividad de grabación de los sensores de vibración de un puente ferroviario a lo largo de un día, a partir de los archivos CSV generados por los sensores.

## Requisitos

- Python 3.7+
- Paquetes: matplotlib, numpy, pandas

## Instalación

Instala las dependencias con:

```sh
pip install -r requirements.txt
```

## Uso básico

Ejecuta el script indicando la ruta base:

```sh
python train-schedule.py --root /ruta/a/la/carpeta/base
```

Parámetros útiles:
- `--yesterday HH:MM` : Hora para procesar también el día anterior (opcional, por defecto: 00:15)

## Salida

El gráfico PDF se guarda en cada carpeta de día procesada:

```
/ruta/a/la/carpeta/base/.../<año>/<mes>/<día>/train_recording_schedule.pdf
```

## Notas
- Los datos deben organizarse en carpetas `raw` (datos originales) y `report` (informes generados) dentro de la carpeta del puente, estructuradas en subcarpetas por año/mes/día.
- Los archivos CSV deben seguir el formato requerido por el script para ser procesados correctamente.
- Para más detalles, consulta la documentación completa del proyecto

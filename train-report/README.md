# Train Report - SmartBridges

Generación de informes PDF a partir de ficheros CSV con los datos de los acelerómetros, agrupando lógicamente los eventos de vibración detectados para determinar qué vibraciones corresponden al paso de los trenes y su posterior resumen en un fichero de reportes.

Para más detalles sobre el módulo, veáse la [documentación](https://shrondi.github.io/smart-bridges/train-report.html) completa del módulo.

## Requisitos

- Python 3.7+
- Paquetes: matplotlib, numpy, pandas, seaborn, scipy, PyPDF2

## Instalación

Instala las dependencias con:

```sh
pip install -r requirements.txt
```

## Uso básico

Ejecuta el script indicando la ruta al puente:

```sh
python train-report-smartbridges.py --bridge_path /ruta/al/puente
```

Parámetros útiles:
- `--date YYYYMMDD` : Fecha a procesar (opcional)
- `--regenerar-hora HH:MM:SS [HH:MM:SS ...]` : Regenera páginas concretas (opcional)

## Salida

El informe PDF se guarda en:

```
/ruta/al/puente/report/<año>/<mes>/<día>/train_report_<fecha>.pdf
```

## Notas
- Los datos deben organizarse en carpetas `raw` (datos originales) y `report` (informes generados) dentro de la carpeta del puente, estructuradas en subcarpetas por año/mes/día.
- Los archivos CSV deben seguir el formato requerido por el script para ser procesados correctamente.
- Para más detalles, consulta la documentación completa del proyecto




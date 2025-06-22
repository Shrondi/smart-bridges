# Train Report - SmartBridges <!-- omit from toc -->

Generación de informes PDF a partir de ficheros CSV con los datos de los acelerómetros, agrupando lógicamente los eventos de vibración detectados para determinar que vibraciones corresponden al paso de los trenes y su posterior resumen en un fichero de reportes.

## Índice <!-- omit from toc -->

- [Train Report - SmartBridges](#train-report---smartbridges)
  - [Descripción](#descripción)
  - [Uso](#uso)
    - [Requisitos](#requisitos)
    - [Ejecución básica](#ejecución-básica)
    - [Regenerar una página concreta del informe](#regenerar-una-página-concreta-del-informe)
    - [Salida](#salida)
- [Documentación adicional](#documentación-adicional)


## Descripción

El script `train-report-smartbridges.py` procesa los archivos CSV de los datos en crudo para cada uno de los sensores, agrupando los eventos de trenes detectados en función de la coincidencia temporal entre sensores. 

Una vez determinadas que vibraciones corresponden a un tren, se genera una página en un informe PDF con gráficas de aceleración por cada eje y la transformada FFT.

Para una explicación más detallada del proceso, véase [Funcionamiento](./EXPLANATION.md)

## Uso

### Requisitos

- Python 3.7+
- Paquetes: pandas, numpy, matplotlib, seaborn, scipy, PyPDF2

Instalación de dependencias (recomendado en un entorno virtual):

```sh
pip install pandas numpy matplotlib seaborn scipy PyPDF2
```

### Ejecución básica

Para generar el informe completo del día anterior:

```sh
python train-report-smartbridges.py --bridge_path /ruta/al/puente
```

Para especificar una fecha concreta (formato YYYYMMDD):

```sh
python train-report-smartbridges.py --bridge_path /ruta/al/puente --date 20240601
```

Parámetros opcionales:

- `--min_sensors N` : Número mínimo de sensores para considerar un evento (por defecto: 5)
- `--workers N`     : Número de hilos para el procesamiento paralelo (por defecto: 2 x núcleos CPU)

### Regenerar una página concreta del informe

Si necesitas regenerar la página de un tren concreto (por ejemplo, el tren detectado a las 12:34:56):

```sh
python train-report-smartbridges.py --bridge_path /ruta/al/puente --date 20240601 --regenerar-hora 12:34:56
```

Puedes pasar varias horas separadas por espacio:

```sh
python train-report-smartbridges.py --bridge_path /ruta/al/puente --date 20240601 --regenerar-hora 12:34:56 13:45:00
```

### Salida

El informe PDF se guarda en:

```
/ruta/al/puente/report/<año>/<mes>/<día>/train_report_<fecha>.pdf
```

# Documentación adicional
Para más detalles, consulta la documentación asociada:
- [Funcionamiento detallado](./EXPLANATION.md)



---


